"""Process-wide construction of the optional server-side phonetic backend."""

from functools import lru_cache

from app.lexicon import EN_US_LEXICON, EN_US_SENTENCE_LEXICON
from app.config import get_settings
from app.ensemble_inference import (
    get_ensemble_decision_engine,
    voice_frame_ratio,
)
from app.phonetic_pipeline import (
    DtwWordIdentityGate,
    PronunciationPipeline,
    SentencePronunciationPipeline,
    cleaning_audio_preprocessor,
)
from app.pronunciation_engine import TransformersWav2Vec2PhonemeModel
from app.streaming_inference import FinalEvaluator, database_distance_provider


@lru_cache(maxsize=1)
def get_phoneme_model() -> TransformersWav2Vec2PhonemeModel:
    """Load one process-wide model shared by word and sentence pipelines."""
    settings = get_settings()
    return TransformersWav2Vec2PhonemeModel(
        quantize_cpu=settings.phonetic_cpu_quantization,
    )


@lru_cache(maxsize=1)
def get_final_evaluator() -> FinalEvaluator:
    distances = database_distance_provider()
    pipeline = PronunciationPipeline(
        word_gate=DtwWordIdentityGate(
            distances,
            mismatch_margin_threshold=(
                get_settings().word_gate_mismatch_margin_threshold
            ),
        ),
        phoneme_model=get_phoneme_model(),
        lexicon=EN_US_LEXICON,
        audio_preprocessor=cleaning_audio_preprocessor,
    )
    ensemble = get_ensemble_decision_engine()

    def evaluate(pcm: bytes, word: str) -> dict:
        result = pipeline.evaluate(pcm, word).as_dict()
        return ensemble.evaluate(
            result,
            pcm,
            voice_frame_ratio=voice_frame_ratio(pcm),
        )

    return evaluate


@lru_cache(maxsize=1)
def get_sentence_final_evaluator():
    pipeline = SentencePronunciationPipeline(
        phoneme_model=get_phoneme_model(),
        lexicon=EN_US_SENTENCE_LEXICON,
        audio_preprocessor=cleaning_audio_preprocessor,
    )
    ensemble = get_ensemble_decision_engine()

    def evaluate(pcm: bytes, text: str, word: str) -> dict:
        result = pipeline.evaluate(pcm, text, word).as_dict()
        tokens = __import__("re").findall(r"[A-Za-z']+", text.casefold())
        scored = [phone for phone in result.get("phones", []) if phone["expected"]]
        offset = 0
        words = []
        for index, token in enumerate(tokens):
            expected = EN_US_SENTENCE_LEXICON[token][0]
            word_phones = scored[offset:offset + len(expected)]
            offset += len(expected)
            statuses = {phone["status"] for phone in word_phones}
            if "FAIL" in statuses:
                status = "INCORRECT"
            elif word_phones and statuses == {"PASS"}:
                status = "CORRECT"
            else:
                status = "UNDECIDABLE"
            words.append({
                "index": index,
                "word": token,
                "status": status,
                "pronunciation_score": (
                    round(sum(phone["score"] for phone in word_phones) / len(word_phones), 6)
                    if word_phones else None
                ),
                "phones": word_phones,
            })
        result["words"] = words
        target_pcm = _target_word_pcm(pcm, words, word)
        return ensemble.evaluate(
            result,
            target_pcm,
            voice_frame_ratio=voice_frame_ratio(target_pcm),
        )

    return evaluate


def _target_word_pcm(pcm: bytes, words: list[dict], target_word: str) -> bytes:
    """Slice the forced-aligned focus word for word-level benchmark scorers."""
    for word in words:
        if str(word.get("word", "")).casefold() != target_word.casefold():
            continue
        phones = word.get("phones") or []
        if not phones:
            return b""
        start_ms = max(0, min(int(phone["start_ms"]) for phone in phones) - 20)
        end_ms = max(int(phone["end_ms"]) for phone in phones) + 20
        bytes_per_ms = 32  # mono 16 kHz signed int16
        return pcm[start_ms * bytes_per_ms:end_ms * bytes_per_ms]
    return b""
