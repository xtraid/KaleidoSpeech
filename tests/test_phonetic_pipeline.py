import numpy as np

from app.phonetic_pipeline import (
    FinalStatus,
    PhoneStatus,
    QualityEvidence,
    ScoredPhone,
    WordIdentityEvidence,
    AlignedPhone,
    AlignmentStatus,
    ScoringThresholds,
    SentencePronunciationPipeline,
    ctc_force_align,
    decide_pronunciation,
    score_phones,
)
from app.pronunciation_engine import PhonemeEvidence


class _SentenceModel:
    version = "synthetic-sentence-ctc"

    def infer(self, _pcm: bytes, _sample_rate: int) -> PhonemeEvidence:
        labels = ("<pad>", "s", "b", "ɜː", "d", "x")
        winners = (0, 1, 0, 2, 0, 3, 0, 4, 0)
        probabilities = np.full((len(winners), len(labels)), 0.005)
        for frame, winner in enumerate(winners):
            probabilities[frame, winner] = 0.975
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return PhonemeEvidence(
            np.log(probabilities), labels, 0, 20, self.version
        )


def _identity(status: str = "MATCH") -> WordIdentityEvidence:
    return WordIdentityEvidence(
        "bird", "bird" if status != "MISMATCH" else "happy",
        0.92, 0.92, status, "test-word-gate",
    )


def _phone(status: PhoneStatus) -> ScoredPhone:
    return ScoredPhone("ɜː", "ɜː", 20, 80, 1.2, 0.9, status)


def test_decision_cascade_requires_positive_phonetic_evidence() -> None:
    poor = decide_pronunciation(
        target_word="bird",
        quality=QualityEvidence(False, reason="NO_SPEECH"),
        identity=_identity("MISMATCH"),
    )
    wrong = decide_pronunciation(
        target_word="bird", quality=QualityEvidence(True),
        identity=_identity("MISMATCH"),
    )
    uncertain = decide_pronunciation(
        target_word="bird", quality=QualityEvidence(True), identity=_identity(),
        phones=[_phone(PhoneStatus.BORDERLINE)],
    )
    correct = decide_pronunciation(
        target_word="bird", quality=QualityEvidence(True), identity=_identity(),
        phones=[_phone(PhoneStatus.PASS)],
    )

    assert poor.status is FinalStatus.RETRY
    assert wrong.status is FinalStatus.INCORRECT
    assert wrong.reason_codes == ("WRONG_WORD",)
    assert uncertain.status is FinalStatus.UNDECIDABLE
    assert correct.status is FinalStatus.CORRECT


def test_sentence_pipeline_scores_every_phone_in_known_text() -> None:
    pipeline = SentencePronunciationPipeline(
        phoneme_model=_SentenceModel(),
        lexicon={"say": [["s"]], "bird": [["b", "ɜː", "d"]]},
    )

    result = pipeline.evaluate(b"audio", "Say bird", "bird")

    assert result.status is FinalStatus.CORRECT
    assert [phone.expected for phone in result.phones] == ["s", "b", "ɜː", "d"]
    assert result.reason_codes == ("POSITIVE_PHONETIC_EVIDENCE",)


def test_sentence_pipeline_skips_variant_absent_from_model_vocabulary() -> None:
    pipeline = SentencePronunciationPipeline(
        phoneme_model=_SentenceModel(),
        lexicon={
            "say": [["s"]],
            "bird": [["b", "ɜː", "d"], ["b", "ɝ", "d"]],
        },
    )

    result = pipeline.evaluate(b"audio", "Say bird", "bird")

    assert result.status is FinalStatus.CORRECT
    assert [phone.expected for phone in result.phones] == ["s", "b", "ɜː", "d"]


def test_ctc_alignment_localizes_a_phone_substitution() -> None:
    labels = ("<pad>", "b", "ɜː", "d", "ɪ")
    probabilities = np.full((9, len(labels)), 0.01)
    winners = [0, 1, 1, 0, 4, 4, 0, 3, 0]
    for frame, winner in enumerate(winners):
        probabilities[frame, winner] = 0.92
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    evidence = PhonemeEvidence(
        np.log(probabilities), labels, 0, 20, "synthetic-ctc"
    )

    aligned = ctc_force_align(evidence, ["b", "ɜː", "d"])
    scored = score_phones(aligned)

    assert [phone.expected for phone in aligned] == ["b", "ɜː", "d"]
    assert aligned[1].observed == "ɪ"
    assert aligned[1].status is AlignmentStatus.SUBSTITUTION
    assert aligned[1].start_ms < aligned[1].end_ms
    assert scored[1].status is PhoneStatus.FAIL


def test_unreliable_alignment_can_never_be_correct() -> None:
    result = decide_pronunciation(
        target_word="bird", quality=QualityEvidence(True), identity=_identity(),
        phones=[_phone(PhoneStatus.NOT_SCORABLE)],
    )

    assert result.status is FinalStatus.UNDECIDABLE
    assert result.reason_codes == ("ALIGNMENT_UNRELIABLE",)


def test_duration_normalization_penalizes_abnormal_phone_span() -> None:
    aligned = AlignedPhone("b", "b", 0, 400, 0.8, 0.2, 0.8)
    baseline = score_phones([aligned])[0]
    normalized = score_phones(
        [aligned],
        ScoringThresholds(
            expected_duration_ms={"b": 100},
            duration_penalty_weight=1.0,
        ),
    )[0]

    assert normalized.score < baseline.score


def test_ctc_alignment_reports_unexpected_phone_in_blank_span() -> None:
    labels = ("<pad>", "b", "x")
    probabilities = np.full((6, len(labels)), 0.02)
    for frame, winner in enumerate((0, 1, 0, 2, 2, 0)):
        probabilities[frame, winner] = 0.96
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    evidence = PhonemeEvidence(
        np.log(probabilities), labels, 0, 20, "synthetic-ctc"
    )

    aligned = ctc_force_align(evidence, ["b"])

    insertion = next(
        phone for phone in aligned if phone.status is AlignmentStatus.INSERTION
    )
    assert insertion.observed == "x"
    assert score_phones(aligned)[-1].status is PhoneStatus.FAIL
