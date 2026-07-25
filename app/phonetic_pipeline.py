"""Conservative word identity, CTC alignment, scoring and final policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
import re
from typing import Callable, Protocol, Sequence

import numpy as np

from app.pronunciation_engine import PhonemeEvidence, PhonemeModel


PIPELINE_VERSION = "pronunciation-decision-v2"
ALIGNMENT_VERSION = "numpy-ctc-viterbi-v1"
CALIBRATION_VERSION = "en-us-mvp-uncalibrated-v1"


class FinalStatus(StrEnum):
    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    UNDECIDABLE = "UNDECIDABLE"
    RETRY = "RETRY"


class PhoneStatus(StrEnum):
    PASS = "PASS"
    BORDERLINE = "BORDERLINE"
    FAIL = "FAIL"
    NOT_SCORABLE = "NOT_SCORABLE"


class AlignmentStatus(StrEnum):
    MATCH = "MATCH"
    SUBSTITUTION = "SUBSTITUTION"
    INSERTION = "INSERTION"
    DELETION = "DELETION"


@dataclass(frozen=True)
class QualityEvidence:
    acceptable: bool
    snr_db: float | None = None
    clipping: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class WordIdentityEvidence:
    target_word: str
    recognized_word: str | None
    target_probability: float
    recognized_probability: float
    status: str
    model_version: str
    alternatives: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass(frozen=True)
class AlignedPhone:
    expected: str
    observed: str | None
    start_ms: int
    end_ms: int
    expected_probability: float
    alternative_probability: float
    alignment_confidence: float
    status: AlignmentStatus = AlignmentStatus.MATCH
    duration_anomaly: bool = False


@dataclass(frozen=True)
class ScoredPhone:
    expected: str
    observed: str | None
    start_ms: int
    end_ms: int
    score: float
    confidence: float
    status: PhoneStatus


@dataclass(frozen=True)
class PronunciationDecision:
    status: FinalStatus
    reason_codes: tuple[str, ...]
    target_word: str
    recognized_word: str | None
    word_identity_confidence: float
    pronunciation_score: float | None
    phones: tuple[ScoredPhone, ...]
    quality: QualityEvidence
    versions: dict[str, str]

    def as_dict(self) -> dict:
        value = asdict(self)
        value["status"] = self.status.value
        for phone in value["phones"]:
            phone["status"] = phone["status"].value
        return value


class WordIdentityGate(Protocol):
    version: str

    def evaluate(self, pcm_s16le: bytes, target_word: str) -> WordIdentityEvidence:
        ...


class DtwWordIdentityGate:
    """Closed-vocabulary gate backed by the existing contrastive DTW models."""

    version = "closed-vocabulary-dtw-gate-v1"

    def __init__(
        self,
        distance_provider: Callable[[bytes], dict[str, float]],
        *,
        mismatch_margin_threshold: float = 0.05,
    ) -> None:
        if mismatch_margin_threshold < 0:
            raise ValueError("mismatch_margin_threshold cannot be negative")
        self._distance_provider = distance_provider
        self.mismatch_margin_threshold = mismatch_margin_threshold

    def evaluate(self, pcm_s16le: bytes, target_word: str) -> WordIdentityEvidence:
        distances = self._distance_provider(pcm_s16le)
        target = target_word.casefold()
        if target not in distances or len(distances) < 2:
            return WordIdentityEvidence(target, None, 0.0, 0.0, "UNCERTAIN", self.version)
        ordered = sorted(distances.items(), key=lambda item: item[1])
        recognized_word, _ = ordered[0]
        # A normalized closed-set distribution makes confidence observable.
        logits = -np.asarray([distance for _, distance in ordered], dtype=np.float64)
        probabilities = np.exp(logits - np.max(logits))
        probabilities /= np.sum(probabilities)
        by_word = {
            word: float(probability)
            for (word, _), probability in zip(ordered, probabilities)
        }
        margin = ordered[1][1] - ordered[0][1]
        if margin < self.mismatch_margin_threshold:
            status = "UNCERTAIN"
        else:
            status = "MATCH" if recognized_word == target else "MISMATCH"
        return WordIdentityEvidence(
            target_word=target,
            recognized_word=recognized_word,
            target_probability=round(by_word[target], 6),
            recognized_probability=round(by_word[recognized_word], 6),
            status=status,
            model_version=self.version,
            alternatives={
                word: round(probability, 6)
                for word, probability in by_word.items()
                if word != recognized_word
            },
            confidence=round(by_word[recognized_word], 6),
        )


def _token_id(labels: tuple[str, ...], phone: str) -> int:
    candidates = (phone, phone.casefold(), phone.upper())
    for candidate in candidates:
        if candidate in labels:
            return labels.index(candidate)
    raise ValueError(f"Phone {phone!r} is absent from the model vocabulary")


def ctc_force_align(
    evidence: PhonemeEvidence,
    expected_phones: Sequence[str],
) -> tuple[AlignedPhone, ...]:
    """Viterbi-align a phone sequence to CTC frames.

    This implementation is intentionally local because torchaudio's forced
    alignment API is deprecated. It supports repeated phones through the
    standard blank-interleaved CTC state graph.
    """
    if not expected_phones:
        raise ValueError("Expected phone sequence cannot be empty")
    phone_ids = [_token_id(evidence.labels, phone) for phone in expected_phones]
    states = [evidence.blank_id]
    for phone_id in phone_ids:
        states.extend((phone_id, evidence.blank_id))

    frames = evidence.log_probabilities
    if len(frames) < len(phone_ids):
        raise ValueError("Too few acoustic frames for the expected phones")
    scores = np.full((len(frames), len(states)), -np.inf, dtype=np.float64)
    back = np.full((len(frames), len(states)), -1, dtype=np.int32)
    scores[0, 0] = frames[0, states[0]]
    if len(states) > 1:
        scores[0, 1] = frames[0, states[1]]

    for frame in range(1, len(frames)):
        for state, label_id in enumerate(states):
            predecessors = [state]
            if state > 0:
                predecessors.append(state - 1)
            if (
                state > 1
                and label_id != evidence.blank_id
                and label_id != states[state - 2]
            ):
                predecessors.append(state - 2)
            best = max(predecessors, key=lambda index: scores[frame - 1, index])
            scores[frame, state] = scores[frame - 1, best] + frames[frame, label_id]
            back[frame, state] = best

    possible_ends = [len(states) - 1, len(states) - 2]
    state = max(possible_ends, key=lambda index: scores[-1, index])
    if not np.isfinite(scores[-1, state]):
        raise ValueError("No valid CTC alignment path")
    path = [state]
    for frame in range(len(frames) - 1, 0, -1):
        state = int(back[frame, state])
        path.append(state)
    path.reverse()

    aligned: list[AlignedPhone] = []
    for phone_index, (phone, phone_id) in enumerate(zip(expected_phones, phone_ids)):
        state_id = phone_index * 2 + 1
        indices = np.flatnonzero(np.asarray(path) == state_id)
        if not len(indices):
            aligned.append(
                AlignedPhone(
                    phone,
                    None,
                    0,
                    0,
                    0.0,
                    0.0,
                    0.0,
                    AlignmentStatus.DELETION,
                )
            )
            continue
        segment = frames[indices]
        probabilities = np.exp(segment)
        expected_probability = float(np.mean(probabilities[:, phone_id]))
        alternatives = probabilities.copy()
        alternatives[:, (evidence.blank_id, phone_id)] = 0.0
        alternative_id = int(np.argmax(np.mean(alternatives, axis=0)))
        alternative_probability = float(np.mean(alternatives[:, alternative_id]))
        observed = (
            evidence.labels[alternative_id]
            if alternative_probability > expected_probability
            else phone
        )
        confidence = max(expected_probability, alternative_probability)
        status = (
            AlignmentStatus.DELETION
            if confidence < 0.05
            else (
                AlignmentStatus.SUBSTITUTION
                if observed != phone
                else AlignmentStatus.MATCH
            )
        )
        duration_ms = (indices[-1] - indices[0] + 1) * evidence.frame_stride_ms
        aligned.append(
            AlignedPhone(
                expected=phone,
                observed=observed,
                start_ms=round(indices[0] * evidence.frame_stride_ms),
                end_ms=round((indices[-1] + 1) * evidence.frame_stride_ms),
                expected_probability=expected_probability,
                alternative_probability=alternative_probability,
                # Alignment confidence measures whether this span contains a
                # stable phone, not whether it is the expected phone. Keeping
                # these concepts separate is what makes substitutions scorable.
                alignment_confidence=confidence,
                status=status,
                duration_anomaly=duration_ms < 20 or duration_ms > 500,
            )
        )

    # Unexpected high-confidence labels in CTC blank spans are insertions.
    path_array = np.asarray(path)
    insertion_frames: list[tuple[int, int]] = []
    substitution_labels = {
        phone.observed
        for phone in aligned
        if phone.status is AlignmentStatus.SUBSTITUTION
    }
    for frame, state_id in enumerate(path_array):
        if states[int(state_id)] != evidence.blank_id:
            continue
        row = np.exp(frames[frame]).copy()
        row[evidence.blank_id] = 0.0
        label_id = int(np.argmax(row))
        if row[label_id] >= 0.5 and evidence.labels[label_id] not in substitution_labels:
            insertion_frames.append((frame, label_id))
    start = 0
    while start < len(insertion_frames):
        end = start + 1
        first_frame, label_id = insertion_frames[start]
        while (
            end < len(insertion_frames)
            and insertion_frames[end][0] == insertion_frames[end - 1][0] + 1
            and insertion_frames[end][1] == label_id
        ):
            end += 1
        last_frame = insertion_frames[end - 1][0]
        segment_probability = float(
            np.mean(np.exp(frames[first_frame : last_frame + 1, label_id]))
        )
        aligned.append(
            AlignedPhone(
                expected="",
                observed=evidence.labels[label_id],
                start_ms=round(first_frame * evidence.frame_stride_ms),
                end_ms=round((last_frame + 1) * evidence.frame_stride_ms),
                expected_probability=0.0,
                alternative_probability=segment_probability,
                alignment_confidence=segment_probability,
                status=AlignmentStatus.INSERTION,
            )
        )
        start = end
    return tuple(sorted(aligned, key=lambda phone: (phone.start_ms, not phone.expected)))


@dataclass(frozen=True)
class ScoringThresholds:
    fail_margin: float = -0.35
    pass_margin: float = 0.35
    minimum_alignment_confidence: float = 0.20
    per_phone: dict[str, tuple[float, float]] = field(default_factory=dict)
    expected_duration_ms: dict[str, float] = field(default_factory=dict)
    duration_penalty_weight: float = 0.0

    def margins_for(self, phone: str) -> tuple[float, float]:
        fail, passed = self.per_phone.get(
            phone, (self.fail_margin, self.pass_margin)
        )
        if fail >= passed:
            raise ValueError("Phone fail threshold must be below pass threshold")
        return fail, passed


def score_phones(
    phones: Sequence[AlignedPhone],
    thresholds: ScoringThresholds = ScoringThresholds(),
) -> tuple[ScoredPhone, ...]:
    scored: list[ScoredPhone] = []
    for phone in phones:
        if phone.status in {AlignmentStatus.INSERTION, AlignmentStatus.DELETION}:
            scored.append(
                ScoredPhone(
                    expected=phone.expected,
                    observed=phone.observed,
                    start_ms=phone.start_ms,
                    end_ms=phone.end_ms,
                    score=-20.0,
                    confidence=round(phone.alignment_confidence, 6),
                    status=PhoneStatus.FAIL,
                )
            )
            continue
        fail_margin, pass_margin = thresholds.margins_for(phone.expected)
        margin = float(
            np.log(max(phone.expected_probability, 1e-9))
            - np.log(max(phone.alternative_probability, 1e-9))
        )
        expected_duration = thresholds.expected_duration_ms.get(phone.expected)
        if expected_duration is not None:
            if expected_duration <= 0:
                raise ValueError("Expected phone duration must be positive")
            actual_duration = max(phone.end_ms - phone.start_ms, 1)
            margin -= thresholds.duration_penalty_weight * abs(
                float(np.log(actual_duration / expected_duration))
            )
        if phone.alignment_confidence < thresholds.minimum_alignment_confidence:
            status = PhoneStatus.NOT_SCORABLE
        elif margin <= fail_margin:
            status = PhoneStatus.FAIL
        elif margin >= pass_margin:
            status = PhoneStatus.PASS
        else:
            status = PhoneStatus.BORDERLINE
        scored.append(
            ScoredPhone(
                expected=phone.expected,
                observed=phone.observed,
                start_ms=phone.start_ms,
                end_ms=phone.end_ms,
                score=round(margin, 6),
                confidence=round(phone.alignment_confidence, 6),
                status=status,
            )
        )
    return tuple(scored)


def decide_pronunciation(
    *,
    target_word: str,
    quality: QualityEvidence,
    identity: WordIdentityEvidence | None,
    phones: Sequence[ScoredPhone] = (),
    versions: dict[str, str] | None = None,
) -> PronunciationDecision:
    """Apply the non-negotiable cascade; CORRECT needs positive evidence."""
    reasons: tuple[str, ...]
    if not quality.acceptable:
        status, reasons = FinalStatus.RETRY, (quality.reason or "POOR_QUALITY",)
    elif identity is None or identity.status == "UNCERTAIN":
        status, reasons = FinalStatus.UNDECIDABLE, ("WORD_IDENTITY_UNCERTAIN",)
    elif identity.status == "MISMATCH":
        status, reasons = FinalStatus.INCORRECT, ("WRONG_WORD",)
    elif not phones or any(p.status is PhoneStatus.NOT_SCORABLE for p in phones):
        status, reasons = FinalStatus.UNDECIDABLE, ("ALIGNMENT_UNRELIABLE",)
    elif any(p.status is PhoneStatus.FAIL for p in phones):
        status, reasons = FinalStatus.INCORRECT, ("PHONE_ERROR",)
    elif all(p.status is PhoneStatus.PASS for p in phones):
        status, reasons = FinalStatus.CORRECT, ("POSITIVE_PHONETIC_EVIDENCE",)
    else:
        status, reasons = FinalStatus.UNDECIDABLE, ("PHONETIC_EVIDENCE_AMBIGUOUS",)

    pronunciation_score = (
        round(float(np.mean([phone.score for phone in phones])), 6)
        if phones else None
    )
    return PronunciationDecision(
        status=status,
        reason_codes=reasons,
        target_word=target_word,
        recognized_word=identity.recognized_word if identity else None,
        word_identity_confidence=identity.target_probability if identity else 0.0,
        pronunciation_score=pronunciation_score,
        phones=tuple(phones),
        quality=quality,
        versions={
            "alignment": ALIGNMENT_VERSION,
            "calibration": CALIBRATION_VERSION,
            "decision": PIPELINE_VERSION,
            **(versions or {}),
        },
    )


class PronunciationPipeline:
    def __init__(
        self,
        *,
        word_gate: WordIdentityGate,
        phoneme_model: PhonemeModel,
        lexicon: dict[str, list[list[str]]],
        audio_preprocessor: (
            Callable[[bytes], tuple[bytes, QualityEvidence]] | None
        ) = None,
        thresholds: ScoringThresholds = ScoringThresholds(),
    ) -> None:
        self.word_gate = word_gate
        self.phoneme_model = phoneme_model
        self.lexicon = {word.casefold(): variants for word, variants in lexicon.items()}
        self.audio_preprocessor = audio_preprocessor or (
            lambda pcm: (pcm, QualityEvidence(True))
        )
        self.thresholds = thresholds

    def evaluate(self, pcm_s16le: bytes, target_word: str) -> PronunciationDecision:
        prepared_pcm, quality = self.audio_preprocessor(pcm_s16le)
        if not quality.acceptable:
            return decide_pronunciation(
                target_word=target_word, quality=quality, identity=None
            )
        identity = self.word_gate.evaluate(prepared_pcm, target_word)
        if identity.status != "MATCH":
            return decide_pronunciation(
                target_word=target_word, quality=quality, identity=identity
            )
        variants = self.lexicon.get(target_word.casefold())
        if not variants:
            return decide_pronunciation(
                target_word=target_word,
                quality=quality,
                identity=identity,
                versions={"lexicon": "missing"},
            )
        evidence = self.phoneme_model.infer(prepared_pcm, 16_000)
        candidates: list[tuple[ScoredPhone, ...]] = []
        for variant in variants:
            candidates.append(score_phones(ctc_force_align(evidence, variant), self.thresholds))
        phones = max(
            candidates,
            key=lambda candidate: sum(phone.score for phone in candidate),
        )
        return decide_pronunciation(
            target_word=target_word,
            quality=quality,
            identity=identity,
            phones=phones,
            versions={
                "word_identity": self.word_gate.version,
                "phoneme_model": evidence.model_version,
                "lexicon": "en-us-espeak-v1",
            },
        )


class SentencePronunciationPipeline:
    """Forced-align and score every phone in a known sentence.

    ``target_word`` remains the pedagogical focus reported to the UI, while
    the decision covers the complete ``expected_text``.
    """

    def __init__(
        self,
        *,
        phoneme_model: PhonemeModel,
        lexicon: dict[str, list[list[str]]],
        audio_preprocessor: (
            Callable[[bytes], tuple[bytes, QualityEvidence]] | None
        ) = None,
        thresholds: ScoringThresholds = ScoringThresholds(),
    ) -> None:
        self.phoneme_model = phoneme_model
        self.lexicon = {
            word.casefold(): variants for word, variants in lexicon.items()
        }
        self.audio_preprocessor = audio_preprocessor or (
            lambda pcm: (pcm, QualityEvidence(True))
        )
        self.thresholds = thresholds

    def evaluate(
        self, pcm_s16le: bytes, expected_text: str, target_word: str
    ) -> PronunciationDecision:
        prepared_pcm, quality = self.audio_preprocessor(pcm_s16le)
        target = target_word.casefold()
        if not quality.acceptable:
            return decide_pronunciation(
                target_word=target, quality=quality, identity=None
            )
        words = re.findall(r"[A-Za-z']+", expected_text.casefold())
        if target not in words or any(word not in self.lexicon for word in words):
            return decide_pronunciation(
                target_word=target,
                quality=quality,
                identity=None,
                versions={"lexicon": "sentence-missing"},
            )
        target_word_index = words.index(target)
        evidence = self.phoneme_model.infer(prepared_pcm, 16_000)
        identity = WordIdentityEvidence(
            target_word=target,
            recognized_word=target,
            target_probability=1.0,
            recognized_probability=1.0,
            status="MATCH",
            model_version="known-text-forced-alignment-v1",
            confidence=1.0,
        )
        candidates: list[tuple[ScoredPhone, ...]] = []
        for target_variant in self.lexicon[target]:
            sentence_phones: list[str] = []
            for index, word in enumerate(words):
                phones = (
                    target_variant
                    if index == target_word_index
                    else self.lexicon[word][0]
                )
                sentence_phones.extend(phones)
            try:
                aligned = ctc_force_align(evidence, sentence_phones)
            except ValueError as error:
                # Lexicons may contain regional phones not emitted by a
                # particular checkpoint. Ignore only that incompatible
                # variant; another supported pronunciation can still provide
                # valid positive evidence.
                if "absent from the model vocabulary" in str(error):
                    continue
                raise
            candidates.append(score_phones(aligned, self.thresholds))
        if not candidates:
            return decide_pronunciation(
                target_word=target,
                quality=quality,
                identity=identity,
                versions={
                    "phoneme_model": evidence.model_version,
                    "lexicon": "sentence-model-incompatible",
                },
            )
        phones = max(
            candidates,
            key=lambda candidate: sum(phone.score for phone in candidate),
        )
        return decide_pronunciation(
            target_word=target,
            quality=quality,
            identity=identity,
            phones=phones,
            versions={
                "word_identity": identity.model_version,
                "phoneme_model": evidence.model_version,
                "lexicon": "en-us-espeak-sentence-v1",
            },
        )


def cleaning_audio_preprocessor(
    pcm_s16le: bytes,
) -> tuple[bytes, QualityEvidence]:
    """Adapt the existing cleaner/report to the phonetic quality contract."""
    from app.cleaning import clean_pcm_with_report

    try:
        cleaned, report = clean_pcm_with_report(pcm_s16le)
    except ValueError as error:
        return b"", QualityEvidence(False, reason=f"CLEANING_ERROR:{error}")
    acceptable = report.status != "retry_recommended"
    return cleaned, QualityEvidence(
        acceptable=acceptable,
        snr_db=report.snr_db,
        clipping=report.input_clipped_ratio > 0.01,
        reason=None if acceptable else "CLEANING_RETRY_RECOMMENDED",
    )
