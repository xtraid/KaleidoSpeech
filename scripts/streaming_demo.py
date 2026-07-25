"""Record microphone input in 40 ms blocks and print streaming observations."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
import time

from app.streaming_inference import (
    DistanceProvider,
    FinalEvaluator,
    StreamingConfig,
    StreamingInferenceSession,
    database_distance_provider,
)
from app.temporal_benchmark import _contrastive_decision
from app.temporal_repository import TemporalBenchmark, find_temporal_benchmark


@dataclass
class LiveDecisionTracker:
    target: TemporalBenchmark
    distance_provider: DistanceProvider
    end_silence_frames: int = 3
    speaking: bool = False
    silence_frames: int = 0
    result: str | None = None
    pre_roll: deque[bytes] | None = None
    utterance: list[bytes] | None = None
    final_evaluator: FinalEvaluator | None = None

    def __post_init__(self) -> None:
        self.pre_roll = deque(maxlen=3)
        self.utterance = []

    def update(self, event: dict, frame: bytes) -> str:
        assert self.pre_roll is not None
        assert self.utterance is not None
        if event["voice_activity"]:
            if not self.speaking:
                self.result = None
                self.utterance = list(self.pre_roll)
            self.speaking = True
            self.silence_frames = 0
            self.utterance.append(frame)
            return "ASCOLTO"

        if not self.speaking:
            self.pre_roll.append(frame)
            return self.result or "ATTENDI"

        self.utterance.append(frame)
        self.silence_frames += 1
        if self.silence_frames < self.end_silence_frames:
            return "ASCOLTO"

        self.speaking = False
        utterance_pcm = b"".join(self.utterance)
        self.utterance = []
        self.pre_roll.clear()
        if self.final_evaluator is not None:
            result = self.final_evaluator(utterance_pcm, self.target.word)
            self.result = {
                "CORRECT": "CORRETTO",
                "INCORRECT": "NON CORRETTO",
                "UNDECIDABLE": "INCERTO",
                "RETRY": "RIPETI",
            }[result["status"]]
            reason = ",".join(result.get("reason_codes", []))
            return f"{self.result} [{reason}]" if reason else self.result
        distances = self.distance_provider(utterance_pcm)
        if not distances:
            self.result = "RIPETI"
            return self.result
        decision = _contrastive_decision(
            self.target,
            distances,
            cleaning_status="accept",
        )
        self.result = {
            "CORRECT": "CORRETTO",
            "INCORRECT": "NON CORRETTO",
            "UNDECIDABLE": "INCERTO",
            "RETRY": "RIPETI",
        }[decision.status.value]
        return self.result


def main() -> None:
    import sounddevice as sd

    parser = argparse.ArgumentParser()
    parser.add_argument("words", nargs="+")
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--seed", default="demo")
    parser.add_argument(
        "--voice-threshold",
        type=float,
        default=0.012,
        help="RMS threshold separating ambient noise from speech (default: 0.012).",
    )
    parser.add_argument(
        "--decision-only",
        action="store_true",
        help="Print only a provisional target-word decision every 40 ms.",
    )
    parser.add_argument(
        "--phonetic",
        action="store_true",
        help="Run the utterance-level phoneme model after each detected word.",
    )
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow downloading the selected model if it is not cached.",
    )
    arguments = parser.parse_args()
    target = None
    config = None
    final_evaluator = None
    if arguments.decision_only:
        if len(arguments.words) != 1:
            parser.error("--decision-only requires exactly one target word")
        target = find_temporal_benchmark(arguments.words[0])
        if target is None:
            parser.error(
                f"no active temporal benchmark for {arguments.words[0]!r}"
            )
        # Audio feedback is emitted every 40 ms, while the expensive DTW pass
        # stays asynchronous at 200 ms so microphone capture remains real-time.
        config = StreamingConfig(
            voice_rms_threshold=arguments.voice_threshold,
        )
        if arguments.phonetic:
            from app.lexicon import EN_US_LEXICON
            from app.phonetic_pipeline import (
                DtwWordIdentityGate,
                PronunciationPipeline,
                cleaning_audio_preprocessor,
            )
            from app.pronunciation_engine import (
                TransformersWav2Vec2PhonemeModel,
            )

            provider = database_distance_provider()
            pipeline = PronunciationPipeline(
                word_gate=DtwWordIdentityGate(provider),
                phoneme_model=TransformersWav2Vec2PhonemeModel(
                    local_files_only=not arguments.allow_model_download
                ),
                lexicon=EN_US_LEXICON,
                audio_preprocessor=cleaning_audio_preprocessor,
            )
            final_evaluator = (
                lambda pcm, word: pipeline.evaluate(pcm, word).as_dict()
            )

    session = StreamingInferenceSession(
        arguments.words,
        distance_provider=(lambda _: {}) if arguments.decision_only else None,
        seed=arguments.seed,
        config=config,
    )
    tracker = (
        LiveDecisionTracker(
            target, database_distance_provider(),
            final_evaluator=final_evaluator,
        )
        if arguments.decision_only and target is not None
        else None
    )
    if arguments.decision_only:
        print(f"Pronuncia: {arguments.words[0]}")
    else:
        print(json.dumps(session.start_event(), indent=2))
    input(
        "Premi Invio, poi pronuncia la parola..."
        if arguments.decision_only
        else "Premi Invio, poi leggi la frase..."
    )
    frame_count = round(arguments.duration * 1000 / session.config.frame_ms)
    with sd.RawInputStream(
        samplerate=session.config.sample_rate,
        blocksize=session.config.sample_rate * session.config.frame_ms // 1000,
        channels=1,
        dtype="int16",
    ) as microphone:
        for _ in range(frame_count):
            frame, overflow = microphone.read(
                session.config.sample_rate * session.config.frame_ms // 1000
            )
            event = session.push_frame(bytes(frame))
            if arguments.decision_only:
                assert tracker is not None
                label = (
                    "ERRORE AUDIO"
                    if overflow
                    else tracker.update(event, bytes(frame))
                )
                print(
                    f"{event['elapsed_ms']:05d} ms  {label}",
                    flush=True,
                )
            else:
                event["input_overflow"] = bool(overflow)
                print(json.dumps(event), flush=True)
            time.sleep(0)
    final = session.finish()
    if not arguments.decision_only:
        print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
