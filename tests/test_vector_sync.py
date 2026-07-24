import json

from app.acoustic_repository import StoredRecording
from app.vector_sync import (
    normalization_key,
    query_recording_neighbors,
    sync_recording_embeddings,
)


def _recording(recording_id, word, values):
    return StoredRecording(
        id=recording_id,
        word=word,
        speaker_hash=f"speaker-{recording_id}",
        dataset_split="train",
        source_hash=f"hash-{recording_id}",
        duration_ms=500,
        feature=values,
        quality_status="ACCEPT",
        cleaning_version="clean-v1",
        extractor_version="mfcc-summary-v1",
    )


class RedisSearchDouble:
    def __init__(self):
        self.values = {}
        self.hashes = {}
        self.search_commands = []

    def execute_command(self, *command):
        if command[0] == "FT.INFO":
            return [b"index_name", b"idx:acoustic_embeddings"]
        if command[0] == "FT.SEARCH":
            self.search_commands.append(command)
            return [
                1,
                b"acoustic:embedding:1",
                [b"word", b"bird", b"locale", b"en-US",
                 b"distance", b"0"],
            ]
        raise AssertionError(command)

    def pipeline(self, transaction=False):
        assert transaction is False
        return self

    def hset(self, key, mapping):
        self.hashes[key] = mapping

    def set(self, key, value):
        self.values[key] = value

    def get(self, key):
        return self.values.get(key)

    def execute(self):
        return []


def test_sync_is_normalized_and_queryable(monkeypatch):
    recordings = [
        _recording(1, "bird", [1.0] * 26),
        _recording(2, "happy", [3.0] * 26),
    ]
    monkeypatch.setattr(
        "app.vector_sync.list_recordings", lambda **_: recordings
    )
    redis = RedisSearchDouble()

    report = sync_recording_embeddings(redis)
    neighbors = query_recording_neighbors(redis, 1, k=1)

    assert report.indexed == 2
    assert report.words == ("bird", "happy")
    assert len(redis.hashes) == 2
    metadata = json.loads(redis.values[normalization_key("en-US")])
    assert metadata["center"] == [2.0] * 26
    assert metadata["scale"] == [1.0] * 26
    assert neighbors[0]["word"] == "bird"
    assert neighbors[0]["distance"] == 0.0
    assert "@locale:{en\\-US}" in redis.search_commands[-1][2]


def test_normalizations_and_queries_are_isolated_by_locale(monkeypatch):
    redis = RedisSearchDouble()
    current_locale = "en-US"

    def recordings_for_locale(*, locale, **_):
        nonlocal current_locale
        current_locale = locale
        offset = 1.0 if locale == "en-US" else 10.0
        return [
            _recording(1, "bird", [offset] * 26),
            _recording(2, "happy", [offset + 2.0] * 26),
        ]

    monkeypatch.setattr(
        "app.vector_sync.list_recordings", recordings_for_locale
    )
    sync_recording_embeddings(redis, locale="en-US")
    sync_recording_embeddings(redis, locale="en-GB")

    assert normalization_key("en-US") in redis.values
    assert normalization_key("en-GB") in redis.values
    assert redis.values[normalization_key("en-US")] != (
        redis.values[normalization_key("en-GB")]
    )

    query_recording_neighbors(redis, 1, locale="en-US", k=1)
    assert "@locale:{en\\-US}" in redis.search_commands[-1][2]
