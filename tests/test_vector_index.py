import numpy as np
import pytest

from app.vector_index import ensure_index, nearest, upsert_embedding


class FakeRedis8:
    def __init__(self):
        self.commands = []
        self.hashes = {}

    def execute_command(self, *command):
        self.commands.append(command)
        if command[0] == "FT.INFO":
            raise RuntimeError("Unknown index name")
        if command[0] == "FT.SEARCH":
            return [
                1, b"acoustic:embedding:7",
                [b"word", b"bird", b"locale", b"en-US",
                 b"distance", b"0.12"],
            ]
        return b"OK"

    def hset(self, key, mapping):
        self.hashes[key] = mapping


def test_vector_index_create_upsert_and_query():
    client = FakeRedis8()
    ensure_index(client)
    key = upsert_embedding(
        client, 7, np.zeros(26), word="Bird", locale="en-US", split="train"
    )
    results = nearest(client, np.ones(26), locale="en-US", k=1)

    assert key == "acoustic:embedding:7"
    assert len(client.hashes[key]["vector"]) == 26 * 4
    assert client.hashes[key]["word"] == "bird"
    assert results == [{
        "word": "bird", "locale": "en-US", "distance": 0.12,
        "key": "acoustic:embedding:7",
    }]
    search = next(command for command in client.commands
                  if command[0] == "FT.SEARCH")
    assert "@locale:{en\\-US}" in search[2]


def test_vector_shape_is_checked():
    with pytest.raises(ValueError):
        upsert_embedding(
            FakeRedis8(), 1, np.zeros(25),
            word="bird", locale="en-US", split="train",
        )


def test_nearest_rejects_unbounded_k():
    with pytest.raises(ValueError, match="between 1 and 1000"):
        nearest(FakeRedis8(), np.zeros(26), locale="en-US", k=0)


def test_nearest_clamps_float32_cosine_roundoff():
    client = FakeRedis8()
    original = client.execute_command

    def response(*command):
        if command[0] == "FT.SEARCH":
            return [
                1, b"acoustic:embedding:1",
                [b"word", b"bird", b"locale", b"en-US",
                 b"distance", b"-0.000000119"],
            ]
        return original(*command)

    client.execute_command = response
    assert nearest(
        client, np.zeros(26), locale="en-US", k=1
    )[0]["distance"] == 0.0
