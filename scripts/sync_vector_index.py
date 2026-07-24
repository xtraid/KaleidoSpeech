"""Populate and inspect the Redis 8 acoustic HNSW index."""

from __future__ import annotations

import argparse
import json

from app.redis_bus import client, probe_capabilities
from app.vector_sync import query_recording_neighbors, sync_recording_embeddings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--locale", default="en-US")
    parser.add_argument("--query-recording-id", type=int)
    parser.add_argument("-k", type=int, default=5)
    arguments = parser.parse_args()

    capabilities = probe_capabilities(redis_client=client)
    if not capabilities.reachable or not capabilities.search:
        raise SystemExit("Redis 8 with Search is required; run scripts.check_redis")

    report = sync_recording_embeddings(client, locale=arguments.locale)
    payload: dict = {"sync": report.as_dict()}
    if arguments.query_recording_id is not None:
        payload["neighbors"] = query_recording_neighbors(
            client,
            arguments.query_recording_id,
            locale=arguments.locale,
            k=arguments.k,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
