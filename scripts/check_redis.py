"""Verify that the configured Redis runtime exposes v3 capabilities."""

import json

from app.redis_bus import probe_capabilities


def main() -> None:
    capabilities = probe_capabilities()
    print(json.dumps(capabilities.as_dict(), indent=2, sort_keys=True))
    if not capabilities.reachable:
        raise SystemExit("Redis is unreachable")
    if not capabilities.search:
        raise SystemExit(
            "Redis Search is missing. Use Redis Open Source 8, not a core-only "
            "Redis/Valkey package."
        )


if __name__ == "__main__":
    main()
