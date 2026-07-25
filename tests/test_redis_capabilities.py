from redis.exceptions import ConnectionError

from app.redis_bus import probe_capabilities


class Redis8Double:
    def ping(self):
        return True

    def info(self, section):
        assert section == "server"
        return {"redis_version": "8.8.0"}

    def execute_command(self, *command):
        assert command[:2] == ("COMMAND", "INFO")
        if command[2] in {"FT.SEARCH", "VSIM"}:
            return [[command[2].lower()]]
        return [None]


class CoreOnlyDouble(Redis8Double):
    def execute_command(self, *command):
        return [None]


class OfflineDouble:
    def ping(self):
        raise ConnectionError("connection refused")


def test_probe_detects_redis_8_search_and_vector_sets():
    capabilities = probe_capabilities(redis_client=Redis8Double())
    assert capabilities.as_dict() == {
        "reachable": True,
        "version": "8.8.0",
        "search": True,
        "vector_set": True,
        "error": None,
    }


def test_probe_distinguishes_reachable_core_only_runtime():
    capabilities = probe_capabilities(redis_client=CoreOnlyDouble())
    assert capabilities.reachable is True
    assert capabilities.search is False
    assert capabilities.vector_set is False


def test_probe_reports_connection_failure_without_raising():
    capabilities = probe_capabilities(redis_client=OfflineDouble())
    assert capabilities.reachable is False
    assert capabilities.error == "ConnectionError: connection refused"
