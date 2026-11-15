"""The MCP server exposes the read-only tools, and only those (in-memory client, no subprocess)."""
import asyncio
import json

from mcp import Client

from oncall.simulator import Fleet
from oncall.tools import IRREVERSIBLE, READ_ONLY


def with_client(impl, coro, access="internal"):
    fleet = Fleet(seed=0, faults={"query-api": "memory_leak"})
    server = impl.mcp_server.build_server(fleet, access)

    async def main():
        async with Client(server) as client:
            return await coro(client)
    return asyncio.run(main()), fleet


def payload(result):
    return json.loads(result.content[0].text)


def test_lists_exactly_the_read_only_tools(impl):
    tools, _ = with_client(impl, lambda c: c.list_tools())
    names = {t.name for t in tools.tools}
    assert names == set(READ_ONLY)
    assert not names & set(IRREVERSIBLE)
    by_name = {t.name: t for t in tools.tools}
    assert set(by_name["get_metrics"].input_schema["required"]) == {"service", "metric"}
    assert "metrics-db" in json.dumps(by_name["get_metrics"].input_schema)      # the service enum is published
    assert all(t.description for t in tools.tools)
    assert all(t.annotations and t.annotations.read_only_hint for t in tools.tools)


def test_calls_a_tool_through_the_protocol(impl):
    result, fleet = with_client(impl, lambda c: c.call_tool("get_metrics", {"service": "query-api", "metric": "memory_pct"}))
    assert not result.is_error
    assert payload(result)["status"] == "critical"
    assert fleet.actions == []


def test_action_tools_are_not_callable(impl):
    async def attempt(client):
        try:
            r = await client.call_tool("restart_service", {"service": "query-api", "reason": "testing"})
            return r.is_error
        except Exception:
            return True
    refused, fleet = with_client(impl, attempt)
    assert refused and fleet.actions == []


def test_access_level_comes_from_the_server_not_the_caller(impl):
    q = {"query": "leaked API key rotation break-glass", "top_k": 5}
    public, _ = with_client(impl, lambda c: c.call_tool("search_runbooks", q), access="public")
    restricted, _ = with_client(impl, lambda c: c.call_tool("search_runbooks", q), access="restricted")
    assert "SEC-ROTATE-7731" not in public.content[0].text
    assert "SEC-ROTATE-7731" in restricted.content[0].text
