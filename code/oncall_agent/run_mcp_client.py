#!/usr/bin/env python3
"""Start the MCP server as a subprocess (stdio) and use it from a client.

    python run_mcp_client.py                       # solution server, access level "internal"
    python run_mcp_client.py --impl starter --access public
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from mcp import Client, StdioServerParameters


async def main(impl: str, access: str) -> None:
    server = StdioServerParameters(command=sys.executable, args=["-m", f"{impl}.mcp_server"],
                                   env={**os.environ, "ONCALL_ACCESS": access, "ONCALL_BACKENDS": "fallback"})
    async with Client(server) as client:
        tools = (await client.list_tools()).tools
        print("tools:", ", ".join(t.name for t in tools))
        calls = [("get_metrics", {"service": "query-api", "metric": "error_rate"}),
                 ("search_runbooks", {"query": "leaked API key rotation", "top_k": 2}),
                 ("restart_service", {"service": "query-api", "reason": "not exposed"})]
        for name, args in calls:
            try:
                result = await client.call_tool(name, args)
                text = result.content[0].text if result.content else ""
                print(f"{name}: {'ERROR ' if result.is_error else ''}{' '.join(text.split())[:160]}")
            except Exception as e:                      # an unknown tool is refused by the server
                print(f"{name}: refused ({type(e).__name__}: {str(e)[:80]})")
        hits = json.loads((await client.call_tool("search_runbooks", {"query": "leaked API key"})).content[0].text)
        print(f"access={access}: runbook levels returned:", sorted({h["access"] for h in hits["results"]}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impl", choices=["solution", "starter"], default="solution")
    ap.add_argument("--access", choices=["public", "internal", "restricted"], default="internal")
    a = ap.parse_args()
    asyncio.run(main(a.impl, a.access))
