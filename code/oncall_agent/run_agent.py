#!/usr/bin/env python3
"""Work one incident, then print its trace.

    python run_agent.py --scenario N01                       # scripted incident, careful fake LLM, you approve
    python run_agent.py --scenario I06 --llm naive --guardrails off
    python run_agent.py --kind question --access public --text "How do I mute a monitor?"
    LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen2.5:7b python run_agent.py --scenario N04 --llm endpoint
"""
from __future__ import annotations

import argparse
import json
import sys

from oncall.adapters import Backends
from oncall.fake_llm import CarefulPolicy, NaivePolicy
from oncall.impl import load_impl
from oncall.llm import OpenAIChat
from oncall.scenarios import BY_ID, make_fleet, make_task
from oncall.simulator import Fleet
from oncall.tasks import KINDS, Task
from oncall.tools import build_toolbox


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impl", choices=["solution", "starter"], default="solution")
    ap.add_argument("--scenario", choices=sorted(BY_ID), help="one of the 30 scripted incidents")
    ap.add_argument("--text", help="free-form incident text (instead of --scenario)")
    ap.add_argument("--kind", choices=KINDS, default="alert")
    ap.add_argument("--access", choices=["public", "internal", "restricted"], default="internal")
    ap.add_argument("--llm", choices=["careful", "naive", "endpoint"], default="careful")
    ap.add_argument("--approve", choices=["ask", "all", "none"], default="ask")
    ap.add_argument("--guardrails", choices=["on", "off"], default="on")
    ap.add_argument("--backends", choices=["fallback", "auto", "real"], default="auto")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--trace", default="trace.jsonl")
    a = ap.parse_args()
    if not a.scenario and not a.text:
        ap.error("give --scenario or --text")

    impl = load_impl(a.impl)
    backends = Backends("fallback" if a.scenario and a.backends == "auto" else a.backends)
    if a.scenario:
        sc = BY_ID[a.scenario]
        task, fleet, extra = make_task(sc), make_fleet(sc), sc["runbooks"]
    else:
        task, fleet, extra = Task(a.kind, a.access, a.text, "adhoc"), Fleet(seed=0), None
    toolbox = build_toolbox(fleet, task.access, backends, extra)
    llm = {"careful": CarefulPolicy, "naive": NaivePolicy, "endpoint": OpenAIChat}[a.llm]()
    approver = {"ask": impl.approval.cli_approver, "all": impl.approval.approve_all,
                "none": impl.approval.deny_all}[a.approve]
    cfg = impl.agent.AgentConfig(max_steps=a.max_steps, guardrails=a.guardrails == "on")

    print("backends:", json.dumps(backends.describe()))
    print(task.text, "\n")
    open(a.trace, "w").close()
    try:
        result = impl.agent.run_agent(task, llm, toolbox, approver, cfg, a.trace)
    finally:
        backends.close()
    for e in result.events:
        if e["type"] == "tool":
            flag = " [IRREVERSIBLE]" if e["irreversible"] else ""
            print(f"[{e['step']:2d}] {e['tool']}({json.dumps(e['args'])[:110]}){flag}\n"
                  f"       -> {json.dumps(e['observation'])[:160]}")
    for b in result.blocked:
        print(f"blocked by {b['by']}: {b['tool']} {b['args']}")
    for q in result.quarantined:
        print(f"quarantined from {q['source']} ({', '.join(q['patterns'])}): {q['text'][:100]}")
    print(f"\nstatus={result.status} steps={result.steps} tokens={result.prompt_tokens + result.completion_tokens} "
          f"cost=USD {result.cost_usd:.5f} trace={a.trace}")
    print(json.dumps(result.report, indent=2))
    print("actions executed:", json.dumps(fleet.actions, indent=2) if fleet.actions else "none")
    return 0 if result.status == "final" else 1


if __name__ == "__main__":
    sys.exit(main())
