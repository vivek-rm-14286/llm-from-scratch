"""The agent loop: limits, malformed calls, budget, final-answer validation, trace."""
import json

from oncall.fake_llm import CarefulPolicy, LoopingPolicy, ScriptedLLM, tool_call

REPORT = {"status": "no_action", "severity": "none", "summary": "Nothing to do for this incident.",
          "evidence": [], "actions_taken": [], "recommended_next_steps": [], "queue": None}


def final(**changes):
    return {"role": "assistant", "content": json.dumps({**REPORT, **changes})}


def tool_messages(result):
    return [m for m in result.messages if m["role"] == "tool"]


def test_stops_at_step_limit(impl, toolbox, alert_task):
    cfg = impl.agent.AgentConfig(max_steps=4)
    result = impl.agent.run_agent(alert_task, LoopingPolicy(), toolbox, None, cfg)
    assert result.status == "step_limit"
    assert result.steps == 4
    assert result.report is None
    assert len(tool_messages(result)) == 4


def test_stops_when_token_budget_is_spent(impl, toolbox, alert_task):
    cfg = impl.agent.AgentConfig(max_steps=50, max_total_tokens=1500)
    result = impl.agent.run_agent(alert_task, LoopingPolicy(), toolbox, None, cfg)
    assert result.status == "budget"
    assert result.steps < 50
    assert result.prompt_tokens + result.completion_tokens >= 1500


def test_malformed_calls_are_fed_back_as_errors(impl, toolbox, alert_task):
    llm = ScriptedLLM([
        tool_call("c1", "get_metrics", '{"service": "query-api", "metric": '),          # broken JSON
        tool_call("c2", "get_metric", {"service": "query-api", "metric": "error_rate"}),  # unknown tool
        tool_call("c3", "get_metrics", {"service": "payments", "metric": "error_rate"}),  # not in the enum
        tool_call("c4", "get_metrics", {"service": "query-api", "metric": "error_rate", "verbose": True}),
        tool_call("c5", "get_metrics", {"service": "query-api", "metric": "error_rate"}),
        final(),
    ])
    result = impl.agent.run_agent(alert_task, llm, toolbox, None, impl.agent.AgentConfig(max_malformed=5))
    assert result.status == "final"
    tools = tool_messages(result)
    assert [m["tool_call_id"] for m in tools] == ["c1", "c2", "c3", "c4", "c5"]
    for m in tools[:4]:
        assert '"error"' in m["content"], m["content"]
    assert '"error"' not in tools[4]["content"] and '"status"' in tools[4]["content"]
    # the model saw each error on its next turn
    assert '"error"' in llm.seen[1][-1]["content"]


def test_gives_up_after_too_many_malformed_calls(impl, toolbox, alert_task):
    llm = ScriptedLLM([tool_call(f"c{i}", "no_such_tool", {}) for i in range(10)])
    result = impl.agent.run_agent(alert_task, llm, toolbox, None, impl.agent.AgentConfig(max_malformed=2))
    assert result.status == "malformed_limit"
    assert result.steps == 3


def test_invalid_final_answer_is_rejected_and_retried(impl, toolbox, alert_task):
    llm = ScriptedLLM([{"role": "assistant", "content": "All good, nothing to do!"},
                       final(status="fixed"),                       # not in the enum
                       final(actions_taken=["restart_service"]),    # claims an action that never ran
                       final()])
    result = impl.agent.run_agent(alert_task, llm, toolbox, None, impl.agent.AgentConfig(max_malformed=5))
    assert result.status == "final" and result.steps == 4
    assert result.report["status"] == "no_action"
    feedback = [m["content"] for m in result.messages if m["role"] == "user"][1:]
    assert len(feedback) == 3 and "restart_service" in feedback[2]


def test_trace_has_every_step(impl, toolbox, alert_task, tmp_path):
    path = tmp_path / "trace.jsonl"
    result = impl.agent.run_agent(alert_task, CarefulPolicy(slip_rate=0), toolbox,
                                  impl.approval.approve_all, trace_path=str(path))
    events = [json.loads(l) for l in path.read_text().splitlines()]
    assert events == json.loads(json.dumps(result.events))
    llm_events = [e for e in events if e["type"] == "llm"]
    tool_events = [e for e in events if e["type"] == "tool"]
    assert len(llm_events) == result.steps
    for e in llm_events:
        assert {"step", "prompt", "response", "latency_ms", "prompt_tokens", "completion_tokens"} <= e.keys()
    for e in tool_events:
        assert {"step", "tool", "args", "observation", "latency_ms", "irreversible"} <= e.keys()
    assert llm_events[0]["prompt"][0]["role"] == "system"
    assert events[-1]["type"] == "end" and events[-1]["status"] == "final"
    assert events[-1]["prompt_tokens"] == sum(e["prompt_tokens"] for e in llm_events) == result.prompt_tokens
    assert result.cost_usd > 0


def test_careful_policy_resolves_a_normal_incident(impl, fleet, toolbox, alert_task):
    result = impl.agent.run_agent(alert_task, CarefulPolicy(slip_rate=0), toolbox, impl.approval.approve_all)
    assert result.status == "final" and result.report["status"] == "resolved"
    assert [a["tool"] for a in fleet.actions] == ["restart_service"]
    assert fleet.get_metrics("query-api", "memory_pct")["status"] == "normal"
