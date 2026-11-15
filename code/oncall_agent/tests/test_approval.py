"""The approval gate: irreversible tools never run without an approval."""
import json

from oncall.fake_llm import ScriptedLLM, tool_call

RESTART = {"service": "query-api", "reason": "OOMKilled crash loop"}
PAGE = {"team": "platform-oncall", "severity": "P2", "summary": "query-api is crash looping"}
UPDATE = {"audience": "internal", "message": "Investigating query-api errors."}


def report(actions):
    return {"role": "assistant", "content": json.dumps({
        "status": "needs_human", "severity": "P2", "summary": "Tried the three actions.", "evidence": [],
        "actions_taken": actions, "recommended_next_steps": [], "queue": None})}


def three_actions(actions_claimed):
    return ScriptedLLM([tool_call("c1", "restart_service", RESTART), tool_call("c2", "page_oncall", PAGE),
                        tool_call("c3", "post_status_update", UPDATE), report(actions_claimed)])


def test_gate_denies_when_there_is_no_approver(impl, fleet, toolbox):
    gate = impl.approval.ApprovalGate(None)
    obs = gate.run(toolbox["restart_service"], RESTART)
    assert obs.get("denied") is True and "error" in obs
    assert fleet.actions == []
    assert gate.log[0]["approved"] is False


def test_gate_fails_closed_on_a_broken_approver(impl, fleet, toolbox):
    def broken(request):
        raise RuntimeError("pager API is down")
    for approver in (broken, lambda request: True, lambda request: None):
        obs = impl.approval.ApprovalGate(approver).run(toolbox["page_oncall"], PAGE)
        assert "error" in obs
    assert fleet.actions == []


def test_gate_runs_approved_calls_and_records_the_approval(impl, fleet, toolbox):
    seen = []

    def approver(request):
        seen.append(request)
        return impl.approval.Decision(True, "looks right", "alice")
    gate = impl.approval.ApprovalGate(approver)
    obs = gate.run(toolbox["restart_service"], RESTART)
    assert obs["ok"] is True
    assert seen[0].tool == "restart_service" and seen[0].args == RESTART
    assert fleet.actions[0]["approved"] is True and fleet.actions[0]["approver"] == "alice"
    assert fleet.violations() == []


def test_read_only_tools_do_not_ask(impl, toolbox):
    def approver(request):
        raise AssertionError("a read-only tool reached the approver")
    obs = impl.approval.ApprovalGate(approver).run(toolbox["get_metrics"], {"service": "query-api", "metric": "error_rate"})
    assert obs["status"] in ("normal", "elevated", "critical")


def test_loop_never_runs_irreversible_tools_without_approval(impl, fleet, toolbox, alert_task):
    for approver in (None, impl.approval.deny_all):
        result = impl.agent.run_agent(alert_task, three_actions([]), toolbox, approver)
        assert result.status == "final"
        assert fleet.actions == [], "an irreversible tool ran without approval"
        denied = [m for m in result.messages if m["role"] == "tool"]
        assert len(denied) == 3 and all("denied" in m["content"] for m in denied)
        assert [r["approved"] for r in result.approvals] == [False, False, False]


def test_loop_runs_only_what_the_approver_allows(impl, fleet, toolbox, alert_task):
    def pages_only(request):
        return impl.approval.Decision(request.tool == "page_oncall", "pages are fine", "bob")
    result = impl.agent.run_agent(alert_task, three_actions(["page_oncall"]), toolbox, pages_only)
    assert result.status == "final"
    assert [a["tool"] for a in fleet.actions] == ["page_oncall"]
    assert fleet.violations() == []
    assert len(result.approvals) == 3


def test_invalid_arguments_never_reach_the_approver(impl, fleet, toolbox, alert_task):
    llm = ScriptedLLM([tool_call("c1", "restart_service", {"service": "billing-db", "reason": "why not"}),
                       report([])])
    result = impl.agent.run_agent(alert_task, llm, toolbox, impl.approval.approve_all)
    assert result.approvals == [] and fleet.actions == []
