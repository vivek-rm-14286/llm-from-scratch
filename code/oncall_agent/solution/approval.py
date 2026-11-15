"""The human-approval gate: irreversible tools never run unless code outside the model says yes."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from oncall.simulator import Approval
from oncall.tools import Tool


@dataclass
class ApprovalRequest:
    """Exactly what will happen, shown to whoever decides."""
    tool: str
    args: dict
    task_id: str = ""
    task_kind: str = ""
    warnings: list[str] = field(default_factory=list)   # e.g. "untrusted content was quarantined in this run"


@dataclass
class Decision:
    approved: bool
    reason: str = ""
    approver: str = "policy"


Approver = Callable[[ApprovalRequest], Decision]


class ApprovalGate:
    """Every tool call goes through `run`. Read-only tools pass straight through. Irreversible
    tools need a Decision(approved=True) from the approver. The gate fails closed: no approver,
    an approver that raises, or anything that is not a Decision all mean "denied"."""

    def __init__(self, approver: Approver | None = None):
        self.approver = approver
        self.log: list[dict] = []          # every request that reached the approver, with the decision

    def run(self, tool: Tool, args: dict, request: ApprovalRequest | None = None) -> dict:
        if not tool.irreversible:
            return tool.run(args)
        request = request or ApprovalRequest(tool=tool.name, args=args)
        decision = self._decide(request)
        self.log.append({"tool": tool.name, "args": args, "approved": decision.approved,
                         "approver": decision.approver, "reason": decision.reason,
                         "warnings": list(request.warnings)})
        if not decision.approved:
            return {"error": f"denied: {decision.reason or 'not approved'}", "denied": True}
        return tool.run(args, approval=Approval(True, decision.approver, decision.reason))

    def _decide(self, request: ApprovalRequest) -> Decision:
        if self.approver is None:
            return Decision(False, "no approver is configured", "gate")
        try:
            decision = self.approver(request)
        except Exception as e:                       # a broken approver must not open the gate
            return Decision(False, f"approver failed: {type(e).__name__}: {e}", "gate")
        if not isinstance(decision, Decision):
            return Decision(False, "approver returned no decision", "gate")
        return decision


# ---- approvers ------------------------------------------------------------------------------
def deny_all(request: ApprovalRequest) -> Decision:
    return Decision(False, "this session is read-only", "deny-all")


def approve_all(request: ApprovalRequest) -> Decision:
    """A rubber stamp. It models approval fatigue; use it only to measure what the other layers stop."""
    return Decision(True, "auto-approved", "rubber-stamp")


def cli_approver(request: ApprovalRequest) -> Decision:
    """Interactive mode: show the exact call and wait for a typed 'y'."""
    print("\n=== APPROVAL NEEDED: irreversible action ===")
    print(f"  tool: {request.tool}\n  args: {json.dumps(request.args, indent=2)}")
    for w in request.warnings:
        print(f"  WARNING: {w}")
    answer = input("Run this? [y/N] ").strip().lower()
    return Decision(answer == "y", "typed y" if answer == "y" else "declined at the prompt", "human")
