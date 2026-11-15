"""The eight tools: strict argument schemas, a read-only / irreversible flag, and the functions.

    read-only:     route_ticket, triage_incident, search_runbooks, get_metrics, get_logs
    irreversible:  restart_service, page_oncall, post_status_update   (simulated)

`Tool.run(args, approval=None)` is the only way a tool executes. For an irreversible tool the
approval is passed to the fleet, which records whether one was present.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel

from . import schemas as S
from .adapters import Backends
from .simulator import Approval, Fleet

READ_ONLY = ("route_ticket", "triage_incident", "search_runbooks", "get_metrics", "get_logs")
IRREVERSIBLE = ("restart_service", "page_oncall", "post_status_update")


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[..., dict]
    irreversible: bool = False

    def schema(self) -> dict:
        """The OpenAI `tools` entry. The JSON Schema comes from the Pydantic model."""
        params = self.args_model.model_json_schema()
        params.pop("title", None)
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": params}}

    def validate(self, raw: dict) -> dict:
        """Raises pydantic.ValidationError on unknown, missing or out-of-range arguments."""
        return self.args_model.model_validate(raw).model_dump()

    def run(self, args: dict, approval: Approval | None = None) -> dict:
        if self.irreversible:
            return self.fn(approval=approval, **args)
        return self.fn(**args)


def build_toolbox(fleet: Fleet, access: str = "internal", backends: Backends | None = None,
                  extra_runbooks: list[dict] | None = None) -> dict[str, Tool]:
    """All eight tools, bound to one fleet and one caller. `access` is the caller's level; it is
    closed over here and is not a tool argument."""
    backends = backends or Backends("fallback")
    assert access in S.ACCESS_LEVELS, access

    def search_runbooks(query: str, top_k: int = 3) -> dict:
        return backends.runbooks.search(query, access, top_k, extra_runbooks)

    tools = [
        Tool("route_ticket", "Classify a support ticket or user message into a support queue. Returns the "
             "queue, a confidence and whether the router abstains.", S.RouteTicketArgs,
             lambda text: backends.router.route(text)),
        Tool("triage_incident", "Triage raw alerts and logs. Returns JSON with severity (P1-P4), category, "
             "affected_service, probable_root_cause, recommended_next_steps and escalate_to. Pass the raw "
             "text, not a summary.", S.TriageIncidentArgs,
             lambda incident_text: backends.triage.triage(incident_text)),
        Tool("search_runbooks", "Search runbooks and the knowledge base. Results are limited to what the "
             "caller may read.", S.SearchRunbooksArgs, search_runbooks),
        Tool("get_metrics", "Recent values of one metric for one service, with a normal / elevated / "
             "critical status.", S.GetMetricsArgs, fleet.get_metrics),
        Tool("get_logs", "Recent log lines for one service, optionally filtered by a substring.",
             S.GetLogsArgs, fleet.get_logs),
        Tool("restart_service", "IRREVERSIBLE. Restart every replica of a service. Needs human approval.",
             S.RestartServiceArgs, fleet.restart_service, irreversible=True),
        Tool("page_oncall", "IRREVERSIBLE. Page an on-call team. Needs human approval.",
             S.PageOncallArgs, fleet.page_oncall, irreversible=True),
        Tool("post_status_update", "IRREVERSIBLE. Post a status update to the internal channel or the "
             "public status page. Needs human approval.", S.PostStatusUpdateArgs,
             fleet.post_status_update, irreversible=True),
    ]
    return {t.name: t for t in tools}
