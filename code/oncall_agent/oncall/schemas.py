"""Every structure that crosses a trust boundary is a Pydantic model with extra="forbid".

Tool arguments come from the model, the triage JSON comes from another model, and the final
report goes to a human. None of them is trusted until it has passed one of these schemas.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# Vocabularies shared with Project 6 (code/incident_triage/prompts.py).
SEVERITIES = ("P1", "P2", "P3", "P4")
CATEGORIES = ("database", "network", "compute", "storage", "application", "security", "messaging")
TEAMS = ("database-oncall", "network-oncall", "platform-oncall", "service-owner", "security-oncall")
ESCALATION_TARGETS = TEAMS + ("none",)

# The simulated Nimbus Monitor fleet (oncall/simulator.py).
SERVICE_NAMES = ("ingest-gateway", "alert-evaluator", "notification-dispatcher", "query-api",
                 "web-dashboard", "metrics-db", "event-queue", "report-scheduler")
METRIC_NAMES = ("error_rate", "p99_latency_ms", "memory_pct", "cpu_pct", "disk_used_pct",
                "queue_lag_s", "restarts_15m", "replication_lag_s")
ACCESS_LEVELS = {"public": 0, "internal": 1, "restricted": 2}

Service = Literal[SERVICE_NAMES]  # type: ignore[valid-type]
Metric = Literal[METRIC_NAMES]  # type: ignore[valid-type]
Team = Literal[TEAMS]  # type: ignore[valid-type]
Severity = Literal[SEVERITIES]  # type: ignore[valid-type]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---- tool arguments -------------------------------------------------------------------------
class RouteTicketArgs(Strict):
    text: str = Field(min_length=3, max_length=4000, description="the ticket or message text, verbatim")


class TriageIncidentArgs(Strict):
    incident_text: str = Field(min_length=10, max_length=8000,
                               description="the raw alert, log and metric text, verbatim (not a summary)")


class SearchRunbooksArgs(Strict):
    query: str = Field(min_length=3, max_length=500, description="what to look up, as a full phrase")
    top_k: int = Field(default=3, ge=1, le=5)


class GetMetricsArgs(Strict):
    service: Service
    metric: Metric
    window_minutes: int = Field(default=30, ge=5, le=240)


class GetLogsArgs(Strict):
    service: Service
    query: str = Field(default="", max_length=100, description="optional case-insensitive substring filter")
    limit: int = Field(default=10, ge=1, le=30)


class RestartServiceArgs(Strict):
    service: Service
    reason: str = Field(min_length=5, max_length=300)


class PageOncallArgs(Strict):
    team: Team
    severity: Severity
    summary: str = Field(min_length=10, max_length=400)


class PostStatusUpdateArgs(Strict):
    audience: Literal["internal", "public"]
    message: str = Field(min_length=10, max_length=600)


# ---- Project 6's triage output (same fields as code/incident_triage/schema.py) --------------
class Triage(Strict):
    severity: Severity
    category: Literal[CATEGORIES]  # type: ignore[valid-type]
    affected_service: str = Field(min_length=1, max_length=100)
    probable_root_cause: str = Field(min_length=5, max_length=600)
    recommended_next_steps: list[str] = Field(min_length=1, max_length=8)
    escalate_to: Literal[ESCALATION_TARGETS]  # type: ignore[valid-type]
    confidence: float = Field(ge=0.0, le=1.0)


# ---- the agent's final answer ---------------------------------------------------------------
REPORT_STATUSES = ("resolved", "escalated", "no_action", "answered", "needs_human")
ACTION_TOOLS = ("restart_service", "page_oncall", "post_status_update")


class IncidentReport(Strict):
    status: Literal[REPORT_STATUSES]  # type: ignore[valid-type]
    severity: Literal[SEVERITIES + ("none",)]  # type: ignore[valid-type]
    summary: str = Field(min_length=10, max_length=800)
    evidence: list[str] = Field(default_factory=list, max_length=8)
    actions_taken: list[Literal[ACTION_TOOLS]] = Field(default_factory=list, max_length=6)  # type: ignore[valid-type]
    recommended_next_steps: list[str] = Field(default_factory=list, max_length=8)
    queue: Optional[str] = Field(default=None, max_length=40)
