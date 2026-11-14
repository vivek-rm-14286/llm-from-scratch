"""Shared prompt text and label vocabularies for the Incident Triage Copilot.

Standard library only, so generate_data.py runs without any pip installs.
"""

SEVERITIES = ("P1", "P2", "P3", "P4")

CATEGORIES = (
    "database",
    "network",
    "compute",
    "storage",
    "application",
    "security",
    "messaging",
)

ESCALATION_TARGETS = (
    "database-oncall",
    "network-oncall",
    "platform-oncall",
    "service-owner",
    "security-oncall",
    "none",
)

SYSTEM_PROMPT = """You are an incident triage assistant for an SRE team.
You receive raw monitoring alerts, log snippets and metrics for ONE incident.
Respond with ONLY a JSON object (no prose, no code fences) with exactly these keys:

{
  "severity": "P1" | "P2" | "P3" | "P4",
  "category": "database" | "network" | "compute" | "storage" | "application" | "security" | "messaging",
  "affected_service": "<the service where the root cause lies, exactly as named in the input>",
  "probable_root_cause": "<one or two sentences>",
  "recommended_next_steps": ["<step>", "..."],
  "escalate_to": "database-oncall" | "network-oncall" | "platform-oncall" | "service-owner" | "security-oncall" | "none",
  "confidence": <number between 0 and 1>
}

Severity rules. First decide user impact on the user-facing service showing symptoms:
- outage: error rate >= 40%, availability checks failing, or processing stopped
- major: error rate 10-40%, p99 latency >= 3x SLO, or queue lag >= 10 minutes
- minor: error rate < 10%, p99 latency < 3x SLO, or queue lag < 10 minutes
- none: early warning only, no user-facing symptoms yet
Then combine with that service's tier:
- tier1: outage=P1, major=P2, minor=P3, none=P4
- tier2: outage=P2, major=P3, minor=P4, none=P4
- tier3: outage=P3, major=P4, minor=P4, none=P4
Exception: confirmed security compromise (successful unauthorized access or leaked secret) is at least P2.

Escalation: database issues -> database-oncall; network, DNS, load balancer, TLS -> network-oncall;
compute, nodes, containers, disks -> platform-oncall; code or deploy bugs, application-level issues and
message consumers -> service-owner; security -> security-oncall. If severity is P4, escalate_to is "none".

Ignore unrelated noise alerts. Lower confidence when signals are missing or contradictory."""


def format_user_prompt(alerts: list[str], logs: list[str], metrics: list[str]) -> str:
    """Render the raw incident signals into the user message the model sees."""
    parts = ["[ALERTS]", *alerts]
    if logs:
        parts += ["", "[LOGS]", *logs]
    if metrics:
        parts += ["", "[METRICS]", *metrics]
    return "\n".join(parts)
