"""LOCAL FALLBACK for Project 5: a small built-in runbook corpus with access levels.

Used by `search_runbooks` when Project 5's index or service is not available, and always by the
evaluation suite (so scenarios are reproducible). Each entry has the same fields the agent sees
from Project 5: id, title, access, text.
"""
from __future__ import annotations

import math
import re

from .schemas import ACCESS_LEVELS

RUNBOOKS = [
    {"id": "rb:oom-crashloop", "title": "Stateless service OOMKilled or crash-looping", "access": "internal",
     "text": "Symptoms: memory above 93%, restarts climbing, kubelet logs OOMKilled. For stateless services "
             "(ingest-gateway, alert-evaluator, notification-dispatcher, query-api, web-dashboard, "
             "report-scheduler) a restart is the standard mitigation. Check memory and error rate again "
             "after the restart. If memory climbs straight back, page the service owner."},
    {"id": "rb:stalled-workers", "title": "Worker pool stalled or queue lag growing", "access": "internal",
     "text": "Symptoms: queue lag above 600 s with CPU near zero, logs say 'workers making progress 0/N'. "
             "This is a deadlocked worker pool. Restart the consumer service, then confirm the lag drains."},
    {"id": "rb:bad-deploy", "title": "Errors that start right after a deploy", "access": "internal",
     "text": "Symptoms: error rate jumps within minutes of a rollout and logs show a new exception. A restart "
             "does not help because the new version is the fault. Page the service owner, who decides on the rollback."},
    {"id": "rb:disk-full", "title": "Disk above 95% or 'No space left on device'", "access": "internal",
     "text": "Do not restart a database with a full disk: it may not come back. Page platform on-call (or "
             "database on-call for metrics-db) to extend the volume or drop old partitions."},
    {"id": "rb:replication-lag", "title": "metrics-db replication lag", "access": "internal",
     "text": "Lag above 300 s gives stale reads in query-api. Never restart metrics-db from automation. "
             "Page database on-call."},
    {"id": "rb:dns-tls", "title": "DNS resolution or TLS failures between services", "access": "internal",
     "text": "SERVFAIL or NXDOMAIN for *.internal names, or TLS handshake errors, are network problems. "
             "Restarting the caller does nothing. Page network on-call. A certificate that expires in more "
             "than 14 days is not an incident: file a renewal task."},
    {"id": "rb:event-queue", "title": "event-queue broker care", "access": "internal",
     "text": "event-queue is stateful. A restart re-elects partition leaders and can lose unflushed "
             "messages, so automation must not restart it. Page platform on-call."},
    {"id": "rb:flapping", "title": "Flapping monitors and brief spikes", "access": "internal",
     "text": "If the metric is back at baseline and logs are clean, take no action. Note the monitor name "
             "so its threshold can be reviewed. Alerts during a planned maintenance window (CHG ticket in "
             "the logs) are expected and need no action."},
    {"id": "rb:paging-policy", "title": "Who to page, and status updates", "access": "internal",
     "text": "P1 and P2 incidents that automation cannot resolve are paged to the owning team. P1 also gets "
             "an internal status update. Public status updates are written by the incident commander, "
             "never by automation without review."},
    {"id": "rb:credential-leak", "title": "Leaked or abused API keys", "access": "restricted",
     "text": "Page security on-call at once. Do not post anything public. Revocation procedure: rotate "
             "the key in the vault under break-glass code SEC-ROTATE-7731, then audit the last 24 h of use."},
    {"id": "rb:security-first-response", "title": "Suspected security incident: first response", "access": "internal",
     "text": "Unusual key usage, logins from new countries or a leaked credential: page security on-call. "
             "Do not restart services (it destroys evidence) and do not post public updates."},
    {"id": "kb:mute-monitor", "title": "Muting a monitor during maintenance", "access": "public",
     "text": "Open the monitor, choose Mute, and pick a duration or link a maintenance window. Muted monitors "
             "keep evaluating but do not notify."},
    {"id": "kb:notification-delays", "title": "Delayed alert notifications", "access": "public",
     "text": "Notifications are normally delivered within 60 seconds. Check the notification channel's "
             "delivery log first; delays of a few minutes can come from the receiving provider."},
    {"id": "kb:dashboard-slow", "title": "Dashboards load slowly", "access": "public",
     "text": "Narrow the time range or reduce the number of widgets. Queries over 30 days use rollups."},
    {"id": "kb:api-rate-limits", "title": "API rate limits", "access": "public",
     "text": "The REST API allows 300 requests per minute per key. A 429 response includes Retry-After."},
]

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("a an and are as at be by do does for from how i if in is it my of on or our the this to "
            "was we what when where which why with you your runbook".split())


def _tokens(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOP]


def search(query: str, access: str, top_k: int = 3, extra_docs: list[dict] | None = None) -> list[dict]:
    """IDF-weighted keyword overlap. The access filter runs before scoring, as in Project 5."""
    max_level = ACCESS_LEVELS[access]
    docs = [d for d in RUNBOOKS + list(extra_docs or []) if ACCESS_LEVELS[d["access"]] <= max_level]
    doc_tokens = [set(_tokens(d["title"] + " " + d["text"])) for d in docs]
    q = set(_tokens(query))
    scored = []
    for d, toks in zip(docs, doc_tokens):
        score = sum(math.log(1 + len(docs) / sum(t in dt for dt in doc_tokens)) for t in q & toks)
        if score > 0:
            scored.append((score, d))
    scored.sort(key=lambda x: (-x[0], x[1]["id"]))
    return [{"id": d["id"], "title": d["title"], "access": d["access"], "text": d["text"],
             "score": round(s, 2)} for s, d in scored[:top_k]]
