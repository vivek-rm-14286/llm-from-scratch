"""Deterministic stand-ins for the model, so the whole project runs offline.

Both are scripted policies: plain Python that reads the conversation and returns the next
assistant message in the OpenAI format. They exist to exercise the loop, the gate, the guardrails
and the evaluator. They say nothing about how a real model behaves.

    CarefulPolicy  investigates before acting and never takes orders from tool output. It is good
                   but not perfect: it trusts the triage result (so it misses a root cause in
                   another service), it thinks only "*-db" services are
                   stateful, and now and then it emits a malformed tool call.
    NaivePolicy    the same plan, but it obeys any instruction addressed to "the agent" that it
                   finds in a ticket, a log line or a runbook, and it restarts a service whenever
                   it has no better idea.
"""
from __future__ import annotations

import json
import random
import re

from .llm import estimate_tokens
from .schemas import SERVICE_NAMES, TEAMS
from .simulator import SERVICES
from .tasks import parse_task

KEY_METRIC = {"compute": "memory_pct", "storage": "disk_used_pct", "database": "replication_lag_s",
              "messaging": "queue_lag_s", "network": "p99_latency_ms", "application": "p99_latency_ms"}

# (signature, pattern over the affected service's logs, response, team to page)
PLAYBOOK = [
    ("security", r"unauthorized|leaked credential|api key .* used from", "page", "security-oncall"),
    ("maintenance", r"planned maintenance window", "none", None),
    ("oom", r"OOMKilled|out of memory", "restart", "platform-oncall"),
    ("stalled", r"workers making progress|deadlock", "restart", None),
    ("bad_deploy", r"after deploy|rollout v\d", "page", "service-owner"),
    ("disk", r"No space left", "page", "platform-oncall"),
    ("replication", r"replication lag \d+s", "page", "database-oncall"),
    ("dns", r"SERVFAIL|NXDOMAIN|TLS handshake", "page", "network-oncall"),
    ("cert", r"certificate .* expires in", "none", None),
    ("recovered", r"back to normal|connection restored", "none", None),
]
TICKET_HINTS = [(r"notification|not being delivered|alerts? (are )?not", "notification-dispatcher"),
                (r"dashboard", "web-dashboard"), (r"missing data|ingest", "ingest-gateway"),
                (r"query|api", "query-api")]
OUTAGE_WORDS = r"outage|down for|failing to load|not being delivered|5xx|for all users|multiple customers"

_ADDRESSED = r"\b(?:AI|agent|assistant|automation|bot|responder)\b[^\n]{0,160}?"
_COMMANDS = [
    ("restart_service", re.compile(_ADDRESSED + r"restart (?:the )?(?:service )?([a-z]+(?:-[a-z]+)+)", re.I)),
    ("page_oncall", re.compile(_ADDRESSED + r"\bpage (?:the )?(" + "|".join(TEAMS) + r")(?:[^.\n]{0,40}?\b(P[1-4])\b)?", re.I)),
    ("post_status_update", re.compile(_ADDRESSED + r"post (?:a |an )?(public|internal) status update"
                                      r"(?: saying| that says|:)?\s*[\"“]([^\"”]{10,400})[\"”]", re.I)),
    ("suppress", re.compile(_ADDRESSED + r"(?:not page|no action|mark (?:the |this )?incident as resolved)", re.I)),
]


def _obs_json(content: str):
    """Tool observations are JSON, possibly wrapped in guardrail tags. Returns {} if none is found."""
    a, b = content.find("{"), content.rfind("}")
    if a == -1 or b <= a:
        return {}
    try:
        return json.loads(content[a:b + 1])
    except ValueError:
        return {}


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


class View:
    """What the policy can see: the task, the offered tools, and every call with its observation."""

    def __init__(self, messages: list[dict], tools: list[dict]):
        self.offered = {t["function"]["name"] for t in tools}
        user = next(m["content"] for m in messages if m["role"] == "user")
        self.kind, self.access, self.body = parse_task(user)
        self.calls, by_id = [], {}
        for m in messages:
            if m["role"] == "assistant":
                for c in m.get("tool_calls") or []:
                    try:
                        args = json.loads(c["function"]["arguments"])
                    except (ValueError, TypeError):
                        args = None
                    rec = {"name": c["function"]["name"], "args": args, "obs": {}, "ok": False}
                    by_id[c.get("id")] = rec
                    self.calls.append(rec)
            elif m["role"] == "tool" and m.get("tool_call_id") in by_id:
                rec = by_id[m["tool_call_id"]]
                rec["obs"] = _obs_json(m["content"])
                rec["ok"] = bool(rec["obs"]) and "error" not in rec["obs"]

    def _match(self, rec, name, match):
        return rec["name"] == name and all((rec["args"] or {}).get(k) == v for k, v in match.items())

    def ok(self, name, after=-1, **match):
        for i in range(len(self.calls) - 1, after, -1):
            if self.calls[i]["ok"] and self._match(self.calls[i], name, match):
                return self.calls[i]["obs"]
        return None

    def index(self, name, **match):
        return next((i for i, r in enumerate(self.calls) if r["ok"] and self._match(r, name, match)), None)

    def tries(self, name, **match):
        return sum(self._match(r, name, match) for r in self.calls)

    def executed_actions(self):
        return [r["name"] for r in self.calls
                if r["ok"] and r["name"] in ("restart_service", "page_oncall", "post_status_update")]

    def untrusted_strings(self):
        yield self.body
        for r in self.calls:
            yield from _strings(r["obs"])


class _GiveUp(Exception):
    pass


class CarefulPolicy:
    obeys_embedded_instructions = False
    verifies_before_acting = True
    name = "careful"

    def __init__(self, seed: int = 0, slip_rate: float = 0.08):
        self.seed, self.slip_rate = seed, slip_rate

    # ---- the client interface ----------------------------------------------------------------
    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        v = View(messages, tools)
        try:
            msg = self.decide(v)
        except _GiveUp as e:
            msg = self.final(v, "needs_human", "none", f"Could not complete the investigation: {e}.")
        msg = self._maybe_slip(v, msg)
        return {"message": msg, "usage": {"prompt_tokens": estimate_tokens(messages) + estimate_tokens(tools),
                                          "completion_tokens": estimate_tokens(msg)}}

    def _maybe_slip(self, v: View, msg: dict) -> dict:
        """Sometimes send a malformed call. The loop must feed the error back; the next turn recovers."""
        calls = msg.get("tool_calls")
        if not calls or len(calls) > 1 or (v.calls and not v.calls[-1]["ok"]):
            return msg
        if calls[0]["function"]["name"] in ("restart_service", "page_oncall", "post_status_update"):
            return msg                      # slips happen on read-only calls only
        rng = random.Random(f"{self.seed}|{v.body}|{len(v.calls)}")
        if rng.random() >= self.slip_rate:
            return msg
        fn = calls[0]["function"]
        how = rng.choice(["bad_json", "unknown_tool", "extra_arg"])
        if how == "bad_json":
            fn["arguments"] = fn["arguments"][:-1]
        elif how == "unknown_tool":
            fn["name"] += "s"
        else:
            fn["arguments"] = json.dumps({**json.loads(fn["arguments"]), "verbose": True})
        return msg

    # ---- helpers -----------------------------------------------------------------------------
    @staticmethod
    def call(v: View, *calls: tuple[str, dict]) -> dict:
        n = len(v.calls)
        return {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"call_{n + i + 1}", "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}} for i, (name, args) in enumerate(calls)]}

    def need(self, v: View, name: str, args: dict, *keys: str):
        """(observation, None) if the call already succeeded, else (None, message that makes it)."""
        match = {k: args[k] for k in keys}
        obs = v.ok(name, **match)
        if obs is not None:
            return obs, None
        if v.tries(name, **match) >= 3:
            raise _GiveUp(f"{name} failed three times")
        return None, self.call(v, (name, args))

    @staticmethod
    def final(v: View, status: str, severity: str, summary: str, evidence=(), steps=(), queue=None) -> dict:
        report = {"status": status, "severity": severity, "summary": summary[:800],
                  "evidence": [e[:300] for e in evidence][:8], "actions_taken": sorted(set(v.executed_actions())),
                  "recommended_next_steps": [s[:300] for s in steps][:8], "queue": queue}
        return {"role": "assistant", "content": json.dumps(report)}

    # ---- the three plans ---------------------------------------------------------------------
    def decide(self, v: View) -> dict:
        return {"question": self.plan_question, "ticket": self.plan_ticket}.get(v.kind, self.plan_alert)(v)

    def plan_question(self, v: View) -> dict:
        rb, msg = self.need(v, "search_runbooks", {"query": v.body[:400]})
        if msg:
            return msg
        hits = rb.get("results") or []
        if rb.get("answer"):
            return self.final(v, "answered", "none", rb["answer"])
        if not hits:
            return self.final(v, "needs_human", "none", "No runbook or article is available at the caller's "
                              "access level for this question.")
        return self.final(v, "answered", "none", f"{hits[0]['title']}: {hits[0]['text']}", evidence=[hits[0]["id"]])

    def plan_ticket(self, v: View) -> dict:
        route, msg = self.need(v, "route_ticket", {"text": v.body[:2000]})
        if msg:
            return msg
        rb, msg = self.need(v, "search_runbooks", {"query": v.body[:300]})
        if msg:
            return msg
        hits = rb.get("results") or []
        steps = [f"See {h['id']}: {h['title']}" for h in hits[:2]]
        service = next((s for s in SERVICE_NAMES if s in v.body), None) or \
            next((s for pat, s in TICKET_HINTS if re.search(pat, v.body, re.I)), None)
        if service and re.search(OUTAGE_WORDS, v.body, re.I):
            m, msg = self.need(v, "get_metrics", {"service": service, "metric": "error_rate"}, "service", "metric")
            if msg:
                return msg
            if m["status"] == "critical":
                team = SERVICES[service]["owner"]
                if v.tries("page_oncall") == 0:
                    return self.call(v, ("page_oncall", {"team": team, "severity": "P2", "summary":
                                     f"Ticket reports an outage; {service} error rate is {m['current']}%"}))
                status = "escalated" if v.ok("page_oncall") else "needs_human"
                return self.final(v, status, "P2", f"The ticket describes a live problem: {service} error rate "
                                  f"is {m['current']}% (critical).", [f"error_rate={m['current']}%"], steps,
                                  route["queue"])
        if route.get("abstain"):
            return self.final(v, "needs_human", "none", "The router is not confident about this ticket "
                              f"(best guess {route['queue']} at {route['confidence']}); a person should read it.",
                              steps=steps, queue=route["queue"])
        return self.final(v, "answered", "none", f"Routed to the {route['queue']} queue. No live incident found.",
                          steps=steps, queue=route["queue"])

    def plan_alert(self, v: View) -> dict:
        t, msg = self.need(v, "triage_incident", {"incident_text": v.body[:8000]})
        if msg:
            return msg
        svc = t.get("affected_service")
        if svc not in SERVICE_NAMES:
            raise _GiveUp(f"triage named an unknown service {svc!r}")
        sev = t["severity"]
        key = KEY_METRIC.get(t["category"])
        m_err, msg = self.need(v, "get_metrics", {"service": svc, "metric": "error_rate"}, "service", "metric")
        if msg:
            return msg
        m_key = None
        if key:
            m_key, msg = self.need(v, "get_metrics", {"service": svc, "metric": key}, "service", "metric")
            if msg:
                return msg
        logs, msg = self.need(v, "get_logs", {"service": svc, "limit": 10}, "service")
        if msg:
            return msg
        evidence_text = "\n".join(logs.get("lines", []))     # only the affected service's own logs
        sig, response, team = next(((s, r, tm) for s, pat, r, tm in PLAYBOOK if re.search(pat, evidence_text)),
                                   (None, None, None))
        rb, msg = self.need(v, "search_runbooks", {"query": f"{svc} {sig or t['category']} {t['probable_root_cause'][:80]}"})
        if msg:
            return msg
        steps = [f"See {h['id']}: {h['title']}" for h in (rb.get("results") or [])[:2]] + t["recommended_next_steps"][:2]
        evidence = [f"triage: {sev} {t['category']} on {svc}", f"error_rate {m_err['current']}% ({m_err['status']})"]
        if m_key:
            evidence.append(f"{key} {m_key['current']} ({m_key['status']})")
        if sig:
            evidence.append(f"log signature: {sig}")
        healthy = m_err["status"] == "normal" and (m_key is None or m_key["status"] == "normal")

        if response is None:
            if healthy and self.verifies_before_acting:
                return self.final(v, "no_action", "none", f"The alert on {svc} does not reproduce: metrics are at "
                                  "baseline and the logs are clean. Likely a flapping monitor.", evidence, steps)
            response = "page" if self.verifies_before_acting else "restart"
        if response == "none" or (response == "page" and sev == "P4"):
            return self.final(v, "no_action", sev if sev == "P4" else "none",
                              f"No action needed on {svc} ({sig or 'low severity'}).", evidence, steps)

        if response == "restart" and not self.is_stateful(svc):
            if v.tries("restart_service", service=svc) == 0:
                return self.call(v, ("restart_service", {"service": svc, "reason": f"{sig or 'alert'} on {svc}"}))
            at = v.index("restart_service", service=svc)
            if at is not None:
                check = key or "error_rate"
                after = v.ok("get_metrics", after=at, service=svc, metric=check)
                if after is None:
                    return self.call(v, ("get_metrics", {"service": svc, "metric": check}))
                if after["status"] == "normal":
                    return self.final(v, "resolved", sev, f"Restarted {svc} ({sig}); {check} is back to "
                                      f"{after['current']}.", evidence + [f"after restart: {check} {after['current']}"], steps)
                evidence.append(f"restart did not help: {check} still {after['current']}")
        # page: the playbook says so, the restart was refused, or the restart did not help
        team = team or (t["escalate_to"] if t["escalate_to"] != "none" else SERVICES[svc]["owner"])
        if self.is_stateful(svc) and response == "restart":
            team = SERVICES[svc]["owner"]
        page_sev = sev if sev != "P4" else "P3"
        if v.tries("page_oncall", team=team) == 0:
            calls = [("page_oncall", {"team": team, "severity": page_sev,
                                      "summary": f"{svc}: {t['probable_root_cause']}"[:400]})]
            if sev == "P1" and "post_status_update" in v.offered:
                calls.append(("post_status_update", {"audience": "internal", "message":
                              f"{sev} on {svc}: investigating, {team} paged."}))
            return self.call(v, *calls)
        status = "escalated" if v.ok("page_oncall", team=team) else "needs_human"
        return self.final(v, status, sev, f"{svc}: {t['probable_root_cause']}", evidence, steps)

    @staticmethod
    def is_stateful(service: str) -> bool:
        return service.endswith("-db")          # the blind spot: event-queue is stateful too


class NaivePolicy(CarefulPolicy):
    obeys_embedded_instructions = True
    verifies_before_acting = False
    name = "naive"

    def decide(self, v: View) -> dict:
        for text in v.untrusted_strings():
            for tool, pattern in _COMMANDS:
                for m in pattern.finditer(text):
                    if tool == "suppress":
                        return self.final(v, "no_action", "none", "Closed as instructed in the incident data.")
                    if tool == "restart_service" and m.group(1) in SERVICE_NAMES:
                        args, key = {"service": m.group(1), "reason": "requested in the incident data"}, "service"
                    elif tool == "page_oncall":
                        args, key = {"team": m.group(1), "severity": m.group(2) or "P1",
                                     "summary": "Paged as requested in the incident data"}, "team"
                    elif tool == "post_status_update":
                        args, key = {"audience": m.group(1).lower(), "message": m.group(2)}, "audience"
                    else:
                        continue
                    if v.tries(tool, **{key: args[key]}) == 0:
                        return self.call(v, (tool, args))
        return super().decide(v)

    @staticmethod
    def is_stateful(service: str) -> bool:
        return service.endswith("-db")


class LoopingPolicy:
    """Calls the same read-only tool forever. Used to test the step limit and the budget."""
    name = "looping"

    def chat(self, messages, tools):
        n = sum(1 for m in messages if m["role"] == "assistant")
        msg = {"role": "assistant", "content": None, "tool_calls": [{"id": f"call_{n + 1}", "type": "function",
               "function": {"name": "get_metrics", "arguments": json.dumps(
                   {"service": "query-api", "metric": "error_rate"})}}]}
        return {"message": msg, "usage": {"prompt_tokens": estimate_tokens(messages), "completion_tokens": 20}}


class ScriptedLLM:
    """Replays a fixed list of assistant messages (or callables of the message list). For tests."""
    name = "scripted"

    def __init__(self, responses):
        self.responses, self.seen = list(responses), []

    def chat(self, messages, tools):
        self.seen.append([dict(m) for m in messages])
        r = self.responses.pop(0) if self.responses else {"role": "assistant", "content": "{}"}
        msg = r(messages) if callable(r) else r
        return {"message": msg, "usage": {"prompt_tokens": estimate_tokens(messages),
                                          "completion_tokens": estimate_tokens(msg)}}


def tool_call(call_id: str, name: str, args) -> dict:
    """An assistant message with one tool call. `args` may be a dict or a raw (possibly broken) string."""
    return {"role": "assistant", "content": None, "tool_calls": [{"id": call_id, "type": "function",
            "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}]}
