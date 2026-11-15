"""Shared code that is already written: simulator, tools, fallbacks. Passes for both implementations."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from pydantic import ValidationError

from oncall import adapters
from oncall.schemas import Triage
from oncall.simulator import Approval, Fleet
from oncall.tools import IRREVERSIBLE, READ_ONLY, build_toolbox

ALERT = ("[ALERTS]\n2026-11-14T09:20:00Z [CRITICAL] Monitor 'PodCrashLooping' triggered on service:query-api "
         "(tier:1) - 14 restarts in 15m, reason=OOMKilled\n2026-11-14T09:21:00Z [CRITICAL] Monitor 'HighErrorRate' "
         "triggered on service:query-api (tier:1) - HTTP 5xx rate 22% > 5% for 5m")


def test_simulator_is_seeded_and_restart_clears_a_fixable_fault():
    a, b = Fleet(1, {"query-api": "memory_leak"}), Fleet(1, {"query-api": "memory_leak"})
    assert a.get_metrics("query-api", "memory_pct") == b.get_metrics("query-api", "memory_pct")
    assert a.get_logs("query-api") == b.get_logs("query-api")
    assert a.get_metrics("query-api", "memory_pct")["status"] == "critical"
    assert any("OOMKilled" in l for l in a.get_logs("query-api")["lines"])
    a.restart_service("query-api", "test", approval=Approval(True, "test"))
    assert a.get_metrics("query-api", "memory_pct")["status"] == "normal"
    assert a.violations() == []


def test_tool_schemas_are_strict():
    toolbox = build_toolbox(Fleet())
    assert set(toolbox) == set(READ_ONLY) | set(IRREVERSIBLE)
    assert {n for n, t in toolbox.items() if t.irreversible} == set(IRREVERSIBLE)
    schema = toolbox["restart_service"].schema()["function"]
    assert schema["parameters"]["additionalProperties"] is False
    with pytest.raises(ValidationError):
        toolbox["restart_service"].validate({"service": "not-a-service", "reason": "because"})
    with pytest.raises(ValidationError):
        toolbox["get_logs"].validate({"service": "query-api", "limit": 500})


def test_fallbacks_are_labelled_and_fit_the_schemas():
    b = adapters.Backends("fallback")
    route = b.router.route("We were charged twice on our invoice, please refund")
    assert route["backend"].startswith("fallback:") and route["queue"] == "billing" and not route["abstain"]
    assert b.router.route("hmm")["abstain"] is True
    triage = b.triage.triage(ALERT)
    assert triage["backend"].startswith("fallback:")
    Triage.model_validate({k: v for k, v in triage.items() if k not in ("backend", "fallback_reason")})
    assert (triage["severity"], triage["category"], triage["affected_service"]) == ("P2", "compute", "query-api")
    hits = b.runbooks.search("service OOMKilled crash loop", "internal")
    assert hits["backend"].startswith("fallback:") and hits["results"][0]["id"] == "rb:oom-crashloop"


def test_auto_mode_falls_back_and_says_why(monkeypatch):
    for var in ("TRIAGE_BASE_URL", "RAG_URL", "RUNBOOK_BACKEND"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(adapters, "ROUTER_DIR", adapters.CODE_DIR / "no_such_project")
    b = adapters.Backends("auto")
    assert "fallback_reason" in b.router.route("login fails with SSO")
    assert "TRIAGE_BASE_URL" in b.triage.triage(ALERT)["fallback_reason"]
    assert b.runbooks.search("disk full", "internal")["backend"] == "fallback:local-runbooks"
    with pytest.raises(adapters.BackendUnavailable):
        adapters.TriageAdapter("real")


@pytest.mark.skipif(not (adapters.TRIAGE_DIR / "inference.py").exists(), reason="Project 6 code not present")
def test_triage_adapter_runs_project6_pipeline_against_a_stub_endpoint(monkeypatch):
    """Project 6's real inference.triage() (validate, business rules, retry) with a canned model reply.
    The first reply breaks a business rule, so the pipeline must retry."""
    good = {"severity": "P2", "category": "compute", "affected_service": "query-api",
            "probable_root_cause": "query-api pods exceed their memory limit and are OOMKilled.",
            "recommended_next_steps": ["Restart query-api"], "escalate_to": "platform-oncall", "confidence": 0.9}
    replies = [json.dumps({**good, "affected_service": "payments-api"}), json.dumps(good)]
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps({"choices": [{"message": {"role": "assistant", "content": replies[len(calls) - 1]}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("TRIAGE_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
        out = adapters.TriageAdapter("real").triage(ALERT)
    finally:
        server.shutdown()
    assert out["backend"] == "project6:inference.triage"
    assert out["affected_service"] == "query-api" and len(calls) == 2
    assert "rejected" in calls[1]["messages"][-1]["content"]
