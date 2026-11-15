"""Thin adapters to the earlier projects, each with a clearly labelled LOCAL FALLBACK.

    route_ticket     -> Project 3  code/ticket_router   (solution.router.load_router)
    triage_incident  -> Project 6  code/incident_triage (inference.triage against a served model)
    search_runbooks  -> Project 5  code/support_rag     (POST /ask, or retrieve.Retriever in-process)

Every result carries a "backend" field that says which one answered, for example
"project3:router" or "fallback:keyword-router". Modes:

    fallback  never touch the other projects (the evaluation suite uses this: reproducible)
    auto      use the real project when it is available, else fall back and say why
    real      use the real project or raise
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

from . import runbooks
from .schemas import SERVICE_NAMES, Triage
from .simulator import SERVICES

CODE_DIR = Path(__file__).resolve().parents[2]          # .../code
ROUTER_DIR = Path(os.getenv("TICKET_ROUTER_DIR", CODE_DIR / "ticket_router"))
TRIAGE_DIR = Path(os.getenv("INCIDENT_TRIAGE_DIR", CODE_DIR / "incident_triage"))
RAG_DIR = Path(os.getenv("SUPPORT_RAG_DIR", CODE_DIR / "support_rag"))


class BackendUnavailable(RuntimeError):
    pass


# =============================================================================================
# Project 3: ticket router
# =============================================================================================
QUEUES = ("billing", "ingestion", "alerting", "agent-install", "api", "access", "other")

_ROUTER_KEYWORDS = {   # LOCAL FALLBACK vocabulary, same queue names as Project 3
    "billing": "invoice charge charged refund plan payment billing credit card price",
    "ingestion": "ingest ingestion missing dropped late metrics logs datapoints gap rejected",
    "alerting": "alert alerts monitor monitors notification notifications paging page mute "
                "maintenance delivered firing dashboard dashboards",
    "agent-install": "agent install upgrade host nimbus-agent package config yaml",
    "api": "api key keys rate limit sdk terraform webhook 429 rest",
    "access": "login sso password two-factor 2fa role roles invite invitation user permission",
}

# Project 3 and this project both have a package named `solution`, so the router runs in its own
# process (its own sys.path), and speaks JSON lines over stdin/stdout.
_ROUTER_WORKER = r"""
import json, sys
sys.path.insert(0, ".")
from solution.router import load_router
router = load_router(sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None)
print(json.dumps({"ready": True, "name": getattr(router, "name", "router")}), flush=True)
for line in sys.stdin:
    try:
        out = router.route(json.loads(line)["text"])
        print(json.dumps({"queue": out["queue"], "confidence": float(out["confidence"]),
                          "abstain": bool(out["abstain"]),
                          "probs": {k: float(v) for k, v in out["probs"].items()}}), flush=True)
    except Exception as e:
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}), flush=True)
"""


class RouterAdapter:
    def __init__(self, mode: str = "auto"):
        self.mode, self.proc, self.reason, self.name = mode, None, "mode=fallback", ""
        if mode in ("auto", "real"):
            try:
                self._start()
            except Exception as e:
                self.proc, self.reason = None, f"{type(e).__name__}: {e}"
                if mode == "real":
                    raise BackendUnavailable(f"Project 3 router: {self.reason}") from e

    def _start(self) -> None:
        if not (ROUTER_DIR / "solution" / "router.py").exists():
            raise BackendUnavailable(f"{ROUTER_DIR}/solution/router.py not found")
        python = os.getenv("TICKET_ROUTER_PYTHON", sys.executable)
        self.proc = subprocess.Popen(
            [python, "-c", _ROUTER_WORKER, os.getenv("TICKET_ROUTER_MODEL", "")], cwd=ROUTER_DIR,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        line = self.proc.stdout.readline()
        if '"ready"' not in line:
            err = (self.proc.stderr.read() or "").strip().splitlines()
            self.proc.kill()
            raise BackendUnavailable(err[-1] if err else "router worker did not start")
        self.name = json.loads(line).get("name", "router")   # which model Project 3 loaded (it has its own
                                                             # keyword baseline when nothing is trained)

    def route(self, text: str) -> dict:
        if self.proc is not None:
            self.proc.stdin.write(json.dumps({"text": text}) + "\n")
            self.proc.stdin.flush()
            out = json.loads(self.proc.stdout.readline() or '{"error": "router worker exited"}')
            if "error" not in out:
                top = dict(sorted(out["probs"].items(), key=lambda kv: -kv[1])[:3])
                return {"queue": out["queue"], "confidence": round(out["confidence"], 3),
                        "abstain": out["abstain"], "probs": {k: round(v, 3) for k, v in top.items()},
                        "backend": f"project3:router ({self.name})"}
            self.reason = out["error"]
        return {**fallback_route(text), "backend": "fallback:keyword-router", "fallback_reason": self.reason}

    def close(self) -> None:
        if self.proc is not None:
            self.proc.kill()
            self.proc = None


def fallback_route(text: str) -> dict:
    """LOCAL FALLBACK: keyword counts turned into a distribution. Abstains below 0.5."""
    words = re.findall(r"[a-z0-9-]+", text.lower())
    counts = {q: sum(w in kw.split() for w in words) for q, kw in _ROUTER_KEYWORDS.items()}
    total = sum(counts.values())
    if total == 0:
        probs = {q: round(1 / len(QUEUES), 3) for q in QUEUES}
        return {"queue": "other", "confidence": probs["other"], "abstain": True, "probs": probs}
    probs = {q: (c + 0.25) / (total + 0.25 * len(QUEUES)) for q, c in counts.items()}
    probs["other"] = 0.25 / (total + 0.25 * len(QUEUES))
    queue = max(probs, key=probs.get)
    top = dict(sorted(probs.items(), key=lambda kv: -kv[1])[:3])
    return {"queue": queue, "confidence": round(probs[queue], 3), "abstain": probs[queue] < 0.5,
            "probs": {k: round(v, 3) for k, v in top.items()}}


# =============================================================================================
# Project 6: incident triage
# =============================================================================================
class _UrllibBackend:
    """The `backend.generate(messages)` object Project 6's triage() expects, without the openai package."""

    def __init__(self, base_url: str, model: str, api_key: str):
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key

    def generate(self, messages: list[dict], max_tokens: int = 512) -> str:
        body = json.dumps({"model": self.model, "messages": messages, "temperature": 0,
                           "max_tokens": max_tokens}).encode()
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)["choices"][0]["message"]["content"] or ""


class TriageAdapter:
    """Real path: Project 6's inference.triage() (generate -> validate -> retry) against the model
    served at TRIAGE_BASE_URL. Fallback: a rule-based triager that fills the same schema."""

    def __init__(self, mode: str = "auto"):
        self.mode, self.triage_fn, self.backend, self.reason = mode, None, None, "mode=fallback"
        if mode in ("auto", "real"):
            try:
                self._load()
            except Exception as e:
                self.triage_fn, self.reason = None, f"{type(e).__name__}: {e}"
                if mode == "real":
                    raise BackendUnavailable(f"Project 6 triage: {self.reason}") from e

    def _load(self) -> None:
        base_url = os.getenv("TRIAGE_BASE_URL")
        if not base_url:
            raise BackendUnavailable("TRIAGE_BASE_URL is not set (no served triage model)")
        if not (TRIAGE_DIR / "inference.py").exists():
            raise BackendUnavailable(f"{TRIAGE_DIR}/inference.py not found")
        sys.path.insert(0, str(TRIAGE_DIR))       # Project 6 uses top-level modules: prompts, schema, inference
        try:
            import inference as p6_inference
        finally:
            sys.path.remove(str(TRIAGE_DIR))
        self.triage_fn = p6_inference.triage
        self.backend = _UrllibBackend(base_url, os.getenv("TRIAGE_MODEL", "triage"),
                                      os.getenv("TRIAGE_API_KEY", "not-needed"))

    def triage(self, incident_text: str) -> dict:
        if self.triage_fn is not None:
            try:
                out = self.triage_fn(self.backend, incident_text)
                if out["ok"]:
                    return {**out["triage"], "backend": "project6:inference.triage"}
                self.reason = f"model output rejected after {out['attempts']} attempts: {out['error']}"
            except Exception as e:
                self.reason = f"{type(e).__name__}: {e}"
        return {**fallback_triage(incident_text), "backend": "fallback:rule-triage",
                "fallback_reason": self.reason}


_ALERT = re.compile(r"\[(CRITICAL|WARNING|INFO)\] Monitor '([^']+)' triggered on service:([\w-]+) "
                    r"\(tier:(\d)\) - (.*)")
_CATEGORY_RULES = [
    ("security", r"unauthori[sz]ed|leaked|credential|brute|api key"),
    ("storage", r"disk|no space"),
    ("database", r"replication|connection pool|lock contention"),
    ("network", r"dns|servfail|nxdomain|tls|certificate|packet loss|unhealthy targets"),
    ("messaging", r"queue|consumer lag|dlq|lag \d"),
    ("compute", r"oom|memory|crashloop|restarts|cpu|node"),
]
_TEAM_FOR = {"database": "database-oncall", "network": "network-oncall", "compute": "platform-oncall",
             "storage": "platform-oncall", "application": "service-owner", "messaging": "service-owner",
             "security": "security-oncall"}
_MATRIX = {1: {"outage": "P1", "major": "P2", "minor": "P3", "none": "P4"},
           2: {"outage": "P2", "major": "P3", "minor": "P4", "none": "P4"},
           3: {"outage": "P3", "major": "P4", "minor": "P4", "none": "P4"}}


def fallback_triage(incident_text: str) -> dict:
    """LOCAL FALLBACK for Project 6. Rules, not a model: it takes the first CRITICAL alert as the
    affected service, which is wrong whenever the first alert is only a symptom."""
    alerts = [m.groups() for m in _ALERT.finditer(incident_text)]
    primary = next((a for a in alerts if a[0] == "CRITICAL"), alerts[0] if alerts else None)
    if primary is None:
        service = next((s for s in SERVICE_NAMES if s in incident_text), None)
        if service is None:
            return {"error": "no alert line and no known service name in the input"}
        primary = ("WARNING", "Unknown", service, str(SERVICES[service]["tier"]), incident_text[:200])
    level, monitor, service, tier, message = primary
    own = " ".join(l for l in incident_text.splitlines() if service in l).lower()
    category = next((c for c, pat in _CATEGORY_RULES if re.search(pat, own)), "application")

    text = incident_text.lower()
    rates = [float(x) for x in re.findall(r"rate (\d+(?:\.\d+)?)%", text)]
    lags = [float(x) for x in re.findall(r"lag (\d+)s", text)]
    if re.search(r"availability check failing|processing stopped|is down", text) or (rates and max(rates) >= 40):
        impact = "outage"
    elif (rates and max(rates) >= 10) or (lags and max(lags) >= 600):
        impact = "major"
    elif rates or lags:
        impact = "minor"
    else:
        impact = "none"
    severity = _MATRIX[int(tier)][impact]
    if category == "security" and re.search(r"leaked|successful", text) and severity in ("P3", "P4"):
        severity = "P2"
    triage = Triage(
        severity=severity, category=category, affected_service=service,
        probable_root_cause=f"{monitor} on {service}: {message}"[:600],
        recommended_next_steps=[f"Check recent metrics and logs for {service}",
                                f"Look up the {category} runbook"],
        escalate_to="none" if severity == "P4" else _TEAM_FOR[category],
        confidence=0.6 if len([a for a in alerts if a[0] == "CRITICAL"]) <= 1 else 0.45)
    return triage.model_dump()


# =============================================================================================
# Project 5: runbook and knowledge-base search
# =============================================================================================
class RunbookAdapter:
    """Real paths: POST {RAG_URL} with the caller's own API key, or Project 5's Retriever in-process
    (RUNBOOK_BACKEND=project5). Fallback: oncall/runbooks.py. The access level always comes from
    the caller's session; the model has no argument that can raise it."""

    def __init__(self, mode: str = "auto"):
        self.mode, self.kind, self.retriever, self.reason = mode, "fallback", None, "mode=fallback"
        if mode in ("auto", "real"):
            try:
                self._load()
            except Exception as e:
                self.kind, self.reason = "fallback", f"{type(e).__name__}: {e}"
                if mode == "real":
                    raise BackendUnavailable(f"Project 5 retrieval: {self.reason}") from e

    def _load(self) -> None:
        if os.getenv("RAG_URL"):
            self.kind = "ask"
            return
        if os.getenv("RUNBOOK_BACKEND") != "project5" and self.mode != "real":
            raise BackendUnavailable("RAG_URL is not set and RUNBOOK_BACKEND is not 'project5'")
        if not (RAG_DIR / "index" / "chunks.jsonl").exists():
            raise BackendUnavailable(f"{RAG_DIR}/index/chunks.jsonl not found (run Project 5's ingest.py)")
        sys.path.insert(0, str(RAG_DIR))          # Project 5 uses top-level modules: settings, retrieve, ingest
        try:
            from retrieve import Retriever
            self.retriever = Retriever()
        finally:
            sys.path.remove(str(RAG_DIR))
        self.kind = "retriever"

    def search(self, query: str, access: str, top_k: int = 3, extra_docs: list[dict] | None = None) -> dict:
        try:
            if self.kind == "ask":
                return self._ask(query, access, top_k)
            if self.kind == "retriever":
                chunks = self.retriever.search(query, access=access, k=top_k,
                                               mode=os.getenv("RUNBOOK_MODE", "hybrid"))
                return {"results": [{"id": c["chunk_id"], "title": c["title"], "access": c["access"],
                                     "text": c["text"][:1200]} for c in chunks],
                        "access": access, "backend": "project5:retrieve.Retriever"}
        except Exception as e:
            if self.mode == "real":
                raise
            self.reason = f"{type(e).__name__}: {e}"
        return {"results": runbooks.search(query, access, top_k, extra_docs), "access": access,
                "backend": "fallback:local-runbooks", "fallback_reason": self.reason}

    def _ask(self, query: str, access: str, top_k: int) -> dict:
        keys = json.loads(os.getenv("RAG_API_KEYS", '{"public": "public-demo-key", '
                          '"internal": "support-demo-key", "restricted": "security-demo-key"}'))
        req = urllib.request.Request(os.environ["RAG_URL"],
                                     data=json.dumps({"question": query, "top_k": top_k}).encode(),
                                     headers={"Content-Type": "application/json", "X-API-Key": keys[access]})
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.load(r)
        return {"answer": out["answer"], "citations": out.get("citations", []),
                "abstained": out.get("abstained", False), "access": out.get("access", access),
                "backend": "project5:/ask"}

    def close(self) -> None:
        if self.retriever is not None:
            self.retriever.client.close()
            self.retriever = None


class Backends:
    def __init__(self, mode: str = "auto"):
        self.router, self.triage, self.runbooks = RouterAdapter(mode), TriageAdapter(mode), RunbookAdapter(mode)

    def describe(self) -> dict:
        return {"route_ticket": f"project3:router ({self.router.name})" if self.router.proc else f"fallback ({self.router.reason})",
                "triage_incident": "project6:inference.triage" if self.triage.triage_fn
                                   else f"fallback ({self.triage.reason})",
                "search_runbooks": {"ask": "project5:/ask", "retriever": "project5:retrieve.Retriever"}.get(
                    self.runbooks.kind, f"fallback ({self.runbooks.reason})")}

    def close(self) -> None:
        self.router.close()
        self.runbooks.close()
