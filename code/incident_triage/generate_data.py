#!/usr/bin/env python3
"""Synthetic incident generator for the Incident Triage Copilot.

Produces chat-format JSONL (system / user / assistant messages) where the user message
is a realistic bundle of monitoring alerts, log lines and metrics, and the assistant
message is the ground-truth triage JSON.

Standard library only. Example:

    python generate_data.py --out data --n-per-template 120 --seed 7

Leakage control: whole scenario templates are held out for the test split, so the test
set measures generalisation to incident *types* the model has never seen, not memorisation
of near-duplicate examples. Validation examples come from training templates (in-distribution,
for early stopping and loss curves).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from collections import Counter
from pathlib import Path

from prompts import SYSTEM_PROMPT, format_user_prompt

# --------------------------------------------------------------------------------------
# Service catalog
# --------------------------------------------------------------------------------------
HTTP_SERVICES = {
    "checkout-api": 1, "payments-service": 1, "auth-service": 1, "api-gateway": 1,
    "search-api": 2, "user-profile-service": 2, "notification-api": 2,
    "recommendation-service": 3, "reporting-ui": 3,
}
DATABASES = {"orders-db": 1, "accounts-db": 1, "inventory-db": 2, "analytics-db": 3}
DB_CLIENTS = {
    "orders-db": ["checkout-api", "payments-service"],
    "accounts-db": ["auth-service", "user-profile-service"],
    "inventory-db": ["search-api", "checkout-api"],
    "analytics-db": ["reporting-ui", "recommendation-service"],
}
CONSUMERS = {"order-events-consumer": 1, "email-events-consumer": 2, "analytics-ingest-consumer": 3}
TOPICS = {"order-events-consumer": "orders.created", "email-events-consumer": "notifications.email",
          "analytics-ingest-consumer": "clickstream.raw"}
ALL_TIERS = {**HTTP_SERVICES, **DATABASES, **CONSUMERS}

IMPACTS = ("outage", "major", "minor", "none")
SEVERITY_MATRIX = {
    1: {"outage": "P1", "major": "P2", "minor": "P3", "none": "P4"},
    2: {"outage": "P2", "major": "P3", "minor": "P4", "none": "P4"},
    3: {"outage": "P3", "major": "P4", "minor": "P4", "none": "P4"},
}


class Incident:
    """Mutable builder for one synthetic incident."""

    def __init__(self, rng: random.Random, style: str):
        self.rng = rng
        self.style = style
        self.t0 = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(
            seconds=rng.randint(0, 300 * 24 * 3600))
        self.alerts: list[tuple[int, str]] = []
        self.logs: list[tuple[int, str]] = []
        self.metrics: list[str] = []

    def ts(self, offset: int) -> str:
        return (self.t0 + dt.timedelta(seconds=offset)).strftime("%Y-%m-%dT%H:%M:%SZ")

    def alert(self, offset: int, level: str, svc: str, name: str, text: str) -> None:
        tier = ALL_TIERS.get(svc, 3)
        t = self.ts(offset)
        if self.style == "prometheus":
            line = (f"[FIRING:1] {name} severity={level.lower()} service={svc} tier={tier} "
                    f"- {text} (startsAt={t})")
        elif self.style == "datadog":
            line = f"{t} [{level}] Monitor '{name}' triggered on service:{svc} (tier:{tier}) - {text}"
        else:  # nagios-like
            line = f"** PROBLEM {svc}/{name} is {level} ** tier{tier} | {text} | {t}"
        self.alerts.append((offset, line))

    def log(self, offset: int, level: str, svc: str, msg: str) -> None:
        t = self.ts(offset)
        if self.rng.random() < 0.4:
            line = json.dumps({"ts": t, "level": level, "service": svc, "msg": msg})
        else:
            line = f"{t} {level:<5} [{svc}] {msg}"
        self.logs.append((offset, line))

    def metric(self, name: str, svc: str, value: str) -> None:
        if self.style == "prometheus":
            self.metrics.append(f'{name}{{service="{svc}"}} {value}')
        else:
            self.metrics.append(f"{svc}.{name} = {value}")


# --------------------------------------------------------------------------------------
# Shared symptom helpers: these make "impact" observable in the signals
# --------------------------------------------------------------------------------------
def pick_impact(rng: random.Random, allow_none: bool = True) -> str:
    weights = [0.25, 0.3, 0.3, 0.15] if allow_none else [0.3, 0.35, 0.35, 0.0]
    return rng.choices(IMPACTS, weights=weights)[0]


def http_symptom(inc: Incident, svc: str, impact: str, offset: int = 60) -> None:
    r = inc.rng
    if impact == "outage":
        if r.random() < 0.5:
            err = r.uniform(40, 95)
            inc.alert(offset, "CRITICAL", svc, "HighErrorRate", f"HTTP 5xx rate {err:.1f}% > 5% for 5m")
            inc.metric("http_5xx_ratio", svc, f"{err / 100:.3f}")
        else:
            inc.alert(offset, "CRITICAL", svc, "SyntheticCheckFailed",
                      "availability checks failing from 3/3 regions")
            inc.metric("synthetic_success_ratio", svc, "0.000")
    elif impact == "major":
        if r.random() < 0.5:
            err = r.uniform(10, 38)
            inc.alert(offset, "CRITICAL", svc, "HighErrorRate", f"HTTP 5xx rate {err:.1f}% > 5% for 5m")
            inc.metric("http_5xx_ratio", svc, f"{err / 100:.3f}")
        else:
            slo = r.choice([200, 300, 500])
            p99 = slo * r.uniform(3.2, 9)
            inc.alert(offset, "CRITICAL", svc, "LatencySLOBreach",
                      f"p99 latency {p99:.0f}ms (SLO {slo}ms)")
            inc.metric("http_p99_latency_ms", svc, f"{p99:.0f}")
    elif impact == "minor":
        if r.random() < 0.5:
            err = r.uniform(1, 8)
            inc.alert(offset, "WARNING", svc, "ElevatedErrorRate", f"HTTP 5xx rate {err:.1f}% > 1% for 10m")
            inc.metric("http_5xx_ratio", svc, f"{err / 100:.3f}")
        else:
            slo = r.choice([200, 300, 500])
            p99 = slo * r.uniform(1.3, 2.6)
            inc.alert(offset, "WARNING", svc, "LatencySLOBreach",
                      f"p99 latency {p99:.0f}ms (SLO {slo}ms)")
            inc.metric("http_p99_latency_ms", svc, f"{p99:.0f}")
    else:
        inc.metric("http_5xx_ratio", svc, f"{r.uniform(0, 0.004):.3f}")


def consumer_symptom(inc: Incident, svc: str, impact: str, offset: int = 60) -> None:
    r = inc.rng
    topic = TOPICS[svc]
    if impact == "outage":
        mins = r.randint(30, 120)
        inc.alert(offset, "CRITICAL", svc, "ConsumerStalled",
                  f"0 msgs/s consumed from {topic} for {mins}m, lag growing")
        inc.metric("consumer_lag_seconds", svc, str(mins * 60))
    elif impact == "major":
        mins = r.randint(10, 45)
        inc.alert(offset, "CRITICAL", svc, "ConsumerLagHigh", f"consumer lag on {topic} is {mins}m")
        inc.metric("consumer_lag_seconds", svc, str(mins * 60))
    elif impact == "minor":
        mins = r.randint(2, 9)
        inc.alert(offset, "WARNING", svc, "ConsumerLagHigh", f"consumer lag on {topic} is {mins}m")
        inc.metric("consumer_lag_seconds", svc, str(mins * 60))
    else:
        inc.metric("consumer_lag_seconds", svc, str(r.randint(0, 30)))


def symptom(inc: Incident, svc: str, impact: str, offset: int = 60) -> None:
    if svc in CONSUMERS:
        consumer_symptom(inc, svc, impact, offset)
    else:
        http_symptom(inc, svc, impact, offset)


def steps(rng: random.Random, required: list[str], optional: list[str], k: int = 2) -> list[str]:
    return required + rng.sample(optional, k=min(k, len(optional)))


# --------------------------------------------------------------------------------------
# Scenario templates. Each returns (incident, label_without_severity, impacted_service, impact)
# --------------------------------------------------------------------------------------
def db_conn_pool_exhausted(inc: Incident):
    r = inc.rng
    db = r.choice(list(DATABASES))
    app = r.choice(DB_CLIENTS[db])
    impact = pick_impact(r, allow_none=False)
    max_conn = r.choice([200, 300, 500])
    inc.alert(0, "CRITICAL", db, "PostgresConnectionsNearLimit",
              f"active connections {max_conn - r.randint(0, 3)}/{max_conn}")
    symptom(inc, app, impact)
    inc.log(30, "ERROR", app, "HikariPool-1 - Connection is not available, request timed out after 30000ms")
    inc.log(35, "FATAL", db, "sorry, too many clients already")
    inc.log(40, "ERROR", app, f"could not obtain connection to {db}: remaining connection slots are reserved")
    inc.metric("pg_stat_activity_count", db, str(max_conn))
    inc.metric("pg_idle_in_transaction", db, str(r.randint(max_conn // 3, max_conn // 2)))
    cause = r.choice([
        f"{db} has hit max_connections; many sessions are idle in transaction, so {app} cannot get connections from its pool.",
        f"Connection exhaustion on {db}: leaked or idle-in-transaction sessions are holding all slots and {app} requests time out waiting for a connection.",
    ])
    nxt = steps(r, [f"Inspect pg_stat_activity on {db} for idle-in-transaction or long-running sessions",
                    "Terminate stuck idle sessions to free connection slots"],
                [f"Check recent deploys of {app} for connection leaks or larger pool sizes",
                 f"Put a connection pooler (PgBouncer) in front of {db} or lower per-pod pool size",
                 f"Watch the {app} error rate until it returns to baseline"])
    return {"category": "database", "affected_service": db, "probable_root_cause": cause,
            "recommended_next_steps": nxt, "team": "database-oncall"}, app, impact


def db_replication_lag(inc: Incident):
    r = inc.rng
    db = r.choice(list(DATABASES))
    app = r.choice(DB_CLIENTS[db])
    impact = pick_impact(r)
    lag = r.randint(120, 1800)
    inc.alert(0, "WARNING" if impact in ("none", "minor") else "CRITICAL", db, "ReplicationLagHigh",
              f"replica {db}-replica-1 lag {lag}s")
    symptom(inc, app, impact)
    inc.log(20, "WARN", db, f"replica {db}-replica-1 replay lag {lag}s, WAL receiver behind primary")
    inc.log(25, "INFO", db, "checkpoint starting: wal")
    if impact != "none":
        inc.log(50, "WARN", app, "read-after-write check failed: record not found on replica, retrying on primary")
    inc.metric("pg_replication_lag_seconds", db, str(lag))
    inc.metric("disk_write_iops", f"{db}-replica-1", str(r.randint(2800, 3000)))
    cause = r.choice([
        f"The read replica of {db} is lagging the primary ({lag}s), so {app} reads stale data.",
        f"Replication lag on {db}: the replica cannot replay WAL fast enough, causing stale reads for {app}.",
    ])
    nxt = steps(r, [f"Check replica {db}-replica-1 IO and replay rate",
                    "Identify large write transactions or bulk jobs on the primary"],
                [f"Temporarily route {app} reads to the primary",
                 "Pause or throttle batch writers until the replica catches up",
                 "Consider a larger instance class or faster storage for the replica"])
    return {"category": "database", "affected_service": db, "probable_root_cause": cause,
            "recommended_next_steps": nxt, "team": "database-oncall"}, app, impact


def db_lock_contention(inc: Incident):
    r = inc.rng
    db = r.choice(list(DATABASES))
    app = r.choice(DB_CLIENTS[db])
    impact = pick_impact(r, allow_none=False)
    table = r.choice(["orders", "accounts", "sessions", "inventory_items", "payments"])
    inc.alert(0, "CRITICAL", db, "PostgresLockWaits", f"{r.randint(40, 300)} sessions waiting on locks")
    symptom(inc, app, impact)
    inc.log(15, "LOG", db, f"process {r.randint(1000, 99999)} still waiting for ShareLock on transaction after 1000.123 ms")
    inc.log(18, "ERROR", db, "deadlock detected")
    inc.log(22, "LOG", db, f"statement: ALTER TABLE {table} ADD COLUMN migrated_at timestamptz")
    inc.log(45, "ERROR", app, "query canceled due to statement timeout")
    inc.metric("pg_locks_waiting", db, str(r.randint(40, 300)))
    cause = r.choice([
        f"A schema migration (ALTER TABLE {table}) on {db} is holding an exclusive lock, blocking queries from {app}.",
        f"Lock contention on {db}: a long-running migration on table {table} blocks other transactions and {app} queries time out.",
    ])
    nxt = steps(r, [f"Find the blocking PID in pg_locks / pg_stat_activity on {db}",
                    "Cancel or terminate the blocking migration if safe"],
                ["Re-run the migration with lock_timeout and in a low-traffic window",
                 f"Confirm {app} recovers after the lock is released",
                 "Review the migration process for online-safe DDL"])
    return {"category": "database", "affected_service": db, "probable_root_cause": cause,
            "recommended_next_steps": nxt, "team": "database-oncall"}, app, impact


def dns_failure(inc: Incident):
    r = inc.rng
    app = r.choice(list(HTTP_SERVICES))
    impact = pick_impact(r, allow_none=False)
    dep = r.choice(["payments-provider.example.net", "auth.internal.svc", "geo-api.example.com"])
    inc.alert(0, "CRITICAL", "coredns", "DNSErrorsHigh", f"SERVFAIL rate {r.uniform(20, 80):.0f}%")
    symptom(inc, app, impact)
    inc.log(10, "ERROR", app, f"getaddrinfo ENOTFOUND {dep}")
    inc.log(12, "ERROR", app, f"dial tcp: lookup {dep}: i/o timeout")
    inc.log(14, "WARN", "coredns", f"[ERROR] plugin/errors: 2 {dep}. A: read udp 10.0.0.12:53: i/o timeout")
    inc.metric("coredns_dns_responses_servfail_ratio", "coredns", f"{r.uniform(0.2, 0.8):.2f}")
    cause = r.choice([
        f"DNS resolution is failing in coredns (upstream resolver timeouts), so {app} cannot resolve {dep}.",
        f"coredns is returning SERVFAIL for external lookups; {app} fails to reach {dep}.",
    ])
    nxt = steps(r, ["Check coredns pod health and upstream resolver reachability",
                    f"Test resolution of {dep} from an affected pod"],
                ["Scale coredns or restart unhealthy pods", "Review recent network policy or resolver config changes",
                 "Enable node-local DNS cache if not present"])
    return {"category": "network", "affected_service": "coredns", "probable_root_cause": cause,
            "recommended_next_steps": nxt, "team": "network-oncall"}, app, impact


def lb_unhealthy_targets(inc: Incident):
    r = inc.rng
    app = r.choice(list(HTTP_SERVICES))
    impact = pick_impact(r, allow_none=False)
    total = r.choice([6, 8, 12])
    healthy = {"outage": 0, "major": total // 3, "minor": total - 1}[impact]
    lb = f"{app}-alb"
    inc.alert(0, "CRITICAL", lb, "UnhealthyTargets", f"{healthy}/{total} targets healthy")
    symptom(inc, app, impact)
    inc.log(5, "WARN", lb, f"health check failed: GET /healthz returned 503 for target 10.2.{r.randint(0,9)}.{r.randint(2,250)}:8080")
    inc.log(9, "INFO", lb, "target deregistration in progress")
    inc.metric("lb_healthy_host_count", lb, str(healthy))
    inc.metric("lb_http_502_count", lb, str(r.randint(50, 5000)))
    cause = r.choice([
        f"The load balancer {lb} marks most targets unhealthy because /healthz returns 503, so requests to {app} get 502s.",
        f"Health checks on {lb} are failing and targets are deregistered, reducing capacity for {app}.",
    ])
    nxt = steps(r, [f"Check why /healthz on {app} returns 503 (dependency check or startup probe)",
                    f"Compare health check config on {lb} with the app's actual health endpoint"],
                ["Roll back recent health-check or security group changes",
                 "Scale out healthy targets", f"Monitor healthy host count on {lb}"])
    return {"category": "network", "affected_service": lb, "probable_root_cause": cause,
            "recommended_next_steps": nxt, "team": "network-oncall"}, app, impact


def tls_cert_expiry(inc: Incident):
    r = inc.rng
    app = r.choice(list(HTTP_SERVICES))
    impact = r.choices(["outage", "none"], weights=[0.4, 0.6])[0]
    domain = f"{app.split('-')[0]}.example.com"
    if impact == "outage":
        inc.alert(0, "CRITICAL", app, "TLSCertificateExpired", f"certificate for {domain} (managed by cert-manager) expired 2h ago")
        symptom(inc, app, "outage")
        inc.log(20, "ERROR", "api-gateway", f"x509: certificate has expired or is not yet valid for {domain}")
        inc.log(30, "ERROR", "cert-manager", f"Failed to renew certificate {domain}: ACME challenge failed")
        cause = f"The TLS certificate for {domain} expired because cert-manager renewal failed; clients reject TLS handshakes to {app}."
        req = [f"Manually renew and deploy the certificate for {domain}",
               "Fix the cert-manager ACME challenge failure"]
    else:
        days = r.randint(3, 13)
        inc.alert(0, "WARNING", app, "TLSCertificateExpiringSoon", f"certificate for {domain} (managed by cert-manager) expires in {days} days")
        symptom(inc, app, "none")
        inc.log(30, "WARN", "cert-manager", f"Certificate {domain} renewal attempt failed: ACME challenge failed, will retry")
        cause = f"Automatic renewal of the TLS certificate for {domain} is failing in cert-manager; it will expire in {days} days."
        req = ["Investigate the cert-manager ACME challenge failure", f"Renew the certificate for {domain} before expiry"]
    inc.metric("cert_expiry_seconds", app, str(r.randint(-7200, -60) if impact == "outage" else days * 86400))
    nxt = steps(r, req, ["Add alerting at 30 and 14 days before expiry", "Verify DNS/HTTP-01 challenge routing",
                         "Audit other certificates managed by the same issuer"])
    return {"category": "network", "affected_service": "cert-manager", "probable_root_cause": cause,
            "recommended_next_steps": nxt, "team": "network-oncall"}, app, impact


def packet_loss(inc: Incident):
    r = inc.rng
    app = r.choice(list(HTTP_SERVICES))
    impact = pick_impact(r)
    az_a, az_b = r.sample(["us-east-1a", "us-east-1b", "us-east-1c"], 2)
    loss = {"outage": r.uniform(30, 60), "major": r.uniform(8, 25), "minor": r.uniform(1, 5),
            "none": r.uniform(0.3, 1)}[impact]
    inc.alert(0, "CRITICAL" if loss > 5 else "WARNING", "network-fabric", "InterAZPacketLoss",
              f"packet loss {loss:.1f}% between {az_a} and {az_b}")
    symptom(inc, app, impact)
    inc.log(10, "WARN", app, f"upstream request retry 3/3: connection reset by peer (zone {az_b})")
    inc.log(12, "WARN", "network-fabric", f"tcp retransmits elevated on links {az_a}<->{az_b}")
    inc.metric("node_netstat_tcp_retrans_ratio", "network-fabric", f"{loss / 100:.3f}")
    cause = f"Packet loss ({loss:.1f}%) on the network path between {az_a} and {az_b} causes connection resets and retries for {app}."
    nxt = steps(r, [f"Confirm loss with mtr/iperf between hosts in {az_a} and {az_b}",
                    "Check cloud provider status page and transit gateway / VPC metrics"],
                [f"Shift {app} traffic away from {az_b} if possible", "Open a ticket with the cloud provider",
                 "Correlate with recent network ACL or routing changes"])
    return {"category": "network", "affected_service": "network-fabric", "probable_root_cause": cause,
            "recommended_next_steps": nxt, "team": "network-oncall"}, app, impact


def oom_crashloop(inc: Incident):
    r = inc.rng
    app = r.choice(list(HTTP_SERVICES) + list(CONSUMERS))
    impact = pick_impact(r, allow_none=False)
    limit = r.choice([512, 1024, 2048])
    restarts = r.randint(5, 40)
    inc.alert(0, "CRITICAL", app, "PodCrashLooping", f"{restarts} restarts in 15m, reason=OOMKilled")
    symptom(inc, app, impact)
    inc.log(5, "WARN", "kubelet", f"Container {app} in pod {app}-7d9f{r.randint(10,99)} OOMKilled (limit {limit}Mi)")
    inc.log(8, "INFO", "kubelet", f"Back-off restarting failed container {app}")
    inc.metric("container_memory_working_set_mb", app, str(limit))
    inc.metric("kube_pod_container_status_restarts_total", app, str(restarts))
    cause = r.choice([
        f"Pods of {app} exceed their {limit}Mi memory limit and are OOMKilled, causing a crash loop.",
        f"{app} containers are being OOMKilled at the {limit}Mi limit and restart repeatedly.",
    ])
    nxt = steps(r, [f"Check memory usage trend and recent changes to {app}",
                    f"Temporarily raise the memory limit for {app} or roll back the last deploy"],
                ["Capture a heap profile from a running pod", "Verify JVM/runtime heap settings fit within the container limit",
                 "Check if traffic spiked beyond normal"])
    team = "platform-oncall"
    return {"category": "compute", "affected_service": app, "probable_root_cause": cause,
            "recommended_next_steps": nxt, "team": team}, app, impact


def cpu_saturation(inc: Incident):
    r = inc.rng
    app = r.choice(list(HTTP_SERVICES))
    impact = pick_impact(r)
    cpu = r.uniform(92, 100)
    throttle = r.uniform(25, 70)
    inc.alert(0, "WARNING", app, "CPUThrottlingHigh", f"CPU throttled {throttle:.0f}% of periods, usage {cpu:.0f}% of limit")
    symptom(inc, app, impact)
    inc.log(20, "WARN", app, "request handler took 2840ms (threshold 1000ms)")
    inc.log(25, "INFO", "hpa", f"{app}: desired replicas {r.randint(10, 20)} exceeds maxReplicas {r.randint(6, 9)}")
    inc.metric("container_cpu_usage_ratio", app, f"{cpu / 100:.2f}")
    inc.metric("container_cpu_cfs_throttled_ratio", app, f"{throttle / 100:.2f}")
    cause = f"{app} is CPU-saturated and throttled because the autoscaler is capped at maxReplicas while load increased."
    nxt = steps(r, [f"Raise HPA maxReplicas or CPU limits for {app}",
                    "Check whether traffic growth or a hot code path is driving CPU"],
                ["Profile the service under load", "Verify cluster has capacity for more replicas",
                 "Consider rate limiting abusive clients"])
    return {"category": "compute", "affected_service": app, "probable_root_cause": cause,
            "recommended_next_steps": nxt, "team": "platform-oncall"}, app, impact


def node_not_ready(inc: Incident):
    r = inc.rng
    app = r.choice(list(HTTP_SERVICES))
    impact = pick_impact(r)
    node = f"ip-10-1-{r.randint(0, 255)}-{r.randint(0, 255)}"
    n = r.randint(1, 4)
    inc.alert(0, "CRITICAL", "k8s-cluster", "KubeNodeNotReady", f"{n} node(s) NotReady including {node}")
    symptom(inc, app, impact)
    inc.log(3, "WARN", "kubelet", f"node {node}: PLEG is not healthy: pleg was last seen active 3m10s ago")
    inc.log(6, "INFO", "kube-controller-manager", f"Evicting pods from NotReady node {node}")
    inc.log(9, "WARN", "scheduler", f"0/12 nodes are available: insufficient cpu; {app} pods Pending")
    inc.metric("kube_node_status_ready_false", "k8s-cluster", str(n))
    cause = f"{n} Kubernetes node(s) went NotReady (kubelet/PLEG unhealthy); evicted {app} pods cannot be rescheduled due to insufficient capacity."
    nxt = steps(r, [f"Inspect kubelet and container runtime on {node}",
                    "Cordon and replace unhealthy nodes"],
                ["Add capacity to the node group so Pending pods schedule", "Check for a kernel or runtime update rolled out recently",
                 f"Verify {app} has a PodDisruptionBudget and spreads across zones"])
    return {"category": "compute", "affected_service": "k8s-cluster", "probable_root_cause": cause,
            "recommended_next_steps": nxt, "team": "platform-oncall"}, app, impact


def disk_full(inc: Incident):
    r = inc.rng
    app = r.choice(list(HTTP_SERVICES))
    impact = r.choices(["outage", "major", "none"], weights=[0.3, 0.3, 0.4])[0]
    host = f"{app}-node-{r.randint(1, 9)}"
    used = {"outage": 100.0, "major": r.uniform(99, 99.9), "none": r.uniform(85, 94)}[impact]
    inc.alert(0, "CRITICAL" if used >= 99 else "WARNING", host, "DiskSpaceLow", f"/var {used:.1f}% used")
    symptom(inc, app, impact)
    inc.log(10, "WARN", host, f"/var/log/{app}/debug.log grew 40GB in 6h")
    if impact != "none":
        inc.log(15, "ERROR", app, "write /var/lib/app/tmp/upload-123: no space left on device")
    inc.metric("node_filesystem_used_ratio", host, f"{used / 100:.3f}")
    cause = f"The /var filesystem on {host} is {used:.0f}% full because debug logging for {app} filled the disk."
    nxt = steps(r, [f"Free space on {host} (rotate or truncate {app} debug logs)",
                    f"Disable debug log level for {app}"],
                ["Fix log rotation configuration", "Expand the volume if growth is legitimate",
                 "Add a disk growth-rate alert"])
    return {"category": "storage", "affected_service": host, "probable_root_cause": cause,
            "recommended_next_steps": nxt, "team": "platform-oncall"}, app, impact


def iops_throttling(inc: Incident):
    r = inc.rng
    db = r.choice(list(DATABASES))
    app = r.choice(DB_CLIENTS[db])
    impact = pick_impact(r)
    vol = f"{db}-data-volume"
    inc.alert(0, "WARNING", vol, "EBSBurstBalanceLow", f"burst balance {r.randint(0, 5)}%, IOPS capped at baseline")
    symptom(inc, app, impact)
    inc.log(20, "WARN", db, "checkpoint took 94s, longer than checkpoint_timeout")
    inc.log(25, "WARN", app, f"slow query 4.2s on {db}")
    inc.metric("volume_queue_length", vol, f"{r.uniform(30, 120):.0f}")
    inc.metric("volume_read_latency_ms", vol, f"{r.uniform(20, 90):.0f}")
    cause = f"The storage volume {vol} exhausted its IOPS burst balance and is throttled, so {db} queries from {app} are slow."
    nxt = steps(r, [f"Confirm IOPS throttling on {vol} in cloud metrics",
                    f"Move {vol} to provisioned IOPS / gp3 with higher baseline"],
                ["Identify the IO-heavy workload (vacuum, batch jobs, missing index)", "Reschedule heavy batch jobs off-peak",
                 "Add a burst-balance alert at 30%"])
    return {"category": "storage", "affected_service": vol, "probable_root_cause": cause,
            "recommended_next_steps": nxt, "team": "platform-oncall"}, app, impact


def bad_deploy(inc: Incident):
    r = inc.rng
    app = r.choice(list(HTTP_SERVICES))
    impact = pick_impact(r, allow_none=False)
    ver = f"v{r.randint(1, 4)}.{r.randint(0, 30)}.{r.randint(0, 9)}"
    inc.alert(-300, "INFO", app, "DeploymentCompleted", f"rollout of {app} {ver} completed")
    symptom(inc, app, impact)
    exc = r.choice(["NullPointerException at OrderMapper.toDto(OrderMapper.java:88)",
                    "KeyError: 'currency' in pricing.py line 142",
                    "TypeError: Cannot read properties of undefined (reading 'id')"])
    inc.log(70, "ERROR", app, f"Unhandled exception: {exc}")
    inc.log(75, "ERROR", app, f"500 POST /api/v2/orders version={ver}")
    inc.metric("deploy_version_info", app, ver)
    cause = r.choice([
        f"The deploy of {app} {ver} introduced a regression ({exc.split()[0].rstrip(':')}) that started errors right after rollout.",
        f"A code regression in {app} {ver}: errors began immediately after the rollout with {exc.split()[0].rstrip(':')}.",
    ])
    nxt = steps(r, [f"Roll back {app} to the previous version",
                    f"Confirm the error rate drops after rollback"],
                [f"Diff {ver} against the previous release for the failing code path",
                 "Add a regression test for the failing request", "Use canary analysis for the next rollout"])
    return {"category": "application", "affected_service": app, "probable_root_cause": cause,
            "recommended_next_steps": nxt, "team": "service-owner"}, app, impact


def memory_leak(inc: Incident):
    r = inc.rng
    app = r.choice(list(HTTP_SERVICES))
    impact = pick_impact(r)
    inc.alert(0, "WARNING", app, "MemoryGrowthAnomaly", f"heap usage grew from 35% to {r.randint(80, 97)}% over 18h")
    symptom(inc, app, impact)
    inc.log(30, "WARN", app, "GC overhead high: 38% of time spent in GC over last 5m")
    inc.log(40, "INFO", app, f"cache entries: {r.randint(2, 9)}.{r.randint(0, 9)}M (no eviction configured)")
    inc.metric("jvm_heap_used_ratio", app, f"{r.uniform(0.8, 0.97):.2f}")
    inc.metric("jvm_gc_pause_seconds_p99", app, f"{r.uniform(0.8, 3):.2f}")
    cause = f"{app} has a memory leak: an in-process cache without eviction grows steadily, driving heap usage and GC pauses."
    nxt = steps(r, [f"Capture a heap dump from {app} and identify the growing object",
                    "Configure a size or TTL bound on the in-process cache"],
                ["Rolling-restart pods as a temporary mitigation", "Add heap usage alerts per pod",
                 "Check which release introduced the cache"])
    return {"category": "application", "affected_service": app, "probable_root_cause": cause,
            "recommended_next_steps": nxt, "team": "service-owner"}, app, impact


def upstream_timeout(inc: Incident):
    r = inc.rng
    app = r.choice(["checkout-api", "payments-service", "notification-api", "search-api"])
    impact = pick_impact(r, allow_none=False)
    vendor = {"checkout-api": "tax-calculation-vendor", "payments-service": "card-processor",
              "notification-api": "sms-gateway", "search-api": "geo-lookup-vendor"}[app]
    inc.alert(0, "WARNING", app, "UpstreamLatencyHigh", f"calls to {vendor} p99 {r.randint(5, 30)}s")
    symptom(inc, app, impact)
    inc.log(20, "ERROR", app, f"timeout after 5000ms calling https://{vendor}.example.net/v1/")
    inc.log(22, "WARN", app, f"circuit breaker for {vendor} is HALF_OPEN")
    inc.metric("upstream_timeout_total", app, str(r.randint(100, 9000)))
    cause = f"The third-party dependency {vendor} is slow or timing out, and {app} has no fallback, so its requests fail."
    nxt = steps(r, [f"Check {vendor} status page and contact the vendor",
                    f"Enable or tighten the circuit breaker and fallback for {vendor} in {app}"],
                ["Reduce client timeout to fail fast", "Queue requests for retry where the business flow allows",
                 "Communicate degraded functionality to support"])
    return {"category": "application", "affected_service": app, "probable_root_cause": cause,
            "recommended_next_steps": nxt, "team": "service-owner"}, app, impact


def brute_force(inc: Incident):
    r = inc.rng
    app = "auth-service"
    success = r.random() < 0.3
    impact = "minor" if not success else r.choice(["minor", "major"])
    attempts = r.randint(5000, 90000)
    ip = f"185.{r.randint(0, 255)}.{r.randint(0, 255)}.{r.randint(1, 254)}"
    inc.alert(0, "CRITICAL", app, "LoginFailureSpike", f"{attempts} failed logins in 10m from {r.randint(1, 40)} IPs")
    symptom(inc, app, impact)
    inc.log(5, "WARN", app, f"failed login for user=admin@example.com from {ip} reason=bad_password")
    if success:
        inc.log(50, "INFO", app, f"successful login for user=ops-admin@example.com from {ip} after 312 failures")
        cause = f"Credential-stuffing attack against {app}; at least one account (ops-admin) was successfully accessed from {ip}."
        req = ["Lock the compromised account and revoke its sessions and tokens",
               f"Block {ip} and related IPs at the WAF"]
    else:
        cause = f"Credential-stuffing / brute-force attack against {app} from a small set of IPs; no successful logins observed."
        req = [f"Block {ip} and related IPs at the WAF", "Enable stricter rate limiting on the login endpoint"]
    inc.metric("auth_login_failures_total", app, str(attempts))
    nxt = steps(r, req, ["Enforce MFA for privileged accounts", "Review audit logs for activity by affected accounts",
                         "Notify the security team and open a security incident"])
    sev_floor = "P2" if success else None
    return {"category": "security", "affected_service": app, "probable_root_cause": cause,
            "recommended_next_steps": nxt, "team": "security-oncall", "sev_floor": sev_floor}, app, impact


def leaked_secret(inc: Incident):
    r = inc.rng
    app = r.choice(list(HTTP_SERVICES))
    impact = "none"
    key = r.choice(["AWS access key", "Stripe secret key", "database password"])
    inc.alert(0, "CRITICAL", app, "SecretDetectedInLogs", f"{key} pattern detected in application logs")
    inc.log(10, "DEBUG", app, "config loaded: {'db_password': '****redacted-by-scanner****'}")
    inc.log(12, "INFO", "log-scanner", f"match rule={key.replace(' ', '_').lower()} source={app} count={r.randint(1, 300)}")
    symptom(inc, app, impact)
    cause = f"{app} is writing a {key} to its logs (debug config dump), exposing the secret to anyone with log access."
    nxt = steps(r, [f"Rotate the exposed {key} immediately", f"Remove the debug config dump from {app} logging"],
                ["Purge or restrict access to the affected log indices", "Check access logs for use of the leaked secret",
                 "Add secret redaction in the logging pipeline"])
    return {"category": "security", "affected_service": app, "probable_root_cause": cause,
            "recommended_next_steps": nxt, "team": "security-oncall", "sev_floor": "P2"}, app, impact


def waf_bot_spike(inc: Incident):
    r = inc.rng
    app = r.choice(["search-api", "checkout-api", "api-gateway"])
    impact = pick_impact(r)
    rps = r.randint(5, 40) * 1000
    inc.alert(0, "WARNING", "edge-waf", "RequestRateAnomaly", f"{rps} req/s to {app}, 12x baseline, 80% from one ASN")
    symptom(inc, app, impact)
    inc.log(10, "INFO", "edge-waf", f"rule bot-score<10 matched {r.randint(1, 9)}M requests user_agent=python-requests/2.31")
    inc.metric("waf_requests_per_second", "edge-waf", str(rps))
    cause = f"Automated scraping/bot traffic from a single ASN is flooding {app} at {rps} req/s, overloading it."
    nxt = steps(r, ["Enable a WAF rate-limit or block rule for the offending ASN / bot signature",
                    f"Confirm {app} recovers once bot traffic is filtered"],
                ["Add bot-management challenges to the affected endpoints", "Review whether the traffic is a known partner",
                 "Scale out temporarily if legitimate traffic is affected"])
    return {"category": "security", "affected_service": "edge-waf", "probable_root_cause": cause,
            "recommended_next_steps": nxt, "team": "security-oncall"}, app, impact


def kafka_consumer_lag(inc: Incident):
    r = inc.rng
    svc = r.choice(list(CONSUMERS))
    impact = pick_impact(r)
    topic = TOPICS[svc]
    inc.alert(0, "WARNING", svc, "ConsumerRebalanceStorm", f"{r.randint(10, 60)} rebalances in 15m on group {svc}")
    symptom(inc, svc, impact)
    inc.log(5, "WARN", svc, "Member consumer-1 sending LeaveGroup request: consumer poll timeout has expired (max.poll.interval.ms=300000)")
    inc.log(8, "INFO", svc, f"Revoking previously assigned partitions [{topic}-0, {topic}-3]")
    inc.metric("kafka_consumergroup_rebalances_total", svc, str(r.randint(10, 60)))
    cause = f"{svc} processes batches slower than max.poll.interval.ms, so consumers are kicked from the group and rebalance repeatedly, building lag on {topic}."
    nxt = steps(r, [f"Reduce max.poll.records or increase max.poll.interval.ms for {svc}",
                    "Find the slow downstream call inside the message handler"],
                [f"Scale {svc} consumers up to the partition count", "Check broker health and partition skew",
                 "Add a lag-based alert per partition"])
    return {"category": "messaging", "affected_service": svc, "probable_root_cause": cause,
            "recommended_next_steps": nxt, "team": "service-owner"}, svc, impact


def dlq_growth(inc: Incident):
    r = inc.rng
    svc = r.choice(list(CONSUMERS))
    impact = pick_impact(r)
    topic = TOPICS[svc]
    n = r.randint(500, 200000)
    inc.alert(0, "WARNING", svc, "DeadLetterQueueGrowing", f"{n} messages in {topic}.dlq in 30m")
    symptom(inc, svc, impact)
    inc.log(5, "ERROR", svc, "Failed to deserialize message: Unknown magic byte! schema id 4117 not found")
    inc.log(7, "INFO", "schema-registry", f"new schema version registered for subject {topic}-value")
    inc.metric("dlq_messages_total", svc, str(n))
    cause = f"A producer started publishing {topic} with a new incompatible schema; {svc} cannot deserialize messages and sends them to the DLQ."
    nxt = steps(r, [f"Identify the producer and schema version change for {topic}",
                    "Roll back the producer or deploy a consumer that understands the new schema"],
                ["Replay DLQ messages after the fix", "Enforce schema compatibility mode in the registry",
                 "Add a DLQ growth alert per topic"])
    return {"category": "messaging", "affected_service": svc, "probable_root_cause": cause,
            "recommended_next_steps": nxt, "team": "service-owner"}, svc, impact


TEMPLATES = {
    "db_conn_pool_exhausted": db_conn_pool_exhausted, "db_replication_lag": db_replication_lag,
    "db_lock_contention": db_lock_contention, "dns_failure": dns_failure,
    "lb_unhealthy_targets": lb_unhealthy_targets, "tls_cert_expiry": tls_cert_expiry,
    "packet_loss": packet_loss, "oom_crashloop": oom_crashloop, "cpu_saturation": cpu_saturation,
    "node_not_ready": node_not_ready, "disk_full": disk_full, "iops_throttling": iops_throttling,
    "bad_deploy": bad_deploy, "memory_leak": memory_leak, "upstream_timeout": upstream_timeout,
    "brute_force": brute_force, "leaked_secret": leaked_secret, "waf_bot_spike": waf_bot_spike,
    "kafka_consumer_lag": kafka_consumer_lag, "dlq_growth": dlq_growth,
}

# One template per category is held out entirely for the test split.
TEST_TEMPLATES = {"db_lock_contention", "packet_loss", "node_not_ready", "iops_throttling",
                  "upstream_timeout", "waf_bot_spike", "dlq_growth"}

NOISE = [
    ("reporting-ui", "WARNING", "SlowPageLoad", "dashboard render p95 3.1s"),
    ("analytics-db", "WARNING", "VacuumRunning", "autovacuum running on table events for 25m"),
    ("ci-runner", "WARNING", "BuildQueueLong", "12 jobs queued"),
    ("backup-job", "INFO", "BackupCompleted", "nightly snapshot completed in 41m"),
    ("recommendation-service", "WARNING", "ModelStale", "model artifact older than 7 days"),
    ("vpn-gateway", "INFO", "TunnelFlap", "tunnel to office-2 re-established"),
]


def severity_for(label: dict, impacted: str, impact: str) -> str:
    sev = SEVERITY_MATRIX[ALL_TIERS.get(impacted, 3)][impact]
    floor = label.pop("sev_floor", None)
    if floor and int(sev[1]) > int(floor[1]):
        sev = floor
    return sev


def build_example(name: str, rng: random.Random) -> dict:
    inc = Incident(rng, rng.choice(["prometheus", "datadog", "nagios"]))
    label, impacted, impact = TEMPLATES[name](inc)
    severity = severity_for(label, impacted, impact)
    team = label.pop("team")
    # Noise alerts the model must ignore.
    for svc, lvl, alert_name, text in rng.sample(NOISE, k=rng.randint(0, 3)):
        inc.alert(rng.randint(-1800, 600), lvl, svc, alert_name, text)
    # Randomly drop a signal type; confidence reflects how much evidence is present.
    confidence = 0.9
    if rng.random() < 0.15:
        inc.logs = []
        confidence -= 0.2
    if rng.random() < 0.15:
        inc.metrics = []
        confidence -= 0.1
    inc.alerts.sort(key=lambda x: x[0])
    inc.logs.sort(key=lambda x: x[0])
    rng.shuffle(inc.metrics)
    target = {
        "severity": severity,
        "category": label["category"],
        "affected_service": label["affected_service"],
        "probable_root_cause": label["probable_root_cause"],
        "recommended_next_steps": label["recommended_next_steps"],
        "escalate_to": "none" if severity == "P4" else team,
        "confidence": round(confidence, 2),
    }
    user = format_user_prompt([a for _, a in inc.alerts], [l for _, l in inc.logs], inc.metrics)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(target)},
        ],
        "meta": {"template": name, "impact": impact, "impacted_service": impacted},
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-per-template", type=int, default=120)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train, val, test = [], [], []
    seen_inputs: set[str] = set()
    for name in TEMPLATES:
        for _ in range(args.n_per_template):
            ex = build_example(name, rng)
            key = ex["messages"][1]["content"]
            if key in seen_inputs:  # exact-duplicate guard
                continue
            seen_inputs.add(key)
            if name in TEST_TEMPLATES:
                test.append(ex)
            elif rng.random() < args.val_frac:
                val.append(ex)
            else:
                train.append(ex)
    for rows in (train, val, test):
        rng.shuffle(rows)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    write_jsonl(out / "test.jsonl", test)

    for split, rows in (("train", train), ("val", val), ("test", test)):
        sev = Counter(json.loads(r["messages"][2]["content"])["severity"] for r in rows)
        cat = Counter(json.loads(r["messages"][2]["content"])["category"] for r in rows)
        print(f"{split:5s} n={len(rows):5d}  severity={dict(sorted(sev.items()))}  categories={dict(sorted(cat.items()))}")
    print(f"wrote {out}/train.jsonl, val.jsonl, test.jsonl")
    print("\nexample user message:\n" + train[0]["messages"][1]["content"])
    print("\nexample target:\n" + train[0]["messages"][2]["content"])


if __name__ == "__main__":
    main()
