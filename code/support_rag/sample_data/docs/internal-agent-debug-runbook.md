---
title: Internal Runbook - Debugging the Agent
access: internal
url: https://wiki.nimbus-internal.example/support/agent-debug-runbook
updated: 2026-11-03
---
# Internal Runbook: Debugging the Agent (Support Only)

Internal page. Do not share with customers.

## Getting a diagnostics bundle

Ask the customer to run `nimbusctl diagnose --bundle` as root. It writes a tar.gz with the redacted configuration, the last 24 hours of agent logs and connectivity test results. Have them attach it to the ticket, then open it with the internal `agentscope` viewer.

## Debug logging

Ask for `log_level: debug` only for as long as needed, and never for more than 24 hours: debug logs can reach several GB per day and include metric names and tag values.

## Known memory leak in 3.3.0 to 3.3.2

Agents 3.3.0, 3.3.1 and 3.3.2 leak memory when `container_discovery` is enabled on hosts with frequent container churn, ending in repeated NM-2010 restarts. The fix is in 3.3.3 and all 3.4 releases. Upgrade is the fix; as a stopgap, raise `memory_limit_mb` to 512.

## Agent connects but no host appears

If the log shows successful flushes and the host is still missing, look up the ingestion key in the admin console. The usual causes are a key from a different workspace (often a sandbox workspace) and a duplicate hostname hiding the host behind another one.

## When to escalate to agent engineering

Escalate in `#agent-oncall` when a crash reproduces on the latest version, or when more than 3 customers report the same symptom within a week. Attach the diagnostics bundle.
