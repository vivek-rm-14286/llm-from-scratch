---
title: Troubleshooting and Error Codes
access: public
url: https://docs.nimbus-monitor.example/troubleshooting
updated: 2026-10-20
---
# Troubleshooting and Error Codes

The agent logs error codes in `/var/log/nimbus/agent.log` (Linux) or the Windows Event Log.

## NM-1001: cannot connect to ingestion endpoint

The agent cannot open an HTTPS connection to the ingestion endpoint. Check outbound access on port 443, DNS resolution of the endpoint, and the `proxy_url` setting if you use a proxy. EU workspaces must use `ingest.eu1.nimbus-monitor.example`.

## NM-1002: invalid ingestion key

The key in `agent.yaml` is wrong, revoked, or belongs to another workspace. Create a new ingestion key and restart the agent.

## NM-4021: clock skew detected

The host clock differs from Nimbus servers by more than 60 seconds, so data points are rejected. Enable time synchronization (chrony or systemd-timesyncd on Linux, the Windows Time service on Windows) and restart the agent.

## NM-4029: payload too large

A single batch exceeded 5 MB, usually because of very high-cardinality custom metrics. Reduce tag cardinality or lower `batch_size` in `agent.yaml`.

## NM-5003: ingestion temporarily unavailable

The ingestion service is degraded. The agent buffers up to 30 minutes of data on disk and retries automatically. Check status.nimbus-monitor.example.

## Host shows as offline but agent is running

Check for NM-1001 or NM-4021 in the agent log. A host is marked offline after 10 minutes without data.
