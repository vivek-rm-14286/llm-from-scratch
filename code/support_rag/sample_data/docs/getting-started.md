---
title: Getting Started with Nimbus Monitor
access: public
url: https://docs.nimbus-monitor.example/getting-started
updated: 2026-09-02
---
# Getting Started with Nimbus Monitor

Nimbus Monitor is a hosted infrastructure and application monitoring service. It collects host metrics, container metrics and logs through the Nimbus agent, evaluates alert rules, and notifies your team through channels such as email, Slack and PagerDuty.

## Core concepts

- **Workspace**: the top-level container for your organization. Billing, users and API keys belong to a workspace.
- **Agent**: a lightweight process installed on each host or as a Kubernetes DaemonSet. It sends data to the ingestion endpoint `ingest.nimbus-monitor.example` over HTTPS (port 443).
- **Monitor**: an alert rule that evaluates a metric query against a threshold.
- **Dashboard**: a collection of charts built from metric queries.

## First steps

1. Create a workspace at app.nimbus-monitor.example and invite your team under **Settings > Members**.
2. Create an ingestion key under **Settings > API Keys > Ingestion keys**. Ingestion keys can only send data; they cannot read data or change configuration.
3. Install the agent on one host (see Agent Installation) and confirm the host appears under **Infrastructure > Hosts** within about two minutes.
4. Create your first monitor from the **CPU usage** template.

## Supported regions

Workspaces are hosted in either the US region (`us1`) or the EU region (`eu1`). The region is chosen when the workspace is created and cannot be changed later. EU workspaces use the ingestion endpoint `ingest.eu1.nimbus-monitor.example`.
