---
title: Internal Runbook - Ingestion Issues
access: internal
url: https://wiki.nimbus-internal.example/support/ingestion-runbook
updated: 2026-10-28
---
# Internal Runbook: Ingestion Issues (Support Only)

This page is for Nimbus support engineers. Do not share it with customers.

## Checking a workspace's ingestion health

Open the admin console at admin.nimbus-internal.example, search for the workspace ID, and check the **Ingestion** tab. The "rejected points" counter shows data points dropped and the reason (clock skew, invalid key, payload too large).

## Re-driving stuck ingestion

If a workspace shows a growing ingestion backlog for more than 15 minutes, run the `redrive-partition` job for that workspace from the admin console. Only on-call ingestion engineers may run it for more than 3 workspaces at once.

## Temporary rate limit increases

Support may grant a temporary API rate limit increase of up to 2x for 72 hours using the `api_rate_limit_override` feature flag. Larger or permanent increases need approval from the account's Customer Success Manager.

## Escalation

Escalate to the ingestion on-call via the `#ingest-oncall` Slack channel if the backlog is growing for more than 30 minutes or affects more than 10 workspaces.
