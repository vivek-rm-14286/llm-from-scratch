---
title: Log Management
access: public
url: https://docs.nimbus-monitor.example/logs
updated: 2026-10-02
---
# Log Management

The agent tails files and container output and ships log lines to Nimbus, where pipelines parse and index them.

## Enabling log collection

Log collection is off by default. Set `logs.enabled: true` in `agent.yaml` and list files under `logs.paths`. On Kubernetes, set `logs.enabled: true` in the chart values to collect stdout and stderr of every container.

## Pipelines and parsing rules

A pipeline is an ordered list of processors applied to logs that match a filter. Grok parsing rules extract fields from unstructured lines; JSON logs are parsed automatically. Fields extracted by a pipeline can be used in searches, monitors and dashboards.

## Exclusion filters

Exclusion filters drop matching logs after ingestion but before indexing. Excluded logs do not count toward your indexed log volume, so filters on debug-level or health-check lines are the main way to bring a log bill down. A filter can also keep a sample, for example 10% of matching lines.

## Daily quota

Each workspace has a daily indexed-log quota that resets at 00:00 UTC. When it is reached, further logs are dropped until the reset. Admins can raise the quota under **Settings > Usage**.

## Scrubbing sensitive data

Use `redact_patterns` in `agent.yaml` to mask values before they leave the host, or add a scrubbing processor to a pipeline to mask them at ingestion. Agent-side redaction is preferable because the raw value never leaves your network.

## Archives

Enterprise workspaces can archive all ingested logs, including excluded ones, to their own S3 bucket and re-index a time range later (rehydration).
