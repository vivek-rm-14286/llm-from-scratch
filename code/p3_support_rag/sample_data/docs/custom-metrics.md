---
title: Custom Metrics with StatsD
access: public
url: https://docs.nimbus-monitor.example/custom-metrics
updated: 2026-09-22
---
# Custom Metrics with StatsD

Applications send their own metrics to the agent's built-in StatsD server over UDP.

## Sending metrics

The agent listens on UDP port 8125 on localhost. Supported types are counter (`c`), gauge (`g`), histogram (`h`) and set (`s`). Tags are appended after `|#`, for example `checkout.orders:1|c|#env:prod,region:eu`.

## Naming

Metric names must start with a letter and contain only lowercase letters, digits, underscores and dots, up to 200 characters. Names that do not fit are rejected.

## What counts as one custom metric

Each unique combination of metric name and tag values is one series, and each series counts as one custom metric. A histogram produces five series per tag combination (avg, count, max, median, p95).

## Keeping the series count under control

Never use unbounded values such as user IDs, request IDs, email addresses or timestamps as tag values: every new value creates a new series. Prefer a small fixed set of values such as status class (`2xx`, `5xx`) or region. A single metric name may have at most 50,000 series.

## Receiving StatsD from other hosts or containers

By default the server only accepts packets from localhost. To accept metrics from containers on the same node, set `statsd_non_local: true`.
