---
title: Advanced Monitor Types
access: public
url: https://docs.nimbus-monitor.example/advanced-monitors
updated: 2026-10-08
---
# Advanced Monitor Types

Threshold monitors are covered in Alerting and Monitors. This page covers the other monitor types.

## Anomaly monitors

An anomaly monitor learns the usual daily and weekly shape of a metric and alerts when the value leaves the expected band. It needs at least 2 weeks of history for the metric before it can be enabled, and it suits metrics with a regular rhythm, such as request volume.

## Forecast monitors

A forecast monitor extrapolates a trend and alerts when the metric is predicted to cross a threshold within a chosen horizon of up to 7 days. The common use is disk space: alert when a volume is expected to fill within 48 hours.

## Composite monitors

A composite monitor combines up to 10 existing monitors with boolean logic, such as `A && !B`. Use it to page only when error rate is high and a deploy is not in progress, or when several symptoms occur together.

## SLO monitors

Define a service level objective as a target percentage over 7, 30 or 90 days. A burn-rate alert fires when the error budget is being consumed too fast, for example 14.4 times the sustainable rate over one hour.

## Outlier monitors

An outlier monitor compares the members of a group with each other and alerts when one behaves differently from its peers, such as one host in a pool with far higher latency.
