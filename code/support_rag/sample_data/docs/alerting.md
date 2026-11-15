---
title: Alerting and Monitors
access: public
url: https://docs.nimbus-monitor.example/alerting
updated: 2026-09-20
---
# Alerting and Monitors

A monitor evaluates a metric query every evaluation interval (default 60 seconds) and changes state between OK, WARN and ALERT.

## Thresholds

Each monitor has an alert threshold and an optional warning threshold. A monitor only changes state after the condition has held for the **evaluation window** (default 5 minutes). Short spikes shorter than the window do not trigger alerts.

## Recovery and flapping

To reduce flapping, set a **recovery threshold** that is lower than the alert threshold. For example, alert at 90% CPU but recover only below 80%.

## Maintenance windows

Maintenance windows silence notifications during planned work. Create one under **Monitors > Maintenance Windows**, choose a start and end time, and scope it by tag (for example `env:staging` or `service:checkout`). Monitors keep evaluating during a maintenance window, but no notifications are sent. Maintenance windows can be at most 7 days long, and recurring windows can be defined with a cron expression.

## No-data alerts

Enable **Notify on no data** to alert when a metric stops reporting for longer than the no-data timeframe (default 10 minutes). This catches crashed agents or broken pipelines.

## Alert grouping

Multi-alert monitors create one alert per group (for example per host). Use **Group notifications** to send a single notification when many groups trigger within 5 minutes.
