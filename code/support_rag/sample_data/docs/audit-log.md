---
title: Audit Log
access: public
url: https://docs.nimbus-monitor.example/audit-log
updated: 2026-08-19
---
# Audit Log

The audit log records who did what in a workspace.

## Recorded events

Sign-ins, failed sign-ins, role changes, API key creation and revocation, and every create, update or delete of a monitor, dashboard, integration or maintenance window are recorded with the actor, time, source IP address and a before/after diff.

## Viewing and retention

Admins can browse the audit log under **Settings > Audit Log**. Events are kept for 90 days on the Team plan and 13 months on the Enterprise plan. The Free plan has no audit log.

## Exporting to a SIEM

Enterprise workspaces can stream audit events to an HTTPS endpoint or an S3 bucket in near real time. Events can also be fetched through the REST API endpoint `GET /v2/audit/events`.

## Finding who changed a monitor

Open the monitor, choose **History**, and select a version to see the diff and the member or API key that made the change. A change made with an API key shows the key's name and the service account that owns it.
