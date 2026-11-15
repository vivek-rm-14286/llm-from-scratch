---
title: Security and Data Residency
access: public
url: https://docs.nimbus-monitor.example/security
updated: 2026-09-09
---
# Security and Data Residency

## Where data is stored

Data for `us1` workspaces is stored in the United States and data for `eu1` workspaces in the European Union. Data is never copied between regions, including backups.

## Encryption

Data is encrypted in transit with TLS 1.2 or newer and at rest with AES-256. Ingestion keys and API keys are stored hashed and are shown only once, when they are created.

## Compliance reports

Nimbus Monitor holds a SOC 2 Type II report and ISO 27001 certification. Customers on any paid plan can download the reports from **Settings > Security > Compliance**.

## Firewall allow-listing

Agents only make outbound connections. If your firewall restricts egress by destination address, allow the ranges published at `https://ip-ranges.nimbus-monitor.example/ranges.json`. The ranges can change; the file is updated at least 14 days before a new range is used.

## What happens to data when a workspace is closed

When a workspace is closed, its data is deleted within 30 days. Backups that contain the data expire within a further 35 days.
