---
title: Plans, Billing and Data Retention
access: public
url: https://docs.nimbus-monitor.example/plans
updated: 2026-07-30
---
# Plans, Billing and Data Retention

| Plan | Metric retention | Log retention | Hosts included |
|---|---|---|---|
| Free | 7 days | 3 days | 5 |
| Team | 30 days | 15 days | 50 |
| Enterprise | 15 months | 30 days (extendable to 365) | Custom |

## Billing

Team plans are billed monthly per host, based on the 99th percentile of hourly host count, so short autoscaling bursts do not increase your bill. Containers are not billed separately.

## Upgrading and downgrading

Upgrades take effect immediately. Downgrades take effect at the end of the billing period. When you downgrade, data older than the new plan's retention period is deleted and cannot be recovered.

## SSO

SAML single sign-on is available on the Enterprise plan only.
