---
title: Classic Alert Rules (Deprecated)
access: public
url: https://docs.nimbus-monitor.example/legacy/classic-alert-rules
updated: 2024-01-22
---
# Classic Alert Rules (Deprecated)

Classic alert rules are the alerting system that existed before monitors. They can no longer be created, and existing rules stop evaluating on 2027-03-31. Migrate them with **Monitors > Import classic rules**.

## Evaluation (classic)

A classic rule is evaluated every 5 minutes and fires as soon as one evaluation breaches the threshold. There is no evaluation window and no warning threshold.

## Mute schedules (classic)

Classic rules are silenced with mute schedules, created under **Alerts > Mute schedules**. A mute schedule applies to named rules, not tags, lasts at most 24 hours and cannot recur. A muted rule is not evaluated at all.

## Flapping (classic)

Classic rules have no recovery threshold. The only way to reduce flapping is the **Renotify after** setting, which limits repeat notifications to one every 30 minutes.

## Notifications (classic)

Classic rules can notify email addresses and one Slack channel per rule. PagerDuty, Microsoft Teams and webhooks are not supported.
