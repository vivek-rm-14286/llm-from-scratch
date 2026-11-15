---
title: Security Incident Handling (Restricted)
access: restricted
url: https://wiki.nimbus-internal.example/security/incident-process
updated: 2026-06-14
---
# Security Incident Handling (Restricted)

Access limited to the security team and support leads.

## When a customer reports a leaked API key

1. Revoke the key immediately from the admin console using **Force revoke**, even if the customer cannot log in.
2. Pull API audit logs for the key for the last 30 days and attach them to the security ticket.
3. Page the security on-call through PagerDuty service `sec-oncall`.

## Customer data deletion requests

Full workspace deletion requests are processed by the security team within 30 days. Support must not run the deletion job directly; open a ticket in the SEC queue instead.
