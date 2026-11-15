---
title: Internal - Support Severity and Escalation Matrix
access: internal
url: https://wiki.nimbus-internal.example/support/escalation-matrix
updated: 2026-10-21
---
# Support Severity and Escalation Matrix (Support Only)

Internal page. Do not share with customers.

## Severity levels

S1: production monitoring is down or data is being lost for an Enterprise customer; first response within 15 minutes, around the clock. S2: a major feature is unusable with no workaround; first response within 1 hour in business hours. S3: everything else; first response within 8 business hours.

## Who to page

Agent and Kubernetes problems go to `#agent-oncall`. Ingestion delays and data loss go to `#ingest-oncall`. Alert evaluation or notification delays go to `#alerting-oncall`. Sign-in, SAML and SCIM problems go to `#identity-oncall`. Page only for S1; for S2 post in the channel without paging.

## Involving the Customer Success Manager

Tell the account's CSM about every S1, about any ticket open for more than 5 business days, and whenever the customer mentions cancelling or a contract renewal.

## Credits and refunds

Support engineers may grant a goodwill credit of up to USD 500 per incident. Anything larger, and every refund to a card, needs approval from the billing lead in `#billing-approvals`. SLA credits follow the contract and are calculated by the billing team, not by support.
