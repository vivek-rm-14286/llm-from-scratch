---
title: Internal Runbook - Billing Adjustments
access: internal
url: https://wiki.nimbus-internal.example/support/billing-adjustments
updated: 2026-10-17
---
# Internal Runbook: Billing Adjustments (Support Only)

Internal page. Do not share with customers.

## Extending a trial

A trial can be extended once, by at most 14 days, from the workspace page in the admin console (**Plan > Extend trial**). A second extension needs the sales owner's approval.

## Host-count disputes

When a customer disputes the billed host count, export the hourly host report for the month from the admin console (**Usage > Hourly hosts > Export**) and show them the 99th-percentile hour. The most common cause is hosts reporting under two names after a hostname change.

## Pausing a suspension

If a customer is about to be suspended for non-payment and has a payment in progress, set the `billing_hold` flag on the workspace. It delays suspension by 7 days and can be set only once per invoice.

## Applying a credit

Credits are applied from **Billing > Adjustments** in the admin console and appear on the next invoice. Enter the ticket number in the reason field. Limits on who can approve which amount are in the escalation matrix.
