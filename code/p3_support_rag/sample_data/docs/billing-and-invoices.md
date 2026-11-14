---
title: Invoices, Payments and Usage
access: public
url: https://docs.nimbus-monitor.example/billing
updated: 2026-10-12
---
# Invoices, Payments and Usage

## Invoices and payment methods

Invoices are issued on the 1st of each month for the previous month's usage. Monthly plans are paid by credit card; annual Enterprise contracts can pay by bank transfer against an invoice with 30-day terms.

## Usage beyond what the plan includes

Hosts above the number included in a Team plan are charged at the per-host overage rate shown under **Settings > Usage**. Custom metrics are pooled: each billed host adds 100 custom metric series to the workspace allotment, and series above the pool are billed per 100 series.

## Cancelling

Cancel from **Settings > Plan**. The workspace stays active until the end of the current billing period and then moves to the Free plan. Usage in the final period is still invoiced on the 1st of the following month, so one more invoice after cancelling is expected.

## Unpaid invoices

If an invoice is unpaid for 15 days, admins see a warning banner. After 30 days the workspace is suspended: data sent by agents is rejected and monitors stop evaluating. Paying the invoice lifts the suspension within an hour.

## Changing the billing contact

Invoices are emailed to the billing contact. An Admin or a member with the Billing role can change it under **Settings > Plan > Billing contact**.
