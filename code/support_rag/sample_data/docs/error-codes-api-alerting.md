---
title: Error Code Reference: API and Alerting (NM-6xxx and NM-7xxx)
access: public
url: https://docs.nimbus-monitor.example/error-codes-api-alerting
updated: 2026-10-29
---
# Error Code Reference: API and Alerting (NM-6xxx and NM-7xxx)

The most common codes (NM-1001, NM-1002, NM-4021, NM-4029, NM-5003) are explained in Troubleshooting and Error Codes.

## NM-6001: API key expired

The API key had an expiry date and it has passed. Create a new key; expired keys cannot be renewed.

## NM-6003: API key lacks the required scope

The key is read-only and the endpoint writes, or the key is limited to resources that do not include this one.

## NM-6010: query timed out

A metric query ran for more than 30 seconds. Shorten the time range, add a filter, or group by fewer tags.

## NM-6011: query returns too many series

The query would return more than 5,000 series. Add a filter or use a top-N function.

## NM-6020: dashboard widget limit reached

The dashboard already has 60 widgets.

## NM-6102: export too large

A CSV export would exceed 1 million rows. Export a shorter time range.

## NM-7001: notification delivery failed

The receiving service returned an error or did not answer. Delivery is retried up to 3 times; the last error is shown under the monitor's event history.

## NM-7002: notification channel disabled

A channel is disabled automatically after 50 consecutive failed deliveries, so a broken webhook does not delay other notifications. Fix the receiver and re-enable the channel on the integrations page.

## NM-7010: monitor query no longer valid

A metric or tag used by the monitor's query no longer exists, typically after a rename. The monitor shows No Data until the query is updated.

## NM-7012: monitor evaluation delayed

Evaluation is running more than 2 minutes behind. This is a platform-side condition; check the status page.

## NM-7021: escalation policy has nobody on call

An alert tried to page an escalation policy whose current level has no user on call, so nobody was notified. Add a fallback user or team to every level.

## NM-7100: webhook secret rotated

The webhook secret was rotated. For 24 hours, requests are signed with both the old and the new secret so receivers can be updated without missing deliveries.

## NM-7102: webhook endpoint TLS error

The webhook URL presented an expired, self-signed or mismatched certificate. Deliveries fail until the certificate is fixed.
