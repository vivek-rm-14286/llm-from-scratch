---
title: Notification Integrations
access: public
url: https://docs.nimbus-monitor.example/integrations
updated: 2026-08-15
---
# Notification Integrations

Nimbus Monitor sends alert notifications to email, Slack, Microsoft Teams, PagerDuty and generic webhooks.

## Slack

Install the Nimbus app from **Settings > Integrations > Slack** and choose channels. To mention a channel in a monitor message, use `@slack-<channel-name>`. Private channels require inviting the Nimbus app to the channel first.

## PagerDuty

Add a PagerDuty service integration key under **Settings > Integrations > PagerDuty**. Monitors in ALERT state trigger PagerDuty incidents, and recovery automatically resolves them. Mention it in a monitor message as `@pagerduty-<service-name>`.

## Webhooks

Webhooks send an HTTP POST with a JSON payload to your URL. Every webhook request includes the header `X-Nimbus-Signature`, an HMAC-SHA256 of the raw request body using your webhook secret. Verify this signature before trusting the payload. Webhook deliveries time out after 10 seconds and are retried up to 3 times with exponential backoff. Only HTTPS URLs are allowed.

## Email

Email notifications are sent from alerts@nimbus-monitor.example. If emails are missing, check spam filtering and allow-list that address.
