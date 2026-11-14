---
title: Service Status and SLA
access: public
url: https://docs.nimbus-monitor.example/sla
updated: 2026-07-30
---
# Service Status and SLA

## Status page

Current and past incidents are published at status.nimbus-monitor.example. Subscribe by email, RSS or webhook to be told when an incident opens, changes or closes. Each region is reported separately.

## Uptime commitment

Enterprise contracts include a monthly uptime commitment of 99.9% for data ingestion and alert evaluation. Scheduled maintenance announced at least 5 days ahead does not count as downtime.

## Service credits

If a month falls below the commitment, the customer can claim a credit: 10% of that month's fee below 99.9%, and 25% below 99.0%. Claims must be made within 30 days of the end of the month through a support ticket.

## Scheduled maintenance

Maintenance is announced on the status page and by email to Admins. During ingestion maintenance the agent buffers data on disk and sends it afterwards, so no data is lost for outages shorter than the buffer limit.
