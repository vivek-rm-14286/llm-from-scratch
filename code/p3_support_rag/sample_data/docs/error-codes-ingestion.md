---
title: Error Code Reference: Ingestion (NM-4xxx and NM-5xxx)
access: public
url: https://docs.nimbus-monitor.example/error-codes-ingestion
updated: 2026-10-29
---
# Error Code Reference: Ingestion (NM-4xxx and NM-5xxx)

The most common codes (NM-1001, NM-1002, NM-4021, NM-4029, NM-5003) are explained in Troubleshooting and Error Codes.

## NM-4012: timestamp too old

Data points older than 4 hours are rejected. This happens when a host resumes from suspend or a buffer is replayed after a long outage. Older data can be loaded through the backfill API, which accepts points up to 7 days old.

## NM-4020: timestamp in the future

Data points stamped more than 10 minutes ahead of server time are rejected. Check the host's time zone and hardware clock.

## NM-4022: invalid metric name

The metric name does not follow the naming rules: it must start with a letter and use only lowercase letters, digits, underscores and dots, up to 200 characters. The data point is dropped.

## NM-4030: tag value too long

Tag values longer than 200 characters are cut to 200 characters. The data point is still accepted.

## NM-4031: too many tags on a data point

A single data point may carry at most 50 tags. Extra tags are dropped; the data point is still accepted.

## NM-4040: custom metric allotment reached

The workspace has used its whole pool of custom metric series. On Free workspaces new series are dropped; on paid workspaces they are accepted and billed as overage.

## NM-4041: series limit for one metric name reached

One metric name has more than 50,000 series. New series for that name are dropped until the count falls. The cause is nearly always a tag with unbounded values.

## NM-4102: log line too large

Log lines over 256 KB are truncated to 256 KB and tagged `truncated:true`.

## NM-4120: daily log quota reached

The workspace reached its daily indexed-log quota. Logs are dropped until 00:00 UTC. An Admin can raise the quota.

## NM-4201: ingestion key rate limited

One ingestion key sent more than 10,000 data points per second. The agent backs off and retries. Use separate keys for separate fleets.

## NM-4210: ingestion key disabled

The key was disabled by a workspace Admin. Data sent with it is rejected with HTTP 403. Re-enable the key or deploy a new one.

## NM-5001: wrong region for this key

The ingestion key belongs to a workspace in another region. The server answers HTTP 421 and names the correct region. Change `endpoint` in `agent.yaml` to that region's endpoint.

## NM-5002: ingestion request timed out

A batch was not acknowledged within 30 seconds, usually because of a slow or lossy network path. The agent keeps the batch and retries.

## NM-5013: workspace suspended

The workspace is suspended for non-payment and all data is rejected with HTTP 402. Ingestion resumes within an hour of payment.

## NM-5030: ingestion under maintenance

The region is in a scheduled maintenance period. The agent buffers on disk and sends the data when maintenance ends.
