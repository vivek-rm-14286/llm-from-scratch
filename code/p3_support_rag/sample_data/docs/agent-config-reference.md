---
title: Agent Configuration Reference (agent.yaml)
access: public
url: https://docs.nimbus-monitor.example/agent-config-reference
updated: 2026-10-25
---
# Agent Configuration Reference (agent.yaml)

Every setting in `/etc/nimbus/agent.yaml` for agent 3.4. Restart the agent after changing the file.

## ingestion_key

`ingestion_key`: The ingestion key used to authenticate the agent. Required. No default.

## endpoint

`endpoint`: Hostname of the ingestion endpoint. Default `ingest.nimbus-monitor.example`. EU workspaces set `ingest.eu1.nimbus-monitor.example`.

## proxy_url

`proxy_url`: HTTP proxy for all outbound traffic, for example `http://proxy.internal:3128`. Empty by default, meaning a direct connection.

## ca_bundle_path

`ca_bundle_path`: Path to a PEM file of extra root certificates to trust, used when a proxy re-signs TLS traffic. Empty by default, meaning the system trust store.

## tls_min_version

`tls_min_version`: Lowest TLS version the agent will negotiate. Default `1.2`. Set `1.3` to refuse TLS 1.2.

## batch_size

`batch_size`: Maximum number of data points per request. Default 500. Lower it if batches are rejected as too large.

## flush_interval_s

`flush_interval_s`: How often, in seconds, collected data is sent. Default 10. Higher values mean fewer requests and more delay before data is visible.

## collection_interval_s

`collection_interval_s`: How often, in seconds, host and integration metrics are sampled. Default 15. The minimum is 5.

## buffer_max_mb

`buffer_max_mb`: Maximum disk space, in megabytes, for data waiting to be sent while the endpoint is unreachable. Default 500. When it is full the oldest data is dropped.

## buffer_max_age_s

`buffer_max_age_s`: Maximum age, in seconds, of buffered data. Default 1800 (30 minutes). Data older than this is discarded even if there is disk space left.

## max_retries

`max_retries`: How many times a failed request is retried before the batch goes to the disk buffer. Default 5.

## retry_backoff_max_s

`retry_backoff_max_s`: Upper bound, in seconds, for the wait between retries. Default 300. Waits double from 1 second up to this value.

## memory_limit_mb

`memory_limit_mb`: Memory ceiling for the agent process in megabytes. Default 256. The agent restarts itself when it is exceeded.

## check_timeout_s

`check_timeout_s`: Maximum run time, in seconds, of one integration check. Default 15. A check that exceeds it is cancelled.

## hostname_override

`hostname_override`: Name to report instead of the operating system hostname. Empty by default. Needed when several machines share a hostname, such as cloned VMs.

## tags

`tags`: List of `key:value` tags added to everything this agent sends, for example `[env:prod, team:payments]`. At most 100.

## log_level

`log_level`: Verbosity of the agent's own log: `error`, `warn`, `info` or `debug`. Default `info`.

## log_file

`log_file`: Where the agent writes its own log. Default `/var/log/nimbus/agent.log`. The file is rotated at 50 MB and 5 files are kept.

## logs.enabled

`logs.enabled`: Turns log collection on. Default `false`.

## logs.paths

`logs.paths`: List of files or glob patterns to tail, for example `[/var/log/app/*.log]`. Only read when `logs.enabled` is true.

## logs.exclude_patterns

`logs.exclude_patterns`: List of regular expressions. Lines that match any of them are dropped on the host and never sent.

## logs.multiline_pattern

`logs.multiline_pattern`: Regular expression that marks the first line of a log entry. Lines that do not match are appended to the previous entry, which keeps stack traces together. Empty by default: every line is its own entry.

## redact_patterns

`redact_patterns`: List of regular expressions whose matches are replaced with `[REDACTED]` in logs and tag values before they leave the host.

## network_tracing

`network_tracing`: Collects per-connection network metrics with eBPF. Default `true` on Linux kernels 5.4 and newer.

## container_discovery

`container_discovery`: Finds running containers and collects their metrics and labels. Default `true` when a container runtime socket is present.

## statsd_port

`statsd_port`: UDP port of the built-in StatsD server. Default 8125. Set 0 to turn the server off.

## statsd_non_local

`statsd_non_local`: Accept StatsD packets from other hosts and containers, not only localhost. Default `false`.
