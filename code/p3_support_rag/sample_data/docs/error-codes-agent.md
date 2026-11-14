---
title: Error Code Reference: Agent (NM-1xxx and NM-2xxx)
access: public
url: https://docs.nimbus-monitor.example/error-codes-agent
updated: 2026-10-29
---
# Error Code Reference: Agent (NM-1xxx and NM-2xxx)

The most common codes (NM-1001, NM-1002, NM-4021, NM-4029, NM-5003) are explained in Troubleshooting and Error Codes.

## NM-1003: ingestion key missing

The agent started without an ingestion key. Set `ingestion_key` in `agent.yaml`, or pass `NIMBUS_KEY` to the install script, then restart the agent.

## NM-1004: TLS certificate verification failed

The certificate presented for the ingestion endpoint is not trusted by the host. This almost always means a corporate proxy is inspecting TLS traffic. Point `ca_bundle_path` at a bundle that contains the proxy's root certificate.

## NM-1010: DNS resolution failed

The host could not resolve the name of the ingestion endpoint. Check `/etc/resolv.conf` and whether the host can resolve external names at all.

## NM-1011: proxy authentication required

The HTTP proxy answered 407. Put the credentials in `proxy_url`, in the form `http://user:password@proxy.internal:3128`.

## NM-1100: configuration file could not be parsed

`agent.yaml` is not valid YAML, usually because of tabs or wrong indentation. Run `nimbusctl config validate` to see the line number.

## NM-1101: unknown configuration key

A key in `agent.yaml` is not recognized and was ignored. This is a warning; the agent keeps running. It is usually a typo or a key from Collector 2.x.

## NM-1110: configuration file permissions too open

`agent.yaml` is readable by other users. Because it contains the ingestion key, the agent refuses to start until the file is owned by `nimbus` with mode 0640.

## NM-2001: disk buffer full

The on-disk buffer reached `buffer_max_mb` while the agent could not send. The oldest buffered data is dropped first. Fix the connection problem or raise `buffer_max_mb`.

## NM-2002: disk buffer corrupted

The buffer directory could not be read, typically after a disk-full event or an unclean shutdown. Stop the agent, delete `/var/lib/nimbus/buffer`, and start it again. Buffered data is lost.

## NM-2010: agent memory limit exceeded

The agent's memory use passed `memory_limit_mb` (default 256) and it restarted itself. Occasional restarts on very busy hosts are harmless; repeated ones mean the limit is too low for the workload.

## NM-2012: check timed out

An integration check ran longer than `check_timeout_s` (default 15 seconds) and was cancelled. The other checks are not affected. Raise the timeout for slow targets or fix the target.

## NM-2021: kernel does not support network tracing

Network tracing uses eBPF and needs Linux kernel 5.4 or newer. On older kernels set `network_tracing: false` to silence the error.

## NM-2102: container runtime socket not accessible

The agent cannot read the container runtime socket, so container metrics are missing. Add the `nimbus` user to the `docker` group, or mount the socket into the agent container.

## NM-2120: log file not readable

A file listed under `logs.paths` cannot be opened by the `nimbus` user. Grant read permission on the file and execute permission on its parent directories.

## NM-2201: hostname conflict

Two agents are reporting under the same hostname, so their data is merged and the host seems to flip between two sets of values. Give one of them a unique `hostname_override`.

## NM-2210: host tag limit exceeded

A host may carry at most 100 tags. Tags beyond the limit are dropped in alphabetical order.
