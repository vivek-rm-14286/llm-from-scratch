---
title: nimbusctl Command Reference
access: public
url: https://docs.nimbus-monitor.example/nimbusctl-reference
updated: 2026-10-25
---
# nimbusctl Command Reference

`nimbusctl` is installed with the agent and must be run as root or as the `nimbus` user.

## nimbusctl status

Prints whether the agent is running, its version, the time of the last successful flush and the number of data points waiting in the buffer. Exit code 0 means healthy.

## nimbusctl config validate

Parses `agent.yaml` and reports syntax errors with line numbers and unknown keys. It does not contact the server and does not restart the agent.

## nimbusctl config show --redact

Prints the effective configuration after defaults and environment variables are applied. With `--redact` the ingestion key and proxy password are masked, so the output is safe to paste into a ticket.

## nimbusctl selftest --endpoint

Tests DNS, TCP, TLS and authentication against the ingestion endpoint, one step at a time, and prints the first step that fails. `--endpoint <host>` tests a different endpoint than the configured one.

## nimbusctl diagnose --bundle

Runs every diagnostic and, with `--bundle`, writes the results, the redacted configuration and the last 24 hours of agent logs to a tar.gz file in the current directory.

## nimbusctl check run --dry-run

Runs one integration check once, by name, and prints the metrics it produced. With `--dry-run` nothing is sent to Nimbus, which makes it the way to test a new check configuration.

## nimbusctl flush --force

Sends everything collected so far right away instead of waiting for the next interval. `--force` also retries batches that are currently backing off.

## nimbusctl buffer stats

Shows the size of the on-disk buffer, the age of the oldest entry and how much was dropped since the agent started.

## nimbusctl buffer purge --older-than

Deletes buffered data. `--older-than <duration>`, for example `--older-than 2h`, removes only entries older than that; without it the whole buffer is removed. Purged data is never sent.

## nimbusctl key rotate

Replaces the ingestion key in `agent.yaml` with a new one given on the command line and reloads the agent without dropping buffered data.

## nimbusctl logs tail --level

Follows the agent's own log. `--level warn` shows only warnings and errors.

## nimbusctl integration list

Lists available integrations and whether each is enabled on this host.

## nimbusctl integration enable

Turns an integration on by creating its configuration file from the template under `/etc/nimbus/integrations.d/` and reloading the agent.

## nimbusctl restart --graceful

Restarts the agent. With `--graceful` it flushes pending data first and waits up to 30 seconds for the flush to finish.

## nimbusctl version --check-update

Prints the installed version. With `--check-update` it also reports whether a newer stable version exists; it never installs anything.

## nimbusctl uninstall --purge

Removes the agent. With `--purge` it also deletes `agent.yaml`, the buffer and the logs.

## Global flags

`--config <path>` reads a configuration file other than `/etc/nimbus/agent.yaml`. `--json` prints machine-readable output. `--timeout <seconds>` bounds any command that talks to the agent or the server; the default is 20.
