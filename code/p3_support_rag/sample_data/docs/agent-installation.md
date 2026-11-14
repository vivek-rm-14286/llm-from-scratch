---
title: Installing the Nimbus Agent
access: public
url: https://docs.nimbus-monitor.example/agent/install
updated: 2026-10-11
---
# Installing the Nimbus Agent

The Nimbus agent (`nimbus-agent`) runs on Linux, Windows and Kubernetes. The current stable version is 3.4.

## Linux

Run the install script with your ingestion key:

```
curl -fsSL https://get.nimbus-monitor.example/install.sh | sudo NIMBUS_KEY=<ingestion-key> bash
```

The agent runs as the systemd service `nimbus-agent`. Its configuration file is `/etc/nimbus/agent.yaml` and logs are written to `/var/log/nimbus/agent.log`.

## Windows

Download the MSI installer from **Settings > Agents** and run it as Administrator. The agent runs as the Windows service `NimbusAgent`. The configuration file is `C:\ProgramData\Nimbus\agent.yaml`. Windows Server 2016 and newer are supported.

## Kubernetes

Install the Helm chart, which deploys the agent as a DaemonSet:

```
helm repo add nimbus https://charts.nimbus-monitor.example
helm install nimbus-agent nimbus/nimbus-agent --set ingestionKey=<ingestion-key>
```

The chart requires Kubernetes 1.26 or newer.

## Proxies

If hosts reach the internet through an HTTP proxy, set `proxy_url` in `agent.yaml`, for example `proxy_url: http://proxy.internal:3128`. The agent does not read the `HTTPS_PROXY` environment variable.

## Upgrading

On Linux, upgrade with your package manager (`apt upgrade nimbus-agent` or `yum update nimbus-agent`). Agents older than version 3.0 cannot be upgraded in place and must be reinstalled.
