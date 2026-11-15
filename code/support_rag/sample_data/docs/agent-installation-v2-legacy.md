---
title: Installing Nimbus Collector 2.x (Legacy)
access: public
url: https://docs.nimbus-monitor.example/legacy/collector-2x-install
updated: 2024-03-11
---
# Installing Nimbus Collector 2.x (Legacy)

This page describes version 2.x of the agent, then called Nimbus Collector. Version 2.x reached end of support on 2025-06-30. New installations should use Nimbus agent 3.x.

## Linux (2.x)

Add the package repository and install the `nimbus-collector` package:

```
echo "deb https://apt.nimbus-monitor.example stable main" | sudo tee /etc/apt/sources.list.d/nimbus.list
sudo apt update && sudo apt install nimbus-collector
```

Put your key in `/etc/nimbus-collector/collector.conf` as `api_key = <key>` and start the service `nimbus-collector`. Logs are written to `/var/log/nimbus-collector/collector.log`.

## Windows (2.x)

Run the `NimbusCollectorSetup.exe` installer. The service is named `NimbusCollector` and the configuration file is `C:\Program Files\NimbusCollector\collector.conf`. Windows Server 2012 R2 and newer are supported.

## Kubernetes (2.x)

Apply the static manifest `collector-daemonset.yaml` with `kubectl apply -f`. There is no Helm chart for 2.x. Kubernetes 1.19 or newer is required.

## Proxies (2.x)

Collector 2.x reads the standard `HTTP_PROXY` and `HTTPS_PROXY` environment variables. There is no proxy setting in `collector.conf`.

## Upgrading (2.x)

Upgrade between 2.x versions in place with `apt upgrade nimbus-collector`. Moving from 2.x to 3.x is a reinstall: remove `nimbus-collector`, then install `nimbus-agent`. Settings from `collector.conf` are not migrated.
