---
title: Kubernetes Deployment Guide
access: public
url: https://docs.nimbus-monitor.example/kubernetes
updated: 2026-10-06
---
# Kubernetes Deployment Guide

The Helm chart `nimbus/nimbus-agent` deploys the agent as a DaemonSet (one pod per node) plus an optional cluster agent Deployment.

## Chart values you will usually set

Set `clusterName` so hosts from different clusters can be told apart, and provide the ingestion key either inline with `ingestionKey` or, preferably, from an existing Secret with `existingSecret: <name>` (the Secret must contain the key `ingestion-key`). EU workspaces must also set `endpoint: ingest.eu1.nimbus-monitor.example`.

## Resource requests and limits

Each DaemonSet pod requests 200m CPU and 256Mi memory by default, with a memory limit of 512Mi. On nodes that run more than about 150 pods, raise `resources.limits.memory` to 1Gi; otherwise the kubelet may kill the pod with reason OOMKilled.

## Control-plane and tainted nodes

By default the DaemonSet does not schedule onto tainted nodes, so control-plane nodes and dedicated node pools are not monitored. Add the matching entries under `tolerations` in your values file to cover them. To monitor every node regardless of taints, set `tolerations: [{operator: Exists}]`.

## Cluster agent and cluster-level metrics

Deployment, StatefulSet, Job and node-condition metrics (the `kube_*` family) come from the cluster agent, not from the DaemonSet. Enable it with `clusterAgent.enabled: true`. It runs two replicas with leader election so cluster-level metrics are reported once.

## Excluding namespaces and containers

Use `containerExclude` to stop collecting from noisy workloads, for example `containerExclude: "kube_namespace:kube-system image:pause"`. Excluded containers do not count toward custom metric or log usage.
