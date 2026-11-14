---
title: On-Call Schedules and Escalation Policies
access: public
url: https://docs.nimbus-monitor.example/on-call
updated: 2026-10-15
---
# On-Call Schedules and Escalation Policies

Nimbus can page people directly, without PagerDuty, using schedules and escalation policies.

## Schedules and rotations

A schedule is made of one or more rotations. Each rotation lists people and a hand-off time, daily or weekly. To cover someone for a few hours, add an override; the override takes priority over the rotation and ends automatically.

## Escalation policies

An escalation policy is an ordered list of levels. When an alert pages the policy, level 1 is notified first. If nobody acknowledges within the acknowledgment timeout (default 15 minutes), the next level is notified, and so on. The last level can repeat the whole policy up to 3 times.

## Using a policy from a monitor

Mention a policy in the monitor message with `@oncall-<policy-name>`. A policy whose current level has nobody on call cannot page anyone, so every level should end with a fallback user or team.

## Quiet hours

Low-priority alerts can be held during quiet hours and delivered as a digest when quiet hours end. High-priority alerts always page immediately.

## Acknowledge and resolve

Acknowledging stops further escalation but keeps the alert open. The alert resolves by itself when the monitor recovers, or it can be resolved by hand.
