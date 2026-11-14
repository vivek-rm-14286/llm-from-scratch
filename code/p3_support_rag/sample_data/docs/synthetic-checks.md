---
title: Synthetic Checks
access: public
url: https://docs.nimbus-monitor.example/synthetics
updated: 2026-08-27
---
# Synthetic Checks

Synthetic checks probe your endpoints from Nimbus-managed locations so you find out about an outage before your users do.

## HTTP checks

An HTTP check requests a URL on a schedule and asserts on status code, response time, headers or body content. Checks run from up to 8 managed locations. The minimum interval is 1 minute on Team and Enterprise plans and 5 minutes on the Free plan.

## Reducing false alarms

A single location can fail because of a network problem near that location. Set **Alert when N locations fail** to 2 or more so one bad probe does not page anyone, and enable **Retry once before failing**.

## Certificate expiry

Every HTTPS check also records the certificate's expiry date. The built-in certificate monitor alerts 14 days before expiry by default; the lead time can be set between 1 and 60 days.

## Multi-step API tests

A multi-step test chains up to 10 requests and can pass values, such as a token from a login response, from one step to the next. Secrets used in steps are stored as encrypted global variables and are hidden in results.

## Private locations

To test endpoints that are not reachable from the internet, run a private location: a container inside your network that pulls test definitions over outbound HTTPS. Private locations are available on the Enterprise plan.
