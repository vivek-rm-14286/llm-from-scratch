---
title: Internal - Current Known Issues
access: internal
url: https://wiki.nimbus-internal.example/support/known-issues
updated: 2026-11-10
---
# Current Known Issues (Support Only)

Internal page. Share the workaround with customers, not the issue number or the internal details.

## KI-204: Slack notifications stop after a Slack workspace rename

Renaming the Slack workspace invalidates the app token. Notifications fail silently until an Admin reinstalls the Nimbus Slack app from the integrations page. A fix is planned for the December release.

## KI-211: CSV exports truncated at 100,000 rows in eu1

In the EU region, widget CSV exports stop at 100,000 rows instead of 1 million. Workaround: export shorter time ranges, or use the REST API. Engineering ticket ENG-8841.

## KI-219: False failures from the Sydney synthetic location

The Sydney location intermittently reports timeouts for healthy endpoints because of an upstream network provider. Workaround: require failures from 2 locations, or remove Sydney from the check until resolved.

## KI-223: SCIM group changes delayed

Group membership changes pushed over SCIM can take up to 40 minutes to apply to roles. User deactivation is not affected.

## KI-230: Forecast monitors ignore maintenance windows

Forecast monitors still notify during a maintenance window. Workaround: mute the monitor by hand for the duration of the work.
