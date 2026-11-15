---
title: Dashboards
access: public
url: https://docs.nimbus-monitor.example/dashboards
updated: 2026-09-18
---
# Dashboards

A dashboard is a grid of widgets, each backed by one or more metric or log queries.

## Widgets and limits

Widget types are timeseries, top list, single value, heatmap, table and note. A dashboard can hold at most 60 widgets; beyond that, split it into several dashboards and link them with a note widget. Each widget query may return at most 5,000 series.

## Template variables

Template variables add drop-downs at the top of a dashboard that filter every widget by a tag, such as `env` or `service`. The list of values is built from tags seen in the last 24 hours, and a brand-new tag value can take up to 10 minutes to show up in the drop-down.

## Sharing outside your workspace

A public link gives read-only access to one dashboard without a login. Public links expire after 30 days and must be regenerated; the expiry cannot be extended. Admins can turn public links off for the whole workspace under **Settings > Security**.

## Comparing with an earlier period

Enable **Compare to** on a timeseries widget to overlay the same query shifted by an hour, a day, a week or a month. The overlay is drawn as a dashed line.

## Export and import

Every dashboard can be exported as JSON from the cog menu and imported into another workspace. Widget data can be exported to CSV, up to 1 million rows per export.
