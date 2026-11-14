---
title: REST API v1 Reference (Deprecated)
access: public
url: https://docs.nimbus-monitor.example/legacy/api-v1
updated: 2024-05-02
---
# REST API v1 Reference (Deprecated)

API v1 is deprecated and will be switched off on 2027-01-31. Use API v2 for all new work. The v1 base URL is `https://api.nimbus-monitor.example/v1`.

## Authentication (v1)

Send the key in the `X-Api-Key` header. Keys in v1 are not scoped: every key can read and write.

## Rate limits (v1)

API v1 allows 300 requests per minute per key. Requests over the limit receive HTTP 429 without a `Retry-After` header; wait 60 seconds before retrying.

## Pagination (v1)

List endpoints use page numbers: pass `page` and `per_page` (maximum 50). The response contains `total_pages`.

## Migrating to v2

Replace the `X-Api-Key` header with `Authorization: Bearer <key>`, switch from page numbers to the `cursor` parameter, and create scoped keys. v1 keys continue to work with v2.
