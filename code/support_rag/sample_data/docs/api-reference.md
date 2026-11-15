---
title: REST API Reference
access: public
url: https://docs.nimbus-monitor.example/api
updated: 2026-10-01
---
# REST API Reference

The REST API base URL is `https://api.nimbus-monitor.example/v2` (EU: `https://api.eu1.nimbus-monitor.example/v2`).

## Authentication

Send an API key in the `Authorization: Bearer <key>` header. API keys are created under **Settings > API Keys** and can be scoped to read-only or read-write. Ingestion keys are not accepted by the REST API.

## Rate limits

The API allows 600 requests per minute per workspace. When you exceed the limit you receive HTTP 429 with a `Retry-After` header giving the number of seconds to wait. Enterprise plans can request higher limits.

## Pagination

List endpoints return at most 100 items per page. Use the `next_cursor` value from the response as the `cursor` query parameter to fetch the next page.

## Common error responses

- `401 Unauthorized`: missing or invalid API key.
- `403 Forbidden`: the key does not have the required scope (for example a read-only key calling a write endpoint).
- `422 Unprocessable Entity`: the request body failed validation; the response lists the invalid fields.
- `429 Too Many Requests`: rate limit exceeded.
