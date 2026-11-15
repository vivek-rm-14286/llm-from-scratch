---
title: Key Compromise Forensics (Restricted)
access: restricted
url: https://wiki.nimbus-internal.example/security/key-compromise-forensics
updated: 2026-08-05
---
# Key Compromise Forensics (Restricted)

Access limited to the security team and support leads. Follows on from Security Incident Handling.

## Deciding whether a leaked key was misused

Query the internal SIEM `seclake` for all requests made with the key ID in the exposure period. Signs of misuse are requests from IP addresses or ASNs the customer has never used, bulk reads of dashboards or monitor definitions, and creation of new API keys or webhooks.

## If misuse is found

Revoke every key and webhook created with the compromised key, open a P1 security incident, and notify the customer's Admins in writing within 72 hours using the template `SEC-NOTIFY-02`. The security lead, not support, sends the notification.

## If no misuse is found

Record the query and its result in the security ticket, tell the customer that the audit showed no use from unknown addresses, and close the ticket after the customer confirms the key has been replaced.
