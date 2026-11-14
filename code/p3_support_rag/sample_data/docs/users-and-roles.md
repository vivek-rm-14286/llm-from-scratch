---
title: Users, Roles and Sign-in
access: public
url: https://docs.nimbus-monitor.example/users-and-roles
updated: 2026-09-30
---
# Users, Roles and Sign-in

## Roles

There are four roles. **Admin** can change every setting. **Editor** can create and edit monitors, dashboards and integrations. **Viewer** is read-only. **Billing** can see invoices and change the payment method but cannot see monitoring data.

## Two-factor authentication

Admins can require two-factor authentication for all password-based sign-ins under **Settings > Security**. Members who have not enrolled are forced to enroll at their next sign-in. Sessions expire after 12 hours of inactivity.

## SAML sign-in

Enterprise workspaces can connect a SAML 2.0 identity provider. The NameID must be the user's email address, and the identity provider's clock must be within 3 minutes of real time or assertions are rejected. Users are created on first sign-in with the default role chosen in the SAML settings.

## Automatic provisioning with SCIM

With SCIM enabled, users and group memberships are created, updated and deactivated from the identity provider. Deactivating a user in the identity provider ends their Nimbus sessions within 5 minutes.

## Service accounts

API keys belong to the workspace, not to a person, but each key records who created it. For automation, create a service account and issue keys from it so that keys keep working when an employee leaves.

## Removing a member

Removing a member revokes their sessions immediately. Dashboards and monitors they created stay in the workspace.
