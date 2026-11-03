---
title: Incident Response Runbook
last_updated: 2026-07-02
---

# Incident Response Runbook

This runbook describes how Aurora Labs detects, handles and learns from production incidents.

## Severity levels

SEV1 means a full outage or data loss affecting many customers. The on-call engineer must acknowledge a SEV1 page within 5 minutes and an incident commander is appointed immediately. SEV2 is a major feature outage or severe degradation, acknowledged within 15 minutes. SEV3 is a minor degradation with a workaround, handled during business hours. SEV4 is a cosmetic issue or a request that is tracked as a normal ticket.

## Roles

Every SEV1 and SEV2 incident has three roles. The incident commander coordinates the response and makes decisions. The communications lead posts updates to the status page and to affected customers. The scribe records a timeline of actions and decisions in the incident channel.

## Communication cadence

For a SEV1 the status page is updated every 30 minutes until the incident is resolved, even if there is no new information. For a SEV2 updates go out every hour. Internal updates are posted in the incident channel as events happen.

## Resolution and follow-up

An incident is resolved only when customer impact has ended and monitoring has been stable for 30 minutes. A blameless postmortem must be written within 5 business days of any SEV1 or SEV2. The postmortem lists contributing factors, what went well, and action items with named owners and due dates.

## Escalation

If the on-call engineer does not acknowledge a page within the target time, it automatically escalates to the secondary on-call and then to the engineering manager.
