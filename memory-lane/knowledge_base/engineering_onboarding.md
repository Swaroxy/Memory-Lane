---
title: Engineering Onboarding Guide
last_updated: 2026-08-20
---

# Engineering Onboarding Guide

Welcome to Aurora Labs engineering. This guide covers your first 90 days.

## Week one

On day one you receive your laptop and an onboarding buddy. By the end of the first week you should have your development environment running, merged a small documentation fix, and met every member of your team. Access to repositories, the CI system and the staging cluster is requested through the IT portal; approvals normally take one business day.

## Development environment

We standardise on Python 3.12 managed with uv, Node 22 for front-end work, and Docker for local services. Clone the monorepo, run `make bootstrap`, and then `make test` to confirm everything passes before you change anything. If bootstrap fails, ask in the engineering-help channel rather than working around it.

## Code review

Every change needs two approvals before it can merge, and at least one reviewer must come from outside your immediate team. Keep pull requests under 400 lines where possible. Reviewers aim to respond within one business day. Never merge your own change without the required approvals, even for urgent fixes; use the emergency-merge process instead.

## On-call

You shadow the on-call engineer starting in week four and join the rotation after your first 90 days. Before joining the rotation, read the incident response runbook and complete the pager training session.

## 30, 60 and 90 day goals

By day 30 you ship your first feature behind a flag. By day 60 you own a small service or component. By day 90 you lead a design review and join the on-call rotation.
