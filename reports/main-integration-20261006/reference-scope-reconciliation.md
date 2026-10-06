# Prepared point-reference scope reconciliation

State: implemented but unverified (source review only).

Scope: `src/anygeometry/prepared_member_sheet_component.py`,
`src/anygeometry/prepared_member_sheet_network.py`, and their matching test files.
The parent owns the remapper repair, frozen-builder attributes and living integration
record. Existing unrelated dirty work is outside this assignment. Mistral was
reported unavailable; this narrow source reconciliation used the assigned OpenAI
worker under policy `ANY_ECOSYSTEM_RISK_PROPORTIONATE_ENGINEERING_V1`, revision
`2026-09-24.2`.

## Question and source evidence

The failed scoped attempt
`check-20261006T222547-d3d18c89/stdout.txt` reports 61 failures and 32 passes.
The parent identifies it as a timeout with failed taskkill, not a completed gate.
Repeated receipt construction fails at `point Attachment station or retained fields
changed`, preventing later forged/stale/callback assertions from running.

`split_edge_attachments` preserves a point attachment's ID and constructs lineage
by stable deduplication of old lineage, `('attachment', attachment.id)`, then
`('edge', split_edge_id)`. Each subsequent split retains the point ID and appends
the next split edge in order. The component guard expected old lineage plus only
the root edge. The network guard expected old lineage plus only the unique persisted
edge replacement path. Those exact guards therefore conflict with current owner
provenance. Source review also found two repeated-cut tests asserting the obsolete
edge-only length of four.

## Candidate and preserved contracts

The component now includes the original retained attachment ID before its split
root edge. The network includes that same ID before the unique persisted ordered
split-edge path. Both continue to require exact equality, including order and
stable deduplication. Unsplit attachments retain original lineage unchanged; the
network now explicitly avoids deduplicating untouched authored lineage.

No optional old/new lineage acceptance was introduced. Station/target predicates,
retained fields, vertex/source identity, coordinate tolerances, replacement-path
uniqueness, epoch/document seals, qualification flags, cancellation and binding
validation are unchanged. The parent adopted this exact lineage compatibility
reconciliation; no additional reserved contract decision is proposed.

## Regression source and handoff

- Component: plain and seeded authored-lineage positive cases; omitted, substituted,
  reordered and duplicated attachment provenance must refuse without model mutation.
- Network: seeded split and unsplit lineage cases; the same four provenance
  adversaries extend existing target/edge-lineage forgeries. Repeated-cut checks
  now explicitly require the attachment self entry, original root edge and five
  entries (one attachment plus four split edges).
- Seeded lineage includes a duplicate original vertex entry: split rewriting
  deduplicates it in original order, while an unsplit carrier preserves it.

No Python, imports, AST checks, unit/native/mesher runs, scripts, Git mutations or
numerical budget were used. Only source reads, patching and diff inspection were
performed. The fresh 60-second geometry authority remains consumed as reported by
the parent: 58.13752980006393 seconds used, 1.8624701999360695 remaining and unusable
for checks. Both older ten-shot campaigns remain closed. Full failed evidence is
preserved.

Remaining: independent source review by the parent, then focused regression and
broader failed-slice verification only under fresh execution authority. This
candidate does not establish test success, scientific acceptance, merge readiness
or release eligibility. The common construction failure masks later assertions;
source review cannot determine whether additional runtime failures remain after
this reconciliation.
