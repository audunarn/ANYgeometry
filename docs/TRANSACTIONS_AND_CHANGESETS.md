# Transactions and ChangeSets

Every public mutation is owned by `GeometryModel.transaction()`. Operations
may nest, but only the outer transaction validates, commits, increments the
revision, and notifies observers.

## Journal semantics

The journal records the original value on first write and tracks additions,
removals, immutable replacements, lineage, ownership, structural uses,
groups, tags, feature history, cache invalidations, and old/new AABBs. It does
not snapshot or copy unrelated topology.

On success the changed dependency closure is validated and index/cache deltas
are applied. On failure only journalled state is restored. Allocator
high-water marks do not rewind, so a failed or undone edit may leave an ID
gap. Rollback invalidates any provisional cache or spatial state that could
have been observed inside the transaction.

## ChangeSet contract

A deterministic `ChangeSet` records:

- revision before and after;
- added, removed, and modified entity keys;
- replacement and ownership changes;
- member, attachment, group, tag, and feature-history changes;
- changed AABBs and spatial-index updates;
- invalidated cache keys.

Net-zero topology transactions may leave the model revision unchanged, but
any lazily materialized provisional index entry is still reconciled and
reported. A successful non-empty outer mutation increments revision exactly
once.

## Observers

Change hooks are post-commit observers. They run only after topology,
structural state, feature history, caches, and indices describe the same
revision. Reentrant mutation is rejected. Hook exceptions cannot roll back a
published commit and do not prevent later hooks from observing it.

Downstream code should invalidate by `ChangeSet` and source revision. It must
not infer changes by comparing mutable objects or polling allocator values.

## Expensive compatibility snapshots

Topology/design snapshots remain explicit compatibility and undo tools. A
restore stages and validates the complete candidate state, publishes it
atomically as a new revision, preserves allocator high-water rules, and emits
one full-change notification. Snapshots are not the implementation mechanism
for ordinary local edits.

## Changed-region qualification

`audit_changed_region(model, change_set, policy=None)` expands changed AABBs
through maintained incidence and spatial candidates. Its report scope is
`CHANGED_REGION`; it can reject the affected closure but can never certify the
complete model. Full certification requires a separate full-model audit at
the exact revision being handed downstream.
