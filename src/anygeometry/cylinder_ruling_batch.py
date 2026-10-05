"""Authenticated BATCH of exact whole straight Cylinder ruling lifts.

This module extends the singleton producer of
:mod:`anygeometry.cylinder_angular_lifts` with ONE reusable batch query and
binding validator for a complete finite set of actual face/straight-ruling
occurrences.  Instead of calling the scalar query N times, the batch shares
one immutable model capture and at most TWO full-document hashes (entry and
final freshness) across every operand, and compiles all per-item proofs
under ONE aggregate arithmetic work budget: a single
``cylinder_charts._Proof`` whose counted charges span every operand, so the
recorded ``CylinderAtlasPolicy`` limits apply to the whole batch and are
never reset per edge.  The proof's own memoized enclosures replay their
recorded charges on reuse, so cached exact work is never laundered out of
the accounting, and no fixed model-count ceiling (in particular no new
256-face cap) is imposed on the selection.

Scope is exactly the scalar scope, repeated: whole straight boundary rulings
of active Cylinder faces on RAW stored coefficients.  Nothing here claims a
cylinder whole-material, source-current, reference or meshing acceptance, or
any "1000" acceptance; historical ``CylinderAtlasPolicy`` face-use and
occurrence admission semantics are untouched, and the batch budget concerns
only the actual counted producer work.

Safety discipline mirrors the scalar contract, adapted to the batch shape:

* Every mathematical input -- every occurrence, raw frame, frozen LinePath
  endpoints and edge vertices of every operand -- is captured detached from
  the live model BEFORE any cancellation callback runs, and the proofs
  compile only that immutable entry evidence.  A callback may therefore
  temporarily mutate and restore content; only entry evidence is published.
* During callbacks the guards run only inexpensive revision/busy checks
  (no per-surface rescan at every arithmetic callback, which would be
  quadratic in the operand count).  After the FINAL callback the guards
  recheck, for EVERY selected surface, the CURRENT raw frame pin -- which
  includes the stored circumferential coefficient that the document
  checksum does not serialize -- and then the full document checksum.  A
  finally installed surface change, including a replacement object with
  equal serialized fields and a different private circumferential
  coefficient, refuses typed.
* The caller receives either one complete immutable deterministic
  ``CylinderRulingBatchLift`` or a typed refusal: unsupported operands,
  malformed or duplicate selections, cancellation, stale or busy models,
  aggregate budget exhaustion, binding failures and callback tampering
  never produce partial accepted receipts and never mutate the model.

Determinism: the selection is compiled in canonical (face ID, edge ID)
order regardless of caller order, so the same set of occurrences always
yields the same receipt, and a reversed selection yields the identical
digest.  Per-item receipts are scalar ``CylinderRulingLift`` records whose
``work_counts`` are the CUMULATIVE aggregate proof counts at that item's
completion -- clear batch budget/work provenance, and precisely because
scalar digests bind ``work_counts``, these cumulative records can never
masquerade as standalone scalar receipts.  The batch receipt also carries
the final aggregate counts and the aggregate policy limits.

Narrow unpublished input contract: ``selection`` must be a PLAIN finite
``tuple`` or ``list`` (exact types, no subclasses) whose every entry is a
plain two-item ``tuple``/``list`` ``(face ID, edge ID)`` pair.  Custom
iterables, iterators, generators and container subclasses are refused
typed BEFORE any iteration, so a lazy or unbounded iterable is never
consumed and can never run code under the query; there is no fixed
model-count limit on the plain selection length.  The entry revision is
pinned BEFORE selection normalization and rechecked (busy plus unchanged
revision/expected revision) before any capture, so a selected input
conversion that commits an edit can never bind a newer revision under the
caller's expected revision.

Binding validation pins the caller's batch digest BEFORE any callback runs,
rederives the COMPLETE batch live under the recorded aggregate policy (one
batch rederivation, not per-ruling document hashes), and accepts only a
fresh batch whose digest equals the pinned digest while the caller's
receipt stayed unchanged -- never an object a callback repaired in flight.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
import json
from numbers import Integral
from uuid import UUID

from .cylinder_charts import CylinderAtlasPolicy, _COUNTS, _Proof, _Refusal
from .cylinder_angular_lifts import (
    CylinderAngularLiftErrorCode,
    _canonical,
    _capture_entry,
    _check_receipt,
    _document_checksum,
    _error,
    _lift,
    _raw_pin,
)
from .errors import GeometryError
from .model import GeometryModel
from .surfaces import Cylinder

__all__ = [
    "CylinderRulingBatchLift",
    "cylinder_ruling_batch_digest",
    "query_cylinder_ruling_batch",
    "validate_cylinder_ruling_batch_binding",
]


@dataclass(frozen=True, slots=True)
class CylinderRulingBatchLift:
    """Immutable authenticated batch of whole straight Cylinder rulings.

    ``items`` are per-operand scalar receipts in canonical (face ID, edge ID)
    order; each item's ``work_counts`` are the cumulative aggregate proof
    counts at that item's completion.  ``work_counts`` is the final aggregate
    count tuple of the one shared proof, and ``policy_limits`` records the
    aggregate policy the whole batch was compiled under.
    """

    model_id: UUID
    revision: int
    source_checksum: str
    items: tuple
    policy_limits: tuple
    work_counts: tuple


def cylinder_ruling_batch_digest(receipt):
    """Content digest of a batch receipt for binding comparison."""
    if type(receipt) is not CylinderRulingBatchLift:
        raise _error(CylinderAngularLiftErrorCode.INVALID_RESULT,
                     "CylinderRulingBatchLift required")
    payload = json.dumps(_canonical(receipt), sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def _pair(item):
    if type(item) not in (tuple, list):
        raise _error(CylinderAngularLiftErrorCode.INVALID_REQUEST,
                     "each selection entry must be a plain (face ID, edge ID) pair")
    if len(item) != 2:
        raise _error(CylinderAngularLiftErrorCode.INVALID_REQUEST,
                     "each selection entry must be exactly a (face ID, edge ID) pair")
    return item[0], item[1]


def _selection(selection):
    """Canonical (face ID, edge ID) order; malformed or duplicate refuses.

    Narrow unpublished input contract: the outer selection and every entry
    must be PLAIN finite ``tuple``/``list`` containers (exact types, no
    subclasses, no custom iterables, iterators or generators), refused
    BEFORE any iteration so a lazy or unbounded iterable is never consumed.
    There is no fixed model-count limit on the plain selection length.
    """
    if type(selection) not in (tuple, list):
        raise _error(CylinderAngularLiftErrorCode.INVALID_REQUEST,
                     "selection must be a plain tuple or list of "
                     "(face ID, edge ID) pairs")
    # Detach the entire outer container AND all pair values before coercion:
    # an Integral subclass may otherwise grow or rewrite the caller's lists.
    raw_pairs = tuple(_pair(item) for item in tuple(selection))
    pairs = []
    for face_id, edge_id in raw_pairs:
        for name, value in (("face ID", face_id), ("edge ID", edge_id)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise _error(CylinderAngularLiftErrorCode.INVALID_REQUEST,
                             f"positive integer {name} required")
        pairs.append((int(face_id), int(edge_id)))
    if not pairs:
        raise _error(CylinderAngularLiftErrorCode.INVALID_REQUEST,
                     "selection requires at least one occurrence")
    ordered = sorted(pairs)
    for index in range(len(ordered) - 1):
        if ordered[index] == ordered[index + 1]:
            raise _error(CylinderAngularLiftErrorCode.INVALID_REQUEST,
                         "duplicate selection entries refuse")
    return ordered


def _batch_request(model, selection, expected_revision, cancellation_check, policy):
    if not isinstance(model, GeometryModel):
        raise TypeError("GeometryModel required")
    revision = model.revision
    if expected_revision is not None and (isinstance(expected_revision, bool)
                                          or not isinstance(expected_revision, Integral)
                                          or expected_revision < 0):
        raise _error(CylinderAngularLiftErrorCode.INVALID_REQUEST,
                     "expected revision must be a non-negative integer")
    if cancellation_check is not None and not callable(cancellation_check):
        raise _error(CylinderAngularLiftErrorCode.INVALID_REQUEST,
                     "cancellation check must be callable")
    if policy is not None and type(policy) is not CylinderAtlasPolicy:
        raise _error(CylinderAngularLiftErrorCode.INVALID_REQUEST,
                     "CylinderAtlasPolicy required")
    if model._transaction_journal is not None or model._notifying_hooks:
        raise _error(CylinderAngularLiftErrorCode.BUSY_MODEL,
                     "committed model required")
    if expected_revision is not None and int(expected_revision) != model.revision:
        raise _error(CylinderAngularLiftErrorCode.STALE_REVISION,
                     "model revision differs from the expected revision")
    pairs = _selection(selection)
    if model._transaction_journal is not None or model._notifying_hooks:
        raise _error(CylinderAngularLiftErrorCode.BUSY_MODEL,
                     "committed model required after request normalization")
    if expected_revision is not None and int(expected_revision) != model.revision:
        raise _error(CylinderAngularLiftErrorCode.STALE_REVISION,
                     "model revision changed during request normalization")
    _cheap_recheck(model, revision)
    return pairs


def _cheap_recheck(model, revision):
    """Inexpensive guard for use at EVERY cancellation callback."""
    try:
        changed = (model.revision != revision
                   or model._transaction_journal is not None
                   or model._notifying_hooks)
    except GeometryError as error:
        raise _error(CylinderAngularLiftErrorCode.CHANGED_MODEL,
                     str(error)) from error
    if changed:
        raise _error(CylinderAngularLiftErrorCode.CHANGED_MODEL,
                     "the model changed during the query")


def _batch_guard(callback, model, revision):
    if callback is None:
        return None

    def checked(phase):
        if callback(phase):
            raise _error(CylinderAngularLiftErrorCode.CANCELLED, phase)
        _cheap_recheck(model, revision)

    return checked


def _final_recheck(model, entries, revision, source):
    """Final freshness: raw frame pins for every selected surface, then hash.

    Runs once after the FINAL cancellation callback.  The pins read the
    CURRENT installed surface of every selected face, so a replacement
    object with equal serialized fields and a different private
    circumferential coefficient refuses here even though the document
    checksum alone cannot see it; the full checksum then covers all
    remaining content.
    """
    try:
        changed = (model.revision != revision
                   or model._transaction_journal is not None
                   or model._notifying_hooks)
        if not changed:
            pins = {}
            for entry in entries:
                pins.setdefault(entry["face_id"], entry["frame"]["pin"])
            for face_id in sorted(pins):
                current_face = model.faces.get(face_id)
                if (current_face is None
                        or type(current_face.surface) is not Cylinder
                        or _raw_pin(current_face.surface) != pins[face_id]):
                    changed = True
                    break
        if not changed:
            changed = _document_checksum(model) != source
    except (GeometryError, _Refusal) as error:
        raise _error(CylinderAngularLiftErrorCode.CHANGED_MODEL,
                     str(error)) from error
    if changed:
        raise _error(CylinderAngularLiftErrorCode.CHANGED_MODEL,
                     "the model changed during the query")


def query_cylinder_ruling_batch(model, selection, *,
                                expected_revision=None,
                                cancellation_check=None, policy=None):
    """Prove and return the exact batch lift of a set of straight rulings.

    Every operand's proof input is captured detached from the live model
    before any cancellation callback runs, all items compile under one
    aggregate counted-arithmetic budget in canonical order, and the caller
    receives one complete immutable receipt or a typed refusal.  A refusal
    never returns partial receipts and never mutates the model.
    """
    pairs = _batch_request(model, selection, expected_revision,
                           cancellation_check, policy)
    if policy is None:
        policy = CylinderAtlasPolicy()
    revision = model.revision
    source = _document_checksum(model)
    entries = []
    frames = {}
    try:
        for face_id, edge_id in pairs:
            frame = frames.get(face_id)
            entry = _capture_entry(model, face_id, edge_id, frame)
            if frame is None:
                frames[face_id] = entry["frame"]
            entries.append(entry)
    except _Refusal as error:
        raise _error(CylinderAngularLiftErrorCode.UNQUALIFIED,
                     str(error)) from error
    proof = _Proof(policy, _batch_guard(cancellation_check, model, revision))
    items = []
    try:
        proof.cancel("cylinder ruling batch: request")
        for entry in entries:
            items.append(_lift(entry, revision, source, proof,
                              "cylinder ruling batch item"))
    except _Refusal as error:
        if error is proof.callback_error:
            raise
        if str(error).startswith("qualification_budget_exhausted"):
            raise _error(CylinderAngularLiftErrorCode.BUDGET_EXHAUSTED,
                         str(error)) from error
        raise _error(CylinderAngularLiftErrorCode.UNQUALIFIED,
                     str(error)) from error
    proof.cancel("cylinder ruling batch: final")
    _final_recheck(model, entries, revision, source)
    return CylinderRulingBatchLift(
        model.model_id, revision, source, tuple(items),
        tuple(int(getattr(policy, field.name)) for field in fields(policy)),
        tuple(proof.counts.items()),
    )


def _checked_counts(counts):
    if (type(counts) is not tuple
            or any(type(row) is not tuple or len(row) != 2
                   or type(row[0]) is not str or type(row[1]) is not int
                   or row[1] < 0 for row in counts)
            or tuple(row[0] for row in counts) != tuple(_COUNTS)):
        raise _error(CylinderAngularLiftErrorCode.INVALID_RESULT,
                     "receipt work counts require the complete canonical inventory")
    return dict(counts)


def _check_batch_receipt(receipt):
    if type(receipt) is not CylinderRulingBatchLift:
        raise _error(CylinderAngularLiftErrorCode.INVALID_RESULT,
                     "CylinderRulingBatchLift required")
    if not isinstance(receipt.model_id, UUID):
        raise _error(CylinderAngularLiftErrorCode.INVALID_RESULT,
                     "receipt model identity is invalid")
    if type(receipt.revision) is not int or receipt.revision < 0:
        raise _error(CylinderAngularLiftErrorCode.INVALID_RESULT,
                     "receipt revision is invalid")
    if (type(receipt.source_checksum) is not str or len(receipt.source_checksum) != 64
            or any(character not in "0123456789abcdef"
                   for character in receipt.source_checksum)):
        raise _error(CylinderAngularLiftErrorCode.INVALID_RESULT,
                     "receipt source checksum is invalid")
    if not isinstance(receipt.items, tuple) or not receipt.items:
        raise _error(CylinderAngularLiftErrorCode.INVALID_RESULT,
                     "receipt items are invalid")
    previous_key = None
    previous_counts = None
    for item in receipt.items:
        _check_receipt(item)
        if (item.model_id != receipt.model_id or item.revision != receipt.revision
                or item.source_checksum != receipt.source_checksum
                or item.policy_limits != receipt.policy_limits):
            raise _error(CylinderAngularLiftErrorCode.INVALID_RESULT,
                         "batch item provenance does not match the batch")
        key = (item.face_id, item.edge_id)
        if previous_key is not None and key <= previous_key:
            raise _error(CylinderAngularLiftErrorCode.INVALID_RESULT,
                         "batch items must be canonical and without duplicates")
        counts = _checked_counts(item.work_counts)
        if previous_counts is not None and any(
                counts[name] < previous_counts[name] for name in counts):
            raise _error(CylinderAngularLiftErrorCode.INVALID_RESULT,
                         "batch item work provenance is not cumulative")
        previous_key, previous_counts = key, counts
    if (not isinstance(receipt.policy_limits, tuple) or len(receipt.policy_limits) != 6
            or any(type(value) is not int for value in receipt.policy_limits)):
        raise _error(CylinderAngularLiftErrorCode.INVALID_RESULT,
                     "receipt policy limits are invalid")
    aggregate = _checked_counts(receipt.work_counts)
    final_item = _checked_counts(receipt.items[-1].work_counts)
    if any(aggregate[name] != final_item[name] for name in final_item
           if name != "cancellation_checks"):
        raise _error(CylinderAngularLiftErrorCode.INVALID_RESULT,
                     "aggregate work counts do not extend the final item")
    if aggregate["cancellation_checks"] != final_item["cancellation_checks"] + 1:
        raise _error(CylinderAngularLiftErrorCode.INVALID_RESULT,
                     "aggregate work counts do not extend the final item")


def validate_cylinder_ruling_batch_binding(model, receipt, *,
                                           cancellation_check=None):
    """Authenticate a batch receipt by complete live batch rederivation.

    The entry batch digest and the rederivation request are pinned BEFORE
    any callback runs, the COMPLETE batch rederives live under the recorded
    aggregate policy (one batch query: entry and final freshness hashes
    only), and acceptance requires the caller's receipt to be unchanged and
    the fresh batch digest to equal the pinned digest -- never a caller
    object a callback could have repaired in flight.
    """
    if not isinstance(model, GeometryModel):
        raise TypeError("GeometryModel required")
    _check_batch_receipt(receipt)
    if model._transaction_journal is not None or model._notifying_hooks:
        raise _error(CylinderAngularLiftErrorCode.BUSY_MODEL,
                     "committed model required")
    if receipt.model_id != model.model_id:
        raise _error(CylinderAngularLiftErrorCode.WRONG_MODEL,
                     "ruling batch belongs to another model")
    if receipt.revision != model.revision:
        raise _error(CylinderAngularLiftErrorCode.STALE_REVISION,
                     "ruling batch is stale")
    try:
        current = _document_checksum(model)
    except GeometryError as error:
        raise _error(CylinderAngularLiftErrorCode.CHANGED_MODEL,
                     str(error)) from error
    if current != receipt.source_checksum:
        raise _error(CylinderAngularLiftErrorCode.CHANGED_MODEL,
                     "model content changed since the ruling batch")
    entry_digest = cylinder_ruling_batch_digest(receipt)
    selection = [(item.face_id, item.edge_id) for item in receipt.items]
    revision = receipt.revision
    policy = CylinderAtlasPolicy(*receipt.policy_limits)
    fresh = query_cylinder_ruling_batch(
        model, selection, expected_revision=revision,
        cancellation_check=cancellation_check, policy=policy)
    if cylinder_ruling_batch_digest(receipt) != entry_digest:
        raise _error(CylinderAngularLiftErrorCode.CHANGED_MODEL,
                     "the receipt changed during binding validation")
    if cylinder_ruling_batch_digest(fresh) != entry_digest:
        raise _error(CylinderAngularLiftErrorCode.CHANGED_MODEL,
                     "ruling batch differs from the live derivation")
