"""Atomic intersection preparation over original operand material domains.

Planning completes every narrow phase and arrangement before mutation begins.
Application journals all edge splits, canonicalization and face replacements in
one transaction. A resource refusal never returns partially accepted topology.

Planning mode (ADR-T2 v2, T2-1b; opt-in, default exact). ``IntersectionBatchPolicy.planning_mode``
is ``None`` (the default sentinel), ``"exact"`` or ``"topology_certified"``. With the sentinel the
environment variable ``ANYGEOMETRY_PLANNING_MODE`` may select the mode; an invalid value raises
``GeometryError``. A host application (ANYfem) passes its project setting explicitly through this
policy field: that is the parameter hook. Budgets (``max_predicates``/``max_candidate_pairs``) and
non-default planning context variables force the exact mode. The effective mode is resolved once
per planning call. Exact mode is the unchanged Tier-1 route and produces an ``IntersectionPlan``
without any route field (plan state E). ``topology_certified`` is
topology-certified planning (experimental): it produces a ``TopologyCertifiedPlan`` (state T) whose
``content_checksum`` keeps its meaning and whose ``route_checksum`` binds the mode. Applying a state-T plan persists its route
identity (mode, r_max, certificate version, tolerance, both checksums) in a
``TopologyCertifiedChangeSet`` and in the model document record ``anygeometry:intersection_planning``;
exact applications write neither.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from itertools import combinations
import math
import os
from contextlib import nullcontext
from uuid import UUID

import numpy as np
from .runtime_diagnostics import _timed_phase, _increment

from ._cross3 import _cross3
from .analytic_roots import root_isolation_memo
from .analytic_supports import plane_cylinder_support, cylinder_cylinder_support
from .arrangement_geometry import (LinePath, BezierPath, freeze_edge,
                                   point_parameters, curve_junctions, _junction_bounds_policy)
from .entities import EntityRef, OrientedEdge
from .curves import Straight
from .errors import GeometryError
from .exact_curves import EllipticArc, CylinderIntersectionCurve
from .extruded_supports import extruded_support
from .branch_curves import BezierQuadricCurve
from .quadric_curves import QuadricIntersectionCurve
from .identity import EntityHandle
from .material_arrangement import (ArrangementPath, ArrangementPoint, ArrangementCell, MaterialDomain,
                                   MaterialArrangement, arrange_material, _clip, _clip_intervals)
from .member_arrangements import (MemberAxisArrangement, MemberPointContact,
                                  plan_member_arrangements)
from .predicates import IntersectionDimension, IntersectionKind, qualified_plane_plane
from .serialization import to_dict, _serialized_model_state, _qualified_model_state
from .structural import ConnectionIntent, Orientation
from .surfaces import Cone, Cylinder, ExtrudedSurface, Plane
from .transactions import AABBChange, ChangeSet
from .definition_binding import definition_checksum


PLANNING_MODES = ("exact", "topology_certified")
PLANNING_MODE_ENVIRONMENT = "ANYGEOMETRY_PLANNING_MODE"


def _check_planning_mode_value(value):
    if value is not None and (type(value) is not str or value not in PLANNING_MODES):
        raise GeometryError(f"intersection planning_mode must be one of {PLANNING_MODES} or None")


@dataclass(frozen=True, slots=True)
class IntersectionBatchPolicy:
    intent: ConnectionIntent = ConnectionIntent.CONNECT
    max_predicates: int | None = None
    max_candidate_pairs: int | None = None
    cancellation_check: object = None
    face_connections: bool = True
    member_connections: bool = True
    member_face_connections: bool = True
    # ADR-T2 v2 section 1: None is the default sentinel (environment variable, else exact). The
    # field is outside the definition expansion and, when None, outside the pickled state, so a
    # default policy (and every exact plan holding one) serializes exactly as before T2-1b.
    planning_mode: str | None = field(default=None, metadata={"definition": False})

    def __getstate__(self):
        values = [getattr(self, item.name) for item in fields(self)]
        return values[:-1] if self.planning_mode is None else values

    def __setstate__(self, state):
        names = [item.name for item in fields(self)]
        state = list(state)
        if len(state) == len(names)-1:
            state.append(None)
        if len(state) != len(names):
            raise GeometryError("intersection batch policy state is malformed")
        for name, value in zip(names, state):
            object.__setattr__(self, name, value)
        _check_planning_mode_value(self.planning_mode)

    def __post_init__(self):
        _check_planning_mode_value(self.planning_mode)
        object.__setattr__(self, "intent", ConnectionIntent(self.intent))
        _check_policy_fields(self)


def _check_policy_fields(policy):
    """The field checks of ``IntersectionBatchPolicy.__post_init__`` after its intent coercion."""
    if policy.intent not in (ConnectionIntent.CONNECT,ConnectionIntent.IMPRINT,
                             ConnectionIntent.REUSE_EXISTING):
        raise GeometryError("batch preparation requires connect, imprint or reuse_existing intent")
    if policy.max_predicates is not None and (type(policy.max_predicates) is not int or policy.max_predicates < 1):
        raise GeometryError("intersection predicate budget must be a positive integer")
    if policy.max_candidate_pairs is not None and (
            type(policy.max_candidate_pairs) is not int or policy.max_candidate_pairs < 1):
        raise GeometryError("intersection candidate pair budget must be a positive integer")
    if policy.cancellation_check is not None and not callable(policy.cancellation_check):
        raise GeometryError("intersection cancellation check must be callable")
    for name in ("face_connections", "member_connections", "member_face_connections"):
        if type(getattr(policy,name)) is not bool:
            raise GeometryError(f"intersection {name} policy must be boolean")


def _check_policy_in_place(policy):
    """Every ``__post_init__`` check on an existing policy object, without copying or coercing it.

    For routes that must keep the caller's object (identity) yet may receive one changed after
    construction (``object.__setattr__``, or a state restored by ``__setstate__``, which checks only
    the mode): a noncanonical intent is refused here instead of being normalised by a copy."""
    _check_planning_mode_value(policy.planning_mode)
    if type(policy.intent) is not ConnectionIntent:
        raise GeometryError("intersection policy intent is noncanonical (not a ConnectionIntent)")
    _check_policy_fields(policy)


@dataclass(frozen=True, slots=True)
class FacePointContact:
    face_ids: tuple[int, ...]
    position: tuple[float, float, float]
    world_tolerance: float


@dataclass(frozen=True, slots=True, weakref_slot=True)
class IntersectionPlan:
    model_id: UUID
    revision: int
    source_checksum: str
    operands: tuple[EntityHandle, ...]
    policy: IntersectionBatchPolicy
    arrangements: tuple[MaterialArrangement, ...]
    axes: tuple[MemberAxisArrangement, ...] = ()
    contacts: tuple[MemberPointContact, ...] = ()
    content_checksum: str = ''
    face_contacts: tuple[FacePointContact, ...] = ()


@dataclass(frozen=True, slots=True)
class IntersectionApplication:
    plan: IntersectionPlan
    change_set: ChangeSet
    joint_edges: tuple[EntityHandle, ...]
    reused: bool = False


@dataclass(frozen=True, slots=True)
class TopologyCertifiedPlan(IntersectionPlan):
    """Plan state T (ADR-T2 v2 section 7): a plan made in ``topology_certified`` mode.

    ``content_checksum`` is the unchanged geometry identity. ``route_checksum`` binds the mode:
    definition_checksum((content_checksum, planning_mode, certificate_version, r_max)).
    ``r_pairs`` holds ``(first_face_id, second_face_id, r_pair)`` of every certified pair in
    candidate order (diagnostics; validated against ``r_max``). ``route_tolerance`` is the model
    length tolerance the certificate used: r_max must be its contract value (tol/1000 rounded down)
    at load, and it must equal the model's tolerance at apply. Exact plans are never of this type
    and carry no route field (state E). Construction, unpickling and apply validate every field
    with the one route-carrier validator (``_validate_route_carrier``; section 7 rows 5-12 and 14).
    """
    planning_mode: str | None = None
    route_checksum: str | None = None
    certificate_version: str | None = None
    r_max: float | None = None
    r_pairs: tuple = ()
    route_tolerance: float | None = None

    def __post_init__(self):
        _validate_route_carrier(self)

    def __setstate__(self, state):
        _restore_route_carrier(self, state)


@dataclass(frozen=True, slots=True)
class TopologyCertifiedApplication(IntersectionApplication):
    """Application of a state-T plan: carries the route identity of the applied plan.

    ADR section 7 row 3: the mode, r_max and certificate version are also written into the change
    set (``change_set`` is a ``TopologyCertifiedChangeSet``, the same object that
    ``model.last_change_set`` holds and change hooks receive) and into the model document record
    (``intersection_planning_record``). Construction and unpickling validate every field (row 14):
    the identity is exactly the plan's and the change set's.
    """
    planning_mode: str | None = None
    r_max: float | None = None
    certificate_version: str | None = None

    def __post_init__(self):
        _validate_route_carrier(self)

    def __setstate__(self, state):
        _restore_route_carrier(self, state)


@dataclass(frozen=True, slots=True)
class TopologyCertifiedChangeSet(ChangeSet):
    """The ChangeSet of a topology-certified application (ADR section 7 row 3, matrix row 14).

    Exact applications keep the plain ``ChangeSet`` (byte-identical pickles). Every field is checked
    at construction and at unpickling by the one route-carrier validator, with the same section-7 row
    checks as a state-T plan (rows 12, 11, 10, 5, 7, 8, 6, 14): missing, malformed or mismatched
    fields are rejected.
    """
    planning_mode: str | None = None
    certificate_version: str | None = None
    r_max: float | None = None
    route_tolerance: float | None = None
    content_checksum: str | None = None
    route_checksum: str | None = None

    def __post_init__(self):
        _validate_route_carrier(self)

    def __setstate__(self, state):
        _restore_route_carrier(self, state)


@dataclass(frozen=True, slots=True)
class _RecordEntry:
    """One publication entry of the model record, validated as a route carrier."""
    planning_mode: object
    certificate_version: object
    r_max: object
    route_tolerance: object
    content_checksum: object
    route_checksum: object
    source_revision: object
    source_checksum: object


INTERSECTION_PLANNING_EXTENSION = "anygeometry:intersection_planning"
_RECORD_FORMAT = 1
_ROUTE_IDENTITY_FIELDS = ("planning_mode", "certificate_version", "r_max", "route_tolerance", "content_checksum",
                          "route_checksum")
_RECORD_ENTRY_FIELDS = tuple(item.name for item in fields(_RecordEntry))
_RECORD_FIELDS = frozenset(_RECORD_ENTRY_FIELDS)


def _record_fields(entry):
    """Matrix row 14, entry structure: a dict with exactly the record fields (their values are
    checked by the route-carrier validator)."""
    if type(entry) is not dict or set(entry) != _RECORD_FIELDS:
        raise GeometryError("an entry does not have exactly the record fields")


def _record_entry(entry):
    _record_fields(entry)
    _validate_route_carrier(_RecordEntry(*(entry.get(name) for name in _RECORD_ENTRY_FIELDS)))
    return entry


def validate_intersection_planning_record(value):
    """Load check of the model document record (matrix row 14); returns the entries, or raises
    ``GeometryError``. Called by ``from_dict`` whenever the record is present."""
    try:
        if type(value) is not dict or set(value) != {"format", "publications"}:
            raise GeometryError("the record must have exactly 'format' and 'publications'")
        if type(value["format"]) is not int or value["format"] != _RECORD_FORMAT:
            raise GeometryError(f"the record format must be {_RECORD_FORMAT}")
        publications = value["publications"]
        if type(publications) is not list or not publications:
            raise GeometryError("the record must list at least one publication")
        return tuple(_record_entry(entry) for entry in publications)
    except GeometryError as exc:
        raise GeometryError(f"intersection planning record: {exc}") from None


def intersection_planning_record(model):
    """The validated topology-certified publications recorded on ``model`` (oldest first), or ``()``.

    Each entry holds the route identity of one applied state-T plan (planning_mode, certificate_version,
    r_max, route_tolerance, content_checksum, route_checksum) and its source binding (source_revision,
    source_checksum). Exact applications never write an entry."""
    extensions = getattr(model, "_serialization_extensions", {})
    if INTERSECTION_PLANNING_EXTENSION not in extensions:
        return ()
    return tuple(dict(entry) for entry in validate_intersection_planning_record(
        extensions[INTERSECTION_PLANNING_EXTENSION]))


def _record_publication(model, plan):
    """Append the route identity of the applied state-T ``plan`` to the model document record."""
    extensions = getattr(model, "_serialization_extensions", {})
    if not isinstance(extensions, dict):
        raise GeometryError("model document extensions are malformed")
    previous = (validate_intersection_planning_record(extensions[INTERSECTION_PLANNING_EXTENSION])
                if INTERSECTION_PLANNING_EXTENSION in extensions else ())
    entry = {name: getattr(plan, name) for name in _ROUTE_IDENTITY_FIELDS}
    entry.update(source_revision=plan.revision, source_checksum=plan.source_checksum)
    _record_entry(entry)
    made = dict(extensions)
    made[INTERSECTION_PLANNING_EXTENSION] = {"format": _RECORD_FORMAT,
                                             "publications": [*(dict(item) for item in previous), entry]}
    model._serialization_extensions = made


def _certified_change_set(change_set, plan):
    if isinstance(change_set, TopologyCertifiedChangeSet):
        return change_set
    values = {item.name: getattr(change_set, item.name) for item in fields(ChangeSet)}
    return TopologyCertifiedChangeSet(**values, **{name: getattr(plan, name) for name in _ROUTE_IDENTITY_FIELDS})


def _refuse_journaled_route(plan):
    """A state-T plan is applied only through the committed owner path: its route record is document
    state outside the topology journal, so a rolled-back caller transaction would keep a record of a
    publication that never happened."""
    if isinstance(plan, TopologyCertifiedPlan):
        raise GeometryError("a topology-certified intersection plan is applied only through the committed owner "
                            "path (its route record cannot join the caller's transaction)")


def _publish_topology(model, snapshot, plan):
    """``model.restore_topology(snapshot)``; for a state-T plan the route record is written and the
    committed change set is a ``TopologyCertifiedChangeSet`` (stored and sent to the change hooks)."""
    if not isinstance(plan, TopologyCertifiedPlan):
        model.restore_topology(snapshot)
        return
    # As GeometryModel.restore_design: collect the committed change set, complete the publication,
    # then notify the owner's hooks once with the final change set.
    events = []
    hooks = model._change_hooks
    model._change_hooks = [events.append]
    try:
        model.restore_topology(snapshot)
    finally:
        model._change_hooks = hooks
    _record_publication(model, plan)
    if not events:
        return
    change_set = _certified_change_set(events[0], plan)
    model._last_change_set = change_set
    model._notifying_hooks = True
    try:
        for hook in tuple(hooks):
            try:
                hook(change_set)
            except Exception:
                continue
    finally:
        model._notifying_hooks = False


def _context_forces_exact():
    """Whether a planning context variable is at a non-default value (exact mode is kept then)."""
    from .arrangement_geometry import _JUNCTION_BOUNDS
    from .cylinder_charts import _binding_budget
    from .exact_predicates import _TRACE
    from .curve_parameter_matching import _ACTIVE_WORK
    return (_JUNCTION_BOUNDS.get() is not False or _binding_budget.get() is not None
            or _TRACE.get() != () or _ACTIVE_WORK.get() is not None)


def resolve_planning_mode(policy):
    """The effective planning mode of ``policy`` (ADR-T2 v2 section 1), resolved once per call.

    The policy field wins; with the ``None`` sentinel ``ANYGEOMETRY_PLANNING_MODE`` applies when
    it is set (an invalid value, including an empty one, raises ``GeometryError``); otherwise
    ``"exact"``. Budgets and non-default planning context variables give ``"exact"``.
    """
    policy = _policy(policy)
    mode = policy.planning_mode
    if mode is None:
        raw = os.environ.get(PLANNING_MODE_ENVIRONMENT)
        if raw is None:
            mode = "exact"
        elif raw in PLANNING_MODES:
            mode = raw
        else:
            raise GeometryError(f"{PLANNING_MODE_ENVIRONMENT}={raw!r} is not one of {PLANNING_MODES}")
    if mode != "exact" and (policy.max_predicates is not None or policy.max_candidate_pairs is not None
                            or _context_forces_exact()):
        return "exact"
    return mode


def _mode_agnostic(policy):
    """The policy the mode-agnostic routes (existing-material, bounded) STORE in their plans: no
    mode. It is never the object those routes check or call back through: they receive the
    caller's policy itself (identity preserved, so their policy pins see a change made during the
    call) and this copy only as ``stored_policy``. On apply they validate with the separate
    ``planning_mode="exact"`` override (an explicit "exact" also overrides the environment)."""
    return policy if policy.planning_mode is None else replace(policy, planning_mode=None)


def _route_binding(content_checksum, planning_mode, certificate_version, r_max):
    return definition_checksum((content_checksum, planning_mode, certificate_version, r_max))


def _route_checksum(carrier):
    return _route_binding(carrier.content_checksum, carrier.planning_mode, carrier.certificate_version,
                          carrier.r_max)


def _normal_positive_float(value):
    return type(value) is float and math.isfinite(value) and value >= 2.2250738585072014e-308


# ------------------------------------------------------------------------------------------------
# The route-carrier validator (fix round 5, review R6: one validator, every field, every path).
#
# Route carriers: TopologyCertifiedPlan, TopologyCertifiedChangeSet, TopologyCertifiedApplication and
# _RecordEntry (one publication of the model record). ``_validate_route_carrier`` is the only entry:
# construction (__post_init__), unpickling (__setstate__ via _restore_route_carrier), apply
# (_route_state), document load (validate_intersection_planning_record from from_dict), record read
# and record write. It runs the section-7 row checks of the carrier's type, in a fixed order; each
# field is owned by exactly one row (route_checksum by rows 5 and 6):
#   12 content_checksum   11/10 planning_mode   5/6 route_checksum   7 certificate_version
#   8 r_max, route_tolerance (finite normal floats, consistent)   9 r_pairs (plans)
#   14 every other field (exact type and form, _CARRIER_FIELDS; an application's identity is its plan's).
# Types are exact (bool is not an int, numpy scalars are not floats); every checksum is a SHA-256
# hex digest, the form of definition_checksum and of the document checksum. A dataclass carrier's
# state must have exactly its fields (_restore_route_carrier); a record entry exactly the record
# fields (_record_fields).

_CHECKSUM_DIGITS = frozenset("0123456789abcdef")


def _is_checksum(value):
    return type(value) is str and len(value) == 64 and _CHECKSUM_DIGITS.issuperset(value)


def _is_revision(value):
    return type(value) is int and value >= 0


def _is_tuple_of(value, valid):
    return type(value) is tuple and all(valid(item) for item in value)


def _of_type(kind):
    return lambda value: type(value) is kind


def _tuple_of_type(kind):
    return lambda value: _is_tuple_of(value, _of_type(kind))


def _is_entity_key(value):
    return type(value) is tuple and len(value) == 2 and type(value[0]) is str and type(value[1]) is int


def _is_replacement(value):
    return (type(value) is tuple and len(value) == 2 and type(value[0]) is EntityRef
            and _is_tuple_of(value[1], _of_type(EntityRef)))


_ENTITY_KEYS = lambda value: _is_tuple_of(value, _is_entity_key)
# Row 14: the fields of each carrier that are not owned by rows 5-12.
_CARRIER_FIELDS = {
    TopologyCertifiedPlan: {
        "model_id": _of_type(UUID), "revision": _is_revision, "source_checksum": _is_checksum,
        "operands": _tuple_of_type(EntityHandle), "policy": _of_type(IntersectionBatchPolicy),
        "arrangements": _tuple_of_type(MaterialArrangement), "axes": _tuple_of_type(MemberAxisArrangement),
        "contacts": _tuple_of_type(MemberPointContact), "face_contacts": _tuple_of_type(FacePointContact)},
    TopologyCertifiedChangeSet: {
        "revision_before": _is_revision, "revision_after": _is_revision,
        "added": _ENTITY_KEYS, "removed": _ENTITY_KEYS, "modified": _ENTITY_KEYS,
        "replacements": lambda value: _is_tuple_of(value, _is_replacement),
        "ownership_changes": _ENTITY_KEYS, "member_changes": _ENTITY_KEYS, "attachment_changes": _ENTITY_KEYS,
        "group_changes": _tuple_of_type(str), "tag_changes": _ENTITY_KEYS,
        "affected_aabbs": _tuple_of_type(AABBChange), "invalidated_caches": _ENTITY_KEYS,
        "spatial_updates": _ENTITY_KEYS, "feature_history_changed": _of_type(bool),
        "document_settings_changed": _of_type(bool)},
    _RecordEntry: {"source_revision": _is_revision, "source_checksum": _is_checksum},
    TopologyCertifiedApplication: {
        "plan": _of_type(TopologyCertifiedPlan), "change_set": _of_type(TopologyCertifiedChangeSet),
        "joint_edges": _tuple_of_type(EntityHandle), "reused": _of_type(bool)},
}
_IDENTITY_ROWS = (12, 11, 10, 5, 7, 8, 6, 14)  # persisted route identities carry no r_pairs
_CARRIER_ROWS = {TopologyCertifiedPlan: (12, 11, 10, 5, 7, 8, 9, 6, 14),
                 TopologyCertifiedChangeSet: _IDENTITY_ROWS,
                 _RecordEntry: _IDENTITY_ROWS,
                 TopologyCertifiedApplication: (14,)}


def _validate_route_carrier(carrier):
    """THE validator of every route carrier (see the section comment); raises ``GeometryError``."""
    rows = _CARRIER_ROWS.get(type(carrier))
    if rows is None:
        raise GeometryError(f"{type(carrier).__name__} is not a topology-certified route carrier")
    checks = globals()             # looked up at call time: each row has its own unsound test double
    for row in rows:
        checks[f"_route_row_{row}"](carrier)


def _restore_route_carrier(carrier, state):
    """Unpickling of a dataclass route carrier: exactly its fields, then the validator."""
    names = [item.name for item in fields(carrier)]
    state = list(state) if type(state) in (list, tuple) else None
    if state is None or len(state) != len(names):
        raise GeometryError(f"{type(carrier).__name__} state is malformed")
    for name, value in zip(names, state):
        object.__setattr__(carrier, name, value)
    _validate_route_carrier(carrier)


# ADR section 7 matrix: one function per row, each with its own unsound test double (review R5).
# They are looked up by name at call time, in the order of _CARRIER_ROWS (rows 11-6 keep the order
# of the checks before fix round 4, so the first failing row and its message are unchanged).

def _route_row_12(carrier):
    """Row 12: content_checksum is a SHA-256 hex digest; a plan's recomputes from its content."""
    value = carrier.content_checksum
    if isinstance(carrier, IntersectionPlan):
        try:
            bound = _is_checksum(value) and value == _plan_content(carrier)
        except (GeometryError, TypeError, ValueError, AttributeError):     # content that cannot be bound
            bound = False
        if not bound:
            raise GeometryError("intersection plan analytic content binding changed")
    elif not _is_checksum(value):
        raise GeometryError("topology-certified route content_checksum is not a SHA-256 hex digest")


def _same_value(first, second):
    return type(first) is type(second) and first == second


def _route_row_14(carrier):
    """Row 14: every field not owned by rows 5-12 has its exact type and form; an application's
    identity is exactly its plan's (mode, r_max, version) and its change set's (all six fields)."""
    for name, valid in _CARRIER_FIELDS[type(carrier)].items():
        if not valid(getattr(carrier, name)):
            raise GeometryError(f"topology-certified route carrier field {name} is malformed")
    if type(carrier) is TopologyCertifiedChangeSet and carrier.revision_after < carrier.revision_before:
        raise GeometryError("topology-certified route carrier field revision_after is malformed")
    if type(carrier) is TopologyCertifiedApplication:
        for name in ("planning_mode", "r_max", "certificate_version"):
            if not _same_value(getattr(carrier, name), getattr(carrier.plan, name)):
                raise GeometryError(f"topology-certified route carrier field {name} is malformed")
        for name in _ROUTE_IDENTITY_FIELDS:
            if not _same_value(getattr(carrier.change_set, name), getattr(carrier.plan, name)):
                raise GeometryError("topology-certified route carrier field change_set is malformed")

def _route_row_11(plan):
    """Row 11: route fields without planning_mode, or planning_mode = "exact" stored explicitly."""
    mode = plan.planning_mode
    if mode is None:
        raise GeometryError("intersection plan carries route fields without planning_mode")
    if type(mode) is str and mode == "exact":
        raise GeometryError("an exact intersection plan carries no route fields")


def _route_row_10(plan):
    """Row 10: planning_mode with any other value."""
    mode = plan.planning_mode
    if type(mode) is not str or mode not in PLANNING_MODES:
        raise GeometryError(f"intersection plan planning_mode {mode!r} is not supported")


def _route_row_5(plan):
    """Row 5: route_checksum missing, empty or not a SHA-256 hex digest."""
    if not _is_checksum(plan.route_checksum):
        raise GeometryError("topology-certified intersection plan has no route_checksum (SHA-256 hex digest)")


def _route_row_7(plan):
    """Row 7: certificate_version missing or not in the supported set."""
    from .separating_plane_certificate import CERTIFICATE_VERSION
    if type(plan.certificate_version) is not str or plan.certificate_version != CERTIFICATE_VERSION:
        raise GeometryError("topology-certified intersection plan has an unsupported certificate_version")


def _route_row_8(plan):
    """Row 8 (load side): r_max a positive normal float, bit-equal to the contract value of the
    stored route tolerance (tol/1000 rounded down)."""
    if not _normal_positive_float(plan.r_max):
        raise GeometryError("topology-certified intersection plan r_max is not a positive normal float")
    if not _normal_positive_float(plan.route_tolerance) or plan.r_max != _contract_r_max(plan.route_tolerance):
        raise GeometryError("topology-certified intersection plan r_max is not the contract value")


def _route_row_8_model(plan, model):
    """Row 8 (apply side): the route tolerance is the model's length tolerance and r_max its
    contract value (addendum A13: r_max is per plan)."""
    if plan.route_tolerance != model.tolerance.length or plan.r_max != _plan_r_max(model):
        raise GeometryError("topology-certified intersection plan r_max is not the contract value")


def _route_row_9(plan):
    """Row 9: every recorded r_pair present, finite and <= r_max (exact float comparison)."""
    if type(plan.r_pairs) is not tuple:
        raise GeometryError("topology-certified intersection plan r_pair diagnostics are malformed")
    for item in plan.r_pairs:
        if (type(item) is not tuple or len(item) != 3 or type(item[0]) is not int or type(item[1]) is not int
                or item[0] < 1 or item[1] < 1
                or type(item[2]) is not float or not math.isfinite(item[2]) or not 0. <= item[2] <= plan.r_max):
            raise GeometryError("topology-certified intersection plan r_pair is missing, non-finite or above r_max")


def _route_row_6(plan):
    """Row 6: the route checksum recomputes from (content_checksum, mode, version, r_max)."""
    try:
        binding = _route_checksum(plan)
    except (TypeError, ValueError):                 # unbindable values (only reachable with row 8 removed)
        binding = None
    if binding is None or plan.route_checksum != binding:
        raise GeometryError("topology-certified intersection plan route binding changed")


def _route_state(plan):
    """``"E"`` or ``"T"``; a malformed state-T plan raises ``GeometryError`` (the route-carrier
    validator: ADR section 7 rows 5-12 and 14)."""
    if not isinstance(plan, TopologyCertifiedPlan):
        return "E"
    _validate_route_carrier(plan)
    return "T"


def _validate_route(plan, state, policy, model=None, *, planning_mode=None):
    """Mode/state agreement (ADR section 7) for a plan of ``state`` (from ``_route_state``); with
    ``model``, r_max against the model tolerance. ``planning_mode`` (internal: the mode-agnostic
    routes pass "exact") replaces the policy's resolved mode without copying the policy."""
    if planning_mode is None:
        mode = resolve_planning_mode(policy)
    elif type(planning_mode) is str and planning_mode in PLANNING_MODES:
        mode = planning_mode
    else:
        raise GeometryError(f"intersection planning_mode override must be one of {PLANNING_MODES}")
    if state == "E" and mode != "exact":
        raise GeometryError("an exact intersection plan cannot be applied in topology_certified mode")
    if state == "T":
        if mode != "topology_certified":
            raise GeometryError("a topology-certified intersection plan is never applied as exact")
        if model is not None:
            _route_row_8_model(plan, model)
    return state


def _contract_r_max(tolerance):
    from .separating_plane_certificate import contract_r_max, SPRefusal
    try:
        return contract_r_max(tolerance)
    except SPRefusal as exc:
        raise GeometryError(f"topology_certified r_max is unavailable for this tolerance: {exc}") from None


def _plan_r_max(model):
    """The per-plan contract bound: tol/1000 of the model length tolerance, rounded down."""
    return _contract_r_max(model.tolerance.length)


def _policy(value):
    if isinstance(value, IntersectionBatchPolicy):
        return value
    try:
        return IntersectionBatchPolicy(ConnectionIntent(value))
    except (TypeError, ValueError) as exc:
        raise GeometryError("intersection batch needs an explicit connection policy") from exc


def _plan_content(plan):
    return definition_checksum((plan.operands,plan.policy.intent,
        plan.policy.face_connections,plan.policy.member_connections,plan.policy.member_face_connections,
        plan.arrangements,plan.axes,plan.contacts,plan.face_contacts))


def _validate_policy(plan, policy, model=None, *, planning_mode=None):
    # A state-T plan is first validated field by field by the route-carrier validator (row 12 is its
    # content binding), so the comparisons below read validated fields; a state-E plan does no work here.
    # ``planning_mode`` is the mode-agnostic routes' explicit override (see _validate_route): the
    # caller's policy object itself is validated, never a copy of it.
    state = _route_state(plan)
    if any(getattr(policy,name) != getattr(plan.policy,name) for name in (
        'intent','face_connections','member_connections','member_face_connections')):
        raise GeometryError("apply policy does not match intersection planning")
    if state == "E" and plan.content_checksum != _plan_content(plan):
        raise GeometryError("intersection plan analytic content binding changed")
    _validate_route(plan, state, policy, model, planning_mode=planning_mode)


def _normalise_cells(arrangement):
    # Kernel loops have at least three edges. Analytic lenses and circles can
    # have two mathematical edges; split one chart without adding a seam.
    split = set()
    for cell in arrangement.cells:
        for loop in (cell.outer, *cell.holes):
            if len(loop) == 2:
                split.add(loop[0][0])
    if not split:
        return arrangement
    paths, mapping = [], {}
    for index, path in enumerate(arrangement.paths):
        indices = []
        for a, b in (((0., .5), (.5, 1.)) if index in split else ((0., 1.),)):
            indices.append(len(paths))
            paths.append(replace(path, curve=path.curve.subcurve(a, b)))
        mapping[index] = tuple(indices)
    def loop(old):
        result = []
        for index, forward in old:
            result.extend((child, forward) for child in (mapping[index] if forward else mapping[index][::-1]))
        return tuple(result)
    return replace(arrangement, paths=tuple(paths), cells=tuple(
        ArrangementCell(loop(cell.outer), tuple(loop(hole) for hole in cell.holes))
        for cell in arrangement.cells))


def _synchronize_boundary_events(arrangements, check):
    """Propagate decomposition ports across all exact shared boundaries.

    A construction seam is local to one domain, but its endpoint splits the
    physical boundary used by every incident domain. Synchronize those events
    after every original-operand arrangement, before any face is rebuilt.
    """
    from .material_arrangement import _parameters
    by_edge, by_face = {}, {}
    for arrangement in arrangements:
        events=[]
        for path in arrangement.paths:
            endpoints=(path.curve.evaluate(0.),path.curve.evaluate(1.))
            events.extend(endpoints)
            if path.source_edge is not None:
                by_edge.setdefault(path.source_edge,[]).extend(endpoints)
        by_face[arrangement.face_id]=events
    result=[]
    for arrangement in arrangements:
        paths,mapping=[],{}
        for index,path in enumerate(arrangement.paths):
            values={0.,1.}
            events=list(by_edge.get(path.source_edge,()))
            if len(path.owners)>1:
                events.extend(point for owner in path.owners for point in by_face.get(owner,()))
            lo,hi=path.curve.bounds()
            if events:
                # One vectorized box test; the same elementwise comparisons
                # as testing each point separately, in the same order.
                points=np.asarray(events)
                outside=(np.any(points<lo-arrangement.world_tolerance,axis=1)
                         | np.any(points>hi+arrangement.world_tolerance,axis=1))
                events=[events[index] for index in np.flatnonzero(~outside)]
            for point in events:
                check()
                values.update(point_parameters(path.curve,point,tolerance=arrangement.world_tolerance))
            parameters=_parameters(values,arrangement.world_tolerance,path.curve)
            children=[]
            for start,end in zip(parameters,parameters[1:]):
                children.append(len(paths))
                paths.append(replace(path,curve=path.curve.subcurve(start,end)))
            mapping[index]=tuple(children)
        def loop(old):
            return tuple((child,forward) for index,forward in old
                         for child in (mapping[index] if forward else mapping[index][::-1]))
        result.append(replace(arrangement,paths=tuple(paths),cells=tuple(
            ArrangementCell(loop(cell.outer),tuple(loop(hole) for hole in cell.holes))
            for cell in arrangement.cells)))
    return tuple(result)


def _synchronize_member_events(arrangements, axes, check):
    """Split member axes at ports introduced while decomposing incident faces."""
    from .material_arrangement import _parameters
    by_member = {}
    values = [set(axis.split_parameters) for axis in axes]
    for index, axis in enumerate(axes):
        for member in axis.member_ids:
            by_member.setdefault(member, []).append(index)
    for arrangement in arrangements:
        for path in arrangement.paths:
            indices = sorted({index for member in path.member_ids
                              for index in by_member.get(member, ())})
            if not indices:
                continue
            for parameter in (0., 1.):
                point = path.curve.evaluate(parameter)
                for index in indices:
                    check()
                    axis = axes[index]
                    values[index].update(point_parameters(
                        axis.curve, point, tolerance=axis.world_tolerance))
    return tuple(replace(axis, split_parameters=_parameters(
        values[index], axis.world_tolerance, axis.curve))
        for index, axis in enumerate(axes))


def _domain_bounds(domain):
    boxes = [path.curve.bounds() for loop in domain.boundaries for path in loop]
    return np.min([box[0] for box in boxes], axis=0), np.max([box[1] for box in boxes], axis=0)


def _coplanar_traces(first, second, tolerance, check, model):
    from .intersections import _sheet_ids_for_face
    same_sheet = bool(set(_sheet_ids_for_face(model, first.face_id)) &
                      set(_sheet_ids_for_face(model, second.face_id)))
    shared_seams = {path.source_edge: path.curve for loop in first.boundaries for path in loop
                    if path.decomposition and path.source_edge is not None} if same_sheet else {}
    shared_seams = {path.source_edge: shared_seams[path.source_edge]
                    for loop in second.boundaries for path in loop
                    if path.decomposition and path.source_edge in shared_seams}
    def physical(curve):
        # Source edge identity proves shared topology, not just common Sheet
        # ownership. For affine paths, endpoint containment certifies the
        # complete interval; other curve families remain conservatively kept.
        return not (isinstance(curve, LinePath) and any(
            isinstance(seam, LinePath) and
            all(point_parameters(seam, point, tolerance=tolerance)
                for point in (curve.start, curve.end))
            for seam in shared_seams.values()))
    if isinstance(first.support, Plane) and isinstance(second.support, Plane) and all(
        isinstance(path.curve, LinePath) for domain in (first, second)
        for loop in domain.boundaries for path in loop):
        # The established polygon predicate is exact for straight trim
        # segments. Reuse its robust point/interval/area distinction; no
        # sampled curved boundary enters this backend.
        from .intersections import _query_planar_faces
        result = _query_planar_faces(model, model.handle("face", first.face_id),
                                    model.handle("face", second.face_id), first.support, second.support)
        if not result.classified:
            raise GeometryError("; ".join(result.diagnostics))
        if result.dimension is IntersectionDimension.REGION:
            raise GeometryError("positive-area overlap requires an explicit ownership policy")
        if result.dimension is not IntersectionDimension.CURVE:
            return ()
        return tuple(LinePath(component.witnesses[0], component.witnesses[-1])
                     for component in result.components if len(component.witnesses) >= 2
                     and physical(LinePath(component.witnesses[0], component.witnesses[-1])))
    # Partition original material by the other domain's complete boundary.
    # Classify cells in both domains; positive-area ownership stays explicit.
    traces = tuple(ArrangementPath(path.curve, owners=(first.face_id, second.face_id))
                   for loop in second.boundaries for path in loop)
    arranged = arrange_material(first, traces, tolerance=tolerance, cancellation_check=lambda: (check() or False))
    for cell in arranged.cells:
        index, forward = cell.outer[0]
        path = arranged.paths[index]
        curve = path.curve if forward else path.curve.subcurve(1., 0.)
        uv = first.uv(curve, .5)
        tangent = first.tangent(curve, .5)
        point = first.support.evaluate(*first.left_probe(uv, tangent, 2*tolerance))
        probe = LinePath(tuple(point), tuple(point))
        if second.contains(probe, .5, tolerance, boundary=False):
            raise GeometryError("positive-area overlap requires an explicit ownership policy")
    # Only common boundary intervals remain. Retain each exact interval in
    # both material domains; point-only contact is not an implicit weld.
    result = []
    for loop in first.boundaries:
        for path in loop:
            for part in _clip(second, path.curve, tolerance, check):
                if any(point_parameters(boundary.curve, part.evaluate(.5), tolerance=tolerance)
                       for boundary_loop in second.boundaries for boundary in boundary_loop):
                    result.append(part)
    return tuple(curve for curve in result if physical(curve))


def _rectangular_cylinder(domain, tolerance):
    """Prove a native rectangle from analytic boundary definitions."""
    if not isinstance(domain.support, Cylinder) or len(domain.boundaries) != 1:
        return False
    support = domain.support
    covered = {side: [] for side in range(4)}
    for path in domain.boundaries[0]:
        curve = path.curve
        start, end = domain.uv(curve, 0.), domain.uv(curve, 1.)
        if isinstance(curve, LinePath):
            direction = np.asarray(curve.end)-curve.start
            if np.linalg.norm(_cross3(direction, support.axis)) > tolerance:
                return False
            coordinate, varying = 0, 1
        elif isinstance(curve, EllipticArc):
            normal = _cross3(curve.u_vector, curve.v_vector)
            normal /= np.linalg.norm(normal)
            if np.linalg.norm(_cross3(normal, support.axis)) > tolerance:
                return False
            center = np.asarray(curve.center)-support.origin
            radial = center-float(center @ support.axis)*support.axis
            if np.linalg.norm(radial) > tolerance or max(abs(np.linalg.norm(vector)-support.radius)
                for vector in (curve.u_vector, curve.v_vector)) > tolerance:
                return False
            coordinate, varying = 1, 0
        else:
            return False
        if max(abs(start[coordinate]), abs(end[coordinate])) <= tolerance:
            side = 2*coordinate
        elif max(abs(start[coordinate]-1), abs(end[coordinate]-1)) <= tolerance:
            side = 2*coordinate+1
        else:
            return False
        covered[side].append(tuple(sorted((start[varying], end[varying]))))
    for intervals in covered.values():
        right = 0.
        for start, end in sorted(intervals):
            if start > right+tolerance:
                return False
            right = max(right, end)
        if abs(right-1) > tolerance:
            return False
    return True


def _coincident_cylinder_rectangles(first, second, model, tolerance):
    """Exact angular/axial interval intersection, including periodic seams."""
    a, b = first.support, second.support
    orientation = 1 if float(a.axis @ b.axis) > 0 else -1
    phase = math.atan2(float(b.radial_direction @ a.circumferential_direction),
                       float(b.radial_direction @ a.radial_direction))
    alo, ahi = sorted((a.start_angle, a.start_angle+a.sweep_angle))
    blo, bhi = sorted((phase+orientation*b.start_angle,
                      phase+orientation*(b.start_angle+b.sweep_angle)))
    second_z = float((b.origin-a.origin) @ a.axis)
    zlo = max(min(0., a.height), min(second_z, second_z+orientation*b.height))
    zhi = min(max(0., a.height), max(second_z, second_z+orientation*b.height))
    if zhi < zlo-tolerance:
        return ()
    angle_tolerance = tolerance/a.radius
    paths = []
    from .intersections import _sheet_ids_for_face
    same_sheet = bool(set(_sheet_ids_for_face(model, first.face_id)) & set(_sheet_ids_for_face(model, second.face_id)))
    for turn in range(math.ceil((alo-bhi-angle_tolerance)/math.tau),
                      math.floor((ahi-blo+angle_tolerance)/math.tau)+1):
        lower, upper = max(alo, blo+turn*math.tau), min(ahi, bhi+turn*math.tau)
        if upper < lower-angle_tolerance:
            continue
        if upper-lower > angle_tolerance and zhi-zlo > tolerance:
            raise GeometryError("positive-area overlap requires an explicit ownership policy")
        if same_sheet:
            # Existing chart boundaries of one source Sheet are decomposition
            # seams. They are already connected by their source topology.
            continue
        if upper-lower <= angle_tolerance and zhi-zlo > tolerance:
            angle = .5*(lower+upper)
            base = a.origin+a.radius*(math.cos(angle)*a.radial_direction+math.sin(angle)*a.circumferential_direction)
            paths.append(LinePath(tuple(base+zlo*a.axis), tuple(base+zhi*a.axis)))
        elif upper-lower > angle_tolerance:
            center = a.origin+.5*(zlo+zhi)*a.axis
            paths.append(EllipticArc(center, a.radius*a.radial_direction, a.radius*a.circumferential_direction,
                                     lower, upper-lower))
    return tuple(paths)


def _face_arrangement_tolerance(model, domain):
    """EXACTLY the planner's per-face arrangement tolerance: ``effective_length(norm(hi - lo))`` of the domain bounds."""
    lo, hi = _domain_bounds(domain)
    return model.tolerance.effective_length(float(np.linalg.norm(hi - lo)))


def _owning_tolerance(tolerance, face_tolerances):
    """The strictest of the pair tolerance and BOTH owning faces' arrangement tolerances.

    The pair tolerance is the one of the LONGER face, so it can exceed either owner's; each face's arrangement
    later uses its own limit, so supports are qualified at the strictest.
    """
    return min(float(tolerance), *(float(v) for v in face_tolerances))


def _mark_coincident(shadow):
    if shadow is not None:
        shadow["coincident"] = True


def _short_reason(reason):
    return ":".join(str(reason).split(":")[:2])


class _TopologyRoute:
    """The T2b route of one topology_certified planning call (ADR-T2 v2 sections 4, 5; addendum A9).

    Per candidate pair, in candidate order: the exact separating-plane certificate on records of
    the planner's own domains (cached per domain object, never by face id); on CERTIFIED the full
    exact support stage runs in shadow (refusal parity: the same code, arguments and point in the
    loop, including production's coincidence test and its overlap check); the stored shared topology
    is output only if production's coincidence decision agrees with the certificate's and, for a
    coincident pair, the supports are identical including the radial direction and the faces have
    equal orientation signs. Anything else returns None: the unchanged exact path for the pair.
    """

    def __init__(self, model, certificate_policy=None):
        from .separating_plane_certificate import SPCertifier, CERTIFICATE_VERSION
        self.model = model
        self.tolerance = model.tolerance.length
        self.r_max = _plan_r_max(model)
        self.version = CERTIFICATE_VERSION
        self.certifier = SPCertifier(certificate_policy)
        self.r_pairs = []
        self._records = {}
        self._orientations = {}

    # --- inputs -------------------------------------------------------------------------------
    def _record(self, domain):
        hit = self._records.get(id(domain))
        if hit is not None and hit[0] is domain:
            return hit[1]
        from .separating_plane_certificate import face_record_from_model, SPRefusal
        try:
            value = face_record_from_model(self.model, domain.face_id, domain=domain)
        except SPRefusal as exc:
            value = "record:" + str(exc)
        except Exception as exc:                            # no callback runs here: no cancellation
            value = "record:" + type(exc).__name__
        self._records[id(domain)] = (domain, value)
        return value

    def _orientation(self, domain):
        """Production's orientation sign of a face (MaterialArrangement.orientation: the sign of the
        outer loop's chart area from ``area_loop``), or None unless |area| exceeds the face's area
        tolerance."""
        hit = self._orientations.get(id(domain))
        if hit is not None and hit[0] is domain:
            return hit[1]
        lo, hi = _domain_bounds(domain)
        area_tolerance = self.model.tolerance.effective_area(float(np.linalg.norm(hi-lo)))
        try:
            area = float(domain.area_loop(domain.boundaries[0], area_tolerance*.05))
        except (GeometryError, ArithmeticError, ValueError):
            area = math.nan
        sign = None if not math.isfinite(area) or abs(area) <= area_tolerance else (1 if area > 0 else -1)
        self._orientations[id(domain)] = (domain, sign)
        return sign

    # --- decision -----------------------------------------------------------------------------
    @staticmethod
    def started():
        _increment("planning_mode:topology_certified")

    def _fallback(self, reason):
        _increment("t2b_fallback:" + _short_reason(reason))
        return None

    def _certify(self, first, second, check):
        _increment("t2b_pairs")
        records = (self._record(first), self._record(second))
        for record in records:
            if isinstance(record, str):
                return self._fallback(record)
        raised = []

        def guarded():
            try:
                check()
            except BaseException as error:              # noqa: BLE001 - recorded, then re-raised
                raised.append(error)
                raise
        try:
            with _timed_phase('t2b_certificate'):
                verdict = self.certifier.certify(records[0], records[1], self.tolerance, check=guarded)
        except Exception as error:
            if any(error is item for item in raised):
                raise                                   # CANCELLED: the callback's exception, unchanged
            return self._fallback("internal:" + type(error).__name__)
        if not verdict.certified:
            return self._fallback(verdict.reason)
        return verdict

    def _coincident_eligible(self, first, second):
        """Section 4.4 prerequisites for coincident supports (A9 items 2-3): identical stored
        cylinder data including the radial direction, and equal orientation signs."""
        from .separating_plane_certificate import CylinderSupport
        from fractions import Fraction
        r1, r2 = self._record(first), self._record(second)
        if isinstance(r1, str) or isinstance(r2, str):
            return False
        s1, s2 = r1.support, r2.support
        if type(s1) is not CylinderSupport or type(s2) is not CylinderSupport:
            return False

        def exact(support):
            return (tuple(map(Fraction, support.origin)), tuple(map(Fraction, support.axis)),
                    Fraction(support.radius), tuple(map(Fraction, support.radial_direction)))
        if exact(s1) != exact(s2):
            return False
        o1, o2 = self._orientation(first), self._orientation(second)
        return o1 is not None and o1 == o2

    def pair(self, first, second, model, tolerance, check, point_contacts, face_tolerances):
        """The certified pair's traces (point contacts appended), or None for the exact path."""
        verdict = self._certify(first, second, check)
        if verdict is None:
            return None
        shadow = {}
        with _timed_phase('t2b_support_shadow'):
            output = _pair_paths_exact(first, second, model, tolerance, check, face_tolerances=face_tolerances,
                                       _shadow=shadow)
        production = shadow.get("coincident", False) is True
        flagged = verdict.evidence.get("coincident_support") is True
        if production != flagged:
            _increment("t2b_coincidence_mismatch")
        if production:
            # Production's coincident branch returns before clipping, so the shadow has already
            # run the exact path completely for this pair, overlap check included: its output IS
            # the exact output and is returned unchanged (identical to exact mode, same-sheet seam
            # rule untouched). A disagreeing certificate flag or an ineligible pair (radial
            # direction, orientation signs) is recorded as a fallback; the output is the same.
            if production != flagged:
                self._fallback("coincidence_mismatch")
            elif not self._coincident_eligible(first, second):
                self._fallback("coincident_support_ineligible:radial_or_orientation")
            else:
                _increment("t2b_coincident_exact_output")
            return tuple(output)
        if flagged:
            return self._fallback("coincidence_mismatch")
        return self._emit(verdict, first, second, tolerance, point_contacts)

    def _emit(self, verdict, first, second, tolerance, point_contacts):
        """T_out: each shared edge's stored curve (in the first face's loop order) as a trace, each
        shared vertex not on a shared edge as a point contact (ADR section 4.3, D1)."""
        r1, r2 = self._record(first), self._record(second)
        uses1 = [use for loop in r1.loops for use in loop]
        shared = {use.edge for use in uses1} & {use.edge for loop in r2.loops for use in loop}
        traces, seen, on_edges = [], set(), set()
        for use in uses1:
            if use.edge in shared and use.edge not in seen:
                seen.add(use.edge)
                on_edges.update((use.start_vertex, use.end_vertex))
                traces.append(use.edge_curve if use.edge_curve is not None else use.curve)
        positions1, ids2 = dict(r1.vertices), {vid for vid, _position in r2.vertices}
        if point_contacts is not None:
            emitted = set()
            for use in uses1:
                # One point per shared vertex IDENTITY: the certified bound r_pair << tol does not
                # allow merging distinct vertices within the pair tolerance.
                vertex = use.start_vertex
                if vertex in ids2 and vertex not in on_edges and vertex not in emitted:
                    emitted.add(vertex)
                    point_contacts.append(tuple(float(x) for x in positions1[vertex]))
        self.r_pairs.append((first.face_id, second.face_id, float(verdict.r_pair)))
        _increment("t2b_certified")
        return tuple(traces)

    def seal(self, plan):
        values = {item.name: getattr(plan, item.name) for item in fields(IntersectionPlan)}
        route_checksum = _route_binding(plan.content_checksum, "topology_certified", self.version, self.r_max)
        return TopologyCertifiedPlan(**values, planning_mode="topology_certified", route_checksum=route_checksum,
                                     certificate_version=self.version, r_max=self.r_max,
                                     r_pairs=tuple(self.r_pairs), route_tolerance=float(self.tolerance))


def _pair_paths(first, second, model, tolerance, check, *, point_contacts=None, face_tolerances=None, route=None):
    """Pair-stage traces of two domains. ``route`` (topology_certified mode only) may replace the
    clipping stage of a certified pair by its stored shared topology (ADR-T2 v2 section 4, T2b);
    without a route this is the unchanged exact path."""
    if route is not None:
        traces = route.pair(first, second, model, tolerance, check, point_contacts, face_tolerances)
        if traces is not None:
            return traces
    return _pair_paths_exact(first, second, model, tolerance, check, point_contacts=point_contacts,
                             face_tolerances=face_tolerances)


def _pair_paths_exact(first, second, model, tolerance, check, *, point_contacts=None, face_tolerances=None,
                      _shadow=None):
    """The exact pair stage. With ``_shadow`` (a dict) it runs only the support stage, in full and
    with production's own coincidence test and overlap check, records ``_shadow["coincident"]``
    and returns None before clipping (the D3 / section 5 shadow of a certified pair)."""
    a, b = first.support, second.support
    family = ('plane_plane' if isinstance(a, Plane) and isinstance(b, Plane) else
              'plane_cylinder' if isinstance(a, (Plane, Cylinder)) and isinstance(b, (Plane, Cylinder))
              and (isinstance(a, Plane) or isinstance(b, Plane)) else
              'cylinder_cylinder' if isinstance(a, Cylinder) and isinstance(b, Cylinder) else 'other')
    _increment('intersection_pair_' + family)
    if isinstance(a, Plane) and isinstance(b, Plane):
        result = qualified_plane_plane(a, b, policy=model.tolerance)
        if not result.classified:
            raise GeometryError("; ".join(result.diagnostics))
        if result.kind is IntersectionKind.DISJOINT:
            return ()
        if result.kind is IntersectionKind.COINCIDENT:
            _mark_coincident(_shadow)
            return _coplanar_traces(first, second, tolerance, check, model)
        component = result.components[0]
        point, direction = np.asarray(component.witnesses[0]), np.asarray(component.direction)
        boxes = (_domain_bounds(first), _domain_bounds(second))
        extent = max(np.linalg.norm(lo-point)+np.linalg.norm(hi-point) for lo, hi in boxes)+1.
        paths = (LinePath(tuple(point-extent*direction), tuple(point+extent*direction)),)
    elif isinstance(a, ExtrudedSurface) or isinstance(b, ExtrudedSurface):
        from .extruded_pair_supports import extruded_pair_support
        result = extruded_pair_support(a, b, tolerance=tolerance, cancellation_check=lambda: (check() or False))
        if result.coincident:
            _mark_coincident(_shadow)
            return _coplanar_traces(first, second, tolerance, check, model)
        paths = (*result.curves, *(LinePath(start, end) for start, end in result.segments))
    elif isinstance(a, Cone) or isinstance(b, Cone):
        from .quadric_supports import cone_support
        result = cone_support(a, b, tolerance=tolerance, cancellation_check=lambda: (check() or False))
        if result.coincident:
            _mark_coincident(_shadow)
            return _coplanar_traces(first, second, tolerance, check, model)
        paths = (*result.curves, *(LinePath(start, end) for start, end in result.segments))
    elif (isinstance(a, Plane) and isinstance(b, Cylinder)) or (isinstance(a, Cylinder) and isinstance(b, Plane)):
        plane, cylinder = (a, b) if isinstance(a, Plane) else (b, a)
        pair = (first.face_id, second.face_id)
        # Supports are qualified per pair, before anything is accumulated, with the STRICTEST owning-face tolerance
        # (never larger than the pair tolerance) against the stored plane and cylinder.
        if face_tolerances is None:
            face_tolerances = (_face_arrangement_tolerance(model, first), _face_arrangement_tolerance(model, second))
        strict = _owning_tolerance(tolerance, face_tolerances)
        result = plane_cylinder_support(plane, cylinder, tolerance=strict,
                                        cancellation_check=lambda: (check() or False), face_pair=pair)
        from .plane_ruled import qualify_plane_cylinder_result
        qualify_plane_cylinder_result(result, plane, cylinder, strict, pair)
        paths = (*result.curves, *(LinePath(start, end) for start, end in result.segments))
    else:
        result = cylinder_cylinder_support(a, b, tolerance=tolerance, cancellation_check=lambda: (check() or False))
        if result.coincident:
            _mark_coincident(_shadow)
            if _rectangular_cylinder(first, tolerance) and _rectangular_cylinder(second, tolerance):
                return _coincident_cylinder_rectangles(first, second, model, tolerance)
            return _coplanar_traces(first, second, tolerance, check, model)
        paths = (*result.curves, *(LinePath(start, end) for start, end in result.segments))
    if _shadow is not None:
        return None
    candidates=list(getattr(result,'points',()))
    def clip(domain,curve):
        if point_contacts is None:
            return _clip(domain,curve,tolerance,check)
        return tuple(curve.subcurve(a,b) for a,b in _clip_intervals(
            domain,curve,tolerance,check,isolated_contacts=candidates))
    with _timed_phase('intersection_clipping'):
        curves=tuple(part for path in paths for clipped in clip(first,path)
                     for part in clip(second,clipped))
    if point_contacts is not None:
        for point in candidates:
            check()
            station=LinePath(tuple(point),tuple(point))
            if (first.contains(station,0.,tolerance) and second.contains(station,0.,tolerance)
                    and not any(point_parameters(curve,point,tolerance=tolerance) for curve in curves)
                    and not any(np.linalg.norm(np.asarray(point)-old)<=tolerance for old in point_contacts)):
                point_contacts.append(tuple(point))
    return curves


def plan_intersections(model, operands, *, policy):
    """Plan intersections between the complete original operand set, read-only."""
    from .preparation_epochs import _epoch_plan_required,_issue_native_plan
    from .native_support_snapshots import capture_native_supports
    operands = tuple(operands)
    from .bounded_nurbs import handles_bounded, plan_bounded
    from .intersections import _normalize_operand
    handles = tuple(sorted(set(_normalize_operand(model, value) for value in operands)))
    effective = _policy(policy)
    mode = resolve_planning_mode(effective)         # once per planning call (ADR-T2 v2 section 1)
    # The existing-material and bounded routes are mode-agnostic: they plan (and apply) exactly as
    # in exact mode and return state-E plans without any mode in their stored policy.
    # They check and call back through the caller's policy object (identity preserved); only the
    # plan they store gets the mode-free copy.
    stored = _mode_agnostic(effective)
    from .existing_material_preparation import try_plan_existing
    existing = try_plan_existing(model, handles, effective, stored_policy=stored)
    if existing is not None:
        return existing
    native=capture_native_supports(model) if _epoch_plan_required(model) else None
    if effective.face_connections and handles_bounded(model, handles):
        plan = plan_bounded(model, handles, effective, stored_policy=stored)
        if native is not None:
            if capture_native_supports(model) != native:
                raise GeometryError('bounded epoch source native geometry changed during planning')
        return plan
    if native is not None and any(kind not in ('plane','cylinder') for _,kind,_ in native):
        raise GeometryError('epoch planning native support binding is unavailable')
    junction_policy = _junction_bounds_policy(False) if effective.max_predicates is not None else nullcontext()
    with root_isolation_memo(), _timed_phase('intersection_planning'), junction_policy:
        plan=_plan_intersections(model, operands, policy=policy, _planning_mode=mode)
    if native is not None:
        if (capture_native_supports(model)!=native or model.model_id!=plan.model_id or
                model.revision!=plan.revision or to_dict(model)['checksum']['value']!=plan.source_checksum):
            raise GeometryError('geometry or native support changed during epoch planning')
        _issue_native_plan(plan,native)
    return plan


def _plan_intersections(model, operands, *, policy, _planning_mode="exact", _certificate_policy=None):
    from .intersections import _normalize_operand
    policy = _policy(policy)
    route = None
    if _planning_mode == "topology_certified":
        route = _TopologyRoute(model, _certificate_policy)
        route.started()
    elif _planning_mode != "exact":
        raise GeometryError(f"intersection planning mode {_planning_mode!r} is not supported")
    elif policy.planning_mode is not None:
        # State E carries no route field anywhere, including the stored policy.
        policy = replace(policy, planning_mode=None)
    if policy.intent not in (ConnectionIntent.CONNECT, ConnectionIntent.IMPRINT, ConnectionIntent.REUSE_EXISTING):
        raise GeometryError("intersection preparation requires CONNECT, IMPRINT or REUSE_EXISTING intent")
    revision, checksum = model.revision, to_dict(model)["checksum"]["value"]
    handles = tuple(sorted(set(_normalize_operand(model, item) for item in operands)))
    faces, members = set(), set()
    for handle in handles:
        if handle.kind == "face":
            faces.add(handle.id)
        elif handle.kind == "sheet":
            faces.update(model.face_uses[use_id].face_id for use_id in model.sheets[handle.id].face_use_ids)
        elif handle.kind == "member":
            members.add(handle.id)
        else:
            raise GeometryError("intersection batch operands must be faces, sheets or members")
    from .boundary_contacts import bilinear_boundary, plane_boundary_contacts
    domains, boundary_only = {}, {}
    for face_id in sorted(faces):
        try:
            domains[face_id] = MaterialDomain.from_model(model, face_id)
        except GeometryError:
            boundary_only[face_id] = bilinear_boundary(model, face_id)
    traces = {face_id: [] for face_id in faces}
    points = {face_id: [] for face_id in faces}
    face_contacts=[]
    examined = 0
    candidate_pairs = 0
    def check_pair():
        nonlocal candidate_pairs
        check()
        candidate_pairs += 1
        if policy.max_candidate_pairs is not None and candidate_pairs > policy.max_candidate_pairs:
            raise GeometryError(
                f"intersection planning exceeds maximum_candidate_pairs={policy.max_candidate_pairs}")
    def check():
        nonlocal examined
        examined += 1
        if policy.cancellation_check is not None and policy.cancellation_check():
            raise GeometryError("intersection planning cancelled")
        if policy.max_predicates is not None and examined > policy.max_predicates:
            raise GeometryError("intersection planning predicate budget exhausted")
    domain_bounds = {face_id: _domain_bounds(domain) for face_id, domain in domains.items()}
    lengths = {face_id: float(np.linalg.norm(hi-lo))
               for face_id, (lo, hi) in domain_bounds.items()}
    if policy.face_connections:
        for face_id, corners in boundary_only.items():
            lo1, hi1 = np.min(corners, axis=0), np.max(corners, axis=0)
            for other_id, domain in domains.items():
                check()
                lo2, hi2 = domain_bounds[other_id]
                tolerance = model.tolerance.effective_length(max(
                    float(np.linalg.norm(hi1-lo1)), lengths[other_id]))
                if np.any(hi1 < lo2-tolerance) or np.any(hi2 < lo1-tolerance):
                    continue
                check_pair()
                traces[other_id].extend(plane_boundary_contacts(
                    model, face_id, corners, domain, tolerance, check))
        for first, second in combinations(sorted(boundary_only), 2):
            a, b = np.asarray(boundary_only[first]), np.asarray(boundary_only[second])
            tolerance = model.tolerance.effective_length(max(
                float(np.linalg.norm(np.ptp(a, axis=0))), float(np.linalg.norm(np.ptp(b, axis=0)))))
            if np.all(a.max(axis=0) >= b.min(axis=0)-tolerance) and np.all(b.max(axis=0) >= a.min(axis=0)-tolerance):
                check_pair()
                raise GeometryError("intersecting nonplanar Coons supports require a qualified surface predicate")
    if members and policy.member_face_connections and boundary_only:
        # These compatibility surfaces have no general axis/support predicate.
        # Never silently omit a requested beam/shell relation.
        raise GeometryError("member intersections with nonplanar Coons supports are unsupported")
    from .runtime_diagnostics import _increment
    ordered_faces = sorted(domains)

    def account_skipped(count):
        """Preserve logical predicate charges; poll cancellation in bounded chunks."""
        nonlocal examined
        while count > 0:
            step = min(count, 4096)
            examined += step
            if policy.cancellation_check is not None and policy.cancellation_check():
                raise GeometryError("intersection planning cancelled")
            if policy.max_predicates is not None and examined > policy.max_predicates:
                raise GeometryError("intersection planning predicate budget exhausted")
            count -= step

    def candidate_face_pairs():
        """The pairs of ``combinations(sorted(domains), 2)`` that can pass the final box predicate, in that order.

        A conservative sorted sweep with the WIDEST pair tolerance (the pair
        tolerance is monotone in the longer face's length, so it never rejects a pair the exact predicate keeps).
        Rejected pairs are charged to the logical predicate budget and the cancellation poll as before; the exact
        per-pair predicate below is unchanged and decides every candidate, so the candidate sequence equals the
        exhaustive one.
        """
        if not policy.face_connections or len(ordered_faces) < 2:
            return
        # Explicit ceilings retain lazy traversal and its refusal order;
        # building a dense candidate list must not precede a small budget.
        if policy.max_predicates is not None or policy.max_candidate_pairs is not None:
            yield from combinations(ordered_faces, 2)
            return
        longest, shortest = max(lengths.values()), min(lengths.values())
        wide = max(model.tolerance.effective_length(longest), model.tolerance.effective_length(shortest))
        count = len(ordered_faces)
        _increment("batch_face_pairs_total", count * (count - 1) // 2)
        from .intersection_candidates import swept_face_pairs
        def poll():
            if policy.cancellation_check is not None and policy.cancellation_check():
                raise GeometryError("intersection planning cancelled")
        positions = {face: i for i, face in enumerate(ordered_faces)}
        last_ordinal = -1
        with _timed_phase('intersection_candidates'):
            selected_pairs = swept_face_pairs(domain_bounds, wide, poll=poll)
        for first, second in selected_pairs:
            i, j = positions[first], positions[second]
            ordinal = i * (2 * count - i - 1) // 2 + j - i - 1
            account_skipped(ordinal - last_ordinal - 1)
            last_ordinal = ordinal
            _increment("batch_face_pairs_candidates")
            yield first, second
        account_skipped(count * (count - 1) // 2 - last_ordinal - 1)

    for first_id, second_id in candidate_face_pairs():
        check()
        first, second = domains[first_id], domains[second_id]
        tolerance = model.tolerance.effective_length(max(lengths[first_id], lengths[second_id]))
        lo1, hi1 = domain_bounds[first_id]; lo2, hi2 = domain_bounds[second_id]
        if np.any(hi1 < lo2-tolerance) or np.any(hi2 < lo1-tolerance):
            continue
        check_pair()
        try:
            pair_points=[]
            with _timed_phase('intersection_support_and_clipping'):
                pair_paths = _pair_paths(first, second, model, tolerance, check,point_contacts=pair_points,
                                         face_tolerances=(model.tolerance.effective_length(lengths[first_id]),
                                                          model.tolerance.effective_length(lengths[second_id])),
                                         route=route)
        except GeometryError as exc:
            raise GeometryError(f"intersection batch face:{first_id}/face:{second_id}: {exc}") from exc
        for curve in pair_paths:
            path = ArrangementPath(curve, owners=(first_id, second_id))
            traces[first_id].append(path); traces[second_id].append(path)
        for point in pair_points:
            face_contacts.append(FacePointContact((first_id,second_id),point,tolerance))
            for face_id in (first_id,second_id):
                points[face_id].append(ArrangementPoint(point))
    axes, contacts = plan_member_arrangements(model, members, domains, traces, points, check,
        include_axis_axis=policy.member_connections, include_axis_face=policy.member_face_connections,
        check_pair=check_pair, domain_bounds=domain_bounds)
    arrangements = []
    for face_id, domain in domains.items():
        if not traces[face_id] and not points[face_id]:
            continue
        try:
            with _timed_phase('intersection_arrangements'):
                arrangement = arrange_material(domain, traces[face_id],
                    tolerance=model.tolerance.effective_length(lengths[face_id]),
                    area_tolerance=model.tolerance.effective_area(lengths[face_id]),
                    points=points[face_id],
                    cancellation_check=lambda: (check() or False))
        except GeometryError as exc:
            raise GeometryError(f"intersection material arrangement face:{face_id}: {exc}") from exc
        arrangements.append(_normalise_cells(arrangement))
    if model.revision != revision or _qualified_model_state(model)["checksum"]["value"] != checksum:
        raise GeometryError("geometry changed during intersection planning")
    arrangements=_synchronize_boundary_events(arrangements,check)
    axes=_synchronize_member_events(arrangements,axes,check)
    plan=IntersectionPlan(model.model_id, revision, checksum, handles, policy,
                          arrangements, axes, contacts,face_contacts=tuple(face_contacts))
    plan=replace(plan,content_checksum=_plan_content(plan))
    return plan if route is None else route.seal(plan)


def _coincident(first, second, tolerance):
    if isinstance(first, LinePath) and isinstance(second, LinePath):
        return True  # Caller already established both matching endpoints.
    if isinstance(first, EllipticArc) and isinstance(second, EllipticArc):
        inverse = np.linalg.pinv(np.column_stack((first.u_vector, first.v_vector)))
        normal = _cross3(first.u_vector, first.v_vector)
        normal /= np.linalg.norm(normal)
        center = np.asarray(second.center)-first.center
        u, v = inverse @ second.u_vector, inverse @ second.v_vector
        return (abs(float(normal @ center)) <= tolerance and np.linalg.norm(inverse @ center) <= tolerance
                and abs(float(u @ u)-1) <= tolerance and abs(float(v @ v)-1) <= tolerance
                and abs(float(u @ v)) <= tolerance
                and bool(point_parameters(first, second.evaluate(.5), tolerance=tolerance)))
    if isinstance(first, BezierPath) and isinstance(second, BezierPath):
        a, b = np.asarray(first.controls), np.asarray(second.controls)
        return a.shape == b.shape and min(np.max(np.linalg.norm(a-b, axis=1)), np.max(np.linalg.norm(a-b[::-1], axis=1))) <= tolerance
    if isinstance(first, QuadricIntersectionCurve) or isinstance(second, QuadricIntersectionCurve):
        from .quadric_events import branch_supports, curve_quadric_roots
        branch, other = (first, second) if isinstance(first, QuadricIntersectionCurve) else (second, first)
        return (all(curve_quadric_roots(other, support, tolerance=tolerance) is None
                    for support in branch_supports(branch))
                and bool(point_parameters(branch, other.evaluate(.5), tolerance=tolerance)))
    if isinstance(first, CylinderIntersectionCurve) and isinstance(second, CylinderIntersectionCurve):
        from .cylinder_curve_events import cylinder_roots
        return (all(cylinder_roots(first,support,transform=second.transform,tolerance=tolerance) is None
                    for support in (second.first,second.second))
                and bool(point_parameters(first,second.evaluate(.5),tolerance=tolerance)))
    if isinstance(first,CylinderIntersectionCurve) and isinstance(second,EllipticArc):
        from .arrangement_geometry import ellipse_on_branch_supports
        return (ellipse_on_branch_supports(second,first,tolerance)
                and bool(point_parameters(first,second.evaluate(.5),tolerance=tolerance)))
    if isinstance(first,EllipticArc) and isinstance(second,CylinderIntersectionCurve):
        return _coincident(second,first,tolerance)
    if isinstance(first, BezierQuadricCurve) or isinstance(second, BezierQuadricCurve):
        from .branch_events import world_quadric
        from .quadric_events import curve_quadric_roots
        branch, other = (first, second) if isinstance(first, BezierQuadricCurve) else (second, first)
        # the other curve lies on the branch's quadric and on its wall (every sample is a point of the branch)
        return (curve_quadric_roots(other, world_quadric(branch), tolerance=tolerance) is None
                and all(point_parameters(branch, other.evaluate(float(t)), tolerance=tolerance)
                        for t in np.linspace(0., 1., 9)))
    return False


def _canonicalize_member_edges(model, curves, axes, joint_ids, check):
    """Share qualified axis/boundary definitions without losing owner records."""
    selected = {member for axis in axes for member in axis.member_ids}
    candidates = sorted({use.edge_id for use in model.member_edge_uses.values()
                         if use.member_id in selected})
    for axis_id in candidates:
        check()
        if axis_id not in model.edges:
            continue
        axis = model.edges[axis_id]
        tolerance = model.tolerance.effective_length(model.edge_length(axis_id))
        duplicates = [identifier for identifier in model.edges_using_vertex(axis.start)
                      if identifier != axis_id and
                      {model.edges[identifier].start, model.edges[identifier].end} == {axis.start, axis.end}
                      and _coincident(curves[axis_id], curves[identifier], tolerance)]
        if not duplicates:
            continue
        canonical_id = min(axis_id, *duplicates)
        canonical = model.edges[canonical_id]
        for duplicate_id in sorted({axis_id, *duplicates}-{canonical_id}):
            if duplicate_id not in model.edges:
                continue
            duplicate = model.edges[duplicate_id]
            direct = duplicate.start == canonical.start
            for use_id in tuple(model._edge_member_uses.get(duplicate_id, ())):
                use = model.member_edge_uses[use_id]
                model._put_structural("member_edge_use", replace(use, edge_id=canonical_id,
                    orientation=use.orientation if direct else (
                        Orientation.REVERSED if use.orientation is Orientation.FORWARD else Orientation.FORWARD)))
            for face_id in tuple(model.faces_using_edge(duplicate_id)):
                face = model.faces[face_id]
                def loop(values):
                    return tuple(OrientedEdge(canonical_id, item.forward == direct)
                                 if item.edge == duplicate_id else item for item in values)
                model._put_entity("face", replace(face, loop=loop(face.loop),
                    holes=tuple(loop(hole) for hole in face.holes)))
            from .structural import ParameterRange
            for attachment_id in tuple(model._target_attachments.get(("edge", duplicate_id), ())):
                attachment = model.attachments[attachment_id]
                parameters = attachment.target_parameters if direct else tuple(
                    ParameterRange(1-value.end, 1-value.start) for value in attachment.target_parameters)
                model._put_structural("attachment", replace(attachment,
                    target_id=canonical_id, target_parameters=parameters))
            for member_id in tuple(model._orientation_members.get(("edge", duplicate_id), ())):
                member = model.members[member_id]
                model._put_structural("member", replace(member, orientation_reference=("edge", canonical_id)))
            for attachment in tuple(model.attachments.values()):
                if attachment.source_kind == "edge" and attachment.source_id == duplicate_id:
                    model._put_structural("attachment", replace(attachment, source_id=canonical_id))
            model._delete_entity("edge", duplicate_id)
            model.record_replacement(EntityRef("edge", duplicate_id), (EntityRef("edge", canonical_id),))
            joint_ids.discard(duplicate_id)
        joint_ids.add(canonical_id)


def _extruded_child_support(model, support, outer, tolerance):
    """Rebase a rectangular child of an extruded surface onto its own parameter ranges (no new geometry)."""
    domain=MaterialDomain(0,support,(tuple(ArrangementPath(freeze_edge(model,use.edge)
                if use.forward else freeze_edge(model,use.edge).subcurve(1.,0.)) for use in outer),))
    rows=[domain.uv(path.curve,0.) for path in domain.boundaries[0]]
    lower,upper=np.min(rows,axis=0),np.max(rows,axis=0)
    (u0,u1),(v0,v1)=support.u_range,support.v_range
    t=u0+np.linspace(0.,1.,33)*(u1-u0)
    speed=min(float(np.max(np.linalg.norm(support.directrix.derivative(t),axis=1)))*abs(u1-u0),
              float(np.linalg.norm(support.vector))*abs(v1-v0))
    native_tolerance=tolerance/speed
    rectangular=all((abs(row[0]-lower[0])<=native_tolerance or abs(row[0]-upper[0])<=native_tolerance)
                    or (abs(row[1]-lower[1])<=native_tolerance or abs(row[1]-upper[1])<=native_tolerance)
                    for row in rows)
    for path in domain.boundaries[0]:
        first,middle,last=domain.uv(path.curve,0.),domain.uv(path.curve,.5),domain.uv(path.curve,1.)
        if isinstance(path.curve,LinePath):
            rectangular &= abs(first[0]-last[0]) <= native_tolerance and abs(first[0]-middle[0]) <= native_tolerance
        else:
            rectangular &= abs(first[1]-last[1]) <= native_tolerance and abs(first[1]-middle[1]) <= native_tolerance
    if not rectangular or not np.all(upper-lower>native_tolerance):
        return outer,support,None,None
    index=min((i for i,row in enumerate(rows) if abs(row[1]-lower[1])<=native_tolerance),key=lambda i:rows[i][0])
    outer=outer[index:]+outer[:index]
    child=support.subpatch((lower[0],upper[0]),(lower[1],upper[1]))
    return outer,child,None,model._detect_corners(outer) if len(outer) >= 4 else None


def _store_extruded_support(model, face_id, support, check):
    """Store the exact support of a recovered extrusion whose loop is no longer four edges.

    A contact that splits a boundary edge leaves the face as one cell with a longer loop, which topology
    recognition cannot recover. The stored support is the same surface in the chart of its own directrix,
    so attachments are remapped by position exactly as for the children of a split.
    """
    from .attachment_remapping import capture_face_attachments, remap_face_attachments
    face = model.faces[face_id]
    if (isinstance(support, ExtrudedSurface) and not isinstance(face.surface, ExtrudedSurface)
            and face.parameterization is None and extruded_support(model, face_id) is None):
        snapshots = capture_face_attachments(model, face_id, check)
        model._put_entity('face', replace(face, surface=support, corners=()))
        remap_face_attachments(model, face, (face_id,), snapshots, check)


def _snapped_chart_bounds(lower, upper, lengths, tolerance):
    """Chart bounds of a rectangular child, with each bound physically within ``tolerance`` of the
    parent's own bound 0 or 1 (``lengths``: physical length of each chart direction) set to it.

    The snaps are taken only together and only when the worst chart corner, which can move in both
    directions, moves at most ``tolerance`` (bounded by the sum of the two directions' moves);
    otherwise the float bounds are returned unchanged.
    """
    moves=[[abs(lower[k])*lengths[k],abs(upper[k]-1.)*lengths[k]] for k in (0,1)]
    moves=[[move if move<=tolerance else 0. for move in pair] for pair in moves]
    if max(moves[0])+max(moves[1])>tolerance:
        return lower,upper
    lower,upper=np.array(lower,dtype=float),np.array(upper,dtype=float)
    for k in (0,1):
        if abs(lower[k])*lengths[k]<=tolerance:
            lower[k]=0.
        if abs(upper[k]-1.)*lengths[k]<=tolerance:
            upper[k]=1.
    return lower,upper


def _child_support(model, face, outer, holes, tolerance):
    """Rebase a proven rectangular support; general trims stay native charts.

    A mapped child must not evaluate the untrimmed parent's whole unit square.
    This changes the parameter extent, never the underlying physical surface.
    """
    support=face.surface
    if holes or not isinstance(support,(Plane,Cylinder,Cone,ExtrudedSurface)):
        return outer,support,face.parameterization,None
    if isinstance(support,ExtrudedSurface):
        return _extruded_child_support(model,support,outer,tolerance)
    points=[model.vertex_position(model.oriented_start_vertex(use)) for use in outer]
    if isinstance(support,(Cylinder,Cone)):
        domain=MaterialDomain(0,support,(tuple(ArrangementPath(freeze_edge(model,use.edge)
                    if use.forward else freeze_edge(model,use.edge).subcurve(1.,0.)) for use in outer),))
        rows=[domain.uv(path.curve,0.) for path in domain.boundaries[0]]
        lower,upper=np.min(rows,axis=0),np.max(rows,axis=0)
        widest=max(support.radius_start,support.radius_end) if isinstance(support,Cone) else support.radius
        native_tolerance=tolerance/min(widest*abs(support.sweep_angle),support.height)
        rectangular=all((abs(row[0]-lower[0])<=native_tolerance or abs(row[0]-upper[0])<=native_tolerance)
                        or (abs(row[1]-lower[1])<=native_tolerance or abs(row[1]-upper[1])<=native_tolerance)
                        for row in rows)
        for path in domain.boundaries[0]:
            first,last=domain.uv(path.curve,0.),domain.uv(path.curve,1.)
            if isinstance(path.curve,LinePath):
                rectangular &= abs(first[0]-last[0]) <= native_tolerance
            elif isinstance(path.curve,EllipticArc):
                rectangular &= (abs(first[1]-last[1])<=native_tolerance
                    and abs(float(np.asarray(path.curve.u_vector) @ support.axis))<=tolerance
                    and abs(float(np.asarray(path.curve.v_vector) @ support.axis))<=tolerance)
            else:
                rectangular=False
        if rectangular and np.all(upper-lower>native_tolerance):
            index=min(range(len(rows)),key=lambda i:(rows[i][1],rows[i][0]))
            # Numerical equality on the lower axial level is qualified in
            # physical tolerance before selecting the lowest angular corner.
            index=min((i for i,row in enumerate(rows) if abs(row[1]-lower[1])<=native_tolerance),
                      key=lambda i:rows[i][0])
            outer=outer[index:]+outer[:index]
            # A bound physically within tolerance of the parent's own bound is that bound: the float
            # local_uv of an unmoved boundary can differ from 0 or 1 by a few ulps between
            # platforms' libm, which must not move the carrier's serialized origin/extent. Each
            # chart direction uses its own physical length (widest arc; slant for a Cone), so a
            # real cut on the longer direction is never snapped away. A chart corner can move in both
            # directions at once: the snaps are taken only when the worst corner's movement, bounded
            # by the sum of the two directions' moves (a path along each), is within tolerance.
            if isinstance(support,Cone):
                axial=math.hypot(support.height,support.radius_end-support.radius_start)
            else:
                axial=support.height
            lower,upper=_snapped_chart_bounds(lower,upper,(widest*abs(support.sweep_angle),axial),tolerance)
            origin=support.origin if lower[1]==0. else support.origin+lower[1]*support.height*support.axis
            height=(upper[1]-lower[1])*support.height
            start=support.start_angle if lower[0]==0. else support.start_angle+lower[0]*support.sweep_angle
            sweep=(upper[0]-lower[0])*support.sweep_angle
            if isinstance(support,Cone):
                def radius(v):
                    return (1-v)*support.radius_start+v*support.radius_end
                support=Cone(origin,support.axis,support.radial_direction,radius(lower[1]),radius(upper[1]),
                             height,start,sweep)
            else:
                support=Cylinder(origin,support.axis,support.radial_direction,support.radius,height,start,sweep)
            return outer,support,None,model._detect_corners(outer) if len(outer) >= 4 else None
        return outer,support,None,None
    corners=model._detect_corners(outer) if len(outer)>=4 else ()
    if len(corners)==4 and all(isinstance(model.edges[use.edge].curve,Straight) for use in outer):
        p0,p1,p2,p3=(points[index] for index in corners)
        if np.linalg.norm(p0+p2-p1-p3)<=tolerance:
            return outer,Plane(p0,p1-p0,p3-p0),None,corners
    return outer,support,None,None


def _face_contact_residual(model, face, point, uv):
    """Residual of a canonical face-point contact station on ``face`` at its stored ``uv``.

    ``uv`` comes from ``model.face_local_uv`` (bounded to [0, 1]).
    """
    target=model.faces[face]
    if target.parameterization is None:
        # Unchanged: uv is in the support's own frame.
        return float(np.linalg.norm(target.surface.evaluate(*uv)-point))
    # face_local_uv inverted the parameterization, so the stored uv is checked with
    # face_point in that frame. That residual alone can be self-confirming (a topology
    # Coons map contains its own merged boundary stations), so the station must ALSO lie
    # on the authoritative support, measured with the support's own inverse and evaluator.
    support_uv=target.surface.local_uv(point)
    if type(target.surface) is not Plane:
        # Bounded charts: the fix1 formula, clipped to the unit square, bit for bit.
        support_uv=np.clip(support_uv,0.,1.)
    # A Plane is unbounded and its stored frame need not cover the face (the tripod web
    # planes), so its uv are NOT clipped: the term is the distance to the plane itself.
    # It is an on-support check only. The face_point term at the clipped uv rejects
    # stations outside the parameterization's patch; trim holes inside the patch are
    # not checked here (as before S0f) and rest on the planned contact itself.
    return max(
        float(np.linalg.norm(model.face_point(face,*uv)-point)),
        float(np.linalg.norm(target.surface.evaluate(
            float(support_uv[0]),float(support_uv[1]))-point)))


def _apply_intersections_in_place(model, plan, *, policy, _edge_preimage_draft=None):
    """Atomically apply a model/revision-bound material arrangement."""
    from .intersections import _merge_vertex
    from .attachment_remapping import capture_face_attachments, remap_face_attachments
    from .edge_attachment_remapping import split_edge_attachments, _member_parameter
    outer_journal=model._transaction_journal
    def keys():
        return {(kind,identifier) for kind in model._next_id for identifier in model._entity_store(kind)} | {
            (kind,identifier) for kind in model._next_structural_id for identifier in model._structural_store(kind)}
    before_keys=keys() if outer_journal is not None else set()
    if not isinstance(plan, IntersectionPlan):
        raise TypeError("apply_intersections needs an IntersectionPlan")
    policy = _policy(policy)
    _validate_policy(plan,policy,model)
    if model.model_id != plan.model_id:
        raise GeometryError("intersection plan belongs to another model")
    if model.revision != plan.revision:
        raise GeometryError("intersection plan is stale")
    if _qualified_model_state(model)["checksum"]["value"] != plan.source_checksum:
        raise GeometryError("intersection plan source binding changed")
    tolerances_by_face = {item.face_id: item.world_tolerance for item in plan.arrangements}
    tolerance = model.tolerance.length
    joint_ids = set()
    examined = 0
    def check():
        nonlocal examined
        examined += 1
        if policy.cancellation_check is not None and policy.cancellation_check():
            raise GeometryError("intersection application cancelled")
        if policy.max_predicates is not None and examined > policy.max_predicates:
            raise GeometryError("intersection application budget exhausted")
    with model.transaction():
        check()
        contact_ranges = {
            (member, parameter): next(model.member_edge_uses[use_id].parent_range
                for use_id in model.members[member].edge_use_ids
                if model.member_edge_uses[use_id].parent_range.contains(parameter,
                    tolerance=model.tolerance.parameter))
            for contact in plan.contacts for member, parameter in contact.member_parameters}
        if policy.intent is ConnectionIntent.CONNECT:
            # Declare physical owners before fragmentation. Creating one
            # owner for each descendant would turn decomposition seams into
            # artificial multi-Sheet joints in downstream mesh preparation.
            for arrangement in plan.arrangements:
                if not any(use.face_id == arrangement.face_id for use in model.face_uses.values()):
                    model.add_sheet((arrangement.face_id,),
                                    name=f"intersection owner for face {arrangement.face_id}")
        # Split original trim edges globally before rebuilding any face. This
        # also updates untouched face loops and MemberEdgeUse parent intervals.
        fractions = {}
        originals = {}
        for axis in plan.axes:
            originals[axis.edge_id] = axis.curve
            fractions[axis.edge_id] = set(axis.split_parameters)
        for arrangement in plan.arrangements:
            for path in arrangement.paths:
                if path.source_edge is None:
                    continue
                edge_id = path.source_edge
                original = originals.setdefault(edge_id, freeze_edge(model, edge_id))
                for parameter in (0., 1.):
                    fractions.setdefault(edge_id, set()).update(point_parameters(original, path.curve.evaluate(parameter), tolerance=tolerance))
        for edge_id, values in sorted(fractions.items()):
            tolerance = min((tolerances_by_face[face_id] for face_id in model.faces_using_edge(edge_id)
                             if face_id in tolerances_by_face),
                            default=next((axis.world_tolerance for axis in plan.axes
                                          if axis.edge_id == edge_id), model.tolerance.length))
            current, previous = edge_id, 0.
            original = originals[edge_id]
            unique = []
            for fraction in sorted(values):
                point = original.evaluate(fraction)
                if min(np.linalg.norm(point-original.evaluate(0.)),
                       np.linalg.norm(point-original.evaluate(1.))) <= tolerance:
                    continue
                if unique and np.linalg.norm(point-original.evaluate(unique[-1])) <= tolerance:
                    continue
                unique.append(fraction)
            for fraction in unique:
                check()
                # Earlier children can use different regularized charts; invert
                # the original station on the current exact descendant.
                current_curve = freeze_edge(model, current)
                stations = point_parameters(current_curve, original.evaluate(fraction), tolerance=tolerance)
                if len(stations) != 1:
                    raise GeometryError("split station has no unique descendant parameter")
                parent_definition = None
                if _edge_preimage_draft is not None:
                    from .edge_subcurve_preimages import _edge_ancestry_definition
                    parent_definition = _edge_ancestry_definition(model, current)
                _vertex, (left, right) = split_edge_attachments(model, current, stations[0], check)
                if _edge_preimage_draft is not None:
                    from .edge_subcurve_preimages import (_drop_edge_subcurve_records,
                        _record_edge_subcurve_split, _SubcurveEnclosureUnavailable)
                    try:
                        _record_edge_subcurve_split(_edge_preimage_draft, current, stations[0],
                            (left, right), tolerance, model=model,
                            parent_definition=parent_definition)
                    except _SubcurveEnclosureUnavailable as exc:
                        if _edge_preimage_draft.enclosure_failure is not exc:
                            raise  # Never reinterpret a callback exception.
                        # Optional approximation proof is absent; geometry keeps
                        # its established acceptance. Requested proof queries
                        # must explicitly refuse these unqualified children, and
                        # the dead parent's captured occurrences are dropped with
                        # its primary record instead of lingering unsealed.
                        _drop_edge_subcurve_records(_edge_preimage_draft,
                            (current, left, right))
                current, previous = right, fraction

        edge_curves = {edge_id: freeze_edge(model, edge_id) for edge_id in model.edges}
        used_vertices = {vertex for edge in model.edges.values() for vertex in (edge.start, edge.end)}
        vertex_sources = {identifier: {
            *[("face", face_id) for edge_id in model.edges_using_vertex(identifier)
              for face_id in model.faces_using_edge(edge_id)],
            *[("member", member_id) for edge_id in model.edges_using_vertex(identifier)
              for member_id in model.members_using_edge(edge_id)]}
            for identifier in used_vertices}
        canonical = []
        canonical_bounded = [True]
        # Uniform grids over the used vertices (one per tolerance in force) so
        # a point is compared only with vertices in the 27 neighbouring cells.
        # Cells are twice the tolerance, so every vertex within tolerance of a
        # point lies in one of them; the exact predicate below is unchanged.
        grids = {}
        def cell(size, position):
            return tuple(math.floor(float(value)/size) for value in position)
        def grid_for(size):
            grid = grids.get(size)
            if grid is None:
                grid = {}
                for identifier in used_vertices:
                    if identifier in model.vertices:
                        grid.setdefault(cell(size, model.vertex_position(identifier)), []).append(identifier)
                grids[size] = grid
            return grid
        def nearby(point):
            size = 2*tolerance
            if not (math.isfinite(size) and size > 0. and np.all(np.isfinite(point))):
                return sorted(used_vertices)
            try:
                grid = grid_for(size)
                x, y, z = cell(size, point)
            except (ValueError, OverflowError):
                return sorted(used_vertices)
            found = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        found.extend(grid.get((x+dx, y+dy, z+dz), ()))
            return sorted(identifier for identifier in found if identifier in used_vertices)
        def vertex(point, sources):
            grid_exhaustive = (canonical_bounded[0] and 1e-100 < tolerance < 1e100
                               and type(point) is np.ndarray and point.shape == (3,) and point.dtype == np.float64
                               and np.geterr()["under"] == "ignore"
                               and all(math.isfinite(float(value)) and abs(float(value)) < 1e150 for value in point))
            candidates = [identifier for identifier in nearby(point) if identifier in model.vertices
                          and vertex_sources.get(identifier, set()) & sources
                          and np.linalg.norm(model.vertex_position(identifier)-point) <= tolerance]
            if candidates:
                chosen = candidates[0]
                vertex_sources[chosen].update(sources)
                for duplicate in candidates[1:]:
                    vertex_sources[chosen].update(vertex_sources.get(duplicate, ()))
                    prior_definitions = {}
                    if _edge_preimage_draft is not None:
                        from .edge_subcurve_preimages import _edge_ancestry_definition
                        prior_definitions = {edge: _edge_ancestry_definition(model, edge)
                            for edge in model.edges_using_vertex(duplicate)}
                    _merge_vertex(model, duplicate, chosen)
                    if _edge_preimage_draft is not None:
                        from .edge_subcurve_preimages import _rebind_edge_subcurve_incidence
                        _rebind_edge_subcurve_incidence(_edge_preimage_draft,
                            prior_definitions, model=model)
                    used_vertices.discard(duplicate)
                return chosen
            for identifier in canonical:
                # Canonical vertices still in used_vertices were offered by
                # nearby() and rejected above, provided its 2*tolerance grid
                # sees every vertex the norm test can accept (no underflow or
                # overflow at a moderate tolerance and point scale).
                if identifier in used_vertices and grid_exhaustive:
                    continue
                if identifier in model.vertices and vertex_sources[identifier] & sources and np.linalg.norm(model.vertex_position(identifier)-point) <= tolerance:
                    vertex_sources[identifier].update(sources)
                    return identifier
            identifier = model.add_point(*point)
            canonical.append(identifier); used_vertices.add(identifier)
            # The canonical skip above is exhaustive only while every stored
            # canonical vertex is also of moderate size (no overflowing norm).
            if not all(abs(float(value)) < 1e150 for value in model.vertex_position(identifier)):
                canonical_bounded[0] = False
            vertex_sources[identifier] = set(sources)
            for size, grid in grids.items():
                grid.setdefault(cell(size, model.vertex_position(identifier)), []).append(identifier)
            return identifier

        def edge(path, face_id):
            check()
            curve = path.curve
            sources = {("face", owner) for owner in (face_id, *path.owners)}
            sources.update(("member", member) for member in path.member_ids)
            start, end = vertex(curve.evaluate(0.), sources), vertex(curve.evaluate(1.), sources)
            candidates = []
            for edge_id in model.edges_using_vertex(start):
                record = model.edges[edge_id]
                if {record.start, record.end} == {start, end} and _coincident(curve, edge_curves[edge_id], tolerance):
                    candidates.append(edge_id)
            if candidates:
                edge_id = min(candidates)
                if _edge_preimage_draft is not None and len(candidates) > 1:
                    # Canonical shared-edge reuse: capture every participating
                    # authenticated occurrence before the duplicates orphan.
                    from .edge_subcurve_preimages import _record_edge_subcurve_unification
                    _record_edge_subcurve_unification(_edge_preimage_draft, edge_id,
                        tuple(candidate for candidate in candidates if candidate != edge_id),
                        model=model)
            elif isinstance(curve, LinePath):
                edge_id = model.add_line(start, end)
            elif isinstance(curve, BezierPath):
                controls = tuple(model.add_point(*point) for point in curve.controls[1:-1])
                model.mark_construction_vertices(controls)
                edge_id = model.add_spline(start, controls, end)
            elif isinstance(curve, EllipticArc) and (
                abs(np.linalg.norm(curve.u_vector)-np.linalg.norm(curve.v_vector)) <= tolerance
                and abs(float(np.asarray(curve.u_vector) @ curve.v_vector)) <= tolerance):
                via = model.add_point(*curve.evaluate(.5)); model.mark_construction_vertices((via,))
                edge_id = model.add_arc(start, via, end)
            else:
                edge_id = model.add_curve(start, end, curve)
            edge_curves.setdefault(edge_id, freeze_edge(model, edge_id))
            if path.owners:
                joint_ids.add(edge_id)
            if path.decomposition:
                model.tag(EntityRef("edge", edge_id), "intersection_decomposition_seam")
            return OrientedEdge(edge_id, model.edges[edge_id].start == start)

        certified_cells = []
        for arrangement in plan.arrangements:
            check()
            tolerance = arrangement.world_tolerance
            face = model.faces[arrangement.face_id]
            mapping = {index: edge(path, arrangement.face_id) for index, path in enumerate(arrangement.paths)}
            def loop(values):
                return tuple(OrientedEdge(mapping[index].edge, mapping[index].forward == forward) for index, forward in values)
            loops = [(loop(cell.outer), tuple(loop(hole) for hole in cell.holes)) for cell in arrangement.cells]
            if getattr(arrangement, "certified", False):
                # T-b/D8 E2a (S10): certified after the whole batch's vertex mapping (below).
                from .material_arrangement import expected_cell_signatures
                certified_cells.append((arrangement.support,
                    [([(item.edge, item.forward) for item in outer],
                      [[(item.edge, item.forward) for item in hole] for hole in holes]) for outer, holes in loops],
                    expected_cell_signatures(arrangement)))
            if arrangement.orientation < 0:
                # The arrangement walks counterclockwise in the support chart; a face stored clockwise
                # (a pointed cone's facets are) keeps its sense so its neighbours still agree with it.
                def reverse(values):
                    return tuple(OrientedEdge(item.edge, not item.forward) for item in reversed(values))
                loops = [(reverse(outer), tuple(reverse(hole) for hole in holes)) for outer, holes in loops]
            if len(loops) == 1 and {item.edge for item in loops[0][0]} == {item.edge for item in face.loop} and (
                    {tuple(sorted(item.edge for item in hole)) for hole in loops[0][1]} ==
                    {tuple(sorted(item.edge for item in hole)) for hole in face.holes}):
                _store_extruded_support(model,face.id,arrangement.support,check)
                continue
            if len(loops) == 1:
                # A one-cell, area-conserving arrangement only updates trim
                # ports and canonical boundary edges. Keep its face identity
                # and support parameter frame, including existing attachments.
                outer,holes=loops[0]
                corners=model._detect_corners(outer)
                model._put_entity('face',replace(face,loop=outer,holes=holes,
                    corners=corners if len(corners)==4 else face.corners))
                _store_extruded_support(model,face.id,arrangement.support,check)
                continue
            snapshots = capture_face_attachments(model, face.id,check)
            child_definition = (replace(face,surface=arrangement.support,parameterization=None)
                                if not isinstance(face.surface,(Plane,Cylinder)) else face)
            model._delete_entity("face", face.id)
            children = []
            for outer, holes in loops:
                outer,support,parameterization,corners=_child_support(model,child_definition,outer,holes,tolerance)
                if corners is not None and len(corners)!=4:
                    corners=None
                child = model.add_face_from_loop(outer, corners, surface=support)
                model._put_entity("face", replace(model.faces[child], holes=holes,
                    metadata=face.metadata, parameterization=parameterization))
                children.append(child)
            remap_face_attachments(model, face, children, snapshots,check)
            model.record_replacement(face.ref, tuple(EntityRef("face", child) for child in children))
        member_contacts=[]
        for contact in plan.contacts:
            check()
            tolerance = contact.world_tolerance
            sources = {("member", member) for member, _parameter in contact.member_parameters}
            if contact.face_id is not None:
                sources.add(("face", contact.face_id))
            member_contacts.append((contact,vertex(np.asarray(contact.position), sources)))
        _canonicalize_member_edges(model, edge_curves, plan.axes, joint_ids, check)
        # A later physical trace can coincide with a previously artificial
        # seam. Reconcile after all faces and member edges are canonicalized:
        # another face's old boundary must not restore the artificial marker.
        for edge_id in sorted(joint_ids):
            check()
            reference = EntityRef("edge", edge_id)
            if "intersection_decomposition_seam" in model.tags_for(reference):
                model.untag(reference, "intersection_decomposition_seam")
        # Contacts in the immutable plan use the original axis coordinates.
        # Resolve their retained world stations after every split/canonicalization.
        member_contacts = [(replace(contact, member_parameters=tuple(
            (member, _member_parameter(model, member, np.asarray(contact.position),
                contact_ranges[member, parameter], contact.world_tolerance))
            for member, parameter in contact.member_parameters)), station)
            for contact, station in member_contacts]
        for contact in plan.face_contacts:
            check()
            tolerance=contact.world_tolerance
            station=vertex(np.asarray(contact.position),{('face',face) for face in contact.face_ids})
            if policy.intent is ConnectionIntent.CONNECT:
                from .joint_edges import query_joint_edge
                incident={face for edge_id in model.edges_using_vertex(station)
                          for face in model.faces_using_edge(edge_id)}
                owners={sheet for face in incident for use_id in model._face_structural_uses.get(face,())
                        for sheet in (model.face_uses[use_id].sheet_id,)}
                selected_owners={sheet for parent in contact.face_ids
                    for ref in model.resolve_ref(EntityRef('face',parent)) if ref.id in incident
                    for use_id in model._face_structural_uses.get(ref.id,())
                    for sheet in (model.face_uses[use_id].sheet_id,)}
                connected={owner:{owner} for owner in owners}
                for edge_id in model.edges_using_vertex(station):
                    joint=query_joint_edge(model,edge_id)
                    if joint.declared:
                        edge_owners={owner.id for owner in joint.sheets}
                        for owner in edge_owners:
                            connected.setdefault(owner,{owner}).update(edge_owners)
                reached=set();pending=list(sorted(selected_owners)[:1])
                while pending:
                    owner=pending.pop()
                    if owner in reached:continue
                    reached.add(owner);pending.extend(connected.get(owner,{owner})-reached)
                if selected_owners and selected_owners<=reached:
                    # Existing declared incident joints also connect their
                    # common station. Descendant point pairs must not add
                    # redundant relations on every idempotent batch replay.
                    continue
                from .structural import AttachmentKind,AttachmentTargetKind,AttachmentEvidence,ParameterRange
                for parent in contact.face_ids:
                    for face in sorted(ref.id for ref in model.resolve_ref(EntityRef('face',parent))
                                       if ref.id in incident):
                        point=model.vertex_position(station)
                        uv=model.face_local_uv(face,point)
                        residual=_face_contact_residual(model,face,point,uv)
                        if residual>tolerance:
                            raise GeometryError('canonical face contact exceeds its qualified tolerance')
                        model.ensure_attachment(None,AttachmentKind.VERTEX_ON_FACE,
                            AttachmentTargetKind.FACE,face,ParameterRange(0.,0.),
                            tuple(ParameterRange(float(value),float(value)) for value in uv),
                            source_kind='vertex',source_id=station,connection_intent=ConnectionIntent.CONNECT,
                            evidence=AttachmentEvidence.EXACT,max_residual=residual,tolerance_used=tolerance,
                            provenance={'contract':'ANYGEOMETRY_ANALYTIC_FACE_POINT_JOINT_V1'})
        if policy.intent in (ConnectionIntent.CONNECT, ConnectionIntent.IMPRINT):
            from .member_joints import declare_member_contacts, declare_member_boundaries
            declare_member_contacts(model,member_contacts,check,
                                    sheet_target_ids={operand.id for operand in plan.operands if operand.kind=='sheet'},
                                    intent=policy.intent)
            declare_member_boundaries(model,(member for axis in plan.axes for member in axis.member_ids),check,
                                      intent=policy.intent)
        if policy.intent is ConnectionIntent.CONNECT:
            from .joint_edges import declare_joint_edge
            for edge_id in sorted(joint_ids):
                declare_joint_edge(model,edge_id,
                    model.tolerance.effective_length(model.edge_length(edge_id)))
        if certified_cells:
            # T-b/D8 E2a (S10): every vertex of the batch is mapped now (later arrangements, member and
            # face contacts can merge vertices of earlier faces); the persisted cells of every
            # certified arrangement must still form its planar subdivision, else the detached
            # candidate is refused before publication.
            from .material_arrangement import certify_persisted_cells
            for support, cells, expected in certified_cells:
                certify_persisted_cells(support, cells,
                                        lambda edge: (model.edges[edge].start, model.edges[edge].end),
                                        model.vertex_position, check, expected=expected)
        check()
    change_set=model.last_change_set
    reused=model.revision == plan.revision
    if outer_journal is not None:
        after_keys=keys()
        added=tuple(sorted(after_keys-before_keys)); removed=tuple(sorted(before_keys-after_keys))
        updated=tuple(sorted((set(outer_journal.changed)|set(outer_journal.structural_before)) &
                             before_keys & after_keys))
        change_set=ChangeSet(plan.revision,model.revision,added=added,removed=removed,modified=updated)
        reused=not (added or removed or updated)
    return IntersectionApplication(plan, change_set,
        tuple(model.handle("edge", identifier) for identifier in sorted(joint_ids)), reused)


def apply_intersections(model, plan, *, policy):
    """Apply on a detached candidate, then publish one complete owner change.

    Failed or cancelled candidates never consume live allocator IDs. This is
    stronger than the normal edit transaction's monotonic allocation rule.
    """
    junction_policy = _junction_bounds_policy(False) if _policy(policy).max_predicates is not None else nullcontext()
    with root_isolation_memo(), _timed_phase('intersection_application'), junction_policy:
        result = _apply_intersections(model, plan, policy=policy)
    if isinstance(plan, TopologyCertifiedPlan) and type(result) is IntersectionApplication:
        # Only reached after _validate_policy accepted the state-T plan (matrix row 3). A committed
        # publication already carries a TopologyCertifiedChangeSet; an unchanged or cached one gets it here.
        result = TopologyCertifiedApplication(result.plan, _certified_change_set(result.change_set, plan),
                                              result.joint_edges, result.reused,
                                              plan.planning_mode, plan.r_max, plan.certificate_version)
    return result


def _apply_intersections(model, plan, *, policy):
    if not isinstance(plan, IntersectionPlan):
        raise TypeError("apply_intersections needs an IntersectionPlan")
    effective = _policy(policy)
    resolve_planning_mode(effective)                # an invalid mode is refused on every route
    if isinstance(plan, TopologyCertifiedPlan):
        # A state-T plan is validated before any route dispatch (matrix rows 3-11).
        _validate_policy(plan, effective, model)
    from .existing_material_preparation import is_existing_plan, eligible, apply_existing
    # Mode-agnostic routes: the caller's policy object is passed through unchanged (their policy
    # pins and callbacks must see a change made during the apply); the mode is a separate override.
    if is_existing_plan(plan):
        return apply_existing(model, plan, effective, planning_mode="exact")
    if effective.face_connections and eligible(model, plan.operands):
        raise GeometryError('same-chart material preparation plan is not owner-issued')
    from .bounded_nurbs import is_bounded_plan, handles_bounded, apply_bounded
    active_faces = all(h.kind == 'face' and h.id in model.faces for h in plan.operands)
    if is_bounded_plan(plan) or (active_faces and handles_bounded(model, plan.operands)):
        return apply_bounded(model, plan, effective, planning_mode="exact")
    _validate_policy(plan,effective,model)
    if model.model_id != plan.model_id:
        raise GeometryError("intersection plan belongs to another model")
    if model.revision != plan.revision:
        receipt = getattr(model, "_intersection_application_receipt", None)
        if (receipt is not None and receipt[0] is plan and receipt[1] == model.revision
                and receipt[2] == to_dict(model)["checksum"]["value"]):
            from .preparation_epochs import _seals
            face_receipt=getattr(model,'_prepared_face_preimages_receipt',None)
            epoch_bound=model in _seals or (face_receipt is not None and
                getattr(face_receipt[0],'attachment_source_ids',None) is not None)
            epoch_digest=None
            if epoch_bound:
                from .prepared_face_preimages import _current_receipt,_binding_checksum
                from .native_support_snapshots import capture_native_supports
                binding=_current_receipt(model)
                if (binding.attachment_source_ids is None or binding.current_native_supports is None or
                        any(kind not in ('plane','cylinder') for _,kind,_ in binding.current_native_supports) or
                        capture_native_supports(model)!=binding.current_native_supports):
                    raise GeometryError('cached epoch native support binding is unavailable or changed')
                epoch_digest=_binding_checksum(binding)
            if effective.cancellation_check is not None and effective.cancellation_check():
                raise GeometryError("intersection application cancelled")
            # A callback can perform unrelated authoring; refuse the cached
            # result without rolling back that authoring.
            if (getattr(model,'_intersection_application_receipt',None) is not receipt or
                    model.revision!=receipt[1] or _qualified_model_state(model)['checksum']['value']!=receipt[2]):
                raise GeometryError('geometry changed during intersection reuse')
            if epoch_bound:
                binding=_current_receipt(model)
                if (_binding_checksum(binding)!=epoch_digest or
                        capture_native_supports(model)!=binding.current_native_supports):
                    raise GeometryError('native support or epoch source binding changed during cached application')
            return IntersectionApplication(plan, ChangeSet(model.revision, model.revision),
                tuple(model.handle("edge", identifier) for identifier in receipt[3]), True)
        raise GeometryError("intersection plan is stale")
    if to_dict(model)["checksum"]["value"] != plan.source_checksum:
        raise GeometryError("intersection plan source binding changed")
    from .preparation_epochs import _epoch_plan_required,_native_plan_supports
    if _epoch_plan_required(model):
        from .native_support_snapshots import capture_native_supports
        if capture_native_supports(model)!=_native_plan_supports(plan):
            raise GeometryError('epoch intersection plan native support binding changed')
    if model._transaction_journal is not None:
        _refuse_journaled_route(plan)
        from .preparation_epochs import _has_epoch_permit
        if _has_epoch_permit(model):
            raise GeometryError('first preparation of a new epoch requires the committed owner path')
        # Join the caller's atomic transaction. Snapshot adoption is forbidden
        # while a journal is active; ordinary owner writes participate in that
        # journal and its complete rollback instead.
        if effective.intent is ConnectionIntent.REUSE_EXISTING:
            candidate=model.clone(preserve_identity=True)
            outcome=_apply_intersections_in_place(candidate,plan,policy=effective)
            if candidate.revision != plan.revision:
                raise GeometryError("REUSE_EXISTING requires compatible existing topology")
            return IntersectionApplication(plan,ChangeSet(model.revision,model.revision),
                                           outcome.joint_edges,True)
        model._transaction_journal.exact_allocator_rollback=True
        return _apply_intersections_in_place(model,plan,policy=effective)
    from .prepared_face_preimages import (_capture_application_preimages,
        _compose_application_preimages, _publish_application_preimages)
    from .edge_subcurve_preimages import (_capture_edge_subcurve_preimages,
        _finalize_edge_subcurve_preimages, _publish_edge_subcurve_preimages)

    def provenance_check(*_stage):
        if effective.cancellation_check is not None and effective.cancellation_check():
            raise GeometryError("intersection edge provenance cancelled")

    authored_preimages = _capture_application_preimages(model, allow_seed=effective.face_connections)
    from .native_support_snapshots import capture_native_supports
    epoch_native_source=capture_native_supports(model) if authored_preimages is not None and authored_preimages.attachment_source_ids is not None else None
    edge_draft = _capture_edge_subcurve_preimages(model, allow_seed=effective.face_connections,
        cancellation_check=provenance_check)
    candidate = model.clone(preserve_identity=True)
    from .preparation_epochs import (_begin_attachment_draft, _finish_attachment_draft,
                                    _discard_attachment_draft, _consume_epoch)
    _begin_attachment_draft(candidate,authored_preimages)
    try:
        outcome = _apply_intersections_in_place(candidate, plan, policy=policy,
            _edge_preimage_draft=edge_draft)
        recorded = isinstance(plan, TopologyCertifiedPlan) and candidate.revision != plan.revision
        if recorded:
            # Row 3: the route record is part of the sealed candidate document; _publish_topology
            # writes the same record on the owner, so both documents stay equal. An application that
            # changes nothing publishes nothing and records nothing.
            _record_publication(candidate, plan)
        finished_sources=_finish_attachment_draft(candidate)
        attachment_sources,derived_sources=finished_sources if finished_sources is not None else (None,None)
    finally:
        _discard_attachment_draft(candidate)
    authored_preimages = _compose_application_preimages(candidate, authored_preimages,
        outcome.change_set.replacements,attachment_source_ids=attachment_sources,epoch_derived_attachment_ids=derived_sources)
    edge_preimages = _finalize_edge_subcurve_preimages(candidate, edge_draft,
        cancellation_check=provenance_check)
    epoch_native_candidate=capture_native_supports(candidate) if epoch_native_source is not None else None
    sealed_candidate_checksum = (edge_preimages.binding.source_checksum if edge_preimages is not None
        else authored_preimages.source_checksum if authored_preimages is not None
        else to_dict(candidate)["checksum"]["value"])
    if effective.intent is ConnectionIntent.REUSE_EXISTING and candidate.revision != plan.revision:
        raise GeometryError("REUSE_EXISTING requires compatible existing topology")
    if effective.cancellation_check is not None and effective.cancellation_check():
        raise GeometryError("intersection application cancelled before commit")
    if model.revision != plan.revision or _qualified_model_state(model)["checksum"]["value"] != plan.source_checksum:
        raise GeometryError("geometry changed before intersection commit")
    # The last policy callback must not alter the detached result after its
    # topology/provenance proofs. No callback follows this preflight before the
    # committed snapshot is adopted.
    if _qualified_model_state(candidate)["checksum"]["value"] != sealed_candidate_checksum:
        raise GeometryError("intersection candidate changed before commit")
    if epoch_native_source is not None and (capture_native_supports(model)!=epoch_native_source or
            capture_native_supports(candidate)!=epoch_native_candidate):
        raise GeometryError('native geometry changed before epoch commit')
    before = model.revision
    original_faces = set(model.faces)
    covered_faces = {operand.id for operand in plan.operands if operand.kind == "face"}
    for operand in plan.operands:
        if operand.kind == "sheet":
            covered_faces.update(model.face_uses[use].face_id
                                 for use in model.sheets[operand.id].face_use_ids)
    complete_material = effective.face_connections and original_faces <= covered_faces
    staged_epoch=authored_preimages is not None and authored_preimages.attachment_source_ids is not None
    if staged_epoch:
        _publish_application_preimages(candidate,authored_preimages,
            tuple(sorted(candidate.faces)) if complete_material else (),sealed_candidate_checksum)
        import weakref
        if edge_preimages is not None and edge_preimages.owner() is not model:
            raise GeometryError('epoch edge provenance belongs to another owner')
        staged_edges=replace(edge_preimages,owner=weakref.ref(candidate)) if edge_preimages is not None else None
        _publish_edge_subcurve_preimages(candidate,staged_edges,sealed_candidate_checksum)
        candidate._intersection_preparation_receipt=(plan,candidate.revision,sealed_candidate_checksum,
            tuple(sorted(candidate.faces)) if complete_material else ())
    if staged_epoch:
        from copy import deepcopy
        saved_state=dict(model.__dict__)
        # Keep the original store/view and feature-owner objects. Snapshot only
        # mutable containers changed by adoption, and restore them in place.
        saved_containers=[(value,deepcopy(value)) for value in saved_state.values()
                          if type(value) in (dict,list,set)]
        from .preparation_epochs import _seals
        saved_seal=_seals.get(model)
    try:
        _publish_topology(model, candidate.topology_snapshot(), plan if recorded else None)
        result = IntersectionApplication(plan,
            model.last_change_set if model.revision != before else ChangeSet(before, before),
            tuple(model.handle("edge", edge.id) for edge in outcome.joint_edges), model.revision == before)
        # Bind the exact committed result, including an unchanged/idempotent batch.
        # Only complete original-face classification certifies material ownership.
        checksum = to_dict(model)["checksum"]["value"]
        if staged_epoch:
            from .native_support_snapshots import capture_native_supports
            if capture_native_supports(model)!=capture_native_supports(candidate):
                raise GeometryError('native geometry changed during epoch publication')
        if model.revision != before:
            model._intersection_application_receipt = (plan, model.revision,
                checksum, tuple(edge.id for edge in result.joint_edges))
        model._intersection_preparation_receipt = (plan, model.revision, checksum,
            tuple(sorted(model.faces)) if complete_material else ())
        if staged_epoch:
            _publish_application_preimages(model,authored_preimages,
                model._intersection_preparation_receipt[3],checksum)
            _publish_edge_subcurve_preimages(model,edge_preimages,checksum)
        else:
            _publish_application_preimages(model, authored_preimages,
                                          model._intersection_preparation_receipt[3], checksum)
            _publish_edge_subcurve_preimages(model, edge_preimages, checksum)
        if authored_preimages is not None and edge_preimages is not None:
            _consume_epoch(model)
        return result
    except BaseException:
        if staged_epoch:
            for container,contents in saved_containers:
                if type(container) is list:container[:]=contents
                else:container.clear();container.update(contents)
            model.__dict__.clear();model.__dict__.update(saved_state)
            if saved_seal is None:_seals.pop(model,None)
            else:_seals[model]=saved_seal
        raise



def has_current_intersection_preparation(model, *, face_ids=None):
    """Whether the owner has a current, complete material-preparation receipt.

    This read-only query certifies a previous complete batch, not arbitrary
    existing topology. Receipts are local, revision/checksum bound and excluded
    from documents. Edits, clone/load/undo and partial batches require fresh
    classification. Invalid requested faces raise GeometryError.
    """
    from .intersections import _normalize_operand
    from .serialization import _serialized_model_state
    selected = set(model.faces) if face_ids is None else set()
    if face_ids is not None:
        for value in face_ids:
            operand = _normalize_operand(model, value)
            if operand.kind != "face":
                raise GeometryError("preparation binding requires face operands")
            selected.add(operand.id)
    receipt = getattr(model, "_intersection_preparation_receipt", None)
    if receipt is None:
        return False
    plan, revision, checksum, coverage = receipt
    eligible = bool(coverage and selected <= set(coverage)
                and model.revision == revision
                and model.model_id == plan.model_id
                and plan.content_checksum == _plan_content(plan))
    if not eligible:
        return False
    if _serialized_model_state(model)["checksum"]["value"] == checksum:
        return True
    # A changed raw state no longer has prior qualification. Preserve the
    # public invalid-topology error before reporting a valid stale receipt.
    # Exact unchanged content keeps the fingerprint-only read path above.
    to_dict(model)
    return False


def clone_prepared_geometry(model, *, new_preparation_epoch=False):
    """Detach an exact mesh-attempt copy, preserving a valid local owner proof.

    Ordinary unprepared copies retain normal clone behavior. Prepared copies
    preserve logical document identity and their complete bound receipt only
    after verifying the source and copy have identical committed checksums.
    The source stays unchanged; edits to either copy invalidate its binding.
    """
    if type(new_preparation_epoch) is not bool:
        raise GeometryError('new_preparation_epoch must be a plain boolean')
    if new_preparation_epoch:
        if model._transaction_journal is not None:
            raise GeometryError('a new preparation epoch requires committed source geometry')
        from .native_support_snapshots import capture_native_supports
        before=to_dict(model);native=capture_native_supports(model)
        made=model.clone(preserve_identity=False)
        copied=to_dict(made)
        source_payload={k:v for k,v in before.items() if k not in ('model_id','checksum')}
        clone_payload={k:v for k,v in copied.items() if k not in ('model_id','checksum')}
        if (made.model_id==model.model_id or source_payload!=clone_payload or
                capture_native_supports(made)!=native or capture_native_supports(model)!=native or to_dict(model)!=before):
            raise GeometryError('preparation epoch clone changed source or native geometry')
        from .preparation_epochs import _issue_epoch
        _issue_epoch(made)
        from .prepared_material_source_transfer import _register_source_clone
        _register_source_clone(model, made)
        return made
    prepared = has_current_intersection_preparation(model)
    receipt = getattr(model, "_intersection_preparation_receipt", None)
    made = model.clone(preserve_identity=prepared)
    if prepared:
        checksum = receipt[2]
        if (to_dict(model)["checksum"]["value"] != checksum
                or to_dict(made)["checksum"]["value"] != checksum):
            raise GeometryError("prepared geometry changed during detached copying")
        made._intersection_preparation_receipt = receipt
        from .prepared_face_preimages import _copy_current_preimages
        _copy_current_preimages(model, made)
        from .edge_subcurve_preimages import _copy_current_edge_subcurve_preimages
        _copy_current_edge_subcurve_preimages(model, made)
    from .prepared_material_source_transfer import _register_source_clone
    _register_source_clone(model, made)
    return made


def set_prepared_face_corners(model, updates):
    """Atomically change corner indices without losing a current material proof.

    Only explicit analytic supports without a separate parameterization qualify:
    their material is defined by the unchanged support and trim loops, not corner
    indices. A stale or partial preparation, topology-backed map, nested transaction
    or any other document change is rejected. This does not refresh application
    receipts or make old plans applicable after an edit.

    As with batch application, change hooks precede publication of the new
    receipt. The proof query is fail-closed during hooks and current on return.
    """
    if model._transaction_journal is not None:
        raise GeometryError("prepared corner editing requires no active transaction")
    if not has_current_intersection_preparation(model):
        raise GeometryError("prepared corner editing requires a current complete preparation")
    original = to_dict(model)
    receipt = model._intersection_preparation_receipt
    from .prepared_face_preimages import _current_receipt, _publish_application_preimages
    from .edge_subcurve_preimages import (_capture_edge_subcurve_preimages,
        _finalize_edge_subcurve_preimages, _publish_edge_subcurve_preimages)
    edge_draft = _capture_edge_subcurve_preimages(model, allow_seed=False)
    try:
        authored_preimages = _current_receipt(model)
    except GeometryError:
        authored_preimages = None
    try:
        requested = {key: tuple(value) for key, value in updates.items()}
    except (AttributeError, TypeError) as exc:
        raise GeometryError("prepared corner updates must map face IDs to corner indices") from exc
    for identifier in requested:
        if type(identifier) is not int or identifier not in model.faces:
            raise GeometryError("prepared corner update references an invalid face")
        face = model.faces[identifier]
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
               for value in requested[identifier]):
            raise GeometryError("prepared corner indices must be integers")
        if (type(face.surface) not in (Plane, Cylinder, Cone, ExtrudedSurface)
                or face.parameterization is not None):
            raise GeometryError("prepared corner editing requires an explicit corner-independent support")
    candidate = clone_prepared_geometry(model)
    with candidate.transaction():
        for identifier in sorted(requested):
            candidate.set_face_corners(identifier, requested[identifier])
    actual = to_dict(candidate)
    # Compare the entire certified document, not only the edited faces. Even
    # derived feature/ownership changes must fail this narrow operation closed.
    expected = dict(original)
    expected['faces'] = [dict(face, corners=list(candidate.faces[face['id']].corners))
                         if face['id'] in requested else face for face in original['faces']]
    expected['revision'], expected['checksum'] = actual['revision'], actual['checksum']
    if actual != expected:
        raise GeometryError("prepared corner update changed fields other than corner indices")
    if to_dict(model) != original:
        raise GeometryError("geometry changed during prepared corner editing")
    if actual == original:
        return
    edge_preimages = _finalize_edge_subcurve_preimages(candidate, edge_draft)
    model.restore_topology(candidate.topology_snapshot())
    plan, _revision, _checksum, coverage = receipt
    model._intersection_preparation_receipt = (
        plan, model.revision, to_dict(model)['checksum']['value'], coverage)
    _publish_application_preimages(model, authored_preimages, coverage,
                                  model._intersection_preparation_receipt[2])
    _publish_edge_subcurve_preimages(model, edge_preimages,
                                   model._intersection_preparation_receipt[2])
