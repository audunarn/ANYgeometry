"""Plan transport: a closed codec for intersection plans and the owner's verified adoption of one.

Design A of the verified-adoption note (2026-11-11): a worker process *plans only*; the owner process
decodes the plan with this closed codec, checks it against its **own** model, operands and policy, and
then runs the ordinary, unchanged ``apply_intersections``. The owner's preparation, application and
preimage receipts are therefore minted by the existing owner path in the owner process. This module
adds **no receipt-writing and no registry-writing path**: it never sets a receipt attribute and never
touches ``preparation_epochs`` / existing-material / bounded registries.

Guarantee (and its limit). Adoption gives *fresh owner application of a trusted plan*. The checks bind
the plan to the owner's model id, revision, freshly computed source checksum, normalised operands and
policy, and recompute its content checksum. They do NOT re-derive the planner's arrangement: a
self-consistent wrong plan (a wrong arrangement with a correctly recomputed content checksum) is applied
and certified. Trust in the planner's execution rests on the caller's runtime checks (same source digest,
runtime lock, smoke digest). Out of scope: a malicious worker or a local attacker who can write the
worker's output or the package itself.

The wire is canonical JSON (UTF-8). Grammar: primitives, finite floats as IEEE-754 hex, NumPy float64 /
int64 / bool scalars, C-contiguous finite float64 arrays, tuples, lists, ``UUID``, ``ConnectionIntent``,
the policy's plain fields, and an exact-type record registry (declared init fields in source order).
No pickle, no import by name, no callable. Shared objects keep their sharing (ordinal references);
cycles, unknown types, unknown or missing fields, non-empty unknown caches, overlapping array memory and
any non-canonical packet refuse. Decoding reconstructs through the ordinary constructors and must
re-encode to the identical bytes.
"""
import hashlib
import json
import math
import struct
from dataclasses import fields
from uuid import UUID

import numpy as np

from .errors import GeometryError

__all__ = ("PlanAdoptionRefused", "encode_intersection_plan", "adopt_intersection_plan",
           "PLAN_CODEC_FORMAT", "PLAN_CODEC_VERSION")

PLAN_CODEC_FORMAT = "anygeometry.intersection-plan"
PLAN_CODEC_VERSION = 1
_MAX_PACKET_BYTES = 64_000_000
_MAX_NODES = 5_000_000
# Bound on the plan's EXPANDED weight (every shared reference counted at its full size; one unit per node
# plus one per 8 payload bytes of a string or array): the owner's content hash, constructors and the apply
# walk the expansion, so a small packet of nested shared references to small or large payloads must not
# expand without bound. Measured real plans (sector60 / r7 / N2c-size surrogate) weigh far less (see the
# A0/A1 reports); a plan beyond the bound is refused and planned serially.
_MAX_EXPANDED = 2_000_000
_CONTAINERS = frozenset(('t', 'l', 'a', 'p', 'R'))


def _children(node):
    return node[3] if node[0] == 'R' else node[2] if node[0] in ('t', 'l', 'p') else ()


def _weight(node, sizes):
    """Expanded weight of a well-formed wire node whose container ordinals are already in ``sizes``."""
    tag = node[0]
    if tag == 'r' or tag in _CONTAINERS:
        return sizes[node[1]]
    if tag == 's':
        return 1 + len(node[1]) // 8
    return 1


def _array_weight(hex_text):
    return 1 + len(hex_text) // 16


class PlanAdoptionRefused(GeometryError):
    """A plan packet was refused; the caller runs the ordinary serial planner instead.

    ``reason`` is a stable short code (see ``REASONS``); the message adds detail. A refusal never
    changes the model, its receipts or any owner registry.
    """

    REASONS = ("argument", "journaled_model", "epoch_model", "registry_route", "packet",
               "codec", "record_type", "content", "route", "operands", "model", "revision",
               "source", "policy", "job", "encode")

    def __init__(self, reason, message):
        if reason not in self.REASONS:
            raise ValueError(f"unknown plan adoption refusal reason {reason!r}")
        super().__init__(f"plan adoption refused ({reason}): {message}")
        self.reason = reason


class _Refuse(Exception):
    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason


# --------------------------------------------------------------------------------------------- registry

def _registry():
    """Exact record classes and their declared init field names, in source order (the wire table)."""
    from .batch_intersections import IntersectionPlan, TopologyCertifiedPlan, FacePointContact
    from .identity import EntityHandle
    from .member_arrangements import MemberAxisArrangement, MemberPointContact
    from .surfaces import Plane, Cylinder
    from .arrangement_geometry import LinePath
    from .material_arrangement import ArrangementPath, ArrangementPoint, ArrangementCell, MaterialArrangement
    from .exact_curves import EllipticArc, CylinderIntersectionCurve, _CylinderSupport
    from .branch_curves import BezierQuadricCurve
    from .quadric_algebra import QuadricSupport
    from .plane_ruled import PlaneRuledSupport
    return (IntersectionPlan, TopologyCertifiedPlan, FacePointContact, EntityHandle, MemberAxisArrangement,
            MemberPointContact, Plane, Cylinder, LinePath, ArrangementPath, ArrangementPoint, ArrangementCell,
            MaterialArrangement, EllipticArc, CylinderIntersectionCurve, _CylinderSupport, BezierQuadricCurve,
            QuadricSupport, PlaneRuledSupport)


_PLAN_FIELDS = ('model_id', 'revision', 'source_checksum', 'operands', 'policy', 'arrangements', 'axes',
                'contacts', 'content_checksum', 'face_contacts')
# Field order is copied from the owner declarations; a changed declaration disables the codec (refusal),
# it never silently re-maps a field.
_RECORD_FIELDS = (
    ('IntersectionPlan', _PLAN_FIELDS),
    ('TopologyCertifiedPlan', _PLAN_FIELDS + ('planning_mode', 'route_checksum', 'certificate_version', 'r_max',
                                              'r_pairs', 'route_tolerance', 'seam_policy', 'seam_policy_version')),
    ('FacePointContact', ('face_ids', 'position', 'world_tolerance')),
    ('EntityHandle', ('model_id', 'kind', 'id')),
    ('MemberAxisArrangement', ('edge_id', 'member_ids', 'curve', 'split_parameters', 'world_tolerance')),
    ('MemberPointContact', ('face_id', 'position', 'member_parameters', 'edge_ids', 'world_tolerance')),
    ('Plane', ('origin', 'u_vector', 'v_vector')),
    ('Cylinder', ('origin', 'axis', 'radial_direction', 'radius', 'height', 'start_angle', 'sweep_angle')),
    ('LinePath', ('start', 'end')),
    ('ArrangementPath', ('curve', 'source_edge', 'owners', 'decomposition', 'member_ids')),
    ('ArrangementPoint', ('position', 'member_ids')),
    ('ArrangementCell', ('outer', 'holes')),
    ('MaterialArrangement', ('face_id', 'paths', 'cells', 'area', 'world_tolerance', 'native_area_tolerance',
                             'support', 'orientation', 'certified')),
    ('EllipticArc', ('center', 'u_vector', 'v_vector', 'start_angle', 'sweep_angle')),
    ('CylinderIntersectionCurve', ('first', 'second', 'start_angle', 'sweep_angle', 'branch', 'parameterization',
                                   'transform')),
    ('_CylinderSupport', ('origin', 'axis', 'radial_direction', 'radius', 'height', 'start_angle', 'sweep_angle')),
    ('BezierQuadricCurve', ('first', 'second', 'start', 'sweep', 'branch', 'parameterization', 'transform')),
    ('QuadricSupport', ('kind', 'origin', 'axis', 'radius', 'slope', 'matrix', 'linear', 'constant', 'u_vector',
                        'v_vector')),
    ('PlaneRuledSupport', ('base', 'directrix_vector', 'ruling', 't0', 'length')),
)
# Policy plain fields; ``cancellation_check`` (a process-local callable) never crosses.
_POLICY_FIELDS = ('intent', 'max_predicates', 'max_candidate_pairs', 'face_connections', 'member_connections',
                  'member_face_connections', 'planning_mode', 'seam_policy')
# Derived state that is not transmitted: (class name, attribute, kind). "recomputed" values are checked
# bit-equal to a fresh constructor product at encode; "memo" is a pure answer cache started empty.
_DERIVED = {'Plane': {'normal': 'recomputed'}, 'Cylinder': {'_circumferential': 'recomputed'}}
_INIT_FALSE = {'BezierQuadricCurve': {'_inversions': 'memo'}}


def _schema():
    from .batch_intersections import IntersectionBatchPolicy
    classes = _registry()
    runtime = tuple((cls.__name__, tuple(f.name for f in fields(cls) if f.init)) for cls in classes)
    if runtime != _RECORD_FIELDS:
        raise _Refuse("codec", "an owner record declaration changed; the plan codec is disabled")
    for cls in classes:
        unknown = {f.name for f in fields(cls) if not f.init} - set(_INIT_FALSE.get(cls.__name__, {}))
        if unknown:
            raise _Refuse("codec", f"{cls.__name__} has undeclared derived fields {sorted(unknown)}")
    if tuple(f.name for f in fields(IntersectionBatchPolicy)) != (
            _POLICY_FIELDS[:3] + ('cancellation_check',) + _POLICY_FIELDS[3:]):
        raise _Refuse("codec", "the intersection policy declaration changed; the plan codec is disabled")
    digest = hashlib.sha256(_canonical([_RECORD_FIELDS, _POLICY_FIELDS, sorted(_DERIVED), sorted(_INIT_FALSE),
                                        "plan-codec-1"])).hexdigest()
    return {cls.__name__: cls for cls in classes}, digest


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')


# ------------------------------------------------------------------------------------------------ equality

def _same(a, b):
    """Bit-exact structural equality used for the constructor-idempotency check."""
    t = type(a)
    if t is not type(b):
        return False
    if a is b and t is not float and t is not np.float64:
        return True
    if t is float:
        return struct.pack('!d', a) == struct.pack('!d', b)
    if t in (np.float64, np.int64, np.bool_):
        return a.tobytes() == b.tobytes()
    if t is np.ndarray:
        return (a.dtype == b.dtype and a.shape == b.shape and a.flags.c_contiguous == b.flags.c_contiguous
                and a.tobytes() == b.tobytes())
    if t in (tuple, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if t in (str, int, bool, type(None), UUID):
        return a == b
    from .structural import ConnectionIntent
    if t is ConnectionIntent:
        return a is b
    if hasattr(t, '__dataclass_fields__'):
        return all(_same(getattr(a, f.name), getattr(b, f.name)) for f in fields(a) if f.init)
    return False


# ------------------------------------------------------------------------------------------------- encode

def _encode_tree(plan, classes):
    from .batch_intersections import IntersectionBatchPolicy
    from .structural import ConnectionIntent
    names = {cls: name for name, cls in classes.items()}
    memo, active, spans, sizes = {}, set(), [], {}
    nodes = 0

    def float_hex(value):
        if not math.isfinite(value):
            raise _Refuse("encode", "non-finite float")
        return struct.pack('!d', value).hex()

    def encode(v):
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_NODES:
            raise _Refuse("encode", "plan node limit")
        node = _encode_node(v)
        if node[0] in _CONTAINERS:
            # Records are bounded before their constructor runs (below); the plan itself is a record.
            sizes[node[1]] = (_array_weight(node[3]) if node[0] == 'a' else
                              1 + sum(_weight(child, sizes) for child in _children(node)))
        return node

    def _encode_node(v):
        t = type(v)
        if v is None:
            return ['n']
        if t is bool:
            return ['b', v]
        if t is int:
            return ['i', v]
        if t is str:
            return ['s', v]
        if t is float:
            return ['f', float_hex(v)]
        if t is np.float64:
            if not np.isfinite(v):
                raise _Refuse("encode", "non-finite NumPy scalar")
            return ['f8', v.tobytes().hex()]
        if t is np.int64:
            return ['i8', v.tobytes().hex()]
        if t is np.bool_:
            return ['b1', v.tobytes().hex()]
        if t is UUID:
            return ['u', str(v)]
        if t is ConnectionIntent:
            return ['e', v.value]
        if t not in (tuple, list, np.ndarray, IntersectionBatchPolicy) and t not in names:
            raise _Refuse("encode", f"unsupported plan value type {t.__module__}.{t.__qualname__}")
        if id(v) in active:
            raise _Refuse("encode", "cyclic plan value")
        if id(v) in memo:
            return ['r', memo[id(v)]]
        ordinal = len(memo)
        memo[id(v)] = ordinal
        active.add(id(v))
        try:
            if t in (tuple, list):
                return ['t' if t is tuple else 'l', ordinal, [encode(x) for x in v]]
            if t is np.ndarray:
                if v.dtype != np.dtype('<f8') or not v.flags.c_contiguous or not np.all(np.isfinite(v)):
                    raise _Refuse("encode", "array schema (finite C-contiguous float64 only)")
                if v.nbytes:
                    start = v.__array_interface__['data'][0]
                    spans.append((start, start + v.nbytes))
                return ['a', ordinal, list(v.shape), v.tobytes().hex(), bool(v.flags.writeable)]
            if t is IntersectionBatchPolicy:
                # Plain fields only: the process-local cancellation callable never crosses.
                return ['p', ordinal, [encode(getattr(v, name)) for name in _POLICY_FIELDS]]
            name = names[t]
            declared = dict(_RECORD_FIELDS)[name]
            values = {key: getattr(v, key) for key in declared}
            args = [encode(values[key]) for key in declared]
            if 1 + sum(_weight(x, sizes) for x in args) > _MAX_EXPANDED:
                raise _Refuse("encode", "plan expanded size limit")
            _check_record_state(v, name, t, values)
            return ['R', ordinal, name, args]
        finally:
            active.discard(id(v))

    tree = encode(plan)
    spans.sort()
    for (_, end), (start, _) in zip(spans, spans[1:]):
        if start < end:
            raise _Refuse("encode", "distinct plan arrays share memory")
    return tree


def _check_record_state(value, name, cls, values):
    """The record is a pure constructor product: re-construction is bit-identical; derived state is known."""
    try:
        fresh = cls(**values)
    except Exception as exc:
        raise _Refuse("encode", f"{name} does not re-construct: {type(exc).__name__}") from exc
    if not _same(value, fresh):
        raise _Refuse("encode", f"{name} constructor changes its fields")
    extra = set(getattr(value, '__dict__', ()))
    extra -= set(f.name for f in fields(cls))
    allowed = _DERIVED.get(name, {})
    if extra - set(allowed):
        raise _Refuse("encode", f"{name} carries unknown state {sorted(extra - set(allowed))}")
    for key in extra:
        if not _same(getattr(value, key), getattr(fresh, key)):
            raise _Refuse("encode", f"{name}.{key} differs from its recomputation")
    for f in fields(cls):
        if not f.init:
            current = getattr(value, f.name)
            if _INIT_FALSE.get(name, {}).get(f.name) != 'memo' or type(current) is not dict:
                raise _Refuse("encode", f"{name}.{f.name} is not a known answer memo")


def _plan_route(plan):
    """The producer's resolved (planning mode, seam policy), bound in the envelope beside the record type."""
    from .batch_intersections import TopologyCertifiedPlan
    if type(plan) is TopologyCertifiedPlan:
        return [plan.planning_mode, plan.seam_policy]
    return ["exact", "baseline"]


def encode_intersection_plan(plan, *, job_id=None):
    """Encode an owner-made intersection plan as canonical bytes (closed codec, no pickle).

    ``job_id`` (a non-empty string) binds the packet to one worker job; ``adopt_intersection_plan``
    refuses a packet whose job id differs from the caller's. Plans issued by a registry route
    (existing-material, bounded, epoch native binding) cannot be transported and refuse.
    Raises ``PlanAdoptionRefused`` (reason ``encode`` or ``codec``).
    """
    from .batch_intersections import IntersectionPlan, TopologyCertifiedPlan, _plan_content
    try:
        if type(plan) not in (IntersectionPlan, TopologyCertifiedPlan):
            raise _Refuse("encode", "not an IntersectionPlan or TopologyCertifiedPlan")
        if job_id is not None and (type(job_id) is not str or not job_id):
            raise _Refuse("argument", "job_id must be a non-empty string")
        from .existing_material_preparation import is_existing_plan
        from .bounded_nurbs import is_bounded_plan
        from .preparation_epochs import _plan_bindings
        if is_existing_plan(plan) or is_bounded_plan(plan) or id(plan) in _plan_bindings:
            raise _Refuse("encode", "registry-issued plans cannot be transported")
        classes, schema = _schema()
        tree = _encode_tree(plan, classes)          # cycle guard and size bounds before any content hash
        if plan.content_checksum != _plan_content(plan):
            raise _Refuse("encode", "plan content checksum does not recompute")
        body = _canonical(tree)
        return _canonical({"format": PLAN_CODEC_FORMAT, "version": PLAN_CODEC_VERSION, "schema": schema,
                           "job_id": job_id, "route": _plan_route(plan),
                           "plan_sha256": hashlib.sha256(body).hexdigest(), "plan": tree})
    except _Refuse as exc:
        raise PlanAdoptionRefused(exc.reason, str(exc)) from None
    except PlanAdoptionRefused:
        raise
    except (GeometryError, TypeError, ValueError, AttributeError, KeyError, OverflowError, RecursionError) as exc:
        raise PlanAdoptionRefused("encode", f"{type(exc).__name__}: {exc}") from None


# ------------------------------------------------------------------------------------------------- decode

def _hex_bytes(text, size=None):
    if type(text) is not str:
        raise _Refuse("packet", "byte field is not a string")
    raw = bytes.fromhex(text)
    if raw.hex() != text or (size is not None and len(raw) != size):
        raise _Refuse("packet", "non-canonical byte field")
    return raw


def _decode_tree(tree, classes):
    from .batch_intersections import IntersectionBatchPolicy
    from .structural import ConnectionIntent
    record_fields = dict(_RECORD_FIELDS)
    memo, sizes = [], []
    nodes = 0

    def bound(children):
        # Expanded size of a container from its already decoded children; checked BEFORE any constructor.
        size = 1 + sum(_weight(c, sizes) for c in children)
        if size > _MAX_EXPANDED:
            raise _Refuse("packet", "plan expanded size limit")
        return size

    def decode(p):
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_NODES:
            raise _Refuse("packet", "plan node limit")
        if type(p) is not list or not p or type(p[0]) is not str:
            raise _Refuse("packet", "malformed value")
        tag, n = p[0], len(p)
        if tag == 'n' and n == 1:
            return None
        if tag == 'b' and n == 2 and type(p[1]) is bool:
            return p[1]
        if tag == 'i' and n == 2 and type(p[1]) is int:
            return p[1]
        if tag == 's' and n == 2 and type(p[1]) is str:
            return p[1]
        if tag == 'f' and n == 2:
            value = struct.unpack('!d', _hex_bytes(p[1], 8))[0]
            if not math.isfinite(value):
                raise _Refuse("packet", "non-finite float")
            return value
        if tag in ('f8', 'i8', 'b1') and n == 2:
            dtype = {'f8': np.float64, 'i8': np.int64, 'b1': np.bool_}[tag]
            value = np.frombuffer(_hex_bytes(p[1], np.dtype(dtype).itemsize), dtype=dtype)[0]
            if type(value) is not dtype or (tag == 'f8' and not np.isfinite(value)):
                raise _Refuse("packet", "scalar schema")
            if tag == 'b1' and p[1] not in ('00', '01'):
                raise _Refuse("packet", "non-canonical NumPy bool")
            return value
        if tag == 'u' and n == 2 and type(p[1]) is str:
            value = UUID(p[1])
            if str(value) != p[1]:
                raise _Refuse("packet", "non-canonical UUID")
            return value
        if tag == 'e' and n == 2 and type(p[1]) is str:
            return ConnectionIntent(p[1])
        if tag == 'r' and n == 2 and type(p[1]) is int:
            if not 0 <= p[1] < len(memo) or memo[p[1]] is None:
                raise _Refuse("packet", "invalid reference")
            return memo[p[1]]
        if n < 3 or type(p[1]) is not int or p[1] != len(memo):
            raise _Refuse("packet", "object ordinal")
        ordinal = len(memo)
        memo.append(None)
        sizes.append(None)
        if tag in ('t', 'l') and n == 3 and type(p[2]) is list:
            items = [decode(x) for x in p[2]]
            sizes[ordinal] = bound(p[2])
            value = tuple(items) if tag == 't' else items
        elif tag == 'a' and n == 5 and type(p[2]) is list and type(p[4]) is bool:
            if any(type(k) is not int or k < 0 for k in p[2]):
                raise _Refuse("packet", "array shape")
            value = np.frombuffer(_hex_bytes(p[3]), dtype='<f8').copy().reshape(p[2])
            if not np.all(np.isfinite(value)):
                raise _Refuse("packet", "non-finite array")
            value.flags.writeable = p[4]
            sizes[ordinal] = _array_weight(p[3])
        elif tag == 'p' and n == 3 and type(p[2]) is list and len(p[2]) == len(_POLICY_FIELDS):
            items = [decode(x) for x in p[2]]
            sizes[ordinal] = bound(p[2])
            value = IntersectionBatchPolicy(**dict(zip(_POLICY_FIELDS, items)))
        elif tag == 'R' and n == 4 and type(p[2]) is str and p[2] in classes and type(p[3]) is list:
            declared = record_fields[p[2]]
            if len(p[3]) != len(declared):
                raise _Refuse("packet", f"{p[2]} field count")
            items = [decode(x) for x in p[3]]
            sizes[ordinal] = bound(p[3])
            value = classes[p[2]](**dict(zip(declared, items)))
            if type(value) is not classes[p[2]]:
                raise _Refuse("record_type", f"{p[2]} constructor returned another type")
        else:
            raise _Refuse("packet", f"unknown value tag {tag!r}")
        memo[ordinal] = value
        return value

    return decode(tree)


def _decode_packet(packet, classes, schema):
    from .batch_intersections import IntersectionPlan, TopologyCertifiedPlan
    if type(packet) is not bytes:
        raise _Refuse("argument", "packet must be bytes")
    if len(packet) > _MAX_PACKET_BYTES:
        raise _Refuse("packet", "packet size limit")
    try:
        envelope = json.loads(packet.decode('ascii'))
    except (UnicodeDecodeError, ValueError) as exc:
        raise _Refuse("packet", f"not canonical JSON: {type(exc).__name__}") from None
    if (type(envelope) is not dict or
            set(envelope) != {"format", "version", "schema", "job_id", "route", "plan_sha256", "plan"}):
        raise _Refuse("packet", "envelope fields")
    if envelope["format"] != PLAN_CODEC_FORMAT or type(envelope["version"]) is not int or \
            envelope["version"] != PLAN_CODEC_VERSION:
        raise _Refuse("packet", "format or version")
    if envelope["schema"] != schema:
        raise _Refuse("codec", "codec schema differs between producer and owner")
    if hashlib.sha256(_canonical(envelope["plan"])).hexdigest() != envelope["plan_sha256"]:
        raise _Refuse("packet", "plan body digest mismatch")
    if _canonical(envelope) != packet:
        raise _Refuse("packet", "non-canonical packet")
    plan = _decode_tree(envelope["plan"], classes)
    if type(plan) not in (IntersectionPlan, TopologyCertifiedPlan):
        raise _Refuse("record_type", "packet does not hold an intersection plan")
    if envelope["route"] != _plan_route(plan):
        raise _Refuse("route", "the plan's record type or route fields differ from the producer's route")
    # The reconstruction must be canonical: the owner's own encoding of the decoded plan is the packet.
    if _canonical(_encode_tree(plan, classes)) != _canonical(envelope["plan"]):
        raise _Refuse("codec", "decoded plan does not re-encode to the packet")
    return plan, envelope["job_id"], envelope["route"]


# -------------------------------------------------------------------------------------------------- adopt

def _expected_stored_policy(policy, mode):
    """The policy the ordinary planner stores in a plan of a non-registry route (callable excluded)."""
    from dataclasses import replace
    return replace(policy, cancellation_check=None, seam_policy=None,
                   planning_mode=(policy.planning_mode if mode == "topology_certified" else None))


def adopt_intersection_plan(model, packet, *, policy, operands, job_id, expected_source_checksum):
    """Decode a worker's plan packet and verify it against this process's own model; return the plan.

    The caller then runs the ordinary ``apply_intersections(model, plan, policy=policy)`` with the SAME
    ``policy`` object, which mints every receipt through the unchanged owner path. This function reads
    only: it writes no receipt attribute and no owner registry, and it does not change the model.

    Checks, all against the owner's objects: argument types; refusal of journaled models, epoch models and
    operands selected by the existing-material or bounded routes (decided from the model and operands,
    never from the plan); canonical closed decode with exact record types; job id; content checksum
    recomputed; route carrier and route/mode/seam agreement; ``plan.operands`` == the normalised
    ``operands``; model id, revision and freshly computed source checksum == ``expected_source_checksum``
    == ``plan.source_checksum``; the stored policy's plain fields == the caller's (``cancellation_check``
    excluded: callables never cross).

    Raises ``PlanAdoptionRefused`` (a ``GeometryError``) with a stable ``reason``.
    """
    try:
        return _adopt(model, packet, policy=policy, operands=operands, job_id=job_id,
                      expected_source_checksum=expected_source_checksum)
    except _Refuse as exc:
        raise PlanAdoptionRefused(exc.reason, str(exc)) from None
    except PlanAdoptionRefused:
        raise
    except (GeometryError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError,
            RecursionError) as exc:
        raise PlanAdoptionRefused("packet", f"{type(exc).__name__}: {exc}") from None


def _adopt(model, packet, *, policy, operands, job_id, expected_source_checksum):
    from .batch_intersections import (IntersectionBatchPolicy, _check_policy_in_place, _plan_content,
                                      _validate_policy, resolve_planning_mode, resolve_seam_policy)
    from .model import GeometryModel
    from .serialization import to_dict
    from .intersections import _normalize_operand
    if not isinstance(model, GeometryModel):
        raise _Refuse("argument", "model must be a GeometryModel")
    if type(policy) is not IntersectionBatchPolicy:
        raise _Refuse("argument", "policy must be an IntersectionBatchPolicy object")
    try:
        _check_policy_in_place(policy)
    except GeometryError as exc:
        raise _Refuse("argument", f"policy: {exc}") from None
    if type(job_id) is not str or not job_id:
        raise _Refuse("argument", "job_id must be a non-empty string")
    if (type(expected_source_checksum) is not str or len(expected_source_checksum) != 64 or
            any(c not in '0123456789abcdef' for c in expected_source_checksum)):
        raise _Refuse("argument", "expected_source_checksum must be a SHA-256 hex digest")
    if type(packet) is not bytes:
        raise _Refuse("argument", "packet must be bytes")
    # Owner-side refusals, decided from the owner's model and operands only.
    if model._transaction_journal is not None:
        raise _Refuse("journaled_model", "the model has an active transaction journal")
    from .preparation_epochs import _epoch_plan_required
    if _epoch_plan_required(model):
        raise _Refuse("epoch_model", "epoch models plan in the owner process (v1)")
    handles = tuple(sorted(set(_normalize_operand(model, value) for value in tuple(operands))))
    from .existing_material_preparation import eligible
    from .bounded_nurbs import handles_bounded
    if eligible(model, handles) or handles_bounded(model, handles):
        raise _Refuse("registry_route", "the operands select an owner-issued (registry) planning route")
    classes, schema = _schema()
    plan, packet_job, route = _decode_packet(packet, classes, schema)
    if packet_job != job_id:
        raise _Refuse("job", "the packet belongs to another job")
    if route != [resolve_planning_mode(policy), resolve_seam_policy(policy)]:
        raise _Refuse("route", "the producer's planning mode / seam policy differ from the owner's")
    if plan.content_checksum != _plan_content(plan):
        raise _Refuse("content", "plan content checksum does not recompute")
    if plan.operands != handles or any(type(h) is not type(k) for h, k in zip(plan.operands, handles)):
        raise _Refuse("operands", "the plan's operands are not the requested operands")
    if plan.model_id != model.model_id:
        raise _Refuse("model", "the plan belongs to another model")
    if type(plan.revision) is not int or plan.revision != model.revision:
        raise _Refuse("revision", "the plan was made at another model revision")
    current = to_dict(model)["checksum"]["value"]
    if not (current == expected_source_checksum == plan.source_checksum):
        raise _Refuse("source", "the model source changed (stale plan or stale job)")
    if plan.policy != _expected_stored_policy(policy, resolve_planning_mode(policy)):
        raise _Refuse("policy", "the plan's stored policy differs from the owner's policy")
    try:
        # The owner's own validator: route carrier (state T), mode / seam agreement, content (state E).
        _validate_policy(plan, policy, model)
    except GeometryError as exc:
        raise _Refuse("route", str(exc)) from None
    if model._transaction_journal is not None or model.revision != plan.revision:
        raise _Refuse("model", "the model changed during adoption")
    return plan
