"""Owner-bound whole-interval evidence for captured native Arc functions.

The constants produced by curves.arc_frame are binary rationals. This contract
compares their mathematical harmonic functions, not floating libm evaluations.
A bounded affine proposal does not prove exact parameter or material retention.
"""
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from fractions import Fraction as F
from weakref import WeakKeyDictionary, ref

from .curves import arc_frame
from .cylinder_charts import _Proof, _Refusal
from .definition_binding import definition_checksum
from .edge_subcurve_preimages import query_prepared_edge_subcurve_preimages, _unpack
from .errors import GeometryError


class NativeArcParameterMapError(GeometryError):
    """Invalid binding or aggregate proof exhaustion; no partial receipt."""


@dataclass(frozen=True, slots=True)
class NativeArcParameterMapPolicy:
    """Aggregate arithmetic budget, independent of model or occurrence counts."""
    max_interval_operations: int = 200000

    def __post_init__(self):
        if type(self.max_interval_operations) is not int or self.max_interval_operations <= 0:
            raise NativeArcParameterMapError('invalid aggregate work budget')


@dataclass(frozen=True, slots=True)
class NativeArcFrame:
    center: tuple
    cosine: tuple
    sine: tuple
    sweep: F


@dataclass(frozen=True, slots=True)
class NativeArcParameterMap:
    edge_id: int
    ancestor_edge_id: int
    ancestor_model_id: object
    ancestor_revision: int
    ancestor_source_checksum: str
    interval: tuple
    source_frame: NativeArcFrame
    current_frame: NativeArcFrame
    classification: str
    residual_bound: F
    tolerance: F | None
    geometric_classification: str
    reason: str | None


@dataclass(frozen=True, slots=True, weakref_slot=True)
class PreparedNativeArcParameterMaps:
    model_id: object
    revision: int
    source_checksum: str
    ancestry_digest: str
    policy: NativeArcParameterMapPolicy
    requested_edge_ids: tuple | None
    selected_edge_ids: tuple
    records: tuple
    alias_records: tuple
    unavailable_edge_ids: tuple
    work_counts: tuple


_issued = WeakKeyDictionary()


def _digest(value):
    def canonical(item):
        if isinstance(item, F):
            return ('rational', item.numerator, item.denominator)
        if is_dataclass(item):
            return (type(item).__name__, tuple((f.name, canonical(getattr(item, f.name)))
                for f in fields(item)))
        if isinstance(item, tuple):
            return tuple(map(canonical, item))
        return item
    return definition_checksum(canonical(value))


def _frame(definition):
    points = tuple(tuple(float(_unpack(v)) for v in p) for p in definition.positions)
    frame = arc_frame(*points)
    radius = F(float(frame.radius))
    return NativeArcFrame(tuple(F(float(x)) for x in frame.center),
        tuple(radius * F(float(x)) for x in frame.e1),
        tuple(radius * F(float(x)) for x in frame.e2), F(float(frame.sweep)))


def _bound(proof, source, current, interval):
    """Conservative harmonic Lipschitz bound, valid for every s in [0,1]."""
    a, b = interval
    phase, beta = source.sweep*a, source.sweep*(b-a)
    # Arc reversal can flip the frame normal/e2 while preserving sweep sign.
    # (V,w) and (-V,-w) are identically the same harmonic function. Select the
    # representation closer to beta; the authenticated interval stays untouched.
    sweep, current_sine = current.sweep, current.sine
    if abs(-sweep-beta) < abs(sweep-beta):
        sweep, current_sine = -sweep, tuple(-x for x in current_sine)
    # Structural identity avoids finite-grid enclosure noise; no epsilon exact.
    if (phase == 0 and beta == sweep and source.center == current.center
            and source.cosine == current.cosine and source.sine == current_sine):
        return F(0)
    if phase == 0:
        sine, cosine = proof.i(0), proof.i(1)
    else:
        sine, cosine = proof.sincos(phase)
    frequency = min(F(2), abs(sweep-beta))
    bounds = []
    sup = lambda x: max(abs(x.lo), abs(x.hi))
    for c, d, u, v, uc, vc in zip(source.center, current.center,
            source.cosine, source.sine, current.cosine, current_sine):
        rotated_u = proof.add(proof.mul(u, cosine), proof.mul(v, sine))
        rotated_v = proof.add(proof.mul(-u, sine), proof.mul(v, cosine))
        bound = abs(d-c) + sup(proof.sub(uc, rotated_u)) + sup(proof.sub(vc, rotated_v))
        bound += (sup(rotated_u)+sup(rotated_v))*frequency
        bounds.append(bound)
    return proof.sqrt(proof.i(sum(x*x for x in bounds))).hi


def _entry(proof, row, source, current):
    proof.charge()
    interval = tuple(map(_unpack, row.interval))
    bound = _bound(proof, source, current, interval)
    tolerance = None if row.tolerance is None else _unpack(row.tolerance)
    if row.classification == 'refused':
        classification, reason = 'refused', 'geometric_ancestry_refused'
    elif bound == 0:
        classification, reason = 'exact', None
    elif tolerance is not None and bound <= tolerance:
        classification, reason = 'bounded', None
    else:
        classification, reason = 'refused', 'native_residual_exceeds_recorded_tolerance'
    return NativeArcParameterMap(row.edge_id, row.ancestor.definition.edge_id,
        row.ancestor.model_id, row.ancestor.revision, row.ancestor.source_checksum,
        interval, source, current, classification, bound, tolerance,
        row.classification, reason)


def query_prepared_native_arc_parameter_maps(model, *, edge_ids=None,
        expected_revision=None, policy=None, cancellation_check=None):
    """Prove actual prepared Arc ancestry in one aggregate, atomic batch.

    Returns exact captured-function identities, bounded affine proposals, and
    explicit refusals. It confers no material/reference/mesh admission.
    """
    if edge_ids is not None and type(edge_ids) not in (tuple, list):
        raise NativeArcParameterMapError('edge request must be a finite list or tuple')
    requested = None if edge_ids is None else tuple(edge_ids)
    if requested is not None and (any(type(x) is not int or x <= 0 for x in requested) or
            len(set(requested)) != len(requested)):
        raise NativeArcParameterMapError('invalid edge request')
    if expected_revision is not None and (type(expected_revision) is not int or expected_revision < 0):
        raise NativeArcParameterMapError('invalid expected revision')
    if policy is not None and type(policy) is not NativeArcParameterMapPolicy:
        raise NativeArcParameterMapError('invalid proof policy')
    policy = NativeArcParameterMapPolicy() if policy is None else NativeArcParameterMapPolicy(**{
        f.name: getattr(policy, f.name) for f in fields(policy)})
    # Complete receipt and both frames are detached before any caller callback.
    original = query_prepared_edge_subcurve_preimages(model,
        edge_ids=requested, expected_revision=expected_revision)
    digest = definition_checksum(original)
    snapshot = deepcopy(original)
    rows = snapshot.arc_records + snapshot.arc_alias_records
    if requested is not None:
        if any(type(x) is not int for x in requested):
            raise NativeArcParameterMapError('invalid edge request')
        available = {r.edge_id for r in rows}
        if set(requested)-available:
            raise NativeArcParameterMapError('requested native Arc ancestry unavailable')
        rows = tuple(row for row in rows if row.edge_id in requested)
    proof = _Proof(policy, None)
    frames = {}
    try:
        for row in rows:
            proof.charge()
            for definition in (row.ancestor.definition, row.current_definition):
                if definition not in frames:
                    proof.charge()
                    frames[definition] = _frame(definition)
    except _Refusal as error:
        raise NativeArcParameterMapError(str(error)) from error
    except (GeometryError, ValueError, TypeError, OverflowError) as error:
        raise NativeArcParameterMapError('native Arc frame unavailable') from error
    def check(phase):
        if cancellation_check is not None and cancellation_check(phase):
            raise NativeArcParameterMapError('native Arc parameter maps cancelled')
    proof.callback = check
    try:
        proof.cancel('native Arc parameter maps')
        def entries(items):
            return tuple(_entry(proof, row, frames[row.ancestor.definition],
                frames[row.current_definition]) for row in items
                if requested is None or row.edge_id in requested)
        records, aliases = entries(snapshot.arc_records), entries(snapshot.arc_alias_records)
        proof.cancel('native Arc parameter maps final check')
    except _Refusal as error:
        if error is proof.callback_error:
            raise
        raise NativeArcParameterMapError(str(error)) from error
    # No callback after this final validation, and entry-time digests are pinned.
    current = query_prepared_edge_subcurve_preimages(model)
    if current is not original or definition_checksum(current) != digest:
        raise NativeArcParameterMapError('ancestry changed during native proof')
    result = PreparedNativeArcParameterMaps(snapshot.model_id, snapshot.revision,
        snapshot.source_checksum, digest, policy, requested,
        tuple(sorted({r.edge_id for r in records+aliases})), records, aliases,
        snapshot.unavailable_edge_ids, tuple(sorted(proof.counts.items())))
    issued = _issued.setdefault(model, {})
    key = id(result)
    issued[key] = (ref(result, lambda _: issued.pop(key, None)), _digest(result))
    return result


def _issued_binding_digest(model, binding):
    """Owner binding guard: pinned issuance digest of an owner-issued receipt.

    Raises for receipts not issued to this owner (forged or copied) and for
    receipts whose content changed after issuance.  Owner-internal only.
    """
    issued = _issued.get(model, {}).get(id(binding))
    if issued is None or issued[0]() is not binding:
        raise NativeArcParameterMapError('native Arc receipt was not issued to this owner')
    pinned = issued[1]
    if _digest(binding) != pinned:
        raise NativeArcParameterMapError('native Arc receipt changed')
    return pinned


def validate_prepared_native_arc_parameter_maps_binding(model, binding, *,
        cancellation_check=None):
    """Validate a producer-issued receipt without re-proving or accepting forgery."""
    issued = _issued.get(model, {}).get(id(binding))
    if issued is None or issued[0]() is not binding:
        raise NativeArcParameterMapError('native Arc receipt was not issued to this owner')
    pinned = issued[1]
    if _digest(binding) != pinned:
        raise NativeArcParameterMapError('native Arc receipt changed')
    snapshot = deepcopy(binding)
    original = query_prepared_edge_subcurve_preimages(model)
    ancestry = definition_checksum(original)
    if (snapshot.model_id != model.model_id or snapshot.revision != model.revision
            or snapshot.source_checksum != original.source_checksum
            or snapshot.ancestry_digest != ancestry):
        raise NativeArcParameterMapError('native Arc receipt is stale')
    if cancellation_check is not None:
        if cancellation_check('native Arc parameter maps binding'):
            raise NativeArcParameterMapError('native Arc parameter maps cancelled')
    current = query_prepared_edge_subcurve_preimages(model)
    if (current is not original or definition_checksum(current) != ancestry
            or _digest(binding) != pinned):
        raise NativeArcParameterMapError('native Arc binding changed during validation')
