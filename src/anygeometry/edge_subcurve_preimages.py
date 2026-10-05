"""Transient, owner-bound polynomial split ancestry and rounding enclosures.

Receipts establish ancestry and a whole-interval approximation bound only.
They confer no material containment, analytic substitution or split permission.
Only the detached preparation wrapper records splits; ordinary edits do not.
"""
from dataclasses import dataclass, replace
from fractions import Fraction
from numbers import Integral
from uuid import UUID
import math
import weakref

from .arrangement_geometry import LinePath, BezierPath, freeze_edge
from .definition_binding import definition_checksum
from .errors import GeometryError
from .serialization import to_dict, _checksum, _serialized_model_state


class _SubcurveEnclosureUnavailable(GeometryError):
    """Optional ancestry enclosure exceeds the unchanged owner tolerance."""


def _pack(value):
    value = Fraction(value)
    return value.numerator, value.denominator


def _unpack(value):
    return Fraction(*value)


@dataclass(frozen=True, slots=True)
class PolynomialEdgeDefinition:
    edge_id: int
    start: int
    end: int
    controls: tuple
    checksum: str


@dataclass(frozen=True, slots=True)
class PolynomialEdgeAncestor:
    model_id: UUID
    revision: int
    source_checksum: str
    definition: PolynomialEdgeDefinition


@dataclass(frozen=True, slots=True)
class EdgeSubcurvePreimage:
    edge_id: int
    ancestor: PolynomialEdgeAncestor
    interval: tuple
    current_definition: PolynomialEdgeDefinition
    error_controls: tuple
    coordinate_bounds: tuple
    squared_distance_bound: tuple
    tolerance: tuple | None


@dataclass(frozen=True, slots=True)
class PreparedEdgeSubcurvePreimages:
    model_id: UUID
    revision: int
    source_checksum: str
    records: tuple[EdgeSubcurvePreimage, ...]
    unavailable_edge_ids: tuple[int, ...]
    coverage: tuple[int, ...]


@dataclass
class _Draft:
    owner: object
    model_id: UUID
    source_revision: int
    source_checksum: str
    records: dict
    check: object
    enclosure_failure: object = None


@dataclass(frozen=True)
class _Prepared:
    owner: object
    binding: PreparedEdgeSubcurvePreimages
    checksum: str


def _committed(model):
    if model._transaction_journal is not None or model._notifying_hooks:
        raise GeometryError('edge subcurve provenance requires a committed model')


def _check(callback):
    if callback is not None and callback('edge subcurve provenance'):
        raise GeometryError('edge subcurve provenance cancelled')


def _identifier(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise GeometryError('edge subcurve provenance needs a positive edge ID')
    return int(value)


def _edge_subcurve_definition(model, edge):
    """Freeze the actual edge and every endpoint/control dependency."""
    edge = _identifier(edge)
    if edge not in model.edges:
        raise GeometryError('edge subcurve provenance edge is not active')
    path = freeze_edge(model, edge)
    if type(path) not in (LinePath, BezierPath):
        return None
    entity = model.edges[edge]
    points = (path.start, path.end) if type(path) is LinePath else path.controls
    controls = tuple(tuple(_pack(float(x)) for x in point) for point in points)
    ids = (entity.start, *getattr(entity.curve, 'control_vertices', ()), entity.end)
    checksum = definition_checksum((entity, tuple(model.vertices[i] for i in ids)))
    return PolynomialEdgeDefinition(edge, entity.start, entity.end, controls, checksum)


def _controls(definition):
    return tuple(tuple(_unpack(x) for x in point) for point in definition.controls)


def _restrict_controls(controls, lower, upper):
    """Exact Bernstein restriction, including reverse/zero-length intervals."""
    n = len(controls)-1
    # Blossom: each restricted control uses n-k copies of lower, k of upper.
    result = []
    for k in range(n+1):
        rows = list(controls)
        for t in (lower,)*(n-k)+(upper,)*k:
            rows = [tuple((1-t)*x+t*y for x, y in zip(a, b))
                    for a, b in zip(rows, rows[1:])]
        result.append(rows[0])
    return tuple(result)


def _elevate(controls, degree):
    rows = controls
    while len(rows)-1 < degree:
        n = len(rows)
        rows = (rows[0], *(tuple(Fraction(i, n)*a+(1-Fraction(i, n))*b for a, b in zip(rows[i-1], rows[i]))
                           for i in range(1, n)), rows[-1])
    return rows


def _seal(ancestor, interval, current, tolerance):
    expected = _restrict_controls(_controls(ancestor.definition), *map(_unpack, interval))
    actual = _controls(current)
    degree = max(len(expected), len(actual))-1
    differences = tuple(tuple(a-b for a, b in zip(first, second))
                        for first, second in zip(_elevate(actual, degree), _elevate(expected, degree)))
    coordinate = tuple(max(abs(row[i]) for row in differences) for i in range(3))
    squared = max(sum(x*x for x in row) for row in differences)
    if tolerance is None:
        if squared:
            raise GeometryError('edge subcurve provenance changed without a recorded split')
    elif squared > _unpack(tolerance)**2:
        raise _SubcurveEnclosureUnavailable('edge subcurve rounding enclosure exceeds existing tolerance')
    return EdgeSubcurvePreimage(current.edge_id, ancestor, interval, current,
        tuple(tuple(_pack(x) for x in row) for row in differences),
        tuple(_pack(x) for x in coordinate), _pack(squared), tolerance)


def _current(model):
    _committed(model)
    receipt = getattr(model, '_edge_subcurve_preimages_receipt', None)
    if receipt is None:
        raise GeometryError('edge subcurve provenance is unavailable')
    try:
        binding, checksum, owner = receipt
    except (TypeError, ValueError) as exc:
        raise GeometryError('edge subcurve provenance receipt is invalid') from exc
    if not isinstance(binding, PreparedEdgeSubcurvePreimages) or owner() is not model:
        raise GeometryError('edge subcurve provenance belongs to another owner')
    if binding.model_id != model.model_id:
        raise GeometryError('edge subcurve provenance belongs to another model')
    if binding.revision != model.revision:
        raise GeometryError('edge subcurve provenance is stale')
    if definition_checksum(binding) != checksum:
        raise GeometryError('edge subcurve provenance definition binding changed')
    # Preparation already qualified the immutable receipt. Re-fingerprint all
    # persisted inputs to detect even direct same-revision edits, without
    # repeating whole-model topology qualification for every station query.
    if _serialized_model_state(model)['checksum']['value'] != binding.source_checksum:
        raise GeometryError('edge subcurve provenance source binding changed')
    if binding.coverage != tuple(sorted(model.edges)):
        raise GeometryError('edge subcurve provenance coverage changed')
    return binding


def query_prepared_edge_subcurve_preimages(model, *, edge_ids=None, expected_revision=None,
                                           cancellation_check=None):
    """Return current split ancestry; explicitly requested unavailable edges refuse.

    Default results include all qualified records and unavailable active IDs.
    Ordinary loads/clones do not carry this local proof. A later authorized
    preparation may establish NEW roots from an unfragmented loaded document.
    """
    _committed(model)
    _check(cancellation_check)
    if expected_revision is not None and expected_revision != model.revision:
        raise GeometryError('edge subcurve provenance query revision is stale')
    from .batch_intersections import has_current_intersection_preparation
    if not has_current_intersection_preparation(model):
        raise GeometryError('edge subcurve provenance requires a complete current preparation')
    binding = _current(model)
    if edge_ids is not None:
        requested = tuple(_identifier(value) for value in edge_ids)
        if any(value not in model.edges for value in requested):
            raise GeometryError('edge subcurve provenance requested edge is not active')
        if set(requested) & set(binding.unavailable_edge_ids):
            raise GeometryError('edge subcurve provenance requested edge is unavailable')
    _check(cancellation_check)
    if _current(model) != binding or not has_current_intersection_preparation(model):
        raise GeometryError('edge subcurve provenance changed during query')
    return binding


def validate_prepared_edge_subcurve_preimages_binding(model, binding, *, cancellation_check=None):
    if not isinstance(binding, PreparedEdgeSubcurvePreimages):
        raise GeometryError('edge subcurve provenance needs a PreparedEdgeSubcurvePreimages binding')
    if binding.model_id != model.model_id:
        raise GeometryError('edge subcurve provenance belongs to another model')
    if binding.revision != model.revision:
        raise GeometryError('edge subcurve provenance is stale')
    current = query_prepared_edge_subcurve_preimages(model, cancellation_check=cancellation_check)
    if definition_checksum(binding) != definition_checksum(current):
        raise GeometryError('edge subcurve provenance definition binding changed')


def _capture_edge_subcurve_preimages(model, *, allow_seed=False, cancellation_check=None):
    _committed(model)
    revision = model.revision
    checksum = to_dict(model)['checksum']['value']
    _check(cancellation_check)
    if hasattr(model, '_edge_subcurve_preimages_receipt'):
        try:
            binding = query_prepared_edge_subcurve_preimages(model)
        except GeometryError:
            return None
        records = {record.edge_id: record for record in binding.records}
    else:
        if not allow_seed or any(old.kind == 'edge' for old in model.replacement_history()):
            return None
        records = {}
        for edge in sorted(model.edges):
            _check(cancellation_check)
            definition = _edge_subcurve_definition(model, edge)
            if definition is not None:
                ancestor = PolynomialEdgeAncestor(model.model_id, revision, checksum, definition)
                records[edge] = _seal(ancestor, ((0, 1), (1, 1)), definition, None)
    _check(cancellation_check)
    if model.revision != revision or to_dict(model)['checksum']['value'] != checksum:
        raise GeometryError('edge subcurve provenance source changed during capture')
    return _Draft(weakref.ref(model), model.model_id, revision, checksum, records, cancellation_check)


def _record_edge_subcurve_split(draft, edge, parameter, children, tolerance, *, model, parent_definition):
    """Record one successful actual local split, before any later modification."""
    if draft is None:
        return
    draft.enclosure_failure = None
    _check(draft.check)
    edge = _identifier(edge)
    if model.model_id != draft.model_id:
        raise GeometryError('edge subcurve split belongs to another model')
    record = draft.records.get(edge)
    if record is None:
        return  # Unknown ancestry cannot become a new root through splitting.
    if parent_definition != record.current_definition:
        raise GeometryError('edge subcurve parent definition changed before split')
    ids = tuple(_identifier(value) for value in children)
    if len(ids) != 2 or ids[0] == ids[1] or any(i in draft.records for i in ids) or edge in model.edges:
        raise GeometryError('edge subcurve split children are inconsistent')
    try:
        parameter, tolerance = float(parameter), float(tolerance)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GeometryError('edge subcurve split parameter/tolerance is invalid') from exc
    if not math.isfinite(parameter) or not 0 < parameter < 1 or not math.isfinite(tolerance) or tolerance <= 0:
        raise GeometryError('edge subcurve split parameter/tolerance is invalid')
    definitions = tuple(_edge_subcurve_definition(model, i) for i in ids)
    first, second = definitions
    if (first is None or second is None or first.start != parent_definition.start
            or second.end != parent_definition.end or first.end != second.start):
        raise GeometryError('edge subcurve split orientation/incidence changed')
    a, b = map(_unpack, record.interval)
    middle = a+(b-a)*Fraction(parameter)
    bound = Fraction(tolerance)
    if record.tolerance is not None:
        bound = min(bound, _unpack(record.tolerance))
    try:
        entries = tuple(_seal(record.ancestor, tuple(map(_pack, interval)), definition, _pack(bound))
                        for interval, definition in zip(((a, middle), (middle, b)), definitions))
    except _SubcurveEnclosureUnavailable as error:
        draft.enclosure_failure = error
        raise
    _check(draft.check)
    if any(_edge_subcurve_definition(model, entry.edge_id) != entry.current_definition for entry in entries):
        raise GeometryError('edge subcurve child changed while recording split')
    del draft.records[edge]
    draft.records.update((entry.edge_id, entry) for entry in entries)


def _finalize_edge_subcurve_preimages(model, draft, *, cancellation_check=None):
    if draft is None:
        return None
    _committed(model)
    if model.model_id != draft.model_id:
        raise GeometryError('edge subcurve candidate belongs to another model')
    check = cancellation_check if cancellation_check is not None else draft.check
    _check(check)
    revision = model.revision
    checksum = to_dict(model)['checksum']['value']
    records = []
    for edge, record in sorted(draft.records.items()):
        _check(check)
        if edge not in model.edges:
            continue
        if _edge_subcurve_definition(model, edge) != record.current_definition:
            raise GeometryError('edge subcurve tracked child changed before finalize')
        records.append(record)
    coverage = tuple(sorted(model.edges))
    qualified = {record.edge_id for record in records}
    unavailable = tuple(edge for edge in coverage if edge not in qualified)
    binding = PreparedEdgeSubcurvePreimages(model.model_id, revision, checksum, tuple(records), unavailable, coverage)
    _check(check)
    if model.revision != revision or to_dict(model)['checksum']['value'] != checksum:
        raise GeometryError('edge subcurve candidate changed during finalize')
    owner = draft.owner()
    if owner is None or owner.revision != draft.source_revision or to_dict(owner)['checksum']['value'] != draft.source_checksum:
        raise GeometryError('edge subcurve source changed during detached preparation')
    return _Prepared(draft.owner, binding, definition_checksum(binding))


def _retained_incidence(record, prior, current):
    """Retain an incidence rebinding only through an unchanged-tolerance seal.

    Exact rational controls re-seal to the identical certificate. A recorded
    degree-1 polynomial restriction may otherwise survive only the owner's canonicalizing
    vertex merge, and only while the exact whole-interval residual stays within
    the SAME recorded tolerance. Deleted entries, no-tolerance records without
    exact controls, nonlinear geometry changes and out-of-tolerance
    residuals all become unavailable; nothing here creates ancestry.
    """
    if current is None:
        return None
    if current.controls != prior.controls and (record.tolerance is None
                                               or len(prior.controls) != 2
                                               or len(current.controls) != 2):
        return None
    try:
        return _seal(record.ancestor, record.interval, current, record.tolerance)
    except _SubcurveEnclosureUnavailable:
        return None  # Out-of-tolerance incidence remains unavailable.


def _rebind_edge_subcurve_incidence(draft, prior_definitions, *, model):
    """Reseal only an explicit canonicalization's supplied incidence closure.

    The caller snapshots exactly the edges using the replaced vertex immediately
    before the owned merge. Retention happens only through _seal with the
    existing ancestor, interval and recorded tolerance, proving the exact
    whole-interval residual stays within that unchanged tolerance. This never
    creates ancestry, intervals or tolerances.
    """
    if draft is None:
        return
    _check(draft.check)
    if model.model_id != draft.model_id:
        raise GeometryError('edge subcurve incidence belongs to another model')
    updates = {}
    definitions = {}
    for value, prior in prior_definitions.items():
        edge = _identifier(value)
        record = draft.records.get(edge)
        if record is None:
            continue
        if prior != record.current_definition:
            raise GeometryError('edge subcurve prior definition changed before incidence rebind')
        current = _edge_subcurve_definition(model, edge) if edge in model.edges else None
        definitions[edge] = current
        updates[edge] = _retained_incidence(record, prior, current)
    _check(draft.check)
    for edge, expected in definitions.items():
        current = _edge_subcurve_definition(model, edge) if edge in model.edges else None
        if current != expected:
            raise GeometryError('edge subcurve incidence changed during rebind')
    for edge, record in updates.items():
        if record is None:
            draft.records.pop(edge, None)
        else:
            draft.records[edge] = record


def _publish_edge_subcurve_preimages(model, prepared, checksum):
    """Seal only a finalized detached result after the wrapper's atomic commit."""
    if prepared is None:
        return
    _committed(model)
    if (not isinstance(prepared, _Prepared) or prepared.owner() is not model
            or definition_checksum(prepared.binding) != prepared.checksum
            or prepared.binding.model_id != model.model_id):
        raise GeometryError('edge subcurve publication does not match prepared candidate')
    # An idempotent detached transaction may increment only its revision;
    # restore_topology correctly leaves the committed revision unchanged.
    # Compare every serialized field, allowing only that publication revision
    # difference. This cannot rebind changed geometry, ownership or lineage.
    document = to_dict(model)
    if document['checksum']['value'] != checksum:
        raise GeometryError('edge subcurve publication source changed')
    document['revision'] = prepared.binding.revision
    if _checksum(document)['value'] != prepared.binding.source_checksum:
        raise GeometryError('edge subcurve publication does not match prepared candidate')
    binding = replace(prepared.binding, revision=model.revision, source_checksum=checksum)
    model._edge_subcurve_preimages_receipt = (binding, definition_checksum(binding), weakref.ref(model))


def _copy_current_edge_subcurve_preimages(source, target):
    try:
        binding = query_prepared_edge_subcurve_preimages(source)
    except GeometryError:
        return
    _committed(target)
    if (target.model_id != binding.model_id or target.revision != binding.revision
            or to_dict(target)['checksum']['value'] != binding.source_checksum):
        raise GeometryError('edge subcurve provenance changed during qualified cloning')
    target._edge_subcurve_preimages_receipt = (binding, definition_checksum(binding), weakref.ref(target))
