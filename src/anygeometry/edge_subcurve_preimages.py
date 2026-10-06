"""Transient, owner-bound split ancestry and rounding enclosures.

Receipts establish ancestry and a whole-interval approximation bound only.
They confer no material containment, analytic substitution or split permission.
Only the detached preparation wrapper records splits; ordinary edits do not.

Arc ancestry is a bounded geometric-subarc-image contract, never a
source-parameter restriction.  The numeric local split parameter and the
composed numeric interval are authenticated provenance only; every arc
record carries ``parameter_mapping_qualified = False`` and no record, flag
or doc claims the child equals the ancestor restricted to that interval.
``exact`` certifies only that the child's three actual defining points lie
exactly on the ancestor's exact circumcircle (a common plane/circle, NOT
subarc containment or tiling); ``enclosed`` certifies nonzero bounds within the
recorded tolerance, with the ancestor-plane normal distance included; any
stronger drift is refused with the computed bounds retained as evidence.
"""
from dataclasses import dataclass, replace
from fractions import Fraction
from numbers import Integral
from uuid import UUID
import math
import weakref

from .arrangement_geometry import LinePath, BezierPath, freeze_edge
from .curves import Arc
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
    # Additive producer-authenticated occurrences of unified shared boundaries.
    # Each alias re-seals one participating original boundary's ancestry onto
    # the canonical current edge; records above stay single-ancestor.
    alias_records: tuple[EdgeSubcurvePreimage, ...] = ()
    # Arc split ancestry: bounded geometric-subarc-image evidence per arc
    # child.  Numeric intervals are provenance only; no arc record claims a
    # source-parameter restriction (parameter_mapping_qualified is False).
    arc_records: tuple = ()
    arc_alias_records: tuple = ()


@dataclass(frozen=True, slots=True)
class ArcEdgeDefinition:
    """Actual three-point arc definition; positions are packed rationals."""
    edge_id: int
    start: int
    via: int
    end: int
    positions: tuple
    checksum: str


@dataclass(frozen=True, slots=True)
class ArcEdgeAncestor:
    model_id: UUID
    revision: int
    source_checksum: str
    definition: ArcEdgeDefinition


@dataclass(frozen=True, slots=True)
class ArcSubcurvePreimage:
    """Bounded geometric-subarc-image ancestry evidence for one arc child.

    ``local_split_parameter`` and ``interval`` are authenticated NUMERIC
    PROVENANCE only (the exact binary rational of the float actually used,
    composed through repeated splits); ``parameter_mapping_qualified`` is
    always False and no field claims the child equals the ancestor restricted
    to the interval.  ``classification`` is ``exact`` (the child's three
    actual points lie exactly on the ancestor's exact circumcircle), or
    ``enclosed`` (nonzero certified bounds within ``tolerance``), or
    ``refused`` with ``reason`` and the computed bounds retained as evidence.
    ``anchor_bounds`` holds certified exact-rational (lower, upper) distance
    intervals from each defining point to the ancestor's circumcircle with
    the plane distance included; ``whole_circle_bound`` is a certified upper
    bound over the child's whole circumcircle (None when even the coarse
    tilted bound was not computed).
    """
    edge_id: int
    ancestor: ArcEdgeAncestor
    local_split_parameter: tuple | None
    interval: tuple
    current_definition: ArcEdgeDefinition
    classification: str
    reason: str | None
    direction: str
    anchor_bounds: tuple
    whole_circle_bound: tuple | None
    tolerance: tuple | None
    parameter_mapping_qualified: bool


def _arc_positions(definition):
    return tuple(tuple(_unpack(x) for x in point) for point in definition.positions)


def _vsub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def _vdot(a, b):
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]


def _vcross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _sqrt_interval(value):
    """Genuine rational bounds (floor, ceiling) with floor**2 <= value <= ceiling**2."""
    value = Fraction(value)
    if value < 0:
        raise GeometryError('arc ancestry needs a non-negative bound')
    root = math.isqrt(value.numerator * value.denominator)
    lower = Fraction(root, value.denominator)
    if lower * lower == value:
        return lower, lower
    return lower, lower + Fraction(1, value.denominator)


def _arc_circle(positions):
    """Exact circumcircle of three rational points: (center, radius**2, normal)."""
    first, via, last = positions
    u = _vsub(via, first)
    v = _vsub(last, first)
    uu, vv, uv = _vdot(u, u), _vdot(v, v), _vdot(u, v)
    determinant = uu*vv-uv*uv
    if determinant == 0:
        return None
    alpha = vv*(uu-uv)/(2*determinant)
    beta = uu*(vv-uv)/(2*determinant)
    center = tuple(first[i]+alpha*u[i]+beta*v[i] for i in range(3))
    return center, _vdot(_vsub(center, first), _vsub(center, first)), _vcross(u, v)


def _point_circle_distance(center, radius_squared, normal, point):
    """Certified rational (lower, upper) interval of the exact distance from
    one point to the ancestor circumcircle, plane distance included:
    dist**2 = h**2 + (rho - R)**2 with h the plane-normal distance and rho
    the planar radius.  (rho - R)**2 is convex in rho, so bounds evaluate
    the genuine rho interval at its endpoints only; ceilings are never taken
    inside absolute differences."""
    offset = _vsub(point, center)
    height_dot = _vdot(offset, normal)
    height_squared = Fraction(height_dot*height_dot, _vdot(normal, normal))
    planar_squared = _vdot(offset, offset)-height_squared
    rho_lower, rho_upper = _sqrt_interval(planar_squared)
    radius_lower, radius_upper = _sqrt_interval(radius_squared)
    low = rho_lower-radius_upper
    high = rho_upper-radius_lower
    delta_lower = Fraction(0) if low <= 0 <= high else min(low*low, high*high)
    delta_upper = max(low*low, high*high)
    return (_sqrt_interval(height_squared+delta_lower)[0],
            _sqrt_interval(height_squared+delta_upper)[1])


def _coplanar_circle_bound(center, radius_squared, child_center, child_radius_squared):
    """Whole-circle distance bound for EXACTLY coplanar circles.

    Pair points having the same in-plane radial unit vector. Their distance
    is at most center distance + abs(radius difference), by the triangle
    inequality. Genuine intervals handle overlapping square-root intervals
    without assuming a positive lower bound for their absolute difference.
    """
    offset = _vsub(child_center, center)
    _, d_upper = _sqrt_interval(_vdot(offset, offset))
    r_lower, r_upper = _sqrt_interval(child_radius_squared)
    big_lower, big_upper = _sqrt_interval(radius_squared)
    return d_upper + max(abs(r_lower-big_upper), abs(r_upper-big_lower))


def _classify_arc_child(ancestor_definition, child_definition, tolerance):
    """(classification, reason, direction, anchor_bounds, whole_circle_bound).

    Never false exact: ``exact`` requires the child's three actual points to
    lie exactly on the ancestor's exact circumcircle.  ``enclosed`` requires
    every certified bound to stay within the recorded tolerance.  Anything
    stronger is refused with the computed bounds retained as evidence.
    ``tolerance`` is an exact rational or None (None cannot enclose)."""
    ancestor_circle = _arc_circle(_arc_positions(ancestor_definition))
    child_positions = _arc_positions(child_definition)
    child_circle = _arc_circle(child_positions)
    if ancestor_circle is None or child_circle is None:
        return 'refused', 'arc split child degenerate', 'uncertified', None, None
    first, via, last = child_positions
    if first == via or via == last or first == last:
        return 'refused', 'arc split child degenerate', 'uncertified', None, None
    center, radius_squared, normal = ancestor_circle
    alignment = _vdot(child_circle[2], normal)
    if alignment == 0:
        return 'refused', 'arc split orientation uncertified', 'uncertified', None, None
    direction = 'forward' if alignment > 0 else 'reversed'
    exact = True
    for point in child_positions:
        offset = _vsub(point, center)
        if _vdot(offset, normal) != 0 or _vdot(offset, offset) != radius_squared:
            exact = False
            break
    zero = ((Fraction(0), Fraction(0)),)*3
    if exact:
        return 'exact', None, direction, zero, Fraction(0)
    anchor_bounds = tuple(_point_circle_distance(center, radius_squared, normal, point)
                         for point in child_positions)
    if all(_vdot(_vsub(point, center), normal) == 0 for point in child_positions):
        whole = _coplanar_circle_bound(center, radius_squared,
                                       child_circle[0], child_circle[1])
    else:
        # Arbitrary tilt: the coarse child-centre-to-ancestor-circle distance
        # plus the child radius handles any tilt; when it exceeds the recorded
        # tolerance the record refuses instead of claiming a useful tolerance.
        whole = (_point_circle_distance(center, radius_squared, normal, child_circle[0])[1]
                 + _sqrt_interval(child_circle[1])[1])
    if tolerance is not None and all(upper <= tolerance for _, upper in anchor_bounds) \
            and whole <= tolerance:
        return 'enclosed', None, direction, anchor_bounds, whole
    return 'refused', 'arc split residual exceeds tolerance', direction, anchor_bounds, whole


def _arc_entry(ancestor, parameter, interval, definition, bound):
    """One arc child record; provenance only, never a parameter mapping claim."""
    classification, reason, direction, anchor_bounds, whole = _classify_arc_child(
        ancestor.definition, definition, bound)
    return ArcSubcurvePreimage(definition.edge_id, ancestor, _pack(parameter),
        tuple(map(_pack, interval)), definition, classification, reason, direction,
        tuple(tuple(map(_pack, pair)) for pair in anchor_bounds) if anchor_bounds else (),
        _pack(whole) if whole is not None else None, _pack(bound), False)


def _arc_edge_definition(model, edge):
    """Freeze the actual three-point arc definition and its vertex dependencies."""
    edge = _identifier(edge)
    if edge not in model.edges:
        raise GeometryError('edge subcurve provenance edge is not active')
    entity = model.edges[edge]
    if not isinstance(entity.curve, Arc):
        return None
    ids = (entity.start, entity.curve.via_vertex, entity.end)
    positions = tuple(tuple(_pack(float(x)) for x in model.vertices[i].position) for i in ids)
    checksum = definition_checksum((entity, tuple(model.vertices[i] for i in ids)))
    return ArcEdgeDefinition(edge, ids[0], ids[1], ids[2], positions, checksum)


def _edge_ancestry_definition(model, edge):
    """Combined polynomial-or-arc ancestry definition of one active edge."""
    definition = _edge_subcurve_definition(model, edge)
    if definition is None:
        definition = _arc_edge_definition(model, edge)
    return definition


@dataclass
class _Draft:
    owner: object
    model_id: UUID
    source_revision: int
    source_checksum: str
    records: dict
    check: object
    enclosure_failure: object = None
    aliases: dict = None
    arc_records: dict = None
    arc_aliases: dict = None


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


def _residual(ancestor, interval, current):
    """Exact whole-interval Bernstein residual of one oriented restriction."""
    expected = _restrict_controls(_controls(ancestor.definition), *map(_unpack, interval))
    actual = _controls(current)
    degree = max(len(expected), len(actual))-1
    differences = tuple(tuple(a-b for a, b in zip(first, second))
                        for first, second in zip(_elevate(actual, degree), _elevate(expected, degree)))
    coordinate = tuple(max(abs(row[i]) for row in differences) for i in range(3))
    squared = max(sum(x*x for x in row) for row in differences)
    return differences, coordinate, squared


def _seal(ancestor, interval, current, tolerance):
    differences, coordinate, squared = _residual(ancestor, interval, current)
    if tolerance is None:
        if squared:
            raise GeometryError('edge subcurve provenance changed without a recorded split')
    elif squared > _unpack(tolerance)**2:
        raise _SubcurveEnclosureUnavailable('edge subcurve rounding enclosure exceeds existing tolerance')
    return EdgeSubcurvePreimage(current.edge_id, ancestor, interval, current,
        tuple(tuple(_pack(x) for x in row) for row in differences),
        tuple(_pack(x) for x in coordinate), _pack(squared), tolerance)


def _reseal_occurrence(source, current):
    """Re-seal one recorded occurrence onto a unified edge's exact definition.

    Orientation is fixed by endpoint-ID topology alone, never by residual
    size: the occurrence's sealed current definition and the canonical edge's
    current definition must bind the same two distinct endpoint IDs. The same
    ordered pair preserves the source interval; the swapped pair reverses it;
    any mismatched or degenerate pair is unauthenticated. Only that one
    selected interval is certified, by the exact whole-interval Bernstein
    residual within the occurrence's SAME recorded tolerance. Nothing is
    inferred from proximity, geometry, samples or replacement metadata.
    """
    source_definition = source.current_definition
    if source_definition.start == source_definition.end or current.start == current.end:
        return None  # Degenerate endpoint identity cannot orient the interval.
    pair = (source_definition.start, source_definition.end)
    lower, upper = map(_unpack, source.interval)
    if pair == (current.start, current.end):
        first, second = lower, upper
    elif pair == (current.end, current.start):
        first, second = upper, lower
    else:
        return None  # Mismatched endpoint identity remains unauthenticated.
    differences, coordinate, squared = _residual(source.ancestor,
        (_pack(first), _pack(second)), current)
    if source.tolerance is None:
        if squared:
            return None
        tolerance = None
    elif squared > _unpack(source.tolerance)**2:
        return None  # Out-of-tolerance occurrence remains unauthenticated.
    else:
        tolerance = source.tolerance
    return EdgeSubcurvePreimage(current.edge_id, source.ancestor,
        (_pack(first), _pack(second)), current,
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
    aliases = {}
    arc_aliases = {}
    if hasattr(model, '_edge_subcurve_preimages_receipt'):
        try:
            binding = query_prepared_edge_subcurve_preimages(model)
        except GeometryError:
            return None
        records = {record.edge_id: record for record in binding.records}
        for row in binding.alias_records:
            aliases.setdefault(row.edge_id, []).append(row)
        arc_records = {record.edge_id: record for record in binding.arc_records}
        for row in binding.arc_alias_records:
            arc_aliases.setdefault(row.edge_id, []).append(row)
    else:
        from .preparation_epochs import _has_epoch_permit
        if not allow_seed or (not _has_epoch_permit(model) and any(old.kind == 'edge' for old in model.replacement_history())):
            return None
        records = {}
        arc_records = {}
        for edge in sorted(model.edges):
            _check(cancellation_check)
            definition = _edge_subcurve_definition(model, edge)
            if definition is not None:
                ancestor = PolynomialEdgeAncestor(model.model_id, revision, checksum, definition)
                records[edge] = _seal(ancestor, ((0, 1), (1, 1)), definition, None)
                continue
            arc_definition = _arc_edge_definition(model, edge)
            if arc_definition is None:
                continue
            classification, _, direction, _, _ = _classify_arc_child(
                arc_definition, arc_definition, None)
            if classification != 'exact':
                # A degenerate authored arc cannot seed a sound ancestry root.
                continue
            ancestor = ArcEdgeAncestor(model.model_id, revision, checksum, arc_definition)
            arc_records[edge] = ArcSubcurvePreimage(edge, ancestor, None,
                ((0, 1), (1, 1)), arc_definition, classification, None, direction,
                (((0, 1), (0, 1)),)*3, None, None, False)
    _check(cancellation_check)
    if model.revision != revision or to_dict(model)['checksum']['value'] != checksum:
        raise GeometryError('edge subcurve provenance source changed during capture')
    return _Draft(weakref.ref(model), model.model_id, revision, checksum, records,
        cancellation_check, aliases=aliases, arc_records=arc_records,
        arc_aliases=arc_aliases)


def _drop_edge_subcurve_records(draft, edge_ids):
    """Remove tracked records and captured occurrences of unqualified edges."""
    if draft is None:
        return
    for value in edge_ids:
        edge = _identifier(value)
        draft.records.pop(edge, None)
        if draft.aliases is not None:
            draft.aliases.pop(edge, None)
        if draft.arc_records is not None:
            draft.arc_records.pop(edge, None)
        if draft.arc_aliases is not None:
            draft.arc_aliases.pop(edge, None)


def _reseal_arc_occurrence(source, current):
    """Re-seal one recorded arc occurrence onto a unified edge's definition.

    Orientation is fixed by the ordered start/via/end vertex-ID triple alone,
    never by proximity or residual size: the same ordered triple preserves the
    provenance interval, the reversed triple (end, via, start) reverses it,
    and any other triple is unauthenticated.  The classification is re-derived
    from the canonical edge's actual positions within the occurrence's SAME
    recorded tolerance; a refusal is retained as evidence, never upgraded."""
    source_definition = source.current_definition
    triple = (source_definition.start, source_definition.via, source_definition.end)
    lower, upper = map(_unpack, source.interval)
    if triple == (current.start, current.via, current.end):
        first, second = lower, upper
    elif triple == (current.end, current.via, current.start):
        first, second = upper, lower
    else:
        return None  # Mismatched vertex identity remains unauthenticated.
    bound = source.tolerance
    classification, reason, direction, anchor_bounds, whole = _classify_arc_child(
        source.ancestor.definition, current, _unpack(bound) if bound is not None else None)
    return ArcSubcurvePreimage(current.edge_id, source.ancestor, source.local_split_parameter,
        (_pack(first), _pack(second)), current, classification, reason, direction,
        tuple(tuple(map(_pack, pair)) for pair in anchor_bounds) if anchor_bounds else (),
        _pack(whole) if whole is not None else None, bound, False)


def _record_edge_subcurve_unification(draft, canonical, duplicates, *, model):
    """Capture every authenticated occurrence of a producer-unified boundary.

    The producer calls this at the exact canonical reuse of one current edge
    for coincident duplicates. The canonical edge's actual current definition
    must still match its draft record; each duplicate's actual definition must
    still match the sealed source record it contributes. Each duplicate's
    sealed ancestry, and every occurrence already captured for the duplicate,
    re-seals onto the canonical edge's exact current definition as an oriented
    source interval within the SAME recorded tolerance. A failed seal leaves
    that occurrence unauthenticated; no ancestry is created here.
    """
    if draft is None:
        return
    if draft.aliases is None:
        draft.aliases = {}
    if draft.arc_aliases is None:
        draft.arc_aliases = {}
    _check(draft.check)
    if model.model_id != draft.model_id:
        raise GeometryError('edge subcurve unification belongs to another model')
    canonical = _identifier(canonical)
    current = _edge_ancestry_definition(model, canonical)
    record = draft.records.get(canonical)
    if record is not None:
        if current is None or current != record.current_definition:
            raise GeometryError('edge subcurve canonical definition changed before unification')
    arc_record = (draft.arc_records or {}).get(canonical)
    if arc_record is not None:
        if not isinstance(current, ArcEdgeDefinition) or current != arc_record.current_definition:
            raise GeometryError('edge subcurve canonical definition changed before unification')
    if current is None:
        return
    for value in duplicates:
        _check(draft.check)
        duplicate = _identifier(value)
        if duplicate == canonical:
            continue
        actual = _edge_ancestry_definition(model, duplicate)
        sources = []
        other = draft.records.get(duplicate)
        if other is not None:
            if actual is not None and actual == other.current_definition:
                sources.append(other)
        sources.extend(row for row in draft.aliases.get(duplicate, ())
                        if actual is not None and actual == row.current_definition)
        for source in sources:
            alias = _reseal_occurrence(source, current)
            if alias is None:
                continue
            rows = draft.aliases.setdefault(canonical, [])
            if not any(row.interval == alias.interval and row.ancestor == alias.ancestor
                       for row in rows):
                rows.append(alias)
        arc_other = (draft.arc_records or {}).get(duplicate)
        arc_sources = []
        if arc_other is not None:
            if isinstance(actual, ArcEdgeDefinition) and actual == arc_other.current_definition:
                arc_sources.append(arc_other)
        arc_sources.extend(row for row in (draft.arc_aliases or {}).get(duplicate, ())
                           if isinstance(actual, ArcEdgeDefinition)
                           and actual == row.current_definition)
        for source in arc_sources:
            alias = _reseal_arc_occurrence(source, current)
            if alias is None:
                continue
            rows = draft.arc_aliases.setdefault(canonical, [])
            if not any(row.interval == alias.interval and row.ancestor == alias.ancestor
                       for row in rows):
                rows.append(alias)


def _record_edge_subcurve_split(draft, edge, parameter, children, tolerance, *, model, parent_definition):
    """Record one successful actual local split, before any later modification."""
    if draft is None:
        return
    if draft.aliases is None:
        draft.aliases = {}
    if draft.arc_aliases is None:
        draft.arc_aliases = {}
    draft.enclosure_failure = None
    _check(draft.check)
    edge = _identifier(edge)
    if model.model_id != draft.model_id:
        raise GeometryError('edge subcurve split belongs to another model')
    arc_parent = isinstance(parent_definition, ArcEdgeDefinition)
    records = draft.arc_records if arc_parent else draft.records
    record = records.get(edge)
    aliases = tuple((draft.arc_aliases if arc_parent else draft.aliases).get(edge, ()))
    if record is None and not aliases:
        return  # Unknown ancestry cannot become a new root through splitting.
    if record is not None and parent_definition != record.current_definition:
        raise GeometryError('edge subcurve parent definition changed before split')
    if any(parent_definition != row.current_definition for row in aliases):
        raise GeometryError('edge subcurve parent definition changed before split')
    ids = tuple(_identifier(value) for value in children)
    if len(ids) != 2 or ids[0] == ids[1] or any(i in draft.records or i in draft.arc_records
                                                 for i in ids) or edge in model.edges:
        raise GeometryError('edge subcurve split children are inconsistent')
    try:
        parameter, tolerance = float(parameter), float(tolerance)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GeometryError('edge subcurve split parameter/tolerance is invalid') from exc
    if not math.isfinite(parameter) or not 0 < parameter < 1 or not math.isfinite(tolerance) or tolerance <= 0:
        raise GeometryError('edge subcurve split parameter/tolerance is invalid')
    definitions = tuple(_edge_ancestry_definition(model, i) for i in ids)
    first, second = definitions
    if (type(first) is not type(parent_definition) or type(second) is not type(parent_definition)
            or first.start != parent_definition.start
            or second.end != parent_definition.end or first.end != second.start):
        raise GeometryError('edge subcurve split orientation/incidence changed')
    if arc_parent:
        _record_arc_split(draft, edge, record, aliases, parameter, tolerance,
                          definitions, model=model)
        return
    entries = ()
    if record is not None:
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
    # Captured occurrences compose through the split exactly like the primary
    # ancestry: the same recorded station maps each oriented source interval.
    child_aliases = {ids[0]: [], ids[1]: []}
    for row in aliases:
        _check(draft.check)
        c, d = map(_unpack, row.interval)
        row_middle = c+(d-c)*Fraction(parameter)
        bound = Fraction(tolerance)
        if row.tolerance is not None:
            bound = min(bound, _unpack(row.tolerance))
        for interval, definition, child in (((c, row_middle), first, ids[0]),
                                            ((row_middle, d), second, ids[1])):
            try:
                child_aliases[child].append(_seal(row.ancestor,
                    tuple(map(_pack, interval)), definition, _pack(bound)))
            except _SubcurveEnclosureUnavailable:
                continue  # Out-of-tolerance occurrence becomes unavailable.
    _check(draft.check)
    if any(_edge_subcurve_definition(model, entry.edge_id) != entry.current_definition
           for entry in entries):
        raise GeometryError('edge subcurve child changed while recording split')
    if any(_edge_subcurve_definition(model, row.edge_id) != row.current_definition
           for rows in child_aliases.values() for row in rows):
        raise GeometryError('edge subcurve child changed while recording split')
    if record is not None:
        del draft.records[edge]
        draft.records.update((entry.edge_id, entry) for entry in entries)
    draft.aliases.pop(edge, None)
    for child, rows in child_aliases.items():
        if rows:
            draft.aliases.setdefault(child, []).extend(rows)


def _record_arc_split(draft, edge, record, aliases, parameter, tolerance, definitions, *, model):
    """Record one actual arc split as bounded geometric-image evidence.

    The composed numeric intervals are authenticated provenance only; the
    classification is re-derived per child against the ROOT ancestor's exact
    circumcircle and never claims a source-parameter restriction.  Arc split
    recording never raises enclosure failures: an honest refusal is recorded
    evidence, and geometry keeps its established acceptance.
    """
    station = Fraction(parameter)
    entries = ()
    if record is not None:
        a, b = map(_unpack, record.interval)
        middle = a+(b-a)*station
        bound = Fraction(tolerance)
        if record.tolerance is not None:
            bound = min(bound, _unpack(record.tolerance))
        entries = tuple(_arc_entry(record.ancestor, station, interval, definition, bound)
                        for interval, definition in zip(((a, middle), (middle, b)), definitions))
    child_aliases = {definitions[0].edge_id: [], definitions[1].edge_id: []}
    for row in aliases:
        _check(draft.check)
        c, d = map(_unpack, row.interval)
        row_middle = c+(d-c)*station
        bound = Fraction(tolerance)
        if row.tolerance is not None:
            bound = min(bound, _unpack(row.tolerance))
        for interval, definition in (((c, row_middle), definitions[0]),
                                     ((row_middle, d), definitions[1])):
            child_aliases[definition.edge_id].append(_arc_entry(row.ancestor,
                station, interval, definition, bound))
    _check(draft.check)
    if any(_arc_edge_definition(model, entry.edge_id) != entry.current_definition
           for entry in entries):
        raise GeometryError('edge subcurve child changed while recording split')
    if any(_arc_edge_definition(model, row.edge_id) != row.current_definition
           for rows in child_aliases.values() for row in rows):
        raise GeometryError('edge subcurve child changed while recording split')
    if record is not None:
        del draft.arc_records[edge]
        draft.arc_records.update((entry.edge_id, entry) for entry in entries)
    draft.arc_aliases.pop(edge, None)
    for child, rows in child_aliases.items():
        if rows:
            draft.arc_aliases.setdefault(child, []).extend(rows)


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
    aliases = []
    seen = set()
    for edge, rows in sorted((draft.aliases or {}).items()):
        _check(check)
        if edge not in model.edges:
            continue
        current = _edge_subcurve_definition(model, edge)
        for row in rows:
            if current is None or row.current_definition != current:
                raise GeometryError('edge subcurve tracked child changed before finalize')
            key = (row.edge_id, row.ancestor, row.interval)
            if key in seen:
                continue
            seen.add(key)
            aliases.append(row)
    aliases.sort(key=lambda row: (row.edge_id, row.ancestor.definition.edge_id, row.interval))
    arc_records = []
    for edge, record in sorted((draft.arc_records or {}).items()):
        _check(check)
        if edge not in model.edges:
            continue
        if _arc_edge_definition(model, edge) != record.current_definition:
            raise GeometryError('edge subcurve tracked child changed before finalize')
        arc_records.append(record)
    arc_aliases = []
    seen = set()
    for edge, rows in sorted((draft.arc_aliases or {}).items()):
        _check(check)
        if edge not in model.edges:
            continue
        current = _arc_edge_definition(model, edge)
        for row in rows:
            if current is None or row.current_definition != current:
                raise GeometryError('edge subcurve tracked child changed before finalize')
            key = (row.edge_id, row.ancestor, row.interval)
            if key in seen:
                continue
            seen.add(key)
            arc_aliases.append(row)
    arc_aliases.sort(key=lambda row: (row.edge_id, row.ancestor.definition.edge_id, row.interval))
    coverage = tuple(sorted(model.edges))
    qualified = {record.edge_id for record in records}
    qualified.update(record.edge_id for record in arc_records)
    unavailable = tuple(edge for edge in coverage if edge not in qualified)
    binding = PreparedEdgeSubcurvePreimages(model.model_id, revision, checksum, tuple(records),
        unavailable, coverage, tuple(aliases), tuple(arc_records), tuple(arc_aliases))
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


def _retained_arc_incidence(record, prior, current):
    """Retain an arc incidence rebinding only through an honest re-seal.

    The canonicalizing vertex merge relabels vertices and moves positions
    within the model tolerance; the classification is re-derived from the
    current actual positions against the unchanged ROOT ancestor within the
    SAME recorded tolerance.  Exact may honestly downgrade to enclosed; a
    stronger drift keeps its refusal as recorded evidence.  Deleted edges
    and non-arc replacements become unavailable; nothing here creates
    ancestry, intervals or tolerances.
    """
    if not isinstance(prior, ArcEdgeDefinition) or not isinstance(current, ArcEdgeDefinition):
        return None
    bound = record.tolerance
    classification, reason, direction, anchor_bounds, whole = _classify_arc_child(
        record.ancestor.definition, current, _unpack(bound) if bound is not None else None)
    return ArcSubcurvePreimage(current.edge_id, record.ancestor, record.local_split_parameter,
        record.interval, current, classification, reason, direction,
        tuple(tuple(map(_pack, pair)) for pair in anchor_bounds) if anchor_bounds else (),
        _pack(whole) if whole is not None else None, bound, False)


def _rebind_edge_subcurve_incidence(draft, prior_definitions, *, model):
    """Reseal only an explicit canonicalization's supplied incidence closure.

    The caller snapshots exactly the edges using the replaced vertex immediately
    before the owned merge. Retention happens only through _seal with the
    existing ancestor, interval and recorded tolerance, proving the exact
    whole-interval residual stays within that unchanged tolerance. This never
    creates ancestry, intervals or tolerances. Captured shared-boundary
    occurrences of the same edge re-seal under the identical rule.
    """
    if draft is None:
        return
    if draft.aliases is None:
        draft.aliases = {}
    if draft.arc_aliases is None:
        draft.arc_aliases = {}
    _check(draft.check)
    if model.model_id != draft.model_id:
        raise GeometryError('edge subcurve incidence belongs to another model')
    updates = {}
    alias_updates = {}
    arc_updates = {}
    arc_alias_updates = {}
    definitions = {}
    for value, prior in prior_definitions.items():
        edge = _identifier(value)
        arc_prior = isinstance(prior, ArcEdgeDefinition)
        records = draft.arc_records if arc_prior else draft.records
        record = records.get(edge)
        aliases = tuple((draft.arc_aliases if arc_prior else draft.aliases).get(edge, ()))
        if record is None and not aliases:
            continue
        if record is not None and prior != record.current_definition:
            raise GeometryError('edge subcurve prior definition changed before incidence rebind')
        if any(prior != row.current_definition for row in aliases):
            raise GeometryError('edge subcurve prior definition changed before incidence rebind')
        current = _edge_ancestry_definition(model, edge) if edge in model.edges else None
        definitions[edge] = current
        if arc_prior:
            if record is not None:
                arc_updates[edge] = _retained_arc_incidence(record, prior, current)
            arc_alias_updates[edge] = tuple(filter(None,
                (_retained_arc_incidence(row, prior, current) for row in aliases)))
            continue
        if record is not None:
            updates[edge] = _retained_incidence(record, prior, current)
        alias_updates[edge] = tuple(filter(None,
            (_retained_incidence(row, prior, current) for row in aliases)))
    _check(draft.check)
    for edge, expected in definitions.items():
        current = _edge_ancestry_definition(model, edge) if edge in model.edges else None
        if current != expected:
            raise GeometryError('edge subcurve incidence changed during rebind')
    for edge, record in updates.items():
        if record is None:
            draft.records.pop(edge, None)
        else:
            draft.records[edge] = record
    for edge, rows in alias_updates.items():
        if rows:
            draft.aliases[edge] = list(rows)
        else:
            draft.aliases.pop(edge, None)
    for edge, record in arc_updates.items():
        if record is None:
            draft.arc_records.pop(edge, None)
        else:
            draft.arc_records[edge] = record
    for edge, rows in arc_alias_updates.items():
        if rows:
            draft.arc_aliases[edge] = list(rows)
        else:
            draft.arc_aliases.pop(edge, None)


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
