"""Owner-bound curved source-cell authority, separate from literal facet tests.

Boundary receipts describe source intervals and approximation enclosures. They
alone grant no whole-cell material coverage authority.
"""
from dataclasses import dataclass, fields, is_dataclass
from fractions import Fraction as Q
from itertools import product
from weakref import WeakKeyDictionary, ref
from uuid import UUID
from enum import Enum
import math

import numpy as np

from .errors import GeometryError
from .identity import EntityHandle
from .nurbs import NurbsSurface, CurveOnSurface, _IDENTITY, _float_up
from .support_curves import PlaneNurbsCurve
from .nurbs_intersection_curve import NurbsSurfaceIntersectionCurve
from .surfaces import Plane
from .material_regions import validate_material_surface_regions_binding
from .definition_binding import definition_checksum
from .trimmed_charts import _check


_NATIVE_QUALIFIED = WeakKeyDictionary()      # model -> (complete persisted checksum, actual native snapshot)


def _native_cache_safe(model, native):
    """Whether every non-persisted dependency of topology validity is bound by the native snapshot.

    Unknown semantics are never cached: a nonempty feature history, any ``unsupported*`` snapshot row
    and any edge curve family outside the vertex-defined and runtime-validated families all fall back
    to the full public ``to_dict`` validation on every call.

    Cacheable edge families: ``Straight`` (no evaluated state), and the NURBS families
    ``NurbsCurve``/``CurveOnSurface``/``PlaneNurbsCurve`` whose evaluated kernels, point caches and
    section arcs are validated by ``validate_model_curve_runtimes`` (inside the native snapshot).
    ``Arc`` and legacy ``Spline`` edges depend on evaluated model caches (``_arc_cache``,
    ``_spline_cache``) that no runtime validator covers, so they, and every other family, use the full
    public validation on every call.
    """
    from .curves import Straight
    from .nurbs import NurbsCurve
    from .support_curves import PlaneNurbsCurve
    try:
        if len(model.features) != 0:
            return False
    except (TypeError, AttributeError):
        return False
    if any(str(row[1]).startswith('unsupported') for row in native):
        return False
    known = (Straight, NurbsCurve, CurveOnSurface, PlaneNurbsCurve)
    return all(type(edge.curve) in known for edge in model.edges.values())


def _checksum(model):
    """Complete current persisted-content fingerprint of ``model`` for query binding.

    The owner's read-query contract (``serialization._qualified_model_state``) recomputes the complete
    encoding and reverse-incidence binding on every call. Topology validity can also depend on native
    fields that are not persisted (e.g. a Cylinder's circumferential basis), so full public ``to_dict``
    validation runs before the first local qualification and whenever the complete persisted checksum or
    the ACTUAL stored native snapshot (which also validates the curve runtimes) differs from the last
    locally qualified pair. Unknown dependencies always use ``to_dict``.
    """
    from .serialization import _qualified_model_state, to_dict
    from .native_support_snapshots import capture_native_supports
    checksum = _qualified_model_state(model)['checksum']['value']
    native = capture_native_supports(model)
    if not _native_cache_safe(model, native):
        return to_dict(model)['checksum']['value']
    if _NATIVE_QUALIFIED.get(model) == (checksum, native):
        return checksum
    validated = to_dict(model)['checksum']['value']             # full public validation of every dependency
    if validated != checksum or capture_native_supports(model) != native:
        raise GeometryError('geometry changed during query qualification')
    _NATIVE_QUALIFIED[model] = (checksum, native)
    return checksum


@dataclass(frozen=True)
class BoundaryChordOccurrence:
    face: object
    forward: bool
    uv: tuple
    uv_enclosures: tuple


@dataclass(frozen=True)
class MaterialBoundaryChord:
    edge: object
    parameter_range: tuple
    station_keys: tuple
    xyz: tuple
    occurrences: tuple
    world_error: float
    chart_error: float


@dataclass(frozen=True, eq=False)
class PreparedMaterialBoundaryChords:
    model_id: object
    revision: int
    source_checksum: str
    intervals: tuple
    reference_uv_by_face: tuple
    chord_error: float
    local_field_fingerprint: tuple = ()
    interval_errors: tuple = ()
    local_interval_footprints: tuple = ()


@dataclass(frozen=True, eq=False)
class PreparedMaterialBoundarySampling:
    model_id: object
    revision: int
    source_checksum: str
    edge_parameters: tuple
    chords: PreparedMaterialBoundaryChords
    chord_error: float
    qualification: tuple
    work_counts: tuple
    error_measure: str = "geometric"
    proposal_fraction: float = 1.


@dataclass(frozen=True)
class CurvilinearMaterialCell:
    face: object
    node_ids: tuple
    uv: tuple
    xyz: tuple
    boundary_intervals: tuple


@dataclass(frozen=True, eq=False)
class PreparedMaterialCurvilinearCells:
    model_id: object
    revision: int
    source_checksum: str
    cells: tuple
    surface_error: float
    reference_nodes: tuple = ()
    reference_uv_by_cell: tuple = ()
    source_map_bounds: tuple = ()
    boundary_correspondence: tuple = ()
    local_field_fingerprint: tuple = ()
    surface_errors_by_cell: tuple = ()
    local_footprints_by_cell: tuple = ()


_LOCAL_CHORD_ALLOCATIONS = WeakKeyDictionary()
_LOCAL_CELL_ALLOCATIONS = WeakKeyDictionary()
_LOCAL_RESOLUTION_FAILURES = WeakKeyDictionary()
_CHORDS = WeakKeyDictionary()
_CELLS = WeakKeyDictionary()
_SAMPLINGS = WeakKeyDictionary()


def _literal(value):
    kind = type(value)
    if value is None or kind in (bool, int, float, str):
        return kind.__name__, value
    if kind is Q:
        return 'Fraction', value.numerator, value.denominator
    if kind is UUID:
        return 'UUID', value.hex
    if isinstance(value, Enum):
        return type(value).__qualname__, _literal(value.value)
    if kind is tuple:
        # Exact scalar items are encoded inline; every other item takes the
        # unchanged recursion (identical output and refusals).
        return tuple([(item_kind.__name__, v) if ((item_kind := type(v)) is bool
                      or item_kind is int or item_kind is float or item_kind is str or v is None)
                      else _literal(v) for v in value])
    if kind in (EntityHandle, BoundaryChordOccurrence, MaterialBoundaryChord,
                PreparedMaterialBoundaryChords, PreparedMaterialBoundarySampling,
                CurvilinearMaterialCell, PreparedMaterialCurvilinearCells):
        return kind.__name__, tuple((f.name, _literal(getattr(value, f.name))) for f in fields(value))
    raise GeometryError('curvilinear receipt contains a nonliteral or unsupported field')


def _detached(value):
    """Owner copies of already validated literal fields, without copy hooks."""
    kind = type(value)
    if kind is tuple:
        return tuple(_detached(v) for v in value)
    if kind is Q:
        return Q(value.numerator, value.denominator)
    if kind is UUID:
        return UUID(int=value.int)
    if kind in (EntityHandle, CurvilinearMaterialCell):
        return kind(**{f.name: _detached(getattr(value, f.name)) for f in fields(value)})
    if value is None or kind in (bool, int, float, str):
        return value
    raise GeometryError('source-cell proof input is not a detachable literal')


def _pinned_callback(model, regions, callback, operands=()):
    """No user callback may alter any operand subsequently read by proof.

    Check every callback frontier, not merely initial/final equality: a
    temporary repair followed by restoration must never acquire authority.
    Ordinary callback-free qualification pays no per-proof-step pin cost.
    """
    if callback is None:
        return None
    from .cancellation import is_cancellation_signal
    if is_cancellation_signal(callback):
        # an issued signal executes no caller code, so it cannot mutate any operand between frontiers;
        # every proof check, cancellation test and final definition/runtime gate still applies
        return callback
    from .native_support_snapshots import capture_native_supports
    from .nurbs_runtime_binding import validate_model_curve_runtimes
    from .serialization import _qualified_model_state
    native = capture_native_supports(model)
    revision, checksum = model.revision, _checksum(model)
    region_signature = definition_checksum(regions)
    signatures = tuple(_literal(value) for value in operands)
    def pinned(phase):
        cancelled = callback(phase)
        try:
            # whole evaluated kernels, point caches and section arc/slab/world caches, before any other read:
            # a temporary private poison (and its restoration) inside the callback is refused here
            validate_model_curve_runtimes(model)
            current_native = capture_native_supports(model)
            _qualified_model_state(model,expected_checksum=checksum,
                changed_message='curvilinear proof source changed during callback')
            if (model.revision != revision
                    or current_native != native
                    or definition_checksum(regions) != region_signature
                    or tuple(_literal(value) for value in operands) != signatures):
                raise GeometryError('pinned source or operand differs')
        except GeometryError as cause:
            raise GeometryError(f'curvilinear proof operand changed during callback: {cause}') from cause
        return cancelled
    return pinned


def _positive_error(value, name):
    if isinstance(value, bool):
        raise GeometryError(f'{name} must be finite and positive')
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise GeometryError(f'{name} must be finite and positive') from error
    if not math.isfinite(result) or result <= 0:
        raise GeometryError(f'{name} must be finite and positive')
    return result


def _current(model, regions, expected_revision, cancellation_check):
    _check(cancellation_check, 'curvilinear source binding')
    if expected_revision is not None and expected_revision != model.revision:
        raise GeometryError('curvilinear source revision is stale')
    validate_material_surface_regions_binding(model, regions, cancellation_check=cancellation_check)
    from .nurbs_runtime_binding import validate_model_curve_runtimes
    validate_model_curve_runtimes(model)
    return _checksum(model), definition_checksum(regions)


def _box_corners(box):
    return product(*[(Q(float(a)), Q(float(b))) for a, b in zip(*box)])


def _charged_bounds(curve, lo, hi):
    from .plane_nurbs import charge_box
    charge_box()
    if type(curve) is CurveOnSurface:
        # Endpoint guards, actual-point witnesses AND the unregistered-family
        # fallback need actual definitions, even when a hostile iso cache says
        # an ineligible non-iso curve has a straight iso representation.
        return curve.bounds(lo, hi, fresh=True)
    return curve.bounds(lo, hi)


class _WorldChordNeedsRefinement(GeometryError):
    """A certified source point lies outside the requested chord tube."""
    def __init__(self, parameter, depth):
        super().__init__('source boundary world-chord error exceeds the requested bound')
        self.parameter, self.depth = parameter, depth


# Finite evaluator context for this sufficient specialization. Unknown wrappers
# retain the established enclosure route; no mutable authority cache is added.
from . import nurbs as _chord_nurbs
from . import _local_curve_geometry as _chord_jets
from .nurbs_surface_intersection import _intervals as _chord_intervals

_COS_CHORD_INTERVAL = _chord_jets.I


def _chord_function_pin(function):
    # Same value as the former sorted()/getattr form; only the empty-kwdefaults
    # case (nearly every pinned function) skips building and sorting a dict.
    return (function, getattr(function, '__code__', None),
            getattr(function, '__defaults__', None),
            tuple((key, id(value)) for key, value in sorted(kwdefaults.items()))
            if (kwdefaults := getattr(function, '__kwdefaults__', None)) else ())


from . import _cert_exact_runtime as _exact_runtime

_EXACT_RUNTIME_PINS = tuple((name, _chord_function_pin(getattr(_exact_runtime, name)))
                          for name in ('fraction_kind', 'binding'))


def _exact_arithmetic_context(_runtime=_exact_runtime, _pins=_EXACT_RUNTIME_PINS):
    """Reject renewed admission declarations before capturing a proof context."""
    from . import nurbs_metric as metric, support_curves as section
    if _exact_runtime is not _runtime or _EXACT_RUNTIME_PINS is not _pins:
        raise GeometryError('exact arithmetic admission declaration changed')
    for name, pin in _pins:
        current = getattr(_runtime, name)
        if (current is not pin[0] or current.__code__ is not pin[1]
                or current.__defaults__ is not pin[2] or current.__kwdefaults__ is not None):
            raise GeometryError('exact arithmetic admission calculator changed')
    aliases = (_chord_nurbs._fraction_kind, section._fraction_kind, metric._fraction_kind)
    declaration = _runtime.binding(aliases)
    if declaration is None:
        raise GeometryError('exact arithmetic admission binding changed')
    return (_pins, declaration, aliases, id(section.Fraction), id(_chord_nurbs.Fraction), id(metric.Q))


_EXACT_CONTEXT_PIN = _chord_function_pin(_exact_arithmetic_context)


_COS_CHORD_CONTEXT = tuple((owner, name, _chord_function_pin(getattr(owner, name)))
    for owner, names in (
        (_exact_runtime, ('fraction_kind', 'binding')),
        (_chord_nurbs.CurveOnSurface, ('evaluate', 'bounds', 'exact_point')),
        (_chord_nurbs.NurbsCurve, ('evaluate', 'exact_hull', 'exact_point')),
        (_chord_nurbs.NurbsSurface, ('evaluate', 'evaluate_many', 'exact_net', 'exact_point')),
        (_chord_nurbs, ('_sqrt_up', '_exact_chart', '_exact_span', '_exact_basis')),
        (_chord_nurbs.math, ('isqrt',)),
        (_COS_CHORD_INTERVAL, ('__init__', 'scale', '__add__', '__sub__')),
        (_chord_jets, ('_cos', '_curve_jet', '_surface_jet', '_rational', '_divide',
                       'jet', 'bezier_piece', '_exact_refine')),
    ) for name in names)
_COS_CHORD_CLASSES = (CurveOnSurface, _chord_nurbs.NurbsCurve, NurbsSurface)


def _canonical_cos(curve):
    if (_chord_jets.I is not _COS_CHORD_INTERVAL
            or _chord_intervals.Interval is not _COS_CHORD_INTERVAL):
        return False
    if (_chord_nurbs.CurveOnSurface, _chord_nurbs.NurbsCurve,
            _chord_nurbs.NurbsSurface) != _COS_CHORD_CLASSES:
        return False
    if (type(curve) is not _COS_CHORD_CLASSES[0]
            or type(curve.support) is not _COS_CHORD_CLASSES[2]
            or type(curve.pcurve) is not _COS_CHORD_CLASSES[1]
            or curve.transform != _IDENTITY
            or type(curve.also_on) is not tuple or curve.also_on):
        return False
    if (_exact_arithmetic_context is not _EXACT_CONTEXT_PIN[0]
            or _exact_arithmetic_context.__code__ is not _EXACT_CONTEXT_PIN[1]
            or _exact_arithmetic_context.__defaults__ is not _EXACT_CONTEXT_PIN[2]
            or _exact_arithmetic_context.__kwdefaults__ is not None):
        raise GeometryError('exact arithmetic context calculator changed')
    _exact_arithmetic_context()
    return all(_chord_function_pin(getattr(owner, name)) == pin
               for owner, name, pin in _COS_CHORD_CONTEXT)


def _cos_c2_world_chord_bound(curve, lo, hi, end_error, check, *, parameterized=False):
    """Exact descriptor Taylor bound; unsupported knot spans retain fallback."""
    if not _canonical_cos(curve):
        return None
    from ._local_curve_geometry import _cos, _Unresolved
    from .nurbs_runtime_binding import validate_curve_runtime
    from .plane_nurbs import charge_box
    a, b = sorted((Q(lo), Q(hi)))
    if not 0 <= a < b <= 1:
        return None
    pcurve = curve.pcurve
    u, v = map(Q, pcurve.domain)
    left, right = sorted((u+(v-u)*a, u+(v-u)*b))
    # No derivative jump, even a merely C1 knot, is assumed smooth here.
    if any(left < Q(k) < right for k in pcurve.knots):
        return None
    for knots, degree in ((curve.support.knots_u, curve.support.degree_u),
                          (curve.support.knots_v, curve.support.degree_v)):
        if any(knots.count(k) > degree for k in set(knots) if knots[0] < k < knots[-1]):
            return None
    signature = definition_checksum(curve)
    check()
    if not _canonical_cos(curve):
        raise GeometryError('world-chord COS evaluator changed during qualification')
    if definition_checksum(curve) != signature:
        raise GeometryError('world-chord COS changed during qualification')
    validate_curve_runtime(pcurve)
    charge_box()
    if pcurve.weights is not None and len(set(pcurve.weights)) > 1:
        # Independent numerator/denominator intervals can over-enclose a valid
        # positive-weight rational image beyond [0,1]. That is not evidence
        # that the curve leaves its support. Refuse only actual endpoint
        # witnesses; otherwise let the original world enclosure prove/refuse.
        for t in (a,b):
            if any(not 0 <= coordinate <= 1 for coordinate in pcurve.exact_point(t)):
                raise GeometryError('world-chord COS endpoint leaves the support chart')
        uv,_,_ = _chord_jets._curve_jet(pcurve,a,b)
        if any(value.lo < 0 or value.hi > 1 for value in uv):
            check()
            if not _canonical_cos(curve):
                raise GeometryError('world-chord COS evaluator changed during qualification')
            validate_curve_runtime(pcurve)
            if definition_checksum(curve) != signature:
                raise GeometryError('world-chord COS changed during qualification')
            return None
        charge_box()  # the following jet calculation is distinct work
    try:
        rows = _cos(curve, a, b)
    except _Unresolved:
        return None  # full chart box crosses a support knot; original route
    # _cos is the explicit chain rule in descriptor t, never implicit flow.
    if any(type(value) is not _COS_CHORD_INTERVAL for values in rows[0][1:3] for value in values):
        raise GeometryError('world-chord COS jet has an unknown interval type')
    acceleration = rows[0][2]
    maximum = sum((max(abs(x.lo), abs(x.hi)) for x in acceleration), Q(0))
    # Exact endpoints belong to the same descriptor law as this jet. Charge
    # each point enclosure; no sampled chord direction becomes authority.
    charge_box()
    first = curve.exact_point(a)
    charge_box()
    last = curve.exact_point(b)
    direction = tuple(y-x for x, y in zip(first, last))
    length_squared = sum((d*d for d in direction), Q(0))
    if length_squared > 0 and not parameterized:
        velocity = rows[0][1]
        projection = sum((v.scale(d) for v, d in zip(velocity, direction)), _chord_jets.I(0))
        if projection.lo >= 0:
            # Nondecreasing projection starts at 0 and ends at 1, so the
            # closest point is on the finite segment, not merely its line.
            cross = tuple(acceleration[j].scale(direction[k]) - acceleration[k].scale(direction[j])
                          for j, k in ((1, 2), (2, 0), (0, 1)))
            cross_squared = sum((max(abs(x.lo), abs(x.hi))**2 for x in cross), Q(0))
            perpendicular = _chord_nurbs._sqrt_up(cross_squared / length_squared)
            maximum = min(maximum, perpendicular)
    # Interpolating stored endpoint errors contributes at most end_error,
    # once, for either sufficient remainder bound.
    upper = maximum*(b-a)**2/8 + Q(end_error)
    check()
    if not _canonical_cos(curve):
        raise GeometryError('world-chord COS evaluator changed during qualification')
    validate_curve_runtime(pcurve)
    if definition_checksum(curve) != signature:
        raise GeometryError('world-chord COS changed during qualification')
    return upper


def _cos_chart_chord_bound(support, curve, lo, hi, first, last):
    """Convex distance on positive-weight rational controls in the true chart."""
    if not _canonical_cos(curve) or type(support) is not NurbsSurface or curve.support != support:
        return None
    from .nurbs_domain import _exact_segment_distance
    from .nurbs_runtime_binding import validate_curve_runtime
    validate_curve_runtime(curve.pcurve)
    hull = curve.pcurve.exact_hull(lo, hi)
    return max(_exact_segment_distance(point, first, last) for point in hull)


def _single_c2_world_chord_bound(curve,lo,hi,end_error,check):
    """Certified Taylor remainder for ONE original continuation piece, or None.

    This is a sufficient upper bound only. Failure to meet a requested target
    cannot prove that the actual chord needs refinement.
    """
    if type(curve) is not NurbsSurfaceIntersectionCurve:
        return None
    a,b = min(Q(lo),Q(hi)),max(Q(lo),Q(hi))
    if a == b:
        return None
    start,end = curve.domain
    count = len(curve.owner.segments)
    # breakpoints() is a display float API. Exact source stations are required
    # here so a rounded station cannot hide a derivative jump inside [a,b].
    if any(a < (Q(i,count)-start)/(end-start) < b for i in range(count+1)):
        return None
    check()
    from .plane_nurbs import charge_box
    charge_box()  # one actual owner interval-enclosure query
    bounds = curve.enclose(a,b)
    if len(bounds) != 1 or bounds[0].parameter_range != (a,b):
        return None
    acceleration = bounds[0].world_second_derivative
    maximum = sum((max(abs(x.lo),abs(x.hi)) for x in acceleration),Q(0))
    # L1 bounds Euclidean norm, without any floating rounding. The endpoint
    # term covers interpolation of the stored targets versus true endpoints.
    upper = maximum*(b-a)**2/8+Q(end_error)
    check()
    return upper


def _registered_world_chord(curve, lo, hi, endpoints, error, check):
    from .nurbs_domain import _exact_segment_distance
    from .nurbs import _float_down
    from .plane_nurbs import charge_box, charge_split, depth_limit
    from .registered_world_hulls import fixed_chord_hull_bound
    curve_signature = definition_checksum(curve)
    section_bisections = ((type(curve._INITIAL_ROOT_BISECTIONS), curve._INITIAL_ROOT_BISECTIONS)
                          if type(curve) is PlaneNurbsCurve else None)
    if section_bisections is not None and (section_bisections[0] is not int or section_bisections[1] != 60):
        # Native implicit bounds also encode the sixty-step evaluator law.
        # An unsupported structural hull must not route a changed law into
        # those bounds, including the END_ERROR and midpoint witness boxes.
        raise GeometryError('world-chord section evaluator law is not certified')
    def current():
        check()
        # The caller callback also checks the model, but this local reference
        # may have been detached by an equal-definition edge replacement.
        if definition_checksum(curve) != curve_signature:
            raise GeometryError('world-chord curve changed during qualification')
        if section_bisections is not None:
            value = curve._INITIAL_ROOT_BISECTIONS
            if type(value) is not section_bisections[0] or value != section_bisections[1]:
                raise GeometryError('world-chord section evaluator law changed during qualification')
    def enclosure(a, b):
        # Counts actual owner enclosure queries, including degenerate point
        # boxes; do not additionally charge the enclosing proof-loop tile.
        return _charged_bounds(curve, a, b)
    current()
    exact_ends = [tuple(map(Q, p)) for p in endpoints]
    end_error = max(_exact_segment_distance(p, q, q) for t, q in zip((lo, hi), exact_ends)
                    for p in _box_corners(enclosure(float(t), float(t))))
    pending, accepted, worst = [(min(lo, hi), max(lo, hi), 0)], [], Q(0)
    while pending:
        current()
        a, b, depth = pending.pop()
        lower, upper = max(0., _float_down(a)), min(1., _float_up(b))
        hull_bound = fixed_chord_hull_bound(curve, lower, upper, exact_ends)
        bound = (hull_bound if hull_bound is not None else
                 max(_exact_segment_distance(p, *exact_ends)
                     for p in _box_corners(enclosure(lower, upper))))+end_error
        if bound <= Q(error):
            accepted.append((a, b))
            worst = max(worst, bound)
            continue
        # An actual point enclosure, including on a quarter/subtile station,
        # can prove that the FE chord itself needs refinement. This differs
        # from a merely inconclusive enclosure or exhausted proof budget.
        mid = Q(float((a+b)/2))
        if a < mid < b:
            box = tuple(zip(*enclosure(float(mid), float(mid))))
            centre = tuple((Q(float(x))+Q(float(y)))/2 for x, y in box)
            radius = sum((Q(float(y))-Q(float(x)))/2 for x, y in box)
            direction = tuple(y-x for x, y in zip(*exact_ends))
            delta = tuple(p-x for p, x in zip(centre, exact_ends[0]))
            length2 = sum(x*x for x in direction)
            t = Q(0) if not length2 else max(Q(0), min(Q(1), sum(x*y for x, y in zip(delta, direction))/length2))
            distance2 = sum((x-t*d)**2 for x, d in zip(delta, direction))
            if distance2 > (Q(error)+radius)**2:
                raise _WorldChordNeedsRefinement(mid, depth+1)
        if depth >= depth_limit(40) or len(pending)+len(accepted) >= 32768:
            raise GeometryError('source boundary world-chord error is not certified within the piece budget')
        mid = (a+b)/2
        charge_split()
        pending.extend(((a, mid, depth+1), (mid, b, depth+1)))
    return _float_up(worst), tuple(accepted)


class _ParameterizedChordInconclusive(GeometryError):
    """No boundary violation witness; existing finite proof limit reached."""


def _parameterized_world_chord(curve, lo, hi, endpoints, error, check):
    """Whole-vector interpolation bound, distinct from distance to a segment.

    A failed sufficient bound is not a witness. Only a certified point-box
    lower bound requests new FE stations; unresolved bounds retain refusal.
    """
    from .plane_nurbs import charge_split, depth_limit
    from .nurbs import _float_down
    from .nurbs_domain import _exact_segment_distance
    lo, hi = Q(lo), Q(hi)
    ends = tuple(tuple(map(Q, row)) for row in endpoints)
    signature = definition_checksum(curve)
    pin = _chord_function_pin
    def context():
        return tuple(pin(f) for f in (_parameterized_world_chord,_cos_c2_world_chord_bound,
            _single_c2_world_chord_bound,_charged_bounds,_canonical_cos,_chord_function_pin))
    expected = context()
    def guard():
        if context() != expected:
            raise GeometryError('parameterized boundary numerical context changed')
        check()
        if context() != expected:
            raise GeometryError('parameterized boundary numerical context changed')
        if definition_checksum(curve) != signature:
            raise GeometryError('parameterized boundary source changed')
    def enclosure(a, b):
        return tuple(tuple(Q(float(x)) for x in pair) for pair in
                     zip(*_charged_bounds(curve, _float_down(a), _float_up(b))))
    def linear(t):
        fraction = (t-lo)/(hi-lo)
        return tuple(a+(b-a)*fraction for a,b in zip(*ends))
    guard()
    end_error = max(_exact_segment_distance(point, target, target)
        for t,target in zip((lo,hi),ends)
        for point in product(*enclosure(t,t)))
    fast = (_cos_c2_world_chord_bound(curve,lo,hi,end_error,guard,parameterized=True)
            if type(curve) is CurveOnSurface else
            _single_c2_world_chord_bound(curve,lo,hi,end_error,guard))
    if fast is not None and fast <= Q(error):
        guard()
        return _float_up(fast), ((min(lo,hi),max(lo,hi)),)
    pending, accepted, maximum = [(min(lo,hi),max(lo,hi),0)], [], Q(0)
    while pending:
        guard()
        a,b,depth = pending.pop()
        box = enclosure(a,b)
        first,last = linear(a),linear(b)
        squared = sum(max(abs(x-min(u,v)),abs(y-max(u,v)),
                          abs(x-max(u,v)),abs(y-min(u,v)))**2
                      for (x,y),u,v in zip(box,first,last))
        if squared <= Q(error)**2:
            from .nurbs import _sqrt_up
            maximum = max(maximum,_sqrt_up(squared));accepted.append((a,b));continue
        mid = Q(float((a+b)/2))
        if a < mid < b:
            point = enclosure(mid,mid);target = linear(mid)
            lower = sum(max(x-t,t-y,Q(0))**2 for (x,y),t in zip(point,target))
            if lower > Q(error)**2:
                guard()
                raise _WorldChordNeedsRefinement(mid,depth+1)
        if depth >= depth_limit(40) or len(pending)+len(accepted) >= 32768:
            raise _ParameterizedChordInconclusive('parameterized boundary error is inconclusive within the piece budget')
        charge_split();mid=(a+b)/2
        pending.extend(((a,mid,depth+1),(mid,b,depth+1)))
    guard()
    return _float_up(maximum),tuple(accepted)


def _compute_world_chord(curve, lo, hi, endpoints, error, check):
    if type(curve) not in (CurveOnSurface, NurbsSurfaceIntersectionCurve):
        return _registered_world_chord(curve, lo, hi, endpoints, error, check)
    from .nurbs_domain import _exact_segment_distance
    from .nurbs import _float_down
    from .plane_nurbs import charge_box, charge_split, depth_limit
    from ._boundary_curve_proofs import prepare_iso_hull,IsoHullUnavailable
    iso = None
    if type(curve) is CurveOnSurface:
        try:
            iso = prepare_iso_hull(curve)
        except IsoHullUnavailable:
            pass  # unsupported exact family retains its established proof route
    def enclosure(a, b):
        # Counts actual owner enclosure queries, including degenerate point
        # boxes; do not additionally charge the enclosing proof-loop tile.
        if iso is None:
            return _charged_bounds(curve, a, b)
        charge_box()
        points = iso.hull(a,b)
        return (np.asarray([_float_down(min(p[k] for p in points)-iso.margin[k]) for k in range(3)]),
                np.asarray([_float_up(max(p[k] for p in points)+iso.margin[k]) for k in range(3)]))
    exact_ends = [tuple(map(Q, p)) for p in endpoints]
    end_error = max(_exact_segment_distance(p, q, q) for t, q in zip((lo, hi), exact_ends)
                    for p in _box_corners(enclosure(float(t), float(t))))
    fast = (_cos_c2_world_chord_bound(curve,lo,hi,end_error,check)
            if type(curve) is CurveOnSurface else
            _single_c2_world_chord_bound(curve,lo,hi,end_error,check))
    if fast is not None and fast <= Q(error):
        return _float_up(fast),((min(Q(lo),Q(hi)),max(Q(lo),Q(hi))),)
    pending, accepted, worst = [(min(lo, hi), max(lo, hi), 0)], [], Q(0)
    while pending:
        check()
        a, b, depth = pending.pop()
        if iso is None:
            bound = max(_exact_segment_distance(p, *exact_ends)
                        for p in _box_corners(enclosure(_float_down(a), _float_up(b))))+end_error
        else:
            charge_box()
            points = iso.hull(a,b)
            # Distance to a convex segment is convex. Every positive-weight
            # rational curve point is a convex combination of these controls.
            bound = max(_exact_segment_distance(p,*exact_ends) for p in points)+sum(iso.margin,Q(0))+end_error
        if bound <= Q(error):
            accepted.append((a, b))
            worst = max(worst, bound)
            continue
        # An actual point enclosure, including on a quarter/subtile station,
        # can prove that the FE chord itself needs refinement. This differs
        # from a merely inconclusive enclosure or exhausted proof budget.
        mid = Q(float((a+b)/2))
        if a < mid < b:
            box = tuple(zip(*enclosure(float(mid), float(mid))))
            centre = tuple((Q(float(x))+Q(float(y)))/2 for x, y in box)
            radius = sum((Q(float(y))-Q(float(x)))/2 for x, y in box)
            direction = tuple(y-x for x, y in zip(*exact_ends))
            delta = tuple(p-x for p, x in zip(centre, exact_ends[0]))
            length2 = sum(x*x for x in direction)
            t = Q(0) if not length2 else max(Q(0), min(Q(1), sum(x*y for x, y in zip(delta, direction))/length2))
            distance2 = sum((x-t*d)**2 for x, d in zip(delta, direction))
            if distance2 > (Q(error)+radius)**2:
                raise _WorldChordNeedsRefinement(mid, depth+1)
        if depth >= depth_limit(40) or len(pending)+len(accepted) >= 32768:
            raise GeometryError('source boundary world-chord error is not certified within the piece budget')
        mid = (a+b)/2
        charge_split()
        pending.extend(((a, mid, depth+1), (mid, b, depth+1)))
    return _float_up(worst), tuple(accepted)


def _world_chord_key(curve,lo,hi,endpoints,error,check):
    from ._boundary_curve_proofs import literal
    from .nurbs_intersection_curve import intersection_curve_record
    from .nurbs import nurbs_record
    record = (intersection_curve_record(curve) if type(curve) is NurbsSurfaceIntersectionCurve
              else nurbs_record(curve))
    family = ((type(curve),type(curve.support),type(curve.pcurve),literal(curve.also_on),literal(curve.transform))
              if type(curve) is CurveOnSurface else (type(curve),type(curve.owner)))
    def rational(value):
        value = Q(value)
        return value.numerator,value.denominator
    return ('accepted-world-chord-v1',family,literal(record),rational(lo),rational(hi),
            tuple(tuple(rational(x) for x in p) for p in endpoints),rational(error))


def _accepted_world_chord_payload(curve,lo,hi,endpoints,error,check):
    bound,pieces = _compute_world_chord(curve,lo,hi,endpoints,error,check)
    return bound,tuple(((a.numerator,a.denominator),(b.numerator,b.denominator)) for a,b in pieces)


from .nurbs_intersection_curve import operation_cached
_accepted_world_chord = operation_cached(_world_chord_key)(_accepted_world_chord_payload)


def _world_chord(curve,lo,hi,endpoints,error,check):
    """Reuse only authenticated successful mathematical proofs, never callbacks."""
    if type(curve) not in (CurveOnSurface,NurbsSurfaceIntersectionCurve):
        return _compute_world_chord(curve,lo,hi,endpoints,error,check)
    from .nurbs_runtime_binding import validate_curve_runtime
    def frontier():
        check()
        if type(curve) is CurveOnSurface:
            validate_curve_runtime(curve.pcurve)
        else:
            curve._shape()  # current issuer/configuration checks also on cache hits
    before = _world_chord_key(curve,lo,hi,endpoints,error,check)
    frontier()
    if _world_chord_key(curve,lo,hi,endpoints,error,check) != before:
        raise GeometryError('world chord inputs changed at proof entry')
    if type(curve) is CurveOnSurface:
        from ._boundary_curve_proofs import prepare_iso_hull,IsoHullUnavailable
        try:
            prepare_iso_hull(curve)
        except IsoHullUnavailable:
            result = _compute_world_chord(curve,lo,hi,endpoints,error,frontier)
        else:
            value,pieces = _accepted_world_chord(curve,lo,hi,endpoints,error,frontier)
            result = value,tuple((Q(*a),Q(*b)) for a,b in pieces)
    else:
        value,pieces = _accepted_world_chord(curve,lo,hi,endpoints,error,frontier)
        result = value,tuple((Q(*a),Q(*b)) for a,b in pieces)
    frontier()
    if _world_chord_key(curve,lo,hi,endpoints,error,check) != before:
        raise GeometryError('world chord inputs changed before proof return')
    return result


def _uv_box(support, curve, parameter):
    if isinstance(support, NurbsSurface):
        if isinstance(curve,NurbsSurfaceIntersectionCurve):
            return curve.for_support(support).chart_box(parameter,parameter)
        if isinstance(curve, CurveOnSurface) and curve.support == support and curve.transform == _IDENTITY:
            p = curve.pcurve.exact_point(parameter)
            return ((p[0], p[0]), (p[1], p[1]))
        if isinstance(curve, PlaneNurbsCurve) and curve.surface == support:
            from .nurbs_section_faces import _Chart
            return _Chart(support).rect(*curve.chart_enclosure(float(parameter)))
        raise GeometryError('source boundary has no declared chart on this NURBS support')
    if isinstance(support, Plane):
        from .planar_exact import ExactPlane
        frame = ExactPlane.from_plane(support)
        points = [frame.uv(p) for p in _box_corners(_charged_bounds(curve, float(parameter), float(parameter)))]
        return tuple((min(p[i] for p in points), max(p[i] for p in points)) for i in range(2))
    raise GeometryError('curvilinear boundary chart support is not yet qualified')


def _uv_interval(support, curve, lo, hi):
    if isinstance(support, NurbsSurface):
        if isinstance(curve,NurbsSurfaceIntersectionCurve):
            return curve.for_support(support).chart_box(lo,hi)
        if isinstance(curve, PlaneNurbsCurve) and curve.surface == support:
            from .nurbs_section_faces import _Chart, _piece_rect
            return _piece_rect(curve, _Chart(support), float(lo), float(hi))
        if isinstance(curve, CurveOnSurface) and curve.support == support and curve.transform == _IDENTITY:
            hull = curve.pcurve.exact_hull(lo, hi)
            return tuple((min(p[i] for p in hull), max(p[i] for p in hull)) for i in range(2))
    if isinstance(support, Plane):
        from .planar_exact import ExactPlane
        frame = ExactPlane.from_plane(support)
        points = [frame.uv(p) for p in _box_corners(_charged_bounds(curve, float(lo), float(hi)))]
        return tuple((min(p[i] for p in points), max(p[i] for p in points)) for i in range(2))
    raise GeometryError('source boundary interval has no declared chart enclosure')


def query_prepared_material_boundary_chords(model, regions, edge_intervals, *,
        expected_revision=None, chord_error, cancellation_check=None, local_approximation=None):
    return _query_boundary_chords(model, regions, edge_intervals, expected_revision=expected_revision,
        chord_error=chord_error, cancellation_check=cancellation_check, local_approximation=local_approximation)


def _query_boundary_chords(model, regions, edge_intervals, *, expected_revision=None,
        chord_error, cancellation_check=None, local_approximation=None, references=None,
        frozen_curves=None, terminal_bounds=None):
    # Private operation-local arithmetic only. All live receipt qualification below still runs.
    from .nurbs_metric import certify_projected_chart
    from .nurbs_domain import _exact_segment_distance, ChartContainmentUndecided
    error = _positive_error(chord_error, 'chord_error')
    local_fingerprint=();local_query=None;interval_errors=[];interval_footprints=[]
    if local_approximation is not None:
        from .local_approximation_field import validate_local_approximation_allocation,prepare_local_boundary_limit_query
        local_fingerprint,_=validate_local_approximation_allocation(model,local_approximation,cancellation_check=cancellation_check)
        local_query=prepare_local_boundary_limit_query(model,local_approximation)
    input_rows = tuple(edge_intervals)
    input_signature = _literal(tuple((handle, Q(a), Q(b)) for handle, a, b in input_rows))
    cancellation_check = _pinned_callback(model, regions, cancellation_check, (input_rows,))
    checksum, region_signature = _current(model, regions, expected_revision, cancellation_check)
    revision = model.revision
    from .cancellation import is_cancellation_signal
    ordinary=cancellation_check is not None and not is_cancellation_signal(cancellation_check)
    def check():
        _check(cancellation_check, 'curvilinear boundary qualification')
        if local_approximation is not None and ordinary:
            validate_local_approximation_allocation(model,local_approximation)
    source_faces = {chart.face.id: chart for region in regions.regions for chart in region.sources}
    if references is None:
        references = _sampling_references(model, regions)
    plane_references = {face: pair[1] for face, pair in references.items()}
    frozen_curves = {} if frozen_curves is None else frozen_curves
    occurrence_curves = {}
    anchors = []
    for face_id, chart in sorted(source_faces.items()):
        check()
        if isinstance(chart.support, NurbsSurface):
            certify_projected_chart(chart.support, cancellation_check=lambda: (check() or False))
        elif not isinstance(chart.support, Plane):
            raise GeometryError('curvilinear source support immersion is not yet qualified')
        anchor = None
        anchor_box = ((Q(0), Q(1)), (Q(0), Q(1)))
        if isinstance(chart.support, Plane):
            boxes = [_uv_interval(chart.support, path.curve, Q(0), Q(1)) for loop in chart.boundaries for path in loop]
            anchor_box = tuple((min(box[i][0] for box in boxes), max(box[i][1] for box in boxes)) for i in (0, 1))
        for level in range(1, 6):
            size = 2**level
            for u, v in product(range(1, size, 2), repeat=2):
                point = tuple(a+(b-a)*Q(k, size) for (a, b), k in zip(anchor_box, (u, v)))
                try:
                    inside = chart.domain.contains_loop(chart.boundaries[0], point, chart.world_tolerance, boundary=False)
                    inside &= not any(chart.domain.contains_loop(loop, point, chart.world_tolerance, boundary=True)
                                      for loop in chart.boundaries[1:])
                except ChartContainmentUndecided:
                    continue
                if inside:
                    anchor = tuple(map(float, point))
                    break
            if anchor is not None:
                break
        if anchor is None:
            raise GeometryError('curvilinear source has no certified interior regular reference anchor')
        anchors.append((chart.face, anchor))
    intervals = []
    for handle, t0, t1 in input_rows:
        check()
        if type(handle) is not EntityHandle or handle.kind != 'edge' or handle.model_id != model.model_id:
            raise GeometryError('curvilinear intervals require model-bound edge handles')
        model.handle('edge', handle.id)
        try:
            a, b = Q(t0), Q(t1)
        except (ValueError, TypeError, OverflowError) as exc:
            raise GeometryError('source boundary parameters must be finite') from exc
        if not (0 <= a <= 1 and 0 <= b <= 1 and a != b) or Q(float(a)) != a or Q(float(b)) != b:
            raise GeometryError('source boundary stations require distinct exactly stored double parameters in [0,1]')
        from .arrangement_geometry import freeze_edge, LinePath
        from .nurbs import NurbsCurve
        if handle.id not in frozen_curves:
            frozen_curves[handle.id] = freeze_edge(model, handle.id)
        curve = frozen_curves[handle.id]
        if not isinstance(curve, (CurveOnSurface, PlaneNurbsCurve, NurbsCurve, LinePath,NurbsSurfaceIntersectionCurve)):
            raise GeometryError('curvilinear boundary edge family is not yet qualified')
        edge = model.edges[handle.id]
        if local_approximation is not None:
            from .local_approximation_field import local_boundary_corner_parameters
            if any(min(a,b)<t<max(a,b) for t in local_boundary_corner_parameters(curve)):
                raise GeometryError('local chord crosses a required trim corner station')
        xyz = tuple(tuple(float(v) for v in (model.vertex_position(edge.start) if t == 0 else
                    model.vertex_position(edge.end) if t == 1 else curve.evaluate(float(t)))) for t in (a, b))
        for t, target in zip((a, b), xyz):
            if t in (0, 1):
                endpoint_error = max(sum(abs(x-Q(v)) for x, v in zip(point, target))
                    for point in _box_corners(_charged_bounds(curve, float(t), float(t))))
                extent = float(np.linalg.norm(model.vertex_position(edge.end)-model.vertex_position(edge.start))) or 1.
                if endpoint_error > Q(model.tolerance.effective_length(extent)):
                    raise GeometryError('owner endpoint correction exceeds the existing joint precision')
        interval_error=error
        if local_query is not None:
            from .nurbs import _float_down
            limit,footprint=local_query(handle.id,a,b)
            interval_error=_positive_error(min(error,_float_down(limit)),'local interval error')
            interval_errors.append((handle.id,a,b,Q(interval_error)))
            interval_footprints.append((handle.id,a,b,footprint))
        terminal = None if terminal_bounds is None else terminal_bounds.get((handle.id, min(a,b), max(a,b)))
        if terminal is None:
            world_error, pieces = _world_chord(curve, a, b, xyz, interval_error, check)
        else:
            bound, pieces, qualified_error, qualified_xyz = terminal
            # A full-vector source-reference/descriptor bound also bounds distance
            # to its endpoint segment. Chart occurrence bounds remain independent.
            ordered_xyz = xyz if a < b else xyz[::-1]
            if qualified_xyz != ordered_xyz or qualified_error > Q(interval_error) or bound > Q(interval_error):
                raise GeometryError('terminal boundary arithmetic does not match final qualification')
            check()
            world_error = _float_up(bound)
        occurrences, chart_error = [], Q(0)
        for face_id in model.faces_using_edge(handle.id):
            if face_id not in source_faces:
                continue
            chart = source_faces[face_id]
            from .nurbs_sweep_chart import has_sweep_chart
            occurrence_key = (face_id,handle.id)
            if occurrence_key not in occurrence_curves:
                reference = plane_references[face_id]
                # _sampling_references already resolves the per-face chart model.
                # Reuse that same operation-local owner instead of reconstructing
                # every sweep connector for every accepted dyadic interval.
                owner = reference.owner if hasattr(reference,'owner') else reference.model
                occurrence_curves[occurrence_key] = (owner.edges[handle.id].curve
                    if has_sweep_chart(model.faces[face_id]) else curve)
            occurrence_curve = occurrence_curves[occurrence_key]
            uses = [use for loop in (model.faces[face_id].loop, *model.faces[face_id].holes)
                    for use in loop if use.edge == handle.id]
            if len(uses) != 1:
                raise GeometryError('source boundary edge has ambiguous face occurrence')
            boxes = tuple(_uv_box(chart.support, occurrence_curve, t) for t in (a, b))
            from .regular_source_reference import source_node
            uv = []
            for t in (a, b):
                if t in (0, 1) and isinstance(chart.support, (NurbsSurface, Plane)):
                    vertex = edge.start if t == 0 else edge.end
                    true = plane_references[face_id].node(handle.id, t, vertex)
                    uv.append(tuple(float((lo+hi)/2) for lo, hi in true.box))
                else:
                    uv.append(tuple(float(x) for x in chart.domain.uv(occurrence_curve, float(t))))
            uv = tuple(uv)
            q0, q1 = [tuple(map(Q, q)) for q in uv]
            for lo, hi in pieces:
                bound = _cos_chart_chord_bound(chart.support, occurrence_curve, lo, hi, q0, q1)
                if bound is None:
                    rect = _uv_interval(chart.support, occurrence_curve, lo, hi)
                    bound = max(_exact_segment_distance(p, q0, q1) for p in product(*rect))
                chart_error = max(chart_error, bound)
            occurrences.append(BoundaryChordOccurrence(chart.face, uses[0].forward == (a < b), uv, boxes))
        if not occurrences:
            raise GeometryError('source boundary edge has no selected material occurrence')
        intervals.append(MaterialBoundaryChord(handle, (a, b), ((handle.id, a), (handle.id, b)), xyz,
                                               tuple(occurrences), world_error, _float_up(chart_error)))
    _check(cancellation_check, 'curvilinear boundary qualification complete')
    # Callback mutation of evaluated private kernels is invisible to public
    # serialization. Validate them again after the last callback and before
    # any receipt is registered; legitimate cache fills are permitted.
    from .nurbs_runtime_binding import validate_model_curve_runtimes
    validate_model_curve_runtimes(model)
    if (model.revision != revision or _checksum(model) != checksum
            or definition_checksum(regions) != region_signature
            or _literal(tuple((handle, Q(a), Q(b)) for handle, a, b in input_rows)) != input_signature):
        raise GeometryError('curvilinear source changed during boundary qualification')
    result = PreparedMaterialBoundaryChords(model.model_id, revision, checksum, tuple(intervals), tuple(anchors), error, local_fingerprint,tuple(interval_errors),tuple(interval_footprints))
    if local_approximation is not None:
        validate_local_approximation_allocation(model,local_approximation)
        _LOCAL_CHORD_ALLOCATIONS[result]=local_approximation
    _CHORDS[result] = (ref(model), region_signature, _literal(result))
    return result


def validate_prepared_material_boundary_chords_binding(model, regions, result, *,
        expected_revision=None, cancellation_check=None):
    if type(result) is not PreparedMaterialBoundaryChords or result not in _CHORDS:
        raise GeometryError('curvilinear boundary receipt was not issued by this owner')
    cancellation_check = _pinned_callback(model, regions, cancellation_check, (result,))
    checksum, region_signature = _current(model, regions, expected_revision, cancellation_check)
    source, bound_regions, literal = _CHORDS[result]
    if (source() is not model or result.revision != model.revision or result.source_checksum != checksum
            or region_signature != bound_regions or _literal(result) != literal):
        raise GeometryError('curvilinear boundary receipt is stale or changed')
    if result.local_field_fingerprint:
        from .local_approximation_field import validate_local_approximation_allocation
        fingerprint,_=validate_local_approximation_allocation(model,_LOCAL_CHORD_ALLOCATIONS.get(result))
        if fingerprint!=result.local_field_fingerprint:raise GeometryError('local chord field binding changed')


def _sampling_references(model, regions):
    from .plane_source_reference import PlaneReference
    from .regular_source_reference import NurbsReference
    from .source_cell_complex import _support_owner
    charts = {chart.face.id: chart for region in regions.regions for chart in region.sources}
    result = {}
    for face,chart in charts.items():
        owned = _support_owner(model,face)
        reference = (PlaneReference(owned,owned.faces[face]) if isinstance(chart.support,Plane)
                     else NurbsReference(owned,owned.faces[face]))
        result[face] = (chart.support,reference)
    return result


def _cell_reference_calculator_context():
    """Finite source/memo/metric calculator slice bound before any callback.

    Class identity does not bind its methods. Include private dispatch and
    arithmetic aliases, as in the owner's rational fan proof context.
    """
    if (_exact_arithmetic_context is not _EXACT_CONTEXT_PIN[0]
            or _exact_arithmetic_context.__code__ is not _EXACT_CONTEXT_PIN[1]
            or _exact_arithmetic_context.__defaults__ is not _EXACT_CONTEXT_PIN[2]
            or _exact_arithmetic_context.__kwdefaults__ is not None):
        raise GeometryError('exact arithmetic context calculator changed')
    exact_arithmetic = _exact_arithmetic_context()
    from . import regular_source_reference as source, plane_source_reference as plane
    from . import curvilinear_reference_maps as maps, nurbs_metric as metric
    from . import nurbs_runtime_binding as runtime, native_support_snapshots as native
    from . import _source_box_hull as enclosure, planar_exact as planar
    inventory = (
        (source, ('_plate_joint','_joint_source_node','_constant','_blend',
            '_section_key','_root_node','_path_node','source_node','source_side',
            '_view','_point_rectangle','_pcurve_point_jet','definition_checksum',
            'add','mul','sub','scale')),
        (source.SourceSide, ('_ranges','box','derivative')),
        (source.RationalSourceSide, ('_validate','_ranges','box','derivative')),
        (source.IntersectionSourceSide, ('_view','box','derivative')),
        (source.NurbsReference, ('__init__','node','side')),
        (plane.PlaneReference, ('__init__','_node','node','side')),
        (plane.PlaneSide, ('box','derivative')),
        (plane, ('project','source_node','source_side','derivative_intervals','add','mul')),
        (maps._SideMemo, ('__init__','__getattr__','_ranges','box','derivative')),
        (maps, ('_source_box','_float_down','_float_up')),
        (enclosure, ('source_enclosure','_context','_canonical_literals','_keywords')),
        (planar.ExactPlane, ('__init__','__post_init__','world','uv','uv_direction','offset')),
        (planar, ('_add','_sub','_scale','exact','exact_vector','exact_cross','exact_dot')),
        (metric, ('derivative_intervals','_homogeneous','_rect','_range',
            '_derivative_ranges','add','sub','mul','scale','sub_patches_of')),
        (_chord_jets, ('_curve_jet','_surface_jet','_cos','_rational','_divide',
            'jet','bezier_piece','_exact_refine')),
        (_chord_jets.I, ('__init__','__add__','__sub__','__neg__','__mul__',
            'scale','hull','intersects','contains_zero','excludes_zero')),
        (runtime, ('validate_curve_runtime',)),
        (native, ('_material_support_read_pin',)),
    )
    functions = tuple((owner,name,_chord_function_pin(getattr(owner,name)))
        for owner,names in inventory for name in names)
    functions += tuple((owner,name,_chord_function_pin(getattr(owner,name)))
        for owner,name,_ in _COS_CHORD_CONTEXT)
    functions += ((_chord_function_pin(metric._homogeneous.__wrapped__)),)
    # source_enclosure validates its specialized dispatch and cached net on
    # every call. Bind both those guard declarations and their actual live
    # operands; replacing/renewing the declaration cannot renew this operation.
    enclosure_functions = tuple((owner,name,_chord_function_pin(getattr(owner,name)))
        for owner,name,*_ in enclosure._FUNCTIONS)
    enclosure_bindings = tuple((owner,name,id(getattr(owner,name)))
        for owner,name,_ in enclosure._BINDINGS)
    enclosure_declarations = tuple(id(getattr(enclosure,name)) for name in
        ('_FUNCTIONS','_BINDINGS','_METHODS','_SURFACE','_UNKNOWN','Fraction','MethodType',
         'caps','nurbs','math'))
    # from_plane is a classmethod; pin its underlying function, not the fresh
    # bound-method object produced by descriptor access.
    plane_constructor = _chord_function_pin(planar.ExactPlane.from_plane.__func__)
    classes = (source.SourceNode,source.SourceSide,source.RationalSourceSide,
        source.IntersectionSourceSide,source.NurbsReference,plane.PlaneReference,
        plane.PlaneSide,maps._SideMemo,_chord_jets.I,_chord_intervals.Interval,planar.ExactPlane)
    aliases = (source.Q,plane.Q,maps.Q,metric.Q,_chord_jets.Q,_chord_nurbs.Fraction)
    return (classes,functions,aliases,plane_constructor,enclosure_functions,
        enclosure_bindings,enclosure_declarations,exact_arithmetic,
        _chord_function_pin(_exact_arithmetic_context))


def _cell_reference_world_chord(model, references, edge_id, a, b, endpoints, error, check):
    """Full-vector interpolation bound on EVERY actual incident FE side map.

    SourceSide speed is its certified free-coordinate map, not descriptor t.
    RationalSourceSide retains its actual pcurve parameterization. The bound
    uses a midpoint residual plus integrated interval derivative residual.
    It is never merely distance to the geometric chord.
    """
    from .regular_source_reference import SourceSide, RationalSourceSide
    from .curvilinear_reference_maps import _source_box, _SideMemo
    from .nurbs_metric import derivative_intervals, add, mul, sub
    from .plane_nurbs import charge_box, charge_split, depth_limit
    from .nurbs import _sqrt_up
    calculator_context = _cell_reference_calculator_context()
    ends = tuple(tuple(map(Q,row)) for row in endpoints)
    chord = tuple(y-x for x,y in zip(*ends))
    edge = model.edges[edge_id]
    maximum, selected = Q(0), 0
    for face in model.faces_using_edge(edge_id):
        if face not in references: continue
        selected += 1
        support,reference = references[face]
        nodes = tuple(reference.node(edge_id,t,edge.start if t==0 else edge.end if t==1 else None)
                      for t in (a,b))
        side = reference.side(edge_id,(a,b),*nodes)
        signature=definition_checksum((support,side))
        pin=_chord_function_pin
        context=tuple(pin(f) for f in (_cell_reference_world_chord,_source_box,
            derivative_intervals,type(side).box,type(side).derivative,
            _SideMemo, _SideMemo._ranges, _SideMemo.box, _SideMemo.derivative))
        def guard():
            check()
            if (_cell_reference_calculator_context()!=calculator_context or
                    definition_checksum((support,side))!=signature or
                    tuple(pin(f) for f in (_cell_reference_world_chord,_source_box,
                        derivative_intervals,type(side).box,type(side).derivative,
                        _SideMemo, _SideMemo._ranges, _SideMemo.box, _SideMemo.derivative))!=context):
                raise GeometryError('cell-reference boundary source or calculator changed')
        # Keep the original side in the binding guard; memoize only its exact
        # arithmetic. In particular rational sides use this subdivided full-vector
        # proof rather than an unsubdivided descriptor Taylor enclosure.
        calculator = _SideMemo(side) if type(side) in (SourceSide, RationalSourceSide) else side
        pending=[(Q(0),Q(1),0)]
        while pending:
            guard();charge_box()
            lo,hi,depth=pending.pop();mid=(lo+hi)/2
            center=_source_box(support,calculator.box(mid,mid))
            linear=tuple(x+mid*d for x,d in zip(ends[0],chord))
            lower=sum(max(x-t,t-y,Q(0))**2 for (x,y),t in zip(center,linear))
            if lower>Q(error)**2:
                guard()
                # A true reference witness proves this interval needs a finer
                # station proposal. Its descriptor midpoint is only a proposal.
                station=Q(float((a+b)/2))
                raise _WorldChordNeedsRefinement(station,1)
            box=calculator.box(lo,hi);rates=calculator.derivative(lo,hi)
            # The derivative interval primitive requires positive-area patches.
            # A side may hold one native coordinate exactly constant; enclosing
            # that axis by its whole chart is conservative, never a sampled jet.
            derivative_box = (tuple((Q(0),Q(1)) if x==y else (x,y) for x,y in box)
                              if isinstance(support,NurbsSurface) else box)
            du,dv=derivative_intervals(support,derivative_box)
            derivative=tuple(sub(add(mul(u,rates[0]),mul(v,rates[1])),(d,d))
                for u,v,d in zip(du,dv,chord))
            radius=(hi-lo)/2
            bounds=tuple(max(abs(x-t),abs(y-t))+radius*max(abs(d0),abs(d1))
                for (x,y),t,(d0,d1) in zip(center,linear,derivative))
            squared=sum(value*value for value in bounds)
            if squared<=Q(error)**2:
                maximum=max(maximum,_sqrt_up(squared));continue
            if depth>=depth_limit(40):
                raise _ParameterizedChordInconclusive('cell-reference boundary interpolation bound is inconclusive')
            charge_split();pending.extend(((mid,hi,depth+1),(lo,mid,depth+1)))
        guard()
    if not selected:raise GeometryError('cell-reference sampling has no selected incident face')
    return _float_up(maximum),((min(a,b),max(a,b)),)


def _cell_reference_is_world_affine(model, references, edge_id, a, b, check):
    """Sufficient exact world test; chart/descriptor straightness alone is insufficient."""
    from .regular_source_reference import SourceSide
    from .nurbs_metric import derivative_intervals, add, mul
    from .plane_nurbs import charge_box
    selected = 0
    edge = model.edges[edge_id]
    for face in model.faces_using_edge(edge_id):
        if face not in references:
            continue
        check()
        selected += 1
        support, reference = references[face]
        nodes = tuple(reference.node(edge_id,t,edge.start if t==0 else edge.end if t==1 else None)
                      for t in (a,b))
        side = reference.side(edge_id,(a,b),*nodes)
        if type(side) is not SourceSide or side.curve is not None:
            return False
        charge_box()
        box, rates = side.box(Q(0),Q(1)), side.derivative(Q(0),Q(1))
        patch = (tuple((Q(0),Q(1)) if x==y else (x,y) for x,y in box)
                 if isinstance(support,NurbsSurface) else box)
        du,dv = derivative_intervals(support,patch)
        world = tuple(add(mul(u,rates[0]),mul(v,rates[1])) for u,v in zip(du,dv))
        if any(lo != hi for lo,hi in world):
            return False
    check()
    return bool(selected)


from contextvars import ContextVar as _SamplingContextVar
_SAMPLING_ARITHMETIC = _SamplingContextVar('prepared_material_sampling_arithmetic',default=None)
_SAMPLING_ARITHMETIC_ISSUER = _SamplingContextVar('prepared_material_sampling_arithmetic_issuer',default=None)


def _sampling_arithmetic_variable(variable=_SAMPLING_ARITHMETIC,issuer=_SAMPLING_ARITHMETIC_ISSUER):
    if _SAMPLING_ARITHMETIC is not variable or _SAMPLING_ARITHMETIC_ISSUER is not issuer:
        raise GeometryError('boundary sampling operation arithmetic identity changed')
    return variable,issuer


def _sampling_arithmetic_context():
    from .native_support_snapshots import capture_native_supports
    from .arrangement_geometry import freeze_edge
    from .serialization import _qualified_model_state
    return (_cell_reference_calculator_context(),tuple(_chord_function_pin(f) for f in
        (_cell_reference_calculator_context,_cell_reference_world_chord,
         _cell_reference_is_world_affine,_world_chord,_parameterized_world_chord,
         _sampling_bound_arithmetic,_sampling_terminal_arithmetic,
         _new_sampling_arithmetic,_sampling_arithmetic_context,_sampling_arithmetic_variable,
         _sampling_arithmetic_producer,_sampling_references,_current,_checksum,
         capture_native_supports,freeze_edge,_qualified_model_state)))


def _new_sampling_arithmetic():
    # Rows are closed over and only the opaque owner producer can populate its
    # namespace. Generic caller scratch cannot seed this producer's keys.
    rows, sources = {}, {}
    expected = _sampling_arithmetic_context()
    closed = False
    def guard():
        if closed or _sampling_arithmetic_context() != expected:
            raise GeometryError('boundary sampling operation arithmetic context changed')
    def access(namespace,model,epoch,key,error,builder):
        guard()
        source_key = namespace,id(model)
        previous = sources.get(source_key)
        if previous is not None and (previous[0] is not model or previous[1] != epoch):
            raise GeometryError('boundary sampling operation source changed')
        if previous is None:
            if len(sources) >= 4096:
                rows.clear();sources.clear()  # no retained hit survives source-binding eviction
            sources[source_key] = model,epoch
        cache_key = namespace,id(model),key
        result = rows.get(cache_key)
        if result is not None and result[0] <= error:
            guard()
            return result
        result = builder()
        guard()
        # Only completed successful arithmetic enters the bank. Refusals and
        # counters are never cached; tighter error requests recompute normally.
        value = Q(result[0]),tuple(tuple(map(Q,piece)) for piece in result[1])
        if value[0] > error:
            raise GeometryError('boundary sampling arithmetic exceeds requested error')
        if len(rows) >= 4096:
            rows.clear()  # bounded eviction is only a safe miss, not a new Work allowance
        rows[cache_key] = value
        return value
    def cleanup():
        nonlocal closed
        closed = True
        rows.clear();sources.clear()
    def close():
        if not closed:
            raise GeometryError('boundary sampling arithmetic cleanup is scope-owned')
    functions = access,guard,close
    # The live state exposes no trusted cleanup callable. Its compatibility
    # close hook is pinned, but unwinding calls the separately held closure.
    return (*functions,tuple(_chord_function_pin(f) for f in functions)),cleanup


def _sampling_arithmetic_producer(builder):
    namespace = object()
    def start(model,regions,check):
        from .native_support_snapshots import capture_native_supports
        from .arrangement_geometry import freeze_edge
        from .serialization import _qualified_model_state
        variable,issuer = _sampling_arithmetic_variable()
        state = variable.get()
        if state is not issuer.get():
            raise GeometryError('boundary sampling operation arithmetic issuer changed')
        context = _sampling_arithmetic_context()
        if state is not None:
            state[1]()
        checksum,region_signature = _current(model,regions,model.revision,None)
        epoch = model.revision,checksum,capture_native_supports(model)
        references,curves = None,{}
        def live():
            if (variable.get() is not state or issuer.get() is not state
                    or _sampling_arithmetic_context()!=context):
                raise GeometryError('boundary sampling arithmetic producer context changed')
            if state is not None:
                if tuple(_chord_function_pin(f) for f in state[:3]) != state[3]:
                    raise GeometryError('boundary sampling arithmetic producer calculator changed')
                state[1]()
            # This is an unpublished numerical read window, not receipt
            # qualification. Entry qualified native/runtime state; every actual
            # ordinary external callback is bracketed by _pinned_callback before
            # subsequent math, and public final issuance requalifies it fully.
            # Fingerprint persisted definitions/incidence here without repeatedly
            # walking every evaluated model kernel on each arithmetic interval.
            if (model.revision!=epoch[0]
                    or definition_checksum(regions)!=region_signature):
                raise GeometryError('boundary sampling arithmetic producer source changed')
            _qualified_model_state(model,expected_checksum=epoch[1],
                changed_message='boundary sampling arithmetic producer source changed')
        def evaluate(edge_id,a,b,endpoints,error,measure):
            nonlocal references
            if type(edge_id) is not int or type(measure) is not str or measure not in (
                    'geometric','parameterized','cell_reference'):
                raise GeometryError('boundary sampling arithmetic operands are unsupported')
            a,b=Q(a),Q(b)
            if not 0<=a<b<=1 or Q(float(a))!=a or Q(float(b))!=b:
                raise GeometryError('boundary sampling arithmetic stations are not canonical')
            if (type(endpoints) is not tuple or len(endpoints)!=2 or any(
                    type(row) is not tuple or len(row)!=3 or any(type(x) is not float or
                    not math.isfinite(x) for x in row) for row in endpoints)):
                raise GeometryError('boundary sampling arithmetic endpoints are not literal')
            error=_positive_error(error,'boundary arithmetic error')
            # Identity comes ONLY from the same validated operands subsequently
            # passed to the closed owner builder. No detached key/epoch, alternate
            # source reference, curve or proof-family argument can seed a row.
            key=region_signature,measure,edge_id,a,b,endpoints
            check();live()
            def calculate():
                nonlocal references
                if references is None:
                    references=_sampling_references(model,regions)
                if edge_id not in curves:
                    curves[edge_id]=freeze_edge(model,edge_id)
                def frontier():
                    check()
                return builder(model,references,edge_id,a,b,endpoints,error,frontier,curves[edge_id],measure)
            value=(calculate() if state is None else
                state[0](namespace,model,epoch,key,Q(error),calculate))
            check();live()
            return value
        return evaluate
    return start


def _sampling_bound_arithmetic(model,references,edge_id,a,b,endpoints,error,check,curve,measure):
    if measure == 'cell_reference':
        return _cell_reference_world_chord(model,references,edge_id,a,b,endpoints,error,check)
    prove = _world_chord if measure == 'geometric' else _parameterized_world_chord
    return prove(curve,a,b,endpoints,error,check)


_sampling_terminal_arithmetic = _sampling_arithmetic_producer(_sampling_bound_arithmetic)


class _SamplingWork:
    """Per-query accounting charged simultaneously to an existing owner scope."""
    def __init__(self, local, parent=None):
        self.local, self.parent = local, parent
        self._objects = (local, parent)
        self._parent_counts = None if parent is None else (parent.boxes, parent.subdivisions)
        self._limits = (local.max_boxes, local.max_depth, local.max_newton_iterations)
        self._parent_limits = None if parent is None else (
            parent.max_boxes, parent.max_depth, getattr(parent, 'max_newton_iterations', None))
        self.max_boxes = local.max_boxes
        self.max_depth = min(local.max_depth, getattr(parent, 'max_depth', local.max_depth))
        self.max_newton_iterations = min(local.max_newton_iterations,
            getattr(parent, 'max_newton_iterations', local.max_newton_iterations))
        self._effective = (self.max_boxes, self.max_depth, self.max_newton_iterations)
        self._counts = (local.boxes, local.subdivisions)
        self._take = local.take
        self._parent_take = None if parent is None else parent.take
        self._take_pins = tuple(_chord_function_pin(getattr(f,'__func__',f))
            for f in (self._take,self._parent_take))

    def guard(self):
        from .section_audit import Work
        original_parent = self._objects[1]
        if type(original_parent) is Work:
            counters = (original_parent.boxes, original_parent.subdivisions)
            if any(type(value) is not int or value < before
                   for value,before in zip(counters,self._parent_counts)):
                # Refusal must not erase already observed consumption. Restore
                # only this exact raw owner's known lower-bound watermark.
                restored = tuple(max(value,before) if type(value) is int else before
                                 for value,before in zip(counters,self._parent_counts))
                original_parent.boxes, original_parent.subdivisions = restored
                raise GeometryError('boundary sampling parent work counters decreased; consumed work restored')
            self._parent_counts = counters
        if self.local is not self._objects[0] or self.parent is not self._objects[1]:
            raise GeometryError('boundary sampling work identity changed')
        current = (self.local.take, None if self.parent is None else self.parent.take)
        for expected, effective, captured in zip(self._objects,current,(self._take,self._parent_take)):
            if expected is not None and (getattr(effective,'__self__',None) is not expected
                    or getattr(captured,'__self__',None) is not expected):
                raise GeometryError('boundary sampling work receiver changed')
        if tuple(_chord_function_pin(getattr(f,'__func__',f)) for f in current) != self._take_pins:
            raise GeometryError('boundary sampling work calculator changed')
        if (self.local.boxes, self.local.subdivisions) != self._counts:
            raise GeometryError('boundary sampling work counters changed')
        if ((self.local.max_boxes, self.local.max_depth, self.local.max_newton_iterations) != self._limits
                or (self.max_boxes, self.max_depth, self.max_newton_iterations) != self._effective):
            raise GeometryError('boundary sampling work limits changed')
        if self.parent is not None:
            counters = (self.parent.boxes, self.parent.subdivisions)
            if any(type(value) is not int or value < before
                   for value,before in zip(counters,self._parent_counts)):
                raise GeometryError('boundary sampling parent work counters decreased')
            self._parent_counts = counters
            if (self.parent.max_boxes, self.parent.max_depth,
                    getattr(self.parent, 'max_newton_iterations', None)) != self._parent_limits:
                raise GeometryError('boundary sampling parent work limits changed')
            if type(self.parent) is _SamplingWork:
                self.parent.guard()

    @property
    def boxes(self): return self.local.boxes
    @property
    def subdivisions(self): return self.local.subdivisions
    @subdivisions.setter
    def subdivisions(self, value):
        self.guard()
        delta = value-self.local.subdivisions
        if delta < 0: raise GeometryError('boundary sampling work counter decreased')
        self.local.subdivisions = value
        self._counts = (self.local.boxes, self.local.subdivisions)
        if self.parent is not None: self.parent.subdivisions += delta
        self.guard()  # retain the post-charge parent watermark before callbacks

    def take(self):
        self.guard()
        local = self._take()
        self._counts = (self.local.boxes, self.local.subdivisions)
        parent = True if self._parent_take is None else self._parent_take()
        self.guard()  # a later callback cannot undo this parent charge
        return local and parent


def _sampling_worker(policy):
    from .section_audit import Work
    from .plane_nurbs import CURRENT_WORK
    local = Work(policy.max_boxes_per_pair, policy.max_subdivision_depth)
    local.max_newton_iterations = policy.max_newton_iterations
    return _SamplingWork(local, CURRENT_WORK.get())


def _sampling_policy(qualification):
    from .predicates import IntersectionQualificationPolicy, DEFAULT_INTERSECTION_QUALIFICATION_POLICY
    policy = DEFAULT_INTERSECTION_QUALIFICATION_POLICY if qualification is None else qualification
    if type(policy) is not IntersectionQualificationPolicy:
        raise GeometryError('boundary sampling qualification requires IntersectionQualificationPolicy')
    return IntersectionQualificationPolicy(**{f.name: getattr(policy, f.name) for f in fields(policy)})


from contextlib import contextmanager as _sampling_contextmanager

@_sampling_contextmanager
def prepared_material_boundary_sampling_operation(*, qualification=None):
    """One bounded owner allowance for a complete shared station operation.

    Nested queries retain their own policy limits and local diagnostics, while
    every box and subdivision also charges this scope and any enclosing scope.
    No receipt or proof authority survives this context.
    """
    from .plane_nurbs import CURRENT_WORK
    work = _sampling_worker(_sampling_policy(qualification))
    guard = work.guard
    pin = _chord_function_pin
    def runtime():
        from .section_audit import Work
        return tuple(pin(f) for f in (_SamplingWork, _SamplingWork.__init__,
            _SamplingWork.guard, _SamplingWork.take, _SamplingWork.subdivisions.fset,
            _sampling_worker, Work.take))
    context = runtime()
    arithmetic,arithmetic_cleanup = _new_sampling_arithmetic()
    arithmetic_variable,arithmetic_issuer = _sampling_arithmetic_variable()
    issuer_token = arithmetic_issuer.set(arithmetic)
    arithmetic_token = arithmetic_variable.set(arithmetic)
    token = CURRENT_WORK.set(work)
    failure = None
    try:
        yield work.max_depth
        if CURRENT_WORK.get() is not work:
            raise GeometryError('boundary sampling operation work context changed')
        if runtime() != context:
            raise GeometryError('boundary sampling operation runtime changed')
        guard()
        if (_sampling_arithmetic_variable() != (arithmetic_variable,arithmetic_issuer)
                or arithmetic_variable.get() is not arithmetic or arithmetic_issuer.get() is not arithmetic
                or tuple(pin(f) for f in arithmetic[:3]) != arithmetic[3]):
            raise GeometryError('boundary sampling operation arithmetic identity or calculator changed')
        arithmetic[1]()
    except BaseException as error:
        failure = error
        try:
            guard()
        except BaseException:
            pass  # cleanup/guard failures must not replace the original caller failure
        raise
    finally:
        cleanup_errors = []
        # Each reset is attempted even if an earlier cleanup/reset fails. Never
        # call the exposed mutable close hook while preserving a caller error.
        for cleanup,args in ((arithmetic_cleanup,()),
                (arithmetic_variable.reset,(arithmetic_token,)),
                (arithmetic_issuer.reset,(issuer_token,)),(CURRENT_WORK.reset,(token,))):
            try:
                cleanup(*args)
            except BaseException as error:
                cleanup_errors.append(error)
        if failure is None and cleanup_errors:
            raise cleanup_errors[0]


def query_prepared_material_boundary_sampling(model, regions, edge_parameters, *,
        chord_error, expected_revision=None, cancellation_check=None, qualification=None, local_approximation=None,
        error_measure="geometric", proposal_fraction=1.):
    """Refine actual FE stations, preserving every requested source station.

    Only a certified world-error violation triggers FE refinement. Inconclusive
    predicates, resource exhaustion, unsupported geometry and cancellation
    propagate as refusals, never as a request to try smaller arbitrary pieces.
    """
    from .predicates import IntersectionQualificationPolicy, DEFAULT_INTERSECTION_QUALIFICATION_POLICY
    from .plane_nurbs import CURRENT_WORK
    from .section_audit import Work
    if type(error_measure) is not str or error_measure not in ("geometric", "parameterized", "cell_reference"):
        raise GeometryError("unknown boundary sampling error measure")
    if type(proposal_fraction) not in (float,int) or not math.isfinite(proposal_fraction) or not 0 < proposal_fraction <= 1:
        raise GeometryError("boundary proposal fraction must be finite in (0,1]")
    policy = DEFAULT_INTERSECTION_QUALIFICATION_POLICY if qualification is None else qualification
    if type(policy) is not IntersectionQualificationPolicy:
        raise GeometryError('boundary sampling qualification requires IntersectionQualificationPolicy')
    policy = IntersectionQualificationPolicy(**{f.name: getattr(policy, f.name) for f in fields(policy)})
    settings = tuple((f.name, getattr(policy, f.name)) for f in fields(policy))
    limits = dict(settings)
    work = _sampling_worker(policy)
    token = CURRENT_WORK.set(work)
    try:
        return _query_boundary_sampling(model, regions, edge_parameters, chord_error, expected_revision,
            cancellation_check, settings, work, limits['max_trace_segments'],local_approximation, error_measure, proposal_fraction)
    except BaseException:
        # Preserve a caller's original failure while retaining known consumed
        # parent work even if that caller reset counters before raising.
        try:
            work.guard()
        except BaseException:
            pass  # retain the original caller failure even if cleanup validation fails
        raise
    finally:
        CURRENT_WORK.reset(token)


def _query_boundary_sampling(model, regions, edge_parameters, chord_error, expected_revision,
        cancellation_check, settings, work, segment_limit,local_approximation=None, error_measure="geometric", proposal_fraction=1.):
    from .native_support_snapshots import capture_native_supports
    from .arrangement_geometry import freeze_edge
    from .plane_nurbs import charge_split
    error = _positive_error(chord_error, 'chord_error')
    try:
        inputs = tuple((handle, tuple(Q(t) for t in parameters)) for handle, parameters in edge_parameters)
    except (TypeError, ValueError, OverflowError) as cause:
        raise GeometryError('boundary sampling requires finite source parameter sequences') from cause
    if not inputs:
        raise GeometryError('boundary sampling requires at least one source edge')
    input_signature = _literal(inputs)
    callback = cancellation_check
    pin = _chord_function_pin
    def numerical_context():
        return tuple(pin(f) for f in (_parameterized_world_chord,
            _cos_c2_world_chord_bound,_single_c2_world_chord_bound,_charged_bounds,
            _world_chord,_canonical_cos,_chord_function_pin,
            _sampling_references,_cell_reference_world_chord, _cell_reference_is_world_affine,
            _query_boundary_chords, _cell_reference_calculator_context,
            _sampling_bound_arithmetic,_sampling_terminal_arithmetic,_sampling_arithmetic_context,
            _new_sampling_arithmetic,_sampling_arithmetic_producer,_sampling_arithmetic_variable))
    context = numerical_context()
    calculator_context = _cell_reference_calculator_context()
    arithmetic_variable,arithmetic_issuer = _sampling_arithmetic_variable()
    arithmetic = arithmetic_variable.get()
    if arithmetic is not arithmetic_issuer.get():
        raise GeometryError('boundary sampling operation arithmetic issuer changed')
    from .plane_nurbs import CURRENT_WORK
    work_guard = work.guard
    work_context = tuple(pin(f) for f in (_SamplingWork, _SamplingWork.guard,
        _SamplingWork.take, _SamplingWork.subdivisions.fset, _sampling_worker))
    def context_guard():
        work_guard()
        if (_sampling_arithmetic_variable() != (arithmetic_variable,arithmetic_issuer)
                or arithmetic_variable.get() is not arithmetic or arithmetic_issuer.get() is not arithmetic):
            raise GeometryError('boundary sampling operation arithmetic identity changed')
        if arithmetic is not None:
            if tuple(pin(f) for f in arithmetic[:3]) != arithmetic[3]:
                raise GeometryError('boundary sampling operation arithmetic calculator changed')
            arithmetic[1]()
        if CURRENT_WORK.get() is not work or tuple(pin(f) for f in (
                _SamplingWork, _SamplingWork.guard, _SamplingWork.take,
                _SamplingWork.subdivisions.fset, _sampling_worker)) != work_context:
            raise GeometryError('boundary sampling work context changed')
        if (numerical_context() != context
                or _cell_reference_calculator_context() != calculator_context):
            raise GeometryError('boundary sampling numerical context changed')
    from .cancellation import is_cancellation_signal
    if callback is not None and not is_cancellation_signal(callback):
        original_callback=callback
        def callback(*args):
            context_guard()
            result=original_callback(*args)
            context_guard()
            return result
    # Calculator guards bracket the ACTUAL external code before the generic
    # model pin reads any mutable calculator during its own post-callback check.
    callback = _pinned_callback(model, regions, callback, (inputs,))

    checksum, region_signature = _current(model, regions, expected_revision, callback)
    revision, native = model.revision, capture_native_supports(model)
    local_query=None
    if local_approximation is not None:
        from .local_approximation_field import prepare_local_boundary_limit_query,validate_local_approximation_allocation
        local_query=prepare_local_boundary_limit_query(model,local_approximation)
    from .cancellation import is_cancellation_signal
    ordinary=callback is not None and not is_cancellation_signal(callback)
    def check():
        context_guard()
        _check(callback, 'curvilinear boundary station refinement')
        context_guard()
        if local_approximation is not None and ordinary:
            validate_local_approximation_allocation(model,local_approximation)
    references = _sampling_references(model,regions)
    terminal_arithmetic = _sampling_terminal_arithmetic(model,regions,check)
    frozen_curves, terminal_bounds, affine_edges = {}, {}, {}
    rows, intervals, seen = [], [], set()
    for live_handle, parameters in inputs:
        check()
        if (type(live_handle) is not EntityHandle or live_handle.model_id != model.model_id
                or live_handle.kind != 'edge' or live_handle.id in seen):
            raise GeometryError('boundary sampling requires unique model-bound edge handles')
        handle = _detached(live_handle)
        model.handle('edge', handle.id)
        seen.add(handle.id)
        if (len(parameters) < 2 or any(not 0 <= t <= 1 or Q(float(t)) != t for t in parameters)
                or not (all(a < b for a, b in zip(parameters, parameters[1:]))
                        or all(a > b for a, b in zip(parameters, parameters[1:])))):
            raise GeometryError('boundary seeds must be strictly ordered exact binary64 stations in [0,1]')
        increasing = parameters[0] < parameters[-1]
        ordered = parameters if increasing else parameters[::-1]
        curve, edge = freeze_edge(model, handle.id), model.edges[handle.id]
        frozen_curves[handle.id] = curve
        if local_approximation is not None:
            from .local_approximation_field import local_boundary_corner_parameters
            ordered=tuple(sorted(set(ordered).union(t for t in local_boundary_corner_parameters(curve)
                if ordered[0]<t<ordered[-1])))
        def target(t):
            return tuple(float(x) for x in (model.vertex_position(edge.start) if t == 0 else
                model.vertex_position(edge.end) if t == 1 else curve.evaluate(float(t))))
        accepted = []
        pending = [(a, b, 0) for a, b in reversed(tuple(zip(ordered, ordered[1:])))]
        while pending:
            check()
            a, b, depth = pending.pop()
            interval_error=error
            if local_query is not None:
                from .nurbs import _float_down
                limit,_=local_query(handle.id,a,b)
                interval_error=_positive_error(min(error,_float_down(limit)),'local interval sampling error')
            from .nurbs import _float_down
            fraction = proposal_fraction
            if (error_measure == 'cell_reference' and proposal_fraction != 1.
                    and (handle.id not in affine_edges)):
                # Whole requested span: a sufficient affine result applies to every
                # dyadic subinterval. Failure simply retains the stricter proposal.
                affine_edges[handle.id] = _cell_reference_is_world_affine(
                    model,references,handle.id,ordered[0],ordered[-1],check)
            if error_measure == 'cell_reference' and affine_edges.get(handle.id, False):
                fraction = 1.
            interval_error = _positive_error(_float_down(Q(interval_error)*Q(fraction)), "boundary proposal error")
            try:
                endpoints = target(a),target(b)
                check()  # fresh cancellation, Work, calculator and local-allocation frontier on hits
                bound,pieces = terminal_arithmetic(handle.id,a,b,endpoints,interval_error,error_measure)
                check()
            except _WorldChordNeedsRefinement as need:
                mid = need.parameter
                if (not a < mid < b or Q(float(mid)) != mid
                        or depth+need.depth > work.max_depth):
                    raise GeometryError('boundary sampling cannot separate another canonical dyadic station') from need
                if len(intervals)+len(accepted)+len(pending)+2 > segment_limit:
                    raise GeometryError('boundary sampling trace-segment budget exhausted') from need
                charge_split()
                pending.extend(((mid, b, depth+need.depth), (a, mid, depth+need.depth)))
                continue
            terminal_bounds[(handle.id,a,b)] = (Q(bound),pieces,Q(interval_error),(target(a),target(b)))
            accepted.append((a, b))
            if len(intervals)+len(accepted)+len(pending) > segment_limit:
                raise GeometryError('boundary sampling trace-segment budget exhausted')
        stations = tuple(sorted({t for pair in accepted for t in pair}))
        if not increasing:
            stations = stations[::-1]
        if not set(parameters).issubset(stations):
            raise GeometryError('boundary sampling lost a requested station')
        rows.append((handle, stations))
        intervals.extend((handle, a, b) for a, b in zip(stations, stations[1:]))
    chords = _query_boundary_chords(model, regions, tuple(intervals),
        expected_revision=revision, chord_error=error, cancellation_check=callback,local_approximation=local_approximation,
        references=references, frozen_curves=frozen_curves, terminal_bounds=terminal_bounds)
    _check(callback, 'curvilinear boundary sampling complete')
    if (model.revision != revision or _checksum(model) != checksum
            or capture_native_supports(model) != native or definition_checksum(regions) != region_signature
            or _literal(inputs) != input_signature):
        raise GeometryError('boundary sampling source or requested stations changed during qualification')
    validate_prepared_material_boundary_chords_binding(model, regions, chords, expected_revision=revision)
    context_guard()
    result = PreparedMaterialBoundarySampling(model.model_id, revision, checksum, tuple(rows), chords,
        error, settings, (work.boxes, work.subdivisions, len(intervals)), error_measure, float(proposal_fraction))
    _SAMPLINGS[result] = (ref(model), region_signature, native, _literal(result))
    return result


def validate_prepared_material_boundary_sampling_binding(model, regions, result, *,
        expected_revision=None, cancellation_check=None):
    from .native_support_snapshots import capture_native_supports
    if type(result) is not PreparedMaterialBoundarySampling or result not in _SAMPLINGS:
        raise GeometryError('boundary sampling receipt was not issued by this owner')
    callback = _pinned_callback(model, regions, cancellation_check, (result,))
    checksum, region_signature = _current(model, regions, expected_revision, callback)
    source, bound_regions, native, literal = _SAMPLINGS[result]
    if (source() is not model or result.revision != model.revision or result.source_checksum != checksum
            or region_signature != bound_regions or capture_native_supports(model) != native
            or _literal(result) != literal):
        raise GeometryError('boundary sampling receipt is stale or changed')
    validate_prepared_material_boundary_chords_binding(model, regions, result.chords, expected_revision=model.revision)


def validate_prepared_material_curvilinear_cells(model, regions, chords, cells, *,
        expected_revision=None, surface_error, cancellation_check=None, previous=None, local_approximation=None):
    """Qualify literal curvilinear cells against an authenticated chord receipt.

    ``previous``: an owner receipt this function issued earlier. It is authenticated against the live
    registry and complete source binding before any caller callback can run. Its proof is reused only when
    the supplied chord receipt IS the object it was issued for, the literal cells and the error setting
    equal the issued ones; every other case runs the full proof. Reuse passes the same callbacks, pinning
    and final no-callback gate as a fresh proof and issues a new receipt; ``previous`` gains no authority.
    """
    error = _positive_error(surface_error, 'surface_error')
    live_rows = tuple(cells)
    local_fingerprint,local_limits=(),()
    if local_approximation is not None:
        _LOCAL_RESOLUTION_FAILURES.pop(local_approximation, None)
        from .local_approximation_field import validate_local_approximation_allocation
        local_fingerprint,local_limits=validate_local_approximation_allocation(model,local_approximation,cancellation_check=cancellation_check)
        if chords.local_field_fingerprint!=local_fingerprint:raise GeometryError('cell and chord local fields differ')
    elif chords.local_field_fingerprint:
        raise GeometryError('local chords require their owner local allocation')
    limits=dict(local_limits)
    if local_limits and any(cell.face.id not in limits for cell in live_rows):raise GeometryError('cell missing local field allocation')
    local_errors, local_footprints = (), ()
    local_query = None
    if local_approximation is not None:
        from .local_approximation_field import prepare_local_cell_limit_query
        local_query = prepare_local_cell_limit_query(model, local_approximation)
    input_signature, chord_signature = _literal(live_rows), _literal(chords)
    reuse, previous_signature = False, None
    if previous is not None:
        # no callback exists yet: a forged, copied, stale or foreign receipt is refused here
        validate_prepared_material_curvilinear_cells_binding(model, regions, previous,
                                                             expected_revision=expected_revision)
        previous_signature = _literal(previous)
        reuse = (_CELLS[previous][2] is chords and input_signature == _literal(previous.cells)
                 and _literal(error) == _literal(previous.surface_error)
                 and previous.local_field_fingerprint==local_fingerprint)
    rows = _detached(live_rows)
    operands = (live_rows, chords) if previous is None else (live_rows, chords, previous)
    from .cancellation import is_cancellation_signal
    has_callback = cancellation_check is not None and not is_cancellation_signal(cancellation_check)
    cancellation_check = _pinned_callback(model, regions, cancellation_check, operands)
    checksum, region_signature = _current(model, regions, expected_revision, cancellation_check)
    validate_prepared_material_boundary_chords_binding(model, regions, chords,
        expected_revision=expected_revision, cancellation_check=cancellation_check)
    revision = model.revision
    if reuse:
        local_errors, local_footprints = previous.surface_errors_by_cell, previous.local_footprints_by_cell
        _check(cancellation_check, 'curvilinear source-cell receipt reuse')
        nodes, reference_uv, bounds, boundaries = (previous.reference_nodes, previous.reference_uv_by_cell,
                                                   previous.source_map_bounds, previous.boundary_correspondence)
    else:
        from .source_cell_complex import certify_source_cells
        allocation_records = {}
        callback_failures = []
        def proof_check():
            try:
                _check(cancellation_check, 'curvilinear source-cell qualification')
            except BaseException as failure:
                callback_failures.append(failure)
                raise
            if local_approximation is not None and has_callback:
                validate_local_approximation_allocation(model, local_approximation)
        try:
            from .curvilinear_reference_maps import _closed_fan_steps
            with _closed_fan_steps(cancellation_check) as closed_check:
                nodes, reference_uv, bounds, boundaries = certify_source_cells(model, regions, chords, rows, Q(error),
                    closed_check if closed_check is not None else proof_check,
                    face_error_limits=local_limits, local_limit_query=local_query,
                    local_allocation_records=allocation_records)
        except Exception as failure:
            from .errors import SourceCellRefinementRequired, SourceCellApproximationInconclusive
            if (local_approximation is not None and
                    type(failure) in (SourceCellRefinementRequired, SourceCellApproximationInconclusive)
                    and not any(failure is item for item in callback_failures)):
                key = (failure.face_id, failure.node_ids)
                if key in allocation_records:
                    validate_local_approximation_allocation(model, local_approximation)
                    _LOCAL_RESOLUTION_FAILURES[local_approximation] = (
                        ref(failure), _literal(tuple(sorted(vars(failure).items()))), input_signature, ref(chords),
                        allocation_records[key][0])
            raise
        if local_query is not None:
            ordered = tuple(allocation_records[(cell.face.id, tuple(cell.node_ids))] for cell in rows)
            local_errors = tuple(row[0] for row in ordered)
            local_footprints = tuple(row[1] for row in ordered)
    _check(cancellation_check, 'curvilinear source-cell qualification complete')
    # No user callback follows this final complete definition/input gate.
    from .nurbs_runtime_binding import validate_model_curve_runtimes
    validate_model_curve_runtimes(model)
    if (model.revision != revision or _checksum(model) != checksum
            or definition_checksum(regions) != region_signature
            or _literal(live_rows) != input_signature or _literal(rows) != input_signature
            or _literal(chords) != chord_signature
            or (previous is not None and _literal(previous) != previous_signature)):
        raise GeometryError('curvilinear source or literal cell input changed during qualification')
    validate_prepared_material_boundary_chords_binding(model, regions, chords, expected_revision=revision)
    if previous is not None:
        validate_prepared_material_curvilinear_cells_binding(model, regions, previous, expected_revision=revision)
    result = PreparedMaterialCurvilinearCells(model.model_id, revision, checksum, rows, error,
                                              nodes, reference_uv, bounds, boundaries, local_fingerprint, local_errors, local_footprints)
    if local_approximation is not None:
        validate_local_approximation_allocation(model,local_approximation)
        _LOCAL_CELL_ALLOCATIONS[result]=local_approximation
    _CELLS[result] = (ref(model), region_signature, chords, _literal(result), regions)
    return result


def validate_prepared_material_curvilinear_cells_binding(model, regions, result, *,
        expected_revision=None, cancellation_check=None):
    if type(result) is not PreparedMaterialCurvilinearCells or result not in _CELLS:
        raise GeometryError('curvilinear cell receipt was not issued by this owner')
    source, bound_regions, chords, literal, _bound_region_object = _CELLS[result]
    cancellation_check = _pinned_callback(model, regions, cancellation_check, (result, chords))
    checksum, region_signature = _current(model, regions, expected_revision, cancellation_check)
    if (source() is not model or result.revision != model.revision or result.source_checksum != checksum
            or region_signature != bound_regions or _literal(result) != literal):
        raise GeometryError('curvilinear cell receipt is stale or changed')
    if result.local_field_fingerprint:
        from .local_approximation_field import validate_local_approximation_allocation
        fingerprint,_=validate_local_approximation_allocation(model,_LOCAL_CELL_ALLOCATIONS.get(result))
        if fingerprint!=result.local_field_fingerprint:raise GeometryError('local cell field binding changed')
    validate_prepared_material_boundary_chords_binding(model, regions, chords, expected_revision=model.revision)


def _source_transfer_cell_inputs(model, result):
    """Return authenticated operands, never caller-inferred region ownership."""
    if type(result) is not PreparedMaterialCurvilinearCells or result not in _CELLS:
        raise GeometryError('source transfer requires a live owner cell receipt')
    source, _signature, chords, _literal_signature, regions = _CELLS[result]
    if source() is not model:
        raise GeometryError('source transfer cell receipt belongs to another model')
    validate_prepared_material_curvilinear_cells_binding(model, regions, result)
    return regions, chords


def _local_resolution_failure_limit(model, allocation, failure, cells, chords):
    """Authenticate the last local failure from this live owner operation."""
    from .local_approximation_field import validate_local_approximation_allocation
    validate_local_approximation_allocation(model, allocation)
    record = _LOCAL_RESOLUTION_FAILURES.get(allocation)
    if (record is None or record[0]() is not failure or record[1] != _literal(tuple(sorted(vars(failure).items())))
            or record[2] != _literal(tuple(cells)) or record[3]() is not chords):
        raise GeometryError('local resolution failure has no current owner allocation')
    return record[4]
