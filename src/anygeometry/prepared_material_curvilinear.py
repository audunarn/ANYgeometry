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
        return tuple(_literal(v) for v in value)
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
            if (model.revision != revision or _checksum(model) != checksum
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


def _world_chord(curve, lo, hi, endpoints, error, check):
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


def _uv_box(support, curve, parameter):
    if isinstance(support, NurbsSurface):
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
        expected_revision=None, chord_error, cancellation_check=None):
    from .nurbs_metric import certify_projected_chart
    from .nurbs_domain import _exact_segment_distance, ChartContainmentUndecided
    error = _positive_error(chord_error, 'chord_error')
    input_rows = tuple(edge_intervals)
    input_signature = _literal(tuple((handle, Q(a), Q(b)) for handle, a, b in input_rows))
    cancellation_check = _pinned_callback(model, regions, cancellation_check, (input_rows,))
    checksum, region_signature = _current(model, regions, expected_revision, cancellation_check)
    revision = model.revision
    check = lambda: _check(cancellation_check, 'curvilinear boundary qualification')
    source_faces = {chart.face.id: chart for region in regions.regions for chart in region.sources}
    from .plane_source_reference import PlaneReference
    from .regular_source_reference import NurbsReference
    from .source_cell_complex import _support_owner
    plane_references = {}
    for face, chart in source_faces.items():
        owner = _support_owner(model, face)
        plane_references[face] = (PlaneReference(owner, owner.faces[face]) if isinstance(chart.support, Plane)
                                 else NurbsReference(owner, owner.faces[face]))
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
        curve = freeze_edge(model, handle.id)
        if not isinstance(curve, (CurveOnSurface, PlaneNurbsCurve, NurbsCurve, LinePath)):
            raise GeometryError('curvilinear boundary edge family is not yet qualified')
        edge = model.edges[handle.id]
        xyz = tuple(tuple(float(v) for v in (model.vertex_position(edge.start) if t == 0 else
                    model.vertex_position(edge.end) if t == 1 else curve.evaluate(float(t)))) for t in (a, b))
        for t, target in zip((a, b), xyz):
            if t in (0, 1):
                endpoint_error = max(sum(abs(x-Q(v)) for x, v in zip(point, target))
                    for point in _box_corners(_charged_bounds(curve, float(t), float(t))))
                extent = float(np.linalg.norm(model.vertex_position(edge.end)-model.vertex_position(edge.start))) or 1.
                if endpoint_error > Q(model.tolerance.effective_length(extent)):
                    raise GeometryError('owner endpoint correction exceeds the existing joint precision')
        world_error, pieces = _world_chord(curve, a, b, xyz, error, check)
        occurrences, chart_error = [], Q(0)
        for face_id in model.faces_using_edge(handle.id):
            if face_id not in source_faces:
                continue
            chart = source_faces[face_id]
            from .nurbs_sweep_chart import has_sweep_chart, chart_model
            occurrence_curve = (chart_model(model, face_id).edges[handle.id].curve
                if has_sweep_chart(model.faces[face_id]) else curve)
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
                rect = _uv_interval(chart.support, occurrence_curve, lo, hi)
                chart_error = max(chart_error, max(_exact_segment_distance(p, q0, q1)
                    for p in product(*rect)))
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
    result = PreparedMaterialBoundaryChords(model.model_id, revision, checksum, tuple(intervals), tuple(anchors), error)
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


def query_prepared_material_boundary_sampling(model, regions, edge_parameters, *,
        chord_error, expected_revision=None, cancellation_check=None, qualification=None):
    """Refine actual FE stations, preserving every requested source station.

    Only a certified world-error violation triggers FE refinement. Inconclusive
    predicates, resource exhaustion, unsupported geometry and cancellation
    propagate as refusals, never as a request to try smaller arbitrary pieces.
    """
    from .predicates import IntersectionQualificationPolicy, DEFAULT_INTERSECTION_QUALIFICATION_POLICY
    from .plane_nurbs import CURRENT_WORK
    from .section_audit import Work
    policy = DEFAULT_INTERSECTION_QUALIFICATION_POLICY if qualification is None else qualification
    if type(policy) is not IntersectionQualificationPolicy:
        raise GeometryError('boundary sampling qualification requires IntersectionQualificationPolicy')
    policy = IntersectionQualificationPolicy(**{f.name: getattr(policy, f.name) for f in fields(policy)})
    settings = tuple((f.name, getattr(policy, f.name)) for f in fields(policy))
    limits = dict(settings)
    work = Work(limits['max_boxes_per_pair'], limits['max_subdivision_depth'])
    work.max_newton_iterations = limits['max_newton_iterations']
    token = CURRENT_WORK.set(work)
    try:
        return _query_boundary_sampling(model, regions, edge_parameters, chord_error, expected_revision,
            cancellation_check, settings, work, limits['max_trace_segments'])
    finally:
        CURRENT_WORK.reset(token)


def _query_boundary_sampling(model, regions, edge_parameters, chord_error, expected_revision,
        cancellation_check, settings, work, segment_limit):
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
    callback = _pinned_callback(model, regions, cancellation_check, (inputs,))
    checksum, region_signature = _current(model, regions, expected_revision, callback)
    revision, native = model.revision, capture_native_supports(model)
    check = lambda: _check(callback, 'curvilinear boundary station refinement')
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
        def target(t):
            return tuple(float(x) for x in (model.vertex_position(edge.start) if t == 0 else
                model.vertex_position(edge.end) if t == 1 else curve.evaluate(float(t))))
        accepted = []
        pending = [(a, b, 0) for a, b in reversed(tuple(zip(ordered, ordered[1:])))]
        while pending:
            check()
            a, b, depth = pending.pop()
            try:
                _world_chord(curve, a, b, (target(a), target(b)), error, check)
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
    chords = query_prepared_material_boundary_chords(model, regions, tuple(intervals),
        expected_revision=revision, chord_error=error, cancellation_check=callback)
    _check(callback, 'curvilinear boundary sampling complete')
    if (model.revision != revision or _checksum(model) != checksum
            or capture_native_supports(model) != native or definition_checksum(regions) != region_signature
            or _literal(inputs) != input_signature):
        raise GeometryError('boundary sampling source or requested stations changed during qualification')
    validate_prepared_material_boundary_chords_binding(model, regions, chords, expected_revision=revision)
    result = PreparedMaterialBoundarySampling(model.model_id, revision, checksum, tuple(rows), chords,
        error, settings, (work.boxes, work.subdivisions, len(intervals)))
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
        expected_revision=None, surface_error, cancellation_check=None, previous=None):
    """Qualify literal curvilinear cells against an authenticated chord receipt.

    ``previous``: an owner receipt this function issued earlier. It is authenticated against the live
    registry and complete source binding before any caller callback can run. Its proof is reused only when
    the supplied chord receipt IS the object it was issued for, the literal cells and the error setting
    equal the issued ones; every other case runs the full proof. Reuse passes the same callbacks, pinning
    and final no-callback gate as a fresh proof and issues a new receipt; ``previous`` gains no authority.
    """
    error = _positive_error(surface_error, 'surface_error')
    live_rows = tuple(cells)
    input_signature, chord_signature = _literal(live_rows), _literal(chords)
    reuse, previous_signature = False, None
    if previous is not None:
        # no callback exists yet: a forged, copied, stale or foreign receipt is refused here
        validate_prepared_material_curvilinear_cells_binding(model, regions, previous,
                                                             expected_revision=expected_revision)
        previous_signature = _literal(previous)
        reuse = (_CELLS[previous][2] is chords and input_signature == _literal(previous.cells)
                 and _literal(error) == _literal(previous.surface_error))
    rows = _detached(live_rows)
    operands = (live_rows, chords) if previous is None else (live_rows, chords, previous)
    cancellation_check = _pinned_callback(model, regions, cancellation_check, operands)
    checksum, region_signature = _current(model, regions, expected_revision, cancellation_check)
    validate_prepared_material_boundary_chords_binding(model, regions, chords,
        expected_revision=expected_revision, cancellation_check=cancellation_check)
    revision = model.revision
    if reuse:
        _check(cancellation_check, 'curvilinear source-cell receipt reuse')
        nodes, reference_uv, bounds, boundaries = (previous.reference_nodes, previous.reference_uv_by_cell,
                                                   previous.source_map_bounds, previous.boundary_correspondence)
    else:
        from .source_cell_complex import certify_source_cells
        nodes, reference_uv, bounds, boundaries = certify_source_cells(model, regions, chords, rows, Q(error),
            lambda: _check(cancellation_check, 'curvilinear source-cell qualification'))
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
                                              nodes, reference_uv, bounds, boundaries)
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
