"""Bounded plate / regular NURBS section clipping by shared exact boundary-event stations (UNRUN definitions;
implemented-but-unverified).

Fixture: the paraboloid z = x**2 + y**2 cut by z = .5 is a circle of radius sqrt(.5) about the origin
(tests/test_plane_split.bowl); plates are authored polygons in the plane z = .5.
"""
from dataclasses import replace
from fractions import Fraction as Q

import pytest

from anygeometry import ConnectionIntent, GeometryError, IntersectionBatchPolicy
from anygeometry.bounded_cut_graph import PlateContactRefused, clip_path, interval_inside, plate_roots
from anygeometry.entities import EntityRef, OrientedEdge
from anygeometry.planar_exact import ExactPlane
from anygeometry.plane_section_plate import PlateBoundaryCrossingRequired, plate_loops
from anygeometry.serialization import from_dict, to_dict
from anygeometry.support_curves import PlaneNurbsCurve
from test_plane_split import bowl

AUX = 'anygeometry.auxiliary_partition'
RECT = [(.35, -1.), (1.5, -1.), (1.5, 1.), (.35, 1.)]
L_SHAPE = [(.35, -1.), (1.5, -1.), (1.5, 1.), (.9, 1.), (.9, .2), (.35, .2)]
CAP = [(.70, -.2), (1.5, -.2), (1.5, .2), (.70, .2)]          # x > .70 keeps a sliver of the circle (r = .7071)
NOOP = lambda *_: None


def plate_on(model, points):
    vertices = [model.add_point(x, y, .5) for x, y in points]
    return model.add_plate(vertices)


def fixture(points):
    model, face, plane = bowl()
    plate = plate_on(model, points)
    curve = PlaneNurbsCurve.from_plane(plane, model.faces[face].surface, 0)
    frame = ExactPlane.from_plane(model.faces[plate].surface)
    return model, face, plate, plane, curve, frame


def on_segment(model, plate, plane, curve, frame):
    return [r for r in plate_roots(model, plate, plane, frame, curve, 0, NOOP) if r.on_segment]


@pytest.mark.parametrize('gap,expected', [(Q(1, 2**80), True), (Q(0), False), (-Q(1, 100), False)])
def test_exact_affine_cut_exclusion_distinguishes_gap_double_root_and_crossing(gap, expected):
    from anygeometry.branch_predicates import _affine_quadratic_exclusion
    first = tuple((value,) for value in (Q(1, 4)+gap, -Q(1, 4)+gap, Q(1, 4)+gap))
    assert _affine_quadratic_exclusion(first, ((Q(0), Q(-1)), (Q(1), Q(0)))) is expected


def test_exact_affine_cut_exclusion_preserves_boundary_contacts_and_unsupported_cases():
    from anygeometry.branch_predicates import _affine_quadratic_exclusion as exclude
    diagonal = ((Q(0), Q(-1)), (Q(1), Q(0)))
    assert not exclude(((Q(0),), (Q(0),), (Q(1),)), diagonal)  # double boundary root
    corner = ((Q(0), Q(1)), (Q(1), Q(2)))
    assert not exclude(((Q(0),), (Q(1),)), corner)  # zero line touches box only at root
    assert exclude(((Q(1),), (Q(2),)), corner)  # the same closed endpoint is strictly nonzero
    assert not exclude(((Q(1),), (Q(2),), (Q(3),), (Q(5),)), diagonal)  # genuine cubic
    assert not exclude(((Q(1),),), ((Q(0), Q(-1)), (Q(1), Q(1, 10))))  # non-affine


def test_exact_affine_cut_exclusion_observes_cooperative_cancellation(monkeypatch):
    import anygeometry.branch_predicates as predicates
    error = GeometryError('cancel exact affine exclusion')
    def cancel():
        raise error
    monkeypatch.setattr(predicates, '_cooperative_tick', cancel)
    with pytest.raises(GeometryError) as caught:
        predicates._affine_quadratic_exclusion(((Q(1),),), ((Q(0), Q(-1)), (Q(1), Q(0))))
    assert caught.value is error


# ------------------------------------------------------------------------- exact plate roots

def test_plate_roots_are_isolated_on_segment_roots_of_each_exact_edge_line():
    model, face, plate, plane, curve, frame = fixture(RECT)
    on = on_segment(model, plate, plane, curve, frame)
    assert len(on) == 2 and len({r.plate_edge for r in on}) == 1
    assert on[0].isolator[1] < on[1].isolator[0]
    expected = sorted(((1 + y) / 2) for y in ((.5 - .35**2)**.5, -(.5 - .35**2)**.5))
    found = sorted(float((r.segment_interval[0] + r.segment_interval[1]) / 2) for r in on)
    for got, want in zip(found, expected):                          # supplemental sanity of the exact interval
        assert abs(got - want) < 1e-6
    assert all(0 < r.segment_interval[0] <= r.segment_interval[1] < 1 for r in on)


def test_no_on_segment_root_when_the_plate_contains_or_misses_the_whole_section():
    for points in ([(-2., -2.), (2., -2.), (2., 2.), (-2., 2.)], [(3., 3.), (4., 3.), (4., 4.), (3., 4.)]):
        model, face, plate, plane, curve, frame = fixture(points)
        assert not on_segment(model, plate, plane, curve, frame)


def test_section_through_a_plate_vertex_is_refused():
    # (.5, .5, .5) lies exactly on the circle x**2 + y**2 = .5 and is a plate vertex
    model, face, plate, plane, curve, frame = fixture([(.5, .5), (1.5, .5), (1.5, -1.), (.5, -1.)])
    with pytest.raises(PlateContactRefused):
        plate_roots(model, plate, plane, frame, curve, 0, NOOP)


def test_clip_path_alternates_and_the_wrap_intervals_agree_on_the_closed_circle():
    model, face, plate, plane, curve, frame = fixture(RECT)
    _frame, loops = plate_loops(model, model.faces[plate])
    decisions = clip_path(curve, frame, loops, on_segment(model, plate, plane, curve, frame), True, NOOP)
    assert len(decisions) == 3 and decisions[0][2] == decisions[2][2] != decisions[1][2]
    lo, hi = next((lo, hi) for lo, hi, flag in decisions if flag)
    assert float(curve.evaluate(float((lo + hi) / 2))[0]) > .35 - 1e-9          # the inside arc is the cap x > .35
    outside = next((lo, hi) for lo, hi, flag in decisions if not flag)
    assert interval_inside(curve, frame, loops, *outside, NOOP) is False


def _dyadic_near_tangent_fixture():
    from anygeometry.surfaces import Plane as PlaneSupport
    tiny = 2.0**-54
    model, face, plane = bowl()
    vertices = [model.add_point(x, y, .5) for x, y in ((1., 0.), (tiny, 1.), (-1., 1.), (-1., -1.), (1., -1.))]
    edges = [model.add_line(vertices[i], vertices[(i + 1) % 5]) for i in range(5)]
    plate = model.add_face(edges, surface=PlaneSupport((0., 0., .5), (1., 0., 0.), (0., 1., 0.)))
    curve = PlaneNurbsCurve.from_plane(plane, model.faces[face].surface, 0)
    frame = ExactPlane.from_plane(model.faces[plate].surface)
    return model, face, plate, plane, curve, frame


def test_exact_near_tangent_boundary_events_are_empty_without_geometry_change():
    model, _face, plate, plane, curve, frame = _dyadic_near_tangent_fixture()
    before = to_dict(model)
    assert not on_segment(model, plate, plane, curve, frame)
    assert to_dict(model) == before


def test_dyadic_edge_is_now_expressed_exactly_and_the_near_tangent_circle_is_never_clipped_by_rounding():
    """Exact dyadic plate edge: no rounding-created root; conservative public-plan refusal stays atomic."""
    model, face, plate, plane, curve, frame = _dyadic_near_tangent_fixture()
    assert not on_segment(model, plate, plane, curve, frame)         # no rounding-created root
    from anygeometry.bounded_nurbs import plan_bounded
    before = to_dict(model)
    handles = (model.handle('face', face), model.handle('face', plate))
    with pytest.raises(GeometryError):
        plan_bounded(model, handles, policy_for())
    assert to_dict(model) == before


def test_keep_hook_default_none_is_the_legacy_route_and_a_rejecting_hook_creates_no_trace():
    from anygeometry.plane_split import _stage_plane_split
    from anygeometry.predicates import DEFAULT_INTERSECTION_QUALIFICATION_POLICY as POLICY
    results = {}
    for name, keep in (('legacy', None), ('all', lambda i, lo, hi, start, end: True),
                       ('none', lambda i, lo, hi, start, end: False)):
        model, face, plane = bowl()
        staged = model.clone(preserve_identity=True)
        with staged.transaction():
            mappings, traces, positive, negative = _stage_plane_split(staged, (face,), plane, POLICY, keep=keep)
        results[name] = (len(mappings[0][1]), len(traces), len(staged.faces))
    assert results['legacy'] == results['all']
    assert results['none'] == (1, 0, 1)


# ------------------------------------------------------------- connected bounded route (plan/apply, UNRUN)

def policy_for(**extra):
    return IntersectionBatchPolicy(ConnectionIntent.IMPRINT, **extra)


def bounded_run(points, *, sheet=False, hole=None, **policy_extra):
    from anygeometry.bounded_nurbs import apply_bounded, plan_bounded
    model, face, plane = bowl()
    plate = plate_on(model, points)
    if hole:
        vertices = [model.add_point(x, y, .5) for x, y in hole]
        edges = [model.add_line(vertices[i], vertices[(i + 1) % len(hole)]) for i in range(len(hole))]
        with model.transaction():
            model._put_entity('face', replace(model.faces[plate],
                                              holes=(tuple(OrientedEdge(e, True) for e in edges),)))
    if sheet:
        model.add_sheet((face,))
        model.add_sheet((plate,))
    handles = (model.handle('face', face), model.handle('face', plate))
    policy = policy_for(**policy_extra)
    plan = plan_bounded(model, handles, policy)
    return model, face, plate, plan, apply_bounded(model, plan, policy), policy


def edge_kinds(model):
    traces, auxiliary = [], []
    for edge, item in model.edges.items():
        if AUX in model.tags_for(EntityRef('edge', edge)):
            auxiliary.append(edge)
        elif isinstance(item.curve, PlaneNurbsCurve):
            traces.append(edge)
    return sorted(traces), sorted(auxiliary)


def cell_area_intervals(model):
    """Certified world-area interval per horizontal plate cell (outer minus holes). ``loop_area`` stops at the
    first level whose SIGN is certified, so these are sound containment intervals, NOT accurate areas: the
    tests below only assert that exactly known/analytic values lie inside them, never a midpoint."""
    from anygeometry.plane_section_plate import loop_area
    rows = []
    for face in plate_faces(model):
        item = model.faces[face]
        frame = ExactPlane.from_plane(item.surface)
        lo = hi = Q(0)
        for index, loop in enumerate((item.loop, *item.holes)):
            low, high = loop_area(model, loop, frame, lambda: None)
            low, high = (abs(low), abs(high)) if low > 0 or high > 0 else (abs(high), abs(low))
            sign = 1 if index == 0 else -1
            lo, hi = (lo + low, hi + high) if sign > 0 else (lo - high, hi - low)
        # The owner integrates native affine UV area. These fixtures are
        # horizontal, so the world-area Jacobian is an exact rational.
        assert frame.normal[0] == frame.normal[1] == 0
        jacobian = abs(frame.normal[2])
        rows.append((lo * jacobian, hi * jacobian))
    return rows


def analytic_cap_area(c, r2):
    """Area of the disc x**2 + y**2 <= r2 on the side x >= c: r2 acos(c/r) - c sqrt(r2 - c**2) (reference only)."""
    import math
    r = math.sqrt(r2)
    return r2*math.acos(c/r) - c*math.sqrt(r2 - c*c)


def nurbs_faces(model):
    return [f for f, item in model.faces.items() if not hasattr(item.surface, 'normal')]


def plate_faces(model):
    return [f for f, item in model.faces.items() if hasattr(item.surface, 'normal')]


def test_area_reference_uses_world_units_on_a_span_scaled_plate_chart():
    model, *_ = fixture(RECT)
    expected = (Q(float(1.5)) - Q(float(.35))) * 2
    assert cell_area_intervals(model) == [(expected, expected)]


@pytest.mark.parametrize('points', (RECT, L_SHAPE))
def test_partial_crossing_clips_both_domains_with_physical_traces_and_nurbs_only_auxiliaries(points):
    model, face, plate, plan, result, policy = bounded_run(points)
    assert not model.validate_topology()
    traces, auxiliary = edge_kinds(model)
    assert traces and auxiliary
    assert len(nurbs_faces(model)) == 2                                    # the FULL original plane split of the parent
    assert len(plate_faces(model)) >= 2                                    # the plate is clipped by the inside blocks
    assert all(len(model.faces_using_edge(e)) == 4 for e in traces)        # two NURBS cells + two plate cells
    assert all(len(model.faces_using_edge(e)) == 2 for e in auxiliary)     # NURBS-only partition edges
    assert not set(traces) & set(auxiliary)
    plate_edges = {use.edge for f in plate_faces(model) for loop in (model.faces[f].loop, *model.faces[f].holes)
                   for use in loop}
    assert not plate_edges & set(auxiliary)                                # no auxiliary edge borders a plate cell
    assert set(traces) <= plate_edges                                      # every physical trace borders plate cells
    if points is RECT:
        intervals = cell_area_intervals(model)
        cap = Q(analytic_cap_area(.35, .5))
        total = (Q(float(1.5)) - Q(float(.35))) * 2                         # exact area of the authored rectangle
        assert sum(lo for lo, hi in intervals) <= total <= sum(hi for lo, hi in intervals)
        assert any(lo <= cap <= hi for lo, hi in intervals)                 # the analytic circular cap is one cell
        assert any(lo <= total - cap <= hi for lo, hi in intervals)         # and the rest is the other
    loaded = from_dict(to_dict(model))
    assert not loaded.validate_topology()
    assert edge_kinds(loaded) == edge_kinds(model)                          # persisted tags, same edge ids


def test_plate_with_a_hole_crossing_the_section_keeps_all_material():
    hole = [(.6, -.2), (.6, .2), (.9, .2), (.9, -.2)]
    model, face, plate, plan, result, policy = bounded_run([(-2., -2.), (2., -2.), (2., 2.), (-2., 2.)], hole=hole)
    assert not model.validate_topology()
    traces, auxiliary = edge_kinds(model)
    assert traces and auxiliary
    total = Q(16) - (Q(.9) - Q(.6)) * (Q(.2) + Q(.2))                        # exact area of plate minus hole (floats)
    intervals = cell_area_intervals(model)
    assert sum(lo for lo, hi in intervals) <= total <= sum(hi for lo, hi in intervals)


def test_old_fully_contained_route_is_unchanged_and_creates_no_auxiliary():
    model, face, plate, plan, result, policy = bounded_run([(-2., -2.), (2., -2.), (2., 2.), (-2., 2.)])
    traces, auxiliary = edge_kinds(model)
    assert traces and not auxiliary
    assert not model.validate_topology()


def test_disjoint_plate_changes_nothing():
    from anygeometry.bounded_nurbs import apply_bounded, plan_bounded
    model, face, plane = bowl()
    plate = plate_on(model, [(3., 3.), (4., 3.), (4., 4.), (3., 4.)])
    before = to_dict(model)
    policy = policy_for()
    handles = (model.handle('face', face), model.handle('face', plate))
    apply_bounded(model, plan_bounded(model, handles, policy), policy)
    assert to_dict(model) == before


def test_vertex_contact_and_cancellation_refuse_without_caller_change():
    from anygeometry.bounded_nurbs import plan_bounded
    model, face, plane = bowl()
    plate = plate_on(model, [(.5, .5), (1.5, .5), (1.5, -1.), (.5, -1.)])
    before = to_dict(model)
    handles = (model.handle('face', face), model.handle('face', plate))
    with pytest.raises(GeometryError):
        plan_bounded(model, handles, policy_for())
    assert to_dict(model) == before
    model, face, plane = bowl()
    plate = plate_on(model, RECT)
    before = to_dict(model)
    handles = (model.handle('face', face), model.handle('face', plate))
    with pytest.raises(GeometryError):
        plan_bounded(model, handles, policy_for(cancellation_check=lambda: True))
    assert to_dict(model) == before


def test_raw_edit_makes_a_bounded_plan_stale():
    from anygeometry.bounded_nurbs import apply_bounded, plan_bounded
    model, face, plane = bowl()
    plate = plate_on(model, RECT)
    handles = (model.handle('face', face), model.handle('face', plate))
    policy = policy_for()
    plan = plan_bounded(model, handles, policy)
    model._faces[face] = replace(model.faces[face], metadata={'raw': True})
    with pytest.raises(GeometryError, match='stale|changed'):
        apply_bounded(model, plan, policy)


@pytest.mark.parametrize('points', (RECT, CAP))
def test_sheet_owned_faces_keep_one_use_per_descendant_with_three_coedge_loops(points):
    """Strict (no skip). RECT keeps every existing arc joint; CAP clips a sliver inside one arc, so the physical
    block joining two stations of the same plate edge gets one more owned station before the plate cells exist."""
    model, face, plate, plan, result, policy = bounded_run(points, sheet=True)
    assert not model.validate_topology() and not model._validate_structural()
    assert sorted(use.face_id for use in model.face_uses.values()) == sorted(nurbs_faces(model) + plate_faces(model))
    assert all(len(loop) >= 3 for use in model.face_uses.values() for loop in use.loops)


# ------------------------------------------------------------- canonical exact stations and joints (route C)

def test_station_is_exactly_on_the_original_line_and_the_support_for_axis_aligned_plates():
    from anygeometry._canonical_plate_joint import choose_station, exact
    from anygeometry.bounded_cut_graph import plane_equation
    model, face, plate, plane, curve, frame = fixture(RECT)
    normal, offset = plane_equation(plane)
    for root in on_segment(model, plate, plane, curve, frame):
        item = model.edges[root.plate_edge]
        a, b = exact(model.vertex_position(item.start)), exact(model.vertex_position(item.end))
        q, lam = choose_station(a, b, normal, offset, root.lower, root.upper, model.tolerance.effective_length(1.))
        e = tuple(y - x for x, y in zip(a, b))
        d = tuple(Q(float(x)) - c for x, c in zip(q, a))
        cross = (d[1]*e[2] - d[2]*e[1], d[2]*e[0] - d[0]*e[2], d[0]*e[1] - d[1]*e[0])
        assert not any(cross) and sum(n * Q(float(x)) for n, x in zip(normal, q)) == offset      # exact line + support
        assert all(lo <= Q(float(x)) <= hi for x, lo, hi in zip(q, root.lower, root.upper)) and 0 < lam < 1


def test_unrepresentable_oblique_stations_are_refused_not_nudged():
    from anygeometry._canonical_plate_joint import JointRefused, choose_station, exact
    # Deterministic known-unrepresentable case: the line y = 3 x through (0, 0, .5) and (1, 3, .5). The enclosure
    # holds the single float x0 = float(.3) (an odd 53-bit numerator over 2**54), so y = 3 x0 needs 54 bits and
    # is NOT a float; no float of y lies inside the (width 3 * 2**-60) y range either.
    a, b = exact((0.0, 0.0, .5)), exact((1.0, 3.0, .5))
    x0 = Q(float(.3))
    lower, upper = (x0, 3*x0, Q(1, 2)), (x0 + Q(1, 2**60), 3*x0 + 3*Q(1, 2**60), Q(1, 2))
    with pytest.raises(JointRefused):
        choose_station(a, b, (Q(0), Q(0), Q(1)), Q(1, 2), lower, upper, 1e-9)
    # a plate edge whose endpoints are not exactly on the plate support is refused as well
    with pytest.raises(JointRefused):
        choose_station(exact((0., 0., .5)), exact((1., 0., .5 + 2.0**-40)), (Q(0), Q(0), Q(1)), Q(1, 2),
                       (Q(0), Q(0), Q(1, 2)), (Q(1, 2**60), Q(1, 2**60), Q(1, 2)), 1e-9)


def test_both_source_protocols_consume_the_same_canonical_joint_symbol():
    from anygeometry._canonical_plate_joint import plate_joint
    from anygeometry.plane_source_reference import PlaneReference
    from anygeometry.regular_source_reference import NurbsReference
    model, face, plate, plan, result, policy = bounded_run(RECT)
    traces, auxiliary = edge_kinds(model)
    ends = {}
    for edge in (*traces, *auxiliary):
        item = model.edges[edge]
        for vertex in (item.start, item.end):
            joint = plate_joint(model, vertex, edge)
            if joint is not None:
                ends.setdefault(vertex, set()).add(joint.identity)
    assert ends and all(len(identities) == 1 for identities in ends.values())     # one symbol per station vertex
    symbols = {next(iter(identities)) for identities in ends.values()}
    for plate_cell in plate_faces(model):
        reference = PlaneReference(model, model.faces[plate_cell])
        joints = {v: n for v, n in reference.vertices.items() if v in ends}
        assert joints and all(node.identity == ('plane_root', reference.key, next(iter(ends[v])))
                              for v, node in joints.items())
    for cell in nurbs_faces(model):
        reference = NurbsReference(model, model.faces[cell])
        joints = {v: n for v, n in reference.vertices.items() if v in ends}
        assert all(node.identity == next(iter(ends[v])) for v, node in joints.items())
    assert symbols


def test_native_side_and_derivatives_are_transported_from_the_canonical_joint_isolator():
    from anygeometry._canonical_plate_joint import plate_joint
    from anygeometry.plane_source_reference import PlaneReference
    from anygeometry.regular_source_reference import NurbsReference
    model, face, plate, plan, result, policy = bounded_run(RECT)
    traces, auxiliary = edge_kinds(model)
    checked = 0
    for plate_cell in plate_faces(model):
        item = model.faces[plate_cell]
        reference = PlaneReference(model, item)
        for use in (u for loop in (item.loop, *item.holes) for u in loop):
            edge = model.edges[use.edge]
            if not isinstance(edge.curve, PlaneNurbsCurve):
                continue
            for t, vertex in ((Q(0), edge.start), (Q(1), edge.end)):
                joint = plate_joint(model, vertex, use.edge)
                if joint is None:
                    continue
                first, last = reference.node(use.edge, Q(0), edge.start), reference.node(use.edge, Q(1), edge.end)
                side = reference.side(use.edge, (Q(0), Q(1)), first, last)
                native = side.native.start if t == 0 else side.native.end
                assert native.identity == joint.identity                  # native chart endpoint IS the joint
                lo, hi = (Q(0), Q(1, 16)) if t == 0 else (Q(15, 16), Q(1))
                box, rate = side.native.box(lo, hi), side.native.derivative(lo, hi)
                assert all(a <= b for (a, b) in box) and all(a <= b for (a, b) in rate)
                # the piece starting at the joint encloses the joint's own chart enclosure
                assert all(b0 <= n1 and n0 <= b1 for (b0, b1), (n0, n1) in zip(box, native.box))
                checked += 1
    assert checked


def test_raw_line_regeneration_after_save_load_and_a_remote_endpoint_edit_is_refused():
    import numpy as np
    from anygeometry._canonical_plate_joint import JointRefused, plate_joint
    model, face, plate, plan, result, policy = bounded_run(RECT)
    traces, auxiliary = edge_kinds(model)
    loaded = from_dict(to_dict(model))
    joints = {}
    for edge in (*traces, *auxiliary):
        item = loaded.edges[edge]
        for vertex in (item.start, item.end):
            joint = plate_joint(loaded, vertex, edge)
            if joint is not None:
                joints[(vertex, edge)] = joint
    assert joints                                                       # regenerated purely from the definitions
    (vertex, edge), joint = next(iter(joints.items()))
    straight = next(s for s in loaded.topological_edges_using_vertex(vertex)
                    if type(loaded.edges[s].curve).__name__ == 'Straight')
    item = loaded.edges[straight]
    remote = item.end if item.start == vertex else item.start
    x, y, z = (float(c) for c in loaded.vertex_position(remote))
    before, revision = to_dict(loaded), loaded.revision
    with pytest.raises(GeometryError, match='without a valid evaluable surface'):
        loaded.move_point(remote, x, y, z + 2.0**-40)
    assert loaded.revision == revision and to_dict(loaded) == before

    # Public editing correctly refuses first. Deliberately bypass that guard to
    # exercise regeneration from the ACTUAL raw remote endpoint, not an old
    # projected line or a cached successful joint after save/load.
    raw_vertex = loaded.vertices[remote]
    original = raw_vertex.position
    changed = np.array((x, y, z + 2.0**-40), dtype=float)
    changed.flags.writeable = False
    try:
        object.__setattr__(raw_vertex, 'position', changed)
        assert loaded.vertex_position(remote) is changed
        assert loaded.revision == revision
        assert float(loaded.vertex_position(remote)[2]) - z == 2.0**-40
        with pytest.raises(JointRefused):
            plate_joint(loaded, vertex, edge)
    finally:
        object.__setattr__(raw_vertex, 'position', original)


def _joint_pairs(model):
    from anygeometry._canonical_plate_joint import plate_joint
    traces, auxiliary = edge_kinds(model)
    pairs = []
    for edge in (*traces, *auxiliary):
        item = model.edges[edge]
        for vertex in (item.start, item.end):
            if plate_joint(model, vertex, edge) is not None:
                pairs.append((vertex, edge))
    return pairs


def test_joint_symbol_is_invariant_under_creation_and_regeneration_work_settings():
    """The station was created under the public policy Work (depth-limited refinement); regeneration with no Work
    (deeper refinement), the same depth and other depths must give the SAME symbol whenever they certify, and a
    budget too small to certify must refuse (typed) leaving the model unchanged."""
    from anygeometry._canonical_plate_joint import plate_joint
    from anygeometry.plane_nurbs import CURRENT_WORK
    from anygeometry.section_audit import Work
    model, face, plate, plan, result, policy = bounded_run(RECT)
    pairs = _joint_pairs(model)
    assert pairs
    baseline = {pair: plate_joint(model, *pair).identity for pair in pairs}
    certified = []
    for depth in (24, 32, 40, 48):
        token = CURRENT_WORK.set(Work(10**7, depth))
        try:
            try:
                symbols = {pair: plate_joint(model, *pair).identity for pair in pairs}
            except GeometryError:
                continue                                              # typed refusal at an insufficient depth
        finally:
            CURRENT_WORK.reset(token)
        assert symbols == baseline                                    # refinement never changes the identity
        certified.append(depth)
    assert 32 in certified or 40 in certified                         # at least the creation-like depth certifies
    before = to_dict(model)
    token = CURRENT_WORK.set(Work(3, 32))
    try:
        with pytest.raises(GeometryError):
            plate_joint(model, *pairs[0])
    finally:
        CURRENT_WORK.reset(token)
    assert to_dict(model) == before


def test_joint_symbol_survives_save_and_load_regeneration():
    from anygeometry._canonical_plate_joint import plate_joint
    model, face, plate, plan, result, policy = bounded_run(RECT)
    pairs = _joint_pairs(model)
    loaded = from_dict(to_dict(model))
    assert {pair: plate_joint(loaded, *pair).identity for pair in pairs} == {
        pair: plate_joint(model, *pair).identity for pair in pairs}


def test_a_very_shallow_regeneration_refuses_instead_of_merging_far_roots():
    from anygeometry._canonical_plate_joint import plate_joint
    from anygeometry.plane_nurbs import CURRENT_WORK
    from anygeometry.section_audit import Work
    model, face, plate, plan, result, policy = bounded_run(RECT)
    pairs = _joint_pairs(model)
    token = CURRENT_WORK.set(Work(10**6, 2))
    try:
        with pytest.raises(GeometryError):
            plate_joint(model, *pairs[0])
    finally:
        CURRENT_WORK.reset(token)


def _fake_root(low, high, lower, upper, isolated=True):
    from types import SimpleNamespace
    return SimpleNamespace(isolated=isolated, exact=((Q(low), Q(high)),), lower=lower, upper=upper)


def test_candidate_exclusion_uses_a_strict_distance_and_needs_exactly_one_possible_root():
    """Pure rule check on the exclusion primitive (no geometry claim): a root whose enclosure is farther than the
    tolerance from the stored station is incompatible; exactly one possible root must remain."""
    from anygeometry._canonical_plate_joint import JointRefused, select_candidate, select_roots
    tol = Q(1, 1000)
    near = _fake_root(Q(1, 10), Q(11, 100), (0.0, 0.0, 0.5), (1e-12, 1e-12, 0.5))
    far = _fake_root(Q(5, 10), Q(51, 100), (0.5, 0.5, 0.5), (0.5 + 1e-12, 0.5 + 1e-12, 0.5))
    q = (Q(0), Q(0), Q(1, 2))
    roots = select_roots([far, near])
    assert roots[0] is near and select_candidate(roots, q, tol)[0] == 0          # ordinal is the global order
    with pytest.raises(JointRefused):                                              # zero possible
        select_candidate(roots, (Q(9, 10), Q(9, 10), Q(1, 2)), tol)
    close_both = _fake_root(Q(2, 10), Q(21, 100), (0.0005, 0.0, 0.5), (0.0005 + 1e-12, 1e-12, 0.5))
    with pytest.raises(JointRefused):                                              # two possible: ambiguity refuses
        select_candidate(select_roots([near, close_both]), q, tol)
    with pytest.raises(JointRefused):                                              # a non-isolated root anywhere
        select_roots([near, _fake_root(Q(6, 10), Q(7, 10), (1.0,) * 3, (1.0,) * 3, isolated=False)])
    with pytest.raises(JointRefused):                                              # overlapping isolators
        select_roots([near, _fake_root(Q(105, 1000), Q(12, 100), (0.0,) * 3, (1e-12,) * 3)])


def test_parameter_hull_must_stay_strictly_inside_one_arc_and_within_the_length_tolerance():
    from anygeometry._canonical_plate_joint import JointRefused, check_hull
    curve = bowl_circle_curve()
    count = len(curve._arcs())
    joint = Q(1, count)
    tol = Q(1, 10**9)
    with pytest.raises(JointRefused):                                  # proxy and isolator on both sides of an arc joint
        check_hull(curve, (joint - Q(1, 10**9),), (joint + Q(1, 10**9), joint + Q(2, 10**9)), tol, NOOP)
    inside = (Q(1, 4 * count), Q(1, 4 * count) + Q(1, 10**12))
    check_hull(curve, (Q(1, 4 * count),), inside, Q(1, 10), NOOP)       # tiny hull in one arc, generous tolerance
    with pytest.raises(JointRefused):                                  # same arc, but the hull is far wider than tol
        check_hull(curve, (Q(1, 4 * count),), (Q(3, 4 * count), Q(3, 4 * count) + Q(1, 10**12)), tol, NOOP)


def bowl_circle_curve():
    model, face, plane = bowl()
    return PlaneNurbsCurve.from_plane(plane, model.faces[face].surface, 0)


def test_the_whole_isolator_width_enters_the_parameter_transport_bound():
    from dataclasses import replace as dc_replace
    from anygeometry.regular_source_reference import NurbsReference, parameter_transport_bound
    model, face, plate, plan, result, policy = bounded_run(RECT)
    traces, auxiliary = edge_kinds(model)
    checked = 0
    for cell in nurbs_faces(model):
        reference = NurbsReference(model, model.faces[cell])
        for use in (u for loop in (model.faces[cell].loop, *model.faces[cell].holes) for u in loop):
            item = model.edges[use.edge]
            if not isinstance(item.curve, PlaneNurbsCurve):
                continue
            first = reference.node(use.edge, Q(0), item.start)
            last = reference.node(use.edge, Q(1), item.end)
            if first.path_interval is None and last.path_interval is None:
                continue
            side = reference.side(use.edge, (Q(0), Q(1)), first, last)
            base = parameter_transport_bound(side)
            node = side.start if side.start.path_interval is not None else side.end
            low, high = node.path_interval
            wide = dc_replace(node, path_interval=(low - Q(1, 10**4), high + Q(1, 10**4)))
            widened = dc_replace(side, start=wide) if node is side.start else dc_replace(side, end=wide)
            assert parameter_transport_bound(widened) > base                    # the full interval is charged
            checked += 1
    assert checked


def test_plane_nurbs_and_straight_sides_consume_the_same_joint_interval_and_endpoint_transport():
    from dataclasses import replace as dc_replace
    from anygeometry._canonical_plate_joint import joint_endpoint_distance, plate_joint
    from anygeometry.curves import Straight
    from anygeometry.plane_source_reference import PlaneReference
    from anygeometry.regular_source_reference import NurbsReference
    model, face, plate, plan, result, policy = bounded_run(RECT)
    checked = 0
    for plate_cell in plate_faces(model):
        item = model.faces[plate_cell]
        reference = PlaneReference(model, item)
        assert reference.joints                                            # mixed joints exist in this cell
        for use in (u for loop in (item.loop, *item.holes) for u in loop):
            edge = model.edges[use.edge]
            if type(edge.curve) is not Straight:
                continue
            first, last = reference.node(use.edge, Q(0), edge.start), reference.node(use.edge, Q(1), edge.end)
            side = reference.side(use.edge, (Q(0), Q(1)), first, last)
            expected = sum((joint_endpoint_distance(model, v, reference.joints[v])
                            for v in (edge.start, edge.end) if v in reference.joints), Q(0))
            assert side.endpoint_transport == expected
            assert dc_replace(side).endpoint_transport == expected          # the copy keeps the transport field
            checked += 1
    assert checked
    for cell in nurbs_faces(model):
        nurbs = NurbsReference(model, model.faces[cell])
        for vertex, node in nurbs.vertices.items():
            if node.identity[0] == 'plate_joint':
                assert node.path_interval is not None and node.path_interval[0] <= node.path_parameter <= node.path_interval[1]


def _box_around(point, half):
    return tuple(Q(float(x)) - half for x in point), tuple(Q(float(x)) + half for x in point)


def test_contact_free_gap_witness_classifies_inside_outside_and_across_an_arc_joint():
    from anygeometry._canonical_plate_joint import _classify_gap
    model, face, plate, plane, curve, frame = fixture(RECT)
    root = on_segment(model, plate, plane, curve, frame)[0]
    middle = float((root.isolator[0] + root.isolator[1]) / 2)
    point = curve.evaluate(middle)
    lower, upper = _box_around(point, Q(1, 1000))
    gap = (root.isolator[0] - Q(1, 10**6), root.isolator[1] + Q(1, 10**6))
    assert _classify_gap(curve, lower, upper, *gap, NOOP) == 'in'                  # the whole tiny gap is inside
    far = (tuple(Q(10) for _ in range(3)), tuple(Q(11) for _ in range(3)))
    assert _classify_gap(curve, far[0], far[1], *gap, NOOP) == 'out'               # and clear of a distant box
    count = len(curve._arcs())                                                     # a gap spanning an arc joint
    joint = Q(1, count)
    on_joint = curve.evaluate(float(joint))
    lower, upper = _box_around(on_joint, Q(1, 1000))
    assert _classify_gap(curve, lower, upper, joint - Q(1, 10**7), joint + Q(1, 10**7), NOOP) == 'in'


def test_an_unrepresentable_or_unresolvable_gap_witness_refuses_and_work_and_cancel_are_honoured():
    from anygeometry._canonical_plate_joint import JointRefused, _classify_gap
    from anygeometry.plane_nurbs import CURRENT_WORK
    from anygeometry.section_audit import Work
    model, face, plate, plane, curve, frame = fixture(RECT)
    lower, upper = (Q(-1),) * 3, (Q(1),) * 3
    a = Q(1, 2)
    with pytest.raises(JointRefused):                                              # no float neighbourhood fits strictly inside
        _classify_gap(curve, lower, upper, a, a + Q(1, 2**80), NOOP)

    class Cancelled(GeometryError):
        pass

    def cancel(*_):
        raise Cancelled('cancelled')

    with pytest.raises(Cancelled):
        _classify_gap(curve, lower, upper, Q(1, 4), Q(1, 2), cancel)
    token = CURRENT_WORK.set(Work(1, 32))                                          # a one-box budget cannot decide
    try:
        with pytest.raises(GeometryError):
            _classify_gap(curve, lower, upper, Q(1, 4), Q(1, 2), NOOP)
    finally:
        CURRENT_WORK.reset(token)


def test_the_local_neighbourhood_decision_is_invariant_under_root_refinement_work_profiles():
    from anygeometry._canonical_plate_joint import local_certificate
    from anygeometry.plane_nurbs import CURRENT_WORK
    from anygeometry.section_audit import Work
    tol = None
    outcomes = []
    for work in (None, Work(10**7, 32), Work(10**7, 40)):
        token = CURRENT_WORK.set(work) if work is not None else None
        try:
            model, face, plate, plane, curve, frame = fixture(RECT)
            tol = Q(float(model.tolerance.effective_length(1.)))
            roots = on_segment(model, plate, plane, curve, frame)               # different isolators per profile
            outcomes.append(all(local_certificate(curve, r.isolator, r.lower, r.upper, r.line[0], r.line[1], tol,
                                                  True) is True for r in roots))
        finally:
            if token is not None:
                CURRENT_WORK.reset(token)
    assert outcomes == [True, True, True]


def test_a_nonlocal_same_path_return_inside_the_joint_neighbourhood_is_rejected():
    """The local certificate with an enlarged neighbourhood (explicit tolerance argument, real circle and real
    plate line): the closed section crosses the line twice and lies inside the box, so it must refuse."""
    from anygeometry._canonical_plate_joint import JointRefused, local_certificate
    model, face, plate, plane, curve, frame = fixture(RECT)
    roots = on_segment(model, plate, plane, curve, frame)
    root = roots[0]
    with pytest.raises(JointRefused):
        local_certificate(curve, root.isolator, root.lower, root.upper, root.line[0], root.line[1], Q(2), True)
    # At x=.35 the two roots have |y|=sqrt(.5-.35**2)>.5.
    # A .5-expanded root box excludes the other root and both y=0 turning
    # points, so its one local arc is monotone and must be admitted.
    assert local_certificate(curve, root.isolator, root.lower, root.upper,
                             root.line[0], root.line[1], Q(1, 2), True) is True


@pytest.mark.parametrize('tolerance', (Q(7, 10), Q(11, 10)))
def test_an_extra_plate_event_or_return_anywhere_in_the_whole_local_envelope_is_refused(tolerance):
    """The neighbourhood is enlarged (explicit argument, real circle and plate line) until the local envelope
    contains the second plate event and/or a re-crossing; the certificate proves exactly ONE intended event over
    the whole envelope (bands included), so it must refuse."""
    from anygeometry._canonical_plate_joint import JointRefused, local_certificate
    model, face, plate, plane, curve, frame = fixture(RECT)
    root = on_segment(model, plate, plane, curve, frame)[0]
    with pytest.raises(JointRefused):
        local_certificate(curve, root.isolator, root.lower, root.upper, root.line[0], root.line[1], tolerance, True)


def test_a_proxy_outside_the_strict_gap_refuses():
    from anygeometry._canonical_plate_joint import JointRefused, local_certificate
    model, face, plate, plane, curve, frame = fixture(RECT)
    root = on_segment(model, plate, plane, curve, frame)[0]
    tol = Q(float(model.tolerance.effective_length(1.)))
    far = root.isolator[1] + Q(1, 10)                                     # certainly not inside the joint gap
    with pytest.raises(JointRefused):
        local_certificate(curve, root.isolator, root.lower, root.upper, root.line[0], root.line[1], tol, True,
                          (far,))


def test_local_proof_recursion_respects_cancellation_and_the_active_work_budget():
    from anygeometry._canonical_plate_joint import local_certificate
    from anygeometry.plane_nurbs import CURRENT_WORK
    from anygeometry.section_audit import Work
    model, face, plate, plane, curve, frame = fixture(RECT)
    root = on_segment(model, plate, plane, curve, frame)[0]
    tol = Q(float(model.tolerance.effective_length(1.)))

    class Cancelled(GeometryError):
        pass

    def cancel(*_):
        raise Cancelled('cancelled')

    with pytest.raises(Cancelled):                                         # the caller's check is honoured
        local_certificate(curve, root.isolator, root.lower, root.upper, root.line[0], root.line[1], tol, True,
                          (), cancel)
    token = CURRENT_WORK.set(Work(3, 32))                                  # a tiny box budget refuses, never succeeds
    try:
        with pytest.raises(GeometryError):
            local_certificate(curve, root.isolator, root.lower, root.upper, root.line[0], root.line[1], tol, True)
    finally:
        CURRENT_WORK.reset(token)
    token = CURRENT_WORK.set(Work(10**6, 1))                               # a tiny depth budget refuses as well
    try:
        with pytest.raises(GeometryError):
            local_certificate(curve, root.isolator, root.lower, root.upper, root.line[0], root.line[1], tol, True)
    finally:
        CURRENT_WORK.reset(token)


def test_source_trims_must_be_strictly_clear_of_the_joint_neighbourhood():
    from anygeometry._canonical_plate_joint import JointRefused, exact, trims_clear
    model, face, plane = bowl()
    corner = [Q(float(x)) for x in model.vertex_position(model.edges[model.faces[face].loop[0].edge].start)]
    near = (tuple(x - Q(1, 1000) for x in corner), tuple(x + Q(1, 1000) for x in corner))
    with pytest.raises(JointRefused):                                       # a real original trim passes through it
        trims_clear(model, face, near[0], near[1], None, NOOP)
    far = ((Q(-1, 100), Q(-1, 100), Q(49, 100)), (Q(1, 100), Q(1, 100), Q(51, 100)))      # the bowl floor, trims far
    assert trims_clear(model, face, far[0], far[1], None, NOOP) is None


def test_the_local_certificate_admits_the_ordinary_axis_aligned_joint():
    from anygeometry._canonical_plate_joint import local_certificate
    model, face, plate, plane, curve, frame = fixture(RECT)
    for root in on_segment(model, plate, plane, curve, frame):
        assert local_certificate(curve, root.isolator, root.lower, root.upper, root.line[0], root.line[1],
                                 Q(float(model.tolerance.effective_length(1.))), True) is True


def test_trim_clearance_charges_real_straight_and_pcurve_boxes():
    from anygeometry._canonical_plate_joint import edge_clear
    from anygeometry.plane_nurbs import CURRENT_WORK
    from anygeometry.section_audit import Work
    model, face, plate, *_ = fixture(RECT)
    far_lo, far_hi = (Q(10),) * 3, (Q(11),) * 3
    for edge in (model.faces[face].loop[0].edge, model.faces[plate].loop[0].edge):
        token = CURRENT_WORK.set(Work(0, 32))
        try:
            with pytest.raises(GeometryError, match='budget exhausted'):
                edge_clear(model, edge, far_lo, far_hi, NOOP)
        finally:
            CURRENT_WORK.reset(token)


def test_unregistered_bounds_provider_cannot_certify_trim_clearance():
    from anygeometry._canonical_plate_joint import JointRefused, _certified_world_bounds
    class UnregisteredCurve:
        def bounds(self, *_):
            pytest.fail('an unregistered enclosure provider must never run')
    with pytest.raises(JointRefused, match='no certified world enclosure'):
        _certified_world_bounds(UnregisteredCurve(), 0., 1.)


def test_legacy_contained_route_source_references_are_unchanged():
    from anygeometry.plane_source_reference import PlaneReference
    from anygeometry.regular_source_reference import NurbsReference
    model, face, plate, plan, result, policy = bounded_run([(-2., -2.), (2., -2.), (2., 2.), (-2., 2.)])
    for cell in nurbs_faces(model):
        nodes = NurbsReference(model, model.faces[cell]).vertices
        assert all(node.identity[0] != 'plate_joint' for node in nodes.values())
    for cell in plate_faces(model):
        nodes = PlaneReference(model, model.faces[cell]).vertices
        assert all('plate_joint' not in str(node.identity) for node in nodes.values())


# ------------------------------------------------------------- tips, registry

def test_a_non_incident_trace_never_receives_the_tip_exemption_and_the_created_segment_is_certified(monkeypatch):
    import anygeometry.branch_predicates as predicates
    from anygeometry.bounded_cut_cells import complete_tips
    from anygeometry.plane_split import _stage_plane_split
    from anygeometry.predicates import DEFAULT_INTERSECTION_QUALIFICATION_POLICY as POLICY
    model, face, plane = bowl()
    plate = plate_on(model, [(-2., -2.), (2., -2.), (2., 2.), (-2., 2.)])
    staged = model.clone(preserve_identity=True)
    seen = []

    def recorder(curve, start, end, shared=(), tolerance=0.0, **kwargs):
        seen.append(tuple(shared))
        return True

    monkeypatch.setattr(predicates, 'certify_line_section', recorder)
    with staged.transaction():
        _m, traces, _p, _n = _stage_plane_split(staged, (face,), plane, POLICY)
        assert len(traces) >= 4                                   # a circle of >= 4 arcs has non-adjacent pairs
        first, other = traces[0], traces[len(traces) // 2]
        assert not {staged.edges[first].start, staged.edges[first].end} & {staged.edges[other].start,
                                                                           staged.edges[other].end}
        before = len(seen)
        complete_tips(staged, plate, (first, other), lambda *_: None, None, None)
    assert () in seen[before:] and any(shared for shared in seen[before:])


def test_auxiliary_tag_alone_never_grants_registry_authority():
    import anygeometry.prepared_material_source_transfer as transfer
    model, face, plate, plan, result, policy = bounded_run(RECT)
    loaded = from_dict(to_dict(model))
    traces, auxiliary = edge_kinds(loaded)
    assert auxiliary and transfer._AUXILIARY.get(loaded) is None
    assert not any(transfer._qualified_auxiliary(loaded, e) for e in auxiliary)
    assert all(transfer._qualified_auxiliary(model, e) for e in auxiliary)    # the owner-issued registry


def test_auxiliary_registry_is_carried_by_a_genuine_clone_and_lost_when_a_definition_changes():
    import anygeometry.prepared_material_source_transfer as transfer
    model, face, plate, plan, result, policy = bounded_run(RECT)
    traces, auxiliary = edge_kinds(model)
    clone = model.clone(preserve_identity=False)
    transfer._register_source_clone(model, clone)
    assert all(transfer._qualified_auxiliary(clone, e) for e in auxiliary)
    ordinary = model.clone(preserve_identity=False)                          # no owner clone boundary: no authority
    assert not any(transfer._qualified_auxiliary(ordinary, e) for e in auxiliary)
    transfer._AUXILIARY[ordinary] = {e: 'stale-definition' for e in auxiliary}
    assert not any(transfer._qualified_auxiliary(ordinary, e) for e in auxiliary)


def test_publication_failure_rolls_back_the_auxiliary_registry(monkeypatch):
    import anygeometry.prepared_material_source_transfer as transfer
    from anygeometry.bounded_nurbs import apply_bounded, plan_bounded
    model, face, plane = bowl()
    plate = plate_on(model, RECT)
    handles = (model.handle('face', face), model.handle('face', plate))
    policy = policy_for()
    plan = plan_bounded(model, handles, policy)
    before = to_dict(model)
    original = transfer._publish_source_lineage

    def failing(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError('injected publication failure')

    monkeypatch.setattr(transfer, '_publish_source_lineage', failing)
    with pytest.raises(RuntimeError, match='injected'):
        apply_bounded(model, plan, policy)
    assert transfer._AUXILIARY.get(model) is None and to_dict(model) == before


def test_active_depth_limit_gives_the_same_isolated_decision_as_the_unbudgeted_roots():
    from anygeometry.branch_predicates import section_roots
    from anygeometry.bounded_cut_graph import plane_equation
    from anygeometry.plane_nurbs import CURRENT_WORK
    from anygeometry.section_audit import Work
    model, face, plate, plane, curve, frame = fixture(RECT)
    root = on_segment(model, plate, plane, curve, frame)[0]
    edge = model.edges[root.plate_edge]
    a = tuple(Q(x) for x in frame.world(*frame.uv(model.vertex_position(edge.start))))
    b = tuple(Q(x) for x in frame.world(*frame.uv(model.vertex_position(edge.end))))
    normal_plate, _offset = plane_equation(plane)
    e = tuple(y - x for x, y in zip(a, b))
    normal = (e[1]*normal_plate[2] - e[2]*normal_plate[1], e[2]*normal_plate[0] - e[0]*normal_plate[2],
              e[0]*normal_plate[1] - e[1]*normal_plate[0])
    offset = sum(x*y for x, y in zip(normal, a))
    direct = section_roots(curve, normal, offset)
    token = CURRENT_WORK.set(Work(10**6, 32))
    try:
        limited = section_roots(curve, normal, offset)
    finally:
        CURRENT_WORK.reset(token)
    assert len(limited) == len(direct) and all(r.isolated for r in limited)
    token = CURRENT_WORK.set(Work(10**6, 1))                       # a genuinely too-small depth still refuses
    try:
        # This axis cut already has a valid simple-root certificate at depth0;
        # budget limits do not require artificial subdivision/refusal.
        shallow = section_roots(curve, normal, offset)
        assert len(shallow) == len(direct) and all(r.isolated for r in shallow)
        # The oblique cut needs subdivision to certify both simple crossings.
        with pytest.raises(GeometryError):
            section_roots(curve, (1, 1, 0), Q(9, 10))
    finally:
        CURRENT_WORK.reset(token)
    token = CURRENT_WORK.set(Work(10**6, 32))
    try:
        oblique = section_roots(curve, (1, 1, 0), Q(9, 10))
        assert len(oblique) == 2 and all(r.isolated for r in oblique)
    finally:
        CURRENT_WORK.reset(token)
