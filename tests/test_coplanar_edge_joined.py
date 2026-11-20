"""Owner decision 12 (2026-11-19, HbeamBoth blocker 2): near-coplanar plane faces that share a topological
edge are classified "by edge" (edge-joined) instead of being refused as ``ill_conditioned_plane_plane``.

The rule (docs/coplanar-edge-joined.md) applies only in the plane-plane predicate's ill-conditioned band
(angular tolerance < sine <= sqrt(eps)); it needs a shared topological edge AND a certified bound showing
every material point of each face within the strictest owning-face tolerance of the other face's plane.
Every other near-coplanar pair keeps a typed refusal.
"""
from dataclasses import replace

import numpy as np
import math

import pytest

from anygeometry import GeometryError, GeometryModel, to_dict
from anygeometry.batch_intersections import apply_intersections, plan_intersections
from anygeometry.surfaces import Plane


def _model(second_ring, *, share=True, chart_scale=1.):
    """Plate A = unit square in z=0. Plate B = ring of 4 positions; when ``share`` its first and last
    positions must be A's (1,0,0) and (1,1,0) and B reuses A's edge between them."""
    model = GeometryModel()
    a = model.add_points(((0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.)))
    a_edges = [model.add_line(a[i], a[(i+1) % 4]) for i in range(4)]
    face_a = model.add_face(a_edges)
    if share:
        assert second_ring[0] == (1., 0., 0.) and second_ring[-1] == (1., 1., 0.)
        b = [a[1], *model.add_points(second_ring[1:3]), a[2]]
    else:
        b = list(model.add_points(second_ring))
    b_edges = []
    for i in range(4):
        start, end = b[i], b[(i+1) % 4]
        b_edges.append(a_edges[1] if share and {start, end} == {a[1], a[2]} else model.add_line(start, end))
    face_b = model.add_face(b_edges)
    for face, ring, scale in ((face_a, a, chart_scale), (face_b, b, 1.)):
        p = [model.vertex_position(v) for v in ring]
        model._put_entity("face", replace(model.faces[face],
                                          surface=Plane(p[0], scale*(p[1]-p[0]), scale*(p[3]-p[0]))))
    return model, (model.handle("face", face_a), model.handle("face", face_b))


def _caps(slope, *, folded=False, first_arc_tilt=0.):
    """Two half discs sharing the straight diameter (0,-1,0)-(0,0,0)-(0,1,0) (two model edges): A on x<0 in
    z=0, B on x>0 in z = slope*x (``folded``: B also on x<0, over A). Returns the shared edge ids."""
    model = GeometryModel()
    start, middle, end = model.add_points(((0., -1., 0.), (0., 0., 0.), (0., 1., 0.)))
    side = -1. if folded else 1.
    via_a, via_b = model.add_points(((-1., 0., -first_arc_tilt), (side, 0., slope*side)))
    lower, upper = model.add_line(start, middle), model.add_line(middle, end)
    face_a = model.add_face((lower, upper, model.add_arc(end, via_a, start)))
    face_b = model.add_face((lower, upper, model.add_arc(end, via_b, start)))
    for face, axis in ((face_a, (-1., 0., 0.)), (face_b, (side, 0., slope*side))):
        model._put_entity("face", replace(model.faces[face], surface=Plane((0., 0., 0.), (0., 1., 0.), axis)))
    return model, (model.handle("face", face_a), model.handle("face", face_b)), (lower, upper)


def _joined_by(plan, edge, faces):
    """Every arrangement carries the shared model edge as a path owned by both faces."""
    owners = tuple(sorted(face.id for face in faces))
    return len(plan.arrangements) == 2 and all(
        any(path.source_edge == edge and tuple(sorted(path.owners)) == owners for path in arrangement.paths)
        for arrangement in plan.arrangements)


def _neighbour(slope, length=1.):
    """B continues A across x=1 with z = slope*(x-1): sine of the plane angle ~ slope."""
    return ((1., 0., 0.), (1.+length, 0., slope*length), (1.+length, 1., slope*length), (1., 1., 0.))


def _plan(model, faces):
    return plan_intersections(model, faces, policy="connect")


def _shape(plan):
    """Plan content without ids: per arrangement the sorted rounded trace endpoints."""
    shapes = []
    for arrangement in plan.arrangements:
        paths = []
        for path in getattr(arrangement, "paths", ()):
            curve = getattr(path, "curve", path)
            if isinstance(curve, type(None)):
                continue
            ends = tuple(sorted(tuple(np.round(np.asarray(curve.evaluate(t), float), 6)) for t in (0., 1.)))
            paths.append(ends)
        shapes.append(tuple(sorted(paths)))
    return sorted(shapes)


SINE_IN_BAND = 5.0e-10  # above the default angular tolerance 1e-10, below sqrt(eps) ~ 1.49e-8


def test_in_band_neighbours_sharing_an_edge_are_edge_joined_like_exact_coplanar():
    # Failed on base: "intersection batch face:1/face:2: ill_conditioned_plane_plane".
    model, faces = _model(_neighbour(SINE_IN_BAND))
    exact_model, exact_faces = _model(_neighbour(0.))
    plan = _plan(model, faces)
    assert _shape(plan) == _shape(_plan(exact_model, exact_faces))
    shared = next(edge for edge in model.edges if len(model.faces_using_edge(edge)) == 2)
    assert _joined_by(plan, shared, faces)
    assert _joined_by(_plan(exact_model, exact_faces), shared, exact_faces)
    before = to_dict(model)
    apply_intersections(model, plan, policy="connect")
    assert model.validate_topology() == ()
    assert to_dict(model)["revision"] >= before["revision"]


def test_in_band_neighbours_folded_back_over_each_other_refuse_typed():
    # Shares edge 1 but B lies back over A (x from 1 to 0.5): interiors overlap. Zone contract: A's and B's
    # bottom edges overlap from the shared vertex on, so rule 2 refuses first (contact off the shared edge).
    ring = ((1., 0., 0.), (.5, 0., SINE_IN_BAND*.5), (.5, 1., SINE_IN_BAND*.5), (1., 1., 0.))
    model, faces = _model(ring)
    with pytest.raises(GeometryError, match=CONTACT):
        _plan(model, faces)


def test_in_band_overlapping_plates_without_a_shared_edge_refuse_typed():
    ring = ((.5, 0., 0.), (1.5, 0., SINE_IN_BAND), (1.5, 1., SINE_IN_BAND), (.5, 1., 0.))
    model, faces = _model(ring, share=False)
    with pytest.raises(GeometryError, match="ill_conditioned_plane_plane; near_coplanar_without_shared_edge"):
        _plan(model, faces)


def test_in_band_plates_touching_along_distinct_edges_refuse_typed():
    # Geometrically abutting, but no shared topological edge: coplanarity alone is not evidence.
    model, faces = _model(_neighbour(SINE_IN_BAND), share=False)
    with pytest.raises(GeometryError, match="ill_conditioned_plane_plane; near_coplanar_without_shared_edge"):
        _plan(model, faces)


def test_in_band_coplanar_disjoint_plates_are_unchanged():
    ring = ((2., 0., 0.), (3., 0., SINE_IN_BAND), (3., 1., SINE_IN_BAND), (2., 1., 0.))
    model, faces = _model(ring, share=False)
    assert _plan(model, faces).arrangements == ()


def test_in_band_shared_edge_but_far_corner_outside_face_tolerance_refuses_typed():
    # sine 1e-8 is still in the band, but B's far edge is 1e-7 m off A's plane (> 1e-9 m tolerance):
    # a genuinely transversal pair that the predicate cannot condition, so it keeps a refusal.
    model, faces = _model(_neighbour(1.0e-8, length=10.))
    with pytest.raises(GeometryError, match="ill_conditioned_plane_plane; near_coplanar_outside_face_tolerance"):
        _plan(model, faces)


@pytest.mark.parametrize("slope", (1.0e-6, 1.0e-3, 1.0))
def test_transversal_neighbours_sharing_an_edge_keep_the_crossing_route(slope):
    model, faces = _model(_neighbour(slope))
    plan = _plan(model, faces)
    assert len(plan.arrangements) == 2


def test_perpendicular_neighbour_sharing_an_edge_keeps_the_crossing_route():
    ring = ((1., 0., 0.), (1., 0., 1.), (1., 1., 1.), (1., 1., 0.))
    model, faces = _model(ring)
    assert len(_plan(model, faces).arrangements) == 2


def test_exactly_coplanar_neighbours_do_not_use_the_band_rule():
    # Zone contract: no monkeypatching; the rule's own success counter shows whether it ran.
    from anygeometry.runtime_diagnostics import collect_runtime_diagnostics
    for slope, uses in ((0., 0), (1.0e-6, 0), (SINE_IN_BAND, 1)):
        model, faces = _model(_neighbour(slope))
        with collect_runtime_diagnostics() as counters:
            _plan(model, faces)
        assert counters.get('plane_plane_coplanar_edge_joined', 0) == uses, slope


def test_in_band_edge_join_works_in_world_units_whatever_the_stored_chart_scale():
    # Codex r1 H2: a chart scaled by 1e10 must neither hide an overlap nor lose the shared-edge trace.
    # (On main the exactly coplanar route plans both of these as empty: a pre-existing defect, not this rule.)
    model, faces = _model(_neighbour(SINE_IN_BAND), chart_scale=1e10)
    shared = next(edge for edge in model.edges if len(model.faces_using_edge(edge)) == 2)
    # Pair stage: the shared edge is the trace (the owning-face arrangement of a 1e10 chart refuses later).
    from anygeometry import batch_intersections
    from anygeometry.material_arrangement import MaterialDomain
    first, second = (MaterialDomain.from_model(model, face.id) for face in faces)
    traces = batch_intersections._pair_paths_exact(first, second, model, 1e-9, lambda: None)
    ends = sorted(tuple(sorted(tuple(np.round(np.asarray(t.evaluate(x), float), 9)) for x in (0., 1.)))
                  for t in traces)
    assert ends == [((1., 0., 0.), (1., 1., 0.))]
    try:
        plan = _plan(model, faces)
    except GeometryError as error:
        assert "material conservation" in str(error)
    else:
        assert _joined_by(plan, shared, faces)
    ring = ((1., 0., 0.), (.5, 0., SINE_IN_BAND*.5), (.5, 1., SINE_IN_BAND*.5), (1., 1., 0.))
    model, faces = _model(ring, chart_scale=1e10)
    with pytest.raises(GeometryError, match=CONTACT):  # zone contract: rule 2 refuses the fold first
        _plan(model, faces)


def test_in_band_curved_caps_sharing_a_diameter_refuse_unresolved():
    # Decision 15 round 1: arrange_material's curved route (float cycles, areas and cuts) is not certified
    # at t, so a pair with an arc refuses typed instead of joining (it joined up to e11c2af).
    model, faces, _ = _caps(SINE_IN_BAND)
    with pytest.raises(GeometryError, match=UNRESOLVED):
        _plan(model, faces)


def test_in_band_curved_caps_folded_over_each_other_refuse_positive_area_overlap():
    # Codex r1 M1: the curved backend must see the overlap in the common plane.
    model, faces, _ = _caps(SINE_IN_BAND, folded=True)
    with pytest.raises(GeometryError, match=REFUSED_OVERLAP):
        _plan(model, faces)


def _hinged_fold(length, depth, slope=SINE_IN_BAND):
    """A = [0,length]x[0,1] in z=0; B shares A's edge x=length and folds back over A by ``depth``; B's plane
    is tilted about the x axis (z = slope*(y-0.5)), so both faces stay within tolerance of both planes."""
    model = GeometryModel()
    a = model.add_points(((0., 0., 0.), (length, 0., 0.), (length, 1., 0.), (0., 1., 0.)))
    hinge = model.add_line(a[1], a[2])
    face_a = model.add_face((model.add_line(a[0], a[1]), hinge, model.add_line(a[2], a[3]),
                             model.add_line(a[3], a[0])))
    b = model.add_points(((length-depth, 0., -.5*slope), (length-depth, 1., .5*slope)))
    face_b = model.add_face((model.add_line(a[1], b[0]), model.add_line(b[0], b[1]), model.add_line(b[1], a[2]),
                             hinge))
    model._put_entity("face", replace(model.faces[face_a], surface=Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))))
    model._put_entity("face", replace(model.faces[face_b],
                                      surface=Plane((length, .5, 0.), (-1., 0., 0.), (0., 1., slope))))
    return model, (model.handle("face", face_a), model.handle("face", face_b))


SCALE_REFUSAL = "ill_conditioned_plane_plane; near_coplanar_scale_above_absolute_tolerance"
UNRESOLVED = "ill_conditioned_plane_plane; near_coplanar_overlap_unresolved"
FENCE_REFUSAL = "ill_conditioned_plane_plane; near_coplanar_coordinates_above_fence"
# A certified overlap or a typed unresolved refusal: both refuse the pair (decision 15 round 1).
REFUSED_OVERLAP = "positive-area overlap requires an explicit ownership policy|" + UNRESOLVED
# Zone contract (owner decisions 20-22, 2026-11-20): typed refusals.
CONTACT = "ill_conditioned_plane_plane; near_coplanar_contact_off_shared_edge"
SHORT = "ill_conditioned_plane_plane; near_coplanar_segment_shorter_than_zone"
SAME_SIDE = "ill_conditioned_plane_plane; near_coplanar_same_side_of_shared_edge"
OVERLAP = "positive-area overlap requires an explicit ownership policy"


def test_small_plate_folded_over_a_very_large_one_refuses():
    # Codex r2 H (planned silently on c01c9ad): a 0.5 m^2 fold over a 1e7 m plate. Under option A the rule
    # does not apply above the absolute-tolerance regime at all.
    model, faces = _hinged_fold(1.0e7, .5)
    with pytest.raises(GeometryError, match=SCALE_REFUSAL):
        _plan(model, faces)


@pytest.mark.parametrize("depth", (.5, 1.0e-4))
def test_fold_inside_the_absolute_tolerance_regime_refuses_typed(depth):
    # Zone contract: the folded plate's edges overlap A's from the shared vertices on (rule 2 refuses first).
    model, faces = _hinged_fold(900., depth)  # diagonal < 1000 m: absolute tolerance regime
    with pytest.raises(GeometryError, match=CONTACT):
        _plan(model, faces)


def _codex_r3_cap(length, width, *, folded=True, curved=True, slope=SINE_IN_BAND):
    """Codex r3 H1 construction: A = [0,1]x[0,length]; B shares A's edge x=0 (y in [0,1]) and is a half
    ellipse of depth ``width`` (or a rectangle) folded over A (``folded``) or away from it."""
    import math
    from anygeometry.entities import OrientedEdge as OE
    from anygeometry.exact_curves import EllipticArc
    model = GeometryModel()
    a, b, c, d, e = model.add_points(((0, 0, 0), (0, 1, 0), (0, length, 0), (1, length, 0), (1, 0, 0)))
    hinge = model.add_line(a, b)
    sign = 1 if folded else -1
    face_a = model.add_face_from_loop(
        (*[OE(model.add_line(p, q), True) for p, q in ((b, c), (c, d), (d, e), (e, a))], OE(hinge, True)),
        surface=Plane((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    if curved:
        curve = EllipticArc((0., .5, 0.), (0., .5, 0.), (sign*width, 0., sign*width*slope), 0., math.pi)
        middle = model.add_point(*curve.evaluate(.5))
        paths = (model.add_curve(b, middle, curve.subcurve(0., .5)),
                 model.add_curve(middle, a, curve.subcurve(.5, 1.)))
    else:
        f, g = model.add_points(((sign*width, 1, sign*width*slope), (sign*width, 0, sign*width*slope)))
        paths = (model.add_line(b, f), model.add_line(f, g), model.add_line(g, a))
    face_b = model.add_face_from_loop((OE(hinge, True), *(OE(path, True) for path in paths)),
                                      surface=Plane((0, 0, 0), (1, 0, slope), (0, 1, 0)))
    return model, (model.handle("face", face_a), model.handle("face", face_b))


@pytest.mark.parametrize("length,width,folded,curved", ((1e7, .5, True, True), (1e7, 1e-4, True, True),
                                                        (1e8, 1e-3, True, False), (1e7, 1e-4, False, True)))
def test_codex_r3_caps_beside_a_1e7_m_plate_refuse_above_the_absolute_tolerance(length, width, folded, curved):
    # Planned and applied on e3ffc45 (the folded ones with a missed overlap).
    model, faces = _codex_r3_cap(length, width, folded=folded, curved=curved)
    with pytest.raises(GeometryError, match=SCALE_REFUSAL):
        _plan(model, faces)


def test_codex_r3_two_1e7_m_plates_with_a_local_overlap_refuse_above_the_absolute_tolerance():
    # Codex r3 H2 (planned + applied a false weld on e3ffc45): both faces span 1e7 m, overlap 0.5 m^2.
    from anygeometry.entities import OrientedEdge as OE
    model = GeometryModel()
    length, slope = 1e7, SINE_IN_BAND
    a, b, c, d, e, f, g, h, i = model.add_points(
        ((0, 0, 0), (0, 1, 0), (0, length, 0), (1, length, 0), (1, 0, 0), (.5, 1, .5*slope),
         (.5, -length, .5*slope), (-.5, -length, -.5*slope), (-.5, 0, -.5*slope)))
    hinge = model.add_line(a, b)

    def lines(points):
        return tuple(OE(model.add_line(x, y), True) for x, y in zip(points, points[1:]))
    face_a = model.add_face_from_loop((OE(hinge, True), *lines((b, c, d, e, a))),
                                      surface=Plane((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    face_b = model.add_face_from_loop((OE(hinge, True), *lines((b, f, g, h, i, a))),
                                      surface=Plane((0, 0, 0), (1, 0, slope), (0, 1, 0)))
    with pytest.raises(GeometryError, match=SCALE_REFUSAL):
        _plan(model, (model.handle("face", face_a), model.handle("face", face_b)))


def _lifted_trace(gap, residual, tilt=1.1e-10, curved=False):
    """L-shaped A (curved left side) and square B in its notch share the hinge x=1; A's notch top edge and
    B's top edge are separate edges, B's far corner ``gap`` lower, the two far corners ``residual`` above /
    below z=0."""
    model = GeometryModel()
    a = model.add_points(((0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, residual), (2, 2, 0), (0, 2, 0)))
    hinge = model.add_line(a[1], a[2])
    via = model.add_point(-.3, 1, 0)
    face_a = model.add_face((model.add_line(a[0], a[1]), hinge, model.add_line(a[2], a[3]),
                             model.add_line(a[3], a[4]), model.add_line(a[4], a[5]),
                             model.add_arc(a[5], via, a[0]) if curved else model.add_line(a[5], a[0])))
    b = model.add_points(((2, 0, 0), (2, 1-gap, -residual)))
    face_b = model.add_face((model.add_line(a[1], b[0]), model.add_line(b[0], b[1]), model.add_line(b[1], a[2]),
                             hinge))
    model._put_entity("face", replace(model.faces[face_a], surface=Plane((0, 0, 0), (1, 0, 0), (0, 1, 0))))
    model._put_entity("face", replace(model.faces[face_b], surface=Plane((1.5, .5, 0), (1, 0, 0), (0, 1, tilt))))
    return model, (model.handle("face", face_a), model.handle("face", face_b)), hinge


def test_trace_lifted_off_the_second_trim_refuses_typed():
    # Codex r3 M: a 0.9 nm in-plane gap is accepted on the projected curves, but the lifted source trace is
    # 1.08 nm from B's stored edge (> 1 nm). On e3ffc45 B's arrangement refused with an unresolved bridge.
    model, faces, _ = _lifted_trace(0.9e-9, 0.3e-9)
    # Zone contract: A's notch top edge and B's top edge are separate edges within t (rule 2).
    with pytest.raises(GeometryError, match=CONTACT):
        _plan(model, faces)


def test_trace_beside_an_arc_refuses_unresolved():
    model, faces, _ = _lifted_trace(0., 0., curved=True)
    with pytest.raises(GeometryError, match=UNRESOLVED):
        _plan(model, faces)


def test_duplicate_geometric_edge_beside_the_shared_edge_refuses():
    # Was test_trace_on_both_stored_trims_is_emitted (joined, both edges traced). Zone contract (decision 20): a
    # second, untagged geometric abutment (A's notch top edge and B's top edge, separate model edges at the same
    # position) is contact off the shared edge and refuses typed, as touching along distinct edges does (test
    # above, decision 15 M2).
    model, faces, hinge = _lifted_trace(0., 0.)
    with pytest.raises(GeometryError, match=CONTACT):
        _plan(model, faces)


@pytest.mark.parametrize("folded", (False, True))
def test_caps_with_the_first_arc_off_its_own_flat_support_classify_in_the_common_plane(folded):
    # Codex r2 M: the first face's stored arc is tilted 2e-10 rad off its flat support; classification
    # must still happen in one plane (folded -> overlap refusal, abutting -> joined on both diameter edges).
    model, faces, diameter = _caps(2.0e-10, folded=folded, first_arc_tilt=2.0e-10)
    with pytest.raises(GeometryError, match=REFUSED_OVERLAP if folded else UNRESOLVED):
        _plan(model, faces)


def test_distance_bound_is_exact_for_a_nearly_degenerate_stored_chart():
    # Codex r1 H1: the float normal of nearly parallel chart axes loses digits; the bound must not.
    from fractions import Fraction
    from anygeometry.arrangement_geometry import LinePath
    from anygeometry.batch_intersections import _plane_distance_bound
    u, v = (1., 0., 0.), (1., 1e-9, 3e-17)
    plane = Plane((0., 0., 0.), u, v)
    point = (0.3, 0.7, 1.1e-9)
    F = [Fraction(x) for x in u], [Fraction(x) for x in v], [Fraction(x) for x in point]
    n = (F[0][1]*F[1][2]-F[0][2]*F[1][1], F[0][2]*F[1][0]-F[0][0]*F[1][2], F[0][0]*F[1][1]-F[0][1]*F[1][0])
    exact_squared = sum(a*b for a, b in zip(n, F[2]))**2 / sum(a*a for a in n)
    bound = _plane_distance_bound(LinePath(point, point), plane)
    assert Fraction(bound)**2 >= exact_squared
    assert Fraction(bound)**2 <= exact_squared * (1 + Fraction(1, 10**12))


# Boundary bounds: |n.(p-o)| over each supported curve family, sound (never below the true maximum).

def _true_max(curve, normal, origin):
    t = np.linspace(0., 1., 2001)
    points = np.asarray([curve.evaluate(float(x)) for x in t], float)
    return float(np.max(np.abs((points - origin) @ normal)))


def test_distance_bound_covers_line_arc_bezier_and_rational_nurbs():
    from anygeometry.arrangement_geometry import BezierPath, LinePath
    from anygeometry.batch_intersections import _plane_distance_bound
    from anygeometry.exact_curves import EllipticArc
    from anygeometry.nurbs import NurbsCurve
    normal, origin = np.array([0., 0.6, 0.8]), np.array([.1, .2, .3])
    curves = (LinePath((0., 0., 0.), (1., 2., 3.)),
              EllipticArc((0., 0., 1.), (2., 0., .5), (0., 1., -.25), .3, 2.),
              BezierPath(((0., 0., 0.), (1., 3., -1.), (2., 0., 2.))),
              NurbsCurve(2, (0., 0., 0., 1., 1., 1.), ((1., 0., 0.), (1., 1., 0.), (0., 1., 1.)), (1., .7, 1.)))
    for curve in curves:
        bound = _plane_distance_bound(curve, Plane(tuple(origin), (1., 0., 0.), (0., .8, -.6)))
        assert bound is not None and bound >= _true_max(curve, normal, origin)


def test_distance_bound_refuses_unknown_curve_families():
    from anygeometry.batch_intersections import _plane_distance_bound

    class Unknown:
        def evaluate(self, t):
            return np.zeros(3)

    assert _plane_distance_bound(Unknown(), Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))) is None


# Owner decision 15 (2026-11-19): the rule's own overlap check, absolute and width-based at the strict
# tolerance (Codex a1 H1/H2), and the fenced tolerance in the polygon step (a1 M1). All planned + applied
# on 940c947.

def _oriented_loop(model, hinge, a, b, points, reverse, arc_pair=None, via=None):
    from anygeometry.entities import OrientedEdge as OE
    chain = [OE(hinge, x == a) if {x, y} == {a, b} else
             OE(model.add_arc(x, via, y) if arc_pair and {x, y} == arc_pair else model.add_line(x, y), True)
             for x, y in zip(points, points[1:] + points[:1])]
    return tuple(OE(item.edge, not item.forward) for item in chain[::-1]) if reverse else tuple(chain)


def _a1_polygon(span, dy, dx, reverse, slope=SINE_IN_BAND, tolerance=None):
    """Codex a1 H1: A = [0,1]x[0,span]; B shares the hinge x=0, y in [0,dy], and its next corner (dx, dy) lies
    inside A, so the two overlap in a triangle of legs dx, dy (area dx*dy/2) next to the hinge end."""
    model = GeometryModel() if tolerance is None else GeometryModel(tolerance=tolerance)
    a, b, c, d, e, f, g, h, i = model.add_points(
        ((0, 0, 0), (0, dy, 0), (0, span, 0), (1, span, 0), (1, 0, 0), (dx, dy, dx*slope),
         (.5, -span, .5*slope), (-.5, -span, -.5*slope), (-.5, 0, -.5*slope)))
    hinge = model.add_line(a, b)
    first = model.add_face_from_loop(_oriented_loop(model, hinge, a, b, [a, b, c, d, e], reverse),
                                     surface=Plane((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    ring = [a, b, f, g, h, i] if dx > 0 else [a, b, *model.add_points(
        ((-.5, dy, -.5*slope), (-.5, -span, -.5*slope), (.5, -span, .5*slope)))]
    second = model.add_face_from_loop(_oriented_loop(model, hinge, a, b, ring, False),
                                      surface=Plane((0, 0, 0), (1, 0, slope), (0, 1, 0)))
    return model, (model.handle("face", first), model.handle("face", second)), hinge


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("span,dy,dx", ((100., 1e-6, 1e-5), (999., 1e-5, 1e-4), (1., 1e-8, 5e-8)))
def test_polygon_overlap_below_the_relative_area_tolerance_refuses(span, dy, dx, reverse):
    # a1 H1: the 1e-11 m^2 overlap at 100 m is below the shared backend's relative area tolerance (1e-10 m^2).
    model, faces, _ = _a1_polygon(span, dy, dx, reverse)
    # Zone contract: B's edge from its corner inside A crosses A's bottom edge (rule 2 refuses first).
    with pytest.raises(GeometryError, match=CONTACT):
        _plan(model, faces)


def test_polygon_pair_without_overlap_still_joins():
    model, faces, hinge = _a1_polygon(100., 1e-6, 0., False)
    plan = _plan(model, faces)
    assert _joined_by(plan, hinge, faces)
    apply_intersections(model, plan, policy="connect")
    assert model.validate_topology() == ()


def _a1_curved(dx, reverse_first, reverse_second, slope=SINE_IN_BAND):
    """Codex a1 H2: A = unit square with an arc side; B shares the hinge x=0, y in [0,.3] and its corner (dx,.3)
    lies inside A: a dx x 0.3 m overlap strip."""
    model = GeometryModel()
    a, b, c, d, e, f, g, h, i, via = model.add_points(
        ((0, 0, 0), (0, .3, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0), (dx, .3, dx*slope), (dx, -1, dx*slope),
         (-1, -1, -slope), (-1, 0, -slope), (1.12, .5, 0)))
    hinge = model.add_line(a, b)
    first = model.add_face_from_loop(_oriented_loop(model, hinge, a, b, [a, b, c, d, e], reverse_first, {d, e}, via),
                                     surface=Plane((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    ring = [a, b, f, g, h, i] if dx > 0 else [a, b, *model.add_points(((-1, .3, -slope), (-1, -1, -slope),
                                                                         (1, -1, slope)))]
    second = model.add_face_from_loop(_oriented_loop(model, hinge, a, b, ring, reverse_second),
                                      surface=Plane((0, 0, 0), (1, 0, slope), (0, 1, 0)))
    return model, (model.handle("face", first), model.handle("face", second)), hinge


@pytest.mark.parametrize("reverse_first", (False, True))
@pytest.mark.parametrize("dx", (25e-9, 40e-9, 1e-3))
def test_curved_thin_overlap_refuses(dx, reverse_first):
    # a1 H2: a 25 nm x 0.3 m strip; the shared route's cell probe steps 32 nm and misses it.
    model, faces, _ = _a1_curved(dx, reverse_first, False)
    with pytest.raises(GeometryError, match=REFUSED_OVERLAP):
        _plan(model, faces)


def test_curved_pair_without_overlap_refuses_unresolved():
    model, faces, _ = _a1_curved(0., False, False)
    with pytest.raises(GeometryError, match=UNRESOLVED):
        _plan(model, faces)


@pytest.mark.parametrize("h, joins", ((5e-9, False), (2e-8, True)))
def test_custom_relative_length_keeps_a_short_shared_hinge(h, joins):
    # a1 M1: with relative_length=1e-8 centimetre faces pass the 1 nm fence, but the shared polygon backend
    # recomputed its tolerance with a 1 m extent floor (10 nm) and dropped the genuine 5 nm hinge. Zone contract
    # (decision 21, c = 4): a shared edge shorter than 2 c t = 8 nm refuses typed (never dropped); 20 nm joins.
    from anygeometry import TolerancePolicy
    from anygeometry.entities import OrientedEdge as OE
    model = GeometryModel(tolerance=TolerancePolicy(relative_length=1e-8))
    w, slope = .01, SINE_IN_BAND
    a, b, c, d, e, f = model.add_points(((0, 0, 0), (0, h, 0), (-w, w, 0), (-w, -w, 0),
                                         (w, -w, w*slope), (w, w, w*slope)))
    hinge = model.add_line(a, b)
    first = model.add_face_from_loop((OE(hinge, True), *(OE(model.add_line(x, y), True)
                                                         for x, y in ((b, c), (c, d), (d, a)))),
                                     surface=Plane((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    second = model.add_face_from_loop((OE(hinge, False), *(OE(model.add_line(x, y), True)
                                                           for x, y in ((a, e), (e, f), (f, b)))),
                                      surface=Plane((0, 0, 0), (1, 0, slope), (0, 1, 0)))
    faces = (model.handle("face", first), model.handle("face", second))
    if joins:
        assert _joined_by(_plan(model, faces), hinge, faces)
    else:
        with pytest.raises(GeometryError, match=SHORT):
            _plan(model, faces)


# Decision 15 round 1 (Codex d15): the class of uncertified tolerances on the edge-joined path. All planned on
# e11c2af.

def _d15_strip(reverse_first, reverse_second, curved):
    """Codex d15 High: A has a 20 nm x 5 mm tab at its hinge x=1.01 and (``curved``) an arc far away; B is the
    strip [1.005, 1.01] x [0, w] folded back over the tab."""
    from anygeometry.entities import OrientedEdge as OE
    slope, w = SINE_IN_BAND, 20e-9
    model = GeometryModel()
    p = model.add_points(((0, 0, 0), (1.01, 0, 0), (1.01, w, 0), (1, w, 0), (1, 1, 0), (0, 1, 0)))
    via = model.add_point(-.1, .5, 0)
    hinge = model.add_line(p[1], p[2])
    first = [OE(model.add_line(p[0], p[1]), True), OE(hinge, True), OE(model.add_line(p[2], p[3]), True),
             OE(model.add_line(p[3], p[4]), True), OE(model.add_line(p[4], p[5]), True),
             OE(model.add_arc(p[5], via, p[0]) if curved else model.add_line(p[5], p[0]), True)]
    q = model.add_points(((1.005, 0, slope*(1.005-1.01)), (1.005, w, slope*(1.005-1.01))))
    second = [OE(model.add_line(q[0], p[1]), True), OE(hinge, True), OE(model.add_line(p[2], q[1]), True),
              OE(model.add_line(q[1], q[0]), True)]

    def reverse(chain):
        return tuple(OE(item.edge, not item.forward) for item in chain[::-1])
    face_a = model.add_face_from_loop(reverse(first) if reverse_first else tuple(first),
                                      surface=Plane((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    face_b = model.add_face_from_loop(reverse(second) if reverse_second else tuple(second),
                                      surface=Plane((1.01, 0, 0), (1, 0, slope), (0, 1, 0)))
    return model, (model.handle("face", face_a), model.handle("face", face_b))


@pytest.mark.parametrize("reverse_first,reverse_second", ((False, False), (False, True), (True, False), (True, True)))
@pytest.mark.parametrize("curved", (True, False))
def test_strip_overlap_dropped_by_the_arrangement_refuses(reverse_first, reverse_second, curved):
    # Codex d15 H: with the arc, arrange_material's 32t probe discarded the overlap cycle and its default
    # conservation slack (t used as an area) accepted the missing 1e-10 m^2; the pair planned.
    model, faces = _d15_strip(reverse_first, reverse_second, curved)
    # Zone contract: straight strips refuse at rule 2 (the strip's edges overlap A's from the hinge vertices).
    with pytest.raises(GeometryError, match=UNRESOLVED if curved else CONTACT):
        _plan(model, faces)


def test_witness_closer_than_the_tolerance_to_an_oblique_edge_refuses_unresolved():
    # Codex d15 M: A is a 1.5 nm sliver (h = 0.75 t) rotated by atan(0.1); the witness chosen on e11c2af was
    # only 0.754 t from the hinge, which the ray-crossing containment (horizontal in A's axis-aligned chart,
    # oblique to the edges) did not see. No point of A has a
    # clearance above t, so no witness can be certified.
    from anygeometry.entities import OrientedEdge as OE
    t, slope = 1e-9, SINE_IN_BAND
    h = .75*t

    def mapped(x, y, z=0.):
        return (x - .1*y, y + .1*x, z)
    model = GeometryModel()
    a = model.add_points((mapped(0, -h), mapped(0, h), mapped(1, h), mapped(1, -h)))
    b = model.add_points((mapped(1, h + .5, .5*slope), mapped(0, h + .5, .5*slope)))
    hinge = model.add_line(a[1], a[2])
    first = model.add_face_from_loop((OE(model.add_line(a[0], a[1]), True), OE(hinge, True),
                                      OE(model.add_line(a[2], a[3]), True), OE(model.add_line(a[3], a[0]), True)),
                                     surface=Plane((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    second = model.add_face_from_loop((OE(hinge, False), OE(model.add_line(a[1], b[1]), True),
                                       OE(model.add_line(b[1], b[0]), True), OE(model.add_line(b[0], a[2]), True)),
                                      surface=Plane(mapped(0, h), (1, .1, 0), (-.1, 1, slope)))
    # Zone contract: A's 1.5 nm sides at the shared vertices are shorter than 2 c t (decision 21).
    with pytest.raises(GeometryError, match=SHORT):
        _plan(model, (model.handle("face", first), model.handle("face", second)))


def _far_squares(offset, tilt):
    """Two unit squares sharing an edge, ``offset`` metres from the origin; A lies in the plane z = tilt*(y-offset)
    and B leaves the shared edge with a slope larger by 5e-10 (sine in the band)."""
    from anygeometry.entities import OrientedEdge as OE
    X, eps = offset, 5e-10
    model = GeometryModel()
    a, b, c, d, e, f = model.add_points(((X, X, 0), (X+1, X, 0), (X+1, X+1, tilt), (X, X+1, tilt),
                                         (X+1, X+2, 2*tilt + eps), (X, X+2, 2*tilt + eps)))
    hinge = model.add_line(d, c)
    first = model.add_face_from_loop((OE(model.add_line(a, b), True), OE(model.add_line(b, c), True),
                                      OE(hinge, False), OE(model.add_line(d, a), True)),
                                     surface=Plane((X, X, 0), (1, 0, 0), (0, 1, tilt)))
    second = model.add_face_from_loop((OE(hinge, True), OE(model.add_line(c, e), True),
                                       OE(model.add_line(e, f), True), OE(model.add_line(f, d), True)),
                                      surface=Plane((X, X+1, tilt), (1, 0, 0), (0, 1, tilt + eps)))
    return model, (model.handle("face", first), model.handle("face", second)), hinge


@pytest.mark.parametrize("offset", (0., 1e4, 6e4))
def test_far_from_the_origin_an_exact_projection_still_joins(offset):
    # Inside the absolute-coordinate fence (decision 18: every coordinate below 2^16 m) the rule decides.
    model, faces, hinge = _far_squares(offset, 0.)
    assert _joined_by(_plan(model, faces), hinge, faces)


def test_far_from_the_origin_beyond_the_coordinate_fence_refuses():
    # Decision 18 (recorded fail-closed contract change): this 1e7 m pair joined on 984720f (then a case of
    # the test above), but the proven bound does not hold there (the arrangement's output rounding alone is
    # up to sqrt(3)/2 ulp = 1.6t, outside the k = 7.4 budget), so it now refuses typed.
    model, faces, _ = _far_squares(1e7, 0.)
    with pytest.raises(GeometryError, match=FENCE_REFUSAL):
        _plan(model, faces)


@pytest.mark.parametrize("offset, joins", [
    (65533., True),                               # largest coordinate 65535 m: inside
    (math.nextafter(65534., 0.), True),           # largest coordinate one ulp below 2^16 m: inside
    (65534., False),                              # largest coordinate exactly 2^16 m: the fence is strict
    (65535., False),                              # largest coordinate 65537 m: outside
])
def test_coordinate_fence_boundary(offset, joins):
    from anygeometry.batch_intersections import EDGE_JOINED_MAX_COORDINATE
    assert EDGE_JOINED_MAX_COORDINATE == 2.**16
    model, faces, hinge = _far_squares(offset, 0.)
    if joins:
        assert _joined_by(_plan(model, faces), hinge, faces)
    else:
        with pytest.raises(GeometryError, match=FENCE_REFUSAL):
            _plan(model, faces)


def test_projection_rounding_far_from_the_origin_refuses_at_the_fence():
    # Decision 15 round 1: at 1e7 m on a 45 degree plane the float projection could not apply B's ~0.18 nm
    # in-plane correction. Zone contract: no float projection is used; 1e7 m is beyond the kept 2^16 m fence
    # (decision 22), which now refuses first.
    model, faces, _ = _far_squares(1e7, 1.)
    with pytest.raises(GeometryError, match=FENCE_REFUSAL):
        _plan(model, faces)


# Owner decision 16 (2026-11-19, Codex d15r1): the edge-joined path's own exact straight-line predicates. All
# planned on c320770.

def _ring_face(model, points, surface, shared=None):
    """A face on the closed ring ``points`` (model point ids); ``shared`` = {(p, q): edge} reuses edges."""
    from anygeometry.entities import OrientedEdge as OE
    shared = {} if shared is None else shared
    chain = []
    for p, q in zip(points, points[1:] + points[:1]):
        if (p, q) in shared:
            chain.append(OE(shared[(p, q)], True))
        elif (q, p) in shared:
            chain.append(OE(shared[(q, p)], False))
        else:
            shared[(p, q)] = model.add_line(p, q)
            chain.append(OE(shared[(p, q)], True))
    return model.add_face_from_loop(tuple(chain), surface=surface), shared


def test_witness_past_an_edge_end_is_not_a_boundary_point():
    # d15r1 H (item 10): the shared containment's level-edge test compared the line PARAMETER with [-t, 1+t],
    # i.e. up to 8t past the end of an 8 m edge; a witness 6.75t past B's edge counted as boundary (outside)
    # and a 4.5t overlap planned. Every point of the projected A lies in B. Decision 17: A is only 4.5t wide
    # (< k t), so the contract allows a join; the rule refuses it (no witness clears 3.25t) - typed either way.
    t, L = 1e-9, 8.
    W, k = 9*t, 1.1e-10
    model = GeometryModel()
    a = model.add_points(((L+W/2, -4*t, 0), (L+W, -4*t, 0), (L+W, .5, 0), (L+W/2, .5, 0)))
    face_a, shared = _ring_face(model, list(a), Plane((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    b = model.add_points(((0, 0, k*(-L-W)), (L, 0, k*(-W)), (L, -.5, k*(-W)), (L+W, -.5, 0)))
    face_b, _ = _ring_face(model, [*b, a[1], a[2], model.add_point(0, .5, k*(-L-W))],
                           Plane((L+W, 0, 0), (1, 0, k), (0, 1, 0)), shared)
    # Zone contract: A's 4.5t sides at the shared vertices are shorter than 2 c t (decision 21).
    with pytest.raises(GeometryError, match=SHORT):
        _plan(model, (model.handle("face", face_a), model.handle("face", face_b)))


def test_nearly_degenerate_stored_chart_is_decided_exactly():
    # Was test_projection_drift_is_measured_against_the_stored_plane (d15r1 M item 3: the rebuilt float chart's
    # drift of 3.6e-9 m > t/8 refused unresolved). Zone contract: no float projection or chart is built; the
    # pair is decided exactly on the stored plane and the two faces lie on opposite sides of the shared edge.
    U = (0.7071067811865475, 0.7071067811865475, 0.)
    V = (-0.5773502701566411, 0.5773502701566411, 0.577350267255595)
    W = (-0.5773502699525169, 0.5773502699525169, 0.5773502676638433)

    def add(*vectors):
        return tuple(sum(component) for component in zip(*vectors))

    def neg(vector):
        return tuple(-x for x in vector)
    model = GeometryModel()
    o, u, mv, umv, w, uw = model.add_points(((0., 0., 0.), U, neg(V), add(U, neg(V)), W, add(U, W)))
    face_a, shared = _ring_face(model, [mv, umv, u, o],
                                Plane((0, 0, 0), (1.3, 1.3, 0), (1.30000001, 1.30000003, 1e-8)))
    face_b, _ = _ring_face(model, [o, u, uw, w], Plane((0, 0, 0), U, W), shared)
    faces = (model.handle("face", face_a), model.handle("face", face_b))
    assert _joined_by(_plan(model, faces), shared[(u, o)], faces)


def test_short_collinear_trace_far_out_is_kept_with_both_owners():
    # d15r1 M (item 12): _clip_intervals merged the station at 0.999999999999998 into 1.0 (a fixed 32 eps in
    # parameter space) and dropped the 20.49t trace where B's separate edge (L-w, 0)-(L, 0) lies on A's edge.
    from anygeometry import TolerancePolicy
    L, w, k = 1e7, 20e-9, 5e-10
    model = GeometryModel(tolerance=TolerancePolicy(relative_length=1e-18))
    p = model.add_points(((0, 0, 0), (L, 0, 0), (L+1, 0, 0), (L+1, 1, 0), (0, 1, 0)))
    face_a, shared = _ring_face(model, list(p), Plane((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    q = model.add_points(((L-w, 0, 0), (L-w, -1, -k), (L+1, -1, -k)))
    face_b, shared = _ring_face(model, [p[2], p[1], q[0], q[1], q[2]], Plane((L, 0, 0), (1, 0, 0), (0, 1, k)), shared)
    faces = (model.handle("face", face_a), model.handle("face", face_b))
    # Decision 17: 1e7 m is outside the proven band (faces up to 1000 m), so the rule refuses typed; it must
    # never plan with the trace dropped.
    try:
        plan = _plan(model, faces)
    except GeometryError as error:
        assert "near_coplanar_scale_above_absolute_tolerance" in str(error)
        return
    edge = shared[(p[1], q[0])]
    owners = tuple(sorted(face.id for face in faces))
    assert all(any(tuple(sorted(path.owners)) == owners and path.source_edge in (edge, shared[(p[0], p[1])])
                   and path.source_edge != shared[(p[1], p[2])] for path in arrangement.paths)
               for arrangement in plan.arrangements)


# Owner decision 17 (2026-11-19, Codex d16): proven-bound contract. All planned (or shortened a trace) on 52047ec.

def _xy_pair(first_ring, second_ring, shared_xy, second_plane, tolerance=None, slope=1.1e-10):
    """A on z=0 and B on ``second_plane`` (z given by ``slope`` times the second plane's own coordinate via
    ``second_plane``), both as ordered XY rings; ``shared_xy`` is the reused edge (two XY points, in A's ring)."""
    from anygeometry import TolerancePolicy
    model = GeometryModel() if tolerance is None else GeometryModel(tolerance=TolerancePolicy(relative_length=tolerance))
    ids = {}

    def point(xy, z):
        key = tuple(float(v) for v in xy)
        if key not in ids:
            ids[key] = model.add_point(key[0], key[1], z)
        return ids[key]
    a = [point(xy, 0.) for xy in first_ring]
    face_a, shared = _ring_face(model, a, Plane((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    origin, u, v, height = second_plane
    b = [point(xy, height(*xy)) for xy in second_ring]
    keep = {key: edge for key, edge in shared.items()
            if {ids[tuple(map(float, shared_xy[0]))], ids[tuple(map(float, shared_xy[1]))]} == set(key)}
    face_b, _ = _ring_face(model, b, Plane(origin, u, v), keep)
    return model, (model.handle("face", face_a), model.handle("face", face_b)), ids


SCALE_OR_OVERLAP = ("positive-area overlap requires an explicit ownership policy|" + UNRESOLVED + "|" +
                    "near_coplanar_scale_above_absolute_tolerance")


def test_d16_six_t_overlap_on_a_1e7_m_edge_refuses():
    # Codex d16 H1: _clip_line_exact's shared containment has an 8 eps PARAMETER window (17.76t on a 1e7 m
    # edge) and dropped both cuts bounding a 6t overlap; the pair joined. Decision 17: beyond the proven
    # absolute band (1000 m) the rule refuses; the containment in this call is exact parity anyway.
    t, L, k = 1e-9, 1e7, 1.1e-10
    model, faces, _ = _xy_pair([(0, 0), (0, -L), (1, -L), (1, 1-L), (1, 0)],
                               [(1, -L), (1, 1-L), (1.5, 1-L), (1.5, -12*t), (.3, -12*t), (.3, -6*t), (2, -6*t), (2, -L)],
                               ((1, -L), (1, 1-L)), ((1, 0, 0), (1, 0, k), (0, 1, 0), lambda x, y: k*(x-1)),
                               tolerance=1e-18)
    with pytest.raises(GeometryError, match=SCALE_OR_OVERLAP):
        _plan(model, faces)


def test_d16_trace_at_1e8_m_is_never_silently_shortened():
    # Codex d16 M1: float parameters on the stored source line emitted [-14.9t, 0] of the exact [-20t, 0].
    # Decision 17: refused beyond the proven band, and emitted from exact world end points inside it.
    t, L, k = 1e-9, 1e8, 1.1e-10
    model, faces, ids = _xy_pair([(-L, 0), (0, 0), (1, 0), (1, 1), (-L, 1)],
                                 [(1, 0), (0, 0), (-20*t, 0), (-20*t, -1), (1, -1)],
                                 ((0, 0), (1, 0)), ((0, 0, 0), (1, 0, 0), (0, 1, k), lambda x, y: k*y),
                                 tolerance=1e-18)
    try:
        plan = _plan(model, faces)
    except GeometryError as error:
        assert "near_coplanar_scale_above_absolute_tolerance" in str(error)
        return
    owners = tuple(sorted(face.id for face in faces))
    spans = [path.curve for arrangement in plan.arrangements for path in arrangement.paths
             if tuple(sorted(path.owners)) == owners and min(path.curve.start[0], path.curve.end[0]) < -1e-9]
    assert spans and all(min(curve.start[0], curve.end[0]) <= -20*t for curve in spans)


def test_emitted_trace_is_the_stored_shared_edge():
    # Was test_trace_is_emitted_from_exact_world_end_points (_emit_trace, removed). Zone contract: the trace is
    # the shared model edge's own stored segment (its stored vertex doubles), in both face orders.
    from anygeometry import batch_intersections
    from anygeometry.material_arrangement import MaterialDomain
    model, faces, hinge = _far_squares(6e4, 0.)
    edge = model.edges[hinge]
    expected = (tuple(model.vertex_position(edge.start)), tuple(model.vertex_position(edge.end)))
    for pair in (faces, faces[::-1]):
        first, second = (MaterialDomain.from_model(model, face.id) for face in pair)
        traces = batch_intersections._pair_paths_exact(first, second, model, 1e-9, lambda: None)
        assert [(tuple(t.start), tuple(t.end)) for t in traces] == [expected]


def _d16_band(width):
    """Codex d16 H2: B is a comb whose slot band y in [-width/2, width/2] crosses A completely; the projected
    overlap is [0, 1] x [-width/2, width/2]."""
    h = width/2
    k = 1.1e-10
    return _xy_pair([(0, 0), (0, 1), (1, 1), (1, 0), (1, -.6), (1, -.8), (1, -1), (0, -1)],
                    [(1, -.8), (1, -.6), (1.5, -.6), (1.5, -h), (-1, -h), (-1, h), (2, h), (2, -.8)],
                    ((1, -.8), (1, -.6)), ((1, 0, 0), (1, 0, k), (0, 1, 0), lambda x, y: k*(x-1)))


@pytest.mark.parametrize("factor", (1.2, 1.8, 7.0, 7.8, 22.))
def test_any_overlap_refuses(factor):
    # Replaces test_overlap_just_above_k_t_refuses, test_overlap_within_the_tolerance_model_joins (1.8t joined) and
    # test_overlap_just_below_k_t_is_joined_or_refused_typed. Zone contract (decision 20): every overlap refuses
    # typed, however thin (the k*t allowance of decision 17 is gone).
    model, faces, _ = _d16_band(factor*1e-9)
    with pytest.raises(GeometryError, match=CONTACT + "|" + OVERLAP):
        _plan(model, faces)








def _sloped_contact(length):
    """Codex d16 M2: B's edge from (0,-g) to (.5,-g-.5m) stays within t of A's bottom edge for ``length``
    (from x = 0); the hinge is (2,0)-(3,0)."""
    t, m, k = 1e-9, 1e-4, 1.1e-10
    g = t - length*m
    return _xy_pair([(-1, 0), (1, 0), (2, 0), (3, 0), (3, 1), (-1, 1)],
                    [(3, 0), (2, 0), (2, -.1), (.5, -.1), (.5, -g-.5*m), (0, -g), (0, -1), (3, -1)],
                    ((2, 0), (3, 0)), ((0, 0, 0), (1, 0, 0), (0, 1, k), lambda x, y: k*y))


@pytest.mark.parametrize("length", (3.5e-9, 9e-9))
def test_near_tangent_contact_off_the_shared_edge_refuses(length):
    # Replaces test_near_tangent_contact_shorter_than_k_t_is_a_point_contact (3.5t joined) and
    # ..._longer_than_k_t_refuses_unresolved. Zone contract: any contact within t away from the shared vertices
    # refuses typed, however short.
    model, faces, _ = _sloped_contact(length)
    with pytest.raises(GeometryError, match=CONTACT):
        _plan(model, faces)





def test_faces_beyond_the_proven_band_refuse_scale():
    model, faces, _ = _far_squares(0., 0.)
    plan = _plan(model, faces)
    assert len(plan.arrangements) == 2
    from anygeometry import TolerancePolicy
    big = GeometryModel(tolerance=TolerancePolicy(relative_length=1e-15))
    from anygeometry.entities import OrientedEdge as OE
    L, s = 1500., SINE_IN_BAND
    a, b, c, d, e, f = big.add_points(((0, 0, 0), (L, 0, 0), (L, 1, 0), (0, 1, 0), (L, 2, s), (0, 2, s)))
    hinge = big.add_line(d, c)
    first = big.add_face_from_loop((OE(big.add_line(a, b), True), OE(big.add_line(b, c), True), OE(hinge, False),
                                    OE(big.add_line(d, a), True)), surface=Plane((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    second = big.add_face_from_loop((OE(hinge, True), OE(big.add_line(c, e), True), OE(big.add_line(e, f), True),
                                     OE(big.add_line(f, d), True)), surface=Plane((0, 1, 0), (1, 0, 0), (0, 1, s)))
    with pytest.raises(GeometryError, match=SCALE_REFUSAL):
        _plan(big, (big.handle("face", first), big.handle("face", second)))


def test_far_comb_overlap_refuses():
    # Codex d17 counterexample (log), reproduced on 984720f: B's notch intrudes about 60t into A; it refused at
    # 0 m and 2^28 m from the origin but was accepted at 2^29 m (ulp 119t) although 60t > k*t: the witness
    # search reads the arrangement's rounded output curves (47.7t off there). Decision 18: fence refusal.
    import math
    t, k, X = 1e-9, 1.1e-10, 2.**29
    U, Y = math.ulp(X), 60*t
    model, faces, _ = _xy_pair(
        [(X-1, 0), (X+1, 0), (X+2, 0), (X+2, 1), (X-1, 1)],
        [(X+2, 0), (X+1, 0), (X+1, -.1), (X+2*U, -.1), (X+2*U, -Y), (X, Y), (X-U, Y), (X+U, -Y*3/7), (X+U, -1),
         (X+2, -1)],
        ((X+1, 0), (X+2, 0)), ((X, 0, 0), (1, 0, 0), (0, 1, k), lambda x, y: k*y))
    with pytest.raises(GeometryError, match=FENCE_REFUSAL):
        _plan(model, faces)


def _toothed_contact(offset, teeth, swap=False):
    """Codex d18 H1: B's boundary zigzags below A's bottom edge (vertices 0.9t apart, alternately 3t and 0.05t
    below it). Every single segment pair is a short point contact, but the tips (within t of A's edge) and
    everything between them form one untraced contact (teeth - 1)*1.8t long. ``swap`` gives the toothed ring
    to the first face instead."""
    t, k, X = 1e-9, 1.1e-10, offset
    a = [(-1, 0), (1, 0), (2, 0), (3, 0), (3, 1), (-1, 1)]
    b = [(3, 0), (2, 0), (2, -.1), *[(i*.9*t, -(3 if i % 2 == 0 else .05)*t) for i in range(2*teeth, -1, -1)],
         (-.1, -.1), (-.1, -1), (3, -1)]
    move = lambda ring: [(X + x, y) for x, y in ring]
    rings = (move(b), move(a)) if swap else (move(a), move(b))
    return _xy_pair(*rings, ((X + 2, 0), (X + 3, 0)), ((X, 0, 0), (1, 0, 0), (0, 1, k), lambda x, y: k*y))


@pytest.mark.parametrize("swap", (False, True))
@pytest.mark.parametrize("offset", (0., math.nextafter(65533., 0.)))
def test_toothed_contact_longer_than_k_t_refuses(offset, swap):
    # Codex d18 H1 (inside the fence): contacts are judged as their union along each loop of both faces, not
    # pair by pair; a 10.8t untraced contact refuses unresolved.
    model, faces, _ = _toothed_contact(offset, 7, swap)
    with pytest.raises(GeometryError, match=CONTACT):  # zone contract: contact off the shared edge
        _plan(model, faces)


@pytest.mark.parametrize("swap", (False, True))
def test_toothed_contact_shorter_than_k_t_refuses(swap):
    # Was ..._joins (3.6t < k t joined). Zone contract: point contact off the shared edge refuses typed.
    model, faces, ids = _toothed_contact(0., 2, swap)
    with pytest.raises(GeometryError, match=CONTACT):
        _plan(model, faces)


def _zigzag_over_edge(segments):
    """Codex d18 r2: A's bottom is a zigzag of ``segments`` pieces 1.2t long, 0.33t-1.22t above B's straight top
    edge (no overlap). Seen from B every pair is a trace, seen from A every pair is a point contact, so no trace
    is emitted, while B's edge stays within t of A's boundary for about 1.2t*segments + 2t."""
    t, k = 1e-9, 1.1e-10
    amp, off = 4.419648357655867e-10, 7.759376765497454e-10
    zig = [(i*1.2*t, (amp if i % 2 else -amp) + off) for i in range(segments + 1)]
    a = [(-1, .1), (-6*t, .1), *zig, (13*t, .1), (1, .1), (2, 0), (3, 0), (3, 1), (-1, 1)]
    b = [(3, 0), (2, 0), (2, -.1), (12*t, -.1), (12*t, 0), (-4*t, 0), (-4*t, -1), (3, -1)]
    return _xy_pair(a, b, ((2, 0), (3, 0)), ((0, 0, 0), (1, 0, 0), (0, 1, k), lambda x, y: k*y))


@pytest.mark.parametrize("reverse", (False, True))
def test_contact_hidden_by_a_reverse_only_trace_refuses(reverse):
    # Codex d18 r2: the reverse pass skipped pairs it classified as traces itself, though no trace was emitted
    # for them; it now skips exactly the forward pass's traces (accepted on 8b37822 in both orders).
    model, faces, _ = _zigzag_over_edge(6)
    with pytest.raises(GeometryError, match=CONTACT):  # zone contract: contact off the shared edge
        _plan(model, faces[::-1] if reverse else faces)


def test_short_zigzag_contact_refuses():
    # Was ..._joins (about 4.4t < k t joined). Zone contract: contact off the shared edge refuses typed.
    model, faces, _ = _zigzag_over_edge(2)
    with pytest.raises(GeometryError, match=CONTACT):
        _plan(model, faces)


def _fringe_past_trace(teeth, shift=.8e-9, dx=.82e-9):
    """Codex d18 r3: B's boundary has a wedge whose two arms are both traced on A's bottom edge and meet in a
    tip 0.05t below it at x = 0, so the traced pairs' contact sets reach about 0.9987t past the emitted trace
    end. To the right, teeth (tips 0.05t below the edge, valleys 3t below, ``dx`` apart from ``shift``) carry
    point contacts that start about 0.52t from the trace end. The untraced run is the fringe plus the teeth."""
    t, k = 1e-9, 1.1e-10
    a = [(-1, 0), (1, 0), (2, 0), (3, 0), (3, 1), (-1, 1)]
    zig = [(shift + i*dx, -(3 if i % 2 == 0 else .05)*t) for i in range(2*teeth, 0, -1)]
    b = [(3, 0), (2, 0), (2, -.1), *zig, (.1*t, -3*t), (-1.1*t, -t), (0., 0.),
         (-3*t, 0.), (-3*t, -5*t), (-.1, -.1), (-.1, -1), (3, -1)]
    return _xy_pair(a, b, ((2, 0), (3, 0)), ((0, 0, 0), (1, 0, 0), (0, 1, k), lambda x, y: k*y))


@pytest.mark.parametrize("reverse", (False, True))
def test_contact_past_a_trace_end_counts_as_untraced(reverse):
    # Codex d18 r3 (decision 19): untraced contact is the union of all exact contact sets, traced pairs included,
    # minus the emitted trace intervals; the fringe past the wedge's trace end plus the teeth is about 7.6t >= k t
    # (accepted on b02e41d, where the traced pairs' contact sets were left out of the union).
    model, faces, _ = _fringe_past_trace(4)
    with pytest.raises(GeometryError, match=CONTACT):  # zone contract: contact off the shared edge
        _plan(model, faces[::-1] if reverse else faces)


def test_contact_past_a_trace_end_shorter_than_k_t_refuses():
    # Was ..._joins (about 2.7t < k t joined, with two wedge traces). Zone contract: the wedge's arms touch A's
    # bottom edge away from the shared vertices; refused typed.
    model, faces, _ = _fringe_past_trace(1)
    with pytest.raises(GeometryError, match=CONTACT):
        _plan(model, faces)


@pytest.mark.parametrize("reverse", (False, True))
def test_seam_candidate_does_not_hide_contact_on_a_physical_edge(reverse):
    # Codex d19 High (strict xfail on the parked 078127d; joined with no trace there and on b02e41d): B's
    # physical edge runs within 0.6 nm of A's bottom edge for about 2 m. Zone contract: contact off the shared
    # edge, refused typed in both orders.
    from anygeometry import EntityRef
    k = 1.1e-10
    model, faces, _ = _xy_pair([(0, 0), (3, 0), (3, 1), (0, 1)],
                               [(0, 0), (3, 0), (1, -.6e-9), (-.2, -.7), (-1, .1)],
                               ((0, 0), (3, 0)), ((0, 0, 0), (1, 0, 0), (0, 1, k), lambda x, y: k*y))
    model.tag(EntityRef("edge", 1), "intersection_decomposition_seam")
    model.add_sheet([face.id for face in faces], orientations=(1, -1))
    assert model.validate_topology() == ()
    with pytest.raises(GeometryError, match=CONTACT):
        _plan(model, faces[::-1] if reverse else faces)



# Zone contract (owner decisions 20-22, 2026-11-20). Each refusal below joined on the parked 078127d or is new
# scope; the joins are positive controls.

def _wedge_pair(degrees):
    """Unit square A left of the shared edge x=1 (V = (1,0), W = (1,1)); B right of it, wrapping under A's
    bottom edge with an edge from V at ``degrees`` below it (a sliver wedge of B between the two edges)."""
    k = 1.1e-10
    slope = math.tan(math.radians(degrees))
    return _xy_pair([(0, 0), (1, 0), (1, 1), (0, 1)],
                    [(1, 1), (1, 0), (0, -slope), (2, -1), (2, 1)],
                    ((1, 0), (1, 1)), ((1, 0, 0), (1, 0, k), (0, 1, 0), lambda x, y: k*(x-1)))


@pytest.mark.parametrize("degrees, joins", ((5., False), (14., False), (15., True), (30., True)))
def test_sliver_wedge_at_a_shared_vertex(degrees, joins):
    # Rule 2 at V: the point of A's bottom edge c t = 4t from V is 4t*sin(angle) from B's edge, so wedges below
    # asin(1/c) = 14.48 degrees refuse typed (fail closed: the plates are disjoint) and wider ones join.
    model, faces, ids = _wedge_pair(degrees)
    for pair in (faces, faces[::-1]):
        if joins:
            assert len(_plan(model, pair).arrangements) == 2
        else:
            with pytest.raises(GeometryError, match=CONTACT):
                _plan(model, pair)


def test_face_on_the_same_side_of_the_shared_edge_refuses():
    # Rule 3: triangle B on A's side of the shared edge x=1 (its other edges at 45 degrees, no off-edge contact).
    k = 1.1e-10
    model, faces, _ = _xy_pair([(0, 0), (1, 0), (1, 1), (0, 1)], [(1, 0), (1, 1), (.5, .5)],
                               ((1, 0), (1, 1)), ((1, 0, 0), (1, 0, k), (0, 1, 0), lambda x, y: k*(x-1)))
    for pair in (faces, faces[::-1]):
        with pytest.raises(GeometryError, match=SAME_SIDE):
            _plan(model, pair)


def _holed(inside):
    """A = [0,3]^2 with the square hole [1,2]^2; B = a triangle on the hole's edge x=1 (its other edges at 45
    degrees), inside the hole (``inside``) or in A's material left of it."""
    from anygeometry.entities import OrientedEdge as OE
    k = 1.1e-10
    model = GeometryModel()
    outer = model.add_points(((0, 0, 0), (3, 0, 0), (3, 3, 0), (0, 3, 0)))
    hole = model.add_points(((1, 1, 0), (1, 2, 0), (2, 2, 0), (2, 1, 0)))  # clockwise in the XY chart
    outer_edges = [model.add_line(outer[i], outer[(i+1) % 4]) for i in range(4)]
    hole_edges = [model.add_line(hole[i], hole[(i+1) % 4]) for i in range(4)]
    face_a = model.add_face_from_loop(tuple(OE(edge, True) for edge in outer_edges),
                                      surface=Plane((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    with model.transaction():
        model._put_entity("face", replace(model.faces[face_a], holes=(tuple(OE(edge, True) for edge in hole_edges),)))
    x = 1.5 if inside else .5
    tip = model.add_point(x, 1.5, k*(x-1))
    ring = (OE(hole_edges[0], False), OE(model.add_line(hole[0], tip), True), OE(model.add_line(tip, hole[1]), True))
    face_b = model.add_face_from_loop(ring if inside else tuple(OE(e.edge, not e.forward) for e in ring[::-1]),
                                      surface=Plane((1, 1, 0), (1, 0, k), (0, 1, 0)))
    assert model.validate_topology() == ()
    return model, (model.handle("face", face_a), model.handle("face", face_b)), hole_edges[0]


def test_plate_in_a_hole_sharing_the_hole_edge_joins():
    # Rule 3 with a hole loop: A's material is outside its hole, so B inside the hole is on the other side.
    model, faces, edge = _holed(True)
    for pair in (faces, faces[::-1]):
        assert _joined_by(_plan(model, pair), edge, pair)


def test_plate_over_the_material_beside_a_hole_edge_refuses():
    model, faces, _ = _holed(False)
    for pair in (faces, faces[::-1]):
        with pytest.raises(GeometryError, match=SAME_SIDE + "|" + OVERLAP):
            _plan(model, pair)


def test_same_sheet_seam_shared_edge_is_not_emitted():
    # As main's coplanar route: a shared same-sheet decomposition seam is accepted but not emitted.
    from anygeometry import EntityRef, batch_intersections
    from anygeometry.material_arrangement import MaterialDomain
    model, faces, hinge = _far_squares(0., 0.)
    model.tag(EntityRef("edge", hinge), "intersection_decomposition_seam")
    model.add_sheet([face.id for face in faces], orientations=(1, 1))
    first, second = (MaterialDomain.from_model(model, face.id) for face in faces)
    assert batch_intersections._pair_paths_exact(first, second, model, 1e-9, lambda: None) == ()


@pytest.mark.parametrize("X", (0., 2.**15))
def test_far_comb_inside_the_fence_refuses_typed(X):
    # The Codex d17 comb (B's notch intrudes 60t into A), inside the coordinate fence: contact off the shared edge.
    t, k = 1e-9, 1.1e-10
    U, Y = 1e-6, 60*t
    model, faces, _ = _xy_pair(
        [(X-1, 0), (X+1, 0), (X+2, 0), (X+2, 1), (X-1, 1)],
        [(X+2, 0), (X+1, 0), (X+1, -.1), (X+2*U, -.1), (X+2*U, -Y), (X, Y), (X-U, Y), (X+U, -Y*3/7), (X+U, -1),
         (X+2, -1)],
        ((X+1, 0), (X+2, 0)), ((X, 0, 0), (1, 0, 0), (0, 1, k), lambda x, y: k*y))
    with pytest.raises(GeometryError, match=CONTACT + "|" + OVERLAP):
        _plan(model, faces)



# Codex z1 (zone contract round 1): each reproducer was accepted or refused untyped on 567653d.

def test_stored_plane_angle_above_the_band_refuses_typed():
    # z1 M3: the exact sine of the stored planes is 4.17e-8 > 2^-26, but float cancellation puts the predicate's
    # sine at 1.89e-10 (ill-conditioned band); joined on 567653d. The band is now checked on the exact planes.
    x, y, q, e = 3.141592653589793, 3.141592658589793, 0.9999999115821654, .002
    model = GeometryModel()
    pts = model.add_points(((e, -e, -e), (.1+e, .1-e, -e), (.1, .1, 0), (0, 0, 0), (.1-e, .1+e, e*q), (-e, e, e*q)))
    face_a, shared = _ring_face(model, [pts[0], pts[1], pts[2], pts[3]], Plane((0, 0, 0), (1.3, 1.3, 0), (x, y, (y-x)/2)))
    face_b, _ = _ring_face(model, [pts[3], pts[2], pts[4], pts[5]], Plane((0, 0, 0), (1, 1, 0), (-1, 1, q)), shared)
    assert model.validate_topology() == ()
    with pytest.raises(GeometryError, match="ill_conditioned_plane_plane; near_coplanar_angle_outside_band"):
        _plan(model, (model.handle("face", face_a), model.handle("face", face_b)))


def test_domain_path_off_its_stored_vertex_refuses_typed():
    # z1 M2: at t = 2e-17 m B's reversed domain path starts an ulp (5.6e-17 m) off the stored vertex; on 567653d
    # the pair passed and the arrangement refused untyped ("not closable"). Drift above t/8 now refuses typed.
    from anygeometry import TolerancePolicy
    w, k = 1e-7, 1.1e-10
    model = GeometryModel(tolerance=TolerancePolicy(length=2e-17, relative_length=1e-20))
    p = model.add_points(((0, 1, 0), (0, .3, 0), (-w, .3, 0), (-w, 1, 0), (w, 1, k*w), (w, .3, k*w)))
    face_a, shared = _ring_face(model, [p[0], p[1], p[2], p[3]], Plane((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    face_b, _ = _ring_face(model, [p[1], p[0], p[4], p[5]], Plane((0, 0, 0), (1, 0, k), (0, 1, 0)), shared)
    assert model.validate_topology() == ()
    with pytest.raises(GeometryError, match=UNRESOLVED):
        _plan(model, (model.handle("face", face_a), model.handle("face", face_b)))


def test_hinge_exactly_two_zones_long_joins():
    # z1 L2: a hinge exactly 2 c t = 8t long has its midpoint at distance r; refused on 567653d (rounded step).
    L, k = 8*1e-9, 1.1e-10
    model, faces, _ = _xy_pair([(-1, 0), (0, 0), (0, L), (-1, L)], [(0, L), (0, 0), (1, 0), (1, L)],
                               ((0, 0), (0, L)), ((0, 0, 0), (1, 0, k), (0, 1, 0), lambda x, y: k*x))
    assert len(_plan(model, faces).arrangements) == 2


@pytest.mark.parametrize("gap, joins", ((math.nextafter(1e-9, math.inf), False), (1.0000001e-9, True)))
def test_gap_just_above_t_closes_on_the_second_support(gap, joins):
    # Codex z2 M: B's stored vertices lie 0.25t below both planes; a stored gap one ulp above t between non-shared
    # edges, 1.5 m from the shared vertices, is 0.99999999988t once B's trims are on B's own plane (its material).
    # Joined on 282a2ee. Rule 2 now separates with margin t * 2^-26; a gap 1e-7 relative above t still joins.
    k, h = 5e-10, 2.5e-10
    a = [(-.5, 0), (0, 0), (0, 1), (-.25, 1), (-.25, 2), (0, 2), (0, 3), (-.5, 3)]
    b = [(0, 3), (0, 2), (.25, 1.5), (gap, .75), (gap, .25), (.5, .25), (.5, 3)]
    model, faces, _ = _xy_pair(a, b, ((0, 2), (0, 3)), ((0, 0, 0), (1, 0, k), (0, 1, 0), lambda x, y: -h if x else 0))
    assert model.validate_topology() == ()
    if joins:
        assert len(_plan(model, faces).arrangements) == 2
    else:
        with pytest.raises(GeometryError, match=CONTACT):
            _plan(model, faces)
