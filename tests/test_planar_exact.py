"""Exact planar charts and the planar query route (robust-predicate slice R1a).

Independent references: exact normal-equation solves written here (a different
formula from the Cramer cross-product chart of ``planar_exact``), Shapely on
generic configurations far from degeneracy, and the semantics the engine
already had for gaps it could resolve (a positive gap is disjoint).
"""
from __future__ import annotations

from fractions import Fraction
import math
import random

import numpy as np
import pytest

from anygeometry import GeometryModel, IntersectionKind, Plane, query_intersection
from anygeometry.exact_predicates import Outcome, exact, exact_dot
from anygeometry.intersections import clip_line_to_face
from anygeometry.planar_exact import ExactLine, ExactPlane, clip_line_to_region, plane_plane_line, point_in_region


def normal_equation_uv(plane, point):
    """Independent exact chart coordinates: solve the 2x2 normal equations in Fractions."""
    U, V = [tuple(map(exact, vector)) for vector in (plane.u_vector, plane.v_vector)]
    d = tuple(exact(x) - exact(o) for x, o in zip(point, plane.origin))
    a, b, c = exact_dot(U, U), exact_dot(U, V), exact_dot(V, V)
    p, q = exact_dot(U, d), exact_dot(V, d)
    determinant = a * c - b * b
    return (c * p - b * q) / determinant, (a * q - b * p) / determinant


# ----------------------------------------------------------------------------------- charts and lines


def test_exact_chart_agrees_with_normal_equations_and_with_lstsq_to_rounding():
    rng = np.random.default_rng(1)
    for _ in range(200):
        plane = Plane(rng.uniform(-3, 3, 3), rng.uniform(-2, 2, 3), rng.uniform(-2, 2, 3))
        point = rng.uniform(-5, 5, 3)
        exact_uv = ExactPlane.from_plane(plane).uv(point)
        assert exact_uv == normal_equation_uv(plane, point)
        np.testing.assert_allclose([float(x) for x in exact_uv], plane.local_uv(point), rtol=1e-10, atol=1e-10)


def test_plane_plane_line_lies_exactly_in_both_planes():
    rng = np.random.default_rng(2)
    for _ in range(100):
        first = ExactPlane.from_plane(Plane(rng.uniform(-3, 3, 3), rng.uniform(-2, 2, 3), rng.uniform(-2, 2, 3)))
        second = ExactPlane.from_plane(Plane(rng.uniform(-3, 3, 3), rng.uniform(-2, 2, 3), rng.uniform(-2, 2, 3)))
        line = plane_plane_line(first, second)
        for s in (Fraction(0), Fraction(7, 3), Fraction(-5, 2)):
            point = line.at(s)
            assert exact_dot(first.normal, point) == first.offset()
            assert exact_dot(second.normal, point) == second.offset()
    parallel = Plane((0., 0., 1.), (1., 0., 0.), (0., 1., 0.))
    assert plane_plane_line(ExactPlane.from_plane(parallel), ExactPlane.from_plane(Plane((0., 0., 0.), (2., 0., 0.), (0., 3., 0.)))) is None


# ----------------------------------------------------------------------------------- exact region clipping


SQUARE = [(0, 0), (4, 0), (4, 4), (0, 4)]
HOLE = [(1, 1), (1, 2), (2, 2), (2, 1)]
CONCAVE = [(0, 0), (4, 0), (4, 3), (2, 1), (0, 3)]


# Expected values are the polygon overlay's own components (Shapely), which main reported:
# runs are split at every boundary point they pass (hole vertices, reflex vertices).
@pytest.mark.parametrize("anchor,direction,intervals,points", [
    ((-1, 2), (1, 0), [(1, 2), (2, 3), (3, 5)], []),                 # along the hole's top edge
    ((-1, 1.5), (1, 0), [(1, 2), (3, 5)], []),                       # through the hole
    ((-1, 0), (1, 0), [(1, 5)], []),                                 # along the outer bottom edge
    ((-1, 1), (1, 0), [(1, 2), (2, 3), (3, 5)], []),                 # along the hole's bottom edge
    ((-1, 5), (1, -1), [(1, 3), (3, 5)], []),                        # anti-diagonal through the hole corner (2, 2)
    ((3, 5), (1, -1), [], [Fraction(1)]),                            # touches the corner (4, 4) only
    ((-1, -1), (1, 1), [(1, 2), (3, 5)], []),                        # diagonal through the hole corners
])
def test_clipping_square_with_hole(anchor, direction, intervals, points):
    got_intervals, got_points = clip_line_to_region(anchor, direction, [SQUARE, HOLE])
    assert got_intervals == [(Fraction(a), Fraction(b)) for a, b in intervals]
    assert got_points == points


def test_clipping_concave_notch_vertex_touch():
    # The horizontal line y = 1 touches the notch vertex (2, 1) from below the notch: two intervals meet there.
    intervals, points = clip_line_to_region((0, 1), (1, 0), [CONCAVE])
    assert intervals == [(Fraction(0), Fraction(2)), (Fraction(2), Fraction(4))]      # split at the reflex vertex
    intervals, points = clip_line_to_region((0, 3), (1, 0), [CONCAVE])
    assert intervals == [] and points == [Fraction(0), Fraction(4)]


def test_clipping_matches_shapely_on_generic_lines():
    shapely = pytest.importorskip("shapely")
    from shapely.geometry import LineString, Polygon

    polygon = Polygon(CONCAVE)
    rng = random.Random(5)
    for _ in range(300):
        anchor = (rng.uniform(-1, 5), rng.uniform(-1, 4))
        angle = rng.uniform(0, math.pi)
        direction = (math.cos(angle), math.sin(angle))
        intervals, points = clip_line_to_region(anchor, direction, [CONCAVE])
        far = [(anchor[0] + t * direction[0], anchor[1] + t * direction[1]) for t in (-20.0, 20.0)]
        result = polygon.intersection(LineString(far))
        pieces = [] if result.is_empty else [g for g in getattr(result, "geoms", [result]) if g.geom_type == "LineString"]
        expected = sorted(tuple(sorted(((x - anchor[0]) * direction[0] + (y - anchor[1]) * direction[1])
                                       for x, y in (piece.coords[0], piece.coords[-1]))) for piece in pieces)
        assert len(intervals) == len(expected)                       # same components, not just the same length
        for (a, b), (c, d) in zip(intervals, expected):
            assert float(a) == pytest.approx(c, abs=1e-12) and float(b) == pytest.approx(d, abs=1e-12)


def test_point_in_region_treats_hole_boundaries_as_boundary():
    loops = [SQUARE, HOLE]
    assert point_in_region((Fraction(3), Fraction(3)), loops) is Outcome.ABOVE
    assert point_in_region((Fraction(3, 2), Fraction(3, 2)), loops) is Outcome.BELOW
    assert point_in_region((Fraction(1), Fraction(3, 2)), loops) is Outcome.EQUAL


# ----------------------------------------------------------------------------------- the planar query route


def plates(d):
    """Square A in z = 0 with a corner at the origin; vertical B in x = d (corner gap exactly d)."""
    model = GeometryModel()
    a = model.add_plate(tuple(model.add_points(((-1., -1., 0.), (0., -1., 0.), (0., 0., 0.), (-1., 0., 0.)))))
    b = model.add_plate(tuple(model.add_points(((d, 0., 0.), (d, 1., 0.), (d, 1., 1.), (d, 0., 1.)))))
    return model, model.handle("face", a), model.handle("face", b)


@pytest.mark.parametrize("d", [1e-170, 1e-200, 5e-324, 1e-12, 7.5e-10])
def test_any_positive_corner_gap_is_disjoint_at_every_scale(d):
    # The engine already reported DISJOINT for gaps it could resolve (0.75 tol, tol, 1.5 tol);
    # tiny gaps used to become TOUCH_POINT only because chart rounding swallowed them.
    model, a, b = plates(d)
    assert query_intersection(model, a, b).kind is IntersectionKind.DISJOINT
    assert query_intersection(model, b, a).kind is IntersectionKind.DISJOINT


def test_zero_gap_touches():
    model, a, b = plates(0.0)
    assert query_intersection(model, a, b).kind is IntersectionKind.TOUCH_POINT
    assert query_intersection(model, b, a).kind is IntersectionKind.TOUCH_POINT


@pytest.mark.parametrize("seed", range(1, 9))
def test_plate_on_a_rounded_edge_is_decided_by_the_exact_sign(seed):
    """A plate standing on the rounded evaluation of another face's u = 0 edge.

    The answer follows the exact chart sign of the stored vertices (independent
    normal-equation oracle): inside or on the edge -> the plates meet along a
    curve; outside -> disjoint. Forward and reverse queries agree.
    """
    rng = np.random.default_rng(seed)
    origin, u_vector, v_vector = rng.uniform(-3, 3, 3), rng.uniform(-2, 2, 3), rng.uniform(-2, 2, 3)
    plane = Plane(origin, u_vector, v_vector)
    model = GeometryModel()
    corners = model.add_points(tuple(plane.evaluate(u, v) for u, v in ((0, 0), (1, 0), (1, 1), (0, 1))))
    first = model.add_face(model.add_polyline(corners, close=True), surface=plane)
    a, b = plane.evaluate(0.0, 0.2), plane.evaluate(0.0, 0.8)
    out = plane.normal - 0.5 * u_vector / np.linalg.norm(u_vector)
    second = model.add_plate(tuple(model.add_points((a, b, b + out, a + out))))
    u_a, u_b = normal_equation_uv(plane, a)[0], normal_equation_uv(plane, b)[0]
    forward = query_intersection(model, model.handle("face", first), model.handle("face", second))
    reverse = query_intersection(model, model.handle("face", second), model.handle("face", first))
    assert forward.kind is reverse.kind
    if u_a >= 0 and u_b >= 0:
        assert forward.kind is IntersectionKind.CROSS
    elif u_a < 0 and u_b < 0:
        assert forward.kind is IntersectionKind.DISJOINT


def test_planar_queries_do_not_depend_on_plane_chart_rounding(monkeypatch):
    """The planar route no longer reads Plane.local_uv for decisions: nudging it changes nothing."""
    model, a, b = plates(0.0)
    reference = query_intersection(model, a, b)
    original = Plane.local_uv

    def nudged(self, point):
        u, v = original(self, point)
        return math.nextafter(u, -math.inf), math.nextafter(v, math.inf)

    monkeypatch.setattr(Plane, "local_uv", nudged)
    assert query_intersection(model, a, b).kind is reference.kind
    face = int(a.id)
    assert clip_line_to_face(model, face, (-2., -.5, 0.), (1., 0., 0.)).kind is IntersectionKind.OVERLAP_CURVE


# ----------------------------------------------------------------------------------- review regressions (R1a round 1)


def test_reflex_vertex_split_keeps_main_semantics_for_both_operand_orders():
    """A vertical face meeting the notch line y = 1 of a concave plate: main reported two
    components (split at the reflex vertex) and TOUCH_POINT; merging them made it CROSS."""
    model = GeometryModel()
    concave = model.add_plate(tuple(model.add_points([(x, y, 0.) for x, y in CONCAVE])))
    other = model.add_plate(tuple(model.add_points(((2., 1., -7.5e-10), (3., 1., 1.), (2., 1., 2.), (1., 1., 1.)))))
    a, b = model.handle("face", concave), model.handle("face", other)
    assert query_intersection(model, a, b).kind is IntersectionKind.TOUCH_POINT
    assert query_intersection(model, b, a).kind is IntersectionKind.TOUCH_POINT


def test_tiny_directions_report_physical_parameters_without_overflow():
    model = GeometryModel()
    face = model.add_plate(tuple(model.add_points(((0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.)))))
    result = clip_line_to_face(model, face, (0., .5, 0.), (5e-324, 0., 0.))
    interval = result.components[0].first_parameter_range
    assert result.kind is IntersectionKind.OVERLAP_CURVE and (interval.start, interval.end) == (0.0, 1.0)


def test_coplanar_witnesses_lie_on_the_support_plane():
    model = GeometryModel()
    face = model.add_plate(tuple(model.add_points(((0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.)))))
    result = clip_line_to_face(model, face, (0., .5, 5e-10), (1., 0., 0.))
    assert result.kind is IntersectionKind.OVERLAP_CURVE
    assert all(witness[2] == 0.0 for witness in result.components[0].witnesses)


def test_face_extent_is_canonical():
    from anygeometry.exact_predicates import canonical_extent, correctly_rounded_sqrt
    assert canonical_extent((0, 0, 0), (1, 1, 0)) == math.sqrt(2)          # math.sqrt is correctly rounded
    rng = random.Random(8)
    for _ in range(2000):
        value = Fraction(rng.getrandbits(80), rng.getrandbits(40) + 1)
        assert correctly_rounded_sqrt(value) == math.sqrt(float(value)) or             abs(Fraction(correctly_rounded_sqrt(value)) ** 2 - value) <= abs(Fraction(math.sqrt(float(value))) ** 2 - value)


def test_invalid_regions_are_refused():
    from anygeometry.planar_exact import region_is_simple
    assert not region_is_simple([[(0, 0), (2, 2), (0, 2), (2, 0)]])          # bow tie
    assert not region_is_simple([[(0, 0), (1, 0), (2, 0)]])                   # zero area
    assert region_is_simple([SQUARE, HOLE]) and region_is_simple([CONCAVE])
    with pytest.raises(Exception):
        point_in_region((0, 0), [])


# ----------------------------------------------------------------------------------- review regressions (R1a round 2)


def test_inferred_planar_supports_use_the_exact_vertex_plane():
    """A topology-backed (inferred) planar face exactly in z = x + y meeting a plate in x = 0
    along (0, y, y), 0.25 <= y <= 0.75: the SVD support used to answer DISJOINT."""
    model = GeometryModel()
    a = model.add_face(model.add_polyline(model.add_points(((0., 0., 0.), (1., 0., 1.), (1., 1., 2.), (0., 1., 1.))),
                                          close=True))
    b = model.add_plate(tuple(model.add_points(((0., .25, -.75), (0., .75, -.25), (0., .75, 1.75), (0., .25, 1.25)))))
    first, second = model.handle("face", a), model.handle("face", b)
    forward, reverse = query_intersection(model, first, second), query_intersection(model, second, first)
    assert forward.kind is reverse.kind is IntersectionKind.CROSS
    witnesses = sorted(forward.components[0].witnesses)
    np.testing.assert_allclose(witnesses, [(0., .25, .25), (0., .75, .75)], atol=1e-15)


def test_tolerance_comes_from_stored_vertices_only(monkeypatch):
    model = GeometryModel()
    a = model.add_plate(tuple(model.add_points(((0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.)))))
    w = 0.0014142135623730955
    b = model.add_plate(tuple(model.add_points(((w - 1, .5, -.5), (w, .5, -.5), (w, .5, .5), (w - 1, .5, .5)))))
    first, second = model.handle("face", a), model.handle("face", b)
    reference = query_intersection(model, first, second).kind
    original = Plane.evaluate
    monkeypatch.setattr(Plane, "evaluate", lambda self, u, v: np.nextafter(original(self, u, v), np.inf))
    assert query_intersection(model, first, second).kind is reference


def test_isolated_coplanar_touches_lie_on_the_support_plane():
    model = GeometryModel()
    face = model.add_plate(tuple(model.add_points(((0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.)))))
    result = clip_line_to_face(model, face, (-1., 1., 5e-10), (1., -1., 0.))
    assert result.kind is IntersectionKind.TOUCH_POINT and result.components[0].witnesses[0][2] == 0.0


def test_subnormal_diagonal_directions_report_physical_parameters():
    model = GeometryModel()
    face = model.add_plate(tuple(model.add_points(((0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.)))))
    result = clip_line_to_face(model, face, (0., 0., 0.), (5e-324, 5e-324, 0.))
    interval = result.components[0].first_parameter_range
    assert result.kind is IntersectionKind.OVERLAP_CURVE
    assert (interval.start, interval.end) == (0.0, math.sqrt(2))
    assert result.components[0].direction == (math.sqrt(.5), math.sqrt(.5), 0.0)


def test_holes_must_lie_inside_the_outer_loop_and_not_nest():
    from anygeometry.planar_exact import region_is_simple
    outer = [(0, 0), (4, 0), (4, 4), (0, 4)]
    assert not region_is_simple([outer, [(5, 5), (6, 5), (6, 6), (5, 6)]])                         # outside
    assert not region_is_simple([outer, [(1, 1), (3, 1), (3, 3), (1, 3)],
                                 [(1.5, 1.5), (2.5, 1.5), (2.5, 2.5), (1.5, 2.5)]])                 # nested
    assert region_is_simple([outer, [(1, 1), (2, 1), (2, 2), (1, 2)], [(3, 3), (3.5, 3), (3.5, 3.5)]])


# ----------------------------------------------------------------------------------- review regressions (R1a round 3)


def test_approximately_planar_inferred_faces_keep_mains_fitted_plane_decision():
    """A warped topology-backed face (z = 5e-9 x y, within the planarity tolerance) is decided on
    its fitted plane, as main did (DISJOINT); substituting a vertex plane answered CROSS."""
    model = GeometryModel()
    a = model.add_face(model.add_polyline(model.add_points(((0., 0., 0.), (1., 0., 0.), (1., 1., 5e-9), (0., 1., 0.))),
                                          close=True))
    b = model.add_plate(tuple(model.add_points(((.5, .25, -1.), (.5, .75, -1.), (.5, .75, 5e-10), (.5, .25, 5e-10)))))
    first, second = model.handle("face", a), model.handle("face", b)
    assert query_intersection(model, first, second).kind is IntersectionKind.DISJOINT
    assert query_intersection(model, second, first).kind is IntersectionKind.DISJOINT


def test_inferred_faces_report_uv_in_the_fitted_chart():
    from anygeometry.intersections import _qualified_face_plane
    model = GeometryModel()
    face = model.add_face(model.add_polyline(model.add_points(((0., 0., 0.), (1., 0., 1.), (1., 1., 2.), (0., 1., 1.))),
                                             close=True))
    result = clip_line_to_face(model, face, (.5, .5, 3.), (0., 0., -1.))
    assert result.kind is IntersectionKind.CROSS
    expected = _qualified_face_plane(model, face).local_uv(np.array((.5, .5, 1.)))
    np.testing.assert_allclose(result.components[0].second_parameter, expected, atol=1e-12)
