"""Exact stations for straight arrangement pieces (robust-predicate slice R1b, step 2a).

References: the float rules themselves on generic inputs (the same candidates
away from degeneracy, parameters within rounding), exact distance arithmetic at
the tolerance, and evaluation-rounding perturbation of the arrangement build.
"""
from __future__ import annotations

from fractions import Fraction
import math
import random

import numpy as np
import pytest

from anygeometry import Plane
from anygeometry import planar_stations as st
from anygeometry.arrangement_geometry import LinePath, curve_junctions, point_parameters
from anygeometry.errors import GeometryError
from anygeometry.material_arrangement import ArrangementPath, MaterialDomain, _parameters, arrange_material


def random_line(rng, scale=1.):
    return LinePath(tuple(rng.uniform(-scale, scale) for _ in range(3)),
                    tuple(rng.uniform(-scale, scale) for _ in range(3)))


def planar_line(rng):
    return LinePath((rng.uniform(-2, 2), rng.uniform(-2, 2), 0.), (rng.uniform(-2, 2), rng.uniform(-2, 2), 0.))


@pytest.mark.parametrize("seed", range(3))
def test_junctions_agree_with_the_float_rule_on_generic_pairs(seed):
    rng = random.Random(seed)
    for _ in range(300):
        first, second = planar_line(rng), planar_line(rng)
        expected = curve_junctions(first, second, tolerance=1e-9)
        got = st.junctions(st.segment_of(first), st.segment_of(second), 1e-9)
        assert len(got) == len(expected)
        for (a, b), (x, y) in zip(got, expected):
            assert abs(float(a) - x) <= 1e-12 and abs(float(b) - y) <= 1e-12


@pytest.mark.parametrize("seed", range(3))
def test_point_parameters_agree_with_the_float_rule(seed):
    rng = random.Random(seed)
    for _ in range(300):
        line = planar_line(rng)
        t = rng.choice((0., 1., rng.uniform(0, 1), rng.uniform(-.2, 1.2)))
        point = tuple(np.asarray(line.evaluate(t)) + np.array((0., rng.choice((0., 1e-12, 1e-3)), 0.)))
        expected = point_parameters(line, point, tolerance=1e-9)
        got = st.point_parameters(st.segment_of(line), tuple(map(Fraction, point)), 1e-9)
        assert len(got) == len(expected)
        assert all(abs(float(a) - b) <= 1e-12 for a, b in zip(got, expected))


def test_point_parameters_tolerance_is_exact():
    line = st.segment_of(LinePath((0., 0., 0.), (1., 0., 0.)))
    tol = 1e-9
    for gap, hit in ((math.nextafter(tol, 0.), True), (tol, True), (math.nextafter(tol, 1.), False)):
        assert bool(st.point_parameters(line, (Fraction(1, 2), Fraction(gap), Fraction(0)), tol)) is hit


def test_degenerate_lines_are_refused_as_on_the_float_route():
    point = LinePath((0., 0., 0.), (0., 0., 0.))
    with pytest.raises(GeometryError, match="degenerate arrangement line"):
        point_parameters(point, (1., 0., 0.), tolerance=1e-9)
    with pytest.raises(GeometryError, match="degenerate arrangement line"):
        st.point_parameters(st.segment_of(point), (Fraction(1), Fraction(0), Fraction(0)), 1e-9)
    tiny = LinePath((0., 0., 0.), (1e-200, 0., 0.))             # float squared length underflows
    with pytest.raises(GeometryError, match="degenerate arrangement line"):
        st.point_parameters(st.segment_of(tiny), (Fraction(1), Fraction(0), Fraction(0)), 1e-300)


@pytest.mark.parametrize("seed", range(3))
def test_merge_parameters_agrees_with_the_float_rule(seed):
    rng = random.Random(seed)
    for _ in range(200):
        line = planar_line(rng)
        values = [0., 1.] + [rng.uniform(0, 1) for _ in range(rng.randrange(6))]
        values += [v + rng.choice((0., 1e-17, 1e-11)) for v in values[2:]]
        values = [min(1., v) for v in values]
        expected = _parameters(list(values), 1e-9, line)
        got = st.merge_parameters(values, 1e-9, st.segment_of(line))
        assert len(got) == len(expected)
        assert all(abs(float(a) - b) <= 1e-15 for a, b in zip(got, expected))


@pytest.mark.parametrize("scale,relative", ((1., 1e-9), (1e-100, 1e-9), (1e120, 1e-9), (1., 1e-17)))
def test_vertex_lookup_matches_an_exact_scan(scale, relative):
    """Exact stations: lowest index within tol, with and without the grid (1e-17: rounding > tol/4 -> scan)."""
    rng = random.Random(7)
    tol = relative * scale
    lookup, points = st.ExactVertexLookup(tol), []
    for _ in range(400):
        base = rng.choice(points) if points and rng.random() < .5 else tuple(
            Fraction(rng.uniform(-1, 1) * scale) for _ in range(3))
        offset = rng.choice((0., .5, .999, 1., 1.001, 3.)) * tol
        point = (base[0] + Fraction(offset), base[1], base[2])
        expected = next((i for i, q in enumerate(points) if st.distance2(point, q) <= Fraction(tol) ** 2), None)
        floats = np.asarray(st.rounded(point))
        assert lookup.find(point, floats) == expected
        if expected is None:
            lookup.add(point, floats); points.append(point)
    assert (lookup.grid is None) is (relative < 1e-15 or not 1e-100 < tol < 1e100)


def test_curved_points_keep_the_float_vertex_rule():
    """Review regression: a curved piece's point is compared by main's float rule, not exactly."""
    from anygeometry.arrangement_geometry import BezierPath
    from anygeometry.material_arrangement import _VertexLookup
    curve = BezierPath(((0., 0., 0.), (0., 2**-60, 0.), (1., 0., 0.)))
    q = curve.subcurve(.3, 1.).evaluate(0.)
    tol = 2**-52
    p = (float(q[0]) - tol, float(q[1]) - tol * 2**-27, 0.)
    main_vertices = []
    main = _VertexLookup(main_vertices, tol)
    main.add(np.asarray(p)); main_vertices.append(np.asarray(p))
    lookup = st.ExactVertexLookup(tol)
    lookup.add(tuple(map(Fraction, p)), np.asarray(p))
    assert lookup.find(None, q) == main.find(q) == 0


def test_extension_window_candidates_are_kept():
    """Review regression: points accepted through the [-tol, 1+tol] parameter window are not filtered."""
    tol = 1e-9
    line = LinePath((0., 0., 0.), (1e6, 0., 0.))
    point = (-.0005, 0., 0.)
    assert st.point_parameters(st.segment_of(line), tuple(map(Fraction, point)), tol) == (0,)
    assert point_parameters(line, point, tolerance=tol) == (0.,)
    other = LinePath((-.0005, 0., 0.), (-.0005, 1., 0.))
    assert st.junctions(st.segment_of(line), st.segment_of(other), tol) == ((0, 0),)
    assert curve_junctions(line, other, tolerance=tol) == ((0., 0.),)


@pytest.mark.parametrize("gap,separated", ((1e-9, False), (math.nextafter(1e-9, 1.), True), (.0005, True)))
def test_pair_skip_is_mains_rule_decided_exactly(gap, separated):
    first = st.segment_of(LinePath((0., 0., 0.), (1., 0., 0.)))
    second = st.segment_of(LinePath((-gap, 0., 0.), (-gap, 1., 0.)))
    assert st.boxes_separated(first, second, 1e-9) is separated
    assert st.boxes_separated(second, first, 1e-9) is separated


# --------------------------------------------------------------------------------- arrangement build


def crossing_arrangement(plane, scale=1.):
    s = scale
    square = [(0., 0.), (2., 0.), (2., 2.), (0., 2.)]
    boundary = (tuple(ArrangementPath(LinePath(tuple(plane.evaluate(a[0] * s, a[1] * s)),
                                               tuple(plane.evaluate(b[0] * s, b[1] * s))))
                      for a, b in zip(square, square[1:] + square[:1])),)
    domain = MaterialDomain(1, plane, boundary)
    traces = tuple(ArrangementPath(LinePath(tuple(plane.evaluate(*(p * s for p in a))),
                                            tuple(plane.evaluate(*(p * s for p in b)))), owners=(1, k))
                   for k, (a, b) in enumerate((((-.5, .3), (2.5, 1.7)), ((.7, -.5), (1.3, 2.5)),
                                               ((-.5, 1.1), (2.5, 1.1)))))
    return domain, traces


def summary(arrangement):
    return (len(arrangement.paths), len(arrangement.cells),
            sorted((path.curve.start, path.curve.end) for path in arrangement.paths))


@pytest.mark.parametrize("direction", (-math.inf, math.inf))
def test_the_build_does_not_depend_on_line_evaluation_rounding(monkeypatch, direction):
    plane = Plane((-1., -1., .3), (1., .2, 0.), (-.1, 1., .4))
    domain, traces = crossing_arrangement(plane)
    tolerance = 1e-9
    reference = summary(arrange_material(domain, traces, tolerance=tolerance))
    original = LinePath.evaluate
    monkeypatch.setattr(LinePath, "evaluate", lambda self, t: np.nextafter(original(self, t), direction))
    assert summary(arrange_material(domain, traces, tolerance=tolerance)) == reference


def test_pieces_have_correctly_rounded_exact_ends():
    """Every emitted end equals the correct rounding of an exact crossing of two source lines.

    Exactly coplanar data (z = 0), so each crossing is a unique rational point
    (on a tilted plane the stored lines are skew by rounding)."""
    plane = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    domain, traces = crossing_arrangement(plane)
    arrangement = arrange_material(domain, traces, tolerance=1e-9)
    assert len(arrangement.cells) == 7
    sources = [st.segment_of(path.curve) for loop in domain.boundaries for path in loop]
    sources += [st.segment_of(trace.curve) for trace in traces]
    crossings = {st.rounded(st.at(a_seg, a)) for i, a_seg in enumerate(sources) for b_seg in sources[i + 1:]
                 for a, _b in st.junctions(a_seg, b_seg, 1e-9)}
    for path in arrangement.paths:
        assert path.curve.start in crossings and path.curve.end in crossings


def test_rounded_ends_differ_from_the_float_formula_somewhere():
    """Discrimination: main's ``s + t (e - s)`` ends are not all correctly rounded on this case."""
    from anygeometry.material_arrangement import _parameters
    plane = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    domain, traces = crossing_arrangement(plane)
    exact_ends = {path.curve.end for path in arrange_material(domain, traces, tolerance=1e-9).paths}
    float_ends = set()
    for trace in traces:
        for other in traces + domain.boundaries[0]:
            for a, _b in curve_junctions(trace.curve, other.curve, tolerance=1e-9):
                float_ends.add(tuple(float(v) for v in trace.curve.evaluate(a)))
    near = [point for point in float_ends - exact_ends
            if any(max(abs(a - b) for a, b in zip(point, end)) <= 1e-12 for end in exact_ends)]
    assert near                                    # the same crossing, rounded differently by main's formula


def test_float32_curved_points_switch_to_mains_scan():
    """Review regression: float32 evaluations are compared by main's scan (norm underflow), not the grid."""
    from anygeometry.arrangement_geometry import BezierPath
    from anygeometry.material_arrangement import _VertexLookup
    curve = BezierPath(tuple(tuple(np.float32(v) for v in xyz)
                             for xyz in ((0., 0., 0.), (5e-31, 1e-31, 0.), (1e-30, 0., 0.))))
    p, q = curve.evaluate(0.), curve.evaluate(1.)
    tol = 1e-35
    s = np.array((2e-30, 0., 0.))
    vertices = []
    main = _VertexLookup(vertices, tol)
    for point in (s, p):
        assert main.find(point) is None
        main.add(point); vertices.append(point)
    lookup = st.ExactVertexLookup(tol)
    lookup.add(tuple(map(Fraction, s)), s)
    assert lookup.find(None, p) is None
    lookup.add(None, p)
    assert lookup.find(None, q) == main.find(q) == 1


# --------------------------------------------------------------------------------- seams (step 2b, D6/D7)


def dangling_arrangement(plane):
    """A square with one trace from the boundary to an interior end: two seams through the end."""
    square = [(0., 0.), (2., 0.), (2., 2.), (0., 2.)]
    boundary = (tuple(ArrangementPath(LinePath(tuple(plane.evaluate(*a)), tuple(plane.evaluate(*b))))
                      for a, b in zip(square, square[1:] + square[:1])),)
    domain = MaterialDomain(1, plane, boundary)
    trace = ArrangementPath(LinePath(tuple(plane.evaluate(0., .7)), tuple(plane.evaluate(1.3, 1.1))), owners=(1, 2))
    return domain, (trace,)


@pytest.mark.parametrize("plane", (Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)),
                                   Plane((-1., -1., .3), (1., .2, 0.), (-.1, 1., .4))))
def test_interior_trace_end_gets_two_seams_through_its_station(plane):
    domain, traces = dangling_arrangement(plane)
    arrangement = arrange_material(domain, traces, tolerance=1e-9)
    assert (len(arrangement.cells), len(arrangement.paths)) == (5, 14)       # as main: 4 seam quadrants, one split by the trace
    end = st.segment_of(traces[0].curve)[1]
    seams = [path for path in arrangement.paths if path.decomposition]
    touching = [path for path in seams if end in (tuple(map(Fraction, path.curve.start)),
                                                  tuple(map(Fraction, path.curve.end)))
                or st.rounded(end) in (path.curve.start, path.curve.end)]
    assert len(touching) >= 3                          # the seam pieces meeting at the station


@pytest.mark.parametrize("direction", (-math.inf, math.inf))
def test_seams_do_not_depend_on_evaluation_or_chart_rounding(monkeypatch, direction):
    plane = Plane((-1., -1., .3), (1., .2, 0.), (-.1, 1., .4))
    domain, traces = dangling_arrangement(plane)
    reference = summary(arrange_material(domain, traces, tolerance=1e-9))
    line_evaluate, local_uv = LinePath.evaluate, Plane.local_uv
    monkeypatch.setattr(LinePath, "evaluate", lambda self, t: np.nextafter(line_evaluate(self, t), direction))
    monkeypatch.setattr(Plane, "local_uv", lambda self, point: tuple(
        math.nextafter(float(value), direction) for value in local_uv(self, point)))
    assert summary(arrange_material(domain, traces, tolerance=1e-9)) == reference


def test_ray_dedupe_is_mains_rule_decided_exactly():
    from anygeometry.material_arrangement import _same_ray
    eps = np.finfo(float).eps

    def ray(direction):
        exact = tuple(map(Fraction, direction))
        unit = np.asarray(direction) / np.linalg.norm(direction)
        return exact, unit
    base = ray((1., 0., 0.))
    assert _same_ray(base, ray((3., 0., 0.)))
    assert not _same_ray(base, ray((-1., 0., 0.)))
    assert _same_ray(base, ray((1., 60 * eps, 0.)))
    assert not _same_ray(base, ray((1., 68 * eps, 0.)))
    rng = random.Random(3)
    for _ in range(500):                               # generic directions: same answer as main's float rule
        a = ray(tuple(rng.uniform(-1, 1) for _ in range(3)))
        b = ray(tuple(rng.uniform(-1, 1) for _ in range(3)))
        assert _same_ray(a, b) is bool(np.linalg.norm(a[1] - b[1]) <= 64 * eps)
        assert _same_ray(a, a)


def test_seams_keep_mains_parameterization():
    """Review regression: an oblique seam keeps main's length, so the 32-eps parameter merge in
    clipping sees main's crossing separations (a longer seam merged two crossings and dropped a seam)."""
    from anygeometry.material_arrangement import ArrangementPoint
    plane = Plane((0., 0., 0.), (1., 1., 0.), (-1., 1., 0.))
    uv = [(-6e-15, -5e-11), (6e-15, -5e-11), (6e-15, 5e-11), (-6e-15, 5e-11)]
    boundary = (tuple(ArrangementPath(LinePath(tuple(plane.evaluate(*a)), tuple(plane.evaluate(*b))))
                      for a, b in zip(uv, uv[1:] + uv[:1])),)
    arrangement = arrange_material(MaterialDomain(1, plane, boundary), (), tolerance=1e-18,
                                   points=(ArrangementPoint((0., 0., 0.)),))
    assert (len(arrangement.cells), len(arrangement.paths),
            sum(path.decomposition for path in arrangement.paths)) == (4, 12, 4)       # main's answer


# --------------------------------------------------------------------------------- vertex order (step 2c, D2)


def star_arrangement(plane, angles):
    """A square cut by straight traces through its centre at the given chart angles."""
    square = [(-2., -2.), (2., -2.), (2., 2.), (-2., 2.)]
    boundary = (tuple(ArrangementPath(LinePath(tuple(plane.evaluate(*a)), tuple(plane.evaluate(*b))))
                      for a, b in zip(square, square[1:] + square[:1])),)
    traces = tuple(ArrangementPath(LinePath(tuple(plane.evaluate(-3 * math.cos(t), -3 * math.sin(t))),
                                            tuple(plane.evaluate(3 * math.cos(t), 3 * math.sin(t)))), owners=(1, k))
                   for k, t in enumerate(angles))
    return MaterialDomain(1, plane, boundary), traces


def test_vertex_order_agrees_with_main_on_generic_stars():
    from anygeometry.material_arrangement import _compare_outgoing
    rng = random.Random(5)
    for _ in range(5):
        plane = Plane((rng.uniform(-1, 1), rng.uniform(-1, 1), 0.), (1., rng.uniform(-.3, .3), 0.),
                      (rng.uniform(-.3, .3), 1., rng.uniform(-.3, .3)))
        angles = sorted(rng.uniform(0, math.pi) for _ in range(4))
        domain, traces = star_arrangement(plane, angles)
        arrangement = arrange_material(domain, traces, tolerance=1e-9)
        assert len(arrangement.cells) == 8                     # 4 lines through one point: 8 sectors


def test_nearly_tangent_straight_edges_are_ordered_by_their_exact_directions():
    """Edges 20 eps apart in angle: main grouped them (64 eps) and ordered by index; the exact
    comparison gives the counterclockwise order whatever the index order. (Whole arrangements with
    such slivers still fail at the cell area/probe stage on main and here: step 2d.)"""
    import functools
    from anygeometry.material_arrangement import _compare_outgoing
    from anygeometry.planar_exact import ExactPlane
    chart = ExactPlane.from_plane(Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)))
    eps = np.finfo(float).eps
    origin = (Fraction(0),) * 3
    directions = {0: (1., 1., 0.), 1: (1., 1. + 40 * eps, 0.), 2: (-1., 0., 0.), 3: (1., -1e-300, 0.)}
    segments = {index: (origin, tuple(map(Fraction, d))) for index, d in directions.items()}
    for permutation in ((0, 1, 2, 3), (1, 0, 3, 2), (3, 2, 1, 0)):
        items = [(None, 0., index, True) for index in permutation]
        items.sort(key=functools.cmp_to_key(lambda a, b: _compare_outgoing(chart, segments, a, b)))
        assert [item[2] for item in items] == [0, 1, 2, 3]
    # exactly equal directions keep main's tie order (curvature, index, forward)
    segments[4] = (origin, (Fraction(2), Fraction(2), Fraction(0)))
    items = sorted([(None, 0., 4, True), (None, 0., 0, True)],
                   key=functools.cmp_to_key(lambda a, b: _compare_outgoing(chart, segments, a, b)))
    assert [item[2] for item in items] == [0, 4]


@pytest.mark.parametrize("direction", (-math.inf, math.inf))
def test_vertex_order_does_not_depend_on_chart_rounding(monkeypatch, direction):
    plane = Plane((-1., -1., .3), (1., .2, 0.), (-.1, 1., .4))
    domain, traces = star_arrangement(plane, (.3, 1.1, 2.0, 2.9))
    reference = arrange_material(domain, traces, tolerance=1e-9)
    original = np.linalg.lstsq
    monkeypatch.setattr(np.linalg, "lstsq", lambda *a, **k: (np.nextafter(original(*a, **k)[0], direction),)
                        + tuple(original(*a, **k)[1:]))
    nudged = arrange_material(domain, traces, tolerance=1e-9)
    assert nudged.cells == reference.cells and summary(nudged) == summary(reference)
