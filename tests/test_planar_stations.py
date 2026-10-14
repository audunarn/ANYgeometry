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


def _length_snapped(line, values, tolerance):
    """Main's float point parameters with S1a's endpoint snap (sub-piece no longer than tol)."""
    length = float(np.linalg.norm(np.asarray(line.end) - line.start))
    ends = [t for t in values if t in (0., 1.)]
    result = []
    for value in values:
        value = next((t for t in ends if abs(value - t) * length <= tolerance * (1 + 1e-9)), value)
        if value not in result:
            result.append(value)
    return result


@pytest.mark.parametrize("seed", range(3))
def test_point_parameters_follow_the_float_rule_with_length_snaps(seed):
    rng = random.Random(seed)
    for _ in range(300):
        line = planar_line(rng)
        t = rng.choice((0., 1., rng.uniform(0, 1), rng.uniform(-.2, 1.2)))
        point = tuple(np.asarray(line.evaluate(t)) + np.array((0., rng.choice((0., 1e-12, 1e-3)), 0.)))
        expected = _length_snapped(line, point_parameters(line, point, tolerance=1e-9), 1e-9)
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


def test_merge_parameters_only_removes_exact_duplicates():
    """S2 (deferred removal): split parameters are kept down to any distance; short sub-pieces
    are removed once, on the shared vertex identity (S4)."""
    segment = st.segment_of(LinePath((0., 0., 0.), (1e6, 0., 0.)))
    values = [0., 1., .5, .5, .5 + 1e-17, .5 + 2 ** -53, 1., 0.]
    assert st.merge_parameters(values, 1e-9, segment) == sorted({Fraction(v) for v in values})


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


def _oracle_order(plane, directions):
    """Independent exact counterclockwise order: Gram-matrix chart coordinates, then angle classes."""
    import functools
    u, v = (tuple(map(Fraction, map(float, vector))) for vector in (plane.u_vector, plane.v_vector))
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    g11, g12, g22 = dot(u, u), dot(u, v), dot(v, v)
    det = g11 * g22 - g12 * g12

    def chart(d):
        r1, r2 = dot(u, d), dot(v, d)
        return ((g22 * r1 - g12 * r2) / det, (g11 * r2 - g12 * r1) / det)

    def compare(i, j):
        (a, b), (c, d) = chart(directions[i]), chart(directions[j])
        half_i, half_j = (0 if y > 0 or (y == 0 and x > 0) else 1 for x, y in ((a, b), (c, d)))
        if half_i != half_j:
            return half_i - half_j
        cross = a * d - b * c
        return -1 if cross > 0 else 1 if cross < 0 else i - j
    return sorted(directions, key=functools.cmp_to_key(compare))


def _comparator_order(plane, segments, items):
    import functools
    from anygeometry.material_arrangement import _compare_outgoing
    from anygeometry.planar_exact import ExactPlane
    chart = ExactPlane.from_plane(plane)
    return [item[2] for item in sorted(items, key=functools.cmp_to_key(
        lambda a, b: _compare_outgoing(chart, segments, a, b)))]


def test_ill_conditioned_chart_order_is_exact_where_mains_angles_are_not():
    """Review input: main's least-squares angles mis-order two edges 2,424 eps apart; the exact
    order follows an independent exact oracle."""
    plane = Plane((0., 0., 0.), (1., 1., 0.), (0., 1e-4, 0.))
    start = (-1., -1., 0.)
    ends = {0: (1.0, 1.0000000000000002, 0.), 1: (1.0000000000000004, 1.0000000000000004, 0.), 2: (-2., -1., 0.)}
    segments = {k: (tuple(map(Fraction, start)), tuple(map(Fraction, e))) for k, e in ends.items()}
    directions = {k: tuple(b - a for a, b in zip(*segments[k])) for k in ends}
    expected = _oracle_order(plane, directions)
    assert _comparator_order(plane, segments, [(None, 0., k, True) for k in ends]) == expected
    domain = MaterialDomain(1, plane, ())
    angles = {k: math.atan2(*domain.tangent(LinePath(start, e), 0.)[::-1]) % math.tau for k, e in ends.items()}
    float_order = sorted(ends, key=lambda k: angles[k])
    rotations = [expected[i:] + expected[:i] for i in range(len(expected))]
    assert float_order not in rotations                  # discrimination: main's cyclic order differs


@pytest.mark.parametrize("flip", (False, True))
def test_comparator_follows_the_oracle_for_both_orientations_and_directions(flip):
    rng = random.Random(11)
    for _ in range(50):
        u = (1., rng.uniform(-.5, .5), rng.uniform(-.5, .5))
        v = (rng.uniform(-.5, .5), 1., rng.uniform(-.5, .5))
        plane = Plane((0., 0., 0.), u, tuple(-x for x in v) if flip else v)
        origin = (Fraction(0),) * 3
        segments, items, directions = {}, [], {}
        for k in range(6):
            a, b = rng.uniform(-1, 1), rng.uniform(-1, 1)
            end = tuple(Fraction(a * x + b * y) for x, y in zip(plane.u_vector, plane.v_vector))
            forward = rng.random() < .5
            segments[k] = (origin, end) if forward else (end, origin)
            items.append((None, 0., k, forward))
            directions[k] = end
        assert _comparator_order(plane, segments, items) == _oracle_order(plane, directions)


def test_vertices_with_a_curved_edge_keep_mains_order(monkeypatch):
    """A chord of a circle: both its vertices also carry arcs, so the exact comparator is never used."""
    from anygeometry.exact_curves import EllipticArc
    import anygeometry.material_arrangement as module
    plane = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    circle = (ArrangementPath(EllipticArc((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), 0., math.pi)),
              ArrangementPath(EllipticArc((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), math.pi, math.pi)))
    chord = ArrangementPath(LinePath((-2., .3, 0.), (2., .3, 0.)), owners=(1, 2))

    def forbidden(*args):
        raise AssertionError("exact comparator used at a mixed vertex")
    monkeypatch.setattr(module, "_compare_outgoing", forbidden)
    arrangement = arrange_material(MaterialDomain(1, plane, (circle,)), (chord,), tolerance=1e-9)
    assert len(arrangement.cells) == 2


# --------------------------------------------------------------------------------- cells (step 2d, D3/D5)


def _square_loop(points):
    return tuple(ArrangementPath(LinePath((*a, 0.), (*b, 0.))) for a, b in zip(points, points[1:] + points[:1]))


def test_cell_phase_reproduces_mains_answers_on_merged_and_nested_inputs():
    """Review inputs: a trace end tolerance-merged with a corner (main's area formula sums each
    piece's own stations: 2 cells, area 16), a merged start whose probe main does not certify
    (main refuses: no hole owner), and two nested trace squares 2 tol apart (main's ownership)."""
    plane = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    square = _square_loop([(0., 0.), (4., 0.), (4., 4.), (0., 4.)])

    def run(boundary, traces, **kw):
        return arrange_material(MaterialDomain(1, plane, (boundary,)), traces, tolerance=1e-6, **kw)
    merged = (ArrangementPath(LinePath((5e-7, 5e-7, 0.), (4., 4., 0.)), owners=(1, 2)),)
    for kw in ({}, {"area_tolerance": 1e-12}):
        result = run(square, merged, **kw)
        assert (len(result.cells), result.area) == (2, 16.)
    # main refused "hole has no material owner"; with deferred removal (E1) the refusal identity
    # changes (owner Q-e, listed); the exact answer (2 cells, area 16) is E3's target.
    with pytest.raises(GeometryError, match="failed material conservation"):
        run(_square_loop([(0., 2e-6), (0., 0.), (4., 0.), (4., 4.), (0., 4.)]),
            (ArrangementPath(LinePath((3e-7, 9.5e-7, 0.), (4., 4., 0.)), owners=(1, 2)),), area_tolerance=2e-6)

    def ring(lo, hi, first):
        q = [(lo, lo), (hi, lo), (hi, hi), (lo, hi)]
        return tuple(ArrangementPath(LinePath((*a, 0.), (*b, 0.)), owners=(1, first + i))
                     for i, (a, b) in enumerate(zip(q, q[1:] + q[:1])))
    nested = run(square, ring(1., 3., 2) + ring(1. + 2e-6, 3. - 2e-6, 10), area_tolerance=1e-12)
    assert sorted(len(cell.holes) for cell in nested.cells) == [0, 0, 2] and nested.area == 16.


def _island_arrangement(plane):
    outer = tuple(ArrangementPath(LinePath(tuple(plane.evaluate(*a)), tuple(plane.evaluate(*b))))
                  for a, b in zip(*(lambda q: (q, q[1:] + q[:1]))([(0., 0.), (4., 0.), (4., 4.), (0., 4.)])))
    hole = tuple(ArrangementPath(LinePath(tuple(plane.evaluate(*a)), tuple(plane.evaluate(*b))))
                 for a, b in zip(*(lambda q: (q, q[1:] + q[:1]))([(2.5, 2.5), (2.5, 3.5), (3.5, 3.5), (3.5, 2.5)])))
    island = [(.5, .5), (1.5, .5), (1.5, 1.5), (.5, 1.5)]
    traces = tuple(ArrangementPath(LinePath(tuple(plane.evaluate(*a)), tuple(plane.evaluate(*b))), owners=(1, k))
                   for k, (a, b) in enumerate(zip(island, island[1:] + island[:1])))
    return MaterialDomain(1, plane, (outer, hole)), traces


@pytest.mark.parametrize("plane", (Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)),
                                   Plane((-1., -1., .3), (1., .2, 0.), (-.1, 1., .4))))
def test_islands_and_holes_are_owned_as_on_main(plane):
    """A trace square (island) inside a domain with a hole: the big cell owns the island's outer
    cycle and the domain hole as holes, the island is its own cell (main's answer)."""
    domain, traces = _island_arrangement(plane)
    arrangement = arrange_material(domain, traces, tolerance=1e-9, area_tolerance=1e-12)
    assert len(arrangement.cells) == 2
    assert sorted(len(cell.holes) for cell in arrangement.cells) == [0, 2]     # island outline + domain hole (as main)
    assert arrangement.area == 15.


@pytest.mark.parametrize("direction", (-math.inf, math.inf))
def test_cells_do_not_depend_on_evaluation_or_chart_rounding(monkeypatch, direction):
    plane = Plane((-1., -1., .3), (1., .2, 0.), (-.1, 1., .4))
    domain, traces = _island_arrangement(plane)
    reference = arrange_material(domain, traces, tolerance=1e-9, area_tolerance=1e-12)
    line_evaluate, local_uv, lstsq = LinePath.evaluate, Plane.local_uv, np.linalg.lstsq
    monkeypatch.setattr(LinePath, "evaluate", lambda self, t: np.nextafter(line_evaluate(self, t), direction))
    monkeypatch.setattr(Plane, "local_uv", lambda self, point: tuple(
        math.nextafter(float(value), direction) for value in local_uv(self, point)))
    monkeypatch.setattr(np.linalg, "lstsq", lambda *a, **k: (np.nextafter(lstsq(*a, **k)[0], direction),)
                        + tuple(lstsq(*a, **k)[1:]))
    nudged = arrange_material(domain, traces, tolerance=1e-9, area_tolerance=1e-12)
    assert nudged.cells == reference.cells and nudged.area == reference.area



# --------------------------------------------------------------------------------- T-b/D8 slice E1


@pytest.mark.parametrize("k", (20, 100, 1000))
@pytest.mark.parametrize("tol", (1e-20, 1e-30))
@pytest.mark.parametrize("reverse", (False, True))
def test_nearly_coincident_traces_keep_their_crossing(k, tol, reverse):
    """D8: two lines through (2, 2) at angles 0.3 and 0.3 + k eps. Main merged their split
    parameters within 32 eps on one path while vertex identity kept them apart (unresolved
    bridge); with removal on the shared identity the square has 4 cells and 12 paths."""
    plane = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    eps = np.finfo(float).eps
    traces = []
    for index, angle in enumerate((.3, .3 + k * eps)):
        d = np.array((math.cos(angle), math.sin(angle), 0.))
        traces.append(ArrangementPath(LinePath(tuple(np.array((2., 2., 0.)) - 3 * d),
                                               tuple(np.array((2., 2., 0.)) + 3 * d)), owners=(1, index)))
    if reverse:
        traces.reverse()
    square = _square_loop([(0., 0.), (4., 0.), (4., 4.), (0., 4.)])
    result = arrange_material(MaterialDomain(1, plane, (square,)), tuple(traces), tolerance=tol)
    assert (len(result.cells), len(result.paths)) == (4, 12)


def test_non_transitive_identity_keeps_mains_first_fit_vertices():
    """Three traces fan from separate top points to bottom stations at x = 2, 2 + 0.9 tol and
    2 + 1.8 tol: the second station joins the first vertex, the third (1.8 tol from it) starts a
    new one; the bottom sub-piece between the first two (0.9 tol, one vertex) is removed, the
    one between the second and third (two vertices) is an edge. Hand count: 4 cells; paths 9
    boundary pieces (bottom 3, right 1, top 4, left 1) + 3 traces = 12."""
    plane = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    tol = 1e-6
    traces = []
    for k, (top, bottom) in enumerate(((1., 2.), (2., 2. + .9 * tol), (3., 2. + 1.8 * tol))):
        # the line through (top, 4) and (bottom, 0), extended beyond the square
        slope = (top - bottom) / 4.
        traces.append(ArrangementPath(LinePath((top + slope, 5., 0.), (bottom - slope, -1., 0.)), owners=(1, k)))
    square = _square_loop([(0., 0.), (4., 0.), (4., 4.), (0., 4.)])
    for order in (tuple(traces), tuple(traces[::-1])):
        # The network is not closed (T-b): areas from the pieces' own stations miss ~1.8 tol^1, so
        # conservation at the default area tolerance (= tol) fails here as on main; E2/E3 close it.
        with pytest.raises(GeometryError, match="failed material conservation"):
            arrange_material(MaterialDomain(1, plane, (square,)), order, tolerance=tol)
        result = arrange_material(MaterialDomain(1, plane, (square,)), order, tolerance=tol, area_tolerance=1e-3)
        assert (len(result.cells), len(result.paths)) == (4, 12)


def test_short_clipped_piece_with_overlapping_trace_and_seam():
    """A trace whose clipped piece is 0.5 tol long (deferred removal keeps it to the pair loop)
    and a collinear trace overlapping it that ends inside (two seams through its end): decided
    without refusal in both operand orders -- 4 cells."""
    plane = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    tol = 1e-6
    short = ArrangementPath(LinePath((-1., 2., 0.), (.5 * tol, 2., 0.)), owners=(1, 1))
    long = ArrangementPath(LinePath((-1., 2., 0.), (3., 2., 0.)), owners=(1, 2))
    square = _square_loop([(0., 0.), (4., 0.), (4., 4.), (0., 4.)])
    for order in ((short, long), (long, short)):
        result = arrange_material(MaterialDomain(1, plane, (square,)), order, tolerance=tol)
        assert len(result.cells) == 4


def test_junctions_are_symmetric_in_their_operands():
    """T7 (owner Q-h): the line role follows the canonical data key, so swapping the operands
    swaps every pair, also for skew (tilted-plane) and short pairs."""
    rng = random.Random(12)
    plane = Plane((-1., -1., .3), (1., .2, 0.), (-.1, 1., .4))
    for index in range(10000):
        if index % 3 == 0:
            a, b = planar_line(rng), planar_line(rng)
        elif index % 3 == 1:
            a, b = (LinePath(tuple(plane.evaluate(rng.uniform(-2, 2), rng.uniform(-2, 2))),
                             tuple(plane.evaluate(rng.uniform(-2, 2), rng.uniform(-2, 2)))) for _ in range(2))
        else:
            base = np.array((rng.uniform(-1, 1), rng.uniform(-1, 1), 0.))
            a = LinePath(tuple(base), tuple(base + np.array((rng.uniform(-1, 1) * 1e-9, rng.uniform(-1, 1) * 1e-9, 0.))))
            b = planar_line(rng) if rng.random() < .5 else LinePath(tuple(base), tuple(base + np.array((1e-10, 0., 0.))))
        first, second = st.segment_of(a), st.segment_of(b)
        assert st.junctions(first, second, 1e-9) == tuple(sorted((y, x) for x, y in st.junctions(second, first, 1e-9)))
