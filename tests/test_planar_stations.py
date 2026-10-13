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


@pytest.mark.parametrize("scale", (1., 1e-100, 1e120))
def test_vertex_lookup_matches_an_exact_scan(scale):
    rng = random.Random(7)
    tol = 1e-9 * scale
    lookup, points = st.ExactVertexLookup(tol), []
    for _ in range(400):
        base = rng.choice(points) if points and rng.random() < .5 else tuple(
            Fraction(rng.uniform(-1, 1) * scale) for _ in range(3))
        offset = rng.choice((0., .5, .999, 1., 1.001, 3.)) * tol
        point = (base[0] + Fraction(offset), base[1], base[2])
        expected = next((i for i, q in enumerate(points) if st.distance2(point, q) <= Fraction(tol) ** 2), None)
        assert lookup.find(point) == expected
        if expected is None:
            lookup.add(point); points.append(point)


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
    plane = Plane((-1., -1., .3), (1., .2, 0.), (-.1, 1., .4))
    domain, traces = crossing_arrangement(plane)
    arrangement = arrange_material(domain, traces, tolerance=1e-9)
    assert len(arrangement.cells) == 7
    sources = [st.segment_of(path.curve) for loop in domain.boundaries for path in loop]
    sources += [st.segment_of(trace.curve) for trace in traces]
    for path in arrangement.paths:
        for end in (path.curve.start, path.curve.end):
            # each stored end is the correct rounding of an exact point on a source line
            assert any(st.point_parameters(source, tuple(map(Fraction, end)), 1e-12) for source in sources)
