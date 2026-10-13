"""Exact planar material containment (robust-predicate slice R1b, step 1).

``MaterialDomain.contains_loop``/``contains`` on a Plane support with straight
loops run on exact rationals with the float route's own rules. References: the
float route itself on generic configurations (identical answers away from
degeneracy), exact distance arithmetic at the tolerance boundary, and chart
rounding perturbation (no influence).
"""
from __future__ import annotations

from fractions import Fraction
import math
import random

import numpy as np
import pytest

from anygeometry import GeometryModel, Plane
from anygeometry.arrangement_geometry import LinePath
from anygeometry.material_arrangement import ArrangementPath, MaterialDomain, _contains_loop_exact
from anygeometry.planar_exact import ExactPlane


def domain_of(loops, plane=None):
    plane = plane or Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    boundaries = tuple(
        tuple(ArrangementPath(LinePath(tuple(plane.evaluate(*a)), tuple(plane.evaluate(*b))))
              for a, b in zip(loop, loop[1:] + loop[:1]))
        for loop in loops)
    return MaterialDomain(1, plane, boundaries)


def float_contains_loop(domain, loop, uv, tolerance, boundary=True):
    """The pre-R1b float route (the body after the exact-route dispatch)."""
    planar = MaterialDomain._exact_planar
    try:
        MaterialDomain._exact_planar = lambda self, loops: False
        return domain.contains_loop(loop, np.asarray(uv, dtype=float), tolerance, boundary=boundary)
    finally:
        MaterialDomain._exact_planar = planar


CONCAVE = [(0., 0.), (4., 0.), (4., 3.), (2., 1.), (0., 3.)]
HOLED = ([(0., 0.), (5., 0.), (5., 5.), (0., 5.)], [(1., 1.), (1., 2.), (2., 2.), (2., 1.)])


@pytest.mark.parametrize("seed", range(4))
def test_exact_route_agrees_with_the_float_route_on_generic_points(seed):
    rng = random.Random(seed)
    angle = rng.uniform(0, math.tau)
    plane = Plane((rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-3, 3)),
                  (math.cos(angle) * 1.3, math.sin(angle) * 1.3, 0.4), (-0.2, 0.7, 1.1))
    for loops in ((CONCAVE,), HOLED):
        domain = domain_of(loops, plane)
        chart = ExactPlane.from_plane(plane)
        for _ in range(250):
            uv = (rng.uniform(-1, 6), rng.uniform(-1, 6))
            for loop in domain.boundaries:
                for boundary in (True, False):
                    expected = float_contains_loop(domain, loop, uv, 1e-9, boundary)
                    got = _contains_loop_exact(chart, loop, tuple(map(Fraction, uv)), 1e-9, boundary=boundary)
                    assert got == expected


@pytest.mark.parametrize("gap,inside", [
    (lambda tol: 0.75 * tol, True), (lambda tol: math.nextafter(tol, 0.), True), (lambda tol: tol, True),
    (lambda tol: math.nextafter(tol, 2.), False), (lambda tol: 1.5 * tol, False),
])
def test_the_tolerance_boundary_is_exact(gap, inside):
    # A point exactly d to the right of the square's edge x = 0 (stored exactly: no rounding of the gap)
    # is "boundary" (contained) iff d <= tol.
    tolerance = 1e-9
    domain = domain_of(([(-1., 0.), (0., 0.), (0., 1.), (-1., 1.)],))
    d = gap(tolerance)
    probe = LinePath((d, .5, 0.), (2., .5, 0.))
    assert domain.contains(probe, 0., tolerance) is inside


def test_contains_does_not_depend_on_chart_rounding(monkeypatch):
    tolerance = 1e-9
    domain = domain_of(([(0., 0.), (1., 0.), (1., 1.), (0., 1.)],))
    probe = LinePath((1. + tolerance, .5, 0.), (2., .5, 0.))
    reference = domain.contains(probe, 0., tolerance)
    original = Plane.local_uv
    monkeypatch.setattr(Plane, "local_uv", lambda self, point: tuple(
        math.nextafter(v, math.inf) for v in original(self, point)))
    assert domain.contains(probe, 0., tolerance) is reference


def test_non_straight_domains_keep_the_float_route():
    from anygeometry.exact_curves import EllipticArc
    plane = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    arc = ArrangementPath(EllipticArc((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), 0., math.pi))
    line = ArrangementPath(LinePath((-1., 0., 0.), (1., 0., 0.)))
    domain = MaterialDomain(1, plane, ((arc, line),))
    assert not domain._exact_planar(domain.boundaries)


# ----------------------------------------------------------------------------- review regressions (R1b round 1)


TRANSLATED = Plane((-1., -1., 0.), (1., 0., 0.), (0., 1., 0.))     # chart origin off the data: rounded charts
WORLD_SQUARE = [(-1., 0., 0.), (0., 0., 0.), (0., 1., 0.), (-1., 1., 0.)]


def world_domain(loops, plane=TRANSLATED):
    boundaries = tuple(tuple(ArrangementPath(LinePath(a, b)) for a, b in zip(loop, loop[1:] + loop[:1]))
                       for loop in loops)
    return MaterialDomain(1, plane, boundaries)


def float_contains(domain, curve, parameter, tolerance):
    planar = MaterialDomain._exact_planar
    try:
        MaterialDomain._exact_planar = lambda self, loops: False
        return domain.contains(curve, parameter, tolerance)
    finally:
        MaterialDomain._exact_planar = planar


def nudge_charts(monkeypatch, direction):
    original = Plane.local_uv
    monkeypatch.setattr(Plane, "local_uv", lambda self, point: tuple(
        math.nextafter(v, direction) for v in original(self, point)))


TOL = 1e-9
GAPS = ((math.nextafter(TOL, 0.), True), (TOL, True), (math.nextafter(TOL, 1.), False))


@pytest.mark.parametrize("gap,inside", GAPS)
def test_reference_changes_on_a_translated_chart(gap, inside):
    """The two documented reference changes: exact boundary distance <= tol is boundary (contained).
    The float route rejects prev(tol) and tol here only because of chart rounding."""
    domain = world_domain((WORLD_SQUARE,))
    probe = LinePath((gap, .5, 0.), (2., .5, 0.))
    assert domain.contains(probe, 0., TOL) is inside
    if inside:
        assert float_contains(domain, probe, 0., TOL) is False


@pytest.mark.parametrize("gap,inside", GAPS)
@pytest.mark.parametrize("direction", (-math.inf, math.inf))
def test_translated_chart_rounding_has_no_influence(monkeypatch, gap, inside, direction):
    domain = world_domain((WORLD_SQUARE,))
    probe = LinePath((gap, .5, 0.), (2., .5, 0.))
    nudge_charts(monkeypatch, direction)
    assert domain.contains(probe, 0., TOL) is inside


def test_the_float_route_is_rounding_dependent_on_this_case(monkeypatch):
    """Discrimination check: the case above does flip on the float route."""
    domain = world_domain((WORLD_SQUARE,))
    probe = LinePath((TOL, .5, 0.), (2., .5, 0.))
    answers = {float_contains(domain, probe, 0., TOL)}
    nudge_charts(monkeypatch, -math.inf)
    answers.add(float_contains(domain, probe, 0., TOL))
    assert answers == {True, False}


@pytest.mark.parametrize("direction", (None, -math.inf, math.inf))
def test_curved_queries_keep_the_float_route(monkeypatch, direction):
    from anygeometry.exact_curves import EllipticArc
    domain = world_domain((WORLD_SQUARE,))
    arc = EllipticArc((0., .5, 0.), (TOL, 0., 0.), (0., .1, 0.), 0., math.pi)
    if direction is not None:
        nudge_charts(monkeypatch, direction)
    assert domain.contains(arc, 0., TOL) is float_contains(domain, arc, 0., TOL)


@pytest.mark.parametrize("direction", (None, -math.inf, math.inf))
def test_mixed_domains_keep_the_float_route(monkeypatch, direction):
    from anygeometry.exact_curves import EllipticArc
    outer = world_domain((WORLD_SQUARE,)).boundaries[0]
    hole = (ArrangementPath(EllipticArc((-.5, .5, 0.), (.1, 0., 0.), (0., .1, 0.), 0., math.pi)),
            ArrangementPath(EllipticArc((-.5, .5, 0.), (.1, 0., 0.), (0., .1, 0.), math.pi, math.pi)))
    domain = MaterialDomain(1, TRANSLATED, (outer, hole))
    probe = LinePath((TOL, .5, 0.), (2., .5, 0.))
    if direction is not None:
        nudge_charts(monkeypatch, direction)
    assert domain.contains(probe, 0., TOL) is float_contains(domain, probe, 0., TOL)
    float_uv = domain.uv(probe, 0.)
    assert domain.contains_loop(outer, float_uv, TOL) is float_contains_loop(domain, outer, float_uv, TOL)


def test_float_query_points_keep_the_float_route(monkeypatch):
    domain = world_domain((WORLD_SQUARE,))
    loop = domain.boundaries[0]
    for direction in (None, -math.inf, math.inf):
        if direction is not None:
            nudge_charts(monkeypatch, direction)
        uv = domain.uv(LinePath((TOL, .5, 0.), (2., .5, 0.)), 0.)
        assert domain.contains_loop(loop, uv, TOL) is float_contains_loop(domain, loop, uv, TOL)


def test_degenerate_boundary_lines_are_refused_as_on_the_float_route():
    from anygeometry.errors import GeometryError
    square = [(0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.)]
    plane = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    loop = (ArrangementPath(LinePath((0., 0., 0.), (0., 0., 0.))),) + world_domain((square,), plane).boundaries[0]
    domain = MaterialDomain(1, plane, (loop,))
    with pytest.raises(GeometryError, match="degenerate arrangement line"):
        float_contains_loop(domain, loop, (0., 0.), TOL)
    with pytest.raises(GeometryError, match="degenerate arrangement line"):
        domain.contains_loop(loop, (Fraction(0), Fraction(0)), TOL)


def test_exact_query_points_are_kept_in_any_sequence():
    domain = world_domain((WORLD_SQUARE,))
    chart = ExactPlane.from_plane(TRANSLATED)
    uv = chart.uv((Fraction(TOL) + Fraction(1, 10**30), Fraction(1, 2), Fraction(0)))
    loop = domain.boundaries[0]
    assert domain.contains_loop(loop, uv, TOL) is domain.contains_loop(loop, list(uv), TOL) is False


# Vertex crossings and runs of horizontal edges on integer data (the float route is exact there too).
STAIRS = [(0., 0.), (6., 0.), (6., 2.), (4., 2.), (3., 2.), (3., 4.), (2., 4.), (2., 2.), (0., 2.)]
ZIGZAG = [(0., 0.), (2., 1.), (4., 0.), (6., 1.), (6., 3.), (4., 2.), (2., 3.), (0., 2.)]


@pytest.mark.parametrize("loop", (STAIRS, ZIGZAG))
def test_vertex_levels_and_horizontal_runs_agree_with_the_float_route(loop):
    domain = domain_of((loop,))
    chart = ExactPlane.from_plane(domain.support)
    for v in (0., .5, 1., 2., 3., 4.):
        for u in (-1., 0., 1., 2., 2.5, 3., 3.5, 4., 5., 6., 7.):
            for boundary in (True, False):
                expected = float_contains_loop(domain, domain.boundaries[0], (u, v), 1e-9, boundary)
                got = _contains_loop_exact(chart, domain.boundaries[0], (Fraction(u), Fraction(v)), 1e-9,
                                           boundary=boundary)
                assert got == expected, (u, v, boundary)


@pytest.mark.parametrize("boundary", (True, False))
def test_bezier_stations_on_a_hole_boundary(boundary):
    """A constant Bezier station on the hole edge: boundary of the hole, so material iff boundary=True."""
    from anygeometry.arrangement_geometry import BezierPath
    plane = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    outer = [(0., 0., 0.), (5., 0., 0.), (5., 5., 0.), (0., 5., 0.)]
    hole = [(1., 1., 0.), (1., 2., 0.), (2., 2., 0.), (2., 1., 0.)]
    domain = world_domain((outer, hole), plane)
    station = BezierPath(((1., 1.5, 0.), (1., 1.5, 0.), (1., 1.5, 0.)))
    assert domain.contains(station, .5, TOL, boundary=boundary) is boundary
    inside_hole = BezierPath(((1.5, 1.5, 0.), (1.5, 1.5, 0.)))
    assert domain.contains(inside_hole, .5, TOL, boundary=boundary) is False


@pytest.mark.parametrize("offset", (-2.**-50, -2.**-52, 0., 2.**-52, 2.**-50))
def test_endpoint_windows_near_vertex_levels_agree_with_the_float_route(offset):
    """Query levels within a few eps of a vertex level exercise the 8-eps endpoint windows."""
    for loop in (STAIRS, ZIGZAG):
        domain = domain_of((loop,))
        chart = ExactPlane.from_plane(domain.support)
        for v in (1. + offset, 2. + offset):
            for u in (-1., 1., 2.5, 3.5, 5., 7.):
                expected = float_contains_loop(domain, domain.boundaries[0], (u, v), 1e-9, True)
                got = _contains_loop_exact(chart, domain.boundaries[0], (Fraction(u), Fraction(v)), 1e-9)
                assert got == expected, (loop is STAIRS, u, v)
