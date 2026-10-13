"""Exact containment for loops of lines and elliptic arcs on planes (robust-predicate slice R1c, E4).

References: main's float rules on generic inputs (identical answers away from
degeneracy), exact geometry at degenerate rays (a horizontal ray through an
arc vertex) and at the tolerance band, and chart rounding perturbation.
"""
from __future__ import annotations

from fractions import Fraction
import math
import random

import numpy as np
import pytest

from anygeometry import Plane
from anygeometry.analytic_roots import _first_harmonic_angles
from anygeometry.arrangement_geometry import LinePath, point_parameters
from anygeometry.exact_curves import EllipticArc
from anygeometry.exact_predicates import Canonical
from anygeometry.material_arrangement import ArrangementPath, MaterialDomain
from anygeometry.planar_arcs import arc_point_rule, contains_loop_exact_curved, first_harmonic_angles
from anygeometry.planar_exact import ExactPlane


def float_contains_loop(domain, loop, uv, tolerance, boundary=True):
    planar, curved = MaterialDomain._exact_planar, MaterialDomain._exact_curved_planar
    try:
        MaterialDomain._exact_planar = MaterialDomain._exact_curved_planar = lambda self, loops: False
        return domain.contains_loop(loop, np.asarray(uv, dtype=float), tolerance, boundary=boundary)
    finally:
        MaterialDomain._exact_planar, MaterialDomain._exact_curved_planar = planar, curved


def ring_loop(plane, center, radius, pieces=4, clockwise=False):
    """A circle of ``pieces`` arcs in the plane (world frame of the chart axes)."""
    u = np.asarray(plane.u_vector) / np.linalg.norm(plane.u_vector)
    n = np.cross(plane.u_vector, plane.v_vector)
    v = np.cross(n / np.linalg.norm(n), u)
    c = np.asarray(plane.evaluate(*center))
    sweep = (-1 if clockwise else 1) * math.tau / pieces
    return tuple(ArrangementPath(EllipticArc(tuple(c), tuple(radius * u), tuple(radius * v), k * sweep, sweep))
                 for k in range(pieces))


def square_loop(plane, lo, hi):
    corners = [(lo, lo), (hi, lo), (hi, hi), (lo, hi)]
    return tuple(ArrangementPath(LinePath(tuple(plane.evaluate(*a)), tuple(plane.evaluate(*b))))
                 for a, b in zip(corners, corners[1:] + corners[:1]))


PLANES = (Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)),
          Plane((.37, -1.21, .53), (.8, .2, .1), (-.15, .9, .3)))


def test_first_harmonic_angles_match_mains_roots():
    rng = random.Random(1)
    engine = Canonical(2)
    for _ in range(300):
        c0, c1, c2 = (Fraction(rng.uniform(-1, 1)) for _ in range(3))
        expected = sorted(_first_harmonic_angles(c0, c1, c2))
        got = sorted(float(engine.box(0).lo + (a.theta.box(engine).lo + a.theta.box(engine).hi) / 2)
                     for a in first_harmonic_angles(c0, c1, c2))
        assert len(got) == len(expected)
        assert all(abs(x - y) <= 1e-12 for x, y in zip(got, expected))


@pytest.mark.parametrize("plane", PLANES)
@pytest.mark.parametrize("clockwise", (False, True))
def test_exact_route_agrees_with_the_float_route_on_generic_points(plane, clockwise):
    rng = random.Random(7)
    outer = square_loop(plane, 0., 4.)
    hole = ring_loop(plane, (2., 2.), 1., pieces=rng.choice((2, 3, 4)), clockwise=not clockwise)
    domain = MaterialDomain(1, plane, (outer, hole))
    chart = ExactPlane.from_plane(plane)
    for _ in range(120):
        uv = (rng.uniform(-.5, 4.5), rng.uniform(-.5, 4.5))
        for loop in domain.boundaries:
            for boundary in (True, False):
                expected = float_contains_loop(domain, loop, uv, 1e-9, boundary)
                got = contains_loop_exact_curved(chart, loop, tuple(map(Fraction, uv)), 1e-9, boundary=boundary)
                assert got == expected, (uv, boundary)


def test_ellipse_arcs_agree_with_the_float_route():
    rng = random.Random(9)
    plane = PLANES[1]
    chart = ExactPlane.from_plane(plane)
    center = tuple(plane.evaluate(2., 2.))
    u, v = np.asarray(plane.u_vector) * 1.3, np.asarray(plane.v_vector) * .7     # an ellipse in the plane
    loop = (ArrangementPath(EllipticArc(center, tuple(u), tuple(v), .3, math.pi)),
            ArrangementPath(LinePath(tuple(np.asarray(center) + u * math.cos(.3 + math.pi) + v * math.sin(.3 + math.pi)),
                                     tuple(np.asarray(center) + u * math.cos(.3) + v * math.sin(.3)))))
    domain = MaterialDomain(1, plane, (loop,))
    for _ in range(200):
        uv = (rng.uniform(.3, 3.7), rng.uniform(.3, 3.7))
        expected = float_contains_loop(domain, loop, uv, 1e-9)
        assert contains_loop_exact_curved(chart, loop, tuple(map(Fraction, uv)), 1e-9) == expected, uv


@pytest.mark.parametrize("plane", PLANES)
@pytest.mark.parametrize("direction", (None, -math.inf, math.inf))
def test_ray_through_an_arc_vertex_is_decided_exactly(monkeypatch, plane, direction):
    """Stations at the hole centre's height: the horizontal ray passes through the vertex between
    two arcs (main's half-open vertex rule decided on rounded roots, arc_opening flips)."""
    outer = square_loop(plane, 0., 4.)
    hole = ring_loop(plane, (2., 2.), 1., pieces=4, clockwise=True)
    domain = MaterialDomain(1, plane, (outer, hole))
    original = Plane.local_uv
    if direction is not None:
        monkeypatch.setattr(Plane, "local_uv", lambda self, point: tuple(
            math.nextafter(float(value), direction) for value in original(self, point)))
    for x, material in ((2.6, False), (2.0, False), (.5, True), (3.5, True), (1.1, False)):
        station = LinePath(tuple(plane.evaluate(x, 2.)), tuple(plane.evaluate(x + .01, 2.)))
        assert domain.contains(station, 0., 1e-9) is material, x


@pytest.mark.parametrize("direction", (None, -math.inf, math.inf))
@pytest.mark.parametrize("gap,material", ((-1.5, False), (-.5, True), (.5, True), (1.5, True)))
def test_arc_tolerance_band(monkeypatch, direction, gap, material):
    """Stations ``gap * tol`` (horizontally) from the hole circle on the line y = 2.6 of the flat chart:
    main's rule tests the horizontal distance to the ray crossing, so within tol the hole's
    boundary is not hole (material); beyond tol inside the hole is not material."""
    plane = PLANES[0]
    tolerance = 1e-9
    domain = MaterialDomain(1, plane, (square_loop(plane, 0., 4.), ring_loop(plane, (2., 2.), 1., clockwise=True)))
    crossing = 2. + math.sqrt(1. - .6 ** 2)                    # within 1e-15 of the exact crossing
    station = LinePath((crossing + gap * tolerance, 2.6, 0.), (3.9, 2.6, 0.))
    if direction is not None:
        original = Plane.local_uv
        monkeypatch.setattr(Plane, "local_uv", lambda self, point: tuple(
            math.nextafter(float(value), direction) for value in original(self, point)))
    assert domain.contains(station, 0., tolerance) is material


# --------------------------------------------------------------------------------- review regressions (R1c round 1)


def test_roots_at_huge_start_angles_are_found_by_certified_turns():
    """An arc starting at 1e18 rad: its root at the start lies eight turns from a float-centred
    search; main finds it (True)."""
    s = 1e18
    c, z = math.cos(s), math.sin(s)
    plane = PLANES[0]
    loop = (ArrangementPath(EllipticArc((0., 0., 0.), (c, -z, 0.), (z, c, 0.), s, math.tau)),)
    domain = MaterialDomain(1, plane, (loop,))
    chart = ExactPlane.from_plane(plane)
    expected = float_contains_loop(domain, loop, (1., 0.), 1e-9)
    assert expected is True
    assert contains_loop_exact_curved(chart, loop, (Fraction(1), Fraction(0)), 1e-9) is True


def test_coincident_arcs_accept_points_near_their_endpoints():
    """A flat ellipse (U = (0, 1e-10, 0)) lies on the horizontal chart level: main's point rule
    accepts the query 5e-10 from the first arc's endpoint."""
    plane = PLANES[0]
    loop = (ArrangementPath(EllipticArc((0., 0., 0.), (0., 1e-10, 0.), (1., 0., 0.), 0., math.pi)),
            ArrangementPath(EllipticArc((0., 0., 0.), (0., 1e-10, 0.), (1., 0., 0.), math.pi, math.pi)))
    domain = MaterialDomain(1, plane, (loop,))
    chart = ExactPlane.from_plane(plane)
    uv = (0., 6e-10)
    assert float_contains_loop(domain, loop, uv, 1e-9) is True
    assert contains_loop_exact_curved(chart, loop, tuple(map(Fraction, uv)), 1e-9) is True


def test_clamped_parameters_are_evaluated_at_their_endpoint():
    """A root just past the float end angle: main clamps t to 1 and evaluates the endpoint, which
    is farther than tol (1e-20) from the query."""
    plane = PLANES[0]
    start, sweep = .1, 1.
    arc = EllipticArc((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), start, sweep)
    chord = LinePath(tuple(arc.evaluate(1.)), tuple(arc.evaluate(0.)))
    loop = (ArrangementPath(arc), ArrangementPath(chord))
    domain = MaterialDomain(1, plane, (loop,))
    chart = ExactPlane.from_plane(plane)
    h = Fraction(24524208531525427, 40000000000000000)
    uv = ((1 - h * h) / (1 + h * h), 2 * h / (1 + h * h))
    assert contains_loop_exact_curved(chart, loop, uv, 1e-20) is False


def test_arc_point_rule_agrees_with_point_parameters():
    rng = random.Random(4)
    for _ in range(300):
        center = tuple(rng.uniform(-1, 1) for _ in range(3))
        u = np.array((rng.uniform(.5, 2), rng.uniform(-.3, .3), 0.))
        v = np.array((rng.uniform(-.3, .3), rng.uniform(.5, 2), rng.uniform(-.2, .2)))
        start, sweep = rng.uniform(-math.pi, math.pi), rng.choice((-1, 1)) * rng.uniform(.2, math.tau)
        arc = EllipticArc(center, tuple(u), tuple(v), start, sweep)
        angle = rng.uniform(-4, 4)
        point = tuple(np.asarray(center) + u * math.cos(angle) + v * math.sin(angle)
                      + np.array(rng.choice(((0., 0., 0.), (1e-3, 0., 0.), (0., 0., 1e-12)))))
        expected = bool(point_parameters(arc, point, tolerance=1e-9))
        assert arc_point_rule(arc, tuple(map(Fraction, point)), 1e-9) is expected
