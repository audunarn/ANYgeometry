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
