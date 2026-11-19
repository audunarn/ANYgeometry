"""K6: the rigorous error bound of the float fold coefficients (``exact_curves._fold_coefficient_bound``) and the
undecidable-fold classification it drives (``_fold_undecidable``)."""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import pytest

from anygeometry.exact_curves import _CylinderSupport, _fold_coefficient_bound, _fold_undecidable
from anygeometry.surfaces import Cylinder


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    return vector/np.linalg.norm(vector)


def _pair(rng, scale):
    a1 = _unit(rng.normal(size=3))
    radial = _unit(np.cross(a1, rng.normal(size=3)))
    a2 = _unit(rng.normal(size=3))
    first = Cylinder(origin=tuple(scale*rng.normal(size=3)), axis=tuple(a1), radial_direction=tuple(radial),
                     radius=float(abs(rng.normal())*scale/4+.1), height=10., start_angle=0., sweep_angle=2*math.pi)
    second = Cylinder(origin=tuple(scale*rng.normal(size=3)), axis=tuple(a2),
                      radial_direction=tuple(_unit(np.cross(a2, rng.normal(size=3)))),
                      radius=float(abs(rng.normal())*scale/4+.1), height=10., start_angle=0., sweep_angle=2*math.pi)
    return _CylinderSupport.from_surface(first), _CylinderSupport.from_surface(second)


def _discriminants(first, second, sign):
    """``(D of the float coefficients, exact D of the stored doubles)`` as exact rationals; the irrational
    ``|a1 x a2|`` of the exact D is replaced by a 60-digit rational, far inside the tested bounds."""
    q = lambda vector: [Fraction(float(value)) for value in np.asarray(vector, dtype=float)]
    cross = lambda u, v: [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
    dot = lambda u, v: sum(x*y for x, y in zip(u, v))
    a1, a2 = np.asarray(first.axis), np.asarray(second.axis)
    c = np.cross(a1, a2)
    magnitude = float(np.linalg.norm(c))
    e1 = np.asarray(first.radial_direction)
    n0 = float((np.asarray(first.origin)-np.asarray(second.origin)) @ c)
    nc, ns = first.radius*float(e1 @ c), first.radius*float(np.cross(a1, e1) @ c)
    c0 = n0-sign*magnitude*second.radius
    floated = Fraction(nc)**2+Fraction(ns)**2-Fraction(c0)**2
    A1, E1 = q(a1), q(e1)
    C, E2 = cross(A1, q(a2)), cross(A1, E1)
    N0 = dot([x-y for x, y in zip(q(first.origin), q(second.origin))], C)
    from decimal import Decimal, getcontext
    getcontext().prec = 60
    norm = Fraction((Decimal(dot(C, C).numerator)/Decimal(dot(C, C).denominator)).sqrt())
    R1, R2 = Fraction(float(first.radius)), Fraction(float(second.radius))
    exact = (R1*dot(E1, C))**2+(R1*dot(E2, C))**2-(N0-sign*norm*R2)**2
    return floated, exact


@pytest.mark.parametrize("scale", [1., 50., 1e4])
def test_the_bound_covers_the_float_coefficient_error_on_random_pairs(scale):
    rng = np.random.default_rng(20261119)
    for _ in range(200):
        first, second = _pair(rng, scale)
        for sign in (-1, 1):
            bound = _fold_coefficient_bound(first, second, sign)
            floated, exact = _discriminants(first, second, sign)
            assert abs(floated-exact) <= Fraction(bound), (scale, sign)


@pytest.mark.parametrize("scale", [1., 50., 1e4])
def test_ordinary_random_pairs_have_no_undecidable_fold(scale):
    rng = np.random.default_rng(7)
    for _ in range(200):
        first, second = _pair(rng, scale)
        assert _fold_undecidable(first, second) == {}


def test_the_bound_is_infinite_for_non_finite_input():
    rng = np.random.default_rng(1)
    first, second = _pair(rng, 1.)
    broken = _CylinderSupport(origin=(math.inf, 0., 0.), axis=second.axis, radial_direction=second.radial_direction,
                              radius=second.radius, height=second.height, start_angle=second.start_angle,
                              sweep_angle=second.sweep_angle)
    assert _fold_coefficient_bound(first, broken, 1) == math.inf
