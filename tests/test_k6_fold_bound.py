"""K6: the rigorous error bound of the float fold coefficients (``exact_curves._fold_coefficient_bound``) and the
undecidable-fold classification it drives (``_fold_undecidable``)."""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import pytest

from anygeometry.analytic_supports import undecidable_contact
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


@pytest.mark.parametrize("scale", [2.**-40, 1., 50., 1e4])
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


def test_the_bound_is_none_for_non_finite_input():
    rng = np.random.default_rng(1)
    first, second = _pair(rng, 1.)
    broken = _CylinderSupport(origin=(math.inf, 0., 0.), axis=second.axis, radial_direction=second.radial_direction,
                              radius=second.radius, height=second.height, start_angle=second.start_angle,
                              sweep_angle=second.sweep_angle)
    assert _fold_coefficient_bound(first, broken, 1) is None


def test_the_magnitude_gate_declines_tiny_radii():
    """Codex K6 R1 M7: radii 2^-530 underflow the bound's terms; such pairs are outside the K3 magnitude range and
    are never classified undecidable (main's path)."""
    r1 = 2.**-530
    first = _CylinderSupport.from_surface(Cylinder(origin=(0., 0., 0.), axis=(0., 0., 1.), radial_direction=(0., 1., 0.),
                                                   radius=r1, height=1., start_angle=0., sweep_angle=2*math.pi))
    second = _CylinderSupport.from_surface(Cylinder(origin=(0., 0., 0.), axis=tuple(_unit((1., 1., 0.))),
                                                    radial_direction=(0., 0., 1.), radius=float(np.nextafter(r1, 1.)),
                                                    height=1., start_angle=0., sweep_angle=2*math.pi))
    assert _fold_undecidable(first, second) == {}


def test_an_ill_conditioned_undecidable_pair_offers_no_chart():
    """Codex K6 R1 M5: axes 1.7 degrees apart (sin^2 < 2^-8), equal radii within one ulp, axes intersecting: both folds
    undecidable but no admitted chart (the pair stage then refuses wherever a contact point is in both faces)."""
    u = _unit((.03, 0., 1.))
    first = Cylinder(origin=(0., 0., -1.), axis=(0., 0., 1.), radial_direction=(0., 1., 0.), radius=1., height=2.,
                     start_angle=-.5, sweep_angle=1.)
    second = Cylinder(origin=tuple(-u), axis=tuple(u), radial_direction=(0., 1., 0.), radius=float(np.nextafter(1., 2.)),
                      height=2., start_angle=-.5, sweep_angle=1.)
    contact = undecidable_contact(first, second, tolerance=1e-9)
    assert contact is not None and len(contact.points) == 2 and contact.support is None


def test_the_bound_never_rounds_to_zero():
    """Codex K6 R2 M5: the error propagation is exact rational arithmetic, so a pair whose float terms are tiny still
    gets a positive bound that covers the coefficient error (here axes 2^-29 apart, radii 2^-64)."""
    x = 2.**-29
    r = 2.**-64
    first = _CylinderSupport.from_surface(Cylinder(origin=(0., 0., 0.), axis=(1., 0., 0.), radial_direction=(0., 0., 1.),
                                                   radius=r, height=1., start_angle=0., sweep_angle=2*math.pi))
    second = _CylinderSupport.from_surface(Cylinder(origin=(0., 0., 0.), axis=tuple(_unit((1., x, 0.))),
                                                    radial_direction=(0., 0., 1.), radius=float(np.nextafter(r, 1.)),
                                                    height=1., start_angle=0., sweep_angle=2*math.pi))
    for sign in (-1, 1):
        bound = _fold_coefficient_bound(first, second, sign)
        floated, exact = _discriminants(first, second, sign)
        assert bound > 0 and abs(floated-exact) <= bound


def test_axes_nearer_parallel_than_the_floor_are_not_classified():
    """Codex K6 R2 M5: axes 2^-537 apart are below the classification floor (main's path)."""
    x = 1.25*2.**-537
    r = 2.**-64
    first = _CylinderSupport.from_surface(Cylinder(origin=(0., 0., 0.), axis=(1., 0., 0.), radial_direction=(0., 0., 1.),
                                                   radius=r, height=1., start_angle=0., sweep_angle=2*math.pi))
    second = _CylinderSupport.from_surface(Cylinder(origin=(0., 0., 0.), axis=(1., x, 0.), radial_direction=(0., 0., 1.),
                                                    radius=float(np.nextafter(r, 1.)), height=1., start_angle=0.,
                                                    sweep_angle=2*math.pi))
    assert _fold_undecidable(first, second) == {}


def test_the_ellipse_residual_allowance_uses_unprojected_lengths():
    """Codex K6 R2 H3: origins 1e7 m along the axes; projecting the long axial offset cancels, so an allowance on
    projected lengths certified ellipses 3e-10 m off the cylinder at tolerance 1e-10. They are not certified now."""
    L = 1e7+1/13
    first = Cylinder(origin=tuple(-L*np.asarray((.6, .8, 0.))), axis=(.6, .8, 0.), radial_direction=(0., 0., 1.),
                     radius=1., height=2*L, start_angle=0., sweep_angle=2*math.pi)
    second = Cylinder(origin=(0., 0., -L), axis=(0., 0., 1.), radial_direction=(1., 0., 0.),
                      radius=float(np.nextafter(1., 0.)), height=2*L, start_angle=0., sweep_angle=2*math.pi)
    contact = undecidable_contact(first, second, tolerance=1e-10)
    assert contact is not None and contact.support is None
