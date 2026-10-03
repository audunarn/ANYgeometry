"""Exact algebra of a Bezier extrusion cut with a quadric (the polynomial-chart plan).

``Q(c(t) + s d) = A s^2 + B(t) s + C(t)`` with a constant ``A``: the polynomials are checked exactly against the
quadric at dyadic points, the events against brute force, and the degenerate relations (coincidence, parallel
rulings, tangency) against their closed forms.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from anygeometry.branch_algebra import (BernsteinForm, BezierRuledSupport, build_abc, get_poly_plan, poly_roots)
from anygeometry.errors import GeometryError
from anygeometry.extrusions import BezierDirectrix, EllipseDirectrix
from anygeometry.quadric_algebra import QuadricSupport
from anygeometry.surfaces import Cone, Cylinder, ExtrudedSurface

CUBIC = ((0., 0., 0.), (1., 2., 0.), (2., -1., 0.), (3., 1., 0.))
QUARTIC = ((0., 0., 0.), (1., 2., 0.), (2., -1., 0.), (3., 2., 0.), (4., 0., 0.))
VECTOR = (.25, 0., 1.5)


def ruled(controls=CUBIC, vector=VECTOR):
    return BezierRuledSupport(controls, vector)


def tube():
    return ExtrudedSurface(EllipseDirectrix((1.5, 0., .5), (1., 0., 0.), (0., 0., .7)), (0., 1.2, .3))


def parabola_wall():
    return ExtrudedSurface(BezierDirectrix(((0., 0., 0.), (1., 2., 0.), (2., 0., 0.))), (0., 0., 2.))


SECONDS = {
    "a cylinder": Cylinder((-2., .4, .6), (1., 0., 0.), (0., 1., 0.), .7, 8.),
    "a cone": Cone((-2., .3, .8), (1., 0., 0.), (0., 1., 0.), .15, 1.1, 8.),
    "an elliptic tube": tube(),
    "a parabolic wall": parabola_wall(),
}


def exact_point(support, t, s):
    """``c(t) + s d`` exactly (Bernstein form in Fractions) for Fraction ``t`` and ``s``."""
    n = support.degree
    weights = [math.comb(n, k) * t ** k * (1 - t) ** (n - k) for k in range(n + 1)]
    controls = [[Fraction(v) for v in point] for point in support.controls]
    d = [Fraction(v) for v in support.direction]
    return tuple(sum(w * p[axis] for w, p in zip(weights, controls)) + s * d[axis] for axis in range(3))


def exact_value(quadric, point):
    """``Q(point)`` exactly, from the exact form about the origin ``origin``."""
    m, l, c = quadric.exact_relative((0, 0, 0))
    return (sum(point[i] * m[i][j] * point[j] for i in range(3) for j in range(3))
            + 2 * sum(l[i] * point[i] for i in range(3)) + c)


def horner(poly, x):
    value = Fraction(0)
    for coefficient in reversed(poly):
        value = value * x + coefficient
    return value


@pytest.mark.parametrize("controls", [CUBIC, QUARTIC], ids=["cubic", "quartic"])
@pytest.mark.parametrize("name", SECONDS)
def test_the_polynomials_reproduce_the_quadric_exactly_along_every_ruling(name, controls):
    support, quadric = ruled(controls), QuadricSupport.from_surface(SECONDS[name])
    A, B, C = build_abc(support, quadric)
    assert len(A) == 1 and len(B) <= support.degree + 1 and len(C) <= 2 * support.degree + 1
    for t, s in ((Fraction(3, 8), Fraction(-5, 4)), (Fraction(1, 4), Fraction(7, 8)), (Fraction(0), Fraction(1, 2)),
                 (Fraction(1), Fraction(-1, 8)), (Fraction(5, 16), Fraction(0))):
        expected = exact_value(quadric, exact_point(support, t, s))
        assert A[0] * s * s + horner(B, t) * s + horner(C, t) == expected               # exact, not merely close


def test_the_float_system_agrees_with_the_exact_polynomials_and_their_derivatives():
    support, second = ruled(), SECONDS["a cylinder"]
    plan = get_poly_plan(support, second)
    fs = plan.floats()
    rng = np.random.default_rng(1)
    for t in rng.uniform(0., 1., 6):
        a, b, c = fs.abc_scalar(float(t))
        assert (a, b, c) == pytest.approx((float(plan.A[0]), float(horner(plan.B, Fraction(float(t)))),
                                           float(horner(plan.C, Fraction(float(t))))), rel=1e-12, abs=1e-12)
        step = 1e-6
        for order in (1, 2):                                          # derivatives against central differences
            numeric = [(x - y) / (2 * step) for x, y in zip(fs.abc_scalar(t + step, order - 1),
                                                           fs.abc_scalar(t - step, order - 1))]
            assert fs.abc_scalar(t, order) == pytest.approx(numeric, rel=1e-5, abs=1e-5)
        assert fs.discriminant_scalar(t) == pytest.approx(b * b - 4 * a * c, rel=1e-12, abs=1e-9)
    p, d1, d2 = fs.frame(np.array([.3, .8]))
    assert p.shape == d1.shape == d2.shape == (2, 3)
    assert np.allclose(fs.origin + p, [support_point(support, .3), support_point(support, .8)], atol=1e-13)


def support_point(support, t):
    n = support.degree
    return sum(math.comb(n, k) * t ** k * (1 - t) ** (n - k) * np.asarray(support.controls[k]) for k in range(n + 1))


def test_the_h_form_is_the_exact_difference_quotient_of_the_discriminant():
    """``Delta(anchor + extent u) - Delta(anchor) = extent u h(u)``: the radical of a fold chart, from exact coefficients."""
    plan = get_poly_plan(ruled(), SECONDS["a cone"])
    fs = plan.floats()
    for anchor, extent in ((.3, .4), (.3, -.25), (.7, .3), (.7, -.65), (0., 1.), (1., -1.)):
        form = fs.h_form(anchor, extent)
        for u in (0., 1e-9, 1e-4, .3, .77, 1.):
            exact = horner(plan.disc, Fraction(anchor) + Fraction(extent) * Fraction(u)) - horner(plan.disc, Fraction(anchor))
            made = extent * u * form(u)
            assert abs(made - float(exact)) <= 16 * (form.degree + 2) * np.finfo(float).eps * form.bound * abs(extent * u) + 1e-300
        assert np.array_equal(form.many(np.array([0., .3, 1.])), np.array([form(0.), form(.3), form(1.)]))
    assert fs.h_form(.3, .4) is fs.h_form(.3, .4)                                  # one form per chart end


def test_the_g_form_is_the_radicand_of_a_fold_chart_as_one_polynomial():
    """``g = Delta(anchor + extent u) - Delta(anchor) = extent u h``: the range of ``g`` over a piece has none of the
    dependency that the product of the ranges of ``u`` and ``h`` would carry."""
    plan = get_poly_plan(ruled(), SECONDS["a cone"])
    fs = plan.floats()
    for anchor, extent in ((.3, .4), (.7, -.65), (0., 1.), (1., -1.)):
        g = fs.g_form(anchor, extent)
        assert g(0.) == 0. and g.coeffs[0] == 0.                                   # vanishes at the fold exactly
        for u in (1e-9, 1e-4, .3, .77, 1.):
            exact = horner(plan.disc, Fraction(anchor) + Fraction(extent) * Fraction(u)) - horner(plan.disc, Fraction(anchor))
            assert abs(g(u) - float(exact)) <= 16 * (g.degree + 2) * np.finfo(float).eps * g.bound
            assert g.range(u, min(1., u * 1.05))[0] <= float(exact) <= g.range(u, min(1., u * 1.05))[1]
        assert g.range(0., 1e-9)[0] <= 0. <= g.range(0., 1e-9)[1]
        assert fs.g_form(anchor, extent) is fs.g_form(anchor, extent)
        h = fs.h_form(anchor, extent)
        for u in (.2, .9):
            assert g(u) == pytest.approx(extent * u * h(u), rel=1e-12)             # the two forms are one quotient apart


@pytest.mark.parametrize("name", SECONDS)
def test_the_discriminant_roots_are_the_folds_of_the_branch(name):
    plan = get_poly_plan(ruled(), SECONDS[name])
    fs = plan.floats()
    roots = plan.discriminant_roots()
    grid = np.linspace(0., 1., 4001)
    values = np.array([fs.discriminant_scalar(float(t)) for t in grid])
    changes = grid[:-1][values[:-1] * values[1:] < 0]
    simple = [t for t, m in roots if m % 2 == 1]
    assert len(changes) <= len(simple) + 1
    for t in changes:                                                  # every sign change has a root next to it
        assert min(abs(t - r) for r in simple) < 2.5e-4
    for t in simple:                                                   # and every simple root a sign change
        assert fs.discriminant_scalar(max(0., t - 1e-5)) * fs.discriminant_scalar(min(1., t + 1e-5)) <= 0.


def test_a_double_contact_is_a_root_of_even_multiplicity_found_exactly():
    # a cylinder whose axis touches the cubic's apex generator from one side: the discriminant has a double root
    support = BezierRuledSupport(((0., 0., 0.), (1., 2., 0.), (2., 2., 0.), (3., 0., 0.)), (0., 0., 2.))
    # y(t) = 6 t (1 - t): maximum 1.5 at t = 1/2 where the profile is horizontal, x(t) = 3 t
    touching = Cylinder((-2., 2.5, 1.), (1., 0., 0.), (0., 1., 0.), 1., 8.)         # lowest line y = 1.5 at z = 1
    plan = get_poly_plan(support, touching)
    roots = plan.discriminant_roots()
    assert any(abs(t - .5) < 1e-12 and m == 2 for t, m in roots)


@pytest.mark.parametrize("height", [.2, .35, .6, 1.1])
def test_the_resultant_with_a_plane_is_the_crossing_of_the_wall_section_with_the_pipe(height):
    """At height ``h`` the wall's section is the profile moved by ``h / d_z``; the branch meets the plane exactly
    where that section crosses the pipe, which a sampled sign change of the pipe's implicit value counts."""
    support, second = ruled(), SECONDS["a cylinder"]
    quadric = QuadricSupport.from_surface(second)
    plan = get_poly_plan(support, second)
    roots = plan.resultant_roots(QuadricSupport("plane", (0., 0., height), (0., 0., 1.)))
    shift = np.asarray(support.direction) * (height / support.direction[2])
    grid = np.linspace(0., 1., 4001)
    values = np.array([float(quadric.value(support_point(support, t) + shift)) for t in grid])
    crossings = grid[:-1][values[:-1] * values[1:] < 0]
    assert len(roots) == len(crossings)
    for t, _m in roots:                                                # each root sits next to a sampled crossing
        assert min(abs(t - c) for c in crossings) < 2.5e-4
        assert abs(float(quadric.value(support_point(support, t) + shift))) < 1e-9


def test_the_ruling_stations_where_a_branch_crosses_a_height_are_the_bound_roots():
    support, second = ruled(), SECONDS["a cone"]
    plan = get_poly_plan(support, second)
    for s0 in (.2, .9):
        for t, _m in plan.bound_roots(s0):
            point = fs_point(plan, t, s0)
            assert abs(float(QuadricSupport.from_surface(second).value(point))) < 1e-9


def fs_point(plan, t, s):
    fs = plan.floats()
    return fs.origin + fs.frame(np.array(t))[0] + s * fs.direction


def test_parallel_rulings_give_whole_generators_as_common_roots():
    support = BezierRuledSupport(CUBIC, (0., 0., 2.))
    pipe = Cylinder((1.5, .3, -1.), (0., 0., 1.), (1., 0., 0.), 1.1, 5.)
    plan = get_poly_plan(support, pipe)
    assert plan.linear and all(v == 0 for v in plan.B)                 # no branch: only generators
    roots = plan.common_roots()
    assert roots
    for t, _m in roots:                                                # each root: the ruling lies on the pipe
        for s in (0., .5, 1.):
            point = fs_point(plan, t, s)
            assert abs(float(QuadricSupport.from_surface(pipe).value(point))) < 1e-9


def test_a_quadratic_wall_is_coincident_with_its_own_parabolic_quadric():
    wall = parabola_wall()
    support = BezierRuledSupport.from_surface(wall)
    assert get_poly_plan(support, wall).coincident
    shifted = ExtrudedSurface(BezierDirectrix(((0., .1, 0.), (1., 2., 0.), (2., 0., 0.))), (0., 0., 2.))
    assert not get_poly_plan(support, shifted).coincident


def test_the_support_needs_a_bezier_extrusion_and_a_real_direction():
    with pytest.raises(GeometryError):
        BezierRuledSupport.from_surface(tube())
    with pytest.raises(GeometryError):
        BezierRuledSupport(CUBIC, (0., 0., 0.))
    with pytest.raises(GeometryError):
        BezierRuledSupport(CUBIC[:2], (0., 0., 1.))


def test_polynomial_roots_report_multiplicities_in_a_closed_interval():
    # (t - 1/4)^2 (t - 1/2) (t - 3/2): the root at 3/2 is outside [0, 1]
    poly = [Fraction(-3, 2) * Fraction(1, 16) * -1 * -1]
    poly = [Fraction(1)]
    for root, power in ((Fraction(1, 4), 2), (Fraction(1, 2), 1), (Fraction(3, 2), 1)):
        for _ in range(power):
            poly = [(poly[i - 1] if i else 0) - root * (poly[i] if i < len(poly) else 0) for i in range(len(poly) + 1)]
    found = poly_roots(poly)
    assert [(round(t, 12), m) for t, m in found] == [(.25, 2), (.5, 1)]
    assert [round(t, 12) for t, _m in poly_roots(poly, .3, 2.)] == [.5, 1.5]
    assert poly_roots([Fraction(0), Fraction(0)]) is None
    assert poly_roots([Fraction(3)]) == []


# ---------------------------------------------------------------------------- the Bernstein-basis float system


def random_wall(degree, seed):
    rng = np.random.default_rng(seed)
    ys = np.round(rng.uniform(-1.5, 2.5, degree + 1) * 8) / 8
    return ruled(tuple((float(k), float(y), 0.) for k, y in enumerate(ys)))


@pytest.mark.parametrize("degree", [3, 4, 5, 6, 7, 8])
def test_the_float_polynomials_keep_the_accuracy_of_the_coefficients_whatever_the_degree(degree):
    """In the power basis the discriminant of a degree seven wall is wrong by 1e-10 near ``t = 1``; the Bernstein
    form stays at a few ulp of the coefficients."""
    support = random_wall(degree, degree)
    plan = get_poly_plan(support, SECONDS["a cylinder"])
    fs = plan.floats()
    grid = np.concatenate((np.linspace(0., 1., 61), np.random.default_rng(degree).uniform(.85, 1., 40)))
    for name, exact, form in (("B", plan.B, fs.B_forms[0]), ("C", plan.C, fs.C_forms[0]), ("D", plan.disc, fs.D_forms[0])):
        for t in grid:
            expected = float(horner(exact, Fraction(float(t))))
            assert abs(form(float(t)) - expected) <= 16 * (form.degree + 2) * np.finfo(float).eps * form.bound, (name, t)
        assert np.array_equal(form.many(grid), np.array([form(float(t)) for t in grid]))           # scalar path = vector path
    p, d1, _d2 = fs.frame(grid)
    for k in (0, len(grid) // 2, len(grid) - 1):
        assert np.allclose(fs.origin + p[k], support_point(support, float(grid[k])), atol=1e-14)


def test_the_end_values_are_the_end_coefficients_and_a_constant_is_a_constant():
    plan = get_poly_plan(ruled(), SECONDS["a cylinder"])
    fs = plan.floats()
    for form, exact in ((fs.B_forms[0], plan.B), (fs.C_forms[0], plan.C), (fs.D_forms[0], plan.disc)):
        assert form(0.) == float(horner(exact, Fraction(0))) and form(1.) == float(horner(exact, Fraction(1)))
        assert form.many(np.array([0., 1.]))[0] == form(0.) and form.many(np.array([0., 1.]))[1] == form(1.)
    assert fs.abc_scalar(.3, 1)[0] == 0. and fs.at(np.array([.2, .9]), 2)[0].shape == (2,)


@pytest.mark.parametrize("count", [1, 4, 5, 65])
def test_bernstein_batch_shape_and_scalar_dispatch_have_identical_rounding(count):
    grid = np.random.default_rng(611).uniform(0., 1., (count, 2))[:, :1]
    for degree in (0, 3, 5, 8, 16):
        form = BernsteinForm([Fraction((-1) ** i * (i + 1), 7) for i in range(degree + 1)])
        expected = np.array([form(float(t)) for t in grid.flat]).reshape(grid.shape)
        assert np.array_equal(form.many(grid), expected)


@pytest.mark.parametrize("degree", [3, 5, 7])
def test_a_range_encloses_every_exact_value_and_follows_the_interval(degree):
    support = random_wall(degree, 10 + degree)
    plan = get_poly_plan(support, SECONDS["a cone"])
    fs = plan.floats()
    rng = np.random.default_rng(degree)
    for name, exact, form in (("B", plan.B, fs.B_forms[0]), ("C", plan.C, fs.C_forms[0]), ("D", plan.disc, fs.D_forms[0])):
        for width in (1., .3, 1e-2, 1e-4, 1e-8, 1e-11):
            lo = float(rng.uniform(0., 1. - width))
            hi = lo + width
            values = [float(horner(exact, Fraction(lo + (hi - lo) * k / 8))) for k in range(9)]
            low, high = form.range(lo, hi)
            assert low <= min(values) and max(values) <= high, (name, width)
            # first-order tight: the slack is the Bernstein excess of the piece, at most a few times its variation
            slack = (high - low) - (max(values) - min(values))
            assert slack <= 64 * (degree + 2) * np.finfo(float).eps * form.bound + 4 * width * degree * form.bound, (name, width)
    assert fs.D_forms[0].range(0., 1.)[0] <= min(float(horner(plan.disc, Fraction(k, 64))) for k in range(65))


def test_the_taylor_shift_of_the_discriminant_is_exact_in_rationals():
    from anygeometry.branch_algebra import _exact_taylor
    plan = get_poly_plan(random_wall(6, 3), SECONDS["a cylinder"])
    for anchor in (.0123456789, .5, .987654321):
        x = Fraction(anchor)
        taylor = _exact_taylor(plan.disc, x)
        for delta in (Fraction(3, 1000), Fraction(-1, 7), Fraction(1, 2)):
            assert sum(c * delta ** k for k, c in enumerate(taylor)) == horner(plan.disc, x + delta)       # exactly
