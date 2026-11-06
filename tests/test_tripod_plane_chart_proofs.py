"""Focused regressions for the three proof blockers: nonlinear fold distance, certified point membership, exact root identity."""
from fractions import Fraction
import math

import numpy as np
import pytest

from anygeometry.plane_ruled import PlaneRuledSupportError
from anygeometry.plane_ruled_support import (
    certify_point_membership, events_identical, fold_distance_bound, patch_margins, plane_ruled_cylinder_support,
    sturm_count,
)
from anygeometry.surfaces import Cylinder, Plane


def poly(*roots, scale=1):
    """Monic ascending coefficients of prod (t - root) with Fraction roots."""
    result = [Fraction(scale)]
    for root in roots:
        shifted = [Fraction(0)] + result
        result = [shifted[i] - root * (result[i] if i < len(result) else 0) for i in range(len(shifted))]
    return result


# -- blocker 3: exact root identity -----------------------------------------------------------------------------

def test_sturm_count_counts_distinct_roots_in_a_half_open_interval():
    p = poly(Fraction(1, 3), Fraction(1, 3), Fraction(2, 3))                     # double root is ONE distinct root
    assert sturm_count(p, 0, 1) == 2
    assert sturm_count(p, Fraction(1, 2), 1) == 1
    assert sturm_count(p, 2, 3) == 0


def test_same_root_of_both_polynomials_is_identified_exactly():
    a = Fraction(1, 3)
    d = poly(a, Fraction(7, 10))
    r = poly(a, Fraction(9, 10))
    t = float(a)
    assert events_identical({None: d, 0: r}, [(t, None), (t, 0), (math.nextafter(t, 1.0), 0)])


def test_distinct_nearby_discriminant_root_is_not_exempted():
    """D = (t-a)(t-b), R = (t-a)(t-c): the common root a does NOT identify the nearby distinct roots b or c."""
    a = Fraction(1, 3)
    b = a + Fraction(1, 10 ** 14)                          # both inside one pad-sized interval, all gaps < 1e-12
    c = a + Fraction(2, 10 ** 14)
    d, r = poly(a, b), poly(a, c)
    assert not events_identical({None: d, 0: r}, [(float(a), None), (float(b), None), (float(c), 0)])
    assert not events_identical({None: d, 0: r}, [(float(a), None), (float(c), 0)])    # R has two roots in I
    assert not events_identical({None: d, 0: r}, [(float(b), None), (float(a), 0)])    # D has two roots in I


def test_equal_float_witnesses_of_distinct_exact_roots_are_not_identity():
    """Two distinct exact roots a < b closer than one ulp round to the SAME double: equal witnesses prove nothing."""
    a = Fraction(1, 3)
    b = a + Fraction(1, 2 ** 80)                           # far below one ulp (2^-54 near 1/3)
    assert float(a) == float(b)
    d, r = poly(a, b), poly(a, Fraction(9, 10))
    t = float(a)
    assert not events_identical({None: d, 0: r}, [(t, None), (t, 0)])      # D has two roots in I
    assert not events_identical({0: poly(a, b)}, [(t, 0), (t, 0)])         # one polynomial, two roots, one witness
    assert events_identical({0: poly(a, Fraction(9, 10))}, [(t, 0), (t, 0)])   # a single root: identical


def test_equal_float_witnesses_get_an_outward_bracket_containing_every_root():
    """After identity is False the cluster span is the verified outward bracket, never the zero-width float span."""
    from anygeometry.plane_ruled_support import group_interval
    a = Fraction(1, 3)
    b = a + Fraction(1, 2 ** 80)
    d = poly(a, b)
    t = float(a)
    lo, hi = group_interval({None: d}, [(t, None), (t, None)])
    assert Fraction(lo) <= a and Fraction(hi) >= b and hi > lo                    # width > 0 although the floats are equal
    assert group_interval({None: d}, [(0.9, None)]) is None                        # no root of d near the witness
    assert group_interval({None: d}, [(t, 5)]) is None                             # unknown tag: unresolved


def test_certified_roundings_terminate_and_never_return_zero_for_positive_input():
    from anygeometry.plane_ruled_support import _div_up_certified, _sqrt_down_certified, _sqrt_up_certified
    tiny = Fraction(1, 10 ** 320)                                                  # genuine subnormal-range rational
    assert _sqrt_up_certified(tiny) == math.inf                                    # underflow is refused, no long loop
    assert _sqrt_down_certified(tiny) == 0.0
    assert _div_up_certified(tiny, 1.0) > 0.0 and Fraction(_div_up_certified(tiny, 1.0)) >= tiny
    below_all = Fraction(1, 10 ** 400)                                             # below the smallest subnormal
    assert _div_up_certified(below_all, 1.0) == 5e-324
    x = Fraction(2, 3)
    assert Fraction(_sqrt_up_certified(x)) ** 2 >= x and Fraction(_sqrt_down_certified(x)) ** 2 <= x
    assert _sqrt_up_certified(Fraction(0)) == 0.0


def test_full_sweep_end_seam_plane_is_distinct_from_the_start_plane():
    """The documented cause of the false fold/seam identity: sin(2 pi) makes the stored end plane differ from y = 0."""
    from anygeometry.quadric_supports import _boundary_planes
    planes = _boundary_planes(unit_cylinder())
    assert planes[2].axis != planes[3].axis


def test_doubly_rounded_witness_that_is_not_a_root_is_rejected():
    a = Fraction(1, 3)
    d, r = poly(a, Fraction(7, 10)), poly(a, Fraction(9, 10))
    off = float(a) + 1e-9                                  # a double that is no root of R within 4 eps
    assert not events_identical({None: d, 0: r}, [(float(a), None), (off, 0)])
    assert not events_identical({None: d}, [(float(a), 7)])                           # unknown tag


# -- blocker 2: certified point membership --------------------------------------------------------------------

def unit_cylinder(height=2.0, **kwargs):
    return Cylinder((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), 1.0, height, **kwargs)


def test_membership_is_certified_on_acceptance_and_rejection_sides():
    tol = 1e-10
    cylinder = unit_cylinder()
    inside = (1.0, 0.0, -0.5 * tol)
    assert certify_point_membership(cylinder, inside, 0.1 * tol, tol) is True
    outside = (1.0, 0.0, -3.0 * tol)
    assert certify_point_membership(cylinder, outside, 0.1 * tol, tol) is False


def test_edge_uncertainty_is_refused_not_emitted():
    """margin -0.9 tol with error 0.2 tol: neither accepted (needs margin - error >= -tol) nor rejected."""
    tol = 1e-10
    cylinder = unit_cylinder()
    with pytest.raises(PlaneRuledSupportError) as caught:
        certify_point_membership(cylinder, (1.0, 0.0, -0.9 * tol), 0.2 * tol, tol, face_pair=(1, 2))
    assert caught.value.reason == "point_ambiguous" and caught.value.details["face_pair"] == (1, 2)
    with pytest.raises(PlaneRuledSupportError) as caught:
        certify_point_membership(cylinder, (1.0, 0.0, 0.5), 0.5 * tol, tol)           # error beyond a quarter tolerance
    assert caught.value.reason == "point_unproved"


def test_angular_edge_uses_the_arc_error_not_the_euclidean_one():
    tol = 1e-10
    cylinder = unit_cylinder(start_angle=0.0, sweep_angle=0.5 * math.pi)
    angle = -0.9 * tol                                       # just outside the seam by 0.9 tol of arc
    point = (math.cos(angle), math.sin(angle), 1.0)
    with pytest.raises(PlaneRuledSupportError):
        certify_point_membership(cylinder, point, 0.2 * tol, tol)


# -- blocker 1: nonlinear fold distance -------------------------------------------------------------------------

def test_fold_distance_bound_dominates_the_exact_radial_shift_and_stays_small():
    """The exact distance of a point at Q to the cylinder along the radial line is Q/(r+sqrt(r^2+Q)) for |rho|^2 = r^2+Q:
    the bound must never be below it, and for an ordinary fold it must stay ~E/(2 r sin(phi))."""
    eps, zstar = 1e-3, 2.0
    plane = Plane((1.0 + eps * zstar, 0.0, 0.0), (0.0, 1.0, 0.0), (-eps, 0.0, 1.0))
    cylinder = unit_cylinder(height=4.0)          # the fold (z* = 2) must be INTERIOR: with height 2 it sits on the cap
    result = plane_ruled_cylinder_support(plane, cylinder, tolerance=1e-9)
    curve = next(c for c in result.curves if c.parameterization != "linear")
    plan = curve.plan()
    anchors = (curve.start,) if curve.parameterization == "left_square" else (curve.start + curve.sweep,) \
        if curve.parameterization == "right_square" else (curve.start, curve.start + curve.sweep)
    t_interval = tuple(sorted((curve.start, curve.start + curve.sweep)))
    bound = fold_distance_bound(plan, curve.first, cylinder, anchors, t_interval)
    assert math.isfinite(bound) and bound >= 0.0
    # independent Q offset at the anchor: -D(anchor)/(4A) evaluated exactly
    big_a = Fraction(plan.A[0])
    e_q = max(abs(sum(c * Fraction(float(x)) ** i for i, c in enumerate(plan.disc))) / (4 * big_a) for x in anchors)
    e = float(e_q)
    inward = e / (1.0 + math.sqrt(1.0 - e))                    # Q = -E: radial distance r - sqrt(r^2 - E), the larger
    outward = e / (1.0 + math.sqrt(1.0 + e))                   # Q = +E
    assert bound >= inward * (1.0 - 1e-9) and bound >= outward * (1.0 - 1e-9)
    if e > 0.0:
        assert bound >= e                                      # certified T = 2E/g with g <= 2 r = 2
        # Near a tangent the inverse conditioning is 1/|eps|, not a
        # geometry-independent factor 100. On this exact unit-cylinder
        # fixture g^2 >= 4(1-E) eps^2/(1+eps^2), and T=2E/g.
        analytic_upper = e * math.sqrt(1.0 + eps * eps) / (abs(eps) * math.sqrt(1.0 - e))
        assert bound <= analytic_upper * (1.0 + 1e-9)
    else:
        assert bound == 0.0


def test_both_sine_bound_restricts_h_to_the_range_the_evaluator_actually_uses():
    """both_sine evaluates h only for u = f^2 in [0, 1/2] (left anchor tau <= 1/2, right anchor tau > 1/2), so the derived
    bound of the forced closed ellipse must be small (it was 1e-7 when h was ranged over [0, 1])."""
    from anygeometry.plane_ruled_support import curve_error_bound
    plane = Plane((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 1.0))
    cylinder = unit_cylinder(height=3.0)
    result = plane_ruled_cylinder_support(plane, cylinder, tolerance=1e-9)
    both = [c for c in result.curves if c.parameterization == "both_sine"]
    assert both, "the closed ellipse has a chart between its two folds"
    for curve in both:
        bound, details = curve_error_bound(curve.plan(), curve.first, curve, cylinder)
        assert bound <= 0.25e-9, details
        # the sine charts reach u = f^2 = 1/2 at tau = 1/2 and never beyond
        taus = np.linspace(0.0, 1.0, 101)
        _t, _dt, _anchor, _delta, f, _fp, _side = curve._chart(taus)
        assert float(np.max(f * f)) <= 0.5 + 1e-12


def test_certified_underflow_never_returns_zero_for_a_positive_offset():
    """E = 1e-320 (subnormal): float(2E) style shortcuts could underflow; the exact post-check keeps T > 0."""
    eps = 1e-3
    plane = Plane((1.0 + 2 * eps, 0.0, 0.0), (0.0, 1.0, 0.0), (-eps, 0.0, 1.0))
    cylinder = unit_cylinder(height=4.0)
    from anygeometry.branch_algebra import get_poly_plan
    from anygeometry.plane_ruled import build_plane_ruled_support
    support, _lo, _hi = build_plane_ruled_support(plane, cylinder, tolerance=1e-9)
    plan = get_poly_plan(support, cylinder)
    # the anchor nearest the true fold: D(anchor) is tiny but generally non-zero
    from anygeometry.plane_ruled_support import _in_range
    roots = _in_range([(t, m) for t, m in plan.discriminant_roots() if m == 1], 0.0, 1.0)
    assert roots
    bound = fold_distance_bound(plan, support, cylinder, (roots[0],), (max(0.0, roots[0] - 1e-3), min(1.0, roots[0] + 1e-3)))
    value = abs(sum(c * Fraction(float(roots[0])) ** i for i, c in enumerate(plan.disc)))
    assert (bound > 0.0) == (value != 0)


def test_fold_distance_bound_is_inf_when_the_gradient_lower_bound_fails():
    """An anchor far from the fold has a large Q offset E; once g^2 <= 4 E (K = 1) the bound must be inf."""
    eps = 1e-3
    plane = Plane((1.0 + 2 * eps, 0.0, 0.0), (0.0, 1.0, 0.0), (-eps, 0.0, 1.0))
    cylinder = unit_cylinder()
    from anygeometry.branch_algebra import get_poly_plan
    from anygeometry.plane_ruled import build_plane_ruled_support
    support, _lo, _hi = build_plane_ruled_support(plane, cylinder, tolerance=1e-9)
    plan = get_poly_plan(support, cylinder)
    assert fold_distance_bound(plan, support, cylinder, (0.0,), (0.0, 1.0)) == math.inf     # chart end: D(0) is O(1)
