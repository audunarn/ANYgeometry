"""Curve-level predicates on plane x NURBS section curves (generality slice B1-P4a, architecture 4.4).

References: the analytic section of a cylinder by an oblique plane (an ellipse whose
plan view is the circle x^2 + y^2 = 4), so roots of a second plane are known in closed
form; the B1 bow surface for a doubly curved case (independent float bisection).
"""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import pytest

from anygeometry import Cylinder, Plane
from anygeometry.branch_predicates import certify_line_section, section_roots
from anygeometry.nurbs import NurbsSurface, cylinder_to_nurbs
from anygeometry.support_curves import PlaneNurbsCurve

OBLIQUE = Plane((0., 0., 1.5), (1., 0., .3), (0., 1., .2))          # z = 1.5 + .3 x + .2 y
TOL = 1e-9


def z(x, y):
    return 1.5 + .3 * x + .2 * y


def point(x, y):
    return (x, y, z(x, y))


@pytest.fixture(scope="module")
def half():
    surface = cylinder_to_nurbs(Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 2., 3., 0., math.pi))
    return PlaneNurbsCurve.from_plane(OBLIQUE, surface, 0)              # (2, 0) -> (0, 2) -> (-2, 0)


def contains(root, expected, slack=1e-12):
    return all(lo - slack <= value <= hi + slack for lo, hi, value in zip(root.lower, root.upper, expected))


def test_one_root_encloses_the_analytic_crossing(half):
    roots = section_roots(half, (1, 0, 0), 1)                          # the plane x = 1
    assert len(roots) == 1 and roots[0].isolated
    assert contains(roots[0], point(1., math.sqrt(3.)))
    assert max(hi - lo for lo, hi in zip(roots[0].lower, roots[0].upper)) <= 1e-9
    middle = .5 * (roots[0].t_lo + roots[0].t_hi)
    assert abs(half.evaluate(middle)[0] - 1.) <= 1e-9


def test_two_roots_and_no_roots(half):
    roots = section_roots(half, (0, 1, 0), 1)                          # y = 1: x = +-sqrt(3)
    assert [root.isolated for root in roots] == [True, True]
    assert contains(roots[0], point(math.sqrt(3.), 1.)) and contains(roots[1], point(-math.sqrt(3.), 1.))
    assert roots[0].t_hi < roots[1].t_lo
    assert section_roots(half, (1, 0, 0), 5) == ()                      # x = 5 misses the curve


def test_roots_at_the_curve_ends_stay_at_the_ends(half):
    roots = section_roots(half, (0, 1, 0), 0)                          # y = 0 through both ends
    ends = (half.evaluate(0.), half.evaluate(1.))
    assert roots
    for root in roots:
        assert any(contains(root, end, 1e-9) for end in ends)


def test_sub_curves_report_only_their_own_roots(half):
    first = half.subcurve(0., .5)
    roots = section_roots(first, (0, 1, 0), 1)
    assert len(roots) == 1 and contains(roots[0], point(math.sqrt(3.), 1.))
    assert all(0. <= value <= 1. for value in (roots[0].t_lo, roots[0].t_hi))


def bow_surface():
    """The B1-pre half-bow of ``test_nurbs_bow_audit`` (rational quadratic plan, flared, depth 4)."""
    w = math.cos(math.pi / 4)
    bottom = [(0., 6., 0.), (5.4, 6., 0.), (5.4, 0., 0.)]
    deck = [(0., 7.5, 4.), (6., 7.5, 4.), (6., 0., 4.)]
    return NurbsSurface(2, 1, (0., 0., 0., 1., 1., 1.), (0., 0., 1., 1.),
                        tuple((b, d) for b, d in zip(bottom, deck)), ((1., 1.), (w, w), (1., 1.)))


def test_a_frame_plane_on_the_bow():
    surface = bow_surface()
    deck_edge = PlaneNurbsCurve.from_plane(Plane((0., 0., 3.), (1., 0., 0.), (0., 1., 0.)), surface, 0)
    roots = section_roots(deck_edge, (1, 0, 0), 3)                     # the frame plane x = 3
    assert len(roots) == 1 and roots[0].isolated
    # independent float check: bisection of x(t) - 3 on the evaluated curve
    lo, hi = 0., 1.
    f_lo = deck_edge.evaluate(lo)[0] - 3.
    for _ in range(60):
        mid = .5 * (lo + hi)
        f_mid = deck_edge.evaluate(mid)[0] - 3.
        lo, hi, f_lo = (mid, hi, f_mid) if (f_mid > 0) == (f_lo > 0) else (lo, mid, f_lo)
    reference = deck_edge.evaluate(.5 * (lo + hi))
    assert contains(roots[0], reference, 1e-9)


def test_line_certificates(half):
    start, end = half.evaluate(0.), half.evaluate(1.)
    # the diameter shares both curve ends and meets the curve only there
    assert certify_line_section(half, start, end, (start, end), TOL) is True
    # a chord inside the half disc that does not reach the curve
    assert certify_line_section(half, point(-1., .5), point(1., .5), (), TOL) is True
    # a line leaving the shared end outwards
    assert certify_line_section(half, start, point(3., 0.), (start,), TOL) is True
    # a line crossing the curve at (0, 2)
    assert certify_line_section(half, point(0., -1.), point(0., 3.), (), TOL) is False
    # a crossing that is not at the shared end
    assert certify_line_section(half, start, point(-1., 3.), (start,), TOL) is False


def test_line_certificates_never_accept_unproven_contacts(half):
    start, end = half.evaluate(0.), half.evaluate(1.)
    # (near-)tangent line y = 2: touching or two nearby crossings, never "only shared ends"
    assert certify_line_section(half, point(-3., 2.), point(3., 2.), (), TOL) is not True
    # the diameter without declaring the shared ends: it meets the curve at its own end points
    assert certify_line_section(half, start, end, (), TOL) is not True
    # a segment ending on the curve's interior without a shared point
    on_curve = half.evaluate(.25)
    assert certify_line_section(half, point(0., 0.), on_curve, (), TOL) is not True


# ------------------------------------------------------------------ review B1-P4a round 1


Z0 = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))


def bezier(degree_u, rows):
    knots_u = (0.,) * (degree_u + 1) + (1.,) * (degree_u + 1)
    return NurbsSurface(degree_u, 1, knots_u, (0., 0., 1., 1.), rows)


def test_clipped_end_regions_keep_their_roots():
    # S(u, v) = (u, v, v + u^2 - u + 1/32): the section z = 0 starts at the root (u*, 0) on the edge
    # v = 0, u* = (1 - sqrt(7/8)) / 2 irrational; the arc limit s0 is its interval's midpoint. On a
    # sub-curve inside [s0, u*] the mathematical curve sits at the root, which lies on y = 0 exactly:
    # the clipped piece must still contain it (review B1-P4a round 1: it was excluded).
    rows = tuple(((i / 2, 0., c), (i / 2, 1., c + 1.)) for i, c in enumerate((1 / 32, 1 / 32 - .5, 1 / 32)))
    curve = PlaneNurbsCurve.from_plane(Z0, bezier(2, rows), 0)
    arc = curve._arcs()[0]
    sub = curve.subcurve(2. ** -80, 2. ** -79)
    s_low = arc.s0 + Fraction(sub.domain[0]) * (arc.s1 - arc.s0)
    before_root = (1 - 2 * s_low) ** 2 > Fraction(7, 8)                 # s_low < u*, exactly
    roots = section_roots(sub, (0, 1, 0), 0)
    if before_root:
        assert roots and all(contains(root, sub.evaluate(.5), 1e-12) for root in roots)
    else:
        assert roots == ()


def test_roots_are_roots_of_the_mathematical_curve():
    # S(u, v) = (u, v, 3 v - u - 1/4): on z = 0, y = 1/3 at x = 3/4 (no defined point has y = 1/3 exactly)
    rows = (((0., 0., -.25), (0., 1., 2.75)), ((1., 0., -1.25), (1., 1., 1.75)))
    curve = PlaneNurbsCurve.from_plane(Z0, bezier(1, rows), 0)
    roots = section_roots(curve, (0, 3, 0), 1)
    assert len(roots) == 1 and roots[0].isolated and contains(roots[0], (.75, 1 / 3, 0.))


def test_the_curve_own_plane_is_refused():
    from anygeometry.plane_nurbs import PlaneNurbsNotCertified
    rows = (((0., 0., -.25), (0., 1., 2.75)), ((1., 0., -1.25), (1., 1., 1.75)))
    curve = PlaneNurbsCurve.from_plane(Z0, bezier(1, rows), 0)
    with pytest.raises(PlaneNurbsNotCertified, match="coincident"):
        section_roots(curve, (0, 0, 2), 0)


@pytest.fixture(scope="module")
def level():
    surface = cylinder_to_nurbs(Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 2., 3., 0., math.pi))
    return PlaneNurbsCurve.from_plane(Plane((0., 0., 1.5), (1., 0., 0.), (0., 1., 0.)), surface, 0)


def test_crossings_inside_the_joint_tolerance_are_not_offending(level):
    start = (2., 0., 1.5)
    assert certify_line_section(level, start, (1.99, .4, 1.5), (start,), .11) is not False


def test_segments_off_the_curve_plane_are_undecided(level):
    assert certify_line_section(level, (1., -10., 2.5), (1., 10., 2.5), (), TOL) is None
