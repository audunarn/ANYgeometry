"""Regression tests for the 7 Codex review findings on the serial candidate (cd29a2d).

Contract: every certainly-no-result shortcut gives the same result, the same refusal (type and message) and the
same cancellation poll count as the unfiltered original route. References: ``_junction_bounds_policy(False)``
(junction prefilters), ``ma._point_clear_of_curve -> None`` (point prefilter) and ``frame(t)[order]`` (the
old frame evaluation, restored by patching ``frame_order``; a frozen copy of the pre-739305b expression).
Tests that fail on the candidate document defects; nothing here is xfail or skipped.
"""
import math

import numpy as np
import pytest

import anygeometry.material_arrangement as ma
from anygeometry import Cylinder, Plane
from anygeometry.analytic_supports import plane_cylinder_support
from anygeometry.exact_curves import CylinderIntersectionCurve
from anygeometry.arrangement_geometry import (LinePath, _junction_bounds_policy, curve_junctions,
                                              point_parameters)
from anygeometry.branch_algebra import BezierRuledSupport
from anygeometry.branch_curves import BezierQuadricCurve
from anygeometry.exact_curves import EllipticArc
from anygeometry.quadric_algebra import QuadricSupport

TOLS = (1e-9, 1e-6)


def make_C(controls=((0., 0., 0.), (.5, 0., 0.), (1., 1., 0.)), direction=(0., 0., 1.)):
    first = BezierRuledSupport(controls, direction)
    second = QuadricSupport("plane", (0., 0., 0.), (0., 0., 1.))
    return BezierQuadricCurve(first, second, 0., 1.)


def scaled(curve, k):
    return curve.transformed(((k, 0., 0., 0.), (0., k, 0., 0.), (0., 0., k, 0.), (0., 0., 0., 1.)))


def outcome(fn):
    try:
        return ("ok", repr(fn()))
    except Exception as error:                       # the refusal type and message are part of the contract
        return ("raise", type(error).__name__, str(error))


def both(first, second, tol, **kw):
    with _junction_bounds_policy(False):
        ref = outcome(lambda: curve_junctions(first, second, tolerance=tol, **kw))
    got = outcome(lambda: curve_junctions(first, second, tolerance=tol, **kw))
    return got, ref


def assert_same_junctions(first, second):
    for tol in TOLS:
        for a, b in ((first, second), (second, first)):
            got, ref = both(a, b, tol)
            assert got == ref, f"tol={tol} first={type(a).__name__}: shortcut {got} != original {ref}"


# 1 ---------------------------------------------------------------- transformed branch, padding too small
def test_f1_line_vs_scaled_branch_curve_same_as_original():
    curve = scaled(make_C(), 10000.)
    for tol in TOLS:
        line = LinePath((0., 0., 5000*tol), (10000., 0., 5000*tol))
        for a, b in ((line, curve), (curve, line)):
            got, ref = both(a, b, tol)
            assert got == ref, f"tol={tol}: shortcut {got} != original {ref}"


def test_f1_sibling_sheared_branch_curve_same_as_original():
    shear = ((1., 3e3, 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.))
    curve = make_C().transformed(shear)
    for tol in TOLS:
        line = LinePath((0., 0., 1500*tol), (3000., 0., 1500*tol))
        for a, b in ((line, curve), (curve, line)):
            got, ref = both(a, b, tol)
            assert got == ref, f"tol={tol}: shortcut {got} != original {ref}"


def test_f1_sibling_scaled_branch_vs_ellipse_same_as_original():
    # _curve_pair_clear: transformed branch curve against an ellipse arc
    curve = scaled(make_C(), 10000.)
    for tol in TOLS:
        arc = EllipticArc((0., 0., 5000*tol), (1., 0., 0.), (0., 1., 0.), 0., 2.)
        assert_same_junctions(arc, curve)


def test_f1_sibling_scaled_cylinder_intersection_curve_same_as_original():
    # the same weakness in the CylinderIntersectionCurve branch of _line_clear_of_branch (NOT reproduced on this input)
    a = Cylinder((0, 0, -2), (0, 0, 1), (1, 0, 0), 1, 4)
    b = Cylinder((-2, 0, 0), (1, 0, 0), (0, 1, 0), 1, 4)
    k = 1e4
    for branch in (1, -1):
        curve = CylinderIntersectionCurve(a, b, 0.3, 1.0, branch).transformed(
            ((k, 0, 0, 0), (0, k, 0, 0), (0, 0, k, 0), (0, 0, 0, 1)))
        p = np.asarray(curve.evaluate(0.0))
        d = p - np.asarray(curve.evaluate(.1))
        n = np.cross(d, [0, 0, 1.])
        n /= np.linalg.norm(n)
        for tol in TOLS:
            for off in (.5, .9):
                q = p + off*k*tol*n
                line = LinePath(tuple(q), tuple(q + d/np.linalg.norm(d)*k))
                for x, y in ((line, curve), (curve, line)):
                    got, ref = both(x, y, tol)
                    assert got == ref, (branch, tol, off, got, ref)


# 2 ---------------------------------------------------------------- huge start phase
def test_f2_elliptic_arc_with_huge_start_phase_same_as_original():
    arc = EllipticArc((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), 1e20, 1.)
    assert_same_junctions(arc, LinePath((1., 0., 0.), (2., 0., 0.)))


def test_f2_sibling_arc_vs_arc_huge_phase_same_as_original():
    arc = EllipticArc((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), 1e20, 1.)
    other = EllipticArc((1., 0., 0.), (1., 0., 0.), (0., 1., 0.), 0., 1.)
    assert_same_junctions(arc, other)


# 3 ---------------------------------------------------------------- suppressed unsupported-predicate refusal
def test_f3_coplanar_disjoint_branch_ellipse_keeps_refusal():
    arc = EllipticArc((50., 0., 0.), (1., 0., 0.), (0., 1., 0.), 0., 2.)
    assert_same_junctions(make_C(), arc)


def test_f3_sibling_scaled_coplanar_pair_keeps_refusal():
    arc = EllipticArc((500., 0., 0.), (1., 0., 0.), (0., 1., 0.), 0., 2.)
    assert_same_junctions(scaled(make_C(), 3.), arc)


# 4 ---------------------------------------------------------------- cancellation poll count
def _polls(a, b, tol, cancel_at, off):
    count = [0]

    def check():
        count[0] += 1
        return cancel_at is not None and count[0] >= cancel_at

    def run():
        return outcome(lambda: curve_junctions(a, b, tolerance=tol, cancellation_check=check))
    if off:
        with _junction_bounds_policy(False):
            result = run()
    else:
        result = run()
    return result, count[0]


def test_f4_rejected_pair_polls_cancellation_like_original():
    a = EllipticArc((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), 0., 2.)
    b = EllipticArc((50., 0., 0.), (1., 0., 0.), (0., 1., 0.), 0., 2.)
    for tol in TOLS:
        for cancel_at in (None, 1, 2, 3, 4):
            assert _polls(a, b, tol, cancel_at, False) == _polls(a, b, tol, cancel_at, True), (tol, cancel_at)


def test_f4_sibling_line_vs_branch_polls_like_original():
    curve = make_C()
    far_line = LinePath((0., 50., 0.), (1., 50., 0.))
    for tol in TOLS:
        for cancel_at in (None, 1, 2, 3, 4):
            assert _polls(curve, far_line, tol, cancel_at, False) == _polls(curve, far_line, tol, cancel_at, True), (tol, cancel_at)


def test_f4_sibling_line_vs_ellipse_polls_like_original():
    arc = EllipticArc((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), 0., 2.)
    far_line = LinePath((100., 100., 0.), (101., 100., 0.))
    for tol in TOLS:
        for cancel_at in (None, 1, 2, 3, 4):
            assert _polls(arc, far_line, tol, cancel_at, False) == _polls(arc, far_line, tol, cancel_at, True), (tol, cancel_at)


def _plate(points_factory):
    # copy of tests/test_arrangement_point_prefilter.py::_plate (tests is not a package)
    plane = Plane((0, 0, 0), (1e-16, 0, 1), (0, 1, 0))
    cylinder = Cylinder((0, 0, -2), (0, 0, 1), (1, 0, 0), 1, 4)
    curves = plane_cylinder_support(plane, cylinder).curves
    corners = [tuple(plane.evaluate(u, v)) for u, v in ((-1.5, -1.5), (1.5, -1.5), (1.5, 1.5), (-1.5, 1.5))]
    loop = tuple(ma.ArrangementPath(LinePath(corners[k], corners[(k + 1) % 4])) for k in range(4))
    return ma.MaterialDomain(7, plane, (loop,)), [ma.ArrangementPath(c) for c in curves], points_factory(plane, curves)


# 5 ---------------------------------------------------------------- finite far point that raises FloatingPointError
def test_f5_far_finite_point_filter_implies_unfiltered_returns_empty_without_raising():
    curve = make_C()
    far = ma._point_clear_of_curve(curve, 1e-9)
    assert far is not None
    assert not far((0.5, 0.25, 0.))
    point = (1e308, 1e308, 0.)
    assert far(point)

    def unfiltered():
        with np.errstate(over="raise"):
            return point_parameters(curve, point, tolerance=1e-9)
    # contract of _point_clear_of_curve: far(p) implies the unfiltered route yields () and does not refuse
    assert outcome(unfiltered) == ("ok", repr(()))


def test_f5_split_loop_outcome_same_with_and_without_filter(monkeypatch):
    def points(plane, curves):
        return [ma.ArrangementPoint(tuple(curves[0].evaluate(.4))), ma.ArrangementPoint((1e308, 1e308, 0.))]
    domain, traces, pts = _plate(points)

    def run():
        with np.errstate(over="raise"):
            return outcome(lambda: ma.arrange_material(domain, traces, points=pts, tolerance=1e-9))
    filtered = run()
    monkeypatch.setattr(ma, "_point_clear_of_curve", lambda curve, tol: None)
    assert filtered == run()


# 6 ---------------------------------------------------------------- mixed non-finite point
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("slot", [0, 1, 2])
def test_f6_mixed_nonfinite_point_far_implies_unfiltered_empty(bad, slot):
    curve = make_C()
    far = ma._point_clear_of_curve(curve, 1e-9)
    point = [100., 100., 0.]
    point[slot] = bad
    # contract: far(p) implies the unfiltered route returns () (strengthened to "far is False" if it does not)
    assert (not far(tuple(point))) or outcome(
        lambda: point_parameters(curve, tuple(point), tolerance=1e-9)) == ("ok", repr(()))


# 7 ---------------------------------------------------------------- frame order overflow refusal
def _old_frame(monkeypatch, curve):
    cls = type(curve.plan().floats())
    monkeypatch.setattr(cls, "frame_order", lambda self, t, order: self.frame(t)[order])


def _finding7_curve():
    X = 2.**1021
    return make_C(((0., 0., 0.), (-X, 1., 0.), (-X, 2., 0.), (0., 3., 0.)))


@pytest.mark.parametrize("what", ["evaluate", "derivative"])
def test_f7_frame_order_overflow_refusal_same_as_original(monkeypatch, what):
    curve = _finding7_curve()
    t = [.5]*5

    def run():
        with np.errstate(over="raise"):
            return outcome(lambda: getattr(curve, what)(t))
    new = run()
    _old_frame(monkeypatch, curve)
    assert new == run()
