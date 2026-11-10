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
    is_far = far(point)

    def unfiltered():
        with np.errstate(over="raise"):
            return point_parameters(curve, point, tolerance=1e-9)
    # contract of _point_clear_of_curve: far(p) implies the unfiltered route yields () and does not refuse
    assert (not is_far) or outcome(unfiltered) == ("ok", repr(()))


def test_f5_prefilter_still_filters_moderate_far_points_and_declines_huge_ones():
    curve = make_C()
    far = ma._point_clear_of_curve(curve, 1e-9)
    assert far((1e308, 1e308, 0.)) is False
    moderate = (1e6, 1e6, 0.)
    assert far(moderate) is True

    def unfiltered():
        with np.errstate(over="raise"):
            return point_parameters(curve, moderate, tolerance=1e-9)
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


# ---------------------------------------------------------------- round 2: results and refusals (contract of the owner)
# Contract: with the shortcuts on, the call returns the same values and raises the same typed refusals as the
# original route for all inputs; otherwise the shortcut declines. Cancellation poll counts and exceptions under
# np.errstate(...='raise') are not part of the contract (production never sets errstate raise).
def both_policy_on(first, second, tol):
    """Shortcuts on including the default-off ``_separated_line_pair`` route, against the unfiltered original."""
    with _junction_bounds_policy(False):
        ref = outcome(lambda: curve_junctions(first, second, tolerance=tol))
    with _junction_bounds_policy(True):
        got = outcome(lambda: curve_junctions(first, second, tolerance=tol))
    return got, ref


def test_r2a_bounds_route_huge_phase_arc_vs_line_same_as_original():
    arc = EllipticArc((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), 1e20, 1.)
    line = LinePath((1., 0., 0.), (2., 0., 0.))
    for tol in TOLS:
        for a, b in ((arc, line), (line, arc)):
            got, ref = both_policy_on(a, b, tol)
            assert got == ref, (tol, got, ref)


def _degenerate_directrix_branch():
    return make_C(((50., 0., 0.), (50.5, 0., 0.), (51., 0., 0.)))


def _r2_partners():
    X, Y = (1., 0., 0.), (0., 1., 0.)
    a = Cylinder((0, 0, -2), (0, 0, 1), (1, 0, 0), 1, 4)
    b = Cylinder((-2, 0, 0), (1, 0, 0), (0, 1, 0), 1, 4)
    far_a = Cylinder((100, 0, -2), (0, 0, 1), (1, 0, 0), 1, 4)
    far_b = Cylinder((98, 0, 0), (1, 0, 0), (0, 1, 0), 1, 4)
    return {
        "line": LinePath((0., 0., 5.), (1., 0., 5.)),
        "far_line": LinePath((200., 200., 5.), (201., 200., 5.)),
        "noncoplanar_arc": EllipticArc((50., 0., 3.), X, Y, 0., 2.),
        "branch": make_C(),
        "branch_other_plane": make_C(((0., 0., 9.), (.5, 0., 9.), (1., 1., 9.))),
        "cylinder_curve": CylinderIntersectionCurve(a, b, 0.3, 1.0, 1),
        "far_cylinder_curve": CylinderIntersectionCurve(far_a, far_b, 0.3, 1.0, 1),
    }


@pytest.mark.parametrize("name", sorted(_r2_partners()))
@pytest.mark.parametrize("policy", ["default", "bounds"])
def test_r2b_degenerate_directrix_branch_keeps_its_refusal(name, policy):
    # the original route inverts the branch curve's endpoints and refuses a directrix that degenerates to a line;
    # a shortcut must not turn that refusal into an empty result (any partner kind, both orders)
    bad, partner = _degenerate_directrix_branch(), _r2_partners()[name]
    for tol in TOLS:
        for a, b in ((bad, partner), (partner, bad)):
            if policy == "bounds":
                got, ref = both_policy_on(a, b, tol)
            else:
                got, ref = both(a, b, tol)
            assert got == ref, (name, policy, tol, got, ref)


def test_r2d_bounds_route_scaled_cylinder_curve_vs_line_same_as_original():
    a = Cylinder((0, 0, -2), (0, 0, 1), (1, 0, 0), 1, 4)
    b = Cylinder((-2, 0, 0), (1, 0, 0), (0, 1, 0), 1, 4)
    k = 1e4
    curve = CylinderIntersectionCurve(a, b, 0.3, 1.0, 1).transformed(
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
                got, ref = both_policy_on(x, y, tol)
                assert got == ref, (tol, off, got, ref)


# ---------------------------------------------------------------- round 3: huge-parameter class
def test_r3_cylinder_curve_with_huge_start_phase_vs_line_same_as_original():
    a = Cylinder((0, 0, 0), (0, 0, 1), (-0.020691120661983364, 0.9997859158468633, 0), 1., 4.)
    b = Cylinder((0, 0, 0), (0, 1, 0), (1, 0, 0), 2., 4.)
    curve = CylinderIntersectionCurve(a, b, 870099176652729.5, 0.7942977928370237, 1)
    p = np.asarray(curve.evaluate(.75))
    line = LinePath(tuple(p), tuple(p + (0., 0., 1.)))
    for a_, b_ in ((line, curve), (curve, line)):
        for policy in ("default", "bounds"):
            got, ref = (both(a_, b_, 1e-9) if policy == "default" else both_policy_on(a_, b_, 1e-9))
            assert got == ref, (policy, got, ref)


def _random_curves(rng):
    """Curves of all four kinds with ordinary and huge or odd parameters (angles, radii, axes, control nets)."""
    def phase():
        return float(rng.choice([0., 0.3, 3.0, 40., 1e3, 1e5, 1e10, 1e15, 8.7e14, 1e20, -2.5e16]) + rng.normal()*rng.choice([0, 1e-3, 1]))

    def vec(scale=1.):
        return tuple(float(x) for x in rng.normal(size=3)*scale)
    curves = []
    for _ in range(6):
        u, v = vec(rng.choice([.1, 1., 30.])), vec(rng.choice([.1, 1., 30.]))
        try:
            curves.append(EllipticArc(vec(rng.choice([0., 5., 100.])), u, v, phase(), float(rng.uniform(-6.2, 6.2))))
        except Exception:
            pass
    for _ in range(6):
        try:
            axis_a = rng.normal(size=3); axis_b = rng.normal(size=3)
            cyl_a = Cylinder(vec(2.), tuple(axis_a/np.linalg.norm(axis_a)), tuple(np.cross(axis_a, [0., 0., 1.])/np.linalg.norm(np.cross(axis_a, [0., 0., 1.]))),
                             float(rng.uniform(.3, 3.)), 4.)
            cyl_b = Cylinder(vec(2.), tuple(axis_b/np.linalg.norm(axis_b)), tuple(np.cross(axis_b, [0., 0., 1.])/np.linalg.norm(np.cross(axis_b, [0., 0., 1.]))),
                             float(rng.uniform(.3, 3.)), 4.)
            curves.append(CylinderIntersectionCurve(cyl_a, cyl_b, phase(), float(rng.uniform(-3., 3.)), int(rng.choice([-1, 1]))))
        except Exception:
            pass
    for _ in range(6):
        try:
            scale = float(rng.choice([.5, 1., 1e3, 1e8]))
            n = int(rng.integers(3, 5))
            controls = tuple((float(x)*scale, float(y)*scale, 0.) for x, y in np.cumsum(rng.normal(size=(n, 2)), axis=0))
            curves.append(make_C(controls, tuple(float(x) for x in rng.normal(size=3))))
        except Exception:
            pass
    return curves


def test_r3_randomized_differential_shortcuts_vs_original():
    rng = np.random.default_rng(20261110)
    curves = _random_curves(rng)
    cases = []
    for curve in curves:
        for _ in range(5):
            try:
                p = np.asarray(curve.evaluate(float(rng.uniform(0., 1.))))
            except Exception:
                continue
            d = rng.normal(size=3)
            d *= rng.choice([.5, 2.]) / np.linalg.norm(d)
            shift = rng.normal(size=3)*rng.choice([0., 0., 1e-8, 1e-3])      # through the curve, or just beside it
            cases.append((LinePath(tuple(p + shift - d), tuple(p + shift + d)), curve))
    for i, a in enumerate(curves):
        for b in curves[i+1:i+3]:
            cases.append((a, b))
    assert len(cases) >= 60
    bad = []
    for a, b in cases:
        for x, y in ((a, b), (b, a)):
            for policy in ("default", "bounds"):
                got, ref = (both(x, y, 1e-9) if policy == "default" else both_policy_on(x, y, 1e-9))
                if got != ref:
                    bad.append((type(x).__name__, type(y).__name__, policy, got, ref))
    assert not bad, f"{len(bad)} mismatches, first: {bad[0]}"
