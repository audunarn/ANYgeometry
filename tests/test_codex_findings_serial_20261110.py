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


# ---------------------------------------------------------------- round 4: near-parallel cylinders (conditioning)
def test_r4_near_parallel_cylinders_cylinder_curve_vs_line_same_as_original():
    a = Cylinder((0, 0, 0), (1, 1, 0), (0, 0, 1), 1., 4.)
    b = Cylinder((0, 0, 0), (1, 1+1e-8, 0), (0, 0, 1), 2., 4.)
    curve = CylinderIntersectionCurve(a, b, .3, 1e-12, 1)
    p = np.asarray(curve.evaluate(0.))
    line = LinePath(tuple(p), tuple(p + (0., 0., 1.)))
    for x, y in ((line, curve), (curve, line)):
        for policy in ("default", "bounds"):
            got, ref = (both(x, y, 1e-9) if policy == "default" else both_policy_on(x, y, 1e-9))
            assert got == ref, (policy, got, ref)


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _frame_for(axis, rng):
    ref = np.eye(3)[int(np.argmin(np.abs(axis)))]
    radial = _unit(np.cross(axis, ref))
    angle = rng.uniform(0., 6.3)
    return _unit(np.cos(angle)*radial + np.sin(angle)*np.cross(axis, radial))


def _conditioned_cases(seed, count):
    """Correlated near-parallel cylinder pairs (angles 1e-12 .. 1e-1, log scale), extreme radius ratios, log-scale
    sweeps and origins, axes near coordinate extrema, endpoint and interior witness points, lines through the
    witness along box faces / coordinate axes / random directions, and some non-identity transforms."""
    rng = np.random.default_rng(seed)
    stats = {"built": 0, "tried": 0, "nonempty": 0}
    cases = []
    for _ in range(count):
        stats["tried"] += 1
        if rng.random() < .5:
            axis = _unit(np.eye(3)[int(rng.integers(3))] + rng.normal(size=3)*10**rng.uniform(-12, -2))
        else:
            axis = _unit(rng.normal(size=3))
        theta = 10**rng.uniform(-12, -1)
        pivot = _unit(np.cross(axis, rng.normal(size=3)))
        other = _unit(axis*np.cos(theta) + np.cross(pivot, axis)*np.sin(theta))
        r1 = 10**rng.uniform(-1, 1)
        r2 = r1*10**rng.uniform(-2, 2)
        o1 = rng.normal(size=3)*10**rng.uniform(-1, 3)
        o2 = o1 + rng.normal(size=3)*10**rng.uniform(-2, 1)*(r1 + r2)
        try:
            c1 = Cylinder(tuple(o1), tuple(axis), tuple(_frame_for(axis, rng)), float(r1), 4.)
            c2 = Cylinder(tuple(o2), tuple(other), tuple(_frame_for(other, rng)), float(r2), 4.)
            start = float(rng.choice([0., .3, 1., 3., 5.9, rng.uniform(0., 6.)]))
            sweep = float(rng.choice([1., 3., 6.2, 10**rng.uniform(-12, 0.7)]))*float(rng.choice([1., -1.]))
            curve = CylinderIntersectionCurve(c1, c2, start, sweep, int(rng.choice([-1, 1])))
            curve.evaluate(0.)
            curve.evaluate(1.)
        except Exception:
            continue
        stats["built"] += 1
        if rng.random() < .15:
            k = 10**rng.uniform(-2, 4)
            shear = float(rng.normal())*k
            try:
                curve = curve.transformed(((k, shear, 0., 1.), (0., k, 0., 0.), (0., 0., k, 0.), (0., 0., 0., 1.)))
            except Exception:
                continue
        for u in (0., 1., float(rng.uniform(0.05, .95))):
            try:
                p = np.asarray(curve.evaluate(u))
            except Exception:
                continue
            for direction in (np.array([0., 0., 1.]), np.eye(3)[int(rng.integers(3))], rng.normal(size=3)):
                shift = rng.normal(size=3)*rng.choice([0., 0., 1e-9, 1e-6])
                d = _unit(direction)*rng.choice([.5, 3.])
                cases.append((LinePath(tuple(p + shift - d), tuple(p + shift + d)), curve))
    return cases, stats


def _run_conditioned(seed, count):
    cases, stats = _conditioned_cases(seed, count)
    bad = []
    for line, curve in cases:
        for x, y in ((line, curve), (curve, line)):
            for policy in ("default", "bounds"):
                got, ref = (both(x, y, 1e-9) if policy == "default" else both_policy_on(x, y, 1e-9))
                if ref[0] == "ok" and ref[1] != "()":
                    stats["nonempty"] += 1
                if got != ref:
                    bad.append((type(x).__name__, type(y).__name__, policy, got, ref))
    return bad, stats


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_r4_randomized_conditioned_cylinder_differential(seed):
    bad, stats = _run_conditioned(seed, 70)
    assert stats["built"] >= 15, stats                    # enough constructions succeeded
    assert stats["nonempty"] >= 20, stats                 # and enough cases have a non-empty original result
    assert not bad, f"{len(bad)} mismatches, first: {bad[0]}"


# ---------------------------------------------------------------- round 5: enclosure containment of cylinder curves
import anygeometry.arrangement_geometry as _ag

_B1 = dict(n1=(0.4771723190205203, -0.12821495974963623, -0.8694064078766489),
           n2=(0.7243519434948941, 0.6854406832255213, 0.07406302542092481),
           radial=(0.597711332364669, -0.6778911033720145, 0.4280243160521664),
           origin2=(72435194.3494894, 68544068.32255213, 7406302.542092482), radius2=2., start=.3)
_B2 = dict(n1=(-0.8153544006431129, -0.5336730022260952, 0.22447790102122792),
           n2=(-0.47788401709472794, 0.4792787487641864, 0.7361513072653213),
           radial=(-0.5244632620098799, 0.5166015682258827, -0.6768021176891476),
           origin2=(-47789.42792054749, 47928.88570460989, 73613.80643581653), radius2=1.4674782283599237,
           start=1.0596126305492337)


def _codex_r5_curve(case):
    a = Cylinder((0, 0, 0), case["n1"], case["radial"], 1., 4.)
    b = Cylinder(case["origin2"], case["n2"], case["radial"], case["radius2"], 4.)
    return CylinderIntersectionCurve(a, b, case["start"], 1e-12, 1)


@pytest.mark.parametrize("case", [_B1, _B2], ids=["B1_far_support_origin", "B2_near_tangency"])
def test_r5_codex_counterexamples_same_as_original(case):
    curve = _codex_r5_curve(case)
    p = np.asarray(curve.evaluate(0.))
    line = LinePath(tuple(p), tuple(p + (0., 0., 1.)))
    for x, y in ((line, curve), (curve, line)):
        for policy in ("default", "bounds"):
            got, ref = (both(x, y, 1e-9) if policy == "default" else both_policy_on(x, y, 1e-9))
            assert got == ref, (policy, got, ref)


def _admitted_box(curve, lower, upper, tol):
    """The box the shortcuts use for a curve the guard admits, or None (guard declines)."""
    if not _ag._shortcut_curve_ok(curve, tol):
        return None
    fn = getattr(_ag, "_shortcut_bounds", None)
    try:
        if fn is not None:
            lo, hi = fn(curve, lower, upper)
        else:
            lo, hi = curve.bounds(lower, upper)
    except Exception:
        return None
    return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float), fn is not None


def _adversarial_cylinder_curves(seed, count):
    """B1/B2-style and random pairs: far origins (shifted along the second axis), near tangency, near-parallel and
    anti-parallel axes, huge/tiny radii, start phases up to 1e15, log-scale sweeps."""
    rng = np.random.default_rng(seed)
    out, tried = [], 0
    for index in range(count):
        tried += 1
        n1 = _unit(rng.normal(size=3))
        kind = index % 5
        if kind == 3:                                           # near-parallel / anti-parallel
            theta = 10**rng.uniform(-9, -1)
            pivot = _unit(np.cross(n1, rng.normal(size=3)))
            n2 = _unit((n1*np.cos(theta) + np.cross(pivot, n1)*np.sin(theta))*rng.choice([1., -1.]))
        else:
            n2 = _unit(rng.normal(size=3))
        r1 = float(rng.choice([1., 10**rng.uniform(-3, 3)]))
        r2 = float(rng.choice([2., 10**rng.uniform(-3, 3)]))
        perp = np.cross(n1, n2)
        perp = perp/np.linalg.norm(perp) if np.linalg.norm(perp) > 0 else _unit(rng.normal(size=3))
        sep = float(rng.uniform(0., 1.5))*(r1 + r2)
        if kind == 2:                                           # near tangency: axes separation about r1 + r2
            sep = (r1 + r2)*(1. - 10**rng.uniform(-12, -1))
        far = float(rng.choice([0., 1e3, 1e6, 1e8]))*(1. if kind in (1, 4) else 0.)
        origin2 = far*n2 + sep*perp + rng.normal(size=3)*10**rng.uniform(-3, 0)*(r1 + r2)*(kind == 0)
        radial = _unit(rng.normal(size=3))
        try:
            a = Cylinder((0, 0, 0), tuple(n1), tuple(_frame_for(n1, rng)) if kind != 1 else tuple(radial), r1, 4.)
            b = Cylinder(tuple(origin2), tuple(n2), tuple(_frame_for(n2, rng)) if kind != 1 else tuple(radial), r2, 4.)
            start = float(rng.choice([0., .3, 1., 5.9, rng.uniform(0., 6.), 1e3, 1e15]))
            sweep = float(rng.choice([1., 3., 6.2, 10**rng.uniform(-12, 0.7)]))*float(rng.choice([1., -1.]))
            curve = CylinderIntersectionCurve(a, b, start, sweep, int(rng.choice([-1, 1])))
            curve.evaluate(0.)
        except Exception:
            continue
        out.append(curve)
    return out, tried


def _containment_report(seed, count, tol=1e-9):
    curves, tried = _adversarial_cylinder_curves(seed, count)
    admitted, declined, bad, exact_bad = [], 0, [], []
    rng = np.random.default_rng(seed + 1000)
    for curve in curves:
        ranges = [(0., 1.)] + [(i/8, (i+1)/8) for i in range(8)]
        whole = _admitted_box(curve, 0., 1., tol)
        if whole is None:
            declined += 1
            continue
        admitted.append(curve)
        for lower, upper in ranges:
            box = _admitted_box(curve, lower, upper, tol)
            if box is None:
                continue
            lo, hi, new = box
            ts = np.unique(np.concatenate(([lower, upper], rng.uniform(lower, upper, 24))))
            try:
                pts = np.asarray(curve.evaluate(ts))
            except Exception:
                continue
            top = max(1., float(np.max(np.abs(np.concatenate((lo, hi))))))
            pad = tol*(1 + 1e-9) + 1e3*np.finfo(float).eps*top
            if np.any(pts < lo - pad) or np.any(pts > hi + pad):
                bad.append((lower, upper))
            if new and (np.any(pts < lo) or np.any(pts > hi)):
                exact_bad.append((lower, upper))
    return curves, tried, admitted, declined, bad, exact_bad


@pytest.mark.parametrize("seed", [11, 12, 13, 14])
def test_r5_admitted_cylinder_boxes_contain_every_evaluated_point(seed):
    curves, tried, admitted, declined, bad, exact_bad = _containment_report(seed, 1000)
    assert len(curves) >= 150, (tried, len(curves))                  # enough constructions succeeded
    assert len(admitted) >= 60, (len(admitted), declined)            # enough curves admitted by the guard
    assert not bad, f"{len(bad)} boxes miss an evaluated point (padded); first {bad[0]}; admitted {len(admitted)} declined {declined}"
    assert not exact_bad, f"{len(exact_bad)} boxes miss an evaluated point exactly; admitted {len(admitted)} declined {declined}"


@pytest.mark.parametrize("seed", [21, 22])
def test_r5_filtered_equals_unfiltered_on_admitted_cylinder_curves(seed):
    curves, _tried, admitted, _declined, _bad, _exact = _containment_report(seed, 500)
    assert len(admitted) >= 25, len(admitted)
    rng = np.random.default_rng(seed + 2000)
    nonempty = 0
    for curve in admitted[:40]:
        p = np.asarray(curve.evaluate(float(rng.choice([0., 1., rng.uniform(.05, .95)]))))
        direction = _unit(rng.choice([1., 0.])*np.array([0., 0., 1.]) + rng.normal(size=3)*rng.choice([0., 1.]) + 1e-3)
        line = LinePath(tuple(p - direction), tuple(p + direction))
        for x, y in ((line, curve), (curve, line)):
            for policy in ("default", "bounds"):
                got, ref = (both(x, y, 1e-9) if policy == "default" else both_policy_on(x, y, 1e-9))
                if ref[0] == "ok" and ref[1] != "()":
                    nonempty += 1
                assert got == ref, (policy, got, ref)
    assert nonempty >= 20, nonempty


# ---------------------------------------------------------------- round 6: endpoint snap, completeness table, stronger net
def _snap_curve(L=2.**24, start=0., sweep=1e-12, radius2=2., branch=1):
    a = Cylinder((0., 0., 0.), (0., 0., 1.), (0., 1., 0.), 1., 4.)
    b = Cylinder((0., 0., L), (1., 0., 0.), (0., 1., 0.), radius2, 4.)
    return CylinderIntersectionCurve(a, b, start, sweep, branch)


def test_r6_endpoint_snap_cylinder_curve_same_as_original():
    # evaluate snaps the double-root endpoints to z = -b / (2 a) (here z = L) although the discriminant is positive
    L = 2.**24
    curve = _snap_curve(L)
    scalar = [np.asarray(curve.evaluate(float(t))) for t in (0., 1.)]
    vector = np.asarray(curve.evaluate(np.array([0., .5, 1.])))
    assert abs(scalar[0][2] - L) < 1e-6 and abs(vector[0][2] - L) < 1e-6     # the snap really happens
    line = LinePath((0., 1., L), (0., 1., L + .25))
    for x, y in ((line, curve), (curve, line)):
        for policy in ("default", "bounds"):
            got, ref = (both(x, y, 1e-9) if policy == "default" else both_policy_on(x, y, 1e-9))
            assert got == ref, (policy, got, ref)


def test_r6_endpoint_snap_box_contains_scalar_endpoints_and_vector_samples():
    L = 2.**24
    curve = _snap_curve(L)
    checked = 0
    for lower, upper in [(0., 1.)] + [(i/8, (i+1)/8) for i in range(8)]:
        box = _admitted_box(curve, lower, upper, 1e-9)
        if box is None:
            continue                                            # declined: the original route decides
        lo, hi, _new = box
        ts = np.unique(np.concatenate(([lower, upper], np.linspace(lower, upper, 9))))
        points = [np.asarray(curve.evaluate(float(t))) for t in (lower, upper)] + list(np.asarray(curve.evaluate(ts)))
        for point in points:
            assert not np.any(np.isnan(point))
            assert np.all(point >= lo) and np.all(point <= hi), (lower, upper, point, lo, hi)
        checked += 1
    assert checked >= 1                                         # some piece away from the snapped ends is admitted


# Mechanical completeness check: every conditional, raise and masked operation of the float code that builds the points of
# CylinderIntersectionCurve.evaluate has an interval counterpart in _cylinder_box_once or an explicit decline. The table is
# keyed by the unparsed condition; the source hash of each function pins the table to the reviewed code (change => review).
_EVAL_DISPOSITION = {
    "_angle": {
        "IF self.parameterization == 'left_square'": "decline: shortcut requires parameterization == 'linear'",
        "IF self.parameterization == 'right_square'": "decline: shortcut requires parameterization == 'linear'",
        "IF self.parameterization == 'both_sine'": "decline: shortcut requires parameterization == 'linear'",
    },
    "_root_at_angles": {
        "IF magnitude": "interval: m > 0 required (decline otherwise)",
        "IF discriminant is not None": "decline: only set for non-linear parameterizations",
        "IF np.any(disc < -roundoff)": "interval counterpart: the box relies only on evaluate returning (a refusal is a separate original outcome and is never skipped: curve set-up and both end evaluations must succeed, and disc_hi > 0 is required); for the values evaluate returns, root = sqrt(max(disc, 0)) lies in the clamped sqrt interval. A mixed-sign disc interval does NOT exclude disc < -roundoff, which would make evaluate raise: the shortcut then returns () where the original raises. Accepted only because the guard checks both ends evaluate and the interior of a linear chart is real (the constructor proves the chart real over its whole interval)",
        "CALL np.maximum(disc, 0)": "interval: root lower end clamped at 0",
        "CALL np.any(disc < -roundoff)": "decline: see IF np.any(disc < -roundoff)",
        "CALL np.abs(c)": "not a branch: the roundoff term of the refusal test only",
        "RAISE raise GeometryError('intersection branch leaves the real cylinder intersection')": "decline: see the refusal condition above; curve set-up evaluation at both ends must not raise",
    },
    "_root": {
        "IF self.parameterization != 'linear'": "decline: shortcut requires parameterization == 'linear'",
        "IF np.any(regular)": "decline: regular needs a non-linear parameterization, or the double-root snap, which is declined at t = 0 and t = 1 when the discriminant interval does not prove root^2 > envelope",
        "CALL np.any(regular)": "same as IF np.any(regular)",
        "IF np.any(regular & (root * root > envelope))": "decline: part of the snap branch",
        "CALL np.any(regular & (root * root > envelope))": "decline: part of the snap branch",
        "CALL np.where(regular, -b / (2 * a), z)": "decline: the snapped value; avoided by proving no snap at the end ranges",
        "CALL np.where(regular, 0.0, root)": "decline: the snapped root; avoided by proving no snap",
        "CALL np.abs(-8 * n * dn)": "decline: part of the double-root snap condition (only evaluated at t = 0 or 1)",
        "IF self.parameterization == 'left_square'": "decline: non-linear only",
        "IF self.parameterization == 'right_square'": "decline: non-linear only",
        "IF np.any(endpoint_root * endpoint_root > envelope)": "decline: non-linear only",
        "CALL np.where(left, self.start_angle, self.start_angle + self.sweep_angle)": "decline: non-linear only",
        "CALL np.any(endpoint_root * endpoint_root > envelope)": "decline: non-linear only",
        "RAISE raise GeometryError('regular endpoint is not a discriminant transition')": "decline: non-linear only / snap branch",
        "CALL np.where(left, self.sweep_angle * np.sin(0.5 * np.pi * t) ** 2, -self.sweep_angle * np.cos(0.5 * np.pi * t) ** 2)": "decline: non-linear only",
    },
    "_coefficients": {},
    "evaluate": {},
}
_EVAL_SOURCE_SHA = {}                                           # filled by the first run of the pin test below (see assert)


def _conditionals(function):
    import ast
    import inspect
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.IfExp)):
            found.add("IF " + ast.unparse(node.test))
        elif isinstance(node, ast.Call) and ast.unparse(node.func) in ("np.where", "np.any", "np.maximum", "np.minimum", "np.abs", "np.divide"):
            found.add("CALL " + ast.unparse(node))
        elif isinstance(node, ast.Raise):
            found.add("RAISE " + ast.unparse(node))
    return found


def _normalized_source_hash(function):
    import hashlib
    import inspect
    text = "\n".join(line.strip() for line in inspect.getsource(function).replace("\r\n", "\n").split("\n") if line.strip())
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@pytest.mark.parametrize("name", sorted(_EVAL_DISPOSITION))
def test_r6_every_conditional_of_the_evaluate_path_has_an_interval_counterpart_or_decline(name):
    function = getattr(CylinderIntersectionCurve, name)
    found = _conditionals(function)
    table = set(_EVAL_DISPOSITION[name])
    assert found - table == set(), f"{name}: a new conditional needs an interval counterpart or a decline: {sorted(found - table)}"
    assert table - found == set(), f"{name}: table entries without code (code changed? review the table): {sorted(table - found)}"


def test_r6_evaluate_path_source_is_the_reviewed_one():
    expected = {"_angle": "34791f8ae77d616b", "_coefficients": "74cb10de6e2bd78b", "_root_at_angles": "74606db371e14ba3",
                "_root": "3baa9a815863ae22", "evaluate": "933def1a34a60adf"}
    actual = {name: _normalized_source_hash(getattr(CylinderIntersectionCurve, name)) for name in expected}
    assert actual == expected, f"the float code of the cylinder curve changed; re-derive the interval box and update the table: {actual}"


def _adversarial2(seed, count):
    """Kinds: 0 generic, 1 shift along the second axis (B1), 2 near tangency, 3 near-parallel/anti-parallel, 4 large shift
    along the FIRST axis, 5 large shift along all axes, 6 short sweep from 0 with a large axial shift (endpoint snap)."""
    rng = np.random.default_rng(seed)
    out = []
    for index in range(count):
        kind = index % 7
        n1 = _unit(rng.normal(size=3))
        if kind == 3:
            theta = 10**rng.uniform(-9, -1)
            pivot = _unit(np.cross(n1, rng.normal(size=3)))
            n2 = _unit((n1*np.cos(theta) + np.cross(pivot, n1)*np.sin(theta))*rng.choice([1., -1.]))
        else:
            n2 = _unit(rng.normal(size=3))
        r1 = float(rng.choice([1., 10**rng.uniform(-2, 2)]))
        r2 = float(rng.choice([2., 10**rng.uniform(-2, 2)]))
        perp = np.cross(n1, n2)
        perp = perp/np.linalg.norm(perp) if np.linalg.norm(perp) > 0 else _unit(rng.normal(size=3))
        sep = float(rng.uniform(0., 1.5))*(r1 + r2)
        if kind == 2:
            sep = (r1 + r2)*(1. - 10**rng.uniform(-12, -1))
        shift = np.zeros(3)
        big = float(rng.choice([2.**14, 2.**20, 2.**24, 1e6, 1e8]))
        if kind == 1:
            shift = big*n2
        elif kind == 4:
            shift = big*n1
        origin1 = np.zeros(3)
        if kind == 5:                                           # translate the whole configuration far along a generic direction
            origin1 = big*_unit(rng.normal(size=3))
            shift = origin1 + float(rng.choice([0., 1e3, big]))*n2
        origin2 = shift + sep*perp
        try:
            if kind == 6:
                curve = _snap_curve(float(rng.choice([2.**14, 2.**20, 2.**24, 2.**26])), float(rng.choice([0., 1e-9, .3])),
                                    float(10**rng.uniform(-12, -6))*float(rng.choice([1., -1.])), float(rng.uniform(1.5, 3.)),
                                    int(rng.choice([-1, 1])))
            else:
                a = Cylinder(tuple(origin1), tuple(n1), tuple(_frame_for(n1, rng)), r1, 4.)
                b = Cylinder(tuple(origin2), tuple(n2), tuple(_frame_for(n2, rng)), r2, 4.)
                start = float(rng.choice([0., .3, 1., 5.9, rng.uniform(0., 6.), 1e3, 1e15]))
                sweep = float(rng.choice([1., 3., 6.2, 10**rng.uniform(-12, 0.7)]))*float(rng.choice([1., -1.]))
                curve = CylinderIntersectionCurve(a, b, start, sweep, int(rng.choice([-1, 1])))
            curve.evaluate(0.)
        except Exception:
            continue
        out.append((kind, curve))
    return out


def _containment_report2(seed, count, tol=1e-9):
    built = {k: 0 for k in range(7)}
    admitted_ok = {k: 0 for k in range(7)}
    bad, skipped, checked = [], 0, 0
    rng = np.random.default_rng(seed + 5000)
    for kind, curve in _adversarial2(seed, count):
        built[kind] += 1
        if _admitted_box(curve, 0., 1., tol) is None and _admitted_box(curve, 0., .125, tol) is None:
            continue
        clean, checked_ranges = True, 0
        for lower, upper in [(0., 1.)] + [(i/8, (i+1)/8) for i in range(8)]:
            box = _admitted_box(curve, lower, upper, tol)
            if box is None:
                continue
            lo, hi, exact = box
            ts = np.unique(np.concatenate(([lower, upper], rng.uniform(lower, upper, 12))))
            try:
                pts = np.vstack([np.asarray(curve.evaluate(float(t))) for t in (lower, upper)]    # scalar endpoint path
                                 + [np.asarray(curve.evaluate(ts))])                                  # vector path
            except Exception:
                skipped += 1
                continue
            checked += 1
            checked_ranges += 1
            if np.any(np.isnan(pts)):
                bad.append((kind, lower, upper, "nan"))
                clean = False
                continue
            pad = 0. if exact else tol*(1 + 1e-9) + 1e3*np.finfo(float).eps*max(1., float(np.max(np.abs(np.concatenate((lo, hi))))))
            if not (np.all(pts >= lo - pad) and np.all(pts <= hi + pad)):
                bad.append((kind, lower, upper))
                clean = False
        if clean and checked_ranges >= 1:                       # a curve whose evaluation raised is not counted
            admitted_ok[kind] += 1
    return built, admitted_ok, bad, skipped, checked


@pytest.mark.parametrize("seed", [31, 32, 33])
def test_r6_stronger_containment_net(seed):
    built, admitted_ok, bad, skipped, checked = _containment_report2(seed, 1400)
    assert all(built[k] >= (2 if k == 2 else 12) for k in range(7)), built                         # per-kind constructions
    assert all(admitted_ok[k] >= 6 for k in (0, 1, 3, 4, 5)), (built, admitted_ok)   # per-kind admitted AFTER the point check
    assert checked >= 400 and skipped <= 0.05*(checked + skipped), (checked, skipped)
    assert not bad, f"{len(bad)} box/point violations; first {bad[0]}; built {built} admitted_ok {admitted_ok}"


# ---------------------------------------------------------------- round 7: huge / non-finite coordinates, all kinds
import warnings


def _huge_objects():
    big = 1e308
    return {
        "arc_codex": EllipticArc((-big, -big, 0.), (1., 1., 0.), (1., -1., 0.), 0., 1.),
        "line_codex": LinePath((big, big, 0.), (big, big, 1.)),
    }


@pytest.mark.filterwarnings("ignore")
def test_r7_codex_huge_coordinate_ellipse_and_line_same_as_original():
    # the original route raises ValueError('cannot convert float NaN to integer') (inf - inf in the least squares, then
    # atan2 -> math.ceil); the shortcuts must not turn it into ()
    objs = _huge_objects()
    for x, y in ((objs["line_codex"], objs["arc_codex"]), (objs["arc_codex"], objs["line_codex"])):
        for policy in ("default", "bounds"):
            got, ref = (both(x, y, 1e-9) if policy == "default" else both_policy_on(x, y, 1e-9))
            assert got == ref, (policy, got, ref)
    with _junction_bounds_policy(False):
        ref = outcome(lambda: curve_junctions(objs["line_codex"], objs["arc_codex"], tolerance=1e-9))
    assert ref[0] == "raise", ref                                   # the recorded actual outcome of the original route


def _huge_value(rng):
    return float(rng.choice([1e30, 1e100, 1e200, 1e300, 1e308, -1e308, -1e200, np.nan, np.inf, -np.inf, 1e-300]))


def _huge_kind_objects(rng):
    """One object per kind, ordinary except for a few huge / NaN / inf entries."""
    def vec(scale=1., bad=False):
        v = rng.normal(size=3)*scale
        if bad:
            v[int(rng.integers(3))] = _huge_value(rng)
        return tuple(float(x) for x in v)
    objects = {}
    for tag, bad in (("ok", False), ("bad", True)):
        try:
            objects["line_" + tag] = LinePath(vec(5., bad), vec(5., bad))
        except Exception:
            pass
        try:
            objects["arc_" + tag] = EllipticArc(vec(5., bad), vec(1., bad), vec(1., bad), float(rng.choice([0., .3])), 2.)
        except Exception:
            pass
        try:
            scale = _huge_value(rng) if bad else 1.
            controls = tuple((float(x)*scale, float(y)*scale, 0.) for x, y in np.cumsum(rng.normal(size=(3, 2)), axis=0))
            objects["branch_" + tag] = make_C(controls)
        except Exception:
            pass
        try:
            far = _huge_value(rng) if bad else 0.
            a = Cylinder((0, 0, 0), (0, 0, 1), (1, 0, 0), 1., 4.)
            b = Cylinder((far, 0, 0), (1, 0, 0), (0, 1, 0), 1.5, 4.)
            objects["cyl_" + tag] = CylinderIntersectionCurve(a, b, 0.3, 1.0, 1)
        except Exception:
            pass
    return objects


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("seed", [41, 42, 43, 44])
def test_r7_randomized_huge_and_nonfinite_differential_all_kinds(seed):
    rng = np.random.default_rng(seed)
    objects = {}
    for _ in range(6):
        for name, obj in _huge_kind_objects(rng).items():
            objects[f"{name}{len(objects)}"] = obj
    names = sorted(objects)
    kinds = {n: n.rstrip("0123456789") for n in names}
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i+1:]
             if not (kinds[a].startswith("line") and kinds[b].startswith("line"))]
    rng.shuffle(pairs)
    pairs = pairs[:160]
    mismatches = []
    raised = 0
    for a, b in pairs:
        for x, y in ((objects[a], objects[b]), (objects[b], objects[a])):
            for policy in ("default", "bounds"):
                got, ref = (both(x, y, 1e-9) if policy == "default" else both_policy_on(x, y, 1e-9))
                raised += ref[0] == "raise"
                if got != ref:
                    mismatches.append((a, b, policy, got, ref))
    assert raised >= 5, raised                                            # the net really reaches original exceptions
    assert not mismatches, f"{len(mismatches)} mismatches; first: {mismatches[0]}"


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("seed", [51, 52])
def test_r7_huge_paired_ellipse_line_and_ellipse_ellipse_differential(seed):
    # the Codex pattern generalized: huge centers / line points of either sign, huge or ordinary axes, so that
    # inf - inf, overflow in the least squares and atan2 of NaN are reached through the ellipse inversion
    rng = np.random.default_rng(seed)
    mismatches, raised, total = [], 0, 0
    for _ in range(60):
        s = float(rng.choice([1e30, 1e100, 1e200, 1e300, 1e308]))
        sign = lambda: float(rng.choice([-1., 1.]))
        center = tuple(sign()*s*float(rng.choice([0., 1.])) for _ in range(3))
        scale_u = float(rng.choice([1., 1., s]))
        try:
            arc = EllipticArc(center, (scale_u, sign()*scale_u, 0.), (1., -1., 0.), float(rng.choice([0., .5])), float(rng.choice([1., 2.])))
        except Exception:
            continue
        point = tuple(sign()*s*float(rng.choice([0., 1.])) for _ in range(3))
        other = list(point)
        other[int(rng.integers(3))] += float(rng.choice([1., s]))
        try:
            line = LinePath(point, tuple(other))
            arc2 = EllipticArc(tuple(sign()*s*float(rng.choice([0., 1.])) for _ in range(3)), (1., 1., 0.), (1., -1., 0.), 0., 1.)
        except Exception:
            continue
        for x, y in ((line, arc), (arc, line), (arc, arc2), (arc2, arc)):
            for policy in ("default", "bounds"):
                total += 1
                got, ref = (both(x, y, 1e-9) if policy == "default" else both_policy_on(x, y, 1e-9))
                raised += ref[0] == "raise"
                if got != ref:
                    mismatches.append((type(x).__name__, type(y).__name__, policy, got, ref))
    assert total >= 200 and raised >= 10, (total, raised)
    assert not mismatches, f"{len(mismatches)} mismatches; first: {mismatches[0]}"


# ---------------------------------------------------------------- round 8: allow-list guard (dtypes, containers, degree)
from anygeometry.arrangement_geometry import BezierPath


@pytest.mark.filterwarnings("ignore")
def test_r8_integer_dtype_line_wraps_in_original_and_shortcut_declines():
    z = np.int64(0)
    line = LinePath((z, z, z), (np.int64(2**32), z, z))
    arc = EllipticArc((0., 100., 0.), (1., 0., 0.), (0., 1., 0.), 0., 1.)
    for x, y in ((line, arc), (arc, line)):
        for policy in ("default", "bounds"):
            got, ref = (both(x, y, 1e-9) if policy == "default" else both_policy_on(x, y, 1e-9))
            assert got == ref, (policy, got, ref)
    with _junction_bounds_policy(False):
        assert outcome(lambda: curve_junctions(line, arc, tolerance=1e-9)) == ("raise", "GeometryError", "degenerate arrangement line")


@pytest.mark.filterwarnings("ignore")
def test_r8_bool_coordinate_line_same_as_original():
    line = LinePath((False, False, False), (True, False, False))
    arc = EllipticArc((0., 100., 0.), (1., 0., 0.), (0., 1., 0.), 0., 1.)
    for x, y in ((line, arc), (arc, line)):
        for policy in ("default", "bounds"):
            got, ref = (both(x, y, 1e-9) if policy == "default" else both_policy_on(x, y, 1e-9))
            assert got == ref, (policy, got, ref)


@pytest.mark.filterwarnings("ignore")
def test_r8_list_controls_bezier_path_same_as_original_on_the_bounds_route():
    path = BezierPath([(0., 0., 0.), (1., 0., 0.)])
    line = LinePath((0., 50., 0.), (1., 50., 0.))
    for x, y in ((path, line), (line, path)):
        for policy in ("default", "bounds"):
            got, ref = (both(x, y, 1e-9) if policy == "default" else both_policy_on(x, y, 1e-9))
            assert got == ref, (policy, got, ref)


@pytest.mark.filterwarnings("ignore")
def test_r8_degree_2048_bezier_path_same_as_original_on_the_bounds_route():
    path = BezierPath(((0., 0., 0.),)*2049)
    line = LinePath((50., 0., 0.), (51., 0., 0.))
    for x, y in ((path, line), (line, path)):
        for policy in ("default", "bounds"):
            got, ref = (both(x, y, 1e-9) if policy == "default" else both_policy_on(x, y, 1e-9))
            assert got == ref, (policy, got, ref)
    with _junction_bounds_policy(False):
        assert outcome(lambda: curve_junctions(path, line, tolerance=1e-9))[:2] == ("raise", "OverflowError")


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("seed", [61, 62, 63])
def test_r8_randomized_dtype_container_degree_differential(seed):
    from fractions import Fraction
    rng = np.random.default_rng(seed)
    casts = [float, np.float64, np.float32, np.int64, np.int32, int, bool, Fraction, np.float16]

    def cast(value):
        kind = casts[int(rng.integers(len(casts)))]
        try:
            if kind in (np.int64, np.int32, int, bool):
                return kind(int(round(value)) if kind is not bool else int(value > .5))
            return kind(float(value))
        except Exception:
            return float(value)

    def point(scale, typed):
        values = rng.normal(size=3)*scale
        if rng.random() < .3:
            values[0] = float(rng.choice([2.**32, 2.**40, 1e30, 2.**31]))
        return tuple(cast(v) if typed else float(v) for v in values)

    def container(rows):
        pick = int(rng.integers(3))
        return list(rows) if pick == 0 else tuple(rows) if pick == 1 else np.array(rows, dtype=float)

    objects = []
    for _ in range(40):
        typed = rng.random() < .7
        try:
            objects.append(("line", LinePath(point(5., typed), point(5., typed))))
            objects.append(("arc", EllipticArc(point(5., typed), (1., 0., 0.), (0., 1., 0.), float(rng.choice([0., .5])), 2.)))
            degree = int(rng.choice([1, 2, 3, 5]))
            rows = [tuple(float(v) for v in rng.normal(size=3)) for _ in range(degree + 1)]
            objects.append(("bezier", BezierPath(container(rows))))
        except Exception:
            continue
    assert {k for k, _ in objects} == {"line", "arc", "bezier"}
    mismatches, raised, total = [], 0, 0
    pairs = [(a, b) for i, a in enumerate(objects) for b in objects[i+1:i+4] if not (a[0] == "line" and b[0] == "line")]
    for (ka, a), (kb, b) in pairs:
        for x, y in ((a, b), (b, a)):
            for policy in ("default", "bounds"):
                total += 1
                got, ref = (both(x, y, 1e-9) if policy == "default" else both_policy_on(x, y, 1e-9))
                raised += ref[0] == "raise"
                if got != ref:
                    mismatches.append((ka, kb, policy, got, ref))
    assert total >= 200 and raised >= 10, (total, raised)
    assert not mismatches, f"{len(mismatches)} mismatches; first: {mismatches[0]}"


# ---------------------------------------------------------------- round 9: shapes / lengths / ndim of ordinary containers
# Contract (owner, narrowed): shortcut == original for ORDINARY inputs (exact float / np.float64 leaves, plain tuple or float64
# ndarray containers) in values, typed refusals and exception type / message. Out of contract: ndarray / tuple subclasses and
# objects with overridden __iter__ / __len__ / __array__, mutated private caches, the cancellation exception of the
# default-off bounds route.
@pytest.mark.filterwarnings("ignore")
def test_r9_one_dimensional_bezier_controls_same_as_original():
    first = LinePath((100., 100., 100.), (101., 100., 100.))
    second = BezierPath(np.array([0., 0., 0.], dtype=np.float64))
    for x, y in ((first, second), (second, first)):
        for policy in ("default", "bounds"):
            got, ref = (both(x, y, 1e-9) if policy == "default" else both_policy_on(x, y, 1e-9))
            assert got == ref, (policy, got, ref)
    with _junction_bounds_policy(False):
        assert outcome(lambda: curve_junctions(first, second, tolerance=1e-9))[:2] == ("raise", "IndexError")


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("length", [2, 4, 5])
def test_r9_wrong_length_line_tuples_same_as_original(length):
    zeros = (0.,)*length
    one = (1.,) + (0.,)*(length - 1)
    line = LinePath(zeros, one)
    arc = EllipticArc((0., 100., 0.), (1., 0., 0.), (0., 1., 0.), 0., 1.)
    far_line = LinePath((0., 50., 0.), (1., 50., 0.))
    for other in (arc, far_line):
        for x, y in ((line, other), (other, line)):
            for policy in ("default", "bounds"):
                got, ref = (both(x, y, 1e-9) if policy == "default" else both_policy_on(x, y, 1e-9))
                assert got == ref, (length, type(other).__name__, policy, got, ref)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("shape", [(3,), (1, 3), (4,), (2, 2), (2, 4), (3, 1), (2, 3, 1)])
def test_r9_odd_shape_bezier_controls_same_as_original(shape):
    controls = np.arange(float(np.prod(shape))).reshape(shape) + 100.
    try:
        path = BezierPath(controls)
    except Exception:
        return
    line = LinePath((0., 50., 0.), (1., 50., 0.))
    arc = EllipticArc((0., -100., 0.), (1., 0., 0.), (0., 1., 0.), 0., 1.)
    for other in (line, arc):
        for x, y in ((path, other), (other, path)):
            for policy in ("default", "bounds"):
                got, ref = (both(x, y, 1e-9) if policy == "default" else both_policy_on(x, y, 1e-9))
                assert got == ref, (shape, type(other).__name__, policy, got, ref)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("make", [
    lambda: np.array(50., dtype=np.float64),                         # 0-d array
    lambda: np.array([[50., 50., 50.]]),                             # 2-D
    lambda: np.array([50., 50.]),                                    # length 2
    lambda: np.array([50., 50., 50., 50.]),                          # length 4
    lambda: (50., 50.),                                              # 2-tuple
    lambda: (50., 50., 50., 50.),                                    # 4-tuple
    lambda: np.array([[50.], [50.], [50.]]),                         # (3, 1)
    lambda: 50.,                                                     # bare float
], ids=["0d", "2d", "len2", "len4", "tuple2", "tuple4", "col3", "scalar"])
def test_r9_point_filter_positions_of_other_shapes_decline(make):
    curve = make_C()
    far = ma._point_clear_of_curve(curve, 1e-9)
    assert far is not None
    position = make()
    try:
        is_far = far(position)
    except Exception as error:                                       # the filter itself must never raise
        pytest.fail(f"far() raised {type(error).__name__}: {error}")
    original = outcome(lambda: point_parameters(curve, position, tolerance=1e-9))
    assert (not is_far) or original == ("ok", repr(())), (is_far, original)
    assert is_far is False                                           # ordinary positions are exactly three float coordinates


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("seed", [71, 72])
def test_r9_randomized_shape_length_ndim_differential(seed):
    rng = np.random.default_rng(seed)

    def point():
        length = int(rng.choice([2, 3, 3, 3, 4]))
        values = tuple(float(v) for v in rng.normal(size=length)*float(rng.choice([1., 50.])))
        return values if rng.random() < .7 else np.array(values)

    def controls():
        rows, cols = int(rng.choice([1, 2, 3, 4])), int(rng.choice([2, 3, 3, 3, 4]))
        array = rng.normal(size=(rows, cols))*float(rng.choice([1., 50.]))
        pick = int(rng.integers(5))
        return (array if pick == 0 else tuple(map(tuple, array.tolist())) if pick == 1
                else array.ravel() if pick == 2 else array[0] if pick == 3 else array.reshape(rows, cols, 1))

    objects = []
    for _ in range(30):
        try:
            objects.append(("line", LinePath(point(), point())))
            objects.append(("arc", EllipticArc(point() if rng.random() < .5 else (0., 100., 0.), (1., 0., 0.), (0., 1., 0.), 0., 1.)))
            objects.append(("bezier", BezierPath(controls())))
        except Exception:
            continue
    assert {k for k, _ in objects} == {"line", "arc", "bezier"}
    mismatches, raised, total = [], 0, 0
    pairs = [(a, b) for i, a in enumerate(objects) for b in objects[i+1:i+4] if not (a[0] == "line" and b[0] == "line")]
    for (ka, a), (kb, b) in pairs:
        for x, y in ((a, b), (b, a)):
            for policy in ("default", "bounds"):
                total += 1
                got, ref = (both(x, y, 1e-9) if policy == "default" else both_policy_on(x, y, 1e-9))
                raised += ref[0] == "raise"
                if got != ref:
                    mismatches.append((ka, kb, policy, got, ref))
    assert total >= 200 and raised >= 20, (total, raised)
    assert not mismatches, f"{len(mismatches)} mismatches; first: {mismatches[0]}"


# ---------------------------------------------------------------- round 10: container-valued scalar / point fields
def _set(obj, name, value):
    object.__setattr__(obj, name, value)
    return obj


def _fresh_arc():
    return EllipticArc((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), 0., 1.)


def _fresh_far_arc():
    return EllipticArc((0., 100., 0.), (1., 0., 0.), (0., 1., 0.), 0., 1.)


def _fresh_cyl_curve():
    a = Cylinder((0, 0, 0), (0, 0, 1), (1, 0, 0), 1., 4.)
    b = Cylinder((-2, 0, 0), (1, 0, 0), (0, 1, 0), 1., 4.)
    return CylinderIntersectionCurve(a, b, 0.3, 1.0, 1)


_FAR_LINE = lambda: LinePath((0., 50., 5.), (1., 50., 5.))                       # noqa: E731
_R10_CASES = {
    # Codex round 10: container-valued scalar angle field plus nested-empty point on the line
    "codex_angle_tuple": lambda: (LinePath(((), (), ()), (0., 0., 0.)), _set(_fresh_arc(), "start_angle", (0.,))),
    # siblings of the same class: nested / empty tuples, singleton and 0-d arrays in scalar or point fields of every curve type
    "line_nested_point": lambda: (LinePath(((0.,), (0.,), (0.,)), (1., 0., 0.)), _fresh_far_arc()),
    "line_empty_end": lambda: (LinePath((0., 0., 0.), ((), (), ())), _fresh_far_arc()),
    "arc_center_nested": lambda: (_set(_fresh_arc(), "center", ((0.,), (0.,), (0.,))), _FAR_LINE()),
    "arc_sweep_singleton_array": lambda: (_set(_fresh_arc(), "sweep_angle", np.array([1.])), _FAR_LINE()),
    "arc_start_zero_d_array": lambda: (_set(_fresh_arc(), "start_angle", np.array(0.)), _FAR_LINE()),
    "bezier_rows_nested": lambda: (BezierPath((((0.,), (0.,), (0.,)), ((1.,), (0.,), (0.,)))), _FAR_LINE()),
    "cyl_sweep_tuple": lambda: (_set(_fresh_cyl_curve(), "sweep_angle", (1.,)), _FAR_LINE()),
    "cyl_start_singleton_array": lambda: (_set(_fresh_cyl_curve(), "start_angle", np.array([0.3])), _FAR_LINE()),
    "branch_start_tuple": lambda: (_set(make_C(), "start", (0.,)), _FAR_LINE()),
    "branch_sweep_zero_d_array": lambda: (_set(make_C(), "sweep", np.array(1.)), _FAR_LINE()),
}


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("name", sorted(_R10_CASES))
def test_r10_container_valued_scalar_and_point_fields_same_as_original(name):
    a, b = _R10_CASES[name]()
    for x, y in ((a, b), (b, a)):
        for policy in ("default", "bounds"):
            got, ref = (both(x, y, 1e-9) if policy == "default" else both_policy_on(x, y, 1e-9))
            assert got == ref, (name, policy, got, ref)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("name", sorted(_R10_CASES))
def test_r10_the_guard_never_raises(name):
    a, b = _R10_CASES[name]()
    for obj in (a, b):
        try:
            result = _ag._shortcut_curve_ok(obj, 1e-9)
        except Exception as error:
            pytest.fail(f"guard raised {type(error).__name__}: {error}")
        assert result in (True, False)
