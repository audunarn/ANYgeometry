"""The exact curve family of a ruled support (Cylinder or Cone) cut with a quadric.

A ``QuadricIntersectionCurve`` is one real branch of the ruling equation over an angular chart. These
tests pin what the arrangement engine relies on: points lie on both supports to rounding, the first and
second derivatives agree with differences in every parameterization (including the square-root
reparametrizations that keep a fold regular), sub-charts and affine images stay exact, the interval
bounds enclose the curve, and invalid charts are refused instead of approximated.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from anygeometry import GeometryError
from anygeometry.analytic_supports import cylinder_cylinder_support
from anygeometry.generators import cylinder
from anygeometry.quadric_algebra import QuadricSupport, RuledSupport, get_plan
from anygeometry.quadric_curves import QuadricIntersectionCurve
from anygeometry.quadric_supports import cone_support
from anygeometry.surfaces import Cone, Cylinder, Plane


def _cone(off_degrees=10., *, r0=.5, r1=1., height=5., origin=(0., 0., 0.)):
    a = math.radians(off_degrees)
    return Cone(origin, (math.cos(a), 0., math.sin(a)), (0., 1., 0.), r0, r1, height, 0., math.tau)


def _cylinder(radius=2.):
    return Cylinder((0., 0., -3.), (0., 0., 1.), (1., 0., 0.), radius, 6., 0., math.tau)


def _plate(tilt_degrees):
    t = math.radians(tilt_degrees)
    n = np.array([0., math.sin(t), math.cos(t)])
    u = np.array([1., 0., 0.])
    v = np.cross(n, u)
    return Plane(np.array([0., 0., 1.5]) - 3 * u - 3 * v, 6 * u, 6 * v)


def _tall_cone():
    return Cone((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), .5, 1.5, 3., 0., math.tau)


# (name, cone, other support). The third one folds: its charts end at tangent rulings.
CASES = {
    "closed loop, 10 deg off perpendicular": (_cone(10.), _cylinder()),
    "folds, cone axis near the wall": (_cone(10., origin=(0., 2.2, 0.)), _cylinder()),
    "wide cone crossing the cylinder": (_cone(60., r0=1., r1=3.), _cylinder()),
    "hyperbola on a plate": (_tall_cone(), _plate(80.)),
}


def _charts(name):
    cone, other = CASES[name]
    curves = [c for c in cone_support(cone, other, tolerance=1e-9).curves if isinstance(c, QuadricIntersectionCurve)]
    assert curves, name
    return cone, other, curves


def _distance(support, points):
    quadric = QuadricSupport.from_surface(support)
    return np.abs(quadric.value(points)) / quadric.gradient_norm(points)


@pytest.mark.parametrize("name", CASES)
def test_every_chart_lies_on_both_supports_and_the_loop_closes(name):
    cone, other, curves = _charts(name)
    t = np.linspace(0., 1., 65)
    for curve in curves:
        points = curve.evaluate(t)
        assert np.all(np.isfinite(points))
        assert _distance(cone, points).max() < 1e-12
        assert _distance(other, points).max() < 1e-12
        assert np.allclose(curve.evaluate(0.), points[0]) and np.allclose(curve.evaluate(1.), points[-1])
    if name.startswith("closed loop"):                       # the four quarter charts chain around one loop
        ordered = sorted(curves, key=lambda c: c.start_angle)
        for before, after in zip(ordered, ordered[1:] + ordered[:1]):
            assert np.linalg.norm(before.evaluate(1.) - after.evaluate(0.)) < 1e-12


@pytest.mark.parametrize("name", CASES)
def test_first_derivative_matches_differences_in_every_parameterization(name):
    _cone_, _other, curves = _charts(name)
    for curve in curves:
        for tau in (.07, .3, .5, .8, .95):
            h = 1e-6
            numeric = (curve.evaluate(tau + h) - curve.evaluate(tau - h)) / (2 * h)
            exact = curve.derivative(tau)
            assert np.linalg.norm(exact - numeric) <= 1e-6 * max(1., np.linalg.norm(exact)), (name, curve.parameterization, tau)
        for tau, step in ((0., 1e-7), (1., -1e-7)):          # ends, including the fold ends of square charts
            numeric = (curve.evaluate(tau + step) - curve.evaluate(tau)) / step
            exact = curve.derivative(tau)
            assert np.all(np.isfinite(exact))
            assert np.linalg.norm(exact - numeric) <= 1e-4 * max(1., np.linalg.norm(exact)), (name, curve.parameterization, tau)


@pytest.mark.parametrize("name", CASES)
def test_second_derivative_matches_differences_of_the_first(name):
    _cone_, _other, curves = _charts(name)
    for curve in curves:
        for tau in (.1, .4, .6, .9):
            h = 1e-5
            numeric = (curve.derivative(tau + h) - curve.derivative(tau - h)) / (2 * h)
            exact = curve.second_derivative(tau)
            assert np.linalg.norm(exact - numeric) <= 1e-5 * max(1., np.linalg.norm(exact)), (name, curve.parameterization, tau)


def test_a_fold_chart_is_regular_at_its_fold_end_and_its_branches_meet_there():
    cone, other, curves = _charts("folds, cone axis near the wall")
    folded = [c for c in curves if c.parameterization != "linear"]
    assert {c.parameterization for c in folded} >= {"left_square", "right_square"}
    assert {c.branch for c in folded} == {-1, 1}
    for curve in folded:
        end = 0. if curve.parameterization == "left_square" else 1.
        assert 0. < np.linalg.norm(curve.derivative(end)) < np.inf
        assert np.all(np.isfinite(curve.second_derivative(end)))
    # the two branches of one fold join at one point of the (smooth) intersection curve
    starts = [c.evaluate(0.) for c in folded if c.parameterization == "left_square"]
    assert np.linalg.norm(starts[0] - starts[1]) < 1e-12


def test_subcurves_stay_on_the_curve_and_reversal_is_exact():
    _cone_, _other, curves = _charts("closed loop, 10 deg off perpendicular")
    curve = curves[0]
    sub = curve.subcurve(.2, .7)
    t = np.linspace(0., 1., 17)
    for point in sub.evaluate(t):
        found = curve.parameters_of(point, tolerance=1e-9)
        assert any(.2 - 1e-6 <= tau <= .7 + 1e-6 for tau in found)
    assert np.allclose(curve.subcurve(.7, .2).evaluate(t), sub.evaluate(1. - t), atol=1e-13)
    assert np.allclose(sub.derivative(.5), curve.derivative(.45) * .5, rtol=1e-9)
    for chart in _charts("folds, cone axis near the wall")[2]:
        piece = chart.subcurve(.3, .6)
        assert piece.parameterization == "linear"            # away from the fold the angle chart is regular
        assert np.allclose(piece.evaluate(0.), chart.evaluate(.3), atol=1e-12)
        assert np.allclose(piece.evaluate(1.), chart.evaluate(.6), atol=1e-12)
        lower = chart.subcurve(0., .5)                        # a piece that keeps the fold end keeps its chart
        keeps_fold = chart.parameterization in ("left_square", "both_sine")
        assert (lower.parameterization != "linear") == keeps_fold
        with pytest.raises(GeometryError):
            chart.subcurve(.5, .5)


def test_an_affine_image_is_exact_and_composes():
    _cone_, _other, curves = _charts("closed loop, 10 deg off perpendicular")
    curve = curves[0]
    angle = .7
    rotation = np.array([[math.cos(angle), -math.sin(angle), 0.], [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
    first = np.eye(4)
    first[:3, :3] = 1.5 * rotation
    first[:3, 3] = (.3, -.2, .9)
    second = np.eye(4)
    second[:3, :3] = np.array([[1., .1, 0.], [0., 1., 0.], [0., .2, 1.]])
    second[:3, 3] = (-1., 0., .5)
    t = np.linspace(0., 1., 9)
    moved = curve.transformed(first)
    assert np.allclose(moved.evaluate(t), curve.evaluate(t) @ first[:3, :3].T + first[:3, 3], atol=1e-12)
    assert np.allclose(moved.derivative(t), curve.derivative(t) @ first[:3, :3].T, atol=1e-10)
    twice = moved.transformed(second)
    both = second @ first
    assert np.allclose(twice.evaluate(t), curve.evaluate(t) @ both[:3, :3].T + both[:3, 3], atol=1e-11)
    inverse_back = twice.subcurve(.25, .75)
    assert np.allclose(inverse_back.evaluate(0.), twice.evaluate(.25), atol=1e-11)
    with pytest.raises(GeometryError):
        curve.transformed(np.zeros((4, 4)))


@pytest.mark.parametrize("name", CASES)
def test_bounds_enclose_the_curve_tightly_enough_to_prune(name):
    _cone_, _other, curves = _charts(name)
    rng = np.random.default_rng(11)
    for curve in curves:
        for lo, hi in ((0., 1.), (0., .5), (.25, .3), (.9, 1.), tuple(sorted(rng.uniform(0., 1., 2)))):
            low, high = curve.bounds(lo, hi)
            points = curve.evaluate(np.linspace(lo, hi, 401))
            assert np.all(points >= low) and np.all(points <= high), (name, lo, hi)
            assert np.all(np.isfinite(low)) and np.all(np.isfinite(high))


@pytest.mark.parametrize("name", CASES)
def test_roots_on_a_plane_are_exactly_the_sign_changes(name):
    _cone_, _other, curves = _charts(name)
    rng = np.random.default_rng(5)
    for curve in curves:
        point = curve.evaluate(rng.uniform(.2, .8))
        normal = rng.normal(size=3)
        normal /= np.linalg.norm(normal)
        u = np.cross(normal, rng.normal(size=3))
        u /= np.linalg.norm(u)
        plane = Plane(point, u, np.cross(normal, u))
        roots = curve.roots_on(plane)
        t = np.linspace(0., 1., 4001)
        values = (curve.evaluate(t) - point) @ normal
        changes = np.nonzero(np.sign(values[:-1]) * np.sign(values[1:]) < 0)[0]
        for index in changes:
            assert any(abs(root - t[index]) < 1e-3 for root in roots), (name, t[index], roots)
        for root in roots:
            assert abs(float((curve.evaluate(root) - point) @ normal)) < 1e-9


def test_a_chart_lies_on_its_own_supports_and_parameters_round_trip():
    cone, other, curves = _charts("wide cone crossing the cylinder")
    for curve in curves:
        assert curve.roots_on(cone) is None and curve.roots_on(other) is None
        for tau in (0., .1, .5, .77, 1.):
            found = curve.parameters_of(curve.evaluate(tau), tolerance=1e-9)
            assert any(abs(value - tau) < 1e-6 for value in found), (tau, found)


@pytest.mark.parametrize("make,expected", [
    (lambda c, o: QuadricIntersectionCurve(c, o, 0., 1., 2), "branch"),
    (lambda c, o: QuadricIntersectionCurve(c, o, 0., 1., 1, "cubic"), "parameterization"),
    (lambda c, o: QuadricIntersectionCurve(c, o, 0., 0.), "sweep"),
    (lambda c, o: QuadricIntersectionCurve(c, o, 0., math.inf), "finite"),
    (lambda c, o: QuadricIntersectionCurve(c, o, 0., 7.), "turn"),
    (lambda c, o: QuadricIntersectionCurve(c, o, 0., 1., 1, "linear", ((1, 0, 0), (0, 1, 0), (0, 0, 1))), "transformation"),
])
def test_invalid_definitions_are_refused(make, expected):
    cone, other = CASES["closed loop, 10 deg off perpendicular"]
    with pytest.raises(GeometryError, match=expected):
        make(cone, other)


def test_charts_must_split_at_folds_and_square_ends_must_be_folds():
    cone, other = CASES["folds, cone axis near the wall"]
    plan = get_plan(RuledSupport.from_surface(cone), other)
    fold = min(angle for angle, _multiplicity in plan.discriminant_roots())
    with pytest.raises(GeometryError, match="split"):                         # a fold inside the chart
        QuadricIntersectionCurve(cone, other, fold - .1, .2, 1)
    with pytest.raises(GeometryError, match="regular endpoint"):              # a square end away from any fold
        QuadricIntersectionCurve(cone, other, fold + .2, .1, 1, "left_square")
    with pytest.raises(GeometryError, match="single branch"):                 # a plane has one branch
        QuadricIntersectionCurve(_tall_cone(), _plate(80.), 0., .1, -1)
    with pytest.raises(GeometryError, match="real"):                          # outside the real intersection
        QuadricIntersectionCurve(_cone(10., origin=(0., 3., 0.)), _cylinder(.5), 0., .1, 1)


def test_cylinder_cylinder_charts_agree_with_the_established_curve():
    """The general branch reproduces :class:`CylinderIntersectionCurve` point for point."""
    base = cylinder(1., 3., origin=(0., 0., -1.5), circumferential_segments=8)
    other = cylinder(.75, 3., origin=(0., 0., 0.), axis=(math.cos(math.radians(10)), 0., math.sin(math.radians(10))),
                     radial_direction=(0., 1., 0.), circumferential_segments=8)
    checked = 0
    for a in (f.surface for f in base.faces.values()):
        for b in (f.surface for f in other.faces.values()):
            for old in cylinder_cylinder_support(a, b).curves:
                new = QuadricIntersectionCurve(RuledSupport.from_surface(a), b, old.start_angle, old.sweep_angle,
                                               old.branch, old.parameterization)
                for tau in (0., .1, .5, .9, 1.):
                    assert np.linalg.norm(old.evaluate(tau) - new.evaluate(tau)) < 1e-12
                checked += 1
    assert checked >= 4
