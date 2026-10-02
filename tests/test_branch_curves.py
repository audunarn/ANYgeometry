"""Contract of the exact polynomial-chart curve: a branch of a Bezier extrusion cut with a quadric.

The curve is checked on both supports, its derivatives against differences (including the regular fold ends of the
square charts and the linear branch of a quadric whose generators are parallel to the extrusion), its enclosure,
slicing, affine copies, roots and point inversion, and its refusals.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from anygeometry.branch_algebra import BezierRuledSupport, get_poly_plan
from anygeometry.branch_curves import BezierQuadricCurve
from anygeometry.branch_supports import bezier_quadric_support
from anygeometry.errors import GeometryError
from anygeometry.extrusions import BezierDirectrix
from anygeometry.quadric_algebra import QuadricSupport
from anygeometry.surfaces import Cone, Cylinder, ExtrudedSurface, Plane

CUBIC = ((0., 0., 0.), (1., 2., 0.), (2., -1., 0.), (3., 1., 0.))
QUARTIC = ((0., 0., 0.), (1., 2., 0.), (2., -1., 0.), (3., 2., 0.), (4., 0., 0.))


def wall(controls=CUBIC, vector=(.25, 0., 1.5)):
    return ExtrudedSurface(BezierDirectrix(controls), vector)


CASES = {
    "linear charts across the wall": (wall(), Cylinder((-2., .4, .6), (1., 0., 0.), (0., 1., 0.), .7, 8.)),
    "narrow pipe (sine charts)": (wall(), Cylinder((-2., .5, .7), (1., 0., 0.), (0., 1., 0.), .3, 8.)),
    "cone (square charts)": (wall(), Cone((-2., .5, .8), (1., 0., 0.), (0., 1., 0.), .1, 1., 8.)),
    "quartic against a tilted pipe": (wall(QUARTIC), Cylinder((-2., .5, .7), (math.cos(.17), 0., math.sin(.17)),
                                                               (0., 1., 0.), .4, 8.)),
    # the axis in the plane of the wall's base: every fold lies on the base curve, where B, C and the radical vanish together
    "pipe in the base plane (folds on the base curve)": (wall(), Cylinder((-2., 0., 0.), (1., 0., 0.), (0., 1., 0.), .3, 8.)),
    # rulings nearly along the pipe's axis: A = 1e-4 is tiny, so 1 / (4 A) magnifies the residue of a fold root
    "rulings nearly along the axis (tiny A)": (wall(CUBIC, (1., 0., .01)), Cylinder((-2., .5, .01), (1., 0., 0.), (0., 1., 0.), .3, 8.)),
    # slope 1/8 and the ruling (1, 0, 1/8) are exact doubles, so the ruling is exactly a generator of the cone
    "linear branch (a cone generator along the rulings)": (wall(CUBIC, (1., 0., .125)),
                                                           Cone((-2., .5, .4), (1., 0., 0.), (0., 1., 0.), .25, 1.125, 7.)),
}


def charts(name):
    first, second = CASES[name]
    curves = list(bezier_quadric_support(first, second).curves)
    assert curves, name
    return first, second, curves


def distance(support, points):
    quadric = QuadricSupport.from_surface(support)
    return np.abs(quadric.value(points)) / quadric.gradient_norm(points)


@pytest.mark.parametrize("name", CASES)
def test_every_chart_lies_on_both_supports(name):
    first, second, curves = charts(name)
    t = np.linspace(0., 1., 65)
    for curve in curves:
        points = curve.evaluate(t)
        assert np.all(np.isfinite(points))
        assert np.abs(first.evaluate_many(first.local_uv_many(points)) - points).max() < 1e-12
        assert distance(second, points).max() < 1e-11
        assert np.allclose(curve.evaluate(0.), points[0]) and np.allclose(curve.evaluate(1.), points[-1])
        assert np.array_equal(np.array([curve.evaluate(float(x)) for x in t]), points)       # scalar path = vector path


def test_the_linear_case_is_a_single_rational_branch():
    first, second, curves = charts("linear branch (a cone generator along the rulings)")
    plan = curves[0].plan()
    assert plan.linear and plan.A == [0]
    assert {c.branch for c in curves} == {1}
    with pytest.raises(GeometryError, match="single"):
        BezierQuadricCurve(curves[0].first, curves[0].second, curves[0].start, curves[0].sweep, -1)


@pytest.mark.parametrize("name", CASES)
def test_first_derivative_matches_differences_in_every_parameterization(name):
    _first, _second, curves = charts(name)
    for curve in curves:
        for tau in (.07, .3, .5, .8, .95):
            h = 1e-6
            numeric = (curve.evaluate(tau + h) - curve.evaluate(tau - h)) / (2 * h)
            exact = curve.derivative(tau)
            assert np.linalg.norm(exact - numeric) <= 1e-6 * max(1., np.linalg.norm(exact)), (name, curve.parameterization, tau)
        for tau, side in ((0., 1.), (1., -1.)):              # ends, including the fold ends of square charts
            exact = curve.derivative(tau)
            assert np.all(np.isfinite(exact))
            errors = []
            for step in (side * 10. ** -k for k in range(4, 10)):
                one, two = (curve.evaluate(tau + j * step) - curve.evaluate(tau) for j in (1, 2))
                numeric = 2 * one / step - two / (2 * step)                 # one-sided Richardson
                errors.append(np.linalg.norm(exact - numeric) / max(1., np.linalg.norm(exact)))
            # a chart can start a hair from a fold (a seam crossing 1e-5 away): no single step suits every end, but a
            # wrong end derivative would be off by orders of magnitude at all of them
            assert min(errors) <= 1e-6, (name, curve.parameterization, tau, errors)


@pytest.mark.parametrize("name", CASES)
def test_second_derivative_matches_differences_of_the_first(name):
    _first, _second, curves = charts(name)
    for curve in curves:
        for tau in (.1, .4, .6, .9):
            h = 1e-5
            numeric = (curve.derivative(tau + h) - curve.derivative(tau - h)) / (2 * h)
            exact = curve.second_derivative(tau)
            assert np.linalg.norm(exact - numeric) <= 1e-5 * max(1., np.linalg.norm(exact)), (name, curve.parameterization, tau)
        for tau, step in ((0., 1e-9), (1., -1e-9)):          # the one-sided jet at an end is the limit of the interior
            exact = curve.second_derivative(tau)
            inner = curve.second_derivative(tau + step)
            assert np.all(np.isfinite(exact))
            assert np.linalg.norm(exact - inner) <= 5e-3 * max(1., np.linalg.norm(exact)), (name, curve.parameterization, tau)


def test_a_fold_chart_is_regular_at_its_fold_end_and_its_branches_meet_there():
    _first, _second, curves = charts("cone (square charts)")
    folded = [c for c in curves if c.parameterization != "linear"]
    assert {c.parameterization for c in folded} >= {"left_square"}
    assert {c.branch for c in folded} == {-1, 1}
    for curve in folded:
        end = 0. if curve.parameterization == "left_square" else 1.
        assert 0. < np.linalg.norm(curve.derivative(end)) < np.inf
        assert np.all(np.isfinite(curve.second_derivative(end)))
    starts = [c.evaluate(0.) for c in folded if c.parameterization == "left_square"]
    assert np.array_equal(starts[0], starts[1])                               # one fold point, bit for bit, on both branches
    # the tangents there are parallel: the pair is one smooth curve through the fold
    d0, d1 = (c.derivative(0.) for c in folded if c.parameterization == "left_square")
    assert np.linalg.norm(np.cross(d0, d1)) < 1e-8 * np.linalg.norm(d0) * np.linalg.norm(d1)


def test_subcurves_stay_on_the_curve_and_reversal_is_exact():
    _first, _second, curves = charts("linear charts across the wall")
    curve = max(curves, key=lambda c: abs(c.sweep))
    sub = curve.subcurve(.2, .7)
    t = np.linspace(0., 1., 17)
    for point in sub.evaluate(t):
        found = curve.parameters_of(point, tolerance=1e-9)
        assert found and any(.2 - 1e-6 <= tau <= .7 + 1e-6 for tau in found)
    assert np.allclose(curve.subcurve(.7, .2).evaluate(t), sub.evaluate(1. - t), atol=1e-13)
    assert np.allclose(sub.derivative(.5), curve.derivative(.45) * .5, rtol=1e-9)
    for chart in charts("narrow pipe (sine charts)")[2] + charts("cone (square charts)")[2]:
        for lower, upper in ((0., .5), (.5, 1.), (0., 1.), (.3, .6)):
            piece = chart.subcurve(lower, upper)
            # a slice is reparametrized in t, so only its ends agree parameter by parameter; the rest is the same trace
            for tau, source in ((0., lower), (1., upper)):
                assert np.allclose(piece.evaluate(tau), chart.evaluate(source), atol=1e-12)
            for point in piece.evaluate(np.linspace(0., 1., 9)):
                found = chart.parameters_of(point, tolerance=1e-9)
                assert found and any(lower - 1e-6 <= tau <= upper + 1e-6 for tau in found)
            if lower == 0. and chart.parameterization in ("left_square", "both_sine"):
                assert piece.parameterization in ("left_square", "both_sine")            # a fold end stays a fold end
        assert chart.subcurve(.3, .6).parameterization == "linear"


def test_an_affine_image_is_exact_and_composes():
    _first, _second, curves = charts("narrow pipe (sine charts)")
    curve = curves[0]
    rotation = np.eye(4)
    c, s = math.cos(.4), math.sin(.4)
    rotation[:3, :3] = ((c, -s, 0.), (s, c, 0.), (0., 0., 1.))
    rotation[:3, 3] = (1., -2., .5)
    scale = np.diag((2., .5, 1.5, 1.))
    image = curve.transformed(rotation)
    t = np.linspace(0., 1., 9)
    assert np.allclose(image.evaluate(t), curve.evaluate(t) @ rotation[:3, :3].T + rotation[:3, 3], atol=1e-13)
    twice = image.transformed(scale)
    assert np.allclose(twice.evaluate(t), curve.transformed(scale @ rotation).evaluate(t), atol=1e-12)
    assert np.allclose(twice.derivative(.37), (scale @ rotation)[:3, :3] @ curve.derivative(.37), rtol=1e-12)
    # roots and inversion work in the reference frame of the image
    point = twice.evaluate(.4)
    assert any(abs(tau - .4) < 1e-6 for tau in twice.parameters_of(point, tolerance=1e-9))
    lo, hi = twice.bounds()
    assert np.all(twice.evaluate(t) >= lo - 1e-12) and np.all(twice.evaluate(t) <= hi + 1e-12)


@pytest.mark.parametrize("name", CASES)
def test_bounds_enclose_the_curve_tightly_enough_to_prune(name):
    _first, _second, curves = charts(name)
    for curve in curves:
        t = np.linspace(0., 1., 201)
        points = curve.evaluate(t)
        lo, hi = curve.bounds()
        assert np.all(points >= lo) and np.all(points <= hi)
        assert np.all(hi - lo <= (points.max(axis=0) - points.min(axis=0)) * 1.6 + 1e-6)          # tight, not just valid
        whole = points.max(axis=0) - points.min(axis=0)
        for a, b in ((0., .3), (.3, .8), (.8, 1.), (0., .05), (.45, .55), (.95, 1.)):
            lo, hi = curve.bounds(a, b)
            piece = curve.evaluate(np.linspace(a, b, 400))
            assert np.all(piece >= lo) and np.all(piece <= hi)
            # interval arithmetic cannot see the cancellation between the directrix and the ruling, so a slice
            # carries a slack of the first order in its width; a whole-curve box for every slice would fail this
            assert np.all(hi - lo <= (piece.max(axis=0) - piece.min(axis=0)) * 1.6 + (b - a) * whole + 1e-6), (name, a, b)
    with pytest.raises(GeometryError):
        curves[0].bounds(.6, .2)


@pytest.mark.parametrize("name", CASES)
def test_bounds_enclose_the_curve_over_intervals_of_every_width_and_at_the_fold_ends(name):
    """The enclosure feeds certified disjointness proofs, so it must hold down to the tolerance and at the square-root
    ends of a chart, where the difference form takes over."""
    _first, _second, curves = charts(name)
    rng = np.random.default_rng(11)
    for curve in curves:
        for width in (1., .5, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
            for _ in range(6):
                lo = float(rng.uniform(0., 1. - width))
                lower, upper = curve.bounds(lo, lo + width)
                points = curve.evaluate(np.linspace(lo, lo + width, 7))
                assert np.all(points >= lower) and np.all(points <= upper), (name, curve.parameterization, width, lo)
        for width in (1e-3, 1e-6, 1e-9, 1e-12):                      # the two ends, and the degenerate end intervals
            for lo, hi in ((0., width), (1. - width, 1.), (0., 0.), (1., 1.)):
                lower, upper = curve.bounds(lo, hi)
                points = curve.evaluate(np.linspace(lo, hi, 7))
                assert np.all(points >= lower) and np.all(points <= upper), (name, curve.parameterization, lo, hi)


@pytest.mark.parametrize("name", CASES)
def test_roots_on_a_plane_are_exactly_the_sign_changes(name):
    _first, _second, curves = charts(name)
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
    first, second, curves = charts("linear charts across the wall")
    for curve in curves:
        assert curve.roots_on(second) is None
        for tau in (0., .1, .5, .77, 1.):
            found = curve.parameters_of(curve.evaluate(tau), tolerance=1e-9)
            assert any(abs(value - tau) < 1e-6 for value in found), (tau, found)
        assert curve.parameters_of(curve.evaluate(.5) + (0., 0., .01), tolerance=1e-9) == ()     # off the curve
    assert curves[0].parameters_of(curves[1].evaluate(.5), tolerance=1e-9) == ()                 # another branch


def test_a_curve_is_hashable_and_equal_to_its_rebuilt_self():
    _first, _second, curves = charts("narrow pipe (sine charts)")
    curve = curves[0]
    twin = BezierQuadricCurve(curve.first, curve.second, curve.start, curve.sweep, curve.branch, curve.parameterization)
    assert twin == curve and hash(twin) == hash(curve) and len({curve, twin}) == 1
    assert curves[0] != curves[1]


@pytest.mark.parametrize("make,expected", [
    (lambda f, s: BezierQuadricCurve(f, s, 0., .1, 2), "branch"),
    (lambda f, s: BezierQuadricCurve(f, s, 0., .1, 1, "cubic"), "parameterization"),
    (lambda f, s: BezierQuadricCurve(f, s, 0., 0.), "extent"),
    (lambda f, s: BezierQuadricCurve(f, s, 0., math.inf), "finite"),
    (lambda f, s: BezierQuadricCurve(f, s, .5, 1.), "interval"),
    (lambda f, s: BezierQuadricCurve(f, s, 0., .1, 1, "linear", ((1, 0, 0), (0, 1, 0), (0, 0, 1))), "transformation"),
    (lambda f, s: BezierQuadricCurve(wall(), s, 0., .1), "Bezier ruled support"),
])
def test_invalid_definitions_are_refused(make, expected):
    first, second = CASES["linear charts across the wall"]
    with pytest.raises(GeometryError, match=expected):
        make(BezierRuledSupport.from_surface(first), second)


def test_charts_must_split_at_folds_and_square_ends_must_be_folds():
    first, second = CASES["narrow pipe (sine charts)"]
    support = BezierRuledSupport.from_surface(first)
    plan = get_poly_plan(support, second)
    fold = plan.discriminant_roots()[0][0]
    with pytest.raises(GeometryError, match="split"):                          # a fold inside the chart
        BezierQuadricCurve(support, second, fold - .01, .02, 1)
    with pytest.raises(GeometryError, match="regular endpoint"):               # a square end away from any fold
        BezierQuadricCurve(support, second, fold + .1, .1, 1, "left_square")
    with pytest.raises(GeometryError, match="real"):                           # outside the real intersection
        BezierQuadricCurve(support, second, 0., .01, 1)


# ------------------------------------------------------------------------------- folds on the base curve


def base_plane_pipe_charts():
    """A pipe whose axis lies in the plane of the wall's base (``z = 0``): every fold sits on the base curve (``B = 0``)."""
    return list(bezier_quadric_support(wall(), Cylinder((-2., 0., 0.), (1., 0., 0.), (0., 1., 0.), .3, 8.)).curves)


def test_a_fold_on_the_base_curve_is_evaluated_without_a_ratio_of_residues():
    """There ``B``, ``C`` and the radical all vanish together, so ``C / q`` is a quotient of rounding residues (it put
    the end of a chart 8 units below the wall); the direct root decides instead."""
    curves = base_plane_pipe_charts()
    assert curves
    for curve in curves:
        points = curve.evaluate(np.linspace(0., 1., 129))
        assert points[:, 2].min() >= -1e-12 and points[:, 2].max() <= 1.5 + 1e-12          # inside the wall's height
        assert distance(Cylinder((-2., 0., 0.), (1., 0., 0.), (0., 1., 0.), .3, 8.), points).max() < 1e-12
        for end in (0., 1.):
            if curve._is_fold_end(int(end)):
                assert abs(curve.evaluate(end)[2]) < 1e-14                                  # the fold lies on the base curve


@pytest.mark.parametrize("name", ["narrow pipe (sine charts)", "cone (square charts)", "quartic against a tilted pipe"])
def test_a_plane_a_hair_from_a_fold_end_is_still_crossed_and_the_point_is_found_again(name):
    """Near a fold end ``t -> tau`` is a square root: a crossing a few ulp of ``t`` away keeps only digits of ``tau``,
    which used to move the point by 1e-9 (the root was dropped, and a face boundary lost its winding crossing)."""
    _first, _second, curves = charts(name)
    checked = 0
    for curve in curves:
        for end in (0, 1):
            if not curve._is_fold_end(end):
                continue
            origin = curve.evaluate(float(end))
            ruling = np.asarray(curve.plan().floats().direction)
            ruling = ruling / np.linalg.norm(ruling)                      # the curve leaves the fold along the ruling
            inward = curve.derivative(float(end)) * (1. if end == 0 else -1.)
            sign = 1. if float(inward @ ruling) > 0 else -1.
            e1 = np.cross(ruling, (0., 0., 1.) if abs(ruling[2]) < .9 else (1., 0., 0.))
            for height in (1e-12, 1e-10, 1e-8, 1e-6):
                plane = Plane(origin + sign * height * ruling, e1, np.cross(ruling, e1))
                roots = curve.roots_on(plane)
                near = [tau for tau in roots if abs(tau - end) < 1e-2]
                assert near, (name, curve.parameterization, end, height, roots)
                for tau in near:
                    assert abs(float((curve.evaluate(tau) - plane.origin) @ plane.normal)) < 1e-14, (name, end, height, tau)
                assert any(abs(tau - end) > 0. for tau in near)           # a crossing beside the end, not the end itself
                checked += 1
            for tau in (1e-9, 1e-7, 1e-5):
                if float(np.linalg.norm(curve.derivative(float(end)))) * tau < 1e-9:
                    continue                                              # closer to the end than the tolerance can tell
                where = abs(float(end) - tau)
                found = curve.parameters_of(curve.evaluate(where), tolerance=1e-10)
                assert found and abs(found[0] - where) <= 1e-3 * tau + 1e-15, (name, end, tau, found)
    assert checked


def test_the_enclosure_follows_the_constant_term_that_a_fold_radical_implies():
    """A fold chart's radical is that of ``Delta(t) - Delta(anchor)``: the chart lies on the quadric whose constant
    term is ``C + Delta(anchor) / (4 A)``. With an anchor off the fold by a rounding residue that is invisible; with
    one 2e-9 off (a synthetic chart, which the support builder never makes) the two root formulas of the enclosure
    disagree by 1e-8 unless the second one uses the implied constant."""
    _first, _second, curves = charts("rulings nearly along the axis (tiny A)")
    folded = [c for c in curves if c.parameterization in ("left_square", "right_square")]
    assert folded
    for curve in folded:
        shift = 2e-9 * curve.sweep / abs(curve.sweep)
        if curve.parameterization == "left_square":
            moved = BezierQuadricCurve._make(curve.first, curve.second, curve.start + shift, curve.sweep - shift, curve.branch,
                                             curve.parameterization)
        else:
            moved = BezierQuadricCurve._make(curve.first, curve.second, curve.start, curve.sweep - shift, curve.branch,
                                             curve.parameterization)
        for lo, hi in ((0., 1.), (0., 1e-3), (.4, .6), (1. - 1e-3, 1.), (1e-6, 1e-5)):
            lower, upper = moved.bounds(lo, hi)
            points = moved.evaluate(np.linspace(lo, hi, 33))
            assert np.all(points >= lower) and np.all(points <= upper), (curve.parameterization, lo, hi)


@pytest.mark.parametrize("name", CASES)
def test_an_end_point_is_found_again_as_exactly_that_end(name):
    """Through the square root of a fold end ``t -> tau`` keeps a few digits, but an end is reported as an end."""
    _first, _second, curves = charts(name)
    for curve in curves:
        for end in (0., 1.):
            assert curve.parameters_of(curve.evaluate(end), tolerance=1e-10) == (end,), (name, curve.parameterization, end)


def test_the_answers_to_a_point_inversion_depend_on_the_tolerance_asked():
    """Answers are remembered per curve; a looser tolerance must not leak into a tighter question."""
    _first, _second, curves = charts("linear charts across the wall")
    curve = curves[0]
    point = curve.evaluate(.5) + np.array([3e-9, 0., 0.])                                  # a point 3e-9 off the curve
    assert curve.parameters_of(point, tolerance=1e-7) != ()
    assert curve.parameters_of(point, tolerance=1e-10) == ()
    assert curve.parameters_of(point, tolerance=1e-7) != ()                                # and back
