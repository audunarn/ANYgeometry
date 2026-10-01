"""PROTOTYPE evidence: exact branches of a ruled quadric (Cylinder or Cone) with another quadric.

These tests keep the claims made for ``anygeometry._quadric_branch`` reproducible:

* the trigonometric algebra and the quadratic-in-``s`` identity are exact;
* a Cone is parametrized exactly like the kernel's ``Cone.evaluate``;
* for cylinder/cylinder the generalized branch reproduces the production
  ``cylinder_cylinder_support`` charts (same intervals, same branches, same points);
* for cone/cylinder 10 degrees off perpendicular, and a spread of other cone,
  cylinder and plane pairings, every contour point of an independent grid oracle
  is covered by an exact chart and every chart lies on both surfaces;
* exact branch/quadric elimination finds every sign change along a chart;
* unsupported contact (apex on the other support, higher-order contact, rulings
  parallel to a plane) is refused with a typed error rather than approximated.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from anygeometry import GeometryError
from anygeometry import _quadric_branch as qb
from anygeometry.analytic_supports import cylinder_cylinder_support
from anygeometry.generators import cylinder
from anygeometry.surfaces import Cone, Cylinder, Plane


def _unit(vector):
    return vector / np.linalg.norm(vector)


def _cone_at(off_degrees, r0=.5, r1=1., height=5., origin=(0., 0., 0.)):
    a = math.radians(off_degrees)
    return Cone(origin, (math.cos(a), 0., math.sin(a)), (0., 1., 0.), r0, r1, height, 0., math.tau)


def _tall_cylinder():
    return Cylinder((0., 0., -3.), (0., 0., 1.), (1., 0., 0.), 2., 6., 0., math.tau)


# ---------------------------------------------------------------- algebra


def test_trigonometric_product_and_tangent_half_polynomial_are_exact():
    rng = np.random.default_rng(3)
    for _ in range(60):
        p = tuple(rng.normal(size=2 * int(rng.integers(0, 3)) + 1))
        q = tuple(rng.normal(size=2 * int(rng.integers(0, 3)) + 1))
        t = rng.uniform(-7, 7, 4)
        want = qb.tp_eval(p, t) * qb.tp_eval(q, t)
        assert np.allclose(qb.tp_eval(qb.tp_mul(p, q), t), want, atol=1e-12)
        exact = qb.tp_mul(tuple(Fraction(float(v)) for v in p), tuple(Fraction(float(v)) for v in q))
        assert np.allclose(qb.tp_eval(exact, t), want, atol=1e-12)
    for n in (1, 2, 3, 4):
        p = tuple(rng.normal(size=2 * n + 1))
        poly = qb._tangent_half_polynomial(tuple(Fraction(float(v)) for v in p))
        for t in rng.uniform(-3, 3, 4):
            x = math.tan(t / 2)
            assert sum(float(c) * x ** i for i, c in enumerate(poly)) == pytest.approx(
                float(qb.tp_eval(p, t)) * (1 + x * x) ** n, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("first,second", [
    (_cone_at(10.), _tall_cylinder()),
    (_tall_cylinder(), _cone_at(10.)),
    (_cone_at(10.), Plane((.2, 0., 0.), (0., 1., 0.), (0., 0., 1.))),
    (_cone_at(33., 1., .4, 3.), _cone_at(-20., .3, 1.4, 2., (.4, .1, 0.))),
])
def test_ruled_equation_coefficients_reproduce_the_implicit_value(first, second):
    ruled, quadric = qb.RuledSupport.from_surface(first), qb.Quadric.from_surface(second)
    A, B, C = qb.coefficients(ruled, quadric, exact=False)
    rng = np.random.default_rng(5)
    for _ in range(40):
        t, s = rng.uniform(-4, 4), rng.uniform(-3, 3)
        direct = float(quadric.value(ruled.point(t, s)))
        via = float(qb.tp_eval(A, t) * s * s + qb.tp_eval(B, t) * s + qb.tp_eval(C, t))
        assert via == pytest.approx(direct, rel=1e-11, abs=1e-11)


def test_cone_ruled_parametrization_is_the_kernel_cone():
    cone = _cone_at(10.)
    ruled = qb.RuledSupport.from_surface(cone)
    apex, axis, _k = qb.cone_apex_axis_slope(cone)
    rng = np.random.default_rng(1)
    for _ in range(20):
        u, v = rng.uniform(0, 1, 2)
        s = v * cone.height - float(np.dot(apex - cone.origin, axis))
        assert np.allclose(ruled.point(cone.start_angle + u * cone.sweep_angle, s), cone.evaluate(u, v), atol=1e-13)


def test_root_multiplicity_is_exact():
    # Coefficients must be exactly representable for a true repeated root: these are.
    sin_t, one_minus_cos, one_plus_cos = (0., 0., 1.), (.5, -.5, 0.), (.5, .5, 0.)

    def roots(p):
        return {round(a, 6): m for a, m in qb.tp_roots_multiplicity(p, start=-1., sweep=5., tolerance=1e-12)}

    assert roots(qb.tp_mul(one_minus_cos, one_plus_cos)) == {0.0: 2, round(math.pi, 6): 2}    # sin^2(t)/4
    assert roots(qb.tp_mul(sin_t, one_minus_cos)) == {0.0: 3, round(math.pi, 6): 1}            # a triple root
    assert roots(sin_t) == {0.0: 1, round(math.pi, 6): 1}


# ---------------------------------------------------------------- the cone x cylinder case


@pytest.mark.parametrize("off", (0., 10.))
def test_cone_hitting_cylinder_off_perpendicular_is_one_exact_closed_loop(off):
    cone, cyl = _cone_at(off), _tall_cylinder()
    charts, info = qb.support_intersection(cone, cyl)
    assert not info["folds"] and not info["poles"] and not info["points"]   # generic: no folds, no poles
    assert {chart.branch for chart in charts} == {1}
    # chart intervals tile the full turn, each starting where the previous one ended
    spans = sorted((min(c.start_angle, c.start_angle + c.sweep_angle), max(c.start_angle, c.start_angle + c.sweep_angle))
                   for c in charts)
    assert spans[0][0] == pytest.approx(0.) and spans[-1][1] == pytest.approx(math.tau)
    for (_a, end), (start, _b) in zip(spans, spans[1:]):
        assert end == pytest.approx(start, abs=1e-12)
    for chart in charts:
        points = chart.evaluate(np.linspace(0, 1, 33))
        for support in (cone, cyl):
            quadric = qb.Quadric.from_surface(support)
            distance = np.abs(quadric.value(points)) / quadric.gradient_norm(points)
            assert distance.max() < 1e-13
    # the loop is closed: consecutive charts join at identical points
    ends = [(c.evaluate(0.), c.evaluate(1.)) for c in sorted(charts, key=lambda c: c.start_angle)]
    for (_s0, e0), (s1, _e1) in zip(ends, ends[1:]):
        assert np.linalg.norm(e0 - s1) < 1e-13
    assert np.linalg.norm(ends[-1][1] - ends[0][0]) < 1e-13


def test_cone_first_and_cylinder_first_describe_the_same_curve():
    cone, cyl = _cone_at(10.), _tall_cylinder()
    cone_first, _ = qb.support_intersection(cone, cyl)
    cyl_first, _ = qb.support_intersection(cyl, cone)
    assert len(cone_first) == 4 and len(cyl_first) == 8     # the loop wraps the cone; it folds twice on the cylinder
    cloud = np.vstack([c.evaluate(np.linspace(0, 1, 400)) for c in cone_first])
    for chart in cyl_first:
        for point in chart.evaluate(np.linspace(0, 1, 25)):
            assert np.min(np.linalg.norm(cloud - point, axis=1)) < 2e-2    # sampled-cloud distance only


# ---------------------------------------------------------------- parity with production


@pytest.mark.parametrize("spec", [
    dict(radius=1., origin=(0., -1.5, 0.), axis=(0., 1., 0.), radial_direction=(1., 0., 0.)),     # equal radii
    dict(radius=.8, origin=(-.5, -1., -.5), axis=(1., 2., 1.), radial_direction=(1., 0., 0.)),     # skew
    dict(radius=.75, origin=(0., 0., 0.), axis=(math.cos(math.radians(10)), 0., math.sin(math.radians(10))),
         radial_direction=(0., 1., 0.)),                                                           # 10 degrees off
])
def test_generalized_branch_reproduces_the_production_cylinder_charts(spec):
    base = cylinder(1., 3., origin=(0., 0., -1.5), circumferential_segments=8)
    other = cylinder(height=3., circumferential_segments=8, **spec)

    def key(curve):
        lo, hi = sorted((curve.start_angle, curve.start_angle + curve.sweep_angle))
        return (round(lo, 9), round(hi, 9), curve.branch)

    def invert(curve, angle, left, right):
        w = min(1., max(0., (angle - curve.start_angle) / curve.sweep_angle))
        if left and right:
            return 2 / math.pi * math.asin(math.sqrt(w)) if w <= .5 else 1 - 2 / math.pi * math.asin(math.sqrt(1 - w))
        return math.sqrt(w) if left else 1 - math.sqrt(1 - w) if right else w

    worst = 0.
    for a in (f.surface for f in base.faces.values()):
        for b in (f.surface for f in other.faces.values()):
            old = {key(c): c for c in cylinder_cylinder_support(a, b).curves}
            new, _ = qb.support_intersection(a, b, mid_split=True)
            new = {key(c): c for c in new}
            assert set(old) == set(new)
            for k, curve in old.items():
                mode = curve.parameterization
                for w in np.linspace(.03, .97, 7):
                    angle = k[0] + w * (k[1] - k[0])
                    t_old = invert(curve, angle, mode in ("left_square", "both_sine"), mode in ("right_square", "both_sine"))
                    t_new = invert(new[k], angle, new[k].left_fold, new[k].right_fold)
                    worst = max(worst, float(np.linalg.norm(
                        curve.evaluate(np.array([t_old]))[0] - new[k].evaluate(np.array([t_new]))[0])))
    assert worst < 1e-12


# ---------------------------------------------------------------- independent coverage oracle


def _coverage(first_surface, second_surface, n_angle=541, n_s=271):
    """Contour the second equation on a grid of the first chart; compare with exact charts."""
    charts, info = qb.support_intersection(first_surface, second_surface)
    first = qb.RuledSupport.from_surface(first_surface)
    second = qb.Quadric.from_surface(second_surface)
    angles = first.start_angle + first.sweep_angle * np.linspace(0, 1, n_angle)
    s_values = np.linspace(first.s_range[0], first.s_range[1], n_s)
    field = second.value(first.point(angles[:, None], s_values[None, :]))
    sign = np.sign(field)
    contour = []
    i, j = np.nonzero(sign[:, :-1] * sign[:, 1:] < 0)
    contour.extend(zip(angles[i], s_values[j] + field[i, j] / (field[i, j] - field[i, j + 1]) * (s_values[j + 1] - s_values[j])))
    i, j = np.nonzero(sign[:-1, :] * sign[1:, :] < 0)
    contour.extend(zip(angles[i] + field[i, j] / (field[i, j] - field[i + 1, j]) * (angles[i + 1] - angles[i]), s_values[j]))
    h = max(abs(first.sweep_angle) / (n_angle - 1), (first.s_range[1] - first.s_range[0]) / (n_s - 1))
    near_events = np.array([*info["folds"], *info["poles"], *info["nodes"], *info["events"]])
    uncovered = matched = 0
    for angle, s in contour:
        point = first.point(angle, s)
        if not (qb._inside(first_surface, point, 1e-9) and qb._inside(second_surface, point, 1e-9)):
            continue
        best = np.inf
        for chart in charts:
            lo, hi = sorted((chart.start_angle, chart.start_angle + chart.sweep_angle))
            if lo - 1e-12 <= angle <= hi + 1e-12:
                t = qb.parameter_for_angle(chart, angle)
                best = min(best, float(np.linalg.norm(chart.evaluate(np.array([t]))[0] - point)))
        if best < np.inf:
            matched += 1
        elif not len(near_events) or np.min(np.abs(((near_events - angle + np.pi) % math.tau) - np.pi)) > 4.5 * h:
            uncovered += 1
    worst = 0.
    for chart in charts:
        points = chart.evaluate(np.linspace(0, 1, 41))
        for support in (first_surface, second_surface):
            quadric = qb.Quadric.from_surface(support)
            gradient = quadric.gradient_norm(points)
            good = gradient > 1e-6
            if good.any():
                worst = max(worst, float((np.abs(quadric.value(points))[good] / gradient[good]).max()))
    return uncovered, matched, worst


def _random_surface(rng, kind):
    axis = _unit(rng.normal(size=3))
    helper = _unit(np.cross(axis, rng.normal(size=3)))
    sweep = math.tau if rng.random() < .5 else rng.uniform(.8, 5.5)
    start = rng.uniform(0, math.tau)
    origin = rng.uniform(-1, 1, 3)
    if kind == "cyl":
        return Cylinder(origin, axis, helper, rng.uniform(.5, 2.5), rng.choice([-1, 1]) * rng.uniform(2., 7.), start, sweep)
    if kind == "cone":
        r0, r1 = rng.uniform(.2, 1.5), rng.uniform(.2, 2.5)
        r1 = r1 if abs(r1 - r0) > .1 else r0 + .6
        return Cone(origin, axis, helper, r0, r1, rng.choice([-1, 1]) * rng.uniform(1.5, 6.), start, sweep)
    u = _unit(np.cross(axis, helper)) * rng.uniform(3, 8)
    v = helper * rng.uniform(3, 8)
    return Plane(origin - .5 * u - .5 * v, u, v)


@pytest.mark.parametrize("off", (0., 10.))
@pytest.mark.parametrize("order", ("cone-first", "cylinder-first"))
def test_cone_cylinder_charts_cover_an_independent_contour(off, order):
    cone, cyl = _cone_at(off), _tall_cylinder()
    first, second = (cone, cyl) if order == "cone-first" else (cyl, cone)
    uncovered, matched, worst = _coverage(first, second)
    assert matched > 150 and uncovered == 0 and worst < 1e-12


@pytest.mark.parametrize("pairing", [("cone", "cyl"), ("cyl", "cone"), ("cone", "cone"), ("cone", "plane"), ("cyl", "cyl")])
def test_random_pairings_are_covered_complete_and_on_both_surfaces(pairing):
    rng = np.random.default_rng(100 + sum(map(ord, "".join(pairing))))
    total_matched = 0
    for _ in range(14):
        first, second = (_random_surface(rng, kind) for kind in pairing)
        try:
            uncovered, matched, worst = _coverage(first, second, 361, 181)
        except GeometryError:
            continue                      # a typed refusal of a degenerate pairing is acceptable
        assert uncovered == 0 and worst < 1e-10
        total_matched += matched
    assert total_matched > 500


# ---------------------------------------------------------------- elimination against another quadric


def test_branch_roots_match_sign_changes_along_the_chart():
    rng = np.random.default_rng(8)
    checked = 0
    for kinds in (("cone", "cyl"), ("cyl", "cone"), ("cone", "cone"), ("cone", "plane")):
        for _ in range(4):
            first, second = (_random_surface(rng, kind) for kind in kinds)
            try:
                charts, _ = qb.support_intersection(first, second)
            except GeometryError:
                continue
            for chart in charts[:3]:
                other = qb.Quadric.from_surface(_random_surface(rng, ("cyl", "cone", "plane")[int(rng.integers(0, 3))]))
                exact = qb.branch_roots(chart, other)
                if exact is None:
                    continue
                t = np.linspace(0, 1, 6001)
                x = chart.evaluate(t)
                f = other.value(x) / np.maximum(other.gradient_norm(x), 1e-12)
                numeric = t[:-1][np.sign(f[:-1]) * np.sign(f[1:]) < 0]
                for root in numeric:
                    assert any(abs(root - e) <= 2e-3 for e in exact), (root, exact)
                for e in exact:
                    x_e = chart.evaluate(e)
                    assert abs(float(other.value(x_e))) / max(float(other.gradient_norm(x_e)), 1e-10) <= 1e-10
                checked += 1
    assert checked >= 8


# ---------------------------------------------------------------- fail closed


def test_apex_on_the_other_support_is_refused_not_guessed():
    cyl = _tall_cylinder()
    apex_on_wall = Cone((2., 0., 0.), (-1., 0., 0.), (0., 1., 0.), 0., 1.2, 3., 0., math.tau)
    with pytest.raises(GeometryError, match="explicit decision"):
        qb.support_intersection(apex_on_wall, cyl)


def test_rulings_parallel_to_the_plane_are_refused():
    plane = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    parallel = Cylinder((0., 0., 1.), (1., 0., 0.), (0., 1., 0.), 1., 3., 0., math.tau)
    with pytest.raises(GeometryError, match="parallel to the plane"):
        qb.support_intersection(parallel, plane)
