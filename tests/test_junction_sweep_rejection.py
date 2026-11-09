"""Default-on piecewise line/branch rejection in ``curve_junctions`` is a pure shortcut.

Every result (and every refusal) equals the unfiltered route's, in both operand orders,
including lines that pass within, at, and just beyond the tolerance window.
"""
import numpy as np
import pytest

import anygeometry.arrangement_geometry as ag
from anygeometry import Cylinder, GeometryError, Plane
from anygeometry.analytic_supports import cylinder_cylinder_support, plane_cylinder_support
from anygeometry.arrangement_geometry import LinePath, _junction_bounds_policy, curve_junctions


def _curves():
    cylinder = Cylinder((0, 0, -2), (0, 0, 1), (1, 0, 0), 1, 4)
    crossing = Cylinder((-2, 0, 0), (1, 0, 0), (0, 1, 0), 1, 4)
    tilted = Plane((0, 0, 0), (1e-16, 0, 1), (0, 1, 0))
    slanted = Plane((0, 0, 0), (.3, 0, 1), (0, 1, 0))
    out = [*plane_cylinder_support(tilted, cylinder).curves, *plane_cylinder_support(slanted, cylinder).curves,
           *cylinder_cylinder_support(cylinder, crossing).curves]
    assert {type(c) for c in out} & {ag.BezierQuadricCurve, ag.CylinderIntersectionCurve}
    return out


def reference(first, second, tol):
    with _junction_bounds_policy(False):
        try:
            return curve_junctions(first, second, tolerance=tol)
        except GeometryError as error:
            return ("refused", str(error))


def actual(first, second, tol):
    try:
        return curve_junctions(first, second, tolerance=tol)
    except GeometryError as error:
        return ("refused", str(error))


def test_random_and_near_touching_lines_match_unfiltered_route(monkeypatch):
    rng = np.random.default_rng(20261109)
    calls = {"clear": 0, "total": 0}
    original = ag._line_clear_of_branch

    def spy(*args):
        result = original(*args)
        calls["total"] += 1
        calls["clear"] += bool(result)
        return result

    monkeypatch.setattr(ag, "_line_clear_of_branch", spy)
    curves = _curves()
    for curve in curves:
        points = [np.asarray(curve.evaluate(t)) for t in np.linspace(0., 1., 7)]
        for tol in (1e-9, 1e-6):
            lines = []
            for _ in range(60):
                lines.append(LinePath(tuple(rng.normal(size=3) * 2), tuple(rng.normal(size=3) * 2)))
            for point in points:
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                for offset in (0., .5 * tol, tol, 1.5 * tol, 4 * tol, 1e3 * tol):
                    centre = point + offset * np.cross(direction, [0., 0., 1.])
                    lines.append(LinePath(tuple(centre - .7 * direction), tuple(centre + .7 * direction)))
                    # segment ending near the point: the parameter-extension window
                    lines.append(LinePath(tuple(point - 1.3 * direction), tuple(point - offset * direction)))
                    lines.append(LinePath(tuple(point + 1e-3 * direction), tuple(point + 2e-3 * direction)))
            for line in lines:
                for pair in ((line, curve), (curve, line)):
                    assert actual(*pair, tol) == reference(*pair, tol)
    assert calls["clear"] > 100 and calls["clear"] < calls["total"]


@pytest.mark.parametrize("reverse", (False, True))
def test_degenerate_line_refusal_is_unchanged(reverse):
    curve = _curves()[0]
    pair = [LinePath((5., 5., 5.), (5., 5., 5.)), curve]
    if reverse:
        pair.reverse()
    assert reference(*pair, 1e-9)[0] == "refused" == actual(*pair, 1e-9)[0]


def test_budgeted_route_is_not_filtered(monkeypatch):
    curve = _curves()[0]
    line = LinePath((50., 50., 50.), (51., 50., 50.))
    assert ag._line_clear_of_branch(line, curve, 1e-9)
    with _junction_bounds_policy(False):
        assert not ag._line_clear_of_branch(line, curve, 1e-9)


def test_curve_pairs_and_ellipse_lines_match_unfiltered_route(monkeypatch):
    from anygeometry.exact_curves import EllipticArc
    rng = np.random.default_rng(1110)
    calls = {"clear": 0, "total": 0}
    original = ag._curve_pair_clear

    def spy(*args):
        result = original(*args)
        calls["total"] += 1
        calls["clear"] += bool(result)
        return result

    monkeypatch.setattr(ag, "_curve_pair_clear", spy)
    branches = _curves()
    arcs = []
    for _ in range(8):
        centre = rng.normal(size=3) * .8
        u, v = rng.normal(size=3) * .6, rng.normal(size=3) * .6
        arcs.append(EllipticArc(tuple(centre), tuple(u), tuple(v), float(rng.uniform(-3, 3)), float(rng.uniform(.3, 5.))))
    for branch in branches[:4]:             # arcs through points of the branch curves, offset by multiples of tol
        for t in (0., .3, .7, 1.):
            point = np.asarray(branch.evaluate(t))
            for offset in (0., 1e-9, 2e-9, 1e-7):
                arcs.append(EllipticArc(tuple(point + offset), (.5, 0., 0.), (0., .5, 0.), 0., 2.))
    lines = [LinePath(tuple(rng.normal(size=3) * 2), tuple(rng.normal(size=3) * 2)) for _ in range(20)]
    for arc in arcs[8:12]:
        p = np.asarray(arc.evaluate(0.))
        lines.append(LinePath(tuple(p - (.5, 0., 0.)), tuple(p + (0., 0., 1e-9))))
    groups = [(arc, other) for arc in arcs for other in (*arcs[:10], *branches, *lines)]
    for tol in (1e-9, 1e-6):
        for first, second in groups:
            for pair in ((first, second), (second, first)):
                assert actual(*pair, tol) == reference(*pair, tol)
    for first, second in [(a, b) for a in branches for b in branches]:
        assert actual(first, second, 1e-9) == reference(first, second, 1e-9)
    assert calls["clear"] > 50 and calls["clear"] < calls["total"]


def test_degenerate_line_against_ellipse_refusal_is_unchanged():
    from anygeometry.exact_curves import EllipticArc
    good = EllipticArc((50., 0., 0.), (1., 0., 0.), (0., 1., 0.), 0., 2.)
    point = LinePath((9., 9., 9.), (9., 9., 9.))
    for pair in ((point, good), (good, point)):
        assert actual(*pair, 1e-9) == reference(*pair, 1e-9)
