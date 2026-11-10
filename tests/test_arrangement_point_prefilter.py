"""The points-by-paths prefilter of ``arrange_material`` (``_point_clear_of_curve``) is a pure shortcut.

A point it reports as far certainly has no ``point_parameters`` on the curve; points within, at and just
beyond the tolerance and at the widened enclosure edge are checked against the unfiltered route, and an
arrangement with branch-curve traces and points is identical with and without the filter. The filter is a
pure shortcut, so these tests pin the current results; they cannot fail on the code before it.
"""
import math

import numpy as np
import pytest

import anygeometry.material_arrangement as ma
from anygeometry import Cylinder, Plane
from anygeometry.analytic_supports import cylinder_cylinder_support, plane_cylinder_support
from anygeometry.arrangement_geometry import LinePath, point_parameters
from anygeometry.material_arrangement import ArrangementPath, ArrangementPoint, MaterialDomain, arrange_material

TOL = 1e-9


def _support_curves():
    cylinder = Cylinder((0, 0, -2), (0, 0, 1), (1, 0, 0), 1, 4)
    crossing = Cylinder((-2, 0, 0), (1, 0, 0), (0, 1, 0), 1, 4)
    tilted = Plane((0, 0, 0), (1e-16, 0, 1), (0, 1, 0))
    slanted = Plane((0, 0, 0), (.3, 0, 1), (0, 1, 0))
    return [*plane_cylinder_support(tilted, cylinder).curves, *plane_cylinder_support(slanted, cylinder).curves,
            *cylinder_cylinder_support(cylinder, crossing).curves]


def _branch_curves():
    curves = [c for c in _support_curves() if type(c) is ma.BezierQuadricCurve and c._identity]
    assert curves
    return curves


def test_far_points_never_have_parameters():
    rng = np.random.default_rng(20261110)
    skipped = 0
    for curve in _branch_curves():
        for tol in (TOL, 1e-6):
            far = ma._point_clear_of_curve(curve, tol)
            assert far is not None
            lo, hi = (np.asarray(v, dtype=float) for v in curve.bounds())
            margin = 2 * tol + 1e-6 * max(1.0, float(np.max(np.abs(np.concatenate((lo, hi))))))
            points = []
            for t in np.linspace(0., 1., 11):
                base = np.asarray(curve.evaluate(float(t)))
                for scale in (0., .5, 1., 1. + 1e-9, 2., 3., 10., 1e3, 1e6, 1e9):
                    direction = rng.normal(size=3)
                    points.append(base + scale * tol * direction / np.linalg.norm(direction))
            for axis in range(3):                     # just inside / just outside the widened enclosure
                for edge, sign in ((lo, -1), (hi, 1)):
                    for factor in (1 - 1e-3, 1., 1 + 1e-3, 2.):
                        point = .5 * (lo + hi)
                        point[axis] = edge[axis] + sign * factor * margin
                        points.append(point)
            points.extend(rng.normal(size=(200, 3)) * 3)
            for point in points:
                if far(tuple(point)):
                    skipped += 1
                    assert point_parameters(curve, point, tolerance=tol) == ()
            for t in np.linspace(0., 1., 11):            # points on or within tol of the curve are never far
                base = np.asarray(curve.evaluate(float(t)))
                assert not far(tuple(base))
                assert not far(tuple(base + .5 * tol * np.array((1., 0., 0.))))
    assert skipped > 100


def test_filter_declines_other_curves_transforms_and_odd_points():
    curves = _support_curves()
    for curve in curves:
        if type(curve) is not ma.BezierQuadricCurve:
            assert ma._point_clear_of_curve(curve, TOL) is None
    assert ma._point_clear_of_curve(LinePath((0., 0., 0.), (1., 0., 0.)), TOL) is None
    curve = _branch_curves()[0]
    moved = curve.transformed(((2., 0., 0., 1.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.)))
    assert not moved._identity and ma._point_clear_of_curve(moved, TOL) is None
    for tol in (0., -1., math.inf, math.nan):
        assert ma._point_clear_of_curve(curve, tol) is None
    far = ma._point_clear_of_curve(curve, TOL)
    assert far((1e9, 0., 0.))
    assert not far((math.nan, math.nan, math.nan))
    assert not far((1e9, 0.))                          # not a 3-point: unfiltered route


def _plate(points_factory):
    plane = Plane((0, 0, 0), (1e-16, 0, 1), (0, 1, 0))
    cylinder = Cylinder((0, 0, -2), (0, 0, 1), (1, 0, 0), 1, 4)
    curves = plane_cylinder_support(plane, cylinder).curves
    corners = [tuple(plane.evaluate(u, v)) for u, v in ((-1.5, -1.5), (1.5, -1.5), (1.5, 1.5), (-1.5, 1.5))]
    loop = tuple(ArrangementPath(LinePath(corners[k], corners[(k + 1) % 4])) for k in range(4))
    return MaterialDomain(7, plane, (loop,)), [ArrangementPath(c) for c in curves], points_factory(plane, curves)


@pytest.mark.parametrize("order", ["near_first", "far_first"])
def test_arrangement_is_identical_with_and_without_the_filter(monkeypatch, order):
    def points(plane, curves):
        values = [tuple(curves[0].evaluate(.4)), tuple(curves[1].evaluate(.6)), tuple(plane.evaluate(1., 0.)),
                  tuple(plane.evaluate(-1., .3))]
        return [ArrangementPoint(p) for p in (values if order == "near_first" else values[::-1])]
    domain, traces, arrangement_points = _plate(points)
    seen = {"skipped": 0}
    original = ma._point_clear_of_curve

    def spy(curve, tol):
        far = original(curve, tol)
        if far is None:
            return None

        def counted(position):
            result = far(position)
            seen["skipped"] += result
            return result
        return counted
    monkeypatch.setattr(ma, "_point_clear_of_curve", spy)
    filtered = arrange_material(domain, traces, points=arrangement_points, tolerance=TOL)
    monkeypatch.setattr(ma, "_point_clear_of_curve", lambda curve, tol: None)
    unfiltered = arrange_material(domain, traces, points=arrangement_points, tolerance=TOL)
    np.set_printoptions(floatmode="unique")
    assert repr(filtered) == repr(unfiltered)
    assert seen["skipped"] > 0
