from decimal import Decimal, localcontext
import math

import numpy as np
import pytest

from anygeometry import Cylinder, Plane
from anygeometry.analytic_supports import cylinder_cylinder_support, plane_cylinder_support


def _residual(point, cylinder):
    # Independent high precision evaluation of the implicit support equation.
    with localcontext() as context:
        context.prec = 60
        offset = [Decimal(float(x))-Decimal(float(o)) for x, o in zip(point, cylinder.origin)]
        axis = [Decimal(float(x)) for x in cylinder.axis]
        axial = sum(x*a for x, a in zip(offset, axis))
        radial = [x-axial*a for x, a in zip(offset, axis)]
        return float(abs(sum(x*x for x in radial)-Decimal(float(cylinder.radius))**2))


def test_oblique_plane_cylinder_ellipse_and_finite_height_clipping():
    cylinder = Cylinder((0, 0, -2), (0, 0, 1), (1, 0, 0), 1, 4)
    plane = Plane((0, 0, 0), (1, 0, 1), (0, 1, 0))
    result = plane_cylinder_support(plane, cylinder)
    assert len(result.curves) == 2
    for curve in result.curves:
        for point in curve.evaluate(np.linspace(0, 1, 31)):
            assert _residual(point, cylinder) < 2e-15
            assert point[0] == pytest.approx(point[2], abs=1e-15)
    short = Cylinder((0, 0, -.25), (0, 0, 1), (1, 0, 0), 1, .5)
    clipped = plane_cylinder_support(plane, short)
    assert len(clipped.curves) == 2
    # Analytic domain measure: two intervals around +/-pi/2.
    assert sum(abs(curve.sweep_angle) for curve in clipped.curves) == pytest.approx(4*math.asin(.25), abs=1e-10)


def test_parallel_planes_give_two_generators_or_one_tangent():
    cylinder = Cylinder((0, 0, -2), (0, 0, 1), (1, 0, 0), 1, 4)
    through = Plane((0, 0, 0), (0, 1, 0), (0, 0, 1))
    assert len(plane_cylinder_support(through, cylinder).segments) == 2
    tangent = Plane((1, 0, 0), (0, 1, 0), (0, 0, 1))
    assert len(plane_cylinder_support(tangent, cylinder).segments) == 1
    outside = Plane((2, 0, 0), (0, 1, 0), (0, 0, 1))
    assert plane_cylinder_support(outside, cylinder).segments == ()


@pytest.mark.parametrize("axis,radius", [((0, 1, 0), 1.), ((1, 2, 1), .75), ((0, 1, 0), 2.)])
def test_nonparallel_branch_coverage_and_both_support_equations(axis, radius):
    unit = np.asarray(axis)/np.linalg.norm(axis)
    first = Cylinder((0, 0, -3), (0, 0, 1), (1, 0, 0), 2, 6)
    second = Cylinder(-3*unit, unit, (1, 0, 0), radius, 6)
    result = cylinder_cylinder_support(first, second)
    assert result.curves
    for curve in result.curves:
        for point in curve.evaluate(np.linspace(0, 1, 31)):
            assert _residual(point, first) < 2e-13
            assert _residual(point, second) < 2e-13
            assert -.00000001 <= first.local_uv(point)[1] <= 1.00000001
            assert -.00000001 <= second.local_uv(point)[1] <= 1.00000001
    # Independent analytic root enumeration at angles away from transitions:
    # each allowed quadratic root belongs to exactly one returned chart.
    for angle in (.21, .91, 1.51, 2.11, 3.21, 4.11, 5.01, 5.71):
        base = np.array((2*math.cos(angle), 2*math.sin(angle), -3.))
        projection = np.eye(3)-np.outer(unit, unit)
        q, w = projection @ (base-second.origin), projection @ first.axis
        a, b, c = w @ w, 2*q @ w, q @ q-radius**2
        discriminant = b*b-4*a*c
        expected = 0
        if discriminant >= 0:
            for z in ((-b-math.sqrt(discriminant))/(2*a), (-b+math.sqrt(discriminant))/(2*a)):
                point = base+z*first.axis
                if 0 <= z <= first.height and 0 <= second.local_uv(point)[1] <= 1:
                    expected += 1
        actual = sum(min(curve.start_angle, curve.start_angle+curve.sweep_angle) < angle <
                     max(curve.start_angle, curve.start_angle+curve.sweep_angle) for curve in result.curves)
        assert actual == expected


def test_parallel_cylinders_coincident_tangent_and_finite_height():
    first = Cylinder((0, 0, -2), (0, 0, 1), (1, 0, 0), 1, 4)
    second = Cylinder((1, 0, -1), (0, 0, 1), (1, 0, 0), 1, 2)
    result = cylinder_cylinder_support(first, second)
    assert len(result.segments) == 2
    for start, end in result.segments:
        assert start[0] == pytest.approx(.5)
        assert abs(start[1]) == pytest.approx(math.sqrt(3)/2)
        assert (start[2], end[2]) == pytest.approx((-1, 1))
    tangent = Cylinder((2, 0, -1), (0, 0, 1), (1, 0, 0), 1, 2)
    assert len(cylinder_cylinder_support(first, tangent).segments) == 1
    assert cylinder_cylinder_support(first, first).coincident
