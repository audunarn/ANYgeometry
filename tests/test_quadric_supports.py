"""Support-level intersections of a Cone with a Plane, Cylinder or Cone.

``cone_support`` is the Cone counterpart of ``plane_cylinder_support`` and ``cylinder_cylinder_support``:
exact curves inside both native rectangles. These tests compare it with an independent contour oracle,
pin the conic classification of a plane section (circle, ellipse, parabola, hyperbola, generators through
the apex), the coaxial and coincident relations, and the cost property the engine depends on: the exact
solve is shared by every facet of a shell pair.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from anygeometry import GeometryError
from anygeometry.exact_curves import EllipticArc
from anygeometry.generators import cone as cone_shell, cylinder as cylinder_shell
from anygeometry.quadric_algebra import QuadricSupport, RuledSupport, _cached_plan, get_plan
from anygeometry.quadric_curves import QuadricIntersectionCurve
from anygeometry.quadric_supports import cone_support
from anygeometry.surfaces import Cone, Cylinder, Plane


def _unit(vector):
    return vector / np.linalg.norm(vector)


def _distance(support, points):
    quadric = QuadricSupport.from_surface(support)
    gradient = np.maximum(quadric.gradient_norm(points), 1e-300)
    return np.abs(quadric.value(points)) / gradient, gradient


def _patch_contains(surface, point, tolerance=1e-9):
    if isinstance(surface, Plane):
        return True
    u, v = surface.local_uv(point)
    radius = max(surface.radius_start, surface.radius_end) if isinstance(surface, Cone) else surface.radius
    angular = tolerance / max(radius * abs(surface.sweep_angle), tolerance)
    axial = tolerance / max(abs(surface.height), tolerance)
    return -angular <= u <= 1 + angular and -axial <= v <= 1 + axial


def _random_support(rng, kind, partial):
    axis = _unit(rng.normal(size=3))
    helper = _unit(np.cross(axis, rng.normal(size=3)))
    origin = rng.uniform(-1, 1, 3)
    sweep = math.tau if (not partial or rng.random() < .4) else rng.uniform(.8, 5.5)
    start = rng.uniform(0, 6)
    if kind == "cylinder":
        return Cylinder(origin, axis, helper, rng.uniform(.5, 2.5), rng.choice([-1, 1]) * rng.uniform(2., 7.), start, sweep)
    if kind == "cone":
        r0, r1 = rng.uniform(.2, 1.5), rng.uniform(.2, 2.5)
        r1 = r1 if abs(r1 - r0) > .1 else r0 + .6
        return Cone(origin, axis, helper, r0, r1, rng.choice([-1, 1]) * rng.uniform(1.5, 6.), start, sweep)
    u = _unit(np.cross(axis, helper)) * rng.uniform(3, 8)
    v = helper * rng.uniform(3, 8)
    return Plane(origin - .5 * u - .5 * v, u, v)


def _contour_oracle(cone, other, n_u=500, n_v=260):
    """Points of the other support's implicit surface on the cone's native chart (independent of the exact code)."""
    quadric = QuadricSupport.from_surface(other)
    u, v = np.linspace(0, 1, n_u), np.linspace(0, 1, n_v)
    angle = cone.start_angle + u * cone.sweep_angle
    radius = cone.radius_start + (cone.radius_end - cone.radius_start) * v
    e1, e2, axis = (np.asarray(x) for x in (cone.radial_direction, cone.circumferential_direction, cone.axis))
    grid = (np.asarray(cone.origin) + (v * cone.height)[None, :, None] * axis
            + radius[None, :, None] * (np.cos(angle)[:, None, None] * e1 + np.sin(angle)[:, None, None] * e2))
    value = quadric.value(grid)
    sign = np.sign(value)
    found = []
    i, j = np.nonzero(sign[:, :-1] * sign[:, 1:] < 0)
    t = value[i, j] / (value[i, j] - value[i, j + 1])
    found.append(grid[i, j] + t[:, None] * (grid[i, j + 1] - grid[i, j]))
    i, j = np.nonzero(sign[:-1, :] * sign[1:, :] < 0)
    t = value[i, j] / (value[i, j] - value[i + 1, j])
    found.append(grid[i, j] + t[:, None] * (grid[i + 1, j] - grid[i, j]))
    spacing = max(np.linalg.norm(grid[1, 0] - grid[0, 0]), np.linalg.norm(grid[0, 1] - grid[0, 0]))
    return np.vstack(found), spacing


def _farther_than(points, samples, limit):
    count = 0
    for start in range(0, len(points), 256):
        chunk = points[start:start + 256]
        nearest = np.min(np.linalg.norm(chunk[:, None, :] - samples[None, :, :], axis=2), axis=1)
        count += int(np.sum(nearest > limit))
    return count


@pytest.mark.parametrize("seed", range(18))
def test_support_curves_lie_on_both_supports_and_cover_an_independent_contour(seed):
    rng = np.random.default_rng(1000 + seed)
    cone = _random_support(rng, "cone", partial=True)
    other = _random_support(rng, ("cylinder", "cone", "plane")[seed % 3], partial=True)
    try:
        result = cone_support(cone, other)
    except GeometryError:
        pytest.skip("typed refusal of a degenerate random pair")
    samples = []
    for curve in result.curves:
        points = curve.evaluate(np.linspace(0, 1, 1500))
        samples.append(points)
        for support in (cone, other):
            distance, gradient = _distance(support, points)
            ok = gradient > 1e-6
            assert distance[ok].max() < 1e-9
        for point in points[::97]:                                        # curves stay inside both native patches
            assert _patch_contains(cone, point, 1e-7) and _patch_contains(other, point, 1e-7)
    contour, spacing = _contour_oracle(cone, other)
    keep = np.asarray([p for p in contour[::3] if _patch_contains(cone, p) and _patch_contains(other, p)])
    if len(keep):
        assert samples, "an intersection exists but nothing was returned"
        assert _farther_than(keep, np.vstack(samples), 4 * spacing) == 0


# ---------------------------------------------------------------- plane sections of a cone

CONE = Cone((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), .5, 1.5, 3., 0., math.tau)          # generators 18.4 deg off the axis


def _plate(normal_yz_degrees, center_z=1.5, shift=0.):
    t = math.radians(normal_yz_degrees)
    n = np.array([0., math.sin(t), math.cos(t)])
    u = np.array([1., 0., 0.])
    v = np.cross(n, u)
    return Plane(np.array([0., 0., center_z]) + shift * n - 3 * u - 3 * v, 6 * u, 6 * v)


def test_a_perpendicular_plane_cuts_a_circle_and_a_tilted_plane_an_ellipse_both_as_arcs():
    for tilt in (0., 10., 40., 70.):                         # ellipse until the plane is parallel to a generator (71.6 deg)
        result = cone_support(CONE, _plate(tilt))
        assert result.curves and all(isinstance(c, EllipticArc) for c in result.curves), tilt
        assert not result.segments
        for curve in result.curves:
            points = curve.evaluate(np.linspace(0, 1, 33))
            assert _distance(CONE, points)[0].max() < 1e-12 and _distance(_plate(tilt), points)[0].max() < 1e-12


def test_a_steep_plane_cuts_hyperbola_branches_as_exact_charts():
    for tilt, shift in ((80., 0.), (90., .4)):
        result = cone_support(CONE, _plate(tilt, shift=shift))
        assert result.curves and all(isinstance(c, QuadricIntersectionCurve) for c in result.curves), tilt
        assert all(c.branch == 1 for c in result.curves)                   # a plane has a single branch
        for curve in result.curves:
            points = curve.evaluate(np.linspace(0, 1, 33))
            assert _distance(CONE, points)[0].max() < 1e-12 and _distance(_plate(tilt, shift=shift), points)[0].max() < 1e-12


def test_a_plane_parallel_to_a_generator_is_an_exact_parabola():
    cone = Cone((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), .5, 1.5, 2., 0., math.tau)    # slope 1/2 is exact in binary
    plane = Plane(np.array([-2., -2., -.5]), np.array([4., 0., 0.]), np.array([0., 2., 4.]))   # z = 2y - 0.5 + ... exactly parallel
    # the normal (0, 2, -1) is perpendicular to the generator direction (0, 1/2, 1)
    assert plane.normal @ np.array([0., .5, 1.]) == pytest.approx(0., abs=1e-15)
    result = cone_support(cone, plane)
    assert result.curves and all(isinstance(c, QuadricIntersectionCurve) for c in result.curves)
    for curve in result.curves:
        points = curve.evaluate(np.linspace(0, 1, 33))
        assert _distance(cone, points)[0].max() < 1e-12 and _distance(plane, points)[0].max() < 1e-12


def test_a_plane_through_the_apex_cuts_generator_segments_once_each():
    plane = Plane(np.array([-4., 0., -4.]), np.array([8., 0., 0.]), np.array([0., 0., 8.]))   # contains the axis
    result = cone_support(CONE, plane)
    assert not result.curves and len(result.segments) == 2            # the seam of the full turn does not repeat one
    ends = sorted(tuple(np.round(segment[1], 9)) for segment in result.segments)
    assert ends == [(-1.5, 0., 3.), (1.5, 0., 3.)]
    for start, end in result.segments:
        assert np.linalg.norm(np.subtract(start, end)) > 3.


def test_a_plane_through_the_apex_perpendicular_to_the_axis_touches_only_the_apex():
    plane = Plane(np.array([-4., -4., -1.5]), np.array([8., 0., 0.]), np.array([0., 8., 0.]))
    pointed = Cone((0., 0., -1.5), (0., 0., 1.), (1., 0., 0.), 0., 1., 3., 0., math.tau)      # apex at the origin
    result = cone_support(pointed, plane)
    assert not result.curves and not result.segments
    assert len(result.points) == 1 and np.allclose(result.points[0], (0., 0., -1.5))


def test_the_exact_tangent_plane_of_a_cone_meets_it_along_one_generator():
    cone = Cone((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), .5, 1.5, 2., 0., math.tau)    # slope 1/2
    plane = Plane(np.array([.5, -4., 0.]), np.array([0., 8., 0.]), np.array([1., 0., 2.]))  # holds the generator at azimuth 0
    result = cone_support(cone, plane)
    assert not result.curves and len(result.segments) == 1
    start, end = result.segments[0]
    assert np.allclose(start, (.5, 0., 0.)) and np.allclose(end, (1.5, 0., 2.))


# ---------------------------------------------------------------- coaxial and coincident pairs


def test_coaxial_pairs_meet_in_circles():
    cylinder = Cylinder((0., 0., -1.), (0., 0., 1.), (1., 0., 0.), 1., 5., 0., math.tau)
    opposite = Cone((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 1.5, .5, 3., 0., math.tau)       # shrinks as CONE grows
    for other in (cylinder, opposite):
        result = cone_support(CONE, other)
        assert result.curves and all(isinstance(c, EllipticArc) for c in result.curves)
        for curve in result.curves:
            points = curve.evaluate(np.linspace(0, 1, 17))
            assert _distance(CONE, points)[0].max() < 1e-12 and _distance(other, points)[0].max() < 1e-12
            assert np.ptp(points[:, 2]) < 1e-12                                  # one ring


def test_the_same_cone_surface_is_reported_coincident_not_as_a_curve():
    assert cone_support(CONE, Cone((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), .5, 1.5, 3., 0., math.tau)).coincident
    upper = Cone((0., 0., 1.), (0., 0., 1.), (1., 0., 0.), 5. / 6., 1.5, 2., 0., math.tau)      # another patch of the same surface
    assert cone_support(CONE, upper).coincident


def test_disjoint_supports_have_no_intersection():
    far = Cylinder((10., 0., -3.), (0., 0., 1.), (1., 0., 0.), .5, 6., 0., math.tau)
    result = cone_support(CONE, far)
    assert not result.curves and not result.segments and not result.points and not result.coincident


def test_a_parallel_axis_cylinder_crosses_the_cone_in_exact_charts():
    cylinder = Cylinder((.9, 0., -3.), (0., 0., 1.), (1., 0., 0.), .5, 6., 0., math.tau)
    result = cone_support(CONE, cylinder)
    assert result.curves and all(isinstance(c, QuadricIntersectionCurve) for c in result.curves)
    for curve in result.curves:
        points = curve.evaluate(np.linspace(0, 1, 33))
        assert _distance(CONE, points)[0].max() < 1e-12 and _distance(cylinder, points)[0].max() < 1e-12


# ---------------------------------------------------------------- apex contacts and common generators


def _pointed(apex, axis, radial=(0., 1., 0.), radius=1., height=3.):
    return Cone(apex, axis, radial, 0., radius, height, 0., math.tau)


@pytest.mark.parametrize("axis", ((math.cos(math.radians(10)), 0., math.sin(math.radians(10))), (1., 0., 0.)))
def test_an_apex_on_the_other_support_leaves_no_degenerate_curve_and_is_a_point_contact(axis):
    """The branch ``s = 0`` of the rulings is the apex for every angle: a point, never a zero-length curve."""
    cylinder = Cylinder((0., 0., -3.), (0., 0., 1.), (1., 0., 0.), 2., 6., 0., math.tau)
    result = cone_support(_pointed((0., 2., 0.), axis), cylinder)
    assert len(result.curves) == 2 and all(isinstance(c, QuadricIntersectionCurve) for c in result.curves)
    for curve in result.curves:
        assert np.ptp(curve.evaluate(np.linspace(0., 1., 9)), axis=0).max() > .5
        assert _distance(cylinder, curve.evaluate(np.linspace(0., 1., 9)))[0].max() < 1e-12
    assert any(np.allclose(point, (0., 2., 0.)) for point in result.points)


def test_an_apex_on_the_wall_with_nothing_else_is_an_isolated_point():
    cylinder = Cylinder((0., 0., -3.), (0., 0., 1.), (1., 0., 0.), 2., 6., 0., math.tau)
    result = cone_support(_pointed((2., 0., 0.), (-1., 0., 0.)), cylinder)
    assert not result.curves and not result.segments
    assert len(result.points) == 1 and np.allclose(result.points[0], (2., 0., 0.))


def test_cones_sharing_an_apex_meet_along_their_common_generators():
    a = Cone((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), .5, 1.5, 2., 0., math.tau)             # apex (0, 0, -1), slope 1/2
    b = Cone((0., 0., -1.), (1., 0., 0.), (0., 1., 0.), 0., 2., 1., 0., math.tau)             # same apex, axis x, slope 2
    for first, second in ((a, b), (b, a)):
        result = cone_support(first, second)
        assert not result.curves
        assert len(result.segments) == 1                          # the line (1/2, 0, 1) t, clipped to both patches
        (start, end), = result.segments
        assert np.allclose(sorted([start, end], key=lambda p: p[2]), [(.5, 0., 0.), (1., 0., 1.)])


def test_the_plan_finds_the_generators_that_lie_wholly_on_the_other_support():
    a = Cone((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), .5, 1.5, 2., 0., math.tau)
    b = Cone((0., 0., -1.), (1., 0., 0.), (0., 1., 0.), 0., 2., 1., 0., math.tau)
    angles = [angle for angle, _m in get_plan(RuledSupport.from_surface(a), b).common_roots()]
    assert angles == pytest.approx([0., math.pi])                 # the two nappes of the common generator line
    generic = get_plan(RuledSupport.from_surface(CONE), Cylinder((.9, 0., -3.), (0., 0., 1.), (1., 0., 0.), .5, 6., 0., math.tau))
    assert generic.common_roots() == []


# ---------------------------------------------------------------- engine-facing properties


def test_every_configuration_is_numerically_clean():
    """No invalid-value or overflow warnings escape for generic, polar, tangent or disjoint pairs."""
    cases = [(CONE, _plate(80.)), (CONE, _plate(0.)), (CONE, _plate(90., shift=.4)),
             (CONE, Cylinder((.9, 0., -3.), (0., 0., 1.), (1., 0., 0.), .5, 6., 0., math.tau)),
             (CONE, Cylinder((0., 2.2, -3.), (0., 0., 1.), (1., 0., 0.), 2., 6., 0., math.tau))]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for first, second in cases:
            cone_support(first, second)


def test_the_cancellation_check_is_honoured():
    with pytest.raises(GeometryError, match="cancel"):
        cone_support(CONE, _plate(80.), cancellation_check=lambda: True)


def test_all_facets_of_a_shell_pair_share_one_exact_solve():
    """The exact events depend on the support pair only: facets add dictionary lookups, not solves."""
    cone_model = cone_shell(.5, 1.0, 5., origin=(0., 0., 0.), axis=(math.cos(math.radians(10)), 0., math.sin(math.radians(10))),
                            radial_direction=(0., 1., 0.), circumferential_segments=8)
    cylinder_model = cylinder_shell(2., 6., origin=(0., 0., -3.), circumferential_segments=12)
    cones = [f.surface for f in cone_model.faces.values()]
    cylinders = [f.surface for f in cylinder_model.faces.values()]
    _cached_plan.cache_clear()
    for a in cones:
        for b in cylinders:
            cone_support(a, b)
    info = _cached_plan.cache_info()
    # every cone facet is one support and every cylinder facet another: one exact solve for each choice of
    # which supplies the angle, however many facet pairs there are
    assert info.misses <= 2, info
    assert info.hits >= len(cones) * len(cylinders)
