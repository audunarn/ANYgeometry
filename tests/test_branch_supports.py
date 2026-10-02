"""Bezier extrusions of any degree against quadric patches: the exact polynomial-chart branch curves.

Curves are checked on both supports, against an independent grid oracle (the sign changes of the quadric along
fixed-``v`` slices of the wall's chart), against the established angular engine wherever both apply (a quadratic
wall is a parabolic cylinder), and through the degenerate relations: whole generators, exact tangency, patch cuts.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from anygeometry.branch_supports import bezier_quadric_support
from anygeometry.errors import GeometryError
from anygeometry.extrusions import BezierDirectrix, EllipseDirectrix
from anygeometry.quadric_algebra import QuadricSupport
from anygeometry.quadric_supports import quadric_pair_support
from anygeometry.surfaces import Cone, Cylinder, ExtrudedSurface

CUBIC = ((0., 0., 0.), (1., 2., 0.), (2., -1., 0.), (3., 1., 0.))
QUARTIC = ((0., 0., 0.), (1., 2., 0.), (2., -1., 0.), (3., 2., 0.), (4., 0., 0.))
ARCH = ((0., 0., 0.), (1., 2., 0.), (2., 0., 0.))
VECTOR = (.25, 0., 1.5)


def wall(controls=CUBIC, vector=VECTOR, u_range=(0., 1.), v_range=(0., 1.)):
    return ExtrudedSurface(BezierDirectrix(controls), vector, u_range, v_range)


def pipe(radius=.7, y=.4, z=.6, **angles):
    return Cylinder((-2., y, z), (1., 0., 0.), (0., 1., 0.), radius, 8., **angles)


def tube(center=(1.5, -1., .7), a=1., b=.5, vector=(0., 2.5, .3)):
    return ExtrudedSurface(EllipseDirectrix(center, (a, 0., 0.), (0., 0., b)), vector)


SECONDS = {
    "a pipe across the wall": pipe(),
    "a narrow pipe (two folds)": pipe(radius=.3, y=.5, z=.7),
    "a wide pipe": pipe(radius=.9, y=.2, z=.9),
    "a pipe 10 degrees off perpendicular": Cylinder((-2., .5, .7), (math.cos(.17), 0., math.sin(.17)), (0., 1., 0.), .4, 8.),
    "a cone": Cone((-2., .5, .8), (1., 0., 0.), (0., 1., 0.), .1, 1., 8.),
    "a skew pipe along y": Cylinder((1.4, -2., .7), (0., 1., 0.), (1., 0., 0.), .35, 8.),
    "an elliptic tube": tube(),
    "a quadratic wall of another direction": ExtrudedSurface(BezierDirectrix(((0., -1., 0.), (1., 1.5, .2), (2., -1., .4))),
                                                             (0., 1., 1.)),
}


def chart_polyline(curve, surface, samples=600):
    return surface.local_uv_many(np.asarray(curve.evaluate(np.linspace(0., 1., samples))))


def slice_crossings(polylines, level):
    found = []
    for uv in polylines:
        d = uv[:, 1] - level
        for i in np.nonzero(d[:-1] * d[1:] < 0)[0]:
            w = d[i] / (d[i] - d[i + 1])
            found.append(uv[i, 0] + w * (uv[i + 1, 0] - uv[i, 0]))
    return sorted(found)


def grid_crossings(first, second, level, columns=4000):
    u = np.linspace(0., 1., columns + 1)
    values = QuadricSupport.from_surface(second).value(first.evaluate_many(np.column_stack((u, np.full_like(u, level)))))
    return [u[i] + values[i] / (values[i] - values[i + 1]) * (u[i + 1] - u[i])
            for i in np.nonzero(values[:-1] * values[1:] < 0)[0]]


def inside_patch(second, point):
    if isinstance(second, ExtrudedSurface):
        u, v = second.local_uv(point)
        return -1e-9 <= u <= 1 + 1e-9 and -1e-9 <= v <= 1 + 1e-9
    local = np.asarray(point) - second.origin
    z = float(local @ second.axis)
    ok = min(0., second.height) - 1e-9 <= z <= max(0., second.height) + 1e-9
    return ok


@pytest.mark.parametrize("controls", [CUBIC, QUARTIC], ids=["cubic", "quartic"])
@pytest.mark.parametrize("name", SECONDS)
def test_a_wall_meets_a_quadric_on_both_supports_with_the_topology_of_a_grid_oracle(name, controls):
    first, second = wall(controls), SECONDS[name]
    result = bezier_quadric_support(first, second)
    assert result.curves and not result.coincident and not result.segments
    quadric = QuadricSupport.from_surface(second)
    for curve in result.curves:
        points = np.asarray(curve.evaluate(np.linspace(0., 1., 101)))
        assert np.abs(first.evaluate_many(first.local_uv_many(points)) - points).max() < 1e-12
        assert (np.abs(quadric.value(points)) / quadric.gradient_norm(points)).max() < 1e-11
    polylines = [chart_polyline(curve, first) for curve in result.curves]
    for level in np.linspace(.0137, .9871, 13):               # slices of the wall's chart; never a split height
        exact = slice_crossings(polylines, level)
        grid = [u for u in grid_crossings(first, second, level)
                if inside_patch(second, first.evaluate_many(np.array([[u, level]]))[0])]
        assert len(exact) == len(grid), (name, float(level))
        assert np.allclose(exact, grid, atol=2e-3), (name, float(level))


def curve_set(result, samples=800):
    points = [np.asarray(curve.evaluate(np.linspace(0., 1., samples))) for curve in result.curves]
    steps = [np.linalg.norm(np.diff(p, axis=0), axis=1) for p in points]
    return (np.vstack(points) if points else np.empty((0, 3))), sum(float(s.sum()) for s in steps), max(
        (float(s.max()) for s in steps), default=0.)


def one_sided(p, q):
    return float(np.sqrt(((p[:, None, :] - q[None, :, :]) ** 2).sum(-1)).min(axis=1).max())


QUADRATIC_CASES = {
    "pipe across": (wall(ARCH, (0., 0., 2.)), pipe(radius=.5, y=.3, z=1.)),
    "wide pipe cutting the arch twice": (wall(ARCH, (0., 0., 2.)), pipe(radius=.9, y=.2, z=1.)),
    "cone": (wall(ARCH, (.2, 0., 2.)), Cone((-2., .3, 1.), (1., 0., 0.), (0., 1., 0.), .2, 1., 6.)),
    "elliptic tube": (wall(ARCH, (0., 0., 2.)), tube(center=(1., -1., 1.), a=.6, b=.6, vector=(0., 2.5, .3))),
    "cut patch of the wall": (wall(ARCH, (0., 0., 2.), u_range=(.1, .8), v_range=(.15, .7)), pipe(radius=.9, y=.2, z=1.)),
    "cut patch of the pipe": (wall(ARCH, (0., 0., 2.)), pipe(radius=.9, y=.2, z=1., start_angle=.3, sweep_angle=3.)),
}


@pytest.mark.parametrize("name", QUADRATIC_CASES)
def test_a_quadratic_wall_gives_the_same_curves_from_the_polynomial_and_the_angular_engines(name):
    """A quadratic Bezier is a parabola, so the established angular engine (the wall as a parabolic second support)
    computes the same intersection by an entirely different parametrization."""
    first, second = QUADRATIC_CASES[name]
    polynomial = bezier_quadric_support(first, second)
    angular = quadric_pair_support(first, second)
    p, length, spacing_p = curve_set(polynomial)
    q, other, spacing_q = curve_set(angular)
    assert len(p) and len(q)
    assert length == pytest.approx(other, rel=1e-5)
    assert one_sided(p, q) <= .5 * spacing_q + 1e-9 and one_sided(q, p) <= .5 * spacing_p + 1e-9


def test_patch_cuts_of_the_wall_clip_the_curves_exactly():
    second = pipe(radius=.9, y=.2, z=.9)
    whole = bezier_quadric_support(wall(), second)
    whole_length = curve_set(whole)[1]
    for cut in (wall(u_range=(.1, .8)), wall(v_range=(.15, .9)), wall(u_range=(.05, .9), v_range=(.1, .95))):
        result = bezier_quadric_support(cut, second)
        points = np.vstack([c.evaluate(np.linspace(0., 1., 80)) for c in result.curves])
        uv = cut.local_uv_many(points)
        assert uv[:, 1].min() > -1e-9 and uv[:, 1].max() < 1 + 1e-9
        t = cut.directrix.invert(points - cut.extrusion_coordinate(points)[:, None] * np.asarray(cut.vector))
        assert t.min() >= min(cut.u_range) - 1e-9 and t.max() <= max(cut.u_range) + 1e-9
        assert 0. < curve_set(result)[1] < whole_length


def test_a_cut_of_the_second_patch_clips_the_curves():
    cut = bezier_quadric_support(wall(), pipe(radius=.9, y=.2, z=.9, start_angle=.3, sweep_angle=3.))
    points = np.vstack([c.evaluate(np.linspace(0., 1., 80)) for c in cut.curves])
    second = pipe(radius=.9, y=.2, z=.9, start_angle=.3, sweep_angle=3.)
    uv = np.array([second.local_uv(p) for p in points])
    assert uv.min() > -1e-9 and uv.max() < 1 + 1e-9


def test_rulings_parallel_to_the_extrusion_meet_the_wall_in_generators():
    axis = np.asarray(VECTOR) / np.linalg.norm(VECTOR)
    parallel = Cylinder((1.2, .4, -1.), tuple(axis), (0., 1., 0.), .6, 8.)
    result = bezier_quadric_support(wall(), parallel)
    assert not result.curves and len(result.segments) >= 2
    for start, end in result.segments:
        assert np.allclose(np.asarray(end) - np.asarray(start), np.asarray(VECTOR), atol=1e-12)    # the whole ruling
    outside = Cylinder((8., .4, -1.), tuple(axis), (0., 1., 0.), .6, 8.)
    assert not bezier_quadric_support(wall(), outside).segments


def test_a_quadric_touching_the_wall_at_one_point_gives_that_point_exactly():
    apex = ((0., 0., 0.), (1., 2., 0.), (2., 2., 0.), (3., 0., 0.))              # y(t) = 6 t (1 - t): maximum 1.5 at t = 1/2
    touching = Cylinder((-2., 2.5, 1.), (1., 0., 0.), (0., 1., 0.), 1., 8.)       # lowest line y = 1.5 at z = 1
    first = wall(apex, (0., 0., 2.))
    result = bezier_quadric_support(first, touching)
    assert not result.curves and len(result.points) == 1
    assert np.allclose(result.points[0], (1.5, 1.5, 1.), atol=1e-12)


def test_the_same_surface_is_coincident_and_two_cubic_walls_are_refused():
    arch = wall(ARCH, (0., 0., 2.))
    assert bezier_quadric_support(arch, wall(ARCH, (0., 0., 2.), v_range=(.5, 1.5))).coincident
    with pytest.raises(GeometryError):
        bezier_quadric_support(wall(), wall(CUBIC, (0., .5, 1.5)))                 # a cubic is not a quadric
    with pytest.raises(GeometryError):
        bezier_quadric_support(tube(), pipe())                                     # the first support is a Bezier


def test_two_quadratic_walls_of_different_directions_give_the_same_curve_from_either_wall():
    """Neither parabolic cylinder has an angle to supply, so the polynomial engine takes one as its wall; the two
    charts are parametrized by different directrices, and the curve is the same."""
    a = ExtrudedSurface(BezierDirectrix(((0., 0., 0.), (1.5, 2., 0.), (3., 0., 0.))), (0., 0., 2.))
    b = ExtrudedSurface(BezierDirectrix(((1., -1., 0.), (1.6, .8, 0.), (1.2, 2.5, 0.))), (.3, 0., 1.6))
    forward, backward = bezier_quadric_support(a, b), bezier_quadric_support(b, a)
    assert forward.curves and backward.curves and not forward.segments and not backward.segments

    def length(result):
        return sum(float(np.linalg.norm(np.diff(c.evaluate(np.linspace(0., 1., 4001)), axis=0), axis=1).sum()) for c in result.curves)

    assert length(forward) == pytest.approx(length(backward), rel=1e-6)
    for result in (forward, backward):                                          # every point lies on both surfaces
        for curve in result.curves:
            points = np.asarray(curve.evaluate(np.linspace(0., 1., 101)))
            for surface in (a, b):
                assert np.abs(surface.evaluate_many(surface.local_uv_many(points)) - points).max() < 1e-12
    cloud = lambda result: np.vstack([c.evaluate(np.linspace(0., 1., 600)) for c in result.curves])
    p, q = cloud(forward), cloud(backward)
    spacing = max(float(np.linalg.norm(np.diff(c.evaluate(np.linspace(0., 1., 600)), axis=0), axis=1).max())
                  for result in (forward, backward) for c in result.curves)
    assert one_sided(p, q) <= spacing and one_sided(q, p) <= spacing
