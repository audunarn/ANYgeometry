"""Plane x NURBS section topology (generality slice B1-P1, architecture 4.1-4.2).

References: constructed patches whose plane sections are known in closed form
(rulings and ellipses of a rational cylinder, the lines of a quadratic ridge),
the analytic positions of their end points, and the refusals of 4.2.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from anygeometry import Cylinder, Plane
from anygeometry.nurbs import NurbsSurface, cylinder_to_nurbs
from anygeometry.plane_nurbs import (PlaneNurbsClosedSection, PlaneNurbsNotCertified, plane_nurbs_section)


def cylinder(sweep=math.pi):
    return cylinder_to_nurbs(Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 2., 3., 0., sweep))


def horizontal(z):
    return Plane((0., 0., z), (1., 0., 0.), (0., 1., 0.))


def vertical(x):
    return Plane((x, 0., 0.), (0., 1., 0.), (0., 0., 1.))


def ridge(z_controls):
    """z = quadratic in u (constant in v) over the unit square."""
    return NurbsSurface(2, 1, (0., 0., 0., 1., 1., 1.), (0., 0., 1., 1.),
                        tuple(((i / 2, 0., z), (i / 2, 1., z)) for i, z in enumerate(z_controls)))


def on_plane(plane, point, tol=1e-12):
    n = np.cross(plane.u_vector, plane.v_vector)
    return abs(float(n @ (np.asarray(point) - plane.origin))) <= tol * (1 + float(np.linalg.norm(point)))


def end_points(section, surface):
    return [[surface.evaluate(*section.chart_point(ident, surface)) for ident in (path.roots[0], path.roots[-1])]
            for path in section.paths]


@pytest.mark.parametrize("z, expected", ((1.2, 1), (.3, 1), (2.9, 1), (5., 0), (-1., 0)))
def test_horizontal_sections_of_a_cylinder(z, expected):
    surface = cylinder()
    section = plane_nurbs_section(horizontal(z), surface)
    assert len(section.paths) == expected
    for ends in end_points(section, surface):
        for point in ends:
            assert abs(point[2] - z) <= 1e-12 and abs(math.hypot(point[0], point[1]) - 2.) <= 1e-12


@pytest.mark.parametrize("sweep, x, expected", ((math.pi, .7, 1), (math.pi, -1.3, 1), (2 * math.pi, .7, 2),
                                                (2 * math.pi, 1.9, 2), (math.pi, 2.5, 0)))
def test_vertical_sections_are_rulings(sweep, x, expected):
    surface = cylinder(sweep)
    plane = vertical(x)
    section = plane_nurbs_section(plane, surface)
    assert len(section.paths) == expected
    for ends in end_points(section, surface):
        assert on_plane(plane, ends[0]) and on_plane(plane, ends[1])
        assert {round(ends[0][2], 12), round(ends[1][2], 12)} == {0., 3.}          # bottom to top


def test_an_oblique_plane_cuts_a_full_cylinder_in_one_path_across_knot_lines():
    surface = cylinder(2 * math.pi)                         # four rational spans in u
    plane = Plane((0., 0., 1.5), (1., 0., .3), (0., 1., .2))
    section = plane_nurbs_section(plane, surface)
    assert len(section.paths) == 1
    ends = end_points(section, surface)[0]
    assert all(on_plane(plane, point) for point in ends)
    assert len(section.paths[0].arcs) >= 4                  # through every span


def test_a_ridge_cut_at_two_levels():
    surface = ridge((.21, -.29, .21))                       # z = (u - .3)(u - .7), to rounding of the controls
    section = plane_nurbs_section(horizontal(0.), surface)
    assert len(section.paths) == 2
    for ends in end_points(section, surface):
        assert abs(ends[0][0] - ends[1][0]) <= 1e-12        # each path is a line u = const across v
    assert len(plane_nurbs_section(horizontal(-.1), surface).paths) == 0


def test_refusals():
    with pytest.raises(PlaneNurbsNotCertified):             # tangent along a line (double root)
        plane_nurbs_section(horizontal(0.), ridge((.25, -.25, .25)))
    dome = NurbsSurface(2, 2, (0., 0., 0., 1., 1., 1.), (0., 0., 0., 1., 1., 1.),
                        tuple(tuple((i / 2, j / 2, 1. if (i, j) == (1, 1) else 0.) for j in range(3)) for i in range(3)))
    section = plane_nurbs_section(horizontal(.2), dome)
    assert len(section.paths) == 1 and section.paths[0].closed
    assert section.paths[0].roots[0] == section.paths[0].roots[-1]
    # z = 1.5 is v = 1/2, but G is monotone in v over the whole patch (one regular box): no grid line, one path
    assert len(plane_nurbs_section(horizontal(1.5), cylinder()).paths) == 1
    flat = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                        (((0., 0., 0.), (0., 1., 0.)), ((1., 0., 0.), (1., 1., 0.))))
    assert plane_nurbs_section(horizontal(0.), flat).coincident


def test_internal_circle_uses_certified_alternate_subdivision():
    coefficients = (1., -1., 1.)
    surface = NurbsSurface(2, 2, (0., 0., 0., 1., 1., 1.),
                           (0., 0., 0., 1., 1., 1.),
                           tuple(tuple((i - 1., j - 1., coefficients[i] + coefficients[j])
                                       for j in range(3)) for i in range(3)))
    section = plane_nurbs_section(horizontal(.5), surface)
    assert len(section.paths) == 1 and section.paths[0].closed
    path = section.paths[0]
    assert len(path.arcs) >= 2
    assert len(path.roots) == len(path.arcs) + 1
    assert path.roots[0] == path.roots[-1]
    assert len(set(path.arcs)) == len(path.arcs)
    from anygeometry.support_curves import PlaneNurbsCurve
    curve = PlaneNurbsCurve.from_plane(horizontal(.5), surface)
    points = curve.evaluate(np.linspace(0., 1., 19))
    assert np.allclose(points[:, 2], .5, atol=1e-10)
    assert np.allclose(np.sum(points[:, :2] ** 2, axis=1), .5, atol=1e-10)


def boundary_sign_changes(plane, surface, count=4001):
    """Independent check: sign changes of the plane distance along the four chart boundary edges."""
    n = np.cross(plane.u_vector, plane.v_vector)
    t = np.linspace(0., 1., count)
    total = 0
    for uv in (np.column_stack((t, 0 * t)), np.column_stack((1 + 0 * t, t)),
               np.column_stack((t[::-1], 1 + 0 * t)), np.column_stack((0 * t, t[::-1]))):
        values = (surface.evaluate_many(uv) - plane.origin) @ n
        signs = np.sign(values[np.abs(values) > 1e-12])
        total += int(np.count_nonzero(np.diff(signs)))
    return total


def test_random_wavy_patches_match_the_boundary_sign_count():
    rng = np.random.default_rng(5)
    certified = 0
    for _ in range(30):
        controls = np.zeros((5, 4, 3))
        controls[..., 0] = np.linspace(0., 4., 5)[:, None]
        controls[..., 1] = np.linspace(0., 3., 4)[None, :]
        controls[..., 2] = rng.uniform(-1., 1., (5, 4))
        surface = NurbsSurface(3, 2, (0., 0., 0., 0., .5, 1., 1., 1., 1.), (0., 0., 0., .4, 1., 1., 1.),
                               tuple(tuple(map(tuple, row)) for row in controls),
                               tuple(map(tuple, rng.uniform(.6, 2., (5, 4)))))
        plane = Plane((0., 0., rng.uniform(-.3, .3)), (1., 0., rng.uniform(-.2, .2)), (0., 1., rng.uniform(-.2, .2)))
        try:
            section = plane_nurbs_section(plane, surface)
        except PlaneNurbsNotCertified:
            continue
        certified += 1
        assert 2 * sum(not path.closed for path in section.paths) == boundary_sign_changes(plane, surface)
        for ends in end_points(section, surface):
            assert on_plane(plane, ends[0], 1e-9) and on_plane(plane, ends[1], 1e-9)
    assert certified >= 25



def two_span(z_middle):
    return NurbsSurface(1, 1, (0., 0., .5, 1., 1.), (0., 0., 1., 1.),
                        tuple(((u, 0., z), (u, 1., z)) for u, z in ((0., 1.), (.5, z_middle), (1., 1.))))


def test_roots_either_side_of_a_knot_stay_distinct():
    """Review B1-P1: two lines just left and right of the knot u = 1/2."""
    section = plane_nurbs_section(horizontal(0.), two_span(-2. ** -100))
    assert len(section.paths) == 2
    touching = plane_nurbs_section(horizontal(0.), two_span(1.))
    assert len(touching.paths) == 0


def test_partial_coincidence_is_a_typed_refusal():
    with pytest.raises(PlaneNurbsNotCertified, match="partly in the plane"):
        plane_nurbs_section(horizontal(0.), NurbsSurface(
            1, 1, (0., 0., .5, 1., 1.), (0., 0., 1., 1.),
            tuple(((u, 0., z), (u, 1., z)) for u, z in ((0., 0.), (.5, 0.), (1., 1.)))))



def test_the_box_budget_is_shared_by_the_pair(monkeypatch):
    import anygeometry.plane_nurbs as module
    monkeypatch.setattr(module, "MAX_BOXES", 50)
    knots = (0.,) + tuple(float(k) for k in range(0, 9)) + (8.,)
    surface = NurbsSurface(1, 1, knots, knots, tuple(tuple((float(i), float(j), 1.) for j in range(9)) for i in range(9)))
    with pytest.raises(PlaneNurbsNotCertified, match="budget"):
        plane_nurbs_section(horizontal(0.), surface)              # 64 empty sub-patches > 50 boxes
