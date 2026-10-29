"""T2-0: the float ray test of MaterialDomain.contains must not lose a crossing at a boundary vertex.

Reproduces the three-way tripod node defect (faces 961/601, centre_r8.5): the query height lies within
128 eps (in the line parameter) of a generator/cap-arc vertex. The generator reports its endpoint root
t = 1 (snapped; skipped as "the next curve owns it"), the following cap EllipticArc starts 1e-13 above
the ray and reports no root, so the vertex crossing was lost: a point 6 deg outside the strip was
inside, and a point inside at the same height was outside.
"""
import math

import numpy as np
import pytest

from anygeometry.arrangement_geometry import LinePath
from anygeometry.exact_curves import EllipticArc
from anygeometry.material_arrangement import ArrangementPath, MaterialDomain
from anygeometry.surfaces import Cylinder

R = 6.5
SIX = math.radians(6.)


def _strip():
    """Strip 0 <= theta <= 6 deg on a full-turn cylinder whose chart starts at -0.25 rad (as the tripod arms).

    Oblique caps: z(theta) = z0 - sin(theta), so the top cap rises from the 6 deg vertex to the 0 deg vertex.
    """
    support = Cylinder(origin=(0., 0., 0.), axis=(0., 0., 1.), radial_direction=(1., 0., 0.), radius=R,
                       height=52., start_angle=-.25)
    bottom = EllipticArc((0., 0., 4.), (R, 0., 0.), (0., R, -1.), 0., SIX)
    top = EllipticArc((0., 0., 45.), (R, 0., 0.), (0., R, -1.), SIX, -SIX)
    up = LinePath(tuple(map(float, bottom.evaluate(1.))), tuple(map(float, top.evaluate(0.))))
    down = LinePath(tuple(map(float, top.evaluate(1.))), tuple(map(float, bottom.evaluate(0.))))
    loop = tuple(ArrangementPath(c) for c in (bottom, up, top, down))
    return MaterialDomain(0, support, (loop,)), up


def _contains(domain, theta, z):
    point = (R*math.cos(theta), R*math.sin(theta), z)
    return domain.contains(LinePath(point, point), 0., 1e-9)


@pytest.mark.parametrize("delta", (1e-13, 3e-13, 5e-13))
def test_point_outside_strip_at_vertex_height_is_outside(delta):
    domain, up = _strip()
    z = float(up.end[2])-delta            # within 128 eps of the 6 deg vertex in the generator's parameter
    assert not _contains(domain, -SIX, z)
    assert not _contains(domain, -3*SIX, z)


@pytest.mark.parametrize("delta", (1e-13, 3e-13, 5e-13))
def test_point_inside_strip_at_vertex_height_is_inside(delta):
    domain, up = _strip()
    z = float(up.end[2])-delta
    assert _contains(domain, .5*SIX, z)


def _plane_face():
    """Plane face: long right edge up to vertex V at 5 deg on a unit circle, arc 5 -> 15 deg, then back.

    The edge is 100 long, so a query 1e-12 below V snaps its root to t = 1 (128 eps), while the
    arc's start root lies 6e-12 of its sweep outside the arc and is not reported.
    """
    from anygeometry.surfaces import Plane
    arc = EllipticArc((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), math.radians(5.), math.radians(10.))
    v_point, w_point = (tuple(map(float, arc.evaluate(t))) for t in (0., 1.))
    corners = ((v_point[0], -99.5, 0.), v_point, None, w_point, (0., w_point[1], 0.), (0., -99.5, 0.))
    curves = (LinePath(corners[0], corners[1]), arc, LinePath(corners[3], corners[4]),
              LinePath(corners[4], corners[5]), LinePath(corners[5], corners[0]))
    support = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    return MaterialDomain(0, support, (tuple(ArrangementPath(c) for c in curves),)), v_point


@pytest.mark.parametrize("route", ("exact", "float"))
@pytest.mark.parametrize("delta", (1e-12, 2e-12))
def test_plane_vertex_height_line_then_arc(route, delta):
    domain, v_point = _plane_face()
    y = v_point[1]-delta

    def contains(x):
        if route == "exact":           # a line station: exact chart point, contains_loop_exact_curved
            point = (x, y, 0.)
            return domain.contains(LinePath(point, point), 0., 1e-9)
        # an arc station: float chart point, the float MaterialDomain.contains_loop on the Plane
        query = EllipticArc((x-1e-3, y, 0.), (1e-3, 0., 0.), (0., 1e-3, 0.), 0., 1.)
        return domain.contains(query, 0., 1e-9)

    assert not contains(-1.)
    assert contains(.5)
    assert not contains(2.)


def test_ordinary_heights_unchanged():
    domain, up = _strip()
    z = .5*(float(up.start[2])+float(up.end[2]))
    assert _contains(domain, .5*SIX, z)
    assert not _contains(domain, -SIX, z)
    assert not _contains(domain, 2*SIX, z)
    assert _contains(domain, SIX, z)                  # on the boundary generator
    assert _contains(domain, SIX, float(up.end[2]))  # the vertex itself
