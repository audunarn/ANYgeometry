"""Tripod hub: three equal-radius cylinders with coplanar crossing axes, mutually tangent at two points.

Fixture ``_hub_tangency_fixture.hub``: the centre section of a floating tripod, the three crossbeams
cut square 1 m past their hub seams, with the hub seams and longitudinal webs (see the module).  At
the hub axis points (0, 0, zc +- R) every pair of crossbeam cylinders touches tangentially, the
three hub seam ellipses meet, and the webs on the top and bottom skin seams end.  In the stored
doubles these contacts are exact double points: the exact rational discriminant of every
cylinder pair's fold equation is zero (pinned below), so a correct kernel has one vertex there.

Before the fix the cylinder/cylinder branch ending at such a node was evaluated through the
cancelling discriminant b^2 - 4ac: every parameter within about 1e-6 of the node was displaced
by up to sqrt(rounding) (7.95e-8 m for R = 6, tolerance 1e-9), and its interval enclosure was
3.8e-6 m wide however short the piece.  The displaced points produced a spurious web/seam
junction 7.95e-8 m from the bottom vertex ("arrangement contains an unresolved bridge"); the
loose enclosure kept 1e-14 m branch pieces at the node unmerged ("unresolved closed or degenerate
arrangement edge").  Which dimensions passed was luck of rounding.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from anygeometry import plan_intersections
from anygeometry.analytic_supports import cylinder_cylinder_support
from anygeometry.exact_curves import CylinderIntersectionCurve

from _hub_tangency_fixture import hub


def _prepare(**options):
    model, faces = hub(**options)
    return model, faces, plan_intersections(model, tuple(model.faces), policy="connect")


def _no_stub_edges(plan, tolerance=1e-9):
    for arrangement in plan.arrangements:
        for path in arrangement.paths:
            assert np.linalg.norm(path.curve.evaluate(0.) - path.curve.evaluate(1.)) > 1e3 * tolerance, (
                arrangement.face_id, path)


# ---- (a), (b): the owner's centre-section cut, one web on the bottom / top skin seam of arm 1

@pytest.mark.parametrize("webs", ["bottom", "top"])
def test_hub_point_with_a_seam_web_prepares(webs):
    _model, _faces, plan = _prepare(webs=webs, web_arms=(0,))
    _no_stub_edges(plan)


# ---- (c): dimension sweep (R, zc); 6.5 / 15.5, 5 / 12 and 6.3 / 14.7 passed before only by rounding luck

SWEEP = [(6.0, 14.0), (6.5, 15.5), (5.0, 12.0), (6.0, 15.0), (7.0, 16.5), (3.1, 9.9), (6.3, 14.7),
         (6.0, 14.000000000000002)]


@pytest.mark.parametrize("radius, zc", SWEEP)
@pytest.mark.parametrize("webs", ["bottom", "top"])
def test_hub_dimension_sweep_prepares(radius, zc, webs):
    _model, _faces, plan = _prepare(radius=radius, zc=zc, webs=webs, web_arms=(0,))
    _no_stub_edges(plan)


@pytest.mark.parametrize("rotation", ["rational", "float"])
@pytest.mark.parametrize("radius, zc", [(6.0, 14.0), (7.0, 16.5)])
def test_hub_skins_alone_prepare(rotation, radius, zc):
    _model, _faces, plan = _prepare(radius=radius, zc=zc, webs="none", rotation=rotation)
    _no_stub_edges(plan)


def test_float_rotated_hub_with_a_seam_web_prepares():
    _model, _faces, plan = _prepare(webs="bottom", web_arms=(0,), rotation="float")
    _no_stub_edges(plan)


# ---- the node itself: exact double in the stored doubles, accurate evaluation and enclosure

def _exact_fold_sign(first, second, sign):
    """Exact sign of nc^2 + ns^2 - (n0 - sign |a1 x a2| r2)^2 for the stored doubles (zero: exact double)."""
    q = lambda vector: [Fraction(float(value)) for value in vector]
    o1, a1, e1 = q(first.origin), q(first.axis), q(first.radial_direction)
    o2, a2 = q(second.origin), q(second.axis)
    e2 = q(np.cross(first.axis, first.radial_direction))
    cross = lambda u, v: [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
    dot = lambda u, v: sum(x*y for x, y in zip(u, v))
    c = cross(a1, a2)
    r1, r2 = Fraction(float(first.radius)), Fraction(float(second.radius))
    n0, m2 = dot([x-y for x, y in zip(o1, o2)], c), dot(c, c)
    nc, ns = r1*dot(e1, c), r1*dot(e2, c)
    rational, factor = nc*nc+ns*ns-n0*n0-m2*r2*r2, 2*sign*n0*r2
    signum = lambda value: (value > 0) - (value < 0)
    if factor == 0 or signum(rational) in (0, signum(factor)):
        return signum(rational) or signum(factor)
    left, right = rational*rational, factor*factor*m2
    return signum(rational) if left > right else signum(factor) if left < right else 0


def _crossbeams(rotation="rational", radius=6.0, zc=14.0):
    model, faces = hub(radius=radius, zc=zc, webs="none", rotation=rotation)
    return [model.faces[face].surface for face in faces["skins"][::2]]


@pytest.mark.parametrize("rotation", ["rational", "float"])
def test_hub_contacts_are_exact_doubles_in_the_stored_doubles(rotation):
    beams = _crossbeams(rotation)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert {_exact_fold_sign(beams[i], beams[j], sign) for sign in (-1, 1)} == {0}


def _node_branches(rotation="rational"):
    """Arm-1 x arm-2 branch pieces that end at the bottom hub node (0, 0, zc - R)."""
    first, second = _crossbeams(rotation)[:2]
    node = np.array((0., 0., 8.))
    result = []
    for curve in cylinder_cylinder_support(first, second, tolerance=1e-9).curves:
        if not isinstance(curve, CylinderIntersectionCurve):
            continue
        for end in (0., 1.):
            if np.linalg.norm(curve.evaluate(end) - node) < 1e-12:
                result.append((curve, end))
    assert result
    return result, node


def _plane_distance(point, node):
    """Distance from the union of the two bisector planes (the exact intersection ellipses' planes)."""
    offset = point - node
    planes = [np.array((-math.sin(a), math.cos(a), 0.)) for a in (math.pi / 3., 5. * math.pi / 6.)]
    return min(abs(float(offset @ normal)) for normal in planes)


@pytest.mark.parametrize("rotation", ["rational", "float"])
def test_branch_is_accurate_next_to_a_double_node(rotation):
    branches, node = _node_branches(rotation)
    for curve, end in branches:
        for step in (1e-16, 1e-15, 1e-13, 1e-11, 1e-9, 1e-7, 1e-5):
            t = end + (step if end == 0. else -step)
            point = curve.evaluate(t)
            # On the exact intersection: within rounding of one of the two bisector-plane ellipses.
            assert _plane_distance(point, node) <= 1e-13, (step, point)


@pytest.mark.parametrize("rotation", ["rational", "float"])
def test_branch_enclosure_is_tight_next_to_a_double_node(rotation):
    branches, _node = _node_branches(rotation)
    for curve, end in branches:
        for step in (1e-14, 1e-12, 1e-10):
            piece = curve.subcurve(end, end + (step if end == 0. else -step))
            lo, hi = piece.bounds()
            chord = np.linalg.norm(piece.evaluate(1.) - piece.evaluate(0.))
            # A few chords plus the node-angle slack (~1e-12 m), not sqrt(rounding) (3.8e-6 m before the fix).
            assert np.linalg.norm(hi - lo) <= 4 * chord + 1e-11, (step, chord, hi - lo)
