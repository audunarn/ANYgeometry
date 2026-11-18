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


# ---- one certified double fold plus a simple fold: a large pipe with a small one tangent inside it

def _offset_pipes():
    """x-pipe r = 2 and y-pipe r = 1 with axes 1 apart in z: internally tangent at (0, 0, -2) (fold
    sign -1 is an exact double); the other fold is simple (square charts at theta = pi/2, 3 pi/2)."""
    from anygeometry.surfaces import Cylinder
    big = Cylinder((-4., 0., 0.), (1., 0., 0.), (0., 0., -1.), 2., 8.)
    small = Cylinder((0., -4., -1.), (0., 1., 0.), (1., 0., 0.), 1., 8.)
    curves = [c for c in cylinder_cylinder_support(big, small, tolerance=1e-9).curves
              if isinstance(c, CylinderIntersectionCurve)]
    assert {c.parameterization for c in curves} >= {"linear", "left_square", "right_square"}
    return big, small, curves


def test_offset_pipes_have_one_exact_double_fold():
    big, small, _curves = _offset_pipes()
    assert _exact_fold_sign(big, small, -1) == 0 and _exact_fold_sign(big, small, 1) != 0


def test_one_double_pair_is_accurate_next_to_its_node():
    # On the big pipe y = 2 sin(theta), z = -2 cos(theta); on the small one x^2 + (z + 1)^2 = 1, so
    # |x| = 2 sqrt(2) |sin(theta / 2)| sqrt(cos(theta)) without cancellation near the node theta = 0.
    # (The reference is ill-conditioned at the simple folds, so only the node zone is checked; there the
    # worst error was 1.65e-8 before the fix and 1.0e-15 after it, on every chart.)
    _big, _small, curves = _offset_pipes()
    checked = 0
    for curve in curves:
        for t in np.concatenate([np.logspace(-16, -.3, 60), 1 - np.logspace(-16, -.3, 60)]):
            x, y, z = curve.evaluate(t)
            theta = math.atan2(y, -z)
            if abs(math.remainder(theta, math.tau)) > .3:
                continue
            checked += 1
            exact = 2 * math.sqrt(2.) * abs(math.sin(.5 * theta)) * math.sqrt(math.cos(theta))
            assert abs(abs(x) - exact) <= 1e-14, (curve.parameterization, t, x, exact)
    assert checked > 100


@pytest.mark.parametrize("rotation", ["rational", "float"])
def test_branch_tangent_is_accurate_next_to_a_double_node(rotation):
    # Each branch lies in a vertical bisector plane, so its tangent's horizontal part is along that plane.
    # Before the fix the 0/0 quotient -(db z + dc) / root turned it by up to 60 degrees within 1e-8 of the node.
    branches, node = _node_branches(rotation)
    for curve, end in branches:
        far = curve.evaluate(.5) - node
        plane = far[:2] / np.linalg.norm(far[:2])
        for step in (0., 1e-16, 1e-15, 1e-13, 1e-11, 1e-9, 1e-7, 1e-5, 1e-3):
            derivative = curve.derivative(end + (step if end == 0. else -step))
            unit = derivative[:2] / np.linalg.norm(derivative[:2])
            assert abs(unit[0] * plane[1] - unit[1] * plane[0]) <= 1e-13, (step, derivative)


# ---- Codex review round 1 regressions

def test_far_apart_origins_do_not_certify_a_node():
    # Coplanar equal-radius axes whose elimination chart puts its fold 2.2e-6 (half-angle chart) away from
    # the node: certification must fail (an empty search window is not agreement).
    from anygeometry.analytic_supports import _certified_nodes
    from anygeometry.surfaces import Cylinder
    L = 1e11
    first = Cylinder((-4., 0., 14.), (1., 0., 0.), (0., 0., 1.), 6., 8.)
    second = Cylinder((-L / 2, L * math.sqrt(3.) / 2, 14.), (-.5, math.sqrt(3.) / 2, 0.), (0., 0., 1.), 6., 8.)
    assert _certified_nodes(first, second) == {}
    # ... while the hub and the offset pipes stay certified
    big, small, _curves = _offset_pipes()
    assert set(_certified_nodes(big, small)) == {-1}
    beams = _crossbeams()
    assert set(_certified_nodes(beams[0], beams[1])) == {-1, 1}


def _sampled_contained(curve, lower, upper):
    lo, hi = curve.bounds(lower, upper)
    for t in np.linspace(lower, upper, 33):
        point = curve.evaluate(t)
        assert np.all(lo <= point) and np.all(point <= hi), (curve.parameterization, lower, upper, t, point, lo, hi)


@pytest.mark.parametrize("fixture", ["hub", "pipes", "oblique"])
def test_bounds_contain_evaluated_points_near_chart_ends(fixture):
    curves = ([curve for curve, _end in _node_branches()[0]] if fixture == "hub"
              else _offset_pipes()[2] if fixture == "pipes" else _oblique_pipes()[2])
    for curve in curves:
        for step in (1e-15, 1e-12, 1e-9, 1e-6, 1e-3):
            _sampled_contained(curve, 0., step)
            _sampled_contained(curve, 1. - step, 1.)


def _oblique_pipes(tilt=.3):
    """The offset pipes with the small pipe's axis turned by ``tilt`` in the horizontal plane: the node stays
    an exact double (|a1 x a2| = cos(tilt) enters every term exactly) and a square chart starts at it."""
    from anygeometry.surfaces import Cylinder
    big = Cylinder((-4., 0., 0.), (1., 0., 0.), (0., 0., -1.), 2., 8.)
    axis = (math.sin(tilt), math.cos(tilt), 0.)
    small = Cylinder((-4. * axis[0], -4. * axis[1], -1.), axis, (math.cos(tilt), -math.sin(tilt), 0.), 1., 8.)
    curves = [c for c in cylinder_cylinder_support(big, small, tolerance=1e-9).curves
              if isinstance(c, CylinderIntersectionCurve)]
    return big, small, curves


def _forward_difference(function, t, step):
    """Second-order one-sided difference that stays inside [0, 1]."""
    sign = 1. if t + 2 * step <= 1. else -1.
    h = sign * step
    return sign * (-3 * function(t) + 4 * function(t + h) - function(t + 2 * h)) / (2 * step)


@pytest.mark.parametrize("fixture", ["pipes", "oblique"])
def test_first_derivative_is_accurate_on_every_chart_next_to_the_node(fixture):
    _big, _small, curves = _offset_pipes() if fixture == "pipes" else _oblique_pipes()
    checked = 0
    for curve in curves:
        for end in (0., 1.):
            if np.linalg.norm(curve.evaluate(end) - np.array((0., 0., -2.))) > 1e-12:
                continue
            for step in (1e-14, 1e-11, 1e-8):
                t = end + (step if end == 0. else -step)
                reference = _forward_difference(curve.evaluate, t, 1e-6)
                derivative = curve.derivative(t)
                assert np.linalg.norm(derivative - reference) <= 1e-6 * np.linalg.norm(reference), (
                    curve.parameterization, t, derivative, reference)
                checked += 1
    assert checked >= 6


def test_second_derivative_is_bounded_and_accurate_next_to_a_double_node():
    branches, _node = _node_branches()
    for curve, end in branches:
        for step in (1e-12, 1e-9, 1e-6):
            t = end + (step if end == 0. else -step)
            second = curve.second_derivative(t)
            reference = _forward_difference(curve.derivative, t, 1e-6)
            assert np.linalg.norm(second - reference) <= 1e-5 * max(1., np.linalg.norm(reference)), (
                t, second, reference)


@pytest.mark.parametrize("shift", [-.3, 3.1, 0.])
def test_tangent_plane_roots_at_a_node_are_the_node_for_transformed_branches(shift):
    # The plane z = zc - R + shift is tangent to the (translated) branch at its node: its only root near the
    # node is the node itself, not points 1.2e-7 m away (nor no root at all).
    from anygeometry.arrangement_geometry import plane_roots
    branches, node = _node_branches()
    matrix = ((1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., shift), (0., 0., 0., 1.))
    for curve, end in branches:
        moved = curve.transformed(matrix)
        target = node + (0., 0., shift)
        roots = plane_roots(moved, np.array((0., 0., 1.)), float(target[2]), tolerance=1e-9)
        near = [t for t in roots if np.linalg.norm(moved.evaluate(t) - target) < 1e-3]
        assert near, (shift, end, roots)
        for t in near:
            assert np.linalg.norm(moved.evaluate(t) - target) <= 1e-12, (shift, end, t, moved.evaluate(t))
