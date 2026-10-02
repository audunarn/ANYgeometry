"""Junctions of a Bezier quadric branch curve with the other exact arrangement curves.

A branch of a Bezier wall cut with a quadric lies on a quadric and on a wall that is not one. Against every other
family the arrangement asks for all shared points: lines and ellipses through their planes, quadric branches through
their supports, Bezier paths through the roots on the quadric, and curves on the same quadric through certified box
subdivision. Each pair is checked against an independent oracle (a dense grid of the squared distance refined by
Newton iteration) in both orders, and the subdivision against its own contract: disjoint curves are proved disjoint,
transversal crossings are found to the tolerance, tangential contact is refused.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from anygeometry.analytic_supports import cylinder_cylinder_support
from anygeometry.arrangement_geometry import BezierPath, LinePath, curve_junctions
from anygeometry.branch_events import bezier_quadric_junctions, same_family, subdivision_junctions, world_quadric
from anygeometry.branch_supports import bezier_quadric_support
from anygeometry.errors import GeometryError
from anygeometry.exact_curves import CylinderIntersectionCurve, EllipticArc
from anygeometry.extrusions import BezierDirectrix
from anygeometry.quadric_curves import QuadricIntersectionCurve
from anygeometry.quadric_supports import quadric_pair_support
from anygeometry.surfaces import Cone, Cylinder, ExtrudedSurface

CUBIC = ((0., 0., 0.), (1., 2., 0.), (2., -1., 0.), (3., 1., 0.))
CROSSING = ((1., 1.5, 0.), (1.6, .5, 0.), (1.2, -.5, 0.), (1.8, -1.5, 0.))
VECTOR = (.25, 0., 1.5)


def wall(controls=CUBIC, vector=VECTOR, shift=(0., 0., 0.)):
    return ExtrudedSurface(BezierDirectrix(tuple(tuple(np.asarray(p) + shift) for p in controls)), vector)


def pipe(radius=.7, y=.4, z=.6, x=-2.):
    return Cylinder((x, y, z), (1., 0., 0.), (0., 1., 0.), radius, 8.)


def pipe_y(x=1.5, z=.7, radius=.3):
    return Cylinder((x, -2., z), (0., 1., 0.), (1., 0., 0.), radius, 6.)


def charts(first, second):
    return list(bezier_quadric_support(first, second).curves)


def on_pipe(angle, radius=.7, y=.4, z=.6):
    """A generator of ``pipe()`` at an angle, as a line."""
    return LinePath((-2., y + radius * math.cos(angle), z + radius * math.sin(angle)),
                    (6., y + radius * math.cos(angle), z + radius * math.sin(angle)))


def ring(x, radius=.7, y=.4, z=.6, start=0., sweep=math.tau):
    return EllipticArc((x, y, z), (0., radius, 0.), (0., 0., radius), start, sweep)


# ------------------------------------------------------------------------------------------------ the oracle


def sample(curve, n):
    return np.asarray(curve.evaluate(np.linspace(0., 1., n)))


def refine(first, second, t, u, iterations=40):
    """Gauss-Newton on ``|first(t) - second(u)|`` for many starting pairs at once (clamped to the unit square)."""
    t, u = np.array(t, dtype=float), np.array(u, dtype=float)
    for _ in range(iterations):
        difference = first.evaluate(t) - second.evaluate(u)
        a, b = first.derivative(t), -second.derivative(u)
        a11, a12, a22 = (a * a).sum(1), (a * b).sum(1), (b * b).sum(1)
        b1, b2 = -(a * difference).sum(1), -(b * difference).sum(1)
        det = a11 * a22 - a12 * a12
        det = np.where(np.abs(det) < 1e-300, 1e-300, det)
        t = np.clip(t + (b1 * a22 - b2 * a12) / det, 0., 1.)
        u = np.clip(u + (a11 * b2 - a12 * b1) / det, 0., 1.)
    return t, u, np.linalg.norm(first.evaluate(t) - second.evaluate(u), axis=1)


def oracle(first, second, n=600):
    """Shared points from the local minima of the squared distance on a grid, polished by Newton."""
    a, b = sample(first, n), sample(second, n)
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    padded = np.pad(d2, 1, constant_values=np.inf)
    local = np.ones_like(d2, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                local &= d2 <= padded[1 + di:1 + di + n, 1 + dj:1 + dj + n]
    spacing = max(float(np.linalg.norm(np.diff(a, axis=0), axis=1).max()), float(np.linalg.norm(np.diff(b, axis=0), axis=1).max()))
    i, j = np.nonzero(local & (d2 <= (2 * spacing) ** 2))
    if not len(i):
        return []
    t, u, residual = refine(first, second, i / (n - 1), j / (n - 1))
    found = []
    for s, v, r in sorted(zip(t, u, residual)):
        if r <= 1e-9 and not any(abs(s - p) < 1e-6 and abs(v - q) < 1e-6 for p, q in found):
            found.append((float(s), float(v)))
    return sorted(found)


def agree(first, second, tolerance=1e-10, n=600):
    expected = oracle(first, second, n)
    made = curve_junctions(first, second, tolerance=tolerance)
    for t, u in made:                                                    # nothing invented
        assert np.linalg.norm(first.evaluate(t) - second.evaluate(u)) <= 1e-8, (t, u)
    for t, u in expected:                                                # nothing missed
        assert any(abs(t - s) < 1e-6 and abs(u - v) < 1e-6 for s, v in made), (t, u, made)
    swapped = curve_junctions(second, first, tolerance=tolerance)
    assert len(swapped) == len(made)                                     # the same set from either order
    for t, u in made:
        assert any(abs(t - v) < 1e-9 and abs(u - s) < 1e-9 for s, v in swapped), (t, u, swapped)
    return made


def all_charts_pairs(first_charts, second_curves):
    return [(a, b) for a in first_charts for b in second_curves]


# ------------------------------------------------------------------------------------------------ lines, ellipses


@pytest.mark.parametrize("angle,crossings", [(.785, 1), (1.309, 3), (1.571, 3), (1.833, 1), (2.8, 0), (5.498, 1)])
def test_a_generator_of_the_pipe_meets_the_branches_where_the_pipe_meets_the_wall(angle, crossings):
    """A line lying on the branch's quadric is reached through two planes through it."""
    line = on_pipe(angle)
    total = 0
    for curve in charts(wall(), pipe()):
        total += len(agree(curve, line))
    assert total == crossings                                      # (the oracle counts the same crossings)


@pytest.mark.parametrize("x", [.4, 1.3, 2.2])
def test_a_ring_of_the_pipe_meets_the_branches_in_its_plane(x):
    arc = ring(x)
    total = 0
    for curve in charts(wall(), pipe()):
        total += len(agree(curve, arc))
    assert total == 1                                              # the oracle finds the same single crossing


def test_an_arc_of_the_ring_keeps_only_the_crossings_inside_its_sweep():
    whole, half = ring(1.3), ring(1.3, start=0., sweep=math.pi)
    full = sum(len(agree(c, whole)) for c in charts(wall(), pipe()))
    part = sum(len(agree(c, half)) for c in charts(wall(), pipe()))
    assert 0 < part <= full


# ------------------------------------------------------------------------------------------------ Bezier paths


def test_the_walls_own_boundary_curve_ends_the_branches_where_the_pipe_crosses_it():
    bottom = BezierPath(CUBIC)                                         # the wall's edge at ``s = 0`` (``z = 0``): inside the pipe's range
    top = BezierPath(tuple(tuple(np.asarray(p) + VECTOR) for p in CUBIC))
    hits = 0
    for curve in charts(wall(), pipe()):
        hits += len(agree(curve, bottom))
        assert agree(curve, top) == ()                                 # ``z = 1.5`` is above the pipe (``z <= 1.3``)
    assert hits >= 2


def test_a_straight_bezier_path_drawn_on_the_pipe_is_resolved_by_subdivision():
    """Its roots on the quadric vanish identically, so the elimination has nothing to say: the boxes decide."""
    start, end = np.array([-1., .4 + .7 * math.cos(1.2), .6 + .7 * math.sin(1.2)]), np.array([5., .4 + .7 * math.cos(1.2), .6 + .7 * math.sin(1.2)])
    path = BezierPath(tuple(tuple(start + k / 3 * (end - start)) for k in range(4)))
    hits = sum(len(agree(curve, path)) for curve in charts(wall(), pipe()))
    assert hits >= 1


# ------------------------------------------------------------------------------------------------ quadric branches


def quadric_branch_on(first, second, kind):
    result = quadric_pair_support(first, second)
    return [c for c in result.curves if isinstance(c, kind)]


def test_a_branch_of_two_other_quadrics_meets_the_curve_where_the_three_surfaces_meet():
    cone = Cone((1.2, -1., .9), (0., 1., 0.), (1., 0., 0.), .2, .6, 4.)             # a cone across the pipe, cutting the wall
    branches = quadric_branch_on(pipe(), cone, QuadricIntersectionCurve)
    assert branches
    total = sum(len(agree(c, b)) for c in charts(wall(), pipe()) for b in branches)
    assert total == 2                                                  # the oracle finds the same two points


def cylinder_branches():
    cross = pipe_y(x=1.5, z=1.1, radius=.3)                            # a thin pipe across the thick one near its top
    return [c for c in cylinder_cylinder_support(pipe(), cross).curves if isinstance(c, CylinderIntersectionCurve)]


def test_a_cylinder_cylinder_branch_meets_the_curve_where_the_three_surfaces_meet():
    branches = cylinder_branches()
    assert branches
    total = sum(len(agree(c, b)) for c in charts(wall(), pipe()) for b in branches)
    assert total == 2


def test_the_junctions_of_a_cylinder_branch_are_the_same_set_from_either_order():
    curve = charts(wall(), pipe())[2]
    for branch in cylinder_branches():
        forward = bezier_quadric_junctions(curve, branch)
        backward = bezier_quadric_junctions(branch, curve)
        assert len(backward) == len(forward)
        for t, u in forward:
            assert any(abs(t - v) < 1e-9 and abs(u - s) < 1e-9 for s, v in backward)


# ------------------------------------------------------------------------------------------------ two branches


def test_two_walls_on_the_same_pipe_are_resolved_by_subdivision_and_agree_with_the_oracle():
    first, second = charts(wall(), pipe()), charts(wall(CROSSING), pipe())
    assert first and second
    total = 0
    for a in first:
        for b in second:
            assert not same_family(a, b)
            total += len(agree(a, b))
    assert total >= 1                                                  # the walls cross, and so do their traces on the pipe


def test_two_walls_that_never_meet_are_proved_disjoint():
    near = charts(wall(shift=(0., .35, 0.)), pipe())
    assert near
    for a in charts(wall(), pipe()):
        for b in near:
            assert agree(a, b) == ()
            assert bezier_quadric_junctions(a, b) == ()


def test_one_wall_cut_by_two_pipes_is_resolved_exactly_through_the_roots_on_the_other_quadric():
    first, second = charts(wall(), pipe()), charts(wall(), pipe_y(x=1.5, z=1.1, radius=.3))
    assert first and second
    total = 0
    for a in first:
        for b in second:
            assert not same_family(a, b)
            total += len(agree(a, b))
    assert total == 2                                                  # on the wall, the two pipes' traces cross twice
    nested = charts(wall(), pipe_y(x=1.5, z=.7, radius=.3))           # a thin pipe wholly inside the thick one's trace
    assert sum(len(agree(a, b)) for a in first for b in nested) == 0


def test_charts_of_one_family_meet_only_at_their_ends():
    curves = charts(wall(), pipe())
    assert len(curves) >= 2
    for a in curves:
        for b in curves:
            if a is not b:
                assert same_family(a, b)
                assert bezier_quadric_junctions(a, b) == ()
                for t, u in curve_junctions(a, b):                     # what the arrangement names are chart ends
                    assert (t in (0., 1.)) or (u in (0., 1.))
                    assert np.linalg.norm(a.evaluate(t) - b.evaluate(u)) <= 1e-9


def test_a_second_wall_of_the_same_family_under_an_affine_image_is_another_family():
    shifted = [c.transformed(np.array([[1., 0., 0., .5], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]))
               for c in charts(wall(), pipe())]
    original = charts(wall(), pipe())
    assert not same_family(shifted[0], original[0]) and same_family(shifted[0], shifted[1])
    assert world_quadric(shifted[0]).value(shifted[0].evaluate(.5)) == pytest.approx(0., abs=1e-12)


# ------------------------------------------------------------------------------------------------ subdivision


def test_subdivision_finds_transversal_crossings_and_nothing_else():
    first = LinePath((0., 0., 0.), (4., 0., 0.))
    second = EllipticArc((2., 0., 0.), (0., 1., 0.), (0., 0., 1.), 0., math.tau)   # a circle in the plane x = 2, crossing the line once
    assert subdivision_junctions(first, second) == ()                  # through its centre, not across it
    across = LinePath((0., 1., 0.), (4., 1., 0.))                      # pierces the plane at (2, 1, 0), a point of the circle
    found = subdivision_junctions(across, second)
    assert [round(t, 9) for t, _u in found] == [.5, .5]                # the circle's start and end are the same point
    assert sorted(round(u, 9) for _t, u in found) == [0., 1.]
    for t, u in found:
        assert np.linalg.norm(across.evaluate(t) - second.evaluate(u)) <= 1e-10
    assert subdivision_junctions(first, EllipticArc((2., 0., 3.), (0., 1., 0.), (0., 0., 1.), 0., math.tau)) == ()


def test_subdivision_proves_curves_with_overlapping_boxes_disjoint():
    ellipse = EllipticArc((0., 0., 0.), (2., 0., 0.), (0., 2., 0.), 0., math.tau)
    inner = EllipticArc((0., 0., 0.), (1.5, 0., 0.), (0., 1.5, 0.), 0., math.tau)          # nested circles never meet
    assert subdivision_junctions(ellipse, inner) == ()


def test_subdivision_refuses_a_tangential_contact_instead_of_guessing():
    circle = EllipticArc((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), 0., math.tau)
    touching = EllipticArc((2., 0., 0.), (1., 0., 0.), (0., 1., 0.), 0., math.tau)          # externally tangent at one point
    with pytest.raises(GeometryError, match="tangential or overlapping"):
        subdivision_junctions(circle, touching, budget=4000)
    same = EllipticArc((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), 0., math.tau)
    with pytest.raises(GeometryError, match="tangential or overlapping"):
        subdivision_junctions(circle, same, budget=4000)


def test_subdivision_can_be_cancelled():
    first = LinePath((0., 1., 0.), (4., 1., 0.))
    second = EllipticArc((2., 0., 0.), (0., 1., 0.), (0., 0., 1.), 0., math.tau)
    with pytest.raises(GeometryError, match="cancelled"):
        subdivision_junctions(first, second, cancellation_check=lambda: True)


# ------------------------------------------------------------------------------------------------ refusals


def test_an_unsupported_partner_is_refused_with_a_typed_error():
    curve = charts(wall(), pipe())[0]
    with pytest.raises(GeometryError, match="not implemented"):
        bezier_quadric_junctions(curve, LinePath((0., 0., 0.), (1., 0., 0.)))      # lines have their own plane path


def test_the_generator_through_a_fold_touches_the_branch_once():
    """At a fold the branch is tangent to the wall's generator: the exact roots of the two planes through the line
    come out as a pair of crossings a square root of the rounding apart, which the arrangement must see as one."""
    seen = 0
    for curve in charts(wall(), pipe(radius=.3, y=.5, z=.7)):
        direction = np.asarray(curve.plan().floats().direction)
        for end in (0, 1):
            if not curve._is_fold_end(end):
                continue
            point = curve.evaluate(float(end))
            line = LinePath(tuple(point - .3 * direction), tuple(point + .9 * direction))
            junctions = curve_junctions(curve, line, tolerance=1e-9)
            assert len(junctions) == 1 and junctions[0][0] == float(end) and junctions[0][1] == pytest.approx(.25, abs=1e-9)
            seen += 1
    assert seen >= 4
