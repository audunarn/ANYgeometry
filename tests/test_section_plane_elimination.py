"""Exact equality elimination of the generating plane in ``branch_predicates._engine`` (UNRUN definitions).

``G2' = G2 - alpha G1`` has the same roots, signs and tangential derivatives on the true section, so planes
parallel to the generating plane become the nonzero constant ``c W``. The tests compare complete root sets with
independent analytic circle/line intersections and keep the coincidence refusal, the constant-projection
non-monotone answer, custom patches and the Work/depth behaviour.
"""
import math
from dataclasses import replace
from fractions import Fraction as Q

import pytest

from anygeometry import Cylinder, GeometryError, Plane
from anygeometry.branch_predicates import _eliminate_generating_plane, projection_monotone, section_roots
from anygeometry.nurbs import cylinder_to_nurbs
from anygeometry.plane_nurbs import CURRENT_WORK, PlaneNurbsNotCertified
from anygeometry.section_audit import Work
from anygeometry.support_curves import PlaneNurbsCurve
from test_plane_split import bowl

ZERO = Q(0)


def bowl_circle():
    """The circle x**2 + y**2 = 1/2 in the plane z = 1/2 (paraboloid z = x**2 + y**2)."""
    model, face, plane = bowl()
    return PlaneNurbsCurve.from_plane(plane, model.faces[face].surface, 0)


def cylinder_circle():
    """A rational (weighted) full cylinder of radius 2 cut by z = 1.2: the circle x**2 + y**2 = 4."""
    surface = cylinder_to_nurbs(Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 2., 3., 0., 2 * math.pi))
    plane = Plane((0., 0., 1.2), (1., 0., 0.), (0., 1., 0.))
    return PlaneNurbsCurve.from_plane(plane, surface, 0)


def roots_contain(curve, roots, points):
    """Every analytic point lies in some root's exact world enclosure (supplemental slack 1e-9)."""
    for point in points:
        assert any(all(float(lo) - 1e-9 <= x <= float(hi) + 1e-9 for x, lo, hi in zip(point, r.lower, r.upper))
                   for r in roots), point


def test_elimination_is_exact_and_zeroes_a_largest_component():
    n, d = (Q(1), Q(2), Q(3)), Q(5)
    m, d1 = (Q(0), Q(0), Q(2)), Q(1)
    new_n, new_d = _eliminate_generating_plane(n, d, m, d1)
    assert new_n == (Q(1), Q(2), Q(0)) and new_d == Q(5) - Q(3, 2) * Q(1)
    parallel_n, parallel_d = _eliminate_generating_plane((Q(0), Q(0), Q(3)), Q(3, 2) + Q(3, 1 << 20), m, d1)
    assert parallel_n == (Q(0), Q(0), Q(0)) and parallel_d != 0            # a nonzero constant times W


@pytest.mark.parametrize('curve_factory, level', ((bowl_circle, Q(1, 2)), (cylinder_circle, Q(float(1.2)))))
@pytest.mark.parametrize('shift', (Q(1, 1 << 20), -Q(1, 1 << 20), Q(1, 1 << 52), -Q(1, 1 << 52)))
@pytest.mark.parametrize('scale', (Q(1), Q(3)))
def test_parallel_displaced_planes_have_no_roots_on_both_sides(curve_factory, level, shift, scale):
    curve = curve_factory()
    assert section_roots(curve, (ZERO, ZERO, scale), scale * (level + shift)) == ()


def test_parallel_displaced_planes_need_only_a_tiny_work_budget():
    curve = bowl_circle()
    token = CURRENT_WORK.set(Work(2000, 32))
    try:
        assert section_roots(curve, (ZERO, ZERO, Q(1)), Q(1, 2) + Q(1, 1 << 30)) == ()
        assert section_roots(curve, (ZERO, ZERO, Q(1)), Q(1, 2) - Q(1, 1 << 30)) == ()
    finally:
        CURRENT_WORK.reset(token)


@pytest.mark.parametrize('scaled', (Q(1), Q(2)))
def test_coincident_plane_is_refused_before_elimination(scaled):
    curve = bowl_circle()
    with pytest.raises(PlaneNurbsNotCertified):
        section_roots(curve, (ZERO, ZERO, scaled), scaled * Q(1, 2))


def test_transverse_roots_match_the_analytic_circle_line_intersections():
    curve = bowl_circle()
    c = Q(float(.35))
    roots = section_roots(curve, (Q(1), ZERO, ZERO), c)                   # the line x = .35 in the plane z = .5
    assert section_roots(curve, (Q(1), ZERO, Q(2)), c + 1) == roots
    assert len(roots) == 2 and all(r.isolated for r in roots)
    y = math.sqrt(.5 - float(c) ** 2)
    roots_contain(curve, roots, [(float(c), y, .5), (float(c), -y, .5)])
    tangent_free = section_roots(curve, (Q(1), ZERO, ZERO), Q(2))          # x = 2: misses the circle
    assert tangent_free == ()


@pytest.mark.parametrize('reversed_domain', (False, True))
def test_weighted_rational_circle_roots_with_either_domain_direction(reversed_domain):
    curve = cylinder_circle()
    if reversed_domain:
        curve = replace(curve, domain=(1.0, 0.0))
    roots = section_roots(curve, (Q(1), ZERO, ZERO), Q(1))                # x = 1 meets x**2 + y**2 = 4 twice
    assert section_roots(curve, (Q(1), ZERO, Q(2)), Q(1) + 2 * Q(float(1.2))) == roots
    assert len(roots) == 2 and all(r.isolated for r in roots)
    y = math.sqrt(3.0)
    roots_contain(curve, roots, [(1.0, y, 1.2), (1.0, -y, 1.2)])
    for root in roots:                                                    # the root parameter evaluates onto the line
        t = float((root.t_lo + root.t_hi) / 2)
        assert abs(float(curve.evaluate(t)[0]) - 1.0) < 1e-6


def test_a_projection_constant_along_the_section_is_not_strictly_monotone():
    curve = bowl_circle()
    # A constant must never be admitted as strictly monotone. Bound the
    # negative decision rather than exhaust the full predicate allowance.
    for scale in (Q(1), Q(2)):
        token = CURRENT_WORK.set(Work(128, 16))
        try:
            assert projection_monotone(curve, (ZERO, ZERO, scale)) is None
        finally:
            CURRENT_WORK.reset(token)


def test_custom_supplied_patches_still_work_with_a_chart_segment_boundary():
    from anygeometry.section_events import section_segment_events
    model, face, plane = bowl()
    curve = PlaneNurbsCurve.from_plane(plane, model.faces[face].surface, 0)
    boundary = model.edges[model.faces[face].loop[0].edge].curve
    assert section_segment_events(curve, boundary) == ()                  # the bowl edge is far from the circle


def test_exhausted_box_budget_refuses_and_low_depth_preserves_root_accounting():
    curve = bowl_circle()
    c = Q(float(.35))
    token = CURRENT_WORK.set(Work(1, 32))
    try:
        with pytest.raises(GeometryError):
            section_roots(curve, (Q(1), ZERO, ZERO), c)
    finally:
        CURRENT_WORK.reset(token)
    token = CURRENT_WORK.set(Work(10**6, 1))
    try:
        # Depth is an execution bound, not an instruction to refuse an
        # already certified monotone crossing. Unresolved queries may refuse;
        # a successful shallow query must retain both actual circle roots.
        try:
            roots = section_roots(curve, (Q(1), ZERO, ZERO), c)
        except GeometryError:
            return
        assert len(roots) == 2
        y = math.sqrt(.5 - float(c) ** 2)
        roots_contain(curve, roots, [(float(c), y, .5), (float(c), -y, .5)])
    finally:
        CURRENT_WORK.reset(token)
