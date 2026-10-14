"""Certified cells on the closed planar embedding (T-b/D8 slices E2 and E3).

References: hand-derived exact answers (cell counts, hole counts, exact areas),
main's answers where main is not rounding-arbitrary (touching holes), and
the domain validity rules of the design (S8).
"""
from __future__ import annotations

import pytest

from anygeometry import Plane
from anygeometry.arrangement_geometry import LinePath
from anygeometry.errors import GeometryError
from anygeometry.material_arrangement import ArrangementPath, MaterialDomain, arrange_material

PLANE = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
SQUARE = [(0., 0.), (4., 0.), (4., 4.), (0., 4.)]


def loop(points):
    points = [tuple(map(float, p)) for p in points]
    return tuple(ArrangementPath(LinePath((*a, 0.), (*b, 0.))) for a, b in zip(points, points[1:] + points[:1]))


def trace(a, b, k=2):
    return ArrangementPath(LinePath((*map(float, a), 0.), (*map(float, b), 0.)), owners=(1, k))


def run(loops, traces=(), tol=1e-6, **kw):
    return arrange_material(MaterialDomain(1, PLANE, tuple(loops)), tuple(traces), tolerance=tol, **kw)


@pytest.mark.parametrize("start", range(4))
@pytest.mark.parametrize("reverse", (False, True))
def test_thin_cell_along_the_boundary_for_every_start_vertex(start, reverse):
    """A trace 1.2 tol above the bottom edge: main gave 2 cells for one start vertex of the outer loop
    and "failed material conservation" for the other three (uncertified probe); here 2 cells, area 16."""
    a, b = (-1., 1.2e-6), (5., 1.2e-6)
    result = run([loop(SQUARE[start:] + SQUARE[:start])], [trace(*((b, a) if reverse else (a, b)))],
                 area_tolerance=1e-12)
    assert (len(result.cells), result.area) == (2, 16.)


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("hole,area", (([(0., 0.), (1., .5), (.5, 1.)], 15.625),
                                        ([(2., 0.), (3., 1.), (1., 1.)], 15.)))
@pytest.mark.parametrize("cut", (False, True))
def test_holes_touching_the_outer_loop_at_a_vertex(reverse, hole, area, cut):
    """As main: a hole touching the outer corner or an outer edge at one vertex, either winding, with
    and without a crossing trace."""
    traces = [trace((-1., 2.), (5., 2.))] if cut else []
    result = run([loop(SQUARE), loop(hole[::-1] if reverse else hole)], traces)
    assert (len(result.cells), result.area) == (2 if cut else 1, area)
    assert all(len(cell.holes) == 0 for cell in result.cells)


def test_two_holes_on_one_horizontal():
    result = run([loop(SQUARE), loop([(.5, 1.), (1.5, 1.), (1., 2.)][::-1]), loop([(2.5, 1.), (3.5, 1.), (3., 2.)][::-1])])
    assert (len(result.cells), sorted(len(c.holes) for c in result.cells), result.area) == (1, [2], 15.)


def test_ownership_ray_through_a_vertex():
    """The leftmost vertex of the inner square's outline lies on the horizontal through a vertex of
    an island triangle to its left: the ray's first hit is that vertex (wedge rule)."""
    island = [trace(a, b, 10 + k) for k, (a, b) in enumerate((((.5, 1.), (1., 2.)), ((1., 2.), (.5, 3.)),
                                                             ((.5, 3.), (.5, 1.))))]
    inner = [trace(a, b, 20 + k) for k, (a, b) in enumerate((((2., 2.), (3., 1.5)), ((3., 1.5), (3., 2.5)),
                                                             ((3., 2.5), (2., 2.))))]
    result = run([loop(SQUARE)], island + inner, area_tolerance=1e-12)
    assert len(result.cells) == 3
    assert sorted(len(cell.holes) for cell in result.cells) == [0, 0, 2]
    assert result.area == 16.


@pytest.mark.parametrize("loops,traces", (
    # an exterior "hole" whose vertices all lie on the outer notch boundary (S8 rule 3)
    ([[(0, 0), (6, 0), (6, 6), (4, 6), (4, 2), (2, 2), (2, 6), (0, 6)], [(2, 3), (4, 4), (3, 2)]], []),
    # a self-crossing outer loop (S8 rule 0, stored)
    ([[(0., 0.), (4., 4.), (0., 4.), (2., 0.)]], []),
    # a simple stored loop made self-touching by re-anchoring (S8 rule 0, after S3-S5)
    ([[(0., 0.), (6., 0.), (6., 6.), (4., 6.), (3., 5e-7), (2., 6.), (0., 6.)]], [((3., -1.), (3., -.5))]),
    # a hole enclosing another hole
    ([SQUARE, [(1., 1.), (3., 1.), (3., 3.), (1., 3.)][::-1], [(1.5, 1.5), (2.5, 1.5), (2.5, 2.5), (1.5, 2.5)][::-1]], []),
    # a hole sharing an edge with the outer loop (coincident occurrences, owner Q-c)
    ([SQUARE, [(0., 0.), (1., 0.), (0., 1.)]], []),
))
def test_invalid_material_domains_refuse_before_classification(loops, traces):
    with pytest.raises(GeometryError, match="invalid material domain"):
        run([loop(points) for points in loops], [trace(a, b) for a, b in traces])


@pytest.mark.parametrize("scale", (1e-100, 1e3, 1e8))
def test_plate_crossing_has_the_scale_one_topology_at_any_scale(scale):
    """T-b/D8 E5: the plane x plane trace comes from the exact line of the two plane records with a
    box-relative extent. Main's absolute extent floor (+1.) made the rounded trace pass through the
    origin at 1e-100 ("unresolved bridge"); now every scale gives the same shape."""
    import sys
    from pathlib import Path
    from anygeometry import apply_intersections, plan_intersections
    sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
    import adversarial_corpus
    model, context = adversarial_corpus.scaled_crossing(scale)()
    apply_intersections(model, plan_intersections(model, [context["first"], context["second"]], policy="connect"),
                        policy="connect")
    assert (len(model.faces), len(model.edges)) == (8, 23)


def test_an_off_support_trace_absorbed_into_the_boundary_refuses():
    """Review (round 1): boundary at z = 0.75 tol, a coincident bottom-edge trace at z = 1.5 tol
    (off the support z = 0 by more than tol) -- refused before its stations merge into the boundary."""
    tol = 1e-6
    lift = lambda points, z: tuple(ArrangementPath(LinePath((*a, z), (*b, z))) for a, b in zip(points, points[1:] + points[:1]))
    domain = MaterialDomain(1, PLANE, (lift(SQUARE, .75 * tol),))
    trace_ = ArrangementPath(LinePath((-1., 0., 1.5 * tol), (5., 0., 1.5 * tol)), owners=(1, 2))
    with pytest.raises(GeometryError, match="off its support"):
        arrange_material(domain, (trace_,), tolerance=tol)


def test_chart_coincident_stations_are_one_vertex():
    """Review (round 1, S3): square at z = +0.75 tol and a corner-touching triangular hole at
    z = -0.75 tol: the contact corners coincide in the chart (1.5 tol apart in the world) and are
    one vertex, as main's answer for the coplanar fixture (1 cell, area 15.625)."""
    tol = 1e-6
    lift = lambda points, z: tuple(ArrangementPath(LinePath((*a, z), (*b, z))) for a, b in zip(points, points[1:] + points[:1]))
    domain = MaterialDomain(1, PLANE, (lift(SQUARE, .75 * tol), lift([(0., 0.), (1., .5), (.5, 1.)], -.75 * tol)))
    result = arrange_material(domain, (), tolerance=tol)
    assert (len(result.cells), result.area) == (1, 15.625)
