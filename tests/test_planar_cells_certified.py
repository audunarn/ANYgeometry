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
