"""Certification of persisted cells (T-b/D8 slice E2a, design S10).

``certify_persisted_cells`` re-checks applied cells on their stored
coordinates: a planar embedding, simple loops, consistent area signs,
holes inside their outer loop and no overlapping face regions. The cases
are constructed so that one property breaks while the others hold.
"""
from __future__ import annotations

import pytest

from anygeometry import Plane
from anygeometry.errors import GeometryError
from anygeometry.material_arrangement import certify_persisted_cells

PLANE = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))


def build(points, edges):
    """Return (edge_ends, position) callables for vertex positions and edges given as vertex pairs."""
    positions = {name: (float(x), float(y), 0.) for name, (x, y) in points.items()}
    ends = dict(enumerate(edges))
    return (lambda edge: ends[edge]), (lambda vertex: positions[vertex])


def loop(edges, sequence):
    """Oriented edges of the cycle through ``sequence`` (vertex names)."""
    result = []
    for a, b in zip(sequence, sequence[1:] + sequence[:1]):
        index = next(i for i, e in enumerate(edges) if set(e) == {a, b})
        result.append((index, edges[index] == (a, b)))
    return result


# Two unit squares side by side; the left one has a square hole filled by an island face.
POINTS = {"a": (0, 0), "b": (2, 0), "c": (4, 0), "d": (4, 2), "e": (2, 2), "f": (0, 2),
          "p": (.5, .5), "q": (1.5, .5), "r": (1.5, 1.5), "s": (.5, 1.5)}
EDGES = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f"), ("f", "a"), ("b", "e"),
         ("p", "q"), ("q", "r"), ("r", "s"), ("s", "p")]


def cells(edges=EDGES):
    left = (loop(edges, ["a", "b", "e", "f"]), [loop(edges, ["p", "s", "r", "q"])])
    right = (loop(edges, ["b", "c", "d", "e"]), [])
    island = (loop(edges, ["p", "q", "r", "s"]), [])
    return [left, right, island]


def certify(points, edges=EDGES, faces=None):
    edge_ends, position = build(points, edges)
    certify_persisted_cells(PLANE, faces if faces is not None else cells(edges), edge_ends, position)


def test_a_valid_subdivision_with_a_filled_hole_passes():
    certify(POINTS)


def test_a_vertex_moved_onto_another_edge_refuses():
    moved = dict(POINTS, q=(2., .5))                     # the island corner lands on the shared edge b-e
    with pytest.raises(GeometryError, match="cross or overlap"):
        certify(moved)


def test_a_changed_cyclic_order_without_collapse_refuses():
    moved = dict(POINTS, e=(2., -.5))                     # e swings below b: the order at b changes
    with pytest.raises(GeometryError, match="not preserved"):
        certify(moved)


def test_an_island_moved_into_the_neighbouring_face_refuses():
    """Planarity, loop simplicity and area signs survive; only the nesting changes."""
    moved = dict(POINTS, p=(2.5, .5), q=(3.5, .5), r=(3.5, 1.5), s=(2.5, 1.5))
    with pytest.raises(GeometryError, match="hole outside its outer loop|faces overlap"):
        certify(moved)


def test_a_collapsed_edge_refuses():
    moved = dict(POINTS, q=(.5, .5))
    with pytest.raises(GeometryError, match="collapsed edge"):
        certify(moved)


def test_a_non_simple_loop_refuses():
    edges = EDGES + [("a", "e")]
    faces = [(loop(edges, ["a", "b", "e", "f"]) + loop(edges, ["a", "b", "e", "f"])[:0]
              + [(len(EDGES), True), (len(EDGES), False)], [])]
    with pytest.raises(GeometryError, match="slit in a cell"):
        certify(POINTS, edges, faces)



def test_overlapping_rectangles_sharing_split_edges_refuse():
    """Review (E2a round 1): [0,2]x[0,1] and [1,3]x[0,1], boundaries split at x = 1 and x = 2, walks
    starting at (0,0) and (2,0): their interiors overlap; the shared pieces are used in the same
    direction by both cells."""
    points = {"a": (0, 0), "b": (1, 0), "c": (2, 0), "d": (3, 0), "e": (3, 1), "f": (2, 1), "g": (1, 1), "h": (0, 1)}
    edges = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f"), ("f", "g"), ("g", "h"), ("h", "a"),
             ("c", "f"), ("b", "g")]
    faces = [(loop(edges, ["a", "b", "c", "f", "g", "h"]), []), (loop(edges, ["c", "d", "e", "f", "g", "b"]), [])]
    with pytest.raises(GeometryError, match="overlapping cells|cross or overlap"):
        certify(points, edges, faces)


def test_a_hole_inside_another_hole_refuses():
    points = {"a": (0, 0), "b": (10, 0), "c": (10, 10), "d": (0, 10),
              "p": (1, 1), "q": (9, 1), "r": (9, 9), "s": (1, 9),
              "u": (2, 2), "v": (3, 2), "w": (3, 3), "x": (2, 3)}
    edges = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"), ("p", "q"), ("q", "r"), ("r", "s"), ("s", "p"),
             ("u", "v"), ("v", "w"), ("w", "x"), ("x", "u")]
    faces = [(loop(edges, ["a", "b", "c", "d"]), [loop(edges, ["p", "s", "r", "q"]), loop(edges, ["u", "x", "w", "v"])])]
    with pytest.raises(GeometryError, match="hole inside another hole"):
        certify(points, edges, faces)


def test_a_touching_hole_absorbed_into_the_outer_walk_is_accepted():
    """E3 walks a hole touching the outer corner into the outer loop (the corner is visited twice);
    the persisted cell is valid and certifies on its unchanged coordinates."""
    from anygeometry.arrangement_geometry import LinePath
    from anygeometry.material_arrangement import ArrangementPath, MaterialDomain, arrange_material

    def ring(points):
        return tuple(ArrangementPath(LinePath((*a, 0.), (*b, 0.))) for a, b in zip(points, points[1:] + points[:1]))
    arrangement = arrange_material(MaterialDomain(1, PLANE, (ring([(0., 0.), (4., 0.), (4., 4.), (0., 4.)]),
                                                             ring([(0., 0.), (1., .5), (.5, 1.)]))),
                                   (), tolerance=1e-6)
    assert arrangement.certified and len(arrangement.cells) == 1
    vertex_ids, positions, ends = {}, {}, {}
    for index, path in enumerate(arrangement.paths):
        pair = []
        for point in (path.curve.start, path.curve.end):
            key = tuple(point)
            vertex_ids.setdefault(key, len(vertex_ids))
            positions[vertex_ids[key]] = key
            pair.append(vertex_ids[key])
        ends[index] = tuple(pair)
    cells = [(list(cell.outer), [list(hole) for hole in cell.holes]) for cell in arrangement.cells]
    certify_persisted_cells(PLANE, cells, lambda edge: ends[edge], lambda vertex: positions[vertex])


def _touching_hole_arrangement():
    from anygeometry.arrangement_geometry import LinePath
    from anygeometry.material_arrangement import ArrangementPath, MaterialDomain, arrange_material

    def ring(points):
        return tuple(ArrangementPath(LinePath((*a, 0.), (*b, 0.))) for a, b in zip(points, points[1:] + points[:1]))
    return arrange_material(MaterialDomain(1, PLANE, (ring([(0., 0.), (4., 0.), (4., 4.), (0., 4.)]),
                                                      ring([(0., 0.), (1., .5), (.5, 1.)]))), (), tolerance=1e-6)


def _persisted(arrangement, move=lambda key: key):
    vertex_ids, positions, ends = {}, {}, {}
    for index, path in enumerate(arrangement.paths):
        pair = []
        for point in (path.curve.start, path.curve.end):
            key = tuple(point)
            if key not in vertex_ids:
                vertex_ids[key] = len(vertex_ids)
                positions[vertex_ids[key]] = move(key)
            pair.append(vertex_ids[key])
        ends[index] = tuple(pair)
    cells = [(list(cell.outer), [list(hole) for hole in cell.holes]) for cell in arrangement.cells]
    return cells, (lambda edge: ends[edge]), (lambda vertex: positions[vertex])


def test_a_touching_hole_lobe_moved_outside_refuses():
    """Review (round 2): negate the two non-contact triangle vertices -- the clockwise lobe leaves the
    square while every sign stays the same."""
    from anygeometry.material_arrangement import expected_cell_signatures
    arrangement = _touching_hole_arrangement()
    move = lambda key: (-key[0], -key[1], key[2]) if key[:2] in ((1., .5), (.5, 1.)) else key
    cells, ends, position = _persisted(arrangement, move)
    with pytest.raises(GeometryError, match="lobe outside its loop"):
        certify_persisted_cells(PLANE, cells, ends, position, expected=expected_cell_signatures(arrangement))


def test_a_reflected_face_refuses_against_its_certified_winding():
    from anygeometry.material_arrangement import expected_cell_signatures
    arrangement = _touching_hole_arrangement()
    cells, ends, position = _persisted(arrangement, lambda key: (-key[0], key[1], key[2]))
    with pytest.raises(GeometryError, match="winding or lobes changed"):
        certify_persisted_cells(PLANE, cells, ends, position, expected=expected_cell_signatures(arrangement))
    cells, ends, position = _persisted(arrangement)
    certify_persisted_cells(PLANE, cells, ends, position, expected=expected_cell_signatures(arrangement))


def test_nested_touching_hole_lobes_refuse():
    """Review (round 3): two holes touching the square's corner become two lobes of the outer walk;
    moving hole B inside hole A (contact vertex kept) must refuse, as constructing that domain does."""
    from anygeometry.arrangement_geometry import LinePath
    from anygeometry.material_arrangement import (ArrangementPath, MaterialDomain, arrange_material,
                                                  expected_cell_signatures)

    def ring(points):
        return tuple(ArrangementPath(LinePath((*a, 0.), (*b, 0.))) for a, b in zip(points, points[1:] + points[:1]))
    arrangement = arrange_material(MaterialDomain(1, PLANE, (ring([(0., 0.), (4., 0.), (4., 4.), (0., 4.)]),
                                                             ring([(0., 0.), (2., .4), (1., .5)]),
                                                             ring([(0., 0.), (.5, 1.), (.4, 2.)]))), (), tolerance=1e-6)
    assert len(arrangement.cells) == 1 and arrangement.area == 15.4
    cells, ends, position = _persisted(arrangement)
    certify_persisted_cells(PLANE, cells, ends, position, expected=expected_cell_signatures(arrangement))
    moved = {(.5, 1.): (1., .22), (.4, 2.): (.7, .25)}
    cells, ends, position = _persisted(arrangement, lambda key: (*moved.get(key[:2], key[:2]), key[2]))
    with pytest.raises(GeometryError, match="hole inside another hole"):
        certify_persisted_cells(PLANE, cells, ends, position, expected=expected_cell_signatures(arrangement))
