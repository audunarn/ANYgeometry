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
    with pytest.raises(GeometryError, match="not preserved"):
        certify(POINTS, edges, faces)
