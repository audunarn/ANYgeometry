"""Contract tests of the private slab triangulation (plan N2): exact coverage, typed refusals."""
import itertools

import numpy as np
import pytest

from anygeometry import _slab_triangulation as slab
from anygeometry.display_tessellation import DisplayTrimRefused

SQUARE = [(0., 0.), (1., 0.), (1., 1.), (0., 1.)]


def _shoelace(ring):
    r = np.asarray(ring, float)
    r = r - r.min(axis=0)
    return 0.5 * abs(np.dot(r[:, 0], np.roll(r[:, 1], -1)) - np.dot(np.roll(r[:, 0], -1), r[:, 1]))


def _inside(point, ring):
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, np.roll(ring, -1, axis=0)):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _overlap(t1, t2):
    """True when two CCW triangles share interior area (separating-axis test, shared edges allowed)."""
    for tri, other in ((t1, t2), (t2, t1)):
        for k in range(3):
            p, q = tri[k], tri[(k + 1) % 3]
            e = q - p
            n = np.array([-e[1], e[0]])
            scale = np.abs(e).max() * max(1.0, np.abs(tri).max(), np.abs(other).max())
            side = (other - p) @ n
            if np.all(side <= 1e-12 * scale):
                return False
    return True


def _contract(rings, extra=None, exact=1e-12):
    rings = [np.asarray(r, float) for r in rings]
    result = slab.triangulate_rings(rings, extra)
    n = sum(len(r) for r in rings)
    vertices = np.vstack(rings)
    expected = _shoelace(rings[0]) - sum(_shoelace(r) for r in rings[1:])
    pts, tri = result.points, result.triangles
    # vertices only from the input rings (cut points lie on a ring segment)
    assert np.array_equal(pts[:n], vertices)
    for k, ((a, b), f) in enumerate(zip(result.cut_ends, result.cut_fraction)):
        assert 0.0 < f < 1.0
        assert np.allclose(pts[n + k], vertices[a] + f * (vertices[b] - vertices[a]), rtol=0, atol=1e-14)
    assert len(pts) == n + len(result.cut_ends)
    a, b, c = (pts[tri[:, k]] for k in range(3))
    doubled = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    assert np.all(doubled > 0.0)                                  # CCW, none degenerate
    assert 0.5 * doubled.sum() == pytest.approx(expected, rel=exact)
    assert result.area == pytest.approx(expected, rel=exact)
    for t in pts[tri].mean(axis=1):
        assert _inside(t, rings[0]) and not any(_inside(t, h) for h in rings[1:])
    if len(tri) <= 120:
        for i, j in itertools.combinations(range(len(tri)), 2):
            assert not _overlap(pts[tri[i]], pts[tri[j]]), (i, j)
    again = slab.triangulate_rings(rings, extra)
    assert np.array_equal(again.points, pts) and np.array_equal(again.triangles, tri)
    assert np.array_equal(again.cut_ends, result.cut_ends) and np.array_equal(again.cut_fraction, result.cut_fraction)
    return result


def test_refusal_is_the_typed_trim_invalid():
    assert issubclass(slab.SlabInvalid, DisplayTrimRefused)
    with pytest.raises(DisplayTrimRefused) as caught:
        slab.triangulate_rings([[(0., 0.), (1., 0.)]])
    assert caught.value.code == "TRIM_INVALID"


def test_duplicate_consecutive_points():
    ring = [(0., 0.), (0., 0.), (1., 0.), (1., 0.), (1., 1.), (0., 1.), (0., 1.)]
    _contract([ring])


def test_collinear_runs():
    ring = [(0., 0.), (.5, 0.), (1., 0.), (1., .5), (1., 1.), (.5, 1.), (0., 1.), (0., .5)]
    _contract([ring])


def test_vertical_edges_and_equal_x_vertices():
    ring = [(0., 0.), (2., 0.), (2., 1.), (1., 1.), (1., 2.), (0., 2.)]
    _contract([ring])
    _contract([ring[::-1]])
    comb = [(0., 0.), (3., 0.), (3., 2.), (2., 2.), (2., 1.), (1., 1.), (1., 2.), (0., 2.)]
    _contract([comb])


def test_a_hole_touching_nothing_and_many_holes():
    outer = [(0., 0.), (10., 0.), (10., 10.), (0., 10.)]
    holes = [[(1. + 2 * i, 1. + 2 * j), (1. + 2 * i, 2. + 2 * j), (2. + 2 * i, 2. + 2 * j), (2. + 2 * i, 1. + 2 * j)]
             for i in range(4) for j in range(4)]
    _contract([outer, holes[0]])
    _contract([outer, *holes])
    _contract([outer, *[h[::-1] for h in holes]])


def test_holes_sharing_x_with_each_other_and_the_outer_ring():
    outer = [(0., 0.), (4., 0.), (4., 4.), (0., 4.)]
    holes = [[(1., 1.), (2., 1.), (2., 2.)], [(1., 2.5), (2., 3.), (2., 3.5)]]
    _contract([outer, *holes])


@pytest.mark.parametrize("ring", [
    [(0., 0.), (1., 0.), (0., 1.)],
    [(0., 0.), (1., 0.), (.5, 1e-3)],
    [(2., 3.), (2., 5.), (7., 4.)],
])
def test_three_point_rings(ring):
    _contract([ring])
    _contract([ring[::-1]])


def test_far_from_the_origin_keeps_relative_accuracy():
    ring = np.array(SQUARE) * 1e-3 + 1e4
    _contract([ring], exact=1e-6)


@pytest.mark.parametrize("count", [4, 30, 200])
def test_monotone_fast_path_gives_the_general_path_coverage(count):
    u = np.linspace(0., 1., count)
    ring = np.vstack((np.column_stack((u, .1 * np.sin(3 * u))), np.column_stack((u[::-1], 1. + .2 * u[::-1]))))
    fast = slab._monotone(ring, None)
    small = slab._monotone_small(ring, None)
    assert fast is not None and small is not None
    expected = _shoelace(ring)
    for r in (fast, small):
        assert r.area == pytest.approx(expected, rel=1e-12)
    assert sorted(map(tuple, fast.triangles.tolist())) == sorted(map(tuple, small.triangles.tolist()))
    if count <= 30:
        _contract([ring])
    else:
        result = slab.triangulate_rings([ring])
        assert result.area == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("rings", [
    [],
    [[]],
    [[(0., 0.), (1., 0.)]],
    [[(0., 0.), (1., 0.), (2., 0.)]],                                   # zero area
    [[(0., 0.), (0., 0.), (0., 0.), (0., 0.)]],                         # one point repeated
    [[(0., 0.), (1., 1.), (1., 0.), (0., 1.)]],                         # bow tie
    [[(0., 0.), (4., 0.), (4., 4.), (0., 4.), (2., -1.), (3., 5.)]],    # self-intersecting
    [SQUARE, [(5., 5.), (6., 5.), (6., 6.)]],                           # hole outside
    [SQUARE, SQUARE],                                                   # hole equals outer
    [[(0., 0.), (4., 0.), (4., 4.), (0., 4.)], [(1., 1.), (3., 1.), (3., 3.)], [(1.5, 1.), (3.5, 1.), (3.5, 3.)]],
    [[(0., 0.), (1., 0.), (float("nan"), 1.)]],
    [[(0., 0.), (1., 0.), (float("inf"), 1.)]],
    [[(0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.), (0., 0., 1.), (1., 0., 1.)]],   # not 2-D points
    [[0., 0., 1., 0., 1., 1.]],                                         # flat list
    [[("a", "b"), ("c", "d"), ("e", "f")]],
], ids=lambda r: str(r)[:40])
def test_invalid_input_is_a_typed_refusal(rings):
    with pytest.raises(DisplayTrimRefused) as caught:
        slab.triangulate_rings(rings)
    assert caught.value.code == "TRIM_INVALID"


def test_invalid_extra_lines_are_refused():
    with pytest.raises(DisplayTrimRefused):
        slab.triangulate_rings([SQUARE], np.array([.5, float("nan")]))


def test_random_garbage_never_raises_untyped_or_returns_wrong_area():
    rng = np.random.default_rng(7)
    refused = ok = 0
    for _ in range(300):
        n = int(rng.integers(3, 9))
        rings = [rng.integers(0, 5, size=(n, 2)).astype(float)]
        if rng.random() < .4:
            rings.append(rng.integers(0, 5, size=(int(rng.integers(3, 6)), 2)).astype(float))
        try:
            result = slab.triangulate_rings(rings)
        except DisplayTrimRefused:
            refused += 1
            continue
        ok += 1
        a, b, c = (result.points[result.triangles[:, k]] for k in range(3))
        doubled = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        assert np.all(doubled > 0.0)
        expected = _shoelace(rings[0]) - sum(_shoelace(r) for r in rings[1:])
        assert 0.5 * doubled.sum() == pytest.approx(expected, rel=1e-9, abs=1e-12)
        for t in result.points[result.triangles].mean(axis=1):
            assert _inside(t, rings[0]) and not any(_inside(t, h) for h in rings[1:])
    assert refused > 0 and ok > 0
