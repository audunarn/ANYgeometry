"""Slab (trapezoid) triangulation of trim rings (plan S2): exact coverage, no Shapely."""
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import display_tessellation_corpus as corpus

import importlib.util  # noqa: E402

if importlib.util.find_spec("anygeometry._slab_triangulation") is None:   # absent before S2, not broken
    pytest.skip("anygeometry._slab_triangulation is not implemented yet", allow_module_level=True)

from anygeometry import _slab_triangulation as slab  # noqa: E402


def _inside(point, ring):
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, np.roll(ring, -1, axis=0)):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _in_region(point, rings):
    return _inside(point, rings[0]) and not any(_inside(point, r) for r in rings[1:])


def _area(ring):
    ring = np.asarray(ring, float)
    ring = ring - ring.min(axis=0)
    return 0.5 * abs(np.dot(ring[:, 0], np.roll(ring[:, 1], -1)) - np.dot(np.roll(ring[:, 0], -1), ring[:, 1]))


def _doubled(points, triangles):
    a, b, c = (points[triangles[:, k]] for k in range(3))
    return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])


def _check(rings, result, rel=1e-12):
    rings = [np.asarray(r, float) for r in rings]
    expected = _area(rings[0]) - sum(_area(r) for r in rings[1:])
    doubled = _doubled(result.points, result.triangles)
    assert np.all(doubled > 0.0)
    assert 0.5 * doubled.sum() == pytest.approx(expected, rel=rel)
    assert result.area == pytest.approx(expected, rel=rel)
    centroids = result.points[result.triangles].mean(axis=1)
    assert all(_in_region(c, rings) for c in centroids)
    # cut points lie on their ring segment
    vertices = np.vstack(rings)
    n = len(vertices)
    for k, ((a, b), f) in enumerate(zip(result.cut_ends, result.cut_fraction)):
        assert 0.0 < f < 1.0
        assert np.allclose(result.points[n + k], vertices[a] + f * (vertices[b] - vertices[a]), rtol=0, atol=1e-15)
    _no_t_junctions(result)


def _no_t_junctions(result):
    """No point lies strictly inside a triangle edge; interior edges are shared by two triangles."""
    points, triangles = result.points, result.triangles
    edges = np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    for i, j in edges:
        p, q = points[i], points[j]
        d = q - p
        rel = points - p
        t = rel @ d / (d @ d)
        cross = rel[:, 0] * d[1] - rel[:, 1] * d[0]
        on = (np.abs(cross) <= 1e-13 * max(1.0, np.abs(d).max())) & (t > 1e-12) & (t < 1 - 1e-12)
        on[[i, j]] = False
        used = np.zeros(len(points), bool)
        used[np.unique(triangles)] = True
        assert not np.any(on & used), (i, j, np.flatnonzero(on & used))


def _f(*rings):
    """Rings in the input contract's form (owner decision N2 r4): a list of float64 (n, 2) ndarrays."""
    return [np.array(r, dtype=np.float64) for r in rings]


def test_l_shape_is_covered_exactly_with_nothing_in_its_notch():
    rings = _f(corpus.L_RING)
    result = slab.triangulate_rings(rings)
    _check(rings, result)
    centroids = result.points[result.triangles].mean(axis=1)
    assert not np.any((centroids[:, 0] > .4) & (centroids[:, 1] > .4))


@pytest.mark.parametrize("builder", [corpus.cone_comb_r1, corpus.cone_comb_r3, corpus.holed_cone_comb])
def test_lattice_combs_and_holes_are_covered_exactly(builder):
    rings = _f(*builder().info["rings"])
    for ordered in (rings, [r[::-1] for r in rings]):
        result = slab.triangulate_rings(ordered)
        _check(ordered, result)


def test_the_round_3_comb_keeps_its_whole_area():
    """ANYfem round 3: a Shapely-2.0 fallback left 25 % of this comb unfilled."""
    rings = _f(*corpus.cone_comb_r3().info["rings"])
    assert slab.triangulate_rings(rings).area == pytest.approx(6 / 256, rel=1e-12)


def _star(n, r0, r1, centre=(0., 0.), phase=0.):
    k = np.arange(2 * n)
    r = np.where(k % 2, r1, r0)
    a = phase + np.pi * k / n
    return np.column_stack((centre[0] + r * np.cos(a), centre[1] + r * np.sin(a)))


@pytest.mark.parametrize("seed", range(6))
def test_star_with_star_holes(seed):
    rng = np.random.default_rng(seed)
    outer = _star(9 + seed, 1.0, .55, phase=rng.uniform(0, 1))
    holes = [_star(5, .12, .06, (.0, .0), rng.uniform(0, 1))[::-1]]
    result = slab.triangulate_rings([outer, *holes])
    _check([outer, *holes], result, rel=1e-11)


def test_extra_lines_bound_the_slab_width():
    ring = np.array([(0., 0.), (1., 0.), (1., .2), (0., .2)])
    lines = np.linspace(0., 1., 11)
    result = slab.triangulate_rings([ring], lines)
    _check([ring], result)
    xs = np.unique(result.points[np.unique(result.triangles), 0])
    assert np.diff(xs).max() <= .1 + 1e-12


def test_a_thin_lens_is_kept():
    """Width .002, height ~2.5e-7: the lens survives, every triangle positive."""
    lower = [(.499, .4), (.5, .4), (.501, .4)]
    upper = [(.5, .4 + .5 * (1 - math.cos(.001)))]
    ring = np.array(lower + upper)
    _check([ring], slab.triangulate_rings([ring]), rel=1e-9)


@pytest.mark.parametrize("rings", [
    [[(0., 0.), (1., 1.), (1., 0.), (0., 1.)]],                                     # bow tie
    [[(0., 0.), (1., 0.), (1., 1.), (0., 1.)], [(2., 2.), (3., 2.), (3., 3.)]],     # hole outside
    [[(0., 0.), (4., 0.), (4., 4.), (0., 4.)], [(1., 1.), (3., 1.), (3., 3.)], [(1.5, 1.), (3.5, 1.), (3.5, 3.)]],
    [[(0., 0.), (1., 0.)]],                                                         # too few vertices
    [[(0., 0.), (1., 0.), (2., 0.)]],                                               # no area
], ids=("bowtie", "hole_outside", "overlapping_holes", "two_vertices", "collinear"))
def test_invalid_rings_are_refused(rings):
    with pytest.raises(slab.SlabInvalid):
        slab.triangulate_rings(_f(*rings))


def test_non_finite_coordinates_are_out_of_domain():
    """N2 r1 owner decision (2): the declared numeric domain is checked first (as in N1)."""
    from anygeometry.display_tessellation import DisplayDomainRefused
    with pytest.raises(DisplayDomainRefused):
        slab.triangulate_rings(_f([(0., 0.), (1., 0.), (float("nan"), 1.)]))


def test_no_shapely_is_imported():
    source = Path(slab.__file__).read_text(encoding="utf-8")
    assert "shapely" not in source.split('"""', 2)[2]
    probe = ("import sys; import anygeometry._slab_triangulation as s; "
             "import numpy as np; s.triangulate_rings([np.array([(0., 0.), (1., 0.), (0., 1.)])]); print('shapely' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


@pytest.mark.parametrize("count", [6, 40, 200])
def test_monotone_rings_take_the_fast_path_with_the_same_coverage(count):
    """x-monotone rings (most faces) skip the general sweep; small and large fast paths agree."""
    u = np.linspace(0., 1., count)
    lower = np.column_stack((u, .1 * np.sin(3 * u)))
    upper = np.column_stack((u[::-1] * .97 + .015, 1. + .2 * u))
    ring = np.vstack((lower, upper))
    lines = np.linspace(0., 1., 9)
    fast = slab._monotone_small(ring, lines) if 2 * count <= slab._SMALL else slab._monotone(ring, lines)
    assert fast is not None
    result = slab.triangulate_rings([ring], lines)
    _check([ring], result, rel=1e-12)
    assert sorted(map(tuple, result.triangles.tolist())) == sorted(map(tuple, fast.triangles.tolist()))
    both = slab._monotone(ring, lines), slab._monotone_small(ring, lines)
    assert np.array_equal(both[0].points, both[1].points)
    assert sorted(map(tuple, both[0].triangles.tolist())) == sorted(map(tuple, both[1].triangles.tolist()))


@pytest.mark.parametrize("ring", [
    [(0., 0.), (1., 0.), (1., 1.), (.5, .5), (0., 1.)],           # notch from above: monotone, two chains
    [(0., 0.), (1., 0.), (1., 1.), (.5, 1.), (.5, .5), (0., .5)],  # vertical step inside: general path
    [(0., 0.), (1., .5), (0., 1.), (.5, .5)],                      # arrow: not monotone
])
def test_non_trivial_rings_are_covered_either_way(ring):
    ring = np.asarray(ring)
    _check([ring], slab.triangulate_rings([ring]))
    _check([ring], slab.triangulate_rings([ring], _fast=False))
