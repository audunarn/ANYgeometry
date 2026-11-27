"""Contract tests of the private slab triangulation (plan N2): exact coverage, typed refusals.

The oracle here is independent of the module: exact rational (``fractions.Fraction``) predicates
with a float filter whose error bound is Shewchuk's ``orient2d`` bound, and its own brute-force
simplicity check.  N2 round-1 (Codex) reproductions are the ``test_r1_*`` tests.
"""
from __future__ import annotations

import math
from collections import defaultdict
from fractions import Fraction

import numpy as np
import pytest

from anygeometry import _slab_triangulation as slab
from anygeometry.display_tessellation import DisplayTrimRefused

F = Fraction
SQUARE = [(0., 0.), (1., 0.), (1., 1.), (0., 1.)]
SQ = np.array(SQUARE)
#: the input-contract refusal message (owner decision N2 r4 (1))
_FORM = "input must be a list/tuple of float64 (n, 2) ndarrays"
_EPS = 2.0 ** -52
_ORIENT_BOUND = (3.0 + 16.0 * 2.0 ** -53) * 2.0 ** -53       # Shewchuk ccwerrboundA
_TINY = 2.0 ** -1000
_TRIANGULATE = slab.triangulate_rings                      # unwrapped (the fixture below wraps it)


@pytest.fixture(autouse=True)
def _caller_raises_on_every_floating_point_event(monkeypatch):
    """Owner decision N2 r2 (3): results and refusals do not depend on the caller's NumPy error
    state, so every call in these tests runs under ``np.errstate(all="raise")``."""
    def strict(*args, **kwargs):
        with np.errstate(all="raise"):
            return _TRIANGULATE(*args, **kwargs)
    monkeypatch.setattr(slab, "triangulate_rings", strict)


def _f(*rings):
    """Rings in the input contract's form (owner decision N2 r4): a list of float64 (n, 2) ndarrays."""
    return [np.array(r, dtype=np.float64) for r in rings]


def _fx(values):
    """``extra_x`` in the input contract's form: None or a float64 1-D ndarray."""
    return None if values is None else np.array(values, dtype=np.float64).reshape(-1)


# ----------------------------------------------------------------------------- exact oracle
def _fp(p):
    return (F(float(p[0])), F(float(p[1])))


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _orient_signs(a, b, c):
    """Exact sign of orient(a, b, c) for float point arrays (k, 2): filter, then rationals."""
    t1 = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
    t2 = (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    det = t1 - t2
    # the absolute term covers underflow: subnormal products make the relative bound round to 0
    sign = np.where(np.abs(det) > _ORIENT_BOUND * (np.abs(t1) + np.abs(t2)) + _TINY, np.sign(det), 2)
    sign = sign.astype(np.int8)
    for k in np.flatnonzero(sign == 2):
        e = _orient(_fp(a[k]), _fp(b[k]), _fp(c[k]))
        sign[k] = (e > 0) - (e < 0)
    return sign


def _compress(ring):
    """Ring as float tuples without repeated consecutive points (cyclic)."""
    pts = [tuple(map(float, p)) for p in np.asarray(ring, float)]
    out = [p for k, p in enumerate(pts) if p != pts[k - 1]]
    return out if out else pts[:1]


def _signed_area(points):
    s = F(0)
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        s += x1 * y2 - x2 * y1
    return s / 2


class _Ring:
    """Exact closed point location against one ring (rational query points)."""

    def __init__(self, points):
        p = np.asarray(points, float)
        q = np.roll(p, -1, axis=0)
        self.f = [_fp(v) for v in p]
        self.ylo, self.yhi = np.minimum(p[:, 1], q[:, 1]), np.maximum(p[:, 1], q[:, 1])
        self.xhi = np.maximum(p[:, 0], q[:, 0])

    def locate(self, qx, qy):
        """1 strictly inside, 0 on the boundary, -1 outside."""
        fx, fy = float(qx), float(qy)
        sy, sx = abs(fy) * 1e-12 + 1e-300, abs(fx) * 1e-12 + 1e-300
        m = len(self.f)
        inside = False
        for k in np.flatnonzero((self.ylo <= fy + sy) & (self.yhi >= fy - sy) & (self.xhi >= fx - sx)):
            (x1, y1), (x2, y2) = self.f[k], self.f[(k + 1) % m]
            if (_orient((x1, y1), (x2, y2), (qx, qy)) == 0 and min(x1, x2) <= qx <= max(x1, x2)
                    and min(y1, y2) <= qy <= max(y1, y2)):
                return 0
            if (y1 > qy) != (y2 > qy) and qx < x1 + (qy - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
        return 1 if inside else -1


def _locate_region(rings, qx, qy):
    where = rings[0].locate(qx, qy)
    if where <= 0:
        return where
    for hole in rings[1:]:
        h = hole.locate(qx, qy)
        if h >= 0:
            return -h if h > 0 else 0
    return 1


def _segments_meet(p1, p2, p3, p4):
    """Closed segments p1p2 and p3p4 share a point (exact)."""
    d1, d2 = _orient(p3, p4, p1), _orient(p3, p4, p2)
    d3, d4 = _orient(p1, p2, p3), _orient(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    def on(a, b, c):
        return min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
    return ((d1 == 0 and on(p3, p4, p1)) or (d2 == 0 and on(p3, p4, p2)) or (d3 == 0 and on(p1, p2, p3))
            or (d4 == 0 and on(p1, p2, p4)))


def brute_valid(rings) -> bool:
    """Independent exact check of the input contract (all pairs: for small fixtures only)."""
    rings = [_compress(r) for r in rings]
    if any(len(r) < 3 for r in rings):
        return False
    segs = []
    for r, ring in enumerate(rings):
        m = len(ring)
        for k in range(m):
            segs.append((r, k, m, _fp(ring[k]), _fp(ring[(k + 1) % m])))
    for i in range(len(segs)):
        ri, ki, mi, a, b = segs[i]
        for j in range(i + 1, len(segs)):
            rj, kj, mj, c, d = segs[j]
            if (max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0])
                    or max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1])):
                continue
            if ri == rj and (kj == ki + 1 or (ki == 0 and kj == mi - 1)):
                # ring neighbours share exactly one endpoint; only a retrace along it is invalid
                p, q, s = (a, b, d) if kj == ki + 1 else (c, d, b)
                if _orient(p, q, s) == 0 and (q[0] - p[0]) * (s[0] - q[0]) + (q[1] - p[1]) * (s[1] - q[1]) < 0:
                    return False
                continue
            if _segments_meet(a, b, c, d):
                return False
    if _signed_area([_fp(p) for p in rings[0]]) == 0:
        return False
    exact = [_Ring(r) for r in rings]
    for h, hole in enumerate(rings[1:], start=1):
        x, y = _fp(hole[0])
        if exact[0].locate(x, y) <= 0:
            return False
        if any(exact[g].locate(x, y) > 0 for g in range(1, len(rings)) if g != h):
            return False
    return True


def _rounded_ring_ids(rings, result):
    """Point ids of the rings with each cut point inserted on its segment (tracked by the cut's
    segment ``cut_ends[k]`` and its index ``k``, never by coordinates: N2 round 3, Low b)."""
    vertices = np.vstack([np.asarray(r, float) for r in rings])
    n = len(vertices)
    groups = defaultdict(list)
    for k, (a, b) in enumerate(result.cut_ends.tolist()):
        groups[(a, b)].append(n + k)
    used = 0
    out = []
    offset = 0
    for ring in rings:
        pts = [tuple(map(float, p)) for p in np.asarray(ring, float)]
        ids = [offset + k for k, p in enumerate(pts) if p != pts[k - 1]] or [offset]
        offset += len(pts)
        new = []
        for p, q in zip(ids, ids[1:] + ids[:1]):
            new.append(p)
            ks = groups.pop((p, q), []) + groups.pop((q, p), [])
            used += len(ks)
            ks.sort(key=lambda k: result.points[k, 0], reverse=bool(vertices[p, 0] > vertices[q, 0]))
            new.extend(ks)
        out.append(new)
    assert used == len(result.cut_ends), "a cut point lies on no ring segment"
    return out


def _rounded_rings(rings, result):
    """The rings with each cut point inserted on its segment: the polygon the mesh must tile."""
    return [[tuple(result.points[i]) for i in ring] for ring in _rounded_ring_ids(rings, result)]


def _area_allowance(rings, result, bounds):
    """The stated area bound: each cut point k moved by ``bounds[k]`` changes the area by that times
    half the x distance of its two neighbours IN ITS OWN ring (each cut counted once)."""
    n = sum(len(r) for r in rings)
    allowance = F(0)
    for ring in _rounded_ring_ids(rings, result):
        m = len(ring)
        for i, pid in enumerate(ring):
            if pid >= n:
                span = abs(F(float(result.points[ring[(i + 1) % m], 0])) - F(float(result.points[ring[i - 1], 0])))
                allowance += bounds[pid - n] * span / 2
    return allowance


def _candidate_pairs(boxes):
    """Index pairs of boxes (xmin, xmax, ymin, ymax) whose interiors overlap."""
    order = np.argsort(boxes[:, 0], kind="stable")
    b = boxes[order]
    stop = np.searchsorted(b[:, 0], b[:, 1], side="left")
    counts = np.maximum(stop - np.arange(len(b)) - 1, 0)
    i = np.repeat(np.arange(len(b)), counts)
    j = i + 1 + (np.arange(len(i)) - np.repeat(np.cumsum(counts) - counts, counts))
    keep = (b[j, 2] < b[i, 3]) & (b[i, 2] < b[j, 3]) & (b[j, 0] < b[i, 1])
    return order[i[keep]], order[j[keep]]


def _interiors_disjoint(points, triangles):
    t = points[triangles]
    boxes = np.column_stack((t[:, :, 0].min(1), t[:, :, 0].max(1), t[:, :, 1].min(1), t[:, :, 1].max(1)))
    i, j = _candidate_pairs(boxes)
    if not len(i):
        return True, 0
    separated = np.zeros(len(i), dtype=bool)
    for first, second in ((i, j), (j, i)):
        for k in range(3):
            p, q = t[first, k], t[first, (k + 1) % 3]
            outside = np.ones(len(i), dtype=bool)
            for v in range(3):
                outside &= _orient_signs(p, q, t[second, v]) <= 0
            separated |= outside
    return bool(np.all(separated)), len(i)


def check_mesh(rings, result, extra=None):
    """Exact contract of a result: provenance, orientation, disjointness, containment, area."""
    rings = [np.asarray(r, float) for r in rings]
    vertices = np.vstack(rings)
    n = len(vertices)
    pts, tri = result.points, result.triangles
    assert pts.dtype == np.float64 and tri.dtype == np.int64
    assert np.array_equal(pts[:n], vertices) and len(pts) == n + len(result.cut_ends)
    lines = set(vertices[:, 0].tolist()) | set(np.asarray([] if extra is None else extra, float).reshape(-1).tolist())
    # cut provenance, recomputed exactly (not with the production formula)
    bounds = []
    for k, ((a, b), f) in enumerate(zip(result.cut_ends, result.cut_fraction)):
        (xa, ya), (xb, yb) = vertices[a], vertices[b]
        cx, cy = pts[n + k]
        assert xa < cx < xb and cx in lines
        assert 0.0 <= f <= 1.0                     # 0/1 at an interior x: owner decision N2 r2 (1)
        fa, fb, fc = _fp((xa, ya)), _fp((xb, yb)), _fp((cx, cy))
        exact_f = (fc[0] - fa[0]) / (fb[0] - fa[0])
        exact_y = fa[1] + exact_f * (fb[1] - fa[1])
        assert f == float(exact_f), "cut fraction not correctly rounded"      # float(Fraction) is RN
        assert cy == float(exact_y), "cut ordinate not correctly rounded"
        bounds.append(abs(fc[1] - exact_y))
    # exact orientation: every triangle strictly counter-clockwise
    assert len(tri) and np.all(_orient_signs(pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]) > 0)
    # exact area: the triangles tile the rounded polygon exactly; it is within the stated bound
    rounded = _rounded_rings(rings, result)
    fr = [[_fp(p) for p in r] for r in rounded]
    region = abs(_signed_area(fr[0])) - sum(abs(_signed_area(h)) for h in fr[1:])
    fp = [_fp(p) for p in pts]
    total = sum(_orient(fp[a], fp[b], fp[c]) for a, b, c in tri.tolist()) / 2
    assert total == region
    assert result.area == float(total), "area is not the correctly rounded exact area (N2 r4 (3))"
    exact = abs(_signed_area([_fp(p) for p in _compress(rings[0])])) - sum(
        abs(_signed_area([_fp(p) for p in _compress(h)])) for h in rings[1:])
    assert abs(total - exact) <= _area_allowance(rings, result, bounds)
    # pairwise interior-disjoint (exact, all pairs after a bounding-box prefilter)
    disjoint, _ = _interiors_disjoint(pts, tri)
    assert disjoint
    # containment in the rounded region: corners and edge midpoints closed, centroids strictly inside
    exact_rings = [_Ring(r) for r in rounded]
    for p in np.unique(tri):
        assert _locate_region(exact_rings, *fp[p]) >= 0, ("corner outside", p)
    edges = {tuple(sorted(e)) for a, b, c in tri.tolist() for e in ((a, b), (b, c), (c, a))}
    for a, b in edges:
        assert _locate_region(exact_rings, (fp[a][0] + fp[b][0]) / 2, (fp[a][1] + fp[b][1]) / 2) >= 0
    for a, b, c in tri.tolist():
        assert _locate_region(exact_rings, (fp[a][0] + fp[b][0] + fp[c][0]) / 3,
                              (fp[a][1] + fp[b][1] + fp[c][1]) / 3) == 1
    return total, exact


def _contract(rings, extra=None, **kwargs):
    rings = _f(*rings)
    extra = _fx(extra)
    result = slab.triangulate_rings(rings, extra, **kwargs)
    check_mesh(rings, result, extra)
    again = slab.triangulate_rings(rings, extra, **kwargs)
    assert np.array_equal(again.points, result.points) and np.array_equal(again.triangles, result.triangles)
    assert np.array_equal(again.cut_ends, result.cut_ends)
    assert np.array_equal(again.cut_fraction, result.cut_fraction)
    return result


def _triangle_set(result):
    """Triangles as rotation-normalised coordinate tuples (independent of point numbering)."""
    out = set()
    for t in result.points[result.triangles].tolist():
        t = [tuple(p) for p in t]
        k = t.index(min(t))
        out.add(tuple(t[k:] + t[:k]))
    return out


def _refused(code, rings, extra=None, **kwargs):
    with pytest.raises(DisplayTrimRefused) as caught:
        slab.triangulate_rings(rings, extra, **kwargs)
    assert caught.value.code == code, caught.value
    return caught.value


#: default call (fast path where it applies), then the general path forced
_PATHS = ({}, {"_fast": False})


# ----------------------------------------------------------------------------- contract
def test_refusal_is_the_typed_trim_invalid():
    assert issubclass(slab.SlabInvalid, DisplayTrimRefused)
    _refused("TRIM_INVALID", _f([(0., 0.), (1., 0.)]))


def test_duplicate_consecutive_points():
    ring = [(0., 0.), (0., 0.), (1., 0.), (1., 0.), (1., 1.), (0., 1.), (0., 1.)]
    _contract([ring])
    _contract([SQUARE + [SQUARE[0]]])                     # explicitly closed ring


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
    _contract([ring])


def _monotone_ring(count):
    u = np.linspace(0., 1., count)
    return np.vstack((np.column_stack((u, .1 * np.sin(3 * u))), np.column_stack((u[::-1], 1. + .2 * u[::-1]))))


@pytest.mark.parametrize("count", [4, 30, 200])
def test_monotone_fast_path_gives_the_general_path_coverage(count):
    ring = _monotone_ring(count)
    fast = slab._monotone(ring, None)
    small = slab._monotone_small(ring, None)
    assert fast is not None and small is not None
    assert _triangle_set(fast) == _triangle_set(small)
    assert np.array_equal(fast.points, small.points)
    if count <= 30:
        _contract([ring])
    else:
        assert slab.triangulate_rings([ring]).area == pytest.approx(abs(float(_signed_area(
            [_fp(p) for p in ring]))), rel=1e-12)


@pytest.mark.parametrize("rings", [
    [],
    [np.zeros((0, 2))],
    _f([(0., 0.), (1., 0.)]),
    _f([(0., 0.), (1., 0.), (2., 0.)]),                                 # zero area
    _f([(0., 0.), (0., 0.), (0., 0.), (0., 0.)]),                       # one point repeated
    _f([(0., 0.), (1., 1.), (1., 0.), (0., 1.)]),                       # bow tie
    _f([(0., 0.), (4., 0.), (4., 4.), (0., 4.), (2., -1.), (3., 5.)]),  # self-intersecting
    _f(SQUARE, [(5., 5.), (6., 5.), (6., 6.)]),                         # hole outside
    _f(SQUARE, SQUARE),                                                 # hole equals outer
    _f([(0., 0.), (4., 0.), (4., 4.), (0., 4.)], [(1., 1.), (3., 1.), (3., 3.)], [(1.5, 1.), (3.5, 1.), (3.5, 3.)]),
    [np.array([(0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.), (0., 0., 1.), (1., 0., 1.)])],  # not 2-D
    [np.array([0., 0., 1., 0., 1., 1.])],                               # flat array
    [np.array([("a", "b"), ("c", "d"), ("e", "f")])],
], ids=lambda r: str(r)[:40])
def test_invalid_input_is_a_typed_refusal(rings):
    _refused("TRIM_INVALID", rings)


@pytest.mark.parametrize("rings", [
    _f([(0., 0.), (1., 0.), (float("nan"), 1.)]),
    _f([(0., 0.), (1., 0.), (float("inf"), 1.)]),
    _f([(0., 0.), (1., 0.), (0., 1.0000001e9)]),
], ids=("nan", "inf", "above_1e9"))
def test_out_of_domain_input_is_a_typed_refusal(rings):
    """Owner decision N2 r1 (2): the declared numeric domain, like N1 (non-finite: OUT_OF_DOMAIN)."""
    _refused("OUT_OF_DOMAIN", rings)


def test_the_domain_bound_itself_is_admitted():
    ring = np.array([(-1e9, -1e9), (1e9, -1e9), (1e9, 1e9), (-1e9, 1e9)])
    _contract([ring])


def test_invalid_extra_lines_are_refused():
    _refused("OUT_OF_DOMAIN", [SQ], np.array([.5, float("nan")]))
    _refused("TRIM_INVALID", [SQ], ["a"])


# ----------------------------------------------------------------------------- N2 round-1 reproductions
def test_r1_1_a_proper_crossing_is_refused_whatever_the_cuts():
    """High: a cut through the crossing made the order reverse 'between' slabs and it was accepted."""
    ring = np.array([(1, 2), (3, 2), (3, 0), (2, 4), (4, 2)], dtype=float)
    assert not brute_valid([ring])
    for cuts in (None, _fx([2.5]), _fx([2.25, 2.5, 3.5]), np.linspace(1, 4, 13)):
        _refused("TRIM_INVALID", [ring], cuts)
        _refused("TRIM_INVALID", [ring], cuts, _fast=False)


def test_r1_2_a_hole_sharing_a_sloped_edge_is_refused():
    """Medium: a hole sharing the sloped edge y = x; tiny cut gave overlapping triangles, f == 1."""
    x = 2. ** -55
    outer = [(0, 2), (1, 1), (0, 0)]
    hole = [(0, 0), (1, 1), (.5, .75)]
    for cuts in (None, _fx([x])):
        _refused("TRIM_INVALID", _f(outer, hole), cuts)


def test_r1_2_a_tiny_slab_keeps_cut_fractions_inside_and_ordinates_consistent():
    x = 2. ** -55
    outer = [(0., 2.), (1., 1.), (0., 0.)]
    hole = [(.25, .5), (.5, .625), (.5, .75)]
    for rings in ([outer], [outer, hole], [outer[::-1], hole[::-1]]):
        for path in _PATHS:
            result = _contract(rings, [x], **path)
            assert np.all((result.cut_fraction > 0) & (result.cut_fraction < 1))


def test_r1_2_cut_points_do_not_depend_on_the_traversal_direction():
    """Canonical interpolation (lower-x end first): both directions give bit-identical cut points."""
    rng = np.random.default_rng(3)
    ring = np.column_stack((np.cos(np.linspace(0, 2 * np.pi, 41)[:-1]), np.sin(np.linspace(0, 2 * np.pi, 41)[:-1])))
    ring = ring * rng.uniform(.5, 1.5, (40, 1))
    lines = rng.uniform(-1, 1, 25)
    for path in _PATHS:
        a = slab.triangulate_rings([ring], lines, **path)
        b = slab.triangulate_rings([ring[::-1]], lines, **path)
        n = len(ring)
        assert sorted(map(tuple, a.points[n:].tolist())) == sorted(map(tuple, b.points[n:].tolist()))
        assert _triangle_set(a) == _triangle_set(b)


def test_r1_3_a_positive_area_piece_is_never_dropped():
    """Medium: a generic determinant cancelled to 0 and dropped a triangle of area 2**-54."""
    ring = np.array([(0., -1.), (1., 0.), (1., 2. ** -53), (0., 1.)])
    for path in _PATHS:
        total, exact = check_mesh([ring], slab.triangulate_rings([ring], **path))
        assert total == exact


def test_r1_4_a_small_hole_in_a_long_face_is_kept():
    """Medium: the area floor used the extent of ALL rings and refused a representable hole."""
    d = 1 / 64
    outer = [(0, 0), (10 ** 6, 0), (10 ** 6, 1), (0, 1)]
    hole = [(1, .25), (1 + d, .25), (1 + d, .25 + d), (1, .25 + d)]
    _contract([outer, hole])


@pytest.mark.parametrize("rings", [
    [[(0, 0), (2, 0), (2, 2), (0, 2), (0, 0), (0, -1), (-1, -1), (-1, 0)]],    # self-touch, two lobes (r1)
    [[(0, 0), (2, 0), (2, 2), (1, 0), (0, 2)]],                                 # vertex on its own edge
    [[(0, 0), (4, 0), (4, 4), (0, 4)], [(1, 1), (2, 1), (2, 2)], [(2, 2), (3, 2), (3, 3)]],   # holes touch at a point
    [[(0, 0), (4, 0), (4, 4), (0, 4)], [(0, 0), (1, 2), (2, 1)]],               # hole touches outer at a vertex
    [[(0, 0), (4, 0), (4, 4), (0, 4)], [(2, 0), (3, 1), (1, 1)]],               # hole vertex on an outer edge
    [[(0, 0), (4, 0), (4, 4), (0, 4)], [(1, 0), (3, 0), (2, 1)]],               # hole shares an outer edge piece
    [[(0, 0), (4, 0), (4, 4), (0, 4)], [(1, 1), (2, 1), (2, 2), (1, 2)], [(2, 1), (3, 1), (3, 2), (2, 2)]],  # shared edge
    [[(0, 0), (4, 0), (4, 4), (0, 4)], [(4, 1), (4, 3), (3, 2)]],               # hole on a vertical outer edge
    [[(0, 0), (4, 0), (4, 4), (0, 4)], [(1, 1), (3, 1), (3, 3), (1, 3)], [(1.5, 1.5), (2.5, 1.5), (2.5, 2.5)]],  # nested
    [[(0, 0), (2, 0), (2, 2), (1, 2), (1, 3), (1, 1), (0, 1)]],                 # zero-width vertical spike
    [[(0, 0), (3, 0), (3, 1), (2, 1), (1.5, 2), (1, 1), (1.5, 2), (0, 1)]],     # zero-width sloped spike
], ids=("self_touch", "vertex_on_edge", "holes_point", "outer_vertex", "outer_edge_point", "outer_edge_piece",
        "holes_share_edge", "vertical_edge", "nested", "vertical_spike", "sloped_spike"))
def test_r1_4_and_7_contacts_are_refused(rings):
    """Owner decision (1): simple rings, holes strictly inside and disjoint; any contact refuses."""
    rings = _f(*rings)
    assert not brute_valid(rings)
    for path in _PATHS:
        _refused("TRIM_INVALID", rings, **path)
        _refused("TRIM_INVALID", rings, _fx([.5, 1.5, 2.5]), **path)


def test_r1_5_conversion_and_overflow_are_typed_refusals():
    """Medium: oversized ints leaked OverflowError, complex lost its NaN, 1e154 overflowed the shoelace.
    Since N2 r4 (owner decision (1)) the foreign forms are TRIM_INVALID by the input contract (they
    were OUT_OF_DOMAIN after conversion); the overflow case stays OUT_OF_DOMAIN."""
    for rings, extra in (([[(0, 0), (1, 0), (0, 10 ** 400)]], None), ([SQ], [10 ** 400])):
        assert _FORM in str(_refused("TRIM_INVALID", rings, extra))
    z = np.array(SQUARE, dtype=complex)
    z[0, 0] = complex(0, float("nan"))
    for ring in (z, np.array(SQUARE, dtype=complex), np.array(SQUARE, dtype=object)):
        assert _FORM in str(_refused("TRIM_INVALID", [ring]))
    outer = 1e154 * np.array([(0, 0), (.9, .9), (1, 1), (0, 1)])
    hole = 1e154 * np.array([(1.25, 0), (1.5, 0), (1.5, .125), (1.25, .125)])
    with np.errstate(all="raise"):
        _refused("OUT_OF_DOMAIN", [outer, hole])


def _staggered(m):
    def box(a, b, c, d):
        return np.array([(a, c), (b, c), (b, d), (a, d)], dtype=float)
    return [box(0, 3 * m + 1, 0, 2 * m + 1), *[box(i + 1, 2 * m + i + 1, 2 * i + 1, 2 * i + 2) for i in range(m)]]


def test_r1_6_quadratic_trims_are_refused_by_the_preflight_budget():
    """Medium: staggered holes need ~2 m**2 incidences; above the budget OUT_OF_DOMAIN, nothing built."""
    import time
    small = slab.triangulate_rings(_staggered(16))
    assert len(small.cut_ends) == 2 * 16 ** 2 + 2 * 16
    _contract(_staggered(6))
    start = time.perf_counter()
    error = _refused("OUT_OF_DOMAIN", _staggered(400))
    assert "too complex" in str(error) and time.perf_counter() - start < 2.0
    _refused("OUT_OF_DOMAIN", [SQ], np.linspace(0, 1, 200_001))
    assert slab.MAX_SLAB_INCIDENCES == 200_000


def test_r1_7_the_general_path_is_forced_and_matches_the_fast_path():
    for ring, lines in ((_monotone_ring(12), None), (_monotone_ring(12), np.linspace(0, 1, 7)),
                        (_monotone_ring(60), np.linspace(.05, .95, 5)),
                        (np.array([(0., -1.), (1., 0.), (1., 2. ** -53), (0., 1.)]), None)):
        fast = slab.triangulate_rings([ring], lines)
        general = slab.triangulate_rings([ring], lines, _fast=False)
        check_mesh([ring], general, lines)
        assert _triangle_set(fast) == _triangle_set(general)
        assert fast.area == pytest.approx(general.area, rel=1e-14)


def test_r1_7_the_oracle_catches_a_missing_triangle_and_an_overlap():
    """The oracle itself: it fails on a mesh with a triangle removed or duplicated."""
    ring = np.array([(0., -1.), (1., 0.), (1., 2. ** -53), (0., 1.)])
    good = slab.triangulate_rings([ring])
    for triangles in (good.triangles[1:], np.vstack((good.triangles, good.triangles[:1]))):
        bad = slab.SlabTriangulation(good.points, triangles, good.cut_ends, good.cut_fraction, good.area)
        with pytest.raises(AssertionError):
            check_mesh([ring], bad)


def test_validity_does_not_depend_on_extra_lines():
    rng = np.random.default_rng(11)
    for _ in range(150):
        n = int(rng.integers(3, 8))
        rings = [rng.integers(0, 5, size=(n, 2)).astype(float)]
        if rng.random() < .4:
            rings.append(rng.integers(0, 5, size=(int(rng.integers(3, 5)), 2)).astype(float))
        verdicts = set()
        for lines in (None, rng.uniform(-1, 6, 5), np.array([.5, 1.5, 2.5, 3.5])):
            try:
                slab.triangulate_rings(rings, lines)
                verdicts.add("ok")
            except DisplayTrimRefused as error:
                verdicts.add(error.code)
        assert len(verdicts) == 1, verdicts
        assert verdicts == ({"ok"} if brute_valid(rings) else {"TRIM_INVALID"})


def test_random_garbage_matches_the_brute_force_verdict_and_the_oracle():
    rng = np.random.default_rng(7)
    refused = ok = 0
    for _ in range(300):
        n = int(rng.integers(3, 9))
        rings = [rng.integers(0, 5, size=(n, 2)).astype(float)]
        if rng.random() < .4:
            rings.append(rng.integers(0, 5, size=(int(rng.integers(3, 6)), 2)).astype(float))
        valid = brute_valid(rings)
        try:
            result = slab.triangulate_rings(rings)
        except DisplayTrimRefused as error:
            assert error.code == "TRIM_INVALID" and not valid
            refused += 1
            continue
        assert valid
        ok += 1
        check_mesh(rings, result)
    assert refused > 0 and ok > 0


# ----------------------------------------------------------------------------- N2 round-2 reproductions
def _n1_face_ring():
    """A nearly rectangular planar face whose two right corners are one ulp apart in x, sampled by N1."""
    from anygeometry import GeometryModel
    from anygeometry import display_tessellation as dt
    from anygeometry.surfaces import Plane
    x = math.nextafter(1.5, -math.inf)
    g = GeometryModel()
    vertices = [g.add_point(u, v, 0) for u, v in [(-1, 0), (1.5, 0), (x, 1), (-1, 1)]]
    edges = [g.add_line(a, b) for a, b in zip(vertices, vertices[1:] + vertices[:1])]
    g.add_face(edges, surface=Plane((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    tol = dt.DisplayTolerance(.15)
    return np.array([dt.edge_display_samples(g, e, tol).points[0, :2] for e in edges])


def test_r2_1_a_real_n1_face_with_corners_one_ulp_apart_is_accepted():
    """Medium (M1): the bottom edge's cut fraction rounded to 1 in the subtraction; it is correctly
    rounded now (0.9999999999999999) and the face is meshed (owner decision N2 r2 (1))."""
    ring = _n1_face_ring()
    x = math.nextafter(1.5, -math.inf)
    assert ring[2, 0] == x and ring[1, 0] == 1.5
    for path in _PATHS:
        for extra in (None, ring[:, 0]):
            result = _contract([ring], extra, **path)
            assert np.any(result.cut_fraction == float((F(x) + 1) / F(2.5)))
    for path in _PATHS:
        _contract([[(-1, 0), (1.5, 0), (x, 1), (-1, 1)]], **path)
        _contract([[(-1, 0), (1.5, 0), (1.5, 1), (-1, 1)]], [1.5], **path)


def test_r2_1_a_triangular_hole_with_a_cut_one_ulp_from_its_tip_is_accepted():
    """Medium (M1): both hole chains round to the tip's y one ulp right of the tip: a TIE, not a
    reversal; the zero-height pieces are dropped and the mesh tiles the rounded polygon."""
    outer = [(.5, -1), (3, -1), (3, 3), (.5, 3)]
    hole = [(1, 1), (2, .75), (2, 1.25)]
    near = math.nextafter(1.0, 2.0)
    for path in _PATHS:
        _contract([outer, hole], [1.0], **path)
        result = _contract([outer, hole], [near], **path)
        on_line = result.points[:, 0] == near
        assert np.count_nonzero(result.points[on_line, 1] == 1.0) >= 2      # the tie is there
        _contract([outer[:3] + [(near, 3)] + outer[3:], hole], **path)


def test_r2_1_a_cut_fraction_rounding_onto_a_segment_end_is_accepted():
    """Medium (M1/M3): a fraction correctly rounded to 0.0 (or 1.0) at a strictly interior x is
    accepted; the cut point is the exact position rounded (within half an ulp of the fraction)."""
    ring = [(0, 0), (2, 0), (2, 1), (0, 1)]
    tiny = math.ulp(0.0)
    for path in _PATHS:
        result = _contract([ring], [tiny], **path)
        assert np.all(result.cut_fraction == 0.0)
        assert np.all(result.points[len(ring):, 0] == tiny)
    long = [(-1e9, 0), (1, 0), (1, 1), (-1e9, 1)]
    for path in _PATHS:
        result = _contract([long], [math.nextafter(1.0, 0.0)], **path)
        assert np.all(result.cut_fraction == 1.0)


def test_r2_1_correctly_rounded_cut_values_on_hard_directed_inputs():
    """The certified rounding of fraction and ordinate equals float(Fraction) on directed cases:
    sign changes, subnormal slabs and ordinates, the domain bound, one-ulp slopes."""
    rng = np.random.default_rng(2611)
    rows = []
    for _ in range(4000):
        kind = int(rng.integers(6))
        scale = 10.0 ** rng.uniform(-300, 9) if kind == 0 else 10.0 ** rng.uniform(-3, 9)
        xa, xb = np.sort(rng.uniform(-1, 1, 2)) * min(scale, 1e9)
        x = rng.uniform(xa, xb)
        if kind == 1:
            x = np.nextafter(xa, xb)
        elif kind == 2:
            x = np.nextafter(xb, xa)
        ya, yb = rng.uniform(-1, 1, 2) * min(scale, 1e9)
        if kind == 3:
            yb = -ya * (1 + int(rng.integers(-3, 4)) * 2.0 ** -52)       # crosses 0 near the middle
        if kind == 4:
            ya, yb = ya * 1e-300, yb * 1e-300                          # subnormal ordinates
        if kind == 5:
            yb = np.nextafter(ya, np.inf)
        if xa < x < xb:
            rows.append(tuple(float(v) for v in (xa, ya, xb, yb, x)))
    rows += [(0., 0., 2., 1., math.ulp(0.)), (-1., 0., 1.5, 0., math.nextafter(1.5, -math.inf)),
             (-1e9, -1e9, 1e9, 1e9, 0.), (1., 1., 2., .75, math.nextafter(1., 2.)), (0., 1., 1., -1., .5)]
    xa, ya, xb, yb, x = (np.array(c, dtype=float) for c in zip(*rows))
    with np.errstate(all="raise"):
        f, y = slab._cut_values(xa, ya, xb, yb, x)
    small_f, small_y = slab._cut_values_exact(xa.tolist(), ya.tolist(), xb.tolist(), yb.tolist(), x.tolist())
    for k, (a, b, c, d, e) in enumerate(rows):
        fa, fb, fc, fd, fe = map(F, (a, b, c, d, e))
        exact_f = (fe - fa) / (fc - fa)
        assert f[k] == float(exact_f) == small_f[k], (rows[k], f[k])
        assert y[k] == float(fb + exact_f * (fd - fb)) == small_y[k], (rows[k], y[k])


def test_r2_2_vertical_and_duplicate_points_count_in_the_preflight():
    """Medium (M2): one subdivided vertical side made 200002 triangles from 2 incidences."""
    import time
    n = slab.MAX_SLAB_INCIDENCES + 1
    side = np.column_stack((np.ones(n), np.arange(1, n + 1) / n))
    ring = np.vstack(([(0, 0), (1, 0)], side, [(0, 1)]))
    start = time.perf_counter()
    error = _refused("OUT_OF_DOMAIN", [ring])
    assert "too complex" in str(error) and time.perf_counter() - start < 1.0
    assert slab.MAX_SLAB_RING_POINTS == 50_000 and slab.MAX_SLAB_OUTPUT == 600_000
    over = slab.MAX_SLAB_RING_POINTS - 2
    dup = np.vstack(([(0., 0.)] * over, [(1., 0.), (1., 1.), (0., 1.)]))    # duplicates count too
    assert "too complex" in str(_refused("OUT_OF_DOMAIN", [dup]))
    _contract([dup[-(slab.MAX_SLAB_RING_POINTS - 10):]])


def test_r2_2_the_output_bound_is_checked_before_allocation():
    """Below the point and incidence caps, the triangle bound I + 2 (N + cuts) still binds."""
    k = 20_000
    side = np.column_stack((np.zeros(k), np.linspace(1, 0, k + 2)[1:-1]))
    ring = np.vstack(([(0., 0.), (1., 0.), (1., 1.), (0., 1.)], side))
    lines = np.linspace(0, 1, 95_002)[1:-1]
    error = _refused("OUT_OF_DOMAIN", [ring], lines)
    assert "too complex" in str(error) and "MAX_SLAB_OUTPUT" in str(error)
    result = slab.triangulate_rings([ring], lines[::2])
    assert len(result.triangles) <= slab.MAX_SLAB_OUTPUT


def test_r2_3_the_callers_numpy_error_state_does_not_matter():
    """Medium (M3): under np.errstate(all='raise') the interpolation's underflow escaped raw."""
    ring = np.array([(0, 0), (2, 0), (2, 1), (0, 1)], dtype=float)
    tiny = _fx([math.ulp(0.0)])
    for path in _PATHS:
        with np.errstate(all="ignore"):
            quiet = _TRIANGULATE([ring], tiny, **path)
        with np.errstate(all="raise"):
            strict = _TRIANGULATE([ring], tiny, **path)
            _refused("OUT_OF_DOMAIN", _f([(0., 0.), (1., 0.), (float("nan"), 1.)]), **path)
        assert np.array_equal(quiet.points, strict.points) and np.array_equal(quiet.triangles, strict.triangles)
        assert np.array_equal(quiet.cut_fraction, strict.cut_fraction)


def test_r2_4_coordinates_not_exactly_binary64_are_refused_before_anything_else():
    """Medium (M4): narrowing before compression lost a retrace; integers above 2**53 likewise.
    Since N2 r4 (owner decision (1)) nothing but float64 is admitted: every other dtype is
    TRIM_INVALID by the input contract, before any conversion (nothing can be narrowed)."""
    big = 2 ** 53 + 1
    rings = [np.array([(0, 0), (value, 0), (0, 1)], dtype=np.int64) for value in (big, -big, 2 ** 53)]
    rings += [np.array([(0, 0), (2 ** 64 - 1, 0), (0, 1)], dtype=np.uint64), np.array(SQUARE, dtype=np.float32),
              np.array([(0, 0), (3, 0), (0, 2)], dtype=np.uint8), np.array(SQUARE, dtype=np.int64)]
    for ring in rings:
        assert _FORM in str(_refused("TRIM_INVALID", [ring]))
    assert _FORM in str(_refused("TRIM_INVALID", [SQ], np.array([big], dtype=np.int64)))


@pytest.mark.skipif(np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant,
                    reason="np.longdouble is binary64 on this platform: no wider value to narrow")
def test_r2_4_a_wider_float_retrace_is_not_narrowed_into_a_valid_square():
    wide = np.longdouble
    x = np.nextafter(wide(1), wide(2))
    assert x > 1 and float(x) == 1
    ring = np.array([(0, 0), (1, 0), (x, 0), (1, 0), (1, 1), (0, 1)], dtype=wide)
    assert _FORM in str(_refused("TRIM_INVALID", [ring]))                  # N2 r4: not float64
    assert _FORM in str(_refused("TRIM_INVALID", [SQ], np.array([x], dtype=wide)))
    assert _FORM in str(_refused("TRIM_INVALID", [np.array(SQUARE, dtype=wide)]))


def test_r2_low_a_the_area_floor_is_one_exact_rule_on_both_paths():
    """Low: the fast path summed slab areas, the general path a cancelling shoelace; they disagreed."""
    d = 2.0 ** -46
    for e, admitted in ((1e-20, True), (0.0, False), (-1e-20, False)):
        ring = np.array([(0., 0.), (1., 1.), (1., 1. + d), (0., d + e)])
        extent = max(1.0, 1.0 + d)
        floor = F(16.0 * 4 * _EPS * extent * extent)
        assert (abs(_signed_area([_fp(p) for p in ring])) > floor) == admitted
        verdicts = []
        for path in _PATHS:
            try:
                slab.triangulate_rings([ring], **path)
                verdicts.append("ok")
            except DisplayTrimRefused as error:
                verdicts.append(error.code)
        assert verdicts == (["ok", "ok"] if admitted else ["OUT_OF_DOMAIN", "OUT_OF_DOMAIN"]), (e, verdicts)


def test_r2_low_b_the_oracle_orientation_is_exact_under_underflow():
    """Low: subnormal products made the oracle's filter threshold 0 and skip the rational check."""
    a = np.array([(-4.089060633435915e-164, -1.2267181900307745e-163)])
    b = np.array([(1.9914039133227464e-155, 5.974211739968239e-155)])
    c = np.array([(4.191769962437771e-155, 1.2575309887313312e-154)])
    assert _orient(_fp(a[0]), _fp(b[0]), _fp(c[0])) == 0
    assert _orient_signs(a, b, c).tolist() == [0]


def test_r2_1_ties_are_accepted_and_the_reversal_guard_never_fires():
    """Owner decision N2 r2 (1): valid trims with near-contacts at the ulp level (lines one ulp from
    a vertex, snapped holes) are accepted and tile exactly; the reversal refusal stays unreachable."""
    rng = np.random.default_rng(26112)
    accepted = 0
    for case in range(120):
        k = int(rng.integers(3, 16))
        a = np.sort(rng.uniform(0, 2 * np.pi, k))
        outer = np.column_stack(np.array([np.cos(a), np.sin(a)]) * rng.uniform(.4, 1., k))
        rings = [outer]
        if case % 2:
            c = rng.uniform(-.1, .1, 2)
            b = np.sort(rng.uniform(0, 2 * np.pi, 3))
            rings.append(c + .05 * np.column_stack((np.cos(b), np.sin(b))))
        rings = [np.round(r * 2 ** 12) / 2 ** 12 for r in rings]
        v = np.vstack(rings)[:, 0]
        lines = np.concatenate((np.nextafter(v, np.inf), np.nextafter(v, -np.inf), rng.uniform(-1, 1, 5)))
        valid = brute_valid(rings)
        for path in _PATHS:
            try:
                result = slab.triangulate_rings(rings, lines, **path)
            except DisplayTrimRefused as error:
                assert "reverse" not in str(error)
                assert error.code == "TRIM_INVALID" and not valid or "below display resolution" in str(error)
                continue
            assert valid
            accepted += 1
            check_mesh(rings, result, lines)
    assert accepted >= 100


# ----------------------------------------------------------------------------- N2 round 3 (Codex)
_DOCS = __import__("pathlib").Path(__file__).resolve().parents[1] / "docs" / "DISPLAY_TESSELLATION.md"


def _lift_by_fraction(result):
    """Each cut point rebuilt at ``cut_fraction`` on its segment (exact, then rounded): the lift the
    round-2 docs described, which N2 does NOT certify (owner decision N2 r3 (1))."""
    uv = [_fp(p) for p in result.points]
    n = len(result.points) - len(result.cut_ends)
    lifted = uv[:n]
    for (a, b), f in zip(result.cut_ends.tolist(), result.cut_fraction.tolist()):
        lifted.append(tuple(F(float(x + F(f) * (y - x))) for x, y in zip(uv[a], uv[b])))
    return uv, lifted


def test_r3_1_the_certified_product_is_the_2d_mesh_and_a_fraction_lift_can_invert_it():
    """Medium (M1): rebuilding the cut points at cut_fraction moved an outer cut past a hole tip and
    inverted triangle [7, 4, 10].  The 2D mesh is certified; any lift must check orientation itself."""
    outer = [(-1., -1.), (1., -1.), (1., 3.), (-1., 3.)]
    hole = [(-.21, 1.), (.75, .75), (.75, 1.25)]
    extra = [math.nextafter(-.21, -math.inf)]
    for path in _PATHS:
        result = _contract([outer, hole], extra, **path)            # the 2D mesh passes the exact oracle
        uv, lifted = _lift_by_fraction(result)
        triangles = result.triangles.tolist()
        assert all(_orient(*(uv[i] for i in t)) > 0 for t in triangles)
        inverted = [t for t in triangles if _orient(*(lifted[i] for i in t)) < 0]
        assert inverted, path
        if not path:
            assert any(t[k:] + t[:k] == [7, 4, 10] for t in inverted for k in range(3)), inverted
    text = " ".join(_DOCS.read_text(encoding="utf-8").split())
    assert "must not rely on cut_fraction reconstruction for orientation or coverage" in text
    assert "metadata" in text and "[7, 4, 10]" in text


def _traced_peak(call):
    import tracemalloc
    tracemalloc.start()
    try:
        call()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


#: largest traced allocation (bytes) allowed before a size-cap refusal (owner decision N2 r3 (2))
_PREFLIGHT_PEAK = 1_000_000


def test_r3_2_size_caps_are_checked_from_shapes_before_any_conversion():
    """Medium (M2): a million-point broadcast view (16 bytes of storage) was converted and scanned
    (about 32 MB traced) before OUT_OF_DOMAIN; a non-contiguous extra_x was copied by reshape.
    Bound: under 1 MB of traced peak allocation before every size-cap refusal."""
    square = np.array(SQUARE)
    base = np.zeros((2000, 2000))                                   # allocated before tracing
    many_points = np.zeros((200_002, 2))                            # (N2 r4: arrays; lists are refused)
    many_lines = np.full(300_000, 0.5)
    flat = np.zeros(2_000_000)
    cases = [
        ([np.broadcast_to(np.zeros((1, 2)), (1_000_000, 2))], None),
        ([np.broadcast_to(np.zeros((1, 2)), (100_000, 2))], None),
        ([square] * 20_000, None),                                  # 80000 points in many rings
        ([many_points], None),
        ([square], np.broadcast_to(0.5, (1_000_000,))),
        ([square], flat[::2]),                                      # a copy would be 8 MB
        ([square], many_lines),
    ]
    for rings, extra in cases:
        caught = []

        def call():
            try:
                slab.triangulate_rings(rings, extra)
            except DisplayTrimRefused as error:
                caught.append(error)
        peak = _traced_peak(call)
        assert caught and caught[0].code == "OUT_OF_DOMAIN" and "too complex" in str(caught[0]), caught
        assert peak < _PREFLIGHT_PEAK, (peak, str(caught[0]))
    # non-sized iterables are typed refusals, never consumed (N2 r4: by the one input-contract rule)
    assert _FORM in str(_refused("TRIM_INVALID", (r for r in [SQ])))
    assert _FORM in str(_refused("TRIM_INVALID", [iter(SQUARE)]))
    assert _FORM in str(_refused("TRIM_INVALID", [SQ], (x for x in [0.5])))
    # the admitted forms (a list or tuple of float64 arrays, views included) give the same result
    reference = slab.triangulate_rings([square], np.array([0.25, 0.5]))
    wide = np.array([(0., 9.), (1., 9.), (1., 9.), (0., 9.), (0., 9.), (1., 9.), (1., 9.), (0., 9.)])
    wide[::2, 0], wide[::2, 1] = square[:, 0], square[:, 1]
    for rings, extra in (((square,), np.array([0.25, 0.5])), ([wide[::2]], np.array([0.25, 9., 0.5, 9.])[::2]),
                         ([np.asfortranarray(square)], np.array([0.25, 0.5]))):
        again = slab.triangulate_rings(rings, extra)
        assert np.array_equal(again.points, reference.points) and np.array_equal(again.triangles, reference.triangles)


def test_r3_3_masked_arrays_and_every_other_subclass_are_refused(tmp_path):
    """Medium (M3): np.asarray dropped the mask, so the hidden backing value decided the mesh.
    Since N2 r4 (owner decision (1), Codex r4 M2) every ndarray subclass is refused too (they were
    read as stored, through np.asarray): only exact np.ndarray is admitted."""
    rings = []
    for hidden in (1., 2.):
        ring = np.ma.array([(0., 0.), (hidden, 0.), (1., 1.), (0., 1.)])
        ring[1, 0] = np.ma.masked
        rings.append(ring)
    assert np.ma.allequal(*rings)
    for ring in rings + [np.ma.array(SQUARE), np.ma.array(SQUARE, mask=False)]:
        assert _FORM in str(_refused("TRIM_INVALID", [ring]))
    assert _FORM in str(_refused("TRIM_INVALID", [SQ], np.ma.array([0.5])))
    assert _FORM in str(_refused("TRIM_INVALID", [SQ], [np.ma.masked]))
    assert _FORM in str(_refused("TRIM_INVALID", [[(0., 0.), np.ma.array((1., 0.)), (1., 1.), (0., 1.)]]))
    assert _FORM in str(_refused("TRIM_INVALID", [[(0., 0.), (np.ma.masked, 0.), (1., 1.)]]))
    import warnings
    stored = np.memmap(tmp_path / "ring.bin", dtype=float, mode="w+", shape=(4, 2))
    stored[:] = SQUARE
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        matrix = np.matrix(SQUARE)
    for ring in (matrix, stored):
        assert _FORM in str(_refused("TRIM_INVALID", [ring], np.array([0.5])))
    assert _FORM in str(_refused("TRIM_INVALID", [SQ], np.memmap(tmp_path / "x.bin", dtype=float, mode="w+",
                                                                 shape=(1,))))
    del stored


def test_r3_low_a_extra_cuts_never_underflow_the_area():
    """Low: a square of side 2**-536 had area 2**-1072 uncut and 0.0 in 16 slabs (both paths)."""
    s = 2.0 ** -536
    square = np.array([(0., 0.), (s, 0.), (s, s), (0., s)])
    lines = np.array([s * k / 16 for k in range(1, 16)])
    for path in _PATHS:
        assert slab.triangulate_rings([square], **path).area == 2.0 ** -1072
        result = _contract([square], lines, **path)
        assert result.area == 2.0 ** -1072, (path, result.area)


def test_r3_low_b_the_area_allowance_counts_each_cut_once_in_its_own_ring():
    """Low: the oracle matched cuts by rounded coordinates across rings, so an outer cut rounded onto
    a hole vertex was charged to both rings."""
    yb = 0.1 * 3                                                    # the cut at x = 1 rounds UP
    tip = float(F(yb) / 3)
    assert F(tip) > F(yb) / 3
    outer = [(0., 0.), (3., yb), (3., 4.), (0., 4.)]
    hole = [(.5, .5), (1., tip), (2., 1.), (1., 2.)]
    rings = [np.asarray(r, float) for r in (outer, hole)]
    n = 8
    for path in _PATHS:
        result = _contract([outer, hole], **path)
        points = result.points
        bounds = []
        for k, (a, b) in enumerate(result.cut_ends.tolist()):
            fa, fb, fc = _fp(points[a]), _fp(points[b]), _fp(points[n + k])
            exact_y = fa[1] + (fc[0] - fa[0]) / (fb[0] - fa[0]) * (fb[1] - fa[1])
            bounds.append(abs(fc[1] - exact_y))
        tied = [k for k in range(len(result.cut_ends)) if tuple(points[n + k]) == (1., tip)]
        assert len(tied) == 1 and bounds[tied[0]] > 0                # the outer cut rounded onto the hole tip
        # independent count: each cut once, with the x distance of its neighbours on its own segment
        expected = F(0)
        ends = result.cut_ends.tolist()
        for k, (a, b) in enumerate(ends):
            on_segment = sorted([points[a, 0], points[b, 0]] + [points[n + j, 0] for j in range(len(ends))
                                                                if ends[j] == [a, b]])
            i = on_segment.index(points[n + k, 0])
            expected += bounds[k] * (F(on_segment[i + 1]) - F(on_segment[i - 1])) / 2
        assert _area_allowance(rings, result, bounds) == expected


# ----------------------------------------------------------------------------- N2 round 4 (Codex)
class _ScaledOnRead(np.ndarray):
    """Codex r4 M2: a subclass whose indexing doubles what is stored."""

    def __getitem__(self, key):
        return 2 * np.asarray(super().__getitem__(key))


class _ListSubclass(list):
    pass


def _foreign_forms():
    """Every input-form reproduction of Codex N2 rounds 1-4, and other forms the strict input
    contract refuses (owner decision N2 r4 (1)): only a list/tuple of exact float64 ndarrays."""
    stored = np.array([SQUARE])
    point = np.array(["0", "1"], dtype="U16384")
    z = np.array(SQUARE, dtype=complex)
    z[0, 0] = complex(0, float("nan"))
    masked = np.ma.array(SQUARE)
    masked[1, 0] = np.ma.masked
    return [
        ("r1_5_huge_int_ring", [[(0, 0), (1, 0), (0, 10 ** 400)]], None),
        ("r1_5_huge_int_extra", [SQ], [10 ** 400]),
        ("r1_5_complex_nan", [z], None),
        ("r2_4_int64_above_2_53", [np.array([(0, 0), (2 ** 53 + 1, 0), (0, 1)], dtype=np.int64)], None),
        ("r2_4_uint64", [np.array([(0, 0), (2 ** 64 - 1, 0), (0, 1)], dtype=np.uint64)], None),
        ("r3_2_generator_rings", (r for r in [SQ]), None),
        ("r3_2_iterator_ring", [iter(SQUARE)], None),
        ("r3_2_generator_extra", [SQ], (x for x in [0.5])),
        ("r3_3_masked_ring", [masked], None),
        ("r3_3_unmasked_masked_array", [np.ma.array(SQUARE, mask=False)], None),
        ("r3_3_masked_extra", [SQ], np.ma.array([0.5])),
        ("r3_3_masked_scalar_in_list", [SQ], [np.ma.masked]),
        ("r4_m1_stacked_empty_rings", np.empty((100_000, 0, 2)), None),
        ("r4_m1_wide_dtype_points", [[point] * 128], None),
        ("r4_m1_huge_range_extra", [SQ], range(10 ** 40)),
        ("r4_m1_huge_range_rings", range(10 ** 40), None),
        ("r4_m2_stacked_subclass", stored.view(_ScaledOnRead), None),
        ("r4_m2_subclass_ring", [stored[0].view(_ScaledOnRead)], None),
        ("r4_low_c_nested_extra", [SQ], [[0.25], [0.5]]),
        ("stacked_ndarray", stored, None),
        ("list_ring", [SQUARE], None),
        ("tuple_ring", [tuple(SQUARE)], None),
        ("list_of_arrays_ring", [[np.array(p) for p in SQUARE]], None),
        ("list_extra", [SQ], [0.5]),
        ("tuple_extra", [SQ], (0.5,)),
        ("scalar_extra", [SQ], 0.5),
        ("numpy_scalar_extra", [SQ], np.float64(0.5)),
        ("2d_extra", [SQ], np.array([[0.25], [0.5]])),
        ("int_extra", [SQ], np.array([1], dtype=np.int64)),
        ("list_subclass_rings", _ListSubclass([SQ]), None),
        ("dict_rings", {0: SQ}, None),
        ("none_rings", None, None),
        ("str_rings", "abc", None),
        ("bool_ring", [np.array(SQUARE, dtype=bool)], None),
        ("float32_ring", [np.array(SQUARE, dtype=np.float32)], None),
        ("int64_ring", [np.array(SQUARE, dtype=np.int64)], None),
        ("object_ring", [np.array(SQUARE, dtype=object)], None),
        ("string_ring", [np.array(SQUARE).astype(str)], None),
        ("big_endian_ring", [SQ.astype(">f8")], None),
        ("ring_three_columns", [np.zeros((4, 3))], None),
        ("ring_one_dimensional", [np.zeros(8)], None),
        ("ring_three_dimensional", [np.zeros((1, 4, 2))], None),
        ("ring_two_points", [np.zeros((2, 2))], None),
        ("no_rings", [], None),
    ]


#: largest traced allocation (bytes) allowed for an input-form refusal (it reads types and shapes only)
_FORM_PEAK = 100_000
_FORMS = _foreign_forms()


@pytest.mark.parametrize("name, rings, extra", _FORMS, ids=[f[0] for f in _FORMS])
def test_r4_the_input_contract_refuses_every_foreign_form_before_touching_data(name, rings, extra):
    """Medium (M1, M2) and Low (c), Codex r4, and the input-form reproductions of rounds 1-3: one
    strict rule (owner decision N2 r4 (1)), checked from types and shapes only: no conversion, no
    iteration of a foreign container, no raw exception (a huge range raised OverflowError), no large
    allocation (an empty stacked array and wide string points allocated 14 and 17 MB), and no
    subclass indexing (a stacked subclass meshed a stored unit square to area 4)."""
    caught = []

    def call():
        try:
            slab.triangulate_rings(rings, extra)
        except DisplayTrimRefused as error:
            caught.append(error)
    peak = _traced_peak(call)
    assert caught and caught[0].code == "TRIM_INVALID", caught
    assert _FORM in str(caught[0]), str(caught[0])
    assert peak < _FORM_PEAK, (peak, str(caught[0]))


def test_r4_m2_the_stored_values_are_what_is_meshed():
    stored = np.array([SQUARE])
    assert slab.triangulate_rings([stored[0]]).area == 1.0
    assert slab.triangulate_rings(list(stored)).area == 1.0                 # rows of a stacked array: views
    for rings in (stored.view(_ScaledOnRead), [stored[0].view(_ScaledOnRead)]):
        _refused("TRIM_INVALID", rings)


def test_r4_m3_a_lift_must_certify_coverage_as_well_as_orientation():
    """Medium (M3): fraction reconstruction separates an outer cut from a coincident hole tip;
    every lifted triangle stays positive, yet a point the certified 2D mesh covers falls into a gap.
    The doc says only: N3 must not rely on cut_fraction reconstruction."""
    yb = .1 * 3
    tip = float(F(yb) / 3)
    outer = [(0, 0), (3, yb), (3, 4), (0, 4)]
    hole = [(.5, .5), (1, tip), (2, 1), (1, 2)]
    m = _contract([outer, hole])                                    # the 2D mesh passes the exact oracle
    q = [tuple(F(float(x)) for x in p) for p in m.points]
    r = q[:8] + [tuple(F(float(x + F(float(f)) * (y - x))) for x, y in zip(q[a], q[b]))
                 for (a, b), f in zip(m.cut_ends.tolist(), m.cut_fraction.tolist())]

    def covered(p, pts):
        return any(min(_orient(pts[a], pts[b], p), _orient(pts[b], pts[c], p), _orient(pts[c], pts[a], p)) >= 0
                   for a, b, c in m.triangles.tolist())
    assert q[5] == q[9] and r[5] != r[9]                            # an outer cut equal to the hole tip
    assert all(_orient(*(r[i] for i in t)) > 0 for t in m.triangles.tolist())   # orientation survives
    p = F(3, 2), (r[5][1] + r[9][1] + 2) / 4
    assert covered(p, q) and not covered(p, r)                      # coverage does not
    text = " ".join(_DOCS.read_text(encoding="utf-8").split())
    assert "must not rely on cut_fraction reconstruction for orientation or coverage" in text


def test_r4_low_a_area_is_the_correctly_rounded_exact_area_on_every_path():
    """Low (a): extra cuts zeroed a subnormal area on both paths; and the fast and general paths
    disagreed in the last bits of `area` on 23 of the round-3 trims.  Owner decision N2 r4 (3): the
    correctly rounded exact area of the emitted triangulation, identical on every path."""
    s = 2.0 ** -537
    w = math.nextafter(s, math.inf)
    rect = np.array([(0., 0.), (w, 0.), (w, s / 2), (0., s / 2)])
    assert float(F(w) * F(s / 2)) == 5e-324
    for path in _PATHS:
        for extra in (None, np.array([.09 * s, .34 * s])):
            assert slab.triangulate_rings([rect], extra, **path).area == 5e-324, (path, extra)
    rng = np.random.default_rng(2604)
    checked = 0
    for case in range(160):
        k = int(rng.integers(2, 60))
        u = np.sort(rng.uniform(0, 1, k))
        u[0], u[-1] = 0., 1.
        low = rng.uniform(-.1, .1, k)
        ring = np.vstack((np.column_stack((u, low)), np.column_stack((u, low + rng.uniform(.01, 1, k)))[::-1]))
        ring = ring * 10.0 ** rng.uniform(-3, 3) + rng.uniform(-1e3, 1e3, 2) * (case % 2)
        extra = rng.uniform(ring[:, 0].min(), ring[:, 0].max(), int(rng.integers(0, 20))) if case % 3 else None
        fast = slab.triangulate_rings([ring], extra)
        general = slab.triangulate_rings([ring], extra, _fast=False)
        assert fast.area == general.area, (case, fast.area, general.area)
        if case % 8 == 0:
            check_mesh([ring], general, extra)                      # area == float(exact triangle sum)
            checked += 1
    assert checked == 20


# ----------------------------------------------------------------------------- N2 round 5
class _NameRaises(type):
    def __getattribute__(cls, name):
        if name == "__name__":
            raise RuntimeError("executed caller code")
        return super().__getattribute__(name)


class _Foreign(metaclass=_NameRaises):
    pass


def test_r5_m1_refusals_run_no_caller_code_and_allocate_little():
    """Medium (M1): a metaclass raised a raw RuntimeError from the refusal message (``__name__``),
    and an eight-element array with a 2e6-character dtype field name allocated about 6 MB while it
    was refused.  Owner decision N2 r5 (1): fixed refusal text, no caller object or dtype formatted."""
    import tracemalloc
    for rings, extra in (([_Foreign()], None), (_Foreign(), None), ([SQ], _Foreign())):
        _refused("TRIM_INVALID", rings, extra)
    bad = np.empty((4, 2), dtype=[("x" * 2_000_000, "f8")])
    bad_x = np.empty(4, dtype=[("x" * 2_000_000, "f8")])
    for rings, extra in (([bad], None), ([SQ], bad_x)):
        tracemalloc.start()
        try:
            _refused("TRIM_INVALID", rings, extra)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert peak < 100_000, peak


def test_r5_m2_a_zero_width_trim_interval_is_not_covered():
    """Medium (M2): a hole edge and an outer edge round onto the same segment.  The exact oracle
    passes, the triangles tile the positive-area pieces, and no triangle covers the interior of the
    coincident segment.  Owner decision N2 r5 (2): the mesh covers exactly the CLOSURE OF THE
    POSITIVE-AREA INTERIOR of the rounded trim; zero-width parts are not covered."""
    outer = np.array([(0., np.nextafter(1., 0.)), (4., 1.), (4., 3.), (0., 3.)])
    hole = np.array([(2.5, 1.), (3.5, 1.), (3.5, 2.), (2.5, 2.)])
    m = _contract([outer, hole])                                    # the exact oracle passes
    assert np.array_equal(m.points[8:10], hole[:2])
    q = [tuple(F(float(v)) for v in p) for p in m.points.tolist()]
    for p in ((F(3), F(1)), (F(11, 4), F(1)), (F(13, 4), F(1))):    # interior of the coincident edges
        assert not any(min(_orient(q[a], q[b], p), _orient(q[b], q[c], p), _orient(q[c], q[a], p)) >= 0
                       for a, b, c in m.triangles.tolist())

    # area accounting: the triangles' exact area is the rounded polygon's (rings with the cut points
    # inserted) - the zero-width part adds nothing - correctly rounded
    def poly_area(pts):
        return sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(pts, pts[1:] + pts[:1])) / 2
    exact = sum(abs(_orient(q[a], q[b], q[c])) for a, b, c in m.triangles.tolist()) / 2
    assert m.area == float(exact)
    hole_q = [tuple(F(float(v)) for v in p) for p in hole.tolist()]
    outer_q = [tuple(F(float(v)) for v in p) for p in outer.tolist()]
    assert exact <= poly_area(outer_q) - abs(poly_area(hole_q))     # the rounded polygon, not the exact
    text = " ".join(_DOCS.read_text(encoding="utf-8").split())
    assert "CLOSURE OF THE POSITIVE-AREA INTERIOR of the rounded trim" in text
    assert "N3 must not assume every trim boundary segment is a triangle edge" in text


def test_r5_low_area_may_round_to_zero_for_subnormal_scale_trims():
    """Low: a square of side 2**-538 is accepted on both paths with two strictly positive triangles,
    yet its exact area 2**-1076 rounds to 0.0.  Owner decision N2 r5 (4): the guarantee is correct
    rounding, including to zero."""
    s = 2.0 ** -538
    sq = np.array([(0., 0.), (s, 0.), (s, s), (0., s)])
    for path in _PATHS:
        m = slab.triangulate_rings([sq], **path)
        assert m.area == 0.0 and len(m.triangles) == 2
        q = [tuple(F(float(v)) for v in p) for p in m.points.tolist()]
        assert all(_orient(*(q[i] for i in t)) > 0 for t in m.triangles.tolist())
    text = " ".join(_DOCS.read_text(encoding="utf-8").split())
    assert "may round to 0 for subnormal-scale trims" in text
