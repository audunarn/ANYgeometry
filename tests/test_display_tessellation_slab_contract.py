"""Contract tests of the private slab triangulation (plan N2): exact coverage, typed refusals.

The oracle here is independent of the module: exact rational (``fractions.Fraction``) predicates
with a float filter whose error bound is Shewchuk's ``orient2d`` bound, and its own brute-force
simplicity check.  N2 round-1 (Codex) reproductions are the ``test_r1_*`` tests.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction

import numpy as np
import pytest

from anygeometry import _slab_triangulation as slab
from anygeometry.display_tessellation import DisplayTrimRefused

F = Fraction
SQUARE = [(0., 0.), (1., 0.), (1., 1.), (0., 1.)]
_EPS = 2.0 ** -52
_ORIENT_BOUND = (3.0 + 16.0 * 2.0 ** -53) * 2.0 ** -53       # Shewchuk ccwerrboundA
_ORDINATE_BOUND = 4.0 * _EPS                                 # the module's stated cut ordinate bound
_TINY = 2.0 ** -1000


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
    sign = np.where(np.abs(det) > _ORIENT_BOUND * (np.abs(t1) + np.abs(t2)), np.sign(det), 2).astype(np.int8)
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


def _rounded_rings(rings, result):
    """The rings with each cut point inserted on its segment: the polygon the mesh must tile."""
    vertices = np.vstack([np.asarray(r, float) for r in rings])
    n = len(vertices)
    groups = defaultdict(list)
    for k, (a, b) in enumerate(result.cut_ends):
        groups[(tuple(vertices[a]), tuple(vertices[b]))].append(n + k)
    used = 0
    out = []
    for ring in rings:
        c = _compress(ring)
        new = []
        for p, q in zip(c, c[1:] + c[:1]):
            new.append(p)
            ks = groups.get((p, q), []) + groups.get((q, p), [])
            used += len(ks)
            ks.sort(key=lambda k: result.points[k, 0], reverse=p[0] > q[0])
            new.extend(tuple(result.points[k]) for k in ks)
        out.append(new)
    assert used == len(result.cut_ends), "a cut point lies on no ring segment"
    return out


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
        assert 0.0 < f < 1.0
        fa, fb, fc = _fp((xa, ya)), _fp((xb, yb)), _fp((cx, cy))
        exact_f = (fc[0] - fa[0]) / (fb[0] - fa[0])
        assert abs(F(float(f)) - exact_f) <= 4 * F(_EPS)
        bound = F(_ORDINATE_BOUND * (abs(ya) + abs(yb)) + _TINY)
        assert abs(fc[1] - (fa[1] + exact_f * (fb[1] - fa[1]))) <= bound
        bounds.append(bound)
    # exact orientation: every triangle strictly counter-clockwise
    assert len(tri) and np.all(_orient_signs(pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]) > 0)
    # exact area: the triangles tile the rounded polygon exactly; it is within the stated bound
    rounded = _rounded_rings(rings, result)
    fr = [[_fp(p) for p in r] for r in rounded]
    region = abs(_signed_area(fr[0])) - sum(abs(_signed_area(h)) for h in fr[1:])
    fp = [_fp(p) for p in pts]
    total = sum(_orient(fp[a], fp[b], fp[c]) for a, b, c in tri.tolist()) / 2
    assert total == region
    exact = abs(_signed_area([_fp(p) for p in _compress(rings[0])])) - sum(
        abs(_signed_area([_fp(p) for p in _compress(h)])) for h in rings[1:])
    allowance = F(0)
    for ring in rounded:                                       # moving a cut by d changes the area by d w / 2
        m = len(ring)
        index = {p: i for i, p in enumerate(ring)}
        for k in range(len(result.cut_ends)):
            i = index.get(tuple(pts[n + k]))
            if i is not None:
                allowance += bounds[k] * abs(F(ring[(i + 1) % m][0]) - F(ring[i - 1][0])) / 2
    assert abs(total - exact) <= allowance
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
    rings = [np.asarray(r, float) for r in rings]
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
    _refused("TRIM_INVALID", [[(0., 0.), (1., 0.)]])


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
    [[]],
    [[(0., 0.), (1., 0.)]],
    [[(0., 0.), (1., 0.), (2., 0.)]],                                   # zero area
    [[(0., 0.), (0., 0.), (0., 0.), (0., 0.)]],                         # one point repeated
    [[(0., 0.), (1., 1.), (1., 0.), (0., 1.)]],                         # bow tie
    [[(0., 0.), (4., 0.), (4., 4.), (0., 4.), (2., -1.), (3., 5.)]],    # self-intersecting
    [SQUARE, [(5., 5.), (6., 5.), (6., 6.)]],                           # hole outside
    [SQUARE, SQUARE],                                                   # hole equals outer
    [[(0., 0.), (4., 0.), (4., 4.), (0., 4.)], [(1., 1.), (3., 1.), (3., 3.)], [(1.5, 1.), (3.5, 1.), (3.5, 3.)]],
    [[(0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.), (0., 0., 1.), (1., 0., 1.)]],   # not 2-D points
    [[0., 0., 1., 0., 1., 1.]],                                         # flat list
    [[("a", "b"), ("c", "d"), ("e", "f")]],
], ids=lambda r: str(r)[:40])
def test_invalid_input_is_a_typed_refusal(rings):
    _refused("TRIM_INVALID", rings)


@pytest.mark.parametrize("rings", [
    [[(0., 0.), (1., 0.), (float("nan"), 1.)]],
    [[(0., 0.), (1., 0.), (float("inf"), 1.)]],
    [[(0., 0.), (1., 0.), (0., 1.0000001e9)]],
    [np.array(SQUARE, dtype=bool)],
], ids=("nan", "inf", "above_1e9", "bool"))
def test_out_of_domain_input_is_a_typed_refusal(rings):
    """Owner decision N2 r1 (2): the declared numeric domain, like N1 (non-finite: OUT_OF_DOMAIN)."""
    _refused("OUT_OF_DOMAIN", rings)


def test_the_domain_bound_itself_is_admitted():
    ring = np.array([(-1e9, -1e9), (1e9, -1e9), (1e9, 1e9), (-1e9, 1e9)])
    _contract([ring])
    _contract([np.array(SQUARE, dtype=np.int64)])


def test_invalid_extra_lines_are_refused():
    _refused("OUT_OF_DOMAIN", [SQUARE], np.array([.5, float("nan")]))
    _refused("TRIM_INVALID", [SQUARE], ["a"])


# ----------------------------------------------------------------------------- N2 round-1 reproductions
def test_r1_1_a_proper_crossing_is_refused_whatever_the_cuts():
    """High: a cut through the crossing made the order reverse 'between' slabs and it was accepted."""
    ring = [(1, 2), (3, 2), (3, 0), (2, 4), (4, 2)]
    assert not brute_valid([ring])
    for cuts in (None, [2.5], [2.25, 2.5, 3.5], np.linspace(1, 4, 13)):
        _refused("TRIM_INVALID", [ring], cuts)
        _refused("TRIM_INVALID", [ring], cuts, _fast=False)


def test_r1_2_a_hole_sharing_a_sloped_edge_is_refused():
    """Medium: a hole sharing the sloped edge y = x; tiny cut gave overlapping triangles, f == 1."""
    x = 2. ** -55
    outer = [(0, 2), (1, 1), (0, 0)]
    hole = [(0, 0), (1, 1), (.5, .75)]
    for cuts in (None, [x]):
        _refused("TRIM_INVALID", [outer, hole], cuts)


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
    ring = [(0., -1.), (1., 0.), (1., 2. ** -53), (0., 1.)]
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
    assert not brute_valid(rings)
    for path in _PATHS:
        _refused("TRIM_INVALID", rings, **path)
        _refused("TRIM_INVALID", rings, [.5, 1.5, 2.5], **path)


def test_r1_5_conversion_and_overflow_are_typed_refusals():
    """Medium: oversized ints leaked OverflowError, complex lost its NaN, 1e154 overflowed the shoelace."""
    _refused("OUT_OF_DOMAIN", [[(0, 0), (1, 0), (0, 10 ** 400)]])
    _refused("OUT_OF_DOMAIN", [SQUARE], [10 ** 400])
    z = np.array(SQUARE, dtype=complex)
    z[0, 0] = complex(0, float("nan"))
    _refused("OUT_OF_DOMAIN", [z])
    _refused("OUT_OF_DOMAIN", [np.array(SQUARE, dtype=complex)])
    _refused("OUT_OF_DOMAIN", [np.array(SQUARE, dtype=object)])
    outer = 1e154 * np.array([(0, 0), (.9, .9), (1, 1), (0, 1)])
    hole = 1e154 * np.array([(1.25, 0), (1.5, 0), (1.5, .125), (1.25, .125)])
    with np.errstate(all="raise"):
        _refused("OUT_OF_DOMAIN", [outer, hole])


def _staggered(m):
    def box(a, b, c, d):
        return [(a, c), (b, c), (b, d), (a, d)]
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
    _refused("OUT_OF_DOMAIN", [SQUARE], np.linspace(0, 1, 200_001))
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
    ring = [(0., -1.), (1., 0.), (1., 2. ** -53), (0., 1.)]
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
