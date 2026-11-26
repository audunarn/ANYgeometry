"""Exact-coverage triangulation of polygons with holes by vertical slabs (trapezoids).

Private helper of :mod:`anygeometry.display_tessellation`; pure NumPy, no Shapely.

Input contract (owner decision, N2 round 1): the outer ring and every hole are SIMPLE polygons (no
crossing, no self-touch, no zero-width retrace; repeated consecutive points are allowed and
collapse), the holes lie strictly inside the outer ring and no two boundaries share a point.  The
decision is exact (a float filter with a rigorous error bound, exact rationals where it cannot
decide) and does not depend on ``extra_x``; anything else is :class:`SlabInvalid`
(``TRIM_INVALID``).  Inputs outside the declared numeric domain, features below float resolution
and trims above :data:`MAX_SLAB_INCIDENCES` are ``OUT_OF_DOMAIN``
(:class:`~anygeometry.display_tessellation.DisplayDomainRefused`).

The plane is cut by a vertical line through every ring vertex (plus optional extra lines).  In
each slab between two consecutive lines, the ring segments spanning it are sorted by height and
paired bottom/top, which gives trapezoids whose union is exactly the region.  A trapezoid's
corners are ring vertices or *cut points* on a ring segment (their segment and fraction are
returned so a caller can place them on the segment's own image).  Every point lying on a
trapezoid's vertical side is used by its triangles, so the mesh has no T-junctions.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

import numpy as np

from .display_tessellation import MAX_DISPLAY_COORDINATE, DisplayDomainRefused, DisplayTrimRefused

__all__ = ["MAX_SLAB_INCIDENCES", "SlabInvalid", "SlabTriangulation", "triangulate_rings"]

#: Largest number of (segment, slab) incidences (and of ``extra_x`` values) a trim may need; above
#: it the trim is ``OUT_OF_DOMAIN`` ("too complex for display") before anything is allocated, and
#: the caller keeps its old route.  Every result array is linear in it (cut points and triangles
#: are each at most about one per incidence).  See docs/DISPLAY_TESSELLATION.md for the measured
#: ordinary sizes it is set against.
MAX_SLAB_INCIDENCES = 200_000

_EPS = float(np.finfo(float).eps)
#: Absolute term of the ordinate error bound: covers gradual underflow in the four operations
#: (each at most 2**-1075, scaled by at most 2e9) with a wide margin.
_TINY = 2.0 ** -1000
_SMALL = 48   # rings up to this many vertices take the pure-Python monotone path (measured crossover ~50)


class SlabInvalid(DisplayTrimRefused):
    """The rings are not a valid region (not simple, touching, holes outside or nested).

    A typed display refusal with code ``TRIM_INVALID``.
    """

    def __init__(self, message: str):
        super().__init__("TRIM_INVALID", message)


@dataclass(frozen=True, eq=False)
class SlabTriangulation:
    """``points`` are the ring vertices (rings concatenated, in order, unchanged), then the cut points.

    ``triangles`` index ``points`` counter-clockwise (strictly positive area).  Cut point ``k``
    (point ``len(vertices) + k``) lies on the ring segment ``cut_ends[k] = (a, b)``, ``a`` its
    endpoint of lower x, at fraction ``cut_fraction[k]`` (strictly inside ``(0, 1)``) from ``a``;
    ``area`` is the covered area (float sum of the trapezoids).
    """

    points: np.ndarray
    triangles: np.ndarray
    cut_ends: np.ndarray
    cut_fraction: np.ndarray
    area: float


# Ordinate error bound.  A cut point's ordinate is computed as ``ya + f (yb - ya)`` with
# ``f = (x - xa) / (xb - xa)`` from the segment's lower-x end (``0 <= f <= 1``).  Five roundings on
# the product (two subtractions and the quotient in ``f``, the subtraction and the product) and one
# on the sum give at most ``6.01 u (|ya| + |yb|)`` (``u = eps / 2``).  ``B = 4 eps (|ya| + |yb|) +
# _TINY`` is that with a 1.33 margin, which also absorbs the rounding of the filter's own
# difference and sum; ``_TINY`` covers underflow.  Vertices are exact (``B = 0``).


def _numbers(value, what: str) -> np.ndarray:
    """A real numeric array (no arithmetic yet): strings are ``TRIM_INVALID``; complex, boolean,
    object (e.g. integers too large for an int64 or a float) arrays are ``OUT_OF_DOMAIN``."""
    try:
        array = np.asarray(value)
    except OverflowError:
        raise DisplayDomainRefused(f"{what}: a number too large for a float coordinate") from None
    except (TypeError, ValueError) as error:
        raise SlabInvalid(f"{what} must be numbers ({error})") from None
    kind = array.dtype.kind
    if kind in "USV":
        raise SlabInvalid(f"{what} must be numbers, got dtype {array.dtype}")
    if kind not in "fiu":
        raise DisplayDomainRefused(f"{what} must be real numbers (floats or integers), got dtype {array.dtype}")
    return array


def _in_domain(array: np.ndarray, what: str) -> np.ndarray:
    """Finite with ``|c| <= MAX_DISPLAY_COORDINATE``, checked before any conversion or arithmetic."""
    if array.size:
        if array.dtype.kind == "f":
            ok = bool(np.all(np.isfinite(array))) and float(np.abs(array).max()) <= MAX_DISPLAY_COORDINATE
        else:
            ok = -MAX_DISPLAY_COORDINATE <= int(array.min()) and int(array.max()) <= MAX_DISPLAY_COORDINATE
        if not ok:
            raise DisplayDomainRefused(f"{what} must be finite with |coordinate| <= {MAX_DISPLAY_COORDINATE:g}")
    return np.array(array, dtype=np.float64)


def _chains(x):
    """The increasing-x and decreasing-x vertex chains of an x-monotone ring (each listed in
    increasing x; at most one vertical segment at each end), else ``None``."""
    n = len(x)
    step = np.sign(np.roll(x, -1) - x)              # segment k: vertex k -> k + 1
    moving = np.flatnonzero(step != 0)
    if len(moving) < 2:
        return None
    signs = step[moving]
    changes = np.flatnonzero(signs != np.roll(signs, 1))
    if len(changes) != 2:
        return None
    first_up = moving[changes[0]] if signs[changes[0]] > 0 else moving[changes[1]]
    first_down = moving[changes[1]] if signs[changes[1]] < 0 else moving[changes[0]]
    up_segments = (first_up + np.arange((first_down - first_up) % n)) % n
    down_segments = (first_down + np.arange((first_up - first_down) % n)) % n
    for run, sign in ((up_segments, 1), (down_segments, -1)):
        kinds = step[run]
        if not (np.all(kinds[:-1] == sign) and kinds[-1] in (sign, 0)):
            return None
    up_segments = up_segments[step[up_segments] > 0]
    down_segments = down_segments[step[down_segments] < 0]
    up = np.concatenate((up_segments, [(up_segments[-1] + 1) % n]))
    down = np.concatenate((down_segments, [(down_segments[-1] + 1) % n]))[::-1]
    if not (np.all(np.diff(x[up]) > 0) and np.all(np.diff(x[down]) > 0)):
        return None
    if x[down[0]] != x[up[0]] or x[down[-1]] != x[up[-1]]:
        return None
    return up, down


def _monotone(ring, extra_x) -> SlabTriangulation | None:
    """The same trapezoids for one x-monotone ring (most faces), without the general sweep.

    ``ring`` has no repeated consecutive points.  A certified shortcut: it returns a result only
    when the float filter proves the two chains strictly apart at every interior slab line (so the
    ring is simple and the rounded ordinates are ordered); otherwise ``None`` (the general path
    then decides exactly).
    """
    x, y = ring[:, 0], ring[:, 1]
    n = len(ring)
    chains = _chains(x)
    if chains is None:
        return None
    up, down = chains
    lo, hi = x[up[0]], x[up[-1]]
    lines = np.concatenate((x[up], x[down]))
    if extra_x is not None and len(extra_x):
        lines = np.concatenate((lines, extra_x[(extra_x > lo) & (extra_x < hi)]))
    xs = np.unique(lines)

    cut_ends, cut_fraction, cut_points, cut_bound, ids = [], [], [], [], []
    for chain in (up, down):
        cx = x[chain]
        at = np.searchsorted(cx, xs)                 # cx[at] >= xs
        exact = cx[np.minimum(at, len(cx) - 1)] == xs
        ident = np.empty(len(xs), dtype=np.int64)
        ident[exact] = chain[at[exact]]
        inner = np.flatnonzero(~exact)
        a, b = chain[at[inner] - 1], chain[at[inner]]     # canonical: a has the lower x
        f = (xs[inner] - x[a]) / (x[b] - x[a])
        ident[inner] = n + sum(len(c) for c in cut_fraction) + np.arange(len(inner))
        cut_ends.append(np.column_stack((a, b)))
        cut_fraction.append(f)
        cut_points.append(np.column_stack((xs[inner], y[a] + f * (y[b] - y[a]))))
        cut_bound.append(4.0 * _EPS * (np.abs(y[a]) + np.abs(y[b])) + _TINY)
        ids.append(ident)
    fractions = np.concatenate(cut_fraction)
    if np.any(fractions <= 0.0) or np.any(fractions >= 1.0):
        return None
    points = np.vstack([ring, *cut_points])
    bound = np.concatenate([np.zeros(n), *cut_bound])
    y_up, y_down = points[ids[0], 1], points[ids[1], 1]
    middle = len(xs) // 2
    probe = slice(max(middle - 1, 0), middle + 1)
    if np.sum(y_down[probe]) >= np.sum(y_up[probe]):
        bottom, top = ids[0], ids[1]
    else:
        bottom, top = ids[1], ids[0]
    gap = points[top, 1] - points[bottom, 1]
    if not np.all(gap[1:-1] > bound[top[1:-1]] + bound[bottom[1:-1]]):
        return None
    for k in (0, -1):                                # ends: vertices (exact), shared or a vertical side
        if not (top[k] == bottom[k] or gap[k] > 0.0):
            return None
    area = float(np.sum(np.diff(xs) * (gap[:-1] + gap[1:])) * 0.5)
    _floor(area, n, max(hi - lo, float(y.max() - y.min())))
    bl, br, tl, tr = bottom[:-1], bottom[1:], top[:-1], top[1:]
    triangles = np.vstack((np.column_stack((bl, br, tr))[br != tr], np.column_stack((bl, tr, tl))[bl != tl]))
    return SlabTriangulation(points, triangles.astype(np.int64), np.vstack(cut_ends).astype(np.int64),
                             fractions, area)


def _monotone_small(ring, extra_x) -> SlabTriangulation | None:
    """Pure-Python :func:`_monotone` for small rings (NumPy call overhead dominates there); same
    certification, contract and result layout."""
    pts = ring.tolist()
    n = len(pts)
    xs_ = [p[0] for p in pts]
    step = [(1 if xs_[(k + 1) % n] > xs_[k] else -1 if xs_[(k + 1) % n] < xs_[k] else 0) for k in range(n)]
    moving = [k for k in range(n) if step[k]]
    if len(moving) < 2:
        return None
    changes = [i for i in range(len(moving)) if step[moving[i]] != step[moving[i - 1]]]
    if len(changes) != 2:
        return None
    a0, a1 = moving[changes[0]], moving[changes[1]]
    first_up, first_down = (a0, a1) if step[a0] > 0 else (a1, a0)
    up_run = [(first_up + i) % n for i in range((first_down - first_up) % n)]
    down_run = [(first_down + i) % n for i in range((first_up - first_down) % n)]
    for run, sign in ((up_run, 1), (down_run, -1)):
        if any(step[k] != sign for k in run[:-1]) or step[run[-1]] not in (sign, 0):
            return None
    up_seg = [k for k in up_run if step[k] > 0]
    down_seg = [k for k in down_run if step[k] < 0]
    up = up_seg + [(up_seg[-1] + 1) % n]
    down = (down_seg + [(down_seg[-1] + 1) % n])[::-1]
    lo, hi = xs_[up[0]], xs_[up[-1]]
    if xs_[down[0]] != lo or xs_[down[-1]] != hi:
        return None
    lines = set(xs_[k] for k in up)
    lines.update(xs_[k] for k in down)
    if extra_x is not None and len(extra_x):
        lines.update(v for v in extra_x.tolist() if lo < v < hi)
    xs = sorted(lines)
    eps4 = 4.0 * _EPS
    cut_ends, cut_fraction, points, bound = [], [], [list(p) for p in pts], [0.0] * n
    ids = []
    for chain in (up, down):
        ident = []
        i = 0
        last = len(chain) - 1
        for x in xs:
            while i < last and xs_[chain[i + 1]] <= x:
                i += 1
            if xs_[chain[i]] == x:
                ident.append(chain[i])
                continue
            a, b = chain[i], chain[i + 1]            # canonical: a has the lower x
            xa, ya = pts[a]
            xb, yb = pts[b]
            f = (x - xa) / (xb - xa)
            if not 0.0 < f < 1.0:
                return None
            ident.append(len(points))
            points.append([x, ya + f * (yb - ya)])
            bound.append(eps4 * (abs(ya) + abs(yb)) + _TINY)
            cut_ends.append((a, b))
            cut_fraction.append(f)
        ids.append(ident)
    y_up = [points[k][1] for k in ids[0]]
    y_down = [points[k][1] for k in ids[1]]
    m = len(xs) // 2
    if y_down[m - 1] + y_down[m] >= y_up[m - 1] + y_up[m]:
        bottom, top, yb, yt = ids[0], ids[1], y_up, y_down
    else:
        bottom, top, yb, yt = ids[1], ids[0], y_down, y_up
    last = len(xs) - 1
    for j in range(len(xs)):
        gap = yt[j] - yb[j]
        if 0 < j < last:
            if not gap > bound[top[j]] + bound[bottom[j]]:
                return None
        elif not (top[j] == bottom[j] or gap > 0.0):
            return None
    area = 0.0
    tris = []
    for j in range(last):
        area += (xs[j + 1] - xs[j]) * ((yt[j] - yb[j]) + (yt[j + 1] - yb[j + 1]))
        bl, br, tl, tr = bottom[j], bottom[j + 1], top[j], top[j + 1]
        if br != tr:
            tris.append((bl, br, tr))
        if bl != tl:
            tris.append((bl, tr, tl))
    ys = [p[1] for p in pts]
    _floor(0.5 * area, n, max(hi - lo, max(ys) - min(ys)))
    return SlabTriangulation(np.array(points, dtype=float), np.array(tris, dtype=np.int64).reshape(-1, 3),
                             np.array(cut_ends, dtype=np.int64).reshape(-1, 2), np.array(cut_fraction, dtype=float),
                             0.5 * area)


def _prepare(rings, extra_x):
    """Domain, structure and the preflight budget; returns the rings' vertices and segments."""
    try:
        raw = [_numbers(r, "ring coordinates") for r in rings]
    except TypeError:
        raise SlabInvalid("rings must be a sequence of (n, 2) point arrays") from None
    if not raw:
        raise SlabInvalid("no rings")
    for array in raw:
        if array.ndim != 2 or array.shape[1] != 2:
            raise SlabInvalid(f"a ring must be an (n, 2) array of points, got shape {array.shape}")
        if len(array) < 3:
            raise SlabInvalid("every ring needs at least three vertices")
    rings = [_in_domain(a, "ring coordinates") for a in raw]
    extra = None
    if extra_x is not None:
        extra = _numbers(extra_x, "extra_x").reshape(-1)
        if extra.size > MAX_SLAB_INCIDENCES:
            raise DisplayDomainRefused(f"trim too complex for display: {extra.size} extra lines "
                                       f"(MAX_SLAB_INCIDENCES = {MAX_SLAB_INCIDENCES})")
        extra = _in_domain(extra, "extra_x")

    vertices = np.vstack(rings)
    sizes = np.array([len(r) for r in rings])
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    ends = starts + sizes
    previous = np.arange(len(vertices)) - 1
    previous[starts] = ends - 1
    differs = np.any(vertices != vertices[previous], axis=1)
    # canonical vertex id: the first vertex of each run of repeated consecutive points (cyclic)
    canonical = np.empty(len(vertices), dtype=np.int64)
    compressed = []
    for s, e in zip(starts, ends):
        run_starts = np.flatnonzero(differs[s:e]) + s
        if len(run_starts) < 3:
            raise SlabInvalid("every ring needs at least three distinct consecutive vertices")
        canonical[s:e] = run_starts[np.searchsorted(run_starts, np.arange(s, e), side="right") - 1]
        compressed.append(run_starts)
    seg_a = np.concatenate(compressed)
    seg_b = np.concatenate([np.roll(c, -1) for c in compressed])
    seg_ring = np.repeat(np.arange(len(rings)), [len(c) for c in compressed])

    prep = dict(vertices=vertices, extra=extra, canonical=canonical, compressed=compressed, seg_a=seg_a,
                seg_b=seg_b, seg_ring=seg_ring)
    # preflight: each segment spans fewer slabs than there are lines; count exactly only when that
    # bound exceeds the budget (the fast path then never pays for it)
    if len(seg_a) * (len(vertices) + (0 if extra is None else len(extra))) > MAX_SLAB_INCIDENCES:
        _slabs(prep)
    return prep


def _slabs(prep):
    """Slab lines, canonical sloped segments and their slab ranges; refuses above the budget."""
    if "xs" in prep:
        return prep
    vertices, extra, seg_a, seg_b = prep["vertices"], prep["extra"], prep["seg_a"], prep["seg_b"]
    x = vertices[:, 0]
    lines = x
    if extra is not None and len(extra):
        lines = np.concatenate((x, extra[(extra > x.min()) & (extra < x.max())]))
    xs = np.unique(lines)
    xa, xb = x[seg_a], x[seg_b]
    sloped = np.flatnonzero(xa != xb)
    forward = xa[sloped] < xb[sloped]
    sp = np.where(forward, seg_a[sloped], seg_b[sloped])      # canonical: lower-x endpoint first
    sq = np.where(forward, seg_b[sloped], seg_a[sloped])
    jl, jr = np.searchsorted(xs, x[sp]), np.searchsorted(xs, x[sq])
    incidences = int(np.sum(jr - jl))
    if incidences > MAX_SLAB_INCIDENCES:
        raise DisplayDomainRefused(f"trim too complex for display: {incidences} segment-slab incidences "
                                   f"(MAX_SLAB_INCIDENCES = {MAX_SLAB_INCIDENCES})")
    prep.update(xs=xs, sp=sp, sq=sq, sring=prep["seg_ring"][sloped], jl=jl, jr=jr, incidences=incidences)
    return prep


def _floor(area, count, extent):
    """The area floor of :func:`_area_floor` for a ring whose (fast path) area is known."""
    if area <= 16.0 * count * _EPS * extent * extent:
        raise DisplayDomainRefused("a ring's area is below display resolution relative to its own extent")


def _area_floor(vertices, compressed):
    """``OUT_OF_DOMAIN`` for a ring whose float area is not resolved relative to ITS OWN extent
    (a backstop: a simple ring has positive exact area; this refuses slivers below resolution)."""
    for c in compressed:
        ring = vertices[c]
        ring = ring - ring.min(axis=0)
        doubled = abs(float(np.dot(ring[:, 0], np.roll(ring[:, 1], -1)) - np.dot(np.roll(ring[:, 0], -1), ring[:, 1])))
        _floor(0.5 * doubled, len(c), float(ring.max()))


def triangulate_rings(rings: Sequence[np.ndarray], extra_x: np.ndarray | None = None, *,
                      _fast: bool = True) -> SlabTriangulation:
    """Triangulate the region of ``rings[0]`` minus ``rings[1:]`` (each ring closed implicitly).

    ``extra_x`` adds slab lines (inside the region's x range) so no slab is wider than a caller's
    step.  Raises :class:`SlabInvalid` (``TRIM_INVALID``) when a ring is not simple, two rings
    touch, or a hole is outside the outer ring or inside another hole (decided exactly,
    independent of ``extra_x``), and ``DisplayDomainRefused`` (``OUT_OF_DOMAIN``) outside the
    numeric domain, above :data:`MAX_SLAB_INCIDENCES` or for a feature below float resolution.
    ``_fast=False`` (private, for tests) skips the x-monotone shortcut.
    """
    prep = _prepare(rings, extra_x)
    vertices, compressed = prep["vertices"], prep["compressed"]
    if _fast and len(compressed) == 1:
        ring_ids = compressed[0]
        fast = (_monotone_small if len(ring_ids) <= _SMALL else _monotone)(vertices[ring_ids], prep["extra"])
        if fast is not None:
            n, m = len(vertices), len(ring_ids)
            remap = np.concatenate((ring_ids, n + np.arange(len(fast.points) - m)))
            return SlabTriangulation(np.vstack((vertices, fast.points[m:])), remap[fast.triangles],
                                     ring_ids[fast.cut_ends].reshape(-1, 2), fast.cut_fraction, fast.area)
    return _general(prep)


def _general(prep) -> SlabTriangulation:
    _slabs(prep)
    vertices, canonical, xs = prep["vertices"], prep["canonical"], prep["xs"]
    sp, sq, sring, jl, jr = prep["sp"], prep["sq"], prep["sring"], prep["jl"], prep["jr"]
    n = len(vertices)
    xp, yp = vertices[sp, 0], vertices[sp, 1]
    xq, yq = vertices[sq, 0], vertices[sq, 1]

    # (segment, slab) incidences: sloped segment s spans the slabs jl[s] .. jr[s] - 1
    span = jr - jl
    count = prep["incidences"]
    inc_seg = np.repeat(np.arange(len(sp)), span)
    inc_slab = jl[inc_seg] + (np.arange(count) - np.repeat(np.cumsum(span) - span, span))
    # cut points: segment s crosses the interior lines jl < j < jr; canonical form from its lower-x end
    cuts_per = span - 1
    cut_base = np.cumsum(cuts_per) - cuts_per
    cut_seg = np.repeat(np.arange(len(sp)), cuts_per)
    cut_line = jl[cut_seg] + 1 + (np.arange(len(cut_seg)) - np.repeat(cut_base, cuts_per))
    cut_x = xs[cut_line]
    cut_fraction = (cut_x - xp[cut_seg]) / (xq[cut_seg] - xp[cut_seg])
    cut_y = yp[cut_seg] + cut_fraction * (yq[cut_seg] - yp[cut_seg])
    points = np.vstack((vertices, np.column_stack((cut_x, cut_y))))
    py = points[:, 1]
    bound = np.concatenate((np.zeros(n), 4.0 * _EPS * (np.abs(yp) + np.abs(yq))[cut_seg] + _TINY))

    def at_line(line):
        s = inc_seg
        return np.where(line == jl[s], sp[s], np.where(line == jr[s], sq[s], n + cut_base[s] + line - jl[s] - 1))

    left, right = at_line(inc_slab), at_line(inc_slab + 1)

    exact_cache: dict[int, Fraction] = {}

    def exact_y(pid: int) -> Fraction:
        value = exact_cache.get(pid)
        if value is None:
            if pid < n:
                value = Fraction(float(py[pid]))
            else:
                s = cut_seg[pid - n]
                xa, ya, xb, yb = (Fraction(float(v)) for v in (xp[s], yp[s], xq[s], yq[s]))
                value = ya + (Fraction(float(cut_x[pid - n])) - xa) * (yb - ya) / (xb - xa)
            exact_cache[pid] = value
        return value

    def certified_increasing(a, b):
        """True where ``y[b] > y[a]`` is proved by the float filter alone."""
        return (py[b] - py[a]) > (bound[a] + bound[b])

    # 1. exact order of every boundary point on each slab line (vertices by canonical id, cut points)
    vertex_ids = np.unique(canonical)
    line_points = np.concatenate((vertex_ids, np.arange(n, len(points))))
    point_line = np.full(len(points), -1, dtype=np.int64)
    point_line[vertex_ids] = np.searchsorted(xs, vertices[vertex_ids, 0])
    point_line[n:] = cut_line
    by_line = line_points[np.lexsort((line_points, py[line_points], point_line[line_points]))]
    lines_of = point_line[by_line]
    same = np.flatnonzero(lines_of[1:] == lines_of[:-1])
    doubtful = same[~certified_increasing(by_line[same], by_line[same + 1])]
    below_resolution = False
    for line in np.unique(lines_of[doubtful]):
        lo, hi = np.searchsorted(lines_of, [line, line + 1])
        block = by_line[lo:hi].tolist()
        exact_block = sorted(block, key=lambda pid: (exact_y(pid), pid))
        for u, v in zip(exact_block, exact_block[1:]):
            if exact_y(u) == exact_y(v):
                raise SlabInvalid("two boundary points coincide (rings touch or cross at a point)")
        if exact_block != block or any(py[u] >= py[v] for u, v in zip(exact_block, exact_block[1:])):
            below_resolution = True                   # the rounded order is not the exact one
        by_line[lo:hi] = exact_block
    position = np.empty(len(points), dtype=np.int64)
    position[by_line] = np.arange(len(by_line))

    # 2. a vertical segment has no other boundary point on it (its endpoints adjacent on their line)
    seg_a, seg_b = prep["seg_a"], prep["seg_b"]
    vertical = np.flatnonzero(vertices[seg_a, 0] == vertices[seg_b, 0])
    if len(vertical) and np.any(np.abs(position[seg_a[vertical]] - position[seg_b[vertical]]) != 1):
        raise SlabInvalid("a boundary touches or retraces a vertical edge")

    # 3. inside each slab the segments keep their order from the left line to the right line
    order = np.lexsort((position[right], position[left], inc_slab))
    inc_seg, inc_slab, left, right = inc_seg[order], inc_slab[order], left[order], right[order]
    same = inc_slab[1:] == inc_slab[:-1]
    step_left = position[left[1:]] - position[left[:-1]]
    step_right = position[right[1:]] - position[right[:-1]]
    if np.any(same & (step_right < 0)):
        raise SlabInvalid("boundary segments cross inside a slab")
    if np.any(same & (step_left == 0) & (step_right == 0)):
        raise SlabInvalid("boundary segments overlap (zero-width retrace or shared edge)")

    # 4. holes strictly inside the outer ring and not inside each other (each slab walked bottom-up)
    if count:
        ring_of = sring[inc_seg]
        slab_start = np.concatenate(([0], np.flatnonzero(~same) + 1))
        first = np.repeat(slab_start, np.diff(np.concatenate((slab_start, [count]))))
        grouped = np.lexsort((np.arange(count), ring_of, inc_slab))
        new_group = np.ones(count, dtype=bool)
        new_group[1:] = ((inc_slab[grouped][1:] != inc_slab[grouped][:-1])
                         | (ring_of[grouped][1:] != ring_of[grouped][:-1]))
        group_first = np.maximum.accumulate(np.where(new_group, np.arange(count), 0))
        occurrence = np.empty(count, dtype=np.int64)
        occurrence[grouped] = np.arange(count) - group_first
        is_outer = ring_of == 0
        delta = np.where(is_outer, 0, np.where(occurrence % 2 == 0, 1, -1))
        open_after = np.cumsum(delta)
        open_after = open_after - (open_after[first] - delta[first])
        outer_before = np.cumsum(is_outer)
        outer_before = outer_before - (outer_before[first] - is_outer[first]) - is_outer
        entering = delta > 0
        if (np.any(is_outer & (open_after != 0)) or np.any(entering & ((open_after != 1) | (outer_before % 2 == 0)))
                or np.any((delta < 0) & (open_after != 0))):
            raise SlabInvalid("a hole lies outside the outer ring or inside another hole")

    # rounding (the topology above is exact): every computed value must resolve it
    if np.any(cut_fraction <= 0.0) or np.any(cut_fraction >= 1.0):
        raise DisplayDomainRefused("slab too narrow to resolve: a cut fraction rounds onto a segment end")
    if below_resolution:
        raise DisplayDomainRefused("feature below display resolution: rounded ordinates on a slab line "
                                   "are not strictly in their exact order")
    _area_floor(vertices, prep["compressed"])

    # trapezoids: bottom/top pairs in each slab; sides use every point between them on their line
    bottom_l, bottom_r = left[0::2], right[0::2]
    top_l, top_r = left[1::2], right[1::2]
    pair_slab = inc_slab[0::2]
    heights = (py[top_l] - py[bottom_l]) + (py[top_r] - py[bottom_r])
    area = float(np.sum((xs[pair_slab + 1] - xs[pair_slab]) * heights) * 0.5)
    left_lo, left_hi = position[bottom_l], position[top_l]
    right_lo, right_hi = position[bottom_r], position[top_r]
    plain = (left_hi - left_lo <= 1) & (right_hi - right_lo <= 1)

    # Each triangle has one vertical side between two distinct points of one line (strictly ordered
    # floats) and its third corner on the other line: its exact area is (side height) x (slab
    # width) / 2 > 0, so no determinant is evaluated.  Only a side of zero height (one shared point)
    # gives no triangle.
    triangles = []
    p = np.flatnonzero(plain)
    bl, br, tl, tr = bottom_l[p], bottom_r[p], top_l[p], top_r[p]
    triangles.append(np.column_stack((bl, br, tr))[br != tr])
    triangles.append(np.column_stack((bl, tr, tl))[bl != tl])
    for k in np.flatnonzero(~plain):
        lchain = by_line[left_lo[k]:left_hi[k] + 1]
        rchain = by_line[right_lo[k]:right_hi[k] + 1]
        i = j = 0
        made = []
        while i < len(lchain) - 1 or j < len(rchain) - 1:
            advance_left = j == len(rchain) - 1 or (
                i < len(lchain) - 1 and py[lchain[i + 1]] - py[bottom_l[k]] <= py[rchain[j + 1]] - py[bottom_r[k]])
            if advance_left:
                made.append((lchain[i], rchain[j], lchain[i + 1]))
                i += 1
            else:
                made.append((lchain[i], rchain[j], rchain[j + 1]))
                j += 1
        triangles.append(np.asarray(made, dtype=np.int64).reshape(-1, 3))
    triangles = np.vstack(triangles).astype(np.int64)
    cut_ends = np.column_stack((sp[cut_seg], sq[cut_seg])).astype(np.int64).reshape(-1, 2)
    return SlabTriangulation(points, triangles, cut_ends, cut_fraction, area)
