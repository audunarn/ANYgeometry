"""Exact-coverage triangulation of polygons with holes by vertical slabs (trapezoids).

Private helper of :mod:`anygeometry.display_tessellation`; pure NumPy, no Shapely.

The plane is cut by a vertical line through every ring vertex (plus optional extra lines).  In
each slab between two consecutive lines, the non-vertical ring segments that span it are sorted
by height and paired bottom/top (even-odd rule), which gives trapezoids whose union is exactly
the region: no triangulation heuristics, so combs, L-shapes and holes are covered by
construction.  A trapezoid's corners are ring vertices or *cut points* on a ring segment (their
segment and fraction are returned so a caller can place them on the segment's own image).  Any
point lying on a trapezoid's vertical side is used by its triangles, so the mesh has no
T-junctions.  Rings that cross, holes outside the outer ring or overlapping holes are refused
(:class:`SlabInvalid`), never drawn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .display_tessellation import DisplayTrimRefused

__all__ = ["SlabInvalid", "SlabTriangulation", "triangulate_rings"]

_AREA_RTOL = 1.0e-9


class SlabInvalid(DisplayTrimRefused):
    """The rings are not a valid region (crossing, holes outside or overlapping, degenerate).

    A typed display refusal with code ``TRIM_INVALID``.
    """

    def __init__(self, message: str):
        super().__init__("TRIM_INVALID", message)


def _as_ring(ring) -> np.ndarray:
    try:
        array = np.array(ring, dtype=float)
    except (TypeError, ValueError) as error:
        raise SlabInvalid(f"ring coordinates must be numbers ({error})") from None
    if array.ndim != 2 or array.shape[1] != 2:
        raise SlabInvalid(f"a ring must be an (n, 2) array of points, got shape {array.shape}")
    return array


@dataclass(frozen=True, eq=False)
class SlabTriangulation:
    """``points`` are the ring vertices (rings concatenated, in order), then the cut points.

    ``triangles`` index ``points`` counter-clockwise (positive area).  Cut point ``k`` (point
    ``len(vertices) + k``) lies on the ring segment ``cut_ends[k] = (a, b)`` at fraction
    ``cut_fraction[k]`` from ``a``; ``area`` is the covered (even-odd) area.
    """

    points: np.ndarray
    triangles: np.ndarray
    cut_ends: np.ndarray
    cut_fraction: np.ndarray
    area: float


def _signed_area(ring: np.ndarray, origin: np.ndarray) -> float:
    x, y = ring[:, 0] - origin[0], ring[:, 1] - origin[1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _drop_flat(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Zero-area triangles (collinear or coincident corners) carry nothing; remove them."""
    a, b, c = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    doubled = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    return triangles[doubled > 0.0]


def _monotone(ring: np.ndarray, extra_x) -> SlabTriangulation | None:
    """The same trapezoids for one x-monotone ring (most faces), without the general sweep.

    The ring must split into one chain of increasing and one of decreasing x, with at most one
    vertical segment at each end; otherwise (or if the chains touch or cross) ``None``.
    """
    n = len(ring)
    x, y = ring[:, 0], ring[:, 1]
    step = np.sign(np.roll(x, -1) - x)              # segment k: vertex k -> k + 1
    moving = np.flatnonzero(step != 0)
    if len(moving) < 2:
        return None
    signs = step[moving]
    changes = np.flatnonzero(signs != np.roll(signs, 1))
    if len(changes) != 2:
        return None
    # chains as ring vertex indices, each in increasing x; a run may end in one vertical segment
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
    lo, hi = x[up[0]], x[up[-1]]
    if x[down[0]] != lo or x[down[-1]] != hi:
        return None
    lines = np.concatenate((x[up], x[down]))
    if extra_x is not None and len(extra_x):
        extra = np.asarray(extra_x, dtype=float).reshape(-1)
        lines = np.concatenate((lines, extra[(extra > lo) & (extra < hi)]))
    xs = np.unique(lines)

    cut_ends, cut_fraction, cut_points = [], [], []
    ids = []
    for chain in (up, down):
        cx = x[chain]
        at = np.searchsorted(cx, xs)                 # cx[at] >= xs
        exact = cx[np.minimum(at, len(cx) - 1)] == xs
        ident = np.empty(len(xs), dtype=np.int64)
        ident[exact] = chain[at[exact]]
        inner = np.flatnonzero(~exact)
        left, right = chain[at[inner] - 1], chain[at[inner]]
        # the ring segment runs left -> right on the up chain, right -> left on the down chain
        a, b = (left, right) if chain is up else (right, left)
        f = (xs[inner] - x[a]) / (x[b] - x[a])
        ident[inner] = n + sum(len(c) for c in cut_fraction) + np.arange(len(inner))
        cut_ends.append(np.column_stack((a, b)))
        cut_fraction.append(f)
        cut_points.append(np.column_stack((xs[inner], y[a] + f * (y[b] - y[a]))))
        ids.append(ident)
    points = np.vstack([ring, *cut_points])
    y_up, y_down = points[ids[0], 1], points[ids[1], 1]
    middle = len(xs) // 2
    if middle == 0:
        return None
    probe = slice(max(middle - 1, 0), middle + 1)
    if np.sum(y_down[probe]) >= np.sum(y_up[probe]):
        bottom, top, yb, yt = ids[0], ids[1], y_up, y_down
    else:
        bottom, top, yb, yt = ids[1], ids[0], y_down, y_up
    if np.any(yt[1:-1] <= yb[1:-1]) or np.any(yt < yb):
        return None                                   # the chains touch or cross: general path
    area = float(np.sum(np.diff(xs) * ((yt[:-1] - yb[:-1]) + (yt[1:] - yb[1:]))) * 0.5)
    bl, br, tl, tr = bottom[:-1], bottom[1:], top[:-1], top[1:]
    triangles = np.vstack((np.column_stack((bl, br, tr))[br != tr], np.column_stack((bl, tr, tl))[bl != tl]))
    triangles = _drop_flat(points, triangles.astype(np.int64))
    return SlabTriangulation(points, triangles, np.vstack(cut_ends).astype(np.int64),
                             np.concatenate(cut_fraction), area)


def _monotone_small(ring, extra_x):
    """Pure-Python x-monotone slab triangulation for small rings (NumPy call overhead dominates
    there); same contract and result layout as :func:`_monotone`."""
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
    cut_ends, cut_fraction, points = [], [], [list(p) for p in pts]
    ids = []
    for chain, is_up in ((up, True), (down, False)):
        ident = []
        i = 0
        last = len(chain) - 1
        for x in xs:
            while i < last and xs_[chain[i + 1]] <= x:
                i += 1
            if xs_[chain[i]] == x:
                ident.append(chain[i])
                continue
            left, right = chain[i], chain[i + 1]
            a, b = (left, right) if is_up else (right, left)
            xa, ya = pts[a]
            xb, yb = pts[b]
            f = (x - xa) / (xb - xa)
            ident.append(len(points))
            points.append([x, ya + f * (yb - ya)])
            cut_ends.append((a, b))
            cut_fraction.append(f)
        ids.append(ident)
    y_up = [points[k][1] for k in ids[0]]
    y_down = [points[k][1] for k in ids[1]]
    m = len(xs) // 2
    if m == 0:
        return None
    if y_down[m - 1] + y_down[m] >= y_up[m - 1] + y_up[m]:
        bottom, top, yb, yt = ids[0], ids[1], y_up, y_down
    else:
        bottom, top, yb, yt = ids[1], ids[0], y_down, y_up
    if any(t < b for t, b in zip(yt, yb)) or any(t <= b for t, b in zip(yt[1:-1], yb[1:-1])):
        return None
    area = 0.0
    tris = []
    for j in range(len(xs) - 1):
        area += (xs[j + 1] - xs[j]) * ((yt[j] - yb[j]) + (yt[j + 1] - yb[j + 1]))
        bl, br, tl, tr = bottom[j], bottom[j + 1], top[j], top[j + 1]
        if br != tr:
            tris.append((bl, br, tr))
        if bl != tl:
            tris.append((bl, tr, tl))
    points = np.array(points, dtype=float)
    return SlabTriangulation(points, _drop_flat(points, np.array(tris, dtype=np.int64).reshape(-1, 3)),
                             np.array(cut_ends, dtype=np.int64).reshape(-1, 2), np.array(cut_fraction, dtype=float),
                             0.5 * area)


_SMALL = 48   # rings up to this many vertices take the pure-Python monotone path (measured crossover ~50)


def triangulate_rings(rings: Sequence[np.ndarray], extra_x: np.ndarray | None = None) -> SlabTriangulation:
    """Triangulate the region of ``rings[0]`` minus ``rings[1:]`` (each ring closed implicitly).

    ``extra_x`` adds slab lines (inside the region's x range) so no slab is wider than a caller's
    step.  Raises :class:`SlabInvalid` when the rings do not bound a valid region.
    """
    try:
        rings = [_as_ring(r) for r in rings]
    except TypeError:
        raise SlabInvalid("rings must be a sequence of (n, 2) point arrays") from None
    if extra_x is not None:
        try:
            extra_x = np.array(extra_x, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            raise SlabInvalid("extra_x must be a sequence of numbers") from None
        if not np.all(np.isfinite(extra_x)):
            raise SlabInvalid("extra_x must be finite")
    if not rings or any(len(r) < 3 for r in rings):
        raise SlabInvalid("every ring needs at least three vertices")
    if not all(np.all(np.isfinite(r)) for r in rings):
        raise SlabInvalid("ring coordinates must be finite")
    vertices = np.vstack(rings)
    count = len(vertices)
    sizes = np.array([len(r) for r in rings])
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    seg_a = np.arange(count)
    seg_b = seg_a + 1
    ends = starts + sizes
    seg_b[ends - 1] = starts                     # each ring closes on its first vertex

    origin = vertices.min(axis=0)
    ring_areas = [abs(_signed_area(r, origin)) for r in rings]
    extent = float(np.ptp(vertices, axis=0).max())
    if any(a <= 16.0 * len(r) * np.finfo(float).eps * extent * extent for a, r in zip(ring_areas, rings)):
        raise SlabInvalid("a ring has no area (collinear, repeated or self-cancelling vertices)")
    expected = abs(_signed_area(rings[0], origin)) - sum(abs(_signed_area(r, origin)) for r in rings[1:])
    rounding = 16.0 * count * np.finfo(float).eps * extent * extent
    if len(rings) == 1 and expected > 0.0:
        fast = (_monotone_small if count <= _SMALL else _monotone)(vertices, extra_x)
        if fast is not None and abs(fast.area - expected) <= _AREA_RTOL * max(expected, fast.area) + rounding:
            return fast
    xa, ya = vertices[seg_a, 0], vertices[seg_a, 1]
    xb, yb = vertices[seg_b, 0], vertices[seg_b, 1]
    lines = vertices[:, 0]
    if extra_x is not None and len(extra_x):
        extra = np.asarray(extra_x, dtype=float).reshape(-1)
        extra = extra[(extra > lines.min()) & (extra < lines.max())]
        lines = np.concatenate((lines, extra))
    xs = np.unique(lines)

    # (segment, slab) pairs for every non-vertical segment and every slab it spans
    sloped = np.flatnonzero(xa != xb)
    lo = np.minimum(xa[sloped], xb[sloped])
    hi = np.maximum(xa[sloped], xb[sloped])
    jl = np.searchsorted(xs, lo)
    jr = np.searchsorted(xs, hi)
    span = jr - jl
    seg = np.repeat(sloped, span)
    offsets = np.repeat(np.cumsum(span) - span, span)
    slab = np.repeat(jl, span) + (np.arange(len(seg)) - offsets)

    # cut points: segment s crosses the interior lines jl < j < jr
    cuts_per = np.maximum(span - 1, 0)
    cut_base = np.zeros(count, dtype=np.int64)
    cut_base[sloped] = np.cumsum(cuts_per) - cuts_per
    cut_seg = np.repeat(sloped, cuts_per)
    cut_offsets = np.repeat(np.cumsum(cuts_per) - cuts_per, cuts_per)
    cut_line = np.repeat(jl, cuts_per) + 1 + (np.arange(len(cut_seg)) - cut_offsets)
    dx = xb[cut_seg] - xa[cut_seg]
    cut_fraction = (xs[cut_line] - xa[cut_seg]) / dx
    cut_points = np.column_stack((xs[cut_line], ya[cut_seg] + cut_fraction * (yb[cut_seg] - ya[cut_seg])))
    jl_of = np.zeros(count, dtype=np.int64)
    jl_of[sloped] = jl

    def corner(segments, line):
        """Point id of ``segments`` at slab line ``line`` (a vertex, else its cut point)."""
        x = xs[line]
        at_a = x == xa[segments]
        at_b = x == xb[segments]
        cut = count + cut_base[segments] + (line - jl_of[segments] - 1)
        return np.where(at_a, seg_a[segments], np.where(at_b, seg_b[segments], cut))

    points = np.vstack((vertices, cut_points)) if len(cut_points) else vertices.copy()
    left = corner(seg, slab)
    right = corner(seg, slab + 1)
    y0, y1 = points[left, 1], points[right, 1]
    order = np.lexsort((y1, y0, y0 + y1, slab))
    seg, slab, left, right, y0, y1 = (a[order] for a in (seg, slab, left, right, y0, y1))
    if len(slab) and np.any(np.bincount(slab, minlength=len(xs)) % 2):
        raise SlabInvalid("a slab crosses an odd number of boundary segments (open or crossing rings)")
    same = slab[1:] == slab[:-1]
    scale = max(1.0, float(np.abs(points).max(initial=0.0)))
    slack = 64 * np.finfo(float).eps * scale
    if np.any(same & ((y0[1:] < y0[:-1] - slack) | (y1[1:] < y1[:-1] - slack))):
        raise SlabInvalid("boundary segments cross inside a slab")

    bottom_l, bottom_r = left[0::2], right[0::2]
    top_l, top_r = left[1::2], right[1::2]
    pair_slab = slab[0::2]
    x0, x1 = xs[pair_slab], xs[pair_slab + 1]
    heights = (points[top_l, 1] - points[bottom_l, 1]) + (points[top_r, 1] - points[bottom_r, 1])
    area = float(np.sum((x1 - x0) * heights) * 0.5)
    if not (expected > 0.0 and abs(area - expected) <= _AREA_RTOL * max(expected, abs(area)) + rounding):
        raise SlabInvalid(f"rings do not bound a valid region (covered {area!r}, expected {expected!r})")

    # every point on each slab line, ordered by height: vertical sides use all of them
    point_line = np.concatenate((np.searchsorted(xs, vertices[:, 0]), cut_line))
    by_line = np.lexsort((np.arange(len(points)), points[:, 1], point_line))
    position = np.empty(len(points), dtype=np.int64)
    position[by_line] = np.arange(len(points))

    def side(lower, upper):
        a, b = position[lower], position[upper]
        return np.minimum(a, b), np.maximum(a, b)

    left_lo, left_hi = side(bottom_l, top_l)
    right_lo, right_hi = side(bottom_r, top_r)
    plain = (left_hi - left_lo <= 1) & (right_hi - right_lo <= 1)

    triangles = []
    # plain trapezoids: diagonal bottom-left to top-right; a side that is one point gives one triangle
    p = np.flatnonzero(plain)
    bl, br, tl, tr = bottom_l[p], bottom_r[p], top_l[p], top_r[p]
    triangles.append(np.column_stack((bl, br, tr))[br != tr])
    triangles.append(np.column_stack((bl, tr, tl))[bl != tl])
    # trapezoids with further points on a vertical side: zip the two sides bottom to top
    for k in np.flatnonzero(~plain):
        lchain = [bottom_l[k], *by_line[left_lo[k] + 1:left_hi[k]], top_l[k]] if bottom_l[k] != top_l[k] else [bottom_l[k]]
        rchain = [bottom_r[k], *by_line[right_lo[k] + 1:right_hi[k]], top_r[k]] if bottom_r[k] != top_r[k] else [bottom_r[k]]
        i = j = 0
        made = []
        while i < len(lchain) - 1 or j < len(rchain) - 1:
            advance_left = j == len(rchain) - 1 or (
                i < len(lchain) - 1 and points[lchain[i + 1], 1] - points[bottom_l[k], 1]
                <= points[rchain[j + 1], 1] - points[bottom_r[k], 1])
            if advance_left:
                made.append((lchain[i], rchain[j], lchain[i + 1]))
                i += 1
            else:
                made.append((lchain[i], rchain[j], rchain[j + 1]))
                j += 1
        triangles.append(np.asarray(made, dtype=np.int64).reshape(-1, 3))
    triangles = np.vstack(triangles).astype(np.int64) if triangles else np.zeros((0, 3), dtype=np.int64)
    # zero-area triangles (coincident points, touching rings) carry nothing
    a, b, c = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    doubled = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    triangles = triangles[doubled > 0.0]
    cut_ends = np.column_stack((seg_a[cut_seg], seg_b[cut_seg])) if len(cut_seg) else np.zeros((0, 2), dtype=np.int64)
    return SlabTriangulation(points, triangles, cut_ends.astype(np.int64), cut_fraction, area)
