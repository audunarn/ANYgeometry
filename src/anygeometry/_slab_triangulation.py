"""Exact-coverage triangulation of polygons with holes by vertical slabs (trapezoids).

Private helper of :mod:`anygeometry.display_tessellation`; pure NumPy, no Shapely.

Input contract (owner decision, N2 round 1): the outer ring and every hole are SIMPLE polygons (no
crossing, no self-touch, no zero-width retrace; repeated consecutive points are allowed and
collapse), the holes lie strictly inside the outer ring and no two boundaries share a point.  The
decision is exact and does not depend on ``extra_x``; anything else is :class:`SlabInvalid`
(``TRIM_INVALID``).  The input FORM is strict (owner decision, N2 round 4): ``rings`` is a list or
tuple of exact ``numpy.ndarray`` float64 ``(n, 2)`` arrays, ``extra_x`` None or an exact float64 1-D
ndarray; it is checked first, from types and shapes only, and anything else is ``TRIM_INVALID``.
Non-finite coordinates or ``|c| > 1e9``, trims above the preflight budget and, as a fail-closed
guard only, a rounding reversal are ``OUT_OF_DOMAIN``
(:class:`~anygeometry.display_tessellation.DisplayDomainRefused`).

The plane is cut by a vertical line through every ring vertex (plus optional extra lines).  In
each slab between two consecutive lines, the ring segments spanning it are sorted by height and
paired bottom/top, which gives trapezoids whose union is exactly the region.  A trapezoid's
corners are ring vertices or *cut points* on a ring segment (their segment and fraction are
returned as provenance metadata only: the certified product is the 2D mesh, and a lift that
rebuilds cut points from their fractions can invert a triangle or open a gap, so any lift must
certify both the orientation and the coverage of the lifted mesh itself; owner decisions, N2
rounds 3 and 4).  A cut point's fraction and
ordinate are the CORRECTLY ROUNDED values of the exact interpolation (owner decision, N2 round
2): rounding is monotone, so the exact order on a slab line is kept as a non-strict order (ties
allowed, never a reversal).  Every point lying on a trapezoid's vertical side is used by its
triangles, so the mesh has no T-junctions.  ``area`` is the correctly rounded exact area of the
emitted triangles (owner decision, N2 round 4), the same bits on every path.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

import numpy as np

from .display_tessellation import MAX_DISPLAY_COORDINATE, DisplayDomainRefused, DisplayTrimRefused

__all__ = ["MAX_SLAB_INCIDENCES", "MAX_SLAB_OUTPUT", "MAX_SLAB_RINGS", "MAX_SLAB_RING_POINTS", "SlabInvalid",
           "SlabTriangulation", "triangulate_rings"]

#: Largest number of (segment, slab) incidences (and of ``extra_x`` values) a trim may need.
MAX_SLAB_INCIDENCES = 200_000
#: Largest total number of input ring points (every point counts: duplicates and the points of
#: vertical segments are allocated and returned too).
MAX_SLAB_RING_POINTS = 50_000
#: Largest number of rings (every ring needs three points).
MAX_SLAB_RINGS = MAX_SLAB_RING_POINTS // 3
#: Largest preflight bound on the number of output triangles, ``I + 2 (N + cuts)`` (``I``
#: incidences, ``N`` input points, ``cuts = I - sloped segments`` cut points).  Above any of the
#: three the trim is ``OUT_OF_DOMAIN`` ("too complex for display") before anything is built, and
#: the caller keeps its old route.  See docs/DISPLAY_TESSELLATION.md for the ordinary sizes.
MAX_SLAB_OUTPUT = 600_000

_EPS = float(np.finfo(float).eps)
_U = _EPS / 2.0
_SMALL = 48            # rings up to this many vertices take the pure-Python monotone path
_EXACT_ROWS = 64       # up to this many cut points exact integer rounding is cheaper (measured)
_SPLIT = 134217729.0   # 2**27 + 1, Veltkamp's splitter
_SAFE = 2.0 ** -450    # a two_prod operand is zero or at least this (no underflow in its terms)
_ABS = 2.0 ** -1060    # absolute slack for gradual underflow (each rounding at most 2**-1075)
_EXACT_PAIRS = 24      # up to this many trapezoids the exact integer area is cheaper (measured)
_TINY = 2.0 ** -1074   # the smallest subnormal
#: the input-form refusal (owner decision, N2 round 4)
_FORM = "input must be a list/tuple of float64 (n, 2) ndarrays"


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
    endpoint of lower x, at fraction ``cut_fraction[k]`` from ``a`` (the exact fraction correctly
    rounded, in ``[0, 1]``; 0 or 1 only when the exact one is within half an ulp of it).
    ``cut_ends`` and ``cut_fraction`` are provenance metadata, not a certified lift.  ``area`` is
    the exact area of the triangles, correctly rounded (the same bits on every path).
    """

    points: np.ndarray
    triangles: np.ndarray
    cut_ends: np.ndarray
    cut_fraction: np.ndarray
    area: float


# ----------------------------------------------------------------------------- input and domain
def _too_complex(what: str):
    return DisplayDomainRefused(f"trim too complex for display: {what}")


# ----------------------------------------------------------------------------- correctly rounded cuts
# A cut point on the segment from (xa, ya) to (xb, yb) (xa < x < xb, the lower-x end first) has the
# exact fraction f* = N / D and ordinate y* = ya + N M / D, N = x - xa, D = xb - xa, M = yb - ya.
# The returned values are RN(f*) and RN(y*).  Filter: a float candidate c is certified to be RN(v*)
# when the residual R = (v* - c) D, evaluated from error-free transformations (two_sum, Dekker's
# two_prod; all exact for domain values, the products guarded against underflow), satisfies
# |R| + E < h D with h half the smaller gap of c to its float neighbours and E a rigorous bound on
# the residual's evaluation error; or when R is exactly zero.  Rows it cannot certify (residual
# within E of a rounding boundary, a cancelling ordinate near 0, tiny operands) are rounded exactly
# from integers (CPython's int / int is correctly rounded, gradual underflow included).

def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _two_prod(a, b):
    p = a * b
    t = _SPLIT * a
    ah = t - (t - a)
    al = a - ah
    t = _SPLIT * b
    bh = t - (t - b)
    bl = b - bh
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def _safe(*values):
    ok = True
    for v in values:
        m = np.abs(v)
        ok = ok & ((m == 0.0) | (m >= _SAFE))
    return ok


def _terms_f(f, d1, d2, n1, n2):
    """Residual ``N - f D`` as ``s + sum(rest) + sum(u v for cross)``, ``s`` and ``rest`` exact."""
    p, e = _two_prod(f, d1)
    s, t = _two_sum(n1, -p)
    return s, [t, n2, -e], [(-f, d2)], _safe(f, d1)


def _terms_y(y, ya, d1, d2, n1, n2, m1, m2):
    """Residual ``(ya - y) D + N M`` in the same form."""
    a1, a2 = _two_sum(ya, -y)
    p1, e1 = _two_prod(a1, d1)
    p2, e2 = _two_prod(n1, m1)
    s, t = _two_sum(p1, p2)
    return (s, [t, e1, e2], [(a1, d2), (a2, d1), (a2, d2), (n1, m2), (n2, m1), (n2, m2)],
            _safe(a1, d1, n1, m1))


def _residual(s, rest, cross):
    """The float residual ``R = s + S`` and the magnitude sum of the terms of ``S``."""
    total = rest[0]
    magnitude = np.abs(rest[0])
    for t in rest[1:] + [u * v for u, v in cross]:
        total = total + t
        magnitude = magnitude + np.abs(t)
    return s + total, magnitude


def _certified(c, d1, s, rest, cross, guard):
    """True where ``c`` is proved to be the correctly rounded value.

    The exact residual is ``s + sum(rest) + sum(u * v for u, v in cross)`` (``s`` and ``rest``
    exact floats, at most nine terms besides ``s``).  Evaluated as ``R = s + S``: its error is at
    most ``8.01 u`` times the magnitude sum of the terms of ``S`` (summation) plus ``u`` per cross
    product plus ``u |R|``, so ``E = 16 u (sum |terms| + |R|) + _ABS`` bounds it with margin (the
    rounding of ``E`` itself included; ``_ABS`` covers underflow).  ``c`` is RN of the exact value
    when ``|R| + E < h D`` (``h`` half the smaller gap to the float neighbours of ``c``), or when the
    residual is exactly zero (every term zero).
    """
    residual, magnitude = _residual(s, rest, cross)
    error = 16.0 * _U * (magnitude + np.abs(residual)) + _ABS
    gap = np.minimum(c - np.nextafter(c, -np.inf), np.nextafter(c, np.inf) - c)
    room = (0.5 * gap) * d1 * (1.0 - 8.0 * _U) - _ABS            # <= h D (|d2| <= u d1)
    ok = (np.abs(residual) + error) * (1.0 + 8.0 * _U) < room
    zero = s == 0.0
    for t in rest:
        zero &= t == 0.0
    for u, v in cross:
        zero &= (u == 0.0) | (v == 0.0)
    return guard & (ok | zero)


def _cut_values_exact(xa, ya, xb, yb, x):
    """Correctly rounded fractions and ordinates from integers (lists of floats in, lists out)."""
    fs, ys = [], []
    for a, b, c, d, e in zip(xa, ya, xb, yb, x):
        ratios = (a.as_integer_ratio(), b.as_integer_ratio(), c.as_integer_ratio(), d.as_integer_ratio(),
                  e.as_integer_ratio())
        q = max(r[1] for r in ratios)                      # powers of two: q is a common denominator
        pa, pb, pc, pd, pe = (num * (q // den) for num, den in ratios)
        span, part = pc - pa, pe - pa
        fs.append(part / span + 0.0)
        ys.append((pb * span + part * (pd - pb)) / (q * span) + 0.0)
    return fs, ys


def _cut_values(xa, ya, xb, yb, x):
    """``(RN(f*), RN(y*))`` for arrays of cut rows (canonical: ``xa < x < xb``)."""
    if len(x) <= _EXACT_ROWS:
        fs, ys = _cut_values_exact(xa.tolist(), ya.tolist(), xb.tolist(), yb.tolist(), x.tolist())
        return np.array(fs, dtype=float), np.array(ys, dtype=float)
    with np.errstate(all="ignore"):                        # also when called on its own
        return _cut_values_filtered(xa, ya, xb, yb, x)


def _cut_values_filtered(xa, ya, xb, yb, x):
    d1, d2 = _two_sum(xb, -xa)
    n1, n2 = _two_sum(x, -xa)
    m1, m2 = _two_sum(yb, -ya)
    # candidates: the plain formulas, then one correction step c + R / D (no bound needed here)
    f = n1 / d1
    y = ya + f * m1
    f = f + _residual(*_terms_f(f, d1, d2, n1, n2)[:3])[0] / d1
    y = y + _residual(*_terms_y(y, ya, d1, d2, n1, n2, m1, m2)[:3])[0] / d1
    ok = _certified(f, d1, *_terms_f(f, d1, d2, n1, n2))
    ok &= _certified(y, d1, *_terms_y(y, ya, d1, d2, n1, n2, m1, m2))
    rows = np.flatnonzero(~ok)
    if len(rows):
        fs, ys = _cut_values_exact(xa[rows].tolist(), ya[rows].tolist(), xb[rows].tolist(), yb[rows].tolist(),
                                   x[rows].tolist())
        f[rows], y[rows] = fs, ys
    f, y = f + 0.0, y + 0.0
    if not (bool(np.all(np.isfinite(f))) and bool(np.all(np.isfinite(y)))):
        raise DisplayDomainRefused("a cut value is not finite (unreachable inside the domain)")
    return f, y


# ----------------------------------------------------------------------------- area floor
def _floor(ring) -> None:
    """``OUT_OF_DOMAIN`` when the ring's EXACT area is at most ``16 n eps extent^2`` (the float value
    of that expression, ``extent`` the larger float x or y range of THIS ring).  One rule on every
    path (owner decision, N2 round 2): a float filter with a rigorous bound, else exact integers.

    Filter: ``2A = sum (x_i - x_0)(y_{i+1} - y_{i-1})``; each term carries at most three roundings,
    the sum at most ``n - 1``, so ``(n + 8) eps sum |terms|`` (twice that need) plus ``2**-1000``
    bounds the error; decisions keep a ``2**-40`` relative margin against the comparisons' own
    roundings.
    """
    n = len(ring)
    if n <= _SMALL:
        pts = ring.tolist() if isinstance(ring, np.ndarray) else ring
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        extent = max(max(xs) - min(xs), max(ys) - min(ys))
        x0 = xs[0]
        terms = [(xs[i] - x0) * (ys[(i + 1) % n] - ys[i - 1]) for i in range(n)]
        doubled, magnitude = sum(terms), sum(abs(t) for t in terms)
    else:
        x, y = ring[:, 0], ring[:, 1]
        extent = max(float(x.max() - x.min()), float(y.max() - y.min()))
        terms = (x - x[0]) * (np.roll(y, -1) - np.roll(y, 1))
        doubled, magnitude = float(np.sum(terms)), float(np.sum(np.abs(terms)))
        xs, ys = x.tolist(), y.tolist()
    limit = 2.0 * (16.0 * n * _EPS * extent * extent)          # doubled-area threshold (exact x2)
    error = (n + 8) * _EPS * magnitude + 2.0 ** -1000
    if abs(doubled) - error > limit * (1.0 + 2.0 ** -40):
        return
    if (abs(doubled) + error) * (1.0 + 2.0 ** -40) < limit:
        below = True
    else:
        ratios = [v.as_integer_ratio() for v in xs + ys]
        q = max(r[1] for r in ratios)
        ints = [num * (q // den) for num, den in ratios]
        ix, iy = ints[:n], ints[n:]
        exact = abs(sum(ix[i] * (iy[(i + 1) % n] - iy[i - 1]) for i in range(n)))
        below = Fraction(exact, q * q) <= Fraction(limit)
    if below:
        raise DisplayDomainRefused("a ring's area is below display resolution relative to its own extent")


# ----------------------------------------------------------------------------- covered area
# Owner decision, N2 round 4 (3): ``area`` is RN(A), A the EXACT area of the emitted triangles.  The
# triangles tile the trapezoids exactly (a dropped piece has zero area), so
# 2 A = sum over trapezoids of (xr - xl) ((ytl - ybl) + (ytr - ybr)), every value a stored float.
# Small trims: exact integers (a common power-of-two denominator), one correctly rounded division.
# Larger: a certified float evaluation of the same sum, else the integers.  Every path yields RN(A).
def _integers(values):
    """``values`` (floats) as integers over one common power-of-two denominator ``q``: ``v = i / q``."""
    ratios = [v.as_integer_ratio() for v in values]
    bits = max((d.bit_length() for _, d in ratios), default=1)
    return [n << (bits - d.bit_length()) for n, d in ratios], 1 << (bits - 1)


def _area_exact(xl, xr, ybl, ytl, ybr, ytr) -> float:
    """RN of half the exact sum over the trapezoids (lists of floats), from integers."""
    k = len(xl)
    ints, q = _integers((*xl, *xr, *ybl, *ytl, *ybr, *ytr))
    left, right, bl, tl, br, tr = (ints[i * k:(i + 1) * k] for i in range(6))
    twice = sum((r - l) * ((a - b) + (c - d)) for l, r, b, a, d, c in zip(left, right, bl, tl, br, tr))
    return twice / (2 * q * q) + 0.0                       # CPython int / int is correctly rounded


def _area_exact_lines(xs, yb, yt) -> float:
    """:func:`_area_exact` for one trapezoid per slab (lines ``xs``, bottom and top ordinates on them)."""
    m = len(xs)
    ints, q = _integers((*xs, *yb, *yt))
    x = ints[:m]
    g = [t - b for b, t in zip(ints[m:2 * m], ints[2 * m:])]
    twice = sum((x[j + 1] - x[j]) * (g[j] + g[j + 1]) for j in range(m - 1))
    return twice / (2 * q * q) + 0.0


def _area_filtered(xl, xr, ybl, ytl, ybr, ytr):
    """RN(A) from a certified float evaluation (arrays), or ``None`` when it cannot certify.

    Per trapezoid, ``w1 + w2 = xr - xl``, ``d1 + d2 = ytl - ybl``, ``d3 + d4 = ytr - ybr`` exactly
    (two_sum), so ``2 A_k = (w1 d1 + w1 d3) + m_k`` with the main products exact as ``p + e``
    (two_prod; operands zero or at least 2**-450) and the small rest
    ``m_k = w1 (d2 + d4) + w2 (d1 + d2 + d3 + d4)`` evaluated with error at most ``16 u`` times its
    magnitude sum plus 2**-1074 per product (underflow).  The products ``p`` are split exactly
    (``q = (sigma + p) - sigma``, ``r = p - q``; ``sigma`` a power of two at least ``(n + 2) max |p|``)
    so ``tau = sum q`` is exact in any order (multiples of ``u sigma`` below ``sigma``); the rest
    ``s = sum(r, e, m)`` has error ``B`` (``2 (n + 1) u sum |terms|`` plus the ``m`` bound).
    ``tau + s = c + err`` exactly (two_sum): ``c`` is RN(2 A) when ``|err| + B`` is below half the
    gap of ``c`` to its float neighbours; halving is exact for ``c >= 2**-1020``.
    """
    w1, w2 = _two_sum(xr, -xl)
    d1, d2 = _two_sum(ytl, -ybl)
    d3, d4 = _two_sum(ytr, -ybr)
    if not bool(np.all(_safe(w1, d1, d3))):
        return None
    p1, e1 = _two_prod(w1, d1)
    p2, e2 = _two_prod(w1, d3)
    low = d2 + d4
    minor = w1 * low + w2 * ((d1 + d3) + low)
    magnitude = np.abs(w1) * (np.abs(d2) + np.abs(d4)) + np.abs(w2) * (
        (np.abs(d1) + np.abs(d3)) + (np.abs(d2) + np.abs(d4)))
    p = np.concatenate((p1, p2))
    big = float(np.max(np.abs(p)))
    if not big >= 2.0 ** -800:
        return None
    sigma = math.ldexp(1.0, math.frexp(big)[1] + (len(p) + 2).bit_length())
    q = (sigma + p) - sigma
    tau = float(np.sum(q))                                  # exact
    rest = np.concatenate((p - q, e1, e2, minor))
    s = float(np.sum(rest))
    bound = ((2.0 * (len(rest) + 1) * _U) * float(np.sum(np.abs(rest))) + 16.0 * _U * float(np.sum(magnitude))
             + 16.0 * (len(xl) + 1) * _TINY) * (1.0 + 2.0 ** -40)
    c, err = _two_sum(tau, s)
    if not (math.isfinite(c) and c >= 2.0 ** -1020):
        return None
    half_gap = 0.5 * min(c - math.nextafter(c, -math.inf), math.nextafter(c, math.inf) - c)
    if not (abs(err) + bound) * (1.0 + 2.0 ** -40) < half_gap:
        return None
    return 0.5 * c


def _area(xl, xr, ybl, ytl, ybr, ytr) -> float:
    """RN of the exact covered area of trapezoids (float64 arrays), the same bits on every path."""
    if len(xl) > _EXACT_PAIRS:
        area = _area_filtered(xl, xr, ybl, ytl, ybr, ytr)
        if area is not None:
            return area
    return _area_exact(xl.tolist(), xr.tolist(), ybl.tolist(), ytl.tolist(), ybr.tolist(), ytr.tolist())


# ----------------------------------------------------------------------------- monotone fast paths
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
    when the correctly rounded chains are strictly apart at every interior slab line (rounding is
    monotone, so the exact chains are too: the ring is simple); otherwise ``None`` (the general
    path then decides exactly, ties included).
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

    ends, at_x, ids = [], [], []
    count = 0
    for chain in (up, down):
        cx = x[chain]
        at = np.searchsorted(cx, xs)                 # cx[at] >= xs
        exact = cx[np.minimum(at, len(cx) - 1)] == xs
        ident = np.empty(len(xs), dtype=np.int64)
        ident[exact] = chain[at[exact]]
        inner = np.flatnonzero(~exact)
        ends.append(np.column_stack((chain[at[inner] - 1], chain[at[inner]])))   # a has the lower x
        at_x.append(xs[inner])
        ident[inner] = n + count + np.arange(len(inner))
        count += len(inner)
        ids.append(ident)
    cut_ends = np.vstack(ends).astype(np.int64)
    cut_x = np.concatenate(at_x)
    a, b = cut_ends[:, 0], cut_ends[:, 1]
    fractions, cut_y = _cut_values(x[a], y[a], x[b], y[b], cut_x)
    points = np.vstack((ring, np.column_stack((cut_x, cut_y))))
    y_up, y_down = points[ids[0], 1], points[ids[1], 1]
    middle = len(xs) // 2
    probe = slice(max(middle - 1, 0), middle + 1)
    if np.sum(y_down[probe]) >= np.sum(y_up[probe]):
        bottom, top = ids[0], ids[1]
    else:
        bottom, top = ids[1], ids[0]
    gap = points[top, 1] - points[bottom, 1]
    if not np.all(gap[1:-1] > 0.0):
        return None
    for k in (0, -1):                                # ends: vertices (exact), shared or a vertical side
        if not (top[k] == bottom[k] or gap[k] > 0.0):
            return None
    _floor(ring)
    yb, yt = points[bottom, 1], points[top, 1]
    area = _area(xs[:-1], xs[1:], yb[:-1], yt[:-1], yb[1:], yt[1:])
    bl, br, tl, tr = bottom[:-1], bottom[1:], top[:-1], top[1:]
    triangles = np.vstack((np.column_stack((bl, br, tr))[br != tr], np.column_stack((bl, tr, tl))[bl != tl]))
    return SlabTriangulation(points, triangles.astype(np.int64), cut_ends, fractions, area)


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
    cut_ends, cut_x, ids = [], [], []
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
            ident.append(n + len(cut_x))
            cut_ends.append((chain[i], chain[i + 1]))        # canonical: the lower-x end first
            cut_x.append(x)
        ids.append(ident)
    if len(cut_x) <= _EXACT_ROWS:
        fractions, cut_y = _cut_values_exact([pts[a][0] for a, _ in cut_ends], [pts[a][1] for a, _ in cut_ends],
                                             [pts[b][0] for _, b in cut_ends], [pts[b][1] for _, b in cut_ends],
                                             cut_x)
    else:
        e = np.array(cut_ends, dtype=np.int64)
        fractions, cut_y = _cut_values(ring[e[:, 0], 0], ring[e[:, 0], 1], ring[e[:, 1], 0], ring[e[:, 1], 1],
                                       np.array(cut_x))
        fractions, cut_y = fractions.tolist(), cut_y.tolist()
    py = [p[1] for p in pts] + list(cut_y)
    y_up = [py[k] for k in ids[0]]
    y_down = [py[k] for k in ids[1]]
    m = len(xs) // 2
    if y_down[m - 1] + y_down[m] >= y_up[m - 1] + y_up[m]:
        bottom, top, yb, yt = ids[0], ids[1], y_up, y_down
    else:
        bottom, top, yb, yt = ids[1], ids[0], y_down, y_up
    last = len(xs) - 1
    for j in range(len(xs)):
        if not (yt[j] - yb[j] > 0.0 or ((j == 0 or j == last) and top[j] == bottom[j])):
            return None
    _floor(ring)
    if len(xs) - 1 <= _EXACT_PAIRS:
        area = _area_exact_lines(xs, yb, yt)
    else:                                            # many extra lines: the certified float route
        lx, lb, lt = np.array(xs), np.array(yb), np.array(yt)
        area = _area(lx[:-1], lx[1:], lb[:-1], lt[:-1], lb[1:], lt[1:])
    tris = []
    for j in range(last):
        bl, br, tl, tr = bottom[j], bottom[j + 1], top[j], top[j + 1]
        if br != tr:
            tris.append((bl, br, tr))
        if bl != tl:
            tris.append((bl, tr, tl))
    points = np.vstack((ring, np.column_stack((cut_x, cut_y)))) if cut_x else np.array(ring, dtype=float)
    return SlabTriangulation(points, np.array(tris, dtype=np.int64).reshape(-1, 3),
                             np.array(cut_ends, dtype=np.int64).reshape(-1, 2), np.array(fractions, dtype=float),
                             area)


# ----------------------------------------------------------------------------- preparation and budget
# The input contract first (owner decision, N2 round 4 (1)), in O(number of rings) and from types and
# shapes alone: no conversion, no iteration of a caller's container beyond the outer list or tuple,
# no attribute of anything but an exact ndarray, so no foreign object runs code and nothing large is
# allocated before a refusal.  Size caps follow from the same shapes (broadcast and strided views are
# capped without being materialised).  Only then are coordinates read (finiteness and the bound).
def _form_error(detail: str) -> SlabInvalid:
    return SlabInvalid(f"{_FORM} ({detail})")


def _check_input(rings, extra_x) -> tuple:
    """The rings as a tuple of exact float64 (n, 2) ndarrays with n >= 3, after every size cap."""
    if type(rings) is not list and type(rings) is not tuple:
        raise _form_error(f"rings is a {type(rings).__name__}")
    count = len(rings)
    if count == 0:
        raise _form_error("no rings")
    if count > MAX_SLAB_RINGS:
        raise _too_complex(f"{count} rings (MAX_SLAB_RINGS = {MAX_SLAB_RINGS})")
    rings = tuple(rings)                                    # a snapshot of the caller's list
    total = 0
    for k, ring in enumerate(rings):
        if type(ring) is not np.ndarray:
            raise _form_error(f"ring {k} is a {type(ring).__name__}")
        if not (ring.dtype == np.float64 and ring.ndim == 2 and ring.shape[1] == 2 and ring.shape[0] >= 3):
            raise _form_error(f"ring {k} has dtype {ring.dtype} and shape {ring.shape}; every ring needs at "
                              "least three points")
        total += ring.shape[0]
    if total > MAX_SLAB_RING_POINTS:
        raise _too_complex(f"{total} ring points (MAX_SLAB_RING_POINTS = {MAX_SLAB_RING_POINTS})")
    if extra_x is not None:
        if type(extra_x) is not np.ndarray:
            raise _form_error(f"extra_x is a {type(extra_x).__name__}, not None or a float64 1-D ndarray")
        if not (extra_x.dtype == np.float64 and extra_x.ndim == 1):
            raise _form_error(f"extra_x has dtype {extra_x.dtype} and shape {extra_x.shape}, not float64 1-D")
        if extra_x.size > MAX_SLAB_INCIDENCES:
            raise _too_complex(f"{extra_x.size} extra lines (MAX_SLAB_INCIDENCES = {MAX_SLAB_INCIDENCES})")
    return rings


def _in_domain(values: np.ndarray, what: str) -> None:
    """Finite with ``|c| <= MAX_DISPLAY_COORDINATE`` (float64 data, no conversion)."""
    if values.size and not (bool(np.all(np.isfinite(values)))
                            and float(np.abs(values).max()) <= MAX_DISPLAY_COORDINATE):
        raise DisplayDomainRefused(f"{what} must be finite with |coordinate| <= {MAX_DISPLAY_COORDINATE:g}")


def _prepare(rings, extra_x):
    """Input contract and size caps, domain, structure and the preflight budget; returns the rings'
    vertices and segments."""
    rings = _check_input(rings, extra_x)
    vertices = np.vstack(rings)                             # float64, at most MAX_SLAB_RING_POINTS rows
    _in_domain(vertices, "ring coordinates")
    extra = None
    if extra_x is not None:
        extra = np.array(extra_x, dtype=np.float64)         # a copy, at most MAX_SLAB_INCIDENCES values
        _in_domain(extra, "extra_x")
    total = len(vertices)

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
    # preflight: a segment spans fewer slabs than there are lines; count exactly only when that
    # cheap bound could exceed a budget (the fast path then never pays for it)
    bound = len(seg_a) * (total + (0 if extra is None else len(extra)))
    if bound > MAX_SLAB_INCIDENCES or bound + 2 * (total + bound) > MAX_SLAB_OUTPUT:
        _slabs(prep)
    return prep


def _slabs(prep):
    """Slab lines, canonical sloped segments and their slab ranges; refuses above the budgets."""
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
        raise _too_complex(f"{incidences} segment-slab incidences (MAX_SLAB_INCIDENCES = {MAX_SLAB_INCIDENCES})")
    output = incidences + 2 * (len(vertices) + incidences - len(sloped))
    if output > MAX_SLAB_OUTPUT:
        raise _too_complex(f"up to {output} triangles (MAX_SLAB_OUTPUT = {MAX_SLAB_OUTPUT})")
    prep.update(xs=xs, sp=sp, sq=sq, sring=prep["seg_ring"][sloped], jl=jl, jr=jr, incidences=incidences)
    return prep


# ----------------------------------------------------------------------------- entry point
def triangulate_rings(rings: Sequence[np.ndarray], extra_x: np.ndarray | None = None, *,
                      _fast: bool = True) -> SlabTriangulation:
    """Triangulate the region of ``rings[0]`` minus ``rings[1:]`` (each ring closed implicitly).

    ``extra_x`` adds slab lines (inside the region's x range) so no slab is wider than a caller's
    step.  Raises :class:`SlabInvalid` (``TRIM_INVALID``) when a ring is not simple, two rings
    touch, or a hole is outside the outer ring or inside another hole (decided exactly,
    independent of ``extra_x``), and ``DisplayDomainRefused`` (``OUT_OF_DOMAIN``) outside the
    numeric domain, above a preflight budget, or for a ring whose exact area is below the floor.
    Input contract (owner decision, N2 round 4): ``rings`` is a list or tuple of 1 to
    ``MAX_SLAB_RINGS`` exact ``numpy.ndarray`` float64 arrays of shape ``(n, 2)``, ``n >= 3``, at most
    ``MAX_SLAB_RING_POINTS`` points in all; ``extra_x`` None or an exact float64 1-D ndarray of at
    most ``MAX_SLAB_INCIDENCES`` values.  It is checked first, from types and shapes only; any other
    form (lists of points, other dtypes, subclasses such as masked arrays, stacked arrays, iterables)
    is ``TRIM_INVALID`` ("input must be a list/tuple of float64 (n, 2) ndarrays"); the caps are
    ``OUT_OF_DOMAIN`` ("too complex").
    Independent of the caller's NumPy error state (every numeric section runs under
    ``np.errstate(all="ignore")`` with explicit finiteness checks).  ``_fast=False`` (private, for
    tests) skips the x-monotone shortcut.
    """
    with np.errstate(all="ignore"):
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
    cut_fraction, cut_y = _cut_values(xp[cut_seg], yp[cut_seg], xq[cut_seg], yq[cut_seg], cut_x)
    points = np.vstack((vertices, np.column_stack((cut_x, cut_y))))
    py = points[:, 1]

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

    # 1. exact order of every boundary point on each slab line (vertices by canonical id, cut points).
    # Every ordinate is the exact one correctly rounded, so a strictly smaller float is a strictly
    # smaller exact value; only equal floats (ties) need the exact comparison.
    vertex_ids = np.unique(canonical)
    line_points = np.concatenate((vertex_ids, np.arange(n, len(points))))
    point_line = np.full(len(points), -1, dtype=np.int64)
    point_line[vertex_ids] = np.searchsorted(xs, vertices[vertex_ids, 0])
    point_line[n:] = cut_line
    by_line = line_points[np.lexsort((line_points, py[line_points], point_line[line_points]))]
    lines_of = point_line[by_line]
    same = np.flatnonzero(lines_of[1:] == lines_of[:-1])
    tied = same[py[by_line[same + 1]] <= py[by_line[same]]]
    reversed_order = False
    for line in np.unique(lines_of[tied]):
        lo, hi = np.searchsorted(lines_of, [line, line + 1])
        block = by_line[lo:hi].tolist()
        exact_block = sorted(block, key=lambda pid: (exact_y(pid), pid))
        for u, v in zip(exact_block, exact_block[1:]):
            if exact_y(u) == exact_y(v):
                raise SlabInvalid("two boundary points coincide (rings touch or cross at a point)")
        if any(py[u] > py[v] for u, v in zip(exact_block, exact_block[1:])):
            reversed_order = True                    # unreachable: correct rounding is monotone
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

    # fail-closed guard (the topology above is exact; correctly rounded ordinates cannot reverse it)
    if reversed_order:
        raise DisplayDomainRefused("rounded ordinates reverse their exact order on a slab line "
                                   "(unreachable: correct rounding is monotone)")
    for c in prep["compressed"]:
        _floor(vertices[c])

    # trapezoids: bottom/top pairs in each slab; sides use every point between them on their line
    bottom_l, bottom_r = left[0::2], right[0::2]
    top_l, top_r = left[1::2], right[1::2]
    pair_slab = inc_slab[0::2]
    area = _area(xs[pair_slab], xs[pair_slab + 1], py[bottom_l], py[top_l], py[bottom_r], py[top_r])
    if not np.isfinite(area):
        raise DisplayDomainRefused("the covered area is not finite (unreachable inside the domain)")
    left_lo, left_hi = position[bottom_l], position[top_l]
    right_lo, right_hi = position[bottom_r], position[top_r]
    plain = (left_hi - left_lo <= 1) & (right_hi - right_lo <= 1)

    # Each triangle has one vertical side between two points of one line and its third corner on
    # the other line: its exact area is (side height) x (slab width) / 2, so no determinant is
    # evaluated.  The side's height is the difference of two floats, exactly signed: a positive one
    # gives a strictly positive triangle; a zero one (a shared point, or a TIE of two distinct
    # points rounded to the same ordinate) gives an exactly degenerate piece, which is dropped.
    triangles = []
    p = np.flatnonzero(plain)
    bl, br, tl, tr = bottom_l[p], bottom_r[p], top_l[p], top_r[p]
    triangles.append(np.column_stack((bl, br, tr))[py[tr] > py[br]])
    triangles.append(np.column_stack((bl, tr, tl))[py[tl] > py[bl]])
    for k in np.flatnonzero(~plain):
        lchain = by_line[left_lo[k]:left_hi[k] + 1]
        rchain = by_line[right_lo[k]:right_hi[k] + 1]
        i = j = 0
        made = []
        while i < len(lchain) - 1 or j < len(rchain) - 1:
            advance_left = j == len(rchain) - 1 or (
                i < len(lchain) - 1 and py[lchain[i + 1]] - py[bottom_l[k]] <= py[rchain[j + 1]] - py[bottom_r[k]])
            if advance_left:
                if py[lchain[i + 1]] > py[lchain[i]]:
                    made.append((lchain[i], rchain[j], lchain[i + 1]))
                i += 1
            else:
                if py[rchain[j + 1]] > py[rchain[j]]:
                    made.append((lchain[i], rchain[j], rchain[j + 1]))
                j += 1
        triangles.append(np.asarray(made, dtype=np.int64).reshape(-1, 3))
    triangles = np.vstack(triangles).astype(np.int64)
    cut_ends = np.column_stack((sp[cut_seg], sq[cut_seg])).astype(np.int64).reshape(-1, 2)
    return SlabTriangulation(points, triangles, cut_ends, cut_fraction, area)
