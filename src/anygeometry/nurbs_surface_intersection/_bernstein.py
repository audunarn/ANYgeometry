"""Exact homogeneous Bezier pieces and Bernstein range enclosures (Fractions).

Grids are ``g[i][j]`` of equal-length tuples of Fractions (here the homogeneous
vector ``(wX, wY, wZ, w)``).  Everything is factored per surface: a four
parameter box is two independent 2D boxes; no 4D tensor is ever built.
"""
from __future__ import annotations

from fractions import Fraction

from ..errors import GeometryError
from ..nurbs import _exact_refine          # read-only private primitive (exact knot insertion), isolated here
from ._intervals import Interval

__all__ = ["bezier_piece", "subbox", "derivative_grid", "grid_ranges", "ZERO_TOLERANT_DEGENERATE"]

ZERO_TOLERANT_DEGENERATE = True   # a degenerate sub-box (a == b) yields the exact point value


def bezier_piece(surface, net, u_box, v_box):
    """Homogeneous Bernstein grid of the surface over a knot box inside one span.

    ``u_box``/``v_box`` are Fraction pairs in knot coordinates.  Raises
    ``GeometryError`` when the box straddles an interior knot or leaves the
    knot domain (the caller must split at knots first).
    """
    p, q = surface.degree_u, surface.degree_v
    count_v = len(net[0])
    s0, s1 = u_box
    t0, t1 = v_box
    if not (s0 < s1 and t0 < t1):
        raise GeometryError("Bezier piece needs a box of positive size")
    rows = [sum(row, ()) for row in net]
    knots_u, rows, active_u = _exact_refine(p, [Fraction(k) for k in surface.knots_u], rows, s0, s1)
    columns = [sum((rows[i][4 * j: 4 * j + 4] for i in active_u), ()) for j in range(count_v)]
    knots_v, columns, active_v = _exact_refine(q, [Fraction(k) for k in surface.knots_v], columns, t0, t1)
    if len(active_u) != p + 1 or len(active_v) != q + 1:
        raise GeometryError("box straddles an interior knot; split at knots first")
    lo_u, hi_u = Fraction(surface.knots_u[p]), Fraction(surface.knots_u[len(net)])
    lo_v, hi_v = Fraction(surface.knots_v[q]), Fraction(surface.knots_v[count_v])
    if s0 < lo_u or s1 > hi_u or t0 < lo_v or t1 > hi_v:
        raise GeometryError("box leaves the knot domain")
    return tuple(tuple(tuple(columns[j][4 * a: 4 * a + 4]) for j in active_v) for a in range(p + 1))


def _lerp(x, y, tau):
    return tuple((1 - tau) * a + tau * b for a, b in zip(x, y))


def _blossom_axis_reference(vectors, a, b):
    """Coefficients of the same polynomial on ``[a, b]`` (parameters may leave [0, 1]) by blossoming."""
    n = len(vectors) - 1
    out = []
    for k in range(n + 1):
        work = list(vectors)
        for tau in [a] * (n - k) + [b] * k:
            work = [_lerp(work[i], work[i + 1], tau) for i in range(len(work) - 1)]
        out.append(work[0])
    return out


def _split_axis(vectors,tau):
    work=list(vectors)
    left,right=[work[0]],[work[-1]]
    while len(work)>1:
        work=[_lerp(work[i],work[i+1],tau) for i in range(len(work)-1)]
        left.append(work[0])
        right.append(work[-1])
    return left,list(reversed(right))


def _blossom_axis(vectors,a,b):
    """Exact restriction via two splits; preserve reference for extrapolation."""
    n=len(vectors)-1
    if n==0:
        return list(vectors)
    if a==b:
        work=list(vectors)
        while len(work)>1:
            work=[_lerp(work[i],work[i+1],a) for i in range(len(work)-1)]
        return [work[0]]*(n+1)
    if not 0<=a<b<=1:
        return _blossom_axis_reference(vectors,a,b)
    right=list(vectors) if a==0 else _split_axis(vectors,a)[1]
    if b==1:
        return right
    return _split_axis(right,(b-a)/(1-a))[0]


def subbox(grid, u0, u1, v0, v1):
    """Bernstein grid of the same piece on the sub-box ``[u0, u1] x [v0, v1]`` of its [0, 1]^2."""
    u0, u1, v0, v1 = (Fraction(x) for x in (u0, u1, v0, v1))
    columns = [_blossom_axis([grid[i][j] for i in range(len(grid))], u0, u1) for j in range(len(grid[0]))]
    rows = [[columns[j][i] for j in range(len(columns))] for i in range(len(columns[0]))]
    return tuple(tuple(_blossom_axis(row, v0, v1)) for row in rows)


def derivative_grid(grid, axis: int):
    """Bernstein grid of the partial derivative (w.r.t. the piece's own [0, 1] parameter)."""
    n_u, n_v = len(grid) - 1, len(grid[0]) - 1
    size = len(grid[0][0])
    zero = (Fraction(0),) * size
    if axis == 0:
        if n_u == 0:
            return ((zero,) * (n_v + 1),)
        return tuple(tuple(tuple(n_u * (c - d) for c, d in zip(grid[i + 1][j], grid[i][j]))
                           for j in range(n_v + 1)) for i in range(n_u))
    if n_v == 0:
        return tuple((zero,) for _ in range(n_u + 1))
    return tuple(tuple(tuple(n_v * (c - d) for c, d in zip(grid[i][j + 1], grid[i][j]))
                       for j in range(n_v)) for i in range(n_u + 1))


def grid_ranges(grid):
    """Per-component ``Interval`` of the Bernstein coefficients (convex hull property)."""
    size = len(grid[0][0])
    return tuple(Interval(min(c[k] for row in grid for c in row), max(c[k] for row in grid for c in row))
                 for k in range(size))
