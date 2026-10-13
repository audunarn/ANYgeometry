"""References for B-spline / NURBS curve kernels (same signatures as ``anygeometry._core``).

Algorithms follow Piegl and Tiller, *The NURBS Book* (2nd ed.): span search
(A2.1), basis functions and their derivatives (A2.2/A2.3), curve derivatives
(A3.2), rational curve derivatives (A4.2), knot insertion (A5.1) and
decomposition into Bezier segments (A5.6).

Inputs are assumed valid (the curve classes validate): a non-decreasing
clamped knot vector of length ``n + p + 2`` for ``n + 1`` control points of
degree ``p``, positive weights, parameters inside ``[knots[p], knots[n+1]]``.
"""
from __future__ import annotations

import math

import numpy as np


def find_spans(degree: int, knots: np.ndarray, count: int, t: np.ndarray) -> np.ndarray:
    """Index ``i`` with ``knots[i] <= t < knots[i+1]`` per parameter (A2.1).

    ``count`` is the number of control points. The last parameter value of the
    domain belongs to the last non-empty span.
    """
    n = count - 1
    spans = np.searchsorted(knots, t, side="right") - 1
    spans = np.clip(spans, degree, n)
    # t == knots[n+1] (the end of the domain): use the last non-empty span.
    at_end = t >= knots[n + 1]
    if np.any(at_end):
        last = n
        while last > degree and knots[last] == knots[n + 1]:
            last -= 1
        spans = np.where(at_end, last, spans)
    return spans


def basis_derivatives(degree: int, knots: np.ndarray, spans: np.ndarray, t: np.ndarray, order: int) -> np.ndarray:
    """Non-zero basis functions and derivatives, shape ``(m, order+1, degree+1)`` (A2.3)."""
    p, m = degree, len(t)
    ndu = np.zeros((m, p + 1, p + 1))
    ndu[:, 0, 0] = 1.0
    left = np.zeros((m, p + 1))
    right = np.zeros((m, p + 1))
    for j in range(1, p + 1):
        left[:, j] = t - knots[spans + 1 - j]
        right[:, j] = knots[spans + j] - t
        saved = np.zeros(m)
        for r in range(j):
            ndu[:, j, r] = right[:, r + 1] + left[:, j - r]          # lower triangle: knot differences
            temp = ndu[:, r, j - 1] / ndu[:, j, r]
            ndu[:, r, j] = saved + right[:, r + 1] * temp           # upper triangle: basis functions
            saved = left[:, j - r] * temp
        ndu[:, j, j] = saved
    ders = np.zeros((m, order + 1, p + 1))
    ders[:, 0, :] = ndu[:, :, p]
    top = min(order, p)              # derivatives above the degree vanish
    for r in range(p + 1):
        a = np.zeros((m, 2, p + 1))
        a[:, 0, 0] = 1.0
        s1, s2 = 0, 1
        for k in range(1, top + 1):
            d = np.zeros(m)
            rk, pk = r - k, p - k
            if r >= k:
                a[:, s2, 0] = a[:, s1, 0] / ndu[:, pk + 1, rk]
                d = a[:, s2, 0] * ndu[:, rk, pk]
            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if r - 1 <= pk else p - r
            for j in range(j1, j2 + 1):
                a[:, s2, j] = (a[:, s1, j] - a[:, s1, j - 1]) / ndu[:, pk + 1, rk + j]
                d = d + a[:, s2, j] * ndu[:, rk + j, pk]
            if r <= pk:
                a[:, s2, k] = -a[:, s1, k - 1] / ndu[:, pk + 1, r]
                d = d + a[:, s2, k] * ndu[:, r, pk]
            ders[:, k, r] = d
            s1, s2 = s2, s1
    factor = float(p)
    for k in range(1, top + 1):
        ders[:, k, :] *= factor
        factor *= p - k
    return ders


def bspline_evaluate(degree: int, knots: np.ndarray, controls: np.ndarray, weights: np.ndarray,
                     t: np.ndarray, order: int, out: np.ndarray) -> None:
    """Write curve points and derivatives up to ``order`` (0..2) into ``out``.

    ``controls`` is ``(n+1, dim)``; ``weights`` is ``(n+1,)`` for a rational
    curve or empty for a polynomial one; ``out`` is ``(m, order+1, dim)``.
    """
    p = int(degree)
    count = len(controls)
    spans = find_spans(p, knots, count, t)
    ders = basis_derivatives(p, knots, spans, t, order)
    index = spans[:, None] - p + np.arange(p + 1)[None, :]                   # (m, p+1)
    rational = weights.size > 0
    if rational:
        homogeneous = np.concatenate((controls * weights[:, None], weights[:, None]), axis=1)
    else:
        homogeneous = controls
    local = homogeneous[index]                                               # (m, p+1, dim(+1))
    values = np.einsum("mkj,mjd->mkd", ders, local)                         # A3.2
    if not rational:
        out[...] = values
        return
    a, w = values[:, :, :-1], values[:, :, -1]
    result = np.empty_like(a)
    for k in range(order + 1):                                               # A4.2
        v = a[:, k, :].copy()
        for i in range(1, k + 1):
            v -= math.comb(k, i) * w[:, i, None] * result[:, k - i, :]
        result[:, k, :] = v / w[:, 0, None]
    out[...] = result


def insert_knot(degree: int, knots: np.ndarray, controls: np.ndarray, weights: np.ndarray | None,
                value: float, times: int = 1):
    """Insert ``value`` ``times`` times (A5.1); the curve is unchanged.

    Returns ``(knots, controls, weights)`` of the refined curve (weights None
    for a polynomial curve). Works in homogeneous coordinates.
    """
    p = int(degree)
    knots = np.asarray(knots, dtype=float)
    rational = weights is not None
    points = np.asarray(controls, dtype=float)
    if rational:
        w = np.asarray(weights, dtype=float)
        points = np.concatenate((points * w[:, None], w[:, None]), axis=1)
    for _ in range(times):
        n = len(points) - 1
        k = int(find_spans(p, knots, n + 1, np.array([value]))[0])
        new = np.empty((n + 2, points.shape[1]))
        new[: k - p + 1] = points[: k - p + 1]
        new[k + 1:] = points[k:]
        for i in range(k - p + 1, k + 1):
            alpha = (value - knots[i]) / (knots[i + p] - knots[i])
            new[i] = alpha * points[i] + (1.0 - alpha) * points[i - 1]
        knots = np.concatenate((knots[: k + 1], [value], knots[k + 1:]))
        points = new
    if rational:
        return knots, points[:, :-1] / points[:, -1:], points[:, -1].copy()
    return knots, points, None


def bezier_segments(degree: int, knots: np.ndarray, controls: np.ndarray, weights: np.ndarray | None):
    """Decompose into Bezier segments by knot insertion (equivalent to A5.6).

    Returns a list of ``(t0, t1, controls, weights)`` for every non-empty span.
    """
    p = int(degree)
    knots = np.asarray(knots, dtype=float)
    distinct = sorted(set(knots[p + 1: len(knots) - p - 1].tolist()))
    k, c, w = knots, np.asarray(controls, dtype=float), None if weights is None else np.asarray(weights, dtype=float)
    for value in distinct:
        multiplicity = int(np.count_nonzero(k == value))
        if multiplicity < p:
            k, c, w = insert_knot(p, k, c, w, value, p - multiplicity)
    breaks = [k[p], *distinct, k[-p - 1]]
    segments = []
    for index, (t0, t1) in enumerate(zip(breaks, breaks[1:])):
        rows = slice(index * p, index * p + p + 1)
        segments.append((t0, t1, c[rows].copy(), None if w is None else w[rows].copy()))
    return segments
