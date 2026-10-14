"""Audit certificates for curves on NURBS supports (generality slice A3b, B-S4).

``certify_only_shared_ends(first, second)`` proves, in exact arithmetic, that
two edge curves have no common point other than their shared end points, or
returns ``None`` (the audit then reports the pair as unclassified).

Both curves must expose exact control hulls of their sub-intervals without
widening: curves on NURBS supports whose pcurve is an iso-line (the exact
iso-curve, slice B-S1). Pairs of pieces are processed by branch and bound:

* pieces whose exact hulls are strictly separated (disjoint boxes, or a plane
  with every hull point of one strictly on one side and of the other strictly
  on the other) share no point;
* pieces that both end at a shared end point ``V`` are separated by a plane
  through ``V`` with normal ``n = t_a/|t_a| - t_b/|t_b|`` (the curves' unit
  tangents leaving ``V``): every hull point ``P`` of the first piece has
  ``n.(P - V) > 0`` and of the second ``< 0``, except ``V`` itself (exact
  insertion puts ``V`` among both hulls). Then ``V`` is their only common point;
* otherwise the pieces are halved, within a budget.

Exact hulls contain the exact curves (positive weights, convex hull), so the
certificate is about the exact edge curves.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from .nurbs import CurveOnSurface, NurbsCurve

_BUDGET = 20000


def _exact_hull(curve, lower: float, upper: float):
    """Exact hull of a world edge curve's piece: iso-line curves on NURBS supports; 2D pcurves in the chart."""
    if isinstance(curve, CurveOnSurface) and curve._iso_world() is not None:
        return curve._iso_hull(lower, upper, exact=True)
    if isinstance(curve, NurbsCurve) and curve.dimension == 2:
        return curve.exact_hull(lower, upper)
    return None


def _strictly_separated(first, second) -> bool:
    dimension = len(first[0])
    for k in range(dimension):
        if max(p[k] for p in first) < min(q[k] for q in second) or max(q[k] for q in second) < min(p[k] for p in first):
            return True
    centre_first = [sum(p[k] for p in first) / len(first) for k in range(dimension)]
    centre_second = [sum(q[k] for q in second) / len(second) for k in range(dimension)]
    normal = [b - a for a, b in zip(centre_first, centre_second)]
    if not any(normal):
        return False
    return max(sum(n * p[k] for k, n in enumerate(normal)) for p in first) < \
        min(sum(n * q[k] for k, n in enumerate(normal)) for q in second)


def _leaving_tangent(curve, end: int) -> np.ndarray:
    derivative = np.asarray(curve.derivative(float(end)), dtype=float)
    return derivative if end == 0 else -derivative


def _separated_at(first, second, vertex, normal) -> bool:
    def side(point):
        return sum(n * (p - v) for n, p, v in zip(normal, point, vertex))

    for point in first:
        value = side(point)
        if value < 0 or (value == 0 and tuple(point) != tuple(vertex)):
            return False
    for point in second:
        value = side(point)
        if value > 0 or (value == 0 and tuple(point) != tuple(vertex)):
            return False
    return True


def certify_only_shared_ends(first, second, budget: int = _BUDGET) -> bool | None:
    """``True`` when the exact curves meet at most at shared end points; ``None`` when not certified."""
    if _exact_hull(first, 0.0, 1.0) is None or _exact_hull(second, 0.0, 1.0) is None:
        return None
    ends_first = {0: first.exact_point(0), 1: first.exact_point(1)}
    ends_second = {0: second.exact_point(0), 1: second.exact_point(1)}
    shared = [(a, b) for a in (0, 1) for b in (0, 1) if ends_first[a] == ends_second[b]]
    normals = {}
    for a, b in shared:
        t_a, t_b = _leaving_tangent(first, a), _leaving_tangent(second, b)
        length_a, length_b = float(np.linalg.norm(t_a)), float(np.linalg.norm(t_b))
        if length_a == 0.0 or length_b == 0.0:
            return None
        normal = t_a / length_a - t_b / length_b
        if not np.any(normal):
            return None                                    # tangent at the shared end: not certified here
        normals[a, b] = [Fraction(float(value)) for value in normal]
    queue = [((0.0, 1.0), (0.0, 1.0))]
    work = 0
    while queue:
        work += 1
        if work > budget:
            return None
        (a0, a1), (b0, b1) = queue.pop()
        hull_first, hull_second = _exact_hull(first, a0, a1), _exact_hull(second, b0, b1)
        if _strictly_separated(hull_first, hull_second):
            continue
        at_vertex = [(a, b) for a, b in shared
                     if ((a == 0 and a0 == 0.0) or (a == 1 and a1 == 1.0))
                     and ((b == 0 and b0 == 0.0) or (b == 1 and b1 == 1.0))]
        if any(_separated_at(hull_first, hull_second, ends_first[a], normals[a, b]) for a, b in at_vertex):
            continue
        if a1 - a0 <= 1e-9 and b1 - b0 <= 1e-9:
            return None
        am, bm = 0.5 * (a0 + a1), 0.5 * (b0 + b1)
        for first_piece in ((a0, am), (am, a1)):
            for second_piece in ((b0, bm), (bm, b1)):
                queue.append((first_piece, second_piece))
    return True


def shared_ends_consistent(first_edge, second_edge) -> bool:
    """Whether exact end coincidences of two edge curves are exactly their shared topology vertices.

    Every exact end coincidence must join the same vertex ID, and every shared
    vertex ID must be an exact end coincidence; only then may
    ``certify_only_shared_ends`` stand for the topology (review A3b B-S4).
    """
    first_ids, second_ids = (first_edge.start, first_edge.end), (second_edge.start, second_edge.end)
    matches = [(a, b) for a in (0, 1) for b in (0, 1)
               if first_edge.curve.exact_point(a) == second_edge.curve.exact_point(b)]
    return (all(first_ids[a] == second_ids[b] for a, b in matches)
            and set(first_ids) & set(second_ids) == {first_ids[a] for a, _b in matches})

