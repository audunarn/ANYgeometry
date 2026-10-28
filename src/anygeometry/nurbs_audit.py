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

_BUDGET = 4000


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


def _certify_pieces(first, first_range, second, second_range, contacts, budget: int, *, _hull=None,
                    _qualification=False, _depth=0) -> bool | None:
    """Exact certificate: the two curve pieces meet at most at the listed contact points.

    ``first_range``/``second_range`` are parameter intervals of the curves
    (dyadic, exact); ``contacts`` lists parameter pairs ``(ta, tb)`` at ends of
    those ranges where the exact points coincide. Any other common point makes
    the pieces inseparable, so the result is ``None`` (review A3b round 3: only
    explicitly allowed contacts; no rounded sub-curves).
    """
    vertices, normals = [], []
    for ta, tb in contacts:
        vertex = first.exact_point(ta)
        if vertex != second.exact_point(tb):
            return None
        t_a = np.asarray(first.derivative(float(ta)), dtype=float) * (1 if ta == first_range[0] else -1)
        t_b = np.asarray(second.derivative(float(tb)), dtype=float) * (1 if tb == second_range[0] else -1)
        length_a, length_b = float(np.linalg.norm(t_a)), float(np.linalg.norm(t_b))
        if length_a == 0.0 or length_b == 0.0:
            return None
        normal = t_a / length_a - t_b / length_b
        if not np.any(normal):
            return None                                    # tangent at the contact: not certified here
        vertices.append(vertex)
        normals.append([Fraction(float(value)) for value in normal])
    queue = [(tuple(first_range), tuple(second_range), _depth)]
    work = 0
    while queue:
        work += 1
        if work > budget:
            return None
        (a0, a1), (b0, b1), depth = queue.pop()
        hull = _exact_hull if _hull is None else _hull
        if _qualification:
            from .plane_nurbs import charge_box, charge_split, depth_limit
            charge_box()
        hull_first = hull(first, a0, a1)
        if _qualification:
            charge_box()
        hull_second = hull(second, b0, b1)
        if hull_first is None or hull_second is None:
            return None
        if _strictly_separated(hull_first, hull_second):
            continue
        if any(ta in (a0, a1) and tb in (b0, b1) and _separated_at(hull_first, hull_second, vertex, normal)
               for (ta, tb), vertex, normal in zip(contacts, vertices, normals)):
            continue
        if a1 - a0 <= 2. ** -16 and b1 - b0 <= 2. ** -16:
            return None                                    # inseparable at this scale: a crossing or a near-touch
        if _qualification:
            if depth >= depth_limit(16):
                return None
            charge_split()
            charge_split()
        am, bm = 0.5 * (a0 + a1), 0.5 * (b0 + b1)
        for first_piece in ((a0, am), (am, a1)):
            for second_piece in ((b0, bm), (bm, b1)):
                queue.append((first_piece, second_piece, depth+1))
    return True


def certify_only_shared_ends(first, second, budget: int = _BUDGET) -> bool | None:
    """``True`` when the exact curves meet at most at their coinciding end points; ``None`` when not certified."""
    contacts = [(float(a), float(b)) for a in (0, 1) for b in (0, 1) if first.exact_point(a) == second.exact_point(b)]
    return _certify_pieces(first, (0.0, 1.0), second, (0.0, 1.0), contacts, budget)


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


def _monotone(hull) -> bool:
    """Whether consecutive hull points advance strictly along one coordinate axis or the chord.

    With positive weights the curve is then monotone in that direction
    (variation diminishing), hence simple.
    """
    dimension = len(hull[0])
    directions = [tuple(1 if k == axis else 0 for k in range(dimension)) for axis in range(dimension)]
    directions.append(tuple(b - a for a, b in zip(hull[0], hull[-1])))
    for direction in directions:
        steps = [sum(d * (q[k] - p[k]) for k, d in enumerate(direction)) for p, q in zip(hull, hull[1:])]
        if all(step > 0 for step in steps) or all(step < 0 for step in steps):
            return True
    return False


def certify_simple(curve, lower: float = 0.0, upper: float = 1.0, depth: int = 0) -> bool | None:
    """``True`` when the exact curve piece ``[lower, upper]`` has no self-intersection (``None``: not certified).

    A piece whose ends coincide is refused (an edge piece returning to its
    start); a piece with a monotone exact hull is simple; otherwise its halves
    must be simple and meet only at the split point, certified on exact
    parameter intervals of the original curve (no rounded sub-curves).
    """
    if curve.exact_point(lower) == curve.exact_point(upper):
        return None
    hull = _exact_hull(curve, lower, upper)
    if hull is None:
        return None
    if _monotone(hull):
        return True
    if depth >= 24:
        return None
    middle = 0.5 * (lower + upper)
    if certify_simple(curve, lower, middle, depth + 1) is not True or \
            certify_simple(curve, middle, upper, depth + 1) is not True:
        return None
    return _certify_pieces(curve, (lower, middle), curve, (middle, upper), [(middle, middle)], _BUDGET)
