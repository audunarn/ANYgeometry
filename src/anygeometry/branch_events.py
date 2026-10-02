"""Junctions of a Bezier quadric branch curve with the other exact arrangement curves.

A :class:`~anygeometry.branch_curves.BezierQuadricCurve` lies on a quadric and on a Bezier-ruled wall that is not a
quadric, so the elimination of :mod:`anygeometry.quadric_events` ("the other curve's roots on both supports") has only
its quadric half. That half is enough whenever the other curve is not contained in the quadric: every shared point is
a root of the other curve on it, and the point inversion on the branch qualifies the candidate. Lines and ellipses
are reached through their planes (see :mod:`anygeometry.arrangement_geometry`) and the branches of quadric
intersections through their own supports. What is left is a curve that lies on the same quadric - a second wall's
trace on the same pipe, or a Bezier path drawn on it - and there the shared points are those of the two walls, which
has no closed form of low degree. For parallel walls rational polynomial elimination isolates their common
generators, including tangent roots; both branch charts then qualify every candidate. Other configurations use
certified box subdivision with outward-rounded enclosures: separated boxes prove curves disjoint,
crossings are bracketed down to the tolerance, and a pair that cannot be resolved within a fixed
budget (a tangential or overlapping contact) is refused instead of guessed.
"""
from __future__ import annotations

import numpy as np

from .branch_curves import BezierQuadricCurve
from .errors import GeometryError

_EPS = float(np.finfo(float).eps)
_BUDGET = 60000                      # box pairs visited before a pair of curves is called unresolvable


def world_quadric(curve):
    """The quadric that a Bezier quadric branch lies on, in world coordinates."""
    support = curve.second
    if not curve._identity:
        support = support.pulled_back(np.linalg.inv(np.asarray(curve.transform)))
    return support


def same_family(first, second):
    """Both curves are charts of the same algebraic curve (the same wall, quadric and affine image)."""
    return first.first == second.first and first.second == second.second and first.transform == second.transform


def _unit(value):
    return min(1.0, max(0.0, float(value)))


def _refine(first, second, t, u, tolerance):
    """Polish a candidate pair by Gauss-Newton on the vector difference; keep the better of start and result."""
    best = (t, u, float(np.linalg.norm(first.evaluate(t) - second.evaluate(u))))
    for _ in range(8):
        difference = first.evaluate(best[0]) - second.evaluate(best[1])
        jacobian = np.column_stack((first.derivative(best[0]), -second.derivative(best[1])))
        if not np.all(np.isfinite(jacobian)):
            break
        step = np.linalg.lstsq(jacobian, -difference, rcond=None)[0]
        if not np.all(np.isfinite(step)):
            break
        moved_t, moved_u = _unit(best[0] + step[0]), _unit(best[1] + step[1])
        residual = float(np.linalg.norm(first.evaluate(moved_t) - second.evaluate(moved_u)))
        if residual >= best[2]:
            break
        best = (moved_t, moved_u, residual)
        if float(np.max(np.abs(step))) <= 4 * _EPS:
            break
    return best


def subdivision_junctions(first, second, *, tolerance=1e-10, cancellation_check=None, budget=_BUDGET):
    """Every ``(first parameter, second parameter)`` at which two curves come within ``tolerance``.

    Both curves are cut at the midpoint of their longer box until the boxes are smaller than the tolerance; a pair
    of boxes separated by more than the tolerance cannot contain a shared point, so nothing is missed. Survivors are
    clustered and polished by Newton iteration.
    """
    stop = max(tolerance / 8, 1e-12)
    stack = [(0.0, 1.0, 0.0, 1.0)]
    survivors = []
    visits = 0
    while stack:
        a0, a1, b0, b1 = stack.pop()
        visits += 1
        if visits > budget:
            raise GeometryError("two exact curves stay within tolerance of each other over a stretch (a tangential or "
                                "overlapping contact), which the box subdivision cannot resolve")
        if cancellation_check is not None and cancellation_check():
            raise GeometryError("curve intersection predicate cancelled")
        lo_a, hi_a = first.bounds(a0, a1)
        lo_b, hi_b = second.bounds(b0, b1)
        if np.any(lo_a > hi_b + tolerance) or np.any(lo_b > hi_a + tolerance):
            continue
        size_a, size_b = float(np.max(hi_a - lo_a)), float(np.max(hi_b - lo_b))
        if max(size_a, size_b) <= stop or (b1 - b0 <= 4 * _EPS and a1 - a0 <= 4 * _EPS):
            survivors.append((.5 * (a0 + a1), .5 * (b0 + b1)))
            continue
        if size_a >= size_b and a1 - a0 > 4 * _EPS:
            mid = .5 * (a0 + a1)
            stack.append((a0, mid, b0, b1))
            stack.append((mid, a1, b0, b1))
        elif b1 - b0 > 4 * _EPS:
            mid = .5 * (b0 + b1)
            stack.append((a0, a1, b0, mid))
            stack.append((a0, a1, mid, b1))
        else:
            mid = .5 * (a0 + a1)
            stack.append((a0, mid, b0, b1))
            stack.append((mid, a1, b0, b1))
    clusters = []
    for t, u in sorted(survivors):
        for cluster in clusters:
            if abs(cluster[-1][0] - t) <= 1e-7 and abs(cluster[-1][1] - u) <= 1e-7:
                cluster.append((t, u))
                break
        else:
            clusters.append([(t, u)])
    result = []
    for cluster in clusters:
        t = sum(item[0] for item in cluster) / len(cluster)
        u = sum(item[1] for item in cluster) / len(cluster)
        t, u, residual = _refine(first, second, t, u, tolerance)
        if residual <= tolerance:
            result.append((t, u))
    return tuple(result)


def bezier_quadric_junctions(first, second, *, tolerance=1e-10, cancellation_check=None):
    """Every isolated shared point (or overlap endpoint) of two curves, at least one a Bezier quadric branch.

    ``first`` and ``second`` may be a :class:`BezierQuadricCurve`, a ``BezierPath`` or a
    ``CylinderIntersectionCurve``; the pairs with lines, ellipses and quadric branches go through their own paths.
    """
    from .arrangement_geometry import BezierPath, point_parameters
    from .exact_curves import CylinderIntersectionCurve
    from .quadric_events import branch_supports, curve_quadric_roots
    swapped = not isinstance(first, BezierQuadricCurve)
    curve, other = (second, first) if swapped else (first, second)           # the branch is ``curve``
    pairs = []

    def add(u, r):
        pair = (r, u) if swapped else (u, r)
        if not any(max(abs(a - pair[0]), abs(b - pair[1])) <= 64 * _EPS for a, b in pairs):
            pairs.append(pair)

    def invert_curve_roots(parameters):
        """Roots of the branch on a support of ``other``: qualify each by the point inversion on ``other``."""
        for u in parameters:
            for r in point_parameters(other, curve.evaluate(u), tolerance=tolerance):
                add(u, r)

    if isinstance(other, BezierPath):
        roots = curve_quadric_roots(other, world_quadric(curve), tolerance=tolerance,
                                    cancellation_check=cancellation_check)
        if roots is None:                                  # the path is drawn on the branch's quadric
            for u, r in subdivision_junctions(curve, other, tolerance=tolerance, cancellation_check=cancellation_check):
                add(u, r)
        else:
            for r in roots:
                for u in point_parameters(curve, other.evaluate(r), tolerance=tolerance):
                    add(u, r)
    elif isinstance(other, CylinderIntersectionCurve):
        constraints = [curve.roots_on(support, tolerance=tolerance, cancellation_check=cancellation_check)
                       for support in branch_supports(other)]
        if all(roots is None for roots in constraints):    # the branch lies on both cylinders: a common stretch
            invert_curve_roots((0., 1., *point_parameters(curve, other.evaluate(0.), tolerance=tolerance),
                                *point_parameters(curve, other.evaluate(1.), tolerance=tolerance)))
        else:
            invert_curve_roots(tuple(u for roots in constraints if roots is not None for u in roots))
    elif isinstance(other, BezierQuadricCurve):
        if same_family(curve, other):
            return ()                                      # charts of one curve meet only at their ends (named by the caller)
        roots = curve.roots_on(world_quadric(other), tolerance=tolerance, cancellation_check=cancellation_check)
        if roots is not None:
            invert_curve_roots(roots)
        else:
            reverse = other.roots_on(world_quadric(curve), tolerance=tolerance, cancellation_check=cancellation_check)
            if reverse is not None:
                for r in reverse:
                    for u in point_parameters(curve, other.evaluate(r), tolerance=tolerance):
                        add(u, r)
            else:
                from .branch_wall_events import parallel_wall_roots, branch_candidates
                generators = parallel_wall_roots(curve, other, cancellation_check=cancellation_check)
                if generators is not None:
                    reverse_generators = parallel_wall_roots(other, curve, cancellation_check=cancellation_check)
                    if reverse_generators is None:
                        raise GeometryError("parallel wall elimination has an unresolved common component")
                    first_candidates = branch_candidates(curve, generators)
                    second_candidates = branch_candidates(other, reverse_generators)
                    # A directrix can revisit a projected point. Qualify every
                    # isolated parameter on both walls, not a nearest-point
                    # inversion that can silently select only one visit.
                    for u, (lo, hi) in first_candidates:
                        for r, (other_lo, other_hi) in second_candidates:
                            if cancellation_check is not None and cancellation_check():
                                raise GeometryError("wall intersection predicate cancelled")
                            if np.any(lo > other_hi + tolerance) or np.any(other_lo > hi + tolerance):
                                continue
                            made_u, made_r, residual = _refine(curve, other, u, r, tolerance)
                            if residual > tolerance:
                                raise GeometryError("parallel wall junction cannot resolve the requested world tolerance")
                            add(made_u, made_r)
                else:                                      # other supports retain certified subdivision
                    for u, r in subdivision_junctions(curve, other, tolerance=tolerance,
                                                      cancellation_check=cancellation_check):
                        add(u, r)
    else:
        raise GeometryError("general analytic curve-pair arrangement predicate is not implemented")
    return tuple(pairs)
