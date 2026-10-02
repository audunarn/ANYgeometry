"""Exact generator events of two parallel Bezier-ruled supports.

Projection along the common ruling reduces wall intersections to intersections
of two polynomial directrices. Rational elimination retains repeated roots;
distance subdivision is not needed to distinguish a tangent from separation.
"""
from fractions import Fraction

import numpy as np

from .analytic_roots import isolate_real_roots
from .bezier_intersections import _resultant
from .cylinder_curve_events import _add, _scale, _trim
from .errors import GeometryError


def _world_power(curve):
    support = curve.first
    rows = support.exact_power()
    rows[0] = tuple(Fraction(x) for x in support.origin)
    transform = [[Fraction(x) for x in row] for row in curve.transform]
    power = [tuple(sum(transform[i][j] * row[j] for j in range(3))
                   + (transform[i][3] if k == 0 else 0) for i in range(3))
             for k, row in enumerate(rows)]
    direction = tuple(sum(transform[i][j] * Fraction(support.direction[j])
                          for j in range(3)) for i in range(3))
    return power, direction


def parallel_wall_roots(first, second, *, cancellation_check=None):
    """Certified first-directrix root intervals for common generators, or ``None``.

    ``None`` leaves non-parallel supports and common projected components to
    their existing predicate. Root candidates still need both curve charts to
    qualify them: projection alone does not select a quadric branch.
    """
    def check():
        if cancellation_check is not None and cancellation_check():
            raise GeometryError("wall intersection predicate cancelled")

    check()
    a, da = _world_power(first)
    b, db = _world_power(second)
    if any(da[i] * db[j] != da[j] * db[i] for i, j in ((0, 1), (0, 2), (1, 2))):
        return None
    pivot = max(range(3), key=lambda i: abs(da[i]))
    axes = [i for i in range(3) if i != pivot]
    # Homogeneous quotient coordinates avoid a rounded rotation or division.
    def project(rows, axis):
        return _trim(tuple(row[axis] * da[pivot] - row[pivot] * da[axis] for row in rows))

    equations = []
    for axis in axes:
        pa, pb = project(a, axis), project(b, axis)
        equations.append((_add(pa, (-pb[0],)), *((-value,) for value in pb[1:])))
    polynomial = _resultant(*equations, check)
    if polynomial == (0,):
        return None
    roots = isolate_real_roots(polynomial, tolerance=4 * np.finfo(float).eps,
                               cancellation_check=cancellation_check)
    return tuple(root for root in roots if root.upper >= 0 and root.lower <= 1)


def parallel_wall_parameters(first, second, *, cancellation_check=None):
    roots = parallel_wall_roots(first, second, cancellation_check=cancellation_check)
    return None if roots is None else tuple(min(1., max(0., root.witness)) for root in roots)


def branch_candidates(curve, roots):
    """Chart witnesses and enclosing boxes, including near-fold conditioning."""
    lower, upper = sorted((curve.start, curve.start + curve.sweep))
    candidates = []
    for root in roots:
        a = max(lower, float(np.nextafter(float(root.lower), -np.inf)))
        b = min(upper, float(np.nextafter(float(root.upper), np.inf)))
        if a > b:
            continue
        u, v = sorted(float(curve.parameter_for_t(t)) for t in (a, b))
        u, v = max(0., float(np.nextafter(u, -np.inf))), min(1., float(np.nextafter(v, np.inf)))
        parameter = float(curve.parameter_for_t(min(b, max(a, root.witness))))
        candidates.append((parameter, curve.bounds(u, v)))
    return candidates
