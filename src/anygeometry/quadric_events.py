"""Events of any exact arrangement curve against a quadric, and junctions with quadric branches.

``curve_quadric_roots`` is the one primitive: the chart parameters where a curve
lies on a quadric (Plane, Cylinder, Cone, ...). It is exact for every curve family
(quadratic for a line, a degree-two trigonometric polynomial for an ellipse, a
degree-``2n`` polynomial for a Bezier curve, resultant elimination for the branch
curves). A :class:`QuadricIntersectionCurve` is the common curve of two quadrics, so
every junction with one reduces to the roots of the *other* curve on those two
supports followed by a point inversion.
"""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np

from .analytic_roots import isolate_real_roots, trigonometric_roots
from .errors import GeometryError
from .quadric_algebra import QuadricSupport
from .quadric_curves import QuadricIntersectionCurve

_EPS = float(np.finfo(float).eps)


def _clip_unit(parameters):
    return tuple(min(1.0, max(0.0, float(p))) for p in parameters)


def _distance(quadric, point):
    return abs(float(quadric.value(point))) / max(float(quadric.gradient_norm(point)), 1e-300)


def _polynomial_mul(p, q):
    out = [Fraction(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a:
            for j, b in enumerate(q):
                out[i + j] += a * b
    return out


def _lies_on(curve, quadric, coefficients, tolerance):
    """The whole curve lies on the quadric: tiny implicit polynomial and every sample within tolerance."""
    total = sum(abs(float(c)) for c in coefficients)
    gradient = float(quadric.gradient_norm(curve.evaluate(.5)))
    if total > 16 * tolerance * (gradient + tolerance):
        return False
    return all(_distance(quadric, curve.evaluate(t)) <= tolerance for t in np.linspace(0.0, 1.0, 9))


def _quadratic_roots(c0, c1, c2):
    """Real roots in ``[0, 1]`` (as the engine clips them) of the exact quadratic ``c0 + c1 t + c2 t^2``."""
    if c2 == 0:
        found = [] if c1 == 0 else [float(-c0 / c1)]
    else:
        discriminant = c1 * c1 - 4 * c0 * c2                  # exact: only its sign and one rounding matter
        if discriminant < 0:
            return ()
        if discriminant == 0:
            found = [float(-c1 / (2 * c2))]
        else:
            radical = math.sqrt(float(discriminant))
            q = -.5 * (float(c1) + math.copysign(radical, float(c1)))     # no cancellation between c1 and the root
            found = sorted({q / float(c2), float(c0) / q}) if q != 0 else [0.0]
    return _clip_unit(r for r in found if -32 * _EPS <= r <= 1 + 32 * _EPS)


_MEMO = {}
_MEMO_SIZE = 8192


def curve_quadric_roots(curve, quadric, *, tolerance=1e-10, cancellation_check=None):
    """Parameters in ``[0, 1]`` where ``curve`` lies on ``quadric``; ``None`` if the whole curve does.

    The answer depends only on the curve and the support, and every facet that shares a boundary edge or
    a trace asks it again, so exact answers are remembered (a cancelled solve is not; curves that cannot
    be hashed are recomputed).
    """
    quadric = QuadricSupport.from_surface(quadric)
    try:
        key = (curve, quadric, tolerance)
        hash(key)
    except TypeError:
        return _curve_quadric_roots(curve, quadric, tolerance, cancellation_check)
    try:
        return _MEMO[key]
    except KeyError:
        pass
    result = _curve_quadric_roots(curve, quadric, tolerance, cancellation_check)
    if len(_MEMO) >= _MEMO_SIZE:
        _MEMO.pop(next(iter(_MEMO)), None)                 # oldest first: plans are local in time
    _MEMO[key] = result
    return result


def _curve_quadric_roots(curve, quadric, tolerance, cancellation_check):
    from .arrangement_geometry import BezierPath, LinePath
    from .exact_curves import CylinderIntersectionCurve, EllipticArc
    if isinstance(curve, QuadricIntersectionCurve):
        return curve.roots_on(quadric, tolerance=tolerance, cancellation_check=cancellation_check)
    if isinstance(curve, CylinderIntersectionCurve):
        from .quadric_curve_events import quadric_roots
        matrix, linear, constant = quadric.general_float()
        return quadric_roots(curve, matrix, linear, constant, tolerance=tolerance,
                             cancellation_check=cancellation_check)
    if isinstance(curve, LinePath):
        start = tuple(float(v) for v in curve.start)
        m, l, c = quadric.exact_relative(start)
        d = tuple(Fraction(float(e)) - Fraction(float(s)) for e, s in zip(curve.end, curve.start))
        md = [sum(m[i][j] * d[j] for j in range(3)) for i in range(3)]
        coefficients = (c, 2 * sum(l[i] * d[i] for i in range(3)), sum(d[i] * md[i] for i in range(3)))
        if all(v == 0 for v in coefficients) or _lies_on(curve, quadric, coefficients, tolerance):
            return None
        return _quadratic_roots(*coefficients)
    if isinstance(curve, EllipticArc):
        center = tuple(float(v) for v in curve.center)
        m, l, c = quadric.exact_relative(center)
        u = tuple(Fraction(float(v)) for v in curve.u_vector)
        v = tuple(Fraction(float(w)) for w in curve.v_vector)

        def form(a, b):
            return sum(a[i] * sum(m[i][j] * b[j] for j in range(3)) for i in range(3))

        uu, vv, uv = form(u, u), form(v, v), form(u, v)
        coefficients = (c + (uu + vv) / 2, 2 * sum(l[i] * u[i] for i in range(3)),
                        2 * sum(l[i] * v[i] for i in range(3)), (uu - vv) / 2, uv)
        if all(x == 0 for x in coefficients) or _lies_on(curve, quadric, coefficients, tolerance):
            return None
        angles = trigonometric_roots(coefficients, start=curve.start_angle, sweep=curve.sweep_angle,
                                     tolerance=4 * _EPS, cancellation_check=cancellation_check)
        return _clip_unit((a - curve.start_angle) / curve.sweep_angle for a in angles)
    if isinstance(curve, BezierPath):
        controls = np.asarray(curve.controls, dtype=float)
        origin = tuple(float(v) for v in controls[0])
        m, l, c = quadric.exact_relative(origin)
        degree = len(controls) - 1
        powers = []
        for coordinate in range(3):
            row = []
            for k in range(degree + 1):
                row.append(sum(Fraction(float(controls[i, coordinate])) * ((-1) ** (k - i) * math.comb(degree, i)
                                                                          * math.comb(degree - i, k - i))
                               for i in range(k + 1)))
            row[0] -= Fraction(origin[coordinate])
            powers.append(row)
        total = [Fraction(0)] * (2 * degree + 1)
        total[0] += c
        for i in range(3):
            for j in range(3):
                if m[i][j]:
                    for index, value in enumerate(_polynomial_mul(powers[i], powers[j])):
                        total[index] += m[i][j] * value
            for index, value in enumerate(powers[i]):
                total[index] += 2 * l[i] * value
        if all(x == 0 for x in total) or _lies_on(curve, quadric, total, tolerance):
            return None
        roots = isolate_real_roots(total, tolerance=4 * _EPS, cancellation_check=cancellation_check)
        return _clip_unit(r.witness for r in roots if -32 * _EPS <= r.witness <= 1 + 32 * _EPS)
    raise GeometryError("curve/quadric predicate is unsupported for this curve family")


def branch_supports(curve):
    """The two quadric supports of a branch curve, in world coordinates."""
    first = QuadricSupport("cone" if curve.first.is_cone else "cylinder", curve.first.origin, curve.first.axis,
                           curve.first.radius, curve.first.slope)
    second = curve.second
    if curve.transform != ((1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.)):
        inverse = np.linalg.inv(np.asarray(curve.transform))
        first, second = first.pulled_back(inverse), second.pulled_back(inverse)
    return first, second


def quadric_pair_junctions(first, second, *, tolerance=1e-10, cancellation_check=None):
    """Every isolated shared point (or overlap endpoint) of two curves, at least one a quadric branch."""
    from .arrangement_geometry import point_parameters
    swapped = not isinstance(first, QuadricIntersectionCurve)       # the branch is the second argument
    branch, other = (second, first) if swapped else (first, second)
    supports = branch_supports(branch)
    constraints = [curve_quadric_roots(other, support, tolerance=tolerance, cancellation_check=cancellation_check)
                   for support in supports]
    if all(roots is None for roots in constraints):
        # the other curve lies on both supports: endpoints of the common stretch
        candidates = (0., 1., *point_parameters(other, branch.evaluate(0.), tolerance=tolerance),
                      *point_parameters(other, branch.evaluate(1.), tolerance=tolerance))
    else:
        # every root on either support is a candidate; point inversion on the branch qualifies it
        candidates = tuple(t for roots in constraints if roots is not None for t in roots)
    result = []
    for t in candidates:
        point = other.evaluate(t)
        for u in point_parameters(branch, point, tolerance=tolerance):
            pair = (t, u) if swapped else (u, t)                 # (first parameter, second parameter)
            if not any(max(abs(a - pair[0]), abs(b - pair[1])) <= 64 * _EPS for a, b in result):
                result.append(pair)
    return tuple(result)
