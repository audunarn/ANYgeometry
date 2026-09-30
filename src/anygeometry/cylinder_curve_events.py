"""Polynomial elimination for intersections involving cylinder branch charts.

Two quadratic cylinder equations eliminate the axial parameter. The resultant
is a degree-eight tan-half-angle polynomial; exact rational Sturm counts isolate
every candidate. Branch membership and full implicit residuals qualify roots.
"""
from __future__ import annotations

from fractions import Fraction
import math
import numpy as np

from .analytic_roots import isolate_real_roots
from .errors import GeometryError
from .exact_curves import CylinderIntersectionCurve


def _trim(p):
    p = list(p)
    while len(p) > 1 and p[-1] == 0:
        p.pop()
    return tuple(p)


def _add(p, q):
    return _trim([(p[i] if i < len(p) else 0)+(q[i] if i < len(q) else 0)
                  for i in range(max(len(p), len(q)))])


def _scale(p, value):
    return _trim([coefficient*value for coefficient in p])


def _multiply(p, q):
    result = [Fraction(0)]*(len(p)+len(q)-1)
    for i, first in enumerate(p):
        for j, second in enumerate(q):
            result[i+j] += first*second
    return _trim(result)


def same_cylinder(first, second, tolerance):
    """Coincidence of untransformed infinite support equations."""
    axis = np.asarray(first.axis)
    other_axis = np.asarray(second.axis)
    offset = np.asarray(second.origin)-first.origin
    return (np.linalg.norm(np.cross(axis, other_axis)) <= 64*np.finfo(float).eps
            and np.linalg.norm(offset-float(offset @ axis)*axis) <= tolerance
            and abs(first.radius-second.radius) <= tolerance)


def _quadratic(curve, support, target_transform):
    matrix = np.linalg.solve(np.asarray(target_transform), np.asarray(curve.transform))
    first = curve.first
    base = matrix[:3, :3] @ first.origin+matrix[:3, 3]-support.origin
    cosine = matrix[:3, :3] @ (first.radius*np.asarray(first.radial_direction))
    sine = matrix[:3, :3] @ (first.radius*np.cross(first.axis, first.radial_direction))
    direction = matrix[:3, :3] @ first.axis
    axis = np.asarray(support.axis)
    project = np.eye(3)-np.outer(axis, axis)
    base, cosine, sine, direction = project @ base, project @ cosine, project @ sine, project @ direction
    # Rational polynomial numerators over R=1+t^2. All subsequent arithmetic
    # is exact over the supplied finite floating descriptor coefficients.
    numerator = [(Fraction(float(o+c)), Fraction(float(2*s)), Fraction(float(o-c)))
                 for o, c, s in zip(base, cosine, sine)]
    w = [Fraction(float(value)) for value in direction]
    a = sum(value*value for value in w)
    b, c = (Fraction(0),), (Fraction(0),)
    for values, weight in zip(numerator, w):
        b = _add(b, _scale(values, 2*weight))
        c = _add(c, _multiply(values, values))
    radius2 = Fraction(float(support.radius))**2
    c = _add(c, _scale((1, 0, 2, 0, 1), -radius2))
    return a, b, c


def _residual(point, support, transform):
    inverse = np.linalg.inv(np.asarray(transform))
    original = inverse[:3, :3] @ point+inverse[:3, 3]
    offset = original-support.origin
    axis = np.asarray(support.axis)
    radial = offset-float(offset @ axis)*axis
    return abs(float(radial @ radial)-support.radius**2)


def cylinder_roots(curve, support, *, transform=None, tolerance=1e-10,
                   cancellation_check=None):
    """All chart roots on another cylinder; None denotes a common branch."""
    if not isinstance(curve, CylinderIntersectionCurve):
        raise GeometryError("cylinder branch elimination needs CylinderIntersectionCurve")
    transform = np.eye(4) if transform is None else np.asarray(transform)
    if np.array_equal(transform, curve.transform) and any(
        same_cylinder(parent, support, tolerance) for parent in (curve.first, curve.second)):
        return None
    a, b, c = _quadratic(curve, curve.second, curve.transform)
    d, e, h = _quadratic(curve, support, transform)
    m = _add(_scale(h, a), _scale(c, -d))
    l = _add(_scale(e, a), _scale(b, -d))
    bh_ec = _add(_multiply(b, h), _scale(_multiply(e, c), -1))
    polynomial = _add(_multiply(m, m), _scale(_multiply(l, bh_ec), -1))
    implicit_tolerance = tolerance*(2*support.radius+tolerance)
    if polynomial == (0,):
        # Algebraic common-factor proof precedes branch selection. A branch
        # chart is split at transitions; checking its interior selects which
        # certified factor it represents, not an interpolating display trace.
        if _residual(curve.evaluate(.5), support, transform) <= implicit_tolerance:
            return None
        return tuple(t for t in (0., 1.) if _residual(curve.evaluate(t), support, transform) <= implicit_tolerance)
    roots = isolate_real_roots(polynomial, tolerance=4*np.finfo(float).eps,
                               cancellation_check=cancellation_check)
    angles = [2*math.atan(root.witness) for root in roots]
    if len(polynomial) < 9:
        angles.append(math.pi)
    lower, upper = sorted((curve.start_angle, curve.start_angle+curve.sweep_angle))
    from .arrangement_geometry import _angle_parameter
    parameters = []
    for angle in angles:
        for turn in range(math.ceil((lower-angle-1e-14)/math.tau), math.floor((upper-angle+1e-14)/math.tau)+1):
            parameter = min(1., max(0., _angle_parameter(curve, angle+turn*math.tau)))
            if _residual(curve.evaluate(parameter), support, transform) <= implicit_tolerance:
                if not any(abs(parameter-old) <= 64*np.finfo(float).eps for old in parameters):
                    parameters.append(parameter)
    return tuple(sorted(parameters))


def branch_junctions(first, second, *, tolerance=1e-10,cancellation_check=None):
    """Every isolated shared point or overlap endpoint of two branch charts."""
    from .arrangement_geometry import point_parameters
    constraints = [cylinder_roots(first, support, transform=second.transform, tolerance=tolerance,
                                 cancellation_check=cancellation_check)
                   for support in (second.first, second.second)]
    if all(roots is None for roots in constraints):
        candidates = (0., 1., *point_parameters(first, second.evaluate(0.), tolerance=tolerance),
                      *point_parameters(first, second.evaluate(1.), tolerance=tolerance))
    else:
        candidates = tuple(parameter for roots in constraints if roots is not None for parameter in roots)
    result = []
    for parameter in candidates:
        point = first.evaluate(parameter)
        for other_parameter in point_parameters(second, point, tolerance=tolerance):
            pair = (parameter, other_parameter)
            if not any(max(abs(a-pair[0]), abs(b-pair[1])) <= 64*np.finfo(float).eps for a, b in result):
                result.append(pair)
    return tuple(result)
