"""Exact support intersections of an extruded surface with a Plane (or with another patch of itself).

A plane not parallel to the extrusion vector ``d`` meets ``c(t) + s d`` in the directrix carried along ``d``
onto the plane, ``c(t) + s(t) d`` with ``s(t) = (h - n.c(t)) / (n.d)``. A Bezier directrix gives a Bezier
curve of the same degree (an affine image of its control points, restricted to the patch in exact rational
arithmetic so a nearly parallel plane cannot spoil the conditioning) and an elliptic directrix gives an
ellipse, as a plane meets a cylinder. A plane parallel to ``d`` meets it in the generators through the
roots of ``n.c(t) = h``.
"""
from __future__ import annotations

from fractions import Fraction
import math
from math import comb

import numpy as np

from .analytic_roots import isolate_real_roots, trigonometric_roots
from .analytic_supports import SupportIntersection
from .arrangement_geometry import BezierPath
from .errors import GeometryError
from .exact_curves import EllipticArc
from .extrusions import BezierDirectrix, EllipseDirectrix
from .surfaces import Cone, Cylinder, ExtrudedSurface, Plane

_EPS = float(np.finfo(float).eps)


def _power_coefficients(values):
    """Ascending power coefficients of the Bernstein-form scalar polynomial with these coefficients."""
    n = len(values) - 1
    return [sum((-1) ** (k - i) * comb(n, i) * comb(n - i, k - i) * values[i] for i in range(k + 1))
            for k in range(n + 1)]


def _split(rows, t):
    work = [list(row) for row in rows]
    left, right = [work[0][:]], [work[-1][:]]
    for size in range(len(work) - 1, 0, -1):
        for i in range(size):
            work[i] = [(1 - t) * x + t * y for x, y in zip(work[i], work[i + 1])]
        left.append(work[0][:])
        right.append(work[size - 1][:])
    return left, right[::-1]


def _restrict(points, lower, upper):
    """Exact de Casteljau restriction of Bezier control points (rows of Fractions) to ``[lower, upper]``."""
    rows = [list(row) for row in points]
    if upper < 1:
        rows = _split(rows, upper)[0]
    if lower > 0:
        rows = _split(rows, lower / upper)[1]
    return rows


def _stretches(breakpoints, inside):
    """Merge consecutive intervals between ordered ``breakpoints`` whose midpoint satisfies ``inside``."""
    merged = []
    for a, b in zip(breakpoints, breakpoints[1:]):
        if a == b or not inside(.5 * (a + b)):
            continue
        if merged and merged[-1][1] == a:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return merged


def _bezier_section(plane, surface, tolerance, cancellation_check):
    normal = np.asarray(plane.normal, dtype=float)
    vector = np.asarray(surface._vector, dtype=float)
    rate = float(normal @ vector)
    directrix = surface.directrix
    t_low, t_high = sorted(surface.u_range)
    s_low, s_high = sorted(surface.v_range)
    controls = [[Fraction(float(x)) for x in row] for row in directrix.controls]
    n_f = [Fraction(float(x)) for x in normal]
    h_f = sum(n_f[i] * Fraction(float(plane.origin[i])) for i in range(3))
    d_f = [Fraction(float(x)) for x in vector]
    values = [sum(n_f[i] * row[i] for i in range(3)) - h_f for row in controls]               # n.c_i - h
    interval = (Fraction(t_low), Fraction(t_high))
    if rate == 0.0:
        coefficients = _power_coefficients(values)
        if not any(coefficients):
            return SupportIntersection(coincident=True)
        segments = []
        for root in isolate_real_roots(coefficients, tolerance=4 * _EPS, cancellation_check=cancellation_check,
                                       interval=interval):
            base = directrix.point(float(root.witness))
            segments.append((tuple(base + s_low * vector), tuple(base + s_high * vector)))
        return SupportIntersection(segments=tuple(segments))
    m_f = sum(n_f[i] * d_f[i] for i in range(3))
    s_values = [-v / m_f for v in values]                                                    # s_i = (h - n.c_i) / m
    moved = [[row[i] + s * d_f[i] for i in range(3)] for row, s in zip(controls, s_values)]
    power = _power_coefficients(s_values)
    breakpoints = {t_low, t_high}
    for bound in (s_low, s_high):
        coefficients = list(power)
        coefficients[0] -= Fraction(bound)
        if any(coefficients):
            for root in isolate_real_roots(coefficients, tolerance=4 * _EPS, cancellation_check=cancellation_check,
                                           interval=interval):
                breakpoints.add(float(root.witness))
    pad = tolerance / float(np.linalg.norm(vector))

    def inside(t):
        value = float(sum(c * Fraction(t) ** k for k, c in enumerate(power)))
        return s_low - pad <= value <= s_high + pad

    curves = []
    for a, b in _stretches(sorted(breakpoints), inside):
        rows = _restrict(moved, Fraction(a), Fraction(b))
        curves.append(BezierPath(tuple(tuple(float(x) for x in row) for row in rows)))
    return SupportIntersection(curves=tuple(curves))


def _ellipse_section(plane, surface, tolerance, cancellation_check):
    normal = np.asarray(plane.normal, dtype=float)
    height = float(normal @ plane.origin)
    vector = np.asarray(surface._vector, dtype=float)
    rate = float(normal @ vector)
    directrix = surface.directrix
    center, u_vec, v_vec = (np.asarray(x, dtype=float) for x in (directrix.center, directrix.u_vector, directrix.v_vector))
    t_low, t_high = sorted(surface.u_range)
    s_low, s_high = sorted(surface.v_range)
    angle_low = directrix.start_angle + t_low * directrix.sweep_angle
    angle_sweep = (t_high - t_low) * directrix.sweep_angle
    if rate == 0.0:
        coefficients = (float(normal @ center) - height, float(normal @ u_vec), float(normal @ v_vec), 0., 0.)
        if not any(coefficients):
            return SupportIntersection(coincident=True)
        segments = []
        for angle in trigonometric_roots(coefficients, start=angle_low, sweep=angle_sweep, tolerance=4 * _EPS,
                                         cancellation_check=cancellation_check):
            base = center + math.cos(angle) * u_vec + math.sin(angle) * v_vec
            segments.append((tuple(base + s_low * vector), tuple(base + s_high * vector)))
        return SupportIntersection(segments=tuple(segments))
    s_center = (height - float(normal @ center)) / rate
    s_u, s_v = -float(normal @ u_vec) / rate, -float(normal @ v_vec) / rate               # s = s_c + s_u cos + s_v sin
    breakpoints = {angle_low, angle_low + angle_sweep}
    for bound in (s_low, s_high):
        for angle in trigonometric_roots((s_center - bound, s_u, s_v, 0., 0.), start=angle_low, sweep=angle_sweep,
                                         tolerance=4 * _EPS, cancellation_check=cancellation_check):
            breakpoints.add(float(angle))
    pad = tolerance / float(np.linalg.norm(vector))

    def inside(angle):
        return s_low - pad <= s_center + s_u * math.cos(angle) + s_v * math.sin(angle) <= s_high + pad

    ordered = sorted(breakpoints)
    new_center, new_u, new_v = center + s_center * vector, u_vec + s_u * vector, v_vec + s_v * vector
    curves = tuple(EllipticArc(tuple(new_center), tuple(new_u), tuple(new_v), a, b - a)
                   for a, b in _stretches(ordered, inside))
    return SupportIntersection(curves=curves)


def plane_extruded_support(plane, surface, *, tolerance=1e-10, cancellation_check=None):
    """Exact intersections of a Plane with an extruded surface, restricted to the surface's patch."""
    if not isinstance(plane, Plane) or not isinstance(surface, ExtrudedSurface):
        raise GeometryError("analytic plane/extruded intersection needs a Plane and an ExtrudedSurface")
    if isinstance(surface.directrix, BezierDirectrix):
        return _bezier_section(plane, surface, tolerance, cancellation_check)
    return _ellipse_section(plane, surface, tolerance, cancellation_check)


def _patch_directrix(surface, origin, normal, direction):
    """The patch's directrix carried along ``direction`` into the plane ``(origin, normal)``, as a path."""
    directrix = surface.directrix
    t_low, t_high = surface.u_range
    direction = np.asarray(direction, dtype=float)
    scale = float(normal @ direction)

    def carry(points):
        points = np.asarray(points, dtype=float)
        return points - (((points - origin) @ normal) / scale)[..., None] * direction

    if isinstance(directrix, BezierDirectrix):
        piece = BezierPath(directrix.controls).subcurve(t_low, t_high)
        return BezierPath(tuple(tuple(row) for row in carry(piece.controls)))
    center, u_vec, v_vec = (np.asarray(x, dtype=float) for x in (directrix.center, directrix.u_vector, directrix.v_vector))

    def linear(vector):
        return vector - float(normal @ vector) / scale * direction

    return EllipticArc(tuple(carry(center)), tuple(linear(u_vec)), tuple(linear(v_vec)),
                       directrix.start_angle + t_low * directrix.sweep_angle, (t_high - t_low) * directrix.sweep_angle)


def parallel_extruded_support(a, b, *, tolerance=1e-10, cancellation_check=None):
    """Two extruded surfaces with parallel vectors meet in the generators through the crossings of their directrices.

    Carried along the common direction into one plane both directrices are coplanar curves; every crossing
    of them is a point shared by a generator of each surface, and the surfaces meet along that line,
    clipped to both patches. Overlapping directrices would make the surfaces overlap in a region, which
    needs an explicit ownership policy.
    """
    from .arrangement_geometry import curve_junctions, point_parameters
    d_a, d_b = np.asarray(a._vector, dtype=float), np.asarray(b._vector, dtype=float)
    if float(np.linalg.norm(np.cross(d_a, d_b))) > 1e-12 * float(np.linalg.norm(d_a) * np.linalg.norm(d_b)):
        raise GeometryError("extruded surfaces with non-parallel extrusion vectors are not supported yet")
    origin, normal = np.asarray(a.profile_origin, dtype=float), np.asarray(a.profile_normal, dtype=float)
    curve_a = _patch_directrix(a, origin, normal, d_a)
    curve_b = _patch_directrix(b, origin, normal, d_a)
    junctions = curve_junctions(curve_a, curve_b, tolerance=tolerance, cancellation_check=cancellation_check)
    ordered = sorted(junctions)
    for (first, _x), (second, _y) in zip(ordered, ordered[1:]):
        if second - first > 1e-9 and point_parameters(curve_b, curve_a.evaluate(.5 * (first + second)), tolerance=tolerance):
            raise GeometryError("positive-area overlap requires an explicit ownership policy")
    alpha = float(d_a @ d_b) / float(d_a @ d_a)                                  # b's vector in units of a's
    a_low, a_high = sorted(a.v_range)
    b_low, b_high = sorted(b.v_range)
    t_a, t_b = a.u_range, b.u_range
    segments = []
    for ta, tb in junctions:
        base = curve_a.evaluate(ta)                                              # on a's profile plane
        # b's curve point sits kappa vectors above its carried image; its generator runs z in b's own units
        t_world = t_b[0] + tb * (t_b[1] - t_b[0])
        point_b = b.directrix.point(t_world)
        kappa = float((point_b - origin) @ normal) / float(normal @ d_a)
        low = max(a_low, min(kappa + alpha * b_low, kappa + alpha * b_high))
        high = min(a_high, max(kappa + alpha * b_low, kappa + alpha * b_high))
        if (high - low) * float(np.linalg.norm(d_a)) <= tolerance:
            continue
        segments.append((tuple(base + low * d_a), tuple(base + high * d_a)))
    return SupportIntersection(segments=tuple(segments))


def _parallel(a, b):
    first, second = np.asarray(a.vector, dtype=float), np.asarray(b.vector, dtype=float)
    return float(np.linalg.norm(np.cross(first, second))) <= 1e-12 * float(np.linalg.norm(first) * np.linalg.norm(second))


def _ruled_quadric(surface):
    """Cylinders, Cones, extruded ellipses and quadratic Bezier extrusions are quadrics; other Beziers are not."""
    return isinstance(surface, (Cylinder, Cone)) or (
        isinstance(surface, ExtrudedSurface) and (isinstance(surface.directrix, EllipseDirectrix) or (
            isinstance(surface.directrix, BezierDirectrix) and surface.directrix.degree == 2)))


def _parabolic(surface):
    """A quadratic Bezier extrusion (a parabolic cylinder)."""
    return isinstance(surface, ExtrudedSurface) and isinstance(surface.directrix, BezierDirectrix)


def _bezier_wall(surface):
    """A Bezier extrusion of degree three or more: not a quadric, so the polynomial-chart engine takes it first."""
    return (isinstance(surface, ExtrudedSurface) and isinstance(surface.directrix, BezierDirectrix)
            and surface.directrix.degree >= 3)


def extruded_pair_support(a, b, *, tolerance=1e-10, cancellation_check=None):
    """Support intersection for a pair that includes an extruded surface."""
    if isinstance(b, ExtrudedSurface) and not isinstance(a, ExtrudedSurface):
        a, b = b, a
    if isinstance(b, Plane):
        return plane_extruded_support(b, a, tolerance=tolerance, cancellation_check=cancellation_check)
    if isinstance(b, ExtrudedSurface):
        if a.support_key() == b.support_key():
            return SupportIntersection(coincident=True)
        if _parallel(a, b):
            return parallel_extruded_support(a, b, tolerance=tolerance, cancellation_check=cancellation_check)
    if _ruled_quadric(a) and _ruled_quadric(b):
        if _parabolic(a) and _parabolic(b):                 # neither has an angle to supply: the first is the polynomial wall
            from .branch_supports import bezier_quadric_support
            return bezier_quadric_support(a, b, tolerance=tolerance, cancellation_check=cancellation_check)
        from .quadric_supports import quadric_pair_support
        return quadric_pair_support(a, b, tolerance=tolerance, cancellation_check=cancellation_check)
    for wall, other in ((a, b), (b, a)):
        if _bezier_wall(wall) and _ruled_quadric(other):
            from .branch_supports import bezier_quadric_support
            return bezier_quadric_support(wall, other, tolerance=tolerance, cancellation_check=cancellation_check)
    raise GeometryError("this extruded surface pair is unsupported: a Bezier extrusion of degree three or more "
                        "meets a Plane, a cylinder, a cone, a quadric extrusion or a parallel extrusion, and "
                        "two non-parallel Bezier extrusions of degree three or more are not supported")
