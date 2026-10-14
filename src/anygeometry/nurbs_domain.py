"""Material domains on NURBS charts (generality slice A1e-lite, architecture 1.4, 4.6, 4.7).

A ``MaterialDomain`` whose support is a ``NurbsSurface`` works in the surface
chart [0, 1]^2: every boundary path is a ``CurveOnSurface`` on that support with
the identity transform, and chart points, tangents and curvature come from its
pcurve; world points are never inverted (architecture 4.6). Anything else is
refused.

Decisions here are on the float route (level roots by Bernstein subdivision);
certified NURBS-chart predicates arrive with A3b/A4. Areas: the chart area is
Green's theorem on the pcurves; the world area is ``oint G dv`` with
``G(u, v) = int_0^u J(s, v) ds`` and ``J = |S_u x S_v|``, both by Gauss rules per
knot span doubled until they settle. ``world_area_estimate`` returns the value
and the last change as an *estimate* (a rigorous bound is slice R5).
"""
from __future__ import annotations

import math

import numpy as np

from ._cross3 import _gauss_legendre
from .errors import GeometryError
from .nurbs import CurveOnSurface, NurbsCurve, NurbsSurface, _scaled_weights
from .reference import bspline as _reference

_EPS = float(np.finfo(float).eps)
_IDENTITY = tuple(tuple(1.0 if i == j else 0.0 for j in range(4)) for i in range(4))


def is_nurbs_domain(domain) -> bool:
    return isinstance(domain.support, NurbsSurface)


def pcurve_of(domain, curve) -> NurbsCurve:
    """The chart pcurve of a boundary curve on the domain's support, or a refusal."""
    if (not isinstance(curve, CurveOnSurface) or curve.support != domain.support
            or curve.transform != _IDENTITY):
        raise GeometryError("a NURBS material domain needs CurveOnSurface boundaries on its own support "
                            "(identity transform)")
    return curve.pcurve


def _chart_tolerance(domain, tolerance: float) -> float:
    """World tolerance to chart tolerance, ``tol / (sqrt(2) sigma_max)`` (architecture 1.4, forward only)."""
    from .surface_capabilities import capabilities
    sigma = capabilities(domain.support).jacobian_bounds(((0.0, 1.0), (0.0, 1.0))).sigma_max
    return tolerance / (math.sqrt(2.0) * sigma) if sigma > 0 else tolerance


def area_jacobian(domain) -> float:
    """An upper bound of the area element ``|S_u x S_v| <= sigma_max^2`` over the chart."""
    from .surface_capabilities import capabilities
    sigma = capabilities(domain.support).jacobian_bounds(((0.0, 1.0), (0.0, 1.0))).sigma_max
    return sigma * sigma


def world_delta(domain, delta, v=0.0, u=0.0) -> np.ndarray:
    s_u, s_v = domain.support.derivatives_many(np.array([[min(1.0, max(0.0, float(u))),
                                                           min(1.0, max(0.0, float(v)))]]))
    return float(delta[0]) * s_u[0] + float(delta[1]) * s_v[0]


def uv(domain, curve, parameter) -> np.ndarray:
    pcurve_of(domain, curve)
    return np.asarray(curve.chart(float(parameter)), dtype=float)


def tangent(domain, curve, parameter) -> np.ndarray:
    pcurve_of(domain, curve)
    return np.asarray(curve.chart_derivative(float(parameter)), dtype=float)


def curvature(domain, curve, parameter, forward) -> float | None:
    pcurve_of(domain, curve)
    first = np.asarray(curve.chart_derivative(float(parameter)), dtype=float) * (1 if forward else -1)
    second = np.asarray(curve.chart_second_derivative(float(parameter)), dtype=float)
    length = float(np.linalg.norm(first))
    if length == 0.0:
        return None
    return float(first[0] * second[1] - first[1] * second[0]) / length ** 3


# ------------------------------------------------------------------ level roots of a pcurve coordinate


def _bernstein_split(coefficients: np.ndarray, at: float = 0.5):
    """De Casteljau split of a Bernstein polynomial on [0, 1] at ``at``."""
    work = coefficients.astype(float).copy()
    left, right = [work[0]], [work[-1]]
    for _ in range(len(work) - 1):
        work = (1 - at) * work[:-1] + at * work[1:]
        left.append(work[0])
        right.append(work[-1])
    return np.asarray(left), np.asarray(right[::-1])


def _bernstein_value(coefficients: np.ndarray, s: float) -> float:
    work = coefficients.astype(float).copy()
    for _ in range(len(work) - 1):
        work = (1 - s) * work[:-1] + s * work[1:]
    return float(work[0])


def _bernstein_roots(coefficients: np.ndarray, floor: float) -> list[float]:
    """Roots in [0, 1] of a Bernstein polynomial by subdivision (float route).

    ``floor``: coefficients within it of zero count as zero (rounding of the
    numerator). Returns sorted parameters; touching roots are reported where the
    polynomial stays within ``floor`` on a tiny interval.
    """
    roots: list[float] = []
    stack = [(0.0, 1.0, coefficients)]
    while stack:
        a, b, c = stack.pop()
        if np.all(c > floor) or np.all(c < -floor):
            continue
        if b - a <= 1e-15 or len(stack) > 4000:
            roots.append(0.5 * (a + b))
            continue
        # A single sign change of the control polygon and differing end signs: bisect directly.
        signs = np.sign(np.where(np.abs(c) <= floor, 0.0, c))
        nonzero = signs[signs != 0]
        if len(nonzero) and np.count_nonzero(np.diff(nonzero)) == 1 and abs(c[0]) > floor and abs(c[-1]) > floor:
            lo, hi, f_lo = 0.0, 1.0, c[0]
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                f_mid = _bernstein_value(c, mid)
                if f_mid == 0.0:
                    lo = hi = mid
                    break
                if (f_mid > 0) == (f_lo > 0):
                    lo, f_lo = mid, f_mid
                else:
                    hi = mid
            roots.append(a + (b - a) * 0.5 * (lo + hi))
            continue
        left, right = _bernstein_split(c)
        middle = 0.5 * (a + b)
        stack.append((middle, b, right))
        stack.append((a, middle, left))
    roots.sort()
    merged: list[float] = []
    for root in roots:
        if not merged or root - merged[-1] > 1e-12:
            merged.append(root)
    return merged


def level_roots(pcurve: NurbsCurve, axis: int, value: float, chart_tolerance: float):
    """Engine parameters where the pcurve's chart coordinate ``axis`` equals ``value``.

    ``None`` when the pcurve lies on that level (every control within the
    tolerance of it), as ``MaterialDomain.horizontal_roots`` reports horizontal
    curves. Per Bezier span the rational numerator ``sum B_i w_i (c_i - value)``
    is solved by Bernstein subdivision; endpoints within ``chart_tolerance``
    are included.
    """
    controls = np.asarray(pcurve.controls, dtype=float)
    if np.all(np.abs(controls[:, axis] - value) <= chart_tolerance):
        return None
    start, end = pcurve.domain
    weights = _scaled_weights(pcurve.weights)
    found: list[float] = []
    for s0, s1, span_controls, span_weights in _reference.bezier_segments(
            pcurve.degree, np.asarray(pcurve.knots), controls, weights):
        low, high = (min(start, end), max(start, end))
        if s1 <= low or s0 >= high:
            continue
        w = np.ones(len(span_controls)) if span_weights is None else np.asarray(span_weights)
        numerator = w * (np.asarray(span_controls)[:, axis] - value)
        floor = 8 * _EPS * float(np.max(np.abs(w * np.asarray(span_controls)[:, axis])) + abs(value) * np.max(w))
        for r in _bernstein_roots(numerator, floor):
            s = s0 + (s1 - s0) * r
            if low <= s <= high:
                found.append((s - start) / (end - start))
    for t in (0.0, 1.0):
        if abs(float(pcurve.evaluate(t)[axis]) - value) <= chart_tolerance:
            found.append(t)
    found = sorted(min(1.0, max(0.0, t)) for t in found)
    unique: list[float] = []
    for t in found:
        if not unique or t - unique[-1] > 128 * _EPS:
            unique.append(t)
        elif t in (0.0, 1.0):
            unique[-1] = t                                     # prefer the exact end
    return tuple(unique)


def horizontal_roots(domain, curve, value, tolerance):
    return level_roots(pcurve_of(domain, curve), 1, float(value), _chart_tolerance(domain, tolerance))


def on_horizontal(domain, curve, uv_point, tolerance) -> bool:
    """Whether a pcurve lying on the level ``v = uv_point[1]`` passes through ``uv_point``."""
    roots = level_roots(pcurve_of(domain, curve), 0, float(uv_point[0]), _chart_tolerance(domain, tolerance))
    return roots is None or bool(roots)


# ------------------------------------------------------------------ areas


def _span_nodes(curve: NurbsCurve, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss nodes and weights on [0, 1] with ``count`` points per span between breakpoints."""
    nodes, weights = _gauss_legendre(count)
    pieces = (0.0, *curve.breakpoints(), 1.0)
    all_nodes, all_weights = [], []
    for a, b in zip(pieces, pieces[1:]):
        all_nodes.append(a + (b - a) * 0.5 * (nodes + 1))
        all_weights.append(0.5 * (b - a) * weights)
    return np.concatenate(all_nodes), np.concatenate(all_weights)


def _settle(evaluate, tolerance: float, name: str) -> tuple[float, float]:
    previous = None
    for count in (8, 16, 32, 64, 128):
        value, magnitude = evaluate(count)
        if previous is not None:
            change = abs(value - previous)
            if change <= max(tolerance, 64 * _EPS * magnitude):
                return value, change
        previous = value
    raise GeometryError(f"{name} did not settle to the requested tolerance")


def area_loop(domain, loop, tolerance) -> float:
    """Signed chart area ``1/2 oint (u dv - v du)`` (positive counterclockwise)."""
    pcurves = [pcurve_of(domain, path.curve) for path in loop]

    def evaluate(count):
        value = magnitude = 0.0
        for pcurve in pcurves:
            nodes, weights = _span_nodes(pcurve, count)
            q = np.clip(pcurve.evaluate(nodes), 0.0, 1.0)
            dq = pcurve.derivative(nodes)
            samples = 0.5 * (q[:, 0] * dq[:, 1] - q[:, 1] * dq[:, 0])
            value += float(weights @ samples)
            magnitude += float(weights @ np.abs(samples))
        return value, magnitude

    return _settle(evaluate, tolerance, "NURBS chart area")[0]


def _u_breakpoints(surface: NurbsSurface) -> np.ndarray:
    """Chart u values in (0, 1) of the surface's interior u knots inside its range."""
    start, end = surface.u_range
    low, high = min(start, end), max(start, end)
    values = sorted({(knot - start) / (end - start) for knot in surface.knots_u if low < knot < high})
    return np.asarray([value for value in values if 0.0 < value < 1.0])


def _area_density(surface: NurbsSurface, uv_points: np.ndarray) -> np.ndarray:
    s_u, s_v = surface.derivatives_many(uv_points)
    return np.linalg.norm(np.cross(s_u, s_v), axis=1)


def _inner_integral(surface: NurbsSurface, u: np.ndarray, v: np.ndarray, count: int) -> np.ndarray:
    """``G(u, v) = int_0^u J(s, v) ds`` per point, Gauss with ``count`` nodes per knot span piece."""
    nodes, weights = _gauss_legendre(count)
    breaks = _u_breakpoints(surface)
    total = np.zeros(len(u))
    # pieces [b_k, b_{k+1}] intersected with [0, u]: integrate each point's own pieces
    edges = np.concatenate(([0.0], breaks, [1.0]))
    for a, b in zip(edges, edges[1:]):
        upper = np.clip(u, a, b)
        width = upper - a
        active = width > 0
        if not np.any(active):
            continue
        s = a + width[active, None] * 0.5 * (nodes[None, :] + 1)
        vv = np.repeat(v[active, None], count, axis=1)
        density = _area_density(surface, np.column_stack((s.ravel(), vv.ravel()))).reshape(s.shape)
        total[active] += 0.5 * width[active] * (density @ weights)
    return total


def world_area_estimate(domain, loop, tolerance) -> tuple[float, float]:
    """Signed world area ``oint G(u, v) dv`` and the last change of the doubling (an estimate, not a bound)."""
    surface = domain.support
    pcurves = [pcurve_of(domain, path.curve) for path in loop]

    def evaluate(count):
        value = magnitude = 0.0
        for pcurve in pcurves:
            nodes, weights = _span_nodes(pcurve, count)
            q = np.clip(pcurve.evaluate(nodes), 0.0, 1.0)
            dq = pcurve.derivative(nodes)
            moving = dq[:, 1] != 0.0
            samples = np.zeros(len(nodes))
            if np.any(moving):
                samples[moving] = _inner_integral(surface, q[moving, 0], q[moving, 1], count) * dq[moving, 1]
            value += float(weights @ samples)
            magnitude += float(weights @ np.abs(samples))
        return value, magnitude

    return _settle(evaluate, tolerance, "NURBS world area")


def world_area_loop(domain, loop, tolerance) -> float:
    return world_area_estimate(domain, loop, tolerance)[0]
