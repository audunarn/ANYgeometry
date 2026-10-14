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
from .exact_levels import event_side, has_root_in, level_set
from fractions import Fraction

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


# ------------------------------------------------------------------ exact level sets and containment


def level_roots(pcurve: NurbsCurve, axis: int, value: float, chart_tolerance: float):
    """Engine parameters where the pcurve's chart coordinate ``axis`` equals ``value`` (decided exactly).

    ``None`` when the whole pcurve lies on the level, as
    ``MaterialDomain.horizontal_roots`` reports horizontal curves. A point
    event (crossing or touch) gives one parameter, a run its two ends (from
    ``exact_levels.level_set``); curve ends within ``chart_tolerance`` of the
    level are included, as for the other supports.
    """
    events = level_set(pcurve, axis, value)
    if events is None:
        return None
    found = []
    for event in events:
        found.append(float(event.t0))
        if event.run:
            found.append(float(event.t1))
    for t in (0.0, 1.0):
        if abs(float(pcurve.evaluate(t)[axis]) - value) <= chart_tolerance:
            found.append(t)
    return tuple(sorted(set(min(1.0, max(0.0, t)) for t in found)))


def horizontal_roots(domain, curve, value, tolerance):
    return level_roots(pcurve_of(domain, curve), 1, float(value), _chart_tolerance(domain, tolerance))


def _exact_sign(pcurve: NurbsCurve, axis: int, value: float, t: Fraction) -> int:
    difference = pcurve.exact_point(t)[axis] - Fraction(value)
    return 0 if difference == 0 else (1 if difference > 0 else -1)


def contains_loop(domain, loop, uv_point, tolerance, *, boundary=True) -> bool:
    """Ray parity in the chart with every decision exact (``exact_levels``).

    Along the loop the sign of ``v - v0`` is a cyclic sequence of non-zero gaps
    and zero stretches (point events, runs, whole curves; stretches meeting at a
    curve joint are one stretch). A stretch between gaps of different sign is a
    crossing and counts when it lies right of the query, decided exactly by
    ``event_side`` (the exact sign of ``u - u0`` at the root). A stretch through
    the query (exactly, or within the tolerance in the world) is the boundary.
    """
    # The query is used exactly (doubles or Fractions; review A3b round 2: a rounded
    # chart vertex can cross a boundary); floats only for the tolerance conversion.
    exact_u0, exact_v0 = Fraction(uv_point[0]), Fraction(uv_point[1])
    u0f, v0f = float(exact_u0), float(exact_v0)
    # The world tolerance as a chart band [u0 - delta, u0 + delta] along the ray,
    # delta = tolerance / |S_u| at the query; membership in it is decided exactly.
    rate = float(np.linalg.norm(world_delta(domain, (1.0, 0.0), v0f, u0f)))
    delta = tolerance / rate if rate > 0 else 0.0
    exact_delta = Fraction(delta)
    band = (exact_u0 - exact_delta, exact_u0 + exact_delta)        # exact edges (no rounding of u0 +- delta)

    items: list = []                 # ("gap", sign) or ("zero", [part, ...]); part = (pcurve, event | None)
    for path in loop:
        pcurve = pcurve_of(domain, path.curve)
        events = level_set(pcurve, 1, exact_v0)
        if events is None:
            items.append(("zero", [(pcurve, None)]))
            continue
        if not events:
            items.append(("gap", _exact_sign(pcurve, 1, exact_v0, Fraction(1, 2))))
            continue
        if events[0].before != 0:
            items.append(("gap", events[0].before))
        for index, event in enumerate(events):
            items.append(("zero", [(pcurve, event)]))
            if event.after != 0:
                items.append(("gap", event.after))
    merged: list = []
    for kind, payload in items:
        if kind == "zero" and merged and merged[-1][0] == "zero":
            merged[-1][1].extend(payload)
        else:
            merged.append((kind, list(payload) if kind == "zero" else payload))
    if len(merged) > 1 and merged[0][0] == "zero" and merged[-1][0] == "zero":
        merged[0][1][:0] = merged.pop()[1]

    def through_query(part) -> bool:
        pcurve, event = part
        if event is not None and not event.run:
            # u(t*) in [u0 - delta, u0 + delta], exactly
            return event_side(pcurve, event, 0, band[0]) >= 0 and event_side(pcurve, event, 0, band[1]) <= 0
        lo, hi = (Fraction(0), Fraction(1)) if event is None else (event.lo, event.hi)
        # a run meets the band if u crosses a band edge or u0 on it, or lies inside the band
        if any(has_root_in(pcurve, 0, edge, lo, hi) for edge in (band[0], exact_u0, band[1])):
            return True
        return _exact_sign(pcurve, 0, band[0], lo) >= 0 and _exact_sign(pcurve, 0, band[1], lo) <= 0

    def right_of_query(part) -> bool:
        pcurve, event = part
        if event is not None and not event.run:
            return event_side(pcurve, event, 0, exact_u0) > 0
        start = Fraction(0) if event is None else event.lo
        return _exact_sign(pcurve, 0, exact_u0, start) > 0

    if not any(kind == "gap" for kind, _payload in merged):
        return boundary if any(through_query(part) for _kind, stretch in merged for part in stretch) else False
    winding = 0
    count = len(merged)
    for index, (kind, stretch) in enumerate(merged):
        if kind != "zero":
            continue
        if any(through_query(part) for part in stretch):
            return boundary
        before = next(merged[(index - k) % count][1] for k in range(1, count + 1)
                      if merged[(index - k) % count][0] == "gap")
        after = next(merged[(index + k) % count][1] for k in range(1, count + 1)
                     if merged[(index + k) % count][0] == "gap")
        if before == after:
            continue                                   # a touch
        if right_of_query(stretch[0]):
            winding += 1 if after > 0 else -1
    return winding != 0


def on_horizontal(domain, curve, uv_point, tolerance) -> bool:
    """Whether a pcurve lying on the level ``v = uv_point[1]`` passes through ``uv_point``."""
    roots = level_roots(pcurve_of(domain, curve), 0, float(uv_point[0]), _chart_tolerance(domain, tolerance))
    return roots is None or bool(roots)


# ------------------------------------------------------------------ areas


def _span_nodes(curve: NurbsCurve, count: int, extra=()) -> tuple[np.ndarray, np.ndarray]:
    """Gauss nodes and weights on [0, 1] with ``count`` points per piece between breakpoints.

    ``extra``: further breakpoints (where the pcurve crosses surface knot lines).
    """
    nodes, weights = _gauss_legendre(count)
    pieces = sorted({0.0, *curve.breakpoints(), *(t for t in extra if 0.0 < t < 1.0), 1.0})
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


def _v_breakpoints(surface: NurbsSurface) -> np.ndarray:
    start, end = surface.v_range
    low, high = min(start, end), max(start, end)
    values = sorted({(knot - start) / (end - start) for knot in surface.knots_v if low < knot < high})
    return np.asarray([value for value in values if 0.0 < value < 1.0])


def _knot_crossings(surface: NurbsSurface, pcurve: NurbsCurve) -> tuple[float, ...]:
    """Pcurve parameters where it meets an interior surface knot line (``u`` or ``v``), decided exactly.

    The world-area integrand has derivative jumps there (the surface's
    continuity), so the outer Gauss rule is split at them; a run along a knot
    line contributes its ends.
    """
    found: list[float] = []
    for axis, breaks in ((0, _u_breakpoints(surface)), (1, _v_breakpoints(surface))):
        for value in breaks:
            events = level_set(pcurve, axis, float(value))
            if events is None:
                continue
            for event in events:
                found.extend((float(event.t0), float(event.t1)))
    return tuple(found)


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
    crossings = [_knot_crossings(surface, pcurve) for pcurve in pcurves]

    def evaluate(count):
        value = magnitude = 0.0
        for pcurve, extra in zip(pcurves, crossings):
            nodes, weights = _span_nodes(pcurve, count, extra)
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


# ------------------------------------------------------------------ certified trim polylines (A3b B-S3)


def _exact_segment_distance(point, a, b) -> Fraction:
    """Exact upper bound (rational sqrt rounded up) of the distance from ``point`` to the segment ``[a, b]``."""
    from .nurbs import _sqrt_up
    direction = [y - x for x, y in zip(a, b)]
    length2 = sum(value * value for value in direction)
    offset = [p - x for p, x in zip(point, a)]
    r = Fraction(0) if length2 == 0 else min(Fraction(1), max(Fraction(0), sum(
        o * d for o, d in zip(offset, direction)) / length2))
    return _sqrt_up(sum((o - r * d) ** 2 for o, d in zip(offset, direction)))


def _segment_distance(point, a, b) -> float:
    direction = b - a
    length2 = float(direction @ direction)
    if length2 == 0.0:
        return float(np.linalg.norm(point - a))
    r = min(1.0, max(0.0, float((point - a) @ direction) / length2))
    return float(np.linalg.norm(point - (a + r * direction)))


def certified_chart_polyline(surface: NurbsSurface, pcurve: NurbsCurve, tolerance: float,
                             max_pieces: int = 1 << 16) -> tuple[np.ndarray, float]:
    """Chart polyline of a pcurve whose surface image deviates from the trim curve by at most the returned bound.

    The polyline lives in the chart (``trim_loops_uv_certified``, architecture
    3.3): the bound is for the *surface image* of each chart chord against the
    trim curve, not for a straight world segment (a mesher maps chart points
    through the surface). Each piece's exact control hull
    (``NurbsCurve._active_controls``, positive weights) lies within ``d`` of its
    chord, so the pcurve piece does too; the chart deviation in the infinity
    norm is at most the Euclidean ``d``, and the world deviation of the mapped
    chord is at most ``sqrt(2) sigma_max d`` (architecture 1.4, ``sigma_max`` of
    the whole patch). ``d`` is computed exactly (rational distances of the hull
    points to the chord through the stored float polyline vertices, rounded up)
    and the product is rounded up. Pieces are halved until it is ``<= tolerance``.
    """
    from .nurbs import _float_up, _sqrt_up
    from .surface_capabilities import capabilities
    sigma = capabilities(surface).jacobian_bounds(((0.0, 1.0), (0.0, 1.0))).sigma_max
    factor = _sqrt_up(Fraction(2)) * Fraction(sigma)
    accepted: list[tuple[float, float, float]] = []
    pending = [(0.0, 1.0)]
    while pending:
        a, b = pending.pop()
        hull = pcurve._active_controls(a, b)
        ends = [tuple(Fraction(float(value)) for value in point)
                for point in np.clip(pcurve.evaluate(np.array([a, b])), 0.0, 1.0)]
        # curve -> chord: the hull's distance to the stored chord; chord -> curve: that
        # distance plus the stored ends' errors against the exact curve ends (the curve
        # runs from near one chord end to near the other; review A3b round 2).
        end_error = max(_exact_segment_distance(pcurve.exact_point(Fraction(t)), stored, stored)
                        for t, stored in ((a, ends[0]), (b, ends[1])))
        deviation = max(_exact_segment_distance(point, ends[0], ends[1]) for point in hull) + end_error
        bound = _float_up(factor * deviation)
        if bound <= tolerance:
            accepted.append((a, b, bound))
            continue
        if len(accepted) + len(pending) >= max_pieces or b - a <= 1e-12:
            raise GeometryError("certified trim polyline: tolerance not reachable within the piece budget")
        middle = 0.5 * (a + b)
        pending.extend(((middle, b), (a, middle)))
    accepted.sort()
    parameters = np.array([piece[0] for piece in accepted] + [1.0])
    return np.clip(pcurve.evaluate(parameters), 0.0, 1.0), max(piece[2] for piece in accepted)

