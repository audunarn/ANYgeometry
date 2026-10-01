"""Chart quantities of an extruded surface for the material arrangement.

The chart is ``(u, v)`` with ``t = t0 + u (t1 - t0)`` along the planar directrix and ``s = s0 + v (s1 - s0)``
along the extrusion vector ``d``. ``s`` is linear in position, so ``v`` and its rates are exact; ``u`` is
the directrix parameter of the projection ``q = p - s d`` and its rate follows from ``q' = c'(t) t'``:

``du/d(tau) = (q' . c') / (c' . c') / (t1 - t0)`` with ``q' = p' - s' d`` and ``s' = (n . p') / (n . d)``.

Everything here is vectorized over points and has no mutable state.
"""
from __future__ import annotations

import numpy as np

from .errors import GeometryError


def _state(surface, points):
    """``(s, q, t, c1)`` for points on the surface: extrusion coordinate, projection, parameter, tangent."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    s = surface.extrusion_coordinate(points)
    q = points - s[:, None] * surface._vector
    t = surface.directrix.invert(q)
    return points, s, q, t, surface.directrix.derivative(t)


def chart_rates(surface, points, velocities):
    """``(du/dtau, dv/dtau)`` rows for points moving on the surface with the given world velocities."""
    points, s, _q, _t, c1 = _state(surface, points)
    velocities = np.atleast_2d(np.asarray(velocities, dtype=float))
    s_rate = velocities @ surface.profile_normal / surface.profile_rate
    planar = velocities - s_rate[:, None] * surface._vector
    norm = np.sum(c1 * c1, axis=1)
    if np.any(norm <= 0.0):
        raise GeometryError("the extruded directrix has a singular point (a cusp) inside the chart")
    t_rate = np.sum(planar * c1, axis=1) / norm
    (u0, u1), (v0, v1) = surface.u_range, surface.v_range
    return np.column_stack((t_rate / (u1 - u0), s_rate / (v1 - v0)))


def chart_accelerations(surface, points, velocities, accelerations):
    """``(d2u/dtau2, d2v/dtau2)`` rows: the chart acceleration of a curve on the surface."""
    points, s, _q, t, c1 = _state(surface, points)
    velocities = np.atleast_2d(np.asarray(velocities, dtype=float))
    accelerations = np.atleast_2d(np.asarray(accelerations, dtype=float))
    normal, rate = surface.profile_normal, surface.profile_rate
    s_rate = velocities @ normal / rate
    s_second = accelerations @ normal / rate
    planar_first = velocities - s_rate[:, None] * surface._vector
    planar_second = accelerations - s_second[:, None] * surface._vector
    c2 = surface.directrix.second_derivative(t)
    norm = np.sum(c1 * c1, axis=1)
    t_rate = np.sum(planar_first * c1, axis=1) / norm
    t_second = (np.sum(planar_second * c1, axis=1) - np.sum(c2 * c1, axis=1) * t_rate ** 2) / norm
    (u0, u1), (v0, v1) = surface.u_range, surface.v_range
    return np.column_stack((t_second / (u1 - u0), s_second / (v1 - v0)))


def area_density(surface, u):
    """World area per unit of chart area at ``u``: ``|c'(t) x d| |t1 - t0| |s1 - s0|``."""
    (u0, u1), (v0, v1) = surface.u_range, surface.v_range
    t = u0 + np.asarray(u, dtype=float) * (u1 - u0)
    c1 = surface.directrix.derivative(t)
    return np.linalg.norm(np.cross(c1, surface._vector), axis=-1) * abs(u1 - u0) * abs(v1 - v0)


def density_bound(surface):
    """An upper bound of :func:`area_density` over the chart (for tolerance scaling only)."""
    samples = np.linspace(0.0, 1.0, 129)
    return float(np.max(area_density(surface, samples))) * 1.05


def world_metric(surface, delta, u):
    """The world displacement of the chart displacement ``delta`` at chart abscissa ``u`` (a 3-vector)."""
    (u0, u1), (v0, v1) = surface.u_range, surface.v_range
    t = u0 + float(u) * (u1 - u0)
    return (float(delta[0]) * (u1 - u0)) * surface.directrix.derivative(t) + (float(delta[1]) * (v1 - v0)) * surface._vector
