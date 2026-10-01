"""Planar directrix curves of extruded surfaces: an exact Bezier polynomial or an ellipse.

An extruded surface is ``S(t, s) = c(t) + s * d``. When the directrix ``c`` lies in a plane (the
profile plane, normal ``n``) the extrusion coordinate is *linear in position*,
``s = n . (p - o) / (n . d)``, so the level lines of ``s`` are plane sections exactly as for a Cylinder,
and the directrix parameter of a surface point is the parameter of its projection ``p - s d`` on the
planar directrix. This module holds the two directrix families and nothing about surfaces, so it can be
imported by :mod:`anygeometry.surfaces` without a cycle.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .errors import GeometryError

TWO_PI = 2.0 * math.pi
_PLANAR_RELATIVE = 1e-10          # control points may leave their common plane by this fraction of the extent
_EPS = float(np.finfo(float).eps)


def _vector3(value, name):
    vector = np.array(value, dtype=float, copy=True)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise GeometryError(f"{name} must be a finite 3-vector")
    return vector


def _tuple3(vector):
    return tuple(float(v) for v in vector)


def _frame_of(normal):
    """Orthonormal ``(e1, e2)`` spanning the plane of a unit ``normal`` (deterministic)."""
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < .9 else np.array([0.0, 1.0, 0.0])
    e1 = helper - float(helper @ normal) * normal
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(normal, e1)


@dataclass(frozen=True, eq=False)
class BezierDirectrix:
    """A planar Bezier curve of degree ``n >= 2`` (``n + 1`` control points), parameter ``t`` in ``[0, 1]``."""

    controls: tuple

    def __post_init__(self):
        array = np.array(self.controls, dtype=float, copy=True)
        if array.ndim != 2 or array.shape[1] != 3 or len(array) < 3 or not np.all(np.isfinite(array)):
            raise GeometryError("a Bezier directrix needs at least three finite control points")
        centroid = array.mean(axis=0)
        _u, singular, vectors = np.linalg.svd(array - centroid)
        extent = float(np.linalg.norm(np.ptp(array, axis=0)))
        if extent <= 0.0 or float(singular[1]) <= 1e-12 * extent:
            raise GeometryError("a Bezier directrix must not degenerate to a line (extrude it as a plane)")
        if float(singular[-1]) > _PLANAR_RELATIVE * extent:
            raise GeometryError("an extruded Bezier directrix must be planar")
        normal = vectors[-1] / np.linalg.norm(vectors[-1])
        # a deterministic sign: the first nonzero component of the normal is positive
        for component in normal:
            if abs(component) > 1e-12:
                normal = normal if component > 0 else -normal
                break
        array.flags.writeable = False
        object.__setattr__(self, "controls", tuple(_tuple3(row) for row in array))
        object.__setattr__(self, "_array", array)
        object.__setattr__(self, "origin", centroid)
        object.__setattr__(self, "normal", normal)

    kind = "bezier"

    @property
    def degree(self):
        return len(self.controls) - 1

    def _key(self):
        return ("bezier", self.controls)

    def __eq__(self, other):
        return isinstance(other, BezierDirectrix) and self.controls == other.controls

    def __hash__(self):
        return hash(self._key())

    # ---------------------------------------------------------------- evaluation

    @staticmethod
    def _de_casteljau(controls, t):
        t = np.asarray(t, dtype=float)
        work = np.broadcast_to(controls, t.shape + controls.shape).copy()
        for size in range(len(controls) - 1, 0, -1):
            work[..., :size, :] = ((1 - t[..., None, None]) * work[..., :size, :]
                                   + t[..., None, None] * work[..., 1:size + 1, :])
        return work[..., 0, :]

    def point(self, t):
        return self._de_casteljau(self._array, t)

    def derivative(self, t):
        degree = self.degree
        return self._de_casteljau(degree * np.diff(self._array, axis=0), t)

    def second_derivative(self, t):
        degree = self.degree
        if degree < 2:
            return np.zeros(np.shape(t) + (3,))
        return self._de_casteljau(degree * (degree - 1) * np.diff(self._array, n=2, axis=0), t)

    def bounds(self):
        return self._array.min(axis=0), self._array.max(axis=0)

    def transformed(self, matrix):
        matrix = np.asarray(matrix, dtype=float)
        return BezierDirectrix(tuple(_tuple3(p) for p in self._array @ matrix[:3, :3].T + matrix[:3, 3]))

    # ---------------------------------------------------------------- inversion

    _TABLE = 129

    def _inversion_data(self):
        """The sample table and the power-basis polynomials of the curve and its two derivatives (built once)."""
        data = self.__dict__.get("_inversion")
        if data is None:
            table_t = np.linspace(0.0, 1.0, self._TABLE)
            table = self.point(table_t)
            degree = self.degree
            power = np.array([math.comb(degree, k) * sum((-1) ** (k - i) * math.comb(k, i) * self._array[i]
                                                          for i in range(k + 1)) for k in range(degree + 1)])
            first = power[1:] * np.arange(1, degree + 1)[:, None]
            second = first[1:] * np.arange(1, degree)[:, None]
            data = (table_t, table, np.einsum("ij,ij->i", table, table), power, first, second)
            object.__setattr__(self, "_inversion", data)
        return data

    @staticmethod
    def _horner(coefficients, t):
        value = np.broadcast_to(coefficients[-1], t.shape + (3,))
        for row in coefficients[-2::-1]:
            value = value * t[:, None] + row
        return value

    def invert(self, points, *, iterations=8):
        """Parameter ``t`` of the curve point nearest each planar point (Newton from a sampled table)."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        table_t, table, table_norm, power, first, second = self._inversion_data()
        # |p - c|^2 = |p|^2 - 2 p.c + |c|^2; the constant |p|^2 does not change the nearest sample
        t = table_t[np.argmin(table_norm[None, :] - 2.0 * (points @ table.T), axis=1)]
        done = np.zeros(len(t), dtype=bool)               # a converged point stops moving, whatever else is in the batch
        for _ in range(iterations):
            offset = self._horner(power, t) - points
            d1, d2 = self._horner(first, t), self._horner(second, t)
            numerator = np.einsum("ij,ij->i", offset, d1)
            denominator = np.einsum("ij,ij->i", d1, d1) + np.einsum("ij,ij->i", offset, d2)
            safe = np.where(np.abs(denominator) > 1e-300, denominator, 1.0)
            step = numerator / safe
            t = np.where(done, t, np.clip(t - step, 0.0, 1.0))
            done |= np.abs(step) <= 4.0 * _EPS
            if done.all():
                break
        return t


@dataclass(frozen=True, eq=False)
class EllipseDirectrix:
    """``center + u cos(a) + v sin(a)`` with ``a = start + t * sweep`` for ``t`` in ``[0, 1]``."""

    center: tuple
    u_vector: tuple
    v_vector: tuple
    start_angle: float = 0.0
    sweep_angle: float = TWO_PI

    def __post_init__(self):
        center, u, v = _vector3(self.center, "center"), _vector3(self.u_vector, "u_vector"), _vector3(self.v_vector, "v_vector")
        cross = np.cross(u, v)
        if float(np.linalg.norm(cross)) <= 1e-14 * max(1.0, float(np.linalg.norm(u) * np.linalg.norm(v))):
            raise GeometryError("ellipse vectors must be independent")
        start, sweep = float(self.start_angle), float(self.sweep_angle)
        if not math.isfinite(start) or not math.isfinite(sweep) or sweep == 0.0 or abs(sweep) > TWO_PI + 1e-12:
            raise GeometryError("ellipse angles must be finite with a nonzero sweep of at most a turn")
        object.__setattr__(self, "center", _tuple3(center))
        object.__setattr__(self, "u_vector", _tuple3(u))
        object.__setattr__(self, "v_vector", _tuple3(v))
        object.__setattr__(self, "start_angle", start)
        object.__setattr__(self, "sweep_angle", sweep)
        object.__setattr__(self, "_c", center)
        object.__setattr__(self, "_u", u)
        object.__setattr__(self, "_v", v)
        object.__setattr__(self, "origin", center)
        object.__setattr__(self, "normal", cross / np.linalg.norm(cross))

    kind = "ellipse"

    def _key(self):
        return ("ellipse", self.center, self.u_vector, self.v_vector, self.start_angle, self.sweep_angle)

    def __eq__(self, other):
        return isinstance(other, EllipseDirectrix) and self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

    def _angle(self, t):
        return self.start_angle + np.asarray(t, dtype=float) * self.sweep_angle

    def point(self, t):
        a = self._angle(t)
        return self._c + np.cos(a)[..., None] * self._u + np.sin(a)[..., None] * self._v

    def derivative(self, t):
        a = self._angle(t)
        return self.sweep_angle * (-np.sin(a)[..., None] * self._u + np.cos(a)[..., None] * self._v)

    def second_derivative(self, t):
        a = self._angle(t)
        return -self.sweep_angle ** 2 * (np.cos(a)[..., None] * self._u + np.sin(a)[..., None] * self._v)

    def bounds(self):
        reach = np.sqrt(self._u ** 2 + self._v ** 2)             # the full ellipse bounds every arc of it
        return self._c - reach, self._c + reach

    def transformed(self, matrix):
        matrix = np.asarray(matrix, dtype=float)
        return EllipseDirectrix(_tuple3(matrix[:3, :3] @ self._c + matrix[:3, 3]), _tuple3(matrix[:3, :3] @ self._u),
                                _tuple3(matrix[:3, :3] @ self._v), self.start_angle, self.sweep_angle)

    def invert(self, points):
        """Parameter ``t`` of each planar point on the ellipse: exact, from a 2 x 2 linear solve."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        matrix = np.column_stack((self._u, self._v))
        solution = np.linalg.lstsq(matrix, (points - self._c).T, rcond=None)[0]
        angle = np.arctan2(solution[1], solution[0])
        raw = angle - self.start_angle
        tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(self.start_angle), abs(self.sweep_angle))
        if self.sweep_angle > 0.0:
            delta = np.mod(raw, TWO_PI)
            delta = np.where(np.abs(delta - TWO_PI) <= tolerance, 0.0, delta)
        else:
            delta = -np.mod(-raw, TWO_PI)
            delta = np.where(np.abs(delta + TWO_PI) <= tolerance, 0.0, delta)
        delta = np.where(np.abs(raw) <= tolerance, 0.0, delta)
        return delta / self.sweep_angle


Directrix = (BezierDirectrix, EllipseDirectrix)
