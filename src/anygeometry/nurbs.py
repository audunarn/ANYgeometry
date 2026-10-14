"""NURBS definitions: curves, pcurves, surfaces and curves on surfaces (slice A0).

Pure definitions with validation, evaluation and schema-7 records. They are
immutable, self-contained values (control points are owned arrays, not
topology vertices) and carry no engine routes yet: model integration,
intersections and meshing consume them in later slices
(docs/agent_plans/GENERALITY_ARCHITECTURE.md, sections 3 and 8).

Parameter convention, as for the other exact curves: the engine parameter
``t`` in [0, 1] maps affinely onto ``domain`` (a sub-interval of the knot
domain, possibly reversed); a surface chart point ``(u, v)`` in [0, 1]^2 maps
affinely onto ``u_range`` x ``v_range``.

Validation (typed ``GeometryError``):

* degree 1..``MAX_DEGREE`` (7);
* clamped knot vectors (end multiplicity exactly degree + 1, interior
  multiplicity at most the degree, non-empty spans above 64 eps of the knot
  scale, which also bounds every span below by about 2**-46 of the range) and
  knot magnitudes at most 2**100;
* finite controls whose nonzero coordinates lie in [2**-500, 2**500];
* weights ``None`` (polynomial) or positive and finite with max/min <= 1e6.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
import math
from typing import Sequence

import numpy as np

from . import backend
from .errors import GeometryError
from .reference import bspline as _reference

__all__ = [
    "cylinder_to_nurbs",
    "nurbs_circular_arc",
    "MAX_DEGREE",
    "NurbsCurve",
    "NurbsSurface",
    "CurveOnSurface",
    "nurbs_record",
    "nurbs_from_record",
]

MAX_DEGREE = 7
_KNOT_MAGNITUDE = 2.0 ** 100
_KNOT_RANGE = 2.0 ** -100
_IDENTITY = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))


# ----------------------------------------------------------------------------------- validation helpers


def _real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise GeometryError(f"{name} must be a real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise GeometryError(f"{name} must be a finite real number") from error
    if not math.isfinite(result):
        raise GeometryError(f"{name} must be finite")
    return result


def _degree(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 1 <= int(value) <= MAX_DEGREE:
        raise GeometryError(f"{name} must be an integer from 1 to {MAX_DEGREE}")
    return int(value)


def _sequence(values: object, name: str) -> Sequence:
    if (isinstance(values, (str, bytes)) or not isinstance(values, (Sequence, np.ndarray))
            or (isinstance(values, np.ndarray) and values.ndim == 0)):
        raise GeometryError(f"{name} must be a sequence")
    return values


def _ratio_exceeds(largest: float, smallest: float, limit: float) -> bool:
    """``largest / smallest > limit`` decided exactly (no rounding at subnormal magnitudes)."""
    return Fraction(largest) > Fraction(limit) * Fraction(smallest)


def _real_tuple(values: object, name: str) -> tuple[float, ...]:
    return tuple(_real(value, name) for value in _sequence(values, name))


def _knots(degree: int, values: object, count: int, name: str) -> tuple[float, ...]:
    knots = _real_tuple(values, name)
    try:
        backend.check_clamped_knots(degree, np.asarray(knots, dtype=float), count)
    except ValueError as error:
        raise GeometryError(f"{name}: {error}") from None
    if max(abs(knots[0]), abs(knots[-1])) > _KNOT_MAGNITUDE:
        raise GeometryError(f"{name}: knot magnitudes must not exceed 2**100")
    if knots[-1] - knots[0] < _KNOT_RANGE:
        raise GeometryError(f"{name}: the knot range must be at least 2**-100")
    return knots


def _check_coordinates(values: Sequence[float], name: str) -> None:
    for value in values:
        if value != 0.0 and not 2.0 ** -500 <= abs(value) <= 2.0 ** 500:
            raise GeometryError(f"{name}: nonzero coordinates must lie within [2**-500, 2**500] in magnitude")


def _weights(values: object, count: int, name: str) -> tuple[float, ...] | None:
    if values is None:
        return None
    weights = _real_tuple(values, name)
    if len(weights) != count or min(weights) <= 0.0:
        raise GeometryError(f"{name} must hold one positive weight per control point")
    if _ratio_exceeds(max(weights), min(weights), backend.MAX_WEIGHT_RATIO):
        raise GeometryError(f"{name}: weight ratio max/min must not exceed 1e6")
    return weights


def _interval(value: object, lower: float, upper: float, name: str) -> tuple[float, float]:
    if value is None:
        return lower, upper
    pair = _real_tuple(value, name)
    if len(pair) != 2 or pair[0] == pair[1] or not all(lower <= item <= upper for item in pair):
        raise GeometryError(f"{name} must be two distinct values inside [{lower!r}, {upper!r}]")
    return pair


def _affine(matrix: object, name: str = "transformation") -> np.ndarray:
    rows = [_real_tuple(row, name) for row in _sequence(matrix, name)]
    values = np.asarray(rows, dtype=float) if len(rows) == 4 and all(len(row) == 4 for row in rows) else np.zeros(0)
    if values.shape != (4, 4) or not np.array_equal(values[3], (0.0, 0.0, 0.0, 1.0)):
        raise GeometryError(f"{name} must be a finite 4x4 affine matrix")
    # Nonsingular exactly: the determinant of the stored doubles in rationals
    # (no underflow and no condition-number threshold).
    (a, b, c), (d, e, f), (g, h, i) = ([Fraction(float(x)) for x in row[:3]] for row in values[:3])
    if a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g) == 0:
        raise GeometryError(f"{name} must be nonsingular")
    return values


def _parameters(values: object, name: str = "curve parameters") -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise GeometryError(f"{name} must be finite and in [0, 1]")
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise GeometryError(f"{name} must be finite and in [0, 1]") from error
    if not np.all(np.isfinite(result)) or np.any((result < 0.0) | (result > 1.0)):
        raise GeometryError(f"{name} must be finite and in [0, 1]")
    return result


def _pair(first: object, second: object) -> np.ndarray:
    """Two scalar parameters in [0, 1] as an array; anything else raises GeometryError."""
    return _parameters(np.array([_real(first, "parameter"), _real(second, "parameter")]))


def _call(function, *args):
    try:
        return function(*args)
    except ValueError as error:
        raise GeometryError(str(error)) from None


def _map(t: np.ndarray, start: float, end: float) -> np.ndarray:
    """Endpoint-exact affine map of [0, 1] onto [start, end] (t = 0 -> start, t = 1 -> end)."""
    s = (1.0 - t) * start + t * end
    return np.clip(s, min(start, end), max(start, end))


def _scaled_weights(weights) -> np.ndarray | None:
    """Weights scaled by an exact power of two so the largest lies in [0.5, 1); the curve is unchanged."""
    if weights is None:
        return None
    values = np.asarray(weights, dtype=float)
    return np.ldexp(values, -np.frexp(values.max())[1])


# ----------------------------------------------------------------------------------- curves


@dataclass(frozen=True, slots=True)
class NurbsCurve:
    """Clamped B-spline or NURBS curve in 3D, or in a 2D chart (a pcurve).

    ``controls`` are 2- or 3-tuples (all the same); ``weights`` is ``None`` for
    a polynomial curve. ``domain`` is the sub-interval of the knot domain that
    the engine parameter [0, 1] maps onto (default: all of it; reversed allowed).
    """

    degree: int
    knots: tuple[float, ...]
    controls: tuple[tuple[float, ...], ...]
    weights: tuple[float, ...] | None = None
    domain: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        degree = _degree(self.degree, "NURBS degree")
        controls = tuple(_real_tuple(point, "NURBS control point")
                         for point in _sequence(self.controls, "NURBS controls"))
        dimensions = {len(point) for point in controls}
        if len(controls) < degree + 1 or len(dimensions) != 1 or dimensions.pop() not in (2, 3):
            raise GeometryError("NURBS controls must be at least degree + 1 points of dimension 2 or 3")
        _check_coordinates([value for point in controls for value in point], "NURBS controls")
        knots = _knots(degree, self.knots, len(controls), "NURBS knots")
        weights = _weights(self.weights, len(controls), "NURBS weights")
        domain = _interval(self.domain, knots[degree], knots[len(controls)], "NURBS domain")
        object.__setattr__(self, "degree", degree)
        object.__setattr__(self, "knots", knots)
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "domain", domain)

    @property
    def dimension(self) -> int:
        return len(self.controls[0])

    @property
    def rational(self) -> bool:
        return self.weights is not None

    def _knot_parameters(self, parameters) -> tuple[np.ndarray, float]:
        t = _parameters(parameters)
        start, end = self.domain
        return _map(t, start, end), end - start

    def derivatives(self, parameters, order: int = 0) -> np.ndarray:
        """Point and derivatives with respect to the engine parameter, shape ``(..., order+1, dim)``.

        Raises ``GeometryError`` when a derivative in the knot parameter is not
        representable in float64 (extreme knot spans and coordinates).
        """
        if type(order) is not int or order not in (0, 1, 2):
            raise GeometryError("derivative order must be the integer 0, 1 or 2")
        s, scale = self._knot_parameters(parameters)
        values = _call(backend.bspline_curve, self.degree, self.knots, self.controls, self.weights,
                       np.ravel(s), order)
        for k in range(1, order + 1):                 # chain rule one factor at a time (no scale**k)
            values[:, k:, :] *= scale
        if not np.all(np.isfinite(values)):
            raise GeometryError("NURBS derivative is not representable in float64")
        return values.reshape(np.shape(s) + (order + 1, self.dimension))

    def evaluate(self, parameters) -> np.ndarray:
        return self.derivatives(parameters, 0)[..., 0, :]

    def derivative(self, parameters) -> np.ndarray:
        return self.derivatives(parameters, 1)[..., 1, :]

    def second_derivative(self, parameters) -> np.ndarray:
        return self.derivatives(parameters, 2)[..., 2, :]

    def subcurve(self, lower, upper) -> "NurbsCurve":
        s = self._knot_parameters(_pair(lower, upper))[0]
        if s[0] == s[1]:
            raise GeometryError("curve interval must have positive length")
        return replace(self, domain=(float(s[0]), float(s[1])))

    def _active_controls(self, lower=0.0, upper=1.0) -> list:
        """Exact projected controls whose hull contains the curve on ``[lower, upper]``.

        The chart interval (clipped to [0, 1] like ``_map``) is mapped exactly
        and widened by the float map's error bound (``_knot_interval``), so it
        covers both the exact and the float-evaluated parameters. Its ends are
        inserted exactly (rational A5.1 on the homogeneous controls) up to
        multiplicity ``degree``; the controls of the basis functions active on
        it bound the curve there (positive weights: convex hull). No rounding
        until the caller rounds outward (review A2a round 2).
        """
        if lower > upper:
            raise GeometryError("curve bound interval is reversed")
        low, high = (min(1.0, max(0.0, _real(value, "curve parameter"))) for value in (lower, upper))
        s0, s1 = _knot_interval(low, high, self.domain)
        weights = self.weights if self.weights is not None else (1.0,) * len(self.controls)
        points = []
        for point, weight in zip(self.controls, weights):
            w = Fraction(weight)
            points.append(tuple(Fraction(value) * w for value in point) + (w,))
        _, refined, active = _exact_refine(self.degree, [Fraction(k) for k in self.knots], points, s0, s1)
        return [tuple(value / refined[i][-1] for value in refined[i][:-1]) for i in active]

    def bounds(self, lower=0.0, upper=1.0):
        """Outward-rounded box containing the curve on ``[lower, upper]`` (see ``_active_controls``).

        The exact hull box is widened by ``16 (degree + 1) eps`` of the control
        magnitude so the engine's float evaluations (which round) also lie inside;
        ``distance_lower_bound`` uses the unwidened exact hull.
        """
        points = self._active_controls(lower, upper)
        dimension = self.dimension
        scale = np.max(np.abs(np.asarray(self.controls)), axis=0)
        margin = 16 * (self.degree + 1) * np.finfo(float).eps * scale
        low = np.array([_float_down(min(point[k] for point in points)) for k in range(dimension)])
        high = np.array([_float_up(max(point[k] for point in points)) for k in range(dimension)])
        with np.errstate(over="ignore"):
            return np.nextafter(low - margin, -np.inf), np.nextafter(high + margin, np.inf)

    def distance_lower_bound(self, lower, upper, target) -> float:
        """A lower bound of the distance from ``target`` to the curve on ``[lower, upper]``.

        Exact: the larger of the squared distance to the exact control box and
        the squared half-space bound ``(min_i n . (P_i - x))^2 / |n|^2`` (when
        positive) for ``n`` the float direction towards the mid-interval curve
        point (any non-zero ``n`` is valid: every hull point lies on the far side
        of the plane). The hull hugs the curve to second order, so the bound
        converges quadratically (projection branch and bound). Rounded down.
        """
        points = self._active_controls(lower, upper)
        x = tuple(Fraction(float(value)) for value in np.asarray(target, dtype=float))
        squared = Fraction(0)
        for k in range(self.dimension):
            low, high = min(point[k] for point in points), max(point[k] for point in points)
            gap = max(low - x[k], x[k] - high, Fraction(0))
            squared += gap * gap
        with np.errstate(over="ignore", invalid="ignore"):
            probe = self.evaluate(.5 * (float(lower) + float(upper))) - np.asarray(target, dtype=float)
        if np.all(np.isfinite(probe)) and np.any(probe != 0.0):
            n = tuple(Fraction(float(value)) for value in probe)
            margin = min(sum(a * (b - c) for a, b, c in zip(n, point, x)) for point in points)
            if margin > 0:
                squared = max(squared, margin * margin / sum(a * a for a in n))
        return _float_down(_sqrt_down(squared))

    def breakpoints(self) -> tuple[float, ...]:
        """Engine parameters in (0, 1) of the distinct interior knots inside the domain (span ends)."""
        start, end = self.domain
        low, high = min(start, end), max(start, end)
        values = sorted({(knot - start) / (end - start) for knot in self.knots if low < knot < high})
        return tuple(value for value in values if 0. < value < 1.)

    def exact_point(self, parameter) -> tuple[Fraction, ...]:
        """The exact rational point at the engine parameter ``parameter`` (a Fraction or double).

        De Boor in homogeneous form (Piegl & Tiller A4.1) in exact arithmetic on
        the stored doubles; used for decisions on curve ends (architecture 3.4).
        """
        t = Fraction(parameter)
        if not 0 <= t <= 1:
            raise GeometryError("curve parameter must lie in [0, 1]")
        start, end = (Fraction(value) for value in self.domain)
        s = start + t * (end - start)
        knots = [Fraction(value) for value in self.knots]
        p, count = self.degree, len(self.controls)
        span = p
        while span < count - 1 and knots[span + 1] <= s:
            span += 1
        weights = [Fraction(1)] * count if self.weights is None else [Fraction(w) for w in self.weights]
        points = [[Fraction(c) * weights[i] for c in self.controls[i]] + [weights[i]] for i in range(count)]
        d = [list(points[span - p + j]) for j in range(p + 1)]
        for r in range(1, p + 1):
            for j in range(p, r - 1, -1):
                i = span - p + j
                denominator = knots[i + p - r + 1] - knots[i]
                alpha = Fraction(0) if denominator == 0 else (s - knots[i]) / denominator
                d[j] = [(1 - alpha) * a + alpha * b for a, b in zip(d[j - 1], d[j])]
        homogeneous = d[p]
        return tuple(value / homogeneous[-1] for value in homogeneous[:-1])

    def transformed(self, matrix) -> "NurbsCurve":
        if self.dimension != 3:
            raise GeometryError("only 3D NURBS curves have a model transformation")
        values = _affine(matrix)
        points = np.asarray(self.controls) @ values[:3, :3].T + values[:3, 3]
        return replace(self, controls=tuple(tuple(map(float, point)) for point in points))


# ----------------------------------------------------------------------------------- surfaces


@dataclass(frozen=True, slots=True)
class NurbsSurface:
    """Clamped tensor-product B-spline or NURBS surface.

    ``controls[i][j]`` is the 3D control point of u-index ``i`` and v-index
    ``j``; ``weights`` has the same layout or is ``None``. The chart [0, 1]^2
    maps affinely onto ``u_range`` x ``v_range`` (default: the knot domains).
    """

    degree_u: int
    degree_v: int
    knots_u: tuple[float, ...]
    knots_v: tuple[float, ...]
    controls: tuple[tuple[tuple[float, float, float], ...], ...]
    weights: tuple[tuple[float, ...], ...] | None = None
    u_range: tuple[float, float] | None = None
    v_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        degree_u = _degree(self.degree_u, "NURBS surface degree_u")
        degree_v = _degree(self.degree_v, "NURBS surface degree_v")
        controls = tuple(tuple(_real_tuple(point, "NURBS surface control point")
                               for point in _sequence(row, "NURBS surface control row"))
                         for row in _sequence(self.controls, "NURBS surface controls"))
        count_u = len(controls)
        count_v = len(controls[0]) if controls else 0
        if (count_u < degree_u + 1 or count_v < degree_v + 1 or any(len(row) != count_v for row in controls)
                or any(len(point) != 3 for row in controls for point in row)):
            raise GeometryError("NURBS surface controls must be a full grid of 3D points, "
                                "at least (degree_u + 1) x (degree_v + 1)")
        _check_coordinates([value for row in controls for point in row for value in point], "NURBS surface controls")
        knots_u = _knots(degree_u, self.knots_u, count_u, "NURBS surface knots_u")
        knots_v = _knots(degree_v, self.knots_v, count_v, "NURBS surface knots_v")
        weights = None
        if self.weights is not None:
            if len(_sequence(self.weights, "NURBS surface weights")) != count_u:
                raise GeometryError("NURBS surface weights must match the control grid")
            weights = tuple(_weights(_sequence(row, "NURBS surface weight row"), count_v, "NURBS surface weights")
                            for row in self.weights)
            flat = [value for row in weights for value in row]
            if _ratio_exceeds(max(flat), min(flat), backend.MAX_WEIGHT_RATIO):
                raise GeometryError("NURBS surface weights: weight ratio max/min must not exceed 1e6")
        u_range = _interval(self.u_range, knots_u[degree_u], knots_u[count_u], "NURBS surface u_range")
        v_range = _interval(self.v_range, knots_v[degree_v], knots_v[count_v], "NURBS surface v_range")
        for name, value in (("degree_u", degree_u), ("degree_v", degree_v), ("knots_u", knots_u),
                            ("knots_v", knots_v), ("controls", controls), ("weights", weights),
                            ("u_range", u_range), ("v_range", v_range)):
            object.__setattr__(self, name, value)

    def support_key(self) -> tuple:
        """Identity of the untrimmed support (independent of the chart sub-rectangle)."""
        return (self.degree_u, self.degree_v, self.knots_u, self.knots_v, self.controls, self.weights)

    def evaluate_many(self, uv) -> np.ndarray:
        """Surface points for chart points ``uv`` of shape ``(..., 2)`` in [0, 1]^2."""
        uv = _parameters(uv, "surface chart points")
        if uv.shape[-1:] != (2,):
            raise GeometryError("surface chart points must have shape (..., 2)")
        flat = uv.reshape(-1, 2)
        u = _map(flat[:, 0], *self.u_range)
        v = _map(flat[:, 1], *self.v_range)
        knots_u, knots_v = np.asarray(self.knots_u), np.asarray(self.knots_v)
        count_u, count_v = len(self.controls), len(self.controls[0])
        spans_u = _reference.find_spans(self.degree_u, knots_u, count_u, u)
        spans_v = _reference.find_spans(self.degree_v, knots_v, count_v, v)
        basis_u = _reference.basis_derivatives(self.degree_u, knots_u, spans_u, u, 0)[:, 0, :]
        basis_v = _reference.basis_derivatives(self.degree_v, knots_v, spans_v, v, 0)[:, 0, :]
        points = np.asarray(self.controls)
        weights = np.ones((count_u, count_v)) if self.weights is None else np.asarray(self.weights)
        weights = _scaled_weights(weights)
        homogeneous = np.concatenate((points * weights[..., None], weights[..., None]), axis=-1)
        rows = spans_u[:, None] - self.degree_u + np.arange(self.degree_u + 1)[None, :]
        cols = spans_v[:, None] - self.degree_v + np.arange(self.degree_v + 1)[None, :]
        local = homogeneous[rows[:, :, None], cols[:, None, :]]                 # (m, p+1, q+1, 4)
        values = np.einsum("mi,mj,mijd->md", basis_u, basis_v, local)
        result = values[:, :3] / values[:, 3:]
        if not np.all(np.isfinite(result)):
            raise GeometryError("NURBS surface evaluation is not representable in float64")
        return result.reshape(uv.shape[:-1] + (3,))

    def evaluate(self, u: float, v: float) -> np.ndarray:
        return self.evaluate_many(_pair(u, v)[None, :])[0]

    def derivatives_many(self, uv) -> tuple[np.ndarray, np.ndarray]:
        """First chart derivatives ``(S_u, S_v)`` for chart points of shape ``(n, 2)`` in [0, 1]^2.

        ``S_u = sum N_i' M_j w_ij (P_ij - S) / w`` (and likewise ``S_v``) in a frame
        anchored at the first active control: ``P_ij - S = D_ij - E`` with
        ``D_ij = P_ij - P_anchor`` and ``E = sum N M w D / w``, so a translated
        surface has the same derivatives and a constant row gives exact zeros.
        """
        uv = _parameters(uv, "surface chart points")
        if uv.ndim != 2 or uv.shape[1] != 2:
            raise GeometryError("surface chart points must have shape (n, 2)")
        u = _map(uv[:, 0], *self.u_range)
        v = _map(uv[:, 1], *self.v_range)
        knots_u, knots_v = np.asarray(self.knots_u), np.asarray(self.knots_v)
        count_u, count_v = len(self.controls), len(self.controls[0])
        spans_u = _reference.find_spans(self.degree_u, knots_u, count_u, u)
        spans_v = _reference.find_spans(self.degree_v, knots_v, count_v, v)
        basis_u = _reference.basis_derivatives(self.degree_u, knots_u, spans_u, u, 1)
        basis_v = _reference.basis_derivatives(self.degree_v, knots_v, spans_v, v, 1)
        points = np.asarray(self.controls)
        weights = _scaled_weights(np.ones((count_u, count_v)) if self.weights is None else np.asarray(self.weights))
        rows = spans_u[:, None] - self.degree_u + np.arange(self.degree_u + 1)[None, :]
        cols = spans_v[:, None] - self.degree_v + np.arange(self.degree_v + 1)[None, :]
        local_points = points[rows[:, :, None], cols[:, None, :]]                 # (m, p+1, q+1, 3)
        local_weights = weights[rows[:, :, None], cols[:, None, :]]               # (m, p+1, q+1)
        w = np.einsum("mi,mj,mij->m", basis_u[:, 0], basis_v[:, 0], local_weights)
        # Local frame anchored at the first active control (as the curve kernel):
        # offsets are small differences, so translation adds no rounding.
        anchor = local_points[:, 0, 0, :]
        delta = local_points - anchor[:, None, None, :]
        offset = np.einsum("mi,mj,mij,mijd->md", basis_u[:, 0], basis_v[:, 0], local_weights, delta) / w[:, None]
        differences = delta - offset[:, None, None, :]
        s_u = np.einsum("mi,mj,mij,mijd->md", basis_u[:, 1], basis_v[:, 0], local_weights, differences) / w[:, None]
        s_v = np.einsum("mi,mj,mij,mijd->md", basis_u[:, 0], basis_v[:, 1], local_weights, differences) / w[:, None]
        s_u *= self.u_range[1] - self.u_range[0]
        s_v *= self.v_range[1] - self.v_range[0]
        if not (np.all(np.isfinite(s_u)) and np.all(np.isfinite(s_v))):
            raise GeometryError("NURBS surface derivative is not representable in float64")
        return s_u, s_v

    def exact_net(self) -> tuple[tuple[tuple[Fraction, ...], ...], ...]:
        """Homogeneous control net ``(w x, w y, w z, w)`` as exact rationals of the stored doubles."""
        count_v = len(self.controls[0])
        weights = self.weights if self.weights is not None else ((1.0,) * count_v,) * len(self.controls)
        net = []
        for row, weight_row in zip(self.controls, weights):
            entries = []
            for point, weight in zip(row, weight_row):
                w = Fraction(weight)
                entries.append(tuple(Fraction(value) * w for value in point) + (w,))
            net.append(tuple(entries))
        return tuple(net)

    def exact_point(self, u, v, net=None) -> tuple[Fraction, Fraction, Fraction]:
        """The exact rational point at the chart point ``(u, v)`` (Fractions or doubles).

        The chart map is the exact affine map of [0, 1] onto the ranges (the
        engine's float evaluation rounds it; see ``patch_bounds`` for the slack).
        """
        net = self.exact_net() if net is None else net
        s, t = _exact_chart(u, self.u_range), _exact_chart(v, self.v_range)
        knots_u, knots_v = [Fraction(k) for k in self.knots_u], [Fraction(k) for k in self.knots_v]
        p, q, count_u, count_v = self.degree_u, self.degree_v, len(net), len(net[0])
        span_u, span_v = _exact_span(p, knots_u, count_u, s), _exact_span(q, knots_v, count_v, t)
        basis_u, basis_v = _exact_basis(p, knots_u, span_u, s), _exact_basis(q, knots_v, span_v, t)
        total = [Fraction(0)] * 4
        for a, nu in enumerate(basis_u):
            for b, nv in enumerate(basis_v):
                entry = net[span_u - p + a][span_v - q + b]
                factor = nu * nv
                total = [x + factor * y for x, y in zip(total, entry)]
        return tuple(value / total[3] for value in total[:3])

    def patch_bounds(self, box, net=None) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...], Fraction]:
        """Exact box and squared Jacobian bound of the sub-patch over a chart box.

        ``box = ((u0, u1), (v0, v1))`` inside [0, 1]^2 (either order). The knot
        intervals are the exact chart images widened by ``4 ulp`` of the range
        ends (the engine's float chart map ``_map`` errs by less than that) and
        clipped to the ranges, so the result also covers the float-evaluated
        parameters. The sub-patch net comes from exact knot insertion of the
        interval ends to multiplicity ``degree`` (A5.1); its projected controls
        bound the patch (convex hull property, positive weights).

        The squared bound ``sigma2`` of the largest singular value of
        ``[S_u S_v]`` (chart derivatives) is ``|S_u|^2 + |S_v|^2`` bounded with
        ``S_u = sum N_i' M_j w_ij (P_ij - S) / w`` summed by parts:
        ``S_u = p sum_i N_{i,p-1} / (k_{i+p} - k_i) sum_j M_j c_ij / w`` with
        ``c_ij = w_ij (P_ij - P_{i-1,j}) + (w_ij - w_{i-1,j}) (P_{i-1,j} - S)``, so
        ``|S_u| <= p |scale| max_i max_j (w_ij |P_ij - P_{i-1,j}| + |w_ij - w_{i-1,j}| D)
        / (k_{i+p} - k_i) / w_min`` over the degree ``p - 1`` basis functions active on
        the interval (``D`` the sub-net box diameter bounds ``|P - S|``; the
        partitions of unity and ``w >= w_min`` do the rest). On a shrinking box the
        control differences converge to the derivative and the weight
        differences vanish, so the bound converges to ``|S_u|`` (up to the
        knot-spacing ratio).
        """
        net = self.exact_net() if net is None else net
        (u0, u1), (v0, v1) = box
        s0, s1 = _knot_interval(u0, u1, self.u_range)
        t0, t1 = _knot_interval(v0, v1, self.v_range)
        p, q, count_v = self.degree_u, self.degree_v, len(net[0])
        rows = [sum(row, ()) for row in net]                       # each row one flat vector (a u-curve "point")
        knots_u, rows, active_u = _exact_refine(p, [Fraction(k) for k in self.knots_u], rows, s0, s1)
        columns = [sum((rows[i][4 * j: 4 * j + 4] for i in active_u), ()) for j in range(count_v)]
        knots_v, columns, active_v = _exact_refine(q, [Fraction(k) for k in self.knots_v], columns, t0, t1)
        grid = {}                                                  # (refined u index, refined v index) -> (P, w)
        for j in active_v:
            column = columns[j]
            for a, i in enumerate(active_u):
                x, y, z, w = column[4 * a: 4 * a + 4]
                grid[i, j] = ((x / w, y / w, z / w), w)
        points = [entry[0] for entry in grid.values()]
        weights = [entry[1] for entry in grid.values()]
        lower = tuple(min(point[k] for point in points) for k in range(3))
        upper = tuple(max(point[k] for point in points) for k in range(3))
        diameter = _sqrt_up(sum(((b - a) ** 2 for a, b in zip(lower, upper)), Fraction(0)))
        w_min = min(weights)
        bound_u = _derivative_bound(p, knots_u, s0, s1, self.u_range, active_u, active_v,
                                    lambda i, j: grid[i, j], diameter) / w_min
        bound_v = _derivative_bound(q, knots_v, t0, t1, self.v_range, active_v, active_u,
                                    lambda j, i: grid[i, j], diameter) / w_min
        sigma2 = bound_u ** 2 + bound_v ** 2
        return lower, upper, sigma2

    def iso_curve(self, fixed: str, value: float) -> "NurbsCurve":
        """The curve ``u = value`` (``fixed="u"``, running in v) or ``v = value`` (running in u).

        Computed exactly at the engine's float knot parameter of ``value`` and
        rounded once (controls and weights to nearest): the curve parameter is the
        other chart parameter, so ``curve.evaluate(t)`` matches
        ``surface.evaluate(value, t)`` (or ``(t, value)``) to rounding.
        """
        if fixed not in ("u", "v"):
            raise GeometryError('iso_curve: fixed must be "u" or "v"')
        chart = _real(value, "iso-curve parameter")
        if not 0.0 <= chart <= 1.0:
            raise GeometryError("iso-curve parameter must lie in [0, 1]")
        net = self.exact_net()
        if fixed == "v":
            net = tuple(zip(*net))
        degree, knots, chart_range = ((self.degree_u, self.knots_u, self.u_range) if fixed == "u"
                                      else (self.degree_v, self.knots_v, self.v_range))
        other = (self.degree_v, self.knots_v, self.v_range) if fixed == "u" else (self.degree_u, self.knots_u,
                                                                                    self.u_range)
        s = Fraction(float(_map(np.array([chart]), *chart_range)[0]))
        exact_knots = [Fraction(k) for k in knots]
        span = _exact_span(degree, exact_knots, len(net), s)
        basis = _exact_basis(degree, exact_knots, span, s)
        controls, weights = [], []
        for j in range(len(net[0])):
            total = [Fraction(0)] * 4
            for a, n in enumerate(basis):
                total = [x + n * y for x, y in zip(total, net[span - degree + a][j])]
            weights.append(float(total[3]))
            controls.append(tuple(float(x / total[3]) for x in total[:3]))
        rational = self.weights is not None
        tiny = [value for point in controls for value in point if value != 0.0 and abs(value) < 2.0 ** -500]
        if tiny:
            raise GeometryError("iso-curve controls fall below the NURBS coordinate range (nonzero |x| < 2**-500)")
        return NurbsCurve(other[0], other[1], tuple(controls), tuple(weights) if rational else None, other[2])

    def transformed(self, matrix) -> "NurbsSurface":
        values = _affine(matrix)
        points = np.asarray(self.controls) @ values[:3, :3].T + values[:3, 3]
        return replace(self, controls=tuple(tuple(tuple(map(float, point)) for point in row) for row in points))


@dataclass(frozen=True, slots=True)
class CurveOnSurface:
    """The exact composition ``transform(S(q(t)))`` of a NURBS support and a chart pcurve.

    The pcurve is the definition: the curve lies on ``support`` by
    construction. ``also_on`` (exact sharing with another support) needs the
    proof of slice A3b and must be empty until then.
    """

    support: NurbsSurface
    pcurve: NurbsCurve
    also_on: tuple = ()
    transform: tuple = _IDENTITY

    def __post_init__(self) -> None:
        if not isinstance(self.support, NurbsSurface):
            raise GeometryError("a curve on a surface needs a NurbsSurface support")
        if not isinstance(self.pcurve, NurbsCurve) or self.pcurve.dimension != 2:
            raise GeometryError("a curve on a surface needs a 2D NurbsCurve pcurve")
        if not isinstance(self.also_on, (tuple, list)) or len(self.also_on) != 0:
            raise GeometryError("also_on requires an exact shared-boundary proof (not available yet)")
        # Supported domain, deliberately narrower than "curve inside the chart":
        # the control polygon lies in [0, 1]^2, which with positive weights
        # keeps the whole pcurve in the chart.
        if any(not (0.0 <= value <= 1.0) for point in self.pcurve.controls for value in point):
            raise GeometryError("the pcurve control polygon must lie inside the chart [0, 1]^2")
        matrix = _affine(self.transform)
        object.__setattr__(self, "also_on", ())
        object.__setattr__(self, "transform", tuple(tuple(map(float, row)) for row in matrix))

    def evaluate(self, parameters) -> np.ndarray:
        chart = np.clip(self.pcurve.evaluate(parameters), 0.0, 1.0)
        points = self.support.evaluate_many(chart)
        matrix = np.asarray(self.transform)
        with np.errstate(over="ignore", invalid="ignore"):
            result = points @ matrix[:3, :3].T + matrix[:3, 3]
        if not np.all(np.isfinite(result)):
            raise GeometryError("curve on surface is not representable in float64 after its transform")
        return result

    def subcurve(self, lower, upper) -> "CurveOnSurface":
        return replace(self, pcurve=self.pcurve.subcurve(lower, upper))

    def transformed(self, matrix) -> "CurveOnSurface":
        return replace(self, transform=tuple(map(tuple, _affine(matrix) @ np.asarray(self.transform))))


# ----------------------------------------------------------------------------------- records (schema 7)


def _exact(record: dict, fields: set[str], name: str) -> None:
    if not isinstance(record, dict) or set(record) != fields:
        missing = fields - set(record) if isinstance(record, dict) else fields
        extra = set(record) - fields if isinstance(record, dict) else set()
        raise GeometryError(f"{name} record must have exactly the fields {sorted(fields)} "
                            f"(missing {sorted(missing)}, unexpected {sorted(extra)})")


def nurbs_record(value) -> dict:
    """Schema-7 record of a NURBS definition (``nurbs_curve``, ``nurbs_pcurve``, ``nurbs_surface``, ``curve_on_surface``)."""
    if isinstance(value, NurbsCurve):
        return {"type": "nurbs_curve" if value.dimension == 3 else "nurbs_pcurve",
                "degree": value.degree, "knots": list(value.knots),
                "controls": [list(point) for point in value.controls],
                "weights": None if value.weights is None else list(value.weights),
                "domain": list(value.domain)}
    if isinstance(value, NurbsSurface):
        return {"type": "nurbs_surface", "degree_u": value.degree_u, "degree_v": value.degree_v,
                "knots_u": list(value.knots_u), "knots_v": list(value.knots_v),
                "controls": [[list(point) for point in row] for row in value.controls],
                "weights": None if value.weights is None else [list(row) for row in value.weights],
                "u_range": list(value.u_range), "v_range": list(value.v_range)}
    if isinstance(value, CurveOnSurface):
        return {"type": "curve_on_surface", "support": nurbs_record(value.support),
                "pcurve": nurbs_record(value.pcurve), "also_on": [],
                "transform": [list(row) for row in value.transform]}
    raise GeometryError(f"not a NURBS definition: {type(value).__name__}")


_CURVE_FIELDS = {"type", "degree", "knots", "controls", "weights", "domain"}
_SURFACE_FIELDS = {"type", "degree_u", "degree_v", "knots_u", "knots_v", "controls", "weights", "u_range", "v_range"}
_ON_SURFACE_FIELDS = {"type", "support", "pcurve", "also_on", "transform"}


def nurbs_from_record(record: object):
    """Decode and validate a schema-7 NURBS record (strict field sets)."""
    if not isinstance(record, dict) or not isinstance(record.get("type"), str):
        raise GeometryError("NURBS record must be an object with a type")
    kind = record["type"]
    if kind in ("nurbs_curve", "nurbs_pcurve"):
        _exact(record, _CURVE_FIELDS, kind)
        curve = NurbsCurve(record["degree"], record["knots"], record["controls"], record["weights"], record["domain"])
        if curve.dimension != (3 if kind == "nurbs_curve" else 2):
            raise GeometryError(f"{kind} record has controls of dimension {curve.dimension}")
        return curve
    if kind == "nurbs_surface":
        _exact(record, _SURFACE_FIELDS, kind)
        return NurbsSurface(record["degree_u"], record["degree_v"], record["knots_u"], record["knots_v"],
                            record["controls"], record["weights"], record["u_range"], record["v_range"])
    if kind == "curve_on_surface":
        _exact(record, _ON_SURFACE_FIELDS, kind)
        if record["also_on"] != []:
            raise GeometryError("also_on requires an exact shared-boundary proof (not available yet)")
        support, pcurve = nurbs_from_record(record["support"]), nurbs_from_record(record["pcurve"])
        if not isinstance(support, NurbsSurface) or not (isinstance(pcurve, NurbsCurve) and pcurve.dimension == 2):
            raise GeometryError("curve_on_surface needs a nurbs_surface support and a nurbs_pcurve")
        return CurveOnSurface(support, pcurve, (), record["transform"])
    raise GeometryError(f"unsupported NURBS record type {kind!r}")



def nurbs_circular_arc(center, x_axis, y_axis, radius: float, start: float, sweep: float) -> NurbsCurve:
    """A circular arc as a rational quadratic NURBS curve (Piegl & Tiller A7.1).

    The arc lies in the plane of ``x_axis`` and ``y_axis``, which are
    orthonormalised (``x`` normalised, ``y`` made orthogonal to it and
    normalised, as ``Cylinder`` treats its radial direction); zero or parallel
    axes are refused. It runs from angle ``start`` over ``sweep``
    (``0 < |sweep| <= 2 pi``), split into at most four segments of <= 90 degrees
    with middle weights ``cos(delta / 2)``. Segment ends are rotations of the
    start ray by ``k delta`` (libm reduces ``start`` exactly; no rounded sums of
    large angles). The curve is exact in its stored doubles; it is a different
    exact object from an ``EllipticArc`` or a ``Cylinder`` (architecture 3.5:
    never reclassified).
    """
    center = np.asarray(_real_tuple(center, "arc centre"))
    x_axis, y_axis = np.asarray(_real_tuple(x_axis, "arc x axis")), np.asarray(_real_tuple(y_axis, "arc y axis"))
    radius, start, sweep = _real(radius, "arc radius"), _real(start, "arc start"), _real(sweep, "arc sweep")
    if radius <= 0 or sweep == 0 or abs(sweep) > 2 * math.pi:
        raise GeometryError("a NURBS circular arc needs a positive radius and 0 < |sweep| <= 2 pi")
    if center.shape != x_axis.shape or center.shape != y_axis.shape or len(center) not in (2, 3):
        raise GeometryError("arc centre and axes must be 2D or 3D vectors of one dimension")
    x_length = float(np.linalg.norm(x_axis))
    if not np.isfinite(x_length) or x_length == 0.0:
        raise GeometryError("arc x axis must be a non-zero finite vector")
    raw_x, raw_y = x_axis, y_axis
    x_axis = x_axis / x_length
    y_scale = float(np.linalg.norm(y_axis))
    # y - x (x.y)/(x.x) in exact rationals on the stored axes, rounded once: the
    # arc's plane is the input plane to rounding even for nearly parallel axes.
    exact_x = [Fraction(float(value)) for value in raw_x]
    exact_y = [Fraction(float(value)) for value in raw_y]
    xx = sum(a * a for a in exact_x)
    xy = sum(a * b for a, b in zip(exact_x, exact_y))
    y_axis = np.array([float(b * xx - a * xy) for a, b in zip(exact_x, exact_y)]) / float(xx)
    y_length = float(np.linalg.norm(y_axis))
    if not np.isfinite(y_length) or y_length <= 1e-12 * y_scale:
        raise GeometryError("arc y axis must be non-zero and not parallel to the x axis")
    y_axis = y_axis / y_length
    segments = max(1, math.ceil(abs(sweep) / (math.pi / 2) - 1e-12))
    delta = sweep / segments
    middle_weight = math.cos(delta / 2)
    c0, s0 = math.cos(start), math.sin(start)

    def ray(offset):                                  # (cos, sin)(start + offset) by rotation
        c, s = math.cos(offset), math.sin(offset)
        return c0 * c - s0 * s, s0 * c + c0 * s

    def at(direction):
        return center + radius * (direction[0] * x_axis + direction[1] * y_axis)

    controls, weights = [tuple(map(float, at((c0, s0))))], [1.0]
    for k in range(segments):
        first, second = ray(k * delta), ray((k + 1) * delta)
        tangent0 = -first[1] * x_axis + first[0] * y_axis
        middle = at(first) + radius * math.tan(delta / 2) * tangent0       # tangent intersection
        controls += [tuple(map(float, middle)), tuple(map(float, at(second)))]
        weights += [middle_weight, 1.0]
    knots = [0.0, 0.0, 0.0]
    for k in range(1, segments):
        knots += [k / segments, k / segments]
    knots += [1.0, 1.0, 1.0]
    return NurbsCurve(2, tuple(knots), tuple(controls), tuple(weights))


# ------------------------------------------------------------- exact rational helpers (A2a/A3a)


def _exact_chart(value, chart_range) -> Fraction:
    """Exact affine image of a chart parameter in [0, 1] on ``chart_range``."""
    start, end = (Fraction(item) for item in chart_range)
    return start + Fraction(value) * (end - start)


def _knot_interval(low, high, chart_range) -> tuple[Fraction, Fraction]:
    """Knot interval covering the chart interval [low, high] (either order) under exact and float maps.

    ``_map`` evaluates ``(1 - t) a + t b`` in floats with an absolute error below
    ``4 u max(|a|, |b|) < 4 ulp(max(|a|, |b|))``; the exact image is widened by that
    and clipped to the range.
    """
    values = [_real(item, "chart box bound") for item in (low, high)]
    if not all(0.0 <= item <= 1.0 for item in values):
        raise GeometryError("chart box must lie inside [0, 1]^2")
    s0, s1 = sorted(_exact_chart(item, chart_range) for item in values)
    slack = 4 * Fraction(math.ulp(max(abs(chart_range[0]), abs(chart_range[1]))))
    lower, upper = (Fraction(item) for item in sorted(chart_range))
    return max(lower, s0 - slack), min(upper, s1 + slack)


def _exact_span(p: int, knots: list, count: int, s: Fraction) -> int:
    """Span as ``reference.find_spans``: ``knots[i] <= s < knots[i+1]``, the domain end in the last non-empty span."""
    if s >= knots[count]:
        span = count - 1
        while span > p and knots[span] == knots[count]:
            span -= 1
        return span
    span = p
    while span < count - 1 and knots[span + 1] <= s:
        span += 1
    return span


def _exact_basis(p: int, knots: list, span: int, s: Fraction) -> list:
    """The non-zero basis values ``N_{span-p..span}(s)`` in exact arithmetic (A2.2)."""
    values = [Fraction(1)] + [Fraction(0)] * p
    left, right = [Fraction(0)] * (p + 1), [Fraction(0)] * (p + 1)
    for j in range(1, p + 1):
        left[j] = s - knots[span + 1 - j]
        right[j] = knots[span + j] - s
        saved = Fraction(0)
        for r in range(j):
            temp = values[r] / (right[r + 1] + left[j - r])
            values[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        values[j] = saved
    return values


def _exact_insert(p: int, knots: list, points: list, value: Fraction, times: int) -> tuple[list, list]:
    """Insert ``value`` (strictly inside the domain) ``times`` times (A5.1); points are exact vectors."""
    for _ in range(times):
        k = _exact_span(p, knots, len(points), value)
        new = points[: k - p + 1]
        for i in range(k - p + 1, k + 1):
            alpha = (value - knots[i]) / (knots[i + p] - knots[i])
            new.append(tuple(alpha * a + (1 - alpha) * b for a, b in zip(points[i], points[i - 1])))
        new += points[k:]
        knots = knots[: k + 1] + [value] + knots[k + 1:]
        points = new
    return knots, points


def _exact_refine(p: int, knots: list, points: list, s0: Fraction, s1: Fraction) -> tuple[list, list, list]:
    """Insert the interval ends to multiplicity ``p``; return the knots, points and the active indices.

    Active: the basis functions not identically zero on the interval (open
    interval; for ``s0 == s1`` those not zero at the point, a superset). Their
    controls bound the curve on the interval whatever the multiplicities.
    """
    for value in sorted({s0, s1}):
        if knots[p] < value < knots[len(points)]:
            times = p - knots.count(value)
            if times > 0:
                knots, points = _exact_insert(p, knots, points, value, times)
    count = len(points)
    if s0 < s1:
        active = [i for i in range(count) if knots[i] < s1 and knots[i + p + 1] > s0]
    else:
        active = [i for i in range(count) if knots[i] <= s0 <= knots[i + p + 1]]
    return knots, points, active


def _sqrt_up(value: Fraction) -> Fraction:
    """An exact rational upper bound of ``sqrt(value)`` (relative excess below ``2**-60``)."""
    if value == 0:
        return Fraction(0)
    shift = max(0, 64 - (value.numerator.bit_length() - value.denominator.bit_length()) // 2)
    scaled = value * (1 << (2 * shift))
    whole = scaled.numerator // scaled.denominator
    root = math.isqrt(whole)
    exact_root = root * root == whole and scaled.denominator == 1
    return Fraction(root if exact_root else root + 1, 1 << shift)


def _sqrt_down(value: Fraction) -> Fraction:
    """An exact rational lower bound of ``sqrt(value)`` (relative deficit below ``2**-60``)."""
    if value == 0:
        return Fraction(0)
    shift = max(0, 64 - (value.numerator.bit_length() - value.denominator.bit_length()) // 2)
    scaled = value * (1 << (2 * shift))
    return Fraction(math.isqrt(scaled.numerator // scaled.denominator), 1 << shift)


_FLOAT_MAX = Fraction(np.finfo(float).max)


def _float_down(value: Fraction) -> float:
    """The largest double <= ``value`` (saturating to ``-inf`` / ``max``)."""
    if value < -_FLOAT_MAX:
        return -math.inf
    if value > _FLOAT_MAX:
        return float(np.finfo(float).max)
    rounded = float(value)
    return math.nextafter(rounded, -math.inf) if Fraction(rounded) > value else rounded


def _float_up(value: Fraction) -> float:
    """The smallest double >= ``value`` (saturating to ``inf`` / ``-max``)."""
    if value > _FLOAT_MAX:
        return math.inf
    if value < -_FLOAT_MAX:
        return -float(np.finfo(float).max)
    rounded = float(value)
    return math.nextafter(rounded, math.inf) if Fraction(rounded) < value else rounded


def _derivative_bound(p: int, knots: list, s0: Fraction, s1: Fraction, chart_range, active: list,
                      other: list, entry, diameter: Fraction) -> Fraction:
    """``p |scale| max_i max_j (w_ij |P_ij - P_{i-1,j}| + |w_ij - w_{i-1,j}| D) / (k_{i+p} - k_i)``.

    ``i`` runs over the degree ``p - 1`` basis functions not identically zero on
    the interval (for ``s0 == s1``: not zero at the point, a superset); both
    ``i - 1`` and ``i`` are then active degree-``p`` controls. Divide by ``w_min``.
    """
    present = set(active)
    worst = Fraction(0)
    for i in range(1, len(knots) - p):
        length = knots[i + p] - knots[i]
        if length <= 0:
            continue
        if s0 < s1:
            live = knots[i] < s1 and knots[i + p] > s0
        else:
            live = knots[i] <= s0 <= knots[i + p]
        if not live:
            continue
        if i not in present or i - 1 not in present:
            raise GeometryError("NURBS sub-patch derivative bound: inconsistent active controls")
        for j in other:
            (point, weight), (previous, previous_weight) = entry(i, j), entry(i - 1, j)
            step = _sqrt_up(sum(((a - b) ** 2 for a, b in zip(point, previous)), Fraction(0)))
            term = (weight * step + abs(weight - previous_weight) * diameter) / length
            worst = max(worst, term)
    scale = abs(Fraction(chart_range[1]) - Fraction(chart_range[0]))
    return p * scale * worst


def cylinder_to_nurbs(cylinder) -> "NurbsSurface":
    """The cylinder patch as a rational NURBS surface of degree (2, 1).

    u runs along a ``nurbs_circular_arc`` of the cylinder's start/sweep (at most
    four <= 90 degree segments), v along the axis over the height. The surface
    is the same point set to rounding of the controls, but the u chart is the
    rational arc parameter, not the angle (architecture 3.5: a NURBS copy is a
    different exact object and is never reclassified as the cylinder).
    """
    arc = nurbs_circular_arc(cylinder.origin, cylinder.radial_direction, cylinder.circumferential_direction,
                             cylinder.radius, cylinder.start_angle, cylinder.sweep_angle)
    lift = np.asarray(cylinder.axis, dtype=float) * float(cylinder.height)
    controls = tuple((point, tuple(map(float, np.asarray(point) + lift))) for point in arc.controls)
    weights = tuple((weight, weight) for weight in arc.weights)
    return NurbsSurface(2, 1, arc.knots, (0.0, 0.0, 1.0, 1.0), controls, weights)
