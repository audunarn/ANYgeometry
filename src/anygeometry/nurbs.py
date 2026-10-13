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

    def bounds(self, lower=0.0, upper=1.0):
        """Outward-rounded box containing the curve on ``[lower, upper]`` (control hulls of its spans)."""
        s = np.sort(self._knot_parameters(_pair(lower, upper))[0])
        if lower > upper:
            raise GeometryError("curve bound interval is reversed")
        rows = []
        for t0, t1, controls, _ in _reference.bezier_segments(
                self.degree, np.asarray(self.knots), np.asarray(self.controls), _scaled_weights(self.weights)):
            if t1 >= s[0] and t0 <= s[1]:
                rows.append(controls)
        if not rows or not np.all(np.isfinite(np.concatenate(rows))):
            raise GeometryError("NURBS bounds are not representable")
        points = np.concatenate(rows)
        # Positive weights keep each span inside the hull of its controls; the
        # decomposition rounds, so widen by its scale before rounding outward.
        margin = 64 * np.finfo(float).eps * np.max(np.abs(np.asarray(self.controls)), axis=0)
        return (np.nextafter(points.min(axis=0) - margin, -np.inf),
                np.nextafter(points.max(axis=0) + margin, np.inf))

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
