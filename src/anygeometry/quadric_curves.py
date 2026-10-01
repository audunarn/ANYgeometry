"""Exact intersection curves of a ruled support (Cylinder or Cone) with a quadric.

One real branch of ``s = (-B +- sqrt(B*B - 4*A*C)) / (2*A)`` over an angular chart
``[start, start + sweep]``; see :mod:`anygeometry.quadric_algebra`. Charts never
contain a fold or double contact of the discriminant in their interior. A chart
end that is a simple fold uses a square (``left_square``/``right_square``) or sine
(``both_sine``) reparametrization so the curve stays regular there, exactly as
:class:`~anygeometry.exact_curves.CylinderIntersectionCurve` does. Against a plane
the equation is linear in ``s`` and there is a single rational branch.

Everything that depends only on the support pair (exact events, rounded
coefficients) lives in a shared :class:`~anygeometry.quadric_algebra.Plan`; a curve
holds only its chart, so slicing a curve with ``subcurve`` is cheap.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

from .errors import GeometryError
from .quadric_algebra import (EllipticRuledSupport, QuadricSupport, RuledSupport, TWO_PI, circle_angles, get_plan,
                              ruled_support)

IDENTITY = ((1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.))
MODES = ("linear", "left_square", "right_square", "both_sine")
_EPS = float(np.finfo(float).eps)
_INF = float("inf")


def _unit_parameters(value):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise GeometryError("curve parameters must be finite and in [0, 1]") from error
    if not np.all(np.isfinite(result)) or np.any((result < 0) | (result > 1)):
        raise GeometryError("curve parameters must be finite and in [0, 1]")
    return result


def _affine(value):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise GeometryError("curve transformation must be a finite 4x4 matrix")
    if not np.array_equal(matrix[3], (0, 0, 0, 1)) or np.linalg.det(matrix[:3, :3]) == 0:
        raise GeometryError("curve transformation must be nonsingular and affine")
    return matrix


def _quadratic_root(a, b, c, radical, branch):
    """``(-b + branch*radical)/(2a)`` without cancellation (and finite where ``a`` vanishes)."""
    sign_b = np.where(b >= 0, 1.0, -1.0)
    q = -.5 * (b + sign_b * radical)
    with np.errstate(divide="ignore", invalid="ignore"):
        big = q / a
        small = np.where(q != 0, c / q, 0.0)
    plus = np.where(sign_b > 0, small, big)
    minus = np.where(sign_b > 0, big, small)
    return plus if branch > 0 else minus


@dataclass(frozen=True, slots=True)
class QuadricIntersectionCurve:
    """One real branch of a ruled support cut with a quadric, over an angular chart.

    ``first`` supplies the angle about its axis. ``branch`` selects ``-1`` or ``+1``
    of the quadratic root (``+1`` only against a plane). ``transform`` keeps an exact
    affine image of the whole definition.
    """
    first: RuledSupport
    second: QuadricSupport
    start_angle: float
    sweep_angle: float
    branch: int = 1
    parameterization: str = "linear"
    transform: tuple = IDENTITY

    def __post_init__(self):
        first = ruled_support(self.first)
        second = QuadricSupport.from_surface(self.second)
        object.__setattr__(self, "first", first)
        object.__setattr__(self, "second", second)
        try:
            start, sweep = float(self.start_angle), float(self.sweep_angle)
        except (TypeError, ValueError) as error:
            raise GeometryError("curve angles must be finite") from error
        if not math.isfinite(start) or not math.isfinite(sweep) or sweep == 0:
            raise GeometryError("curve angles must be finite with a nonzero sweep")
        if abs(sweep) > TWO_PI + 16 * _EPS:
            raise GeometryError("one curve interval cannot exceed a full turn")
        object.__setattr__(self, "start_angle", start)
        object.__setattr__(self, "sweep_angle", sweep)
        if type(self.branch) is not int or self.branch not in (-1, 1):
            raise GeometryError("intersection branch must be -1 or 1")
        if self.parameterization not in MODES:
            raise GeometryError("invalid intersection branch parameterization")
        matrix = _affine(self.transform)
        object.__setattr__(self, "transform", tuple(tuple(float(v) for v in row) for row in matrix))
        self._validate_chart()

    # ------------------------------------------------------------------ construction without revalidation

    @classmethod
    def _make(cls, first, second, start, sweep, branch, mode, transform=IDENTITY):
        """Build a chart already known to be valid (support builders prove it by their event split)."""
        curve = object.__new__(cls)
        for name, value in (("first", first), ("second", second), ("start_angle", float(start)),
                            ("sweep_angle", float(sweep)), ("branch", branch), ("parameterization", mode),
                            ("transform", transform)):
            object.__setattr__(curve, name, value)
        return curve

    def _derived(self, start, sweep, mode, transform=None):
        """A sub-chart of this validated curve (never re-qualified: it inherits the parent's proof)."""
        curve = object.__new__(QuadricIntersectionCurve)
        for name, value in (("first", self.first), ("second", self.second), ("start_angle", float(start)),
                            ("sweep_angle", float(sweep)), ("branch", self.branch),
                            ("parameterization", mode),
                            ("transform", self.transform if transform is None else transform)):
            object.__setattr__(curve, name, value)
        return curve

    # ------------------------------------------------------------------ shared exact data

    def plan(self):
        return get_plan(self.first, self.second)

    @property
    def linear(self):
        return self.plan().linear

    def _angle_tolerance(self):
        return 1e-12 * max(1.0, self.first.radius, abs(self.first.slope))

    def _validate_chart(self):
        plan = self.plan()
        if plan.linear and self.branch != 1:
            raise GeometryError("a plane intersection has a single branch (+1)")
        lo, hi = sorted((self.start_angle, self.start_angle + self.sweep_angle))
        tolerance = self._angle_tolerance()
        roots = plan.discriminant_roots()
        if any(lo + tolerance < a < hi - tolerance for a, _m in circle_angles(roots, lo, hi, tolerance)):
            raise GeometryError("intersection branch chart must split at every discriminant transition")
        if not plan.linear:
            fs = plan.floats()
            if fs.discriminant_scalar(float(self.start_angle + .5 * self.sweep_angle)) <= 0:
                raise GeometryError("intersection branch leaves the real quadric intersection")
        for end, angle in ((0, self.start_angle), (1, self.start_angle + self.sweep_angle)):
            if not self._is_fold_end(end):
                continue
            if not any(m == 1 and abs(((a - angle + math.pi) % TWO_PI) - math.pi) <= tolerance for a, m in roots):
                raise GeometryError("regular endpoint is not a discriminant transition")

    # ------------------------------------------------------------------ chart map

    def _is_fold_end(self, end):
        mode = self.parameterization
        return (mode in ("left_square", "both_sine") and end == 0) or (mode in ("right_square", "both_sine") and end == 1)

    def _chart(self, tau):
        """``(theta, dtheta, anchor, delta, f, f', side)``; anchor is ``None`` in a linear chart."""
        start, sweep = self.start_angle, self.sweep_angle
        mode = self.parameterization
        if mode == "linear":
            return start + sweep * tau, np.full_like(tau, sweep), None, None, None, None, None
        end = start + sweep
        if mode == "left_square":
            f, fp, side = tau, np.ones_like(tau), 1.0
            anchor = np.full_like(tau, start)
        elif mode == "right_square":
            f, fp, side = 1.0 - tau, -np.ones_like(tau), -1.0
            anchor = np.full_like(tau, end)
        else:
            left = tau <= .5
            half = .5 * math.pi * tau
            f = np.where(left, np.sin(half), np.cos(half))
            fp = np.where(left, .5 * math.pi * np.cos(half), -.5 * math.pi * np.sin(half))
            side = np.where(left, 1.0, -1.0)
            anchor = np.where(left, start, end)
        delta = side * sweep * f * f
        return anchor + delta, 2 * side * sweep * f * fp, anchor, delta, f, fp, side

    def _chart_second(self, tau):
        """Second derivative of the angle with respect to the chart parameter."""
        mode = self.parameterization
        sweep = self.sweep_angle
        if mode == "linear":
            return np.zeros_like(tau)
        if mode == "both_sine":
            return .5 * math.pi ** 2 * sweep * np.cos(math.pi * tau)
        return np.full_like(tau, 2 * sweep * (1.0 if mode == "left_square" else -1.0))

    def parameter_for_angle(self, angle):
        """Invert the chart map: the parameter whose angle is ``angle`` (clamped into the chart)."""
        w = np.clip((np.asarray(angle, dtype=float) - self.start_angle) / self.sweep_angle, 0.0, 1.0)
        mode = self.parameterization
        if mode == "left_square":
            return np.sqrt(w)
        if mode == "right_square":
            return 1.0 - np.sqrt(1.0 - w)
        if mode == "both_sine":
            return np.where(w <= .5, 2 / math.pi * np.arcsin(np.sqrt(w)),
                            1.0 - 2 / math.pi * np.arcsin(np.sqrt(1.0 - w)))
        return w

    # ------------------------------------------------------------------ evaluation

    @property
    def _identity(self):
        return self.transform == IDENTITY

    def _apply_transform(self, points):
        matrix = np.asarray(self.transform)
        return points @ matrix[:3, :3].T + matrix[:3, 3]

    def _frame(self, theta):
        """Support rows at ``theta`` and their first two angular derivatives."""
        fs = self.plan().floats()
        c, s = np.cos(theta)[..., None], np.sin(theta)[..., None]
        p0, p1, p2 = (np.asarray(row) for row in fs.p_rows)
        d0, d1, d2 = (np.asarray(row) for row in fs.d_rows)
        return (fs, p0 + c * p1 + s * p2, d0 + c * d1 + s * d2, -s * p1 + c * p2, -s * d1 + c * d2,
                -c * p1 - s * p2, -c * d1 - s * d2)

    def _radical(self, fs, theta, anchor, delta, f, side):
        if anchor is None:
            return np.sqrt(np.maximum(fs.discriminant(theta), 0.0))
        return f * np.sqrt(np.maximum(side * self.sweep_angle * fs.difference(anchor, delta), 0.0))

    def _root(self, fs, theta, anchor, delta, f, side):
        a, b, c = fs.at(theta)
        if self.plan().linear:
            with np.errstate(divide="ignore", invalid="ignore"):
                return -c / b, np.zeros_like(theta)
        radical = self._radical(fs, theta, anchor, delta, f, side)
        return _quadratic_root(a, b, c, radical, self.branch), radical

    def _evaluate_scalar(self, tau):
        """One point without numpy: the arrangement calls this tens of thousands of times."""
        plan = self.plan()
        fs = plan.floats()
        start, sweep = self.start_angle, self.sweep_angle
        mode = self.parameterization
        fold = mode != "linear"
        if not fold:
            theta = start + sweep * tau
        else:
            end = start + sweep
            if mode == "left_square":
                f, side, anchor = tau, 1.0, start
            elif mode == "right_square":
                f, side, anchor = 1.0 - tau, -1.0, end
            elif tau <= .5:
                f, side, anchor = math.sin(.5 * math.pi * tau), 1.0, start
            else:
                f, side, anchor = math.cos(.5 * math.pi * tau), -1.0, end
            delta = side * sweep * f * f
            theta = anchor + delta
        a, b, c = fs.abc_scalar(theta)
        if plan.linear:
            s = -c / b if b != 0.0 else _INF
        else:
            if fold:
                radical = f * math.sqrt(max(side * sweep * fs.difference_scalar(anchor, delta), 0.0))
            else:
                radical = math.sqrt(max(fs.discriminant_scalar(theta), 0.0))
            sign_b = 1.0 if b >= 0 else -1.0
            q = -.5 * (b + sign_b * radical)
            big = q / a if a != 0.0 else (_INF if q >= 0 else -_INF)
            small = c / q if q != 0.0 else 0.0
            plus, minus = (small, big) if sign_b > 0 else (big, small)
            s = plus if self.branch > 0 else minus
        ct, st = math.cos(theta), math.sin(theta)
        (p0, p1, p2), (d0, d1, d2) = fs.p_rows, fs.d_rows
        o = fs.origin
        point = [o[i] + p0[i] + ct * p1[i] + st * p2[i] + s * (d0[i] + ct * d1[i] + st * d2[i]) for i in range(3)]
        if not self._identity:
            m = self.transform
            point = [m[i][0] * point[0] + m[i][1] * point[1] + m[i][2] * point[2] + m[i][3] for i in range(3)]
        return np.array(point)

    def evaluate(self, parameters):
        if isinstance(parameters, (float, int, np.floating, np.integer)) and not isinstance(parameters, bool):
            tau = float(parameters)
            if not 0.0 <= tau <= 1.0:
                raise GeometryError("curve parameters must be finite and in [0, 1]")
            return self._evaluate_scalar(tau)
        tau = _unit_parameters(parameters)
        theta, _dth, anchor, delta, f, _fp, side = self._chart(np.atleast_1d(tau))
        fs, P, D, *_rest = self._frame(theta)
        s, _radical = self._root(fs, theta, anchor, delta, f, side)
        points = fs.origin + P + s[..., None] * D
        if not self._identity:
            points = self._apply_transform(points)
        return points[0] if tau.ndim == 0 else points

    # ------------------------------------------------------------------ derivatives

    def derivative(self, parameters):
        tau = _unit_parameters(parameters)
        scalar = tau.ndim == 0
        tau = np.atleast_1d(tau)
        theta, dth, anchor, delta, f, fp, side = self._chart(tau)
        fs, P, D, Pd, Dd, _Pdd, _Ddd = self._frame(theta)
        s, radical = self._root(fs, theta, anchor, delta, f, side)
        a, b, c = fs.at(theta)
        a1, b1, c1 = fs.at(theta, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.plan().linear:
                ds = (-c1 * b + c * b1) * dth / (b * b)
            else:
                g1 = a1 * s * s + b1 * s + c1
                if anchor is None:
                    rho = dth / radical
                else:
                    rho = 2 * side * self.sweep_angle * fp / np.sqrt(
                        np.maximum(side * self.sweep_angle * fs.difference(anchor, delta), 1e-300))
                ds = -self.branch * g1 * rho
        for index in range(len(tau)):
            if tau[index] in (0.0, 1.0):
                jet = self._end_jet(int(tau[index]))
                if jet is not None:
                    ds[index] = jet[0]
        result = (Pd + s[..., None] * Dd) * dth[..., None] + ds[..., None] * D
        if not self._identity:
            result = result @ np.asarray(self.transform)[:3, :3].T
        return result[0] if scalar else result

    def second_derivative(self, parameters):
        """Analytic second derivative; exact one-sided jets at fold and double-contact ends."""
        tau = _unit_parameters(parameters)
        scalar = tau.ndim == 0
        tau = np.atleast_1d(tau)
        theta, dth, anchor, delta, f, fp, side = self._chart(tau)
        ddth = self._chart_second(tau)
        fs, P, D, Pd, Dd, Pdd, Ddd = self._frame(theta)
        s, radical = self._root(fs, theta, anchor, delta, f, side)
        a, b, c = fs.at(theta)
        a1, b1, c1 = fs.at(theta, 1)
        a2, b2, c2 = fs.at(theta, 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.plan().linear:
                n = -c1 * b + c * b1
                ds = n * dth / (b * b)
                dds = ((-c2 * b + c * b2) * dth * dth + n * ddth) / (b * b) - 2 * n * b1 * dth * dth / b ** 3
            else:
                g1 = a1 * s * s + b1 * s + c1
                if anchor is None:
                    rho, inv = dth / radical, 1.0 / radical
                else:
                    rho = 2 * side * self.sweep_angle * fp / np.sqrt(
                        np.maximum(side * self.sweep_angle * fs.difference(anchor, delta), 1e-300))
                    inv = rho / dth
                ds = -self.branch * g1 * rho
                numerator = ((a2 * s * s + b2 * s + c2) * dth * dth + g1 * ddth
                             + 2 * (2 * a1 * s + b1) * dth * ds + 2 * a * ds * ds)
                dds = -self.branch * numerator * np.where(np.isfinite(inv), inv, 0.0)
        result = ((Pdd + s[..., None] * Ddd) * (dth * dth)[..., None] + (Pd + s[..., None] * Dd) * ddth[..., None]
                  + 2 * (ds * dth)[..., None] * Dd + dds[..., None] * D)
        for index in range(len(tau)):
            if tau[index] in (0.0, 1.0):
                end = int(tau[index])
                if self._end_jet(end) is not None:
                    result[index] = self._end_second(end)
        if not self._identity:
            result = result @ np.asarray(self.transform)[:3, :3].T
        return result[0] if scalar else result

    # -- jets at fold and double-contact ends

    def _fold_jet(self, end):
        """``(s0, ds/dtau, d2s/dtau2, d2theta/dtau2)`` at a fold end."""
        fs = self.plan().floats()
        angle = float(self.start_angle if end == 0 else self.start_angle + self.sweep_angle)
        sweep = self.sweep_angle
        scale = .25 * math.pi ** 2 if self.parameterization == "both_sine" else 1.0
        kappa = (sweep if end == 0 else -sweep) * scale
        a0, b0, _c0 = fs.abc_scalar(angle)
        a1, b1, _c1 = fs.abc_scalar(angle, 1)
        s0 = -b0 / (2 * a0)
        h0 = fs.discriminant_scalar(angle, 1)                       # H(0) = Delta'(anchor)
        c1 = self.branch * math.sqrt(max(kappa * h0, 0.0)) / (2 * a0)
        c2 = -(2 * a1 * s0 + b1) * kappa / (2 * a0)
        direction = 1.0 if end == 0 else -1.0                      # d(end variable)/d(tau)
        return s0, direction * c1, 2 * c2, 2 * kappa

    def _node_jet(self, end):
        """``(s0, ds/dtau, d2s/dtau2)`` at a linear-chart end that is a double contact."""
        fs = self.plan().floats()
        angle = float(self.start_angle if end == 0 else self.start_angle + self.sweep_angle)
        (a0, b0, _), (a1, b1, _), (a2, b2, _), (a3, b3, c3) = (fs.abc_scalar(angle, order) for order in range(4))
        s0 = -b0 / (2 * a0)
        eps = math.copysign(1.0, self.sweep_angle) * (1 if end == 0 else -1)      # interior side in angle
        h = max(.5 * fs.discriminant_scalar(angle, 2), 0.0)
        slope = (-b1 + self.branch * eps * math.sqrt(h)) / (2 * a0) - s0 * a1 / a0
        g11, g21 = 2 * a1 * s0 + b1, (2 * a2 * s0 + b2) / 2
        g30 = (a3 * s0 * s0 + b3 * s0 + c3) / 6
        q = -(g30 + g21 * slope + a1 * slope * slope) / (2 * a0 * slope + g11)
        return s0, slope * self.sweep_angle, 2 * q * self.sweep_angle ** 2

    def _end_jet(self, end):
        """``(s', s'')`` at an end that needs a limit (fold or double contact), else ``None``."""
        if self._is_fold_end(end):
            _s0, s1, s2, _t2 = self._fold_jet(end)
            return s1, s2
        if self.parameterization == "linear" and not self.plan().linear:
            fs = self.plan().floats()
            angle = float(self.start_angle if end == 0 else self.start_angle + self.sweep_angle)
            if math.sqrt(max(fs.discriminant_scalar(angle), 0.0)) < 1e-7 * math.sqrt(fs.scale):
                _s0, s1, s2 = self._node_jet(end)
                return s1, s2
        return None

    def _end_second(self, end):
        """Second derivative of the position at a fold or double-contact end (before the transform)."""
        theta_end = float(self.start_angle if end == 0 else self.start_angle + self.sweep_angle)
        _fs, _P, D, Pd, Dd, Pdd, Ddd = self._frame(np.asarray([theta_end]))
        if self._is_fold_end(end):
            s0, _s1, s2, theta2 = self._fold_jet(end)
            return ((Pd + s0 * Dd) * theta2 + s2 * D)[0]
        s0, s1, s2 = self._node_jet(end)
        sweep = self.sweep_angle
        slope, q = s1 / sweep, s2 / (2 * sweep * sweep)
        return (sweep * sweep * ((Pdd + s0 * Ddd) + 2 * slope * Dd + 2 * q * D))[0]

    # ------------------------------------------------------------------ slicing and copying

    def subcurve(self, lower, upper):
        lower, upper = (float(v) for v in _unit_parameters((lower, upper)))
        if lower == upper:
            raise GeometryError("curve interval must have positive length")
        angles = self._chart(np.asarray((lower, upper)))[0]
        low, high = min(lower, upper), max(lower, upper)
        mode = self.parameterization
        left = low == 0 and mode in ("left_square", "both_sine")
        right = high == 1 and mode in ("right_square", "both_sine")
        new = "both_sine" if left and right else "left_square" if left else "right_square" if right else "linear"
        if upper < lower:
            new = {"left_square": "right_square", "right_square": "left_square"}.get(new, new)
        return self._derived(float(angles[0]), float(angles[1] - angles[0]), new)

    def transformed(self, matrix):
        combined = _affine(matrix) @ np.asarray(self.transform)
        return self._derived(self.start_angle, self.sweep_angle, self.parameterization,
                             tuple(tuple(float(v) for v in row) for row in combined))

    # ------------------------------------------------------------------ enclosure

    def bounds(self, lower=0.0, upper=1.0):
        """Outward axis-aligned enclosure of the curve between two parameters (interval arithmetic)."""
        lower, upper = (float(v) for v in _unit_parameters((lower, upper)))
        if lower > upper:
            raise GeometryError("curve bound interval is reversed")
        lo, hi = _cached_bounds(self, lower, upper)
        return lo.copy(), hi.copy()

    # ------------------------------------------------------------------ events and inversion

    def _reference_copy(self):
        """The same chart without the affine image (event solves work in the reference frame)."""
        return self._derived(self.start_angle, self.sweep_angle, self.parameterization, IDENTITY)

    def roots_on(self, other, *, tolerance=1e-10, cancellation_check=None):
        """Parameters where the curve lies on the quadric/surface ``other``; ``None`` if the whole chart does.

        The shared exact solve (resultant of the two ruling quadratics, Sturm-isolated)
        gives every candidate angle; each is qualified by the implicit distance.
        """
        other = QuadricSupport.from_surface(other)
        if not self._identity:
            return self._reference_copy().roots_on(other.pulled_back(self.transform), tolerance=tolerance,
                                                   cancellation_check=cancellation_check)

        def distance(point):
            return abs(float(other.value(point))) / max(float(other.gradient_norm(point)), 1e-300)

        samples = self.evaluate(np.linspace(0., 1., 9))
        if all(distance(point) <= tolerance for point in samples):
            return None                       # the support itself (or a rounding twin of it): the whole chart lies on it
        roots = self.plan().resultant_roots(other)
        lo, hi = sorted((self.start_angle, self.start_angle + self.sweep_angle))

        if roots is None:
            return tuple(t for t in (0., 1.) if distance(self.evaluate(t)) <= tolerance)
        angles = [a for a, _m in circle_angles(roots, lo, hi, self._angle_tolerance())]
        parameters = []
        for tau in sorted(float(t) for t in np.atleast_1d(self.parameter_for_angle(np.asarray(angles)))) if angles else ():
            if parameters and tau - parameters[-1] <= 64 * _EPS:
                continue
            if distance(self.evaluate(tau)) <= tolerance:
                parameters.append(tau)
        return tuple(parameters)

    def parameters_of(self, point, *, tolerance=1e-10):
        """Every chart parameter whose curve point lies within ``tolerance`` of ``point``."""
        point = np.asarray(point, dtype=float)
        if not self._identity:
            inverse = np.linalg.inv(np.asarray(self.transform))
            return self._reference_copy().parameters_of(inverse[:3, :3] @ point + inverse[:3, 3], tolerance=tolerance)
        theta0 = float(self.first.angle_of(point))
        lo, hi = sorted((self.start_angle, self.start_angle + self.sweep_angle))
        margin = max(1e-9, tolerance / max(1.0, self.first.radius, abs(self.first.slope)))
        result = []
        for turn in range(math.ceil((lo - theta0 - margin) / TWO_PI), math.floor((hi - theta0 + margin) / TWO_PI) + 1):
            tau = float(self.parameter_for_angle(theta0 + turn * TWO_PI))
            if np.linalg.norm(self.evaluate(tau) - point) > tolerance:
                continue
            for end in (0.0, 1.0):                   # prefer the exact end over a square-root amplified neighbour
                if abs(tau - end) <= 1e-6 and np.linalg.norm(self.evaluate(end) - point) <= tolerance:
                    tau = end
            if all(abs(tau - old) > 64 * _EPS for old in result):
                result.append(tau)
        return tuple(sorted(result))


# ---------------------------------------------------------------------------
# Interval enclosure (pure Python; cached per curve and interval)


def _trig_range(lo, hi):
    """Exact range of ``(cos, sin)`` over the angle interval ``[lo, hi]``, padded for rounding."""
    if hi - lo >= TWO_PI:
        return (-1.0, 1.0), (-1.0, 1.0)
    ca, cb, sa, sb = math.cos(lo), math.cos(hi), math.sin(lo), math.sin(hi)
    c_lo, c_hi, s_lo, s_hi = min(ca, cb), max(ca, cb), min(sa, sb), max(sa, sb)
    for m in range(math.ceil(lo / math.pi), math.floor(hi / math.pi) + 1):
        if m % 2 == 0:
            c_hi = 1.0
        else:
            c_lo = -1.0
    for m in range(math.ceil((lo - .5 * math.pi) / math.pi), math.floor((hi - .5 * math.pi) / math.pi) + 1):
        if m % 2 == 0:
            s_hi = 1.0
        else:
            s_lo = -1.0
    pad = 4 * _EPS
    return (max(c_lo - pad, -1.0), min(c_hi + pad, 1.0)), (max(s_lo - pad, -1.0), min(s_hi + pad, 1.0))


def _iadd(x, y):
    return (x[0] + y[0], x[1] + y[1])


def _iscale(x, c):
    return (c * x[0], c * x[1]) if c >= 0 else (c * x[1], c * x[0])


def _imul(x, y):
    products = (x[0] * y[0], x[0] * y[1], x[1] * y[0], x[1] * y[1])
    return (min(products), max(products))


def _fourier(coefficients, lo, hi):
    """Interval of ``c0 + sum a_k cos(k t) + b_k sin(k t)`` for ``t`` in ``[lo, hi]``."""
    total = (coefficients[0], coefficients[0])
    for k in range((len(coefficients) - 1) // 2):
        a_k, b_k = coefficients[2 * k + 1], coefficients[2 * k + 2]
        if a_k == 0.0 and b_k == 0.0:
            continue
        cos_r, sin_r = _trig_range((k + 1) * lo, (k + 1) * hi)
        total = _iadd(_iadd(total, _iscale(cos_r, a_k)), _iscale(sin_r, b_k))
    return total


def _sinc_range(lo, hi):
    if max(abs(lo), abs(hi)) > math.pi:
        return (-.2173, 1.0)
    low_arg, high_arg = (0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi))), max(abs(lo), abs(hi))

    def sinc(x):
        return 1.0 if x == 0 else math.sin(x) / x

    return (sinc(high_arg) - 4 * _EPS, min(sinc(low_arg) + 4 * _EPS, 1.0))


def _difference_interval(fs, anchor, d_lo, d_hi):
    """Interval of ``H(delta) = (Delta(anchor + delta) - Delta(anchor)) / delta`` for ``delta`` in ``[d_lo, d_hi]``."""
    coefficients = fs.disc[0]
    total = (0.0, 0.0)
    for k in range(1, 5):
        x_lo, x_hi = .5 * k * d_lo, .5 * k * d_hi
        sinc = _sinc_range(x_lo, x_hi)
        cos_r, sin_r = _trig_range(k * anchor + x_lo, k * anchor + x_hi)
        bracket = _iadd(_iscale(sin_r, -coefficients[2 * k - 1]), _iscale(cos_r, coefficients[2 * k]))
        total = _iadd(total, _iscale(_imul(sinc, bracket), k))
    return total


def _piece_enclosure(curve, tau_lo, tau_hi):
    """Enclosure of one piece of the curve as ``(lo[3], hi[3])`` in the reference frame (may be infinite)."""
    plan = curve.plan()
    fs = plan.floats()
    start, sweep = curve.start_angle, curve.sweep_angle
    mode = curve.parameterization
    end = start + sweep
    th_a, th_b = (float(v) for v in curve._chart(np.asarray((tau_lo, tau_hi)))[0])
    th_lo, th_hi = min(th_a, th_b), max(th_a, th_b)
    a_iv, b_iv, c_iv = (_fourier(coefficients, th_lo, th_hi) for coefficients in fs.abc[0])
    if plan.linear:
        if b_iv[0] <= 0.0 <= b_iv[1]:
            s_iv = (-_INF, _INF)
        else:
            s_iv = _iscale(_imul(c_iv, (1.0 / b_iv[1], 1.0 / b_iv[0])), -1.0)
    else:
        if mode == "linear":
            d_iv = _fourier(fs.disc[0], th_lo, th_hi)
            r_iv = (math.sqrt(max(d_iv[0], 0.0)), math.sqrt(max(d_iv[1], 0.0)))
        else:
            r_iv = None
            for lo, hi in ((tau_lo, min(tau_hi, .5)), (max(tau_lo, .5), tau_hi)) if mode == "both_sine" else ((tau_lo, tau_hi),):
                if lo > hi:
                    continue
                if mode == "left_square":
                    f_iv, side, anchor = (lo, hi), 1.0, start
                elif mode == "right_square":
                    f_iv, side, anchor = (1.0 - hi, 1.0 - lo), -1.0, end
                elif hi <= .5:
                    f_iv, side, anchor = (math.sin(.5 * math.pi * lo), math.sin(.5 * math.pi * hi)), 1.0, start
                else:
                    f_iv, side, anchor = (math.cos(.5 * math.pi * hi), math.cos(.5 * math.pi * lo)), -1.0, end
                f2 = (f_iv[0] * f_iv[0], f_iv[1] * f_iv[1])
                d_vals = sorted((side * sweep * f2[0], side * sweep * f2[1]))
                d_lo, d_hi = (min(0.0, d_vals[0]), max(0.0, d_vals[1])) if f_iv[0] <= 0 else (d_vals[0], d_vals[1])
                h_iv = _difference_interval(fs, anchor, d_lo, d_hi)
                inner = _iscale(h_iv, side * sweep)
                piece = _imul(f_iv, (math.sqrt(max(inner[0], 0.0)), math.sqrt(max(inner[1], 0.0))))
                r_iv = piece if r_iv is None else (min(r_iv[0], piece[0]), max(r_iv[1], piece[1]))
        branch = curve.branch
        candidates = []
        if not (a_iv[0] <= 0.0 <= a_iv[1]):
            inv_2a = (.5 / a_iv[1], .5 / a_iv[0])
            candidates.append(_imul(_iadd(_iscale(b_iv, -1.0), _iscale(r_iv, branch)), inv_2a))
        denominator = _iadd(_iscale(b_iv, -1.0), _iscale(r_iv, -branch))
        if not (denominator[0] <= 0.0 <= denominator[1]):
            candidates.append(_imul(_iscale(c_iv, 2.0), (1.0 / denominator[1], 1.0 / denominator[0])))
        if not candidates:
            s_iv = (-_INF, _INF)
        else:
            s_iv = (max(c[0] for c in candidates), min(c[1] for c in candidates))
            if s_iv[0] > s_iv[1]:                                  # rounding made the two forms disjoint
                s_iv = (min(c[0] for c in candidates), max(c[1] for c in candidates))
    cos_r, sin_r = _trig_range(th_lo, th_hi)
    lo_box, hi_box = [], []
    for i in range(3):
        p = _iadd(_iadd((fs.p_rows[0][i], fs.p_rows[0][i]), _iscale(cos_r, fs.p_rows[1][i])),
                  _iscale(sin_r, fs.p_rows[2][i]))
        d = _iadd(_iadd((fs.d_rows[0][i], fs.d_rows[0][i]), _iscale(cos_r, fs.d_rows[1][i])),
                  _iscale(sin_r, fs.d_rows[2][i]))
        coordinate = _iadd(p, _imul(s_iv, d)) if math.isfinite(s_iv[0]) and math.isfinite(s_iv[1]) else (-_INF, _INF)
        lo_box.append(coordinate[0] + fs.origin[i])
        hi_box.append(coordinate[1] + fs.origin[i])
    return lo_box, hi_box


@lru_cache(maxsize=8192)
def _cached_bounds(curve, lower, upper):
    """Subdivide until pieces are narrow in angle, enclose each, and take the hull."""
    if lower == upper:
        point = curve.evaluate(lower)
        margin = 512 * _EPS * np.maximum(1.0, np.abs(point))
        lo_a, hi_a = np.nextafter(point - margin, -np.inf), np.nextafter(point + margin, np.inf)
    else:
        theta_a, theta_b = (float(v) for v in curve._chart(np.asarray((lower, upper)))[0])
        pieces = max(1, min(64, math.ceil(abs(theta_b - theta_a) / .25)))
        edges = [lower + (upper - lower) * i / pieces for i in range(pieces + 1)]
        edges[-1] = upper
        lo, hi = [_INF] * 3, [-_INF] * 3
        for index in range(pieces):
            p_lo, p_hi = _piece_enclosure(curve, edges[index], edges[index + 1])
            lo = [min(a, b) for a, b in zip(lo, p_lo)]
            hi = [max(a, b) for a, b in zip(hi, p_hi)]
        lo_a, hi_a = np.asarray(lo), np.asarray(hi)
        if not curve._identity:
            matrix = np.asarray(curve.transform)
            center, half = .5 * (lo_a + hi_a), .5 * (hi_a - lo_a)
            center = matrix[:3, :3] @ center + matrix[:3, 3]
            half = np.abs(matrix[:3, :3]) @ half
            lo_a, hi_a = center - half, center + half
        margin = 512 * _EPS * np.maximum(1.0, np.maximum(np.abs(lo_a), np.abs(hi_a)))
        lo_a, hi_a = np.nextafter(lo_a - margin, -np.inf), np.nextafter(hi_a + margin, np.inf)
    lo_a.flags.writeable = False
    hi_a.flags.writeable = False
    return lo_a, hi_a
