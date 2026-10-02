"""Exact intersection curves of a Bezier extrusion with a quadric, over a polynomial chart.

One real branch of ``s = (-B +- sqrt(B*B - 4*A*C)) / (2*A)`` (``A`` constant) over an interval of the directrix
parameter ``t``; see :mod:`anygeometry.branch_algebra`. It is the polynomial-chart counterpart of
:class:`~anygeometry.quadric_curves.QuadricIntersectionCurve` and follows its conventions exactly: charts never
contain a fold or double contact of the discriminant in their interior, a chart end that is a simple fold uses a
square (``left_square``/``right_square``) or sine (``both_sine``) reparametrization so the curve stays regular
there, and against a quadric whose rulings are parallel to the extrusion the equation is linear in ``s`` and there
is a single rational branch. The parameter ``t`` never wraps around, which makes the chart simpler than the
angular one: a point of the extrusion has exactly one ``t``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import math

import numpy as np

from .branch_algebra import BezierRuledSupport, directrix_of, get_poly_plan
from .errors import GeometryError
from .quadric_algebra import QuadricSupport
from .quadric_curves import (IDENTITY, MODES, _affine, _iadd, _imul, _iscale, _unit_parameters)

_EPS = float(np.finfo(float).eps)
_INF = float("inf")
_TOLERANCE = 1e-12                       # the chart tolerance in ``t`` (the parameter domain is the unit interval)


def _branch_root(a, b, c, radical, branch):
    """``(-b + branch * radical) / (2 a)`` for arrays, by the cancellation-free form where that is safe.

    With ``q = -(b + sign(b) radical) / 2`` one root is ``q / a`` and the other ``c / q``, exact in relative terms
    when ``4 a c`` is small against ``b^2``. But ``c / q`` assumes the discriminant that ``c`` implies, and a fold
    chart's radical is deliberately that of ``Delta(t) - Delta(anchor)``: where ``c`` and ``q`` both vanish (a fold
    on the base curve, ``b = 0``) the quotient is the ratio of two rounding residues. The direct form has an absolute
    error of a few ulp whatever the relation, so it decides: ``c / q`` is kept only where it agrees with it to that.
    """
    sign_b = np.where(b >= 0, 1.0, -1.0)
    q = -.5 * (b + sign_b * radical)
    direct = (-b + branch * radical) / (2 * a)
    with np.errstate(divide="ignore", invalid="ignore"):
        big = q / a
        other = np.where(q != 0, c / q, direct)
    tolerance = 16 * _EPS * (np.abs(b) + radical) / (2 * np.abs(a))
    safe = np.where(np.abs(other - direct) <= tolerance, other, direct)
    root = np.where((sign_b < 0) == (branch > 0), big, safe)                   # the root with the sign of -b is q / a
    return np.where(radical == 0.0, direct, root)                              # the fold itself: one point for both branches


@dataclass(frozen=True, slots=True)
class BezierQuadricCurve:
    """One real branch of a Bezier extrusion cut with a quadric, over a chart ``[start, start + sweep]`` of ``t``.

    ``branch`` selects ``-1`` or ``+1`` of the quadratic root (``+1`` only for a linear branch, where the rulings
    are parallel to an asymptotic direction of the quadric). ``transform`` keeps an exact affine image of the
    whole definition. Endpoint roundoff outside ``[0, 1]`` within the chart tolerance is canonicalized to the
    directrix boundary, so evaluation and interval bounds share the same domain.
    """
    first: BezierRuledSupport
    second: QuadricSupport
    start: float
    sweep: float
    branch: int = 1
    parameterization: str = "linear"
    transform: tuple = IDENTITY
    _inversions: dict = field(default_factory=dict, init=False, repr=False, compare=False, hash=False,
                              metadata={"definition": False})

    def __post_init__(self):
        if not isinstance(self.first, BezierRuledSupport):
            raise GeometryError("the first support of a Bezier quadric curve is a Bezier ruled support")
        object.__setattr__(self, "second", QuadricSupport.from_surface(self.second))
        try:
            start, sweep = float(self.start), float(self.sweep)
        except (TypeError, ValueError) as error:
            raise GeometryError("curve chart must be finite") from error
        if not math.isfinite(start) or not math.isfinite(sweep) or sweep == 0:
            raise GeometryError("curve chart must be finite with a nonzero extent")
        if min(start, start + sweep) < -_TOLERANCE or max(start, start + sweep) > 1 + _TOLERANCE:
            raise GeometryError("a curve chart lies inside the directrix parameter interval [0, 1]")
        end = start + sweep
        if not (0.0 <= start <= 1.0 and 0.0 <= end <= 1.0):
            start, end = min(1.0, max(0.0, start)), min(1.0, max(0.0, end))
            sweep = end - start
            if sweep == 0.0:
                raise GeometryError("curve chart must have a nonzero extent inside the directrix interval")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "sweep", sweep)
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
        """Build a chart already known to be valid (the support builder proves it by its event split)."""
        curve = object.__new__(cls)
        for name, value in (("first", first), ("second", second), ("start", float(start)), ("sweep", float(sweep)),
                            ("branch", branch), ("parameterization", mode), ("transform", transform),
                            ("_inversions", {})):
            object.__setattr__(curve, name, value)
        return curve

    def _derived(self, start, sweep, mode, transform=None):
        """A sub-chart of this validated curve (it inherits the parent's proof)."""
        curve = object.__new__(BezierQuadricCurve)
        for name, value in (("first", self.first), ("second", self.second), ("start", float(start)),
                            ("sweep", float(sweep)), ("branch", self.branch), ("parameterization", mode),
                            ("transform", self.transform if transform is None else transform), ("_inversions", {})):
            object.__setattr__(curve, name, value)
        return curve

    # ------------------------------------------------------------------ shared exact data

    def plan(self):
        return get_poly_plan(self.first, self.second)

    @property
    def linear(self):
        return self.plan().linear

    def _validate_chart(self):
        plan = self.plan()
        if plan.linear and self.branch != 1:
            raise GeometryError("a linear branch is a single rational branch (+1)")
        lo, hi = sorted((self.start, self.start + self.sweep))
        roots = plan.discriminant_roots()
        if any(lo + _TOLERANCE < t < hi - _TOLERANCE for t, _m in roots):
            raise GeometryError("intersection branch chart must split at every discriminant transition")
        if not plan.linear:
            if plan.floats().discriminant_scalar(float(self.start + .5 * self.sweep)) <= 0:
                raise GeometryError("intersection branch leaves the real quadric intersection")
        for end, t in ((0, self.start), (1, self.start + self.sweep)):
            simple_fold = any(m == 1 and abs(r - t) <= _TOLERANCE for r, m in roots)
            if self._is_fold_end(end) and not simple_fold:
                raise GeometryError("regular endpoint is not a discriminant transition")
            if simple_fold and not self._is_fold_end(end):
                raise GeometryError("a simple fold endpoint requires a square or sine parameterization")

    # ------------------------------------------------------------------ chart map

    def _is_fold_end(self, end):
        mode = self.parameterization
        return (mode in ("left_square", "both_sine") and end == 0) or (mode in ("right_square", "both_sine") and end == 1)

    def _chart(self, tau):
        """``(t, dt, anchor, delta, f, f', side)``; anchor is ``None`` in a linear chart."""
        start, sweep = self.start, self.sweep
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
        """Second derivative of ``t`` with respect to the chart parameter."""
        mode = self.parameterization
        sweep = self.sweep
        if mode == "linear":
            return np.zeros_like(tau)
        if mode == "both_sine":
            return .5 * math.pi ** 2 * sweep * np.cos(math.pi * tau)
        return np.full_like(tau, 2 * sweep * (1.0 if mode == "left_square" else -1.0))

    def parameter_for_t(self, t):
        """Invert the chart map: the parameter whose ``t`` is ``t`` (clamped into the chart)."""
        w = np.clip((np.asarray(t, dtype=float) - self.start) / self.sweep, 0.0, 1.0)
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

    def _difference(self, fs, anchor, u):
        """``h(u) = (Delta(anchor + E u) - Delta(anchor)) / (E u)`` with ``E = side * sweep`` at every chart anchor."""
        anchor = np.asarray(anchor, dtype=float)
        u = np.asarray(u, dtype=float)
        result = np.empty(u.shape)
        for value in np.unique(anchor):
            mask = anchor == value
            result[mask] = fs.h_form(float(value), self.sweep if value == self.start else -self.sweep).many(u[mask])
        return result

    def _radical(self, fs, t, anchor, delta, f, side):
        if anchor is None:
            return np.sqrt(np.maximum(fs.discriminant(t), 0.0))
        return f * np.sqrt(np.maximum(side * self.sweep * self._difference(fs, anchor, f * f), 0.0))

    def _root(self, fs, t, anchor, delta, f, side):
        a, b, c = fs.at(t)
        if self.plan().linear:
            with np.errstate(divide="ignore", invalid="ignore"):
                return -c / b, np.zeros_like(t)
        radical = self._radical(fs, t, anchor, delta, f, side)
        return _branch_root(a, b, c, radical, self.branch), radical            # at a fold both branches give one point

    def _evaluate_scalar(self, tau):
        """One point without numpy: the arrangement calls this tens of thousands of times."""
        plan = self.plan()
        fs = plan.floats()
        start, sweep = self.start, self.sweep
        mode = self.parameterization
        fold = mode != "linear"
        if not fold:
            t = start + sweep * tau
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
            t = anchor + delta
        a, b, c = fs.abc_scalar(t)
        if plan.linear:
            s = -c / b if b != 0.0 else _INF
        else:
            if fold:
                radical = f * math.sqrt(max(side * sweep * fs.h_form(anchor, side * sweep)(f * f), 0.0))
            else:
                radical = math.sqrt(max(fs.discriminant_scalar(t), 0.0))
            sign_b = 1.0 if b >= 0 else -1.0
            q = -.5 * (b + sign_b * radical)
            direct = (-b + self.branch * radical) / (2 * a)
            if radical == 0.0:
                s = direct                                 # the fold itself: one point for both branches
            elif sign_b * self.branch < 0:
                s = q / a                                  # the root with the sign of -b needs no quotient
            else:
                s = c / q if q != 0.0 else direct
                if abs(s - direct) > 16 * _EPS * (abs(b) + radical) / (2 * abs(a)):
                    s = direct                             # c / q is a ratio of residues where c and q both vanish
        point = fs.point_scalar(t, s)
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
        t, _dt, anchor, delta, f, _fp, side = self._chart(np.atleast_1d(tau))
        fs = self.plan().floats()
        s, _radical = self._root(fs, t, anchor, delta, f, side)
        points = fs.origin + fs.frame(t)[0] + s[..., None] * fs.direction
        if not self._identity:
            points = self._apply_transform(points)
        return points[0] if tau.ndim == 0 else points

    # ------------------------------------------------------------------ derivatives

    def derivative(self, parameters):
        tau = _unit_parameters(parameters)
        scalar = tau.ndim == 0
        tau = np.atleast_1d(tau)
        t, dt, anchor, delta, f, fp, side = self._chart(tau)
        fs = self.plan().floats()
        _P, Pd, _Pdd = fs.frame(t)
        s, radical = self._root(fs, t, anchor, delta, f, side)
        a, b, c = fs.at(t)
        _a1, b1, c1 = fs.at(t, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.plan().linear:
                ds = (-c1 * b + c * b1) * dt / (b * b)
            else:
                g1 = b1 * s + c1
                if anchor is None:
                    rho = dt / radical
                else:
                    rho = 2 * side * self.sweep * fp / np.sqrt(
                        np.maximum(side * self.sweep * self._difference(fs, anchor, f * f), 1e-300))
                ds = -self.branch * g1 * rho
        for index in range(len(tau)):
            if tau[index] in (0.0, 1.0):
                jet = self._end_jet(int(tau[index]))
                if jet is not None:
                    ds[index] = jet[0]
        result = Pd * dt[..., None] + ds[..., None] * fs.direction
        if not self._identity:
            result = result @ np.asarray(self.transform)[:3, :3].T
        return result[0] if scalar else result

    def second_derivative(self, parameters):
        """Analytic second derivative; exact one-sided jets at fold and double-contact ends."""
        tau = _unit_parameters(parameters)
        scalar = tau.ndim == 0
        tau = np.atleast_1d(tau)
        t, dt, anchor, delta, f, fp, side = self._chart(tau)
        ddt = self._chart_second(tau)
        fs = self.plan().floats()
        _P, Pd, Pdd = fs.frame(t)
        s, radical = self._root(fs, t, anchor, delta, f, side)
        a, b, c = fs.at(t)
        _a1, b1, c1 = fs.at(t, 1)
        _a2, b2, c2 = fs.at(t, 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.plan().linear:
                n = -c1 * b + c * b1
                ds = n * dt / (b * b)
                dds = ((-c2 * b + c * b2) * dt * dt + n * ddt) / (b * b) - 2 * n * b1 * dt * dt / b ** 3
            else:
                g1 = b1 * s + c1
                if anchor is None:
                    rho, inv = dt / radical, 1.0 / radical
                else:
                    rho = 2 * side * self.sweep * fp / np.sqrt(
                        np.maximum(side * self.sweep * self._difference(fs, anchor, f * f), 1e-300))
                    inv = rho / dt
                ds = -self.branch * g1 * rho
                numerator = (b2 * s + c2) * dt * dt + g1 * ddt + 2 * b1 * dt * ds + 2 * a * ds * ds
                dds = -self.branch * numerator * np.where(np.isfinite(inv), inv, 0.0)
        result = Pdd * (dt * dt)[..., None] + Pd * ddt[..., None] + dds[..., None] * fs.direction
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
        """``(s0, ds/dtau, d2s/dtau2, d2t/dtau2)`` at a fold end."""
        fs = self.plan().floats()
        anchor = float(self.start if end == 0 else self.start + self.sweep)
        sweep = self.sweep
        scale = .25 * math.pi ** 2 if self.parameterization == "both_sine" else 1.0
        kappa = (sweep if end == 0 else -sweep) * scale
        a0, b0, _c0 = fs.abc_scalar(anchor)
        _a1, b1, _c1 = fs.abc_scalar(anchor, 1)
        s0 = -b0 / (2 * a0)
        h0 = fs.discriminant_scalar(anchor, 1)                      # H(0) = Delta'(anchor)
        c1 = self.branch * math.sqrt(max(kappa * h0, 0.0)) / (2 * a0)
        c2 = -b1 * kappa / (2 * a0)
        direction = 1.0 if end == 0 else -1.0                      # d(end variable)/d(tau)
        return s0, direction * c1, 2 * c2, 2 * kappa

    def _node_jet(self, end):
        """``(s0, ds/dtau, d2s/dtau2)`` at a linear-chart end that is a double contact."""
        fs = self.plan().floats()
        anchor = float(self.start if end == 0 else self.start + self.sweep)
        (a0, b0, _), (a1, b1, _), (a2, b2, _), (a3, b3, c3) = (fs.abc_scalar(anchor, order) for order in range(4))
        s0 = -b0 / (2 * a0)
        eps = math.copysign(1.0, self.sweep) * (1 if end == 0 else -1)           # interior side in t
        h = max(.5 * fs.discriminant_scalar(anchor, 2), 0.0)
        slope = (-b1 + self.branch * eps * math.sqrt(h)) / (2 * a0) - s0 * a1 / a0
        g11, g21 = 2 * a1 * s0 + b1, (2 * a2 * s0 + b2) / 2
        g30 = (a3 * s0 * s0 + b3 * s0 + c3) / 6
        q = -(g30 + g21 * slope + a1 * slope * slope) / (2 * a0 * slope + g11)
        return s0, slope * self.sweep, 2 * q * self.sweep ** 2

    def _end_jet(self, end):
        """``(s', s'')`` at an end that needs a limit (fold or double contact), else ``None``."""
        if self._is_fold_end(end):
            _s0, s1, s2, _t2 = self._fold_jet(end)
            return s1, s2
        if self.parameterization == "linear" and not self.plan().linear:
            fs = self.plan().floats()
            anchor = float(self.start if end == 0 else self.start + self.sweep)
            double_contact = any(m == 2 and abs(r - anchor) <= _TOLERANCE
                                 for r, m in self.plan().discriminant_roots())
            if double_contact and math.sqrt(max(fs.discriminant_scalar(anchor), 0.0)) < 1e-7 * math.sqrt(fs.scale):
                _s0, s1, s2 = self._node_jet(end)
                return s1, s2
        return None

    def _end_second(self, end):
        """Second derivative of the position at a fold or double-contact end (before the transform)."""
        anchor = float(self.start if end == 0 else self.start + self.sweep)
        fs = self.plan().floats()
        _P, Pd, Pdd = fs.frame(np.asarray([anchor]))
        direction = fs.direction
        if self._is_fold_end(end):
            _s0, _s1, s2, t2 = self._fold_jet(end)
            return (Pd * t2 + s2 * direction)[0]
        _s0, s1, s2 = self._node_jet(end)
        sweep = self.sweep
        slope, q = s1 / sweep, s2 / (2 * sweep * sweep)
        return (sweep * sweep * (Pdd + 2 * q * direction))[0]

    # ------------------------------------------------------------------ slicing and copying

    def subcurve(self, lower, upper):
        lower, upper = (float(v) for v in _unit_parameters((lower, upper)))
        if lower == upper:
            raise GeometryError("curve interval must have positive length")
        ends = self._chart(np.asarray((lower, upper)))[0]
        low, high = min(lower, upper), max(lower, upper)
        mode = self.parameterization
        left = low == 0 and mode in ("left_square", "both_sine")
        right = high == 1 and mode in ("right_square", "both_sine")
        new = "both_sine" if left and right else "left_square" if left else "right_square" if right else "linear"
        if upper < lower:
            new = {"left_square": "right_square", "right_square": "left_square"}.get(new, new)
        return self._derived(float(ends[0]), float(ends[1] - ends[0]), new)

    def transformed(self, matrix):
        combined = _affine(matrix) @ np.asarray(self.transform)
        return self._derived(self.start, self.sweep, self.parameterization,
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
        return self._derived(self.start, self.sweep, self.parameterization, IDENTITY)

    def roots_on(self, other, *, tolerance=1e-10, cancellation_check=None):
        """Parameters where the curve lies on the quadric/surface ``other``; ``None`` if the whole chart does.

        The exact resultant of the two ruling quadratics gives every candidate ``t``; each is qualified by the
        implicit distance.
        """
        other = QuadricSupport.from_surface(other)
        if not self._identity:
            return self._reference_copy().roots_on(other.pulled_back(self.transform), tolerance=tolerance,
                                                   cancellation_check=cancellation_check)

        def distance(point):
            return abs(float(other.value(point))) / max(float(other.gradient_norm(point)), 1e-300)

        samples = self.evaluate(np.linspace(0., 1., 9))
        if all(distance(point) <= tolerance for point in samples):
            return None
        roots = self.plan().resultant_roots(other)
        lo, hi = sorted((self.start, self.start + self.sweep))
        if roots is None:
            return tuple(t for t in (0., 1.) if distance(self.evaluate(t)) <= tolerance)
        values = [t for t, _m in roots if lo - _TOLERANCE <= t <= hi + _TOLERANCE]
        parameters = []
        candidates = sorted(self._polish_root(other, float(x)) for x in np.atleast_1d(self.parameter_for_t(np.asarray(values)))) \
            if values else ()
        for tau in candidates:
            if parameters and tau - parameters[-1] <= 64 * _EPS:
                continue
            if distance(self.evaluate(tau)) <= tolerance:
                parameters.append(tau)
        return tuple(parameters)

    def _polish_root(self, other, tau, iterations=4):
        """Newton in the chart parameter on the implicit value of ``other``.

        Near a fold end ``t -> tau`` is a square root: a root a few ulp of ``t`` from the fold keeps only a few
        digits of ``tau``, which moves the point by far more than the tolerance although the curve is regular in
        ``tau``. The curve is evaluated accurately in ``tau``, so Newton restores the root.
        """
        def at(parameter):
            point = self._evaluate_scalar(parameter)
            return float(other.value(point)), float(other.gradient(point) @ self.derivative(parameter))

        value, slope = at(tau)
        for _ in range(iterations):
            if value == 0.0 or not math.isfinite(slope) or slope == 0.0:
                break
            trial = min(1.0, max(0.0, tau - value / slope))
            if trial == tau:
                break
            trial_value, trial_slope = at(trial)
            if not abs(trial_value) < abs(value):
                break
            tau, value, slope = trial, trial_value, trial_slope
        return tau

    def _polish_point(self, point, tau, iterations=4):
        """Gauss-Newton along the curve towards ``point``: the same repair for a point inversion."""
        error = self._evaluate_scalar(tau) - point
        size = float(np.linalg.norm(error))
        for _ in range(iterations):
            rate = self.derivative(tau)
            norm2 = float(rate @ rate)
            if size == 0.0 or not math.isfinite(norm2) or norm2 == 0.0:
                break
            trial = min(1.0, max(0.0, tau - float(error @ rate) / norm2))
            if trial == tau:
                break
            trial_error = self._evaluate_scalar(trial) - point
            trial_size = float(np.linalg.norm(trial_error))
            if not trial_size < size:
                break
            tau, error, size = trial, trial_error, trial_size
        return tau

    def parameters_of(self, point, *, tolerance=1e-10):
        """Every chart parameter whose curve point lies within ``tolerance`` of ``point`` (at most one).

        The arrangement asks for the same vertex again from every edge and trace that meets it, so answers are kept.
        """
        point = np.asarray(point, dtype=float)
        key = (point.tobytes(), tolerance)
        answer = self._inversions.get(key)
        if answer is None:
            answer = self._parameters_of(point, tolerance)
            if len(self._inversions) < 4096:
                self._inversions[key] = answer
        return answer

    def _parameters_of(self, point, tolerance):
        if not self._identity:
            inverse = np.linalg.inv(np.asarray(self.transform))
            return self._reference_copy().parameters_of(inverse[:3, :3] @ point + inverse[:3, 3], tolerance=tolerance)
        directrix = directrix_of(self.first)
        direction = np.asarray(self.first.direction)
        s = float((point - np.asarray(directrix.origin)) @ directrix.normal / float(directrix.normal @ direction))
        t = directrix.invert_one(point - s * direction)
        lo, hi = sorted((self.start, self.start + self.sweep))
        margin = max(1e-9, tolerance)
        if not lo - margin <= t <= hi + margin:
            return ()
        tau = float(self.parameter_for_t(t))
        for end in (0.0, 1.0):                       # prefer the exact end over a square-root amplified neighbour
            if abs(tau - end) <= 1e-6 and np.linalg.norm(self.evaluate(end) - point) <= tolerance:
                return (end,)
        tau = self._polish_point(point, tau)
        if np.linalg.norm(self.evaluate(tau) - point) > tolerance:
            return ()
        return (tau,)


# ---------------------------------------------------------------------------
# Interval enclosure (pure Python; cached per curve and interval)


def _piece_enclosure(curve, tau_lo, tau_hi):
    """Enclosure of one piece of the curve as ``(lo[3], hi[3])`` in the reference frame (may be infinite)."""
    plan = curve.plan()
    fs = plan.floats()
    start, sweep = curve.start, curve.sweep
    mode = curve.parameterization
    end = start + sweep
    t_a, t_b = (float(v) for v in curve._chart(np.asarray((tau_lo, tau_hi)))[0])
    t_lo, t_hi = min(t_a, t_b), max(t_a, t_b)
    a_iv = (fs.a0, fs.a0)
    b_iv = fs.B_forms[0].range(t_lo, t_hi)
    c_iv = fs.C_forms[0].range(t_lo, t_hi)
    if plan.linear:
        if b_iv[0] <= 0.0 <= b_iv[1]:
            s_iv = (-_INF, _INF)
        else:
            s_iv = _iscale(_imul(c_iv, (1.0 / b_iv[1], 1.0 / b_iv[0])), -1.0)
    else:
        if mode == "linear":
            d_iv = fs.D_forms[0].range(t_lo, t_hi)
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
                radicand = fs.g_form(anchor, side * sweep).range(f_iv[0] * f_iv[0], f_iv[1] * f_iv[1])
                piece = (math.sqrt(max(radicand[0], 0.0)), math.sqrt(max(radicand[1], 0.0)))
                r_iv = piece if r_iv is None else (min(r_iv[0], piece[0]), max(r_iv[1], piece[1]))
        branch = curve.branch
        candidates = []
        if not (a_iv[0] <= 0.0 <= a_iv[1]):
            inv_2a = (.5 / a_iv[1], .5 / a_iv[0])
            candidates.append(_imul(_iadd(_iscale(b_iv, -1.0), _iscale(r_iv, branch)), inv_2a))
        denominator = _iadd(_iscale(b_iv, -1.0), _iscale(r_iv, -branch))
        if not (denominator[0] <= 0.0 <= denominator[1]):
            c_implied = c_iv
            if mode != "linear":         # a fold radical is that of Delta - Delta(anchor): c + Delta(anchor) / (4 A) goes with it
                form = fs.D_forms[0]
                shifts = [form(anchor) / (4 * fs.a0) for anchor in
                          ((start, end) if mode == "both_sine" else (start,) if mode == "left_square" else (end,))]
                slack = 16 * (form.degree + 2) * _EPS * form.bound / (4 * abs(fs.a0))
                c_implied = _iadd(c_iv, (min(shifts) - slack, max(shifts) + slack))
            candidates.append(_imul(_iscale(c_implied, 2.0), (1.0 / denominator[1], 1.0 / denominator[0])))
        if not candidates:
            s_iv = (-_INF, _INF)
        else:
            s_iv = (max(c[0] for c in candidates), min(c[1] for c in candidates))
            if s_iv[0] > s_iv[1]:                                  # rounding made the two forms disjoint
                s_iv = (min(c[0] for c in candidates), max(c[1] for c in candidates))
    lo_box, hi_box = [], []
    for axis in range(3):
        p = fs.P_forms[axis][0].range(t_lo, t_hi)
        d = fs.direction_tuple[axis]
        coordinate = _iadd(p, _imul(s_iv, (d, d))) if math.isfinite(s_iv[0]) and math.isfinite(s_iv[1]) else (-_INF, _INF)
        lo_box.append(coordinate[0] + fs.origin_tuple[axis])
        hi_box.append(coordinate[1] + fs.origin_tuple[axis])
    return lo_box, hi_box


@lru_cache(maxsize=8192)
def _cached_bounds(curve, lower, upper):
    """Subdivide until pieces are narrow in ``t``, enclose each, and take the hull."""
    if lower == upper:
        point = curve.evaluate(lower)
        margin = 512 * _EPS * np.maximum(1.0, np.abs(point))
        lo_a, hi_a = np.nextafter(point - margin, -np.inf), np.nextafter(point + margin, np.inf)
    else:
        t_a, t_b = (float(v) for v in curve._chart(np.asarray((lower, upper)))[0])
        pieces = math.ceil(abs(t_b - t_a) / .0625)
        if curve.parameterization != "linear":                    # a square-root end needs narrow charts in tau as well
            pieces = max(pieces, math.ceil(abs(upper - lower) * 16))
        pieces = max(1, min(64, pieces))
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
