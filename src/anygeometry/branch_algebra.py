"""Exact algebra of a Bezier extrusion cut with a quadric: the polynomial-chart counterpart of ``quadric_algebra``.

A Bezier curve ``c(t)`` of degree ``n`` swept along a constant direction ``d`` is the ruled surface
``c(t) + s d`` (``s`` in units of ``d``). Substituting it into a quadric ``Q`` gives

    ``A s^2 + B(t) s + C(t) = 0``

with a *constant* ``A = d.M.d``, ``B`` of degree ``n`` and ``C`` of degree ``2 n``, all exact rational
polynomials of the stored doubles. A branch is ``s = (-B +- sqrt(B^2 - 4 A C)) / (2 A)`` over a real interval of
``t`` (a single rational branch ``s = -C / B`` when ``A = 0``). Nothing wraps around and nothing has a pole
unless ``A = 0``, so the exact events are the real roots in ``[0, 1]`` of a handful of polynomials: the
discriminant (folds and double contacts), ``B`` (poles of the linear case), the common divisor of ``A``, ``B`` and
``C`` (whole generators on the quadric), ``A s0^2 + B s0 + C`` (the patch limits of the extrusion) and the
resultant with another quadric (the boundaries of the other patch).

Everything that depends only on the support pair lives in a shared :class:`PolyPlan`; a curve holds just its chart.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import math

import numpy as np

from .analytic_roots import _integer_row
from .errors import GeometryError
from .extrusions import BezierDirectrix
from .quadric_algebra import QuadricSupport, _chart_roots, _fraction_vec, _padd, _pmul
from .surfaces import ExtrudedSurface

_EPS = float(np.finfo(float).eps)


def _vec(value, name):
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise GeometryError(f"{name} must be a finite 3-vector") from error
    if len(result) != 3 or not all(math.isfinite(item) for item in result):
        raise GeometryError(f"{name} must be a finite 3-vector")
    return result


@dataclass(frozen=True, slots=True)
class BezierRuledSupport:
    """The ruled surface ``c(t) + s d`` of a Bezier curve with control points ``controls`` and constant direction ``d``.

    A hashable cache key: the patch ranges of an :class:`~anygeometry.surfaces.ExtrudedSurface` are deliberately
    absent, so every patch of one surface shares its plans.
    """
    controls: tuple
    direction: tuple

    def __post_init__(self):
        controls = tuple(_vec(point, "control point") for point in self.controls)
        if len(controls) < 3:
            raise GeometryError("a Bezier ruled support needs at least three control points")
        direction = _vec(self.direction, "ruling direction")
        if not any(direction):
            raise GeometryError("the ruling direction must be non-zero")
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "direction", direction)

    @classmethod
    def from_surface(cls, surface):
        directrix = getattr(surface, "directrix", None)
        if isinstance(surface, ExtrudedSurface) and isinstance(directrix, BezierDirectrix):
            return cls(directrix.controls, surface.vector)
        raise GeometryError("a Bezier ruled support is an extruded Bezier curve")

    @property
    def degree(self):
        return len(self.controls) - 1

    @property
    def origin(self):
        return self.controls[0]

    def exact_power(self):
        """The power-basis coefficients ``a_0 .. a_n`` of ``c(t) - c(0)`` as Fraction vectors (``a_0 = 0``)."""
        degree = self.degree
        points = [_fraction_vec(point) for point in self.controls]
        rows = [(Fraction(0),) * 3]
        for k in range(1, degree + 1):
            weight = math.comb(degree, k)
            rows.append(tuple(weight * sum((-1) ** (k - i) * math.comb(k, i) * points[i][axis]
                                           for i in range(k + 1)) for axis in range(3)))
        return rows

    def quadric(self):
        raise GeometryError("a Bezier extrusion of degree three or more is not a quadric")


# ---------------------------------------------------------------------------
# Exact coefficients


def _trim(poly):
    poly = list(poly)
    while len(poly) > 1 and poly[-1] == 0:
        poly.pop()
    return poly


def _is_zero(poly):
    return all(value == 0 for value in poly)


def build_abc(first, second):
    """``(A, B, C)`` as ascending Fraction polynomials in ``t``: ``Q(c(t) + s d) = A s^2 + B(t) s + C(t)``."""
    second = QuadricSupport.from_surface(second)
    m, l, c0 = second.exact_relative(first.origin)
    d = _fraction_vec(first.direction)
    a = first.exact_power()
    n = first.degree
    md = [sum(m[i][j] * d[j] for j in range(3)) for i in range(3)]
    lead = sum(d[i] * md[i] for i in range(3))
    b = [2 * sum(l[i] * d[i] for i in range(3))] + [2 * sum(md[i] * a[k][i] for i in range(3)) for k in range(1, n + 1)]
    ma = [[sum(m[i][j] * a[k][j] for j in range(3)) for i in range(3)] for k in range(n + 1)]
    c = [c0] + [Fraction(0)] * (2 * n)
    for i in range(1, n + 1):
        for k in range(1, n + 1):
            c[i + k] += sum(a[i][r] * ma[k][r] for r in range(3))
    for j in range(1, n + 1):
        c[j] += 2 * sum(l[r] * a[j][r] for r in range(3))
    return [lead], _trim(b), _trim(c)


def _resultant(A2, B2, C2, A3, B3, C3):
    """Common-root condition of two quadratics in ``s`` (a plane has ``A == 0``); zero means a shared ruling point."""
    zero2, zero3 = _is_zero(A2), _is_zero(A3)

    def mul(*factors):
        result = factors[0]
        for factor in factors[1:]:
            result = _pmul(result, factor)
        return result

    if zero2 and zero3:
        return _padd(mul(B2, C3), mul(B3, C2), -1)
    if zero3:
        return _padd(_padd(mul(A2, C3, C3), mul(B2, B3, C3), -1), mul(C2, B3, B3))
    if zero2:
        return _padd(_padd(mul(A3, C2, C2), mul(B3, B2, C2), -1), mul(C3, B2, B2))
    m = _padd(mul(A2, C3), mul(A3, C2), -1)
    ab = _padd(mul(A2, B3), mul(A3, B2), -1)
    bc = _padd(mul(B2, C3), mul(B3, C2), -1)
    return _padd(mul(m, m), mul(ab, bc), -1)


def poly_roots(poly, lo=0.0, hi=1.0):
    """``[(t, multiplicity)]`` for the real roots of an exact polynomial in ``[lo, hi]`` (``None`` for the zero polynomial)."""
    poly = _trim(poly)
    if _is_zero(poly):
        return None
    integers = list(_integer_row(poly))
    if len(integers) < 2:
        return []
    return _chart_roots(integers, interval=(Fraction(float(lo)), Fraction(float(hi))))


# ---------------------------------------------------------------------------
# The shared plan of one support pair


class PolyPlan:
    """Everything that depends on a (Bezier ruled support, quadric) pair but not on a patch (lazy, shared)."""

    __slots__ = ("first", "second", "A", "B", "C", "disc", "linear", "_disc", "_poles", "_common", "_bound",
                 "_resultant", "_float")

    def __init__(self, first, second):
        self.first, self.second = first, second
        self.A, self.B, self.C = build_abc(first, second)
        self.linear = _is_zero(self.A)
        self.disc = _trim(_padd(_pmul(self.B, self.B), _pmul([4 * self.A[0]], self.C), -1))
        self._disc = self._poles = self._common = self._float = None
        self._bound = {}
        self._resultant = {}

    @property
    def coincident(self):
        """Every ruling lies on the quadric: the two supports are one surface."""
        return _is_zero(self.A) and _is_zero(self.B) and _is_zero(self.C)

    def discriminant_roots(self):
        """``[(t, multiplicity)]`` of ``B^2 - 4 A C`` in ``[0, 1]`` (empty for a linear branch)."""
        if self._disc is None:
            if self.linear:
                self._disc = []
            else:
                roots = poly_roots(self.disc)
                if roots is None:
                    raise GeometryError("the surfaces touch along a whole curve (a double branch): unsupported")
                self._disc = roots
        return self._disc

    def pole_roots(self):
        """Roots of ``B`` where the linear branch ``s = -C / B`` leaves to infinity (none for a quadratic branch)."""
        if self._poles is None:
            roots = poly_roots(self.B) if self.linear else []
            self._poles = [] if roots is None else roots
        return self._poles

    def common_roots(self):
        """``[(t, 1)]``: the parameters whose whole ruling lies on the quadric (roots of ``gcd(A, B, C)``)."""
        if self._common is None:
            from .analytic_roots import _integer_gcd
            current = None
            for poly in (self.A, self.B, self.C):
                if _is_zero(poly):
                    continue
                row = _integer_row(_trim(poly))
                current = row if current is None else _integer_gcd(current, row)
            if current is None or len(current) < 2:
                self._common = []
            else:
                roots = _chart_roots(list(current), interval=(Fraction(0), Fraction(1)))
                self._common = [(t, 1) for t, _m in roots]
        return self._common

    def bound_roots(self, s0):
        """Parameters where the branch crosses the ruling station ``s0``: roots of ``A s0^2 + B s0 + C``."""
        key = float(s0)
        if key not in self._bound:
            s = Fraction(key)
            poly = _padd(_padd([self.A[0] * s * s], [value * s for value in self.B]), self.C)
            roots = poly_roots(poly)
            self._bound[key] = [] if roots is None else roots
        return self._bound[key]

    def resultant_roots(self, other):
        """Parameters where the branch can meet ``other`` (a quadric), or ``None`` when the resultant vanishes."""
        other = QuadricSupport.from_surface(other)
        if other not in self._resultant:
            A3, B3, C3 = build_abc(self.first, other)
            self._resultant[other] = poly_roots(_resultant(self.A, self.B, self.C, A3, B3, C3))
        return self._resultant[other]

    def floats(self):
        """The shared :class:`PolyFloatSystem` (built on first use)."""
        if self._float is None:
            self._float = PolyFloatSystem(self)
        return self._float


@lru_cache(maxsize=256)
def directrix_of(first):
    """The planar :class:`~anygeometry.extrusions.BezierDirectrix` of a ruled support (built once per curve)."""
    return BezierDirectrix(first.controls)


@lru_cache(maxsize=512)
def _cached_poly_plan(first, second):
    return PolyPlan(first, second)


def get_poly_plan(first, second):
    """The shared plan for a pair; ``second`` may be a surface or a quadric."""
    return _cached_poly_plan(first, QuadricSupport.from_surface(second))


# ---------------------------------------------------------------------------
# Floating-point evaluation
#
# The exact polynomials are evaluated in the Bernstein basis of [0, 1]: its values are convex combinations of the
# coefficients, so the error stays a few units of ``eps`` times the size of the coefficients whatever the degree,
# where the power basis loses about one digit per degree to cancellation (``C`` has degree ``2 n`` and reaches
# ``1e-13`` already for a quartic directrix, ``1e-10`` for a degree seven one).


def _exact_taylor(power, anchor):
    """Taylor coefficients ``p^(k)(anchor) / k!`` (``k = 0 .. n``) of exact ascending coefficients, exactly."""
    work = [Fraction(v) for v in power]
    taylor = []
    for _ in range(len(work)):
        carry = Fraction(0)
        for index in range(len(work) - 1, -1, -1):
            carry = carry * anchor + work[index]
            work[index] = carry
        taylor.append(work[0])
        work = work[1:] or [Fraction(0)]
    return taylor


def _casteljau(coefficients, t):
    """The Bernstein coefficients of a polynomial on ``[0, t]`` and on ``[t, 1]`` (convex combinations only)."""
    work = list(coefficients)
    left, right = [work[0]], [work[-1]]
    u = 1.0 - t
    for size in range(len(work) - 1, 0, -1):
        work = [u * work[i] + t * work[i + 1] for i in range(size)]
        left.append(work[0])
        right.append(work[-1])
    return left, right[::-1]


class BernsteinForm:
    """A polynomial of ``t`` on ``[0, 1]`` in the Bernstein basis, with correctly rounded coefficients.

    Built from exact ascending power coefficients (or exact Bernstein coefficients) and evaluated by the
    Bernstein-Horner scheme: on ``t <= 1/2`` the sum is a Horner polynomial in ``t / (1 - t)`` with the weights
    ``b_j C(m, j)``, on ``t > 1/2`` the same in ``(1 - t) / t`` with the weights reversed, so every term is
    non-negative-weighted and the rounding error is that of the Bernstein form. ``range`` encloses the values over
    a sub-interval from the Bernstein coefficients of the piece.
    """

    __slots__ = ("degree", "coeffs", "forward", "backward", "bound")

    def __init__(self, bernstein):
        bernstein = list(bernstein) or [Fraction(0)]
        m = len(bernstein) - 1
        self.degree = m
        self.coeffs = tuple(float(v) for v in bernstein)
        self.forward = tuple(float(bernstein[j] * math.comb(m, j)) for j in range(m + 1))
        self.backward = tuple(float(bernstein[m - j] * math.comb(m, j)) for j in range(m + 1))
        self.bound = max(abs(v) for v in self.coeffs)

    @classmethod
    def from_power(cls, power):
        power = [Fraction(v) for v in power] or [Fraction(0)]
        m = len(power) - 1
        return cls([sum(Fraction(math.comb(j, k), math.comb(m, k)) * power[k] for k in range(j + 1))
                    for j in range(m + 1)])

    def __call__(self, t):
        """The value at one float ``t`` (pure Python)."""
        m = self.degree
        if m == 0:
            return self.coeffs[0]
        if t <= .5:
            weights, ratio, scale = self.forward, t / (1.0 - t), (1.0 - t) ** m
        else:
            weights, ratio, scale = self.backward, (1.0 - t) / t, t ** m
        value = weights[m]
        for j in range(m - 1, -1, -1):
            value = value * ratio + weights[j]
        return value * scale

    def many(self, t):
        """The values at an array of ``t`` (any shape)."""
        t = np.asarray(t, dtype=float)
        m = self.degree
        if m == 0:
            return np.full(t.shape, self.coeffs[0])
        if t.size <= 4:                                         # numpy's per-call cost dwarfs four scalar evaluations
            return np.array([self(x) for x in t.ravel().tolist()]).reshape(t.shape)
        flat = t.ravel()
        result = np.empty(flat.shape)
        low = flat <= .5
        for mask, weights, high in ((low, self.forward, False), (~low, self.backward, True)):
            if not mask.any():
                continue
            x = flat[mask]
            other = 1.0 - x
            ratio = other / x if high else x / other
            value = np.full(x.shape, weights[m])
            for j in range(m - 1, -1, -1):
                value = value * ratio + weights[j]
            # NumPy's platform-specific array power can round differently from
            # Python's scalar libm power. Keep the public scalar/batch identity
            # while retaining the vectorized Horner recurrence.
            scale = np.fromiter((float(v) ** m for v in (x if high else other)),
                                dtype=float, count=x.size)
            result[mask] = value * scale
        return result.reshape(t.shape)

    def range(self, lo, hi):
        """An interval that contains the polynomial on ``[lo, hi]`` (a subset of ``[0, 1]``, outward padded)."""
        lo, hi = max(0.0, float(lo)), min(1.0, float(hi))
        coefficients = list(self.coeffs)
        m = self.degree
        if m > 0:
            if hi < 1.0:
                coefficients = _casteljau(coefficients, hi)[0]
            if lo > 0.0 and hi > 0.0:
                coefficients = _casteljau(coefficients, lo / hi)[1]
        pad = 16 * (m + 2) * _EPS * self.bound
        return min(coefficients) - pad, max(coefficients) + pad


def _bernstein_orders(power, orders=4):
    """``BernsteinForm`` of an exact polynomial and of its first ``orders - 1`` derivatives."""
    power = [Fraction(v) for v in power] or [Fraction(0)]
    forms = []
    for _ in range(orders):
        forms.append(BernsteinForm.from_power(power))
        power = [k * power[k] for k in range(1, len(power))] or [Fraction(0)]
    return tuple(forms)


def _control_orders(values, orders=3):
    """``BernsteinForm`` of a Bezier coordinate (exact Bernstein coefficients) and its derivatives."""
    values = [Fraction(v) for v in values]
    forms = []
    for _ in range(orders):
        forms.append(BernsteinForm(values))
        m = len(values) - 1
        values = [m * (values[j + 1] - values[j]) for j in range(m)] or [Fraction(0)]
    return tuple(forms)


class PolyFloatSystem:
    """Correctly rounded Bernstein forms of a plan and its derivatives, for evaluation.

    ``B_forms[i]``/``C_forms[i]``/``D_forms[i]`` are the ``i``-th derivative (``0 .. 3``) of ``B``, ``C`` and the
    discriminant; ``A`` is a constant. ``P_forms[axis][i]`` are the Bernstein forms of the directrix relative to the
    control point ``origin`` (``0 .. 2``); the ruling direction ``D`` is constant.
    """

    __slots__ = ("plan", "origin", "direction", "a0", "B_forms", "C_forms", "D_forms", "P_forms", "scale", "_hforms",
                 "_gforms", "origin_tuple", "direction_tuple")

    def __init__(self, plan):
        first = plan.first
        self.plan = plan
        self.origin = np.asarray(first.origin)
        self.direction = np.asarray(first.direction)
        self.a0 = float(plan.A[0])
        self.B_forms = _bernstein_orders(plan.B)
        self.C_forms = _bernstein_orders(plan.C)
        self.D_forms = _bernstein_orders(plan.disc)
        self.P_forms = tuple(_control_orders([Fraction(point[axis]) - Fraction(first.controls[0][axis])
                                              for point in first.controls]) for axis in range(3))
        self.scale = max([abs(float(v)) for poly in (plan.A, plan.B, plan.C) for v in poly] + [1e-300])
        self._hforms = {}
        self._gforms = {}
        self.origin_tuple = tuple(float(v) for v in first.origin)
        self.direction_tuple = tuple(float(v) for v in first.direction)

    # -- scalars (the arrangement calls these tens of thousands of times)

    def abc_scalar(self, t, order=0):
        return (self.a0 if order == 0 else 0.0), self.B_forms[order](t), self.C_forms[order](t)

    def discriminant_scalar(self, t, order=0):
        return self.D_forms[order](t)

    def h_form(self, anchor, extent):
        """The Bernstein form of ``h(u) = (Delta(anchor + extent u) - Delta(anchor)) / (extent u)`` on ``u`` in ``[0, 1]``.

        ``anchor`` is the end of a chart where the discriminant has a simple root (a fold) and ``extent`` the signed
        length of the chart from it. The radical of the chart is ``sqrt(Delta(t) - Delta(anchor)) = sqrt(extent u
        h(u))``, which vanishes at the anchor exactly instead of leaving the rounding residue of a root, and the
        coefficients come from an exact rational Taylor shift, so ``h`` keeps its relative accuracy all the way in.
        """
        key = (float(anchor), float(extent))
        form = self._hforms.get(key)
        if form is None:
            taylor = _exact_taylor(self.plan.disc, Fraction(key[0]))
            scale = Fraction(key[1])
            form = self._hforms[key] = BernsteinForm.from_power(
                [d * scale ** k for k, d in enumerate(taylor[1:])] or [Fraction(0)])
        return form

    def g_form(self, anchor, extent):
        """The Bernstein form of ``g(u) = Delta(anchor + extent u) - Delta(anchor)`` on ``u`` in ``[0, 1]`` (``g = extent u h``).

        The radicand of a fold chart as one polynomial: its range over a piece has none of the dependency a product
        ``f * sqrt(extent * h(u))`` of two ranges would carry.
        """
        key = (float(anchor), float(extent))
        form = self._gforms.get(key)
        if form is None:
            taylor = _exact_taylor(self.plan.disc, Fraction(key[0]))
            scale = Fraction(key[1])
            form = self._gforms[key] = BernsteinForm.from_power(
                [Fraction(0)] + [d * scale ** (k + 1) for k, d in enumerate(taylor[1:])])
        return form

    def point_scalar(self, t, s):
        """``origin + P(t) + s d`` for one ``t`` and ``s`` without numpy."""
        o, d = self.origin_tuple, self.direction_tuple
        return [o[axis] + self.P_forms[axis][0](t) + s * d[axis] for axis in range(3)]

    # -- vectorized

    def at(self, t, order=0):
        t = np.asarray(t, dtype=float)
        return (self.a0 if order == 0 else 0.0) + 0.0 * t, self.B_forms[order].many(t), self.C_forms[order].many(t)

    def discriminant(self, t, order=0):
        return self.D_forms[order].many(t)

    def frame(self, t):
        """``(P, P', P'')`` at ``t`` (any shape), relative to ``origin``."""
        t = np.asarray(t, dtype=float)
        return tuple(np.stack([self.P_forms[axis][order].many(t) for axis in range(3)], axis=-1) for order in range(3))
