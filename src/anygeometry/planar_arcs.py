"""Exact containment for loops of lines and elliptic arcs on a plane (robust-predicate R1c, E4).

``MaterialDomain.contains_loop`` decides a point against a loop by horizontal
ray parity in the plane chart. For straight loops the rules run on exact
rationals (``material_arrangement._contains_loop_exact``). This module extends
that exact replica to ``EllipticArc`` paths. Every quantity is an exact
rational or a canonical enclosure (``exact_predicates.Canonical``) of an exact
real: the root angles of ``c0 + c1 cos t + c2 sin t`` (main's tan-half-angle
quadratic, so ``x = tan(t / 2)`` is a quadratic irrational), arc points, and
evaluations at main's midpoint parameters. Every decision of main's rules --
the coincidence test, the 4-eps turn window and clamping of trigonometric
roots, the 128-eps endpoint snapping, the world-distance boundary test, the
ray side, the 8-eps endpoint windows, the before/after signs and main's point
rule for coincident arcs -- is taken by ``certified_compare``. An exact tie
that canonical enclosures cannot separate raises ``DegenerateWithinBound`` (a
``GeometryError``): the query is refused rather than decided by rounding.
"""
from __future__ import annotations

import functools
from fractions import Fraction
import math

import numpy as np

from .arrangement_geometry import LinePath
from .errors import GeometryError
from .exact_curves import EllipticArc
from .exact_predicates import (L_MAX, Canonical, DegenerateWithinBound, Outcome,
                               certified_compare, exact, exact_dot, exact_vector)

_EPS = Fraction(float(np.finfo(float).eps))
_TRIG_TOLERANCE = 4 * _EPS                      # main's trigonometric_roots tolerance for arc roots
_TAU_TURNS = 3                                  # turns tried around the sweep (|sweep| <= 2 pi)


# ----------------------------------------------------------------------------------- exact-or-enclosed numbers


class Num:
    """An exact rational or a canonical enclosure ``f(engine) -> Box`` of an exact real."""

    __slots__ = ("exact", "f")

    def __init__(self, exact_value=None, f=None):
        self.exact, self.f = exact_value, f

    @staticmethod
    def of(value) -> "Num":
        return value if isinstance(value, Num) else Num(exact(value))

    def box(self, engine):
        """The canonical enclosure at ``engine``, memoized on that engine only.

        Each certified comparison evaluates on fresh engines; within one
        engine a shared node is evaluated once. The result and the budget
        accounting of a comparison therefore depend only on its expression
        (not on earlier comparisons), so refusals are reproducible.
        """
        if self.exact is not None:
            return engine.box(self.exact)
        memo = engine.__dict__.setdefault("_num_memo", {})
        box = memo.get(id(self))
        if box is None:
            box = memo[id(self)] = self.f(engine)
            engine.__dict__.setdefault("_num_alive", []).append(self)      # keeps ids unique while memoized
        return box

    def _combine(self, other, exact_op, name):
        other = Num.of(other)
        if self.exact is not None and other.exact is not None:
            return Num(exact_op(self.exact, other.exact))
        return Num(f=lambda engine, a=self, b=other: getattr(engine, name)(a.box(engine), b.box(engine)))

    def __add__(self, other):
        return self._combine(other, lambda a, b: a + b, "add")

    __radd__ = __add__

    def __sub__(self, other):
        return self._combine(other, lambda a, b: a - b, "sub")

    def __rsub__(self, other):
        return Num.of(other) - self

    def __mul__(self, other):
        return self._combine(other, lambda a, b: a * b, "mul")

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._combine(other, lambda a, b: a / b, "div")

    def __neg__(self):
        return Num(-self.exact) if self.exact is not None else Num(f=lambda e, a=self: e.neg(a.box(e)))


_PREDICATE = "planar arc containment"


def sign(value) -> int:
    """Exact sign, or the certified sign of an enclosed real (an undecidable zero is refused)."""
    value = Num.of(value)
    if value.exact is not None:
        return (value.exact > 0) - (value.exact < 0)
    outcome = certified_compare(_PREDICATE, value.f, 0)
    return {Outcome.BELOW: -1, Outcome.ABOVE: 1, Outcome.EQUAL: 0}[outcome]


def compare(a, b) -> int:
    return sign(Num.of(a) - Num.of(b))


def _dot(a, b) -> Num:
    total = Num(Fraction(0))
    for x, y in zip(a, b):
        total = total + Num.of(x) * Num.of(y)
    return total


def _pi() -> Num:
    return Num(f=lambda engine: engine.pi())


def _atan(x: Num) -> Num:
    if x.exact == 0:
        return Num(Fraction(0))
    return Num(f=lambda engine, x=x: engine.atan(x.box(engine)))


def _atan2(y: Num, x: Num) -> Num:
    return Num(f=lambda engine, y=y, x=x: engine.atan2(y.box(engine), x.box(engine)))


def _sqrt(value: Fraction) -> Num:
    if value >= 0:
        numerator, denominator = math.isqrt(value.numerator), math.isqrt(value.denominator)
        if numerator * numerator == value.numerator and denominator * denominator == value.denominator:
            return Num(Fraction(numerator, denominator))
    return Num(f=lambda engine, value=value: engine.sqrt(engine.box(value)))


# ----------------------------------------------------------------------------------- arc angles and points


class ArcAngle:
    """An angle ``theta`` (``Num``) with its exact cosine/sine where known.

    Root angles carry ``x = tan(theta / 2)`` (cos and sin are rational in
    ``x``); the angle ``pi`` and stored angles carry exact or enclosed values.
    """

    __slots__ = ("theta", "x", "cs")

    def __init__(self, theta: Num, x: Num | None = None, cs=None):
        self.theta, self.x, self.cs = theta, x, cs

    def cos_sin(self):
        if self.cs is not None:
            return self.cs
        if self.x is not None:
            x = self.x
            denominator = 1 + x * x
            return (1 - x * x) / denominator, (2 * x) / denominator
        if self.theta.exact == 0:
            return Num(Fraction(1)), Num(Fraction(0))
        theta = self.theta
        return (Num(f=lambda e, t=theta: e.sincos(t.box(e))[1]), Num(f=lambda e, t=theta: e.sincos(t.box(e))[0]))


def _pi_angle() -> ArcAngle:
    return ArcAngle(_pi(), None, (Num(Fraction(-1)), Num(Fraction(0))))


def arc_point(curve: EllipticArc, angle: ArcAngle) -> tuple:
    """World position ``center + U cos + V sin`` (three ``Num``)."""
    cos, sin = angle.cos_sin()
    center, u, v = (exact_vector(value) for value in (curve.center, curve.u_vector, curve.v_vector))
    return tuple(Num(c) + Num(a) * cos + Num(b) * sin for c, a, b in zip(center, u, v))


def _end_angle(curve: EllipticArc) -> Fraction:
    """The angle of ``evaluate(1.)``: the float sum ``start + sweep`` (main's evaluation)."""
    return exact(float(curve.start_angle) + float(curve.sweep_angle))


def angle_at(curve: EllipticArc, t: Num) -> ArcAngle:
    """The angle of ``evaluate(t)``: ``start + sweep t`` (endpoints as main evaluates them)."""
    if t.exact is not None and t.exact == 0:
        return ArcAngle(Num(exact(curve.start_angle)))
    if t.exact is not None and t.exact == 1:
        return ArcAngle(Num(_end_angle(curve)))
    return ArcAngle(Num(exact(curve.start_angle)) + Num(exact(curve.sweep_angle)) * t)


def first_harmonic_angles(c0: Fraction, c1: Fraction, c2: Fraction) -> list:
    """Exact replica of ``analytic_roots._first_harmonic_angles`` (angles in ``(-pi, pi]``)."""
    a, b, c = c0 - c1, 2 * c2, c0 + c1
    if a == 0 and b == 0 and c == 0:
        raise GeometryError("root isolation needs a nonzero polynomial and positive tolerance")
    if a == 0:
        angles = [_pi_angle()]
        if b != 0:
            x = Num(-c / b)
            angles.append(ArcAngle(2 * _atan(x), x))
        return angles
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return []
    if discriminant == 0:
        x = Num(-b / (2 * a))
        return [ArcAngle(2 * _atan(x), x)]
    root = _sqrt(discriminant)
    angles = []
    for x in ((-Num(b) + root) / (2 * a), (-Num(b) - root) / (2 * a)):
        angles.append(ArcAngle(2 * _atan(x), x))
    return angles


class ArcRoot:
    """A root of the horizontal-ray constraint on an arc: parameter ``t`` and its angle."""

    __slots__ = ("t", "angle")

    def __init__(self, t: Num, angle: ArcAngle):
        self.t, self.angle = t, angle


def _turned(angle: ArcAngle, turn: int) -> ArcAngle:
    if turn == 0:
        return angle
    return ArcAngle(angle.theta + 2 * turn * _pi(), angle.x, angle.cs)


def _turn_range(lower: Fraction, upper: Fraction, tolerance: Fraction) -> range:
    """All turns ``k`` with ``lower - tol <= theta + 2 pi k <= upper + tol`` for some ``theta`` in ``[-pi, pi]``.

    From a rational enclosure of ``pi`` (canonical, finest level), widened by
    one turn each side; each candidate is then tested by certified
    comparisons. Refuses when the window is not a handful of turns (huge
    angles, where the enclosure of ``2 pi k`` cannot separate turns).
    """
    pi = Canonical(L_MAX).pi()
    # (bound -+ pi) / (2 pi) is monotone in pi: its extremes over the enclosure are at the ends.
    first = min((lower - tolerance - value) / (2 * value) for value in (pi.lo, pi.hi))
    last = max((upper + tolerance + value) / (2 * value) for value in (pi.lo, pi.hi))
    lo, hi = math.floor(first) - 1, math.ceil(last) + 1
    if hi - lo > 2 * _TAU_TURNS + 2:
        raise DegenerateWithinBound(_PREDICATE + ": turn window", None, Fraction(0))
    return range(lo, hi + 1)


def _sweep_roots(curve: EllipticArc, angles, tolerance: Fraction) -> list:
    """Main's ``trigonometric_roots`` turn window and clamp, then ``_angle_parameter``.

    A root angle is kept for every turn with ``lower - tol <= angle <= upper + tol``
    and clamped into ``[lower, upper]`` (``upper`` is the float end angle
    ``start + sweep``); ``_angle_parameter`` maps the stored end angles to
    0 and 1, and clamps a fraction within 128 eps outside ``[0, 1]`` (the float
    end angle can lie past the exact ``start + sweep``) -- the clamped parameter
    is evaluated at its endpoint, as main evaluates it.
    """
    start = exact(curve.start_angle)
    end = _end_angle(curve)
    lower, upper = min(start, end), max(start, end)
    sweep = exact(curve.sweep_angle)
    roots = []
    turns = _turn_range(lower, upper, tolerance)
    for angle in angles:
        for turn in turns:
            candidate = _turned(angle, turn)
            if compare(candidate.theta, lower - tolerance) < 0 or compare(candidate.theta, upper + tolerance) > 0:
                continue
            if compare(candidate.theta, lower) < 0:
                candidate = ArcAngle(Num(lower))
            elif compare(candidate.theta, upper) > 0:
                candidate = ArcAngle(Num(upper))
            if candidate.theta.exact is not None and candidate.theta.exact == start:
                roots.append(ArcRoot(Num(Fraction(0)), ArcAngle(Num(start))))
                continue
            if candidate.theta.exact is not None and candidate.theta.exact == end:
                roots.append(ArcRoot(Num(Fraction(1)), ArcAngle(Num(end))))
                continue
            fraction = (candidate.theta - start) / sweep
            window = 128 * _EPS
            if compare(fraction, -window) >= 0 and compare(fraction, 1 + window) <= 0:
                if compare(fraction, 0) < 0:
                    roots.append(ArcRoot(Num(Fraction(0)), ArcAngle(Num(start))))
                    continue
                if compare(fraction, 1) > 0:
                    roots.append(ArcRoot(Num(Fraction(1)), ArcAngle(Num(end))))
                    continue
            roots.append(ArcRoot(fraction, candidate))
    return roots


def arc_horizontal_roots(chart, curve: EllipticArc, level: Fraction, tolerance: float):
    """Exact replica of ``MaterialDomain.horizontal_roots`` for an ``EllipticArc`` on a Plane.

    ``None`` when the arc lies on the chart level (main's coincidence test of
    the scalar constraint), else the roots (``ArcRoot``) after main's 128-eps
    snapping to endpoints whose scalar residual is within tolerance.
    """
    normal = tuple(value / chart._scale for value in chart._dual_v)
    offset = level + exact_dot(normal, chart.origin)
    bound2 = Fraction(tolerance) ** 2 * exact_dot(normal, normal)
    c0 = exact_dot(normal, exact_vector(curve.center)) - offset
    c1 = exact_dot(normal, exact_vector(curve.u_vector))
    c2 = exact_dot(normal, exact_vector(curve.v_vector))
    if (abs(c0) + abs(c1) + abs(c2)) ** 2 <= bound2:
        return None
    roots = _sweep_roots(curve, first_harmonic_angles(c0, c1, c2), _TRIG_TOLERANCE)
    endpoints = []
    for t in (Fraction(0), Fraction(1)):
        residual = _dot(normal, arc_point(curve, angle_at(curve, Num(t)))) - offset
        if compare(residual * residual, bound2) <= 0:
            endpoints.append(t)
    result = []
    for root in roots:
        for t in endpoints:
            if compare(abs_num(root.t - t), 128 * _EPS) <= 0:
                root = ArcRoot(Num(t), angle_at(curve, Num(t)))
                break
        if not any(root.t.exact is not None and other.t.exact == root.t.exact for other in result):
            result.append(root)
    return result


def abs_num(value: Num) -> Num:
    return value if sign(value) >= 0 else -value


def arc_point_rule(curve: EllipticArc, point, tolerance: float) -> bool:
    """Exact replica of ``point_parameters(EllipticArc, point, tolerance=...)`` being non-empty.

    The public wrapper first accepts an endpoint within ``tol`` (certified
    distance to the evaluated endpoint), then ``_point_parameters``' ellipse
    inversion.
    """
    query = exact_vector(point)
    tol2 = Fraction(tolerance) ** 2
    for t in (Fraction(0), Fraction(1)):
        delta = tuple(a - Num(b) for a, b in zip(arc_point(curve, angle_at(curve, Num(t))), query))
        if compare(_dot(delta, delta), tol2) <= 0:
            return True
    center, u, v = (exact_vector(value) for value in (curve.center, curve.u_vector, curve.v_vector))
    offset = tuple(p - c for p, c in zip(exact_vector(point), center))
    uu, uv, vv = exact_dot(u, u), exact_dot(u, v), exact_dot(v, v)
    determinant = uu * vv - uv * uv
    if determinant == 0:
        raise GeometryError("degenerate arc frame")
    ru, rv = exact_dot(u, offset), exact_dot(v, offset)
    a, b = (vv * ru - uv * rv) / determinant, (uu * rv - uv * ru) / determinant
    q = a * a + b * b - 1
    tol = Fraction(tolerance)
    norm2 = uu + vv                                              # Frobenius norm of [U V], squared
    if (q * q * norm2 > tol * tol) if norm2 >= tol * tol else (q * q > 1):
        return False
    residual = tuple(a * x + b * y - z for x, y, z in zip(u, v, offset))
    if exact_dot(residual, residual) > tol * tol:
        return False
    angle = ArcAngle(_atan2(Num(b), Num(a)))
    return bool(_sweep_roots(curve, [angle], tol))


# ----------------------------------------------------------------------------------- containment


def contains_loop_exact_curved(chart, loop, uv, tolerance: float, *, boundary: bool = True) -> bool:
    """Exact replica of ``MaterialDomain.contains_loop`` for loops of lines and elliptic arcs on a Plane."""
    from .material_arrangement import _exact_horizontal_roots, _exact_point_on_segment

    u_query, v_query = uv
    tol2 = Fraction(tolerance) ** 2
    window = 8 * _EPS
    plane_u, plane_v = chart.u_vector, chart.v_vector

    def roots_of(curve):
        if type(curve) is LinePath:
            line_roots = _exact_horizontal_roots(chart, curve, v_query, tolerance)
            return None if line_roots is None else [ArcRoot(Num(t), None) for t in line_roots]
        return arc_horizontal_roots(chart, curve, v_query, tolerance)

    def chart_uv(point):
        offset = tuple(Num.of(p) - Num(o) for p, o in zip(point, chart.origin))
        return _dot(offset, chart._dual_u) / chart._scale, _dot(offset, chart._dual_v) / chart._scale

    def station_uv(curve, root: ArcRoot):
        if type(curve) is LinePath:
            t = root.t.exact
            start, end = exact_vector(curve.start), exact_vector(curve.end)
            return tuple(Num(value) for value in chart.uv(tuple(a + t * (b - a) for a, b in zip(start, end))))
        return chart_uv(arc_point(curve, root.angle))

    def at_parameter(curve, t: Num) -> ArcRoot:
        return ArcRoot(t, None if type(curve) is LinePath else angle_at(curve, t))

    def ordered(curve, roots):
        # main: sorted(set(min(1, max(0, t)))) and evaluation at the clamped parameter, so a clamped
        # root is evaluated at its endpoint (its angle follows the parameter).
        unique = []
        for root in roots:
            clamped = _clamp(root.t)
            if clamped is not root.t:
                root = at_parameter(curve, clamped)
            if not any(other.t.exact is not None and other.t.exact == root.t.exact for other in unique):
                unique.append(root)
        return sorted(unique, key=functools.cmp_to_key(lambda a, b: compare(a.t, b.t)))

    winding = 0
    for index, path in enumerate(loop):
        curve = path.curve
        roots = roots_of(curve)
        if roots is None:
            world = chart.world(u_query, v_query)
            hit = (_exact_point_on_segment(curve, world, tolerance) if type(curve) is LinePath
                   else arc_point_rule(curve, world, tolerance))
            if hit:
                return boundary
            continue
        roots = ordered(curve, roots)
        for position, root in enumerate(roots):
            point_u, point_v = station_uv(curve, root)
            du, dv = point_u - u_query, point_v - v_query
            delta = tuple(du * a + dv * b for a, b in zip(plane_u, plane_v))
            if compare(_dot(delta, delta), tol2) <= 0:
                return boundary
            if compare(point_u, u_query) <= 0:
                continue
            if compare(root.t, 1 - window) >= 0:
                # main (T2-0): the next curve's start root owns the vertex event; when it reports
                # none, the side change into its first branch is counted here, exactly once.
                following = loop[(index + 1) % len(loop)].curve
                following_roots = roots_of(following)
                if following_roots is None:
                    continue
                following_roots = ordered(following, following_roots)
                if following_roots and compare(following_roots[0].t, window) <= 0:
                    continue
                first = following_roots[0].t if following_roots else Num(Fraction(1))
                after = station_uv(following, at_parameter(following, first / 2))[1] - v_query
                previous_root = roots[position - 1].t if position else Num(Fraction(0))
                before = station_uv(curve, at_parameter(curve, (previous_root + root.t) / 2))[1] - v_query
                before_sign, after_sign = sign(before), sign(after)
                if before_sign <= 0 < after_sign:
                    winding += 1
                elif after_sign <= 0 < before_sign:
                    winding -= 1
                continue
            if compare(root.t, window) <= 0:
                previous = loop[index - 1].curve
                previous_roots = roots_of(previous)
                for distance in range(2, len(loop) + 1):
                    if previous_roots is not None:
                        break
                    previous = loop[index - distance].curve
                    previous_roots = roots_of(previous)
                below = [r.t for r in (previous_roots or ()) if compare(r.t, 1 - window) < 0]
                previous_lower = functools.reduce(lambda a, b: a if compare(a, b) >= 0 else b, below,
                                                  Num(Fraction(0))) if below else Num(Fraction(0))
                before = station_uv(previous, at_parameter(previous, (previous_lower + 1) / 2))[1] - v_query
            else:
                previous_root = roots[position - 1].t if position else Num(Fraction(0))
                before = station_uv(curve, at_parameter(curve, (previous_root + root.t) / 2))[1] - v_query
            next_root = roots[position + 1].t if position + 1 < len(roots) else Num(Fraction(1))
            after = station_uv(curve, at_parameter(curve, (root.t + next_root) / 2))[1] - v_query
            before_sign, after_sign = sign(before), sign(after)
            if before_sign <= 0 < after_sign:
                winding += 1
            elif after_sign <= 0 < before_sign:
                winding -= 1
    return winding != 0


def _clamp(t: Num) -> Num:
    """``min(1, max(0, t))``; returns ``t`` itself when no clamping applies."""
    if t.exact is not None:
        return t if 0 <= t.exact <= 1 else Num(min(Fraction(1), max(Fraction(0), t.exact)))
    if compare(t, 0) < 0:
        return Num(Fraction(0))
    if compare(t, 1) > 0:
        return Num(Fraction(1))
    return t


__all__ = ["ArcAngle", "ArcRoot", "Num", "arc_horizontal_roots", "arc_point", "arc_point_rule",
           "contains_loop_exact_curved", "first_harmonic_angles"]
