"""Exact level sets of NURBS pcurve coordinates (generality slice A1e-lite, owner option a).

For a 2D NURBS curve ``q(t)`` and a level ``c`` of one coordinate, the
crossings, touches and runs of ``q_axis(t) = c`` are decided in exact rational
arithmetic on the stored doubles; only the reported positions are rounded.

* The numerator ``f = sum N_i w_i (q_i - c)`` has the same sign as
  ``q_axis - c`` (positive weights). Exact knot insertion (A5.1) extracts its
  Bezier spans over the curve's domain; a span whose Bernstein coefficients are
  all zero is a run on the level.
* On every other span the distinct roots are isolated by a Sturm sequence of
  the square-free part (exact rational roots are deflated), then refined by
  exact bisection to ``2**-62`` of the span.
* Between consecutive zero events the sign of ``f`` is constant; it is
  evaluated exactly at a rational point of each gap. A zero event whose two
  sides have different signs is a crossing, otherwise a touch.

``level_set`` returns the ordered events of one curve; the containment parity
of ``nurbs_domain`` is built on them.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb

from .nurbs import NurbsCurve, _exact_insert

_WIDTH = Fraction(1, 2 ** 62)


# ------------------------------------------------------------------ exact polynomials (ascending power basis)


def _trim(p: list) -> list:
    while len(p) > 1 and p[-1] == 0:
        p = p[:-1]
    return p


def _value(p: list, x: Fraction) -> Fraction:
    result = Fraction(0)
    for coefficient in reversed(p):
        result = result * x + coefficient
    return result


def _derivative(p: list) -> list:
    return _trim([k * p[k] for k in range(1, len(p))] or [Fraction(0)])


def _divmod(a: list, b: list) -> tuple[list, list]:
    a, b = _trim(list(a)), _trim(list(b))
    if len(b) == 1 and b[0] == 0:
        raise ZeroDivisionError("polynomial division by zero")
    quotient = [Fraction(0)] * max(1, len(a) - len(b) + 1)
    remainder = list(a)
    while len(remainder) >= len(b) and not (len(remainder) == 1 and remainder[0] == 0):
        shift = len(remainder) - len(b)
        factor = remainder[-1] / b[-1]
        quotient[shift] = factor
        for k, coefficient in enumerate(b):
            remainder[shift + k] -= factor * coefficient
        remainder = _trim(remainder[:-1]) if len(remainder) > 1 else [Fraction(0)]
        if len(remainder) < len(b):
            break
    return _trim(quotient), _trim(remainder)


def _is_zero(p: list) -> bool:
    return all(coefficient == 0 for coefficient in p)


def _gcd(a: list, b: list) -> list:
    a, b = _trim(list(a)), _trim(list(b))
    while not _is_zero(b):
        a, b = b, _divmod(a, b)[1]
    return [coefficient / a[-1] for coefficient in a]


def _bernstein_to_power(b: list) -> list:
    p = len(b) - 1
    result = [Fraction(0)] * (p + 1)
    for k, coefficient in enumerate(b):
        if coefficient == 0:
            continue
        scale = coefficient * comb(p, k)
        for j in range(p - k + 1):
            result[k + j] += scale * comb(p - k, j) * (-1) ** j
    return _trim(result)


def _sturm(p: list) -> list[list]:
    sequence = [p, _derivative(p)]
    while not _is_zero(sequence[-1]) and len(sequence[-1]) > 1:
        remainder = _divmod(sequence[-2], sequence[-1])[1]
        if _is_zero(remainder):
            break
        sequence.append([-c for c in remainder])
    return sequence


def _variations(sequence: list[list], x: Fraction) -> int:
    signs = [v for v in (_value(p, x) for p in sequence) if v != 0]
    return sum(1 for a, b in zip(signs, signs[1:]) if (a > 0) != (b > 0))


def _roots_in_unit(p: list) -> list[tuple[Fraction, Fraction]]:
    """Distinct roots of a non-zero polynomial in [0, 1]: exact ``(r, r)`` or isolating ``(lo, hi)`` of width <= 2**-62."""
    p = _trim(p)
    if len(p) == 1:
        return []
    g = _divmod(p, _gcd(p, _derivative(p)))[0]          # square-free part: same distinct roots
    found: list[tuple[Fraction, Fraction]] = []
    for end in (Fraction(0), Fraction(1)):
        if _value(g, end) == 0:
            found.append((end, end))
            g = _divmod(g, [-end, Fraction(1)])[0]
    while len(g) > 1:
        sequence = _sturm(g)
        pending = [(Fraction(0), Fraction(1))]
        exact = None
        isolated: list[tuple[Fraction, Fraction]] = []
        while pending and exact is None:
            a, b = pending.pop()
            count = _variations(sequence, a) - _variations(sequence, b)
            if count == 0:
                continue
            if count == 1 and b - a <= _WIDTH:
                isolated.append((a, b))
                continue
            middle = (a + b) / 2
            if _value(g, middle) == 0:
                exact = middle
                break
            if count == 1:
                # one simple root, signs differ at the ends: bisect by sign
                low_sign = _value(g, a) > 0
                while b - a > _WIDTH:
                    middle = (a + b) / 2
                    value = _value(g, middle)
                    if value == 0:
                        exact = middle
                        break
                    if (value > 0) == low_sign:
                        a = middle
                    else:
                        b = middle
                if exact is not None:
                    break
                isolated.append((a, b))
                continue
            pending.extend(((middle, b), (a, middle)))
        if exact is not None:
            found.append((exact, exact))
            g = _divmod(g, [-exact, Fraction(1)])[0]
            continue
        found.extend(isolated)
        break
    return sorted(found)


# ------------------------------------------------------------------ level sets of a pcurve coordinate


@dataclass(frozen=True, slots=True)
class LevelEvent:
    """A zero stretch of ``q_axis(t) - c``: a point (``t0 == t1``) or a run, with the signs beside it.

    ``t0``/``t1`` are exact engine parameters (isolating-interval midpoints for
    irrational roots); ``before``/``after`` are the signs (-1, 0 or +1) of the
    coordinate minus the level just before and after it on this curve (0 at a
    curve end, where the neighbour decides).
    """
    t0: Fraction
    t1: Fraction
    before: int
    after: int

    @property
    def run(self) -> bool:
        return self.t1 > self.t0

    @property
    def crossing(self) -> bool:
        return self.before != 0 and self.after != 0 and self.before != self.after


def _homogeneous(curve: NurbsCurve, axis: int, level: Fraction) -> list[tuple[Fraction]]:
    weights = curve.weights if curve.weights is not None else (1.0,) * len(curve.controls)
    return [(Fraction(weight) * (Fraction(point[axis]) - level),) for point, weight in zip(curve.controls, weights)]


def _bezier_spans(curve: NurbsCurve, axis: int, level: Fraction):
    """Exact Bezier spans ``(s0, s1, coefficients)`` of the numerator over the curve's domain."""
    p = curve.degree
    knots = [Fraction(k) for k in curve.knots]
    points = _homogeneous(curve, axis, level)
    start, end = (Fraction(value) for value in curve.domain)
    low, high = min(start, end), max(start, end)
    breaks = sorted({k for k in knots[p + 1: len(points)]} | {low, high})
    for value in breaks:
        if knots[p] < value < knots[len(points)]:
            times = p - knots.count(value)
            if times > 0:
                knots, points = _exact_insert(p, knots, points, value, times)
    spans = []
    distinct = sorted(set(knots))
    for s0, s1 in zip(distinct, distinct[1:]):
        if s1 <= low or s0 >= high:
            continue
        first = knots.index(s0) + (knots.count(s0) - 1) - p      # control index of the span's first Bezier control
        coefficients = [points[first + k][0] for k in range(p + 1)]
        spans.append((s0, s1, coefficients))
    return spans


def _bernstein_at(coefficients: list, s: Fraction) -> Fraction:
    work = list(coefficients)
    for _ in range(len(work) - 1):
        work = [(1 - s) * a + s * b for a, b in zip(work, work[1:])]
    return work[0]


def level_set(curve: NurbsCurve, axis: int, value) -> tuple[LevelEvent, ...] | None:
    """Exact zero events of ``q_axis(t) - value`` in engine-parameter order; ``None`` if the whole curve is on it."""
    level = Fraction(value)
    spans = _bezier_spans(curve, axis, level)
    start, end = (Fraction(item) for item in curve.domain)

    def engine(s):
        return (s - start) / (end - start)

    # zero stretches in knot parameter: runs (zero spans) and isolated roots
    stretches: list[tuple[Fraction, Fraction]] = []
    for s0, s1, coefficients in spans:
        if all(c == 0 for c in coefficients):
            stretches.append((s0, s1))
            continue
        for lo, hi in _roots_in_unit(_bernstein_to_power(coefficients)):
            middle = (lo + hi) / 2
            stretches.append((s0 + (s1 - s0) * middle,) * 2)
    if not stretches:
        merged = []
    else:
        stretches.sort()
        merged = [list(stretches[0])]
        for a, b in stretches[1:]:
            if a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
    low, high = min(start, end), max(start, end)
    if merged and merged[0][0] <= low and merged[-1][1] >= high and len(merged) == 1:
        return None

    def sign_at(s):
        for s0, s1, coefficients in spans:
            if s0 <= s <= s1:
                value_at = _bernstein_at(coefficients, (s - s0) / (s1 - s0))
                return 0 if value_at == 0 else (1 if value_at > 0 else -1)
        return 0

    # signs in the gaps (knot order), then events in engine order
    points = [low] + [x for stretch in merged for x in stretch] + [high]
    gaps = []
    for a, b in zip(points[0::2], points[1::2]):
        gaps.append(sign_at((a + b) / 2) if b > a else 0)
    events = []
    for index, (a, b) in enumerate(merged):
        before, after = gaps[index], gaps[index + 1]
        if a <= low:
            before = 0
        if b >= high:
            after = 0
        events.append((a, b, before, after))
    if end > start:
        return tuple(LevelEvent(engine(a), engine(b), before, after) for a, b, before, after in events)
    # a reversed domain: engine order runs against the knot order
    return tuple(LevelEvent(engine(b), engine(a), after, before) for a, b, before, after in reversed(events))
