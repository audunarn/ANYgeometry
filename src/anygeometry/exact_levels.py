"""Exact level sets of NURBS pcurve coordinates (generality slice A1e-lite, owner option a).

For a 2D NURBS curve ``q(t)`` and a level ``c`` of one coordinate, the
crossings, touches and runs of ``q_axis(t) = c`` are decided in exact rational
arithmetic on the stored doubles; only reported positions are rounded.

* The curve is cut into exact Bezier spans over its domain (exact knot
  insertion, A5.1, on the homogeneous controls; cached per curve). On a span
  the numerator ``f = sum B_k (W_k q_k - c W_k)`` has the sign of
  ``q_axis - c`` (positive weights). A span whose coefficients are all zero is a
  run on the level.
* On other spans the distinct roots are isolated by Sturm sequences of the
  square-free part (exact rational roots are deflated) and refined by exact
  bisection; every root keeps its exact isolating interval.
* Signs are evaluated exactly at points of proven root-free gaps (between
  isolating intervals, or at a curve end that is not a root). A zero event
  between gaps of different sign is a crossing, otherwise a touch.
* ``event_side`` decides the sign of another coordinate minus a value at a
  point event exactly (a common root is detected by a gcd, otherwise the
  isolating interval is refined until the other numerator has no root in it);
  ``has_root_in`` counts roots of a level on an exact parameter interval.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import functools
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


def _sign(value: Fraction) -> int:
    return 0 if value == 0 else (1 if value > 0 else -1)


def _derivative(p: list) -> list:
    return _trim([k * p[k] for k in range(1, len(p))] or [Fraction(0)])


def _is_zero(p: list) -> bool:
    return all(coefficient == 0 for coefficient in p)


def _divmod(a: list, b: list) -> tuple[list, list]:
    a, b = _trim(list(a)), _trim(list(b))
    if _is_zero(b):
        raise ZeroDivisionError("polynomial division by zero")
    if len(a) < len(b):
        return [Fraction(0)], a
    quotient = [Fraction(0)] * (len(a) - len(b) + 1)
    remainder = list(a)
    for shift in range(len(a) - len(b), -1, -1):
        factor = remainder[shift + len(b) - 1] / b[-1]
        quotient[shift] = factor
        if factor:
            for k, coefficient in enumerate(b):
                remainder[shift + k] -= factor * coefficient
    return _trim(quotient), _trim(remainder[: len(b) - 1] or [Fraction(0)])


def _gcd(a: list, b: list) -> list:
    a, b = _trim(list(a)), _trim(list(b))
    while not _is_zero(b):
        a, b = b, _divmod(a, b)[1]
    return [coefficient / a[-1] for coefficient in a]


def _square_free(p: list) -> list:
    p = _trim(p)
    if len(p) == 1:
        return p
    return _divmod(p, _gcd(p, _derivative(p)))[0]


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
    while len(sequence[-1]) > 1 or not _is_zero(sequence[-1]):
        remainder = _divmod(sequence[-2], sequence[-1])[1]
        if _is_zero(remainder):
            break
        sequence.append([-c for c in remainder])
        if len(sequence[-1]) == 1:
            break
    return sequence


def _variations(sequence: list[list], x: Fraction) -> int:
    signs = [v for v in (_value(p, x) for p in sequence) if v != 0]
    return sum(1 for a, b in zip(signs, signs[1:]) if (a > 0) != (b > 0))


def _count(g: list, a: Fraction, b: Fraction) -> int:
    """Distinct roots of a square-free ``g`` in the closed interval [a, b]."""
    if len(_trim(g)) == 1:
        return 0 if not _is_zero(g) else 1
    sequence = _sturm(g)
    count = _variations(sequence, a) - _variations(sequence, b)    # roots in (a, b]
    return count + (1 if _value(g, a) == 0 else 0)


def _roots_in_unit(p: list) -> list[tuple[Fraction, Fraction, list]]:
    """Distinct roots of a non-zero polynomial in [0, 1]: ``(lo, hi, g)`` with ``g`` square-free.

    ``lo == hi`` for an exact rational root; otherwise ``(lo, hi)`` isolates one
    simple root of ``g`` with ``g(lo) g(hi) < 0`` and width <= 2**-62.
    """
    g = _square_free(p)
    if len(g) == 1:
        return []
    found: list[tuple[Fraction, Fraction, list]] = []
    base = g
    for end in (Fraction(0), Fraction(1)):
        if _value(g, end) == 0:
            found.append((end, end, base))
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
            middle = (a + b) / 2
            if _value(g, middle) == 0:
                exact = middle
                break
            if count == 1:
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
            found.append((exact, exact, base))
            g = _divmod(g, [-exact, Fraction(1)])[0]
            continue
        exact_roots = [lo for lo, hi, _g in found if lo == hi]
        for a, b in isolated:
            # Keep the interval off every exact root of the original (they were
            # deflated from g): bisect toward g's own root until none lies in [a, b].
            low_sign = _value(g, a) > 0
            while any(a <= r <= b for r in exact_roots):
                middle = (a + b) / 2
                value = _value(g, middle)
                if value == 0:                             # cannot happen for a deflated simple root
                    raise ArithmeticError("isolation met an exact root of the deflated polynomial")
                if (value > 0) == low_sign:
                    a = middle
                else:
                    b = middle
            found.append((a, b, base))
        break
    return sorted(found, key=lambda item: item[0])


# ------------------------------------------------------------------ exact Bezier extraction (cached per curve)


@dataclass(frozen=True)
class _Span:
    s0: Fraction
    s1: Fraction
    points: tuple            # homogeneous (w q_0, w q_1, w) per Bezier control


@functools.lru_cache(maxsize=256)
def _extraction(curve: NurbsCurve) -> tuple[_Span, ...]:
    p = curve.degree
    knots = [Fraction(k) for k in curve.knots]
    weights = curve.weights if curve.weights is not None else (1.0,) * len(curve.controls)
    points = [tuple(Fraction(c) * Fraction(w) for c in point) + (Fraction(w),)
              for point, w in zip(curve.controls, weights)]
    start, end = (Fraction(value) for value in curve.domain)
    low, high = min(start, end), max(start, end)
    inside = {k for k in knots if low < k < high} | {low, high}       # only knots that cut the domain
    for value in sorted(inside):
        if knots[p] < value < knots[len(points)]:
            times = p - knots.count(value)
            if times > 0:
                knots, points = _exact_insert(p, knots, points, value, times)
    spans = []
    distinct = sorted(set(knots))
    for s0, s1 in zip(distinct, distinct[1:]):
        if s1 <= low or s0 >= high:
            continue
        first = knots.index(s0) + knots.count(s0) - 1 - p
        spans.append(_Span(s0, s1, tuple(points[first + k] for k in range(p + 1))))
    return tuple(spans)


def _numerator(span: _Span, axis: int, level: Fraction) -> list:
    return [point[axis] - level * point[-1] for point in span.points]


def _bernstein_at(coefficients: list, s: Fraction) -> Fraction:
    work = list(coefficients)
    for _ in range(len(work) - 1):
        work = [(1 - s) * a + s * b for a, b in zip(work, work[1:])]
    return work[0]


def _sign_at(spans, axis: int, level: Fraction, s: Fraction) -> int:
    for span in spans:
        if span.s0 <= s <= span.s1:
            return _sign(_bernstein_at(_numerator(span, axis, level), (s - span.s0) / (span.s1 - span.s0)))
    raise ValueError("parameter outside the curve's spans")


# ------------------------------------------------------------------ level sets


@dataclass(frozen=True)
class LevelEvent:
    """A zero stretch of ``q_axis(t) - c`` in engine parameters.

    ``lo``/``hi``: exact bounds (a run's ends, an exact root ``lo == hi``, or an
    isolating interval of one root). ``before``/``after``: signs (-1, 0, +1) of
    the coordinate minus the level just before and after it on this curve (0 at
    a curve end, where the neighbour decides).
    """
    lo: Fraction
    hi: Fraction
    before: int
    after: int
    is_run: bool = False
    _root: tuple | None = field(default=None, compare=False, repr=False)   # (span, a, b, g) in local span terms

    @property
    def run(self) -> bool:
        return self.is_run

    @property
    def t0(self) -> Fraction:
        return self.lo if self.is_run else (self.lo + self.hi) / 2

    @property
    def t1(self) -> Fraction:
        return self.hi if self.is_run else (self.lo + self.hi) / 2

    @property
    def crossing(self) -> bool:
        return self.before != 0 and self.after != 0 and self.before != self.after


def level_set(curve: NurbsCurve, axis: int, value) -> tuple[LevelEvent, ...] | None:
    """Exact zero events of ``q_axis(t) - value`` in engine-parameter order; ``None`` if the whole curve is on it."""
    level = Fraction(value)
    spans = _extraction(curve)
    start, end = (Fraction(item) for item in curve.domain)
    low, high = min(start, end), max(start, end)

    # stretches in knot parameter: [lo, hi, is_run, root, ends_are_roots]
    stretches: list[list] = []
    for span in spans:
        coefficients = _numerator(span, axis, level)
        if all(c == 0 for c in coefficients):
            stretches.append([span.s0, span.s1, True, None, True])
            continue
        if all(c > 0 for c in coefficients) or all(c < 0 for c in coefficients):
            continue                                       # exact hull exclusion: no root on this span
        width = span.s1 - span.s0
        for a, b, g in _roots_in_unit(_bernstein_to_power(coefficients)):
            stretches.append([span.s0 + width * a, span.s0 + width * b, False, (span, a, b, g), a == b])
    merged: list[list] = []
    for item in sorted(stretches, key=lambda entry: (entry[0], entry[1])):
        if merged:
            last = merged[-1]
            # Merge a true overlap, or ends that are both roots (a root found from
            # both spans at a knot, a run meeting a root). Isolating intervals end at
            # non-roots and never touch exact roots, so distinct roots stay apart.
            if item[0] < last[1] or (item[0] == last[1] and last[4] and item[4]):
                last[1] = max(last[1], item[1])
                last[2] = last[2] or item[2]
                last[3] = None if last[2] else last[3]
                last[4] = last[4] and item[4]
                continue
        merged.append(item)
    if len(merged) == 1 and merged[0][2] and merged[0][0] <= low and merged[0][1] >= high:
        return None

    def gap_sign(left: Fraction, right: Fraction, shared_is_root: bool) -> int:
        """Sign in the root-free gap [left, right] (a non-root shared end when left == right)."""
        if left < right:
            return _sign_at(spans, axis, level, (left + right) / 2)
        return 0 if shared_is_root else _sign_at(spans, axis, level, left)

    events = []
    for index, (lo, hi, is_run, root, ends_are_roots) in enumerate(merged):
        previous = merged[index - 1][1] if index else low
        following = merged[index + 1][0] if index + 1 < len(merged) else high
        before = gap_sign(previous, lo, ends_are_roots)
        after = gap_sign(hi, following, ends_are_roots)
        events.append((lo, hi, before, after, is_run, root))

    def engine(s):
        return (s - start) / (end - start)

    if end > start:
        return tuple(LevelEvent(engine(lo), engine(hi), before, after, is_run, root)
                     for lo, hi, before, after, is_run, root in events)
    return tuple(LevelEvent(engine(hi), engine(lo), after, before, is_run, root)
                 for lo, hi, before, after, is_run, root in reversed(events))


def event_side(curve: NurbsCurve, event: LevelEvent, axis: int, value) -> int:
    """Exact sign of ``q_axis - value`` at a point event (-1, 0 or +1)."""
    level = Fraction(value)
    if event.is_run:
        raise ValueError("event_side needs a point event")
    if event._root is None or event.lo == event.hi:
        start, end = (Fraction(item) for item in curve.domain)
        s = start + event.lo * (end - start)
        return _sign_at(_extraction(curve), axis, level, s)
    span, a, b, g = event._root
    other = _square_free(_bernstein_to_power(_numerator(span, axis, level)))
    if _is_zero(other):
        return 0
    common = _gcd(g, other)
    if len(common) > 1 and _count(common, a, b) > 0:
        return 0                                           # the event's root is a root of the other coordinate
    low_sign = _value(g, a) > 0
    while _count(other, a, b) > 0:
        middle = (a + b) / 2
        if (_value(g, middle) > 0) == low_sign:
            a = middle
        else:
            b = middle
    # the original numerator (not its square-free part) carries the sign; it has no root in [a, b]
    return _sign(_value(_bernstein_to_power(_numerator(span, axis, level)), a))


def has_root_in(curve: NurbsCurve, axis: int, value, t_lo: Fraction, t_hi: Fraction) -> bool:
    """Whether ``q_axis(t) = value`` somewhere in the exact engine interval [t_lo, t_hi]."""
    level = Fraction(value)
    start, end = (Fraction(item) for item in curve.domain)
    s_a, s_b = sorted((start + t_lo * (end - start), start + t_hi * (end - start)))
    for span in _extraction(curve):
        if span.s1 < s_a or span.s0 > s_b:
            continue
        coefficients = _numerator(span, axis, level)
        if all(c == 0 for c in coefficients):
            return True
        if all(c > 0 for c in coefficients) or all(c < 0 for c in coefficients):
            continue
        width = span.s1 - span.s0
        a = max(Fraction(0), (s_a - span.s0) / width)
        b = min(Fraction(1), (s_b - span.s0) / width)
        if _count(_square_free(_bernstein_to_power(coefficients)), a, b) > 0:
            return True
    return False
