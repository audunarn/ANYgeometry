"""Rigorous interval primitives with exact Fraction endpoints (reference path).

Every operation is exact on rationals, so enclosures are never narrower than the
true range.  Float conversion is the only rounding and it is always outward.
Faster directed-float paths are future work and must be validated against this
module (see tests/nurbs_surface_intersection/test_intervals.py).
"""
from __future__ import annotations

import math
from fractions import Fraction

__all__ = ["Interval", "float_down", "float_up"]


def float_down(value: Fraction) -> float:
    """Largest double <= value (``-inf`` if the value is below the double range)."""
    try:
        made = float(value)
    except OverflowError:
        return -math.inf if value < 0 else math.nextafter(math.inf, 0.0)
    if math.isinf(made):
        return -math.inf if made < 0 else math.nextafter(math.inf, 0.0)
    return math.nextafter(made, -math.inf) if Fraction(made) > value else made


def float_up(value: Fraction) -> float:
    """Smallest double >= value (``inf`` if the value is above the double range)."""
    try:
        made = float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.nextafter(math.inf, 0.0)
    if math.isinf(made):
        return math.inf if made > 0 else -math.nextafter(math.inf, 0.0)
    return math.nextafter(made, math.inf) if Fraction(made) < value else made


def sqrt_up(value: Fraction) -> Fraction:
    """Exact rational upper bound of sqrt(value), value >= 0 (relative excess below 2**-60)."""
    value = Fraction(value)
    if value == 0:
        return Fraction(0)
    shift = 64
    return Fraction(math.isqrt((value.numerator << (2 * shift)) // value.denominator) + 1, 1 << shift)


def diameter_upper(squared: Fraction) -> float:
    """Float ``r`` with ``Fraction(r)**2 >= squared`` (exact postcheck); never rounds down to a false zero."""
    squared = Fraction(squared)
    if squared == 0:
        return 0.0
    reported = float_up(sqrt_up(squared))
    while math.isfinite(reported) and Fraction(reported) ** 2 < squared:
        reported = math.nextafter(reported, math.inf)
    return reported


class Interval:
    """Closed interval ``[lo, hi]`` with Fraction endpoints."""

    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        lo = Fraction(lo)
        hi = lo if hi is None else Fraction(hi)
        if lo > hi:
            raise ValueError("interval endpoints are reversed")
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)

    def __setattr__(self, name, value):
        raise AttributeError("Interval is immutable")

    def __delattr__(self, name):
        raise AttributeError("Interval is immutable")

    def __repr__(self) -> str:
        return f"Interval({self.lo}, {self.hi})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Interval) and self.lo == other.lo and self.hi == other.hi

    def __hash__(self) -> int:
        return hash((self.lo, self.hi))

    def __add__(self, other: "Interval") -> "Interval":
        return Interval(self.lo + other.lo, self.hi + other.hi)

    def __sub__(self, other: "Interval") -> "Interval":
        return Interval(self.lo - other.hi, self.hi - other.lo)

    def __neg__(self) -> "Interval":
        return Interval(-self.hi, -self.lo)

    def __mul__(self, other: "Interval") -> "Interval":
        products = (self.lo * other.lo, self.lo * other.hi, self.hi * other.lo, self.hi * other.hi)
        return Interval(min(products), max(products))

    def scale(self, factor) -> "Interval":
        factor = Fraction(factor)
        a, b = self.lo * factor, self.hi * factor
        return Interval(min(a, b), max(a, b))

    @property
    def mid(self) -> Fraction:
        return (self.lo + self.hi) / 2

    @property
    def width(self) -> Fraction:
        return self.hi - self.lo

    def contains(self, value) -> bool:
        return self.lo <= Fraction(value) <= self.hi

    def contains_zero(self) -> bool:
        return self.lo <= 0 <= self.hi

    def excludes_zero(self) -> bool:
        return self.lo > 0 or self.hi < 0

    def strictly_inside(self, other: "Interval") -> bool:
        return other.lo < self.lo and self.hi < other.hi

    def hull(self, other: "Interval") -> "Interval":
        return Interval(min(self.lo, other.lo), max(self.hi, other.hi))

    def intersects(self, other: "Interval") -> bool:
        return self.lo <= other.hi and other.lo <= self.hi

    def to_floats(self) -> tuple[float, float]:
        return float_down(self.lo), float_up(self.hi)

    @staticmethod
    def sum(items) -> "Interval":
        lo = hi = Fraction(0)
        for item in items:
            lo += item.lo
            hi += item.hi
        return Interval(lo, hi)
