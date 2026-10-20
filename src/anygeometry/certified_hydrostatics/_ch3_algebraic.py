"""CH3 exact kernel, layer 2: real algebraic numbers, enclosures and single-root predicates (design CH3 3.2).

An ``AlgebraicNumber`` is a real root of a square-free primitive integer polynomial ``poly`` (positive leading
coefficient), named by its ``ordinal`` among the real roots of ``poly`` (increasing, from 0), with an isolating
interval that holds exactly one root of ``poly`` by a Sturm count of 1 (never by endpoint signs alone): the
open interval ``(lo, hi)`` with ``lo < hi``, or the degenerate ``[r, r]`` for a rational root ``r``.
``refine()`` bisects exactly.

Identity (3.2): same ``poly`` and same ordinal; for different polynomials, the intersection ``I`` of the
isolating intervals must have ``count_p(I) = count_q(I) = count_gcd(I) = 1``; ``count_gcd(I) = 0`` proves the
numbers distinct, and distinct numbers are ordered by refining until their intervals are disjoint. Overlap of
intervals is never used as evidence of equality.

Single-root predicates (the sign of a rational polynomial at an algebraic number) are exact: zero iff
``gcd(g, poly)`` has a root in the isolating interval, otherwise the interval is refined until the Bernstein
coefficients of ``g`` over it are one-signed (bounded by ``max_isolation_bits``; beyond it the predicate is
refused ``UNDECIDED_PREDICATE``)."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from ._ch3_poly import (Kernel, ZERO, bernstein, cauchy_bound, psub, peval, pgcd, primitive, refusal,
                        squarefree, sturm_sequence, variations_at, variations_at_infinity)
from ._errors import RefusalCategory

__all__ = ["Enclosure", "AlgebraicNumber", "isolate_roots", "algebraic_equal", "compare", "sign_at",
           "count_open", "lower_bound", "upper_bound"]


# ------------------------------------------------------------------ enclosures

@dataclass(frozen=True)
class Enclosure:
    """A closed rational interval ``[lo, hi]`` that contains an exact value."""

    lo: Fraction
    hi: Fraction

    def __post_init__(self) -> None:
        if not isinstance(self.lo, Fraction) or not isinstance(self.hi, Fraction):
            raise TypeError("Enclosure bounds must be Fractions")
        if self.lo > self.hi:
            raise ValueError("Enclosure with lo > hi")

    @property
    def width(self) -> Fraction:
        return self.hi - self.lo

    def contains(self, x) -> bool:
        return self.lo <= x <= self.hi

    def add(self, k: Kernel, other: "Enclosure") -> "Enclosure":
        k.tick("interval_op")
        return Enclosure(k.ar.add(self.lo, other.lo), k.ar.add(self.hi, other.hi))

    def neg(self, k: Kernel) -> "Enclosure":
        k.tick("interval_op")
        return Enclosure(-self.hi, -self.lo)

    def shift(self, k: Kernel, c) -> "Enclosure":
        k.tick("interval_op")
        return Enclosure(k.ar.add(self.lo, c), k.ar.add(self.hi, c))

    def scale(self, k: Kernel, c) -> "Enclosure":
        k.tick("interval_op")
        a, b = k.ar.mul(self.lo, c), k.ar.mul(self.hi, c)
        return Enclosure(a, b) if a <= b else Enclosure(b, a)

    @staticmethod
    def exact(x) -> "Enclosure":
        x = Fraction(x)
        return Enclosure(x, x)


# ------------------------------------------------------------------ algebraic numbers

class AlgebraicNumber:
    """A real root of ``poly`` (square-free, primitive, positive leading coefficient) with its ordinal."""

    __slots__ = ("poly", "ordinal", "lo", "hi", "_sturm")

    def __init__(self, poly: tuple, ordinal: int, lo: Fraction, hi: Fraction, sturm: tuple) -> None:
        self.poly = poly
        self.ordinal = ordinal
        self.lo = lo
        self.hi = hi
        self._sturm = sturm

    @property
    def is_rational(self) -> bool:
        return self.lo == self.hi

    def interval(self) -> Enclosure:
        return Enclosure(self.lo, self.hi)

    def __repr__(self) -> str:
        return (f"AlgebraicNumber(poly={tuple(int(c) for c in self.poly)}, ordinal={self.ordinal}, "
                f"~{float(self.lo + self.hi) / 2:.12g})")

    @classmethod
    def from_interval(cls, k: Kernel, poly: tuple, lo, hi) -> "AlgebraicNumber":
        """The root of ``poly`` isolated by the open interval ``(lo, hi)`` (or ``[r, r]``); the polynomial is
        made square-free and primitive first. Refuses ``UNDECIDED_PREDICATE`` unless the count is exactly 1."""
        lo, hi = Fraction(lo), Fraction(hi)
        p = squarefree(k, poly)
        if len(p) < 2:
            raise ValueError("an algebraic number needs a polynomial of positive degree")
        seq = sturm_sequence(k, p)
        if lo == hi:
            if peval(k, p, lo) != 0:
                raise refusal(RefusalCategory.UNDECIDED_PREDICATE, "the degenerate interval is not a root")
        elif lo > hi or _count_open_seq(k, p, seq, lo, hi) != 1:
            raise refusal(RefusalCategory.UNDECIDED_PREDICATE, "the interval does not isolate exactly one root")
        return cls(p, _ordinal(k, p, seq, lo, hi), lo, hi, seq)

    def refine(self, k: Kernel) -> None:
        """One exact bisection step (``root_refine``); a rational midpoint root makes the interval degenerate."""
        if self.lo == self.hi:
            return
        k.tick("root_refine")
        if self.hi - self.lo <= Fraction(2, 1 << k.limits.max_isolation_bits):
            raise refusal(RefusalCategory.UNDECIDED_PREDICATE,
                          f"isolating interval refined to max_isolation_bits={k.limits.max_isolation_bits}")
        mid = k.ar.div(k.ar.add(self.lo, self.hi), 2)
        if peval(k, self.poly, mid) == 0:
            self.lo = self.hi = mid
        elif _count_open_seq(k, self.poly, self._sturm, self.lo, mid) == 1:
            self.hi = mid
        else:
            self.lo = mid

    def refine_below(self, k: Kernel, width) -> None:
        """Refine until ``hi - lo <= width``."""
        while self.hi - self.lo > width:
            self.refine(k)


def _count_open_seq(k: Kernel, p: tuple, seq: tuple, lo, hi) -> int:
    """Distinct roots of the square-free ``p`` in the open ``(lo, hi)``: ``V(lo) - V(hi)`` counts ``(lo, hi]``."""
    if lo >= hi:
        return 0
    count = variations_at(k, seq, lo) - variations_at(k, seq, hi)
    if peval(k, p, hi) == 0:
        count -= 1
    return count


def _ordinal(k: Kernel, p: tuple, seq: tuple, lo, hi) -> int:
    """Number of real roots of ``p`` below the root isolated by ``(lo, hi)`` / ``[lo, lo]``: the roots in
    ``(-inf, lo]`` (every root in ``(-inf, lo]`` is below the isolated one, which is ``> lo`` or ``= lo``)."""
    below = variations_at_infinity(k, seq, positive=False) - variations_at(k, seq, lo)
    if lo == hi:
        below -= 1                      # the root itself is lo
    return below


def count_open(k: Kernel, p: tuple, lo, hi) -> int:
    """Distinct real roots of the polynomial ``p`` (any, nonzero) in the open interval ``(lo, hi)``."""
    q = squarefree(k, p)
    if len(q) < 2:
        return 0
    return _count_open_seq(k, q, sturm_sequence(k, q), lo, hi)


def isolate_roots(k: Kernel, p: tuple, lo=None, hi=None) -> list[AlgebraicNumber]:
    """All distinct real roots of ``p`` in the open interval ``(lo, hi)`` (``None``: unbounded), increasing."""
    q = squarefree(k, p)
    if len(q) < 2:
        return []
    seq = sturm_sequence(k, q)
    bound = cauchy_bound(k, q)
    a = -bound if lo is None else Fraction(lo)
    b = bound if hi is None else Fraction(hi)
    if a >= b:
        return []
    found: list[AlgebraicNumber] = []
    stack = [(a, b, _count_open_seq(k, q, seq, a, b))]
    min_width = Fraction(1, 2 ** k.limits.max_isolation_bits)
    while stack:
        k.tick("root_refine")
        x, y, c = stack.pop()
        if c == 0:
            continue
        if c == 1:
            found.append(AlgebraicNumber(q, _ordinal(k, q, seq, x, y), x, y, seq))
            continue
        if y - x <= min_width:
            raise refusal(RefusalCategory.UNDECIDED_PREDICATE,
                          f"root isolation reached max_isolation_bits={k.limits.max_isolation_bits}")
        m = k.ar.div(k.ar.add(x, y), 2)
        if peval(k, q, m) == 0:
            found.append(AlgebraicNumber(q, _ordinal(k, q, seq, m, m), m, m, seq))
        left = _count_open_seq(k, q, seq, x, m)
        stack.append((m, y, _count_open_seq(k, q, seq, m, y)))
        stack.append((x, m, left))
    found.sort(key=lambda r: r.ordinal)          # native sort of at most deg(p) items (bounded by the degree)
    return found


# ------------------------------------------------------------------ identity, order, signs

def _as_value(x):
    return x if isinstance(x, AlgebraicNumber) else Fraction(x)


def lower_bound(x) -> Fraction:
    return x.lo if isinstance(x, AlgebraicNumber) else Fraction(x)


def upper_bound(x) -> Fraction:
    return x.hi if isinstance(x, AlgebraicNumber) else Fraction(x)


def _strictly_below(x, y) -> bool:
    """Certainly ``x < y`` from the intervals alone (open intervals, or degenerate points)."""
    xh, yl = upper_bound(x), lower_bound(y)
    if xh < yl:
        return True
    if xh == yl:
        x_point = not isinstance(x, AlgebraicNumber) or x.is_rational
        y_point = not isinstance(y, AlgebraicNumber) or y.is_rational
        return not (x_point and y_point)
    return False


def _rational_value(x):
    if isinstance(x, AlgebraicNumber):
        return x.lo if x.is_rational else None
    return x


def algebraic_equal(k: Kernel, x, y) -> bool:
    """Exact identity of two real algebraic (or rational) numbers by the rules of design 3.2."""
    x, y = _as_value(x), _as_value(y)
    rx, ry = _rational_value(x), _rational_value(y)
    if rx is not None and ry is not None:
        return rx == ry
    if rx is not None or ry is not None:
        r, alpha = (rx, y) if rx is not None else (ry, x)
        k.tick("sign_count")
        return alpha.lo < r < alpha.hi and peval(k, alpha.poly, r) == 0
    if x.poly == y.poly:
        return x.ordinal == y.ordinal
    g = primitive(k, pgcd(k, x.poly, y.poly))
    while True:
        k.tick("root_refine")
        lo, hi = max(x.lo, y.lo), min(x.hi, y.hi)
        if lo >= hi:
            return False
        if len(g) < 2 or count_open(k, g, lo, hi) == 0:
            return False                        # no common root in I, so x != y (both lie in I if equal)
        if count_open(k, x.poly, lo, hi) == 1 and count_open(k, y.poly, lo, hi) == 1 \
                and count_open(k, g, lo, hi) == 1:
            return True
        x.refine(k)
        y.refine(k)
        if x.is_rational or y.is_rational:
            return algebraic_equal(k, x, y)


def compare(k: Kernel, x, y) -> int:
    """``-1``, ``0`` or ``1`` as ``x <, =, > y``; exact (identity first, then refinement to disjointness)."""
    x, y = _as_value(x), _as_value(y)
    if _strictly_below(x, y):
        return -1
    if _strictly_below(y, x):
        return 1
    if algebraic_equal(k, x, y):
        return 0
    while True:
        k.tick("root_refine")
        if _strictly_below(x, y):
            return -1
        if _strictly_below(y, x):
            return 1
        # refine the wider one (a rational is a point and needs no refinement)
        candidates = [z for z in (x, y) if isinstance(z, AlgebraicNumber) and not z.is_rational]
        if not candidates:
            raise ArithmeticError("distinct points with overlapping degenerate intervals")
        widest = max(candidates, key=lambda z: z.hi - z.lo)
        if isinstance(y, Fraction) or (isinstance(y, AlgebraicNumber) and y.is_rational):
            r = _rational_value(y)
            _split_at(k, widest, r)
        elif isinstance(x, Fraction) or (isinstance(x, AlgebraicNumber) and x.is_rational):
            r = _rational_value(x)
            _split_at(k, widest, r)
        else:
            widest.refine(k)


def _split_at(k: Kernel, alpha: AlgebraicNumber, r: Fraction) -> None:
    """Shrink ``alpha``'s interval to one side of the rational ``r`` (``alpha != r`` known)."""
    if not (alpha.lo < r < alpha.hi):
        alpha.refine(k)
        return
    k.tick("root_refine")
    if _count_open_seq(k, alpha.poly, alpha._sturm, alpha.lo, r) == 1:
        alpha.hi = r
    else:
        alpha.lo = r


def sign_at(k: Kernel, g: tuple, x) -> int:
    """Exact sign of the rational polynomial ``g`` at the algebraic (or rational) number ``x``."""
    if not g:
        return 0
    x = _as_value(x)
    r = _rational_value(x)
    if r is not None:
        v = peval(k, g, r)
        return (v > 0) - (v < 0)
    h = pgcd(k, g, x.poly)
    if len(h) > 1 and count_open(k, h, x.lo, x.hi) >= 1:
        return 0                                # the only root of poly in (lo, hi) is x, and h | poly
    while True:
        k.tick("root_refine")
        coeffs = bernstein(k, g, x.lo, x.hi)
        sign = 0
        uniform = True
        for c in coeffs:
            k.tick("bernstein_coefficient")
            s = (c > 0) - (c < 0)
            if s == 0 or (sign and s != sign):
                uniform = False
                break
            sign = s
        if uniform:
            return sign
        x.refine(k)
        if x.is_rational:
            v = peval(k, g, x.lo)
            return (v > 0) - (v < 0)


def sign_of_difference(k: Kernel, g: tuple, h: tuple, x) -> int:
    """Sign of ``g - h`` at ``x`` (convenience)."""
    return sign_at(k, psub(k, g, h), x)
