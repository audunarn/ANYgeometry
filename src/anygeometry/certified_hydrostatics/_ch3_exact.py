"""CH3a2: exact points of a curved model and certified predicates on them (design CH3 sections 3.2 and 3.3).

A vertex of a model with curved faces is either a **rational** point (the exact stored float coordinates) or an
**algebraic** point: the corner ``u*`` of a curved face (an irrational ``AlgebraicNumber``) with world coordinates
``E_c(u*) / M(u*)``. ``sign`` decides the sign of a polynomial expression of such points:

* an ``Enclosure`` evaluation (interval arithmetic over the isolating intervals) certifies any non-zero sign;
* if the enclosure straddles zero and all algebraic points share **one** root, the expression is evaluated in
  ``Q(u*)`` (rational functions of that root) and its sign is decided exactly by a single-root test;
* otherwise the roots are refined, and the predicate refuses ``UNDECIDED_PREDICATE`` at ``PREDICATE_BITS``.

An expression is a function of an *algebra* ``alg`` (``coord``, ``const``, ``add``, ``sub``, ``mul``), so the same
formula is evaluated both ways. Every loop reserves a unit before its work."""
from __future__ import annotations

from fractions import Fraction

from ._ch3_admission_trims import rational_enclosure
from ._ch3_algebraic import (AlgebraicNumber, Enclosure, algebraic_equal, compare, isolate_roots, lower_bound,
                             sign_at, upper_bound)
from ._ch3_poly import Kernel, ONE, ZERO, at_most_pow2, bernstein, padd, peval, pmul, psub, refusal
from ._errors import RefusalCategory

__all__ = ["Pt", "EncAlg", "RFAlg", "sign", "points_distinct", "simplest_rational", "arc_box", "point_box",
           "boxes_disjoint", "box_union", "poly_sign_on_open", "PREDICATE_BITS"]

PREDICATE_BITS = 192


class Pt:
    """An exact point: ``rat`` (three Fractions) or ``root``/``num``/``den`` (an irrational root and three
    polynomials over a polynomial)."""

    __slots__ = ("vertex", "root", "num", "den", "rat")

    def __init__(self, vertex, rat=None, root=None, num=None, den=None) -> None:
        self.vertex, self.rat, self.root, self.num, self.den = vertex, rat, root, num, den

    @staticmethod
    def from_corner(k: Kernel, vertex: int, root: AlgebraicNumber, E: tuple, M: tuple) -> "Pt":
        """The point ``E(root) / M(root)``; a rational root gives a rational point."""
        if root.is_rational:
            x = root.lo
            den = peval(k, M, x)
            if den == 0:
                raise refusal(RefusalCategory.UNSUPPORTED_TRIM, "the arc denominator vanishes at a corner",
                              "denominator_vanishes")
            return Pt(vertex, rat=tuple(k.ar.div(peval(k, e, x), den) for e in E))
        return Pt(vertex, root=root, num=tuple(E), den=M)


# ------------------------------------------------------------------ algebras

def _emul(k: Kernel, a: Enclosure, b: Enclosure) -> Enclosure:
    k.tick("interval_op")
    ps = [k.ar.mul(x, y) for x in (a.lo, a.hi) for y in (b.lo, b.hi)]
    return Enclosure(min(ps), max(ps))


class EncAlg:
    """Interval evaluation over the points' isolating intervals."""

    def __init__(self, k: Kernel) -> None:
        self.k = k

    def coord(self, pt: Pt, c: int) -> Enclosure:
        if pt.rat is not None:
            return Enclosure.exact(pt.rat[c])
        return rational_enclosure(self.k, pt.num[c], pt.den, pt.root)

    def const(self, f) -> Enclosure:
        return Enclosure.exact(f)

    def add(self, a, b):
        return a.add(self.k, b)

    def sub(self, a, b):
        return a.add(self.k, b.neg(self.k))

    def mul(self, a, b):
        return _emul(self.k, a, b)


class RFAlg:
    """Exact evaluation in ``Q(u*)``: rational functions ``(num, den)`` of the one shared root ``r0``."""

    def __init__(self, k: Kernel, r0: AlgebraicNumber) -> None:
        self.k, self.r0 = k, r0

    def coord(self, pt: Pt, c: int):
        if pt.rat is not None:
            return ((pt.rat[c],) if pt.rat[c] else (), (ONE,))
        return (pt.num[c], pt.den)

    def const(self, f):
        f = Fraction(f)
        return ((f,) if f else (), (ONE,))

    def add(self, a, b):
        k = self.k
        return (padd(k, pmul(k, a[0], b[1]), pmul(k, b[0], a[1])), pmul(k, a[1], b[1]))

    def sub(self, a, b):
        k = self.k
        return (psub(k, pmul(k, a[0], b[1]), pmul(k, b[0], a[1])), pmul(k, a[1], b[1]))

    def mul(self, a, b):
        return (pmul(self.k, a[0], b[0]), pmul(self.k, a[1], b[1]))

    def sign(self, v) -> int:
        k = self.k
        s_den = sign_at(k, v[1], self.r0)
        if s_den == 0:
            raise refusal(RefusalCategory.UNSUPPORTED_TRIM, "a point denominator vanishes at its root",
                          "denominator_vanishes")
        return sign_at(k, v[0], self.r0) * s_den


def _roots_of(k: Kernel, pts) -> list:
    roots: list = []
    for p in pts:
        k.tick("admission_element")
        if p.root is not None and not any(r is p.root or algebraic_equal(k, p.root, r) for r in roots):
            roots.append(p.root)
    return roots


def sign(k: Kernel, pts, expr) -> int:
    """The exact sign of ``expr(alg)`` for the points ``pts`` (see the module docstring)."""
    roots = _roots_of(k, pts)
    bits = min(k.limits.max_isolation_bits, PREDICATE_BITS)
    while True:
        k.tick("predicate_round")
        e = expr(EncAlg(k))
        if e.lo > 0:
            return 1
        if e.hi < 0:
            return -1
        if e.lo == e.hi:
            return 0
        if len(roots) == 1:
            alg = RFAlg(k, roots[0])
            return alg.sign(expr(alg))
        for r in roots:
            if at_most_pow2(r.hi - r.lo, bits, 2):
                raise refusal(RefusalCategory.UNDECIDED_PREDICATE,
                              f"a predicate on points with several algebraic roots is not decided at {bits} bits "
                              "(it may be exactly zero)", "multi_root_zero")
            r.refine(k)


def points_distinct(k: Kernel, a: Pt, b: Pt) -> bool:
    """True iff the points differ (a coordinate difference has a certified non-zero sign); False iff exactly
    equal; refuses ``UNDECIDED_PREDICATE`` if neither is established."""
    if a.vertex == b.vertex:
        return False
    for c in range(3):
        k.tick("admission_element")
        if sign(k, (a, b), lambda alg, c=c: alg.sub(alg.coord(a, c), alg.coord(b, c))) != 0:
            return True
    return False


def simplest_rational(k: Kernel, lo: Fraction, hi: Fraction) -> Fraction:
    """The rational of least denominator in the closed interval ``[lo, hi]``."""
    if lo > hi:
        raise ValueError("empty interval")
    if lo <= 0 <= hi:
        return ZERO
    if hi < 0:
        return -simplest_rational(k, -hi, -lo)
    prefix = []
    while True:
        k.tick("admission_element")
        fl = lo.numerator // lo.denominator
        if fl == lo:
            result = Fraction(fl)
            break
        if fl + 1 <= hi:
            result = Fraction(fl + 1)
            break
        prefix.append(fl)
        lo, hi = 1 / (hi - fl), 1 / (lo - fl)
    for a in reversed(prefix):
        k.tick("admission_element")
        result = a + 1 / result
    return result


# ------------------------------------------------------------------ boxes (tuples of three Enclosures)

def point_box(k: Kernel, pt: Pt) -> tuple:
    alg = EncAlg(k)
    return tuple(alg.coord(pt, c) for c in range(3))


def _bounds(k: Kernel, p: tuple, lo, hi):
    low = high = None
    for c in bernstein(k, p, lo, hi):
        k.tick("bernstein_coefficient")
        if low is None or c < low:
            low = c
        if high is None or c > high:
            high = c
    return low, high


def arc_box(k: Kernel, E: tuple, M: tuple, lo, hi, pieces: int = 8) -> tuple:
    """A box containing ``E(u) / M(u)`` for ``u`` in ``[lo, hi]`` (rationals; ``M`` one-signed there), by interval
    division on ``pieces`` equal sub-intervals."""
    lows = [None] * 3
    highs = [None] * 3
    width = k.ar.sub(hi, lo)
    for i in range(pieces):
        k.tick("admission_element")
        a = k.ar.add(lo, k.ar.div(k.ar.mul(width, i), pieces))
        b = k.ar.add(lo, k.ar.div(k.ar.mul(width, i + 1), pieces))
        ml, mh = _bounds(k, M, a, b)
        if not (ml > 0 or mh < 0):
            raise refusal(RefusalCategory.UNSUPPORTED_TRIM, "an arc denominator is not certified one-signed",
                          "denominator_not_certified")
        for c in range(3):
            nl, nh = _bounds(k, E[c], a, b)
            qs = [k.ar.div(n, m) for n in (nl, nh) for m in (ml, mh)]
            lo_c, hi_c = min(qs), max(qs)
            if lows[c] is None or lo_c < lows[c]:
                lows[c] = lo_c
            if highs[c] is None or hi_c > highs[c]:
                highs[c] = hi_c
    return tuple(Enclosure(lows[c], highs[c]) for c in range(3))


def boxes_disjoint(a: tuple, b: tuple) -> bool:
    return any(a[c].hi < b[c].lo or b[c].hi < a[c].lo for c in range(3))


def box_union(boxes) -> tuple:
    return tuple(Enclosure(min(b[c].lo for b in boxes), max(b[c].hi for b in boxes)) for c in range(3))


def poly_sign_on_open(k: Kernel, g: tuple, lo, hi) -> tuple:
    """``(sign, root_inside)`` of the polynomial ``g`` on the open interval between ``lo < hi`` (rational or
    algebraic): the common non-zero sign if ``g`` has no root there; ``(0, False)`` if ``g`` is identically zero;
    ``(0, True)`` if it has a root there."""
    if not g:
        return 0, False
    for r in isolate_roots(k, g, lower_bound(lo) - 1, upper_bound(hi) + 1):
        k.tick("root_refine")
        if compare(k, r, lo) > 0 and compare(k, r, hi) < 0:
            return 0, True
    while upper_bound(lo) >= lower_bound(hi):
        k.tick("root_refine")
        for x in (lo, hi):
            if isinstance(x, AlgebraicNumber) and not x.is_rational:
                x.refine(k)
    sample = k.ar.div(k.ar.add(upper_bound(lo), lower_bound(hi)), 2)
    v = peval(k, g, sample)
    return (v > 0) - (v < 0), False
