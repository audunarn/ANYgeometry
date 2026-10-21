"""CH3 exact kernel, layer 4: bivariate resultants and the ``Q(u*)`` gcd validation (design CH3 section 3.3).

A bivariate polynomial ``F(u, s)`` is a tuple of univariate ``u``-polynomials indexed by the power of ``s``
(``F = sum F[i](u) s^i``). ``resultant_s`` computes ``Res_s(F, G)`` by fraction-free (Bareiss) elimination of the
Sylvester matrix over ``Q[u]``.

Validation of a resultant root ``u*`` (3.3, with the round-4 obligation): the Euclidean algorithm for
``gcd_s(F(u*, s), G(u*, s))`` runs over ``Q[u]/(q)`` where ``q`` starts as the square-free polynomial of
``u*`` and is **refined to a factor containing ``u*`` whenever a zero test meets a nontrivial gcd**: for a
coefficient ``c``, ``g = gcd(c, q)``; if ``g`` has ``u*`` as a root (a Sturm count of ``g`` on ``u*``'s
isolating interval) then ``q := g`` and ``c`` is zero; otherwise ``q := q / g`` and ``c`` is a unit. So every
leading coefficient used by the algorithm is invertible modulo the current ``q`` and every dropped coefficient
vanishes at ``u*``: the computed remainder sequence specialises under ``u -> u*`` to the Euclidean sequence of
``F(u*, s)`` and ``G(u*, s)``, and the computed gcd has exactly the degree of their gcd over ``C``. ``q`` need not
be irreducible; it only has to make every inverse that the algorithm needs exist, and the refinement ensures
that (it reaches the irreducible factor of ``u*`` only as far as the computation requires).

``u*`` is a genuine crossing iff the gcd has positive degree and a real root ``s*`` in the open ``s``-interval;
the count is a Sturm count over ``Q(u*)`` whose every sign is a single-root test."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from ._ch3_algebraic import AlgebraicNumber, Enclosure, compare, count_open, isolate_roots, lower_bound, \
    sign_at, upper_bound
from ._ch3_poly import (Kernel, ONE, ZERO, at_most_pow2, padd, pdivmod, pexact_div, pgcd, pmul, pneg, ppow, primitive, psub,
                        pxgcd, refusal, squarefree)
from ._errors import RefusalCategory

__all__ = ["bivariate", "biv_trim", "resultant_s", "NumberField", "field_gcd", "validate_crossing",
           "CrossingValidation", "curve_crossings"]


# ------------------------------------------------------------------ bivariate polynomials

def biv_trim(k: Kernel, F: list) -> tuple:
    F = list(F)
    while F and not F[-1]:
        k.tick("poly_coefficient")
        F.pop()
    return tuple(F)


def bivariate(k: Kernel, rows) -> tuple:
    """``rows[i]`` is the ``u``-polynomial coefficient of ``s^i`` (tuples of Fractions)."""
    out = []
    for r in rows:
        k.tick("poly_coefficient")                  # every row is a checkpoint, empty or not
        row = []
        for c in r:
            k.tick("poly_coefficient")
            row.append(k.ar.fit(Fraction(c)))
        out.append(tuple(row))
    return biv_trim(k, out)


def _det_bareiss(k: Kernel, M: list) -> tuple:
    """Determinant of a square matrix over ``Q[u]`` by fraction-free elimination (exact divisions)."""
    n = len(M)
    if n == 0:
        return (ONE,)
    copy = []
    for row in M:
        k.tick("poly_coefficient")
        copy.append(list(row))
    M = copy
    sign = 1
    prev = (ONE,)
    for c in range(n - 1):
        k.tick("poly_coefficient")
        pivot = None
        for r in range(c, n):
            k.tick("poly_coefficient")
            if M[r][c]:
                pivot = r
                break
        if pivot is None:
            return ()
        if pivot != c:
            M[c], M[pivot] = M[pivot], M[c]
            sign = -sign
        for i in range(c + 1, n):
            for j in range(c + 1, n):
                k.tick("poly_coefficient")
                num = psub(k, pmul(k, M[i][j], M[c][c]), pmul(k, M[i][c], M[c][j]))
                M[i][j] = pexact_div(k, num, prev)
            M[i][c] = ()
        prev = M[c][c]
    det = M[n - 1][n - 1]
    return det if sign > 0 else pneg(k, det)


def resultant_s(k: Kernel, F: tuple, G: tuple) -> tuple:
    """``Res_s(F, G)`` in ``Q[u]`` (``()`` when either is zero or they share a factor of positive ``s``-degree)."""
    if not F or not G:
        return ()
    m, n = len(F) - 1, len(G) - 1
    if m == 0 and n == 0:
        return (ONE,)
    if m == 0:
        return ppow(k, F[0], n)
    if n == 0:
        return ppow(k, G[0], m)
    size = m + n
    rows = []
    for i in range(n):
        row = [()] * size
        for j in range(m + 1):
            k.tick("poly_coefficient")
            row[i + j] = F[m - j]
        rows.append(row)
    for i in range(m):
        row = [()] * size
        for j in range(n + 1):
            k.tick("poly_coefficient")
            row[i + j] = G[n - j]
        rows.append(row)
    return _det_bareiss(k, rows)


# ------------------------------------------------------------------ Q[u]/(q) with factor refinement

class NumberField:
    """Arithmetic in ``Q[u]/(q)`` that refines ``q`` to the factor containing ``root`` at every zero test."""

    __slots__ = ("k", "root", "q", "splits")

    def __init__(self, k: Kernel, root: AlgebraicNumber, q: tuple | None = None) -> None:
        self.k = k
        self.root = root
        q = root.poly if q is None else squarefree(k, q)
        if len(q) < 2 or sign_at(k, q, root) != 0:
            raise ValueError("q must be a square-free polynomial vanishing at the root")
        if q != root.poly:               # make the interval isolate the root among the roots of q
            while not root.is_rational and count_open(k, q, root.lo, root.hi) > 1:
                root.refine(k)
        self.q = q
        self.splits = 0

    def reduce(self, c: tuple) -> tuple:
        return pdivmod(self.k, c, self.q)[1] if len(c) >= len(self.q) else c

    def is_zero(self, c: tuple) -> bool:
        """Exact zero test of ``c(root)``; refines ``q`` when ``gcd(c, q)`` is nontrivial."""
        k = self.k
        c = self.reduce(c)
        if not c:
            return True
        g = primitive(k, pgcd(k, c, self.q))
        if len(g) < 2:
            return False
        k.tick("root_refine")
        self.splits += 1
        if self._contains_root(g):
            self.q = g
            return True
        self.q = primitive(k, pexact_div(k, self.q, g))
        return False

    def _contains_root(self, g: tuple) -> bool:
        """Whether the root is a root of ``g`` (``g`` divides the root's square-free polynomial's multiple
        ``q``, so its roots in the isolating interval can only be the root itself)."""
        r = self.root
        if r.is_rational:
            return sign_at(self.k, g, r.lo) == 0
        return count_open(self.k, g, r.lo, r.hi) >= 1

    def inv(self, c: tuple) -> tuple:
        if self.is_zero(c):
            raise ZeroDivisionError("inverse of zero in Q(u*)")
        g, s, _ = pxgcd(self.k, self.reduce(c), self.q)
        if g != (ONE,):
            raise ArithmeticError("no inverse although the zero test passed")
        return self.reduce(s)

    def mul(self, a: tuple, b: tuple) -> tuple:
        return self.reduce(pmul(self.k, a, b))

    def sub(self, a: tuple, b: tuple) -> tuple:
        return self.reduce(psub(self.k, a, b))

    def sign(self, c: tuple) -> int:
        """Exact sign of ``c(root)``."""
        if self.is_zero(c):
            return 0
        return sign_at(self.k, self.reduce(c), self.root)

    def normalize(self, P: tuple) -> tuple:
        """Drop leading coefficients (in ``s``) that vanish at the root."""
        P = list(P)
        while P and self.is_zero(P[-1]):
            self.k.tick("poly_coefficient")
            P.pop()
        out = []
        for c in P:
            self.k.tick("poly_coefficient")
            out.append(self.reduce(c))
        return tuple(out)

    def divmod_s(self, A: tuple, B: tuple) -> tuple[tuple, tuple]:
        """Division in ``s`` over the field; ``B`` normalised (nonzero leading coefficient)."""
        k = self.k
        A = list(self.normalize(A))
        B = self.normalize(B)
        if not B:
            raise ZeroDivisionError("division by the zero polynomial in s")
        inv = self.inv(B[-1])
        db = len(B) - 1
        quot = [()] * max(len(A) - db, 0)
        for shift in range(len(A) - 1 - db, -1, -1):
            k.tick("poly_coefficient")
            c = self.mul(A[shift + db], inv)
            quot[shift] = c
            if c:
                for j in range(db + 1):
                    k.tick("poly_coefficient")
                    A[shift + j] = self.sub(A[shift + j], self.mul(c, B[j]))
            A[shift + db] = ()
        return self.normalize(quot), self.normalize(A[:db])

    def eval_s(self, P: tuple, s: Fraction) -> tuple:
        """``P(u, s)`` at a rational ``s``, as an element of the field."""
        acc = ()
        for c in reversed(P):
            self.k.tick("poly_coefficient")
            acc = self.reduce(padd(self.k, pmul(self.k, acc, (s,)), c))
        return acc

    def deriv_s(self, P: tuple) -> tuple:
        out = []
        for i in range(1, len(P)):
            self.k.tick("poly_coefficient")
            out.append(self.reduce(pmul(self.k, P[i], (Fraction(i),))))
        return self.normalize(out)


def field_gcd(nf: NumberField, A: tuple, B: tuple) -> tuple:
    """Monic ``gcd_s(A(u*, s), B(u*, s))`` by the Euclidean algorithm over ``Q[u]/(q)``."""
    k = nf.k
    A, B = nf.normalize(A), nf.normalize(B)
    while B:
        k.tick("poly_coefficient")
        _, r = nf.divmod_s(A, B)
        A, B = B, r
    if not A:
        return ()
    inv = nf.inv(A[-1])
    out = []
    for c in A:
        k.tick("poly_coefficient")
        out.append(nf.mul(c, inv))
    return nf.normalize(tuple(out))


def _field_sturm(nf: NumberField, P: tuple) -> list:
    """A Sturm sequence of the square-free (over ``Q(u*)``) ``P``: ``P, P', -rem, ...``."""
    k = nf.k
    seq = [P, nf.deriv_s(P)]
    while len(seq[-1]) > 1:
        k.tick("sign_count")
        _, r = nf.divmod_s(seq[-2], seq[-1])
        if not r:
            break
        neg = []
        for c in r:
            k.tick("poly_coefficient")
            neg.append(nf.sub((), c))
        seq.append(tuple(neg))
    return seq


def _field_variations(nf: NumberField, seq: list, s: Fraction) -> int:
    count = 0
    last = 0
    for P in seq:
        nf.k.tick("sign_count")
        v = nf.sign(nf.eval_s(P, s))
        if v:
            if last and v != last:
                count += 1
            last = v
    return count


def _field_count_open(nf: NumberField, seq: list, lo: Fraction, hi: Fraction) -> int:
    count = _field_variations(nf, seq, lo) - _field_variations(nf, seq, hi)
    if nf.sign(nf.eval_s(seq[0], hi)) == 0:
        count -= 1
    return count


@dataclass(frozen=True)
class CrossingValidation:
    """Result of the ``Q(u*)`` validation of one resultant root."""

    u: AlgebraicNumber
    genuine: bool
    gcd_degree: int
    s_roots: tuple                 # Enclosures of the real roots s* in the open s-interval, increasing
    final_q: tuple                 # the refined modulus (a factor of the starting q containing u*)
    splits: int


def _s_bound(k: Kernel, x):
    """An ``s`` bound: a rational (size-checked) or a genuinely irrational ``AlgebraicNumber``."""
    if isinstance(x, AlgebraicNumber):
        return k.ar.fit(x.lo) if x.is_rational else x
    return k.ar.fit(Fraction(x))


def _side(nf: NumberField, h: tuple, enc: Enclosure, S) -> int | None:
    """``-1``, ``0``, ``1`` as the single root in ``enc`` is below, equal to or above ``S``; ``None`` if the
    intervals do not decide it yet. ``enc`` isolates one root of ``h`` in its open interval (or is a point)."""
    if isinstance(S, AlgebraicNumber) and S.is_rational:
        S = S.lo
    if isinstance(S, AlgebraicNumber):                   # irrational: never equal to an enclosure's rational end
        if enc.hi <= S.lo:
            return -1
        if enc.lo >= S.hi:
            return 1
        return None
    if enc.lo == enc.hi:
        return (enc.lo > S) - (enc.lo < S)
    if enc.hi <= S:
        return -1
    if enc.lo >= S:
        return 1
    return 0 if nf.sign(nf.eval_s(h, S)) == 0 else None     # S strictly inside: equal, or bisect further


def _refine_s_root(nf: NumberField, seq: list, h: tuple, enc: Enclosure) -> Enclosure:
    """One exact bisection of an enclosure holding exactly one root of ``h`` in its open interval."""
    k = nf.k
    a, b = enc.lo, enc.hi
    if a == b:
        return enc
    if at_most_pow2(b - a, k.limits.max_isolation_bits):
        raise refusal(RefusalCategory.UNDECIDED_PREDICATE, "s-root isolation reached max_isolation_bits")
    k.tick("root_refine")
    m = k.ar.div(k.ar.add(a, b), 2)
    if nf.sign(nf.eval_s(h, m)) == 0:
        return Enclosure(m, m)
    return Enclosure(a, m) if _field_count_open(nf, seq, a, m) == 1 else Enclosure(m, b)


def _inside_algebraic_bounds(nf: NumberField, seq: list, h: tuple, enc: Enclosure, checks: list):
    """``enc`` (refined as far as needed) if its root is certified strictly inside every algebraic bound
    ``(S, want)`` (``want`` is the required side), else ``None``. A root on a bound is never decided by
    intervals alone and ends in ``UNDECIDED_PREDICATE`` at ``max_isolation_bits`` (fail closed)."""
    k = nf.k
    k.tick("root_refine")                           # one checkpoint per candidate root, even when decided at once
    while True:
        undecided = []
        for S, want in checks:
            rel = _side(nf, h, enc, S)
            if rel is None:
                undecided.append(S)
            elif rel != want:
                return None
        if not undecided:
            return enc
        k.tick("root_refine")
        for S in undecided:
            S.refine(k)
        enc = _refine_s_root(nf, seq, h, enc)


def validate_crossing(k: Kernel, F: tuple, G: tuple, u: AlgebraicNumber, s_lo, s_hi, q: tuple | None = None,
                      s_width=Fraction(1, 2 ** 40)) -> CrossingValidation:
    """Validate the resultant root ``u`` in ``Q(u)``: genuine iff ``gcd_s(F(u, s), G(u, s))`` has positive degree
    and a real root in the open interval ``(s_lo, s_hi)``. Each such root is enclosed to ``s_width``.

    ``s_lo``, ``s_hi`` may be rational or ``AlgebraicNumber``: the Sturm count runs on the outer rational
    bounds and each root is then kept only when its enclosure is certified strictly inside every algebraic
    bound (refining both); a root on an algebraic bound refuses ``UNDECIDED_PREDICATE`` at the bit limit."""
    s_lo, s_hi = _s_bound(k, s_lo), _s_bound(k, s_hi)
    outer_lo = s_lo.lo if isinstance(s_lo, AlgebraicNumber) else s_lo
    outer_hi = s_hi.hi if isinstance(s_hi, AlgebraicNumber) else s_hi
    checks = [(S, want) for S, want in ((s_lo, 1), (s_hi, -1)) if isinstance(S, AlgebraicNumber)]
    nf = NumberField(k, u, q)
    h = field_gcd(nf, F, G)
    degree = len(h) - 1 if h else -1
    if degree <= 0:
        return CrossingValidation(u, False, max(degree, 0), (), nf.q, nf.splits)
    # square-free part over the field, then a Sturm count on the open s-interval
    dh = nf.deriv_s(h)
    common = field_gcd(nf, h, dh)
    if len(common) > 1:
        h, _ = nf.divmod_s(h, common)
    seq = _field_sturm(nf, h)
    count = _field_count_open(nf, seq, outer_lo, outer_hi)
    roots = []
    stack = [(outer_lo, outer_hi, count)]
    while stack:
        k.tick("root_refine")
        a, b, c = stack.pop()
        if c == 0:
            continue
        if c == 1 and b - a <= s_width:
            roots.append(Enclosure(a, b))
            continue
        if at_most_pow2(b - a, k.limits.max_isolation_bits):
            raise refusal(RefusalCategory.UNDECIDED_PREDICATE, "s-root isolation reached max_isolation_bits")
        m = k.ar.div(k.ar.add(a, b), 2)
        if nf.sign(nf.eval_s(h, m)) == 0:
            roots.append(Enclosure(m, m))
        stack.append((m, b, _field_count_open(nf, seq, m, b)))
        stack.append((a, m, _field_count_open(nf, seq, a, m)))
    roots.sort(key=lambda e: e.lo)
    if checks:                      # algebraic bounds: keep the roots certified strictly inside them
        kept = (_inside_algebraic_bounds(nf, seq, h, e, checks) for e in roots)
        roots = [e for e in kept if e is not None]
        count = len(roots)
    return CrossingValidation(u, count >= 1, degree, tuple(roots), nf.q, nf.splits)


def curve_crossings(k: Kernel, F: tuple, G: tuple, u_lo, u_hi, s_lo, s_hi,
                    s_width=Fraction(1, 2 ** 40)) -> tuple[tuple, list[CrossingValidation]]:
    """All roots of the raw resultant in the open ``u``-interval, each validated in ``Q(u*)``.

    ``u_lo``, ``u_hi`` may be rational or algebraic (roots are isolated on the outer bounds and filtered by
    exact comparison). Returns ``(R, validations)``; ``R`` identically zero refuses ``UNDECIDED_PREDICATE``
    (a common component). The starting modulus is the square-free part of ``R``, never a quotient by a
    presumed spurious factor."""
    R = resultant_s(k, F, G)
    if not R:
        raise refusal(RefusalCategory.UNDECIDED_PREDICATE, "the resultant vanishes identically (common component)")
    if len(R) < 2:
        return R, []
    roots = isolate_roots(k, R, lower_bound(u_lo), upper_bound(u_hi))
    out = []
    for r in roots:
        k.tick("root_refine")
        if compare(k, r, u_lo) <= 0 or compare(k, r, u_hi) >= 0:
            continue
        out.append(validate_crossing(k, F, G, r, s_lo, s_hi, None, s_width))
    return R, out
