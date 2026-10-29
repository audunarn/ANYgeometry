"""CH3 exact kernel, layer 1: univariate polynomials over Q (design CH3 sections 3.2, 4.3 and 7).

A polynomial is a tuple of ``Fraction`` coefficients from the constant term up, with no trailing zero; the
zero polynomial is ``()``. Every routine takes a ``Kernel`` (the call's ``Meter``, its bit-capped ``Ar`` and the
policy limits) and is written as explicit loops that reserve one unit *before* each element's work (one
coefficient product, one division step, one Bernstein coefficient, one Sturm evaluation step), so every loop is
a checkpoint for the budget, the deadline and the caller's cancellation check (planar design section 7, span
bound, amendment 6d). The only non-preemptible spans are single native ``Fraction``/``int`` operations on values
whose size ``Ar`` has checked against ``max_bits`` before the operation runs.

This module is the kernel's own implementation (D-CH3-6): it does not use ``analytic_roots``,
``exact_levels`` or ``exact_predicates``."""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from ._arith import Ar
from ._errors import CertifiedHydrostaticsRefusal, RefusalCategory
from ._work import Meter

__all__ = ["KernelLimits", "Kernel", "poly_from", "degree", "lc", "padd", "psub", "pneg", "pscale", "pmul",
           "pderiv", "peval", "pdivmod", "pexact_div", "pgcd", "pxgcd", "ppow", "primitive", "positive_primitive",
           "squarefree", "squarefree_factors", "pshift", "pscale_var", "bernstein", "abs_upper", "signed_lower",
           "sturm_sequence", "variations_at", "variations_at_infinity", "coprime_basis", "partial_fractions",
           "pintegrate_definite", "cauchy_bound", "at_most_pow2"]

ZERO = Fraction(0)
ONE = Fraction(1)


# ------------------------------------------------------------------ context

@dataclass(frozen=True)
class KernelLimits:
    """Policy limits that bound every loop of the kernel (D-CH3-7 defaults; order 16 by owner decision 2026-10-21, 32 is opt-in).

    ``max_isolation_bits``: an isolating interval is never refined below width ``2**-max_isolation_bits``;
    ``max_taylor_order``: the highest Taylor order of a term enclosure; ``max_pieces_per_integral``: the
    number of Taylor sub-intervals one integral may use; ``max_refinement_rounds``: recorded for CH3c (the
    kernel itself does not consume it)."""

    max_isolation_bits: int = 4096
    max_taylor_order: int = 16
    max_pieces_per_integral: int = 1024
    max_refinement_rounds: int = 8

    def __post_init__(self) -> None:
        for name in ("max_isolation_bits", "max_taylor_order", "max_pieces_per_integral", "max_refinement_rounds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_taylor_order < 1:
            raise ValueError("max_taylor_order must be at least 1")


class Kernel:
    """One call's metering context: the ``Meter``, its bit-capped arithmetic ``Ar`` and the limits."""

    __slots__ = ("meter", "ar", "limits")

    def __init__(self, meter: Meter, limits: KernelLimits | None = None) -> None:
        if not isinstance(meter, Meter):
            raise TypeError("meter must be a Meter")
        self.meter = meter
        self.ar = Ar(meter)
        self.limits = KernelLimits() if limits is None else limits
        if not isinstance(self.limits, KernelLimits):
            raise TypeError("limits must be a KernelLimits or None")

    def tick(self, kind: str) -> None:
        """One checkpoint: reserve one unit of ``kind`` before the element's work."""
        self.meter.unit(kind)


def refusal(category: RefusalCategory, message: str, sub_reason: str | None = None) -> CertifiedHydrostaticsRefusal:
    return CertifiedHydrostaticsRefusal(category, message, sub_reason=sub_reason)


def at_most_pow2(w: Fraction, e: int, mult: int = 1) -> bool:
    """Exactly ``w <= mult * 2**-e`` for a policy exponent ``e`` of any size, without building ``2**e``
    (a large ``max_isolation_bits`` must not allocate more than the values it is compared with)."""
    if w <= 0:
        return True
    n, d = w.numerator, w.denominator
    bound = mult * d
    if n.bit_length() + e - 1 >= bound.bit_length():
        return False                    # n 2^e >= 2^(bits(n) - 1 + e) >= 2^bits(bound) > bound
    return (n << e) <= bound            # here e < bits(bound): the shifted value is no larger than the operands


# ------------------------------------------------------------------ construction

def poly_from(k: Kernel, coefficients) -> tuple:
    """A normalised polynomial from ints/Fractions (constant term first); each coefficient is size-checked."""
    out = []
    for c in coefficients:
        k.tick("poly_coefficient")
        if isinstance(c, bool) or not isinstance(c, (int, Fraction)):
            raise TypeError("polynomial coefficients must be int or Fraction")
        out.append(k.ar.fit(Fraction(c)))
    return _trim(k, out)


def _trim(k: Kernel, coefficients: list) -> tuple:
    while coefficients and coefficients[-1] == 0:
        k.tick("poly_coefficient")
        coefficients.pop()
    return tuple(coefficients)


def degree(p: tuple) -> int:
    """Degree; ``-1`` for the zero polynomial."""
    return len(p) - 1


def lc(p: tuple) -> Fraction:
    return p[-1] if p else ZERO


# ------------------------------------------------------------------ ring operations

def padd(k: Kernel, p: tuple, q: tuple) -> tuple:
    ar = k.ar
    n = max(len(p), len(q))
    out = []
    for i in range(n):
        k.tick("poly_coefficient")
        a = p[i] if i < len(p) else ZERO
        b = q[i] if i < len(q) else ZERO
        out.append(ar.add(a, b))
    return _trim(k, out)


def psub(k: Kernel, p: tuple, q: tuple) -> tuple:
    ar = k.ar
    n = max(len(p), len(q))
    out = []
    for i in range(n):
        k.tick("poly_coefficient")
        a = p[i] if i < len(p) else ZERO
        b = q[i] if i < len(q) else ZERO
        out.append(ar.sub(a, b))
    return _trim(k, out)


def pneg(k: Kernel, p: tuple) -> tuple:
    out = []
    for c in p:
        k.tick("poly_coefficient")
        out.append(-c)
    return tuple(out)


def pscale(k: Kernel, p: tuple, c) -> tuple:
    if c == 0:
        return ()
    ar = k.ar
    out = []
    for a in p:
        k.tick("poly_coefficient")
        out.append(ar.mul(a, c))
    return tuple(out)


def pmul(k: Kernel, p: tuple, q: tuple) -> tuple:
    """Product; one unit per coefficient product (``len(p) * len(q)`` units)."""
    if not p or not q:
        return ()
    ar = k.ar
    out = [ZERO] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            k.tick("poly_coefficient")
            out[i + j] = ar.add(out[i + j], ar.mul(a, b))
    return _trim(k, out)


def ppow(k: Kernel, p: tuple, e: int) -> tuple:
    """``p ** e``; one unit per iteration (a zero ``p`` multiplies for free, so the loop itself is metered)."""
    out = (ONE,)
    for _ in range(e):
        k.tick("poly_coefficient")
        out = pmul(k, out, p)
    return out


def pderiv(k: Kernel, p: tuple) -> tuple:
    ar = k.ar
    out = []
    for i in range(1, len(p)):
        k.tick("poly_coefficient")
        out.append(ar.mul(p[i], i))
    return _trim(k, out)


def peval(k: Kernel, p: tuple, x) -> Fraction:
    """Horner evaluation at a rational ``x``; one unit per coefficient."""
    ar = k.ar
    acc = ZERO
    for c in reversed(p):
        k.tick("poly_coefficient")
        acc = ar.add(ar.mul(acc, x), c)
    return acc


def pdivmod(k: Kernel, p: tuple, q: tuple) -> tuple[tuple, tuple]:
    """Quotient and remainder over Q; one unit per coefficient update of each division step."""
    if not q:
        raise ZeroDivisionError("polynomial division by zero")
    ar = k.ar
    rem = list(p)
    dq = len(q) - 1
    if len(rem) - 1 < dq:
        return (), tuple(rem)
    quot = [ZERO] * (len(rem) - dq)
    inv = ar.div(ONE, q[-1])
    for shift in range(len(rem) - 1 - dq, -1, -1):
        k.tick("poly_coefficient")
        c = ar.mul(rem[shift + dq], inv)
        quot[shift] = c
        if c != 0:
            for j in range(dq + 1):
                k.tick("poly_coefficient")
                rem[shift + j] = ar.sub(rem[shift + j], ar.mul(c, q[j]))
        rem[shift + dq] = ZERO                      # exact by construction
    return _trim(k, quot), _trim(k, rem[:dq] if dq > 0 else [])


def pexact_div(k: Kernel, p: tuple, q: tuple) -> tuple:
    quotient, remainder = pdivmod(k, p, q)
    if remainder:
        raise ArithmeticError("inexact polynomial division in an exact-division step")
    return quotient


def pmonic(k: Kernel, p: tuple) -> tuple:
    if not p:
        return p
    inv = k.ar.div(ONE, p[-1])
    return pscale(k, p, inv)


def pgcd(k: Kernel, p: tuple, q: tuple) -> tuple:
    """Monic gcd over Q by the Euclidean algorithm (``()`` only when both are zero)."""
    a, b = p, q
    while b:
        k.tick("poly_coefficient")
        _, r = pdivmod(k, a, b)
        a, b = b, pmonic(k, r) if r else r
    return pmonic(k, a)


def pxgcd(k: Kernel, p: tuple, q: tuple) -> tuple[tuple, tuple, tuple]:
    """``(g, s, t)`` with ``s p + t q = g``, ``g`` the monic gcd."""
    r0, r1 = p, q
    s0, s1 = (ONE,), ()
    t0, t1 = (), (ONE,)
    while r1:
        k.tick("poly_coefficient")
        quotient, r2 = pdivmod(k, r0, r1)
        r0, r1 = r1, r2
        s0, s1 = s1, psub(k, s0, pmul(k, quotient, s1))
        t0, t1 = t1, psub(k, t0, pmul(k, quotient, t1))
    if not r0:
        return (), (), ()
    inv = k.ar.div(ONE, r0[-1])
    return pscale(k, r0, inv), pscale(k, s0, inv), pscale(k, t0, inv)


def _content(k: Kernel, p: tuple) -> Fraction:
    """The positive rational ``c`` with ``p / c`` an integer polynomial of coprime coefficients."""
    den = 1
    num = 0
    ar = k.ar
    for c in p:
        k.tick("poly_coefficient")
        g = math.gcd(den, c.denominator)
        den = ar.fit(ar.mul(den // g, c.denominator))
    for c in p:
        k.tick("poly_coefficient")
        num = math.gcd(num, ar.mul(c, den).numerator)
    return Fraction(num, den) if num else ONE


def positive_primitive(k: Kernel, p: tuple) -> tuple:
    """``p`` divided by its positive content (signs kept): integer coefficients without a common factor."""
    if not p:
        return p
    c = _content(k, p)
    return pscale(k, p, k.ar.div(ONE, c))


def primitive(k: Kernel, p: tuple) -> tuple:
    """Primitive integer polynomial with a positive leading coefficient (the canonical form of 3.2)."""
    q = positive_primitive(k, p)
    if q and q[-1] < 0:
        q = pneg(k, q)
    return q


def squarefree(k: Kernel, p: tuple) -> tuple:
    """The primitive square-free part ``p / gcd(p, p')``."""
    if len(p) <= 2:
        return primitive(k, p)
    g = pgcd(k, p, pderiv(k, p))
    return primitive(k, pexact_div(k, p, g))


def squarefree_factors(k: Kernel, p: tuple) -> list[tuple[tuple, int]]:
    """Yun's square-free factorisation: ``[(f_i, i)]`` with ``p = c * prod f_i**i``, each ``f_i`` primitive,
    square-free, of positive degree and pairwise coprime."""
    out: list[tuple[tuple, int]] = []
    if len(p) <= 1:
        return out
    dp = pderiv(k, p)
    a = pgcd(k, p, dp)
    b = pexact_div(k, p, a)
    c = pexact_div(k, dp, a)
    d = psub(k, c, pderiv(k, b))
    i = 1
    while len(b) > 1:
        k.tick("poly_coefficient")
        a = pgcd(k, b, d)
        if len(a) > 1:
            out.append((primitive(k, a), i))
        b = pexact_div(k, b, a)
        c = pexact_div(k, d, a)
        d = psub(k, c, pderiv(k, b))
        i += 1
    return out


# ------------------------------------------------------------------ changes of variable, Bernstein bounds

def pshift(k: Kernel, p: tuple, c) -> tuple:
    """``p(x + c)`` by repeated synthetic division; one unit per inner step."""
    ar = k.ar
    a = list(p)
    n = len(a)
    for i in range(n - 1):
        for j in range(n - 2, i - 1, -1):
            k.tick("poly_coefficient")
            a[j] = ar.add(a[j], ar.mul(c, a[j + 1]))
    return _trim(k, a)


def pscale_var(k: Kernel, p: tuple, s) -> tuple:
    """``p(s x)``."""
    ar = k.ar
    out = []
    power = ONE
    for c in p:
        k.tick("poly_coefficient")
        out.append(ar.mul(c, power))
        power = ar.mul(power, s)
    return _trim(k, out)


def bernstein(k: Kernel, p: tuple, a, b) -> list:
    """Bernstein coefficients of ``p`` over ``[a, b]`` in degree ``max(deg p, 0)``; ``p`` lies in their hull.

    ``p(a + (b - a) x) = sum c_j x^j``; ``b_i = sum_{j <= i} C(i, j) / C(n, j) c_j``. One unit per
    ``bernstein_coefficient`` step."""
    ar = k.ar
    if not p:
        return [ZERO]
    shifted = pshift(k, p, a)
    local = list(pscale_var(k, shifted, ar.sub(b, a)))
    n = len(p) - 1
    local += [ZERO] * (n + 1 - len(local))
    # binomials of row n, then the lower-triangular sums
    binom_n = [ONE]
    for j in range(1, n + 1):
        k.tick("bernstein_coefficient")
        binom_n.append(ar.div(ar.mul(binom_n[-1], n - j + 1), j))
    scaled = []
    for j in range(n + 1):
        k.tick("bernstein_coefficient")
        scaled.append(ar.div(local[j], binom_n[j]))
    out = []
    for i in range(n + 1):
        acc = ZERO
        binom_i = ONE
        for j in range(i + 1):
            k.tick("bernstein_coefficient")
            acc = ar.add(acc, ar.mul(binom_i, scaled[j]))
            binom_i = ar.div(ar.mul(binom_i, i - j), j + 1)
        out.append(acc)
    return out


def abs_upper(k: Kernel, p: tuple, a, b) -> Fraction:
    """An upper bound of ``|p|`` on ``[a, b]``: the largest absolute Bernstein coefficient."""
    best = ZERO
    for c in bernstein(k, p, a, b):
        k.tick("bernstein_coefficient")
        m = -c if c < 0 else c
        if m > best:
            best = m
    return best


def signed_lower(k: Kernel, p: tuple, a, b) -> tuple[int, Fraction]:
    """``(s, m)``: when every Bernstein coefficient over ``[a, b]`` has the strict sign ``s``, ``|p| >= m > 0`` on
    ``[a, b]`` (convex hull property); otherwise ``(0, 0)``."""
    coeffs = bernstein(k, p, a, b)
    sign = 0
    low = None
    for c in coeffs:
        k.tick("bernstein_coefficient")
        s = (c > 0) - (c < 0)
        if s == 0 or (sign and s != sign):
            return 0, ZERO
        sign = s
        m = -c if c < 0 else c
        if low is None or m < low:
            low = m
    return sign, low


# ------------------------------------------------------------------ Sturm sequences

def sturm_sequence(k: Kernel, p: tuple) -> tuple:
    """Sturm sequence of a square-free ``p``: ``p, p', -rem, ...``, each divided by its positive content."""
    seq = [positive_primitive(k, p)]
    if len(p) <= 1:
        return tuple(seq)
    seq.append(positive_primitive(k, pderiv(k, p)))
    while len(seq[-1]) > 1:
        k.tick("sign_count")
        _, r = pdivmod(k, seq[-2], seq[-1])
        if not r:
            break
        seq.append(positive_primitive(k, pneg(k, r)))
    if len(seq[-1]) > 1:
        raise ArithmeticError("Sturm sequence of a polynomial that is not square-free")
    return tuple(seq)


def variations_at(k: Kernel, seq: tuple, x) -> int:
    """Sign variations of the sequence at a rational ``x`` (zeros dropped)."""
    count = 0
    last = 0
    for f in seq:
        k.tick("sign_count")
        v = peval(k, f, x)
        s = (v > 0) - (v < 0)
        if s:
            if last and s != last:
                count += 1
            last = s
    return count


def variations_at_infinity(k: Kernel, seq: tuple, positive: bool) -> int:
    count = 0
    last = 0
    for f in seq:
        k.tick("sign_count")
        if not f:
            continue
        s = 1 if f[-1] > 0 else -1
        if not positive and (len(f) - 1) % 2 == 1:
            s = -s
        if last and s != last:
            count += 1
        last = s
    return count


def cauchy_bound(k: Kernel, p: tuple) -> Fraction:
    """A positive rational strictly larger than the absolute value of every complex root."""
    ar = k.ar
    lead = abs(p[-1])
    best = ZERO
    for c in p[:-1]:
        k.tick("poly_coefficient")
        r = ar.div(abs(c), lead)
        if r > best:
            best = r
    return ar.add(best, ONE)


# ------------------------------------------------------------------ coprime basis and partial fractions

def coprime_basis(k: Kernel, polys) -> tuple[list[tuple], list[tuple[Fraction, list[int]]]]:
    """A pairwise-coprime basis by gcd refinement, and each input as ``(constant, exponents)`` over it.

    Inputs are first split by Yun's square-free factorisation; pairs with a nontrivial gcd ``g`` are
    replaced by ``a / g``, ``g``, ``b / g`` until all are coprime (the total degree drops with every step, so
    the loop ends). Every basis element is primitive with a positive leading coefficient and of positive
    degree. Each input is then divided out exactly; a non-constant leftover is an internal error."""
    items: list[tuple] = []
    for p in polys:
        k.tick("poly_coefficient")
        if not p:
            raise ZeroDivisionError("the zero polynomial has no coprime factorisation")
        for f, _ in squarefree_factors(k, p):
            k.tick("poly_coefficient")
            items.append(f)
    changed = True
    while changed:
        k.tick("poly_coefficient")
        changed = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                k.tick("poly_coefficient")
                g = pgcd(k, items[i], items[j])
                if len(g) > 1:
                    a = pexact_div(k, items[i], g)
                    b = pexact_div(k, items[j], g)
                    rest = []
                    for t in range(len(items)):
                        k.tick("poly_coefficient")
                        if t not in (i, j):
                            rest.append(items[t])
                    for f in (a, g, b):
                        k.tick("poly_coefficient")
                        if len(f) > 1:
                            rest.append(primitive(k, f))
                    items = rest
                    changed = True
                    break
            if changed:
                break
    # remove duplicates (equal primitive polynomials are not coprime, so none can remain; keep the check)
    basis: list[tuple] = []
    for f in items:
        k.tick("poly_coefficient")
        basis.append(f)
    expansions = []
    for p in polys:
        k.tick("poly_coefficient")
        exps = []
        rest = p
        for f in basis:
            e = 0
            while True:
                k.tick("poly_coefficient")
                quotient, remainder = pdivmod(k, rest, f)
                if remainder:
                    break
                rest = quotient
                e += 1
            exps.append(e)
        if len(rest) != 1:
            raise ArithmeticError("coprime basis does not factor an input")
        expansions.append((rest[0], exps))
    return basis, expansions


def partial_fractions(k: Kernel, num: tuple, factors: list[tuple[tuple, int]]) -> tuple[tuple, list]:
    """``num / prod(F_i ** e_i)`` with pairwise-coprime ``F_i`` as ``poly + sum L / F_i ** j``.

    Returns ``(poly_part, [(L, F_i, j)])`` with ``deg L < deg F_i`` and ``L`` nonzero. Coprime splitting by
    the extended Euclidean algorithm, then the ``F_i``-adic expansion of each numerator."""
    den = (ONE,)
    for f, e in factors:
        k.tick("poly_coefficient")
        den = pmul(k, den, ppow(k, f, e))
    poly_part, rem = pdivmod(k, num, den)
    terms = []
    rest_den = den
    for idx, (f, e) in enumerate(factors):
        k.tick("poly_coefficient")
        g_i = ppow(k, f, e)
        other = pexact_div(k, rest_den, g_i)
        if idx == len(factors) - 1:
            a_i = rem
            rem = ()
        else:
            g, _, t = pxgcd(k, g_i, other)
            if g != (ONE,):
                raise ArithmeticError("partial fractions over factors that are not coprime")
            _, a_i = pdivmod(k, pmul(k, rem, t), g_i)
            rem = pexact_div(k, psub(k, rem, pmul(k, a_i, other)), g_i)
        rest_den = other
        # F-adic expansion: a_i = sum c_j f**j, c_j / f**(e - j)
        j = 0
        cur = a_i
        while cur:
            k.tick("poly_coefficient")
            quotient, c_j = pdivmod(k, cur, f)
            if c_j:
                terms.append((c_j, f, e - j))
            cur = quotient
            j += 1
        if j > e:
            raise ArithmeticError("partial-fraction numerator of too high degree")
    if rem:
        raise ArithmeticError("partial fractions left a remainder")
    return poly_part, terms


def pintegrate_definite(k: Kernel, p: tuple, a, b) -> Fraction:
    """``int_a^b p`` exactly (rational ends)."""
    ar = k.ar
    anti = [ZERO]
    for i, c in enumerate(p):
        k.tick("poly_coefficient")
        anti.append(ar.div(c, i + 1))
    anti_t = tuple(anti)
    return ar.sub(peval(k, anti_t, b), peval(k, anti_t, a))
