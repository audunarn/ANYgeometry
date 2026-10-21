"""CH3 exact kernel, layer 3: rigorous enclosures of 1-D rational integrals and the ``1/|n|`` bounds (design CH3
sections 4.3 and 4.4).

Reduction: the denominator's factors are refined to a pairwise-coprime basis (``coprime_basis``), and the
integrand is split by partial fractions into an exact polynomial part and terms ``L / Q^k`` with
``deg L < deg Q``.

Term enclosure on a rational piece ``[a, b]`` (``c`` the midpoint, ``h`` the half width, ``g = Q^{-k}``)::

    g^{(j)} = P_j / Q^{k+j},  P_0 = 1,  P_{j+1} = P_j' Q - (k + j) P_j Q'
    main = int_a^b L t_K   (t_K the order-K Taylor polynomial of g at c; exact)
    B_g  = max|P_{K+1}| / (min|Q|)^{K+1+k} / (K+1)!     (Bernstein bounds over [a, b]; Q's Bernstein
                                                           coefficients must be one-signed there)
    rho  = B_g * L_max * 2 h^{K+2} / (K + 2)            (L_max from |Bernstein coefficients of L|)
    T in [main - rho, main + rho]

The bound factors are rounded *upward* to 64-bit-mantissa dyadics (each rounding enlarges ``rho``, so the bound
stays rigorous) and the term enclosure is rounded *outward* to a dyadic grid finer than a sixteenth of its
allowance, so sums of many enclosures keep small denominators. An algebraic end ``alpha`` contributes the end
enclosure ``[-e, e]``, ``e = (hi - lo) max|Num| / min|Den|`` over its isolating interval (the whole integrand's
bound; refined by bisection), and the inner piece starts at the rational interval end.

Allocation: the target ``tau`` of an integral is split equally over its pieces; in each piece the algebraic
ends draw first (each refined until its width is at most a quarter of the piece's share), the rest is split
equally over the partial-fraction terms; orders run 8, 16 by default (32 only if opted in via ``max_taylor_order``; capped by it) before a term
piece is bisected, and each child gets half of its parent's allowance. The total width is the sum of every term
and end width (the enclosures are added exactly). Failure within ``max_pieces_per_integral`` refuses
``ACCURACY_UNACHIEVABLE`` with the achieved width."""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable

from ._ch3_algebraic import AlgebraicNumber, Enclosure, compare, lower_bound, upper_bound
from ._ch3_poly import (Kernel, ONE, ZERO, abs_upper, at_most_pow2, coprime_basis, partial_fractions, pderiv, pintegrate_definite,
                        pmul, pscale, pshift, psub, refusal, signed_lower)
from ._errors import RefusalCategory

__all__ = ["round_up", "round_down", "grid_floor", "grid_ceil", "grid_exponent", "taylor_term",
           "enclose_term", "enclose_rational_integral", "enclose_pieces", "IntegralEnclosure", "AllocationRecord",
           "divide_by_norm", "NormQuotient", "MANTISSA_BITS"]

MANTISSA_BITS = 64


# ------------------------------------------------------------------ directed dyadic rounding

def round_up(k: Kernel, x: Fraction) -> Fraction:
    """The least dyadic ``m 2^-s`` with a ``MANTISSA_BITS``-bit mantissa that is ``>= x`` (``x >= 0``)."""
    if x <= 0:
        if x < 0:
            raise ValueError("round_up expects a non-negative value")
        return ZERO
    k.tick("interval_op")
    n, d = x.numerator, x.denominator
    s = MANTISSA_BITS - (n.bit_length() - d.bit_length())
    if s >= 0:
        k.ar.guard(n.bit_length() + s)
        m = -((-(n << s)) // d)
        return k.ar.fit(Fraction(m, 1 << s))
    k.ar.guard(d.bit_length() - s)
    m = -((-n) // (d << -s))
    return k.ar.fit(Fraction(m << -s))


def round_down(k: Kernel, x: Fraction) -> Fraction:
    """The greatest dyadic with a ``MANTISSA_BITS``-bit mantissa that is ``<= x`` (``x > 0``; may be 0 never)."""
    if x <= 0:
        raise ValueError("round_down expects a positive value")
    k.tick("interval_op")
    n, d = x.numerator, x.denominator
    s = MANTISSA_BITS - (n.bit_length() - d.bit_length())
    if s >= 0:
        k.ar.guard(n.bit_length() + s)
        m = (n << s) // d
        return k.ar.fit(Fraction(m, 1 << s))
    k.ar.guard(d.bit_length() - s)
    m = n // (d << -s)
    return k.ar.fit(Fraction(m << -s))


def grid_exponent(k: Kernel, allowance: Fraction, divisor: int = 16) -> int:
    """The least ``g`` with ``2^-g <= allowance / divisor`` (``allowance > 0``)."""
    if allowance <= 0:
        raise ValueError("allowance must be positive")
    target = k.ar.div(allowance, divisor)
    n, d = target.numerator, target.denominator
    g = d.bit_length() - n.bit_length()
    while True:
        k.tick("interval_op")
        # 2^-g <= n/d  <=>  d <= n 2^g
        ok = (d <= (n << g)) if g >= 0 else ((d << -g) <= n)
        if ok:
            prev = g - 1
            ok_prev = (d <= (n << prev)) if prev >= 0 else ((d << -prev) <= n)
            if not ok_prev:
                return g
            g = prev
        else:
            g += 1


def grid_floor(k: Kernel, x: Fraction, g: int) -> Fraction:
    k.tick("interval_op")
    if g >= 0:
        k.ar.guard(abs(x.numerator).bit_length() + g)
        return k.ar.fit(Fraction((x.numerator << g) // x.denominator, 1 << g))
    k.ar.guard(x.denominator.bit_length() - g)
    return k.ar.fit(Fraction((x.numerator // (x.denominator << -g)) << -g))


def grid_ceil(k: Kernel, x: Fraction, g: int) -> Fraction:
    return -grid_floor(k, -x, g)


def _mul_up(k: Kernel, x: Fraction, y: Fraction) -> Fraction:
    return round_up(k, k.ar.mul(x, y))


# ------------------------------------------------------------------ the term enclosure

def taylor_term(k: Kernel, L: tuple, Q: tuple, kpow: int, a: Fraction, b: Fraction, order: int,
                _cache: dict | None = None):
    """``(main, rho)`` of ``int_a^b L / Q^kpow`` at Taylor order ``order`` (rational ``a < b``), or ``None``
    when ``Q``'s Bernstein coefficients over ``[a, b]`` are not one-signed (the bound is then unavailable).

    ``main`` is exact; ``rho`` is the complete remainder bound of design 4.3, rounded upward."""
    ar = k.ar
    if _cache is None:
        _cache = {}
    if "Qs" not in _cache:
        c = ar.div(ar.add(a, b), 2)
        h = ar.div(ar.sub(b, a), 2)
        Qs = pshift(k, Q, c)
        Ls = pshift(k, L, c)
        sign, q_min = signed_lower(k, Qs, -h, h)
        _cache.update(c=c, h=h, Qs=Qs, Ls=Ls, dQs=pderiv(k, Qs), sign=sign, q_min=q_min, P=[(ONE,)],
                      L_max=abs_upper(k, Ls, -h, h) if sign else None)
    if not _cache["sign"]:
        return None
    h, Qs, Ls, dQs, P = _cache["h"], _cache["Qs"], _cache["Ls"], _cache["dQs"], _cache["P"]
    while len(P) < order + 2:                                   # P_0 .. P_{order+1}
        j = len(P) - 1
        k.tick("poly_coefficient")
        P.append(psub(k, pmul(k, pderiv(k, P[j]), Qs), pscale(k, pmul(k, P[j], dQs), kpow + j)))
    # Taylor coefficients g_j = P_j(0) / (Qs(0)^{kpow+j} j!)
    q0 = Qs[0]
    inv_q0 = ar.div(ONE, q0)
    scale = ONE
    for _ in range(kpow):
        k.tick("poly_coefficient")
        scale = ar.mul(scale, inv_q0)
    g = []
    fact = ONE
    for j in range(order + 1):
        k.tick("poly_coefficient")
        if j:
            fact = ar.mul(fact, j)
            scale = ar.mul(scale, inv_q0)
        pj0 = P[j][0] if P[j] else ZERO
        g.append(ar.div(ar.mul(pj0, scale), fact))
    # main = sum_{i,j} l_i g_j int_{-h}^{h} t^{i+j} dt, the odd powers vanish
    main = ZERO
    h_pow = [h]                                                  # h^{m+1}
    for i, li in enumerate(Ls):
        for j, gj in enumerate(g):
            k.tick("poly_coefficient")
            m = i + j
            while len(h_pow) <= m:
                h_pow.append(ar.mul(h_pow[-1], h))
            if m % 2 == 0 and li and gj:
                main = ar.add(main, ar.div(ar.mul(ar.mul(li, gj), ar.mul(2, h_pow[m])), m + 1))
    # rho = max|P_{K+1}| (1/min|Q|)^{K+1+k} / (K+1)! * L_max * 2 h^{K+2} / (K+2), every factor rounded up
    p_max = abs_upper(k, P[order + 1], -h, h)
    inv_qmin = round_up(k, ar.div(ONE, _cache["q_min"]))
    bound = round_up(k, p_max)
    for _ in range(order + 1 + kpow):
        k.tick("interval_op")
        bound = _mul_up(k, bound, inv_qmin)
    for j in range(2, order + 2):
        k.tick("interval_op")
        bound = round_up(k, ar.div(bound, j))
    bound = _mul_up(k, bound, _cache["L_max"])
    for _ in range(order + 2):
        k.tick("interval_op")
        bound = _mul_up(k, bound, h)
    bound = round_up(k, ar.div(ar.mul(bound, 2), order + 2))
    return main, bound


def _orders(k: Kernel) -> list[int]:
    top = k.limits.max_taylor_order
    out = [o for o in (8, 16, 32) if o < top]
    out.append(top)
    return out


@dataclass
class AllocationRecord:
    """Untrusted bookkeeping of one integral: per term piece and end, its allowance and achieved width."""

    tau: Fraction
    pieces: list = field(default_factory=list)       # (term index, a, b, depth, order, allowance, width)
    ends: list = field(default_factory=list)         # (which, allowance, width)
    taylor_pieces: int = 0

    @property
    def total_width(self) -> Fraction:
        return sum((p[6] for p in self.pieces), ZERO) + sum((e[2] for e in self.ends), ZERO)

    @property
    def total_allowance(self) -> Fraction:
        return sum((p[5] for p in self.pieces), ZERO) + sum((e[1] for e in self.ends), ZERO)


def enclose_term(k: Kernel, L: tuple, Q: tuple, kpow: int, a: Fraction, b: Fraction, allowance: Fraction,
                 record: AllocationRecord | None = None, term_index: int = 0) -> Enclosure:
    """An enclosure of ``int_a^b L / Q^kpow`` (rational ``a < b``) of width ``<= allowance``."""
    if not a < b:
        raise ValueError("enclose_term needs a < b")
    if allowance <= 0:
        raise ValueError("allowance must be positive")
    record = record if record is not None else AllocationRecord(allowance)
    orders = _orders(k)
    total = Enclosure(ZERO, ZERO)
    stack = [(a, b, allowance, 0)]
    if record.taylor_pieces + 1 > k.limits.max_pieces_per_integral:
        raise refusal(RefusalCategory.ACCURACY_UNACHIEVABLE,
                      f"the integral needs more than max_pieces_per_integral={k.limits.max_pieces_per_integral} "
                      f"Taylor pieces (one per partial-fraction term at least)")
    record.taylor_pieces += 1
    while stack:
        k.tick("piece_split")
        x, y, share, depth = stack.pop()
        cache: dict = {}
        best = None
        done = None
        for order in orders:
            result = taylor_term(k, L, Q, kpow, x, y, order, cache)
            if result is None:
                break
            main, rho = result
            g = grid_exponent(k, share)
            lo = grid_floor(k, k.ar.sub(main, rho), g)
            hi = grid_ceil(k, k.ar.add(main, rho), g)
            width = hi - lo
            best = width if best is None or width < best else best
            if width <= share:
                done = (Enclosure(lo, hi), order)
                break
        if done is not None:
            enc, order = done
            record.pieces.append((term_index, x, y, depth, order, share, enc.width))
            total = total.add(k, enc)
            continue
        if record.taylor_pieces + 1 > k.limits.max_pieces_per_integral:
            achieved = "unbounded (denominator not one-signed)" if best is None else f"{float(best):.3e}"
            raise refusal(RefusalCategory.ACCURACY_UNACHIEVABLE,
                          f"term enclosure needs more than max_pieces_per_integral="
                          f"{k.limits.max_pieces_per_integral} pieces; achieved width {achieved} "
                          f"for allowance {float(share):.3e} on a piece of width {float(y - x):.3e}")
        record.taylor_pieces += 1
        mid = k.ar.div(k.ar.add(x, y), 2)
        half = k.ar.div(share, 2)
        stack.append((mid, y, half, depth + 1))
        stack.append((x, mid, half, depth + 1))
    return total


# ------------------------------------------------------------------ whole integrals

@dataclass(frozen=True)
class IntegralEnclosure:
    enclosure: Enclosure
    exact: Fraction | None              # the exact value when no algebraic end and no enclosed term entered
    record: AllocationRecord


def _decompose(k: Kernel, num: tuple, dens: list) -> tuple:
    polys = []
    for f, e in dens:
        k.tick("poly_coefficient")
        if e < 1:
            raise ValueError("denominator exponents must be positive")
        polys.append(f)
    basis, expansions = coprime_basis(k, polys)
    constant = ONE
    exps = [0] * len(basis)
    for (f, e), (c, ex) in zip(dens, expansions):
        k.tick("poly_coefficient")
        for _ in range(e):
            k.tick("poly_coefficient")
            constant = k.ar.mul(constant, c)
        for i in range(len(basis)):
            k.tick("poly_coefficient")
            exps[i] += ex[i] * e
    factors = [(basis[i], exps[i]) for i in range(len(basis)) if exps[i] > 0]
    scaled = pscale(k, num, k.ar.div(ONE, constant))
    poly_part, terms = partial_fractions(k, scaled, factors)
    return poly_part, terms, factors, constant


def _end_bound(k: Kernel, num: tuple, factors: list, constant: Fraction, lo: Fraction, hi: Fraction):
    """``(hi - lo) max|Num| / min|Den|`` over ``[lo, hi]`` rounded up, or ``None`` if the denominator's
    factors are not certified one-signed there."""
    ar = k.ar
    den_low = round_down(k, abs(constant)) if constant else None
    if den_low is None:
        return None
    for f, e in factors:
        k.tick("interval_op")
        sign, low = signed_lower(k, f, lo, hi)
        if not sign:
            return None
        low = round_down(k, low)
        for _ in range(e):
            k.tick("interval_op")
            den_low = round_down(k, ar.mul(den_low, low))
    num_max = round_up(k, abs_upper(k, num, lo, hi))
    return round_up(k, ar.div(ar.mul(num_max, ar.sub(hi, lo)), den_low))


def compare_bounds(a, b) -> bool:
    """Whether the isolating intervals of ``a < b`` still overlap (``hi(a) >= lo(b)``)."""
    return upper_bound(a) >= lower_bound(b)


def enclose_rational_integral(k: Kernel, num: tuple, dens: list, a, b, tau: Fraction,
                              record: AllocationRecord | None = None) -> IntegralEnclosure:
    """``int_a^b num / prod(f ** e for f, e in dens)`` enclosed to width ``<= tau``.

    ``a``, ``b`` are ``Fraction``s or ``AlgebraicNumber``s; the denominator must not vanish on the closed
    interval (Taylor bounds need one-signed Bernstein coefficients; a vanishing denominator ends in
    ``ACCURACY_UNACHIEVABLE``)."""
    tau = Fraction(tau)
    if tau <= 0:
        raise ValueError("tau must be positive")
    record = record if record is not None else AllocationRecord(tau)
    order = compare(k, a, b)
    if order == 0:
        return IntegralEnclosure(Enclosure(ZERO, ZERO), ZERO, record)
    if order > 0:
        inner = enclose_rational_integral(k, num, dens, b, a, tau, record)
        neg = inner.enclosure.neg(k)
        return IntegralEnclosure(neg, None if inner.exact is None else -inner.exact, record)
    poly_part, terms, factors, constant = _decompose(k, num, dens)
    end_share = k.ar.div(tau, 4)
    end_enclosures = []
    while compare_bounds(a, b):             # separate the isolating intervals (a < b is certain)
        k.tick("root_refine")
        for x in (a, b):
            if isinstance(x, AlgebraicNumber) and not x.is_rational:
                x.refine(k)
    inner = {"start": upper_bound(a), "end": lower_bound(b)}
    for which, x in (("start", a), ("end", b)):
        if not (isinstance(x, AlgebraicNumber) and not x.is_rational):
            continue
        k.tick("piece_split")
        while True:
            # the inner end is snapped outward to a dyadic grid at the interval's own scale (short numbers for
            # the Taylor pieces); the end piece then spans [lo, snapped] or [snapped, hi], at most twice the width
            g = grid_exponent(k, k.ar.sub(x.hi, x.lo), 1)
            if which == "start":
                snapped = grid_ceil(k, x.hi, g)
                span = (x.lo, snapped)
                ordered = snapped < inner["end"]
            else:
                snapped = grid_floor(k, x.lo, g)
                span = (snapped, x.hi)
                ordered = inner["start"] < snapped
            e = _end_bound(k, num, factors, constant, span[0], span[1]) if ordered else None
            if e is not None and 2 * e <= end_share:
                inner[which] = snapped
                break
            if at_most_pow2(x.hi - x.lo, k.limits.max_isolation_bits, 2):
                achieved = "unbounded" if e is None else f"{float(2 * e):.3e}"
                raise refusal(RefusalCategory.ACCURACY_UNACHIEVABLE,
                              f"end enclosure at max_isolation_bits: achieved width {achieved}")
            x.refine(k)
            if x.is_rational:
                inner[which] = x.lo
                e = ZERO
                break
        if e:
            end_enclosures.append(Enclosure(-e, e))
            record.ends.append((which, end_share, 2 * e))
    a_in, b_in = inner["start"], inner["end"]
    used = ZERO
    for enc in end_enclosures:
        k.tick("interval_op")
        used = k.ar.add(used, enc.width)
    exact_part = pintegrate_definite(k, poly_part, a_in, b_in)
    total = Enclosure(exact_part, exact_part)
    for enc in end_enclosures:
        total = total.add(k, enc)
    if terms:
        share = k.ar.div(k.ar.sub(tau, used), len(terms))
        for index, (L, Q, j) in enumerate(terms):
            k.tick("interval_op")
            total = total.add(k, enclose_term(k, L, Q, j, a_in, b_in, share, record, index))
    algebraic_end = any(isinstance(x, AlgebraicNumber) and not x.is_rational for x in (a, b))
    exact = exact_part if not terms and not end_enclosures and not algebraic_end else None
    return IntegralEnclosure(total, exact, record)


def enclose_pieces(k: Kernel, pieces: list, tau: Fraction) -> tuple[Enclosure, list[AllocationRecord]]:
    """Sum of ``enclose_rational_integral`` over ``pieces = [(num, dens, a, b)]``, ``tau`` split equally."""
    tau = Fraction(tau)
    if not pieces:
        return Enclosure(ZERO, ZERO), []
    share = tau / len(pieces)
    total = Enclosure(ZERO, ZERO)
    records = []
    for num, dens, a, b in pieces:
        k.tick("piece_split")
        result = enclose_rational_integral(k, num, dens, a, b, share)
        records.append(result.record)
        total = total.add(k, result.enclosure)
    return total, records


# ------------------------------------------------------------------ 1/|n| (design 4.4)

@dataclass(frozen=True)
class NormQuotient:
    core: Enclosure              # contains numerator / |n|; width <= a / 3
    claim: Enclosure             # [core.lo - w, core.hi + w], width <= a
    numerator: Enclosure
    m: int
    s_lo: Fraction
    s_hi: Fraction
    tau_numerator: Fraction


def _sqrt_bounds(k: Kernel, n2: Fraction, m: int) -> Fraction:
    """``s_lo = isqrt(floor(|n|^2 2^{2m})) / 2^m``; the radicand is stored and size-checked before ``isqrt``."""
    k.tick("interval_op")
    num, den = n2.numerator, n2.denominator
    k.ar.guard(num.bit_length() + 2 * m)
    radicand = k.ar.fit((num << (2 * m)) // den)
    return k.ar.fit(Fraction(k.ar.isqrt(radicand), 1 << m))


def divide_by_norm(k: Kernel, n2, numerator: Callable[[Fraction], Enclosure], a) -> NormQuotient:
    """Enclose ``N / |n|`` for ``|n|^2 = n2`` with the square-root allocation in output units (design 4.4).

    ``numerator(tau_N)`` must return an enclosure of ``N`` of width ``<= tau_N``; ``tau_N = (a/6) s_lo``, and
    ``m`` is raised until ``max(|N_lo|, |N_hi|) 2^-m / s_lo^2 <= a/6``, so the core is ``<= a/3`` wide and
    the published claim ``<= a``."""
    ar = k.ar
    n2, a = Fraction(n2), Fraction(a)
    if n2 <= 0 or a <= 0:
        raise ValueError("|n|^2 and a must be positive")
    # the least m with s_lo > 0: floor(n2 4^m) >= 1
    m = max(0, (n2.denominator.bit_length() - n2.numerator.bit_length() + 2) // 2)
    s_lo = _sqrt_bounds(k, n2, m)
    while s_lo == 0:
        m += 1
        s_lo = _sqrt_bounds(k, n2, m)
    tau_n = ar.mul(ar.div(a, 6), s_lo)
    enc = numerator(tau_n)
    if not isinstance(enc, Enclosure) or enc.width > tau_n:
        raise ValueError("the numerator enclosure does not meet its target")
    n_max = max(abs(enc.lo), abs(enc.hi))
    target = ar.div(a, 6)
    if n_max:
        # a logarithmic first guess: 2^m >= n_max / (target s_lo^2), from bit lengths, then confirm exactly
        ratio = ar.div(n_max, ar.mul(target, ar.mul(s_lo, s_lo)))
        guess = ratio.numerator.bit_length() - ratio.denominator.bit_length()
        m = max(m, guess)
        while True:
            k.tick("interval_op")
            s_lo = _sqrt_bounds(k, n2, m)
            term = ar.div(n_max, ar.mul(ar.mul(s_lo, s_lo), Fraction(1 << m)))
            if term <= target:
                break
            m += 1
    else:
        s_lo = _sqrt_bounds(k, n2, m)
    s_hi = ar.add(s_lo, Fraction(1, 1 << m))
    lo = min(ar.div(enc.lo, s_lo), ar.div(enc.lo, s_hi))
    hi = max(ar.div(enc.hi, s_lo), ar.div(enc.hi, s_hi))
    core = Enclosure(lo, hi)
    width = core.width
    if width > ar.div(a, 3):
        raise ArithmeticError("the 1/|n| core exceeds a/3 (allocation invariant broken)")
    claim = Enclosure(ar.sub(lo, width), ar.add(hi, width))
    return NormQuotient(core, claim, enc, m, s_lo, s_hi, tau_n)
