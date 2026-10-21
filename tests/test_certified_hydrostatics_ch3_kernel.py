"""CH3b: the exact kernel of certified hydrostatics for curved shells (design CERTIFIED_HYDROSTATICS_CH3_DESIGN.md,
section 9, CH3b row)."""
from __future__ import annotations

import math
import random
import time
from fractions import Fraction as F

import pytest

from anygeometry.certified_hydrostatics import CertifiedHydrostaticsRefusal, RefusalCategory, WorkBudget
from anygeometry.certified_hydrostatics._ch3_algebraic import (AlgebraicNumber, Enclosure, algebraic_equal, compare,
                                                              count_open, isolate_roots, sign_at)
from anygeometry.certified_hydrostatics._ch3_integrate import (AllocationRecord, divide_by_norm, enclose_pieces,
                                                              enclose_rational_integral, enclose_term, taylor_term)
from anygeometry.certified_hydrostatics._ch3_poly import (Kernel, KernelLimits, bernstein, coprime_basis, padd, pdivmod,
                                                         peval, pgcd, pmul, poly_from, ppow, partial_fractions,
                                                         pshift, sturm_sequence, variations_at)
from anygeometry.certified_hydrostatics._ch3_range import graph_in_range
from anygeometry.certified_hydrostatics._ch3_resultant import (bivariate, curve_crossings, resultant_s,
                                                              validate_crossing)
from anygeometry.certified_hydrostatics._work import Meter

mpmath = pytest.importorskip("mpmath")
C = RefusalCategory
SEED = 20261020


def kern(max_bits=4096, limits=None, budget=None, cancel=None):
    budget = budget if budget is not None else WorkBudget(None, max_bits)
    return Kernel(Meter(budget, cancel), limits)


K0 = kern()


def P(*c):
    return poly_from(K0, c)


def mp_poly(p, x):
    acc = mpmath.mpf(0)
    for c in reversed(p):
        acc = acc * x + mpmath.mpf(c.numerator) / c.denominator
    return acc


def mp_integral(num, dens, a, b):
    with mpmath.workdps(60):
        def f(x):
            d = mpmath.mpf(1)
            for q, e in dens:
                d *= mp_poly(q, x) ** e
            return mp_poly(num, x) / d
        return mpmath.quad(f, [mpmath.mpf(a.numerator) / a.denominator, mpmath.mpf(b.numerator) / b.denominator])


def contains(enc, ref):
    with mpmath.workdps(60):
        return mpmath.mpf(enc.lo.numerator) / enc.lo.denominator <= ref <= \
            mpmath.mpf(enc.hi.numerator) / enc.hi.denominator


# ------------------------------------------------------------------ polynomials, coprime basis, partial fractions

def test_polynomial_ring_and_division():
    k = kern()
    p, q = P(1, 2, 3), P(-1, 1)
    assert pmul(k, p, q) == P(-1, -1, -1, 3)
    quot, rem = pdivmod(k, pmul(k, p, q), q)
    assert quot == p and rem == ()
    quot, rem = pdivmod(k, P(1, 0, 0, 1), P(1, 1))
    assert padd(k, pmul(k, quot, P(1, 1)), rem) == P(1, 0, 0, 1)
    assert pgcd(k, pmul(k, P(-2, 0, 1), P(1, 1)), pmul(k, P(-2, 0, 1), P(3, 1))) == P(-2, 0, 1)
    assert pshift(k, P(0, 0, 1), F(1)) == P(1, 2, 1)
    # Bernstein coefficients enclose the range: u(1-u) on [0, 1] -> (0, 1/2, 0)
    assert bernstein(k, P(0, 1, -1), F(0), F(1)) == [0, F(1, 2), 0]


def test_coprime_basis_and_partial_fractions_recombine_exactly():
    k = kern()
    W, D = P(1, -1, 2), P(2, 1)                      # coprime
    A = pmul(k, W, P(5, 1))                          # shares W
    basis, expansions = coprime_basis(k, [pmul(k, ppow(k, W, 2), D), A, ppow(k, D, 3)])
    for f in basis:
        for g in basis:
            if f is not g:
                assert len(pgcd(k, f, g)) == 1
    for (c, exps), original in zip(expansions, [pmul(k, ppow(k, W, 2), D), A, ppow(k, D, 3)]):
        rebuilt = (c,)
        for f, e in zip(basis, exps):
            rebuilt = pmul(k, rebuilt, ppow(k, f, e))
        assert rebuilt == original
    num = P(3, -1, 4, 0, 2, 7, 1, 1)
    factors = [(W, 3), (D, 2)]
    poly, terms = partial_fractions(k, num, factors)
    den = pmul(k, ppow(k, W, 3), ppow(k, D, 2))
    total = pmul(k, poly, den)
    for L, Q, j in terms:
        assert len(L) < len(Q)
        total = padd(k, total, pmul(k, L, pdivmod(k, den, ppow(k, Q, j))[0]))
    assert total == num


# ------------------------------------------------------------------ algebraic numbers: identity and order

def test_identity_by_poly_and_ordinal_and_three_counts():
    k = kern()
    p = P(6, 0, -5, 0, 1)                            # (x^2 - 2)(x^2 - 3)
    roots = isolate_roots(k, p)
    assert [r.ordinal for r in roots] == [0, 1, 2, 3]
    assert [round(float(r.lo + r.hi) / 2, 0) for r in roots] == [-2, -1, 1, 2]
    sqrt2 = AlgebraicNumber.from_interval(k, p, 1, F(8, 5))
    assert sqrt2.ordinal == 2
    sqrt3 = AlgebraicNumber.from_interval(k, P(-3, 0, 1), F(3, 2), 2)
    # the overlapping intervals (1, 8/5) and (3/2, 2): I = (3/2, 8/5) holds no common root, so distinct
    assert count_open(k, pgcd(k, p, P(-3, 0, 1)), F(3, 2), F(8, 5)) == 0
    assert not algebraic_equal(k, sqrt2, sqrt3)
    assert compare(k, sqrt2, sqrt3) == -1 and compare(k, sqrt3, sqrt2) == 1
    assert sqrt2.hi <= sqrt3.lo                       # certified by refining to disjoint intervals
    # same number, different polynomials: the three counts are 1
    other = AlgebraicNumber.from_interval(k, P(-2, 0, 1), F(5, 4), F(3, 2))
    assert algebraic_equal(k, AlgebraicNumber.from_interval(k, p, 1, F(8, 5)), other)
    assert compare(k, other, AlgebraicNumber.from_interval(k, p, 1, F(8, 5))) == 0
    # same polynomial: ordinals decide
    assert algebraic_equal(k, roots[2], AlgebraicNumber.from_interval(k, p, F(13, 10), F(3, 2)))
    assert not algebraic_equal(k, roots[2], roots[3])
    # (x^2-2)(x^2-3) ordinal 3 versus x^2 - 3 on (3/2, 2): equal
    assert algebraic_equal(k, roots[3], sqrt3)
    # rational points
    half = AlgebraicNumber.from_interval(k, P(-1, 2), F(1, 2), F(1, 2))
    assert half.is_rational and algebraic_equal(k, half, F(1, 2)) and compare(k, half, sqrt2) == -1
    assert compare(k, F(3, 2), sqrt2) == 1 and compare(k, F(7, 5), sqrt2) == -1


def test_an_interval_that_does_not_isolate_is_refused():
    k = kern()
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        AlgebraicNumber.from_interval(k, P(6, 0, -5, 0, 1), 1, 2)          # holds sqrt2 and sqrt3
    assert info.value.category is C.UNDECIDED_PREDICATE


def test_single_root_sign_tests():
    k = kern()
    sqrt2 = AlgebraicNumber.from_interval(k, P(-2, 0, 1), 1, 2)
    assert sign_at(k, P(-2, 0, 1), sqrt2) == 0
    assert sign_at(k, pmul(k, P(-2, 0, 1), P(5, 1)), sqrt2) == 0       # gcd with a root in the interval
    assert sign_at(k, P(F(-141421356, 100000000), 1), sqrt2) == 1      # x - 1.41421356 > 0
    assert sign_at(k, P(F(-141421357, 100000000), 1), sqrt2) == -1
    assert sign_at(k, P(-3, 0, 1), sqrt2) == -1


def test_isolation_refuses_beyond_max_isolation_bits():
    # two roots 2^-30 apart cannot be separated with 20 isolation bits
    k = kern(limits=KernelLimits(max_isolation_bits=20))
    eps = F(1, 2 ** 30)
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        isolate_roots(k, pmul(k, P(-F(1, 3), 1), P(-F(1, 3) - eps, 1)))
    assert info.value.category is C.UNDECIDED_PREDICATE


# ------------------------------------------------------------------ the term integrator

def test_finding_2_term_rho_bounds_the_actual_error():
    """The review's term: L = 10^6, Q = 1 + u on [0, 1], K = 8: rho = 195.3125 >= error 1.133."""
    k = kern()
    main, rho = taylor_term(k, P(10 ** 6), P(1, 1), 1, F(0), F(1), 8)
    error = abs(float(main) - 1e6 * math.log(2))
    assert 1.13 < error < 1.14
    assert F(1953125, 10000) <= rho <= F(1953125, 10000) * (1 + F(1, 10 ** 15))
    assert rho >= F(error)
    enc = enclose_term(k, P(10 ** 6), P(1, 1), 1, F(0), F(1), F(1, 10 ** 12))
    with mpmath.workdps(60):
        ref = mpmath.mpf(10) ** 6 * mpmath.log(2)
    assert enc.width <= F(1, 10 ** 12) and contains(enc, ref)


def _random_term(rng):
    while True:
        deg = rng.choice((1, 2))
        Q = P(*[rng.randint(-9, 9) for _ in range(deg + 1)])
        if len(Q) != deg + 1:
            continue
        a = F(rng.randint(-2000, 2000), 1000)
        b = a + F(rng.randint(50, 1000), 1000)
        if count_open(K0, Q, a - F(1, 10), b + F(1, 10)) or peval(K0, Q, a - F(1, 10)) == 0:
            continue
        if min(abs(float(peval(K0, Q, a + (b - a) * i / 20))) for i in range(21)) < 0.25:
            continue
        L = P(*[rng.randint(-20, 20) for _ in range(deg)])
        if not L:
            continue
        return L, Q, a, b


def test_fifty_random_term_enclosures_contain_mpmath_references():
    rng = random.Random(SEED)
    worst_bits = 0
    widths = []
    for case in range(50):
        L, Q, a, b = _random_term(rng)
        kpow = rng.randint(1, 5)
        tau = F(1, 10 ** rng.randint(6, 20))
        budget = WorkBudget(None, 4096)
        k = kern(budget=budget)
        enc = enclose_rational_integral(k, L, [(Q, kpow)], a, b, tau).enclosure
        ref = mp_integral(L, [(Q, kpow)], a, b)
        assert enc.width <= tau, case
        assert contains(enc, ref), (case, L, Q, kpow, a, b)
        worst_bits = max(worst_bits, budget.counts()["max_bits_seen"])
        widths.append(enc.width / tau)
    print(f"\nCH3b random terms: max bits {worst_bits}, max width/tau {float(max(widths)):.3f}")
    assert worst_bits <= 4096


def test_composite_integrands_with_algebraic_ends_contain_mpmath_references():
    """int Num / (W^3 D^3) between algebraic corners, through the coprime basis and partial fractions."""
    rng = random.Random(SEED + 1)
    W = P(1, F(-1, 2), F(1, 2))                       # positive weight-like quadratic
    worst = 0
    for case in range(6):
        D = P(rng.randint(3, 6), rng.choice((-1, 1)))
        num = P(*[rng.randint(-5, 5) for _ in range(rng.randint(1, 9))]) or P(1)
        a = AlgebraicNumber.from_interval(K0, P(-F(1, 10), 0, 1), F(3, 10), F(4, 10))       # sqrt(0.1) = 0.316
        b = AlgebraicNumber.from_interval(K0, P(-F(1, 2), 0, 1), F(7, 10), F(3, 4))          # sqrt(0.5) = 0.707
        tau = F(1, 10 ** rng.randint(8, 16))
        dens = [(W, 3), (D, 3)]
        # DEFAULT limits (policy max_taylor_order = 16, owner decision 2026-10-21): the long inner ends no longer
        # exceed max_bits (the order-32 refusal is pinned by the design-issue test below)
        budget = WorkBudget(None, 4096)
        result = enclose_rational_integral(kern(budget=budget), num, dens, a, b, tau)
        worst = max(worst, budget.counts()["max_bits_seen"])
        with mpmath.workdps(60):
            def f(x):
                return mp_poly(num, x) / (mp_poly(W, x) ** 3 * mp_poly(D, x) ** 3)
            ref = mpmath.quad(f, [mpmath.sqrt(mpmath.mpf(1) / 10), mpmath.sqrt(mpmath.mpf(1) / 2)])
        assert result.exact is None and result.enclosure.width <= tau and contains(result.enclosure, ref), case
    print(f"\nCH3b composite W^3 D^3 with algebraic ends, default limits: max bits {worst}")
    # reversed orientation negates
    a = AlgebraicNumber.from_interval(K0, P(-F(1, 10), 0, 1), F(3, 10), F(4, 10))
    low = KernelLimits()
    fwd = enclose_rational_integral(kern(limits=low), P(1), [(W, 2)], a, F(1), F(1, 10 ** 10)).enclosure
    a = AlgebraicNumber.from_interval(K0, P(-F(1, 10), 0, 1), F(3, 10), F(4, 10))
    bwd = enclose_rational_integral(kern(limits=low), P(1), [(W, 2)], F(1), a, F(1, 10 ** 10)).enclosure
    assert bwd.lo <= -fwd.lo and -fwd.hi <= bwd.hi and max(fwd.lo, -bwd.hi) <= min(fwd.hi, -bwd.lo)


def test_design_issue_long_centre_bits_at_order_32_exceed_the_cap():
    """Design issue (reported, not worked around; order 32 is now opt-in, passed explicitly): algebraic ends refined for tau = 1e-14 give inner ends with
    ~46-bit denominators; with the design's midpoint centre, g_j = P_j(c) / (Q(c)^{k+j} j!) at k = 3, K = 32 needs
    about 6575 bits, beyond max_bits = 4096. The kernel refuses (fail closed) rather than changing the formula."""
    rng = random.Random(SEED + 1)
    W = P(1, F(-1, 2), F(1, 2))
    D = P(rng.randint(3, 6), rng.choice((-1, 1)))
    num = P(*[rng.randint(-5, 5) for _ in range(rng.randint(1, 9))]) or P(1)
    a = AlgebraicNumber.from_interval(K0, P(-F(1, 10), 0, 1), F(3, 10), F(4, 10))
    b = AlgebraicNumber.from_interval(K0, P(-F(1, 2), 0, 1), F(7, 10), F(3, 4))
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        enclose_rational_integral(kern(limits=KernelLimits(max_taylor_order=32)), num, [(W, 3), (D, 3)], a, b,
                                  F(1, 10 ** 14))
    assert info.value.category is C.BIT_LENGTH_EXCEEDED


def test_policy_default_max_taylor_order_is_16_and_32_is_opt_in():
    """2026-10-21 owner decision (a): the default order is 16 (bit cap: ~140 bits per end-denominator bit at order
    32). The long-end case that refuses BIT_LENGTH_EXCEEDED at order 32 succeeds with default limits."""
    assert KernelLimits().max_taylor_order == 16
    assert KernelLimits(max_taylor_order=32).max_taylor_order == 32
    rng = random.Random(SEED + 1)
    W = P(1, F(-1, 2), F(1, 2))
    D = P(rng.randint(3, 6), rng.choice((-1, 1)))
    num = P(*[rng.randint(-5, 5) for _ in range(rng.randint(1, 9))]) or P(1)
    a = AlgebraicNumber.from_interval(K0, P(-F(1, 10), 0, 1), F(3, 10), F(4, 10))
    b = AlgebraicNumber.from_interval(K0, P(-F(1, 2), 0, 1), F(7, 10), F(3, 4))
    tau = F(1, 10 ** 14)
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        enclose_rational_integral(kern(limits=KernelLimits(max_taylor_order=32)), num, [(W, 3), (D, 3)], a, b, tau)
    assert info.value.category is C.BIT_LENGTH_EXCEEDED
    budget = WorkBudget(None, 4096)
    result = enclose_rational_integral(kern(budget=budget), num, [(W, 3), (D, 3)], a, b, tau)
    assert result.enclosure.width <= tau and budget.counts()["max_bits_seen"] <= 4096


def test_polynomial_integrals_with_rational_ends_are_exact():
    result = enclose_rational_integral(kern(), P(0, 0, 3), [(P(2), 1)], F(0), F(1), F(1, 10 ** 12))
    assert result.exact == F(1, 2) and result.enclosure == Enclosure.exact(F(1, 2))


def test_allocation_sums_and_children_get_half_the_parent_allowance():
    k = kern(limits=KernelLimits(max_taylor_order=8))          # force bisection
    tau = F(1, 10 ** 14)
    num, dens = P(1, 2), [(P(1, 1), 3), (P(3, -1), 2)]
    a = AlgebraicNumber.from_interval(K0, P(-2, 0, 1), 1, F(3, 2))
    result = enclose_rational_integral(k, num, dens, F(0), a, tau)
    rec = result.record
    assert len(rec.ends) == 1 and rec.ends[0][2] <= rec.ends[0][1] == tau / 4
    used = rec.ends[0][2]
    # the width is the sum of every term and end width; the leaves' allowances plus the end's use make tau
    assert result.enclosure.width == rec.total_width <= tau
    assert sum(p[5] for p in rec.pieces) + used == tau
    n_terms = len({p[0] for p in rec.pieces})
    share = (tau - used) / n_terms
    assert n_terms == 5                                       # 3 terms over 1+u, 2 over 3-u
    assert any(p[3] > 0 for p in rec.pieces)
    for term, x, y, depth, order, allowance, width in rec.pieces:
        assert allowance == share / 2 ** depth and width <= allowance and order <= 8
    for term in range(n_terms):                              # the children tile each term's interval
        segs = sorted((p[1], p[2]) for p in rec.pieces if p[0] == term)
        assert segs[0][0] == 0 and all(segs[i][1] == segs[i + 1][0] for i in range(len(segs) - 1))
    with mpmath.workdps(60):
        ref = mpmath.quad(lambda x: (1 + 2 * x) / ((1 + x) ** 3 * (3 - x) ** 2), [0, mpmath.sqrt(2)])
    assert contains(result.enclosure, ref)
    # pieces of one quantity share tau equally
    total, records = enclose_pieces(kern(), [(num, dens, F(0), F(1, 2)), (num, dens, F(1, 2), F(1))], tau)
    assert [r.tau for r in records] == [tau / 2, tau / 2] and total.width <= tau


def test_accuracy_unachievable_within_the_piece_limit():
    k = kern(limits=KernelLimits(max_taylor_order=8, max_pieces_per_integral=3))
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        enclose_rational_integral(k, P(1), [(P(1, 1), 5)], F(0), F(1), F(1, 10 ** 30))
    assert info.value.category is C.ACCURACY_UNACHIEVABLE and "achieved width" in str(info.value)


def test_order_32_k5_bit_size_and_time_are_recorded():
    """Design 10, risk 1: bits and time of one k = 5, order-32 term."""
    budget = WorkBudget(None, 4096)
    k = kern(budget=budget)
    t0 = time.perf_counter()
    main, rho = taylor_term(k, P(3, -2), P(1, F(-1, 2), F(1, 2)), 5, F(1, 5), F(2, 5), 32)
    elapsed = time.perf_counter() - t0
    bits = budget.counts()["max_bits_seen"]
    print(f"\nCH3b k=5 order 32: max bits {bits}, rho {float(rho):.3e}, {elapsed * 1e3:.1f} ms")
    assert bits <= 1024 and rho < F(1, 10 ** 30)


# ------------------------------------------------------------------ 1/|n|

def test_sqrt_allocation_case():
    a = F(1, 2 ** 300)
    q = divide_by_norm(kern(), F(2 ** 201), lambda tau: Enclosure.exact(2 ** 100), a)
    assert q.m == 202
    assert q.core.width <= a / 3 and q.claim.width <= a
    assert q.s_lo ** 2 <= 2 ** 201 < q.s_hi ** 2
    assert float(q.core.width / a) == pytest.approx(0.125, rel=1e-3)
    # an enclosed numerator meets its target and the claim still meets a
    q2 = divide_by_norm(kern(), F(13, 7), lambda tau: Enclosure(F(5) - tau / 2, F(5) + tau / 2), F(1, 10 ** 12))
    with mpmath.workdps(40):
        v = 5 / mpmath.sqrt(mpmath.mpf(13) / 7)
    assert q2.claim.width <= F(1, 10 ** 12) and contains(q2.core, v) and contains(q2.claim, v)


def test_sqrt_radicand_is_size_checked_before_isqrt(monkeypatch):
    calls = []
    real = math.isqrt
    monkeypatch.setattr(math, "isqrt", lambda n: calls.append(n.bit_length()) or real(n))
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        divide_by_norm(kern(max_bits=500), F(2 ** 201), lambda tau: Enclosure.exact(2 ** 100), F(1, 2 ** 300))
    assert info.value.category is C.BIT_LENGTH_EXCEEDED
    assert all(b <= 500 for b in calls)


# ------------------------------------------------------------------ resultants and the Q(u*) validation

def review_pair():
    """C_1 = (1/u, 1/(u+1)), C_2 = (1/s + 1/(s+1), 1/s - 1/(s+1)); M_1 = u(u+1), M_2 = s(s+1)."""
    M1, A1, B1 = P(0, 1, 1), P(1, 1), P(0, 1)
    # F = A1(u) M2(s) - A2(s) M1(u), A2 = 2s + 1;  G = B1(u) M2(s) - B2(s) M1(u), B2 = 1
    Fb = bivariate(K0, [P(*[-c for c in M1]), padd(K0, A1, P(*[-2 * c for c in M1])), A1])
    Gb = bivariate(K0, [P(*[-c for c in M1]), B1, B1])
    return Fb, Gb


U_STAR, S_STAR = 0.5850432433, 0.8546376797


def test_review_pair_resultant_and_crossing():
    k = kern()
    Fb, Gb = review_pair()
    R = resultant_s(k, Fb, Gb)
    expected = pmul(k, pmul(k, P(0, 0, 1), ppow(k, P(1, 1), 2)), P(1, 2, -4, -4))
    assert len(R) == 8 and pdivmod(k, R, expected)[1] == () and len(pdivmod(k, R, expected)[0]) == 1
    _, validations = curve_crossings(k, Fb, Gb, F(1, 4), F(2), F(1, 4), F(2))
    assert len(validations) == 1
    v = validations[0]
    assert v.genuine and v.gcd_degree == 1 and len(v.s_roots) == 1
    assert abs(float(v.u.lo) - U_STAR) < 1e-9 and abs(float(v.s_roots[0].lo) - S_STAR) < 1e-9
    u, s = (v.u.lo + v.u.hi) / 2, (v.s_roots[0].lo + v.s_roots[0].hi) / 2
    assert abs(1 / u - (1 / s + 1 / (s + 1))) < 1e-9 and abs(1 / (u + 1) - (1 / s - 1 / (s + 1))) < 1e-9


def test_round_4_obligation_gcd_over_the_factor_containing_u_star():
    """Start from the reducible square-free q = u(u+1)(1+2u-4u^2-4u^3): M_1 = u(u+1) is not invertible modulo q
    although M_1(u*) != 0; the zero tests must refine q to the factor containing u*."""
    k = kern()
    Fb, Gb = review_pair()
    q = pmul(k, P(0, 1, 1), P(1, 2, -4, -4))
    roots = [r for r in isolate_roots(k, q, F(1, 4), F(2))]
    assert len(roots) == 1 and roots[0].poly == pmul(k, P(0, 1, 1), P(-1, -2, 4, 4))
    v = validate_crossing(k, Fb, Gb, roots[0], F(1, 4), F(2), q=q)
    assert v.splits >= 1 and v.final_q == P(-1, -2, 4, 4)          # the irreducible cubic of u*
    assert v.genuine and v.gcd_degree == 1
    assert abs(float(roots[0].lo) - U_STAR) < 1e-9 and abs(float(v.s_roots[0].lo) - S_STAR) < 1e-9
    # the leading coefficient u + 1 of F is a unit at u*, yet not modulo the starting q
    assert pgcd(k, P(1, 1), q) == P(1, 1)


def test_pole_factor_root_is_rejected():
    """With the u-interval widened over the pole u = 0 of C_1, the resultant root u = 0 (a factor of M_1^2)
    gives the gcd M_2(s) = s(s+1), whose roots are outside the open s-interval: rejected."""
    k = kern()
    Fb, Gb = review_pair()
    _, validations = curve_crossings(k, Fb, Gb, F(-1, 2), F(2), F(1, 4), F(2))
    # roots in (-1/2, 2): the pole u = 0, the cubic's root near -0.344 and u*
    assert len(validations) == 3
    pole = [v for v in validations if algebraic_equal(k, v.u, F(0))]
    assert len(pole) == 1 and pole[0].gcd_degree >= 1 and not pole[0].genuine and pole[0].s_roots == ()
    genuine = [v for v in validations if v.genuine]
    assert len(genuine) == 1 and abs(float(genuine[0].u.lo) - U_STAR) < 1e-9


def test_spurious_leading_coefficient_root_is_rejected():
    """C_1 = (u, u^2), C_2 = ((s^2/2 + s) / (s^2 + 1), (s^2/4 + 1) / (s^2 + 1)): C_2(inf) = (1/2, 1/4) = C_1(1/2), so
    both leading coefficients in s vanish at u = 1/2 and the resultant vanishes there; the gcd has degree 0."""
    k = kern()
    Fb = bivariate(K0, [P(0, 1), P(-1), P(F(-1, 2), 1)])                 # u (s^2 + 1) - (s^2/2 + s)
    Gb = bivariate(K0, [P(-1, 0, 1), P(), P(F(-1, 4), 0, 1)])            # u^2 (s^2 + 1) - (s^2/4 + 1)
    R = resultant_s(k, Fb, Gb)
    assert peval(k, R, F(1, 2)) == 0
    _, validations = curve_crossings(k, Fb, Gb, F(0), F(1), F(-10), F(10))
    spurious = [v for v in validations if v.u.is_rational and v.u.lo == F(1, 2)]
    assert len(spurious) == 1 and not spurious[0].genuine and spurious[0].gcd_degree == 0
    for v in validations:                                                # every accepted crossing is real
        if v.genuine:
            u, s = float(v.u.lo + v.u.hi) / 2, float(v.s_roots[0].lo + v.s_roots[0].hi) / 2
            assert abs(u - (s * s / 2 + s) / (s * s + 1)) < 1e-9 and abs(u * u - (s * s / 4 + 1) / (s * s + 1)) < 1e-9


def test_identically_vanishing_resultant_is_refused():
    Fb, _ = review_pair()
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        curve_crossings(kern(), Fb, Fb, F(1, 4), F(2), F(1, 4), F(2))
    assert info.value.category is C.UNDECIDED_PREDICATE


# ------------------------------------------------------------------ the whole-edge range test

def test_range_test_refuses_the_review_trim():
    """r = 12u - 12u^2 - 2 on [1/4, 3/4]: corners r = 1/4 pass, the interior extremum r(1/2) = 1 = v_1 refuses."""
    k = kern()
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        graph_in_range(k, P(-2, 12, -12), P(1), F(1, 4), F(3, 4), 0, 1)
    assert info.value.category is C.UNSUPPORTED_TRIM and info.value.sub_reason == "leaves_range_box"
    assert graph_in_range(k, P(-2, 12, -12), P(1), F(1, 4), F(3, 4), 0, F(11, 10)) == "exact"
    assert graph_in_range(k, P(1, 1), P(4), F(1, 4), F(3, 4), 0, 1) == "bernstein"


def test_range_test_with_algebraic_corners_and_a_rational_graph():
    k = kern()
    a = AlgebraicNumber.from_interval(k, P(-1, 0, 8), F(1, 4), F(2, 5))       # 1/sqrt(8) = 0.354
    b = AlgebraicNumber.from_interval(k, P(-1, 0, 2), F(7, 10), F(3, 4))      # 1/sqrt(2) = 0.707
    N, D = P(F(1, 2), 1, -1), P(2, 1)                                          # (1/2 + u - u^2) / (2 + u)
    assert graph_in_range(k, N, D, a, b, F(1, 10), F(1, 2)) in ("bernstein", "exact")
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:                 # max of r is about 0.2808 > 0.28
        graph_in_range(k, N, D, a, b, F(1, 10), F(28, 100))
    assert info.value.sub_reason == "leaves_range_box"
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:                 # D vanishes inside
        graph_in_range(k, N, P(F(-1, 2), 1), a, b, 0, 1)
    assert info.value.category is C.UNSUPPORTED_TRIM and info.value.sub_reason == "denominator_vanishes"
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:                 # D vanishes at a rational corner
        graph_in_range(k, N, P(F(-1, 4), 1), F(1, 4), F(3, 4), 0, 1)
    assert info.value.sub_reason == "denominator_vanishes"


# ------------------------------------------------------------------ work: per-element checkpoints, deadlines, bit cap

def _ops():
    """Each kernel loop family exercised by one call, on a fresh kernel."""
    Fb, Gb = review_pair()
    sq2 = lambda k: AlgebraicNumber.from_interval(k, P(-2, 0, 1), 1, 2)
    return {
        "pmul": lambda k: pmul(k, P(1, 2, 3, 4), P(5, 6, 7)),
        "pdivmod": lambda k: pdivmod(k, P(1, 2, 3, 4, 5, 6), P(1, 1, 2)),
        "pgcd": lambda k: pgcd(k, P(6, 0, -5, 0, 1), P(-2, 0, 1, 0, 3)),
        "bernstein": lambda k: bernstein(k, P(1, -3, 2, 5, -1), F(1, 3), F(2, 3)),
        "sturm": lambda k: variations_at(k, sturm_sequence(k, P(6, 0, -5, 0, 1)), F(1, 3)),
        "isolate": lambda k: isolate_roots(k, P(6, 0, -5, 0, 1)),
        "refine": lambda k: sq2(k).refine_below(k, F(1, 2 ** 20)),
        "identity": lambda k: algebraic_equal(k, AlgebraicNumber.from_interval(k, P(6, 0, -5, 0, 1), 1, F(8, 5)),
                                              AlgebraicNumber.from_interval(k, P(-3, 0, 1), F(3, 2), 2)),
        "sign_at": lambda k: sign_at(k, P(F(-141421356, 10 ** 8), 1), sq2(k)),
        "coprime": lambda k: coprime_basis(k, [P(1, 2, 1), P(1, 1, 0, 1), P(-1, 0, 1)]),
        "partial": lambda k: partial_fractions(k, P(1, 2, 3, 4, 5), [(P(1, 1), 3), (P(2, 0, 1), 1)]),
        "term": lambda k: enclose_term(k, P(3, 1), P(2, -1, 1), 3, F(0), F(1), F(1, 10 ** 15)),
        "integral": lambda k: enclose_rational_integral(k, P(1, 2), [(P(1, 1), 2)], F(0), sq2(k), F(1, 10 ** 10)),
        "norm": lambda k: divide_by_norm(k, F(2 ** 201), lambda tau: Enclosure.exact(2 ** 100), F(1, 2 ** 300)),
        "resultant": lambda k: resultant_s(k, Fb, Gb),
        "crossing": lambda k: curve_crossings(k, Fb, Gb, F(1, 4), F(2), F(1, 4), F(2)),
        "range": lambda k: graph_in_range(k, P(-2, 12, -12), P(1), F(1, 4), F(3, 4), 0, F(11, 10)),
    }


def _units(op):
    budget = WorkBudget(None, 4096)
    op(kern(budget=budget))
    return budget.counts()


@pytest.mark.parametrize("name", sorted(_ops()))
def test_cancellation_at_element_k_and_deadline_per_loop(name):
    op = _ops()[name]
    done = _units(op)
    total = done["operations"]
    assert total > 0
    for stop in sorted({0, 1, total // 3, total // 2, total - 1}):
        budget = WorkBudget(None, 4096)
        with pytest.raises(CertifiedHydrostaticsRefusal) as info:
            op(kern(budget=budget, cancel=lambda b=budget, s=stop: b.operations >= s))
        assert info.value.category is C.CANCELLED and budget.operations == stop
        holder = []
        budget = WorkBudget(None, 4096, deadline=5.0,
                            clock=lambda s=stop: 10.0 if holder[0].operations >= s else 0.0)
        holder.append(budget)
        with pytest.raises(CertifiedHydrostaticsRefusal) as info:
            op(kern(budget=budget))
        assert info.value.category is C.DEADLINE_EXCEEDED and budget.operations == stop
    # every unit kind of the loop is a checkpoint: cancel at the first unit of each kind
    for kind, count in done["units"].items():
        budget = WorkBudget(None, 4096)
        with pytest.raises(CertifiedHydrostaticsRefusal) as info:
            op(kern(budget=budget, cancel=lambda b=budget, kd=kind: b.counts()["units"].get(kd, 0) >= 1))
        assert info.value.category is C.CANCELLED


@pytest.mark.parametrize("name", ["pmul", "pdivmod", "bernstein", "sturm", "identity", "resultant", "range"])
def test_every_unit_is_a_budget_checkpoint(name):
    op = _ops()[name]
    total = _units(op)["operations"]
    for limit in range(total):
        budget = WorkBudget(limit, 4096)
        with pytest.raises(CertifiedHydrostaticsRefusal) as info:
            op(kern(budget=budget))
        assert info.value.category is C.BUDGET_EXHAUSTED and budget.operations == limit


def test_coefficient_loops_reserve_one_unit_per_element():
    budget = WorkBudget(None, 4096)
    k = kern(budget=budget)
    pmul(k, P(1, 2, 3, 4), P(5, 6, 7))
    assert budget.counts()["units"] == {"poly_coefficient": 12}
    budget = WorkBudget(None, 4096)
    peval(kern(budget=budget), P(1, 2, 3, 4, 5), F(1, 3))
    assert budget.counts()["units"] == {"poly_coefficient": 5}


def test_bit_cap_refuses_before_the_native_operation():
    big = P(2 ** 600 + 1, 3)
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        pmul(kern(max_bits=1000), big, big)                     # the 1200-bit product is never formed
    assert info.value.category is C.BIT_LENGTH_EXCEEDED and "intermediate" in str(info.value)
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:   # Taylor recurrences under a small cap
        taylor_term(kern(max_bits=200), P(3, -2), P(1, F(-1, 2), F(1, 2)), 5, F(1, 5), F(2, 5), 32)
    assert info.value.category is C.BIT_LENGTH_EXCEEDED
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:   # the resultant pipeline under a tiny cap
        curve_crossings(kern(max_bits=8), *review_pair(), F(1, 4), F(2), F(1, 4), F(2))
    assert info.value.category is C.BIT_LENGTH_EXCEEDED


# ---------------------------------------------------------------- Codex review round 1 (FIXES1)
# Every expectation below is a hard-coded input and a hard-coded tiny budget: none derives a stop position
# from the implementation's own counters.

import tracemalloc

from anygeometry.certified_hydrostatics._arith import Ar


def _peak_bytes(fn):
    tracemalloc.start()
    try:
        try:
            fn()
        except CertifiedHydrostaticsRefusal:
            pass
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def _huge_limit_ops():
    Fb, Gb = review_pair()
    sq2 = lambda k: AlgebraicNumber.from_interval(k, P(-2, 0, 1), 1, 2)
    return {
        "refine": lambda k: sq2(k).refine(k),
        "isolate": lambda k: isolate_roots(k, P(-2, 0, 1)),
        "integral": lambda k: enclose_rational_integral(k, P(1, 2), [(P(1, 1), 2)], F(0), sq2(k), F(1, 10 ** 6)),
        "range": lambda k: graph_in_range(k, P(1), P(F(-141, 100), 1), sq2(k), F(3, 2), -5, 5),
        "crossing": lambda k: curve_crossings(k, Fb, Gb, F(1, 4), F(2), F(1, 4), F(2)),
    }


@pytest.mark.parametrize("name", sorted(_huge_limit_ops()))
def test_isolation_threshold_is_never_built_as_a_power_of_two(name):
    """Finding 1: ``1 << max_isolation_bits`` allocated 12.5 MB here although only a few bits were seen."""
    op = _huge_limit_ops()[name]
    limits = KernelLimits(max_isolation_bits=10 ** 8)
    peak = _peak_bytes(lambda: op(kern(max_bits=4096, limits=limits)))
    assert peak < 1_000_000


def test_threshold_comparison_is_exact_at_the_boundary():
    from anygeometry.certified_hydrostatics._ch3_poly import at_most_pow2
    for e in (1, 2, 5, 64):
        for mult in (1, 2):
            edge = F(mult, 2 ** e)
            assert at_most_pow2(edge, e, mult) and not at_most_pow2(edge + F(1, 2 ** (e + 70)), e, mult)
            assert at_most_pow2(F(0), e, mult) and not at_most_pow2(F(7), e, mult)
    assert not at_most_pow2(F(1, 3), 10 ** 12) and not at_most_pow2(F(1, 10 ** 400), 10 ** 12)
    assert at_most_pow2(F(1, 2 ** 5000), 4000)


def test_bivariate_construction_is_metered_and_size_checked():
    """Finding 2: coefficients were converted with no checkpoint and no size check."""
    big = [(F(1 << 100),)]
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        bivariate(kern(budget=WorkBudget(0, 8), cancel=lambda: True), big)
    assert info.value.category is C.CANCELLED
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        bivariate(kern(max_bits=8), big)
    assert info.value.category is C.BIT_LENGTH_EXCEEDED
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        bivariate(kern(budget=WorkBudget(2, 4096)), [(1, 2, 3, 4, 5)])
    assert info.value.category is C.BUDGET_EXHAUSTED


def test_ppow_of_zero_is_metered_per_iteration():
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        ppow(kern(cancel=lambda: True), (), 100000)
    assert info.value.category is C.CANCELLED
    budget = WorkBudget(5, 4096)
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        ppow(kern(budget=budget), (), 100000)
    assert info.value.category is C.BUDGET_EXHAUSTED and budget.operations == 5


def test_denominator_exponent_loop_is_metered_per_iteration(monkeypatch):
    """Finding 2: ``1 / 1**10000`` ran 10 000 native multiplications under an operation budget of four."""
    calls = [0]
    original = Ar.mul

    def counting(self, a, b):
        calls[0] += 1
        return original(self, a, b)

    monkeypatch.setattr(Ar, "mul", counting)
    budget = WorkBudget(4, 4096)
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        enclose_rational_integral(kern(budget=budget), P(1), [(P(1), 10000)], F(0), F(1), F(1, 10))
    assert info.value.category is C.BUDGET_EXHAUSTED and budget.operations == 4
    assert calls[0] < 100
    calls[0] = 0
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        enclose_rational_integral(kern(cancel=lambda: True), P(1), [(P(1), 10000)], F(0), F(1), F(1, 10))
    assert info.value.category is C.CANCELLED and calls[0] < 100


def test_input_conversions_are_size_checked():
    big = F(1 << 300)
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        AlgebraicNumber.from_interval(kern(max_bits=64), P(-2, 0, 1), F(1), big)
    assert info.value.category is C.BIT_LENGTH_EXCEEDED
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        enclose_rational_integral(kern(max_bits=64), P(1), [(P(1, 1), 1)], F(0), F(1), F(1, 1 << 300))
    assert info.value.category is C.BIT_LENGTH_EXCEEDED
    Fb, Gb = review_pair()
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        curve_crossings(kern(max_bits=4096), Fb, Gb, F(1, 4), F(2), F(1, 4), F(1 << 5000))
    assert info.value.category is C.BIT_LENGTH_EXCEEDED


def _sqrt_frac(num, den, lo, hi):
    return lambda k: AlgebraicNumber.from_interval(k, P(-num, 0, den), lo, hi)


def _vertical_pair():
    Fb = bivariate(K0, [P(0, 1), P(-1)])                    # u - s
    Gb = bivariate(K0, [P(F(-1, 2), 0, 1)])                 # u^2 - 1/2 (a vertical line)
    return Fb, Gb


def test_crossing_with_algebraic_s_bounds():
    """Finding 3: algebraic bounds of the second arc raised TypeError."""
    k = kern()
    Fb, Gb = _vertical_pair()
    s_lo = _sqrt_frac(1, 10, 0, 1)(k)                        # sqrt(1/10)
    s_hi = _sqrt_frac(9, 10, 0, 1)(k)                        # sqrt(9/10)
    _, validations = curve_crossings(k, Fb, Gb, F(0), F(1), s_lo, s_hi)
    assert len(validations) == 1 and validations[0].genuine
    (root,) = validations[0].s_roots
    assert root.width <= F(1, 2 ** 40) and root.lo <= 1 / math.sqrt(2) <= root.hi


def test_crossing_with_algebraic_s_bounds_excluding_the_root():
    k = kern()
    Fb, Gb = _vertical_pair()
    s_lo = _sqrt_frac(8, 10, 0, 1)(k)                        # sqrt(0.8) > 1/sqrt(2)
    s_hi = _sqrt_frac(9, 10, 0, 1)(k)
    _, validations = curve_crossings(k, Fb, Gb, F(0), F(1), s_lo, s_hi)
    assert len(validations) == 1 and not validations[0].genuine and validations[0].s_roots == ()
    s_lo = _sqrt_frac(1, 10, 0, 1)(k)
    s_hi = _sqrt_frac(3, 10, 0, 1)(k)                        # sqrt(0.3) < 1/sqrt(2)
    _, validations = curve_crossings(k, Fb, Gb, F(0), F(1), s_lo, s_hi)
    assert len(validations) == 1 and not validations[0].genuine


def test_crossing_with_a_mixed_rational_and_algebraic_bound_and_a_tie():
    k = kern(limits=KernelLimits(max_isolation_bits=64))
    Fb, Gb = _vertical_pair()
    s_hi = _sqrt_frac(9, 10, 0, 1)(k)
    _, validations = curve_crossings(k, Fb, Gb, F(0), F(1), F(1, 2), s_hi)
    assert len(validations) == 1 and validations[0].genuine
    # the root s = 1/sqrt(2) sits exactly on the algebraic bound: undecidable by intervals, so refused (fail closed)
    s_tie = _sqrt_frac(1, 2, 0, 1)(k)
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        curve_crossings(k, Fb, Gb, F(0), F(1), F(0), s_tie)
    assert info.value.category is C.UNDECIDED_PREDICATE


def test_initial_term_pieces_respect_max_pieces_per_integral():
    """Finding 4: two terms produced two Taylor pieces under ``max_pieces_per_integral=1``."""
    k = kern(limits=KernelLimits(max_pieces_per_integral=1))
    num, dens = P(1), [(P(1, 1), 1), (P(2, 1), 1)]            # 1 / ((1 + u)(2 + u))
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        enclose_rational_integral(k, num, dens, F(0), F(1, 10), F(1, 10 ** 6))
    assert info.value.category is C.ACCURACY_UNACHIEVABLE
    ok = enclose_rational_integral(kern(limits=KernelLimits(max_pieces_per_integral=2)), num, dens, F(0), F(1, 10),
                                   F(1, 10 ** 6))
    assert ok.record.taylor_pieces == 2
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:   # the same through the term entry point
        enclose_term(kern(limits=KernelLimits(max_pieces_per_integral=1)), P(1), P(1, 1), 1, F(0), F(1, 10),
                     F(1, 10 ** 6), AllocationRecord(F(1, 10 ** 6), taylor_pieces=1))
    assert info.value.category is C.ACCURACY_UNACHIEVABLE


def test_algebraic_s_bound_path_is_a_budget_and_cancellation_checkpoint():
    Fb, Gb = _vertical_pair()

    def run(k):
        s_lo = _sqrt_frac(1, 10, 0, 1)(k)
        s_hi = _sqrt_frac(9, 10, 0, 1)(k)
        return curve_crossings(k, Fb, Gb, F(0), F(1), s_lo, s_hi)

    for limit in (200, 3000, 15000):                       # the two bounds cost 172 units; the rest is the crossing
        budget = WorkBudget(limit, 4096)
        with pytest.raises(CertifiedHydrostaticsRefusal) as info:
            run(kern(budget=budget))
        assert info.value.category is C.BUDGET_EXHAUSTED and budget.operations == limit
    budget = WorkBudget(None, 4096)
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        run(kern(budget=budget, cancel=lambda: budget.operations >= 5000))
    assert info.value.category is C.CANCELLED and budget.operations == 5000


def test_bivariate_empty_rows_are_metered_and_cancellable():
    """Re-review finding 1: 100000 empty rows were consumed for one operation."""
    rows = [()] * 100000 + [(F(1),)]
    budget = WorkBudget(1, 4096)
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        bivariate(kern(budget=budget), rows)
    assert info.value.category is C.BUDGET_EXHAUSTED and budget.operations == 1
    budget = WorkBudget(None, 4096)
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        bivariate(kern(budget=budget, cancel=lambda: budget.operations >= 10), rows)
    assert info.value.category is C.CANCELLED and budget.operations == 10


def test_algebraic_filter_loop_is_a_budget_and_cancellation_checkpoint(monkeypatch):
    """Re-review finding 2: stops inside ``_inside_algebraic_bounds`` (first entered at operation 22675).
    The hard-coded positions are verified by instrumenting the filter."""
    import anygeometry.certified_hydrostatics._ch3_resultant as R
    Fb, Gb = _vertical_pair()
    inner = R._inside_algebraic_bounds
    entries = []

    def spy(nf, seq, h, enc, checks):
        entries.append(1)
        return inner(nf, seq, h, enc, checks)

    monkeypatch.setattr(R, "_inside_algebraic_bounds", spy)

    def run(k):
        s_lo = _sqrt_frac(1, 10, 0, 1)(k)
        s_hi = _sqrt_frac(9, 10, 0, 1)(k)
        return curve_crossings(k, Fb, Gb, F(0), F(1), s_lo, s_hi)

    for limit in (22700, 23000, 23300):          # the filter runs from 22675 to 23350
        entries.clear()
        budget = WorkBudget(limit, 4096)
        with pytest.raises(CertifiedHydrostaticsRefusal) as info:
            run(kern(budget=budget))
        assert info.value.category is C.BUDGET_EXHAUSTED and budget.operations == limit
        assert len(entries) >= 1, limit
    entries.clear()
    budget = WorkBudget(None, 4096)
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        run(kern(budget=budget, cancel=lambda: budget.operations >= 23000))
    assert info.value.category is C.CANCELLED and budget.operations == 23000
    assert len(entries) >= 1
