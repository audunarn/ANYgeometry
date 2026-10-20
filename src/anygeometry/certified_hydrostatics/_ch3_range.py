"""CH3 exact kernel, layer 5: the whole-edge range test of admission check C2 (design CH3 section 2.3, revision 4).

A trim edge is the graph ``v = r(u) = N(u) / D(u)`` over the closed ``u``-interval ``[u_a, u_b]`` whose ends are
the corners (rational or algebraic). The test proves ``v_0 < r(u) < v_1`` on the whole closed interval:

1. ``D`` must be certified one-signed on the outer bounds ``[lo(u_a), hi(u_b)]`` by its Bernstein coefficients
   (algebraic ends refined to the policy depth), else ``UNSUPPORTED_TRIM / denominator_not_certified``;
2. sufficient test: the Bernstein coefficients of ``N - v_0 D`` and of ``v_1 D - N`` over the outer bounds all
   have the strict sign of ``D``;
3. otherwise exact: at both corners and at every root of ``N' D - N D'`` strictly inside ``(u_a, u_b)`` (the
   interior extrema of ``r``), ``r`` is compared with ``v_0`` and ``v_1`` by single-root sign tests. A value
   ``<= v_0`` or ``>= v_1`` anywhere refuses ``UNSUPPORTED_TRIM / leaves_range_box``. ``r`` is continuous on the
   interval (step 1), so its extreme values are taken at the corners or the interior extrema."""
from __future__ import annotations

from fractions import Fraction

from ._ch3_algebraic import AlgebraicNumber, compare, isolate_roots, lower_bound, sign_at, upper_bound
from ._ch3_poly import Kernel, at_most_pow2, bernstein, pderiv, pmul, pscale, psub, refusal, signed_lower
from ._errors import RefusalCategory

__all__ = ["graph_in_range"]


def _strict_sign(k: Kernel, p: tuple, lo, hi, sign: int) -> bool:
    for c in bernstein(k, p, lo, hi):
        k.tick("bernstein_coefficient")
        if ((c > 0) - (c < 0)) != sign:
            return False
    return True


def graph_in_range(k: Kernel, N: tuple, D: tuple, u_a, u_b, v_0, v_1) -> str:
    """Prove ``v_0 < N/D < v_1`` on ``[u_a, u_b]``; returns ``"bernstein"`` or ``"exact"`` (the method that
    decided), or refuses ``UNSUPPORTED_TRIM``."""
    v_0, v_1 = Fraction(v_0), Fraction(v_1)
    if compare(k, u_a, u_b) >= 0:
        raise ValueError("the edge interval needs u_a < u_b")
    # 1. D must not vanish on the closed interval (exact), then its sign on the outer bounds
    vanishes = sign_at(k, D, u_a) == 0 or sign_at(k, D, u_b) == 0
    for r in isolate_roots(k, D, lower_bound(u_a), upper_bound(u_b)):
        k.tick("root_refine")
        vanishes = vanishes or (compare(k, r, u_a) >= 0 and compare(k, r, u_b) <= 0)
    if vanishes:
        raise refusal(RefusalCategory.UNSUPPORTED_TRIM, "the graph denominator vanishes on the edge interval",
                      "denominator_vanishes")
    while True:
        k.tick("root_refine")
        lo, hi = lower_bound(u_a), upper_bound(u_b)
        sign, _ = signed_lower(k, D, lo, hi)
        if sign:
            break
        movable = [x for x in (u_a, u_b) if isinstance(x, AlgebraicNumber) and not x.is_rational
                   and not at_most_pow2(x.hi - x.lo, k.limits.max_isolation_bits, 2)]
        if not movable:
            raise refusal(RefusalCategory.UNSUPPORTED_TRIM, "the graph denominator is not certified one-signed "
                          "on the edge interval", "denominator_not_certified")
        for x in movable:
            x.refine(k)
    below = psub(k, N, pscale(k, D, v_0))            # N - v_0 D
    above = psub(k, pscale(k, D, v_1), N)            # v_1 D - N
    # 2. sufficient Bernstein test on the outer bounds
    if _strict_sign(k, below, lo, hi, sign) and _strict_sign(k, above, lo, hi, sign):
        return "bernstein"
    # 3. exact decision at the corners and the interior extrema
    points = [u_a, u_b]
    critical = psub(k, pmul(k, pderiv(k, N), D), pmul(k, N, pderiv(k, D)))
    if critical:
        for r in isolate_roots(k, critical, lower_bound(u_a), upper_bound(u_b)):
            k.tick("root_refine")
            if compare(k, r, u_a) > 0 and compare(k, r, u_b) < 0:
                points.append(r)
    for x in points:
        k.tick("sign_count")
        if sign_at(k, below, x) * sign <= 0 or sign_at(k, above, x) * sign <= 0:
            raise refusal(RefusalCategory.UNSUPPORTED_TRIM, "the trim graph leaves the open range box",
                          "leaves_range_box")
    return "exact"
