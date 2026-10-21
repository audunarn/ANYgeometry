"""CH3a2: loops of graph pieces on a curved face (design CH3 section 3.3).

A curved face's region is the interior of its closed loop of pieces ``v = r_k(u) = N_k / D_k`` over the exact
``u``-intervals between corners. ``check_loop`` decides the **four simplicity tests** exactly (``INVALID_LOOP``):

(i)   non-adjacent pieces have no crossing in the open intersection of their ``u``-intervals (roots of
      ``N_i D_j - N_j D_i``), and adjacent pieces none except at their shared corner;
(ii)  endpoint-interior: no corner lies on a non-incident piece (single-root exact test);
(iii) non-adjacent corners are different points;
(iv)  no two pieces coincide over an open ``u``-interval (``N_i D_j - N_j D_i`` identically zero).

``loop_sigma`` is ``sigma_chart = sign(-oint v du)``, certified by an enclosure (``UNDECIDED_PREDICATE`` if it
cannot be); ``region_contains`` is the vertical-ray parity of a rational point; ``is_v_monotone`` decides exactly
whether every vertical line meets the region in at most one interval (every elementary ``u``-interval between
sorted corners is covered by exactly two pieces)."""
from __future__ import annotations

from fractions import Fraction

from ._ch3_admission_trims import CurvedFace
from ._ch3_algebraic import AlgebraicNumber, compare, isolate_roots, lower_bound, sign_at, upper_bound
from ._ch3_integrate import enclose_rational_integral
from ._ch3_poly import Kernel, ONE, ZERO, peval, pmul, psub, refusal
from ._errors import RefusalCategory

__all__ = ["loop_vertices", "check_loop", "loop_sigma", "region_contains", "is_v_monotone"]

IL = RefusalCategory.INVALID_LOOP


def loop_vertices(face: CurvedFace) -> list:
    """The loop vertex that starts each piece (loop order)."""
    return [t.start_vertex if t.forward else t.end_vertex for t in face.trims]


def _diff(k: Kernel, a, b):
    return psub(k, pmul(k, a.plane.N, b.plane.D), pmul(k, b.plane.N, a.plane.D))


def _max(k, x, y):
    return x if compare(k, x, y) >= 0 else y


def _min(k, x, y):
    return x if compare(k, x, y) <= 0 else y


def check_loop(k: Kernel, face: CurvedFace) -> None:
    trims = face.trims
    n = len(trims)
    starts = loop_vertices(face)
    roots = [face.corners[v].number for v in starts]
    for i in range(n):
        for j in range(i + 1, n):
            k.tick("loop_pair")
            adjacent = j == i + 1 or (i == 0 and j == n - 1)
            a, b = trims[i], trims[j]
            R = _diff(k, a, b)
            lo, hi = _max(k, a.u_lo, b.u_lo), _min(k, a.u_hi, b.u_hi)
            overlap = compare(k, lo, hi) < 0
            if not R:
                if overlap:
                    raise refusal(IL, f"pieces {i} and {j} of face {face.face} lie on the same graph",
                                  "coincident_pieces")
                continue
            if overlap:
                for r in isolate_roots(k, R, lower_bound(lo), upper_bound(hi)):
                    k.tick("root_refine")
                    if compare(k, r, lo) > 0 and compare(k, r, hi) < 0:
                        raise refusal(IL, f"pieces {i} and {j} of face {face.face} cross",
                                      "adjacent_crossing" if adjacent else "crossing")
    for c in range(n):                                   # (ii) corner c is the start of piece c
        incident = {(c - 1) % n, c}
        for m in range(n):
            k.tick("loop_pair")
            if m in incident:
                continue
            t = trims[m]
            if compare(k, roots[c], t.u_lo) >= 0 and compare(k, roots[c], t.u_hi) <= 0:
                if sign_at(k, _diff(k, trims[m], trims[c]), roots[c]) == 0:
                    raise refusal(IL, f"corner {c} of face {face.face} lies on piece {m}", "endpoint_contact")
    for c in range(n):                                   # (iii)
        for d in range(c + 1, n):
            k.tick("loop_pair")
            if d == c + 1 or (c == 0 and d == n - 1):
                continue
            if compare(k, roots[c], roots[d]) == 0 and sign_at(k, _diff(k, trims[c], trims[d]), roots[c]) == 0:
                raise refusal(IL, f"corners {c} and {d} of face {face.face} are the same point", "repeated_corner")


def loop_sigma(k: Kernel, face: CurvedFace, start_bits: int = 6, stop_bits: int = 60) -> int:
    """``sigma_chart = sign(-oint v du)`` in the loop direction (``+1`` anticlockwise in the chart)."""
    bits = start_bits
    while True:
        k.tick("sigma_round")
        tau = Fraction(1, 1 << bits) / len(face.trims)
        total = ZERO
        low = high = ZERO
        for t in face.trims:
            k.tick("loop_pair")
            a, b = (t.u_lo, t.u_hi) if t.loop_increasing else (t.u_hi, t.u_lo)
            D = t.plane.D
            if len(D) == 1:
                num, dens = tuple(k.ar.div(c, D[0]) for c in t.plane.N), []
            else:
                num, dens = t.plane.N, [(D, 1)]
            if not num:
                continue
            e = enclose_rational_integral(k, num, dens, a, b, tau).enclosure
            low, high = k.ar.add(low, e.lo), k.ar.add(high, e.hi)
        if high < 0:
            return 1                                    # oint v du < 0
        if low > 0:
            return -1
        if bits >= stop_bits:
            raise refusal(RefusalCategory.UNDECIDED_PREDICATE, f"sigma_chart of face {face.face} is not certified",
                          "sigma_chart")
        bits += 8


def region_contains(k: Kernel, face: CurvedFace, u_q, v_q) -> bool:
    """Vertical-ray parity: the number of pieces whose open ``u``-interval contains the rational ``u_q`` and
    whose graph is above ``v_q`` is odd. ``u_q`` on a corner or ``v_q`` on a piece is ``UNDECIDED_PREDICATE``."""
    u_q, v_q = Fraction(u_q), Fraction(v_q)
    count = 0
    for t in face.trims:
        k.tick("loop_pair")
        c_lo, c_hi = compare(k, t.u_lo, u_q), compare(k, u_q, t.u_hi)
        if c_lo == 0 or c_hi == 0:
            raise refusal(RefusalCategory.UNDECIDED_PREDICATE, "the query abscissa is a corner", "breakpoint")
        if c_lo < 0 and c_hi < 0:
            d = peval(k, t.plane.D, u_q)
            r = k.ar.div(peval(k, t.plane.N, u_q), d)
            if r == v_q:
                raise refusal(RefusalCategory.UNDECIDED_PREDICATE, "the query point is on the boundary", "on_boundary")
            if r > v_q:
                count += 1
    return count % 2 == 1


def is_v_monotone(k: Kernel, face: CurvedFace) -> bool:
    """Exact: every vertical line ``u = const`` meets the region in at most one interval (the loop is a lower
    graph and an upper graph over one ``u``-interval)."""
    points: list = []
    for v in loop_vertices(face):
        k.tick("loop_pair")
        number = face.corners[v].number
        if not any(compare(k, number, p) == 0 for p in points):
            points.append(number)
    # insertion sort by exact comparison (one tick per comparison)
    ordered: list = []
    for p in points:
        i = len(ordered)
        while i > 0 and compare(k, ordered[i - 1], p) > 0:
            k.tick("loop_pair")
            i -= 1
        ordered.insert(i, p)
    for a, b in zip(ordered, ordered[1:]):
        covering = 0
        for t in face.trims:
            k.tick("loop_pair")
            if compare(k, t.u_lo, a) <= 0 and compare(k, b, t.u_hi) <= 0:
                covering += 1
        if covering != 2:
            return False
    return True
