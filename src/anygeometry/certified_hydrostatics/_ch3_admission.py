"""CH3a1: the exact model of a curved face of the ruled rational class (design CH3 sections 2.1-2.3).

``classify_surface`` decides **C1** (the class) and **C4** (chart injectivity) and builds the knot-space net
``A~, B~`` (homogeneous rows, power basis in the knot parameter ``u``) with the common weight polynomial ``W``;
``plane_graph`` turns a trim plane into the exact graph ``v = r(u) = N(u) / D(u)`` (reduced, integer coefficients,
``D`` with a positive leading coefficient) and the world curve ``E(u) / M(u)`` (section 2.2). Everything is exact
rational arithmetic through the CH3b kernel: every loop reserves one unit before each element, and every exact
value goes through the bit-capped ``Ar``. Refusals are typed (``UNSUPPORTED_SURFACE`` with a sub-kind in
``sub_reason``; ``UNSUPPORTED_TRIM`` for the planes).

Knot parameters ``(u, v)`` are used throughout (design 2.2); the chart maps are never used here."""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from ..nurbs import NurbsSurface
from ._ch3_poly import (Kernel, ONE, ZERO, bernstein, padd, pgcd, pexact_div, pmul, poly_from, ppow, pscale, psub,
                        refusal)
from ._errors import RefusalCategory

__all__ = ["RuledSupport", "TrimPlane", "classify_surface", "plane_graph", "exact_float", "poly_dot3"]

US = RefusalCategory.UNSUPPORTED_SURFACE
UT = RefusalCategory.UNSUPPORTED_TRIM


def exact_float(k: Kernel, value) -> Fraction:
    """The exact value of a stored float (``NONFINITE_INPUT`` if it is not finite), size-guarded."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        number = math.nan
    if not math.isfinite(number):
        raise refusal(RefusalCategory.NONFINITE_INPUT, "a stored coordinate is not finite")
    return k.ar.exact_float(number)


@dataclass(frozen=True)
class RuledSupport:
    """The exact data of a face support of the ruled rational class.

    ``A``, ``B``: the homogeneous rows ``(x, y, z, w)`` of polynomials in ``u`` (bottom row ``v = k1``, top row
    ``v = k2``); ``W`` the common weight polynomial; ``span_u = (k_p, k_{p+1})``, ``span_v = (k1, k2)``;
    ``u_range``, ``v_range`` the face's range box (exact rationals, inside the spans); ``chart_normal`` the
    rational ``m`` and ``c0``, ``c1`` with ``m . S = c0 + c1 (v - k1) / (k2 - k1)`` (C4)."""

    degree_u: int
    span_u: tuple
    span_v: tuple
    u_range: tuple
    v_range: tuple
    A: tuple
    B: tuple
    W: tuple
    chart_normal: tuple
    chart_c0: Fraction
    chart_c1: Fraction
    key: tuple


@dataclass(frozen=True)
class TrimPlane:
    """A trim plane and its exact graph on a support: ``v = N / D`` and the world curve ``E / M`` (``E``: three
    polynomials). ``N`` may be the zero polynomial (the graph ``v = 0``); ``D`` is nonzero."""

    origin: tuple
    u_vector: tuple
    v_vector: tuple
    normal: tuple
    offset: Fraction
    N: tuple
    D: tuple
    E: tuple
    M: tuple


def poly_dot3(k: Kernel, m: tuple, rows: tuple) -> tuple:
    """``sum_c m_c rows_c`` for three polynomials."""
    out = ()
    for c in range(3):
        k.tick("admission_element")
        out = padd(k, out, pscale(k, rows[c], m[c]))
    return out


# ------------------------------------------------------------------ C1 and the net

def _check_class(k: Kernel, surface) -> int:
    """C1 up to the weights; returns ``degree_u``."""
    if not isinstance(surface, NurbsSurface):
        raise refusal(US, "the support is not a NURBS surface", "not_nurbs")
    if surface.degree_v != 1:
        raise refusal(US, f"degree_v is {surface.degree_v}, the class has 1", "degree_v")
    p = surface.degree_u
    if p not in (1, 2):
        raise refusal(US, f"degree_u is {p}, the class has 1 or 2", "degree_u")
    count_u, count_v = len(surface.controls), len(surface.controls[0])
    if count_v != 2:
        raise refusal(US, "the v direction has more than two control rows (an interior knot)", "interior_knot")
    if count_u != p + 1:
        raise refusal(US, "the u direction has interior knots", "interior_knot")
    ku, kv = surface.knots_u, surface.knots_v
    for _ in range(len(ku) + len(kv)):
        k.tick("admission_element")
    if not (len(ku) == 2 * p + 2 and ku[:p + 1] == (ku[0],) * (p + 1) and ku[p + 1:] == (ku[-1],) * (p + 1)
            and ku[0] < ku[-1] and kv[0] == kv[1] < kv[2] == kv[3]):
        raise refusal(US, "the knot vectors are not clamped single spans", "knots_not_clamped")
    return p


def _net(k: Kernel, surface: NurbsSurface, p: int, span_u: tuple, weights: tuple):
    """``(A, B, W)``: the homogeneous rows as power-basis polynomials in the knot parameter ``u``."""
    ka, kb = span_u
    width = k.ar.sub(kb, ka)
    t = poly_from(k, [k.ar.div(-ka, width), k.ar.div(ONE, width)])
    omt = poly_from(k, [k.ar.div(kb, width), k.ar.div(-ONE, width)])
    basis = []
    for i in range(p + 1):
        k.tick("admission_element")
        basis.append(pscale(k, pmul(k, ppow(k, t, i), ppow(k, omt, p - i)), math.comb(p, i)))
    rows = []
    for j in (0, 1):
        coordinates = []
        for c in range(4):
            acc = ()
            for i in range(p + 1):
                k.tick("admission_element")
                point = surface.controls[i][j]
                value = weights[i][j] if c == 3 else k.ar.mul(weights[i][j], exact_float(k, point[c]))
                acc = padd(k, acc, pscale(k, basis[i], value))
            coordinates.append(acc)
        rows.append(tuple(coordinates))
    return rows[0], rows[1], rows[0][3]


# ------------------------------------------------------------------ C4

def _controls(k: Kernel, surface: NurbsSurface, p: int) -> list:
    out = []
    for i in range(p + 1):
        k.tick("admission_element")
        out.append([tuple(exact_float(k, c) for c in surface.controls[i][j]) for j in (0, 1)])
    return out


def _chart_normal(k: Kernel, pts: list, p: int):
    """``(m, c0, c1)`` with all bottom-row control points on ``m . x = c0`` and all top-row points on
    ``m . x = c0 + c1``, ``c1 != 0``, ``m`` rational; or ``None``. (Equal weight rows make this equivalent to
    ``m . A~_xyz = c0 W`` and ``m . (B~_xyz - A~_xyz) = c1 W``.)"""
    ar = k.ar
    g = ar.sub3(pts[0][1], pts[0][0])
    diffs = []
    for i in range(1, p + 1):
        k.tick("admission_element")
        diffs.append(ar.sub3(pts[i][0], pts[0][0]))
        diffs.append(ar.sub3(pts[i][1], pts[0][1]))
    nonzero = [d for d in diffs if any(d)]
    if not nonzero:
        m = g
    else:
        d1 = nonzero[0]
        d2 = None
        for d in nonzero[1:]:
            k.tick("admission_element")
            if any(ar.cross(d1, d)):
                d2 = d
                break
        if d2 is not None:
            m = ar.cross(d1, d2)
        else:
            gg = ar.dot(d1, d1)
            dg = ar.dot(d1, g)
            m = tuple(ar.sub(ar.mul(g[c], gg), ar.mul(d1[c], dg)) for c in range(3))
    if not any(m):
        return None
    c0 = ar.dot(m, pts[0][0])
    c1 = ar.sub(ar.dot(m, pts[0][1]), c0)
    if c1 == 0:
        return None
    for i in range(p + 1):
        k.tick("admission_element")
        if ar.dot(m, pts[i][0]) != c0 or ar.dot(m, pts[i][1]) != ar.add(c0, c1):
            return None
    return m, c0, c1


def _rows_noncollinear(k: Kernel, pts: list, p: int) -> bool:
    """Every ``v``-row's control points are non-collinear (``p = 2``) or distinct (``p = 1``) for all ``s`` in
    ``[0, 1]`` (``s = (v - k1) / (k2 - k1)``): a component of the cross product (``p = 2``) or of the difference
    (``p = 1``) has strictly one-signed Bernstein coefficients (sufficient, as in the design)."""
    def row_point(i):          # P_i(s) as three linear polynomials in s
        return tuple(poly_from(k, [pts[i][0][c], k.ar.sub(pts[i][1][c], pts[i][0][c])]) for c in range(3))

    rows = [row_point(i) for i in range(p + 1)]
    a = tuple(psub(k, rows[1][c], rows[0][c]) for c in range(3))
    if p == 1:
        candidates = list(a)
    else:
        b = tuple(psub(k, rows[2][c], rows[0][c]) for c in range(3))
        candidates = [psub(k, pmul(k, a[(c + 1) % 3], b[(c + 2) % 3]), pmul(k, a[(c + 2) % 3], b[(c + 1) % 3]))
                      for c in range(3)]
    for q in candidates:
        k.tick("admission_element")
        if not q:
            continue
        sign = 0
        ok = True
        for c in bernstein(k, q, ZERO, ONE):
            k.tick("bernstein_coefficient")
            s = (c > 0) - (c < 0)
            if s == 0 or (sign and s != sign):
                ok = False
                break
            sign = s
        if ok:
            return True
    return False


def classify_surface(k: Kernel, surface) -> RuledSupport:
    """C1 (class) and C4 (chart injectivity) of a face support, and its exact knot-space net."""
    p = _check_class(k, surface)
    ku, kv = surface.knots_u, surface.knots_v
    span_u = (exact_float(k, ku[0]), exact_float(k, ku[-1]))
    span_v = (exact_float(k, kv[0]), exact_float(k, kv[-1]))
    u_range = tuple(exact_float(k, x) for x in surface.u_range)
    v_range = tuple(exact_float(k, x) for x in surface.v_range)
    for (lo, hi), (a, b) in ((u_range, span_u), (v_range, span_v)):
        k.tick("admission_element")
        if not (a <= lo < hi <= b):
            raise refusal(US, "a parameter range lies outside the knot domain", "range")
    if surface.weights is None:
        weights = tuple((ONE, ONE) for _ in range(p + 1))
    else:
        weights = []
        for row in surface.weights:
            k.tick("admission_element")
            if row[0] != row[1]:
                raise refusal(US, "the two weight rows differ", "unequal_weight_rows")
            weights.append((exact_float(k, row[0]),) * 2)
        weights = tuple(weights)
    A, B, W = _net(k, surface, p, span_u, weights)
    pts = _controls(k, surface, p)
    chart = _chart_normal(k, pts, p)
    if chart is None or not _rows_noncollinear(k, pts, p):
        raise refusal(US, "the chart is not certified injective (no rational direction fixes v, or a degenerate row)",
                      "not_graph_like")
    return RuledSupport(p, span_u, span_v, u_range, v_range, A, B, W, chart[0], chart[1], chart[2],
                        surface.support_key())


# ------------------------------------------------------------------ plane graphs (section 2.2)

def _joint_integer_form(k: Kernel, N: tuple, D: tuple):
    """``(N, D)`` scaled by one rational so that all coefficients are integers with gcd 1 and ``D`` has a
    positive leading coefficient."""
    den = 1
    for c in N + D:
        k.tick("poly_coefficient")
        den = k.ar.fit(den // math.gcd(den, c.denominator) * c.denominator)
    g = 0
    for c in N + D:
        k.tick("poly_coefficient")
        g = math.gcd(g, int(k.ar.mul(c, den)))
    scale = Fraction(den, g)
    if D[-1] < 0:
        scale = -scale
    return pscale(k, N, scale), pscale(k, D, scale)


def plane_graph(k: Kernel, support: RuledSupport, origin: tuple, u_vector: tuple, v_vector: tuple) -> TrimPlane:
    """The graph ``v = N / D`` of the plane ``m . x = d`` (``m = u_vector x v_vector``) on ``support``.

    Refuses ``UNSUPPORTED_TRIM``: ``zero_plane`` (``m = 0``), ``face_in_plane`` (``alpha = beta = 0``) and
    ``ruling_trim`` (``beta = 0``: the plane is parallel to every ruling, so it is no graph)."""
    ar = k.ar
    m = ar.cross(u_vector, v_vector)
    if not any(m):
        raise refusal(UT, "the trim plane is degenerate", "zero_plane")
    d = ar.dot(m, origin)
    k1, k2 = support.span_v
    width = ar.sub(k2, k1)
    mA = poly_dot3(k, m, support.A[:3])
    mB = poly_dot3(k, m, support.B[:3])
    alpha = psub(k, pscale(k, psub(k, pscale(k, mA, k2), pscale(k, mB, k1)), ar.div(ONE, width)),
                 pscale(k, support.W, d))
    beta = pscale(k, psub(k, mB, mA), ar.div(ONE, width))
    if not beta:
        raise refusal(UT, "the plane is parallel to the rulings" if alpha else "the face lies in the plane",
                      "ruling_trim" if alpha else "face_in_plane")
    N = psub(k, (), alpha)
    D = beta
    g = pgcd(k, N, D) if N else ()
    if g and len(g) > 1:
        N, D = pexact_div(k, N, g), pexact_div(k, D, g)
    if not N:
        D = (ONE,)
    N, D = _joint_integer_form(k, N, D)
    # E = (k2 D - N) A_xyz + (N - k1 D) B_xyz, M = (k2 - k1) W D, common factor removed
    first = psub(k, pscale(k, D, k2), N)
    second = psub(k, N, pscale(k, D, k1))
    E = []
    for c in range(3):
        k.tick("admission_element")
        E.append(padd(k, pmul(k, first, support.A[c]), pmul(k, second, support.B[c])))
    M = pscale(k, pmul(k, support.W, D), width)
    common = M
    for c in range(3):
        if E[c]:
            common = pgcd(k, common, E[c])
    if len(common) > 1:
        M = pexact_div(k, M, common)
        E = [pexact_div(k, e, common) if e else e for e in E]
    return TrimPlane(origin, u_vector, v_vector, m, d, N, D, tuple(E), M)
