"""CH3c: the curved query engine (design CH3 sections 3.3, 4 and 5, narrowed plan of 2026-10-21).

``evaluate_curved`` answers a half-space query ``n . r <= d`` on a curved (B1-class) preparation. The solid
``K = S & {n . r <= d}`` is bounded by the clipped planar faces, the clipped curved faces and the cap; every
quantity is a sum of 1-D integrals of rational functions of the knot parameter ``u`` (arcs, graph pieces) and
of tetrahedron / triangle terms of the exact straight segments, evaluated as rigorous enclosures:

* a quantity is a list of **terms**: ``("i", coefficient, numerator, denominator factors, a, b)`` (a 1-D integral
  enclosed by the CH3b integrator, ``a``, ``b`` rational or algebraic) and ``("p", f)`` (a point term, ``f(bits)``
  an ``Enclosure`` computed by outward-rounded interval arithmetic from corner and cut-point enclosures refined to
  ``2**-bits``). ``Builder.evaluate`` splits the target width over the integrals and refines the points.
* the claim protocol (design 4.4): the core has width ``<= a'/3`` (``a' = 3a/4``), the published claim is
  ``[lo - w, hi + w]`` with ``w`` the core width, so the claim is at most ``3a/4`` wide and leaves a quarter of
  the accuracy ``a`` for the outward binary64 / decimal rounding of the forms.

Supported planes (narrowed plan, D-CH3-5 amendment): generic planes only. A plane through a vertex (E1), along or
tangent to an edge (E2, E5), containing a ruling (E4, zero-beta faces) or lying in a face, and the
extra-conservative case ``beta`` vanishing inside a cut face's ``u``-extent, refuse
``CURVED_SECTION_DEGENERACY`` with a sub-reason. Dry / full planes give ``EMPTY`` / ``FULL`` with the section
absent.

Orientation (design 3.3): the model's loops are anticlockwise about the outward normal, so every chain carries the
outward orientation: curved faces ``V_f = -(1/3) oint F_V du``, ``J_f,i = -(1/4) oint F_J,i du``; clipped planar
faces keep their loop direction with the chord closing exit -> enter; the cap is every cut piece reversed. A kept
graph portion of a curved face is directed ``-sign(beta) * sigma_chart`` (``+1``: increasing ``u``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from ._arith import Ar
from ._ch3_admission import RuledSupport, plane_alpha_beta
from ._ch3_admission_trims import enclose_coordinates
from ._ch3_algebraic import (AlgebraicNumber, Enclosure, compare, isolate_roots, lower_bound, sign_at,
                             upper_bound)
from ._ch3_integrate import divide_by_norm, enclose_rational_integral, grid_ceil, grid_floor
from ._ch3_loops import region_contains
from ._ch3_poly import (Kernel, ONE, ZERO, padd, pderiv, peval, pexact_div, pgcd, pmul, ppow, pscale, psub,
                        refusal, squarefree)
from ._errors import CertifiedHydrostaticsRefusal, RefusalCategory
from ._forms import build_enclosed_quantity
from ._query import WATERPLANE_NAMES
from ._result import (CertifiedHydrostaticsResult, Contacts, SectionResult, Topology, TopologyLoop, proof_document,
                      write_canonical)
from ._work import Meter

__all__ = ["evaluate_curved"]

CSD = RefusalCategory.CURVED_SECTION_DEGENERACY
VOLUME_TERMS = ("V", "Jx", "Jy", "Jz")


def _csd(message: str, sub_reason: str) -> CertifiedHydrostaticsRefusal:
    return refusal(CSD, message, sub_reason)


# ------------------------------------------------------------------ polynomial vectors

def _cross(k: Kernel, a: tuple, b: tuple) -> tuple:
    return tuple(psub(k, pmul(k, a[(c + 1) % 3], b[(c + 2) % 3]), pmul(k, a[(c + 2) % 3], b[(c + 1) % 3]))
                 for c in range(3))


def _dot_const(k: Kernel, q: tuple, v: tuple) -> tuple:
    out = ()
    for c in range(3):
        out = padd(k, out, pscale(k, v[c], q[c]))
    return out


def _det(k: Kernel, a: tuple, b: tuple, c: tuple) -> tuple:
    cr = _cross(k, b, c)
    out = ()
    for i in range(3):
        out = padd(k, out, pmul(k, a[i], cr[i]))
    return out


def _vec_deriv(k: Kernel, v: tuple) -> tuple:
    return tuple(pderiv(k, x) for x in v)


def _const_vec(q: tuple) -> tuple:
    return tuple((Fraction(c),) if c else () for c in q)


def _sign(p: tuple) -> int:
    return (p[-1] > 0) - (p[-1] < 0) if p else 0


def _clone(x):
    if isinstance(x, AlgebraicNumber):
        return AlgebraicNumber(x.poly, x.ordinal, x.lo, x.hi, x._sturm)
    return x


# ------------------------------------------------------------------ rational space curves E/M on a support

@dataclass
class Curve:
    """The curve ``E / M`` of a graph ``v = N / D`` on a support (unreduced): ``M = w W D`` split as
    ``scalar * prod(factors)`` over the non-constant polynomials ``W`` and ``D``."""

    support: RuledSupport
    N: tuple
    D: tuple
    E: tuple
    M: tuple
    scalar: Fraction
    factors: tuple

    def dens(self, e: int, ar: Ar):
        """``(scalar ** e, [(factor, e)])``: ``M ** e = scalar**e * prod(factor ** e)``."""
        scalar = ONE
        for _ in range(e):
            scalar = ar.mul(scalar, self.scalar)
        return scalar, [(f, e) for f in self.factors]


def make_curve(k: Kernel, support: RuledSupport, N: tuple, D: tuple) -> Curve:
    ar = k.ar
    k1, k2 = support.span_v
    w = ar.sub(k2, k1)
    first = psub(k, pscale(k, D, k2), N)
    second = psub(k, N, pscale(k, D, k1))
    E = tuple(padd(k, pmul(k, first, support.A[c]), pmul(k, second, support.B[c])) for c in range(3))
    scalar = w
    factors = []
    for f in (support.W, D):
        if len(f) == 1:
            scalar = ar.mul(scalar, f[0])
        else:
            factors.append(f)
    M = pscale(k, pmul(k, support.W, D), w)
    return Curve(support, N, D, E, M, scalar, tuple(factors))


def mixed_dens(k: Kernel, support: RuledSupport, D: tuple, e_w: int, e_d: int):
    """``(scalar, dens)`` for ``W ** e_w * D ** e_d`` (constant factors folded into the scalar)."""
    ar = k.ar
    scalar = ONE
    dens = []
    for f, e in ((support.W, e_w), (D, e_d)):
        if len(f) == 1:
            for _ in range(e):
                scalar = ar.mul(scalar, f[0])
        elif e:
            dens.append((f, e))
    return scalar, dens


@dataclass
class FacePolys:
    """The polynomials of a curved face's Green primitives (design 4.1): ``P = a + v b`` per coordinate,
    ``T0 = det(a, a', b)``, ``T1 = det(a, b', b)`` (so ``P . (P_u x P_v) = T0 + v T1``), and the coefficient lists
    (by power of ``v``) of ``F_V`` and ``F_J,i``."""

    FV: list
    FJ: list


def face_polys(k: Kernel, support: RuledSupport) -> FacePolys:
    ar = k.ar
    k1, k2 = support.span_v
    inv_w = ar.div(ONE, ar.sub(k2, k1))
    a, b = [], []
    for c in range(3):
        a.append(pscale(k, psub(k, pscale(k, support.A[c], k2), pscale(k, support.B[c], k1)), inv_w))
        b.append(pscale(k, psub(k, support.B[c], support.A[c]), inv_w))
    a, b = tuple(a), tuple(b)
    da, db = _vec_deriv(k, a), _vec_deriv(k, b)
    T0, T1 = _det(k, a, da, b), _det(k, a, db, b)
    k1_2, k1_3 = ar.mul(k1, k1), ar.mul(ar.mul(k1, k1), k1)
    half, third = Fraction(1, 2), Fraction(1, 3)
    FV = [psub(k, pscale(k, T0, -k1), pscale(k, T1, ar.mul(half, k1_2))), T0, pscale(k, T1, half)]
    FJ = []
    for i in range(3):
        aT0, aT1b0 = pmul(k, a[i], T0), padd(k, pmul(k, a[i], T1), pmul(k, b[i], T0))
        bT1 = pmul(k, b[i], T1)
        c0 = psub(k, psub(k, pscale(k, aT0, -k1), pscale(k, aT1b0, ar.mul(half, k1_2))),
                  pscale(k, bT1, ar.mul(third, k1_3)))
        FJ.append([c0, aT0, pscale(k, aT1b0, half), pscale(k, bT1, third)])
    return FacePolys(FV, FJ)


def substitute_graph(k: Kernel, coeffs: list, N: tuple, D: tuple) -> tuple:
    """``(num, m)`` with ``sum_j f_j (N/D)**j = num / D**m``, ``m = len(coeffs) - 1``."""
    m = len(coeffs) - 1
    num = ()
    n_pow = [(ONE,)]
    for _ in range(m):
        n_pow.append(pmul(k, n_pow[-1], N))
    d_pow = [(ONE,)]
    for _ in range(m):
        d_pow.append(pmul(k, d_pow[-1], D))
    for j, f in enumerate(coeffs):
        k.tick("curved_element")
        num = padd(k, num, pmul(k, f, pmul(k, n_pow[j], d_pow[m - j])))
    return num, m


# ------------------------------------------------------------------ interval arithmetic (outward rounded)

class IA:
    """Interval arithmetic on ``Enclosure``s; results are rounded outward to the grid ``2**-g`` only when their
    exact form has grown beyond ``4 g`` bits (so rational inputs stay exact)."""

    def __init__(self, k: Kernel, bits: int) -> None:
        self.k = k
        self.g = bits + 32

    def _round(self, lo: Fraction, hi: Fraction) -> Enclosure:
        limit = 4 * self.g
        if max(abs(lo.numerator).bit_length(), lo.denominator.bit_length(), abs(hi.numerator).bit_length(),
               hi.denominator.bit_length()) > limit:
            lo, hi = grid_floor(self.k, lo, self.g), grid_ceil(self.k, hi, self.g)
        return Enclosure(lo, hi)

    def const(self, c) -> Enclosure:
        return Enclosure.exact(c)

    def add(self, a: Enclosure, b: Enclosure) -> Enclosure:
        ar = self.k.ar
        return self._round(ar.add(a.lo, b.lo), ar.add(a.hi, b.hi))

    def sub(self, a: Enclosure, b: Enclosure) -> Enclosure:
        ar = self.k.ar
        return self._round(ar.sub(a.lo, b.hi), ar.sub(a.hi, b.lo))

    def mul(self, a: Enclosure, b: Enclosure) -> Enclosure:
        ar = self.k.ar
        self.k.tick("interval_op")
        products = [ar.mul(x, y) for x in (a.lo, a.hi) for y in (b.lo, b.hi)]
        return self._round(min(products), max(products))

    def scale(self, a: Enclosure, c) -> Enclosure:
        ar = self.k.ar
        x, y = ar.mul(a.lo, c), ar.mul(a.hi, c)
        return self._round(min(x, y), max(x, y))

    def div(self, a: Enclosure, b: Enclosure) -> Enclosure:
        ar = self.k.ar
        self.k.tick("interval_op")
        if b.lo <= 0 <= b.hi:
            raise _NeedBits()
        quotients = [ar.div(x, y) for x in (a.lo, a.hi) for y in (b.lo, b.hi)]
        return self._round(min(quotients), max(quotients))

    def dot_const(self, q: tuple, v: tuple) -> Enclosure:
        out = Enclosure.exact(ZERO)
        for c in range(3):
            out = self.add(out, self.scale(v[c], q[c]))
        return out

    def cross(self, a: tuple, b: tuple) -> tuple:
        return tuple(self.sub(self.mul(a[(c + 1) % 3], b[(c + 2) % 3]), self.mul(a[(c + 2) % 3], b[(c + 1) % 3]))
                     for c in range(3))


class _NeedBits(Exception):
    """An interval division met a zero-containing denominator: the caller retries with more bits."""


# ------------------------------------------------------------------ the engine

@dataclass
class Arc:
    edge: int
    fid: int
    trim: object
    curve: Curve
    g: tuple
    lo: AlgebraicNumber
    hi: AlgebraicNumber
    lo_vertex: int
    hi_vertex: int
    s_lo: int = 0
    roots: tuple = ()


def _pieces_of(nodes: list, inside: list, ascending: bool) -> list:
    pieces = [(nodes[i], nodes[i + 1], inside[i]) for i in range(len(inside))]
    if not ascending:
        pieces = [(b, a, ins) for (a, b, ins) in reversed(pieces)]
    return pieces


class Builder:
    def __init__(self, prepared, query, meter: Meter) -> None:
        self.prepared = prepared
        self.snapshot = prepared.snapshot
        self.query = query
        self.k = Kernel(meter, prepared.policy.kernel_limits())
        self.ar = self.k.ar
        self.curved = self.snapshot.curved
        self.n = query.normal
        self.d = query.offset
        self.n2 = query.normal_squared
        self.o = query.waterplane_origin
        self.e1, self.e2 = query.waterplane_axes
        ar = self.ar
        # fan apexes: the projection of the body origin (V, J) and of the waterplane origin (waterplane moments)
        self.q0 = tuple(ar.div(ar.mul(self.d, c), self.n2) for c in self.n)
        t = ar.div(ar.sub(self.d, ar.dot(self.n, self.o)), self.n2)
        self.qw = tuple(ar.add(self.o[c], ar.mul(t, self.n[c])) for c in range(3))
        rel = ar.sub3(self.qw, self.o)
        self.qw_uv = (ar.dot(self.e1, rel), ar.dot(self.e2, rel))
        self.terms: dict = {name: [] for name in VOLUME_TERMS + WATERPLANE_NAMES}
        self.pos_cache: dict = {}
        self.face_polys: dict = {}
        self.vsign: dict = {}
        self.vcorner: dict = {}
        self.arcs: dict = {}
        self.line_cuts: dict = {}
        self.cut_roots: dict = {}
        self.exec_info = {"integrals": 0, "taylor_pieces": 0, "point_bits": 0, "core": {}}
        self.vertex_clone: dict = {}

    # ---------------------------------------------------------------- setup and analysis

    def analyse(self) -> None:
        k, ar = self.k, self.ar
        curved = self.curved
        for edge, (fid, trim) in sorted(curved.arcs.items()):
            k.tick("curved_element")
            support = curved.faces[fid].support
            curve = make_curve(k, support, trim.plane.N, trim.plane.D)
            g = psub(k, _dot_const(k, self.n, curve.E), pscale(k, curve.M, self.d))
            lo, hi = _clone(trim.u_lo), _clone(trim.u_hi)
            if trim.curve_increasing:
                lo_vertex, hi_vertex = trim.start_vertex, trim.end_vertex
            else:
                lo_vertex, hi_vertex = trim.end_vertex, trim.start_vertex
            self.arcs[edge] = Arc(edge, fid, trim, curve, g, lo, hi, lo_vertex, hi_vertex)
        for edge, arc in self.arcs.items():            # a plane containing a whole edge is named before a corner
            if not arc.g:
                raise _csd(f"edge {edge} lies in the plane", "edge_in_plane")
        for edge, arc in self.arcs.items():
            self._analyse_arc(arc)
        for fid, face in curved.faces.items():
            for vertex, corner in face.corners.items():
                if vertex not in self.vcorner:
                    for edge, arc in self.arcs.items():
                        if vertex in (arc.lo_vertex, arc.hi_vertex):
                            self.vcorner[vertex] = (_clone(corner.number), arc.curve)
                            break
        for vertex, position in sorted(self.snapshot.positions.items()):
            k.tick("curved_element")
            if vertex in self.vsign:
                continue
            h = ar.sub(ar.dot(self.n, position), self.d)
            if h == 0:
                raise _csd(f"the plane passes through vertex {vertex}", "plane_through_vertex")
            self.vsign[vertex] = 1 if h > 0 else -1
        for edge, (s, t) in sorted(self.snapshot.edges.items()):
            k.tick("curved_element")
            if edge not in self.arcs and self.vsign[s] != self.vsign[t]:
                self.line_cuts[edge] = (s, t)

    def _height_sign(self, arc: Arc, x) -> int:
        k = self.k
        s = sign_at(k, arc.g, x)
        if s == 0:
            raise _csd(f"the plane passes through a corner of edge {arc.edge}", "plane_through_vertex")
        return s * sign_at(k, arc.curve.M, x)

    def _outer(self, a, b) -> tuple:
        return lower_bound(a) - 1, upper_bound(b) + 1

    def _roots_between(self, p: tuple, a, b, closed: bool) -> list:
        """The distinct real roots of ``p`` in the open (or closed) interval between the algebraic numbers."""
        k = self.k
        out = []
        if len(squarefree(k, p)) < 2:
            return out
        lo, hi = self._outer(a, b)
        for r in isolate_roots(k, p, lo, hi):
            k.tick("curved_element")
            c_lo, c_hi = compare(k, r, a), compare(k, r, b)
            if (c_lo > 0 and c_hi < 0) or (closed and c_lo >= 0 and c_hi <= 0):
                out.append(r)
        return out

    def _analyse_arc(self, arc: Arc) -> None:
        k = self.k
        g = arc.g
        if not g:
            raise _csd(f"edge {arc.edge} lies in the plane", "edge_in_plane")
        s_lo = self._height_sign(arc, arc.lo)
        s_hi = self._height_sign(arc, arc.hi)
        self.vsign[arc.lo_vertex] = s_lo
        self.vsign[arc.hi_vertex] = s_hi
        multiple = pgcd(k, g, pderiv(k, g))
        if len(multiple) > 1 and self._roots_between(multiple, arc.lo, arc.hi, True):
            raise _csd(f"the plane is tangent to edge {arc.edge}", "edge_tangency")
        roots = self._roots_between(g, arc.lo, arc.hi, False)
        if s_hi != s_lo * (-1) ** len(roots):
            raise _csd(f"inconsistent cut count on edge {arc.edge}", "inconsistent_section")
        arc.s_lo = s_lo
        arc.roots = tuple(roots)
        for i, r in enumerate(roots):
            self.cut_roots[(arc.edge, i)] = r

    # ---------------------------------------------------------------- occupancy

    def occupancy(self) -> str:
        if self.line_cuts or self.cut_roots:
            return "PARTIAL"
        signs = set(self.vsign.values())
        return "FULL" if signs == {-1} else "EMPTY"

    def check_curved_faces(self, partial: bool) -> None:
        k = self.k
        self.extent: dict = {}
        self.beta_sign: dict = {}
        for fid, face in self.curved.faces.items():
            k.tick("curved_element")
            support = face.support
            alpha, beta = plane_alpha_beta(k, support, self.n, self.d)
            lo = hi = None
            for trim in face.trims:
                a, b = _clone(trim.u_lo), _clone(trim.u_hi)
                lo = a if lo is None or compare(k, a, lo) < 0 else lo
                hi = b if hi is None or compare(k, b, hi) > 0 else hi
            self.extent[fid] = (lo, hi)
            if not beta:
                # zero-beta face (design 5.1, row Z): the plane is parallel to every ruling, f = alpha(u). A root of
                # alpha in the extent is a ruling in the plane; otherwise the face lies wholly on one side (the sign
                # of alpha), it has no cut arc, and its corner signs already say which side.
                if not alpha:
                    raise _csd(f"face {fid} lies in the plane", "face_in_plane")
                if len(alpha) > 1 and self._roots_between(alpha, lo, hi, True):
                    raise _csd(f"the plane contains a ruling of face {fid}", "ruling_in_plane")
                continue
            common = pgcd(k, alpha, beta) if alpha else beta
            if len(common) > 1 and self._roots_between(common, lo, hi, True):
                raise _csd(f"the plane contains a ruling of face {fid}", "ruling_in_plane")
            if partial and self._cuts_of_face(fid):
                if len(beta) > 1 and self._roots_between(beta, lo, hi, True):
                    raise _csd(f"beta vanishes inside the extent of face {fid} (outside the supported class)",
                               "beta_root_in_extent")
                self.beta_sign[fid] = sign_at(k, beta, lo)

    def _cuts_of_face(self, fid: int) -> list:
        out = []
        for trim in self.curved.faces[fid].trims:
            arc = self.arcs[trim.edge]
            out.extend((arc.edge, i) for i in range(len(arc.roots)))
        return out

    # ---------------------------------------------------------------- positions

    def _vertex_pos(self, v: int, bits: int) -> tuple:
        key = ("v", v, bits)
        if key not in self.pos_cache:
            if v in self.vcorner:
                number, curve = self.vcorner[v]
                self.pos_cache[key] = enclose_coordinates(self.k, curve.E, curve.M, number,
                                                          Fraction(1, 1 << bits))
            else:
                self.pos_cache[key] = tuple(Enclosure.exact(c) for c in self.snapshot.positions[v])
        return self.pos_cache[key]

    def pos(self, node: tuple, bits: int) -> tuple:
        kind, ident = node
        if kind == "v":
            return self._vertex_pos(ident, bits)
        key = ("c", ident, bits)
        if key in self.pos_cache:
            return self.pos_cache[key]
        edge = ident[0]
        if edge in self.arcs:
            arc = self.arcs[edge]
            value = enclose_coordinates(self.k, arc.curve.E, arc.curve.M, self.cut_roots[ident], Fraction(1, 1 << bits))
        else:
            s, t = self.line_cuts[edge]
            extra = bits
            while True:
                ia = IA(self.k, extra)
                a, b = self._vertex_pos(s, extra), self._vertex_pos(t, extra)
                try:
                    ha = ia.sub(ia.dot_const(self.n, a), Enclosure.exact(self.d))
                    hb = ia.sub(ia.dot_const(self.n, b), Enclosure.exact(self.d))
                    ratio = ia.div(ha, ia.sub(ha, hb))
                    value = tuple(ia.add(a[c], ia.mul(ratio, ia.sub(b[c], a[c]))) for c in range(3))
                    break
                except _NeedBits:
                    extra += 32
                    if extra > self.k.limits.max_isolation_bits:
                        raise refusal(RefusalCategory.UNDECIDED_PREDICATE, "cut point not resolved",
                                      "cut_point") from None
        self.pos_cache[key] = value
        return value

    def node_u(self, arc: Arc, node: tuple):
        kind, ident = node
        if kind == "v":
            return arc.lo if ident == arc.lo_vertex else arc.hi
        return self.cut_roots[ident]

    # ---------------------------------------------------------------- pieces of loops

    def arc_pieces(self, arc: Arc, ascending: bool) -> list:
        nodes = [("v", arc.lo_vertex)] + [("c", (arc.edge, i)) for i in range(len(arc.roots))] \
            + [("v", arc.hi_vertex)]
        inside = [arc.s_lo * (-1) ** i < 0 for i in range(len(arc.roots) + 1)]
        return _pieces_of(nodes, inside, ascending)

    def line_pieces(self, edge: int, start: int, end: int) -> list:
        if edge in self.line_cuts:
            c = ("c", (edge, 0))
            return [(("v", start), c, self.vsign[start] < 0), (c, ("v", end), self.vsign[end] < 0)]
        return [(("v", start), ("v", end), self.vsign[start] < 0)]

    # ---------------------------------------------------------------- terms

    def add_int(self, name: str, coef, num: tuple, scalar, dens: list, a, b) -> None:
        if coef == 0 or not num:
            return
        self.terms[name].append(("i", self.ar.div(coef, scalar), num, dens, a, b))

    def add_point(self, name: str, fn) -> None:
        self.terms[name].append(("p", fn))

    def segment_terms(self, apex: tuple, p: tuple, q: tuple) -> None:
        """Tetrahedra (V, J) of the segment ``p -> q`` with the fan apex ``apex``."""
        cache: dict = {}

        def det(bits):
            if bits not in cache:
                ia = IA(self.k, bits)
                a, b = self.pos(p, bits), self.pos(q, bits)
                cache[bits] = (ia, a, b, ia.dot_const(apex, ia.cross(a, b)))
            return cache[bits]

        self.add_point("V", lambda bits: det(bits)[0].scale(det(bits)[3], Fraction(1, 6)))
        for i, name in enumerate(("Jx", "Jy", "Jz")):
            def fn(bits, i=i):
                ia, a, b, dt = det(bits)
                s = ia.add(ia.add(a[i], b[i]), Enclosure.exact(apex[i]))
                return ia.scale(ia.mul(dt, s), Fraction(1, 24))
            self.add_point(name, fn)

    def triangle_terms(self, p: tuple, q: tuple) -> None:
        """Waterplane triangle ``(qw, p, q)`` (a cap piece in the cap's direction)."""
        cache: dict = {}
        n, o, e1, e2, qw = self.n, self.o, self.e1, self.e2, self.qw

        def data(bits):
            if bits not in cache:
                ia = IA(self.k, bits)
                a, b = self.pos(p, bits), self.pos(q, bits)
                ea = tuple(ia.sub(a[c], Enclosure.exact(qw[c])) for c in range(3))
                eb = tuple(ia.sub(b[c], Enclosure.exact(qw[c])) for c in range(3))
                weight = ia.scale(ia.dot_const(n, ia.cross(ea, eb)), Fraction(1, 2))
                us, vs = [Enclosure.exact(self.qw_uv[0])], [Enclosure.exact(self.qw_uv[1])]
                for pt in (a, b):
                    rel = tuple(ia.sub(pt[c], Enclosure.exact(o[c])) for c in range(3))
                    us.append(ia.dot_const(e1, rel))
                    vs.append(ia.dot_const(e2, rel))
                cache[bits] = (ia, weight, us, vs)
            return cache[bits]

        def total(ia, items):
            out = Enclosure.exact(ZERO)
            for x in items:
                out = ia.add(out, x)
            return out

        def square_sum(ia, xs):                     # sum x_i^2 + sum_{i<j} x_i x_j
            return total(ia, [ia.mul(xs[i], xs[i]) for i in range(3)]
                         + [ia.mul(xs[i], xs[j]) for i, j in ((0, 1), (0, 2), (1, 2))])

        def cross_sum(ia, xs, ys):                  # 2 sum x_i y_i + sum_{i != j} x_i y_j
            return total(ia, [ia.scale(ia.mul(xs[i], ys[i]), 2) for i in range(3)]
                         + [ia.mul(xs[i], ys[j]) for i in range(3) for j in range(3) if i != j])

        self.add_point("A", lambda bits: data(bits)[1])

        def first(bits, which):
            ia, weight, us, vs = data(bits)
            return ia.scale(ia.mul(weight, total(ia, us if which == 0 else vs)), Fraction(1, 3))
        self.add_point("Mu", lambda bits: first(bits, 0))
        self.add_point("Mv", lambda bits: first(bits, 1))

        def second(bits, which):
            ia, weight, us, vs = data(bits)
            if which == 0:
                return ia.scale(ia.mul(weight, square_sum(ia, us)), Fraction(1, 6))
            if which == 1:
                return ia.scale(ia.mul(weight, square_sum(ia, vs)), Fraction(1, 6))
            return ia.scale(ia.mul(weight, cross_sum(ia, us, vs)), Fraction(1, 12))
        self.add_point("Iuu", lambda bits: second(bits, 0))
        self.add_point("Ivv", lambda bits: second(bits, 1))
        self.add_point("Iuv", lambda bits: second(bits, 2))

    def fan_arc_terms(self, curve: Curve, apex: tuple, sign: int, a, b) -> None:
        """``sign * int_a^b`` of the V and J fan integrands of ``curve`` (design 4.1, planar face and cap arcs)."""
        k, ar = self.k, self.ar
        E, M = curve.E, curve.M
        dE = _vec_deriv(k, E)
        X = _cross(k, E, dE)
        det = _dot_const(k, apex, X)
        scalar2, dens2 = curve.dens(2, ar)
        self.add_int("V", Fraction(sign, 6), det, scalar2, dens2, a, b)
        scalar3, dens3 = curve.dens(3, ar)
        for i, name in enumerate(("Jx", "Jy", "Jz")):
            f = padd(k, pscale(k, M, apex[i]), pscale(k, E[i], 2))
            self.add_int(name, Fraction(sign, 24), pmul(k, det, f), scalar3, dens3, a, b)

    def waterplane_arc_terms(self, curve: Curve, sign: int, a, b) -> None:
        k, ar = self.k, self.ar
        E, M = curve.E, curve.M
        dE, dM = _vec_deriv(k, E), pderiv(k, M)
        qv = _const_vec(self.qw)
        X = _cross(k, E, dE)
        qE, qdE = _cross(k, qv, E), _cross(k, qv, dE)
        inner = tuple(padd(k, psub(k, X[c], pmul(k, M, qdE[c])), pmul(k, dM, qE[c])) for c in range(3))
        Z = pscale(k, _dot_const(k, self.n, inner), Fraction(1, 2))
        eo = (ar.dot(self.e1, self.o), ar.dot(self.e2, self.o))
        Pu = psub(k, _dot_const(k, self.e1, E), pscale(k, M, eo[0]))
        Pv = psub(k, _dot_const(k, self.e2, E), pscale(k, M, eo[1]))
        qu, qv_ = self.qw_uv
        M2 = pmul(k, M, M)
        s2, d2 = curve.dens(2, ar)
        s3, d3 = curve.dens(3, ar)
        s4, d4 = curve.dens(4, ar)
        self.add_int("A", Fraction(sign), Z, s2, d2, a, b)
        self.add_int("Mu", Fraction(sign, 3), pmul(k, padd(k, pscale(k, M, qu), pscale(k, Pu, 2)), Z), s3, d3, a, b)
        self.add_int("Mv", Fraction(sign, 3), pmul(k, padd(k, pscale(k, M, qv_), pscale(k, Pv, 2)), Z), s3, d3, a, b)

        def quad(x, qx, y, qy, diag):
            # (qx qy M^2 + qx Py M + qy Px M + c Px Py), c = 3 (u,u / v,v: diag) or 3 (u,v)
            out = pscale(k, M2, ar.mul(qx, qy))
            out = padd(k, out, pscale(k, pmul(k, M, y), qx))
            out = padd(k, out, pscale(k, pmul(k, M, x), qy))
            return padd(k, out, pscale(k, pmul(k, x, y), 3))

        uu = padd(k, padd(k, pscale(k, M2, ar.mul(qu, qu)), pscale(k, pmul(k, M, Pu), ar.mul(2, qu))),
                  pscale(k, pmul(k, Pu, Pu), 3))
        vv = padd(k, padd(k, pscale(k, M2, ar.mul(qv_, qv_)), pscale(k, pmul(k, M, Pv), ar.mul(2, qv_))),
                  pscale(k, pmul(k, Pv, Pv), 3))
        uv = quad(Pu, qu, Pv, qv_, False)
        for name, poly in (("Iuu", uu), ("Ivv", vv), ("Iuv", uv)):
            self.add_int(name, Fraction(sign, 6), pmul(k, poly, Z), s4, d4, a, b)

    def curved_piece_terms(self, fid: int, N: tuple, D: tuple, sign: int, a, b) -> None:
        """``-(1/3) sign int F_V(u, N/D) du`` and the J analogues for a chain piece on the graph ``v = N/D``."""
        k, ar = self.k, self.ar
        support = self.curved.faces[fid].support
        fp = self.face_polys[fid]
        num, m = substitute_graph(k, fp.FV, N, D)
        scalar, dens = mixed_dens(k, support, D, 3, m)
        self.add_int("V", Fraction(-sign, 3), num, scalar, dens, a, b)
        for i, name in enumerate(("Jx", "Jy", "Jz")):
            num, m = substitute_graph(k, fp.FJ[i], N, D)
            scalar, dens = mixed_dens(k, support, D, 4, m)
            self.add_int(name, Fraction(-sign, 4), num, scalar, dens, a, b)

    # ---------------------------------------------------------------- the faces

    def order_nodes(self, nodes: list, line_dir: tuple) -> list:
        """Nodes sorted by their coordinate along ``line_dir`` (exact order by refining the enclosures)."""
        k = self.k

        def param(node, bits):
            ia = IA(k, bits)
            return ia.dot_const(line_dir, self.pos(node, bits))

        def less(x, y):
            bits = 64
            while True:
                k.tick("curved_element")
                px, py = param(x, bits), param(y, bits)
                if px.hi < py.lo:
                    return True
                if py.hi < px.lo:
                    return False
                bits += 32
                if bits > k.limits.max_isolation_bits:
                    raise refusal(RefusalCategory.UNDECIDED_PREDICATE, "two cut points of a face are not ordered",
                                  "cut_order")

        ordered: list = []
        for node in nodes:
            i = len(ordered)
            while i > 0 and less(node, ordered[i - 1]):
                i -= 1
            ordered.insert(i, node)
        return ordered

    def planar_face(self, face, cap_links: list) -> None:
        k, ar = self.k, self.ar
        if len(face.loops) != 1:
            raise _csd(f"planar face {face.id} has holes", "planar_holes")
        loop = face.loops[0]
        origin, u_vec, v_vec = face.support
        apex = origin
        count = len(loop.edges)
        pieces: list = []
        for i, ((edge, forward), start) in enumerate(zip(loop.edges, loop.vertices)):
            k.tick("curved_element")
            end = loop.vertices[(i + 1) % count]
            if edge in self.arcs:
                arc = self.arcs[edge]
                ascending = forward == arc.trim.curve_increasing
                for a, b, inside in self.arc_pieces(arc, ascending):
                    pieces.append(("arc", arc, a, b, inside, ascending))
            else:
                for a, b, inside in self.line_pieces(edge, start, end):
                    pieces.append(("line", None, a, b, inside, True))
        exits, enters = [], []
        for i, piece in enumerate(pieces):
            kind, arc, a, b, inside, ascending = piece
            if inside:
                if kind == "line":
                    self.segment_terms(apex, a, b)
                else:
                    lo_node, hi_node = (a, b) if ascending else (b, a)
                    self.fan_arc_terms(arc.curve, apex, 1 if ascending else -1,
                                       self.node_u(arc, lo_node), self.node_u(arc, hi_node))
            nxt = pieces[(i + 1) % len(pieces)]
            if b[0] == "c":
                if nxt[2] != b:
                    raise _csd("inconsistent loop pieces", "inconsistent_section")
                (exits if inside else enters).append(b)
        if not exits and not enters:
            return
        if len(exits) != len(enters):
            raise _csd("unequal exits and enters on a planar face", "inconsistent_section")
        m_f = ar.cross(u_vec, v_vec)
        line_dir = ar.cross(self.n, m_f)
        if not any(line_dir):
            raise _csd("the plane is parallel to a cut planar face", "inconsistent_section")
        ordered = self.order_nodes(exits + enters, line_dir)
        exit_set = set(exits)
        for j in range(0, len(ordered), 2):
            p, q = ordered[j], ordered[j + 1]
            if (p in exit_set) == (q in exit_set):
                raise _csd("a chord of a planar face does not join an exit to an enter", "inconsistent_section")
            ex, en = (p, q) if p in exit_set else (q, p)
            self.segment_terms(apex, ex, en)               # the clipped face closes exit -> enter
            cap_links.append(("seg", en[1], ex[1]))        # the cap runs enter -> exit

    def curved_face(self, fid: int, cap_links: list) -> None:
        k, ar = self.k, self.ar
        face = self.curved.faces[fid]
        support = face.support
        sigma = self.curved.sigma[fid]
        self.face_polys[fid] = face_polys(k, support)
        cuts = self._cuts_of_face(fid)
        for trim in face.trims:
            k.tick("curved_element")
            arc = self.arcs[trim.edge]
            ascending = trim.loop_increasing
            for a, b, inside in self.arc_pieces(arc, ascending):
                if not inside:
                    continue
                lo_node, hi_node = (a, b) if ascending else (b, a)
                self.curved_piece_terms(fid, trim.plane.N, trim.plane.D, 1 if ascending else -1,
                                        self.node_u(arc, lo_node), self.node_u(arc, hi_node))
        if not cuts:
            return
        # the cut graph v = r_w(u) = -alpha / beta and its portions inside the region
        alpha, beta = plane_alpha_beta(k, support, self.n, self.d)
        N, D = psub(k, (), alpha), beta
        g = pgcd(k, N, D) if N else ()
        if g and len(g) > 1:
            N, D = pexact_div(k, N, g), pexact_div(k, D, g)
        if not N:
            D = (ONE,)
        curve_w = make_curve(k, support, N, D)
        ordered: list = []
        for key in cuts:
            i = len(ordered)
            while i > 0 and compare(k, self.cut_roots[key], self.cut_roots[ordered[i - 1]]) < 0:
                k.tick("curved_element")
                i -= 1
            ordered.insert(i, key)
        for x, y in zip(ordered, ordered[1:]):
            if compare(k, self.cut_roots[x], self.cut_roots[y]) == 0:
                raise _csd("two cut points of a face share a parameter", "inconsistent_section")
        beta_sign = self.beta_sign[fid]
        direction = -beta_sign * sigma               # the clipped face's boundary on the graph (+1: increasing u)
        cap_direction = -direction
        inside_flags = []
        for x, y in zip(ordered, ordered[1:]):
            k.tick("curved_element")
            inside_flags.append(self._graph_inside(face, N, D, self.cut_roots[x], self.cut_roots[y]))
        if len(ordered) % 2 or any(inside_flags[i] != (i % 2 == 0) for i in range(len(inside_flags))):
            raise _csd("the cut graph does not alternate inside and outside the face", "inconsistent_section")
        for i in range(0, len(ordered) - 1, 2):
            x, y = ordered[i], ordered[i + 1]
            a, b = self.cut_roots[x], self.cut_roots[y]
            self.curved_piece_terms(fid, N, D, direction, a, b)
            self.fan_arc_terms(curve_w, self.q0, cap_direction, a, b)
            self.waterplane_arc_terms(curve_w, cap_direction, a, b)
            cap_links.append(("arc", x, y) if cap_direction > 0 else ("arc", y, x))

    def _graph_inside(self, face, N: tuple, D: tuple, a, b) -> bool:
        k, ar = self.k, self.ar
        while compare(k, a, b) >= 0 or upper_bound(a) >= lower_bound(b):
            k.tick("curved_element")
            for x in (a, b):
                if isinstance(x, AlgebraicNumber) and not x.is_rational:
                    x.refine(k)
        lo, hi = upper_bound(a), lower_bound(b)
        for t in (Fraction(1, 2), Fraction(1, 3), Fraction(2, 3), Fraction(1, 5), Fraction(4, 5), Fraction(1, 7)):
            u_q = ar.add(lo, ar.mul(ar.sub(hi, lo), t))
            d_q = peval(k, D, u_q)
            if d_q == 0:
                continue
            v_q = ar.div(peval(k, N, u_q), d_q)
            try:
                return region_contains(k, face, u_q, v_q)
            except CertifiedHydrostaticsRefusal as error:
                if error.category is not RefusalCategory.UNDECIDED_PREDICATE:
                    raise
        raise refusal(RefusalCategory.UNDECIDED_PREDICATE, "no test abscissa between two cut points", "test_point")

    # ---------------------------------------------------------------- build and evaluate

    def build(self, occupancy: str) -> dict:
        """Builds the terms of all quantities. Returns the topology information of the cap."""
        k = self.k
        cap_links: list = []
        for face in self.snapshot.faces:
            k.tick("curved_element")
            if face.id in self.curved.faces:
                self.curved_face(face.id, cap_links)
            else:
                self.planar_face(face, cap_links)
        if occupancy == "PARTIAL":
            return self._cap(cap_links)
        return {}

    def _cap(self, cap_links: list) -> dict:
        k = self.k
        succ: dict = {}
        for kind, src, dst in cap_links:
            k.tick("curved_element")
            if src in succ:
                raise _csd("a cut point starts two cap pieces", "inconsistent_section")
            succ[src] = dst
            if kind == "seg":                         # a chord of a planar face, in the cap's direction
                a, b = ("c", src), ("c", dst)
                self.segment_terms(self.q0, a, b)
                self.triangle_terms(a, b)
        keys = set(self.cut_roots) | {(e, 0) for e in self.line_cuts}
        if set(succ) != keys or sorted(succ.values()) != sorted(succ):
            raise _csd("the cut-point graph is not a union of cycles", "inconsistent_section")
        seen, cycles = set(), []
        for start in sorted(keys):
            if start in seen:
                continue
            cycle, key = [], start
            while key not in seen:
                k.tick("curved_element")
                seen.add(key)
                cycle.append(key)
                key = succ[key]
            if key != start:
                raise _csd("the cut-point graph is not a union of cycles", "inconsistent_section")
            cycles.append(tuple(cycle))
        if len(cycles) != 1:
            raise _csd("the section has more than one loop (outside the supported class)", "multiple_loops")
        return {"cycles": cycles}

    def topology(self, cap: dict) -> Topology:
        cycle = cap["cycles"][0]
        identity = tuple(sorted(edge for edge, _ordinal in cycle))
        return Topology((TopologyLoop(identity, "outer"),), ((identity, ()),), tuple(sorted(cycle)))

    # ---------------------------------------------------------------- evaluation

    def evaluate(self, name: str, tau: Fraction) -> Enclosure:
        k = self.k
        terms = self.terms[name]
        ints = [t for t in terms if t[0] == "i"]
        pts = [t for t in terms if t[0] == "p"]
        total = Enclosure(ZERO, ZERO)
        share = tau / 2 if (ints and pts) else tau
        for _kind, coef, num, dens, a, b in ints:
            k.tick("curved_element")
            tau_i = share / (len(ints) * abs(coef))
            result = enclose_rational_integral(k, num, dens, a, b, tau_i)
            self.exec_info["integrals"] += 1
            self.exec_info["taylor_pieces"] += result.record.taylor_pieces
            total = total.add(k, result.enclosure.scale(k, coef))
        if pts:
            bits = max(48, (tau.denominator.bit_length() - tau.numerator.bit_length()) + 24)
            while True:
                k.tick("curved_element")
                try:
                    acc = Enclosure(ZERO, ZERO)
                    ia = IA(k, bits)
                    for _kind, fn in pts:
                        acc = ia.add(acc, fn(bits))
                except _NeedBits:
                    acc = None
                if acc is not None and acc.width <= share:
                    break
                bits += max(16, bits // 2)
                if bits > k.limits.max_isolation_bits:
                    raise refusal(RefusalCategory.ACCURACY_UNACHIEVABLE,
                                  f"the point terms of {name} cannot reach the width {float(share):.3e}")
            self.exec_info["point_bits"] = max(self.exec_info["point_bits"], bits)
            total = total.add(k, acc)
        if total.width > tau:                       # the allocation invariant (integral shares + point share <= tau)
            raise refusal(RefusalCategory.ACCURACY_UNACHIEVABLE,
                          f"{name}: the enclosure is {float(total.width):.3e} wide, the target {float(tau):.3e}")
        return total


# ------------------------------------------------------------------ entry point

def _pairs(enclosure: Enclosure) -> list:
    return [[enclosure.lo.numerator, enclosure.lo.denominator], [enclosure.hi.numerator, enclosure.hi.denominator]]


def _rational_sqrt(x: Fraction):
    """``sqrt(x)`` when it is rational, else ``None``."""
    n, d = math.isqrt(x.numerator), math.isqrt(x.denominator)
    return Fraction(n, d) if n * n == x.numerator and d * d == x.denominator else None


def _default_width(policy) -> Fraction:
    return Fraction(1, 10 ** policy.default_enclosure_digits)


def evaluate_curved(prepared, query, meter: Meter, before: dict) -> CertifiedHydrostaticsResult:
    ar = Ar(meter)
    for number in (*query.normal, query.offset, *query.waterplane_origin, *query.waterplane_axes[0],
                   *query.waterplane_axes[1], *(w for _name, w in query.absolute_accuracy)):
        meter.fit(number)
    b = Builder(prepared, query, meter)
    b.analyse()
    occupancy = b.occupancy()
    b.check_curved_faces(occupancy == "PARTIAL")
    policy = prepared.policy
    default = _default_width(policy)
    forms = query.forms
    results: dict = {}
    topology = None
    if occupancy == "EMPTY":
        from ._forms import build_quantity
        zero = Fraction(0)
        volume = build_quantity("V", zero, None, forms, query.accuracy("V"), meter, always_exact=True)
        moments = tuple(build_quantity(name, zero, None, forms, query.accuracy(name), meter, always_exact=True)
                        for name in ("Jx", "Jy", "Jz"))
        volume = _with_enclosure(volume, (zero, zero))
        moments = tuple(_with_enclosure(m, (zero, zero)) for m in moments)
        sections = (SectionResult("plane", "EMPTY", "ABSENT", None, None),)
        return _finish(prepared, query, meter, before, "EMPTY", volume, moments, sections, b)
    cap = b.build(occupancy)
    quantities = {}
    for name in VOLUME_TERMS:
        a = query.accuracy(name) or default
        a_eff = a * 3 / 4
        core = b.evaluate(name, a_eff / 3)
        b.exec_info["core"][name] = _pairs(core)
        width = core.width
        claim = (core.lo - width, core.hi + width)
        exact = (core.lo, None) if width == 0 else None
        quantities[name] = build_enclosed_quantity(name, claim, exact, forms, query.accuracy(name), a, meter)
    volume = quantities["V"]
    moments = tuple(quantities[n] for n in ("Jx", "Jy", "Jz"))
    if occupancy == "FULL":
        sections = (SectionResult("plane", "FULL", "ABSENT", None, None),)
    else:
        n2 = query.normal_squared
        waterplane = []
        for name in WATERPLANE_NAMES:
            a = query.accuracy(name) or default
            captured: dict = {}

            def numerator(tau, name=name, captured=captured):
                captured["enc"] = b.evaluate(name, tau)
                return captured["enc"]

            nq = divide_by_norm(b.k, n2, numerator, a * 3 / 4)
            b.exec_info["core"][name] = _pairs(nq.core)
            claim = (nq.claim.lo, nq.claim.hi)
            exact = None
            if captured["enc"].width == 0:                     # an exact numerator: the planar exact form
                exact = (captured["enc"].lo, n2)
                root = _rational_sqrt(n2)
                if root is not None:
                    claim = (captured["enc"].lo / root,) * 2
            waterplane.append(build_enclosed_quantity(name, claim, exact, forms, query.accuracy(name), a, meter))
        topology = b.topology(cap)
        sections = (SectionResult("plane", "PARTIAL", "PRESENT", tuple(waterplane), topology),)
    return _finish(prepared, query, meter, before, occupancy, volume, moments, sections, b)


def _with_enclosure(q, enclosure):
    from dataclasses import replace
    return replace(q, enclosure=enclosure)


def _finish(prepared, query, meter, before, occupancy, volume, moments, sections, b) -> CertifiedHydrostaticsResult:
    from dataclasses import replace
    placeholder = {"operations": 0, "units": {}}
    execution = dict(b.exec_info) if b is not None else None
    result = CertifiedHydrostaticsResult(query, occupancy, True, Contacts((), (), ()), volume, moments, sections,
                                         placeholder, execution=execution)
    limits = dict(max_digits=prepared.policy.max_integer_digits, max_bytes=prepared.policy.max_proof_bytes)
    document = proof_document(prepared, query, result.claim(meter, with_work=False), meter)
    taken: dict = {}

    def work_field():
        from ._engine import _delta
        taken["work"] = _delta(meter.budget.counts(), before)
        return taken["work"]
    proof = write_canonical(document, meter, late=(("result",), "work", work_field), **limits)
    return replace(result, work=taken["work"], proof=proof)
