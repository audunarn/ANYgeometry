"""CH3a2: admission checks 1-6 of a shell with curved faces (design CH3 section 2.3, narrowed to the B1 class).

Faces are planar faces (straight edges, or rational-arc edges shared with a curved face) and curved faces of the
ruled rational class. Vertices are ``Pt``: rational, or the algebraic corner of a curved face. The order of the
checks is that of the planar design: 1 supports (vertices and arcs lie in the face planes), 2 loops (the curved
faces' loops were decided when the snapshot was collected, ``_ch3_collect``; planar loops here), 3 holes
(refused), 4 shell combinatorics (reused), 5 embedding, 6 orientation (the signed volume as an enclosure).

**Planar loops**: pieces are segments between exact points and arcs ``E/M``. Segment/segment by exact orientation
signs; segment/arc by substituting the arc into a rational line (axis-aligned or between rational ends); arc/arc
only by exact box separation (adjacent arcs, arcs with overlapping boxes: typed refusal).

**Embedding** (a narrowing of design 2.3 check 5 that needs no point-in-region test): if a face lies in a closed
half-space of another face's plane (all its loop pieces do), its contact with that plane lies on its own loop
(vertices, edges in the plane); two such faces meet exactly where those loop elements meet, and every such
meeting must lie inside a declared shared cell (vertex or edge). For a curved face and a planar face the curved
face must lie on one side of the plane, no ruling may lie in the plane (the extreme of ``alpha + v beta`` over a
fibre is on the loop), and contact elements not shared must be box-disjoint from the planar face. Faces that
straddle each other's planes, coplanar faces, arcs touching a plane inside, curved/curved pairs with overlapping
boxes refuse ``EMBEDDING_UNDECIDED``."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from ._admission import _Face, _check_shell
from ._ch3_admission import plane_alpha_beta
from ._ch3_algebraic import AlgebraicNumber, Enclosure, compare, isolate_roots, lower_bound, sign_at, upper_bound
from ._ch3_exact import (EncAlg, Pt, _bounds, arc_box, box_union, boxes_disjoint, point_box, points_distinct,
                         poly_sign_on_open, sign, simplest_rational)
from ._ch3_loops import loop_vertices
from ._ch3_poly import (Kernel, ONE, ZERO, padd, pderiv, pgcd, pmul, ppow, pscale, psub, refusal)
from ._errors import RefusalCategory
from ._snapshot import ExactSnapshot
from ._work import Meter

__all__ = ["admit_curved", "curved_box"]

IL = RefusalCategory.INVALID_LOOP
EU = RefusalCategory.EMBEDDING_UNDECIDED
UP = RefusalCategory.UNDECIDED_PREDICATE


# ------------------------------------------------------------------ views

@dataclass(eq=False)
class _Arc:
    edge: int
    E: tuple
    M: tuple
    lo: AlgebraicNumber
    hi: AlgebraicNumber
    start_vertex: int
    end_vertex: int
    curve_increasing: bool

    def u_at(self, vertex: int):
        if vertex == self.start_vertex:
            return self.lo if self.curve_increasing else self.hi
        return self.hi if self.curve_increasing else self.lo


@dataclass(eq=False)
class _Piece:
    kind: str                # "seg" or "arc"
    edge: int
    a: Pt                    # loop-order start
    b: Pt
    arc: _Arc | None = None


@dataclass(eq=False)
class _View:
    id: int
    curved: bool
    vids: list
    pieces: list
    vset: frozenset
    eset: frozenset
    n: tuple | None = None
    d: Fraction | None = None
    origin: tuple | None = None
    axes: tuple | None = None
    cface: object = None
    box: tuple | None = None

    def pts(self, pts):
        return [pts[v] for v in self.vids]


def _arc_of(entry) -> _Arc:
    _fid, trim = entry
    return _Arc(trim.edge, trim.plane.E, trim.plane.M, trim.u_lo, trim.u_hi, trim.start_vertex, trim.end_vertex,
                trim.curve_increasing)


def _points(k: Kernel, snapshot: ExactSnapshot) -> dict:
    owner = {}
    for fid, cf in snapshot.curved.faces.items():
        for vertex, corner in cf.corners.items():
            k.tick("admission_element")
            trim = next(t for t in cf.trims if vertex in (t.start_vertex, t.end_vertex))
            owner[vertex] = Pt.from_corner(k, vertex, corner.number, trim.plane.E, trim.plane.M)
    pts = {}
    for vertex in sorted(snapshot.positions):
        k.tick("admission_element")
        pts[vertex] = owner.get(vertex) or Pt(vertex, rat=tuple(snapshot.positions[vertex]))
    return pts


def _build_views(k: Kernel, snapshot: ExactSnapshot, pts: dict) -> list:
    cd = snapshot.curved
    views = []
    for face in snapshot.faces:
        loop = face.loops[0]
        count = len(loop.edges)
        if count < 3:
            raise refusal(IL, f"a loop of face {face.id} has fewer than 3 edges")
        for i, (edge, forward) in enumerate(loop.edges):
            k.tick("loop_pair")
            start, end = snapshot.edges[edge]
            if not forward:
                start, end = end, start
            if loop.vertices[i] != start or loop.vertices[(i + 1) % count] != end:
                raise refusal(IL, f"a loop of face {face.id} is not continuous")
        vset = frozenset(v for lp in face.loops for v in lp.vertices)
        eset = frozenset(e for lp in face.loops for e, _f in lp.edges)
        if face.id in cd.faces:
            cf = cd.faces[face.id]
            pieces = [_Piece("arc", t.edge, pts[loop.vertices[i]], pts[loop.vertices[(i + 1) % count]], _arc_of(
                (face.id, t))) for i, t in enumerate(cf.trims)]
            views.append(_View(face.id, True, list(loop.vertices), pieces, vset, eset, cface=cf))
            continue
        pieces = []
        for i, (edge, _forward) in enumerate(loop.edges):
            k.tick("loop_pair")
            a, b = pts[loop.vertices[i]], pts[loop.vertices[(i + 1) % count]]
            if edge in cd.arcs:
                pieces.append(_Piece("arc", edge, a, b, _arc_of(cd.arcs[edge])))
            else:
                pieces.append(_Piece("seg", edge, a, b))
        origin, u, v = face.support
        ar = k.ar
        n = ar.cross(u, v)
        if not any(n):
            raise refusal(RefusalCategory.NOT_EXACTLY_PLANAR, f"the support of face {face.id} is degenerate")
        magnitude = [abs(c) for c in n]
        drop = magnitude.index(max(magnitude))
        views.append(_View(face.id, False, list(loop.vertices), pieces, vset, eset, n=n, d=ar.dot(n, origin),
                           origin=origin, axes=tuple(a for a in range(3) if a != drop)))
    return views


# ------------------------------------------------------------------ small exact helpers

def _lin(n, d, p):
    return lambda alg: alg.sub(alg.add(alg.add(alg.mul(alg.const(n[0]), alg.coord(p, 0)),
                                                 alg.mul(alg.const(n[1]), alg.coord(p, 1))),
                                       alg.mul(alg.const(n[2]), alg.coord(p, 2))), alg.const(d))


def _side(k: Kernel, p: Pt, n, d) -> int:
    return sign(k, (p,), _lin(n, d, p))


def _arc_g(k: Kernel, arc: _Arc, n, d) -> tuple:
    """``n . E - d M``: the plane ``n . x = d`` on the arc's curve."""
    g = pscale(k, arc.M, -d)
    for c in range(3):
        g = padd(k, g, pscale(k, arc.E[c], n[c]))
    return g


def _orient(k: Kernel, a: Pt, b: Pt, c: Pt, c0: int, c1: int) -> int:
    return sign(k, (a, b, c), lambda alg: alg.sub(
        alg.mul(alg.sub(alg.coord(b, c0), alg.coord(a, c0)), alg.sub(alg.coord(c, c1), alg.coord(a, c1))),
        alg.mul(alg.sub(alg.coord(b, c1), alg.coord(a, c1)), alg.sub(alg.coord(c, c0), alg.coord(a, c0)))))


def _cmp(k: Kernel, p: Pt, q: Pt, c: int) -> int:
    if p.vertex == q.vertex:
        return 0
    return sign(k, (p, q), lambda alg: alg.sub(alg.coord(p, c), alg.coord(q, c)))


def _on_segment(k: Kernel, p: Pt, a: Pt, b: Pt, c0: int, c1: int) -> bool:
    if p.vertex in (a.vertex, b.vertex):
        return True
    if _orient(k, a, b, p, c0, c1) != 0:
        return False
    return all(_cmp(k, p, a, c) * _cmp(k, p, b, c) <= 0 for c in (c0, c1))


def _rational_line(k: Kernel, a: Pt, b: Pt, c0: int, c1: int):
    """``(A, B, C)`` with the line ``A x_c0 + B x_c1 + C = 0`` through ``a``, ``b`` when it has rational
    coefficients (both ends rational, or one coordinate equal to one rational at both ends), else ``None``."""
    if a.rat is not None and b.rat is not None:
        A = k.ar.sub(b.rat[c1], a.rat[c1])
        B = -k.ar.sub(b.rat[c0], a.rat[c0])
        return A, B, -(k.ar.add(k.ar.mul(A, a.rat[c0]), k.ar.mul(B, a.rat[c1])))
    alg = EncAlg(k)
    for axis, line in ((c0, lambda r: (ONE, ZERO, -r)), (c1, lambda r: (ZERO, ONE, -r))):
        k.tick("admission_element")
        ea, eb = alg.coord(a, axis), alg.coord(b, axis)
        lo, hi = max(ea.lo, eb.lo), min(ea.hi, eb.hi)
        if lo > hi:
            continue
        r = simplest_rational(k, lo, hi)
        if all(sign(k, (p,), lambda al, p=p: al.sub(al.coord(p, axis), al.const(r))) == 0 for p in (a, b)):
            return line(r)
    return None


def _piece_box(k: Kernel, piece: _Piece) -> tuple:
    if piece.kind == "seg":
        return box_union([point_box(k, piece.a), point_box(k, piece.b)])
    arc = piece.arc
    return arc_box(k, arc.E, arc.M, lower_bound(arc.lo), upper_bound(arc.hi))


# ------------------------------------------------------------------ check 1: supports

def _check_supports(k: Kernel, views: list) -> None:
    for v in views:
        if v.curved:
            continue
        seen = set()
        for piece in v.pieces:
            for p in (piece.a, piece.b):
                k.tick("plane_test")
                if p.vertex not in seen:
                    seen.add(p.vertex)
                    if _side(k, p, v.n, v.d) != 0:
                        raise refusal(RefusalCategory.NOT_EXACTLY_PLANAR,
                                      f"vertex {p.vertex} of face {v.id} is not exactly in its support plane")
            if piece.kind == "arc" and _arc_g(k, piece.arc, v.n, v.d):
                raise refusal(RefusalCategory.NOT_EXACTLY_PLANAR,
                              f"arc edge {piece.edge} of face {v.id} is not in the plane of the face")
            if piece.kind == "seg" and not points_distinct(k, piece.a, piece.b):
                raise refusal(IL, f"edge {piece.edge} of face {v.id} has zero length")


# ------------------------------------------------------------------ check 2: planar loops

def _seg_seg(k, i, j, first, second, adjacent, v) -> None:
    c0, c1 = v.axes
    if adjacent:
        a, b, c = first.a, first.b, second.b
        if _orient(k, a, b, c, c0, c1) == 0:
            dot = sign(k, (a, b, c), lambda alg: alg.add(
                alg.mul(alg.sub(alg.coord(a, c0), alg.coord(b, c0)), alg.sub(alg.coord(c, c0), alg.coord(b, c0))),
                alg.mul(alg.sub(alg.coord(a, c1), alg.coord(b, c1)), alg.sub(alg.coord(c, c1), alg.coord(b, c1)))))
            if dot > 0:
                raise refusal(IL, f"a loop of face {v.id} doubles back on itself (spike)", "spike")
        return
    p1, p2, q1, q2 = first.a, first.b, second.a, second.b
    d1, d2 = _orient(k, p1, p2, q1, c0, c1), _orient(k, p1, p2, q2, c0, c1)
    d3, d4 = _orient(k, q1, q2, p1, c0, c1), _orient(k, q1, q2, p2, c0, c1)
    if d1 * d2 > 0 or d3 * d4 > 0:
        return
    if d1 * d2 < 0 and d3 * d4 < 0:
        raise refusal(IL, f"a loop of face {v.id} crosses itself", "crossing")
    if (_on_segment(k, q1, p1, p2, c0, c1) or _on_segment(k, q2, p1, p2, c0, c1)
            or _on_segment(k, p1, q1, q2, c0, c1) or _on_segment(k, p2, q1, q2, c0, c1)):
        raise refusal(IL, f"a loop of face {v.id} touches itself", "touch")


def _seg_arc(k, seg: _Piece, arc_piece: _Piece, shared, v) -> None:
    c0, c1 = v.axes
    line = _rational_line(k, seg.a, seg.b, c0, c1)
    if line is None:
        raise refusal(UP, f"the segment line of face {v.id} is not rational", "irrational_segment_line")
    A, B, C = line
    arc = arc_piece.arc
    g = padd(k, padd(k, pscale(k, arc.E[c0], A), pscale(k, arc.E[c1], B)), pscale(k, arc.M, C))
    if not g:
        raise refusal(UP, f"the arc {arc.edge} of face {v.id} lies on a segment line", "arc_on_segment_line")
    for r in isolate_roots(k, g, lower_bound(arc.lo) - 1, upper_bound(arc.hi) + 1):
        k.tick("root_refine")
        if compare(k, r, arc.lo) < 0 or compare(k, r, arc.hi) > 0:
            continue
        if shared is not None and compare(k, r, arc.u_at(shared)) == 0:
            continue
        raise refusal(UP, f"arc {arc.edge} of face {v.id} meets the line of a segment", "arc_line_contact")


def _check_planar_loop(k: Kernel, v: _View) -> None:
    pieces = v.pieces
    n = len(pieces)
    boxes = [_piece_box(k, p) for p in pieces]
    for i in range(n):
        for j in range(i + 1, n):
            k.tick("loop_pair")
            adjacent = j == i + 1 or (i == 0 and j == n - 1)
            if adjacent and j != i + 1:
                first, second = pieces[j], pieces[i]          # the wrap-around pair: j precedes i in loop order
            else:
                first, second = pieces[i], pieces[j]
            if not adjacent and boxes_disjoint(boxes[i], boxes[j]):
                continue
            kinds = (first.kind, second.kind)
            if kinds == ("seg", "seg"):
                _seg_seg(k, i, j, first, second, adjacent, v)
            elif kinds == ("arc", "arc"):
                raise refusal(UP if adjacent else EU, f"two arcs of face {v.id} cannot be separated",
                              "adjacent_arcs" if adjacent else "arc_arc_boxes_overlap")
            else:
                seg, arc = (first, second) if first.kind == "seg" else (second, first)
                shared = None
                if adjacent:
                    shared = first.b.vertex
                _seg_arc(k, seg, arc, shared, v)


def _check_loops(k: Kernel, views: list) -> None:
    for v in views:
        if not v.curved:
            _check_planar_loop(k, v)


def _check_holes(snapshot: ExactSnapshot) -> None:
    for face in snapshot.faces:
        if len(face.loops) > 1:
            raise refusal(RefusalCategory.INVALID_HOLES,
                          f"face {face.id} has holes (not supported in a model with curved faces)", "holes")


# ------------------------------------------------------------------ boxes

def curved_box(k: Kernel, cf) -> tuple:
    """A box containing the curved face: the patch over the loop's ``u`` and ``v`` extents (design 2.3 (5))."""
    sup = cf.support
    k1, k2 = sup.span_v
    ulo = lower_bound(cf.trims[0].u_lo)
    uhi = upper_bound(cf.trims[0].u_hi)
    vlo = vhi = None
    for t in cf.trims:
        k.tick("admission_element")
        ulo, uhi = min(ulo, lower_bound(t.u_lo)), max(uhi, upper_bound(t.u_hi))
        e = arc_box(k, (t.plane.N,) * 3, t.plane.D, lower_bound(t.u_lo), upper_bound(t.u_hi), 4)[0]
        vlo = e.lo if vlo is None or e.lo < vlo else vlo
        vhi = e.hi if vhi is None or e.hi > vhi else vhi
    width = k.ar.sub(k2, k1)
    boxes = []
    for s in (max(ZERO, k.ar.div(k.ar.sub(vlo, k1), width)), min(ONE, k.ar.div(k.ar.sub(vhi, k1), width))):
        nums = tuple(padd(k, sup.A[c], pscale(k, psub(k, sup.B[c], sup.A[c]), s)) for c in range(3))
        boxes.append(arc_box(k, nums, sup.W, ulo, uhi, 8))
    return box_union(boxes)


def _view_box(k: Kernel, v: _View) -> tuple:
    if v.curved:
        return curved_box(k, v.cface)
    return box_union([_piece_box(k, p) for p in v.pieces])


# ------------------------------------------------------------------ check 5: embedding

@dataclass(eq=False)
class _Profile:
    zero_vertices: list
    zero_segments: list
    zero_arcs: list
    signs: set


def _profile(k: Kernel, v: _View, n, d, allow_arc_in_plane: bool) -> "_Profile | None":
    """The side structure of the face's loop against the plane ``n . x = d``; ``None`` if it straddles."""
    signs: set = set()
    value = {}
    for piece in v.pieces:
        k.tick("plane_test")
        for p in (piece.a, piece.b):
            if p.vertex not in value:
                value[p.vertex] = _side(k, p, n, d)
    zero_arcs = []
    for piece in v.pieces:
        if piece.kind != "arc":
            continue
        arc = piece.arc
        g = _arc_g(k, arc, n, d)
        if not g:
            if not allow_arc_in_plane:
                raise refusal(EU, f"arc {arc.edge} lies in the plane of the other face", "arc_in_plane")
            zero_arcs.append(arc)
            continue
        s, root = poly_sign_on_open(k, g, arc.lo, arc.hi)
        if root:
            raise refusal(EU, f"arc {arc.edge} meets the plane of the other face inside", "arc_meets_plane")
        m_sign, m_root = poly_sign_on_open(k, arc.M, arc.lo, arc.hi)
        if m_root or not m_sign:
            raise refusal(RefusalCategory.UNSUPPORTED_TRIM, "an arc denominator is not one-signed", "denominator_not_certified")
        signs.add(s * m_sign)
    for s in value.values():
        if s:
            signs.add(s)
    if len(signs) > 1:
        return None
    zero_vertices = [piece_pt for piece_pt in
                     {p.vertex: p for piece in v.pieces for p in (piece.a, piece.b)}.values()
                     if value[piece_pt.vertex] == 0]
    zero_segments = [(p.a, p.b) for p in v.pieces if p.kind == "seg" and value[p.a.vertex] == 0
                     and value[p.b.vertex] == 0]
    return _Profile(zero_vertices, zero_segments, zero_arcs, signs)


def _t_compare(k: Kernel, direction, p: Pt, q: Pt) -> int:
    if p.vertex == q.vertex:
        return 0

    def t(alg, pt):
        return alg.add(alg.add(alg.mul(alg.const(direction[0]), alg.coord(pt, 0)),
                               alg.mul(alg.const(direction[1]), alg.coord(pt, 1))),
                       alg.mul(alg.const(direction[2]), alg.coord(pt, 2)))

    return sign(k, (p, q), lambda alg: alg.sub(t(alg, p), t(alg, q)))


def _planar_planar(k: Kernel, a: _View, b: _View) -> None:
    pa = _profile(k, a, b.n, b.d, False)
    pb = _profile(k, b, a.n, a.d, False)
    if pa is None or pb is None:
        raise refusal(EU, f"faces {a.id} and {b.id} straddle each other's planes", "straddle")
    if not pa.signs or not pb.signs:
        raise refusal(EU, f"faces {a.id} and {b.id} are coplanar", "coplanar")
    direction = k.ar.cross(a.n, b.n)

    def element(p, q):
        return (p, q) if _t_compare(k, direction, p, q) <= 0 else (q, p)

    elements_a = [(p, p) for p in pa.zero_vertices] + [element(p, q) for p, q in pa.zero_segments]
    elements_b = [(p, p) for p in pb.zero_vertices] + [element(p, q) for p, q in pb.zero_segments]
    by_vertex = {}
    for piece in a.pieces + b.pieces:
        for p in (piece.a, piece.b):
            by_vertex[p.vertex] = p
    cells = [(by_vertex[x], by_vertex[x]) for x in a.vset & b.vset]
    for e in a.eset & b.eset:
        piece = next(p for p in a.pieces if p.edge == e)
        if piece.kind != "seg":
            raise refusal(EU, f"faces {a.id} and {b.id} share a curved edge", "shared_arc")
        cells.append(element(piece.a, piece.b))
    for lo_a, hi_a in elements_a:
        for lo_b, hi_b in elements_b:
            k.tick("interval_pair")
            lo = lo_a if _t_compare(k, direction, lo_a, lo_b) >= 0 else lo_b
            hi = hi_a if _t_compare(k, direction, hi_a, hi_b) <= 0 else hi_b
            if _t_compare(k, direction, lo, hi) > 0:
                continue
            if not any(_t_compare(k, direction, c0, lo) <= 0 and _t_compare(k, direction, hi, c1) <= 0
                       for c0, c1 in cells):
                raise refusal(RefusalCategory.SELF_INTERSECTING,
                              f"faces {a.id} and {b.id} meet beyond their shared cells")


def _ruling_check(k: Kernel, c: _View, p: _View) -> None:
    cf = c.cface
    alpha, beta = plane_alpha_beta(k, cf.support, p.n, p.d)
    if not alpha and not beta:
        raise refusal(EU, f"curved face {c.id} lies in the plane of face {p.id}", "curved_in_plane")
    g0 = pgcd(k, alpha, beta)
    if len(g0) < 2:
        return
    ulo = min((t.u_lo for t in cf.trims), key=lambda x: lower_bound(x))
    uhi = max((t.u_hi for t in cf.trims), key=lambda x: upper_bound(x))
    for r in isolate_roots(k, g0, lower_bound(ulo) - 1, upper_bound(uhi) + 1):
        k.tick("root_refine")
        if compare(k, r, ulo) >= 0 and compare(k, r, uhi) <= 0:
            raise refusal(EU, f"a ruling of curved face {c.id} lies in the plane of face {p.id}", "ruling_in_plane")


def _curved_planar(k: Kernel, c: _View, p: _View) -> None:
    prof = _profile(k, c, p.n, p.d, True)
    if prof is None:
        raise refusal(EU, f"curved face {c.id} straddles the plane of face {p.id}", "straddle")
    if not prof.signs:
        raise refusal(EU, f"curved face {c.id} lies in the plane of face {p.id}", "curved_in_plane")
    _ruling_check(k, c, p)
    for corner in prof.zero_vertices:
        k.tick("admission_element")
        if corner.vertex not in p.vset and not boxes_disjoint(point_box(k, corner), p.box):
            raise refusal(EU, f"corner {corner.vertex} of curved face {c.id} touches the plane of face {p.id} "
                          "outside the face's declared vertices", "corner_contact")
    for arc in prof.zero_arcs:
        k.tick("admission_element")
        if arc.edge in p.eset:
            continue
        box = arc_box(k, arc.E, arc.M, lower_bound(arc.lo), upper_bound(arc.hi))
        if not boxes_disjoint(box, p.box):
            raise refusal(EU, f"trim {arc.edge} of curved face {c.id} lies in the plane of face {p.id} "
                          "without being its edge", "trim_contact")


def _check_embedding(k: Kernel, views: list, shared_check=True) -> None:
    for v in views:
        v.box = _view_box(k, v)
    for i, a in enumerate(views):
        for b in views[i + 1:]:
            k.tick("bbox_pair")
            if boxes_disjoint(a.box, b.box):
                if a.vset & b.vset or a.eset & b.eset:
                    raise refusal(RefusalCategory.SELF_INTERSECTING,
                                  f"faces {a.id} and {b.id} share cells but their boxes are disjoint")
                continue
            if a.curved and b.curved:
                raise refusal(EU, f"curved faces {a.id} and {b.id} have overlapping boxes", "curved_curved")
            if a.curved or b.curved:
                c, p = (a, b) if a.curved else (b, a)
                _curved_planar(k, c, p)
            else:
                _planar_planar(k, a, b)


# ------------------------------------------------------------------ check 6: signed volume

def _pcross(k: Kernel, a: tuple, b: tuple) -> list:
    return [psub(k, pmul(k, a[(c + 1) % 3], b[(c + 2) % 3]), pmul(k, a[(c + 2) % 3], b[(c + 1) % 3]))
            for c in range(3)]


def _pdot(k: Kernel, a, b) -> tuple:
    out = ()
    for c in range(3):
        out = padd(k, out, pmul(k, a[c], b[c]))
    return out


def _curved_integrand(k: Kernel, cf, trim) -> tuple:
    """``(numerator, [(W, 3), (D, 2)])`` of ``F_V(u, N/D)`` for a piece (design 4.1)."""
    sup = cf.support
    k1, k2 = sup.span_v
    A = sup.A[:3]
    delta = tuple(psub(k, sup.B[c], sup.A[c]) for c in range(3))
    dA = tuple(pderiv(k, A[c]) for c in range(3))
    dDelta = tuple(pderiv(k, delta[c]) for c in range(3))
    c0 = _pdot(k, A, _pcross(k, dA, delta))
    c1 = _pdot(k, A, _pcross(k, dDelta, delta))
    N, D = trim.plane.N, trim.plane.D
    sigma = pscale(k, psub(k, N, pscale(k, D, k1)), k.ar.div(ONE, k.ar.sub(k2, k1)))
    num = padd(k, pmul(k, pmul(k, c0, sigma), D), pscale(k, pmul(k, c1, pmul(k, sigma, sigma)), Fraction(1, 2)))
    dens = [(sup.W, 3)] + ([(D, 2)] if len(D) > 1 else [])
    if len(D) == 1:
        num = pscale(k, num, k.ar.div(ONE, k.ar.mul(D[0], D[0])))
    return num, dens


def _den_poly(k: Kernel, dens: list) -> tuple:
    out = (ONE,)
    for f, e in dens:
        out = pmul(k, out, ppow(k, f, e))
    return out


def _dyadic(x: Fraction, up: bool) -> Fraction:
    """``x`` rounded outward to a dyadic with a 64-bit mantissa (``up``: not below ``x``; else not above)."""
    if x == 0:
        return ZERO
    s = 64 - (x.numerator.bit_length() - x.denominator.bit_length())
    scaled = x * (1 << s) if s >= 0 else x / (1 << -s)
    whole = -((-scaled.numerator) // scaled.denominator) if up else scaled.numerator // scaled.denominator
    return Fraction(whole, 1 << s) if s >= 0 else Fraction(whole * (1 << -s))


def _riemann(k: Kernel, num: tuple, den: tuple, a, b, pieces: int) -> Enclosure:
    """A rigorous enclosure of ``int_a^b num / den du`` (``den`` nonvanishing on the closed interval, ends
    rational or algebraic) from the Bernstein range of ``num`` and ``den`` on ``pieces`` equal sub-intervals plus
    the two end slivers of the isolating intervals. Used for the sign of the signed volume (check 6) only."""
    direction = 1
    if compare(k, a, b) > 0:
        a, b, direction = b, a, -1
    while upper_bound(a) >= lower_bound(b):
        k.tick("root_refine")
        for x in (a, b):
            if isinstance(x, AlgebraicNumber) and not x.is_rational:
                x.refine(k)

    def value_range(lo, hi) -> Enclosure:
        nl, nh = _bounds(k, num, lo, hi)
        dl, dh = _bounds(k, den, lo, hi)
        if not (dl > 0 or dh < 0):
            raise refusal(UP, "the integrand denominator is not certified one-signed", "volume_denominator")
        qs = [k.ar.div(n_, d_) for n_ in (nl, nh) for d_ in (dl, dh)]
        return Enclosure(_dyadic(min(qs), False), _dyadic(max(qs), True))

    total = Enclosure(ZERO, ZERO)
    grid = 1 << 30                       # all ends on a coarse dyadic grid keep the sub-interval numbers short
    a_out = Fraction((lower_bound(a) * grid).__floor__(), grid)
    a_in = Fraction(-((-upper_bound(a) * grid).__floor__()), grid)
    b_in = Fraction((lower_bound(b) * grid).__floor__(), grid)
    b_out = Fraction(-((-upper_bound(b) * grid).__floor__()), grid)
    if a_in >= b_in:
        raise refusal(UP, "the integration ends are too close for the volume bound", "volume_ends")
    for lo, hi in ((a_out, a_in), (b_in, b_out)):
        k.tick("volume_piece")
        if lo < hi:
            f = value_range(lo, hi)
            m = max(abs(f.lo), abs(f.hi))
            w = _dyadic(k.ar.mul(m, k.ar.sub(hi, lo)), True)
            total = total.add(k, Enclosure(-w, w))
    h = k.ar.div(k.ar.sub(b_in, a_in), pieces)
    for i in range(pieces):
        k.tick("volume_piece")
        lo = k.ar.add(a_in, k.ar.mul(h, i))
        f = value_range(lo, k.ar.add(lo, h))
        scaled = f.scale(k, h)
        total = total.add(k, Enclosure(_dyadic(scaled.lo, False), _dyadic(scaled.hi, True)))
    return total if direction == 1 else total.neg(k)


def _volume(k: Kernel, snapshot: ExactSnapshot, views: list, pts: dict, pieces: int, tau: Fraction) -> Enclosure:
    cd = snapshot.curved
    jobs = sum(len(v.pieces) for v in views)
    share = k.ar.div(tau, jobs + 1)
    for p in pts.values():
        if p.root is not None:
            p.root.refine_below(k, k.ar.div(share, 1 << 20))
    total = Enclosure(ZERO, ZERO)
    alg = EncAlg(k)
    for v in views:
        if v.curved:
            for piece in v.pieces:
                k.tick("volume_piece")
                trim = next(t for t in v.cface.trims if t.edge == piece.edge)
                a, b = (trim.u_lo, trim.u_hi) if trim.loop_increasing else (trim.u_hi, trim.u_lo)
                num, dens = _curved_integrand(k, v.cface, trim)
                if not num:
                    continue
                e = _riemann(k, num, _den_poly(k, dens), a, b, pieces)
                total = total.add(k, e.scale(k, Fraction(-1, 3)))
            continue
        q = v.origin
        for piece in v.pieces:
            k.tick("volume_piece")
            if piece.kind == "seg":
                a, b = piece.a, piece.b
                cr = lambda al: al.sub(al.mul(al.coord(a, 1), al.coord(b, 2)), al.mul(al.coord(a, 2), al.coord(b, 1)))
                cs = lambda al: al.sub(al.mul(al.coord(a, 2), al.coord(b, 0)), al.mul(al.coord(a, 0), al.coord(b, 2)))
                ct = lambda al: al.sub(al.mul(al.coord(a, 0), al.coord(b, 1)), al.mul(al.coord(a, 1), al.coord(b, 0)))
                term = alg.add(alg.add(alg.mul(alg.const(q[0]), cr(alg)), alg.mul(alg.const(q[1]), cs(alg))),
                               alg.mul(alg.const(q[2]), ct(alg)))
                total = total.add(k, term.scale(k, Fraction(1, 6)))
            else:
                arc = piece.arc
                E, M = arc.E, arc.M
                dE = tuple(pderiv(k, e) for e in E)
                cross = _pcross(k, E, dE)
                num = ()
                for c in range(3):
                    num = padd(k, num, pscale(k, cross[c], q[c]))
                if not num:
                    continue
                a = arc.u_at(piece.a.vertex)
                b = arc.u_at(piece.b.vertex)
                e = _riemann(k, pscale(k, num, Fraction(1, 6)), ppow(k, M, 2), a, b, pieces)
                total = total.add(k, e)
    return total


def _check_orientation(k: Kernel, snapshot: ExactSnapshot, views: list, pts: dict) -> None:
    n = 64                       # only the sign of the volume is needed: interval Riemann sums, refined by 4
    while True:
        k.tick("volume_round")
        total = _volume(k, snapshot, views, pts, n, Fraction(1, 1 << 20))
        if total.lo > 0:
            return
        if total.hi < 0:
            raise refusal(RefusalCategory.INVERTED_ORIENTATION, "the shell encloses a negative signed volume")
        if n >= 64 * 4 ** 4:
            raise refusal(UP, "the signed volume is not certified positive", "signed_volume")
        n *= 4


# ------------------------------------------------------------------ driver

def admit_curved(snapshot: ExactSnapshot, meter: Meter, policy) -> None:
    """Checks 1-6 for a snapshot with curved faces (see the module docstring)."""
    k = Kernel(meter, policy.kernel_limits())
    pts = _points(k, snapshot)
    views = _build_views(k, snapshot, pts)
    _check_supports(k, views)
    _check_loops(k, views)
    _check_holes(snapshot)
    shell = []
    for face in snapshot.faces:
        f = _Face()
        f.id = face.id
        f.vids = [lp.vertices for lp in face.loops]
        f.eids = [lp.edges for lp in face.loops]
        shell.append(f)
    _check_shell(shell, meter)
    _check_embedding(k, views)
    _check_orientation(k, snapshot, views, pts)
