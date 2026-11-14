"""Optional exact pre-pass for plane x NURBS sections (monotone-cell certificate).

One operand must be an exactly parallelogram bilinear patch (a plane patch); the other a NURBS surface
with uniform positive weights.  With ``F = n . S(u, v) - c`` on the NURBS surface, every Bezier cell is, with
exact ``Fraction`` arithmetic, one of

* EXCLUDED     - all Bernstein coefficients of ``F`` strictly share one sign (no zero in the cell);
* CERTIFIED    - the section tangent ``t = (F_v, -F_u)`` has all Bernstein coefficients (after exact degree
                 elevation) of ``d . t`` strictly positive or strictly negative for one direction ``d``, so
                 ``grad F != 0`` and the level curve is a graph monotone in ``d`` (no loop, no tangency), AND
                 the four cell edges carry 0 or exactly 2 simple roots of ``F`` (no root at a corner);
* SPLIT        - otherwise (depth cap :data:`MAX_DEPTH` then DECLINED).

Cells meeting those conditions have only arcs with both endpoints at simple edge roots; with exactly two
roots the pairing is forced.  Edge roots are isolated once per (line, span) polynomial by Sturm sequences, on
the dyadic grid, so two cells sharing (part of) an edge name the same root.  Components are stitched only if
every interior root is used by exactly two arcs and every domain-boundary root by one; anything else DECLINES.
DECLINED means: the caller must use the unchanged atlas route.  Nothing here is ever a partial answer.
No module state; no caching beyond one call.  See docs/plane-prepass-soundness.md.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from fractions import Fraction
from math import comb

from ._bernstein import bezier_piece, derivative_grid, subbox

__all__ = ["PrepassStatus", "PrepassDecline", "PrepassRoot", "PrepassArc", "PrepassComponent", "PrepassResult",
           "plane_nurbs_prepass", "MAX_DEPTH", "excludes_cell", "monotone_direction"]

MAX_DEPTH = 10
MAX_CELLS = 20000


class PrepassStatus(enum.Enum):
    DISJOINT = "disjoint"
    DECIDED = "decided"
    DECLINED = "declined"


class PrepassDecline(enum.Enum):
    NOT_PLANE_NURBS = "not_plane_nurbs"
    RANGES_NOT_FULL = "ranges_not_full"
    NOT_UNIFORM_WEIGHTS = "not_uniform_weights"
    KNOT_DISCONTINUITY = "knot_discontinuity"
    PLANE_FOOTPRINT = "plane_footprint"
    ZERO_LINE_POLYNOMIAL = "zero_line_polynomial"
    MULTIPLE_EDGE_ROOT = "multiple_edge_root"
    CORNER_OR_DYADIC_ROOT = "corner_or_dyadic_root"
    DEPTH_CAP = "depth_cap"
    CELL_CAP = "cell_cap"
    ROOT_DEGREE = "root_degree"
    ROOT_IDENTITY = "root_identity"


class _Decline(Exception):
    def __init__(self, code: PrepassDecline):
        super().__init__(code.value)
        self.code = code


@dataclass(frozen=True, slots=True)
class PrepassRoot:
    """A simple root of F on a cell-edge line: ``line`` identifies it, ``(lo, hi)`` is the dyadic interval."""
    axis: int           # 0: the line is u = const (the root runs along v), 1: v = const
    transverse_span: int   # index of the knot span of the direction the root runs along
    line: Fraction      # absolute native coordinate of the line
    lo: Fraction        # absolute native interval enclosing the root along the line
    hi: Fraction
    boundary: bool      # the line is a boundary of the surface domain


@dataclass(frozen=True, slots=True)
class PrepassArc:
    cell: tuple         # (span_u, span_v, depth, i, j)
    first_root: int
    second_root: int
    native_box: tuple   # ((u0, u1), (v0, v1)) absolute exact


@dataclass(frozen=True, slots=True)
class PrepassComponent:
    closed: bool
    roots: tuple        # root indices in path order (closed: first repeated last)
    arcs: tuple         # PrepassArc in path order


@dataclass(frozen=True, slots=True)
class PrepassResult:
    status: PrepassStatus
    reason: PrepassDecline | None
    cells_tested: int
    excluded: int
    certified: int
    splits: int
    max_depth: int
    roots: tuple = ()
    components: tuple = ()


# ---------------------------------------------------------------- univariate exact helpers
def _power(bern):
    n = len(bern) - 1
    return [sum((bern[i] * comb(n, i) * comb(n - i, k - i) * (-1) ** (k - i) for i in range(k + 1)), Fraction(0))
            for k in range(n + 1)]


def _trim(p):
    while p and p[-1] == 0:
        p = p[:-1]
    return p


def _rem(a, b):
    a = _trim(list(a))
    while len(a) >= len(b):
        factor = a[-1] / b[-1]
        shift = len(a) - len(b)
        for i, c in enumerate(b):
            a[shift + i] -= factor * c
        a = _trim(a)
    return a


def _eval(p, x):
    total = Fraction(0)
    for c in reversed(p):
        total = total * x + c
    return total


class _Sturm:
    """Sturm chain of a squarefree nonzero polynomial; refuses (typed) otherwise."""

    def __init__(self, power):
        p = _trim(list(power))
        if not p:
            raise _Decline(PrepassDecline.ZERO_LINE_POLYNOMIAL)
        chain = [p]
        if len(p) > 1:
            chain.append(_trim([i * c for i, c in enumerate(p)][1:]))
            while len(chain[-1]) > 1:
                r = _rem(chain[-2], chain[-1])
                if not r:
                    raise _Decline(PrepassDecline.MULTIPLE_EDGE_ROOT)      # gcd(p, p') is not constant
                chain.append([-c for c in r])
        self.chain = chain

    def variations(self, x):
        signs = []
        for q in self.chain:
            v = _eval(q, x)
            if v:
                signs.append(v > 0)
        return sum(1 for a, b in zip(signs, signs[1:]) if a != b)

    def count(self, lo, hi):
        """Exact number of roots in (lo, hi); both endpoint values must be nonzero."""
        if _eval(self.chain[0], lo) == 0 or _eval(self.chain[0], hi) == 0:
            raise _Decline(PrepassDecline.CORNER_OR_DYADIC_ROOT)
        return self.variations(lo) - self.variations(hi)


# ---------------------------------------------------------------- cell certificates (pure, unit-testable)
def _flat(grid):
    return [c[0] for row in grid for c in row]


def excludes_cell(coefficients) -> bool:
    """True iff every Bernstein coefficient is strictly positive or every one strictly negative."""
    values = list(coefficients)
    return min(values) > 0 or max(values) < 0


def _elevate(rows):
    """Degree elevation along the first index of a list-of-rows scalar grid."""
    n = len(rows) - 1
    out = []
    for i in range(n + 2):
        a = Fraction(i, n + 1)
        out.append([(a * rows[i - 1][j] if i > 0 else 0) + ((1 - a) * rows[i][j] if i <= n else 0)
                    for j in range(len(rows[0]))])
    return out


def _transpose(rows):
    return [list(col) for col in zip(*rows)]


def monotone_direction(grid):
    """A direction ``d`` with ``d . (F_v, -F_u)`` strictly one-signed on the cell, else ``None``.

    ``grid`` is the scalar Bernstein grid of ``F`` as 1-tuples.  Exact; coefficients of the derivative
    grids (and, for the mean direction, of the exactly degree-elevated combination) must be STRICTLY one-signed.
    """
    fu = [[c[0] for c in row] for row in derivative_grid(grid, 0)]      # p x (q+1)
    fv = [[c[0] for c in row] for row in derivative_grid(grid, 1)]      # (p+1) x q
    for d, values in (((1, 0), [x for r in fv for x in r]), ((0, 1), [-x for r in fu for x in r])):
        if min(values) > 0 or max(values) < 0:
            return d
    tx = _transpose(_elevate(_transpose(fv)))                           # (p+1) x (q+1)
    ty = [[-x for x in r] for r in _elevate(fu)]                        # (p+1) x (q+1)
    count = len(tx) * len(tx[0])
    mx = sum((x for r in tx for x in r), Fraction(0)) / count
    my = sum((x for r in ty for x in r), Fraction(0)) / count
    if mx or my:
        values = [mx * a + my * b for ra, rb in zip(tx, ty) for a, b in zip(ra, rb)]
        if min(values) > 0 or max(values) < 0:
            return (mx, my)
    return None


# ---------------------------------------------------------------- operand analysis
def _as_plane(surface):
    """``(normal, offset, origin, e1, e2)`` if ``surface`` is a full-range bilinear parallelogram, else None."""
    if (surface.degree_u, surface.degree_v) != (1, 1) or len(surface.controls) != 2 or len(surface.controls[0]) != 2:
        return None
    w = surface.weights
    if w is not None and len({Fraction(x) for r in w for x in r}) != 1:
        return None
    ku, kv = surface.knots_u, surface.knots_v
    if len(ku) != 4 or len(kv) != 4 or ku[0] != ku[1] or ku[2] != ku[3] or kv[0] != kv[1] or kv[2] != kv[3]:
        return None
    if tuple(surface.u_range) != (ku[1], ku[2]) or tuple(surface.v_range) != (kv[1], kv[2]):
        raise _Decline(PrepassDecline.RANGES_NOT_FULL)
    p = [[tuple(Fraction(x) for x in point) for point in row] for row in surface.controls]
    e1 = tuple(a - b for a, b in zip(p[1][0], p[0][0]))
    e2 = tuple(a - b for a, b in zip(p[0][1], p[0][0]))
    if tuple(a - b for a, b in zip(p[1][1], p[0][1])) != e1 or tuple(a - b for a, b in zip(p[1][1], p[1][0])) != e2:
        return None
    n = (e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0])
    if not any(n):
        return None
    return n, sum(a * b for a, b in zip(n, p[0][0])), p[0][0], e1, e2


def _dot(a, b):
    return sum((x * y for x, y in zip(a, b)), Fraction(0))


def _footprint_inside(plane, net):
    """Every control point of the NURBS operand projects inside the closed plane patch (so every
    intersection point, which lies in the convex hull projection, lies in the patch)."""
    n, c, origin, e1, e2 = plane
    g11, g12, g22 = _dot(e1, e1), _dot(e1, e2), _dot(e2, e2)
    det = g11 * g22 - g12 * g12
    for row in net:
        for hx, hy, hz, w in row:
            d = (hx / w - origin[0], hy / w - origin[1], hz / w - origin[2])
            b1, b2 = _dot(d, e1), _dot(d, e2)
            s = (b1 * g22 - b2 * g12) / det
            t = (b2 * g11 - b1 * g12) / det
            if not (0 <= s <= 1 and 0 <= t <= 1):
                return False
    return True


def _knot_spans(knots, degree, count):
    domain = [Fraction(k) for k in knots[degree: count + 1]]
    breaks = sorted(set(domain))
    for b in breaks[1:-1]:
        if domain.count(b) > degree:
            raise _Decline(PrepassDecline.KNOT_DISCONTINUITY)
    return breaks


# ---------------------------------------------------------------- the pre-pass
class _Run:
    def __init__(self, surface, plane):
        self.surface = surface
        self.plane = plane
        net = surface.exact_net()
        self.net = net
        n_ = plane[0]
        self.normal, self.offset = n_, plane[1]
        ws = {row_point[3] for row in net for row_point in row}
        if len(ws) != 1 or min(ws) <= 0:
            raise _Decline(PrepassDecline.NOT_UNIFORM_WEIGHTS)
        if tuple(surface.u_range) != (surface.knots_u[surface.degree_u], surface.knots_u[len(net)]) or \
           tuple(surface.v_range) != (surface.knots_v[surface.degree_v], surface.knots_v[len(net[0])]):
            raise _Decline(PrepassDecline.RANGES_NOT_FULL)
        if not _footprint_inside(plane, net):
            raise _Decline(PrepassDecline.PLANE_FOOTPRINT)
        self.bu = _knot_spans(surface.knots_u, surface.degree_u, len(net))
        self.bv = _knot_spans(surface.knots_v, surface.degree_v, len(net[0]))
        self.span_grid = {}
        self.line_cache = {}          # (axis, absolute line, other span) -> (power poly, Sturm, bernstein)
        self.tested = self.excluded = self.certified = self.splits = 0
        self.max_depth = 0

    # F grid of a span as 1-tuples
    def span_f(self, si, sj):
        key = (si, sj)
        if key not in self.span_grid:
            piece = bezier_piece(self.surface, self.net, (self.bu[si], self.bu[si + 1]),
                                 (self.bv[sj], self.bv[sj + 1]))
            nx, ny, nz = self.normal
            self.span_grid[key] = tuple(tuple((nx * p[0] + ny * p[1] + nz * p[2] - self.offset * p[3],)
                                              for p in row) for row in piece)
        return self.span_grid[key]

    def line(self, axis, si, sj, alpha):
        """Sturm data of F along the line at local coordinate ``alpha`` of the span's ``axis`` direction."""
        a, b = (self.bu[si], self.bu[si + 1]) if axis == 0 else (self.bv[sj], self.bv[sj + 1])
        absolute = a + alpha * (b - a)
        other = sj if axis == 0 else si
        key = (axis, absolute, other)
        grid = self.span_f(si, sj)
        sub = subbox(grid, alpha, alpha, 0, 1) if axis == 0 else subbox(grid, 0, 1, alpha, alpha)
        bern = [c[0] for c in (sub[0] if axis == 0 else [row[0] for row in sub])]
        if key in self.line_cache:
            if self.line_cache[key][2] != bern:
                raise _Decline(PrepassDecline.ROOT_IDENTITY)       # inconsistent shared line: never expected
            return key, self.line_cache[key][1]
        power = _power(bern)
        self.line_cache[key] = (power, _Sturm(power), bern)
        return key, self.line_cache[key][1]

    def edges(self, si, sj, d, i, j):
        """Four edges ``(axis, alpha, segment_lo, segment_hi)`` of a dyadic cell, local to the span."""
        n = 1 << d
        u0, u1, v0, v1 = Fraction(i, n), Fraction(i + 1, n), Fraction(j, n), Fraction(j + 1, n)
        return ((0, u0, v0, v1), (0, u1, v0, v1), (1, v0, u0, u1), (1, v1, u0, u1))

    def cell_roots(self, si, sj, d, i, j):
        total = 0
        for axis, alpha, lo, hi in self.edges(si, sj, d, i, j):
            _, sturm = self.line(axis, si, sj, alpha)
            total += sturm.count(lo, hi)
        return total

    def classify(self):
        leaves = []                                     # (si, sj, d, i, j)
        stack = []
        for si in range(len(self.bu) - 1):
            for sj in range(len(self.bv) - 1):
                stack.append((si, sj, 0, 0, 0, self.span_f(si, sj)))
        while stack:
            si, sj, d, i, j, grid = stack.pop()
            self.tested += 1
            self.max_depth = max(self.max_depth, d)
            if self.tested > MAX_CELLS:
                raise _Decline(PrepassDecline.CELL_CAP)
            if excludes_cell(_flat(grid)):
                self.excluded += 1
                continue
            if monotone_direction(grid) is not None:
                roots = self.cell_roots(si, sj, d, i, j)
                if roots in (0, 2):
                    self.certified += 1
                    if roots:
                        leaves.append((si, sj, d, i, j))
                    continue
            if d >= MAX_DEPTH:
                raise _Decline(PrepassDecline.DEPTH_CAP)
            self.splits += 1
            h = Fraction(1, 2)
            for a in (0, 1):
                for b in (0, 1):
                    stack.append((si, sj, d + 1, 2 * i + a, 2 * j + b, subbox(grid, h * a, h * a + h, h * b, h * b + h)))
        return leaves

    def isolate(self, key, sturm, depth):
        """Canonical dyadic isolating intervals (local to [0,1]) of every root of the line polynomial."""
        out, todo = [], [(Fraction(0), Fraction(1))]
        while todo:
            lo, hi = todo.pop()
            c = sturm.count(lo, hi)
            if c == 0:
                continue
            if c == 1 and hi - lo <= Fraction(1, 1 << depth):
                out.append((lo, hi))
                continue
            mid = (lo + hi) / 2
            todo.append((lo, mid))
            todo.append((mid, hi))
        return sorted(out)

    def stitch(self, leaves):
        depth = self.max_depth + 1
        iso = {}
        roots, index = [], {}
        arcs = []
        boundary_lines = ((0, self.bu[0]), (0, self.bu[-1]), (1, self.bv[0]), (1, self.bv[-1]))
        for si, sj, d, i, j in leaves:
            ids = []
            for axis, alpha, lo, hi in self.edges(si, sj, d, i, j):
                key, sturm = self.line(axis, si, sj, alpha)
                if key not in iso:
                    iso[key] = self.isolate(key, sturm, depth)
                span = (self.bv[sj], self.bv[sj + 1]) if axis == 0 else (self.bu[si], self.bu[si + 1])
                for r_lo, r_hi in iso[key]:
                    if lo <= r_lo and r_hi <= hi:
                        ident = (key, r_lo, r_hi)
                        if ident not in index:
                            a0, a1 = span
                            index[ident] = len(roots)
                            roots.append(PrepassRoot(axis, key[2], key[1], a0 + r_lo * (a1 - a0), a0 + r_hi * (a1 - a0),
                                                     (axis, key[1]) in boundary_lines))
                        ids.append(index[ident])
                    elif not (r_hi <= lo or hi <= r_lo):
                        raise _Decline(PrepassDecline.ROOT_IDENTITY)
            if len(ids) != 2 or ids[0] == ids[1]:
                raise _Decline(PrepassDecline.ROOT_IDENTITY)
            n = 1 << d
            (u0, u1), (v0, v1) = ((self.bu[si] + Fraction(i, n) * (self.bu[si + 1] - self.bu[si]),
                                   self.bu[si] + Fraction(i + 1, n) * (self.bu[si + 1] - self.bu[si])),
                                  (self.bv[sj] + Fraction(j, n) * (self.bv[sj + 1] - self.bv[sj]),
                                   self.bv[sj] + Fraction(j + 1, n) * (self.bv[sj + 1] - self.bv[sj])))
            arcs.append(PrepassArc((si, sj, d, i, j), ids[0], ids[1], ((u0, u1), (v0, v1))))
        incident = {}
        for k, arc in enumerate(arcs):
            incident.setdefault(arc.first_root, []).append(k)
            incident.setdefault(arc.second_root, []).append(k)
        for r, root in enumerate(roots):
            if len(incident.get(r, ())) != (1 if root.boundary else 2):
                raise _Decline(PrepassDecline.ROOT_DEGREE)
        return tuple(roots), self._paths(roots, arcs, incident)

    @staticmethod
    def _paths(roots, arcs, incident):
        used = set()
        components = []

        def walk(start_root, first_arc):
            path_roots, path_arcs = [start_root], []
            root, k = start_root, first_arc
            while True:
                used.add(k)
                arc = arcs[k]
                nxt = arc.second_root if arc.first_root == root else arc.first_root
                path_arcs.append(arc)
                path_roots.append(nxt)
                if nxt == start_root:
                    return PrepassComponent(True, tuple(path_roots), tuple(path_arcs))
                rest = [x for x in incident[nxt] if x != k]
                if not rest:
                    return PrepassComponent(False, tuple(path_roots), tuple(path_arcs))
                root, k = nxt, rest[0]

        for r, root in enumerate(roots):
            if root.boundary and incident[r][0] not in used:
                components.append(walk(r, incident[r][0]))
        for k, arc in enumerate(arcs):
            if k not in used:
                components.append(walk(arc.first_root, k))
        return tuple(components)


def plane_nurbs_prepass(first, second) -> PrepassResult:
    """Decide the plane x NURBS section exactly, or DECLINE (typed) so the caller keeps the existing route."""
    run = None
    try:
        pair = None
        for plane_surface, other in ((first, second), (second, first)):
            plane = _as_plane(plane_surface)
            if plane is not None:
                pair = (plane, other)
                break
        if pair is None:
            raise _Decline(PrepassDecline.NOT_PLANE_NURBS)
        run = _Run(pair[1], pair[0])
        leaves = run.classify()
        if not leaves:
            return PrepassResult(PrepassStatus.DISJOINT, None, run.tested, run.excluded, run.certified,
                                 run.splits, run.max_depth)
        roots, components = run.stitch(leaves)
        return PrepassResult(PrepassStatus.DECIDED, None, run.tested, run.excluded, run.certified, run.splits,
                             run.max_depth, roots, components)
    except _Decline as refusal:
        t = (run.tested, run.excluded, run.certified, run.splits, run.max_depth) if run else (0, 0, 0, 0, 0)
        return PrepassResult(PrepassStatus.DECLINED, refusal.code, *t)
