"""Plane x NURBS section topology (generality slice B1-P1, architecture 4.1-4.2).

Everything is decided in exact rational arithmetic on the stored doubles (the
oracle of architecture 5.3); only reported positions are rounded.

* **Substitution (4.1).** The plane ``n . x = d`` (``n = u_vector x v_vector``,
  ``d = n . origin``, exact) and the homogeneous net ``H_ij = (w P, w)`` give
  ``G = n . H[:3] - d H[3]``, a polynomial of bidegree ``(p, q)`` in Bernstein
  form on every Bezier sub-patch (exact knot insertion over the surface's
  ranges). Its zero set is the plane section (positive weights).
* **Quad-tree (4.2 step 1).** A box is *empty* when its coefficients share a
  strict sign; *regular in x* when the coefficients of ``G_x`` share a strict
  sign (``x`` = v or u): there the zero set is the graph ``x = f(s)`` of the
  other coordinate ``s``; otherwise it is halved in both directions, to a
  depth limit (then refused: singular boxes, 4.2 step 2).
* **Box-edge roots (step 3).** The restriction of ``G`` to each edge of a
  regular box is a univariate Bernstein polynomial; its distinct roots are
  isolated exactly (``exact_levels``); a multiple root (the section tangent to
  a box edge) refuses. Roots on one global line are identified by an exact root
  count on the union of their intervals (the same function on that line, also
  across knot lines by continuity).
* **Clipping contract (step 3).** In a regular box the events (its edge roots,
  projected to ``s``) are separated and sorted; between consecutive events the
  graph is inside the box iff ``G(s_m, x)`` changes sign over the box's ``x``
  range at a rational ``s_m`` in the gap (``G`` is monotone in ``x`` there):
  exact, no Newton. Inside gaps are arcs joining their two events.
* **Accounting.** Roots on the patch boundary have degree 1, all others 2;
  the arcs form paths between boundary roots; cycles, other degrees or
  leftovers refuse (``PlaneNurbsClosedSection`` / ``PlaneNurbsNotCertified``).
* **Coincidence (step 5).** ``G`` identically zero: the surface lies in the plane.

The result lists paths of arcs (box, slicing, event roots); slice P2 adds the
per-piece interval Newton certificates and evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .errors import GeometryError
from .exact_levels import (_bernstein_to_power, _count, _derivative, _gcd, _roots_in_unit, _sign,
                           _square_free, _trim, _value)
from .nurbs import NurbsSurface, _exact_insert

MAX_DEPTH = 12
MAX_BOXES = 20000


class PlaneNurbsNotCertified(GeometryError):
    """The plane x NURBS section could not be certified (singular box, tangency, budget)."""


class PlaneNurbsClosedSection(PlaneNurbsNotCertified):
    """The section contains a closed loop inside the patch (outside the certified scope)."""


# ------------------------------------------------------------------ exact Bezier sub-patches


@dataclass(frozen=True)
class SubPatch:
    """A Bezier sub-patch: knot ranges ``[u0, u1] x [v0, v1]`` and the Bernstein coefficients of ``G``."""
    index: int
    u_range: tuple
    v_range: tuple
    coefficients: tuple            # (p + 1) x (q + 1) Fractions


def _sub_patches(plane, surface: NurbsSurface) -> list[SubPatch]:
    origin = [Fraction(float(value)) for value in plane.origin]
    a = [Fraction(float(value)) for value in plane.u_vector]
    b = [Fraction(float(value)) for value in plane.v_vector]
    n = (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
    if not any(n):
        raise GeometryError("plane parameter vectors must be independent")
    d = sum(x * y for x, y in zip(n, origin))
    net = surface.exact_net()
    p, q = surface.degree_u, surface.degree_v
    g = [[n[0] * h[0] + n[1] * h[1] + n[2] * h[2] - d * h[3] for h in row] for row in net]
    knots_u, knots_v = [Fraction(k) for k in surface.knots_u], [Fraction(k) for k in surface.knots_v]
    low_u, high_u = sorted(Fraction(value) for value in surface.u_range)
    low_v, high_v = sorted(Fraction(value) for value in surface.v_range)
    # insert every knot inside the ranges (and the range ends) to multiplicity degree, along u then v
    rows = [tuple(row) for row in g]
    for value in sorted({k for k in knots_u if low_u < k < high_u} | {low_u, high_u}):
        if knots_u[p] < value < knots_u[len(rows)]:
            times = p - knots_u.count(value)
            if times > 0:
                knots_u, rows = _exact_insert(p, knots_u, rows, value, times)
    columns = [tuple(row[j] for row in rows) for j in range(len(rows[0]))]
    for value in sorted({k for k in knots_v if low_v < k < high_v} | {low_v, high_v}):
        if knots_v[q] < value < knots_v[len(columns)]:
            times = q - knots_v.count(value)
            if times > 0:
                knots_v, columns = _exact_insert(q, knots_v, columns, value, times)
    patches = []
    distinct_u, distinct_v = sorted(set(knots_u)), sorted(set(knots_v))
    for u0, u1 in zip(distinct_u, distinct_u[1:]):
        if u1 <= low_u or u0 >= high_u:
            continue
        i0 = knots_u.index(u0) + knots_u.count(u0) - 1 - p
        for v0, v1 in zip(distinct_v, distinct_v[1:]):
            if v1 <= low_v or v0 >= high_v:
                continue
            j0 = knots_v.index(v0) + knots_v.count(v0) - 1 - q
            coefficients = tuple(tuple(columns[j0 + j][i0 + i] for j in range(q + 1)) for i in range(p + 1))
            patches.append(SubPatch(len(patches), (u0, u1), (v0, v1), coefficients))
    return patches


# ------------------------------------------------------------------ Bernstein boxes


def _split(row: list, at: Fraction = Fraction(1, 2)):
    left, right, work = [row[0]], [row[-1]], list(row)
    for _ in range(len(row) - 1):
        work = [(1 - at) * a + at * b for a, b in zip(work, work[1:])]
        left.append(work[0])
        right.append(work[-1])
    return left, right[::-1]


def _split_u(coefficients):
    columns = list(zip(*coefficients))
    halves = [_split(list(column)) for column in columns]
    left = tuple(tuple(column[i] for column, _ in halves) for i in range(len(coefficients)))
    right = tuple(tuple(column[i] for _, column in halves) for i in range(len(coefficients)))
    return left, right


def _split_v(coefficients):
    halves = [_split(list(row)) for row in coefficients]
    return tuple(tuple(h[0]) for h in halves), tuple(tuple(h[1]) for h in halves)


@dataclass(frozen=True)
class Box:
    """A quad-tree leaf: knot box ``[u0, u1] x [v0, v1]`` of a sub-patch with its Bernstein coefficients."""
    patch: int
    u_range: tuple
    v_range: tuple
    coefficients: tuple
    kind: str                       # "empty", "u" (graph u = f(v)), "v" (graph v = f(u))


def _all_sign(values) -> int:
    values = list(values)
    if all(v > 0 for v in values):
        return 1
    if all(v < 0 for v in values):
        return -1
    return 0


def _classify(coefficients) -> str | None:
    flat = [c for row in coefficients for c in row]
    if _all_sign(flat):
        return "empty"
    p, q = len(coefficients) - 1, len(coefficients[0]) - 1
    if q > 0 and _all_sign(coefficients[i][j + 1] - coefficients[i][j] for i in range(p + 1) for j in range(q)):
        return "v"
    if p > 0 and _all_sign(coefficients[i + 1][j] - coefficients[i][j] for i in range(p) for j in range(q + 1)):
        return "u"
    return None


def _boxes(patch: SubPatch) -> list[Box]:
    leaves: list[Box] = []
    pending = [(patch.u_range, patch.v_range, patch.coefficients, 0)]
    while pending:
        u_range, v_range, coefficients, depth = pending.pop()
        kind = _classify(coefficients)
        if kind is not None:
            leaves.append(Box(patch.index, u_range, v_range, coefficients, kind))
            continue
        if all(c == 0 for row in coefficients for c in row):
            raise GeometryError("coincident")              # handled by the caller (whole-surface check first)
        if depth >= MAX_DEPTH or len(leaves) + len(pending) > MAX_BOXES:
            raise PlaneNurbsNotCertified(f"singular box at depth {depth}: u {tuple(map(float, u_range))}, "
                                         f"v {tuple(map(float, v_range))}")
        um, vm = (u_range[0] + u_range[1]) / 2, (v_range[0] + v_range[1]) / 2
        left, right = _split_u(coefficients)
        for (ua, ub), half in (((u_range[0], um), left), ((um, u_range[1]), right)):
            low, high = _split_v(half)
            pending.append(((ua, ub), (v_range[0], vm), low, depth + 1))
            pending.append(((ua, ub), (vm, v_range[1]), high, depth + 1))
    return leaves


# ------------------------------------------------------------------ edge roots and their global identity


@dataclass
class Root:
    """A zero of ``G`` on a box edge: on the line ``axis = value`` with the other coordinate in ``[lo, hi]``."""
    axis: int                       # 0: the line u = value (root's v in [lo, hi]); 1: the line v = value
    value: Fraction
    lo: Fraction
    hi: Fraction
    poly: list                      # square-free restriction in the edge's local parameter (power basis)
    edge: tuple                     # (a, b): the edge's other-coordinate range, local t = (y - a) / (b - a)
    ident: int = -1

    @property
    def exact(self) -> bool:
        return self.lo == self.hi

    def point(self):
        y = (self.lo + self.hi) / 2
        return (self.value, y) if self.axis == 0 else (y, self.value)

    def refine(self):
        """Halve the isolating interval (bisection on the restriction's sign; exact midpoints)."""
        if self.exact:
            return
        a, b = self.edge
        t_lo, t_hi = (self.lo - a) / (b - a), (self.hi - a) / (b - a)
        middle = (t_lo + t_hi) / 2
        value = _value(self.poly, middle)
        if value == 0:
            self.lo = self.hi = a + middle * (b - a)
        elif (value > 0) == (_value(self.poly, t_lo) > 0):
            self.lo = a + middle * (b - a)
        else:
            self.hi = a + middle * (b - a)


def _edge_roots(box: Box) -> list[Root]:
    c = box.coefficients
    edges = [  # (axis, fixed value, other range, Bernstein restriction)
        (0, box.u_range[0], box.v_range, list(c[0])),
        (0, box.u_range[1], box.v_range, list(c[-1])),
        (1, box.v_range[0], box.u_range, [row[0] for row in c]),
        (1, box.v_range[1], box.u_range, [row[-1] for row in c]),
    ]
    roots = []
    for axis, value, (a, b), bernstein in edges:
        if all(x == 0 for x in bernstein):
            raise PlaneNurbsNotCertified("the section runs along a box edge")
        power = _bernstein_to_power(bernstein)
        if len(_trim(power)) == 1:
            continue
        repeated = _gcd(power, _derivative(power))
        for lo, hi, g in _roots_in_unit(power):
            if len(repeated) > 1 and _count(repeated, lo, hi) > 0:
                raise PlaneNurbsNotCertified("the section is tangent to a box edge")
            roots.append(Root(axis, value, a + lo * (b - a), a + hi * (b - a), g, (a, b)))
    return roots


def _identify(roots: list[Root]) -> list[Root]:
    """Give every root a global id: exact points by equality, isolated roots on one line by an exact count."""
    distinct: list[Root] = []
    for root in roots:
        match = None
        for other in distinct:
            if root.exact and other.exact:
                if root.point() == other.point():
                    match = other
            elif root.axis == other.axis and root.value == other.value:
                if root.lo <= other.hi and other.lo <= root.hi:
                    # same function on this line: the union holds one root iff they are the same
                    lo, hi = min(root.lo, other.lo), max(root.hi, other.hi)
                    a, b = root.edge
                    poly = root.poly
                    if _count(poly, (lo - a) / (b - a), (hi - a) / (b - a)) == 1:
                        match = other
            elif root.exact != other.exact:
                # an exact (corner) point and an isolated root on a line through it
                exact, isolated = (root, other) if root.exact else (other, root)
                px = exact.point()
                if isolated.axis == 0 and px[0] == isolated.value and isolated.lo <= px[1] <= isolated.hi \
                        or isolated.axis == 1 and px[1] == isolated.value and isolated.lo <= px[0] <= isolated.hi:
                    a, b = isolated.edge
                    t = ((px[1] if isolated.axis == 0 else px[0]) - a) / (b - a)
                    if _value(isolated.poly, t) == 0:
                        match = other
            if match is not None:
                break
        if match is None:
            root.ident = len(distinct)
            distinct.append(root)
        else:
            root.ident = match.ident
            if root.exact and not match.exact:
                match.lo, match.hi, match.axis, match.value = root.lo, root.hi, root.axis, root.value
    return distinct


# ------------------------------------------------------------------ clipping contract per regular box


def _graph_inside(box: Box, s: Fraction) -> bool:
    """Whether the graph is inside the box over the (event-free) parameter ``s``: G changes sign in x there."""
    c = box.coefficients
    if box.kind == "v":                                    # s = u, x = v: evaluate the rows at u = s
        a, b = box.u_range
        t = (s - a) / (b - a)
        column = [_de_casteljau([row[j] for row in c], t) for j in range(len(c[0]))]
    else:                                                  # s = v, x = u
        a, b = box.v_range
        t = (s - a) / (b - a)
        column = [_de_casteljau(list(row), t) for row in c]
    first, last = column[0], column[-1]
    if first == 0 or last == 0:
        raise PlaneNurbsNotCertified("a gap midpoint lies on the section")
    return (first > 0) != (last > 0)


def _de_casteljau(values: list, t: Fraction) -> Fraction:
    work = list(values)
    for _ in range(len(work) - 1):
        work = [(1 - t) * a + t * b for a, b in zip(work, work[1:])]
    return work[0]


def _arcs(box: Box, roots: list[Root]) -> list[tuple[int, int]]:
    """In-box arcs of a regular box as pairs of global root ids (ordered by the slicing parameter)."""
    slicing = 0 if box.kind == "v" else 1                  # the independent coordinate: u for "v" boxes
    unique = {}
    for root in roots:
        unique.setdefault(root.ident, root)
    events = list(unique.values())

    def projection(root):
        point_axis = root.axis                             # the root's line fixes this coordinate
        if point_axis == slicing:                          # on an edge of constant s
            return (root.value, root.value)
        return (root.lo, root.hi)

    for _ in range(200):
        spans = sorted((projection(root), root.ident) for root in events)
        overlapping = [(i, j) for i in range(len(spans)) for j in range(i + 1, len(spans))
                       if spans[i][0][0] <= spans[j][0][1] and spans[j][0][0] <= spans[i][0][1]]
        if not overlapping:
            break
        for root in events:
            root.refine()
    else:
        raise PlaneNurbsNotCertified("two section events in one box cannot be separated")
    spans = sorted((projection(root), root.ident) for root in events)
    s_low, s_high = (box.u_range if slicing == 0 else box.v_range)
    arcs = []
    for (first, first_id), (second, second_id) in zip(spans, spans[1:]):
        middle = (first[1] + second[0]) / 2
        if _graph_inside(box, middle):
            arcs.append((first_id, second_id))
    # a gap before the first or after the last event cannot be inside (it would end in an event)
    for (span, _ident), end in ((spans[0], s_low), (spans[-1], s_high)) if spans else ():
        boundary_gap = (end, span[0]) if end < span[0] else (span[1], end) if span[1] < end else None
        if boundary_gap is not None and boundary_gap[0] < boundary_gap[1] and \
                _graph_inside(box, (boundary_gap[0] + boundary_gap[1]) / 2):
            raise PlaneNurbsNotCertified("the section leaves a box between events")
    return arcs


# ------------------------------------------------------------------ the section


@dataclass(frozen=True)
class SectionPath:
    """One section path: global root ids in path order (first and last on the patch boundary) and its arcs."""
    roots: tuple
    arcs: tuple                     # (box index, first root id, second root id) per arc, in path order


@dataclass(frozen=True)
class PlaneNurbsSection:
    coincident: bool
    paths: tuple
    roots: tuple                    # global Root records by id
    boxes: tuple
    sub_patches: tuple

    def chart_point(self, ident: int, surface: NurbsSurface) -> tuple[float, float]:
        """The chart point (rounded) of a root: knot coordinates mapped through the surface ranges."""
        u, v = self.roots[ident].point()
        (r0, r1), (s0, s1) = ((Fraction(x) for x in surface.u_range), (Fraction(x) for x in surface.v_range))
        return float((u - r0) / (r1 - r0)), float((v - s0) / (s1 - s0))


def plane_nurbs_section(plane, surface: NurbsSurface) -> PlaneNurbsSection:
    """Certified topology of ``plane`` x ``surface`` (architecture 4.2); refuses outside the scope."""
    patches = _sub_patches(plane, surface)
    if all(c == 0 for patch in patches for row in patch.coefficients for c in row):
        return PlaneNurbsSection(True, (), (), (), tuple(patches))
    boxes = [box for patch in patches for box in _boxes(patch)]
    regular = [box for box in boxes if box.kind != "empty"]
    per_box = [_edge_roots(box) for box in regular]
    distinct = _identify([root for roots in per_box for root in roots])
    low_u, high_u = sorted(Fraction(value) for value in surface.u_range)
    low_v, high_v = sorted(Fraction(value) for value in surface.v_range)

    def on_boundary(root: Root) -> bool:
        u, v = root.point()
        if root.exact:
            return u in (low_u, high_u) or v in (low_v, high_v)
        return (root.axis == 0 and root.value in (low_u, high_u)) or (root.axis == 1 and root.value in (low_v, high_v))

    arcs = []
    for index, (box, roots) in enumerate(zip(regular, per_box)):
        for first, second in _arcs(box, roots):
            if first == second:
                raise PlaneNurbsNotCertified("an arc joins a root to itself")
            arcs.append((index, first, second))
    degree = {root.ident: 0 for root in distinct}
    for _box, first, second in arcs:
        degree[first] += 1
        degree[second] += 1
    for root in distinct:
        expected = 1 if on_boundary(root) else 2
        if degree[root.ident] != expected:
            raise PlaneNurbsNotCertified(f"section root {root.ident} has degree {degree[root.ident]} "
                                         f"(expected {expected})")
    adjacency: dict[int, list] = {root.ident: [] for root in distinct}
    for arc in arcs:
        adjacency[arc[1]].append(arc)
        adjacency[arc[2]].append(arc)
    used: set = set()
    paths = []
    for start in sorted(root.ident for root in distinct if on_boundary(root)):
        if any(arc in used for arc in adjacency[start]):
            continue
        order, path_arcs, current = [start], [], start
        while True:
            options = [arc for arc in adjacency[current] if arc not in used]
            if not options:
                break
            arc = options[0]
            used.add(arc)
            current = arc[2] if arc[1] == current else arc[1]
            path_arcs.append(arc if arc[1] == order[-1] else (arc[0], arc[2], arc[1]))
            order.append(current)
        paths.append(SectionPath(tuple(order), tuple(path_arcs)))
    if len(used) != len(arcs):
        raise PlaneNurbsClosedSection("the section contains a closed loop inside the patch")
    boundary_roots = sum(1 for root in distinct if on_boundary(root))
    if boundary_roots != 2 * len(paths):
        raise PlaneNurbsNotCertified("patch-boundary roots do not match the paths")
    return PlaneNurbsSection(False, tuple(paths), tuple(distinct), tuple(regular), tuple(patches))
