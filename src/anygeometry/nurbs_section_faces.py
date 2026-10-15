"""NURBS faces trimmed by straight chart segments and plane x NURBS section curves (generality slice B1-P4c).

A NURBS face (``surface=NurbsSurface``) whose loops contain ``PlaneNurbsCurve`` edges on its own
support is validated in the surface's chart, without sampled trims (architecture 2.6, 3.3):

- every section edge's surface is the face's support (exact equality); every other edge is a
  ``CurveOnSurface`` on the support (identity transform) whose pcurve is a straight chart segment
  (degree 1, two controls; iso-curves are the B1 case). Curved pcurves next to section curves are
  refused in this slice (architecture 4.5: "section x general pcurve refused");
- consecutive edges meet in the chart within the chart tolerance (the joint rule of the P4 sub-plan:
  a section curve ends at an algebraic root, given as an exact chart rectangle);
- pairs: segment x segment exactly (rationals); segment x section by ``segment_section`` (P4a's
  engine with the chart line ``a u + b v - c`` as its second polynomial: exact Bernstein exclusion,
  monotonicity along each arc graph, crossings certified within the chart tolerance of the
  segment); section x section on one section path by their exact domains (P1 paths are simple in
  the chart), on two paths or planes by separated chart enclosures;
- every loop's signed chart area (Green; exact on segments, Gauss-Legendre on section arcs) with a
  rigorous bound: each piece of a section arc is a graph over its slicing coordinate, so the region
  between it and its chord lies in its chart rectangle (``_arc_rects``);
- holes inside the outer loop and outside each other by chart ray parity.

Chart semantics follow A3b: the trims are decided in the chart; the chart tolerance is the world
tolerance divided by ``sqrt(2) sigma_max`` (architecture 1.4, forward only).
"""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np

from .branch_predicates import _engine
from .plane_nurbs import PlaneNurbsNotCertified
from .support_curves import PlaneNurbsCurve

MAX_DEPTH = 48
_GAUSS = np.polynomial.legendre.leggauss(24)
_RAYS = 8


class _Patch:
    __slots__ = ("coefficients", "u_range", "v_range")

    def __init__(self, coefficients, u_range, v_range):
        self.coefficients, self.u_range, self.v_range = coefficients, u_range, v_range


def _line_patches(a, b, c):
    """Per arc box: the bilinear Bernstein form of ``f = a u + b v - c`` (knot coordinates, exact) and
    the constant weight 1, as the engine's second polynomial."""
    def patches(arc):
        box = arc.box
        (u0, u1), (v0, v1) = box.u_range, box.v_range

        def f(u, v):
            return a * u + b * v - c
        return (_Patch(((f(u0, v0), f(u0, v1)), (f(u1, v0), f(u1, v1))), box.u_range, box.v_range),
                _Patch(((Fraction(1), Fraction(1)), (Fraction(1), Fraction(1))), box.u_range, box.v_range))
    return patches


class _Chart:
    """Knot <-> chart coordinates of a surface (exact)."""

    def __init__(self, surface):
        (self.r0, self.r1), (self.q0, self.q1) = (tuple(Fraction(value) for value in rng)
                                                  for rng in (surface.u_range, surface.v_range))
        self.ru, self.rv = self.r1 - self.r0, self.q1 - self.q0

    def point(self, u, v):
        return ((u - self.r0) / self.ru, (v - self.q0) / self.rv)

    def rect(self, u_range, v_range):
        us = sorted((self.point(u, self.q0)[0] for u in u_range))
        vs = sorted((self.point(self.r0, v)[1] for v in v_range))
        return (us[0], us[-1]), (vs[0], vs[-1])

    def line(self, p0, p1):
        """Knot coefficients ``(a, b, c)`` of ``f = n . (chart - p0)``, ``n`` normal to ``p1 - p0``."""
        n0, n1 = -(p1[1] - p0[1]), p1[0] - p0[0]
        return (n0 / self.ru, n1 / self.rv,
                n0 * self.r0 / self.ru + n1 * self.q0 / self.rv + n0 * p0[0] + n1 * p0[1])


def _corners(rect):
    (u0, u1), (v0, v1) = rect
    return [(u0, v0), (u0, v1), (u1, v0), (u1, v1)]


def _rect_point_gap2(rect, point):
    (u0, u1), (v0, v1) = rect
    du = max(u0 - point[0], point[0] - u1, Fraction(0))
    dv = max(v0 - point[1], point[1] - v1, Fraction(0))
    return du * du + dv * dv


def _rect_gap2(first, second):
    (a0, a1), (b0, b1) = first
    (c0, c1), (d0, d1) = second
    du = max(a0 - c1, c0 - a1, Fraction(0))
    dv = max(b0 - d1, d0 - b1, Fraction(0))
    return du * du + dv * dv


def _rect_spread2(first, second):
    """The largest squared distance between a point of one rectangle and a point of the other."""
    (a0, a1), (b0, b1) = first
    (c0, c1), (d0, d1) = second
    du = max(abs(a0 - c1), abs(a1 - c0))
    dv = max(abs(b0 - d1), abs(b1 - d0))
    return du * du + dv * dv


def _segment_distance2(point, p0, p1):
    e = (p1[0] - p0[0], p1[1] - p0[1])
    ee = e[0] * e[0] + e[1] * e[1]
    lam = Fraction(0) if ee == 0 else min(Fraction(1), max(Fraction(0),
                                                            ((point[0] - p0[0]) * e[0] + (point[1] - p0[1]) * e[1]) / ee))
    du, dv = point[0] - p0[0] - lam * e[0], point[1] - p0[1] - lam * e[1]
    return du * du + dv * dv


def _context_rect(chart, context):
    u_range, v_range = context.view.uv(context.s_range, context.x_range)
    return chart.rect(u_range, v_range)


def segment_section(curve: PlaneNurbsCurve, p0, p1, shared, tolerance):
    """Whether the chart segment ``p0 -> p1`` (exact chart points) meets the section curve's chart
    image only within ``tolerance`` (chart units) of the ``shared`` chart points: ``True``; a contact
    certified within the tolerance of the segment and away from them: ``False``; else ``None``.
    The semantics of ``certify_line_section`` (P4a) in the chart."""
    chart = _Chart(curve.surface)
    a, b, c = chart.line(p0, p1)
    if a == 0 and b == 0:
        return None
    tol2 = Fraction(tolerance) ** 2
    e = (p1[0] - p0[0], p1[1] - p0[1])
    ee = e[0] * e[0] + e[1] * e[1]
    shared_set = {tuple(p) for p in shared}
    crossing = [False]

    def classify(context) -> str:
        rect = _context_rect(chart, context)
        corners = _corners(rect)
        if any(all((x - p[0]) ** 2 + (y - p[1]) ** 2 <= tol2 for x, y in corners) for p in shared):
            return "done"
        lambdas = [((x - p0[0]) * e[0] + (y - p0[1]) * e[1]) / ee for x, y in corners]
        # beyond an end the nearest segment point is that end: cleared when farther than the tolerance
        # from it, or when that end is a shared joint (its contacts are joint contacts; review B1-P4c
        # round 1: pieces straddling the joint ball's rim were never resolved)
        if (max(lambdas) < 0 and (tuple(p0) in shared_set or _rect_point_gap2(rect, p0) > tol2)) or \
                (min(lambdas) > 1 and (tuple(p1) in shared_set or _rect_point_gap2(rect, p1) > tol2)):
            return "done"                       # also a piece on the segment's line beyond its ends
        low, high = context.g2_range()
        if low > 0 or high < 0:
            return "done"
        if low == high == 0 and context.solid():
            # a section piece on the segment's line, not beyond its ends nor at a joint: an overlap
            # (review B1-P4c round 1: a collinear continuation is cleared above, not refused)
            if all(_rect_point_gap2(rect, p) > tol2 for p in shared) and \
                    all(_segment_distance2(corner, p0, p1) <= tol2 for corner in corners):
                crossing[0] = True
                return "done"
            return "split"
        signs = (context.sign_from, context.sign_to)
        if 0 not in signs and context.view.monotone(context.s_range, context.x_range):
            if signs[0] == signs[1]:
                return "done"
            if all(_rect_point_gap2(rect, p) > tol2 for p in shared) and \
                    all(_segment_distance2(corner, p0, p1) <= tol2 for corner in corners):
                crossing[0] = True
                return "done"
        width = (rect[0][1] - rect[0][0]) ** 2 + (rect[1][1] - rect[1][0]) ** 2
        if tol2 > 0 and width * 256 <= tol2:
            raise PlaneNurbsNotCertified("chart segment x section contact undecided at the tolerance scale")
        return "split"

    try:
        _engine(curve, None, None, classify, MAX_DEPTH, patches=_line_patches(a, b, c))
    except PlaneNurbsNotCertified:
        return False if crossing[0] else None
    return False if crossing[0] else True


def ray_crossings(curve: PlaneNurbsCurve, origin, direction, tolerance):
    """Number of crossings of the chart ray ``origin + r direction`` (r > 0) with the section curve's chart
    image, or ``None`` when undecided (a crossing near the origin, tangency, an unresolved root)."""
    chart = _Chart(curve.surface)
    p1 = (origin[0] + direction[0], origin[1] + direction[1])
    a, b, c = chart.line(origin, p1)
    count = [0]
    tol2 = Fraction(tolerance) ** 2

    def classify(context) -> str:
        low, high = context.g2_range()
        if low > 0 or high < 0:
            return "done"
        rect = _context_rect(chart, context)
        along = [(x - origin[0]) * direction[0] + (y - origin[1]) * direction[1] for x, y in _corners(rect)]
        if max(along) < 0 and _rect_point_gap2(rect, origin) > tol2:
            return "done"
        signs = (context.sign_from, context.sign_to)
        if 0 not in signs and context.view.monotone(context.s_range, context.x_range):
            if signs[0] == signs[1]:
                return "done"
            if min(along) > 0 and _rect_point_gap2(rect, origin) > tol2:
                count[0] += 1
                return "done"
        width = (rect[0][1] - rect[0][0]) ** 2 + (rect[1][1] - rect[1][0]) ** 2
        if tol2 > 0 and width * 256 <= tol2:
            raise PlaneNurbsNotCertified("chart ray undecided")
        return "split"

    try:
        _engine(curve, None, None, classify, MAX_DEPTH, patches=_line_patches(a, b, c))
    except PlaneNurbsNotCertified:
        return None
    return count[0]


# ------------------------------------------------------------------ face validation


def _is_segment_pcurve(curve) -> bool:
    pcurve = curve.pcurve
    return pcurve.degree == 1 and len(pcurve.controls) == 2


def _chart_tolerance(surface, tolerance) -> Fraction:
    from .surface_capabilities import capabilities
    sigma = capabilities(surface).jacobian_bounds(((0.0, 1.0), (0.0, 1.0))).sigma_max
    return Fraction(tolerance / (math.sqrt(2.0) * sigma)) if sigma > 0 else Fraction(tolerance)


def _ends(model, item, chart):
    """Chart sets of an oriented edge's start and end: exact points (pcurves) or exact rectangles
    (the section's end-root enclosures)."""
    curve = model.edges[item.edge].curve
    if isinstance(curve, PlaneNurbsCurve):
        sets = []
        for parameter in (0.0, 1.0):
            (u_lo, u_hi), (v_lo, v_hi) = curve.chart_enclosure(parameter)
            sets.append(chart.rect((u_lo, u_hi), (v_lo, v_hi)))
    else:
        sets = [((point[0], point[0]), (point[1], point[1]))
                for point in (curve.pcurve.exact_point(0), curve.pcurve.exact_point(1))]
    return (sets[0], sets[1]) if item.forward else (sets[1], sets[0])


def _center(rect):
    return ((rect[0][0] + rect[0][1]) / 2, (rect[1][0] + rect[1][1]) / 2)


def _segments_meet(p0, p1, q0, q1) -> bool:
    """Exact: closed chart segments share a point."""
    def orient(a, b, c):
        value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return (value > 0) - (value < 0)

    def on(a, b, c):
        return min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
    o1, o2, o3, o4 = orient(p0, p1, q0), orient(p0, p1, q1), orient(q0, q1, p0), orient(q0, q1, p1)
    if o1 != o2 and o3 != o4:
        return True
    return any(o == 0 and on(a, b, c) for o, a, b, c in
               ((o1, p0, p1, q0), (o2, p0, p1, q1), (o3, q0, q1, p0), (o4, q0, q1, p1)))


def _section_pieces(curve: PlaneNurbsCurve, level: int):
    """Parameter pieces inside single arcs (between breakpoints), each cut into ``2**level``."""
    knots = sorted({0.0, 1.0, *curve.breakpoints()})
    split = 2 ** level
    return [(lo + (hi - lo) * k / split, lo + (hi - lo) * (k + 1) / split)
            for lo, hi in zip(knots, knots[1:]) for k in range(split)]


def _piece_rect(curve: PlaneNurbsCurve, chart, lo, hi):
    """Chart rectangle holding C* (and the defined points) over the parameter piece."""
    rects = curve._arc_rects(min(lo, hi), max(lo, hi))
    us, vs = [], []
    for arc, s_range, x_range in rects:
        u_range, v_range = (s_range, x_range) if arc.slicing == 0 else (x_range, s_range)
        (cu, cv) = chart.rect(u_range, v_range)
        us.extend(cu)
        vs.extend(cv)
    return (min(us), max(us)), (min(vs), max(vs))


def _sections_apart(first, second, chart, tolerance, max_level: int = 8):
    """Chart enclosures of two section curves separated by more than the tolerance (refined)."""
    tol2 = Fraction(tolerance) ** 2
    pieces = [[(0.0, 1.0)], [(0.0, 1.0)]]
    for _level in range(max_level + 1):
        rects = [[_piece_rect(curve, chart, lo, hi) for lo, hi in spans] for curve, spans in zip((first, second), pieces)]
        close = set()
        for i, a in enumerate(rects[0]):
            for j, b in enumerate(rects[1]):
                if _rect_gap2(a, b) <= tol2:
                    close.update(((0, i), (1, j)))
        if not close:
            return True
        for side, index in sorted(close, reverse=True):
            lo, hi = pieces[side][index]
            pieces[side][index:index + 1] = [(lo, .5 * (lo + hi)), (.5 * (lo + hi), hi)]
    return False


def chart_loop_area(model, loop, surface, level: int = 3) -> tuple[float, float]:
    """``(area, bound)``: the signed chart area of a loop (Green) and a rigorous bound of its error.

    Segments exactly; section edges: Gauss-Legendre per arc (value), the chord polyline through
    pieces inside single arcs (each a graph over its slicing coordinate: the region between a piece
    and its chord lies in the piece's chart rectangle), bound = rectangle areas + |gauss - chord|
    + rounding scaled by the summed products.
    """
    chart = _Chart(surface)
    origin = _center(_ends(model, loop[0], chart)[0])
    ox, oy = float(origin[0]), float(origin[1])
    nodes, weights = _GAUSS
    exact = Fraction(0)
    gauss = chord = bound = magnitude = 0.0
    for item in loop:
        curve = model.edges[item.edge].curve
        if not isinstance(curve, PlaneNurbsCurve):
            start, end = (curve.pcurve.exact_point(0), curve.pcurve.exact_point(1))
            if not item.forward:
                start, end = end, start
            x0, y0, x1, y1 = start[0] - origin[0], start[1] - origin[1], end[0] - origin[0], end[1] - origin[1]
            exact += (x0 * y1 - x1 * y0) / 2
            continue
        sign = 1.0 if item.forward else -1.0
        knots = sorted({0.0, 1.0, *curve.breakpoints()})
        for lo, hi in zip(knots, knots[1:]):
            t = .5 * (hi - lo) * (nodes + 1.) + lo
            points = curve.chart(t) - (ox, oy)
            derivative = curve.chart_derivative(t)
            first, second = points[:, 0] * derivative[:, 1], points[:, 1] * derivative[:, 0]
            gauss += sign * .25 * (hi - lo) * float(weights @ (first - second))
            magnitude += .25 * (hi - lo) * float(weights @ (np.abs(first) + np.abs(second)))
        pieces = _section_pieces(curve, level)
        ts = [pieces[0][0]] + [hi for _lo, hi in pieces]
        polyline = curve.chart(np.asarray(ts)) - (ox, oy)
        products = (polyline[:-1, 0] * polyline[1:, 1], polyline[1:, 0] * polyline[:-1, 1])
        chord += sign * .5 * float(np.sum(products[0] - products[1]))
        magnitude += .5 * float(np.sum(np.abs(products[0]) + np.abs(products[1])))
        for lo, hi in pieces:
            (u0, u1), (v0, v1) = _piece_rect(curve, chart, lo, hi)
            bound += float((u1 - u0) * (v1 - v0)) * (1 + 1e-12)
    eps = np.finfo(float).eps
    area = float(exact) + gauss
    return area, bound + abs(gauss - chord) + 64 * eps * magnitude + 4 * eps * abs(float(exact))


def has_section_edges(model, face) -> bool:
    return any(isinstance(model.edges[item.edge].curve, PlaneNurbsCurve)
               for loop in (face.loop,) + tuple(face.holes) for item in loop)


def validate_nurbs_section_face(model, face_id: int) -> list[str]:
    """Validation messages of a NURBS face trimmed by chart segments and section curves (empty: valid)."""
    from .nurbs import CurveOnSurface, _IDENTITY
    face = model.faces[face_id]
    surface = face.surface
    loops = (face.loop,) + tuple(face.holes)
    result: list[str] = []
    for loop in loops:
        for item in loop:
            curve = model.edges[item.edge].curve
            if isinstance(curve, PlaneNurbsCurve):
                if curve.surface != surface:
                    result.append(f"face {face_id} edge {item.edge} is a section curve of another surface")
            elif isinstance(curve, CurveOnSurface) and curve.support == surface and curve.transform == _IDENTITY:
                if not _is_segment_pcurve(curve):
                    result.append(f"face {face_id} edge {item.edge}: only straight chart segments may trim a "
                                  "NURBS face next to section curves (B1-P4c)")
            else:
                result.append(f"face {face_id} edge {item.edge} of a NURBS face is not an exact chart curve "
                              "on the face's support")
    if result:
        return result
    starts = [model.oriented_start_vertex(item) for loop in loops for item in loop]
    if len(set(starts)) != len(starts):
        return [f"face {face_id} loops pass a vertex more than once"]
    chart = _Chart(surface)
    positions = np.asarray([model.vertices[v].position for v in starts], dtype=float)
    extent = float(np.linalg.norm(np.ptp(positions, axis=0))) or 1.0
    tolerance = _chart_tolerance(surface, model.tolerance.effective_length(extent))
    tol2 = tolerance * tolerance
    ends = {(index, position): _ends(model, item, chart)
            for index, loop in enumerate(loops) for position, item in enumerate(loop)}
    # chart continuity (the joint rule): exact between two pcurve ends; at a section end every point
    # of its end-root rectangle within the chart tolerance of the other end (review B1-P4c round 1)
    for index, loop in enumerate(loops):
        for position in range(len(loop)):
            following = (position + 1) % len(loop)
            end, start = ends[(index, position)][1], ends[(index, following)][0]
            exact_pair = not any(isinstance(model.edges[loop[k].edge].curve, PlaneNurbsCurve)
                                 for k in (position, following))
            if (end != start) if exact_pair else _rect_spread2(end, start) > tol2:
                result.append(f"face {face_id} loop {index} is not continuous in the chart at edge "
                              f"{loop[following].edge}")
    if result:
        return result
    # chart location of each topology vertex (exact where an edge gives an exact point)
    located: dict = {}
    for index, loop in enumerate(loops):
        for position, item in enumerate(loop):
            vertex = model.oriented_start_vertex(item)
            start = ends[(index, position)][0]
            exact_point = start[0][0] == start[0][1] and start[1][0] == start[1][1]
            if vertex not in located or exact_point:
                located[vertex] = _center(start)
    items = [(index, position, item) for index, loop in enumerate(loops) for position, item in enumerate(loop)]
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (la, pa, first), (lb, pb, second) = items[i], items[j]
            a, b = model.edges[first.edge], model.edges[second.edge]
            shared_ids = {a.start, a.end} & {b.start, b.end}
            shared = [located[v] for v in shared_ids if v in located]
            count = len(loops[la])
            neighbours = la == lb and (abs(pa - pb) == 1 or abs(pa - pb) == count - 1)
            verdict = _pair(model, a, b, chart, shared, shared_ids, tolerance, neighbours)
            if verdict is True:
                continue
            where = f"loop {la}" if la == lb else f"loops {la} and {lb}"
            if verdict is False:
                result.append(f"face {face_id} {where} self-intersects in the chart (edges {first.edge}, "
                              f"{second.edge})" if la == lb else
                              f"face {face_id} {where} intersect in the chart (edges {first.edge}, {second.edge})")
            else:
                result.append(f"face {face_id} {where}: edges {first.edge} and {second.edge} not certified "
                              "disjoint in the chart")
    if result:
        return result
    for index, loop in enumerate(loops):
        centers = [_center(ends[(index, position)][0]) for position in range(len(loop))]
        span = max(max(abs(float(p[0] - q[0])), abs(float(p[1] - q[1]))) for p in centers for q in centers) or 1.0
        for level in (3, 5, 7):
            try:
                area, bound = chart_loop_area(model, loop, surface, level)
            except PlaneNurbsNotCertified as error:
                result.append(f"face {face_id} loop {index}: {error}")
                break
            if abs(area) > bound + 1e-12 * span * span:
                break
        else:
            result.append(f"face {face_id} loop {index} has zero chart area")
    for index in range(1, len(loops)):
        point = _center(ends[(index, 0)][0])
        inside = _inside(model, loops[0], point, chart, tolerance)
        if inside is not True:
            result.append(f"face {face_id} hole {index} is not strictly inside the outer loop" if inside is False
                          else f"face {face_id} hole {index} containment not certified")
        for other in range(1, len(loops)):
            if other != index and _inside(model, loops[other], point, chart, tolerance) is not False:
                result.append(f"face {face_id} holes {index} and {other} overlap")
    return result


def _pair(model, a, b, chart, shared, shared_ids, tolerance, neighbours=False):
    first_section, second_section = isinstance(a.curve, PlaneNurbsCurve), isinstance(b.curve, PlaneNurbsCurve)
    if not first_section and not second_section:
        p0, p1 = a.curve.pcurve.exact_point(0), a.curve.pcurve.exact_point(1)
        q0, q1 = b.curve.pcurve.exact_point(0), b.curve.pcurve.exact_point(1)
        common = {tuple(p) for p in (p0, p1)} & {tuple(q) for q in (q0, q1)}
        if not common:
            return False if _segments_meet(p0, p1, q0, q1) else True
        if not (neighbours and shared_ids and len(common) == 1):
            return False                        # contact between non-neighbours (or a repeated segment)
        # neighbours: they may share only their joint (no fold back along one line)
        far_a = [p for p in (p0, p1) if tuple(p) not in common]
        far_b = [q for q in (q0, q1) if tuple(q) not in common]
        if any(_segment_distance2(p, q0, q1) <= tolerance * tolerance for p in far_a) or \
                any(_segment_distance2(q, p0, p1) <= tolerance * tolerance for q in far_b):
            return False
        return True
    if first_section and second_section:
        if (a.curve.plane_record, a.curve.surface, a.curve.path) == (b.curve.plane_record, b.curve.surface, b.curve.path):
            low_a, high_a = sorted(a.curve.domain)
            low_b, high_b = sorted(b.curve.domain)
            if min(high_a, high_b) > max(low_a, low_b):
                return False                                    # overlapping pieces of one section path
            if high_a == low_b or high_b == low_a:
                return True if shared_ids else None             # touching domain ends: a shared vertex
            return True                                         # P1 paths are simple in the chart
        if shared_ids:
            # two section curves meeting at a corner (B1-P5a): P4b's world piece certificate (non-neighbour
            # pieces separated, the corner's pieces departing on opposite sides); a surface map is
            # injective on a NURBS face's chart, so world simplicity gives chart simplicity
            from .entities import OrientedEdge
            from .section_faces import _sections_simple
            world_tolerance = model.tolerance.effective_length(_pair_extent(a, b))
            loose, _pieces = _sections_simple(model, ((OrientedEdge(a.id, True), OrientedEdge(b.id, True)),),
                                              world_tolerance)
            return True if not loose else None
        return True if _sections_apart(a.curve, b.curve, chart, tolerance) else None
    section, segment = (a, b) if first_section else (b, a)
    p0, p1 = segment.curve.pcurve.exact_point(0), segment.curve.pcurve.exact_point(1)
    return segment_section(section.curve, p0, p1, shared, tolerance)


def _pair_extent(a, b) -> float:
    lows, highs = [], []
    for curve in (a.curve, b.curve):
        low, high = curve.bounds()
        lows.append(low)
        highs.append(high)
    return float(np.linalg.norm(np.max(highs, axis=0) - np.min(lows, axis=0))) or 1.0


def _inside(model, loop, point, chart, tolerance):
    """Chart ray parity over a fixed sequence of rational directions; ``None`` when no ray decides."""
    tol2 = tolerance * tolerance
    for k in range(_RAYS):
        direction = (Fraction(1000 + 37 * k), Fraction(377 - 101 * k))
        count, decided = 0, True
        for item in loop:
            curve = model.edges[item.edge].curve
            if isinstance(curve, PlaneNurbsCurve):
                crossings = ray_crossings(curve, point, direction, tolerance)
                if crossings is None:
                    decided = False
                    break
                count += crossings
                continue
            p0, p1 = curve.pcurve.exact_point(0), curve.pcurve.exact_point(1)
            side0 = (p0[0] - point[0]) * direction[1] - (p0[1] - point[1]) * direction[0]
            side1 = (p1[0] - point[0]) * direction[1] - (p1[1] - point[1]) * direction[0]
            if side0 == 0 or side1 == 0:
                decided = False                                 # the ray passes a vertex
                break
            if (side0 > 0) != (side1 > 0):
                r = side0 / (side0 - side1)
                crossing = (p0[0] + r * (p1[0] - p0[0]), p0[1] + r * (p1[1] - p0[1]))
                along = (crossing[0] - point[0]) * direction[0] + (crossing[1] - point[1]) * direction[1]
                if (crossing[0] - point[0]) ** 2 + (crossing[1] - point[1]) ** 2 <= tol2:
                    decided = False
                    break
                if along > 0:
                    count += 1
        if decided:
            return count % 2 == 1
    return None
