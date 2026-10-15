"""Certified strict-audit classifications around plane x NURBS sections (generality slice B1-P4d).

Contract (owner decision 2026-10-15, option a: strict uniform tolerance). Two entities are clean
when every pair of their points closer than the tolerance lies in the *departure stretch* of a
joint they share (a shared vertex, or a shared section edge for two faces); everywhere else they
are certified farther apart than the tolerance. At a joint ``X`` both sides must leave along a
separating direction ``w``: each side's piece at ``X`` is certified strictly monotone in ``w . P``
(a plane face lies at a constant offset, a NURBS face patch within a small intrusion), on opposite
sides. With ``E`` = the joint-end offsets + ``tolerance |w|``, a point beyond the offset plane
``w . (P - X) = +-E`` of its side is farther than the tolerance from the whole other side, so
contacts can only lie in the stretches before those planes; a stretch longer than ``E / (|w|
sin(angular tolerance))`` (a departure flatter than the angular tolerance) refuses.

Exception (owner decision 2026-10-15, option b): a plane face and a NURBS face that share section
edges are certified on their exact intersection only (it must be exactly their shared section edges);
the strict tolerance band along the shared edge is not certified yet (review B1-P4d round 3: chart
transversality does not bound the band in world space). Without shared section edges they must be
certified farther apart than the tolerance.

Every subdivision counts against ``Work`` (the audit's ``max_boxes_per_pair`` and
``max_subdivision_depth``), also inside the delegated P1/P4a/P4b predicates (``CURRENT_WORK``);
the work is reported back. Each classifier returns ``True`` (certified clean) or ``None``
(refused: not certified, or out of scope).
"""
from __future__ import annotations

from contextlib import contextmanager
from fractions import Fraction
import math

import numpy as np

from .curves import Straight
from .plane_nurbs import CURRENT_WORK, PlaneNurbsNotCertified, plane_nurbs_section
from .support_curves import PlaneNurbsCurve


class Work:
    """The audit's qualification budget for one pair, and the work consumed (reported back)."""

    def __init__(self, max_boxes: int = 200_000, max_depth: int = 32):
        self.max_boxes, self.max_depth = int(max_boxes), int(max_depth)
        self.boxes = self.subdivisions = 0

    def take(self) -> bool:
        self.boxes += 1
        return self.boxes <= self.max_boxes


@contextmanager
def _charging(work: Work):
    token = CURRENT_WORK.set(work)
    try:
        yield
    finally:
        CURRENT_WORK.reset(token)


def _ex(vector):
    return [Fraction(float(value)) for value in vector]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _norm_up(vector) -> Fraction:
    """A rational upper bound of the Euclidean norm."""
    squared = sum(Fraction(float(x)) ** 2 for x in vector)
    root = Fraction(math.sqrt(float(squared))) * (1 + Fraction(1, 2 ** 40))
    while root * root < squared:
        root *= 1 + Fraction(1, 2 ** 30)
    return root


def _corners(lower, upper):
    return [[Fraction(lower[k] if (i >> k) & 1 == 0 else upper[k]) for k in range(3)] for i in range(8)]


def _box_gap2(first, second) -> Fraction:
    (la, ua), (lb, ub) = first, second
    total = Fraction(0)
    for k in range(3):
        gap = max(Fraction(la[k]) - Fraction(ub[k]), Fraction(lb[k]) - Fraction(ua[k]), Fraction(0))
        total += gap * gap
    return total


def _offsets(box, w, x):
    values = [_dot(w, [c - y for c, y in zip(corner, x)]) for corner in _corners(*box)]
    return min(values), max(values)


# ------------------------------------------------------------------ curve-like entities


class _Curve:
    """A curve edge in its own parameter ``[0, 1]``: pieces, exact world boxes, monotone projections."""

    def __init__(self, model, edge_id):
        edge = model.edges[edge_id]
        self.edge_id, self.start, self.end, self.curve = edge_id, edge.start, edge.end, edge.curve
        self.model = model

    def end_param(self, vertex):
        return 0.0 if vertex == self.start else 1.0


class _Section(_Curve):
    def point(self, t):
        return np.asarray(self.curve.evaluate(float(t)), dtype=float)

    def box(self, a, b):
        lower, upper = self.curve.bounds(min(a, b), max(a, b))
        return _ex(lower), _ex(upper)

    def monotone(self, a, b, w):
        from .branch_predicates import projection_monotone
        return projection_monotone(self.curve.subcurve(a, b), w)

    def tangent(self, t):
        return np.asarray(self.curve.derivative(t), dtype=float)


class _Line(_Curve):
    def __init__(self, model, edge_id):
        super().__init__(model, edge_id)
        self.a = _ex(model.vertices[self.start].position)
        self.b = _ex(model.vertices[self.end].position)

    def point_exact(self, t):
        t = Fraction(t)
        return [x + t * (y - x) for x, y in zip(self.a, self.b)]

    def point(self, t):
        return np.asarray([float(v) for v in self.point_exact(t)])

    def box(self, a, b):
        p, q = self.point_exact(a), self.point_exact(b)
        return [min(x, y) for x, y in zip(p, q)], [max(x, y) for x, y in zip(p, q)]

    def monotone(self, a, b, w):
        value = _dot(_ex(w), [y - x for x, y in zip(self.a, self.b)])
        return 1 if value > 0 else -1 if value < 0 else None

    def tangent(self, t):
        return np.asarray([float(y - x) for x, y in zip(self.a, self.b)])


class _ChartSegment(_Curve):
    """A ``CurveOnSurface`` whose pcurve is a straight chart segment (degree 1, two controls)."""

    def __init__(self, model, edge_id):
        super().__init__(model, edge_id)
        self.surface = self.curve.support
        self.p0, self.p1 = self.curve.pcurve.exact_point(0), self.curve.pcurve.exact_point(1)
        self.net = self.surface.exact_net()

    def chart(self, t):
        t = Fraction(t)
        return [x + t * (y - x) for x, y in zip(self.p0, self.p1)]

    def point(self, t):
        return np.asarray([float(v) for v in self.surface.exact_point(*self.chart(t))])

    def chart_box(self, a, b):
        p, q = self.chart(a), self.chart(b)
        lo = [min(x, y) for x, y in zip(p, q)]
        hi = [max(x, y) for x, y in zip(p, q)]
        return tuple((max(0.0, math.nextafter(float(l), -math.inf)), min(1.0, math.nextafter(float(h), math.inf)))
                     for l, h in zip(lo, hi))

    def box(self, a, b):
        lower, upper, _sigma = self.surface.patch_bounds(self.chart_box(a, b), self.net)
        return list(lower), list(upper)

    def monotone(self, a, b, w):
        return _chart_projection_sign(self.surface, self.chart_box(a, b),
                                      [y - x for x, y in zip(self.p0, self.p1)], w)

    def tangent(self, t):
        uv = np.asarray([[float(value) for value in self.chart(t)]])
        s_u, s_v = self.surface.derivatives_many(uv)
        d = [float(y - x) for x, y in zip(self.p0, self.p1)]
        return s_u[0] * d[0] + s_v[0] * d[1]


def _chart_projection_sign(surface, chart_box, direction, w):
    """Sign of ``d/dt (w . S(p(t)))`` for the chart path ``p`` along ``direction`` inside ``chart_box``:
    the interval of ``D(A) W - A D(W)`` (``A = w . H``, ``D`` the directional derivative in knot units)
    over every Bezier sub-patch meeting the box; ``None`` when undecided."""
    from .branch_predicates import _derivative_ranges, _multiply, _patches, _range, _rect, _subtract
    n = tuple(Fraction(float(value)) for value in w)
    a_patches = _patches(surface, n, Fraction(0))
    w_patches = _patches(surface, (Fraction(0),) * 3, Fraction(-1))
    (r0, r1), (q0, q1) = (tuple(Fraction(v) for v in rng) for rng in (surface.u_range, surface.v_range))
    knot_u = sorted(r0 + Fraction(c) * (r1 - r0) for c in chart_box[0])
    knot_v = sorted(q0 + Fraction(c) * (q1 - q0) for c in chart_box[1])
    du, dv = Fraction(direction[0]) * (r1 - r0), Fraction(direction[1]) * (q1 - q0)
    sign = None
    for a_patch, w_patch in zip(a_patches, w_patches):
        u_lo, u_hi = max(knot_u[0], a_patch.u_range[0]), min(knot_u[1], a_patch.u_range[1])
        v_lo, v_hi = max(knot_v[0], a_patch.v_range[0]), min(knot_v[1], a_patch.v_range[1])
        if u_lo > u_hi or v_lo > v_hi:
            continue
        a = _rect(a_patch.coefficients, a_patch.u_range, a_patch.v_range, (u_lo, u_hi), (v_lo, v_hi))
        ww = _rect(w_patch.coefficients, w_patch.u_range, w_patch.v_range, (u_lo, u_hi), (v_lo, v_hi))

        def directional(coefficients):
            # D = du d/du + dv d/dv over the rectangle (a zero width needs a zero component)
            if (u_hi == u_lo and du != 0) or (v_hi == v_lo and dv != 0):
                return None
            (gu, gv) = _derivative_ranges(coefficients, (u_hi - u_lo) or Fraction(1), (v_hi - v_lo) or Fraction(1))
            terms = []
            if du != 0:
                terms.append(_multiply(gu, (du, du)))
            if dv != 0:
                terms.append(_multiply(gv, (dv, dv)))
            low = sum(t[0] for t in terms)
            high = sum(t[1] for t in terms)
            return low, high
        da, dw = directional(a), directional(ww)
        if da is None or dw is None:
            return None
        total = _subtract(_multiply(da, _range(ww)), _multiply(_range(a), dw))
        piece = 1 if total[0] > 0 else -1 if total[1] < 0 else None
        if piece is None or (sign is not None and piece != sign):
            return None
        sign = piece
    return sign


def _curve_of(model, edge_id):
    from .nurbs import CurveOnSurface, _IDENTITY
    curve = model.edges[edge_id].curve
    if isinstance(curve, PlaneNurbsCurve):
        return _Section(model, edge_id)
    if isinstance(curve, Straight):
        return _Line(model, edge_id)
    if (isinstance(curve, CurveOnSurface) and curve.transform == _IDENTITY and curve.pcurve.degree == 1
            and len(curve.pcurve.controls) == 2):
        return _ChartSegment(model, edge_id)
    return None


# ------------------------------------------------------------------ joints


def _stretch_box(curve, a, b, at_start, x, w, limit, side, work):
    """World box of the part of ``curve[a, b]`` next to its joint end where ``side * w . (P - x)`` may
    still be at most ``limit`` (bisection: the rest is certified beyond ``limit``)."""
    lo, hi = (a, b) if at_start else (b, a)                    # lo: the joint end
    near, far = lo, hi
    for _ in range(work.max_depth):
        if not work.take():
            return None
        middle = .5 * (near + far)
        low, high = _offsets(curve.box(min(middle, hi), max(middle, hi)), w, x)
        beyond = (low > limit) if side > 0 else (high < -limit)
        if beyond:
            far = middle
        else:
            near = middle
        if abs(far - near) <= 1e-300:
            break
    end = far
    beyond_low, beyond_high = _offsets(curve.box(min(end, hi), max(end, hi)), w, x)
    if not ((beyond_low > limit) if side > 0 else (beyond_high < -limit)) and end != hi:
        return None
    return curve.box(min(lo, end), max(lo, end))


def _within(box, x, radius2) -> bool:
    return all(sum((c - y) ** 2 for c, y in zip(corner, x)) <= radius2 for corner in _corners(*box))


def _joint_curves(first, pa, second, pb, vertex, tolerance, angular, work) -> bool:
    """Two curve pieces at their shared vertex: departure along ``w = unit(away_b) - unit(away_a)``."""
    x_float = np.asarray(first.model.vertices[vertex].position, dtype=float)
    x = _ex(x_float)
    pieces = []
    for curve, (a, b) in ((first, pa), (second, pb)):
        at_start = curve.end_param(vertex) == a
        tangent = curve.tangent(a if at_start else b)
        norm = float(np.linalg.norm(tangent))
        if not norm or not math.isfinite(norm):
            return False
        pieces.append((curve, a, b, at_start, (tangent if at_start else -tangent) / norm))
    w = pieces[1][4] - pieces[0][4]
    if not np.all(np.isfinite(w)) or float(np.linalg.norm(w)) <= 1e-12:
        return False
    wx = _ex(w)
    sides, offsets = [], []
    for curve, a, b, at_start, _away in pieces:
        sign = curve.monotone(a, b, wx)
        if sign is None:
            return False
        sides.append(sign if at_start else -sign)
        t = a if at_start else b
        low, high = _offsets(curve.box(t, t), wx, x)
        offsets.append(max(abs(low), abs(high)))
    if sides != [-1, 1]:
        return False
    norm_w = _norm_up(w)
    limit = offsets[0] + offsets[1] + Fraction(float(tolerance)) * norm_w
    radius = limit / norm_w / Fraction(math.sin(angular)) + Fraction(float(tolerance))
    for (curve, a, b, at_start, _away), side in zip(pieces, sides):
        box = _stretch_box(curve, a, b, at_start, x, wx, limit, side, work)
        if box is None or not _within(box, x, radius * radius):
            return False
    return True


# ------------------------------------------------------------------ pair engines


def _definitely_close(p, q, tolerance) -> bool:
    """Two actual points (rounded within ulps; section points within delta) certainly closer than the
    tolerance: a proven near-contact, refused at once instead of refining to the budget."""
    distance = float(np.linalg.norm(np.asarray(p, dtype=float) - np.asarray(q, dtype=float)))
    margin = 1e-6 * tolerance + 8 * np.finfo(float).eps * float(np.max(np.abs(np.concatenate((p, q)))) or 1.0)
    return distance + margin < tolerance


def _halve(spans, key):
    a, b = key
    index = spans.index(key)
    spans[index:index + 1] = [(a, .5 * (a + b)), (.5 * (a + b), b)]


def _curves_clean(first, second, shared, tolerance, angular, work) -> bool:
    """Two curve entities: non-joint piece pairs separated by more than the tolerance; joint pieces
    (both touching one shared vertex) departing (``_joint_curves``)."""
    tol2 = Fraction(float(tolerance)) ** 2
    spans = [[(0.0, .5), (.5, 1.0)], [(0.0, .5), (.5, 1.0)]]
    boxes, joints = {}, {}
    for level in range(work.max_depth + 1):
        loose = set()
        for pa in spans[0]:
            for pb in spans[1]:
                vertex = next((v for v in shared if first.end_param(v) in pa and second.end_param(v) in pb), None)
                if vertex is not None:
                    key = (pa, pb, vertex)
                    if key not in joints:
                        joints[key] = _joint_curves(first, pa, second, pb, vertex, tolerance, angular, work)
                    if not joints[key]:
                        loose.update(((0, pa), (1, pb)))
                    continue
                for side, piece, curve in ((0, pa, first), (1, pb, second)):
                    if (side, piece) not in boxes:
                        if not work.take():
                            return False
                        boxes[(side, piece)] = curve.box(*piece)
                if _box_gap2(boxes[(0, pa)], boxes[(1, pb)]) <= tol2:
                    # pieces near a shared vertex: the departure certificate over the spans from the joint
                    # through these pieces (review B1-P4d round 3: a piece not touching the joint may still
                    # lie in its departure stretch)
                    if any(_span_joint(first, pa, second, pb, v, tolerance, angular, work, joints) for v in shared):
                        continue
                    if _definitely_close(first.point(.5 * sum(pa)), second.point(.5 * sum(pb)), tolerance):
                        return False                            # a proven near-contact away from the joints
                    loose.update(((0, pa), (1, pb)))
        if not loose:
            return True
        if level == work.max_depth:
            return False
        for side, piece in loose:
            work.subdivisions += 1
            _halve(spans[side], piece)
    return False


def _span_joint(first, pa, second, pb, vertex, tolerance, angular, work, memo) -> bool:
    spans = []
    for curve, (a, b) in ((first, pa), (second, pb)):
        t = curve.end_param(vertex)
        spans.append((t, b) if t == 0.0 else (a, t))
    key = (tuple(spans[0]), tuple(spans[1]), vertex, "span")
    if key not in memo:
        memo[key] = _joint_curves(first, spans[0], second, spans[1], vertex, tolerance, angular, work)
    return memo[key]


def _curve_plane_clean(curve, plane, shared, tolerance, angular, work) -> bool:
    """A curve entity against a plane face (the whole plane, conservative)."""
    from .section_faces import _cross, _exact
    n = _cross(_exact(plane.u_vector), _exact(plane.v_vector))
    d = _dot(n, _exact(plane.origin))
    norm_n = _norm_up(n)
    tol = Fraction(float(tolerance))
    spans = [(0.0, .5), (.5, 1.0)]
    verdicts = {}
    for level in range(work.max_depth + 1):
        loose = set()
        for piece in spans:
            if piece in verdicts:
                if not verdicts[piece]:
                    loose.add(piece)
                continue
            if not work.take():
                return False
            vertex = next((v for v in shared if curve.end_param(v) in piece), None)
            if vertex is None:
                low, high = _offsets(curve.box(*piece), n, [Fraction(0)] * 3)
                ok = (low - d > tol * norm_n) or (high - d < -tol * norm_n)
                if not ok:
                    middle = curve.point(.5 * sum(piece))
                    gap = abs(float(_dot(n, _ex(middle)) - d)) / float(norm_n)
                    if gap + 1e-6 * float(tol) + 8 * np.finfo(float).eps * float(np.max(np.abs(middle)) or 1.0) < float(tol):
                        return False                            # a proven near-contact with the plane
            else:
                ok = _joint_plane(curve, piece, vertex, n, d, norm_n, tol, angular, work)
            verdicts[piece] = ok
            if not ok:
                loose.add(piece)
        if not loose:
            return True
        if level == work.max_depth:
            return False
        for piece in loose:
            work.subdivisions += 1
            del verdicts[piece]
            _halve(spans, piece)
    return False


def _joint_plane(curve, piece, vertex, n, d, norm_n, tol, angular, work) -> bool:
    """A curve piece leaving a plane face at a shared vertex: monotone in ``n . P`` away from the plane."""
    a, b = piece
    at_start = curve.end_param(vertex) == a
    x = _ex(curve.model.vertices[vertex].position)
    tangent = curve.tangent(a if at_start else b)
    away = tangent if at_start else -tangent
    orient = 1 if float(np.dot([float(v) for v in n], away)) > 0 else -1
    w = [orient * value for value in n]
    sign = curve.monotone(a, b, w)
    if sign is None or (sign if at_start else -sign) != 1:
        return False
    t = a if at_start else b
    low, high = _offsets(curve.box(t, t), w, x)
    plane_offset = abs(_dot(n, x) - d)                          # the plane face's offset from the joint
    limit = max(abs(low), abs(high)) + plane_offset + tol * norm_n
    radius = limit / norm_n / Fraction(math.sin(angular)) + tol
    box = _stretch_box(curve, a, b, at_start, x, w, limit, 1, work)
    return box is not None and _within(box, x, radius * radius)


def _face_chart_box(model, face):
    lows, highs = [], []
    for loop in (face.loop,) + tuple(face.holes):
        for item in loop:
            curve = model.edges[item.edge].curve
            low, high = curve.chart_bounds() if isinstance(curve, PlaneNurbsCurve) else curve.pcurve.bounds()
            lows.append(low)
            highs.append(high)
    low = np.clip(np.min(lows, axis=0), 0.0, 1.0)
    high = np.clip(np.max(highs, axis=0), 0.0, 1.0)
    return (float(low[0]), float(high[0])), (float(low[1]), float(high[1]))


def _segment_frame(a, b):
    e = [y - x for x, y in zip(a, b)]
    k = min(range(3), key=lambda i: abs(e[i]))
    axis = [Fraction(1) if i == k else Fraction(0) for i in range(3)]
    n1 = [e[1] * axis[2] - e[2] * axis[1], e[2] * axis[0] - e[0] * axis[2], e[0] * axis[1] - e[1] * axis[0]]
    n2 = [e[1] * n1[2] - e[2] * n1[1], e[2] * n1[0] - e[0] * n1[2], e[0] * n1[1] - e[1] * n1[0]]
    return a, e, n1, n2


def _box_segment_far(box, frame, tol2) -> bool:
    """Exact: every point of the box farther than the tolerance from the segment (orthogonal frame)."""
    a, e, n1, n2 = frame
    corners = [[c - y for c, y in zip(corner, a)] for corner in _corners(*box)]
    total = Fraction(0)
    for axis, segment_range in ((e, None), (n1, (Fraction(0), Fraction(0))), (n2, (Fraction(0), Fraction(0)))):
        norm2 = _dot(axis, axis)
        if norm2 == 0:
            return False
        values = [_dot(axis, c) for c in corners]
        lo, hi = (Fraction(0), norm2) if segment_range is None else segment_range
        gap = max(min(values) - hi, lo - max(values), Fraction(0))
        total += gap * gap / norm2
    return total > tol2


def _line_face_clean(line, model, face, shared, tolerance, angular, work) -> bool:
    """A straight line against a NURBS face: face patches (exact sub-patch hulls over the face's chart box)
    separated from the line pieces, or, at a shared vertex, a joint: the line departs along ``w`` (the
    surface normal at the vertex, oriented along the line) and the patch intrudes at most the tolerance."""
    surface = face.surface
    net = surface.exact_net()
    tol = Fraction(float(tolerance))
    tol2 = tol * tol
    chart = {}
    for vertex in shared:                                       # the joint's chart point (exact where given)
        chart[vertex] = _vertex_chart(model, face, vertex)
        if chart[vertex] is None:
            return False
    pending = [(_face_chart_box(model, face), 0)]
    while pending:
        rect, depth = pending.pop()
        if not work.take():
            return False
        lower, upper, _sigma = surface.patch_bounds(rect, net)
        box = (list(lower), list(upper))
        joint = next((v for v, (cu, cv) in chart.items()
                      if rect[0][0] <= cu <= rect[0][1] and rect[1][0] <= cv <= rect[1][1]), None)
        if joint is None:
            if _box_segment_far(box, _segment_frame(line.a, line.b), tol2):
                continue
        elif _joint_line_patch(line, joint, rect, surface, chart[joint], tol, angular, work):
            continue
        if depth >= work.max_depth:
            return False
        work.subdivisions += 1
        (u0, u1), (v0, v1) = rect
        um, vm = .5 * (u0 + u1), .5 * (v0 + v1)
        for us in ((u0, um), (um, u1)):
            for vs in ((v0, vm), (vm, v1)):
                pending.append(((us, vs), depth + 1))
    return True


def _vertex_chart(model, face, vertex):
    for loop in (face.loop,) + tuple(face.holes):
        for item in loop:
            if model.oriented_start_vertex(item) != vertex:
                continue
            curve = model.edges[item.edge].curve
            parameter = 0.0 if item.forward else 1.0
            if isinstance(curve, PlaneNurbsCurve):
                return tuple(float(v) for v in curve.chart(parameter))
            point = curve.pcurve.exact_point(parameter)
            return float(point[0]), float(point[1])
    return None


def _plane_function_max(surface, rect, n, d):
    """An upper bound of ``(n . S - d)`` over the chart rectangle (exact Bernstein ranges of ``n . H - d W``
    divided by the weight's range; positive weights)."""
    from .branch_predicates import _patches, _range, _rect
    g_patches = _patches(surface, tuple(n), d)
    w_patches = _patches(surface, (Fraction(0),) * 3, Fraction(-1))
    (r0, r1), (q0, q1) = (tuple(Fraction(v) for v in rng) for rng in (surface.u_range, surface.v_range))
    knot_u = sorted(r0 + Fraction(c) * (r1 - r0) for c in rect[0])
    knot_v = sorted(q0 + Fraction(c) * (q1 - q0) for c in rect[1])
    best = None
    for g_patch, w_patch in zip(g_patches, w_patches):
        u_lo, u_hi = max(knot_u[0], g_patch.u_range[0]), min(knot_u[1], g_patch.u_range[1])
        v_lo, v_hi = max(knot_v[0], g_patch.v_range[0]), min(knot_v[1], g_patch.v_range[1])
        if u_lo > u_hi or v_lo > v_hi:
            continue
        g_high = _range(_rect(g_patch.coefficients, g_patch.u_range, g_patch.v_range, (u_lo, u_hi), (v_lo, v_hi)))[1]
        w_low, w_high = _range(_rect(w_patch.coefficients, w_patch.u_range, w_patch.v_range, (u_lo, u_hi), (v_lo, v_hi)))
        value = g_high / (w_low if g_high > 0 else w_high)
        best = value if best is None else max(best, value)
    return best


def _joint_line_patch(line, vertex, rect, surface, chart_point, tol, angular, work) -> bool:
    x = _ex(line.model.vertices[vertex].position)
    s_u, s_v = surface.derivatives_many(np.asarray([chart_point]))
    normal = np.cross(s_u[0], s_v[0])
    if not np.all(np.isfinite(normal)) or not np.any(normal):
        return False
    at_start = line.end_param(vertex) == 0.0
    away = line.tangent(0.0) if at_start else -line.tangent(0.0)
    w = _ex(normal if float(np.dot(normal, away)) > 0 else -normal)
    sign = line.monotone(0.0, 1.0, w)
    if sign is None or (sign if at_start else -sign) != 1:
        return False
    # the patch's reach towards the line's side: exact (n . S - n . X over the patch), not its box
    intrusion = _plane_function_max(surface, rect, w, _dot(w, x))
    if intrusion is None:
        return False
    norm_w = _norm_up(w)
    if intrusion > tol * norm_w:
        return False                                            # refine the patch first
    limit = max(intrusion, Fraction(0)) + tol * norm_w
    radius = limit / norm_w / Fraction(math.sin(angular)) + tol
    stretch = _stretch_box(line, 0.0, 1.0, at_start, x, w, limit, 1, work)
    return stretch is not None and _within(stretch, x, radius * radius)


# ------------------------------------------------------------------ public classifiers


def edge_edge(model, first_id: int, second_id: int, tolerance: float, work: Work | None = None,
              angular: float = 1e-10):
    """Edge pairs with a section edge or a straight chart segment (two chart segments on one support
    stay with A3b's exact chart certificate: ``None`` here means "not this route")."""
    work = work or Work()
    first, second = _curve_of(model, first_id), _curve_of(model, second_id)
    if first is None or second is None:
        return None
    shared = sorted({first.start, first.end} & {second.start, second.end})
    try:
        with _charging(work):
            if isinstance(first, _Section) and isinstance(second, _Section):
                c1, c2 = first.curve, second.curve
                if (c1.plane_record, c1.surface, c1.path) == (c2.plane_record, c2.surface, c2.path):
                    low_a, high_a = sorted(c1.domain)
                    low_b, high_b = sorted(c2.domain)
                    if min(high_a, high_b) > max(low_a, low_b):
                        return None                             # overlapping pieces of one path
            return True if _curves_clean(first, second, shared, tolerance, angular, work) else None
    except PlaneNurbsNotCertified:
        return None


def edge_face(model, edge_id: int, face_id: int, tolerance: float, work: Work | None = None,
              angular: float = 1e-10):
    """A section edge, straight line or straight chart segment against a plane face; a straight line
    against a NURBS face. ``None`` also when the pair is out of scope (the caller refuses)."""
    from .surfaces import Plane
    from .nurbs import NurbsSurface
    work = work or Work()
    edge, face = model.edges[edge_id], model.faces[face_id]
    curve = _curve_of(model, edge_id)
    if curve is None:
        return None
    face_vertices = {model.oriented_start_vertex(item) for loop in (face.loop,) + tuple(face.holes) for item in loop}
    shared = sorted({edge.start, edge.end} & face_vertices)
    try:
        with _charging(work):
            if isinstance(face.surface, Plane):
                if isinstance(curve, _Section):
                    from .section_faces import _same_plane
                    if _same_plane(face.surface, curve.curve):
                        # a section edge lying in the face's plane (not on its boundary): outside the face
                        # (B1-P5b). The face's own section edges must be other pieces of the same section
                        # path (a P1 path is simple: they meet only at shared domain ends).
                        own = (curve.curve.plane_record, curve.curve.surface, curve.curve.path)
                        low, high = sorted(curve.curve.domain)
                        for loop in (face.loop,) + tuple(face.holes):
                            for item in loop:
                                other = model.edges[item.edge].curve
                                if not isinstance(other, PlaneNurbsCurve):
                                    continue
                                if (other.plane_record, other.surface, other.path) != own:
                                    return None
                                other_low, other_high = sorted(other.domain)
                                if min(high, other_high) > max(low, other_low):
                                    return None
                        return True if _outside_plane_face(model, face, curve.curve, tolerance, work) else None
                return True if _curve_plane_clean(curve, face.surface, shared, tolerance, angular, work) else None
            if isinstance(face.surface, NurbsSurface) and isinstance(curve, _Line):
                return True if _line_face_clean(curve, model, face, shared, tolerance, angular, work) else None
    except PlaneNurbsNotCertified:
        return None
    return None


def vertex_face(model, vertex_id: int, face_id: int, tolerance: float, work: Work | None = None):
    """A vertex against a NURBS face: certified farther than the tolerance from the face's support over
    the face's chart box (exact sub-patch hulls), else ``None``."""
    work = work or Work()
    face = model.faces[face_id]
    surface = face.surface
    point = _ex(model.vertices[vertex_id].position)
    tol2 = Fraction(float(tolerance)) ** 2
    net = surface.exact_net()
    pending = [(_face_chart_box(model, face), 0)]
    while pending:
        rect, depth = pending.pop()
        if not work.take():
            return None
        lower, upper, _sigma = surface.patch_bounds(rect, net)
        gap = sum(max(lo - x, x - hi, Fraction(0)) ** 2 for lo, hi, x in zip(lower, upper, point))
        if gap > tol2:
            continue
        if depth >= work.max_depth:
            return None
        work.subdivisions += 1
        (u0, u1), (v0, v1) = rect
        um, vm = .5 * (u0 + u1), .5 * (v0 + v1)
        for us in ((u0, um), (um, u1)):
            for vs in ((v0, vm), (vm, v1)):
                pending.append(((us, vs), depth + 1))
    return True


def _paths(plane, surface):
    try:
        section = plane_nurbs_section(plane, surface)
    except PlaneNurbsNotCertified:
        return None
    if section.coincident:
        return None
    try:
        return tuple(PlaneNurbsCurve.from_plane(plane, surface, k) for k in range(len(section.paths)))
    except PlaneNurbsNotCertified:
        return None


def face_face(model, first_id: int, second_id: int, tolerance: float, work: Work | None = None):
    """A plane face against a NURBS face (owner option b): with shared section edges, every path of their
    exact section must be covered by them (their exact intersection is the shared edges; the tolerance
    band along them is not certified yet); without, every surface patch of the NURBS face's chart box
    must be farther than the tolerance from the plane."""
    from .surfaces import Plane
    from .nurbs import NurbsSurface
    from .section_faces import _cross, _exact, _same_plane
    work = work or Work()
    first, second = model.faces[first_id], model.faces[second_id]
    if not isinstance(first.surface, Plane):
        first, second = second, first
    plane, surface = first.surface, second.surface
    if not isinstance(plane, Plane) or not isinstance(surface, NurbsSurface):
        return None
    edges_first = {item.edge for loop in (first.loop,) + tuple(first.holes) for item in loop}
    edges_second = {item.edge for loop in (second.loop,) + tuple(second.holes) for item in loop}
    shared = [model.edges[e].curve for e in edges_first & edges_second
              if isinstance(model.edges[e].curve, PlaneNurbsCurve)
              and model.edges[e].curve.surface == surface and _same_plane(plane, model.edges[e].curve)]
    if len({curve.plane_record for curve in shared}) > 1:
        return None
    try:
        with _charging(work):
            paths = _paths(shared[0].plane if shared else plane, surface)
            if paths is None:
                return None
            for path in paths:
                covering = sorted(sorted(curve.domain) for curve in shared
                                  if curve.plane_record == path.plane_record and curve.path == path.path)
                uncovered, reach = [], 0.0
                for low, high in covering:
                    if low > reach:
                        uncovered.append((reach, low))
                    reach = max(reach, high)
                if reach < 1.0:
                    uncovered.append((reach, 1.0))
                # parts of the section outside the shared edges must lie outside the plane face (B1-P5b:
                # the plane cuts the whole support, the faces only between their trims)
                for low, high in uncovered:
                    if not _outside_plane_face(model, first, path.subcurve(low, high), tolerance, work):
                        return None
            if shared:
                return True                     # option b: the exact intersection is the shared edges
            n = _cross(_exact(plane.u_vector), _exact(plane.v_vector))
            d = _dot(n, _exact(plane.origin))
            # nothing shared: certified farther than the tolerance (no patch may approach the plane)
            return True if _surface_band_clean(surface, _face_chart_box(model, second), n, d, tolerance, work,
                                               allow_crossing=False) else None
    except PlaneNurbsNotCertified:
        return None


def _outside_plane_face(model, face, piece, tolerance, work) -> bool:
    """A section piece lying in a plane face's plane is outside the face (up to its end points): it meets
    the face's straight boundary edges only within the tolerance of their end vertices (P4a line
    certificate, the piece's ends are where it leaves the shared edges), and one interior point is outside
    the outer loop by ray parity (P4b). Other boundary kinds refuse."""
    from .branch_predicates import certify_line_section
    from .section_faces import _Frame, _inside
    ends = [np.asarray(point, dtype=float) for point in piece.evaluate(np.array([0.0, 1.0]))]
    for loop in (face.loop,) + tuple(face.holes):
        for item in loop:
            edge = model.edges[item.edge]
            if isinstance(edge.curve, PlaneNurbsCurve):
                continue                                        # the shared section edges themselves
            if not isinstance(edge.curve, Straight):
                return False
            start, end = model.vertices[edge.start].position, model.vertices[edge.end].position
            joints = [p for p in (start, end) if any(float(np.linalg.norm(p - q)) <= tolerance for q in ends)]
            if certify_line_section(piece, start, end, joints, tolerance) is not True:
                return False
    frame = _Frame(face.surface, model.vertices[model.oriented_start_vertex(face.loop[0])].position)
    middle = piece.evaluate(.5)
    return _inside(model, face.loop, middle, frame, tolerance) is False


def plane_faces(model, first_id: int, second_id: int, tolerance: float, work: Work | None = None,
                angular: float = 1e-10):
    """Two plane faces (B1-P5b; at least one with section edges): every boundary edge of each face that is
    not shared with the other is certified against the other's plane (strict contract: departure at shared
    vertices, else farther than the tolerance). A planar region whose non-shared boundary keeps off the
    other plane meets it only along the shared edges, so the faces' intersection is their shared edges
    (option-b reading for face pairs: the band along a shared edge follows from the edges' departure)."""
    from .section_faces import _same_plane, _cross, _exact
    from .surfaces import Plane
    work = work or Work()
    first, second = model.faces[first_id], model.faces[second_id]
    if not (isinstance(first.surface, Plane) and isinstance(second.surface, Plane)):
        return None
    n1 = _cross(_exact(first.surface.u_vector), _exact(first.surface.v_vector))
    n2 = _cross(_exact(second.surface.u_vector), _exact(second.surface.v_vector))
    if not any(_cross(n1, n2)):
        # parallel planes: separated by more than the tolerance, or coplanar (refused)
        d1 = _dot(n1, _exact(first.surface.origin))
        d2 = _dot(n1, _exact(second.surface.origin))
        gap = abs(d1 - d2)
        return True if gap * gap > Fraction(float(tolerance)) ** 2 * _dot(n1, n1) else None
    edges = [{item.edge for loop in (face.loop,) + tuple(face.holes) for item in loop} for face in (first, second)]
    vertices = [{model.oriented_start_vertex(item) for loop in (face.loop,) + tuple(face.holes) for item in loop}
                for face in (first, second)]
    try:
        with _charging(work):
            for index, (own, other_face) in enumerate(((first, second), (second, first))):
                for edge_id in edges[index] - edges[1 - index]:
                    curve = _curve_of(model, edge_id)
                    if curve is None:
                        return None
                    edge = model.edges[edge_id]
                    shared = sorted({edge.start, edge.end} & vertices[1 - index])
                    if isinstance(curve, _Section) and _same_plane(other_face.surface, curve.curve):
                        return None
                    if not _curve_plane_clean(curve, other_face.surface, shared, tolerance, angular, work):
                        return None
    except PlaneNurbsNotCertified:
        return None
    return True


def _surface_band_clean(surface, chart_box, n, d, tolerance, work, allow_crossing: bool = True) -> bool:
    from .branch_predicates import _derivative_ranges, _multiply, _patches, _range, _rect, _subtract
    g_patches = _patches(surface, tuple(n), d)
    w_patches = _patches(surface, (Fraction(0),) * 3, Fraction(-1))
    (r0, r1), (q0, q1) = (tuple(Fraction(v) for v in rng) for rng in (surface.u_range, surface.v_range))
    norm_n = _norm_up(n)
    tol = Fraction(float(tolerance))
    pending = [(chart_box, 0)]
    while pending:
        rect, depth = pending.pop()
        if not work.take():
            return False
        knot_u = sorted(r0 + Fraction(c) * (r1 - r0) for c in rect[0])
        knot_v = sorted(q0 + Fraction(c) * (q1 - q0) for c in rect[1])
        decided = True
        for g_patch, w_patch in zip(g_patches, w_patches):
            u_lo, u_hi = max(knot_u[0], g_patch.u_range[0]), min(knot_u[1], g_patch.u_range[1])
            v_lo, v_hi = max(knot_v[0], g_patch.v_range[0]), min(knot_v[1], g_patch.v_range[1])
            if u_lo >= u_hi or v_lo >= v_hi:
                continue
            g = _rect(g_patch.coefficients, g_patch.u_range, g_patch.v_range, (u_lo, u_hi), (v_lo, v_hi))
            ww = _rect(w_patch.coefficients, w_patch.u_range, w_patch.v_range, (u_lo, u_hi), (v_lo, v_hi))
            g_low, g_high = _range(g)
            w_high = _range(ww)[1]
            if g_low > tol * norm_n * w_high or g_high < -tol * norm_n * w_high:
                continue                                        # farther than the tolerance from the plane
            corners = [g[0][0], g[0][-1], g[-1][0], g[-1][-1]]  # exact values at the sub-rectangle corners
            crossed = min(corners) < 0 < max(corners)
            (gu, gv), (wu, wv) = (_derivative_ranges(c, u_hi - u_lo, v_hi - v_lo) for c in (g, ww))
            transversal = False
            for dg, dw in ((gu, wu), (gv, wv)):
                total = _subtract(_multiply(dg, _range(ww)), _multiply(_range(g), dw))
                if total[0] > 0 or total[1] < 0:
                    transversal = True
            if not (allow_crossing and crossed and transversal):
                decided = False
                break
        if decided:
            continue
        if depth >= work.max_depth:
            return False
        work.subdivisions += 1
        (u0, u1), (v0, v1) = rect
        um, vm = .5 * (u0 + u1), .5 * (v0 + v1)
        for us in ((u0, um), (um, u1)):
            for vs in ((v0, vm), (vm, v1)):
                pending.append(((us, vs), depth + 1))
    return True
