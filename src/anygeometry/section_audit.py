"""Certified strict-audit classifications around plane x NURBS sections (generality slice B1-P4d).

Contract (owner decision 2026-10-15, option a: strict uniform tolerance). Two entities are clean
when every pair of their points closer than the tolerance lies in the *departure stretch* of a
joint they share (a shared vertex, or a shared section edge for two faces); everywhere else they
are certified farther apart than the tolerance. At a joint ``X`` both sides must leave along a
separating direction ``w``: each side's piece at ``X`` is certified strictly monotone in ``w . P``
(a plane face lies at a constant offset, a NURBS face patch within a small intrusion), on opposite
sides. With ``E`` = the joint-end offsets + ``tolerance |w|``, a point beyond the offset plane
``w . (P - X) = +-E`` of its side is farther than the tolerance from the whole other side, so
contacts can only lie in the stretches before those planes; each stretch must lie within the joint
ball of ``JOINT_BALL`` tolerances around ``X`` (review B1-P5 round 3: a radius scaled by
``1 / sin(angular tolerance)`` admitted contacts 50 tolerances out), so flat departures refuse.

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

from .branch_predicates import JOINT_BALL
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


class _Piece(_Section):
    """A section piece that is not a model edge (an uncovered part of a section path), with the model
    vertices at its ends (``< 0``: none) for the joint logic."""

    def __init__(self, model, curve, start, end):
        self.edge_id, self.start, self.end, self.curve, self.model = None, start, end, curve, model


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
    still be at most ``limit`` (bisection: the rest is certified beyond ``limit``). Callers certify
    ``side * w . P`` strictly increasing away from the joint on ``curve[a, b]``, so the enclosure of
    the cut point alone decides the rest (B1-P5 round 3: tight cuts for the fixed joint ball)."""
    lo, hi = (a, b) if at_start else (b, a)                    # lo: the joint end
    near, far = lo, hi
    for _ in range(work.max_depth):
        if not work.take():
            return None
        middle = .5 * (near + far)
        low, high = _offsets(curve.box(middle, middle), w, x)
        beyond = (low > limit) if side > 0 else (high < -limit)
        if beyond:
            far = middle
        else:
            near = middle
        if abs(far - near) <= 1e-300:
            break
    end = far
    beyond_low, beyond_high = _offsets(curve.box(end, end), w, x)
    if not ((beyond_low > limit) if side > 0 else (beyond_high < -limit)) and end != hi:
        return None
    return curve.box(min(lo, end), max(lo, end))


class _JointTooLong(PlaneNurbsNotCertified):
    """A joint's departure stretch ends inside the piece yet leaves the joint ball: the departure is too
    flat, which smaller pieces cannot change; the pair is refused at once (review B1-P5 round 3: retrying
    on subdivided spans ran the audit's whole budget)."""


def _joint_within(curve, a, b, at_start, x, w, limit, side, work, radius, final: bool = True) -> bool:
    """The departure stretch of ``curve[a, b]`` within the joint ball; ``False`` to retry on smaller
    pieces, ``_JointTooLong`` when the stretch ends inside the piece outside the ball and ``limit`` cannot
    shrink with smaller pieces (``final``)."""
    box = _stretch_box(curve, a, b, at_start, x, w, limit, side, work)
    if box is None:
        return False
    if _within(box, x, radius * radius):
        return True
    hi = b if at_start else a
    low, high = _offsets(curve.box(hi, hi), w, x)
    if final and ((low > limit) if side > 0 else (high < -limit)):
        raise _JointTooLong("a joint's departure stretch leaves the joint ball")
    return False


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
    radius = JOINT_BALL * Fraction(float(tolerance))
    for (curve, a, b, at_start, _away), side in zip(pieces, sides):
        if not _joint_within(curve, a, b, at_start, x, wx, limit, side, work, radius):
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
    radius = JOINT_BALL * Fraction(tol)
    return _joint_within(curve, a, b, at_start, x, w, limit, 1, work, radius)


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
    radius = JOINT_BALL * Fraction(tol)
    # the limit includes the patch's intrusion, which a smaller patch may lower: retry, not refuse
    return _joint_within(line, 0.0, 1.0, at_start, x, w, limit, 1, work, radius, final=False)


# ------------------------------------------------------------------ public classifiers
# Contract (a) for face pairs too (owner 2026-10-15, P4e): points of two faces closer than the tolerance
# must lie in the joint band (within JOINT_BALL tolerances of a shared vertex or edge).


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
    """A plane face against a NURBS face, contract (a) with the joint band (owner 2026-10-15, P4e-3).

    With shared section edges: (1) every path of their exact section is covered by them, or lies outside
    the plane face with a clearance of the band radius plus the tolerance, ``(JOINT_BALL + 1) tol``,
    except in joint departure stretches at the shared corners; (2) the band (``_descent_band_clean``): every
    point of the NURBS face within the tolerance of the plane reaches the exact section along a certified
    monotone chart line within ``JOINT_BALL tol``, so its points within the tolerance of the plane face lie
    within the band of the shared edges (the section elsewhere being cleared by (1); near a shared corner
    the clearance's own joint stretches reach JOINT_BALL (JOINT_BALL + 1) tolerances). Without shared edges every surface patch of the NURBS face's chart
    box must be farther than the tolerance from the plane."""
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
                    if not _outside_plane_face(model, first, path.subcurve(low, high),
                                               (JOINT_BALL + 1) * tolerance, work):
                        return None
            if shared:
                n = _cross(_exact(plane.u_vector), _exact(plane.v_vector))
                d = _dot(n, _exact(plane.origin))
                return True if _descent_band_clean(surface, _face_chart_box(model, second), n, d, tolerance,
                                                   work) else None
            n = _cross(_exact(plane.u_vector), _exact(plane.v_vector))
            d = _dot(n, _exact(plane.origin))
            # nothing shared: certified farther than the tolerance (no patch may approach the plane)
            return True if _surface_band_clean(surface, _face_chart_box(model, second), n, d, tolerance, work,
                                               allow_crossing=False) else None
    except PlaneNurbsNotCertified:
        return None


def _outside_plane_face(model, face, piece, tolerance, work) -> bool:
    """A section piece lying in a plane face's plane is outside the face except in the joint balls at its
    ends (contract (a) for face pairs, P4e-2): it comes within the tolerance of the face's straight
    boundary edges only in certified joint departure stretches (P4a line certificate with the joint ball
    of ``JOINT_BALL`` tolerances; the piece's ends are where it leaves the shared edges), so away from
    those balls it is wholly inside or outside, and one interior point is outside the outer loop by ray
    parity (P4b). Other boundary kinds refuse. (A piece entering the face within a corner's ball, e.g. by
    a sub-tolerance offset of the corner, is a joint contact.)"""
    from .section_faces import _Frame, _inside
    ends = [np.asarray(point, dtype=float) for point in piece.evaluate(np.array([0.0, 1.0]))]
    # the piece as a curve whose ends are the face's vertices it leaves from (if any), for the
    # distance-based pair engine (review P4e round 1: an intersection certificate is not a clearance)
    face_vertices = {model.oriented_start_vertex(item) for loop in (face.loop,) + tuple(face.holes)
                     for item in loop}

    def end_vertex(point):
        near = [v for v in face_vertices
                if float(np.linalg.norm(model.vertices[v].position - point)) <= float(tolerance)]
        return near[0] if len(near) == 1 else -1 - len(near)
    curve = _Piece(model, piece, end_vertex(ends[0]), end_vertex(ends[1]))
    own = (piece.plane_record, piece.surface, piece.path)
    low, high = sorted(piece.domain)
    for loop in (face.loop,) + tuple(face.holes):
        for item in loop:
            edge = model.edges[item.edge]
            if isinstance(edge.curve, PlaneNurbsCurve):
                # skipped only when it is another piece of the same section path, disjoint in its domain
                # (a P1 path is simple): any other section boundary refuses (review B1-P5 round 1)
                other = edge.curve
                other_low, other_high = sorted(other.domain)
                if (other.plane_record, other.surface, other.path) != own or \
                        min(high, other_high) > max(low, other_low):
                    return False
                continue
            if not isinstance(edge.curve, Straight):
                return False
            line = _Line(model, item.edge)
            shared = sorted({line.start, line.end} & {curve.start, curve.end})
            if not _curves_clean(curve, line, shared, tolerance, 1e-10, work):
                return False
    frame = _Frame(face.surface, model.vertices[model.oriented_start_vertex(face.loop[0])].position)
    middle = piece.evaluate(.5)
    return _inside(model, face.loop, middle, frame, tolerance) is False


def plane_faces(model, first_id: int, second_id: int, tolerance: float, work: Work | None = None,
                angular: float = 1e-10):
    """Two plane faces (B1-P5b; at least one with section edges), contract (a) with the joint band (owner
    2026-10-15, P4e-1). Every boundary edge of each face that is not shared with the other is certified
    against the other's plane (departure at shared vertices within the joint ball, else farther than the
    tolerance). Then each face meets the other plane's tolerance slab, away from those joint stretches,
    exactly in its inside stretches of the planes' line L times the slab's strip (its boundary does not
    enter the slab there), so the faces' points closer than the tolerance lie where their stretches come
    within the tolerance of each other. Those places must lie in the joint band: a shared edge's span or a
    shared vertex's ball (near L), widened by ``JOINT_BALL`` tolerances; along a shared edge the
    dihedral angle must keep the band within the ball (``sin theta >= 1 / JOINT_BALL``, exact)."""
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
    # each face's inside stretches of L come from where its boundary actually crosses the other plane
    # (exact line parameters, certified section roots; uncertain root boxes count as inside)
    direction = np.asarray([float(v) for v in _cross(n1, n2)], dtype=float)
    direction /= float(np.linalg.norm(direction))
    slack = 0.0                     # no coordinate slack (review P4e round 1: it deleted a 4e-8 intersection)
    tol = float(tolerance)
    reach = JOINT_BALL * tol
    shared_edges = edges[0] & edges[1]
    if shared_edges:
        # the band along a shared edge: points within the tolerance of the other face lie within
        # tol / sin(theta) of the edge; within the ball when sin(theta) >= 1 / JOINT_BALL
        cross = _cross(n1, n2)
        if _dot(cross, cross) * JOINT_BALL ** 2 < _dot(n1, n1) * _dot(n2, n2):
            return None
    anchor = _line_anchor(first, second, direction)
    if anchor is None:
        return None
    joints = []                                                 # (s_lo, s_hi): the joint band along L
    for edge_id in shared_edges:
        edge = model.edges[edge_id]
        s0, s1 = (float(direction @ model.vertices[v].position) for v in (edge.start, edge.end))
        joints.append((min(s0, s1) - reach, max(s0, s1) + reach))
    for vertex in vertices[0] & vertices[1]:
        position = model.vertices[vertex].position
        offset = float(np.linalg.norm((position - anchor) - float((position - anchor) @ direction) * direction))
        if offset < reach:                                      # the vertex's ball meets L in a chord
            s, half = float(direction @ position), float(np.sqrt(reach * reach - offset * offset))
            joints.append((s - half, s + half))
    insides = []
    for own, other in ((first, second), (second, first)):
        stretches = _inside_stretches(model, own, other, direction, tolerance, slack,
                                      joints)
        if stretches is None:
            return None
        insides.append(stretches)
    merged = []
    for lo, hi in sorted(joints):
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    for lo_a, hi_a in insides[0]:
        for lo_b, hi_b in insides[1]:
            lo, hi = max(lo_a, lo_b) - tol, min(hi_a, hi_b) + tol
            if hi < lo:
                continue                                        # farther apart than the tolerance along L
            if not any(j_lo <= lo and hi <= j_hi for j_lo, j_hi in merged):
                return None                                     # contact outside the joint band
    return True


def _inside_stretches(model, face, other, direction, tolerance, slack, shared_spans=()):
    """The stretches ``(s0, s1)`` (``s = direction . X``) of the planes' intersection line inside ``face``:
    the line's crossings with the face's boundary (straight edges: exact parameters; section edges:
    certified roots on the other plane, an unresolved root's box as an uncertain stretch taken as
    inside; an edge lying on the line contributes its ends), the stretches between consecutive crossings
    classified by ray parity at their midpoints; ``None`` if undecided. Stretches no longer than
    ``slack`` (rounding) are dropped.
    Stretches within ``shared_spans`` (the joint band, where contact is allowed) count as inside
    unclassified: their midpoints may lie within the tolerance of the boundary."""
    from .branch_predicates import section_roots
    from .section_faces import _Frame, _inside
    n = _cross(_ex(other.surface.u_vector), _ex(other.surface.v_vector))
    d = _dot(n, _ex(other.surface.origin))
    values, uncertain = [], []
    for loop in (face.loop,) + tuple(face.holes):
        for item in loop:
            edge = model.edges[item.edge]
            if isinstance(edge.curve, Straight):
                a, b = _ex(model.vertices[edge.start].position), _ex(model.vertices[edge.end].position)
                fa, fb = _dot(n, a) - d, _dot(n, b) - d
                if fa == 0 and fb == 0:
                    values += [float(direction @ model.vertices[v].position) for v in (edge.start, edge.end)]
                elif (fa <= 0 <= fb) or (fb <= 0 <= fa):
                    r = fa / (fa - fb)
                    point = [x + r * (y - x) for x, y in zip(a, b)]
                    values.append(float(direction @ np.asarray([float(v) for v in point])))
            elif isinstance(edge.curve, PlaneNurbsCurve):
                try:
                    roots = section_roots(edge.curve, n, d)
                except PlaneNurbsNotCertified:
                    return None
                for root in roots:
                    centre = .5 * (np.asarray(root.lower) + np.asarray(root.upper))
                    if root.isolated:
                        # the crossing lies somewhere in the root box: a box wider than the slack along
                        # the line is an uncertain stretch, taken as inside (review B1-P5 round 3)
                        box = np.array([[root.lower[k] if (i >> k) & 1 == 0 else root.upper[k] for k in range(3)]
                                        for i in range(8)])
                        reach = box @ direction
                        if float(reach.max() - reach.min()) > slack:
                            uncertain.append((float(reach.min()), float(reach.max())))
                            values += [float(reach.min()), float(reach.max())]
                            continue
                    else:
                        # an unresolved root (e.g. a corner on the other plane): the crossing lies in its
                        # box; the box's stretch of L is uncertain and taken as inside (P4e-1)
                        box = np.array([[root.lower[k] if (i >> k) & 1 == 0 else root.upper[k] for k in range(3)]
                                        for i in range(8)])
                        reach = box @ direction
                        uncertain.append((float(reach.min()), float(reach.max())))
                        values += [float(reach.min()), float(reach.max())]
                        continue
                    values.append(float(direction @ centre))
            else:
                return None
    values = sorted(values)
    frame = _Frame(face.surface, model.vertices[model.oriented_start_vertex(face.loop[0])].position)
    # a point of the line at parameter s: on both planes; anchor from any crossing point
    anchor = _line_anchor(face, other, direction)
    if anchor is None:
        return None
    stretches = []
    for s0, s1 in zip(values, values[1:]):
        if s1 - s0 <= slack:
            continue
        if any(lo <= s0 and s1 <= hi for lo, hi in list(shared_spans) + uncertain):
            stretches.append((s0, s1))
            continue
        middle = anchor + (.5 * (s0 + s1) - float(direction @ anchor)) * direction
        inside = _inside(model, face.loop, middle, frame, tolerance)
        if inside is None:
            return None
        if inside:
            stretches.append((s0, s1))
    return stretches


def _line_anchor(first, second, direction):
    """A point on both planes (least squares of the two plane equations and ``direction . X = 0``)."""
    rows, rhs = [], []
    for face in (first, second):
        normal = np.cross(face.surface.u_vector, face.surface.v_vector)
        rows.append(normal)
        rhs.append(float(normal @ np.asarray(face.surface.origin, dtype=float)))
    rows.append(direction)
    rhs.append(0.0)
    try:
        return np.linalg.solve(np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float))
    except np.linalg.LinAlgError:
        return None


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _descent_band_clean(surface, chart_box, n, d, tolerance, work) -> bool:
    """The band certificate for a NURBS face against the plane ``n . X = d`` (P4e-3, review P4e round 1).

    The face's chart box is cut into knot rectangles. Each is farther than the tolerance from the plane,
    or, along one chart axis ``s``, the plane distance ``dist = G / (|n| W)`` is certified monotone (the
    exact interval of ``G_s W - G W_s`` has one strict sign on every Bezier piece, so across knots too:
    ``dist`` is continuous) and the plane is crossed: the rectangle's two ``s`` edges have strictly
    opposite signs of ``G``, or the rectangle is extended along ``s`` (towards decreasing ``|G|``, within
    the surface's chart, else refused) until its far edge has the opposite sign. On that region the
    world speed bound ``M >= |dS/ds|`` and the slope bound ``m <= |d dist / ds|`` give ``M / m <=
    JOINT_BALL`` (exact). Then every point of the face within the tolerance of the plane reaches the
    exact section along its ``s``-line within ``JOINT_BALL`` tolerances, inside that region (no descent
    through a knot crease or out of the chart). Subdivision to the work budget; refused otherwise."""
    from .branch_predicates import _derivative_ranges, _multiply, _patches, _range, _rect, _subtract
    zero = Fraction(0)
    g_patches = _patches(surface, tuple(n), d)
    w_patches = _patches(surface, (zero,) * 3, Fraction(-1))
    x_patches = [_patches(surface, tuple(Fraction(int(i == k)) for i in range(3)), zero) for k in range(3)]
    norm_n2, norm_up = _dot(n, n), _norm_up(n)
    tol = Fraction(float(tolerance))
    j2 = Fraction(JOINT_BALL) ** 2
    limits = [tuple(sorted(Fraction(v) for v in surface.u_range)), tuple(sorted(Fraction(v) for v in surface.v_range))]

    def chart_to_knot(axis, value):
        a, b = (Fraction(v) for v in (surface.u_range if axis == 0 else surface.v_range))
        return a + Fraction(float(value)) * (b - a)

    def pieces(ranges):
        for index, patch in enumerate(g_patches):
            sub = []
            for axis, own in enumerate((patch.u_range, patch.v_range)):
                low, high = max(ranges[axis][0], own[0]), min(ranges[axis][1], own[1])
                sub.append((low, high))
            if sub[0][0] < sub[0][1] and sub[1][0] < sub[1][1]:
                yield index, sub

    def restricted(patches, index, sub):
        patch = patches[index]
        return _rect(patch.coefficients, patch.u_range, patch.v_range, sub[0], sub[1])

    def far(ranges) -> bool:
        for index, sub in pieces(ranges):
            g_low, g_high = _range(restricted(g_patches, index, sub))
            w_high = _range(restricted(w_patches, index, sub))[1]
            if not (g_low > tol * norm_up * w_high or g_high < -tol * norm_up * w_high):
                return False
        return True

    def strict_sign(ranges) -> int:
        signs = set()
        for index, sub in pieces(ranges):
            g_low, g_high = _range(restricted(g_patches, index, sub))
            signs.add(1 if g_low > 0 else -1 if g_high < 0 else 0)
        return signs.pop() if len(signs) == 1 else 0

    def edge_sign(ranges, axis, high_end) -> int:
        line = ranges[axis][1] if high_end else ranges[axis][0]
        signs = set()
        for index, sub in pieces(ranges):
            if sub[axis][1 if high_end else 0] != line:
                continue
            g = restricted(g_patches, index, sub)
            if axis == 0:
                values = list(g[-1] if high_end else g[0])
            else:
                values = [row[-1] if high_end else row[0] for row in g]
            signs.add(1 if all(v > 0 for v in values) else -1 if all(v < 0 for v in values) else 0)
        return signs.pop() if len(signs) == 1 else 0

    def monotone(ranges, axis):
        """The strict sign of d dist / ds on the region if certified, with M / m <= JOINT_BALL; else None."""
        signs, m_low, s2_high, w_low, w_high = set(), None, zero, None, None
        for index, sub in pieces(ranges):
            g, ww = restricted(g_patches, index, sub), restricted(w_patches, index, sub)
            du, dv = sub[0][1] - sub[0][0], sub[1][1] - sub[1][0]
            g_range, w_range = _range(g), _range(ww)
            g_s = _derivative_ranges(g, du, dv)[axis]
            w_s = _derivative_ranges(ww, du, dv)[axis]
            total = _subtract(_multiply(g_s, w_range), _multiply(g_range, w_s))
            if total[0] > 0:
                signs.add(1)
            elif total[1] < 0:
                signs.add(-1)
            else:
                return None
            low = min(abs(total[0]), abs(total[1]))
            m_low = low if m_low is None else min(m_low, low)
            s2 = zero
            for k in range(3):
                xc = restricted(x_patches[k], index, sub)
                x_s = _derivative_ranges(xc, du, dv)[axis]
                speed = _subtract(_multiply(x_s, w_range), _multiply(_range(xc), w_s))
                s2 += max(speed[0] * speed[0], speed[1] * speed[1])
            s2_high = max(s2_high, s2)
            w_low = w_range[0] if w_low is None else min(w_low, w_range[0])
            w_high = w_range[1] if w_high is None else max(w_high, w_range[1])
        if len(signs) != 1 or not m_low or not w_low or w_low <= 0:
            return None
        # M / m = sqrt(s2) |n| w_high^2 / (w_low^2 m_low) <= JOINT_BALL
        if s2_high * norm_n2 * w_high ** 4 > j2 * w_low ** 4 * m_low * m_low:
            return None
        return signs.pop()

    def certified(ranges) -> bool:
        if far(ranges):
            return True
        for axis in (0, 1):
            slope = monotone(ranges, axis)
            if slope is None:
                continue
            low_sign, high_sign = edge_sign(ranges, axis, False), edge_sign(ranges, axis, True)
            if low_sign and high_sign and low_sign == -high_sign:
                return True                                     # crossed inside the rectangle
            sign = strict_sign(ranges)
            if not sign:
                continue
            towards_high = slope * sign < 0                     # |G| decreases towards +s
            width = ranges[axis][1] - ranges[axis][0]
            extension = width
            for _ in range(48):
                if not work.take():
                    return False
                grown = list(ranges)
                if towards_high:
                    grown[axis] = (ranges[axis][0], min(limits[axis][1], ranges[axis][1] + extension))
                else:
                    grown[axis] = (max(limits[axis][0], ranges[axis][0] - extension), ranges[axis][1])
                if monotone(grown, axis) != slope:
                    break
                if edge_sign(grown, axis, towards_high) == -sign:
                    return True                                 # the section is reached in the region
                if grown[axis] == (limits[axis][0], ranges[axis][1]) or grown[axis] == (ranges[axis][0], limits[axis][1]):
                    break                                       # descent would leave the chart
                extension *= 2
        return False

    box = [tuple(sorted(chart_to_knot(axis, value) for value in chart_box[axis])) for axis in (0, 1)]
    pending = [(sub, 0) for _index, sub in pieces(box)]
    while pending:
        ranges, depth = pending.pop()
        if not work.take():
            return False
        if certified(ranges):
            continue
        if depth >= work.max_depth:
            return False
        work.subdivisions += 1
        (u0, u1), (v0, v1) = ranges
        um, vm = (u0 + u1) / 2, (v0 + v1) / 2
        for us in ((u0, um), (um, u1)):
            for vs in ((v0, vm), (vm, v1)):
                pending.append(([us, vs], depth + 1))
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
