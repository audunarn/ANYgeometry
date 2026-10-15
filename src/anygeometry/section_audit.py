"""Certified strict-audit classifications around plane x NURBS sections (generality slice B1-P4d).

The generic audit routes accepted these pairs without a certificate (a shared boundary curve
assumed to be the only contact of a plane face and a NURBS face; sampled curve routes for
section edges). Here every pair is decided by the exact machinery of P1/P4a/P4c or refused:

- ``edge_edge``: a section edge against a straight line (``certify_line_section``), a straight
  chart segment on its surface (``segment_section`` in the chart) or another section edge (one
  path: exact domains; otherwise separated enclosures);
- ``edge_face``: a section edge against a plane face (its roots on the face's plane all within
  the tolerance of shared vertices); a straight chart segment (``CurveOnSurface``) against a
  plane face (the plane's section paths on the segment's support, met only at shared chart
  points); a straight line against a NURBS face (the section paths of a plane through the line
  meet the line only at shared vertices);
- ``vertex_face``: a vertex against a NURBS face (exact sub-patch hulls of the face's chart box
  farther than the tolerance);
- ``face_face``: a plane face against a NURBS face: every path of their exact section (P1) is
  covered by shared section edges (same plane, surface and path; their domains cover it).

Each returns ``True`` (certified: no contact beyond the shared topology), ``False`` (a certified
contact) or ``None`` (not certified). Contacts are judged against the whole support (not only
the trimmed region): conservative.

Convention (review B1-P4d round 1): a pair WITHOUT shared topology must be certified farther apart
than the tolerance (exact world boxes: ``_far_from_plane``, ``_far_from_segment``; an empty exact
intersection is not enough); a pair WITH shared vertices or edges has its exact contacts only at
the shared topology (the existing straight routes' rule for joints). All subdivision counts
against ``Work`` (the audit's qualification budget, reported back).
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from .curves import Straight
from .plane_nurbs import PlaneNurbsNotCertified, plane_nurbs_section
from .support_curves import PlaneNurbsCurve


class Work:
    """The audit's qualification budget for one pair, and the work consumed (reported back)."""

    def __init__(self, max_boxes: int = 200_000, max_depth: int = 32):
        self.max_boxes, self.max_depth = int(max_boxes), int(max_depth)
        self.boxes = self.subdivisions = 0

    def take(self) -> bool:
        self.boxes += 1
        return self.boxes <= self.max_boxes


def _exact3(vector):
    return [Fraction(float(value)) for value in vector]


def _box_plane_far(lower, upper, n, d, nn, tol2) -> bool:
    values = [sum(a * Fraction(c) for a, c in zip(n, (lower[0] if i & 1 == 0 else upper[0],
                                                       lower[1] if i & 2 == 0 else upper[1],
                                                       lower[2] if i & 4 == 0 else upper[2]))) - d
              for i in range(8)]
    low, high = min(values), max(values)
    gap = low if low > 0 else -high if high < 0 else Fraction(0)
    return gap > 0 and gap * gap > tol2 * nn


def _segment_frame(start, end):
    a, b = _exact3(start), _exact3(end)
    e = [y - x for x, y in zip(a, b)]
    k = min(range(3), key=lambda i: abs(e[i]))
    axis = [Fraction(1) if i == k else Fraction(0) for i in range(3)]
    n1 = [e[1] * axis[2] - e[2] * axis[1], e[2] * axis[0] - e[0] * axis[2], e[0] * axis[1] - e[1] * axis[0]]
    n2 = [e[1] * n1[2] - e[2] * n1[1], e[2] * n1[0] - e[0] * n1[2], e[0] * n1[1] - e[1] * n1[0]]
    return a, e, n1, n2


def _box_segment_far(lower, upper, frame, tol2) -> bool:
    """Exact: every point of the box is farther than the tolerance from the segment (an orthogonal
    frame ``e, n1, n2``: |v|^2 >= sum of the squared axis gaps)."""
    a, e, n1, n2 = frame
    corners = [(Fraction(lower[0] if i & 1 == 0 else upper[0]) - a[0],
                Fraction(lower[1] if i & 2 == 0 else upper[1]) - a[1],
                Fraction(lower[2] if i & 4 == 0 else upper[2]) - a[2]) for i in range(8)]
    total = Fraction(0)
    for axis, segment_range in ((e, None), (n1, (Fraction(0), Fraction(0))), (n2, (Fraction(0), Fraction(0)))):
        norm2 = sum(x * x for x in axis)
        if norm2 == 0:
            return False
        values = [sum(x * y for x, y in zip(axis, c)) for c in corners]
        lo, hi = (Fraction(0), norm2) if segment_range is None else segment_range
        gap = max(min(values) - hi, lo - max(values), Fraction(0))
        total += gap * gap / norm2
    return total > tol2


def _surface_boxes_far(surface, chart_box, far, work: Work) -> bool:
    """Exact sub-patch hulls of ``surface`` over ``chart_box`` all satisfy ``far(lower, upper)``."""
    net = surface.exact_net()
    pending = [(chart_box[0], chart_box[1], 0)]
    while pending:
        u_range, v_range, depth = pending.pop()
        if not work.take():
            return False
        lower, upper, _sigma = surface.patch_bounds((u_range, v_range), net)
        if far(lower, upper):
            continue
        if depth >= work.max_depth:
            return False
        work.subdivisions += 1
        um, vm = .5 * (u_range[0] + u_range[1]), .5 * (v_range[0] + v_range[1])
        for us in ((u_range[0], um), (um, u_range[1])):
            for vs in ((v_range[0], vm), (vm, v_range[1])):
                pending.append((us, vs, depth + 1))
    return True


def _curve_pieces_far(curve, far, work: Work) -> bool:
    """Enclosures ``curve.bounds(t0, t1)`` of a section curve's pieces all satisfy ``far``."""
    pending = [(0.0, 1.0, 0)]
    while pending:
        lo, hi, depth = pending.pop()
        if not work.take():
            return False
        lower, upper = curve.bounds(lo, hi)
        if far(lower, upper):
            continue
        if depth >= work.max_depth:
            return False
        work.subdivisions += 1
        middle = .5 * (lo + hi)
        pending.extend(((lo, middle, depth + 1), (middle, hi, depth + 1)))
    return True


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


def _plane_exact(plane):
    from .section_faces import _cross, _exact
    n = _cross(_exact(plane.u_vector), _exact(plane.v_vector))
    return n, sum(x * y for x, y in zip(n, _exact(plane.origin))), sum(x * x for x in n)


def _plane_of(face):
    from .surfaces import Plane
    return face.surface if isinstance(face.surface, Plane) else None


def _nurbs_of(face):
    from .nurbs import NurbsSurface
    return face.surface if isinstance(face.surface, NurbsSurface) else None


def _shared_positions(model, first_vertices, second_vertices):
    return [model.vertices[v].position for v in set(first_vertices) & set(second_vertices)]


def _face_vertices(model, face):
    return {model.oriented_start_vertex(item) for loop in (face.loop,) + tuple(face.holes) for item in loop}


def _segment_pcurve(curve):
    from .nurbs import CurveOnSurface, _IDENTITY
    return (isinstance(curve, CurveOnSurface) and curve.transform == _IDENTITY
            and curve.pcurve.degree == 1 and len(curve.pcurve.controls) == 2)


def _paths(plane, surface):
    """The section curves of every path of ``plane x surface`` (``()`` when disjoint), or ``None``."""
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


def _combine(verdicts):
    verdicts = list(verdicts)
    if any(v is False for v in verdicts):
        return False
    return True if all(v is True for v in verdicts) else None


# ------------------------------------------------------------------ edge x edge


def edge_edge(model, first_id: int, second_id: int, tolerance: float, work: Work | None = None):
    from .branch_predicates import certify_line_section
    work = work or Work()
    a, b = model.edges[first_id], model.edges[second_id]
    if not isinstance(a.curve, PlaneNurbsCurve):
        a, b = b, a
    shared_ids = {a.start, a.end} & {b.start, b.end}
    shared = [model.vertices[v].position for v in shared_ids]
    curve = a.curve
    other = b.curve
    tol2 = Fraction(float(tolerance)) ** 2
    try:
        if isinstance(other, Straight):
            start, end = model.vertices[b.start].position, model.vertices[b.end].position
            if not shared_ids:                                  # no shared topology: separation
                frame = _segment_frame(start, end)
                return True if _curve_pieces_far(curve, lambda lo, hi: _box_segment_far(lo, hi, frame, tol2),
                                                 work) else None
            return certify_line_section(curve, start, end, shared, tolerance)
        if _segment_pcurve(other) and other.support == curve.surface:
            from .nurbs_section_faces import _chart_tolerance, segment_section
            p0, p1 = other.pcurve.exact_point(0), other.pcurve.exact_point(1)
            chart_shared = [p for v, p in ((b.start, p0), (b.end, p1)) if v in shared_ids]
            return segment_section(curve, p0, p1, chart_shared, _chart_tolerance(curve.surface, tolerance))
        if isinstance(other, PlaneNurbsCurve):
            if (curve.plane_record, curve.surface, curve.path) == (other.plane_record, other.surface, other.path):
                low_a, high_a = sorted(curve.domain)
                low_b, high_b = sorted(other.domain)
                if min(high_a, high_b) > max(low_a, low_b):
                    return False                                # overlapping pieces of one path
            # world simplicity of the two together (disjoint parameters do not separate a path that
            # crosses itself in the world, review B1-P4d round 1): P4b's piece certificate
            from .entities import OrientedEdge
            from .section_faces import _sections_simple
            loose, _pieces = _sections_simple(model, ((OrientedEdge(a.id, True), OrientedEdge(b.id, True)),),
                                              tolerance)
            return True if not loose else None
    except PlaneNurbsNotCertified:
        return None
    return None


# ------------------------------------------------------------------ edge x face


def edge_face(model, edge_id: int, face_id: int, tolerance: float, work: Work | None = None):
    """``None`` also when the pair is outside this module's scope (the caller then refuses)."""
    from .branch_predicates import certify_line_section, section_roots
    work = work or Work()
    edge, face = model.edges[edge_id], model.faces[face_id]
    plane, surface = _plane_of(face), _nurbs_of(face)
    shared_ids = {edge.start, edge.end} & _face_vertices(model, face)
    shared = [model.vertices[v].position for v in shared_ids]
    tol2 = Fraction(float(tolerance)) ** 2
    if not shared_ids:
        # no shared topology: certified separation (an empty exact intersection is not enough)
        try:
            if plane is not None and isinstance(edge.curve, PlaneNurbsCurve):
                n, d, nn = _plane_exact(plane)
                return True if _curve_pieces_far(edge.curve, lambda lo, hi: _box_plane_far(lo, hi, n, d, nn, tol2),
                                                 work) else None
            if plane is not None and _segment_pcurve(edge.curve):
                n, d, nn = _plane_exact(plane)
                p0, p1 = edge.curve.pcurve.exact_point(0), edge.curve.pcurve.exact_point(1)
                box = ((float(min(p0[0], p1[0])), float(max(p0[0], p1[0]))),
                       (float(min(p0[1], p1[1])), float(max(p0[1], p1[1]))))
                if box[0][0] == box[0][1] or box[1][0] == box[1][1]:          # iso segment: a thin chart box
                    return True if _surface_boxes_far(edge.curve.support, box,
                                                      lambda lo, hi: _box_plane_far(lo, hi, n, d, nn, tol2),
                                                      work) else None
                return None
            if surface is not None and isinstance(edge.curve, Straight):
                frame = _segment_frame(model.vertices[edge.start].position, model.vertices[edge.end].position)
                return True if _surface_boxes_far(surface, _face_chart_box(model, face),
                                                  lambda lo, hi: _box_segment_far(lo, hi, frame, tol2),
                                                  work) else None
        except PlaneNurbsNotCertified:
            return None
        return None
    try:
        if isinstance(edge.curve, PlaneNurbsCurve) and plane is not None:
            from .section_faces import _cross, _exact
            n = _cross(_exact(plane.u_vector), _exact(plane.v_vector))
            d = sum(x * y for x, y in zip(n, _exact(plane.origin)))
            roots = section_roots(edge.curve, n, d)
            for root in roots:
                corners = np.array([[root.lower[k] if (i >> k) & 1 == 0 else root.upper[k] for k in range(3)]
                                    for i in range(8)])
                if not any(float(np.max(np.linalg.norm(corners - p, axis=1))) <= tolerance for p in shared):
                    return None                                 # a contact with the plane (maybe outside the face)
            return True
        if _segment_pcurve(edge.curve) and plane is not None:
            from .nurbs_section_faces import _chart_tolerance, segment_section
            paths = _paths(plane, edge.curve.support)
            if paths is None:
                return None
            p0, p1 = edge.curve.pcurve.exact_point(0), edge.curve.pcurve.exact_point(1)
            chart_shared = [p for v, p in ((edge.start, p0), (edge.end, p1)) if v in shared_ids]
            chart_tol = _chart_tolerance(edge.curve.support, tolerance)
            return _combine(segment_section(path, p0, p1, chart_shared, chart_tol) for path in paths)
        if isinstance(edge.curve, Straight) and surface is not None:
            from .surfaces import Plane
            start, end = model.vertices[edge.start].position, model.vertices[edge.end].position
            direction = np.asarray(end, dtype=float) - start
            axis = np.zeros(3)
            axis[int(np.argmin(np.abs(direction)))] = 1.0
            through = Plane(tuple(start), tuple(direction), tuple(axis))
            paths = _paths(through, surface)
            if paths is None:
                return None
            return _combine(certify_line_section(path, start, end, shared, tolerance) for path in paths)
    except PlaneNurbsNotCertified:
        return None
    return None


# ------------------------------------------------------------------ vertex x face


def vertex_face(model, vertex_id: int, face_id: int, tolerance: float, work: Work | None = None):
    """A vertex against a NURBS face: certified farther than the tolerance from the face's support over
    the face's chart box (exact sub-patch hulls), else ``None``."""
    work = work or Work()
    face = model.faces[face_id]
    surface = _nurbs_of(face)
    if surface is None:
        return None
    lows, highs = [], []
    for loop in (face.loop,) + tuple(face.holes):
        for item in loop:
            curve = model.edges[item.edge].curve
            low, high = curve.chart_bounds() if isinstance(curve, PlaneNurbsCurve) else curve.pcurve.bounds()
            lows.append(low)
            highs.append(high)
    low = np.clip(np.min(lows, axis=0), 0.0, 1.0)
    high = np.clip(np.max(highs, axis=0), 0.0, 1.0)
    point = [Fraction(float(x)) for x in model.vertices[vertex_id].position]
    tol2 = Fraction(float(tolerance)) ** 2
    net = surface.exact_net()
    pending = [((float(low[0]), float(high[0])), (float(low[1]), float(high[1])), 0)]
    examined = 0
    while pending:
        u_range, v_range, depth = pending.pop()
        examined += 1
        if not work.take():
            return None
        lower, upper, _sigma = surface.patch_bounds((u_range, v_range), net)
        gap = sum(max(lo - x, x - hi, Fraction(0)) ** 2 for lo, hi, x in zip(lower, upper, point))
        if gap > tol2:
            continue
        size = sum((hi - lo) ** 2 for lo, hi in zip(lower, upper))
        if size * 64 <= tol2 or depth >= work.max_depth:
            return None                                         # within (about) the tolerance of the support
        work.subdivisions += 1
        um, vm = .5 * (u_range[0] + u_range[1]), .5 * (v_range[0] + v_range[1])
        for us in ((u_range[0], um), (um, u_range[1])):
            for vs in ((v_range[0], vm), (vm, v_range[1])):
                pending.append((us, vs, depth + 1))
    return True


# ------------------------------------------------------------------ face x face


def face_face(model, first_id: int, second_id: int, tolerance: float, work: Work | None = None):
    """A plane face against a NURBS face: every path of their section covered by shared section edges."""
    first, second = model.faces[first_id], model.faces[second_id]
    if _plane_of(first) is None:
        first, second = second, first
    plane, surface = _plane_of(first), _nurbs_of(second)
    if plane is None or surface is None:
        return None
    from .section_faces import _same_plane
    edges_first = {item.edge for loop in (first.loop,) + tuple(first.holes) for item in loop}
    edges_second = {item.edge for loop in (second.loop,) + tuple(second.holes) for item in loop}
    shared = [model.edges[e].curve for e in edges_first & edges_second
              if isinstance(model.edges[e].curve, PlaneNurbsCurve)
              and model.edges[e].curve.surface == surface and _same_plane(plane, model.edges[e].curve)]
    records = {curve.plane_record for curve in shared}
    if len(records) > 1:
        return None                                             # one plane in several parameterisations
    # the section computed with the shared curves' own plane record: path indices then match exactly
    paths = _paths(shared[0].plane if shared else plane, surface)
    if paths is None:
        return None
    if not paths:
        # no exact intersection: certified farther than the tolerance (review B1-P4d round 1)
        if shared:
            return None
        n, d, nn = _plane_exact(plane)
        tol2 = Fraction(float(tolerance)) ** 2
        return True if _surface_boxes_far(surface, _face_chart_box(model, second),
                                          lambda lo, hi: _box_plane_far(lo, hi, n, d, nn, tol2),
                                          work or Work()) else None
    for path in paths:
        covering = sorted(sorted(curve.domain) for curve in shared
                          if curve.plane_record == path.plane_record and curve.path == path.path)
        reach = 0.0
        for low, high in covering:
            if low > reach:
                return None                                     # a part of the section is not a shared edge
            reach = max(reach, high)
        if reach < 1.0:
            return None
    return True
