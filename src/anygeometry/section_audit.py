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
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from .curves import Straight
from .plane_nurbs import PlaneNurbsNotCertified, plane_nurbs_section
from .support_curves import PlaneNurbsCurve


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


def edge_edge(model, first_id: int, second_id: int, tolerance: float):
    from .branch_predicates import certify_line_section
    a, b = model.edges[first_id], model.edges[second_id]
    if not isinstance(a.curve, PlaneNurbsCurve):
        a, b = b, a
    shared_ids = {a.start, a.end} & {b.start, b.end}
    shared = [model.vertices[v].position for v in shared_ids]
    curve = a.curve
    other = b.curve
    try:
        if isinstance(other, Straight):
            return certify_line_section(curve, model.vertices[b.start].position, model.vertices[b.end].position,
                                        shared, tolerance)
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
                    return False
                if high_a == low_b or high_b == low_a:
                    return True if shared_ids else None
                return True
            (la, ua), (lb, ub) = curve.bounds(), other.bounds()
            gap = np.maximum(np.maximum(la - ub, lb - ua), 0.0)
            return True if float(np.linalg.norm(gap)) > tolerance else None
    except PlaneNurbsNotCertified:
        return None
    return None


# ------------------------------------------------------------------ edge x face


def edge_face(model, edge_id: int, face_id: int, tolerance: float):
    """``None`` also when the pair is outside this module's scope (the caller then refuses)."""
    from .branch_predicates import certify_line_section, section_roots
    edge, face = model.edges[edge_id], model.faces[face_id]
    plane, surface = _plane_of(face), _nurbs_of(face)
    shared_ids = {edge.start, edge.end} & _face_vertices(model, face)
    shared = [model.vertices[v].position for v in shared_ids]
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


def vertex_face(model, vertex_id: int, face_id: int, tolerance: float, budget: int = 4096):
    """A vertex against a NURBS face: certified farther than the tolerance from the face's support over
    the face's chart box (exact sub-patch hulls), else ``None``."""
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
        if examined > budget:
            return None
        lower, upper, _sigma = surface.patch_bounds((u_range, v_range), net)
        gap = sum(max(lo - x, x - hi, Fraction(0)) ** 2 for lo, hi, x in zip(lower, upper, point))
        if gap > tol2:
            continue
        size = sum((hi - lo) ** 2 for lo, hi in zip(lower, upper))
        if size * 64 <= tol2 or depth >= 30:
            return None                                         # within (about) the tolerance of the support
        um, vm = .5 * (u_range[0] + u_range[1]), .5 * (v_range[0] + v_range[1])
        for us in ((u_range[0], um), (um, u_range[1])):
            for vs in ((v_range[0], vm), (vm, v_range[1])):
                pending.append((us, vs, depth + 1))
    return True


# ------------------------------------------------------------------ face x face


def face_face(model, first_id: int, second_id: int, tolerance: float):
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
        return True                                             # the plane misses the support
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
