"""Plane faces trimmed by straight lines and plane x NURBS section curves (generality slice B1-P4b).

A plane face whose loops contain ``PlaneNurbsCurve`` edges is validated without sampled
trims (architecture 2.6, "no approximate trims"):

- two-support rule (3.3): every section curve's plane is the face's plane (exact: parallel
  normals and the curve plane's origin on the face plane, in rationals); other curved
  edge kinds are refused on such a face in this slice;
- vertices within the surface residual of the plane (as straight plane faces);
- pairwise trim certificates: straight x straight by the existing planar segment test;
  straight x section by ``certify_line_section`` (P4a, intersections within the length
  tolerance of shared vertices only); section x section on the same section path by
  their exact parameter domains, on different supports by separated bounds (else not
  certified);
- every loop's signed area as a Green line integral (Gauss-Legendre along the section
  arcs, exact shoelace on lines) with a rigorous bound from the curve's enclosures
  (``loop_area``); a loop whose area is not certified away from zero is refused;
- holes inside the outer loop and outside each other by ray parity, the crossings of a
  section curve counted with ``section_roots`` (P4a) on the plane through the ray.

Anything not certified is reported (the face is refused).
"""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np

from .curves import Straight
from .errors import GeometryError
from .plane_nurbs import PlaneNurbsNotCertified
from .support_curves import PlaneNurbsCurve

_PIECES = 32                        # polyline pieces per section edge for the area bound
_GAUSS = np.polynomial.legendre.leggauss(24)
_RAYS = 8


def has_section_edges(model, face) -> bool:
    return any(isinstance(model.edges[item.edge].curve, PlaneNurbsCurve)
               for loop in (face.loop,) + tuple(face.holes) for item in loop)


def _exact(vector):
    return [Fraction(float(value)) for value in vector]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _same_plane(plane, curve: PlaneNurbsCurve) -> bool:
    """Exact: the curve's plane is the face's plane (parallel normals, the curve's origin on it)."""
    n = _cross(_exact(plane.u_vector), _exact(plane.v_vector))
    origin, u, v = (_exact(vector) for vector in curve.plane_record)
    m = _cross(u, v)
    if any(_cross(n, m)) or not any(n):
        return False
    return sum(a * (b - c) for a, b, c in zip(n, origin, _exact(plane.origin))) == 0


class _Frame:
    """Orthonormal in-plane coordinates (floats) of the face's plane."""

    def __init__(self, plane):
        normal = np.cross(plane.u_vector, plane.v_vector)
        self.normal = normal / float(np.linalg.norm(normal))
        self.first = np.asarray(plane.u_vector, dtype=float) / float(np.linalg.norm(plane.u_vector))
        self.second = np.cross(self.normal, self.first)
        self.origin = np.asarray(plane.origin, dtype=float)

    def xy(self, points):
        offset = np.asarray(points, dtype=float) - self.origin
        return np.stack((offset @ self.first, offset @ self.second), axis=-1)


def _edge_points(model, oriented):
    edge = model.edges[oriented.edge]
    start, end = model.vertices[edge.start].position, model.vertices[edge.end].position
    return (start, end) if oriented.forward else (end, start)


def _section_area(curve: PlaneNurbsCurve, frame: _Frame, forward: bool, pieces: int = _PIECES):
    """``(value, bound)`` of ``1/2 oint (x dy - y dx)`` along a section edge in its loop direction.

    ``value``: Gauss-Legendre per arc piece (the curve is smooth between breakpoints).
    ``bound``: rigorous for the region between the curve and its ``_PIECES``-chord polyline,
    measured against the polyline area: each piece lies in its enclosure ``bounds(t_k, t_k+1)``
    (as does its chord), so their difference is at most that box's in-plane area; the polyline
    value comes with it so the caller can use the polyline, and the Gauss value is checked
    against the polyline within the bound.
    """
    knots = sorted({0.0, 1.0, *curve.breakpoints()})
    nodes, weights = _GAUSS
    value = 0.0
    for lo, hi in zip(knots, knots[1:]):
        t = .5 * (hi - lo) * (nodes + 1.) + lo
        points = frame.xy(curve.evaluate(t))
        derivatives = np.stack([(curve.derivative(float(item)) @ frame.first,
                                 curve.derivative(float(item)) @ frame.second) for item in t])
        integrand = points[:, 0] * derivatives[:, 1] - points[:, 1] * derivatives[:, 0]
        value += .5 * .5 * (hi - lo) * float(weights @ integrand)
    ts = np.linspace(0.0, 1.0, pieces + 1)
    polyline = frame.xy(curve.evaluate(ts))
    chord = .5 * float(np.sum(polyline[:-1, 0] * polyline[1:, 1] - polyline[1:, 0] * polyline[:-1, 1]))
    bound = 0.0
    for lo, hi in zip(ts, ts[1:]):
        lower, upper = curve.bounds(float(lo), float(hi))
        corners = frame.xy(np.array([[lower[k] if (index >> k) & 1 == 0 else upper[k] for k in range(3)]
                                     for index in range(8)]))
        width = np.ptp(corners, axis=0)
        bound += float(width[0] * width[1])
    if not forward:
        value, chord = -value, -chord
    return value, chord, bound


def loop_area(model, loop, plane, pieces: int = _PIECES) -> tuple[float, float]:
    """``(area, bound)``: the signed area of a plane-face loop (positive anticlockwise about the
    plane's ``u x v``) and a bound on its error (section pieces: enclosure areas; floats: a
    relative allowance)."""
    frame = _Frame(plane)
    value, polygon_value, bound = 0.0, 0.0, 0.0
    for item in loop:
        curve = model.edges[item.edge].curve
        if isinstance(curve, PlaneNurbsCurve):
            gauss, chord, piece_bound = _section_area(curve, frame, item.forward, pieces)
            value += gauss
            polygon_value += chord
            bound += piece_bound
        else:
            start, end = frame.xy(np.asarray(_edge_points(model, item)))
            term = .5 * float(start[0] * end[1] - end[0] * start[1])
            value += term
            polygon_value += term
    if abs(value - polygon_value) > bound * (1 + 1e-9) + 1e-12 * (abs(value) + 1):
        raise GeometryError("plane face loop area quadrature disagrees with its certified polyline bound")
    return value, bound + 64 * np.finfo(float).eps * (abs(value) + 1)


def _pair_verdict(model, first, second, tolerance):
    """True: the two edges meet only within tolerance of their shared vertices; False: they
    certainly meet elsewhere; None: not certified."""
    a, b = model.edges[first.edge], model.edges[second.edge]
    shared_ids = {a.start, a.end} & {b.start, b.end}
    shared = [model.vertices[vertex].position for vertex in shared_ids]
    if isinstance(a.curve, PlaneNurbsCurve) and isinstance(b.curve, PlaneNurbsCurve):
        if (a.curve.plane_record, a.curve.surface, a.curve.path) == \
                (b.curve.plane_record, b.curve.surface, b.curve.path):
            low_a, high_a = sorted(a.curve.domain)
            low_b, high_b = sorted(b.curve.domain)
            if min(high_a, high_b) > max(low_a, low_b):
                return False                                    # overlapping pieces of one section path
            if high_a == low_b or high_b == low_a:
                return True if shared_ids else None            # touching ends must be a shared vertex
            return True                                         # disjoint pieces of one simple path
        (la, ua), (lb, ub) = a.curve.bounds(), b.curve.bounds()
        gap = np.maximum(np.maximum(la - ub, lb - ua), 0.0)
        return True if float(np.linalg.norm(gap)) > tolerance else None
    if isinstance(a.curve, PlaneNurbsCurve) or isinstance(b.curve, PlaneNurbsCurve):
        curve_edge, line_edge = (a, b) if isinstance(a.curve, PlaneNurbsCurve) else (b, a)
        start, end = (model.vertices[line_edge.start].position, model.vertices[line_edge.end].position)
        return certify_line_section_safe(curve_edge.curve, start, end, shared, tolerance)
    return None


def certify_line_section_safe(curve, start, end, shared, tolerance):
    from .branch_predicates import certify_line_section
    try:
        return certify_line_section(curve, start, end, shared, tolerance)
    except PlaneNurbsNotCertified:
        return None


def _crossings(model, loop, point, direction, frame, tolerance):
    """Crossings of the ray ``point + r direction`` (r > 0, in the plane) with a loop, or ``None``
    when this ray is not decidable (through a vertex, tangent, an unresolved crossing)."""
    normal = frame.normal
    across = np.cross(normal, direction)                        # the ray's plane normal (in-plane)
    count = 0
    for item in loop:
        edge = model.edges[item.edge]
        if isinstance(edge.curve, PlaneNurbsCurve):
            from .branch_predicates import section_roots
            n = _exact(across)
            d = sum(x * y for x, y in zip(n, _exact(point)))
            try:
                roots = section_roots(edge.curve, n, d)
            except PlaneNurbsNotCertified:
                return None
            for root in roots:
                corners = np.array([[root.lower[k] if (index >> k) & 1 == 0 else root.upper[k] for k in range(3)]
                                    for index in range(8)])
                along = (corners - point) @ direction
                if np.all(along < -tolerance):
                    continue
                if not (np.all(along > tolerance) and root.isolated):
                    return None
                count += 1
        else:
            start, end = (np.asarray(p, dtype=float) for p in _edge_points(model, item))
            side_start, side_end = float((start - point) @ across), float((end - point) @ across)
            if abs(side_start) <= tolerance or abs(side_end) <= tolerance:
                if float((start - point) @ direction) > -tolerance or float((end - point) @ direction) > -tolerance:
                    return None                                 # the ray passes a vertex
                continue
            if (side_start > 0) != (side_end > 0):
                r = side_start / (side_start - side_end)
                crossing = start + r * (end - start)
                along = float((crossing - point) @ direction)
                if abs(along) <= tolerance:
                    return None
                if along > 0:
                    count += 1
    return count


def _inside(model, loop, point, frame, tolerance):
    """Ray parity over a fixed sequence of directions; ``None`` when no ray decides."""
    for k in range(_RAYS):
        angle = .7390851332151607 + k * math.pi * (math.sqrt(5.) - 1.)
        direction = math.cos(angle) * frame.first + math.sin(angle) * frame.second
        count = _crossings(model, loop, np.asarray(point, dtype=float), direction, frame, tolerance)
        if count is not None:
            return count % 2 == 1
    return None


def validate_plane_section_face(model, face_id: int) -> list[str]:
    """Validation messages of a plane face trimmed by lines and section curves (empty: valid)."""
    face = model.faces[face_id]
    plane = face.surface
    loops = (face.loop,) + tuple(face.holes)
    result: list[str] = []
    for loop in loops:
        for item in loop:
            curve = model.edges[item.edge].curve
            if isinstance(curve, PlaneNurbsCurve):
                if not _same_plane(plane, curve):
                    result.append(f"face {face_id} edge {item.edge} is a section curve of another plane")
            elif not isinstance(curve, Straight):
                result.append(f"face {face_id} edge {item.edge}: only straight and plane x NURBS section edges "
                              "may trim a plane face with section edges (B1-P4b)")
    if result:
        return result
    frame = _Frame(plane)
    vertices = np.asarray([model.vertices[model.oriented_start_vertex(item)].position
                           for loop in loops for item in loop], dtype=float)
    lows, highs = [vertices.min(axis=0)], [vertices.max(axis=0)]
    for loop in loops:
        for item in loop:
            curve = model.edges[item.edge].curve
            if isinstance(curve, PlaneNurbsCurve):
                low, high = curve.bounds()
                lows.append(low)
                highs.append(high)
    extent = float(np.linalg.norm(np.max(highs, axis=0) - np.min(lows, axis=0)))
    residual = model.tolerance.effective_surface_residual(extent)
    if float(np.max(np.abs((vertices - frame.origin) @ frame.normal), initial=0.0)) > residual:
        return [f"face {face_id} boundary is inconsistent with its explicit surface"]
    tolerance = model.tolerance.effective_length(extent)
    # pairwise trim certificates (adjacent edges share their joint vertex)
    items = [(index, position, item) for index, loop in enumerate(loops) for position, item in enumerate(loop)]
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (loop_a, pos_a, first), (loop_b, pos_b, second) = items[i], items[j]
            a, b = model.edges[first.edge], model.edges[second.edge]
            if isinstance(a.curve, Straight) and isinstance(b.curve, Straight):
                count = len(loops[loop_a])
                adjacent = loop_a == loop_b and (abs(pos_a - pos_b) == 1 or abs(pos_a - pos_b) == count - 1)
                if adjacent:
                    continue
                p, q = frame.xy(np.asarray(_edge_points(model, first)))
                r, s = frame.xy(np.asarray(_edge_points(model, second)))
                if model._segments_intersect_2d(p, q, r, s, tolerance):  # noqa: SLF001
                    verdict = False
                else:
                    continue
            else:
                verdict = _pair_verdict(model, first, second, tolerance)
            if verdict is True:
                continue
            where = (f"loop {loop_a}" if loop_a == loop_b else f"loops {loop_a} and {loop_b}")
            if verdict is False:
                result.append(f"face {face_id} {where} self-intersects (edges {first.edge}, {second.edge})"
                              if loop_a == loop_b else
                              f"face {face_id} {where} intersect (edges {first.edge}, {second.edge})")
            else:
                result.append(f"face {face_id} {where}: edges {first.edge} and {second.edge} not certified disjoint")
    if result:
        return result
    for index, loop in enumerate(loops):
        # the bound shrinks with the pieces (enclosure areas ~ L**2 / pieces): refined while undecided
        for pieces in (_PIECES, 4 * _PIECES, 16 * _PIECES, 64 * _PIECES):
            try:
                area, bound = loop_area(model, loop, plane, pieces)
            except GeometryError as error:
                result.append(f"face {face_id} loop {index}: {error}")
                break
            if abs(area) > bound + model.tolerance.effective_area(extent):
                break
        else:
            result.append(f"face {face_id} loop {index} has zero area")
    for index in range(1, len(loops)):
        point = model.vertices[model.oriented_start_vertex(loops[index][0])].position
        inside = _inside(model, loops[0], point, frame, tolerance)
        if inside is not True:
            result.append(f"face {face_id} hole {index} is not strictly inside the outer loop"
                          if inside is False else f"face {face_id} hole {index} containment not certified")
        for other in range(1, len(loops)):
            if other != index and _inside(model, loops[other], point, frame, tolerance) is not False:
                result.append(f"face {face_id} holes {index} and {other} overlap")
    return result
