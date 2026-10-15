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
    """Orthonormal in-plane coordinates (floats) of the face's plane, about a local ``origin``
    (a loop vertex: far plane origins cancelled the area, review B1-P4b round 1)."""

    def __init__(self, plane, origin=None):
        normal = np.cross(plane.u_vector, plane.v_vector)
        self.normal = normal / float(np.linalg.norm(normal))
        self.first = np.asarray(plane.u_vector, dtype=float) / float(np.linalg.norm(plane.u_vector))
        self.second = np.cross(self.normal, self.first)
        self.origin = np.asarray(plane.origin if origin is None else origin, dtype=float)

    def xy(self, points):
        offset = np.asarray(points, dtype=float) - self.origin
        return np.stack((offset @ self.first, offset @ self.second), axis=-1)


def _edge_points(model, oriented):
    edge = model.edges[oriented.edge]
    start, end = model.vertices[edge.start].position, model.vertices[edge.end].position
    return (start, end) if oriented.forward else (end, start)


def _section_area(curve: PlaneNurbsCurve, frame: _Frame, forward: bool, pieces=_PIECES):
    """``(gauss, chord, bound, magnitude)`` of ``1/2 oint (x dy - y dx)`` along a section edge in its
    loop direction.

    ``gauss``: Gauss-Legendre per arc piece (the curve is smooth between breakpoints). ``chord``: the
    polyline through the piece ends. ``pieces``: a count (uniform pieces) or the parameter pieces;
    for a rigorous ``bound`` they must be certified monotone in some direction (``_sections_simple``):
    a monotone piece and its chord are two graphs over that direction, so the region between them is
    covered once and lies in the piece's enclosure ``bounds(t_k, t_k+1)``: at most that box's in-plane
    area (review B1-P4b round 2: containment alone does not bound winding). The defined points lie
    within ``delta`` of C*: the chord length times ``2 delta`` is added. ``magnitude``: the sum of the
    absolute products (rounding scale).
    """
    knots = sorted({0.0, 1.0, *curve.breakpoints()})
    nodes, weights = _GAUSS
    gauss, magnitude = 0.0, 0.0
    for lo, hi in zip(knots, knots[1:]):
        t = .5 * (hi - lo) * (nodes + 1.) + lo
        points = frame.xy(curve.evaluate(t))
        derivatives = np.stack([(curve.derivative(float(item)) @ frame.first,
                                 curve.derivative(float(item)) @ frame.second) for item in t])
        first, second = points[:, 0] * derivatives[:, 1], points[:, 1] * derivatives[:, 0]
        gauss += .25 * (hi - lo) * float(weights @ (first - second))
        magnitude += .25 * (hi - lo) * float(weights @ (np.abs(first) + np.abs(second)))
    if isinstance(pieces, int):
        ts = np.linspace(0.0, 1.0, pieces + 1)
    else:
        ts = np.asarray([pieces[0][0]] + [b for _a, b in pieces], dtype=float)
    polyline = frame.xy(curve.evaluate(ts))
    products = (polyline[:-1, 0] * polyline[1:, 1], polyline[1:, 0] * polyline[:-1, 1])
    chord = .5 * float(np.sum(products[0] - products[1]))
    magnitude += .5 * float(np.sum(np.abs(products[0]) + np.abs(products[1])))
    bound = 0.0
    for lo, hi in zip(ts, ts[1:]):
        lower, upper = curve.bounds(float(lo), float(hi))
        corners = frame.xy(np.array([[lower[k] if (index >> k) & 1 == 0 else upper[k] for k in range(3)]
                                     for index in range(8)]))
        width = np.ptp(corners, axis=0) + 8 * np.finfo(float).eps * float(np.max(np.abs(corners)))
        bound += float(width[0] * width[1])
    delta = math.sqrt(float(curve._world_bound()[3]))
    bound += 2 * delta * float(np.sum(np.linalg.norm(np.diff(polyline, axis=0), axis=1)))
    if not forward:
        gauss, chord = -gauss, -chord
    return gauss, chord, bound, magnitude


def loop_area(model, loop, plane, pieces=_PIECES) -> tuple[float, float]:
    """``(area, bound)``: the signed area of a plane-face loop (positive anticlockwise about the
    plane's ``u x v``) and a rigorous bound of its error.

    In coordinates about the loop's first vertex. ``area`` is the Gauss value; with ``|true -
    chord| <= B`` (enclosures), ``|true - gauss| <= B + |chord - gauss|`` plus rounding, scaled by
    the magnitude of the summed products (review B1-P4b round 1).
    """
    frame = _Frame(plane, model.vertices[model.oriented_start_vertex(loop[0])].position)
    value, polygon_value, bound, magnitude = 0.0, 0.0, 0.0, 0.0
    for item in loop:
        curve = model.edges[item.edge].curve
        if isinstance(curve, PlaneNurbsCurve):
            spans = pieces if isinstance(pieces, int) else pieces[item.edge]
            gauss, chord, piece_bound, piece_magnitude = _section_area(curve, frame, item.forward, spans)
            value += gauss
            polygon_value += chord
            bound += piece_bound
            magnitude += piece_magnitude
        else:
            start, end = frame.xy(np.asarray(_edge_points(model, item)))
            first, second = float(start[0] * end[1]), float(end[0] * start[1])
            value += .5 * (first - second)
            polygon_value += .5 * (first - second)
            magnitude += .5 * (abs(first) + abs(second))
    eps = np.finfo(float).eps
    return value, bound * (1 + 16 * eps) + abs(value - polygon_value) + 64 * eps * magnitude


def _pair_verdict(model, first, second, tolerance, plane_tolerance=None):
    """True: the two edges meet only within tolerance of their shared vertices; False: they
    certainly meet elsewhere; None: not certified."""
    a, b = model.edges[first.edge], model.edges[second.edge]
    shared_ids = {a.start, a.end} & {b.start, b.end}
    shared = [model.vertices[vertex].position for vertex in shared_ids]
    if isinstance(a.curve, PlaneNurbsCurve) and isinstance(b.curve, PlaneNurbsCurve):
        return True                                             # certified by _sections_simple
    if isinstance(a.curve, PlaneNurbsCurve) or isinstance(b.curve, PlaneNurbsCurve):
        curve_edge, line_edge = (a, b) if isinstance(a.curve, PlaneNurbsCurve) else (b, a)
        start, end = (model.vertices[line_edge.start].position, model.vertices[line_edge.end].position)
        return certify_line_section_safe(curve_edge.curve, start, end, shared, tolerance, plane_tolerance)
    return None


def _box_gap(first, second) -> float:
    (la, ua), (lb, ub) = first, second
    return float(np.linalg.norm(np.maximum(np.maximum(np.asarray(la) - ub, np.asarray(lb) - ua), 0.0)))


def _joint_ok(model, p, p_start, q, q_start, point, w, tolerance) -> bool:
    """Neighbouring pieces ``p``, ``q`` (``(edge, a, b)``, joined at ``point``) meet only within ``tolerance``
    of it (review B1-P4b round 2: a crossing near the joint plus a far-end side allowed an excursion).

    Each piece is certified strictly monotone in ``w . X`` (``projection_monotone``: injective), the two
    moving to opposite sides away from the joint. With ``e`` bounding both pieces' offsets
    ``|w . (C(joint end) - point)|``, a piece's points beyond the offset plane ``w . (X - point) = +-e``
    (its side) cannot meet the other piece, whose points stay below ``e`` on that side; the stretch
    before it (up to the plane's single root) must lie within ``tolerance`` of the joint.
    """
    from .branch_predicates import projection_monotone, section_roots
    n = _exact(w)
    x = _exact(point)
    eps = np.finfo(float).eps
    sides, offsets, deltas = [], [], []
    for edge, a, b in (p, q):
        curve = model.edges[edge].curve
        piece = curve.subcurve(a, b)
        sign = projection_monotone(piece, n)
        if sign is None:
            return False
        sides.append(sign if (p_start if (edge, a, b) == p else q_start) else -sign)
        joint_t = a if (p_start if (edge, a, b) == p else q_start) else b
        # the offset of C* at the joint end, bounded exactly over the curve's certified enclosure there
        # (bounds(t, t) holds C* and the defined point): translation invariant, and tight where the
        # piece ends at an exact root (review B1-P4b round 3: a floor proportional to the absolute
        # coordinates refused valid faces far from the origin)
        deltas.append(math.sqrt(float(curve._world_bound()[3])))
        lower, upper = curve.bounds(joint_t, joint_t)
        offsets.append(max(abs(sum(u * (Fraction(float(lower[k] if (index >> k) & 1 == 0 else upper[k])) - v)
                                   for k, (u, v) in enumerate(zip(n, x))))
                           for index in range(8)))
    if sides[0] == sides[1]:
        return False
    e = sum(offsets) * (1 + Fraction(1, 2 ** 40))
    # the joint ball: the tolerance widened by the representation errors of both sides (the curves'
    # delta and the rounding of the joint point)
    radius = tolerance + 2 * max(deltas) + 2 * math.sqrt(3.0) * math.ulp(float(np.max(np.abs(point))) or 1e-300)
    for (edge, a, b), at_start, side in ((p, p_start, sides[0]), (q, q_start, sides[1])):
        curve = model.edges[edge].curve
        piece = curve.subcurve(a, b)
        d = sum(u * v for u, v in zip(n, x)) + side * e
        try:
            roots = section_roots(piece, n, d)
        except PlaneNurbsNotCertified:
            return False
        if len(roots) > 1:
            return False                                        # impossible for a monotone piece
        if roots:
            t_lo, t_hi = roots[0].t_lo, roots[0].t_hi
            near = (0.0, t_hi) if at_start else (t_lo, 1.0)
        else:
            near = (0.0, 1.0)                                   # the piece never leaves the joint's slab
        lo, hi = a + near[0] * (b - a), a + near[1] * (b - a)
        lower, upper = curve.bounds(min(lo, hi), max(lo, hi))
        corners = np.array([[lower[k] if (index >> k) & 1 == 0 else upper[k] for k in range(3)] for index in range(8)])
        if float(np.max(np.linalg.norm(corners - point, axis=1))) > radius:
            return False
    return True


def _sections_simple(model, loops, tolerance, max_level: int = 12):
    """Section edges whose world simplicity (as a set together, within ``tolerance`` of their joints)
    is not certified (review B1-P4b round 1: a simple chart path can cross itself in the world).

    The section edges are cut into pieces; pieces that are not neighbours must have enclosures more
    than the tolerance apart; neighbours (consecutive pieces of an edge, or pieces of two edges at a
    shared vertex) must pass ``_joint_ok`` with ``w = unit(away_q) - unit(away_p)`` (both strictly
    monotone in ``w``, on opposite sides). Every piece takes part in a joint, so every piece is
    certified injective. Undecided pieces are halved up to ``max_level`` times; the remaining ones
    are reported (with the certified pieces, for the area).
    """
    edges = sorted({item.edge for loop in loops for item in loop
                    if isinstance(model.edges[item.edge].curve, PlaneNurbsCurve)})
    if not edges:
        return [], {}
    pieces = {edge: [(0.0, .5), (.5, 1.0)] for edge in edges}
    loose: set = set()
    boxes: dict = {}                                            # memoised across levels: only new pieces cost
    verdicts: dict = {}
    from .plane_nurbs import charge_split, depth_limit
    max_level = depth_limit(max_level)
    for level in range(max_level + 1):
        flat = [(edge, a, b) for edge in edges for a, b in pieces[edge]]
        for key in flat:
            if key not in boxes:
                from .plane_nurbs import charge_box
                charge_box()
                boxes[key] = model.edges[key[0]].curve.bounds(key[1], key[2])
        ends: dict = {}                                         # vertex -> pieces touching it at an edge end
        for key in flat:
            edge = model.edges[key[0]]
            if key[1] == 0.0:
                ends.setdefault(edge.start, []).append((key, True))
            if key[2] == 1.0:
                ends.setdefault(edge.end, []).append((key, False))
        joints = []                                             # (p, p_at_start, q, q_at_start, point)
        for edge in edges:
            curve = model.edges[edge].curve
            for (a, b), (c, e) in zip(pieces[edge], pieces[edge][1:]):
                joints.append(((edge, a, b), False, (edge, c, e), True, curve.evaluate(b)))
        for vertex, touching in ends.items():
            for i in range(len(touching)):
                for j in range(i + 1, len(touching)):
                    (p, p_start), (q, q_start) = touching[i], touching[j]
                    if p[0] != q[0]:
                        joints.append((p, p_start, q, q_start, model.vertices[vertex].position))
        neighbours = {frozenset((p, q)) for p, _ps, q, _qs, _x in joints}
        loose = set()
        for i in range(len(flat)):
            for j in range(i + 1, len(flat)):
                if frozenset((flat[i], flat[j])) in neighbours:
                    continue
                if _box_gap(boxes[flat[i]], boxes[flat[j]]) <= tolerance:
                    loose.update((flat[i], flat[j]))
        for p, p_start, q, q_start, point in joints:
            if p in loose or q in loose:
                continue                                        # split anyway: decided after the split
            memo = (p, p_start, q, q_start)
            if memo in verdicts:
                if not verdicts[memo]:
                    loose.update((p, q))
                continue
            point = np.asarray(point, dtype=float)
            directions = []
            for key, at_start in ((p, p_start), (q, q_start)):
                curve = model.edges[key[0]].curve
                tangent = curve.derivative(key[1] if at_start else key[2])
                away = tangent if at_start else -tangent
                directions.append(away / float(np.linalg.norm(away)))
            normal = directions[1] - directions[0]
            if not np.all(np.isfinite(normal)) or not np.any(normal):
                verdicts[memo] = False                          # a fold straight back at the joint
                loose.update((p, q))
                continue
            verdicts[memo] = _joint_ok(model, p, p_start, q, q_start, point, normal, tolerance)
            if not verdicts[memo]:
                loose.update((p, q))
        if not loose:
            return [], pieces
        if level == max_level:
            break
        for edge, a, b in loose:
            middle = .5 * (a + b)
            spans = pieces[edge]
            index = spans.index((a, b))
            charge_split()
            spans[index:index + 1] = [(a, middle), (middle, b)]
    return sorted({key[0] for key in loose}), pieces


def certify_line_section_safe(curve, start, end, shared, tolerance, plane_tolerance=None):
    from .branch_predicates import certify_line_section
    try:
        return certify_line_section(curve, start, end, shared, tolerance,
                                    coplanar_tolerance=max(tolerance, plane_tolerance or 0.0))
    except PlaneNurbsNotCertified:
        return None


def _segment_distance(point, start, end) -> float:
    direction = np.asarray(end, dtype=float) - start
    length = float(direction @ direction)
    lam = 0.0 if length == 0 else min(1.0, max(0.0, float((np.asarray(point) - start) @ direction) / length))
    return float(np.linalg.norm(np.asarray(point) - (start + lam * direction)))


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
    starts = [model.oriented_start_vertex(item) for loop in loops for item in loop]
    if len(set(starts)) != len(starts):
        return [f"face {face_id} loops pass a vertex more than once"]
    frame = _Frame(plane, model.vertices[model.oriented_start_vertex(loops[0][0])].position)
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
    # exact residuals ((n . (V - O))**2 <= r**2 |n|**2 in rationals of the doubles): a far plane origin
    # cancelled the float residual (review B1-P4b round 3)
    n_exact = _cross(_exact(plane.u_vector), _exact(plane.v_vector))
    nn = sum(value * value for value in n_exact)
    o_exact = _exact(plane.origin)
    limit = Fraction(float(residual)) ** 2 * nn
    for vertex in vertices:
        if sum(a * (b - c) for a, b, c in zip(n_exact, _exact(vertex), o_exact)) ** 2 > limit:
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
                p, q = frame.xy(np.asarray(_edge_points(model, first)))
                r, s = frame.xy(np.asarray(_edge_points(model, second)))
                if adjacent:
                    # neighbours share one vertex: they overlap when one's far end lies on the other
                    # (a fold back along the same line, review B1-P4b round 1)
                    shared_ids = {a.start, a.end} & {b.start, b.end}
                    far_first = [model.vertices[v].position for v in (a.start, a.end) if v not in shared_ids]
                    far_second = [model.vertices[v].position for v in (b.start, b.end) if v not in shared_ids]
                    if any(_segment_distance(frame.xy(x), r, s) <= tolerance for x in far_first) or \
                            any(_segment_distance(frame.xy(x), p, q) <= tolerance for x in far_second):
                        verdict = False
                    else:
                        continue
                elif model._segments_intersect_2d(p, q, r, s, tolerance):  # noqa: SLF001
                    verdict = False
                else:
                    continue
            else:
                verdict = _pair_verdict(model, first, second, tolerance, residual)
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
    loose_edges, certified = _sections_simple(model, loops, tolerance)
    for edge in loose_edges:
        result.append(f"face {face_id} section edge {edge} is not certified simple (it may cross itself "
                      "or another section edge)")
    if result:
        return result
    for index, loop in enumerate(loops):
        loop_points = frame.xy(np.asarray([model.vertices[model.oriented_start_vertex(item)].position
                                           for item in loop], dtype=float))
        loop_extent = float(np.linalg.norm(np.ptp(loop_points, axis=0)))
        for item in loop:
            if isinstance(model.edges[item.edge].curve, PlaneNurbsCurve):
                low, high = model.edges[item.edge].curve.bounds()
                loop_extent = max(loop_extent, float(np.linalg.norm(np.asarray(high) - low)))
        # the bound shrinks with the pieces (enclosure areas ~ L**2 / pieces): refined while undecided
        for split in (1, 4, 16, 64):
            # the certified monotone pieces, each cut into ``split`` (sub-pieces stay monotone)
            pieces = {edge: [(a + (b - a) * k / split, a + (b - a) * (k + 1) / split)
                             for a, b in spans for k in range(split)] for edge, spans in certified.items()}
            try:
                area, bound = loop_area(model, loop, plane, pieces)
            except GeometryError as error:
                result.append(f"face {face_id} loop {index}: {error}")
                break
            if abs(area) > bound + model.tolerance.effective_area(loop_extent):
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
