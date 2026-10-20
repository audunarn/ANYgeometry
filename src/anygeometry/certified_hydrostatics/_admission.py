"""Admission: the six exact checks of design section 2, decided in rational arithmetic.

Order of the checks (the first failing check decides the category): 1 supports (exact planarity),
2 loops, 3 holes, 4 shell combinatorics, 5 embedding, 6 orientation. Every check reserves its
primitive units before running them (``Meter``), so budget, deadline and cancellation are polled at
every predicate."""
from __future__ import annotations

import functools
from contextvars import ContextVar
from fractions import Fraction

import numpy as np

from ._errors import RefusalCategory, refuse
from ._snapshot import ExactSnapshot
from ._work import Meter

__all__ = ["admit"]

ZERO = Fraction(0)
_ACTIVE: ContextVar = ContextVar("certified_hydrostatics_meter", default=None)


def _fit(value):
    """Check an exact result against the active bit-length cap (every predicate result goes through here)."""
    meter = _ACTIVE.get()
    if meter is not None:
        meter.fit(value)
    return value


def _fit_all(values):
    for value in values:
        _fit(value)
    return values


def _sub(a, b):
    return _fit_all((a[0] - b[0], a[1] - b[1], a[2] - b[2]))


def _cross(a, b):
    return _fit_all((a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]))


def _dot(a, b):
    return _fit(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def _is_zero(a) -> bool:
    return not (a[0] or a[1] or a[2])


def _sign(x) -> int:
    return (x > 0) - (x < 0)


def _orient(p, q, r):
    return _fit((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))


def _on_segment(p, a, b) -> bool:
    return (_orient(a, b, p) == 0 and min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))


def _segments_touch(p1, p2, q1, q2) -> bool:
    """Closed segments share a point (exact)."""
    d1, d2 = _orient(p1, p2, q1), _orient(p1, p2, q2)
    d3, d4 = _orient(q1, q2, p1), _orient(q1, q2, p2)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return (_on_segment(q1, p1, p2) or _on_segment(q2, p1, p2)
            or _on_segment(p1, q1, q2) or _on_segment(p2, q1, q2))


def _segment_intersection(p1, p2, q1, q2):
    """``None``, ``("p", x)`` or ``("s", x, y)``: the exact intersection of two closed 2D segments."""
    r = (p2[0] - p1[0], p2[1] - p1[1])
    s = (q2[0] - q1[0], q2[1] - q1[1])
    qp = (q1[0] - p1[0], q1[1] - p1[1])
    denominator = r[0] * s[1] - r[1] * s[0]
    if denominator != 0:
        t = _fit(Fraction(qp[0] * s[1] - qp[1] * s[0]) / denominator)
        u = _fit(Fraction(qp[0] * r[1] - qp[1] * r[0]) / denominator)
        if 0 <= t <= 1 and 0 <= u <= 1:
            return ("p", _fit_all((p1[0] + t * r[0], p1[1] + t * r[1])))
        return None
    if qp[0] * r[1] - qp[1] * r[0] != 0:
        return None
    rr = r[0] * r[0] + r[1] * r[1]
    t0 = _fit(Fraction(qp[0] * r[0] + qp[1] * r[1]) / rr)
    t1 = _fit(Fraction((q2[0] - p1[0]) * r[0] + (q2[1] - p1[1]) * r[1]) / rr)
    low, high = max(ZERO, min(t0, t1)), min(Fraction(1), max(t0, t1))
    if low > high:
        return None
    start = _fit_all((p1[0] + low * r[0], p1[1] + low * r[1]))
    if low == high:
        return ("p", start)
    return ("s", start, _fit_all((p1[0] + high * r[0], p1[1] + high * r[1])))


def _point_in_loop(point, loop, meter: Meter) -> int:
    """1 strictly inside, 0 on the boundary, -1 outside (exact ray parity, half-open rule)."""
    inside = False
    count = len(loop)
    for i in range(count):
        meter.unit("point_edge_test")
        a, b = loop[i], loop[(i + 1) % count]
        if _on_segment(point, a, b):
            return 0
        if (a[1] > point[1]) != (b[1] > point[1]):
            crossing = _fit(a[0] + Fraction((point[1] - a[1]) * (b[0] - a[0])) / (b[1] - a[1]))
            if point[0] < crossing:
                inside = not inside
    return 1 if inside else -1


def _region_state(point, loops2, meter: Meter) -> int:
    """1 strictly inside the face region, 0 on its boundary, -1 outside (outer loop first, then holes)."""
    outer = _point_in_loop(point, loops2[0], meter)
    if outer < 0:
        return -1
    state = outer
    for hole in loops2[1:]:
        inside_hole = _point_in_loop(point, hole, meter)
        if inside_hole > 0:
            return -1
        if inside_hole == 0:
            state = 0
    return state


class _Face:
    """A face prepared for the exact checks: loops as 3D and projected 2D points, Newell normal, box."""

    __slots__ = ("id", "support_normal", "vids", "eids", "points", "loops2", "normal", "drop", "offset",
                 "vertex_points", "edge_points", "vertex_set", "edge_set", "box")


def _project(point, drop: int):
    return tuple(point[axis] for axis in range(3) if axis != drop)


def _newell(points):
    base = points[0]
    total = (ZERO, ZERO, ZERO)
    for i in range(1, len(points) - 1):
        c = _cross(_sub(points[i], base), _sub(points[i + 1], base))
        total = _fit_all((total[0] + c[0], total[1] + c[1], total[2] + c[2]))
    return total


def _signed_area2(loop2):
    base = loop2[0]
    total = ZERO
    for i in range(1, len(loop2) - 1):
        total = _fit(total + _orient(base, loop2[i], loop2[i + 1]))
    return total


# ------------------------------------------------------------------ checks 1-3 (per face)


def _check_supports(snapshot: ExactSnapshot, meter: Meter) -> None:
    for face in snapshot.faces:
        origin, u, v = face.support
        normal = _cross(u, v)
        if _is_zero(normal):
            raise refuse(RefusalCategory.NOT_EXACTLY_PLANAR, f"the support of face {face.id} is degenerate")
        for loop in face.loops:
            for vertex in loop.vertices:
                meter.unit("plane_test")
                if meter.fit(_dot(normal, _sub(snapshot.positions[vertex], origin))) != 0:
                    raise refuse(RefusalCategory.NOT_EXACTLY_PLANAR,
                                 f"vertex {vertex} of face {face.id} is not exactly in its support plane")


def _simple_loop(face_id: int, loop2, meter: Meter) -> None:
    count = len(loop2)
    for i in range(count):
        meter.unit("edge_pair")
        a, b, c = loop2[i - 1], loop2[i], loop2[(i + 1) % count]
        if _orient(a, b, c) == 0 and (a[0] - b[0]) * (c[0] - b[0]) + (a[1] - b[1]) * (c[1] - b[1]) > 0:
            raise refuse(RefusalCategory.INVALID_LOOP, f"a loop of face {face_id} doubles back on itself (spike)")
    for i in range(count):
        for j in range(i + 2, count):
            if i == 0 and j == count - 1:
                continue
            meter.unit("edge_pair")
            if _segments_touch(loop2[i], loop2[(i + 1) % count], loop2[j], loop2[(j + 1) % count]):
                raise refuse(RefusalCategory.INVALID_LOOP, f"a loop of face {face_id} touches itself")


def _build_faces(snapshot: ExactSnapshot, meter: Meter) -> list:
    positions = snapshot.positions
    faces = []
    for face in snapshot.faces:
        prepared = _Face()
        prepared.id = face.id
        prepared.support_normal = _cross(face.support[1], face.support[2])
        prepared.vids, prepared.eids, prepared.points = [], [], []
        for loop in face.loops:
            count = len(loop.edges)
            if count < 3:
                raise refuse(RefusalCategory.INVALID_LOOP, f"a loop of face {face.id} has fewer than 3 edges")
            for i, (edge, forward) in enumerate(loop.edges):
                meter.unit("loop_edge")
                start, end = snapshot.edges[edge]
                if not forward:
                    start, end = end, start
                if loop.vertices[i] != start or loop.vertices[(i + 1) % count] != end:
                    raise refuse(RefusalCategory.INVALID_LOOP, f"a loop of face {face.id} is not continuous")
                if positions[start] == positions[end]:
                    raise refuse(RefusalCategory.INVALID_LOOP, f"edge {edge} of face {face.id} has zero length")
            prepared.vids.append(loop.vertices)
            prepared.eids.append(loop.edges)
            prepared.points.append([positions[v] for v in loop.vertices])
        normals = []
        for points in prepared.points:
            meter.unit("loop_area", len(points))
            normal = tuple(meter.fit(c) for c in _newell(points))
            if _is_zero(normal):
                raise refuse(RefusalCategory.INVALID_LOOP, f"a loop of face {face.id} has zero area")
            normals.append(normal)
        prepared.normal = normals[0]
        magnitude = [abs(c) for c in prepared.normal]
        prepared.drop = magnitude.index(max(magnitude))
        prepared.loops2 = [[_project(p, prepared.drop) for p in points] for points in prepared.points]
        prepared.offset = _dot(prepared.normal, prepared.points[0][0])
        prepared.vertex_points = {v: positions[v] for vids in prepared.vids for v in vids}
        prepared.vertex_set = frozenset(prepared.vertex_points)
        prepared.edge_set = frozenset(edge for loop in prepared.eids for edge, _ in loop)
        prepared.edge_points = {}
        for vids, eids in zip(prepared.vids, prepared.eids):
            for i, (edge, _forward) in enumerate(eids):
                prepared.edge_points[edge] = (positions[vids[i]], positions[vids[(i + 1) % len(vids)]])
        prepared.box = None
        faces.append(prepared)
    return faces


def _check_loops(faces: list, meter: Meter) -> None:
    for face in faces:
        for loop2 in face.loops2:
            _simple_loop(face.id, loop2, meter)
            meter.unit("loop_area", len(loop2))
            if _signed_area2(loop2) == 0:
                raise refuse(RefusalCategory.INVALID_LOOP, f"a loop of face {face.id} has zero projected area")


def _check_holes(faces: list, meter: Meter) -> None:
    for face in faces:
        if len(face.loops2) == 1:
            continue
        outer_sign = _sign(_signed_area2(face.loops2[0]))
        for index, hole in enumerate(face.loops2[1:], 1):
            if _sign(_signed_area2(hole)) == outer_sign:
                raise refuse(RefusalCategory.INVALID_HOLES,
                             f"hole {index} of face {face.id} winds like its outer loop")
        for index, hole in enumerate(face.loops2[1:], 1):
            for a in range(len(hole)):
                for b in range(len(face.loops2[0])):
                    meter.unit("edge_pair")
                    if _segments_touch(hole[a], hole[(a + 1) % len(hole)], face.loops2[0][b],
                                       face.loops2[0][(b + 1) % len(face.loops2[0])]):
                        raise refuse(RefusalCategory.INVALID_HOLES,
                                     f"hole {index} of face {face.id} touches its outer loop")
            if _point_in_loop(hole[0], face.loops2[0], meter) <= 0:
                raise refuse(RefusalCategory.INVALID_HOLES,
                             f"hole {index} of face {face.id} is not inside its outer loop")
        for first in range(1, len(face.loops2)):
            for second in range(first + 1, len(face.loops2)):
                one, other = face.loops2[first], face.loops2[second]
                for a in range(len(one)):
                    for b in range(len(other)):
                        meter.unit("edge_pair")
                        if _segments_touch(one[a], one[(a + 1) % len(one)], other[b], other[(b + 1) % len(other)]):
                            raise refuse(RefusalCategory.INVALID_HOLES,
                                         f"holes {first} and {second} of face {face.id} touch")
                if _point_in_loop(one[0], other, meter) > 0 or _point_in_loop(other[0], one, meter) > 0:
                    raise refuse(RefusalCategory.INVALID_HOLES,
                                 f"holes {first} and {second} of face {face.id} are nested")


# ------------------------------------------------------------------ check 4: shell combinatorics


class _Union:
    def __init__(self) -> None:
        self.parent: dict = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a, b) -> None:
        self.parent[self.find(a)] = self.find(b)


def _check_shell(faces: list, meter: Meter) -> None:
    uses: dict = {}
    for face in faces:
        for loop in face.eids:
            for edge, forward in loop:
                meter.unit("edge_use")
                uses.setdefault(edge, []).append((face.id, forward))
    for edge, owners in uses.items():
        if len(owners) < 2:
            raise refuse(RefusalCategory.NOT_CLOSED, f"edge {edge} bounds only one face")
    for edge, owners in uses.items():
        if len(owners) > 2:
            raise refuse(RefusalCategory.NON_MANIFOLD, f"edge {edge} bounds {len(owners)} faces")
        if owners[0][1] == owners[1][1]:
            raise refuse(RefusalCategory.NON_MANIFOLD, f"edge {edge} runs the same way in both of its faces")
    components = _Union()
    for face in faces:
        components.find(face.id)
    for owners in uses.values():
        components.union(owners[0][0], owners[1][0])
    if len({components.find(face.id) for face in faces}) != 1:
        raise refuse(RefusalCategory.NON_MANIFOLD, "the faces do not form one connected shell")
    link = _Union()
    vertices = set()
    for face in faces:
        for vids, eids in zip(face.vids, face.eids):
            for i, vertex in enumerate(vids):
                meter.unit("link_corner")
                link.union((vertex, eids[i - 1][0]), (vertex, eids[i][0]))
                vertices.add(vertex)
    roots: dict = {}
    for (vertex, _edge) in list(link.parent):
        roots.setdefault(vertex, set()).add(link.find((vertex, _edge)))
    for vertex, found in roots.items():
        if len(found) != 1:
            raise refuse(RefusalCategory.NON_MANIFOLD, f"the faces around vertex {vertex} do not form one cycle")


# ------------------------------------------------------------------ check 5: embedding


def _sorted(items, meter: Meter) -> list:
    """Sort with one work unit per comparison."""
    def compare(a, b):
        meter.unit("sort_comparison")
        return (a > b) - (a < b)
    return sorted(items, key=functools.cmp_to_key(compare))


def _merge(intervals: list, meter: Meter) -> list:
    merged: list = []
    for low, high in _sorted(intervals, meter):
        if merged and low <= merged[-1][1]:
            if high > merged[-1][1]:
                merged[-1] = (merged[-1][0], high)
        else:
            merged.append((low, high))
    return merged


def _closed_member(face: _Face, point, meter: Meter) -> bool:
    return _region_state(_project(point, face.drop), face.loops2, meter) >= 0


def _line_intervals(face: _Face, values: dict, direction, point_at, meter: Meter) -> list:
    """The face's section with the line, as merged closed intervals of ``direction . x``."""
    critical = set()
    for vids, points in zip(face.vids, face.points):
        count = len(vids)
        for i in range(count):
            meter.unit("edge_plane")
            g0, g1 = values[vids[i]], values[vids[(i + 1) % count]]
            p, q = points[i], points[(i + 1) % count]
            if g0 == 0:
                critical.add(_dot(direction, p))
            elif g1 != 0 and (g0 > 0) != (g1 > 0):
                ratio = _fit(g0 / (g0 - g1))
                critical.add(_dot(direction, _fit_all(tuple(p[a] + ratio * (q[a] - p[a]) for a in range(3)))))
    ordered = _sorted(critical, meter)
    found = []
    for t in ordered:
        if _closed_member(face, point_at(t), meter):
            found.append((t, t))
    for t0, t1 in zip(ordered, ordered[1:]):
        if _closed_member(face, point_at(_fit((t0 + t1) / 2)), meter):
            found.append((t0, t1))
    return _merge(found, meter)


def _covered(item, cells: list, meter: Meter) -> bool:
    """A 2D point or segment lies inside the union of the shared cells (points and closed segments)."""
    points = [c[1] for c in cells if c[0] == "v"]
    segments = [c[1:] for c in cells if c[0] == "e"]
    if item[0] == "p":
        x = item[1]
        return any(x == p for p in points) or any(_on_segment(x, a, b) for a, b in segments)
    x, y = item[1], item[2]
    direction = (y[0] - x[0], y[1] - x[1])
    norm = direction[0] ** 2 + direction[1] ** 2
    spans = []
    for a, b in segments:
        if _orient(x, y, a) == 0 and _orient(x, y, b) == 0:
            ta = _fit(Fraction((a[0] - x[0]) * direction[0] + (a[1] - x[1]) * direction[1]) / norm)
            tb = _fit(Fraction((b[0] - x[0]) * direction[0] + (b[1] - x[1]) * direction[1]) / norm)
            spans.append((min(ta, tb), max(ta, tb)))
    return any(low <= 0 and high >= 1 for low, high in _merge(spans, meter))


def _coplanar_ok(a: _Face, b: _Face, shared_vertices, shared_edges, meter: Meter) -> bool:
    drop = a.drop
    edges_a = {e: tuple(_project(p, drop) for p in a.edge_points[e]) for e in a.edge_set}
    edges_b = {e: tuple(_project(p, drop) for p in b.edge_points[e]) for e in b.edge_set}
    cells = [("v", _project(a.vertex_points[v], drop)) for v in shared_vertices]
    cells += [("e", *edges_a[e]) for e in shared_edges]
    for ea, (p1, p2) in edges_a.items():
        for eb, (q1, q2) in edges_b.items():
            meter.unit("edge_pair")
            if ea == eb:
                continue
            hit = _segment_intersection(p1, p2, q1, q2)
            if hit is not None and not _covered(hit, cells, meter):
                return False
    for own, other, edges in ((a, b, edges_a), (b, a, edges_b)):
        for edge, (p, q) in edges.items():
            if edge in shared_edges:
                continue
            middle = _fit_all(((p[0] + q[0]) / 2, (p[1] + q[1]) / 2))
            if _region_state(middle, other.loops2, meter) >= 0:
                return False
    for edge in shared_edges:
        meter.unit("edge_side")
        start_a, end_a = a.edge_points[edge]          # stored in the face's traversal order
        start_b, end_b = b.edge_points[edge]
        tangent_a, tangent_b = _sub(end_a, start_a), _sub(end_b, start_b)
        if _dot(_cross(a.normal, tangent_a), _cross(b.normal, tangent_b)) >= 0:
            return False
    return True


def _pair_embeds(a: _Face, b: _Face, meter: Meter) -> bool:
    shared_vertices = a.vertex_set & b.vertex_set
    shared_edges = a.edge_set & b.edge_set
    values_b = {}
    for vertex, point in b.vertex_points.items():
        meter.unit("plane_test")
        values_b[vertex] = meter.fit(_dot(a.normal, point) - a.offset)
    signs = {_sign(value) for value in values_b.values()}
    if signs == {1} or signs == {-1}:
        return not (shared_vertices or shared_edges)
    values_a = {}
    for vertex, point in a.vertex_points.items():
        meter.unit("plane_test")
        values_a[vertex] = meter.fit(_dot(b.normal, point) - b.offset)
    signs = {_sign(value) for value in values_a.values()}
    if signs == {1} or signs == {-1}:
        return not (shared_vertices or shared_edges)
    if all(value == 0 for value in values_b.values()):
        return _coplanar_ok(a, b, shared_vertices, shared_edges, meter)
    direction = _cross(a.normal, b.normal)
    norm = _dot(direction, direction)
    x0 = _fit_all(tuple(_fit((a.offset * c1 + b.offset * c2) / norm)
                        for c1, c2 in zip(_cross(b.normal, direction), _cross(direction, a.normal))))
    t0 = _dot(direction, x0)

    def point_at(t):
        scale = _fit((t - t0) / norm)
        return _fit_all(tuple(x0[i] + scale * direction[i] for i in range(3)))

    meet = _merge_intersection(_line_intervals(a, values_a, direction, point_at, meter),
                               _line_intervals(b, values_b, direction, point_at, meter), meter)
    cells = [(_dot(direction, a.vertex_points[v]),) * 2 for v in shared_vertices]
    for edge in shared_edges:
        start, end = a.edge_points[edge]
        cells.append(tuple(sorted((_dot(direction, start), _dot(direction, end)))))
    return meet == _merge(cells, meter)


def _merge_intersection(first: list, second: list, meter: Meter) -> list:
    found = []
    for low1, high1 in first:
        for low2, high2 in second:
            meter.unit("interval_pair")
            low, high = max(low1, low2), min(high1, high2)
            if low <= high:
                found.append((low, high))
    return _merge(found, meter)


def _check_embedding(faces: list, meter: Meter) -> None:
    for face in faces:
        meter.unit("box_vertex", len(face.vertex_points))
        coordinates = np.array([[float(c) for c in p] for p in face.vertex_points.values()])
        face.box = (coordinates.min(axis=0), coordinates.max(axis=0))
    low = np.array([face.box[0] for face in faces])
    high = np.array([face.box[1] for face in faces])
    for i, a in enumerate(faces):
        meter.unit("bbox_pair", len(faces) - i - 1)
        mask = np.all((low[i] <= high[i + 1:]) & (low[i + 1:] <= high[i]), axis=1)
        for offset in np.nonzero(mask)[0]:
            b = faces[i + 1 + int(offset)]
            if not _pair_embeds(a, b, meter):
                raise refuse(RefusalCategory.SELF_INTERSECTING,
                             f"faces {a.id} and {b.id} meet beyond their shared cells")


# ------------------------------------------------------------------ check 6: orientation


def _check_orientation(faces: list, meter: Meter) -> None:
    base = faces[0].points[0][0]
    total = ZERO
    for face in faces:
        for points in face.points:
            relative = [_sub(p, base) for p in points]
            for i in range(1, len(relative) - 1):
                meter.unit("triangle_sum")
                total = _fit(total + _dot(relative[0], _cross(relative[i], relative[i + 1])))
    if meter.fit(total) <= 0:
        raise refuse(RefusalCategory.INVERTED_ORIENTATION, "the shell encloses a non-positive signed volume")


def admit(snapshot: ExactSnapshot, meter: Meter) -> None:
    """Raise ``CertifiedHydrostaticsRefusal`` unless the snapshot is the oriented boundary of a solid."""
    token = _ACTIVE.set(meter)
    try:
        _check_supports(snapshot, meter)
        faces = _build_faces(snapshot, meter)
        _check_loops(faces, meter)
        _check_holes(faces, meter)
        _check_shell(faces, meter)
        _check_embedding(faces, meter)
        _check_orientation(faces, meter)
    finally:
        _ACTIVE.reset(token)
