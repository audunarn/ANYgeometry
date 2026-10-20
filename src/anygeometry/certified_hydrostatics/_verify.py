"""Independent verifier of certified hydrostatics results (design section 6, slice CH1c).

Written from the design specification only: nothing here uses the snapshot, admission, section, form or query
code of the provider. ``verify_certified_hydrostatics`` re-derives every claim in exact rational arithmetic from
the model and the face list and compares it with the complete result object.

Steps (design section 6): (0) the bound query equals the caller's query; (1) the proof is parsed (bounded),
recomputed from the result object and compared with ``expected``; (2) the snapshot, admission, model identity and
artifact descriptor are rebuilt from the model and the running environment; (3) V, J, contacts, occupancies,
cut points, segments, loops, topology and the waterplane quantities are recomputed; (4) exact equality and the
form rules; (5) completeness. Any discrepancy raises ``VERIFICATION_FAILED`` with ``error.sub_reason``.

Conventions used here (the specification leaves them open):
* ``expected`` is a mapping with ``artifact_files`` (sequence of ``(relative path, sha256 hex)``; an
  ``ImplementationIdentity`` or its dict under ``artifact`` is also accepted), ``convention_version``, ``units``
  and ``policy`` (a policy object or its dict).
* the snapshot digest is SHA-256 of the canonical JSON (sorted keys, no whitespace) of the document described in
  ``_snapshot_document`` (format inferred from the public ``ExactSnapshot.canonical`` output, see the report).
"""
from __future__ import annotations

import hashlib
import importlib.metadata as _metadata
import json
import math
import os
import platform
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Mapping

from ..curves import Straight
from ..surfaces import Plane
from ._errors import CertifiedHydrostaticsRefusal, RefusalCategory
from ._work import Meter, WorkBudget

__all__ = ["VerifiedHydrostatics", "verify_certified_hydrostatics"]

C = RefusalCategory
Z = Fraction(0)
_CHUNK = 65536
_QUANTITIES = ("A", "Mu", "Mv", "Iuu", "Ivv", "Iuv")
_MOMENTS = ("Jx", "Jy", "Jz")
_SUB_REASONS = ("artifact", "model", "face set", "snapshot", "convention", "policy", "query", "request", "value",
                "enclosure", "accuracy flag", "state", "topology")


# ------------------------------------------------------------------ refusals


def _fail(sub: str, message: str, cause: BaseException | None = None) -> CertifiedHydrostaticsRefusal:
    assert sub in _SUB_REASONS
    error = CertifiedHydrostaticsRefusal(C.VERIFICATION_FAILED, f"[{sub}] {message}")
    error.sub_reason = sub
    if cause is not None:
        error.__cause__ = cause
    return error


def _refuse(category: RefusalCategory, message: str) -> CertifiedHydrostaticsRefusal:
    error = CertifiedHydrostaticsRefusal(category, message)
    error.sub_reason = None
    return error


_RESOURCE = (C.BUDGET_EXHAUSTED, C.DEADLINE_EXCEEDED, C.CANCELLED, C.BIT_LENGTH_EXCEEDED)


# ------------------------------------------------------------------ small exact helpers


def _pair(x: Fraction) -> list:
    return [x.numerator, x.denominator]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub3(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add3(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale3(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _argmax_abs(v) -> int:
    best = 0
    for i in (1, 2):
        if abs(v[i]) > abs(v[best]):
            best = i
    return best


def _sign(x) -> int:
    return (x > 0) - (x < 0)


def _canonical_json(document) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _fraction_of_float(value) -> Fraction:
    x = float(value)
    if not math.isfinite(x):
        raise _refuse(C.NONFINITE_INPUT, "a stored coordinate is not finite")
    return Fraction(x)


# ------------------------------------------------------------------ polynomials in epsilon (simulation of simplicity)


def _padd(p, q):
    n = max(len(p), len(q))
    return tuple((p[i] if i < len(p) else Z) + (q[i] if i < len(q) else Z) for i in range(n))


def _psub(p, q):
    n = max(len(p), len(q))
    return tuple((p[i] if i < len(p) else Z) - (q[i] if i < len(q) else Z) for i in range(n))


def _pmul(p, q):
    out = [Z] * (len(p) + len(q) - 1)
    for i, x in enumerate(p):
        if x:
            for j, y in enumerate(q):
                out[i + j] += x * y
    return tuple(out)


def _psign(p) -> int:
    for c in p:
        if c:
            return 1 if c > 0 else -1
    return 0


# ------------------------------------------------------------------ the exact snapshot (own implementation)


class _Face:
    __slots__ = ("id", "loops", "support", "normal", "w", "axis", "loops2d", "vids", "eids", "box")


class _Snapshot:
    def __init__(self):
        self.faces: list[_Face] = []
        self.pos: dict[int, tuple] = {}
        self.edges: dict[int, tuple[int, int]] = {}


def _read_snapshot(model, faces, meter: Meter) -> _Snapshot:
    snap = _Snapshot()
    for fid in faces:
        if fid not in model.faces:
            raise _refuse(C.INVALID_LOOP, f"face {fid} is not in the model")
        face = model.faces[fid]
        surface = face.surface
        if not isinstance(surface, Plane):
            raise _refuse(C.UNSUPPORTED_SURFACE, f"face {fid} is not supported by a Plane")
        f = _Face()
        f.id = fid
        support = []
        for vec in (surface.origin, surface.u_vector, surface.v_vector):
            support.append(tuple(meter.fit(_fraction_of_float(x)) for x in vec))
        f.support = tuple(support)
        f.loops = []
        for loop in [face.loop, *face.holes]:
            items = []
            for oriented in loop:
                meter.unit("snapshot_read")
                edge = model.edges[oriented.edge]
                if not isinstance(edge.curve, Straight):
                    raise _refuse(C.UNSUPPORTED_SURFACE, f"edge {edge.id} is not straight")
                start, end = (edge.start, edge.end) if oriented.forward else (edge.end, edge.start)
                for vid in (edge.start, edge.end):
                    if vid not in snap.pos:
                        snap.pos[vid] = tuple(meter.fit(_fraction_of_float(x)) for x in model.vertex_position(vid))
                if snap.edges.setdefault(edge.id, (edge.start, edge.end)) != (edge.start, edge.end):
                    raise _refuse(C.INVALID_LOOP, "inconsistent edge record")
                items.append((edge.id, bool(oriented.forward), start, end))
            if len(items) < 3:
                raise _refuse(C.INVALID_LOOP, f"a loop of face {fid} has fewer than three edges")
            for i, item in enumerate(items):
                if item[3] != items[(i + 1) % len(items)][2]:
                    raise _refuse(C.NOT_CLOSED, f"a loop of face {fid} is not a closed chain")
            f.loops.append(items)
        snap.faces.append(f)
    return snap


def _snapshot_document(snap: _Snapshot) -> dict:
    faces = []
    for f in snap.faces:
        faces.append({"id": f.id, "support": [[_pair(x) for x in vec] for vec in f.support],
                      "loops": [[[eid, fwd, start] for eid, fwd, start, _end in loop] for loop in f.loops]})
    uses: dict[int, list] = {}
    vertex_faces: dict[int, set] = {}
    vertex_edges: dict[int, set] = {}
    for f in snap.faces:
        for loop in f.loops:
            for eid, fwd, start, end in loop:
                uses.setdefault(eid, []).append([f.id, fwd])
                for vid in (start, end):
                    vertex_faces.setdefault(vid, set()).add(f.id)
                    vertex_edges.setdefault(vid, set()).add(eid)
    return {
        "faces": faces,
        "vertices": [[vid, [_pair(x) for x in snap.pos[vid]]] for vid in sorted(vertex_faces)],
        "edges": [[eid, snap.edges[eid][0], snap.edges[eid][1], sorted(uses[eid])] for eid in sorted(uses)],
        "vertex_incidence": [[vid, sorted(vertex_faces[vid]), sorted(vertex_edges[vid])] for vid in sorted(vertex_faces)],
    }


def _sha256_chunks(data: bytes, meter: Meter, kind: str) -> str:
    h = hashlib.sha256()
    for i in range(0, max(len(data), 1), _CHUNK):
        meter.unit(kind)
        h.update(data[i:i + _CHUNK])
    return h.hexdigest()


# ------------------------------------------------------------------ admission (section 2), own implementation


def _project(point, axis):
    return (point[(axis + 1) % 3], point[(axis + 2) % 3])


def _orient2(p, q, r):
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def _on_segment2(p, q, r) -> bool:
    """p on the closed segment q-r (collinearity assumed checked by the caller)."""
    return min(q[0], r[0]) <= p[0] <= max(q[0], r[0]) and min(q[1], r[1]) <= p[1] <= max(q[1], r[1])


def _segments_meet(p1, p2, q1, q2) -> bool:
    d1, d2 = _orient2(p1, p2, q1), _orient2(p1, p2, q2)
    d3, d4 = _orient2(q1, q2, p1), _orient2(q1, q2, p2)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return ((d1 == 0 and _on_segment2(q1, p1, p2)) or (d2 == 0 and _on_segment2(q2, p1, p2))
            or (d3 == 0 and _on_segment2(p1, q1, q2)) or (d4 == 0 and _on_segment2(p2, q1, q2)))


def _locate2(p, loops) -> str:
    """'in' / 'on' / 'out' of point ``p`` against polygon-with-holes ``loops`` (lists of 2D points)."""
    for loop in loops:
        m = len(loop)
        for i in range(m):
            q, r = loop[i], loop[(i + 1) % m]
            if _orient2(q, r, p) == 0 and _on_segment2(p, q, r):
                return "on"
    inside = False
    for loop in loops:
        m = len(loop)
        for i in range(m):
            q, r = loop[i], loop[(i + 1) % m]
            q_le, r_le = q[1] <= p[1], r[1] <= p[1]
            if q_le == r_le:
                continue
            det = _orient2(q, r, p)
            if (q_le and det > 0) or ((not q_le) and det < 0):
                inside = not inside
    return "in" if inside else "out"


def _newell(points):
    nx = ny = nz = Z
    m = len(points)
    for i in range(m):
        p, q = points[i], points[(i + 1) % m]
        nx += (p[1] - q[1]) * (p[2] + q[2])
        ny += (p[2] - q[2]) * (p[0] + q[0])
        nz += (p[0] - q[0]) * (p[1] + q[1])
    return (nx / 2, ny / 2, nz / 2)


def _prepare_faces(snap: _Snapshot, meter: Meter) -> None:
    for f in snap.faces:
        o, u, v = f.support
        support_normal = _cross(u, v)
        if support_normal == (0, 0, 0):
            raise _refuse(C.INVALID_LOOP, f"face {f.id} has a degenerate support plane")
        f.loops2d = []
        pts_loops = []
        for loop in f.loops:
            pts = [snap.pos[start] for _e, _f, start, _end in loop]
            for p in pts:
                meter.unit("coplanarity_test")
                if _dot(_sub3(p, o), support_normal) != 0:
                    raise _refuse(C.NOT_EXACTLY_PLANAR, f"face {f.id} has a vertex off its support plane")
            pts_loops.append(pts)
        normals = [_newell(p) for p in pts_loops]
        outer = normals[0]
        if outer == (0, 0, 0):
            raise _refuse(C.INVALID_LOOP, f"face {f.id} has zero area")
        f.normal = outer
        f.w = _dot(outer, pts_loops[0][0])
        f.axis = _argmax_abs(outer)
        for k, pts in enumerate(pts_loops):
            if normals[k] == (0, 0, 0):
                raise _refuse(C.INVALID_LOOP, f"a loop of face {f.id} has zero area")
            f.loops2d.append([_project(p, f.axis) for p in pts])
            if k and _dot(normals[k], outer) >= 0:
                raise _refuse(C.INVALID_HOLES, f"a hole of face {f.id} is not wound against its outer loop")
        f.vids = {start for loop in f.loops for _e, _f, start, _end in loop}
        f.eids = {eid for loop in f.loops for eid, _f, _s, _e2 in loop}


def _check_loops(snap: _Snapshot, meter: Meter) -> None:
    for f in snap.faces:
        for k, loop2 in enumerate(f.loops2d):
            m = len(loop2)
            for i in range(m):
                if loop2[i] == loop2[(i + 1) % m]:
                    raise _refuse(C.INVALID_LOOP, f"face {f.id} has a zero-length edge")
            for i in range(m):
                for j in range(i + 1, m):
                    meter.unit("loop_pair_test")
                    p1, p2 = loop2[i], loop2[(i + 1) % m]
                    q1, q2 = loop2[j], loop2[(j + 1) % m]
                    if j == i + 1 or (i == 0 and j == m - 1):
                        # consecutive edges: they must meet only at the shared vertex
                        if j == i + 1:
                            x, s, y = p1, p2, q2
                        else:
                            x, s, y = q1, q2, p2
                        if _orient2(x, s, y) == 0 and ((x[0] - s[0]) * (y[0] - s[0]) + (x[1] - s[1]) * (y[1] - s[1])) > 0:
                            raise _refuse(C.INVALID_LOOP, f"face {f.id} has a zero-angle spike")
                    elif _segments_meet(p1, p2, q1, q2):
                        raise _refuse(C.INVALID_LOOP, f"a loop of face {f.id} is not simple")
        # holes: strictly inside the outer loop, pairwise disjoint
        outer = f.loops2d[0]
        for k in range(1, len(f.loops2d)):
            hole = f.loops2d[k]
            for i in range(len(hole)):
                for j in range(len(outer)):
                    meter.unit("hole_pair_test")
                    if _segments_meet(hole[i], hole[(i + 1) % len(hole)], outer[j], outer[(j + 1) % len(outer)]):
                        raise _refuse(C.INVALID_HOLES, f"a hole of face {f.id} touches the outer loop")
            if _locate2(hole[0], [outer]) != "in":
                raise _refuse(C.INVALID_HOLES, f"a hole of face {f.id} is not inside the outer loop")
            for k2 in range(1, k):
                other = f.loops2d[k2]
                for i in range(len(hole)):
                    for j in range(len(other)):
                        meter.unit("hole_pair_test")
                        if _segments_meet(hole[i], hole[(i + 1) % len(hole)], other[j], other[(j + 1) % len(other)]):
                            raise _refuse(C.INVALID_HOLES, f"holes of face {f.id} touch")
                if _locate2(hole[0], [other]) != "out" or _locate2(other[0], [hole]) != "out":
                    raise _refuse(C.INVALID_HOLES, f"holes of face {f.id} are nested")


def _check_combinatorics(snap: _Snapshot, meter: Meter) -> None:
    uses: dict[int, list] = {}
    corners: dict[int, list] = {}
    for f in snap.faces:
        for loop in f.loops:
            m = len(loop)
            for i, (eid, fwd, start, _end) in enumerate(loop):
                meter.unit("edge_use")
                uses.setdefault(eid, []).append((f.id, fwd))
                corners.setdefault(start, []).append((loop[i - 1][0], eid))
    parent = {f.id: f.id for f in snap.faces}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for eid, items in uses.items():
        if len(items) != 2:
            raise _refuse(C.NOT_CLOSED if len(items) < 2 else C.NON_MANIFOLD, f"edge {eid} is used {len(items)} times")
        (fa, da), (fb, db) = items
        if da == db:
            raise _refuse(C.INVERTED_ORIENTATION, f"edge {eid} is used twice in the same direction")
        if fa == fb:
            raise _refuse(C.NON_MANIFOLD, f"edge {eid} is used twice by one face")
        parent[find(fa)] = find(fb)
    if len({find(f.id) for f in snap.faces}) != 1:
        raise _refuse(C.NON_MANIFOLD, "the faces do not form one connected shell")
    for vid, cs in corners.items():
        meter.unit("vertex_link")
        adjacency: dict[int, list] = {}
        for a, b in cs:
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
        if any(len(v) != 2 for v in adjacency.values()):
            raise _refuse(C.NON_MANIFOLD, f"the link of vertex {vid} is not a cycle")
        seen, stack = set(), [next(iter(adjacency))]
        while stack:
            e = stack.pop()
            if e not in seen:
                seen.add(e)
                stack.extend(adjacency[e])
        if len(seen) != len(adjacency):
            raise _refuse(C.NON_MANIFOLD, f"the link of vertex {vid} is not a single cycle")


def _merge_intervals(intervals):
    out = []
    for lo, hi in sorted(intervals):
        if out and lo <= out[-1][1]:
            if hi > out[-1][1]:
                out[-1] = (out[-1][0], hi)
        else:
            out.append((lo, hi))
    return out


def _intersect_intervals(a, b):
    out = []
    for lo1, hi1 in a:
        for lo2, hi2 in b:
            lo, hi = max(lo1, lo2), min(hi1, hi2)
            if lo <= hi:
                out.append((lo, hi))
    return _merge_intervals(out)


def _seg_intersection2(p1, p2, q1, q2):
    """Intersection of two closed 2D segments: None, (P, P) for a point or (P, Q) for a collinear overlap."""
    d1 = (p2[0] - p1[0], p2[1] - p1[1])
    d2 = (q2[0] - q1[0], q2[1] - q1[1])
    w = (q1[0] - p1[0], q1[1] - p1[1])
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if den != 0:
        t = (w[0] * d2[1] - w[1] * d2[0]) / den
        u = (w[0] * d1[1] - w[1] * d1[0]) / den
        if 0 <= t <= 1 and 0 <= u <= 1:
            pt = (p1[0] + t * d1[0], p1[1] + t * d1[1])
            return pt, pt
        return None
    if w[0] * d1[1] - w[1] * d1[0] != 0:
        return None
    dd = d1[0] * d1[0] + d1[1] * d1[1]
    t1 = (w[0] * d1[0] + w[1] * d1[1]) / dd
    t2 = ((q2[0] - p1[0]) * d1[0] + (q2[1] - p1[1]) * d1[1]) / dd
    lo, hi = max(Z, min(t1, t2)), min(Fraction(1), max(t1, t2))
    if lo > hi:
        return None
    return ((p1[0] + lo * d1[0], p1[1] + lo * d1[1]), (p1[0] + hi * d1[0], p1[1] + hi * d1[1]))


def _param_on(x, a, b):
    d = (b[0] - a[0], b[1] - a[1])
    return ((x[0] - a[0]) * d[0] + (x[1] - a[1]) * d[1]) / (d[0] * d[0] + d[1] * d[1])


def _covered(piece, shared_pts, shared_segs) -> bool:
    x, y = piece
    if x == y:
        if x in shared_pts:
            return True
        return any(_orient2(a, b, x) == 0 and _on_segment2(x, a, b) for a, b in shared_segs)
    intervals = []
    for a, b in shared_segs:
        if _orient2(a, b, x) == 0 and _orient2(a, b, y) == 0:
            ta, tb = _param_on(a, x, y), _param_on(b, x, y)
            intervals.append((min(ta, tb), max(ta, tb)))
    merged = _merge_intervals(intervals)
    return any(lo <= 0 and hi >= 1 for lo, hi in merged)


def _edges2(face: _Face, snap: _Snapshot):
    """All edges of a face as (eid, forward, p2d, q2d)."""
    out = []
    for loop in face.loops:
        for eid, fwd, start, end in loop:
            out.append((eid, fwd, _project(snap.pos[start], face.axis), _project(snap.pos[end], face.axis)))
    return out


def _coplanar_pair(snap, f, g, shared_edges, shared_vids, meter) -> None:
    shared_pts = {_project(snap.pos[v], f.axis) for v in shared_vids}
    shared_segs = []
    for e in shared_edges:
        s, t = snap.edges[e]
        shared_segs.append((_project(snap.pos[s], f.axis), _project(snap.pos[t], f.axis)))
    ef, eg = _edges2(f, snap), _edges2(g, snap)
    sigma_f, sigma_g = _sign(f.normal[f.axis]), _sign(g.normal[f.axis])
    splits_f = {i: [Z, Fraction(1)] for i in range(len(ef))}
    splits_g = {j: [Z, Fraction(1)] for j in range(len(eg))}
    for i, (_a, _fa, p1, p2) in enumerate(ef):
        for j, (_b, _fb, q1, q2) in enumerate(eg):
            meter.unit("edge_pair_test")
            piece = _seg_intersection2(p1, p2, q1, q2)
            if piece is None:
                continue
            if not _covered(piece, shared_pts, shared_segs):
                raise _refuse(C.SELF_INTERSECTING, f"faces {f.id} and {g.id} meet outside their shared cells")
            for x in set(piece):
                splits_f[i].append(_param_on(x, p1, p2))
                splits_g[j].append(_param_on(x, q1, q2))
            if piece[0] != piece[1]:
                d1 = (p2[0] - p1[0], p2[1] - p1[1])
                d2 = (q2[0] - q1[0], q2[1] - q1[1])
                left1 = (-d1[1] * sigma_f, d1[0] * sigma_f)
                left2 = (-d2[1] * sigma_g, d2[0] * sigma_g)
                if left1[0] * left2[0] + left1[1] * left2[1] > 0:
                    raise _refuse(C.SELF_INTERSECTING, f"faces {f.id} and {g.id} overlap")
    for edges, splits, other in ((ef, splits_f, g.loops2d), (eg, splits_g, f.loops2d)):
        for idx, (_e, _fw, p1, p2) in enumerate(edges):
            ts = sorted(set(splits[idx]))
            for t0, t1 in zip(ts, ts[1:]):
                meter.unit("subpiece_test")
                tm = (t0 + t1) / 2
                mid = (p1[0] + tm * (p2[0] - p1[0]), p1[1] + tm * (p2[1] - p1[1]))
                if _locate2(mid, other) == "in":
                    raise _refuse(C.SELF_INTERSECTING, f"faces {f.id} and {g.id} overlap")


def _line_intervals(face: _Face, snap, other_normal, other_w, direction, meter):
    """Closed intervals (in the coordinate s = direction . r) of ``face`` intersected with the other plane."""
    critical = []
    intervals = []
    for loop in face.loops:
        for eid, _fw, start, end in loop:
            meter.unit("line_edge_test")
            q, r = snap.pos[start], snap.pos[end]
            gq, gr = _dot(other_normal, q) - other_w, _dot(other_normal, r) - other_w
            if gq == 0 and gr == 0:
                sq, sr = _dot(direction, q), _dot(direction, r)
                intervals.append((min(sq, sr), max(sq, sr)))
                critical.append((sq, q))
                critical.append((sr, r))
            elif gq == 0:
                critical.append((_dot(direction, q), q))
            elif gr == 0:
                critical.append((_dot(direction, r), r))
            elif (gq > 0) != (gr > 0):
                tau = gq / (gq - gr)
                point = _add3(q, _scale3(_sub3(r, q), tau))
                critical.append((_dot(direction, point), point))
    critical.sort(key=lambda item: item[0])
    for s, _p in critical:
        intervals.append((s, s))
    for (s0, p0), (s1, p1) in zip(critical, critical[1:]):
        if s0 == s1:
            continue
        meter.unit("line_interval_test")
        mid = _scale3(_add3(p0, p1), Fraction(1, 2))
        if _locate2(_project(mid, face.axis), face.loops2d) != "out":
            intervals.append((s0, s1))
    return _merge_intervals(intervals)


def _noncoplanar_pair(snap, f, g, shared_edges, shared_vids, meter) -> None:
    direction = _cross(f.normal, g.normal)
    i_f = _line_intervals(f, snap, g.normal, g.w, direction, meter)
    i_g = _line_intervals(g, snap, f.normal, f.w, direction, meter)
    touching = _intersect_intervals(i_f, i_g)
    shared = []
    for e in shared_edges:
        s, t = snap.edges[e]
        a, b = _dot(direction, snap.pos[s]), _dot(direction, snap.pos[t])
        shared.append((min(a, b), max(a, b)))
    for v in shared_vids:
        a = _dot(direction, snap.pos[v])
        shared.append((a, a))
    if touching != _merge_intervals(shared):
        raise _refuse(C.SELF_INTERSECTING, f"faces {f.id} and {g.id} meet outside their shared cells")


def _check_embedding(snap: _Snapshot, meter: Meter) -> None:
    scale = 1
    for p in snap.pos.values():
        for x in p:
            scale = max(scale, x.denominator)
    for f in snap.faces:
        coords = [snap.pos[start] for loop in f.loops for _e, _f, start, _end in loop]
        f.box = tuple((min(int(p[i] * scale) for p in coords), max(int(p[i] * scale) for p in coords))
                      for i in range(3))
    order = sorted(range(len(snap.faces)), key=lambda i: snap.faces[i].box[0][0])
    for pos, i in enumerate(order):
        f = snap.faces[i]
        j_pos = pos + 1
        while j_pos < len(order) and snap.faces[order[j_pos]].box[0][0] <= f.box[0][1]:
            g = snap.faces[order[j_pos]]
            j_pos += 1
            meter.unit("box_test")
            if any(g.box[a][0] > f.box[a][1] or f.box[a][0] > g.box[a][1] for a in (1, 2)):
                continue
            shared_edges = f.eids & g.eids
            shared_vids = f.vids & g.vids
            direction = _cross(f.normal, g.normal)
            if direction == (0, 0, 0):
                if _dot(f.normal, snap.pos[next(iter(g.vids))]) != f.w:
                    if shared_vids:
                        raise _refuse(C.SELF_INTERSECTING, "parallel faces share a vertex")
                    continue
                _coplanar_pair(snap, f, g, shared_edges, shared_vids, meter)
            else:
                _noncoplanar_pair(snap, f, g, shared_edges, shared_vids, meter)


def _admit(snap: _Snapshot, meter: Meter) -> Fraction:
    _prepare_faces(snap, meter)
    _check_loops(snap, meter)
    _check_combinatorics(snap, meter)
    _check_embedding(snap, meter)
    volume = Z
    for f in snap.faces:
        g = Z
        for loop in f.loops:
            pts = [snap.pos[start] for _e, _fw, start, _end in loop]
            g += _chain_G(pts, f.axis, meter)[0]
        volume += f.w / f.normal[f.axis] * g / 3
    if volume <= 0:
        raise _refuse(C.INVERTED_ORIENTATION, "the solid has non-positive volume")
    return meter.fit(volume)


# ------------------------------------------------------------------ Green / Mirtich integrals


def _pmul_list(p, q):
    out = [Z] * (len(p) + len(q) - 1)
    for i, x in enumerate(p):
        for j, y in enumerate(q):
            out[i + j] += x * y
    return out


def _edge_G(a0, b0, a1, b1):
    """Green's-theorem edge contributions to the integrals of 1, a, b, a^2, ab, b^2 (CCW positive)."""
    db = b1 - b0
    if db == 0:
        return (Z, Z, Z, Z, Z, Z)
    da = a1 - a0
    A1, B1 = [a0, da], [b0, db]
    A2 = _pmul_list(A1, A1)
    B2 = _pmul_list(B1, B1)

    def integral(poly):
        return sum((c / (k + 1) for k, c in enumerate(poly)), Z)

    ab = _pmul_list(A1, B1)
    return (db * integral(A1),
            db * integral(A2) / 2,
            db * integral(ab),
            db * integral(_pmul_list(A2, A1)) / 3,
            db * integral(_pmul_list(A2, B1)) / 2,
            db * integral(_pmul_list(A1, B2)))


def _chain_G(points, axis, meter: Meter, closed=True):
    """Sum of ``_edge_G`` over a closed chain of 3D points projected on the cyclic axes of ``axis``."""
    total = [Z] * 6
    m = len(points)
    for i in range(m):
        meter.unit("green_edge")
        p, q = points[i], points[(i + 1) % m]
        e = _edge_G(p[(axis + 1) % 3], p[(axis + 2) % 3], q[(axis + 1) % 3], q[(axis + 2) % 3])
        for k in range(6):
            total[k] += e[k]
    return [meter.fit(x) for x in total]


def _segments_G(segments, axes, meter: Meter):
    """Sum over directed segments ((P, Q) 3D points) projected on the coordinate indices ``axes`` = (a, b)."""
    a, b = axes
    total = [Z] * 6
    for p, q in segments:
        meter.unit("green_edge")
        e = _edge_G(p[a], p[b], q[a], q[b])
        for k in range(6):
            total[k] += e[k]
    return [meter.fit(x) for x in total]


def _face_contribution(G, normal, w, axis):
    """(V, Jx, Jy, Jz) contributions of one planar piece with chain integrals G (cyclic projection)."""
    a, b = (axis + 1) % 3, (axis + 2) % 3
    nc = normal[axis]
    g00, g10, g01, g20, g11, g02 = G
    a0, a1, a2 = w / nc, -normal[a] / nc, -normal[b] / nc
    quad = {axis: a0 * a0 * g00 + 2 * a0 * a1 * g10 + 2 * a0 * a2 * g01 + a1 * a1 * g20 + 2 * a1 * a2 * g11
            + a2 * a2 * g02,
            a: g20, b: g02}
    v = w / nc * g00 / 3
    j = [normal[i] / nc * quad[i] / 2 for i in range(3)]
    return v, j


# ------------------------------------------------------------------ section recomputation


@dataclass
class _Side:
    occupancy: str
    section: str
    quantities: dict | None
    topology: dict | None
    chain: list = field(default_factory=list)


def _clip_loop(points, h):
    """Sutherland-Hodgman against h < 0 (the ``below`` convention) -> list of 3D points."""
    out = []
    m = len(points)
    for i in range(m):
        p, q = points[i], points[(i + 1) % m]
        hp, hq = h[i], h[(i + 1) % m]
        p_in, q_in = hp < 0, hq < 0
        if p_in:
            out.append(p)
        if p_in != q_in:
            t = hp / (hp - hq)
            out.append(_add3(p, _scale3(_sub3(q, p), t)))
    return out


class _Plane:
    """The request, exactly."""

    def __init__(self, query):
        self.n = tuple(Fraction(x) for x in query.normal)
        self.d = Fraction(query.offset)
        self.o = tuple(Fraction(x) for x in query.waterplane_origin)
        self.e1, self.e2 = (tuple(Fraction(x) for x in axis) for axis in query.waterplane_axes)
        self.nn = _dot(self.n, self.n)
        self.k = _argmax_abs(self.n)
        self.ab = tuple(i for i in range(3) if i != self.k)
        self.orient = (-1 if self.k % 2 else 1) * _sign(self.n[self.k])


def _affine_vertex(c0, c1):
    return tuple((c0[i], c1[i]) for i in range(3))


def _section_side(snap: _Snapshot, plane: _Plane, side: str, heights: dict, meter: Meter) -> _Side:
    inside = (lambda h: h < 0) if side == "below" else (lambda h: h <= 0)
    tau = 1 if side == "below" else -1
    crossing = {}
    for eid, (s, e) in sorted(snap.edges.items()):
        meter.unit("edge_crossing")
        hs, he = heights[s], heights[e]
        if inside(hs) != inside(he):
            ps, pe = snap.pos[s], snap.pos[e]
            nn = he - hs
            t0 = -hs / nn
            delta = _sub3(pe, ps)
            c0 = tuple(meter.fit(x) for x in _add3(ps, _scale3(delta, t0)))
            c1 = tuple(meter.fit(x) for x in _scale3(delta, -tau / nn))
            crossing[eid] = _affine_vertex(c0, c1)
    values = [inside(h) for h in heights.values()]
    if all(values):
        occupancy = "FULL"
    elif not any(values):
        occupancy = "EMPTY"
    else:
        occupancy = "PARTIAL"
    if not crossing:
        return _Side(occupancy, "ABSENT", None, None)
    segments = []
    for f in snap.faces:
        direction = _cross(plane.n, f.normal)
        cuts = [eid for loop in f.loops for eid, _fw, _s, _e in loop if eid in crossing]
        if not cuts:
            continue
        if len(cuts) % 2 or direction == (0, 0, 0):
            raise _refuse(C.SYMBOLIC_DEGENERACY, f"face {f.id} has an odd or impossible number of cut points")

        def key(eid):
            point = crossing[eid]
            return (sum(direction[i] * point[i][0] for i in range(3)), sum(direction[i] * point[i][1] for i in range(3)))

        meter.unit("sort_comparison", max(1, len(cuts) * max(1, len(cuts).bit_length())))
        cuts.sort(key=key)
        keys = [key(eid) for eid in cuts]
        for k0, k1 in zip(keys, keys[1:]):
            if k0 == k1:
                raise _refuse(C.SYMBOLIC_DEGENERACY, "two distinct cut points coincide on a cut line")
        axis = f.axis
        a_i, b_i = (axis + 1) % 3, (axis + 2) % 3
        loops_const = [[((p[0],), (p[1],)) for p in loop] for loop in f.loops2d]
        for e0, e1 in zip(cuts, cuts[1:]):
            meter.unit("interval_classification", sum(len(loop) for loop in f.loops2d))
            p0, p1 = crossing[e0], crossing[e1]
            mid = tuple(((p0[i][0] + p1[i][0]) / 2, (p0[i][1] + p1[i][1]) / 2) for i in range(3))
            if _inside_sym(mid[a_i], mid[b_i], loops_const):
                segments.append((e0, e1))
    # loops
    nxt, indeg = {}, {}
    for a, b in segments:
        if a in nxt or b in indeg:
            raise _refuse(C.SYMBOLIC_DEGENERACY, "the cut-point graph is not 2-regular")
        nxt[a] = b
        indeg[b] = a
    if set(nxt) != set(indeg) or set(nxt) != set(crossing):
        raise _refuse(C.SYMBOLIC_DEGENERACY, "the cut-point graph has open chains")
    loops, seen = [], set()
    for start in sorted(nxt):
        if start in seen:
            continue
        loop, cur = [], start
        while cur not in seen:
            seen.add(cur)
            loop.append(cur)
            cur = nxt[cur]
        loops.append(loop)
    # waterplane quantities from the limit chain
    a_i, b_i = plane.ab
    chain = [(tuple(crossing[x][i][0] for i in range(3)), tuple(crossing[y][i][0] for i in range(3)))
             for x, y in segments]
    g00, g10, g01, g20, g11, g02 = _segments_G(chain, (a_i, b_i), meter)
    n, k = plane.n, plane.k
    u_c = _basis_coeffs(plane, plane.e1)
    v_c = _basis_coeffs(plane, plane.e2)

    def combine(p, q):
        (p0, p1, p2), (q0, q1, q2) = p, q
        return (p0 * q0 * g00 + (p0 * q1 + p1 * q0) * g10 + (p0 * q2 + p2 * q0) * g01 + p1 * q1 * g20
                + (p1 * q2 + p2 * q1) * g11 + p2 * q2 * g02)

    one = (Fraction(1), Z, Z)
    raw = {"A": g00, "Mu": combine(u_c, one), "Mv": combine(v_c, one), "Iuu": combine(u_c, u_c),
           "Ivv": combine(v_c, v_c), "Iuv": combine(u_c, v_c)}
    factor = plane.nn * plane.orient / abs(n[k])
    quantities = {name: (meter.fit(raw[name] * factor), plane.nn) for name in _QUANTITIES}
    # topology
    roles, polys = [], []
    for loop in loops:
        meter.unit("loop_area", len(loop))
        area = (Z,)
        m = len(loop)
        for i in range(m):
            p, q = crossing[loop[i]], crossing[loop[(i + 1) % m]]
            area = _padd(area, _psub(_pmul(p[a_i], q[b_i]), _pmul(q[a_i], p[b_i])))
        s = _psign(area) * plane.orient
        if s == 0:
            raise _refuse(C.SYMBOLIC_DEGENERACY, "a loop has zero symbolic area")
        roles.append("outer" if s > 0 else "hole")
        polys.append([(crossing[e][a_i], crossing[e][b_i]) for e in loop])
    outers = [i for i, r in enumerate(roles) if r == "outer"]
    holes = [i for i, r in enumerate(roles) if r == "hole"]
    holders: dict[int, list] = {i: [] for i in outers}
    for h in holes:
        probe = polys[h][0]
        cands = [o for o in outers if _inside_sym(probe[0], probe[1], [polys[o]])]
        if not cands:
            raise _refuse(C.SYMBOLIC_DEGENERACY, "a hole loop lies in no outer loop")
        best = [o for o in cands if all(o2 == o or _inside_sym(polys[o][0][0], polys[o][0][1], [polys[o2]])
                                        for o2 in cands)]
        if len(best) != 1:
            raise _refuse(C.SYMBOLIC_DEGENERACY, "the nesting of a hole loop is ambiguous")
        holders[best[0]].append(h)

    def ident(i):
        return tuple(sorted(loops[i]))

    topology = {"loops": sorted((ident(i), roles[i]) for i in range(len(loops))),
                "components": sorted((ident(o), tuple(sorted(ident(h) for h in holders[o]))) for o in outers)}
    return _Side(occupancy, "PRESENT", quantities, topology, chain)


def _basis_coeffs(plane: _Plane, e):
    """Coefficients (b0, b1, b2) of u = e.(r - o) as an affine function of the projection coordinates."""
    n, k = plane.n, plane.k
    a, b = plane.ab
    b0 = e[k] * plane.d / n[k] - _dot(e, plane.o)
    b1 = e[a] - e[k] * n[a] / n[k]
    b2 = e[b] - e[k] * n[b] / n[k]
    return (b0, b1, b2)


def _inside_sym(pa, pb, loops) -> bool:
    inside = False
    for loop in loops:
        m = len(loop)
        for i in range(m):
            qa, qb = loop[i]
            ra, rb = loop[(i + 1) % m]
            q_le = _psign(_psub(pb, qb)) >= 0
            r_le = _psign(_psub(pb, rb)) >= 0
            if q_le == r_le:
                continue
            det = _psub(_pmul(_psub(ra, qa), _psub(pb, qb)), _pmul(_psub(rb, qb), _psub(pa, qa)))
            s = _psign(det)
            if s == 0:
                raise _refuse(C.SYMBOLIC_DEGENERACY, "a test point lies on a loop edge")
            if (q_le and s > 0) or ((not q_le) and s < 0):
                inside = not inside
    return inside


def _recompute(snap: _Snapshot, plane: _Plane, total_volume: Fraction, meter: Meter) -> dict:
    heights = {}
    for vid, p in snap.pos.items():
        meter.unit("vertex_height")
        heights[vid] = meter.fit(_dot(plane.n, p) - plane.d)
    on_plane = sorted(v for v, h in heights.items() if h == 0)
    generic = not on_plane
    contacts_edges = sorted(e for e, (s, t) in snap.edges.items() if heights[s] == 0 and heights[t] == 0)
    contacts_faces = sorted(f.id for f in snap.faces if all(heights[v] == 0 for v in f.vids))
    hv = list(heights.values())
    if all(h >= 0 for h in hv):
        occupancy = "EMPTY"
    elif all(h <= 0 for h in hv):
        occupancy = "FULL"
    else:
        occupancy = "PARTIAL"
    # V and J: clipped faces (below convention) plus the cap
    volume = Z
    moments = [Z, Z, Z]
    for f in snap.faces:
        g = [Z] * 6
        for loop in f.loops:
            pts = [snap.pos[start] for _e, _fw, start, _end in loop]
            h = [heights[start] for _e, _fw, start, _end in loop]
            meter.unit("clip_vertex", len(pts))
            clipped = _clip_loop(pts, h)
            if len(clipped) >= 3:
                part = _chain_G(clipped, f.axis, meter)
                g = [x + y for x, y in zip(g, part)]
        v, j = _face_contribution(g, f.normal, f.w, f.axis)
        volume += v
        for i in range(3):
            moments[i] += j[i]
    below = _section_side(snap, plane, "below", heights, meter)
    sides = {}
    if generic:
        sides["plane"] = below
    else:
        sides["below"] = below
        sides["above"] = _section_side(snap, plane, "above", heights, meter)
    if below.section == "PRESENT":
        cap_segments = below.chain
        k = plane.k
        g = _segments_G(cap_segments, ((k + 1) % 3, (k + 2) % 3), meter)
        v, j = _face_contribution(g, plane.n, plane.d, k)
        volume += v
        for i in range(3):
            moments[i] += j[i]
    volume = meter.fit(volume)
    moments = [meter.fit(x) for x in moments]
    if occupancy == "EMPTY" and (volume != 0 or any(moments)):
        raise _refuse(C.SYMBOLIC_DEGENERACY, "an empty half-space has non-zero volume")
    if occupancy == "FULL" and volume != total_volume:
        raise _refuse(C.SYMBOLIC_DEGENERACY, "a full half-space does not give the solid volume")
    return {"occupancy": occupancy, "generic": generic,
            "contacts": {"vertices": on_plane, "edges": contacts_edges, "faces": contacts_faces},
            "V": volume, "J": moments, "sides": sides}



# ------------------------------------------------------------------ comparisons of forms


def _cmp_exact(r: Fraction, num: Fraction, ns: Fraction | None) -> int:
    """sign(r - x) with x = num (ns None) or num / sqrt(ns)."""
    if ns is None:
        return _sign(r - num)
    num_root, den_root = math.isqrt(ns.numerator), math.isqrt(ns.denominator)
    if num_root * num_root == ns.numerator and den_root * den_root == ns.denominator:
        return _sign(r - num / Fraction(num_root, den_root))
    # compare r * sqrt(ns) with num
    if r >= 0 and num < 0:
        return 1
    if r < 0 and num >= 0:
        return -1
    lhs, rhs = r * r * ns, num * num
    if r >= 0:
        return _sign(lhs - rhs)
    return -_sign(lhs - rhs)


_DECIMAL = re.compile(r"^-?[0-9]+(\.[0-9]+)?$")


def _decimal_fraction(text) -> Fraction:
    if not isinstance(text, str) or not _DECIMAL.match(text) or len(text) > 100000:
        raise _fail("enclosure", "a decimal bound is not a plain decimal string")
    negative = text.startswith("-")
    body = text[1:] if negative else text
    whole, _, frac = body.partition(".")
    value = Fraction(int(whole + frac), 10 ** len(frac))
    return -value if negative else value


def _check_quantity(item, name, num, ns, query, exact_required, meter: Meter) -> None:
    meter.unit("compare_element", 6)
    if getattr(item, "name", None) != name:
        raise _fail("request", f"quantity {name} is missing or misnamed")
    forms = tuple(query.forms)
    if exact_required:
        if item.numerator is None:
            raise _fail("request", f"{name}: exact form missing")
        if item.numerator != num or item.norm_squared != ns:
            raise _fail("value", f"{name}: exact value differs from the recomputed one")
    else:
        if item.numerator is not None or item.norm_squared is not None:
            raise _fail("request", f"{name}: unrequested exact form present")
    for form in ("binary64", "decimal"):
        present = getattr(item, form) is not None
        if present != (form in forms):
            raise _fail("request", f"{name}: form {form} {'present' if present else 'missing'} against the request")
    acc = query.accuracy(name)
    if item.accuracy != acc:
        raise _fail("request", f"{name}: bound accuracy differs from the request")
    met_ok = True
    if item.binary64 is not None:
        lo, hi = item.binary64
        if not (isinstance(lo, float) and isinstance(hi, float) and math.isfinite(lo) and math.isfinite(hi)):
            raise _fail("enclosure", f"{name}: binary64 bounds are not finite doubles")
        flo, fhi = Fraction(lo), Fraction(hi)
        if lo == hi:
            if _cmp_exact(flo, num, ns) != 0:
                raise _fail("enclosure", f"{name}: a point enclosure differs from the exact value")
        else:
            if math.nextafter(lo, math.inf) != hi:
                raise _fail("enclosure", f"{name}: binary64 endpoints are not adjacent doubles")
            if not (_cmp_exact(flo, num, ns) < 0 < _cmp_exact(fhi, num, ns)):
                raise _fail("enclosure", f"{name}: the binary64 enclosure does not contain the exact value")
            if acc is not None and fhi - flo > acc:
                met_ok = False
    if item.decimal is not None:
        lo_s, hi_s, width = item.decimal
        lo, hi = _decimal_fraction(lo_s), _decimal_fraction(hi_s)
        requested = acc if acc is not None else Fraction(1, 10 ** 12)
        if width != requested:
            raise _fail("enclosure", f"{name}: decimal width differs from the requested width")
        if hi - lo > requested or lo > hi:
            raise _fail("enclosure", f"{name}: the decimal enclosure is wider than requested")
        if not (_cmp_exact(lo, num, ns) <= 0 <= _cmp_exact(hi, num, ns)):
            raise _fail("enclosure", f"{name}: the decimal enclosure does not contain the exact value")
    flag = (True if met_ok else False) if acc is not None else None
    if item.accuracy_met != flag or flag is False:
        raise _fail("accuracy flag", f"{name}: accuracy flag differs from the recomputed one")


# ------------------------------------------------------------------ proof (step 1)


def _query_dict(q) -> dict:
    accuracy = {name: _pair(Fraction(v)) for name, v in q.absolute_accuracy}
    return {"absolute_accuracy": accuracy, "forms": list(q.forms),
            "normal": [_pair(Fraction(x)) for x in q.normal], "offset": _pair(Fraction(q.offset)),
            "waterplane_axes": [[_pair(Fraction(x)) for x in axis] for axis in q.waterplane_axes],
            "waterplane_origin": [_pair(Fraction(x)) for x in q.waterplane_origin]}


def _claim_quantity(item) -> dict:
    out: dict = {"accuracy": None if item.accuracy is None else _pair(item.accuracy),
                 "accuracy_met": item.accuracy_met}
    if item.numerator is not None:
        out["exact"] = {"numerator": _pair(item.numerator),
                        "norm_squared": None if item.norm_squared is None else _pair(item.norm_squared)}
    if item.binary64 is not None:
        out["binary64"] = {"lo": float.hex(item.binary64[0]), "hi": float.hex(item.binary64[1])}
    if item.decimal is not None:
        out["decimal"] = {"lo": item.decimal[0], "hi": item.decimal[1], "width": _pair(item.decimal[2])}
    return out


def _claim_section(section) -> dict:
    topology = None
    if section.topology is not None:
        topology = {"loops": [{"identity": list(l.identity), "role": l.role} for l in section.topology.loops],
                    "components": [{"outer": list(o), "holes": [list(h) for h in hs]}
                                   for o, hs in section.topology.components]}
    quantities = None
    if section.quantities is not None:
        quantities = {q.name: _claim_quantity(q) for q in section.quantities}
    return {"occupancy": section.occupancy, "section": section.section, "quantities": quantities,
            "topology": topology}


def _claim_of(result) -> dict:
    return {"occupancy": result.occupancy, "generic": result.generic,
            "contacts": {"vertices": list(result.contacts.vertices), "edges": list(result.contacts.edges),
                         "faces": list(result.contacts.faces)},
            "volume": _claim_quantity(result.volume),
            "moments": {m.name: _claim_quantity(m) for m in result.moments},
            "sections": {s.side: _claim_section(s) for s in result.sections},
            "work": {"operations": result.work["operations"], "units": dict(result.work["units"])}}


def _parse_proof(proof, policy: dict, meter: Meter) -> dict:
    if not isinstance(proof, (bytes, bytearray)):
        raise _refuse(C.MALFORMED_PROOF, "the proof is not bytes")
    if len(proof) > policy["max_proof_bytes"]:
        raise _refuse(C.MALFORMED_PROOF, "the proof exceeds the byte limit")
    limit = policy["max_integer_digits"]

    def parse_int(token):
        if len(token.lstrip("-")) > limit:
            raise _refuse(C.MALFORMED_PROOF, "an integer token exceeds the digit limit")
        try:
            return int(token)
        except ValueError as error:
            raise _refuse(C.MALFORMED_PROOF, "an integer token cannot be converted") from error

    def reject(_token):
        raise _refuse(C.MALFORMED_PROOF, "non-integer numbers and constants are not allowed")

    try:
        text = bytes(proof).decode("utf-8")
        document = json.loads(text, parse_int=parse_int, parse_float=reject, parse_constant=reject)
    except CertifiedHydrostaticsRefusal:
        raise
    except (ValueError, RecursionError) as error:
        raise _refuse(C.MALFORMED_PROOF, f"the proof is not valid JSON: {error}") from error
    if not isinstance(document, dict) or set(document) != {"artifact", "model", "convention", "preparation_policy",
                                                            "query", "result"}:
        raise _refuse(C.MALFORMED_PROOF, "the proof has the wrong top-level structure")
    count = _count_elements(document)
    meter.unit("proof_element", count)
    return document


def _count_elements(node) -> int:
    stack, count = [node], 0
    while stack:
        x = stack.pop()
        count += 1
        if isinstance(x, dict):
            stack.extend(x.values())
        elif isinstance(x, list):
            stack.extend(x)
    return count


def _first_difference(a, b, path=()):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                return path + (key,), "missing"
            found = _first_difference(a[key], b[key], path + (key,))
            if found:
                return found
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return path, "length"
        for i, (x, y) in enumerate(zip(a, b)):
            found = _first_difference(x, y, path + (i,))
            if found:
                return found
        return None
    if type(a) is not type(b) or a != b:
        return path, "value"
    return None


def _classify(path, kind) -> str:
    """Sub-reason of a difference between the proof's result claim and the claim of the result object."""
    if not path:
        return "value"
    head = path[0]
    if head in ("occupancy", "generic"):
        return "state"
    if head in ("contacts", "work"):
        return "value"
    if head == "sections":
        if len(path) <= 2:
            return "state"
        rest = path[2:]
        if rest[0] in ("occupancy", "section"):
            return "state"
        if rest[0] == "topology":
            return "topology"
        if rest[0] == "quantities":
            if len(rest) <= 2:
                return "state" if kind == "value" else "request"
            return _classify_quantity(rest[2:], kind)
        return "state"
    if head in ("volume", "moments"):
        rest = path[1:] if head == "volume" else path[2:]
        if head == "moments" and len(path) < 2:
            return "request"
        return _classify_quantity(rest, kind)
    return "value"


def _classify_quantity(rest, kind) -> str:
    if not rest:
        return "request"
    key = rest[0]
    if key == "exact":
        return "request" if kind == "missing" and len(rest) == 1 else "value"
    if key in ("binary64", "decimal"):
        return "request" if kind == "missing" and len(rest) == 1 else "enclosure"
    if key == "accuracy_met":
        return "accuracy flag"
    if key == "accuracy":
        return "request"
    return "request" if kind == "missing" else "value"


def _policy_dict(policy) -> dict:
    if isinstance(policy, Mapping):
        source = policy
        get = lambda k: source[k]
    else:
        get = lambda k: getattr(policy, k)
    return {"convention_version": get("convention_version"), "max_integer_digits": get("max_integer_digits"),
            "max_proof_bytes": get("max_proof_bytes")}


def _expected_identity(expected) -> dict:
    if not isinstance(expected, Mapping):
        raise TypeError("expected must be a mapping with artifact_files, convention_version, units and policy")
    if "artifact_files" in expected:
        files = expected["artifact_files"]
    elif "artifact" in expected:
        art = expected["artifact"]
        files = art["files"] if isinstance(art, Mapping) else art.files
    else:
        raise ValueError("expected lacks the artifact file hashes")
    for key in ("convention_version", "units", "policy"):
        if key not in expected:
            raise ValueError(f"expected lacks {key}")
    return {"files": sorted([str(p), str(h)] for p, h in files), "convention_version": expected["convention_version"],
            "units": expected["units"], "policy": _policy_dict(expected["policy"])}


# ------------------------------------------------------------------ environment (step 2)


def _environment(meter: Meter) -> dict:
    import anygeometry
    package = os.path.dirname(os.path.abspath(anygeometry.__file__))
    rows = []
    for root, dirs, names in os.walk(package):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in names:
            rows.append(os.path.relpath(os.path.join(root, name), package).replace(os.sep, "/"))
    rows.sort()
    files = []
    for rel in rows:
        h = hashlib.sha256()
        with open(os.path.join(package, *rel.split("/")), "rb") as handle:
            while True:
                meter.unit("package_hash_chunk")
                chunk = handle.read(_CHUNK)
                if not chunk:
                    break
                h.update(chunk)
        files.append([rel, h.hexdigest()])
    try:
        dist = _metadata.distribution("anygeometry")
        dist_version = dist.version
        requires = dist.requires or []
    except _metadata.PackageNotFoundError:
        dist_version, requires = None, []
    deps = []
    for requirement in requires:
        if "extra ==" in requirement.replace("'", '"').replace("extra==", "extra =="):
            continue
        name = re.match(r"[A-Za-z0-9_.\-]+", requirement.strip())
        if name is None:
            continue
        try:
            version = _metadata.version(name.group(0))
        except _metadata.PackageNotFoundError:
            version = None
        deps.append([name.group(0), version])
    return {"distribution": "anygeometry", "distribution_version": dist_version,
            "package_version": getattr(anygeometry, "__version__", None), "files": files,
            "dependencies": sorted(deps, key=lambda x: x[0]),
            "python": f"{platform.python_implementation()} {platform.python_version()}"}


def _document_checksum(model, meter: Meter) -> str:
    from ..serialization import to_dict
    document = to_dict(model)
    payload = {k: v for k, v in document.items() if k != "checksum"}
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"),
                      sort_keys=True).encode("utf-8")
    own = _sha256_chunks(data, meter, "document_hash_chunk")
    if own != document["checksum"]["value"]:
        raise _fail("model", "the document checksum could not be reproduced independently")
    return own


# ------------------------------------------------------------------ result object

@dataclass(frozen=True)
class VerifiedHydrostatics:
    """A result whose every claim was recomputed and found equal (design section 6). Use this, not the raw result."""

    result: Any
    query: Any
    face_ids: tuple
    model_id: str
    revision: int
    document_checksum: str
    snapshot_digest: str
    artifact_files_sha256: str
    proof_sha256: str
    solid_volume: Fraction
    work: dict = field(compare=False)

    def __getattr__(self, name):
        if name.startswith("__") or name == "result":
            raise AttributeError(name)
        return getattr(self.result, name)


# ------------------------------------------------------------------ main entry


def verify_certified_hydrostatics(model, faces, query, result, expected, *, budget: WorkBudget | None = None,
                                  cancellation_check: Callable[[], bool] | None = None) -> VerifiedHydrostatics:
    """Verify ``result`` for ``query`` against ``model`` / ``faces`` and the consumer's ``expected`` identities.

    Raises ``CertifiedHydrostaticsRefusal`` (``VERIFICATION_FAILED`` with ``sub_reason``, ``MALFORMED_PROOF``, or a
    resource category) and never returns a partially verified object."""
    meter = Meter(budget, cancellation_check)
    try:
        return _verify(model, faces, query, result, expected, meter)
    except CertifiedHydrostaticsRefusal as error:
        try:
            request = _query_dict(query)
        except Exception:
            request = None
        error.complete(phase="verify", request=request, work=meter.budget.counts())
        raise


def _verify(model, faces, query, result, expected, meter: Meter) -> VerifiedHydrostatics:
    faces = tuple(faces)
    exp = _expected_identity(expected)
    # (0) the bound query is the caller's query
    try:
        bound = _query_dict(result.query)
    except Exception as error:
        raise _fail("query", "the result carries no readable query", error) from error
    mine = _query_dict(query)
    meter.unit("compare_element", _count_elements(mine))
    if bound != mine:
        raise _fail("query", "the query bound in the result differs from the supplied query")
    plane = _Plane(query)
    if plane.n == (0, 0, 0) or _dot(_cross(plane.e1, plane.e2), plane.n) == 0:
        raise _refuse(C.INVALID_QUERY, "the half-space or waterplane basis is invalid")
    for form in query.forms:
        if form not in ("exact", "binary64", "decimal"):
            raise _refuse(C.INVALID_QUERY, "unknown form")

    # (1) proof
    proof = _parse_proof(result.proof, exp["policy"], meter)
    if proof["query"] != mine:
        raise _fail("query", "the proof's query differs from the supplied query")
    claim = _claim_of(result)
    difference = _first_difference(proof["result"], claim)
    if difference:
        path, kind = difference
        raise _fail(_classify(path, kind), f"the proof's claim differs from the result object at {list(path)}")
    if _canonical_json(proof) != bytes(result.proof):
        raise _fail("value", "the proof is not in canonical form")
    if proof["preparation_policy"] != exp["policy"]:
        raise _fail("policy", "the proof's preparation policy differs from the expected policy")
    convention = proof["convention"]
    if (not isinstance(convention, dict) or convention.get("convention_version") != exp["convention_version"]
            or exp["policy"]["convention_version"] != exp["convention_version"]):
        raise _fail("convention", "the convention version differs from the expected one")
    if convention.get("units") != exp["units"]:
        raise _fail("convention", "the units differ from the expected ones")
    if convention.get("moment_order") != list(_QUANTITIES):
        raise _fail("convention", "the moment order differs from the specification")
    for key in ("body_frame", "half_space", "area_element", "limits", "loop_identity"):
        if not isinstance(convention.get(key), str) or not convention[key]:
            raise _fail("convention", f"convention field {key} is missing")
    artifact = proof["artifact"]
    if not isinstance(artifact, dict) or sorted(map(list, artifact.get("files", []))) != exp["files"]:
        raise _fail("artifact", "the proof's file hashes differ from the expected artifact identity")

    # (2) snapshot, admission, identities
    if [*faces] != proof["model"].get("face_ids"):
        raise _fail("face set", "the supplied face list differs from the proof's face set")
    revision_before = model.revision
    try:
        snap = _read_snapshot(model, faces, meter)
    except CertifiedHydrostaticsRefusal as error:
        if error.category in _RESOURCE:
            raise
        raise _fail("snapshot", f"the model can no longer be read as the prepared snapshot ({error.detail})",
                    error) from error
    if model.revision != revision_before:
        raise _refuse(C.CONCURRENT_MUTATION, "the model changed during verification")
    document = _snapshot_document(snap)
    digest = _sha256_chunks(_canonical_json(document), meter, "snapshot_hash_chunk")
    if digest != proof["model"].get("snapshot_digest"):
        raise _fail("snapshot", "the rebuilt snapshot digest differs from the proof")
    total_volume = _admit(snap, meter)
    model_id = str(model.model_id)
    checksum = _document_checksum(model, meter)
    pm = proof["model"]
    if pm.get("model_id") != model_id or pm.get("revision") != model.revision or pm.get("document_checksum") != checksum:
        raise _fail("model", "the model identity differs from the proof's")
    env = _environment(meter)
    for key, value in env.items():
        if key == "dependencies":
            if sorted(map(list, artifact.get(key, []))) != value:
                raise _fail("artifact", "the claimed dependency versions differ from the running environment")
        elif key == "files":
            if sorted(map(list, artifact.get(key, []))) != sorted(value):
                raise _fail("artifact", "the package file hashes differ from the running environment")
        elif artifact.get(key) != value:
            raise _fail("artifact", f"the claimed {key} differs from the running environment")
    if set(artifact) != set(env):
        raise _fail("artifact", "the artifact descriptor has unexpected fields")

    # (3) recompute
    truth = _recompute(snap, plane, total_volume, meter)

    # (4)/(5) compare, with completeness
    if result.generic != truth["generic"]:
        raise _fail("value", "the degeneracy flag differs")
    contacts = result.contacts
    if (sorted(contacts.vertices) != truth["contacts"]["vertices"] or sorted(contacts.edges) != truth["contacts"]["edges"]
            or sorted(contacts.faces) != truth["contacts"]["faces"]):
        raise _fail("value", "the contact lists differ")
    if result.occupancy != truth["occupancy"]:
        raise _fail("state", "the exact half-space occupancy differs")
    _check_quantity(result.volume, "V", truth["V"], None, query, True, meter)
    if len(result.moments) != 3:
        raise _fail("request", "the first volume moments are incomplete")
    for item, name, value in zip(result.moments, _MOMENTS, truth["J"]):
        _check_quantity(item, name, value, None, query, True, meter)
    if [s.side for s in result.sections] != list(truth["sides"]):
        raise _fail("state", "the set of reported sides differs")
    for section in result.sections:
        side = truth["sides"][section.side]
        if section.occupancy != side.occupancy or section.section != side.section:
            raise _fail("state", f"side {section.side}: occupancy or section flag differs")
        if side.section == "ABSENT":
            if section.quantities is not None:
                raise _fail("state", f"side {section.side}: quantities reported for an absent section")
            if section.topology is not None:
                raise _fail("topology", f"side {section.side}: topology reported for an absent section")
            continue
        if section.quantities is None:
            raise _fail("state", f"side {section.side}: quantities missing for a present section")
        if len(section.quantities) != len(_QUANTITIES):
            raise _fail("request", f"side {section.side}: quantity inventory differs")
        for item, name in zip(section.quantities, _QUANTITIES):
            num, ns = side.quantities[name]
            _check_quantity(item, name, num, ns, query, "exact" in query.forms, meter)
        if section.topology is None:
            raise _fail("topology", f"side {section.side}: topology missing for a present section")
        meter.unit("compare_element", len(side.topology["loops"]) + len(side.topology["components"]) + 1)
        got_loops = sorted((tuple(l.identity), l.role) for l in section.topology.loops)
        got_components = sorted((tuple(o), tuple(sorted(tuple(h) for h in hs)))
                                for o, hs in section.topology.components)
        if got_loops != side.topology["loops"] or got_components != side.topology["components"]:
            raise _fail("topology", f"side {section.side}: loops or components differ")

    return VerifiedHydrostatics(
        result=result, query=query, face_ids=faces, model_id=model_id, revision=model.revision,
        document_checksum=checksum, snapshot_digest=digest,
        artifact_files_sha256=hashlib.sha256(_canonical_json(exp["files"])).hexdigest(),
        proof_sha256=hashlib.sha256(bytes(result.proof)).hexdigest(), solid_volume=total_volume,
        work=meter.budget.counts())
