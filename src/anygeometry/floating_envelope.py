"""Geometry-owned closed displacement envelope (indexed, oriented, identity-bound) for the floating-body workflow.

Identity is the model's own VERTEX identity: two faces share an edge only when they use the same geometry vertex ids (no
coordinate rounding or welding).  The envelope is accepted only when every undirected edge is used by exactly two faces, the
faces can be oriented consistently (propagated from shared edges), the oriented shell has positive volume (outward normals), and
every face is planar within a declared relative tolerance.  Anything else is refused with the offending faces/edges.
"""
from __future__ import annotations

import hashlib
import json
from collections import deque

import numpy as np

from .errors import GeometryError

PLANARITY_RELATIVE_TOLERANCE = 1e-9


def build_envelope(geometry, face_ids) -> dict:
    ids = [int(f) for f in face_ids]
    if len(ids) < 4 or len(set(ids)) != len(ids):
        raise GeometryError('a closed envelope needs at least four distinct faces')
    loops = {f: [int(v) for v in geometry.face_corner_vertices(f)] for f in ids}
    positions = {}
    for loop in loops.values():
        if len(loop) < 3 or len(set(loop)) != len(loop):
            raise GeometryError('envelope faces need simple loops of at least three corners')
        for v in loop:
            positions[v] = np.asarray(geometry.vertex_position(v), dtype=float)
    edge_faces = {}
    for f, loop in loops.items():
        for a, b in zip(loop, loop[1:] + loop[:1]):
            edge_faces.setdefault(frozenset((a, b)), []).append(f)
    bad = [sorted(e) for e, users in edge_faces.items() if len(users) != 2]
    if bad:
        raise GeometryError(f'envelope is not closed: {len(bad)} edges are not used by exactly two faces (first {bad[:3]}); '
                            'select the closing faces explicitly')

    def directed(loop):
        return set(zip(loop, loop[1:] + loop[:1]))

    # orientation propagation: neighbours must traverse the shared edge in opposite directions
    flipped, seen, queue = {ids[0]: False}, {ids[0]}, deque([ids[0]])
    adjacency = {f: set() for f in ids}
    for users in edge_faces.values():
        adjacency[users[0]].add(users[1]); adjacency[users[1]].add(users[0])
    while queue:
        f = queue.popleft()
        mine = directed(loops[f][::-1] if flipped[f] else loops[f])
        for g in adjacency[f]:
            theirs = directed(loops[g])
            shared_same = any((a, b) in theirs for a, b in mine)
            if g not in seen:
                flipped[g] = shared_same            # same direction as ours -> flip to oppose
                seen.add(g); queue.append(g)
            else:
                other = directed(loops[g][::-1] if flipped[g] else loops[g])
                if any((a, b) in other for a, b in mine):
                    raise GeometryError('envelope faces cannot be oriented consistently (non-orientable shell)')
    if len(seen) != len(ids):
        raise GeometryError('envelope faces form more than one shell')
    oriented = {f: (loops[f][::-1] if flipped[f] else loops[f]) for f in ids}

    def volume(orient):
        total = 0.0
        for loop in orient.values():
            p = [positions[v] for v in loop]
            for k in range(1, len(p) - 1):
                total += float(p[0] @ np.cross(p[k], p[k + 1])) / 6.0
        return total

    vol = volume(oriented)
    if vol == 0.0 or not np.isfinite(vol):
        raise GeometryError('envelope encloses no volume')
    if vol < 0.0:
        oriented = {f: loop[::-1] for f, loop in oriented.items()}
        vol = -vol
    scale = max(1.0, max(float(np.abs(p).max()) for p in positions.values()))
    for f, loop in oriented.items():
        p = np.array([positions[v] for v in loop])
        av = 0.5 * np.cross(p, np.roll(p, -1, axis=0)).sum(axis=0)
        n = np.linalg.norm(av)
        if n == 0.0 or np.abs((p - p[0]) @ (av / n)).max() > PLANARITY_RELATIVE_TOLERANCE * scale:
            raise GeometryError(f'envelope face {f} is not planar within the declared tolerance '
                                f'{PLANARITY_RELATIVE_TOLERANCE} (relative)')
    vertex_ids = sorted(positions)
    index = {v: i for i, v in enumerate(vertex_ids)}
    doc = dict(vertex_ids=vertex_ids, vertices=[positions[v].tolist() for v in vertex_ids],
               face_ids=ids, faces=[[index[v] for v in oriented[f]] for f in ids], volume=vol,
               planarity_relative_tolerance=PLANARITY_RELATIVE_TOLERANCE)
    doc['identity'] = hashlib.sha256(json.dumps(doc, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return doc
