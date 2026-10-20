"""The exact snapshot of a face set and its canonical digest (design section 6).

Inputs are converted with ``fractions.Fraction(float_value)`` only (the exact binary value of the stored
float, never a decimal string or a rounded intermediate). The digest is the SHA-256 of a canonical JSON
with rationals as integer pairs; it covers the face ids in order, per face the exact support and loops
(oriented edge identities with their start vertices), the exact vertex coordinates, and the vertex and
edge identities with their incidences. It does not cover model id, revision or document checksum."""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Iterable, Mapping

from ..curves import Straight
from ..surfaces import Plane
from ._errors import RefusalCategory, refuse
from ._identity import canonical_json, sha256_chunks
from ._work import Meter

__all__ = ["ExactSnapshot", "SnapshotFace", "SnapshotLoop", "collect_snapshot"]

Point = tuple  # three Fractions


@dataclass(frozen=True)
class SnapshotLoop:
    edges: tuple          # ((edge id, forward), ...)
    vertices: tuple       # oriented start vertex id of each edge


@dataclass(frozen=True)
class SnapshotFace:
    id: int
    support: tuple        # (origin, u_vector, v_vector), each three Fractions
    loops: tuple          # SnapshotLoop, the outer loop first


@dataclass(frozen=True)
class ExactSnapshot:
    faces: tuple
    positions: Mapping    # vertex id -> (Fraction, Fraction, Fraction)
    edges: Mapping        # edge id -> (start vertex id, end vertex id)

    @property
    def face_ids(self) -> tuple:
        return tuple(face.id for face in self.faces)

    def canonical(self, meter: Meter | None = None) -> dict:
        """The canonical exact document; with ``meter`` one unit per scalar/array element, as it is produced."""
        def tick(n: int = 1) -> None:
            if meter is not None:
                meter.unit("serialize_element", n)

        def pair(value: Fraction) -> list:
            tick(2)
            return [value.numerator, value.denominator]

        def point(p) -> list:
            tick(1)
            return [pair(c) for c in p]

        faces = []
        edge_uses: dict = {}
        vertex_faces: dict = {}
        vertex_edges: dict = {}
        for face in self.faces:
            loops = []
            for loop in face.loops:
                items = []
                for (edge, forward), vertex in zip(loop.edges, loop.vertices):
                    tick(3)
                    items.append([edge, forward, vertex])
                    edge_uses.setdefault(edge, []).append([face.id, forward])
                    vertex_faces.setdefault(vertex, set()).add(face.id)
                    vertex_edges.setdefault(vertex, set()).add(edge)
                tick(1)
                loops.append(items)
            tick(3)
            faces.append({"id": face.id, "support": [point(p) for p in face.support], "loops": loops})
        vertices = []
        for vid in sorted(self.positions):
            tick(1)
            vertices.append([vid, point(self.positions[vid])])
        edges = []
        for edge in sorted(edge_uses):
            tick(3 + 2 * len(edge_uses[edge]))
            edges.append([edge, *self.edges[edge], edge_uses[edge]])
        incidence = []
        for vid in sorted(vertex_faces):
            tick(1 + len(vertex_faces[vid]) + len(vertex_edges[vid]))
            incidence.append([vid, sorted(vertex_faces[vid]), sorted(vertex_edges[vid])])
        return {"faces": faces, "vertices": vertices, "edges": edges, "vertex_incidence": incidence}

    def digest(self, meter: Meter) -> str:
        return sha256_chunks(canonical_json(self.canonical(meter)), meter)


def _exact(value, meter: Meter) -> Fraction:
    number = float(value)
    if not math.isfinite(number):
        raise refuse(RefusalCategory.NONFINITE_INPUT, "a coordinate is not finite")
    return meter.fit(Fraction(number))


def _exact_point(values, meter: Meter) -> Point:
    return tuple(_exact(value, meter) for value in values)


def collect_snapshot(model, faces: Iterable[int], meter: Meter) -> ExactSnapshot:
    """Read ``faces`` of ``model`` into exact rational loops (no admission decisions beyond the types)."""
    face_ids = []
    for face_id in faces:
        if isinstance(face_id, bool) or not hasattr(face_id, "__index__"):
            raise refuse(RefusalCategory.INVALID_QUERY, f"face id {face_id!r} is not an integer")
        face_ids.append(int(face_id))
    if not face_ids:
        raise refuse(RefusalCategory.NOT_CLOSED, "the face set is empty")
    positions: dict = {}
    edges: dict = {}
    result = []
    for face_id in face_ids:
        meter.unit("face_read")
        face = model.faces.get(face_id)
        if face is None:
            raise refuse(RefusalCategory.INVALID_QUERY, f"face {face_id} is not in the model")
        if not isinstance(face.surface, Plane):
            raise refuse(RefusalCategory.UNSUPPORTED_SURFACE,
                         f"face {face_id} has no Plane support ({type(face.surface).__name__})")
        support = (_exact_point(face.surface.origin, meter), _exact_point(face.surface.u_vector, meter),
                   _exact_point(face.surface.v_vector, meter))
        loops = []
        for loop in (face.loop, *face.holes):
            items, starts = [], []
            for item in loop:
                edge = model.edges.get(item.edge)
                if edge is None:
                    raise refuse(RefusalCategory.INVALID_LOOP, f"face {face_id} uses unknown edge {item.edge}")
                if item.edge not in edges:
                    meter.unit("edge_read")
                    if not isinstance(edge.curve, Straight):
                        raise refuse(RefusalCategory.UNSUPPORTED_SURFACE,
                                     f"edge {item.edge} of face {face_id} is not straight")
                    edges[item.edge] = (int(edge.start), int(edge.end))
                start = edges[item.edge][0] if item.forward else edges[item.edge][1]
                for vertex in edges[item.edge]:
                    if vertex not in positions:
                        meter.unit("vertex_read")
                        record = model.vertices.get(vertex)
                        if record is None:
                            raise refuse(RefusalCategory.INVALID_LOOP,
                                         f"edge {item.edge} of face {face_id} ends at the missing vertex {vertex}",
                                         KeyError(vertex))
                        positions[vertex] = _exact_point(record.position, meter)
                items.append((int(item.edge), bool(item.forward)))
                starts.append(start)
            loops.append(SnapshotLoop(tuple(items), tuple(starts)))
        result.append(SnapshotFace(face_id, support, tuple(loops)))
    return ExactSnapshot(tuple(result), MappingProxyType(positions), MappingProxyType(edges))
