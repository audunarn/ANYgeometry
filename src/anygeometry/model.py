"""The neutral structural-surface topology and geometry model.

Modelling is bottom-up and point-driven, which is the paradigm the mapped
mesher wants anyway:

    points  ->  lines between points  ->  faces bounded by line loops
                                      ->  beams carried on lines

Faces may carry explicit planar, cylindrical, conical or ruled surfaces, or a
four-boundary Coons patch.  Four mapped corners remain optional compatibility
metadata; they are not a restriction on neutral face topology.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np

from .curves import (
    Arc,
    ArcFrame,
    CurveShape,
    Spline,
    Straight,
    arc_frame,
    arc_tangent,
    sample_arc,
    sample_straight,
    sample_spline,
    spline_tangent,
    straight_tangent,
)
from .errors import GeometryError
from .entities import Edge, EntityRef, Face, OrientedEdge, Vertex
from .features import FeatureHistory, FeatureRegistry, RegenerationReport
from .surfaces import CoonsSurface, Plane, Surface, closest_uv

# GeometryError is re-exported for temporary compatibility imports.
__all__ = ["GeometryError", "GeometryModel"]


def _rotate_about_axis(
    point: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
    angle: float,
) -> np.ndarray:
    """Rotate a point about an arbitrary axis (Rodrigues' formula)."""

    offset = np.asarray(point, dtype=float) - origin
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return (
        origin
        + offset * cosine
        + np.cross(direction, offset) * sine
        + direction * float(direction @ offset) * (1.0 - cosine)
    )


class GeometryModel:
    """A container of vertices, edges and faces with persistent IDs."""

    def __init__(self) -> None:
        self.vertices: Dict[int, Vertex] = {}
        self.edges: Dict[int, Edge] = {}
        self.faces: Dict[int, Face] = {}
        self._next_id: Dict[str, int] = {"vertex": 1, "edge": 1, "face": 1}
        self._arc_cache: Dict[int, Tuple[int, ArcFrame]] = {}
        # What each removed entity was replaced by, so attributes attached to
        # it can follow.  Splitting a line that carries a load must not throw
        # the load away.
        self._replacements: List[Tuple[EntityRef, Tuple[EntityRef, ...]]] = []
        self._replacement_history: Dict[EntityRef, Tuple[EntityRef, ...]] = {}
        self.groups: Dict[str, Set[EntityRef]] = {}
        self.tags: Dict[EntityRef, Set[str]] = {}
        # Persistent design intent is separate from the materialized topology,
        # but travels with its owner so a geometry document cannot silently
        # lose its editable feature tree.
        self.features = FeatureHistory()

    # ------------------------------------------------------------------
    # replacement log
    # ------------------------------------------------------------------
    def begin_replacement_log(self) -> None:
        """Start recording what replaces what, for the duration of an edit."""

        self._replacements = []

    def replacement_log(self) -> List[Tuple[EntityRef, Tuple[EntityRef, ...]]]:
        """Entities removed during the current edit, and what took their place."""

        return list(self._replacements)

    def record_replacement(
        self, old: EntityRef, new: Sequence[EntityRef]
    ) -> None:
        """Note that one entity has been superseded by others."""

        replacements = tuple(new)
        if (
            old.kind not in self._next_id
            or old.id <= 0
            or old.id >= self._next_id[old.kind]
        ):
            raise GeometryError(f"replacement history references missing entity {old}")
        if (old.kind, old.id) in self.entity_keys():
            raise GeometryError(
                f"cannot record replacement for surviving entity {old}"
            )
        if any(reference.kind not in self._next_id for reference in replacements):
            raise GeometryError("replacement history contains an invalid entity kind")
        if any(reference.kind != old.kind for reference in replacements):
            raise GeometryError("replacement history cannot change entity kind")
        if old in replacements:
            raise GeometryError("an entity cannot replace itself")
        if old in self._replacement_history:
            raise GeometryError(f"replacement history already exists for {old}")
        keys = self.entity_keys()
        for reference in replacements:
            if reference.id <= 0 or reference.id >= self._next_id[reference.kind]:
                raise GeometryError(
                    f"replacement history references missing entity {reference}"
                )
            if (
                (reference.kind, reference.id) not in keys
                and reference not in self._replacement_history
            ):
                raise GeometryError(
                    f"replacement history has an unresolved descendant {reference}"
                )

        def reaches_old(reference: EntityRef, seen: Set[EntityRef]) -> bool:
            if reference == old:
                return True
            if reference in seen:
                return False
            seen.add(reference)
            return any(
                reaches_old(descendant, seen)
                for descendant in self._replacement_history.get(reference, ())
            )

        if any(reaches_old(reference, set()) for reference in replacements):
            raise GeometryError(f"replacement history contains a cycle at {old}")
        self._replacements.append((old, replacements))
        self._replacement_history[old] = replacements
        for name, members in self.groups.items():
            if old in members:
                members.discard(old)
                members.update(replacements)
        inherited = self.tags.pop(old, set())
        for replacement in replacements:
            self.tags.setdefault(replacement, set()).update(inherited)

    def record_replacements_atomic(
        self,
        entries: Iterable[Tuple[EntityRef, Sequence[EntityRef]]],
    ) -> None:
        """Append a complete replacement graph atomically.

        Regeneration may need to reconnect a retained historical graph and a
        set of old-materialization to new-materialization transitions in one
        operation.  Adding those arcs one at a time can temporarily leave a
        descendant unresolved even though the final batch is valid, so this
        method validates and commits the combined graph as a unit.
        """

        normalized: List[Tuple[EntityRef, Tuple[EntityRef, ...]]] = []
        supplied: Dict[EntityRef, Tuple[EntityRef, ...]] = {}
        for old, descendants in entries:
            made = tuple(descendants)
            previous = supplied.get(old)
            if previous is not None and previous != made:
                raise GeometryError(f"conflicting replacement entries for {old}")
            supplied[old] = made
        for old, descendants in supplied.items():
            current = self._replacement_history.get(old)
            if current is not None:
                if current != descendants:
                    raise GeometryError(
                        f"replacement history already exists for {old}"
                    )
                continue
            if (old.kind, old.id) in self.entity_keys():
                raise GeometryError(
                    f"cannot record replacement for surviving entity {old}"
                )
            if (
                old.kind not in self._next_id
                or old.id <= 0
                or old.id >= self._next_id[old.kind]
            ):
                raise GeometryError(
                    f"replacement history references missing entity {old}"
                )
            if any(item.kind != old.kind for item in descendants):
                raise GeometryError("replacement history cannot change entity kind")
            if old in descendants:
                raise GeometryError("an entity cannot replace itself")
            normalized.append((old, descendants))

        previous_history = dict(self._replacement_history)
        previous_log = list(self._replacements)
        previous_groups = {
            name: set(items) for name, items in self.groups.items()
        }
        previous_tags = {
            reference: set(values) for reference, values in self.tags.items()
        }
        try:
            self._replacement_history.update(normalized)
            errors = self._validate_replacement_history()
            if errors:
                raise GeometryError("; ".join(errors))
            for old, descendants in normalized:
                self._replacements.append((old, descendants))
                for members in self.groups.values():
                    if old in members:
                        members.discard(old)
                        members.update(descendants)
                inherited = self.tags.pop(old, set())
                for descendant in descendants:
                    self.tags.setdefault(descendant, set()).update(inherited)
        except Exception:
            self._replacement_history = previous_history
            self._replacements = previous_log
            self.groups = previous_groups
            self.tags = previous_tags
            raise

    def replacement_history(self) -> Dict[EntityRef, Tuple[EntityRef, ...]]:
        """Complete supersession map retained across edit transactions."""

        return dict(self._replacement_history)

    def resolve_ref(self, reference: EntityRef) -> Tuple[EntityRef, ...]:
        """Resolve a stale reference to its current surviving descendants."""

        pending = [reference]
        resolved: List[EntityRef] = []
        seen: Set[EntityRef] = set()
        while pending:
            current = pending.pop(0)
            if current in seen:
                continue
            seen.add(current)
            replacements = self._replacement_history.get(current)
            if replacements is None:
                if (current.kind, current.id) in self.entity_keys():
                    resolved.append(current)
            else:
                pending.extend(replacements)
        return tuple(resolved)

    def add_to_group(self, name: str, references: Iterable[EntityRef]) -> None:
        """Add checked entities to a persistent semantic group."""

        group = self.groups.setdefault(str(name), set())
        for reference in references:
            self.entity_ref(reference.kind, reference.id)
            group.add(reference)

    def group(self, name: str, *, resolve: bool = True) -> Tuple[EntityRef, ...]:
        members = self.groups.get(str(name), set())
        if not resolve:
            return tuple(sorted(members, key=lambda ref: (ref.kind, ref.id)))
        current = {item for member in members for item in self.resolve_ref(member)}
        return tuple(sorted(current, key=lambda ref: (ref.kind, ref.id)))

    def tag(self, reference: EntityRef, *values: str) -> None:
        self.entity_ref(reference.kind, reference.id)
        self.tags.setdefault(reference, set()).update(str(value) for value in values)

    def tags_for(self, reference: EntityRef) -> Tuple[str, ...]:
        return tuple(sorted(self.tags.get(reference, set())))

    # ------------------------------------------------------------------
    # identity
    # ------------------------------------------------------------------
    def _allocate(self, kind: str) -> int:
        entity_id = self._next_id[kind]
        store = {"vertex": self.vertices, "edge": self.edges, "face": self.faces}[kind]
        if entity_id in store:
            raise GeometryError(
                f"{kind} ID {entity_id} is already in use; the ID counter and "
                "the model have gone out of step"
            )
        self._next_id[kind] = entity_id + 1
        return entity_id

    def id_state(self) -> Dict[str, int]:
        """Snapshot the ID counters.

        Undo restores this so a redone operation re-allocates exactly the same
        IDs, which keeps every attribute reference valid across undo and redo.
        """

        return dict(self._next_id)

    def restore_id_state(self, state: Mapping[str, int]) -> None:
        self._next_id = dict(state)

    def reserve_id_state(self, state: Mapping[str, int]) -> None:
        """Raise allocator floors without ever moving a counter backwards."""

        for kind in ("vertex", "edge", "face"):
            if kind not in state:
                raise GeometryError(f"missing {kind} ID counter")
            value = int(state[kind])
            if value < 1:
                raise GeometryError(f"{kind} ID counter must be positive")
            self._next_id[kind] = max(self._next_id[kind], value)

    def topology_snapshot(self) -> Dict[str, object]:
        """Cheap snapshot of the whole topology, for undo.

        Entity objects are referenced rather than copied; face loops and
        corners are captured as the tuples they are, because operations like
        splitting an edge rewrite them in place.
        """

        return {
            "vertices": dict(self.vertices),
            "edges": dict(self.edges),
            "faces": dict(self.faces),
            "vertex_state": {
                vertex_id: vertex.position.copy()
                for vertex_id, vertex in self.vertices.items()
            },
            "edge_state": {
                edge_id: (edge.start, edge.end, edge.curve)
                for edge_id, edge in self.edges.items()
            },
            "face_state": {
                face_id: (
                    face.loop,
                    face.corners,
                    deepcopy(face.metadata),
                    face.holes,
                    deepcopy(face.surface),
                )
                for face_id, face in self.faces.items()
            },
            "ids": dict(self._next_id),
            "groups": {name: set(members) for name, members in self.groups.items()},
            "tags": {reference: set(values) for reference, values in self.tags.items()},
            "replacement_history": dict(self._replacement_history),
            "replacements": list(self._replacements),
        }

    def design_snapshot(self) -> Dict[str, object]:
        """Snapshot topology and persistent feature definitions for undo."""

        return {
            "topology": self.topology_snapshot(),
            "features": self.features.snapshot(),
        }

    def restore_topology(self, snapshot: Mapping[str, object]) -> None:
        """Put the model back exactly as ``topology_snapshot`` found it."""

        self.vertices.clear()
        self.vertices.update(snapshot["vertices"])  # type: ignore[arg-type]
        self.edges.clear()
        self.edges.update(snapshot["edges"])  # type: ignore[arg-type]
        self.faces.clear()
        self.faces.update(snapshot["faces"])  # type: ignore[arg-type]
        for vertex_id, position in snapshot.get("vertex_state", {}).items():  # type: ignore[union-attr]
            self.vertices[vertex_id].position = np.asarray(position, dtype=float).copy()
        for edge_id, (start, end, curve) in snapshot.get("edge_state", {}).items():  # type: ignore[union-attr]
            edge = self.edges[edge_id]
            edge.start = start
            edge.end = end
            edge.curve = deepcopy(curve)
        for face_id, (loop, corners, metadata, holes, surface) in snapshot["face_state"].items():  # type: ignore[union-attr]
            self.faces[face_id].loop = loop
            self.faces[face_id].corners = corners
            self.faces[face_id].metadata = deepcopy(metadata)
            self.faces[face_id].holes = holes
            self.faces[face_id].surface = deepcopy(surface)
        self._next_id = dict(snapshot["ids"])  # type: ignore[arg-type]
        self.groups = {name: set(members) for name, members in snapshot.get("groups", {}).items()}  # type: ignore[union-attr]
        self.tags = {reference: set(values) for reference, values in snapshot.get("tags", {}).items()}  # type: ignore[union-attr]
        self._replacement_history = dict(snapshot.get("replacement_history", {}))  # type: ignore[arg-type]
        self._replacements = list(snapshot.get("replacements", []))  # type: ignore[arg-type]
        self._arc_cache.clear()

    def restore_design(self, snapshot: Mapping[str, object]) -> None:
        """Restore a snapshot made by :meth:`design_snapshot`."""

        self.restore_topology(snapshot["topology"])  # type: ignore[arg-type]
        self.features.restore(snapshot["features"])  # type: ignore[arg-type]

    def clone(self, *, include_features: bool = True) -> "GeometryModel":
        """Return a deep, independently mutable geometry copy."""

        from .serialization import from_dict, to_dict

        return from_dict(to_dict(self, include_features=include_features))

    def insert_model(
        self,
        source: "GeometryModel",
        *,
        matrix: Sequence[Sequence[float]] | None = None,
        group_prefix: str | None = None,
    ):
        """Insert a flattened topology copy with fresh destination IDs."""

        from .editing import insert_model

        return insert_model(
            self, source, matrix=matrix, group_prefix=group_prefix
        )

    def regenerate_features(
        self, registry: FeatureRegistry | None = None
    ) -> RegenerationReport:
        """Replay persistent features with neutral executors by default."""

        if registry is None:
            from .features import builtin_feature_registry

            registry = builtin_feature_registry()
        return self.features.regenerate(self, registry)

    def entity_keys(self) -> Set[Tuple[str, int]]:
        """Every entity in the model, as ``(kind, id)`` pairs."""

        return (
            {("vertex", key) for key in self.vertices}
            | {("edge", key) for key in self.edges}
            | {("face", key) for key in self.faces}
        )

    # ------------------------------------------------------------------
    # dependencies and removal
    # ------------------------------------------------------------------
    def edges_using_vertex(self, vertex_id: int) -> List[int]:
        """Edges that reference a vertex, as an end point or as an arc's via."""

        return [
            edge.id
            for edge in self.edges.values()
            if vertex_id in (edge.start, edge.end)
            or (isinstance(edge.curve, Arc) and edge.curve.via_vertex == vertex_id)
            or (
                isinstance(edge.curve, Spline)
                and vertex_id in edge.curve.control_vertices
            )
        ]

    def faces_using_edge(self, edge_id: int) -> List[int]:
        return [
            face.id
            for face in self.faces.values()
            if any(
                item.edge == edge_id
                for loop in (face.loop,) + tuple(face.holes)
                for item in loop
            )
        ]

    def remove_face(self, face_id: int, *, record: bool = True) -> None:
        self._require_face(face_id)
        del self.faces[face_id]
        if record:
            self.record_replacement(EntityRef("face", face_id), ())

    def remove_edge(self, edge_id: int, *, record: bool = True) -> None:
        self._require_edge(edge_id)
        users = self.faces_using_edge(edge_id)
        if users:
            raise GeometryError(
                f"cannot remove edge {edge_id}: it bounds face(s) {sorted(users)}"
            )
        del self.edges[edge_id]
        self._arc_cache.pop(edge_id, None)
        if record:
            self.record_replacement(EntityRef("edge", edge_id), ())

    def remove_vertex(self, vertex_id: int, *, record: bool = True) -> None:
        self._require_vertex(vertex_id)
        users = self.edges_using_vertex(vertex_id)
        if users:
            raise GeometryError(
                f"cannot remove point {vertex_id}: it is used by edge(s) "
                f"{sorted(users)}"
            )
        del self.vertices[vertex_id]
        if record:
            self.record_replacement(EntityRef("vertex", vertex_id), ())

    def remove_entities(self, keys: Iterable[Tuple[str, int]]) -> None:
        """Remove a set of entities, innermost dependency last."""

        remaining = list(keys)
        order = {"face": 0, "edge": 1, "vertex": 2}
        for kind, entity_id in sorted(remaining, key=lambda k: (order[k[0]], -k[1])):
            if kind == "face":
                self.remove_face(entity_id)
            elif kind == "edge":
                self.remove_edge(entity_id)
            else:
                self.remove_vertex(entity_id)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def add_point(self, x: float, y: float, z: float = 0.0) -> int:
        """Place a point and return its vertex ID."""

        position = np.asarray((x, y, z), dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise GeometryError("point coordinates must be a finite 3-vector")
        vertex_id = self._allocate("vertex")
        self.vertices[vertex_id] = Vertex(
            id=vertex_id, position=position
        )
        return vertex_id

    def add_points(self, positions: Iterable[Sequence[float]]) -> List[int]:
        """Place several points at once."""

        return [self.add_point(*np.asarray(p, dtype=float)) for p in positions]

    def add_line(self, start: int, end: int) -> int:
        """Connect two points with a straight line."""

        self._require_vertex(start)
        self._require_vertex(end)
        if start == end:
            raise GeometryError("a line needs two distinct points")
        if float(
            np.linalg.norm(
                self.vertices[end].position - self.vertices[start].position
            )
        ) <= 0.0:
            raise GeometryError("a line needs two spatially distinct points")
        return self._add_edge(start, end, Straight())

    def add_arc(self, start: int, via: int, end: int) -> int:
        """Connect two points with a circular arc through a third point."""

        self._require_vertex(start)
        self._require_vertex(via)
        self._require_vertex(end)
        if len({start, via, end}) != 3:
            raise GeometryError("an arc needs three distinct points")
        # Resolve now so a bad arc is rejected at modelling time rather than
        # at mesh time, where the diagnostic would be far from the cause.
        arc_frame(
            self.vertices[start].position,
            self.vertices[via].position,
            self.vertices[end].position,
        )
        return self._add_edge(start, end, Arc(via_vertex=via))

    def add_spline(
        self, start: int, control_vertices: Sequence[int], end: int
    ) -> int:
        """Connect two vertices with a lightweight Bezier spline."""

        self._require_vertex(start)
        self._require_vertex(end)
        controls = tuple(int(vertex) for vertex in control_vertices)
        for vertex in controls:
            self._require_vertex(vertex)
        if start == end or len({start, *controls, end}) < 2:
            raise GeometryError("a spline needs two distinct end points")
        return self._add_edge(start, end, Spline(controls))

    def add_polyline(self, vertex_ids: Sequence[int], close: bool = False) -> List[int]:
        """Connect a run of points with straight lines."""

        ids = list(vertex_ids)
        if len(ids) < 2:
            raise GeometryError("a polyline needs at least two points")
        pairs = list(zip(ids, ids[1:]))
        if close:
            pairs.append((ids[-1], ids[0]))
        return [self.add_line(a, b) for a, b in pairs]

    def _add_edge(self, start: int, end: int, curve: CurveShape) -> int:
        edge_id = self._allocate("edge")
        self.edges[edge_id] = Edge(id=edge_id, start=start, end=end, curve=curve)
        return edge_id

    def add_face(
        self,
        edge_ids: Sequence[int],
        corners: Sequence[int] | None = None,
        *,
        surface: Surface | None = None,
    ) -> int:
        """Create a plate bounded by a closed loop of edges.

        The edges may be given in any order and any direction; the loop is
        ordered here.  ``corners`` optionally overrides the four loop indices
        where the sides begin, for faces whose corners are not obvious from
        boundary turn angle.
        """

        loop = self._order_loop(edge_ids)
        if corners is None:
            resolved = self._detect_corners(loop) if len(loop) >= 4 else ()
        else:
            resolved = self._validate_corners(tuple(int(c) for c in corners), len(loop))
        return self._add_face_from_loop(loop, resolved, surface=surface)

    def order_loop(self, edge_ids: Sequence[int]) -> Tuple[OrientedEdge, ...]:
        """Order an unordered edge set into a closed, oriented boundary loop."""

        return self._order_loop(edge_ids)

    def add_face_from_loop(
        self,
        loop: Sequence[OrientedEdge],
        corners: Sequence[int] | None = None,
        *,
        surface: Surface | None = None,
    ) -> int:
        """Create a face from an explicit oriented loop and corner positions.

        Used by the decomposition tools, which know exactly where the corners
        belong and must not leave it to turn-angle detection.
        """

        ordered = tuple(loop)
        if len(ordered) < 3:
            raise GeometryError("a face needs at least three edges")
        for item in ordered:
            self._require_edge(item.edge)
        for current, following in zip(ordered, ordered[1:] + ordered[:1]):
            if self.oriented_end_vertex(current) != self.oriented_start_vertex(
                following
            ):
                raise GeometryError(
                    f"loop is not continuous at edge {following.edge}"
                )
        return self._add_face_from_loop(
            ordered,
            () if corners is None else self._validate_corners(
                tuple(int(c) for c in corners), len(ordered)
            ),
            surface=surface,
        )

    def _add_face_from_loop(
        self,
        loop: Tuple[OrientedEdge, ...],
        corners: Tuple[int, ...],
        *,
        surface: Surface | None = None,
    ) -> int:
        face_id = self._allocate("face")
        self.faces[face_id] = Face(
            id=face_id,
            loop=loop,
            corners=corners,
            surface=surface or (CoonsSurface() if len(corners) == 4 else None),
        )
        return face_id

    def add_plate(self, vertex_ids: Sequence[int]) -> int:
        """Create a plate directly from an ordered ring of points.

        Convenience for the common case: the lines are created too.
        """

        edge_ids = self.add_polyline(vertex_ids, close=True)
        face_id = self.add_face(edge_ids)
        corners = [self.vertex_position(vertex_id) for vertex_id in vertex_ids[:4]]
        if len(corners) >= 3:
            origin = corners[0]
            u_vector = corners[1] - origin
            v_vector = corners[-1] - origin
            if float(np.linalg.norm(np.cross(u_vector, v_vector))) > 1.0e-14:
                self.faces[face_id].surface = Plane(origin, u_vector, v_vector)
        return face_id

    def validate_topology(self) -> Tuple[str, ...]:
        """Return deterministic topology errors without mutating the model."""

        errors: List[str] = []
        for vertex_id, vertex in sorted(self.vertices.items()):
            if vertex.id != vertex_id:
                errors.append(f"vertex key {vertex_id} does not match ID {vertex.id}")
            if vertex.position.shape != (3,) or not np.all(np.isfinite(vertex.position)):
                errors.append(f"vertex {vertex_id} has invalid coordinates")
        for edge_id, edge in sorted(self.edges.items()):
            if edge.id != edge_id:
                errors.append(f"edge key {edge_id} does not match ID {edge.id}")
            for vertex_id in (edge.start, edge.end):
                if vertex_id not in self.vertices:
                    errors.append(f"edge {edge_id} references missing vertex {vertex_id}")
            if edge.start == edge.end:
                errors.append(f"edge {edge_id} has coincident topology endpoints")
            if edge.start in self.vertices and edge.end in self.vertices:
                start = self.vertices[edge.start].position
                end = self.vertices[edge.end].position
                scale = max(
                    float(np.linalg.norm(start)),
                    float(np.linalg.norm(end)),
                    1.0,
                )
                if float(np.linalg.norm(end - start)) <= 1.0e-12 * scale:
                    errors.append(f"edge {edge_id} has zero geometric length")
            if isinstance(edge.curve, Arc) and edge.curve.via_vertex not in self.vertices:
                errors.append(
                    f"edge {edge_id} references missing arc via vertex {edge.curve.via_vertex}"
                )
            elif (
                isinstance(edge.curve, Arc)
                and edge.start in self.vertices
                and edge.end in self.vertices
                and edge.curve.via_vertex in self.vertices
            ):
                try:
                    arc_frame(
                        self.vertices[edge.start].position,
                        self.vertices[edge.curve.via_vertex].position,
                        self.vertices[edge.end].position,
                    )
                except (ValueError, GeometryError) as error:
                    errors.append(f"edge {edge_id} has invalid arc geometry: {error}")
            if isinstance(edge.curve, Spline):
                for vertex_id in edge.curve.control_vertices:
                    if vertex_id not in self.vertices:
                        errors.append(
                            f"edge {edge_id} references missing spline control vertex {vertex_id}"
                        )
        for face_id, face in sorted(self.faces.items()):
            if face.id != face_id:
                errors.append(f"face key {face_id} does not match ID {face.id}")
            if face.corners:
                if (
                    len(face.corners) != 4
                    or len(set(face.corners)) != 4
                    or tuple(sorted(face.corners)) != face.corners
                    or any(index < 0 or index >= len(face.loop) for index in face.corners)
                ):
                    errors.append(f"face {face_id} has invalid mapped corners")
            loops = (face.loop,) + tuple(face.holes)
            for loop_index, loop in enumerate(loops):
                if not loop:
                    errors.append(f"face {face_id} has an empty loop {loop_index}")
                    continue
                minimum = 3 if loop_index == 0 else 2
                if len(loop) < minimum:
                    errors.append(
                        f"face {face_id} loop {loop_index} needs at least "
                        f"{minimum} edges"
                    )
                for item in loop:
                    if item.edge not in self.edges:
                        errors.append(f"face {face_id} references missing edge {item.edge}")
                if len({item.edge for item in loop}) != len(loop):
                    errors.append(f"face {face_id} loop {loop_index} repeats an edge")
                existing = [item for item in loop if item.edge in self.edges]
                for current, following in zip(existing, existing[1:] + existing[:1]):
                    if self.oriented_end_vertex(current) != self.oriented_start_vertex(following):
                        errors.append(f"face {face_id} loop {loop_index} is discontinuous")
                        break
            all_edges = [item.edge for loop in loops for item in loop]
            if len(set(all_edges)) != len(all_edges):
                errors.append(
                    f"face {face_id} reuses an edge across outer and inner loops"
                )
            if not any(
                message.startswith(f"face {face_id} ") for message in errors
            ):
                errors.extend(self._validate_face_geometry(face_id))
        keys = self.entity_keys()
        for name, members in sorted(self.groups.items()):
            for reference in sorted(members, key=lambda item: (item.kind, item.id)):
                if (reference.kind, reference.id) not in keys and reference not in self._replacement_history:
                    errors.append(f"group {name!r} references missing entity {reference}")
        for reference in sorted(self.tags, key=lambda item: (item.kind, item.id)):
            if (reference.kind, reference.id) not in keys and reference not in self._replacement_history:
                errors.append(f"tags reference missing entity {reference}")
        errors.extend(self._validate_replacement_history())
        return tuple(errors)

    @staticmethod
    def _segments_intersect_2d(
        first_start: np.ndarray,
        first_end: np.ndarray,
        second_start: np.ndarray,
        second_end: np.ndarray,
        tolerance: float,
    ) -> bool:
        """Whether two bounded planar segments touch or cross."""

        def cross(first: np.ndarray, second: np.ndarray) -> float:
            return float(first[0] * second[1] - first[1] * second[0])

        first = first_end - first_start
        second = second_end - second_start
        denominator = cross(first, second)
        offset = second_start - first_start
        if abs(denominator) <= tolerance:
            if abs(cross(offset, first)) > tolerance:
                return False
            axis = int(np.argmax(np.abs(first)))
            if abs(float(first[axis])) <= tolerance:
                return float(np.linalg.norm(first_start - second_start)) <= tolerance
            interval = sorted(
                (
                    float((second_start[axis] - first_start[axis]) / first[axis]),
                    float((second_end[axis] - first_start[axis]) / first[axis]),
                )
            )
            return max(0.0, interval[0]) <= min(1.0, interval[1]) + tolerance
        first_parameter = cross(offset, second) / denominator
        second_parameter = cross(offset, first) / denominator
        return (
            -tolerance <= first_parameter <= 1.0 + tolerance
            and -tolerance <= second_parameter <= 1.0 + tolerance
        )

    @classmethod
    def _polygon_self_intersects(
        cls, polygon: np.ndarray, tolerance: float = 1.0e-10
    ) -> bool:
        count = len(polygon)
        for first in range(count):
            first_next = (first + 1) % count
            for second in range(first + 1, count):
                second_next = (second + 1) % count
                if (
                    first == second
                    or first_next == second
                    or second_next == first
                ):
                    continue
                if cls._segments_intersect_2d(
                    polygon[first],
                    polygon[first_next],
                    polygon[second],
                    polygon[second_next],
                    tolerance,
                ):
                    return True
        return False

    def _validation_loop_points(
        self, loop: Sequence[OrientedEdge]
    ) -> np.ndarray:
        points: List[np.ndarray] = []
        for item in loop:
            edge = self.edges[item.edge]
            count = 3 if isinstance(edge.curve, Straight) else 17
            samples = self.sample_edge(item.edge, np.linspace(0.0, 1.0, count))
            if not item.forward:
                samples = samples[::-1]
            points.extend(samples[:-1])
        return np.asarray(points, dtype=float)

    def _validate_face_geometry(self, face_id: int) -> List[str]:
        """Validate trim geometry after its topology references are known valid."""

        face = self.faces[face_id]
        loops = (face.loop,) + tuple(face.holes)
        points_3d = [self._validation_loop_points(loop) for loop in loops]
        if any(len(points) < 3 for points in points_3d):
            return []

        combined = np.vstack(points_3d)
        origin = combined.mean(axis=0)
        _values, singular, vectors = np.linalg.svd(combined - origin)
        scale = max(float(singular[0]), 1.0)
        planar = len(singular) < 3 or float(singular[-1]) <= 1.0e-8 * scale
        surface = face.surface
        if surface is not None and not (
            isinstance(surface, CoonsSurface) and not surface.has_boundaries
        ):
            try:
                for point in combined:
                    uv = surface.local_uv(point)
                    projected = np.asarray(surface.evaluate(*uv), dtype=float)
                    point_scale = max(float(np.linalg.norm(point)), 1.0)
                    if (
                        float(np.linalg.norm(projected - point))
                        > 1.0e-7 * point_scale
                    ):
                        return [
                            f"face {face_id} boundary is inconsistent with its explicit surface"
                        ]
            except (ValueError, GeometryError, np.linalg.LinAlgError) as error:
                return [f"face {face_id} has invalid surface geometry: {error}"]
        if planar:
            polygons = [
                np.column_stack(
                    ((points - origin) @ vectors[0], (points - origin) @ vectors[1])
                )
                for points in points_3d
            ]
        elif surface is not None and not (
            isinstance(surface, CoonsSurface) and not surface.has_boundaries
        ):
            try:
                polygons = [
                    np.asarray([surface.local_uv(point) for point in points])
                    for points in points_3d
                ]
            except (ValueError, GeometryError, np.linalg.LinAlgError) as error:
                return [f"face {face_id} has invalid surface geometry: {error}"]
        elif len(face.corners) == 4:
            try:
                polygons = [
                    np.asarray([self.face_local_uv(face_id, point) for point in points])
                    for points in points_3d
                ]
            except (ValueError, GeometryError, np.linalg.LinAlgError) as error:
                return [f"face {face_id} has invalid Coons geometry: {error}"]
        else:
            return [f"face {face_id} is non-planar and has no explicit surface"]

        result: List[str] = []
        for index, polygon in enumerate(polygons):
            following = np.roll(polygon, -1, axis=0)
            area = 0.5 * abs(
                float(
                    np.sum(
                        polygon[:, 0] * following[:, 1]
                        - following[:, 0] * polygon[:, 1]
                    )
                )
            )
            scale = max(float(np.ptp(polygon, axis=0).max()), 1.0)
            if area <= 1.0e-12 * scale * scale:
                result.append(f"face {face_id} loop {index} has zero area")
            if self._polygon_self_intersects(polygon):
                result.append(f"face {face_id} loop {index} self-intersects")

        outer = polygons[0]
        for index, hole in enumerate(polygons[1:], start=1):
            if not all(
                self._point_in_polygon(point, outer, include_boundary=False)
                for point in hole
            ):
                result.append(
                    f"face {face_id} hole {index} is not strictly inside the outer loop"
                )
            if self._polygons_intersect(outer, hole):
                result.append(f"face {face_id} hole {index} intersects the outer loop")
        for first in range(1, len(polygons)):
            for second in range(first + 1, len(polygons)):
                if (
                    self._polygons_intersect(polygons[first], polygons[second])
                    or self._point_in_polygon(
                        polygons[first][0], polygons[second], include_boundary=True
                    )
                    or self._point_in_polygon(
                        polygons[second][0], polygons[first], include_boundary=True
                    )
                ):
                    result.append(
                        f"face {face_id} holes {first} and {second} overlap"
                    )
        return result

    @classmethod
    def _polygons_intersect(
        cls, first: np.ndarray, second: np.ndarray
    ) -> bool:
        for first_index in range(len(first)):
            for second_index in range(len(second)):
                if cls._segments_intersect_2d(
                    first[first_index],
                    first[(first_index + 1) % len(first)],
                    second[second_index],
                    second[(second_index + 1) % len(second)],
                    1.0e-10,
                ):
                    return True
        return False

    def _validate_replacement_history(self) -> List[str]:
        errors: List[str] = []
        keys = self.entity_keys()
        for old, replacements in sorted(
            self._replacement_history.items(),
            key=lambda item: (str(item[0].kind), item[0].id),
        ):
            if old.kind not in self._next_id or old.id <= 0 or old.id >= self._next_id[old.kind]:
                errors.append(f"replacement history references missing entity {old}")
                continue
            if (old.kind, old.id) in keys:
                errors.append(f"replacement history supersedes surviving entity {old}")
            for replacement in replacements:
                if replacement.kind != old.kind:
                    errors.append(
                        f"replacement history changes entity kind from {old} to {replacement}"
                    )
                    continue
                if (
                    replacement.id <= 0
                    or replacement.id >= self._next_id[replacement.kind]
                ):
                    errors.append(
                        f"replacement history references missing entity {replacement}"
                    )
                elif (
                    (replacement.kind, replacement.id) not in keys
                    and replacement not in self._replacement_history
                ):
                    errors.append(
                        "replacement history has an unresolved descendant "
                        f"{replacement}"
                    )

        visiting: Set[EntityRef] = set()
        visited: Set[EntityRef] = set()

        def visit(reference: EntityRef) -> None:
            if reference in visited or reference not in self._replacement_history:
                return
            if reference in visiting:
                errors.append(
                    f"replacement history contains a cycle at {reference}"
                )
                return
            visiting.add(reference)
            for descendant in self._replacement_history[reference]:
                visit(descendant)
            visiting.remove(reference)
            visited.add(reference)

        for reference in self._replacement_history:
            visit(reference)
        return errors

    # ------------------------------------------------------------------
    # operations
    # ------------------------------------------------------------------
    def extrude(
        self, edge_ids: Sequence[int], vector: Sequence[float]
    ) -> List[int]:
        """Sweep edges along a vector, producing one face per edge.

        Shared points between consecutive edges produce shared swept lines, so
        extruding a chain gives a strip of faces that is conformal by
        construction rather than by coincident-node merging.
        """

        offset = np.asarray(vector, dtype=float)
        if offset.shape != (3,):
            raise GeometryError("extrusion vector must be a 3 component vector")
        if float(np.linalg.norm(offset)) <= 0.0:
            raise GeometryError("extrusion vector must be non-zero")

        swept_vertex: Dict[int, int] = {}
        swept_line: Dict[int, int] = {}

        def translated(vertex_id: int) -> int:
            if vertex_id not in swept_vertex:
                position = self.vertices[vertex_id].position + offset
                swept_vertex[vertex_id] = self.add_point(*position)
            return swept_vertex[vertex_id]

        def connector(vertex_id: int) -> int:
            if vertex_id not in swept_line:
                swept_line[vertex_id] = self.add_line(
                    vertex_id, translated(vertex_id)
                )
            return swept_line[vertex_id]

        face_ids: List[int] = []
        for edge_id in edge_ids:
            edge = self._require_edge(edge_id)
            start_top = translated(edge.start)
            end_top = translated(edge.end)
            if isinstance(edge.curve, Arc):
                via_top = translated(edge.curve.via_vertex)
                top_edge = self.add_arc(start_top, via_top, end_top)
            elif isinstance(edge.curve, Spline):
                controls = tuple(translated(vertex) for vertex in edge.curve.control_vertices)
                top_edge = self.add_spline(start_top, controls, end_top)
            else:
                top_edge = self.add_line(start_top, end_top)

            loop = (
                OrientedEdge(edge_id, True),
                OrientedEdge(connector(edge.end), True),
                OrientedEdge(top_edge, False),
                OrientedEdge(connector(edge.start), False),
            )
            face_ids.append(self._add_face_from_loop(loop, (0, 1, 2, 3)))
        return face_ids

    def revolve(
        self,
        edge_ids: Sequence[int],
        axis_point: Sequence[float],
        axis_direction: Sequence[float],
        angle: float,
        segments: int | None = None,
    ) -> List[int]:
        """Sweep edges about an axis, producing one face per edge per segment.

        The swept boundaries are true arcs, so a revolved profile is exact
        rather than faceted.  The sweep is cut into segments of at most a
        quarter turn, which keeps every arc well conditioned.
        """

        origin = np.asarray(axis_point, dtype=float)
        direction = np.asarray(axis_direction, dtype=float)
        norm = float(np.linalg.norm(direction))
        if origin.shape != (3,) or direction.shape != (3,):
            raise GeometryError("the revolve axis needs a point and a direction")
        if norm <= 0.0:
            raise GeometryError("the revolve axis direction must be non-zero")
        direction = direction / norm
        if not np.isfinite(angle) or angle == 0.0:
            raise GeometryError("the revolve angle must be non-zero")

        profile = list(dict.fromkeys(int(e) for e in edge_ids))
        for edge_id in profile:
            self._require_edge(edge_id)
        self._reject_on_axis(profile, origin, direction)

        if segments is None:
            segments = max(1, int(np.ceil(abs(angle) / (0.5 * np.pi) - 1.0e-9)))
        segments = int(segments)
        if segments < 1:
            raise GeometryError("a revolve needs at least one segment")
        step = float(angle) / segments

        # A full turn must land back on the profile it started from, otherwise
        # the result is a slit cylinder with a seam of coincident-but-separate
        # points rather than a closed one.
        closes = abs(abs(float(angle)) - 2.0 * np.pi) <= 1.0e-9
        start_edges = list(profile)
        start_vertices = {edge_id: edge_id for edge_id in profile}
        edge_origin = {edge_id: edge_id for edge_id in profile}
        vertex_origin: Dict[int, int] = {}
        for edge_id in profile:
            edge = self.edges[edge_id]
            controls = (
                (edge.curve.via_vertex,)
                if isinstance(edge.curve, Arc)
                else edge.curve.control_vertices
                if isinstance(edge.curve, Spline)
                else ()
            )
            for vertex_id in (edge.start, edge.end, *controls):
                vertex_origin[vertex_id] = vertex_id
        del start_vertices

        face_ids: List[int] = []
        for index in range(segments):
            closing = closes and index == segments - 1
            profile, made, edge_origin, vertex_origin = self._revolve_once(
                profile,
                origin,
                direction,
                step,
                edge_origin=edge_origin,
                vertex_origin=vertex_origin,
                closing=closing,
            )
            face_ids.extend(made)
        del start_edges
        return face_ids

    def _revolve_once(
        self,
        profile: Sequence[int],
        origin: np.ndarray,
        direction: np.ndarray,
        step: float,
        *,
        edge_origin: Dict[int, int],
        vertex_origin: Dict[int, int],
        closing: bool = False,
    ) -> Tuple[List[int], List[int], Dict[int, int], Dict[int, int]]:
        swept_vertex: Dict[int, int] = {}
        swept_arc: Dict[int, int] = {}

        def rotated(vertex_id: int) -> int:
            if closing:
                # Land back on the point this one was swept from.
                return vertex_origin[vertex_id]
            if vertex_id not in swept_vertex:
                position = _rotate_about_axis(
                    self.vertices[vertex_id].position, origin, direction, step
                )
                swept_vertex[vertex_id] = self.add_point(*position)
            return swept_vertex[vertex_id]

        def connector(vertex_id: int) -> int:
            if vertex_id not in swept_arc:
                midpoint = _rotate_about_axis(
                    self.vertices[vertex_id].position,
                    origin,
                    direction,
                    0.5 * step,
                )
                via = self.add_point(*midpoint)
                swept_arc[vertex_id] = self.add_arc(
                    vertex_id, via, rotated(vertex_id)
                )
            return swept_arc[vertex_id]

        next_profile: List[int] = []
        face_ids: List[int] = []
        next_edge_origin: Dict[int, int] = {}
        next_vertex_origin: Dict[int, int] = {}

        for edge_id in profile:
            edge = self.edges[edge_id]
            start_top = rotated(edge.start)
            end_top = rotated(edge.end)
            if closing:
                top_edge = edge_origin[edge_id]
            elif isinstance(edge.curve, Arc):
                top_edge = self.add_arc(
                    start_top, rotated(edge.curve.via_vertex), end_top
                )
            elif isinstance(edge.curve, Spline):
                top_edge = self.add_spline(
                    start_top,
                    tuple(rotated(vertex) for vertex in edge.curve.control_vertices),
                    end_top,
                )
            else:
                top_edge = self.add_line(start_top, end_top)

            loop = (
                OrientedEdge(edge_id, True),
                OrientedEdge(connector(edge.end), True),
                OrientedEdge(top_edge, False),
                OrientedEdge(connector(edge.start), False),
            )
            face_ids.append(self._add_face_from_loop(loop, (0, 1, 2, 3)))
            next_profile.append(top_edge)

            next_edge_origin[top_edge] = edge_origin[edge_id]
            next_vertex_origin[start_top] = vertex_origin[edge.start]
            next_vertex_origin[end_top] = vertex_origin[edge.end]
            if isinstance(edge.curve, Arc) and not closing:
                via_top = rotated(edge.curve.via_vertex)
                next_vertex_origin[via_top] = vertex_origin[edge.curve.via_vertex]
            elif isinstance(edge.curve, Spline) and not closing:
                for control in edge.curve.control_vertices:
                    control_top = rotated(control)
                    next_vertex_origin[control_top] = vertex_origin[control]

        return next_profile, face_ids, next_edge_origin, next_vertex_origin

    def _reject_on_axis(
        self,
        edge_ids: Sequence[int],
        origin: np.ndarray,
        direction: np.ndarray,
        tolerance: float = 1.0e-9,
    ) -> None:
        """A point on the axis would sweep into itself, not into an arc."""

        checked: set[int] = set()
        for edge_id in edge_ids:
            edge = self.edges[edge_id]
            vertices = [edge.start, edge.end]
            if isinstance(edge.curve, Arc):
                vertices.append(edge.curve.via_vertex)
            elif isinstance(edge.curve, Spline):
                vertices.extend(edge.curve.control_vertices)
            for vertex_id in vertices:
                if vertex_id in checked:
                    continue
                checked.add(vertex_id)
                offset = self.vertices[vertex_id].position - origin
                radial = offset - float(offset @ direction) * direction
                if float(np.linalg.norm(radial)) <= tolerance:
                    raise GeometryError(
                        f"point {vertex_id} lies on the revolve axis, so it "
                        "would sweep into itself rather than into an arc. "
                        "Move it off the axis, or model the apex region "
                        "separately."
                    )

    # ------------------------------------------------------------------
    # splitting
    # ------------------------------------------------------------------
    def split_edge(
        self, edge_id: int, t: float = 0.5
    ) -> Tuple[int, Tuple[int, int]]:
        """Split a line or arc at parameter ``t``, keeping every face valid.

        Returns the new point and the two replacement edges.  Faces that used
        the original edge have it swapped for the pair in traversal order, and
        their corner indices shift to match, so a side that was one edge simply
        becomes a chain of two.  This is the primitive behind imprinting.
        """

        edge = self._require_edge(edge_id)
        if not 0.0 < float(t) < 1.0:
            raise GeometryError(
                f"split parameter must be strictly between 0 and 1, got {t}"
            )

        new_vertex = self.add_point(*self.sample_edge(edge_id, np.array([t]))[0])
        if isinstance(edge.curve, Arc):
            first_via = self.add_point(
                *self.sample_edge(edge_id, np.array([0.5 * t]))[0]
            )
            second_via = self.add_point(
                *self.sample_edge(edge_id, np.array([0.5 * (1.0 + t)]))[0]
            )
            first = self.add_arc(edge.start, first_via, new_vertex)
            second = self.add_arc(new_vertex, second_via, edge.end)
        elif isinstance(edge.curve, Spline):
            points = self._spline_points(edge)
            left, right = self._split_bezier(points, float(t))
            left_controls = tuple(self.add_point(*point) for point in left[1:-1])
            right_controls = tuple(self.add_point(*point) for point in right[1:-1])
            first = self.add_spline(edge.start, left_controls, new_vertex)
            second = self.add_spline(new_vertex, right_controls, edge.end)
        else:
            first = self.add_line(edge.start, new_vertex)
            second = self.add_line(new_vertex, edge.end)

        for face in self.faces.values():
            self._replace_edge_in_loop(face, edge_id, first, second)

        del self.edges[edge_id]
        self._arc_cache.pop(edge_id, None)
        self.record_replacement(
            EntityRef("edge", edge_id),
            (EntityRef("edge", first), EntityRef("edge", second)),
        )
        return new_vertex, (first, second)

    @staticmethod
    def _replace_edge_in_loop(
        face: Face, edge_id: int, first: int, second: int
    ) -> None:
        def replaced(
            loop: Tuple[OrientedEdge, ...], *, update_corners: bool
        ) -> Tuple[OrientedEdge, ...]:
            positions = [
                index for index, item in enumerate(loop) if item.edge == edge_id
            ]
            for position in reversed(positions):
                item = loop[position]
                if item.forward:
                    replacement = (
                        OrientedEdge(first, True),
                        OrientedEdge(second, True),
                    )
                else:
                    # Traversed backwards, the far half comes first.
                    replacement = (
                        OrientedEdge(second, False),
                        OrientedEdge(first, False),
                    )
                loop = loop[:position] + replacement + loop[position + 1 :]
                if update_corners:
                    # A corner sitting on the split edge still starts where it
                    # did; everything after it moves along by one.
                    face.corners = tuple(  # type: ignore[assignment]
                        corner + 1 if corner > position else corner
                        for corner in face.corners
                    )
            return loop

        face.loop = replaced(face.loop, update_corners=True)
        face.holes = tuple(
            replaced(loop, update_corners=False) for loop in face.holes
        )

    def set_face_corners(self, face_id: int, corners: Sequence[int]) -> None:
        """Override which loop positions begin each of the four sides."""

        face = self._require_face(face_id)
        face.corners = self._validate_corners(
            tuple(int(c) for c in corners), len(face.loop)
        )

    def face_side_lengths(self, face_id: int) -> Tuple[float, float, float, float]:
        face = self._require_face(face_id)
        return tuple(  # type: ignore[return-value]
            self.side_length(side) for side in face.sides()
        )

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------
    def vertex_position(self, vertex_id: int) -> np.ndarray:
        return self._require_vertex(vertex_id).position

    def sample_edge(self, edge_id: int, t: np.ndarray) -> np.ndarray:
        """Sample points along an edge in its own direction.

        ``t`` runs from 0 at the start vertex to 1 at the end vertex.  Uniform
        ``t`` gives uniform arc length for both straight lines and arcs.
        """

        edge = self._require_edge(edge_id)
        start = self.vertices[edge.start].position
        end = self.vertices[edge.end].position
        if isinstance(edge.curve, Arc):
            return sample_arc(self._arc_frame(edge), t)
        if isinstance(edge.curve, Spline):
            return sample_spline(self._spline_points(edge), t)
        return sample_straight(start, end, t)

    def edge_length(self, edge_id: int) -> float:
        edge = self._require_edge(edge_id)
        if isinstance(edge.curve, Arc):
            return self._arc_frame(edge).length
        if isinstance(edge.curve, Spline):
            samples = sample_spline(
                self._spline_points(edge), np.linspace(0.0, 1.0, 65)
            )
            return float(np.linalg.norm(np.diff(samples, axis=0), axis=1).sum())
        start = self.vertices[edge.start].position
        end = self.vertices[edge.end].position
        return float(np.linalg.norm(end - start))

    def edge_tangent(self, edge_id: int, t: float) -> np.ndarray:
        """Unit tangent along the edge's own direction at parameter ``t``."""

        edge = self._require_edge(edge_id)
        if isinstance(edge.curve, Arc):
            return arc_tangent(self._arc_frame(edge), t)
        if isinstance(edge.curve, Spline):
            return spline_tangent(self._spline_points(edge), t)
        return straight_tangent(
            self.vertices[edge.start].position, self.vertices[edge.end].position
        )

    def closest_edge_point(
        self, edge_id: int, point: Sequence[float]
    ) -> Tuple[np.ndarray, float, float]:
        """Closest point, edge parameter and distance on a bounded curve."""

        edge = self._require_edge(edge_id)
        target = np.asarray(point, dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise GeometryError("point must be a finite 3-vector")
        if isinstance(edge.curve, Straight):
            start = self.vertex_position(edge.start)
            vector = self.vertex_position(edge.end) - start
            denominator = float(vector @ vector)
            if denominator <= 0.0:
                raise GeometryError(f"edge {edge_id} has zero length")
            parameter = float(np.clip((target-start) @ vector / denominator, 0.0, 1.0))
            made = start + parameter*vector
            return made, parameter, float(np.linalg.norm(made-target))
        parameters = np.linspace(0.0, 1.0, 65)
        samples = self.sample_edge(edge_id, parameters)
        index = int(np.argmin(np.linalg.norm(samples-target, axis=1)))
        lower = float(parameters[max(0, index-1)])
        upper = float(parameters[min(len(parameters)-1, index+1)])
        ratio = 0.5*(np.sqrt(5.0)-1.0)
        first = upper-ratio*(upper-lower)
        second = lower+ratio*(upper-lower)

        def objective(parameter: float) -> float:
            offset = self.sample_edge(edge_id, np.asarray([parameter]))[0]-target
            return float(offset @ offset)

        first_value, second_value = objective(first), objective(second)
        for _ in range(48):
            if first_value <= second_value:
                upper, second, second_value = second, first, first_value
                first = upper-ratio*(upper-lower)
                first_value = objective(first)
            else:
                lower, first, first_value = first, second, second_value
                second = lower+ratio*(upper-lower)
                second_value = objective(second)
        parameter = 0.5*(lower+upper)
        made = self.sample_edge(edge_id, np.asarray([parameter]))[0]
        return made, parameter, float(np.linalg.norm(made-target))

    def arc_frame(self, edge_id: int) -> ArcFrame:
        """The resolved circle of an arc edge: centre, radius, axes and sweep.

        Public because a mesh backend that rebuilds the model in another kernel
        needs the circle, not just samples along it.  Raises for a straight edge
        rather than returning a degenerate frame.
        """

        edge = self._require_edge(edge_id)
        if not isinstance(edge.curve, Arc):
            raise GeometryError(f"edge {edge_id} is not an arc")
        return self._arc_frame(edge)

    def _arc_frame(self, edge: Edge) -> ArcFrame:
        """Resolve and cache an arc's circle, invalidated when points move."""

        assert isinstance(edge.curve, Arc)
        stamp = self._geometry_stamp(edge)
        cached = self._arc_cache.get(edge.id)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        frame = arc_frame(
            self.vertices[edge.start].position,
            self.vertices[edge.curve.via_vertex].position,
            self.vertices[edge.end].position,
        )
        self._arc_cache[edge.id] = (stamp, frame)
        return frame

    def _geometry_stamp(self, edge: Edge) -> int:
        assert isinstance(edge.curve, Arc)
        return hash(
            (
                self.vertices[edge.start].position.tobytes(),
                self.vertices[edge.curve.via_vertex].position.tobytes(),
                self.vertices[edge.end].position.tobytes(),
            )
        )

    def move_point(self, vertex_id: int, x: float, y: float, z: float = 0.0) -> None:
        """Move a point; every curve referencing it follows."""

        vertex = self._require_vertex(vertex_id)
        position = np.asarray((x, y, z), dtype=float)
        if not np.all(np.isfinite(position)):
            raise GeometryError("point coordinates must be finite")
        if np.array_equal(vertex.position, position):
            return
        snapshot = self.topology_snapshot()
        affected_faces = {
            face_id
            for edge_id in self.edges_using_vertex(vertex_id)
            for face_id in self.faces_using_edge(edge_id)
        }
        try:
            vertex.position = position
            self._arc_cache.clear()
            # A point-wise edit is not, in general, a rigid transform of an
            # analytical surface.  Boundary-backed Coons faces remain exact.
            # Non-mapped faces keep a still-valid explicit surface, or receive
            # a deterministic fitted plane when their edited boundary remains
            # planar; otherwise the edit is rejected rather than leaving an
            # unevaluable face.
            for face_id in affected_faces:
                face = self.faces[face_id]
                if len(face.corners) == 4:
                    face.surface = CoonsSurface()
                    continue
                if face.surface is not None and self._face_matches_surface(
                    face_id, face.surface
                ):
                    continue
                fitted = self._fit_face_plane(face_id)
                if fitted is None:
                    raise GeometryError(
                        f"moving point {vertex_id} would leave face {face_id} "
                        "without a valid evaluable surface"
                    )
                face.surface = fitted
            errors = self.validate_topology()
            if errors:
                raise GeometryError(
                    "point move produced invalid geometry: " + "; ".join(errors)
                )
        except Exception:
            self.restore_topology(snapshot)
            raise

    def _face_matches_surface(self, face_id: int, surface: object) -> bool:
        if isinstance(surface, CoonsSurface) and not surface.has_boundaries:
            return True
        face = self.faces[face_id]
        try:
            for loop in (face.loop,) + face.holes:
                for point in self._validation_loop_points(loop):
                    uv = surface.local_uv(point)  # type: ignore[union-attr]
                    projected = np.asarray(surface.evaluate(*uv), dtype=float)  # type: ignore[union-attr]
                    scale = max(float(np.linalg.norm(point)), 1.0)
                    if float(np.linalg.norm(projected - point)) > 1.0e-7 * scale:
                        return False
        except (AttributeError, ValueError, GeometryError, np.linalg.LinAlgError):
            return False
        return True

    def _fit_face_plane(self, face_id: int) -> Plane | None:
        face = self.faces[face_id]
        points = np.vstack(
            [
                self._validation_loop_points(loop)
                for loop in (face.loop,) + face.holes
            ]
        )
        centre = points.mean(axis=0)
        _values, singular, vectors = np.linalg.svd(points - centre)
        scale = max(float(singular[0]), 1.0)
        if len(singular) >= 3 and float(singular[-1]) > 1.0e-8 * scale:
            return None
        coordinates = np.column_stack(
            ((points - centre) @ vectors[0], (points - centre) @ vectors[1])
        )
        minimum = coordinates.min(axis=0)
        maximum = coordinates.max(axis=0)
        spans = maximum - minimum
        if np.any(spans <= 1.0e-12 * scale):
            return None
        origin = centre + minimum[0] * vectors[0] + minimum[1] * vectors[1]
        return Plane(origin, spans[0] * vectors[0], spans[1] * vectors[1])

    def _spline_points(self, edge: Edge) -> np.ndarray:
        assert isinstance(edge.curve, Spline)
        ids = (edge.start,) + edge.curve.control_vertices + (edge.end,)
        return np.asarray([self.vertices[item].position for item in ids])

    @staticmethod
    def _split_bezier(points: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
        levels = [np.asarray(points, dtype=float)]
        while len(levels[-1]) > 1:
            previous = levels[-1]
            levels.append((1.0 - t) * previous[:-1] + t * previous[1:])
        left = np.asarray([level[0] for level in levels])
        right = np.asarray([level[-1] for level in reversed(levels)])
        return left, right

    # ------------------------------------------------------------------
    # oriented traversal helpers
    # ------------------------------------------------------------------
    def oriented_start_vertex(self, oriented: OrientedEdge) -> int:
        edge = self._require_edge(oriented.edge)
        return edge.start if oriented.forward else edge.end

    def oriented_end_vertex(self, oriented: OrientedEdge) -> int:
        edge = self._require_edge(oriented.edge)
        return edge.end if oriented.forward else edge.start

    def oriented_start_tangent(self, oriented: OrientedEdge) -> np.ndarray:
        if oriented.forward:
            return self.edge_tangent(oriented.edge, 0.0)
        return -self.edge_tangent(oriented.edge, 1.0)

    def oriented_end_tangent(self, oriented: OrientedEdge) -> np.ndarray:
        if oriented.forward:
            return self.edge_tangent(oriented.edge, 1.0)
        return -self.edge_tangent(oriented.edge, 0.0)

    def face_corner_vertices(self, face_id: int) -> Tuple[int, ...]:
        """The four corner points of a face, in loop order."""

        face = self._require_face(face_id)
        return tuple(  # type: ignore[return-value]
            self.oriented_start_vertex(face.loop[index]) for index in face.corners
        )

    def side_length(self, side: Sequence[OrientedEdge]) -> float:
        return float(sum(self.edge_length(item.edge) for item in side))

    def face_point(self, face_id: int, u: float, v: float) -> np.ndarray:
        """Evaluate a face at local coordinates without a mesher dependency."""

        face = self._require_face(face_id)
        if face.surface is not None and (
            not isinstance(face.surface, CoonsSurface) or face.surface.has_boundaries
        ):
            return np.asarray(face.surface.evaluate(float(u), float(v)), dtype=float)
        if len(face.corners) != 4:
            raise GeometryError(
                f"face {face_id} has no explicit surface or four-side Coons mapping"
            )
        sides = face.sides()
        point_a = self._chain_point(sides[0], u)
        point_b = self._chain_point(sides[1], v)
        point_c = self._chain_point(sides[2], 1.0 - u)
        point_d = self._chain_point(sides[3], 1.0 - v)
        corner_00 = self._chain_point(sides[0], 0.0)
        corner_10 = self._chain_point(sides[0], 1.0)
        corner_11 = self._chain_point(sides[2], 0.0)
        corner_01 = self._chain_point(sides[2], 1.0)
        return (
            (1.0 - v) * point_a
            + v * point_c
            + (1.0 - u) * point_d
            + u * point_b
            - (
                (1.0 - u) * (1.0 - v) * corner_00
                + u * (1.0 - v) * corner_10
                + u * v * corner_11
                + (1.0 - u) * v * corner_01
            )
        )

    def face_local_uv(
        self, face_id: int, point: Sequence[float]
    ) -> Tuple[float, float]:
        """Return bounded closest local coordinates on a face."""

        face = self._require_face(face_id)
        if face.surface is not None and (
            not isinstance(face.surface, CoonsSurface) or face.surface.has_boundaries
        ):
            local = face.surface.local_uv(point)
            return (
                float(np.clip(local[0], 0.0, 1.0)),
                float(np.clip(local[1], 0.0, 1.0)),
            )

        model = self

        class _TopologySurface:
            def evaluate(self, u: float, v: float) -> np.ndarray:
                return model.face_point(face_id, u, v)

            def local_uv(self, candidate: object) -> Tuple[float, float]:
                return closest_uv(self, candidate)

        return closest_uv(_TopologySurface(), point)

    def project_to_face(
        self, face_id: int, point: Sequence[float]
    ) -> Tuple[np.ndarray, Tuple[float, float], float]:
        """Project a point to a face and return point, UV and distance."""

        uv = self.face_local_uv(face_id, point)
        projected = self.face_point(face_id, *uv)
        if not self.face_contains_uv(face_id, uv):
            projected = self._closest_face_boundary_point(face_id, point)
            uv = self.face_local_uv(face_id, projected)
        distance = float(np.linalg.norm(projected - np.asarray(point, dtype=float)))
        return projected, uv, distance

    def face_contains_uv(self, face_id: int, uv: Sequence[float]) -> bool:
        """Whether local coordinates lie in the outer trim and outside holes."""

        face = self._require_face(face_id)
        candidate = np.asarray(uv, dtype=float)
        if candidate.shape != (2,):
            raise GeometryError("local coordinates must contain u and v")

        polygons = self.face_trim_loops_uv(face_id)
        if not self._point_in_polygon(candidate, polygons[0]):
            return False
        return not any(
            self._point_in_polygon(candidate, hole, include_boundary=False)
            for hole in polygons[1:]
        )

    def face_trim_loops_uv(
        self, face_id: int, *, curve_samples: int = 17
    ) -> Tuple[np.ndarray, ...]:
        """Return outer and hole trim loops in the face's local UV plane.

        This is the authoritative public bridge for trim-aware tessellation,
        hit testing, and planar export. Curves are sampled deterministically;
        straight segments contribute only their start vertex.
        """

        face = self._require_face(face_id)
        count = int(curve_samples)
        if count < 3:
            raise GeometryError("trim curve sampling needs at least three points")

        def polygon(loop: Sequence[OrientedEdge]) -> np.ndarray:
            points: List[np.ndarray] = []
            for item in loop:
                edge = self.edges[item.edge]
                samples = self.sample_edge(
                    item.edge,
                    np.linspace(
                        0.0,
                        1.0,
                        2 if isinstance(edge.curve, Straight) else count,
                    ),
                )
                if not item.forward:
                    samples = samples[::-1]
                points.extend(samples[:-1])
            return np.asarray(
                [self.face_local_uv(face_id, point) for point in points],
                dtype=float,
            )

        return tuple(polygon(loop) for loop in (face.loop,) + tuple(face.holes))

    @staticmethod
    def _point_in_polygon(
        point: np.ndarray, polygon: np.ndarray, *, include_boundary: bool = True
    ) -> bool:
        if len(polygon) < 3:
            return False
        x, y = float(point[0]), float(point[1])
        inside = False
        previous = polygon[-1]
        for current in polygon:
            x1, y1 = float(previous[0]), float(previous[1])
            x2, y2 = float(current[0]), float(current[1])
            segment = np.asarray((x2 - x1, y2 - y1))
            offset = np.asarray((x - x1, y - y1))
            cross = abs(float(segment[0] * offset[1] - segment[1] * offset[0]))
            if cross <= 1.0e-10 and min(x1, x2)-1e-10 <= x <= max(x1, x2)+1e-10 and min(y1, y2)-1e-10 <= y <= max(y1, y2)+1e-10:
                return include_boundary
            if (y1 > y) != (y2 > y):
                crossing = x1 + (y-y1)*(x2-x1)/(y2-y1)
                if x < crossing:
                    inside = not inside
            previous = current
        return inside

    def _closest_face_boundary_point(
        self, face_id: int, point: Sequence[float]
    ) -> np.ndarray:
        target = np.asarray(point, dtype=float)
        face = self._require_face(face_id)
        best_distance = float("inf")
        best_point: np.ndarray | None = None
        for loop in (face.loop,) + face.holes:
            for item in loop:
                candidate, _parameter, distance = self.closest_edge_point(
                    item.edge, target
                )
                if distance < best_distance:
                    best_distance = distance
                    best_point = candidate
        if best_point is None:
            raise GeometryError(f"face {face_id} has no boundary")
        return best_point

    def face_normal(self, face_id: int, u: float, v: float) -> np.ndarray:
        """Deterministic unit normal, including topology-backed Coons faces."""

        step = 1.0e-6
        u0, v0 = float(u), float(v)
        point = self.face_point(face_id, u0, v0)
        du = self.face_point(face_id, min(1.0, u0 + step), v0) - point
        dv = self.face_point(face_id, u0, min(1.0, v0 + step)) - point
        if float(np.linalg.norm(du)) <= 0.0:
            du = point - self.face_point(face_id, max(0.0, u0 - step), v0)
        if float(np.linalg.norm(dv)) <= 0.0:
            dv = point - self.face_point(face_id, u0, max(0.0, v0 - step))
        normal = np.cross(du, dv)
        length = float(np.linalg.norm(normal))
        if length <= 0.0:
            raise GeometryError(f"face {face_id} has a degenerate normal")
        return normal / length

    def closest_face(
        self, point: Sequence[float], face_ids: Iterable[int] | None = None
    ) -> Tuple[int, np.ndarray, Tuple[float, float], float]:
        """Find the closest face with deterministic ID tie-breaking."""

        candidates = sorted(self.faces if face_ids is None else set(face_ids))
        if not candidates:
            raise GeometryError("closest_face needs at least one face")
        ranked = []
        for face_id in candidates:
            projected, uv, distance = self.project_to_face(face_id, point)
            ranked.append((distance, face_id, projected, uv))
        distance, face_id, projected, uv = min(ranked, key=lambda item: (item[0], item[1]))
        return face_id, projected, uv, distance

    def _chain_point(self, chain: Sequence[OrientedEdge], fraction: float) -> np.ndarray:
        if not chain:
            raise GeometryError("cannot evaluate an empty boundary chain")
        lengths = np.asarray([self.edge_length(item.edge) for item in chain])
        total = float(lengths.sum())
        if total <= 0.0:
            raise GeometryError("cannot evaluate a zero-length boundary chain")
        target = float(np.clip(fraction, 0.0, 1.0)) * total
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        index = min(int(np.searchsorted(cumulative, target, side="right") - 1), len(chain) - 1)
        local = (target - cumulative[index]) / lengths[index]
        item = chain[index]
        parameter = local if item.forward else 1.0 - local
        return self.sample_edge(item.edge, np.asarray([parameter]))[0]

    # ------------------------------------------------------------------
    # loop ordering and corner detection
    # ------------------------------------------------------------------
    def _order_loop(self, edge_ids: Sequence[int]) -> Tuple[OrientedEdge, ...]:
        remaining = list(dict.fromkeys(int(e) for e in edge_ids))
        if len(remaining) < 3:
            raise GeometryError(
                "a face needs at least three edges forming a closed loop"
            )
        for edge_id in remaining:
            self._require_edge(edge_id)

        first = remaining.pop(0)
        loop = [OrientedEdge(first, True)]
        start_vertex = self.edges[first].start
        current = self.edges[first].end

        while remaining:
            for index, edge_id in enumerate(remaining):
                edge = self.edges[edge_id]
                if edge.start == current:
                    loop.append(OrientedEdge(edge_id, True))
                    current = edge.end
                elif edge.end == current:
                    loop.append(OrientedEdge(edge_id, False))
                    current = edge.start
                else:
                    continue
                remaining.pop(index)
                break
            else:
                raise GeometryError(
                    "edges do not form a single closed loop: "
                    f"no edge continues from vertex {current}"
                )

        if current != start_vertex:
            raise GeometryError(
                "edges do not form a closed loop: the chain ends at vertex "
                f"{current} but starts at vertex {start_vertex}"
            )
        return tuple(loop)

    def _detect_corners(
        self, loop: Tuple[OrientedEdge, ...]
    ) -> Tuple[int, int, int, int]:
        """Pick the four sharpest boundary turns as the mapped-face corners."""

        count = len(loop)
        if count < 4:
            raise GeometryError(
                f"a mapped face needs at least four edges, got {count}; "
                "split the boundary so it forms four sides"
            )
        if count == 4:
            return (0, 1, 2, 3)

        deviations = []
        for index in range(count):
            incoming = self.oriented_end_tangent(loop[index - 1])
            outgoing = self.oriented_start_tangent(loop[index])
            cosine = float(np.clip(incoming @ outgoing, -1.0, 1.0))
            deviations.append(float(np.arccos(cosine)))

        sharpest = sorted(
            sorted(range(count), key=lambda i: (-deviations[i], i))[:4]
        )
        return self._validate_corners(tuple(sharpest), count)

    @staticmethod
    def _validate_corners(
        corners: Tuple[int, ...], loop_length: int
    ) -> Tuple[int, int, int, int]:
        if len(corners) != 4:
            raise GeometryError("a mapped face needs exactly four corners")
        if len(set(corners)) != 4:
            raise GeometryError("face corners must be four distinct loop positions")
        if any(not 0 <= c < loop_length for c in corners):
            raise GeometryError("face corner index outside the boundary loop")
        if list(corners) != sorted(corners):
            raise GeometryError("face corners must be given in loop order")
        return tuple(corners)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # lookup
    # ------------------------------------------------------------------
    def _require_vertex(self, vertex_id: int) -> Vertex:
        try:
            return self.vertices[vertex_id]
        except KeyError:
            raise GeometryError(f"no vertex {vertex_id}") from None

    def _require_edge(self, edge_id: int) -> Edge:
        try:
            return self.edges[edge_id]
        except KeyError:
            raise GeometryError(f"no edge {edge_id}") from None

    def _require_face(self, face_id: int) -> Face:
        try:
            return self.faces[face_id]
        except KeyError:
            raise GeometryError(f"no face {face_id}") from None

    def entity_ref(self, kind: str, entity_id: int) -> EntityRef:
        """Build a reference after checking the entity exists."""

        if kind == "vertex":
            self._require_vertex(entity_id)
        elif kind == "edge":
            self._require_edge(entity_id)
        elif kind == "face":
            self._require_face(entity_id)
        else:
            raise GeometryError(f"unknown entity kind {kind!r}")
        return EntityRef(kind, entity_id)  # type: ignore[arg-type]

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"GeometryModel(vertices={len(self.vertices)}, "
            f"edges={len(self.edges)}, faces={len(self.faces)})"
        )
