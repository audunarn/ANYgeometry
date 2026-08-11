"""High-level neutral editing, duplication, and measurement operations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .curves import Arc, Spline
from .entities import EntityRef, OrientedEdge
from .errors import GeometryError
from .model import GeometryModel
from .operations import transform
from .surfaces import CoonsSurface, Cone, Cylinder, Plane, RuledSurface

__all__ = [
    "InsertResult",
    "Measurement",
    "PatternResult",
    "circular_pattern",
    "copy_entities",
    "insert_model",
    "linear_pattern",
    "measure",
    "mirror_entities",
    "reverse_edge",
    "reverse_face",
]


@dataclass(frozen=True)
class InsertResult:
    """Complete identity map produced by inserting or copying topology."""

    entity_map: Mapping[EntityRef, EntityRef]
    groups: Mapping[str, tuple[EntityRef, ...]]
    outputs: Mapping[str, EntityRef]

    def mapped(self, reference: EntityRef) -> EntityRef:
        return self.entity_map[reference]


@dataclass(frozen=True)
class PatternResult:
    instances: tuple[InsertResult, ...]


@dataclass(frozen=True)
class Measurement:
    kind: str
    value: float | tuple[float, ...]
    unit: str
    witnesses: tuple[tuple[float, float, float], ...] = ()


def _affine_matrix(value: Sequence[Sequence[float]] | None) -> np.ndarray:
    if value is None:
        return np.eye(4)
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise GeometryError("transform must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-14):
        raise GeometryError("only affine homogeneous transforms are supported")
    return matrix


def _expanded_refs(
    geometry: GeometryModel, references: Iterable[EntityRef] | None
) -> tuple[set[int], set[int], set[int]]:
    if references is None:
        return set(geometry.vertices), set(geometry.edges), set(geometry.faces)
    current: list[EntityRef] = []
    for reference in references:
        geometry.entity_ref(reference.kind, reference.id)
        current.extend(geometry.resolve_ref(reference))
    vertices = {item.id for item in current if item.kind == "vertex"}
    edges = {item.id for item in current if item.kind == "edge"}
    faces = {item.id for item in current if item.kind == "face"}
    for face_id in faces:
        face = geometry.faces[face_id]
        edges.update(item.edge for loop in (face.loop,) + face.holes for item in loop)
    for edge_id in edges:
        edge = geometry.edges[edge_id]
        vertices.update((edge.start, edge.end))
        if isinstance(edge.curve, Arc):
            vertices.add(edge.curve.via_vertex)
        elif isinstance(edge.curve, Spline):
            vertices.update(edge.curve.control_vertices)
    return vertices, edges, faces


def _insert_selected(
    destination: GeometryModel,
    source: GeometryModel,
    references: Iterable[EntityRef] | None,
    *,
    group_prefix: str | None,
) -> InsertResult:
    vertex_ids, edge_ids, face_ids = _expanded_refs(source, references)
    mapping: dict[EntityRef, EntityRef] = {}

    for vertex_id in sorted(vertex_ids):
        made = destination.add_point(*source.vertex_position(vertex_id))
        mapping[EntityRef("vertex", vertex_id)] = EntityRef("vertex", made)

    for edge_id in sorted(edge_ids):
        edge = source.edges[edge_id]
        start = mapping[EntityRef("vertex", edge.start)].id
        end = mapping[EntityRef("vertex", edge.end)].id
        if isinstance(edge.curve, Arc):
            via = mapping[EntityRef("vertex", edge.curve.via_vertex)].id
            made = destination.add_arc(start, via, end)
        elif isinstance(edge.curve, Spline):
            controls = [
                mapping[EntityRef("vertex", item)].id
                for item in edge.curve.control_vertices
            ]
            made = destination.add_spline(start, controls, end)
        else:
            made = destination.add_line(start, end)
        mapping[EntityRef("edge", edge_id)] = EntityRef("edge", made)

    for face_id in sorted(face_ids):
        face = source.faces[face_id]

        def mapped_loop(loop: Sequence[OrientedEdge]) -> tuple[OrientedEdge, ...]:
            return tuple(
                OrientedEdge(
                    mapping[EntityRef("edge", item.edge)].id,
                    item.forward,
                )
                for item in loop
            )

        made = destination.add_face_from_loop(
            mapped_loop(face.loop),
            face.corners,
            surface=deepcopy(face.surface),
        )
        destination.faces[made].holes = tuple(mapped_loop(loop) for loop in face.holes)
        destination.faces[made].metadata = deepcopy(face.metadata)
        mapping[EntityRef("face", face_id)] = EntityRef("face", made)

    made_groups: dict[str, tuple[EntityRef, ...]] = {}
    prefix = "" if group_prefix is None else str(group_prefix).strip("/") + "/"
    for name in sorted(source.groups):
        members = tuple(
            mapping[item]
            for item in source.group(name)
            if item in mapping
        )
        if not members:
            continue
        target_name = prefix + name
        destination.add_to_group(target_name, members)
        made_groups[target_name] = tuple(
            sorted(members, key=lambda item: (item.kind, item.id))
        )
    for old, new in mapping.items():
        values = source.tags_for(old)
        if values:
            destination.tag(new, *values)

    outputs = {
        f"{old.kind}/{old.id}": new
        for old, new in sorted(
            mapping.items(), key=lambda item: (item[0].kind, item[0].id)
        )
    }
    return InsertResult(dict(mapping), made_groups, outputs)


def insert_model(
    destination: GeometryModel,
    source: GeometryModel,
    *,
    matrix: Sequence[Sequence[float]] | None = None,
    group_prefix: str | None = None,
) -> InsertResult:
    """Insert all current source topology with fresh destination identities.

    Coincident entities are deliberately not welded.  Insertion is a pure
    ownership transfer by copying; source lineage and feature history remain
    in the source document rather than being spliced into the destination.
    """

    errors = source.validate_topology()
    if errors:
        raise GeometryError("cannot insert invalid geometry: " + "; ".join(errors))
    snapshot = destination.topology_snapshot()
    try:
        prepared = source.clone(include_features=False)
        affine = _affine_matrix(matrix)
        if not np.allclose(affine, np.eye(4)):
            transform(prepared, affine)
        result = _insert_selected(
            destination, prepared, None, group_prefix=group_prefix
        )
        errors = destination.validate_topology()
        if errors:
            raise GeometryError("insert produced invalid topology: " + "; ".join(errors))
        return result
    except Exception:
        destination.restore_topology(snapshot)
        raise


def copy_entities(
    geometry: GeometryModel,
    references: Iterable[EntityRef],
    *,
    matrix: Sequence[Sequence[float]] | None = None,
    group_prefix: str | None = None,
) -> InsertResult:
    """Copy a selected topology closure into the same model."""

    selected = tuple(references)
    if not selected:
        raise GeometryError("copy needs at least one entity")
    snapshot = geometry.topology_snapshot()
    try:
        source = geometry.clone(include_features=False)
        affine = _affine_matrix(matrix)
        if not np.allclose(affine, np.eye(4)):
            transform(source, affine, selected)
        result = _insert_selected(
            geometry, source, selected, group_prefix=group_prefix
        )
        errors = geometry.validate_topology()
        if errors:
            raise GeometryError("copy produced invalid topology: " + "; ".join(errors))
        return result
    except Exception:
        geometry.restore_topology(snapshot)
        raise


def linear_pattern(
    geometry: GeometryModel,
    references: Iterable[EntityRef],
    direction: Sequence[float],
    spacing: float,
    count: int,
    *,
    group_prefix: str | None = None,
) -> PatternResult:
    """Create ``count`` translated copies at successive equal spacings."""

    vector = np.asarray(direction, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise GeometryError("pattern direction must be a finite 3-vector")
    length = float(np.linalg.norm(vector))
    if length <= 0.0:
        raise GeometryError("pattern direction must be non-zero")
    if isinstance(count, bool) or int(count) != count or int(count) < 1:
        raise GeometryError("pattern count must be a positive integer")
    spacing = float(spacing)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise GeometryError("pattern spacing must be finite and positive")
    selected = tuple(references)
    snapshot = geometry.topology_snapshot()
    made: list[InsertResult] = []
    try:
        unit = vector / length
        for index in range(1, int(count) + 1):
            matrix = np.eye(4)
            matrix[:3, 3] = unit * spacing * index
            made.append(
                copy_entities(
                    geometry,
                    selected,
                    matrix=matrix,
                    group_prefix=(
                        None
                        if group_prefix is None
                        else f"{group_prefix}/{index}"
                    ),
                )
            )
        return PatternResult(tuple(made))
    except Exception:
        geometry.restore_topology(snapshot)
        raise


def _rotation_matrix(
    point: np.ndarray, direction: np.ndarray, angle: float
) -> np.ndarray:
    x, y, z = direction
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    cross = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    rotation = cosine * np.eye(3) + sine * cross + (1.0 - cosine) * np.outer(direction, direction)
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = point - rotation @ point
    return matrix


def circular_pattern(
    geometry: GeometryModel,
    references: Iterable[EntityRef],
    axis_point: Sequence[float],
    axis_direction: Sequence[float],
    angle_step: float,
    count: int,
    *,
    group_prefix: str | None = None,
) -> PatternResult:
    """Create rotated copies at ``angle_step``, ``2*angle_step``, and so on."""

    point = np.asarray(axis_point, dtype=float)
    direction = np.asarray(axis_direction, dtype=float)
    if (
        point.shape != (3,)
        or direction.shape != (3,)
        or not np.all(np.isfinite(point))
        or not np.all(np.isfinite(direction))
    ):
        raise GeometryError("pattern axis needs a finite point and direction")
    length = float(np.linalg.norm(direction))
    if length <= 0.0:
        raise GeometryError("pattern axis direction must be non-zero")
    if not np.isfinite(angle_step) or float(angle_step) == 0.0:
        raise GeometryError("pattern angle step must be finite and non-zero")
    if isinstance(count, bool) or int(count) != count or int(count) < 1:
        raise GeometryError("pattern count must be a positive integer")
    direction /= length
    selected = tuple(references)
    snapshot = geometry.topology_snapshot()
    made: list[InsertResult] = []
    try:
        for index in range(1, int(count) + 1):
            made.append(
                copy_entities(
                    geometry,
                    selected,
                    matrix=_rotation_matrix(point, direction, float(angle_step) * index),
                    group_prefix=(
                        None
                        if group_prefix is None
                        else f"{group_prefix}/{index}"
                    ),
                )
            )
        return PatternResult(tuple(made))
    except Exception:
        geometry.restore_topology(snapshot)
        raise


def mirror_entities(
    geometry: GeometryModel,
    references: Iterable[EntityRef],
    plane_point: Sequence[float],
    plane_normal: Sequence[float],
    *,
    group_prefix: str | None = None,
) -> InsertResult:
    """Copy selected topology reflected about a plane."""

    point = np.asarray(plane_point, dtype=float)
    normal = np.asarray(plane_normal, dtype=float)
    if (
        point.shape != (3,)
        or normal.shape != (3,)
        or not np.all(np.isfinite(point))
        or not np.all(np.isfinite(normal))
    ):
        raise GeometryError("mirror plane needs a finite point and normal")
    length = float(np.linalg.norm(normal))
    if length <= 0.0:
        raise GeometryError("mirror plane normal must be non-zero")
    normal /= length
    linear = np.eye(3) - 2.0 * np.outer(normal, normal)
    matrix = np.eye(4)
    matrix[:3, :3] = linear
    matrix[:3, 3] = point - linear @ point
    return copy_entities(
        geometry, references, matrix=matrix, group_prefix=group_prefix
    )


def reverse_edge(geometry: GeometryModel, edge_id: int) -> EntityRef:
    """Reverse edge parameterization while preserving every face traversal."""

    edge = geometry._require_edge(int(edge_id))  # noqa: SLF001
    snapshot = geometry.topology_snapshot()
    try:
        edge.start, edge.end = edge.end, edge.start
        if isinstance(edge.curve, Spline):
            edge.curve = Spline(tuple(reversed(edge.curve.control_vertices)))
        for face in geometry.faces.values():
            face.loop = tuple(
                OrientedEdge(item.edge, not item.forward)
                if item.edge == edge.id
                else item
                for item in face.loop
            )
            face.holes = tuple(
                tuple(
                    OrientedEdge(item.edge, not item.forward)
                    if item.edge == edge.id
                    else item
                    for item in loop
                )
                for loop in face.holes
            )
        geometry._arc_cache.pop(edge.id, None)  # noqa: SLF001
        errors = geometry.validate_topology()
        if errors:
            raise GeometryError("edge reversal produced invalid topology: " + "; ".join(errors))
        return edge.ref
    except Exception:
        geometry.restore_topology(snapshot)
        raise


def _reverse_loop(loop: Sequence[OrientedEdge]) -> tuple[OrientedEdge, ...]:
    return tuple(OrientedEdge(item.edge, not item.forward) for item in reversed(loop))


def _reverse_surface(surface: object) -> object:
    if surface is None:
        return None
    if isinstance(surface, Plane):
        return Plane(surface.origin, surface.v_vector, surface.u_vector)
    if isinstance(surface, CoonsSurface):
        if not surface.has_boundaries:
            return surface
        assert surface.bottom is not None and surface.right is not None
        assert surface.top is not None and surface.left is not None
        return CoonsSurface(
            surface.bottom[::-1].copy(),
            surface.left.copy(),
            surface.top[::-1].copy(),
            surface.right.copy(),
        )
    if isinstance(surface, Cylinder):
        return Cylinder(
            surface.origin,
            surface.axis,
            surface.radial_direction,
            surface.radius,
            surface.height,
            surface.start_angle + surface.sweep_angle,
            -surface.sweep_angle,
        )
    if isinstance(surface, Cone):
        return Cone(
            surface.origin,
            surface.axis,
            surface.radial_direction,
            surface.radius_start,
            surface.radius_end,
            surface.height,
            surface.start_angle + surface.sweep_angle,
            -surface.sweep_angle,
        )
    if isinstance(surface, RuledSurface):
        return RuledSurface(
            surface.first_boundary[::-1].copy(),
            surface.second_boundary[::-1].copy(),
        )
    raise GeometryError(f"unsupported surface type {type(surface).__name__}")


def reverse_face(geometry: GeometryModel, face_id: int) -> EntityRef:
    """Reverse a face orientation, including its authoritative surface normal."""

    face = geometry._require_face(int(face_id))  # noqa: SLF001
    snapshot = geometry.topology_snapshot()
    try:
        corner_vertices = set(geometry.face_corner_vertices(face.id))
        face.loop = _reverse_loop(face.loop)
        face.holes = tuple(_reverse_loop(loop) for loop in face.holes)
        if corner_vertices:
            face.corners = tuple(
                index
                for index, item in enumerate(face.loop)
                if geometry.oriented_start_vertex(item) in corner_vertices
            )
        face.surface = _reverse_surface(face.surface)
        errors = geometry.validate_topology()
        if errors:
            raise GeometryError("face reversal produced invalid topology: " + "; ".join(errors))
        return face.ref
    except Exception:
        geometry.restore_topology(snapshot)
        raise


def _edge_samples(geometry: GeometryModel, edge_id: int, count: int = 65) -> np.ndarray:
    return geometry.sample_edge(edge_id, np.linspace(0.0, 1.0, count))


def _loop_points(
    geometry: GeometryModel, loop: Sequence[OrientedEdge]
) -> np.ndarray:
    points: list[np.ndarray] = []
    for item in loop:
        sampled = _edge_samples(geometry, item.edge, 33)
        if not item.forward:
            sampled = sampled[::-1]
        points.extend(sampled[:-1])
    if loop:
        points.append(points[0])
    return np.asarray(points)


def _polygon_area(points: np.ndarray) -> float:
    if len(points) < 4:
        return 0.0
    vector = np.sum(np.cross(points[:-1], points[1:]), axis=0)
    return 0.5 * float(np.linalg.norm(vector))


def _face_area(geometry: GeometryModel, face_id: int) -> float:
    face = geometry.faces[face_id]
    if len(face.corners) == 4 and not face.holes:
        values = np.linspace(0.0, 1.0, 33)
        grid = np.asarray(
            [[geometry.face_point(face_id, float(u), float(v)) for u in values] for v in values]
        )
        area = 0.0
        for j in range(len(values) - 1):
            for i in range(len(values) - 1):
                a, b = grid[j, i], grid[j, i + 1]
                c, d = grid[j + 1, i + 1], grid[j + 1, i]
                area += 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))
                area += 0.5 * float(np.linalg.norm(np.cross(c - a, d - a)))
        return area
    area = _polygon_area(_loop_points(geometry, face.loop))
    return area - sum(_polygon_area(_loop_points(geometry, loop)) for loop in face.holes)


def _edge_centroid(geometry: GeometryModel, edge_id: int) -> np.ndarray:
    points = _edge_samples(geometry, edge_id, 257)
    lengths = np.linalg.norm(points[1:] - points[:-1], axis=1)
    total = float(np.sum(lengths))
    if total <= 0.0:
        raise GeometryError("cannot measure the centroid of a zero-length edge")
    return np.sum(0.5 * (points[1:] + points[:-1]) * lengths[:, None], axis=0) / total


def _loop_area_centroid(points: np.ndarray) -> tuple[float, np.ndarray]:
    ring = np.asarray(points, dtype=float)
    if len(ring) < 4:
        raise GeometryError("cannot measure the centroid of a degenerate loop")
    ring = ring[:-1] if np.allclose(ring[0], ring[-1]) else ring
    origin = np.mean(ring, axis=0)
    area_vector = np.sum(np.cross(ring, np.roll(ring, -1, axis=0)), axis=0)
    norm = float(np.linalg.norm(area_vector))
    if norm <= 0.0:
        raise GeometryError("cannot measure the centroid of a zero-area loop")
    normal = area_vector / norm
    weighted = np.zeros(3)
    signed_area = 0.0
    for first, second in zip(ring, np.roll(ring, -1, axis=0)):
        area = 0.5 * float(np.cross(first - origin, second - origin) @ normal)
        signed_area += area
        weighted += area * (origin + first + second) / 3.0
    if abs(signed_area) <= 0.0:
        raise GeometryError("cannot measure the centroid of a zero-area loop")
    return abs(signed_area), weighted / signed_area


def _face_centroid(geometry: GeometryModel, face_id: int) -> np.ndarray:
    face = geometry.faces[face_id]
    if len(face.corners) == 4 and not face.holes:
        values = np.linspace(0.0, 1.0, 65)
        grid = np.asarray(
            [
                [geometry.face_point(face_id, float(u), float(v)) for u in values]
                for v in values
            ]
        )
        weighted = np.zeros(3)
        total = 0.0
        for row in range(len(values) - 1):
            for column in range(len(values) - 1):
                a, b = grid[row, column], grid[row, column + 1]
                c, d = grid[row + 1, column + 1], grid[row + 1, column]
                for first, second, third in ((a, b, c), (a, c, d)):
                    area = 0.5 * float(
                        np.linalg.norm(np.cross(second - first, third - first))
                    )
                    total += area
                    weighted += area * (first + second + third) / 3.0
        if total <= 0.0:
            raise GeometryError("cannot measure the centroid of a zero-area face")
        return weighted / total

    outer_area, outer_centroid = _loop_area_centroid(
        _loop_points(geometry, face.loop)
    )
    weighted = outer_area * outer_centroid
    total = outer_area
    for loop in face.holes:
        area, centroid = _loop_area_centroid(_loop_points(geometry, loop))
        weighted -= area * centroid
        total -= area
    if total <= 0.0:
        raise GeometryError("face holes remove its complete measurable area")
    return weighted / total


def _samples_for(geometry: GeometryModel, reference: EntityRef) -> np.ndarray:
    if reference.kind == "vertex":
        return np.asarray([geometry.vertex_position(reference.id)])
    if reference.kind == "edge":
        return _edge_samples(geometry, reference.id)
    values = np.linspace(0.0, 1.0, 17)
    return np.asarray(
        [geometry.face_point(reference.id, float(u), float(v)) for v in values for u in values]
    )


def measure(
    geometry: GeometryModel,
    references: EntityRef | Sequence[EntityRef],
    *,
    quantity: str = "auto",
) -> Measurement:
    """Measure one entity or the distance/angle between two entities."""

    refs = (references,) if isinstance(references, EntityRef) else tuple(references)
    if not refs or len(refs) > 2:
        raise GeometryError("measure needs one or two entities")
    for reference in refs:
        geometry.entity_ref(reference.kind, reference.id)
    if len(refs) == 1:
        reference = refs[0]
        if quantity == "auto":
            quantity = {"vertex": "position", "edge": "length", "face": "area"}[reference.kind]
        if quantity in ("position", "coordinates") and reference.kind == "vertex":
            point = tuple(float(item) for item in geometry.vertex_position(reference.id))
            return Measurement(quantity, point, "m", (point,))
        if quantity == "length" and reference.kind == "edge":
            return Measurement("length", geometry.edge_length(reference.id), "m")
        if quantity == "area" and reference.kind == "face":
            return Measurement("area", _face_area(geometry, reference.id), "m^2")
        if quantity == "perimeter" and reference.kind == "face":
            face = geometry.faces[reference.id]
            value = sum(
                geometry.edge_length(item.edge)
                for loop in (face.loop,) + face.holes
                for item in loop
            )
            return Measurement("perimeter", value, "m")
        if quantity == "radius" and reference.kind == "edge":
            edge = geometry.edges[reference.id]
            if not isinstance(edge.curve, Arc):
                raise GeometryError("radius measurement needs a circular arc")
            return Measurement("radius", geometry.arc_frame(reference.id).radius, "m")
        if quantity == "centroid":
            if reference.kind == "vertex":
                value = geometry.vertex_position(reference.id)
            elif reference.kind == "edge":
                value = _edge_centroid(geometry, reference.id)
            else:
                value = _face_centroid(geometry, reference.id)
            point = tuple(float(item) for item in value)
            return Measurement("centroid", point, "m", (point,))
        if quantity == "normal" and reference.kind == "face":
            value = tuple(float(item) for item in geometry.face_normal(reference.id, 0.5, 0.5))
            return Measurement("normal", value, "1")
        raise GeometryError(f"cannot measure {quantity!r} on a {reference.kind}")

    first, second = refs
    if quantity == "angle":
        if first.kind != "edge" or second.kind != "edge":
            raise GeometryError("angle measurement needs two edges")
        a = geometry.edge_tangent(first.id, 0.5)
        b = geometry.edge_tangent(second.id, 0.5)
        cosine = float(np.clip(abs(float(a @ b)), -1.0, 1.0))
        return Measurement("angle", float(np.arccos(cosine)), "rad")
    if quantity not in ("auto", "distance"):
        raise GeometryError(f"unknown two-entity measurement {quantity!r}")
    a, b = _samples_for(geometry, first), _samples_for(geometry, second)
    delta = a[:, None, :] - b[None, :, :]
    distances = np.linalg.norm(delta, axis=2)
    row, column = np.unravel_index(int(np.argmin(distances)), distances.shape)
    witnesses = (
        tuple(float(item) for item in a[row]),
        tuple(float(item) for item in b[column]),
    )
    return Measurement("distance", float(distances[row, column]), "m", witnesses)
