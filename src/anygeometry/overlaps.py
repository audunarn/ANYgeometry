"""Planar plate-overlap auditing and deterministic geometry fragmentation.

Coincident shell area is not a harmless modelling detail: if two faces reach a
solver unchanged, their stiffness and mass are counted twice.  This module
therefore distinguishes boundary contact (valid) from positive-area overlap
and provides an explicit, undo-friendly fragmentation operation.

The selected face order is meaningful.  All geometric area is retained and
split into non-overlapping cells; where several source faces cover one cell,
the earliest selected face owns that cell and therefore supplies its material,
section and other face-scoped attributes through replacement lineage.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence

import numpy as np

from .curves import Straight
from .entities import EntityRef, OrientedEdge
from .errors import GeometryError
from .identity import validate_local_id
from .sketch import SketchPlane, face_sketch_plane
from .spatial import AABB, SpatialKey

__all__ = [
    "FaceOverlap",
    "OverlapFragmentation",
    "find_coplanar_overlaps",
    "fragment_coplanar_overlaps",
]


@dataclass(frozen=True)
class FaceOverlap:
    first: int
    second: int
    area: float


@dataclass(frozen=True)
class OverlapFragmentation:
    """Result of partitioning selected faces into disjoint planar cells."""

    outputs: Mapping[str, EntityRef]
    descendants: Mapping[int, tuple[int, ...]]
    overlap_faces: tuple[int, ...]
    overlap_area: float


def _shapely():
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
    except ImportError as error:  # pragma: no cover - environment diagnostic
        raise GeometryError(
            "planar overlap operations require the ANYgeometry 'planar' extra "
            "(shapely>=2.0)"
        ) from error
    return Polygon, unary_union


def _loop_xy(
    geometry, loop, plane: SketchPlane, *, strict_straight: bool
) -> list[tuple[float, float]]:
    coordinates: list[tuple[float, float]] = []
    for item in loop:
        edge = geometry.edges[item.edge]
        if strict_straight and not isinstance(edge.curve, Straight):
            raise GeometryError(
                "plate overlap fragmentation currently requires straight plate boundaries"
            )
        if isinstance(edge.curve, Straight):
            positions = (geometry.vertex_position(geometry.oriented_start_vertex(item)),)
        else:
            parameters = np.linspace(0.0, 1.0, 33)
            sampled = geometry.sample_edge(item.edge, parameters)
            if not item.forward:
                sampled = sampled[::-1]
            positions = sampled[:-1]
        for position in positions:
            local = plane.local(position)
            coordinates.append((float(local[0]), float(local[1])))
    return coordinates


def _face_polygon(
    geometry, face_id: int, plane: SketchPlane, tolerance: float, *, strict_straight: bool
):
    Polygon, _unary_union = _shapely()
    face = geometry.faces[int(face_id)]
    outer = _loop_xy(geometry, face.loop, plane, strict_straight=strict_straight)
    holes = [
        _loop_xy(geometry, loop, plane, strict_straight=strict_straight)
        for loop in face.holes
    ]
    polygon = Polygon(outer, holes)
    if not polygon.is_valid:
        raise GeometryError(f"plate {face_id} has an invalid planar boundary")
    if polygon.area <= tolerance * tolerance:
        raise GeometryError(f"plate {face_id} has negligible planar area")
    return polygon


def _plane_and_polygons(
    geometry,
    face_ids: Sequence[int],
    tolerance: float,
    *,
    strict_straight: bool,
    absolute_tolerance: bool = False,
) -> tuple[SketchPlane, dict[int, object], float]:
    if not face_ids:
        raise GeometryError("select at least one plate")
    plane = face_sketch_plane(geometry, int(face_ids[0]))
    participating = np.vstack(
        [
            geometry.vertex_position(geometry.oriented_start_vertex(item))
            for face_id in face_ids
            for loop in (
                geometry.faces[int(face_id)].loop,
                *geometry.faces[int(face_id)].holes,
            )
            for item in loop
        ]
    )
    # Classification must be invariant under translating the complete model.
    scale = max(float(np.linalg.norm(np.ptp(participating, axis=0))), 1.0)
    length_tolerance = (
        float(tolerance)
        if absolute_tolerance
        else float(tolerance) * max(scale, 1.0)
    )
    polygons: dict[int, object] = {}
    for face_id in face_ids:
        face = geometry.faces.get(int(face_id))
        if face is None:
            raise GeometryError(f"no plate {face_id}")
        points = np.vstack(
            [
                geometry.vertex_position(geometry.oriented_start_vertex(item))
                for loop in (face.loop,) + tuple(face.holes)
                for item in loop
            ]
        )
        distance = np.abs((points - plane.origin) @ plane.normal)
        if float(distance.max(initial=0.0)) > length_tolerance:
            raise GeometryError(
                "selected plates are not coplanar; crossing plates are imprinted automatically during meshing"
            )
        polygons[int(face_id)] = _face_polygon(
            geometry,
            int(face_id),
            plane,
            length_tolerance,
            strict_straight=strict_straight,
        )
    return plane, polygons, length_tolerance


def _parts(value, minimum_area: float):
    if value.is_empty:
        return []
    if value.geom_type == "Polygon":
        candidates = [value]
    elif value.geom_type == "MultiPolygon":
        candidates = list(value.geoms)
    elif value.geom_type == "GeometryCollection":
        candidates = [item for item in value.geoms if item.geom_type == "Polygon"]
    else:
        candidates = []
    return [item for item in candidates if float(item.area) > minimum_area]


def find_coplanar_overlaps(
    geometry,
    face_ids: Iterable[int] | None = None,
    *,
    changed_aabbs: Iterable[object] | None = None,
    candidate_pairs: Iterable[tuple[int, int] | tuple[SpatialKey, SpatialKey]] | None = None,
    tolerance: float | None = None,
) -> tuple[FaceOverlap, ...]:
    """Return positive-area coplanar overlaps from an indexed candidate set.

    Exactly one optional selector may be supplied:

    * ``face_ids`` qualifies pairs wholly inside a selected face set;
    * ``changed_aabbs`` qualifies face pairs incident to those regions; or
    * ``candidate_pairs`` consumes an already computed narrow-phase worklist.

    With no selector, face pairs come from the model's maintained AABB tree;
    the function never begins with a quadratic nested loop over all faces.
    Shared boundaries remain ignored.
    """

    tolerance_value = None if tolerance is None else float(tolerance)
    if tolerance_value is not None and (
        not np.isfinite(tolerance_value) or tolerance_value <= 0.0
    ):
        raise GeometryError("overlap tolerance must be finite and positive")

    supplied = sum(
        selector is not None
        for selector in (face_ids, changed_aabbs, candidate_pairs)
    )
    if supplied > 1:
        raise GeometryError(
            "face_ids, changed_aabbs, and candidate_pairs are mutually exclusive"
        )

    def normalized_pair(value: object) -> tuple[int, int]:
        try:
            first_raw, second_raw = value  # type: ignore[misc]
        except (TypeError, ValueError) as error:
            raise GeometryError("candidate pairs must contain two face IDs") from error

        def face_identifier(item: object) -> int:
            if (
                isinstance(item, tuple)
                and len(item) == 2
                and item[0] == "face"
            ):
                item = item[1]
            identifier = validate_local_id(item, name="face ID")
            if identifier not in geometry.faces:
                raise GeometryError(f"no plate {identifier}")
            return identifier

        first, second = face_identifier(first_raw), face_identifier(second_raw)
        if first == second:
            raise GeometryError("an overlap candidate must contain two distinct faces")
        return (first, second) if first < second else (second, first)

    if candidate_pairs is not None:
        pairs = tuple(sorted({normalized_pair(pair) for pair in candidate_pairs}))
    else:
        tree = geometry._spatial()  # noqa: SLF001 - kernel-maintained query index

        def face_margin(key: SpatialKey) -> float:
            if tolerance_value is not None:
                return tolerance_value
            bounds = tree.bounds(key)
            extent = float(np.linalg.norm(bounds.extents))
            policy = getattr(geometry, "tolerance", None)
            if policy is None:
                return 1.0e-9 * max(extent, 1.0)
            return max(
                float(getattr(policy, "aabb_padding", 0.0)),
                float(policy.effective_length(extent)),
            )

        def indexed_pairs(
            seeds: Iterable[SpatialKey],
            *,
            selected: frozenset[SpatialKey] | None = None,
        ) -> tuple[tuple[int, int], ...]:
            made: set[tuple[int, int]] = set()
            for seed in sorted(set(seeds)):
                if seed not in tree or seed[0] != "face":
                    continue
                nearby = tree.query(
                    tree.bounds(seed).expanded(face_margin(seed)),
                    kinds=("face",),
                )
                for other in nearby.keys:
                    if other == seed or (selected is not None and other not in selected):
                        continue
                    first_key, second_key = (
                        (seed, other) if seed < other else (other, seed)
                    )
                    made.add((first_key[1], second_key[1]))
            return tuple(sorted(made))

        if face_ids is not None:
            selected = tuple(
                dict.fromkeys(
                    validate_local_id(item, name="face ID") for item in face_ids
                )
            )
            missing = tuple(identifier for identifier in selected if identifier not in geometry.faces)
            if missing:
                raise GeometryError(f"no plate {missing[0]}")
            if len(selected) < 2:
                return ()
            selected_keys = frozenset(("face", identifier) for identifier in selected)
            pairs = indexed_pairs(selected_keys, selected=selected_keys)
        elif changed_aabbs is not None:
            regions: list[AABB] = []
            for value in changed_aabbs:
                if isinstance(value, AABB):
                    regions.append(value)
                    continue
                before = getattr(value, "before", None)
                after = getattr(value, "after", None)
                if hasattr(value, "before") and hasattr(value, "after"):
                    for bounds in (before, after):
                        if bounds is not None:
                            regions.append(AABB(tuple(bounds[:3]), tuple(bounds[3:])))
                    continue
                try:
                    raw = tuple(value)  # type: ignore[arg-type]
                except TypeError as error:
                    raise GeometryError(
                        "changed_aabbs must contain AABB, AABBChange, or six-value bounds"
                    ) from error
                if len(raw) == 6:
                    regions.append(AABB(tuple(raw[:3]), tuple(raw[3:])))
                elif len(raw) == 2:
                    regions.append(AABB(tuple(raw[0]), tuple(raw[1])))
                else:
                    raise GeometryError(
                        "changed_aabbs must contain AABB, AABBChange, or six-value bounds"
                    )
            if not regions:
                return ()
            policy = getattr(geometry, "tolerance", None)
            padding = (
                float(getattr(policy, "aabb_padding", getattr(policy, "length", 1.0e-9)))
                if tolerance_value is None
                else tolerance_value
            )
            regions = [region.expanded(padding) for region in regions]
            changed_faces = tree.query_regions(regions, kinds=("face",)).keys
            if not changed_faces:
                return ()
            pairs = indexed_pairs(changed_faces)
        else:
            pairs = indexed_pairs(key for key in tree.keys if key[0] == "face")

    if not pairs:
        return ()
    overlaps: list[FaceOverlap] = []
    # Each coplanar cluster may use a different deterministic frame.  Trying a
    # pair independently also lets non-coplanar faces remain ordinary models.
    for first, second in pairs:
        try:
            pair_tolerance = tolerance_value
            if pair_tolerance is None:
                first_raw = geometry._entity_bounds(("face", first))  # noqa: SLF001
                second_raw = geometry._entity_bounds(("face", second))  # noqa: SLF001
                if first_raw is None or second_raw is None:
                    raise GeometryError("selected plate cannot be conservatively bounded")
                pair_bounds = AABB(
                    tuple(first_raw[:3]), tuple(first_raw[3:])
                ).union(AABB(tuple(second_raw[:3]), tuple(second_raw[3:])))
                extent = float(np.linalg.norm(pair_bounds.extents))
                policy = getattr(geometry, "tolerance", None)
                pair_tolerance = float(
                    policy.effective_length(extent)
                    if policy is not None and hasattr(policy, "effective_length")
                    else 1.0e-9 * max(extent, 1.0)
                )
            _plane, polygons, length_tolerance = _plane_and_polygons(
                geometry,
                (first, second),
                pair_tolerance,
                strict_straight=False,
                absolute_tolerance=True,
            )
        except GeometryError as error:
            if "not coplanar" in str(error) or "flat plate" in str(error):
                continue
            raise
        area = float(polygons[first].intersection(polygons[second]).area)
        if area > length_tolerance * length_tolerance:
            overlaps.append(FaceOverlap(first, second, area))
    return tuple(overlaps)


def _ring_key(point: Sequence[float], quantum: float) -> tuple[int, int]:
    return tuple(int(round(float(item) / quantum)) for item in point)  # type: ignore[return-value]


def fragment_coplanar_overlaps(
    geometry, face_ids: Sequence[int], *, tolerance: float = 1.0e-9
) -> OverlapFragmentation:
    """Replace selected coplanar faces by a disjoint planar arrangement.

    The first selected face owns any common cell.  No area is deleted.  Old
    faces receive explicit replacement lineage to the cells they own.
    """

    identifiers = tuple(dict.fromkeys(int(item) for item in face_ids))
    if len(identifiers) < 2:
        raise GeometryError("select at least two plates in ownership order")
    with geometry.transaction():
        plane, polygons, length_tolerance = _plane_and_polygons(
            geometry, identifiers, tolerance, strict_straight=True
        )
        _Polygon, unary_union = _shapely()
        minimum_area = length_tolerance * length_tolerance
        cells: list[tuple[object, frozenset[int], int]] = []
        for order, face_id in enumerate(identifiers):
            polygon = polygons[face_id]
            if not cells:
                cells.extend(
                    (part, frozenset((face_id,)), order)
                    for part in _parts(polygon, minimum_area)
                )
                continue
            occupied = unary_union([item[0] for item in cells])
            split: list[tuple[object, frozenset[int], int]] = []
            for cell, memberships, owner in cells:
                split.extend(
                    (part, memberships | {face_id}, owner)
                    for part in _parts(cell.intersection(polygon), minimum_area)
                )
                split.extend(
                    (part, memberships, owner)
                    for part in _parts(cell.difference(polygon), minimum_area)
                )
            split.extend(
                (part, frozenset((face_id,)), order)
                for part in _parts(polygon.difference(occupied), minimum_area)
            )
            cells = split

        overlap_area = sum(
            float(cell.area) for cell, memberships, _owner in cells
            if len(memberships) > 1
        )
        if overlap_area <= minimum_area:
            raise GeometryError("the selected plates have no positive-area overlap")

        old_faces = {face_id: geometry.faces[face_id] for face_id in identifiers}
        old_edges = {
            item.edge
            for face in old_faces.values()
            for loop in (face.loop,) + tuple(face.holes)
            for item in loop
        }
        old_vertices = {
            vertex
            for edge_id in old_edges
            for vertex in (
                geometry.edges[edge_id].start,
                geometry.edges[edge_id].end,
            )
        }
        for face_id in identifiers:
            geometry.remove_face(face_id, record=False)

        quantum = max(length_tolerance, 1.0e-12)
        vertex_by_key: dict[tuple[int, int], int] = {}
        for vertex_id in sorted(old_vertices):
            local = plane.local(geometry.vertex_position(vertex_id))
            vertex_by_key.setdefault(_ring_key(local, quantum), vertex_id)

        edge_by_vertices: dict[tuple[int, int], int] = {}
        for edge_id, edge in sorted(geometry.edges.items()):
            if isinstance(edge.curve, Straight):
                edge_by_vertices.setdefault(
                    tuple(sorted((edge.start, edge.end))), edge_id
                )

        def vertex(point) -> int:
            key = _ring_key(point, quantum)
            existing = vertex_by_key.get(key)
            if existing is not None:
                return existing
            made = geometry.add_point(*plane.world(point))
            vertex_by_key[key] = made
            return made

        def loop(coordinates) -> tuple[OrientedEdge, ...]:
            points = list(coordinates)[:-1]
            vertices = [vertex(item) for item in points]
            result: list[OrientedEdge] = []
            for start, end in zip(vertices, vertices[1:] + vertices[:1]):
                key = tuple(sorted((start, end)))
                edge_id = edge_by_vertices.get(key)
                if edge_id is None:
                    edge_id = geometry.add_line(start, end)
                    edge_by_vertices[key] = edge_id
                edge = geometry.edges[edge_id]
                result.append(OrientedEdge(edge_id, edge.start == start))
            return tuple(result)

        ordered_cells = sorted(
            cells,
            key=lambda item: (
                item[2],
                tuple(identifiers.index(value) for value in identifiers if value in item[1]),
                round(float(item[0].centroid.x), 12),
                round(float(item[0].centroid.y), 12),
                round(float(item[0].area), 12),
            ),
        )
        descendants: dict[int, list[int]] = {face_id: [] for face_id in identifiers}
        overlap_faces: list[int] = []
        outputs: dict[str, EntityRef] = {}
        owner_counts: dict[int, int] = {face_id: 0 for face_id in identifiers}
        for polygon, memberships, owner_order in ordered_cells:
            owner_id = identifiers[owner_order]
            outer = loop(polygon.exterior.coords)
            corners = (  # noqa: SLF001
                geometry._detect_corners(outer) if len(outer) == 4 else None
            )
            owner_face = old_faces[owner_id]
            made = geometry.add_face_from_loop(
                outer,
                corners,
                surface=owner_face.surface,
            )
            geometry._put_entity(  # noqa: SLF001
                "face",
                replace(
                    geometry.faces[made],
                    holes=tuple(loop(ring.coords) for ring in polygon.interiors),
                    surface=owner_face.surface,
                    parameterization=owner_face.parameterization,
                    metadata={
                        **old_faces[owner_id].metadata,
                    "overlap_fragment": True,
                    "source_faces": tuple(sorted(memberships)),
                    "overlap_owner": owner_id,
                    },
                ),
            )
            descendants[owner_id].append(made)
            if len(memberships) > 1:
                overlap_faces.append(made)
            index = owner_counts[owner_id]
            owner_counts[owner_id] += 1
            prefix = "overlap" if len(memberships) > 1 else "plate"
            outputs[f"{prefix}/{owner_order}/{index}"] = EntityRef("face", made)

        for face_id in identifiers:
            geometry.record_replacement(
                EntityRef("face", face_id),
                tuple(EntityRef("face", item) for item in descendants[face_id]),
            )

        # Retire unused source boundary topology.  New fragments intentionally
        # share their constructed vertices and edges, so the resulting model
        # has one conformal boundary rather than duplicate coincident curves.
        for edge_id in sorted(old_edges):
            if edge_id in geometry.edges and not geometry.faces_using_edge(edge_id):
                old_edge = geometry.edges[edge_id]
                start = geometry.vertex_position(old_edge.start)
                end = geometry.vertex_position(old_edge.end)
                direction = end - start
                length_squared = float(direction @ direction)
                replacements: list[tuple[float, EntityRef]] = []
                if isinstance(old_edge.curve, Straight) and length_squared > 0.0:
                    for candidate_id, candidate in geometry.edges.items():
                        if (
                            candidate_id == edge_id
                            or not isinstance(candidate.curve, Straight)
                            or not geometry.faces_using_edge(candidate_id)
                        ):
                            continue
                        parameters: list[float] = []
                        fits = True
                        for vertex_id in (candidate.start, candidate.end):
                            position = geometry.vertex_position(vertex_id)
                            parameter = float((position - start) @ direction / length_squared)
                            projected = start + parameter * direction
                            if (
                                parameter < -tolerance
                                or parameter > 1.0 + tolerance
                                or float(np.linalg.norm(position - projected))
                                > length_tolerance
                            ):
                                fits = False
                                break
                            parameters.append(parameter)
                        if fits:
                            replacements.append(
                                (min(parameters), EntityRef("edge", candidate_id))
                            )
                geometry.remove_edge(edge_id, record=False)
                geometry.record_replacement(
                    EntityRef("edge", edge_id),
                    tuple(
                        reference
                        for _parameter, reference in sorted(
                            replacements, key=lambda item: (item[0], item[1].id)
                        )
                    ),
                )
        for vertex_id in sorted(old_vertices):
            if vertex_id in geometry.vertices and not geometry.edges_using_vertex(vertex_id):
                local = plane.local(geometry.vertex_position(vertex_id))
                target = vertex_by_key.get(_ring_key(local, quantum))
                replacements = (
                    (EntityRef("vertex", target),)
                    if target is not None
                    and target != vertex_id
                    and target in geometry.vertices
                    and geometry.edges_using_vertex(target)
                    else ()
                )
                geometry.remove_vertex(vertex_id, record=False)
                geometry.record_replacement(EntityRef("vertex", vertex_id), replacements)

        errors = geometry.validate_topology()
        if errors:
            raise GeometryError(
                "plate overlap fragmentation produced invalid topology: "
                + "; ".join(errors)
            )
        return OverlapFragmentation(
            outputs,
            {key: tuple(value) for key, value in descendants.items()},
            tuple(overlap_faces),
            float(overlap_area),
        )
