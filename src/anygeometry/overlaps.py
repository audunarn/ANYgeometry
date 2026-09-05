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

from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence
from uuid import UUID

import numpy as np

from .curves import Straight
from .entities import EntityRef, OrientedEdge
from .errors import GeometryError
from .identity import EntityHandle, validate_local_id
from .sketch import SketchPlane, face_sketch_plane
from .spatial import AABB, SpatialKey
from .surfaces import Plane

__all__ = [
    "FaceOverlap",
    "OverlapFragmentation",
    "OverlapFragmentationPlan",
    "OverlapOwnershipPolicy",
    "OverlapQualificationError",
    "plan_coplanar_fragmentation",
    "apply_coplanar_fragmentation",
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
    work_counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))
        object.__setattr__(self, "descendants", MappingProxyType({
            key: tuple(value) for key, value in self.descendants.items()
        }))
        object.__setattr__(self, "overlap_faces", tuple(self.overlap_faces))
        object.__setattr__(self, "work_counts", MappingProxyType(dict(self.work_counts)))


class OverlapOwnershipPolicy(str, Enum):
    FIRST_SELECTED = "first_selected"


class OverlapQualificationError(GeometryError):
    """No complete overlap answer is available; partial answers are withheld."""

    def __init__(self, candidate_pairs, diagnostics):
        self._candidate_pairs = tuple(tuple(pair) for pair in candidate_pairs)
        self._diagnostics = tuple(str(item) for item in diagnostics)
        super().__init__("overlap qualification incomplete: " + "; ".join(self._diagnostics))

    @property
    def candidate_pairs(self) -> tuple[tuple[EntityHandle, EntityHandle], ...]:
        return self._candidate_pairs

    @property
    def diagnostics(self) -> tuple[str, ...]:
        return self._diagnostics


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class OverlapFragmentationPlan:
    """Ephemeral preview; IDs are allocated only when the plan is applied."""

    model_id: UUID
    revision: int
    source_handles: tuple[EntityHandle, ...]
    ownership_policy: OverlapOwnershipPolicy
    tolerance: float
    expected_fragment_counts: Mapping[int, int]
    effects: Mapping[str, object]
    blockers: tuple[str, ...]
    digest: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_handles", tuple(self.source_handles))
        object.__setattr__(self, "expected_fragment_counts", _freeze(dict(self.expected_fragment_counts)))
        object.__setattr__(self, "effects", _freeze(dict(self.effects)))
        object.__setattr__(self, "blockers", tuple(self.blockers))


def _plan_digest(plan: OverlapFragmentationPlan) -> str:
    payload = {
        "model_id": str(plan.model_id), "revision": plan.revision,
        "sources": [(str(h.model_id), h.kind, h.id) for h in plan.source_handles],
        "policy": plan.ownership_policy.value, "tolerance": plan.tolerance,
        "counts": sorted(plan.expected_fragment_counts.items()),
        "effects": _plain(plan.effects), "blockers": plan.blockers,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


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
        if not isinstance(edge.curve, Straight):
            raise GeometryError(
                "plate overlap fragmentation currently requires straight plate boundaries"
            )
        positions = (geometry.vertex_position(geometry.oriented_start_vertex(item)),)
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
    cache: dict | None = None,
) -> tuple[SketchPlane, dict[int, object], float]:
    if not face_ids:
        raise GeometryError("select at least one plate")
    cache = {} if cache is None else cache
    anchor = int(face_ids[0])
    if ("plane", anchor) not in cache:
        cache[("plane", anchor)] = face_sketch_plane(geometry, anchor)
    plane = cache[("plane", anchor)]
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
        key = ("polygon", anchor, int(face_id), length_tolerance)
        if key not in cache:
            cache[key] = _face_polygon(
                geometry, int(face_id), plane, length_tolerance,
                strict_straight=strict_straight,
            )
        polygons[int(face_id)] = cache[key]
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
    unresolved: list[tuple[EntityHandle, EntityHandle]] = []
    diagnostics: list[str] = []
    projected_cache: dict = {}
    # Each coplanar cluster may use a different deterministic frame.  Trying a
    # pair independently also lets non-coplanar faces remain ordinary models.
    for first, second in pairs:
        curved = any(geometry.faces[i].surface is not None
                     and not isinstance(geometry.faces[i].surface, Plane)
                     for i in (first, second)) or any(
            not isinstance(geometry.edges[item.edge].curve, Straight)
            for face_id in (first, second)
            for loop in (geometry.faces[face_id].loop, *geometry.faces[face_id].holes)
            for item in loop
        )
        if curved:
            from .intersections import query_intersection
            from .predicates import IntersectionDimension, IntersectionKind

            handles = (geometry.handle("face", first), geometry.handle("face", second))
            try:
                result = query_intersection(geometry, *handles)
                complete = (
                    result.certificate is not None and result.certificate.complete
                    and all(c.certificate is not None and c.certificate.complete
                            for c in result.components)
                    and (tolerance_value is None or result.tolerance_used == tolerance_value)
                )
                if complete and (
                    result.kind is IntersectionKind.DISJOINT
                    or (result.kind not in (
                        IntersectionKind.UNCLASSIFIED, IntersectionKind.UNSUPPORTED,
                        IntersectionKind.CAPABILITY_MISSING,
                    ) and result.dimension in (IntersectionDimension.POINT, IntersectionDimension.CURVE))
                ):
                    continue
                # The current certificate describes intersections, not a
                # certified curved-region area. Never invent area from chords.
                reason = "certified curved-region area unavailable" if complete else "incomplete intersection certificate"
                diagnostics.append(f"faces {first}/{second}: {reason}; " + "; ".join(result.diagnostics))
            except (GeometryError, ArithmeticError, ValueError, TypeError, AttributeError) as error:
                diagnostics.append(f"faces {first}/{second}: {error}")
            unresolved.append(handles)
            continue
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
                strict_straight=True,
                absolute_tolerance=True,
                cache=projected_cache,
            )
        except GeometryError as error:
            if "not coplanar" in str(error) or "flat plate" in str(error):
                continue
            raise
        area = float(polygons[first].intersection(polygons[second]).area)
        if area > length_tolerance * length_tolerance:
            overlaps.append(FaceOverlap(first, second, area))
    if unresolved:
        raise OverlapQualificationError(unresolved, diagnostics)
    return tuple(overlaps)


def _ring_key(point: Sequence[float], quantum: float) -> tuple[int, int]:
    return tuple(int(round(float(item) / quantum)) for item in point)  # type: ignore[return-value]


def _fragment_ids(geometry, face_ids):
    if isinstance(face_ids, (set, frozenset, Mapping, str, bytes)):
        raise GeometryError("fragmentation requires ordered face IDs")
    try:
        identifiers = tuple(dict.fromkeys(validate_local_id(item, name="face ID") for item in face_ids))
    except TypeError as error:
        raise GeometryError("fragmentation requires ordered face IDs") from error
    if len(identifiers) < 2:
        raise GeometryError("select at least two plates in ownership order")
    for identifier in identifiers:
        geometry._require_face(identifier)
    return identifiers


def _prepare_fragmentation(geometry, identifiers, tolerance):
    if any(geometry.faces[i].surface is not None and not isinstance(geometry.faces[i].surface, Plane)
           for i in identifiers):
        raise GeometryError("overlap fragmentation requires qualified planar support")
    if any(not isinstance(geometry.edges[item.edge].curve, Straight)
           for i in identifiers for loop in (geometry.faces[i].loop, *geometry.faces[i].holes)
           for item in loop):
        raise GeometryError("plate overlap fragmentation currently requires straight plate boundaries")
    points = np.vstack([
        geometry.vertex_position(geometry.oriented_start_vertex(item))
        for face_id in identifiers
        for loop in (geometry.faces[face_id].loop, *geometry.faces[face_id].holes)
        for item in loop
    ])
    if tolerance is None:
        tolerance = geometry.tolerance.effective_length(float(np.linalg.norm(np.ptp(points, axis=0))))
    if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float, np.number))
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise GeometryError("overlap tolerance must be finite and positive")
    plane, polygons, length_tolerance = _plane_and_polygons(
        geometry, identifiers, float(tolerance), strict_straight=True, absolute_tolerance=True
    )
    _Polygon, unary_union = _shapely()
    minimum_area = length_tolerance * length_tolerance
    cells: list[tuple[object, frozenset[int], int]] = []
    for order, face_id in enumerate(identifiers):
        polygon = polygons[face_id]
        if not cells:
            cells.extend((part, frozenset((face_id,)), order) for part in _parts(polygon, minimum_area))
            continue
        occupied = unary_union([item[0] for item in cells])
        split: list[tuple[object, frozenset[int], int]] = []
        for cell, memberships, owner in cells:
            split.extend((part, memberships | {face_id}, owner)
                         for part in _parts(cell.intersection(polygon), minimum_area))
            split.extend((part, memberships, owner)
                         for part in _parts(cell.difference(polygon), minimum_area))
        split.extend((part, frozenset((face_id,)), order)
                     for part in _parts(polygon.difference(occupied), minimum_area))
        cells = split
    overlap_area = sum(float(cell.area) for cell, memberships, _owner in cells if len(memberships) > 1)
    if overlap_area <= minimum_area:
        raise GeometryError("the selected plates have no positive-area overlap")
    return plane, cells, length_tolerance, overlap_area


def _make_fragmentation_plan(geometry, identifiers, ownership_policy, prepared):
    _plane, cells, tolerance, overlap_area = prepared
    counts = {face_id: sum(owner == order for _, _, owner in cells)
              for order, face_id in enumerate(identifiers)}
    faces = {}
    sheets = {}
    blockers = []
    edges = {item.edge for i in identifiers for loop in (geometry.faces[i].loop, *geometry.faces[i].holes) for item in loop}
    vertices = {v for e in edges for v in (geometry.edges[e].start, geometry.edges[e].end)}
    selected_refs = {EntityRef("face", i) for i in identifiers}
    selected_refs.update(EntityRef("edge", i) for i in edges)
    selected_refs.update(EntityRef("vertex", i) for i in vertices)
    groups = {ref: [] for ref in selected_refs}
    for name, members in geometry.groups.items():
        for ref in selected_refs.intersection(members):
            groups[ref].append(name)
    for face_id in identifiers:
        ref = EntityRef("face", face_id)
        faces[str(face_id)] = {
            "groups": sorted(groups[ref]), "tags": sorted(geometry.tags.get(ref, ())),
            "metadata": geometry.faces[face_id].metadata,
            "descendant_count": counts[face_id],
        }
        if geometry._target_attachments.get(("face", face_id)):
            blockers.append(f"face {face_id}: attachments require explicit remapping")
        for use_id in sorted(geometry._face_structural_uses.get(face_id, ())):
            use = geometry.face_uses[use_id]
            sheet = geometry.sheets[use.sheet_id]
            remaining = sum(counts.get(geometry.face_uses[i].face_id, 1) for i in sheet.face_use_ids)
            sheets[str(sheet.id)] = {
                "name": sheet.name, "part_id": sheet.part_id,
                "part_name": geometry.parts[sheet.part_id].name,
                "face_use_count_after": remaining, "removed": remaining == 0,
            }
            if remaining == 0 and (geometry._target_attachments.get(("sheet", sheet.id))
                                   or geometry._sheet_junctions.get(sheet.id)):
                blockers.append(f"sheet {sheet.id}: attachments/junctions prevent removal")
    for edge_id in sorted(edges):
        if geometry._edge_member_uses.get(edge_id) or geometry._target_attachments.get(("edge", edge_id)):
            blockers.append(f"edge {edge_id}: member/attachment dependency requires explicit remapping")
    conflicts = []
    for _cell, members, owner in cells:
        if len(members) > 1:
            owner_id = identifiers[owner]
            for other in sorted(members - {owner_id}):
                if geometry.faces[owner_id].metadata != geometry.faces[other].metadata:
                    conflicts.append((owner_id, other))
    # Preview boundary label disposition from the same arrangement, without
    # allocating descendant IDs or searching unrelated geometry.
    segments = {}
    for cell, _members, _owner in cells:
        for ring in (cell.exterior, *cell.interiors):
            points = tuple(ring.coords)
            for a, b in zip(points, points[1:]):
                key = tuple(sorted((_ring_key(a, tolerance), _ring_key(b, tolerance))))
                segments[key] = (np.asarray(a), np.asarray(b))
    corner_keys = {key for segment in segments for key in segment}
    boundary_labels = {}
    for ref in sorted(selected_refs, key=lambda item: (item.kind, item.id)):
        if ref.kind == "face":
            continue
        if ref.kind == "vertex":
            used_outside = any(set(geometry.faces_using_edge(e)) - set(identifiers)
                               or e not in edges for e in geometry.edges_using_vertex(ref.id))
            count = int(used_outside or _ring_key(_plane.local(geometry.vertex_position(ref.id)), tolerance) in corner_keys)
        else:
            edge = geometry.edges[ref.id]
            a = _plane.local(geometry.vertex_position(edge.start))
            b = _plane.local(geometry.vertex_position(edge.end))
            direction = b - a
            length = float(np.linalg.norm(direction))
            count = 0
            for start, end in segments.values():
                parameters = ((start - a) @ direction / length**2, (end - a) @ direction / length**2)
                if all(-tolerance/length <= t <= 1 + tolerance/length
                       and np.linalg.norm(p - (a + t*direction)) <= tolerance
                       for p, t in zip((start, end), parameters)):
                    count += 1
            if set(geometry.faces_using_edge(ref.id)) - set(identifiers):
                count = 1  # Existing external owner keeps this identity/labels.
        boundary_labels[f"{ref.kind}/{ref.id}"] = {
            "groups": sorted(groups[ref]), "tags": sorted(geometry.tags.get(ref, ())),
            "descendant_count": count, "removed_without_descendant": count == 0,
        }
    effects = {
        "frame": {"origin": tuple(float(x) for x in _plane.origin),
                  "x_axis": tuple(float(x) for x in _plane.x_axis),
                  "y_axis": tuple(float(x) for x in _plane.y_axis)},
        "faces": faces, "sheets": sheets, "boundary_labels": boundary_labels,
        "metadata_conflicts": sorted(set(conflicts)), "overlap_area": overlap_area,
        "cells": [{"owner": identifiers[owner], "sources": sorted(members),
                   "outer": tuple(tuple(p) for p in cell.exterior.coords),
                   "holes": tuple(tuple(tuple(p) for p in ring.coords) for ring in cell.interiors)}
                  for cell, members, owner in cells],
    }
    plan = OverlapFragmentationPlan(
        geometry.model_id, geometry.revision,
        tuple(geometry.handle("face", i) for i in identifiers), ownership_policy,
        tolerance, counts, effects, tuple(sorted(set(blockers))),
    )
    return replace(plan, digest=_plan_digest(plan))


def plan_coplanar_fragmentation(geometry, face_ids: Sequence[int], *, ownership_policy=None,
                               tolerance: float | None = None) -> OverlapFragmentationPlan:
    """Preview explicit first-selected ownership without touching live state."""
    if ownership_policy is None:
        raise GeometryError("overlap fragmentation requires explicit ownership_policy")
    try:
        policy = OverlapOwnershipPolicy(ownership_policy)
    except (ValueError, TypeError) as error:
        raise GeometryError("unknown overlap ownership policy") from error
    identifiers = _fragment_ids(geometry, face_ids)
    return _make_fragmentation_plan(geometry, identifiers, policy,
                                    _prepare_fragmentation(geometry, identifiers, tolerance))


def apply_coplanar_fragmentation(geometry, plan: OverlapFragmentationPlan) -> OverlapFragmentation:
    """Verify both the digest and the live preview before one atomic edit."""
    if not isinstance(plan, OverlapFragmentationPlan):
        raise GeometryError("expected an OverlapFragmentationPlan")
    if plan.model_id != geometry.model_id:
        raise GeometryError("WRONG_MODEL: overlap plan belongs to another model")
    if plan.revision != geometry.revision:
        raise GeometryError("STALE_PLAN: overlap plan revision changed")
    if plan.ownership_policy is not OverlapOwnershipPolicy.FIRST_SELECTED:
        raise GeometryError("invalid overlap ownership policy")
    try:
        if plan.digest != _plan_digest(plan):
            raise GeometryError("invalid overlap plan digest")
        if any(h.model_id != geometry.model_id or h.kind != "face" for h in plan.source_handles):
            raise GeometryError("invalid overlap plan handles")
        identifiers = _fragment_ids(geometry, [h.id for h in plan.source_handles])
        prepared = _prepare_fragmentation(geometry, identifiers, plan.tolerance)
        expected = _make_fragmentation_plan(geometry, identifiers, plan.ownership_policy, prepared)
        if expected.digest != plan.digest:
            raise GeometryError("overlap plan differs from live preview")
        if expected.blockers:
            raise GeometryError("; ".join(expected.blockers))
    except GeometryError:
        raise
    except (TypeError, ValueError, AttributeError) as error:
        raise GeometryError("invalid overlap plan") from error
    return _fragment_coplanar_overlaps(geometry, identifiers, prepared)


def fragment_coplanar_overlaps(geometry, face_ids: Sequence[int], *, ownership_policy=None,
                              tolerance: float | None = None) -> OverlapFragmentation:
    """Plan/apply convenience API; ownership acknowledgment is mandatory."""
    return apply_coplanar_fragmentation(geometry, plan_coplanar_fragmentation(
        geometry, face_ids, ownership_policy=ownership_policy, tolerance=tolerance,
    ))


def _fragment_coplanar_overlaps(geometry, identifiers, prepared) -> OverlapFragmentation:
    plane, cells, length_tolerance, overlap_area = prepared
    edge_comparisons = 0
    with geometry.transaction():
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

        quantum = length_tolerance
        vertex_by_key: dict[tuple[int, int], int] = {}
        for vertex_id in sorted(old_vertices):
            local = plane.local(geometry.vertex_position(vertex_id))
            vertex_by_key.setdefault(_ring_key(local, quantum), vertex_id)

        edge_by_vertices: dict[tuple[int, int], int] = {}
        incident_edges = {edge_id for vertex_id in old_vertices
                          for edge_id in geometry.edges_using_vertex(vertex_id)}
        for edge_id in sorted(incident_edges):
            edge = geometry.edges[edge_id]
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

        label_replacements = []
        for face_id in identifiers:
            replacements = tuple(EntityRef("face", item) for item in descendants[face_id])
            geometry._record_replacement(
                EntityRef("face", face_id),
                replacements, _propagate_labels=False,
            )
            label_replacements.append((EntityRef("face", face_id), replacements))

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
                    for candidate_id in sorted(set(edge_by_vertices.values())):
                        if candidate_id not in geometry.edges:
                            continue
                        edge_comparisons += 1
                        candidate = geometry.edges[candidate_id]
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
                                parameter < -length_tolerance / np.sqrt(length_squared)
                                or parameter > 1.0 + length_tolerance / np.sqrt(length_squared)
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
                descendants_for_edge = tuple(
                        reference
                        for _parameter, reference in sorted(
                            replacements, key=lambda item: (item[0], item[1].id)
                        )
                    )
                geometry._record_replacement(
                    EntityRef("edge", edge_id), descendants_for_edge, _propagate_labels=False,
                )
                label_replacements.append((EntityRef("edge", edge_id), descendants_for_edge))
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
                geometry._record_replacement(EntityRef("vertex", vertex_id), replacements, _propagate_labels=False)
                label_replacements.append((EntityRef("vertex", vertex_id), replacements))

        geometry._propagate_replacement_labels(label_replacements)

        # The outer transaction validates all journaled faces/edges and their
        # incidence closure, including structural ownership and changed labels.
        # Do not validate unrelated faces a second time here. Its existing
        # conservative structural fallback remains owned by the model validator.
        return OverlapFragmentation(
            outputs,
            {key: tuple(value) for key, value in descendants.items()},
            tuple(overlap_faces),
            float(overlap_area),
            {"source_faces": len(identifiers), "source_edges": len(old_edges),
             "incident_edges": len(incident_edges), "edge_comparisons": edge_comparisons,
             "output_faces": len(outputs)},
        )
