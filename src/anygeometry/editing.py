"""High-level neutral editing, duplication, and measurement operations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from itertools import product
from typing import Iterable, Mapping, Sequence

import numpy as np

from .closure import ModelClosure, extract_model_closure
from .curves import Arc, Spline
from .entities import EntityRef, OrientedEdge
from .errors import GeometryError
from .identity import EntityHandle, EntityKey, validate_local_id
from .model import GeometryModel
from .operations import transform
from .structural import (
    Attachment,
    AttachmentTargetKind,
    JunctionMemberUse,
    Orientation,
    ParameterRange,
)
from .surfaces import CoonsSurface, Cone, Cylinder, Plane, RuledSurface
from .transforms import AffineLike, AffineTransform, coerce_affine_transform

__all__ = [
    "InsertResult",
    "Measurement",
    "PatternResult",
    "circular_pattern",
    "copy_entities",
    "copy_rotated",
    "copy_translated",
    "insert_model",
    "linear_pattern",
    "measure",
    "mirror_entities",
    "move_entities",
    "pattern_entities",
    "rectangular_pattern",
    "reverse_edge",
    "reverse_face",
    "rotate_entities",
    "translate_entities",
]


@dataclass(frozen=True)
class InsertResult:
    """Complete identity map produced by inserting or copying topology."""

    entity_map: Mapping[EntityRef, EntityRef]
    groups: Mapping[str, tuple[EntityRef, ...]]
    outputs: Mapping[str, EntityRef]
    handle_map: Mapping[EntityHandle, EntityHandle] = field(default_factory=dict)

    def mapped(self, reference: EntityRef) -> EntityRef:
        return self.entity_map[reference]

    def mapped_handle(self, handle: EntityHandle) -> EntityHandle:
        """Return the fresh model-bound identity corresponding to ``handle``."""

        return self.handle_map[handle]


@dataclass(frozen=True)
class PatternResult:
    instances: tuple[InsertResult, ...]
    transforms: tuple[AffineTransform, ...] = ()


@dataclass(frozen=True)
class Measurement:
    kind: str
    value: float | tuple[float, ...]
    unit: str
    witnesses: tuple[tuple[float, float, float], ...] = ()


Selection = EntityRef | EntityHandle | EntityKey


def _affine_transform(value: AffineLike | None) -> AffineTransform:
    return coerce_affine_transform(value, default_identity=True)


def _expanded_refs(
    geometry: GeometryModel, references: Iterable[EntityRef] | None
) -> tuple[set[int], set[int], set[int]]:
    if references is None:
        return set(geometry.vertices), set(geometry.edges), set(geometry.faces)
    current: list[EntityRef] = []
    for reference in references:
        if not isinstance(reference, EntityRef):
            raise GeometryError("copy selection must contain EntityRef values")
        canonical = geometry.entity_ref(
            reference.kind,
            validate_local_id(reference.id, name="copy selection entity ID"),
        )
        current.extend(geometry.resolve_ref(canonical))
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
        destination._put_entity(  # noqa: SLF001
            "face",
            replace(
                destination.faces[made],
                holes=tuple(mapped_loop(loop) for loop in face.holes),
                metadata=deepcopy(face.metadata),
                parameterization=deepcopy(face.parameterization),
            ),
        )
        mapping[EntityRef("face", face_id)] = EntityRef("face", made)

    structural_mapping = _copy_structural_closure(
        destination,
        source,
        mapping,
        vertex_ids,
        edge_ids,
        face_ids,
        include_empty_parts=references is None,
    )

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
    handle_mapping = {
        source.handle(old.kind, old.id): destination.handle(new.kind, new.id)
        for old, new in mapping.items()
    }
    handle_mapping.update(
        {
            source.handle(kind, identifier): destination.handle(
                mapped_kind, mapped_identifier
            )
            for (kind, identifier), (mapped_kind, mapped_identifier)
            in structural_mapping.items()
        }
    )
    return InsertResult(dict(mapping), made_groups, outputs, handle_mapping)


def _copy_structural_closure(
    destination: GeometryModel,
    source: GeometryModel,
    entity_mapping: Mapping[EntityRef, EntityRef],
    vertex_ids: set[int],
    edge_ids: set[int],
    face_ids: set[int],
    *,
    include_empty_parts: bool,
) -> dict[EntityKey, EntityKey]:
    """Copy every complete structural ownership closure with fresh IDs.

    A partial selection does not invent ownership for an incomplete sheet or
    member.  Complete sheets/members, and relationships whose every parent is
    copied, retain their exact neutral structural records.
    """

    selected_geometry: Mapping[str, set[int]] = {
        "vertex": vertex_ids,
        "edge": edge_ids,
        "face": face_ids,
    }

    def has_selected_geometry(key: EntityKey | None) -> bool:
        return key is None or key[1] in selected_geometry[key[0]]

    eligible_sheets = {
        sheet.id
        for sheet in source.sheets.values()
        if all(
            source.face_uses[use_id].face_id in face_ids
            for use_id in sheet.face_use_ids
        )
    }
    eligible_members = {
        member.id
        for member in source.members.values()
        if all(
            source.member_edge_uses[use_id].edge_id in edge_ids
            for use_id in member.edge_use_ids
        )
        and has_selected_geometry(member.orientation_reference)
    }
    selected_construction = {
        vertex_id: source.construction_vertices[vertex_id]
        for vertex_id in vertex_ids
        if vertex_id in source.construction_vertices
    }
    construction_owner_parts = {
        part_id for part_id in selected_construction.values() if part_id is not None
    }
    eligible_parts = {
        part.id
        for part in source.parts.values()
        if include_empty_parts
        or any(item in eligible_sheets for item in part.sheet_ids)
        or any(item in eligible_members for item in part.member_ids)
        or part.id in construction_owner_parts
    }
    eligible_sheets = {
        item for item in eligible_sheets
        if source.sheets[item].part_id in eligible_parts
    }
    eligible_members = {
        item for item in eligible_members
        if source.members[item].part_id in eligible_parts
    }
    sheet_use_ids = {
        use_id
        for sheet_id in eligible_sheets
        for use_id in source.sheets[sheet_id].face_use_ids
    }
    coedge_ids = {
        coedge_id
        for use_id in sheet_use_ids
        for coedge_id in source.face_uses[use_id].coedge_ids
    }
    member_use_ids = {
        use_id
        for member_id in eligible_members
        for use_id in source.members[member_id].edge_use_ids
    }
    present_parents: Mapping[str, set[int]] = {
        **selected_geometry,
        "member": eligible_members,
        "sheet": eligible_sheets,
    }

    def attachment_is_complete(attachment: Attachment) -> bool:
        assert attachment.source_kind is not None
        assert attachment.source_id is not None
        return (
            attachment.source_id in present_parents[attachment.source_kind]
            and attachment.target_id
            in present_parents[attachment.target_kind.value]
            and (
                attachment.member_id is None
                or attachment.member_id in eligible_members
            )
            and (
                attachment.part_id is None
                or attachment.part_id in eligible_parts
            )
            and (
                attachment.sheet_id is None
                or attachment.sheet_id in eligible_sheets
            )
        )

    eligible_attachments = {
        attachment.id
        for attachment in source.attachments.values()
        if attachment_is_complete(attachment)
    }
    eligible_junctions = {
        junction.id
        for junction in source.junctions.values()
        if all(use.member_id in eligible_members for use in junction.member_uses)
        and all(item in eligible_sheets for item in junction.sheet_ids)
        and all(item in eligible_attachments for item in junction.attachment_ids)
    }

    def fresh(kind: str, identifiers: Iterable[int]) -> dict[int, int]:
        return {
            identifier: destination._allocate_structural(kind)  # noqa: SLF001
            for identifier in sorted(identifiers)
        }

    part_map = fresh("part", eligible_parts)
    sheet_map = fresh("sheet", eligible_sheets)
    face_use_map = fresh("face_use", sheet_use_ids)
    coedge_map = fresh("coedge", coedge_ids)
    member_map = fresh("member", eligible_members)
    member_use_map = fresh("member_edge_use", member_use_ids)
    attachment_map = fresh("attachment", eligible_attachments)
    junction_map = fresh("junction", eligible_junctions)

    structural_maps: Mapping[str, Mapping[int, int]] = {
        "part": part_map,
        "sheet": sheet_map,
        "face_use": face_use_map,
        "coedge": coedge_map,
        "member": member_map,
        "member_edge_use": member_use_map,
        "attachment": attachment_map,
        "junction": junction_map,
    }

    def mapped_geometry_key(key: EntityKey) -> EntityKey:
        mapped = entity_mapping.get(EntityRef(key[0], key[1]))  # type: ignore[arg-type]
        if mapped is None:
            raise GeometryError(
                f"structural copy is missing {key[0]} parent {key[1]}"
            )
        return mapped.kind, mapped.id

    def mapped_parent_id(kind: str, identifier: int) -> int:
        if kind in selected_geometry:
            return mapped_geometry_key((kind, identifier))[1]
        try:
            return structural_maps[kind][identifier]
        except KeyError as error:
            raise GeometryError(
                f"structural copy is missing {kind} parent {identifier}"
            ) from error

    def mapped_lineage_key(key: EntityKey) -> EntityKey:
        kind, identifier = key
        if kind in selected_geometry:
            mapped = entity_mapping.get(EntityRef(kind, identifier))  # type: ignore[arg-type]
            return key if mapped is None else (mapped.kind, mapped.id)
        mapped = structural_maps.get(kind, {}).get(identifier)
        return key if mapped is None else (kind, mapped)

    for old_id in sorted(coedge_ids):
        old = source.coedges[old_id]
        destination._put_structural(  # noqa: SLF001
            "coedge",
            replace(
                old,
                id=coedge_map[old_id],
                face_use_id=face_use_map[old.face_use_id],
                edge_id=entity_mapping[EntityRef("edge", old.edge_id)].id,
            ),
        )
    for old_id in sorted(sheet_use_ids):
        old = source.face_uses[old_id]
        destination._put_structural(  # noqa: SLF001
            "face_use",
            replace(
                old,
                id=face_use_map[old_id],
                sheet_id=sheet_map[old.sheet_id],
                face_id=entity_mapping[EntityRef("face", old.face_id)].id,
                loops=tuple(
                    tuple(coedge_map[item] for item in loop)
                    for loop in old.loops
                ),
            ),
        )
    for old_id in sorted(eligible_sheets):
        old = source.sheets[old_id]
        destination._put_structural(  # noqa: SLF001
            "sheet",
            replace(
                old,
                id=sheet_map[old_id],
                part_id=part_map[old.part_id],
                face_use_ids=tuple(face_use_map[item] for item in old.face_use_ids),
                declared_non_manifold_edges=tuple(
                    entity_mapping[EntityRef("edge", item)].id
                    for item in old.declared_non_manifold_edges
                ),
            ),
        )

    for old_id in sorted(member_use_ids):
        old = source.member_edge_uses[old_id]
        destination._put_structural(  # noqa: SLF001
            "member_edge_use",
            replace(
                old,
                id=member_use_map[old_id],
                member_id=member_map[old.member_id],
                edge_id=entity_mapping[EntityRef("edge", old.edge_id)].id,
            ),
        )
    for old_id in sorted(eligible_members):
        old = source.members[old_id]
        destination._put_structural(  # noqa: SLF001
            "member",
            replace(
                old,
                id=member_map[old_id],
                part_id=part_map[old.part_id],
                edge_use_ids=tuple(
                    member_use_map[item] for item in old.edge_use_ids
                ),
                orientation_reference=(
                    None
                    if old.orientation_reference is None
                    else mapped_geometry_key(old.orientation_reference)
                ),
            ),
        )

    for old_id in sorted(eligible_attachments):
        old = source.attachments[old_id]
        destination._put_structural(  # noqa: SLF001
            "attachment",
            replace(
                old,
                id=attachment_map[old_id],
                member_id=(
                    None
                    if old.member_id is None
                    else member_map[old.member_id]
                ),
                source_id=mapped_parent_id(old.source_kind, old.source_id),
                target_id=mapped_parent_id(
                    old.target_kind.value, old.target_id
                ),
                part_id=(
                    None if old.part_id is None else part_map[old.part_id]
                ),
                sheet_id=(
                    None if old.sheet_id is None else sheet_map[old.sheet_id]
                ),
                lineage=tuple(mapped_lineage_key(key) for key in old.lineage),
            ),
        )
    for old_id in sorted(eligible_junctions):
        old = source.junctions[old_id]
        destination._put_structural(  # noqa: SLF001
            "junction",
            replace(
                old,
                id=junction_map[old_id],
                member_uses=tuple(
                    JunctionMemberUse(member_map[item.member_id], item.member_range)
                    for item in old.member_uses
                ),
                sheet_ids=tuple(sheet_map[item] for item in old.sheet_ids),
                attachment_ids=tuple(
                    attachment_map[item] for item in old.attachment_ids
                ),
            ),
        )

    for old_id in sorted(eligible_parts):
        old = source.parts[old_id]
        destination._put_structural(  # noqa: SLF001
            "part",
            replace(
                old,
                id=part_map[old_id],
                sheet_ids=tuple(
                    sheet_map[item] for item in old.sheet_ids if item in sheet_map
                ),
                member_ids=tuple(
                    member_map[item] for item in old.member_ids if item in member_map
                ),
            ),
        )
    for old_vertex_id, old_part_id in sorted(selected_construction.items()):
        destination.mark_construction_vertices(
            (entity_mapping[EntityRef("vertex", old_vertex_id)].id,),
            part_id=(
                None if old_part_id is None else part_map[old_part_id]
            ),
        )
    # ``_put_structural`` maintains every reverse-incidence bucket above. The
    # surrounding owner transaction validates the complete changed structural
    # closure at commit, so no full destination-model rebuild or scan is
    # needed for each pattern instance.
    return {
        (kind, old_id): (kind, new_id)
        for kind, identifiers in structural_maps.items()
        for old_id, new_id in identifiers.items()
    }


def insert_model(
    destination: GeometryModel,
    source: GeometryModel,
    *,
    matrix: AffineLike | None = None,
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
    with destination.transaction():
        prepared = source.clone(include_features=False)
        affine = _affine_transform(matrix)
        if affine != AffineTransform.identity():
            transform(prepared, affine)
        result = _insert_selected(
            destination, prepared, None, group_prefix=group_prefix
        )
        errors = destination.validate_topology()
        if errors:
            raise GeometryError("insert produced invalid topology: " + "; ".join(errors))
        return InsertResult(
            result.entity_map,
            result.groups,
            result.outputs,
            {
                source.handle(prepared.kind, prepared.id): mapped
                for prepared, mapped in result.handle_map.items()
            },
        )


def _copy_closure(
    geometry: GeometryModel,
    references: Iterable[Selection],
) -> ModelClosure:
    selected = tuple(references)
    if not selected:
        raise GeometryError("copy needs at least one entity")
    return extract_model_closure(
        geometry,
        selected,
        include_structural_closure=True,
        include_features=False,
    )


def _source_result(result: InsertResult, closure: ModelClosure) -> InsertResult:
    """Translate one working-model result back to original model identities."""

    handles: dict[EntityHandle, EntityHandle] = {}
    entities: dict[EntityRef, EntityRef] = {}
    for prepared, destination in result.handle_map.items():
        working = EntityHandle(
            closure.working_model.model_id,
            prepared.kind,
            prepared.id,
        )
        source = closure.work_to_source.get(working)
        if source is None:
            raise GeometryError(
                f"copy result contains unmapped working {prepared.kind} {prepared.id}"
            )
        handles[source] = destination
        if source.kind in ("vertex", "edge", "face"):
            entities[EntityRef(source.kind, source.id)] = EntityRef(  # type: ignore[arg-type]
                destination.kind,
                destination.id,
            )
    outputs = {
        f"{old.kind}/{old.id}": new
        for old, new in sorted(
            entities.items(), key=lambda item: (item[0].kind, item[0].id)
        )
    }
    return InsertResult(entities, result.groups, outputs, handles)


def _insert_closure_instance(
    destination: GeometryModel,
    closure: ModelClosure,
    affine: AffineTransform,
    *,
    group_prefix: str | None,
) -> InsertResult:
    source = closure.working_model
    if affine == AffineTransform.identity():
        prepared = source
    else:
        # Clone only the selected dependency closure, never the complete live
        # destination. Pattern cost is therefore proportional to selected
        # topology and instance count rather than total model size.
        prepared = source.clone(include_features=False)
        transform(prepared, affine)
    return _source_result(
        _insert_selected(destination, prepared, None, group_prefix=group_prefix),
        closure,
    )


def copy_entities(
    geometry: GeometryModel,
    references: Iterable[Selection],
    *,
    matrix: AffineLike | None = None,
    group_prefix: str | None = None,
) -> InsertResult:
    """Copy a selected geometry/structural closure into the same model.

    The selection accepts model-bound handles, local geometry references, and
    compact keys. Copied entities receive fresh IDs and the returned
    ``handle_map`` binds every copied geometry and structural record to the
    destination model.
    """

    closure = _copy_closure(geometry, references)
    affine = _affine_transform(matrix)
    with geometry.transaction():
        return _insert_closure_instance(
            geometry,
            closure,
            affine,
            group_prefix=group_prefix,
        )


def move_entities(
    geometry: GeometryModel,
    references: Iterable[Selection],
    affine: AffineLike,
) -> tuple[EntityRef, ...]:
    """Transform a selected closure in place while preserving every identity."""

    return transform(geometry, coerce_affine_transform(affine), references)


def translate_entities(
    geometry: GeometryModel,
    references: Iterable[Selection],
    vector: Sequence[float],
) -> tuple[EntityRef, ...]:
    return move_entities(geometry, references, AffineTransform.translation(vector))


def rotate_entities(
    geometry: GeometryModel,
    references: Iterable[Selection],
    axis_point: Sequence[float],
    axis_direction: Sequence[float],
    angle: float,
) -> tuple[EntityRef, ...]:
    return move_entities(
        geometry,
        references,
        AffineTransform.rotation(axis_point, axis_direction, angle),
    )


def copy_translated(
    geometry: GeometryModel,
    references: Iterable[Selection],
    vector: Sequence[float],
    *,
    group_prefix: str | None = None,
) -> InsertResult:
    return copy_entities(
        geometry,
        references,
        matrix=AffineTransform.translation(vector),
        group_prefix=group_prefix,
    )


def copy_rotated(
    geometry: GeometryModel,
    references: Iterable[Selection],
    axis_point: Sequence[float],
    axis_direction: Sequence[float],
    angle: float,
    *,
    group_prefix: str | None = None,
) -> InsertResult:
    return copy_entities(
        geometry,
        references,
        matrix=AffineTransform.rotation(axis_point, axis_direction, angle),
        group_prefix=group_prefix,
    )


def pattern_entities(
    geometry: GeometryModel,
    references: Iterable[Selection],
    transforms: Iterable[AffineLike],
    *,
    group_prefix: str | None = None,
) -> PatternResult:
    """Create one fresh selected-closure copy for every supplied transform."""

    made_transforms = tuple(coerce_affine_transform(item) for item in transforms)
    if not made_transforms:
        raise GeometryError("a pattern needs at least one transform")
    closure = _copy_closure(geometry, references)
    instances: list[InsertResult] = []
    with geometry.transaction():
        for index, affine in enumerate(made_transforms, start=1):
            instances.append(
                _insert_closure_instance(
                    geometry,
                    closure,
                    affine,
                    group_prefix=(
                        None
                        if group_prefix is None
                        else f"{group_prefix}/{index}"
                    ),
                )
            )
    return PatternResult(tuple(instances), made_transforms)


def linear_pattern(
    geometry: GeometryModel,
    references: Iterable[Selection],
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
    unit = vector / length
    return pattern_entities(
        geometry,
        references,
        (
            AffineTransform.translation(unit * spacing * index)
            for index in range(1, int(count) + 1)
        ),
        group_prefix=group_prefix,
    )


def circular_pattern(
    geometry: GeometryModel,
    references: Iterable[Selection],
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
    return pattern_entities(
        geometry,
        references,
        (
            AffineTransform.rotation(
                point,
                direction,
                float(angle_step) * index,
            )
            for index in range(1, int(count) + 1)
        ),
        group_prefix=group_prefix,
    )


def rectangular_pattern(
    geometry: GeometryModel,
    references: Iterable[Selection],
    directions: Sequence[Sequence[float]],
    spacings: Sequence[float],
    counts: Sequence[int],
    *,
    group_prefix: str | None = None,
) -> PatternResult:
    """Create a deterministic 1D, 2D, or 3D Cartesian copy array.

    Each count is the number of copied steps beyond the original along that
    axis. For example, counts ``(1, 1)`` produce three copies and, together
    with the untouched original, form a 2x2 array.
    """

    axes = tuple(np.asarray(item, dtype=float) for item in directions)
    spacing_values = tuple(float(item) for item in spacings)
    count_values = tuple(counts)
    if not 1 <= len(axes) <= 3 or not (
        len(axes) == len(spacing_values) == len(count_values)
    ):
        raise GeometryError(
            "rectangular pattern needs matching one to three directions, spacings, and counts"
        )
    units: list[np.ndarray] = []
    for axis in axes:
        if axis.shape != (3,) or not np.all(np.isfinite(axis)):
            raise GeometryError("pattern directions must be finite 3-vectors")
        length = float(np.linalg.norm(axis))
        if length <= 0.0:
            raise GeometryError("pattern directions must be non-zero")
        units.append(axis / length)
    if np.linalg.matrix_rank(np.asarray(units), tol=1.0e-12) != len(units):
        raise GeometryError("rectangular pattern directions must be independent")
    if any(not np.isfinite(item) or item <= 0.0 for item in spacing_values):
        raise GeometryError("pattern spacings must be finite and positive")
    made_counts: list[int] = []
    for item in count_values:
        if isinstance(item, bool) or int(item) != item or int(item) < 1:
            raise GeometryError("pattern counts must be positive integers")
        made_counts.append(int(item))
    offsets = (
        sum(
            (
                units[axis] * spacing_values[axis] * index[axis]
                for axis in range(len(units))
            ),
            start=np.zeros(3),
        )
        for index in product(*(range(item + 1) for item in made_counts))
        if any(index)
    )
    return pattern_entities(
        geometry,
        references,
        (AffineTransform.translation(offset) for offset in offsets),
        group_prefix=group_prefix,
    )


def mirror_entities(
    geometry: GeometryModel,
    references: Iterable[Selection],
    plane_point: Sequence[float],
    plane_normal: Sequence[float],
    *,
    group_prefix: str | None = None,
) -> InsertResult:
    """Copy selected topology reflected about a plane."""

    return copy_entities(
        geometry,
        references,
        matrix=AffineTransform.reflection(plane_point, plane_normal),
        group_prefix=group_prefix,
    )


def reverse_edge(geometry: GeometryModel, edge_id: int) -> EntityRef:
    """Reverse edge parameterization while preserving dependent semantics."""

    edge = geometry._require_edge(  # noqa: SLF001
        validate_local_id(edge_id, name="edge ID")
    )
    with geometry.transaction():
        curve = edge.curve
        if isinstance(curve, Spline):
            curve = Spline(tuple(reversed(curve.control_vertices)))
        geometry._put_entity(  # noqa: SLF001
            "edge", replace(edge, start=edge.end, end=edge.start, curve=curve)
        )
        for use_id in sorted(geometry._edge_member_uses.get(edge.id, ())):  # noqa: SLF001
            use = geometry.member_edge_uses[use_id]
            geometry._put_structural(  # noqa: SLF001
                "member_edge_use",
                replace(
                    use,
                    orientation=(
                        Orientation.REVERSED
                        if use.orientation is Orientation.FORWARD
                        else Orientation.FORWARD
                    ),
                ),
            )
        for attachment in tuple(geometry.attachments.values()):
            if (
                attachment.target_kind is AttachmentTargetKind.EDGE
                and attachment.target_id == edge.id
            ):
                target = attachment.target_parameters[0]
                if not target.is_point:
                    raise GeometryError(
                        "cannot reverse an edge with an attachment whose target "
                        "parameter spans a positive interval"
                    )
                geometry._put_structural(  # noqa: SLF001
                    "attachment",
                    replace(
                        attachment,
                        target_parameters=(
                            ParameterRange.point(1.0 - target.start),
                        ),
                    ),
                )
        for face_id in geometry.faces_using_edge(edge.id):
            face = geometry.faces[face_id]
            geometry._put_entity(  # noqa: SLF001
                "face",
                replace(
                    face,
                    loop=tuple(
                OrientedEdge(item.edge, not item.forward)
                if item.edge == edge.id
                else item
                for item in face.loop
                    ),
                    holes=tuple(
                        tuple(
                            OrientedEdge(item.edge, not item.forward)
                            if item.edge == edge.id
                            else item
                            for item in loop
                        )
                        for loop in face.holes
                    ),
                ),
            )
        errors = geometry.validate_topology()
        if errors:
            raise GeometryError("edge reversal produced invalid topology: " + "; ".join(errors))
        return geometry.edges[edge.id].ref


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


def _reverse_face_attachment_parameters(
    surface: object,
    parameters: tuple[ParameterRange, ...],
) -> tuple[ParameterRange, ...]:
    """Carry a face attachment through the surface reparameterization.

    ``ParameterRange`` is intentionally unoriented, so a positive-length path
    cannot represent a complemented (decreasing) coordinate.  Such curved
    cases fail closed instead of silently moving the declared attachment.
    """

    if len(parameters) != 2:
        raise GeometryError("face attachment requires two target parameter ranges")
    first, second = parameters
    if isinstance(surface, Plane) or (
        isinstance(surface, CoonsSurface) and not surface.has_boundaries
    ):
        # Reversed planar/topology-backed four-side faces transpose u and v.
        return second, first
    if isinstance(surface, (Cylinder, Cone, RuledSurface, CoonsSurface)):
        if not first.is_point:
            raise GeometryError(
                "cannot reverse a face with an attachment whose complemented "
                "surface parameter spans a positive interval"
            )
        return ParameterRange.point(1.0 - first.start), second
    raise GeometryError(
        "cannot reverse a face attachment without a supported surface mapping"
    )


def reverse_face(geometry: GeometryModel, face_id: int) -> EntityRef:
    """Reverse a face orientation, including its authoritative surface normal."""

    face = geometry._require_face(  # noqa: SLF001
        validate_local_id(face_id, name="face ID")
    )
    with geometry.transaction():
        corner_vertices = set(geometry.face_corner_vertices(face.id))
        loop = _reverse_loop(face.loop)
        corners = face.corners
        if corner_vertices:
            corners = tuple(
                index
                for index, item in enumerate(loop)
                if geometry.oriented_start_vertex(item) in corner_vertices
            )
        made = replace(
            face,
            loop=loop,
            holes=tuple(_reverse_loop(item) for item in face.holes),
            corners=corners,
            surface=_reverse_surface(face.surface),
            parameterization=_reverse_surface(face.parameterization),
        )
        geometry._put_entity("face", made)  # noqa: SLF001
        for attachment in tuple(geometry.attachments.values()):
            if (
                attachment.target_kind is AttachmentTargetKind.FACE
                and attachment.target_id == face.id
            ):
                geometry._put_structural(  # noqa: SLF001
                    "attachment",
                    replace(
                        attachment,
                        target_parameters=_reverse_face_attachment_parameters(
                            (
                                face.parameterization
                                if face.parameterization is not None
                                else face.surface
                            ),
                            attachment.target_parameters,
                        ),
                    ),
                )
        errors = geometry.validate_topology()
        if errors:
            raise GeometryError("face reversal produced invalid topology: " + "; ".join(errors))
        return made.ref


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
