"""Identity-safe extraction of a selected model dependency closure."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Iterable, Mapping
from uuid import UUID

from .curves import Arc, Spline
from .entities import EntityRef, Face
from .errors import GeometryError
from .identity import EntityHandle, EntityKey, ResolutionStatus
from .model import GeometryModel
from .structural import (
    Attachment,
    Coedge,
    FaceUse,
    Junction,
    JunctionMemberUse,
    Member,
    MemberEdgeUse,
    Part,
    Sheet,
)

__all__ = ["ModelClosure", "extract_model_closure"]


@dataclass(frozen=True, slots=True)
class ModelClosure:
    """A detached working model plus model-bound bidirectional identity maps."""

    working_model: GeometryModel
    source_to_work: Mapping[EntityHandle, EntityHandle]
    work_to_source: Mapping[EntityHandle, EntityHandle]
    source_model_id: UUID
    source_revision: int
    source_handles: tuple[EntityHandle, ...]

    @property
    def source_to_work_handles(self) -> Mapping[EntityHandle, EntityHandle]:
        return self.source_to_work

    @property
    def work_to_source_handles(self) -> Mapping[EntityHandle, EntityHandle]:
        return self.work_to_source


def _normalize_handles(
    geometry: GeometryModel, handles: Iterable[EntityHandle | EntityRef | EntityKey]
) -> tuple[EntityHandle, ...]:
    made: set[EntityHandle] = set()
    for value in handles:
        if isinstance(value, EntityHandle):
            if value.model_id != geometry.model_id:
                raise GeometryError("closure selection contains a wrong-model handle")
            handle = value
        elif isinstance(value, EntityRef):
            handle = geometry.handle(value.kind, value.id)
        else:
            try:
                kind, identifier = value
            except (TypeError, ValueError) as error:
                raise GeometryError("closure handles must be entity references") from error
            handle = geometry.handle(str(kind), identifier)
        if geometry.resolve_handle(handle).status is not ResolutionStatus.ACTIVE:
            raise GeometryError(f"closure selection is not active: {handle}")
        made.add(handle)
    return tuple(sorted(made))


def _geometry_closure(
    geometry: GeometryModel,
    selected: set[EntityKey],
) -> tuple[set[int], set[int], set[int]]:
    vertices = {identifier for kind, identifier in selected if kind == "vertex"}
    edges = {identifier for kind, identifier in selected if kind == "edge"}
    faces = {identifier for kind, identifier in selected if kind == "face"}
    for face_id in tuple(faces):
        face = geometry.faces[face_id]
        edges.update(
            item.edge for loop in (face.loop,) + face.holes for item in loop
        )
    for edge_id in tuple(edges):
        edge = geometry.edges[edge_id]
        vertices.update((edge.start, edge.end))
        if isinstance(edge.curve, Arc):
            vertices.add(edge.curve.via_vertex)
        elif isinstance(edge.curve, Spline):
            vertices.update(edge.curve.control_vertices)
    return vertices, edges, faces


def extract_model_closure(
    geometry: GeometryModel,
    handles: Iterable[EntityHandle | EntityRef | EntityKey],
    *,
    include_structural_closure: bool = True,
    include_features: bool = False,
) -> ModelClosure:
    """Extract selected definitions and an identity-safe dependency closure.

    The returned model has its own UUID.  Source/work correspondence therefore
    cannot accidentally resolve across documents.  Explicitly selected and
    lineage-selected Parts and Sheets are complete; owner/context parents list
    only copied children.  Attachments and junctions are included only when
    every referenced parent is present.
    """

    if include_features:
        raise GeometryError(
            "feature-history closure extraction is unsupported; select include_features=False"
        )
    source_handles = _normalize_handles(geometry, handles)
    if not include_structural_closure and any(
        handle.kind not in ("vertex", "edge", "face")
        for handle in source_handles
    ):
        raise GeometryError(
            "structural selections require include_structural_closure=True"
        )
    selected: set[EntityKey] = {handle.key for handle in source_handles}

    part_ids = {identifier for kind, identifier in selected if kind == "part"}
    sheet_ids = {identifier for kind, identifier in selected if kind == "sheet"}
    complete_part_ids = set(part_ids)
    complete_sheet_ids = set(sheet_ids)
    member_ids = {identifier for kind, identifier in selected if kind == "member"}
    face_use_ids = {identifier for kind, identifier in selected if kind == "face_use"}
    coedge_ids = {identifier for kind, identifier in selected if kind == "coedge"}
    member_use_ids = {
        identifier for kind, identifier in selected if kind == "member_edge_use"
    }
    requested_attachment_ids = {
        identifier for kind, identifier in selected if kind == "attachment"
    }
    requested_junction_ids = {
        identifier for kind, identifier in selected if kind == "junction"
    }

    structural_selections = {
        "part": part_ids,
        "sheet": sheet_ids,
        "face_use": face_use_ids,
        "coedge": coedge_ids,
        "member": member_ids,
        "member_edge_use": member_use_ids,
        "attachment": requested_attachment_ids,
        "junction": requested_junction_ids,
    }

    def include_active_lineage_parent(parent: EntityKey) -> None:
        """Route one active provenance key into its dependency closure."""

        kind, identifier = parent
        if not geometry._contains_entity(kind, identifier):  # noqa: SLF001
            # Lineage may intentionally retain a superseded source-local key.
            # Such a record cannot be copied into a detached active model.
            return
        if kind in ("vertex", "edge", "face"):
            selected.add(parent)
            return
        try:
            structural_selections[kind].add(identifier)
        except KeyError as error:
            # _contains_entity returning true above establishes that this is a
            # supported active kind.  Keep this fail-closed if the closure's
            # structural routing ever falls behind the model's public kinds.
            raise GeometryError(
                f"closure cannot route active lineage parent {kind} {identifier}"
            ) from error
        if kind == "part":
            complete_part_ids.add(identifier)
        elif kind == "sheet":
            complete_sheet_ids.add(identifier)

    vertices, edges, faces = _geometry_closure(geometry, selected)
    if include_structural_closure:
        previous: tuple[int, ...] | None = None
        while True:
            state = tuple(
                len(values)
                for values in (
                    vertices, edges, faces, part_ids, sheet_ids, member_ids,
                    face_use_ids, coedge_ids, member_use_ids,
                    requested_attachment_ids, requested_junction_ids,
                    complete_part_ids, complete_sheet_ids,
                )
            )
            if state == previous:
                # A Sheet record is intentionally non-empty.  A context-only
                # sheet remains partial when selected dependencies identify
                # specific uses.  If none do, the only truthful non-empty
                # representation is its complete face-use closure.
                empty_partial_sheets = tuple(
                    sorted(
                        sheet_ids
                        - complete_sheet_ids
                        - {
                            geometry.face_uses[face_use_id].sheet_id
                            for face_use_id in face_use_ids
                        }
                    )
                )
                if empty_partial_sheets:
                    for sheet_id in empty_partial_sheets:
                        sheet = geometry.sheets[sheet_id]
                        if not sheet.face_use_ids:
                            raise GeometryError(
                                f"closure cannot materialize empty sheet {sheet_id}"
                            )
                        complete_sheet_ids.add(sheet_id)
                    continue
                break
            previous = state

            # Geometry definitions pull their explicit semantic uses.
            for vertex_id in tuple(vertices):
                owner = geometry.construction_vertices.get(vertex_id)
                if owner is not None:
                    part_ids.add(owner)
                requested_attachment_ids.update(
                    geometry._source_attachments.get(("vertex", vertex_id), ())  # noqa: SLF001
                )
                requested_attachment_ids.update(
                    geometry._target_attachments.get(("vertex", vertex_id), ())  # noqa: SLF001
                )
            for edge_id in tuple(edges):
                coedge_ids.update(geometry.coedges_using_edge(edge_id))
                member_ids.update(geometry.members_using_edge(edge_id))
                requested_attachment_ids.update(
                    geometry._source_attachments.get(("edge", edge_id), ())  # noqa: SLF001
                )
                requested_attachment_ids.update(
                    geometry._target_attachments.get(("edge", edge_id), ())  # noqa: SLF001
                )
            for face_id in tuple(faces):
                face_use_ids.update(geometry._face_structural_uses.get(face_id, ()))  # noqa: SLF001
                requested_attachment_ids.update(
                    geometry._source_attachments.get(("face", face_id), ())  # noqa: SLF001
                )
                requested_attachment_ids.update(geometry.attachments_for_face(face_id))

            for part_id in tuple(complete_part_ids):
                part = geometry.parts[part_id]
                sheet_ids.update(part.sheet_ids)
                complete_sheet_ids.update(part.sheet_ids)
                member_ids.update(part.member_ids)
            for sheet_id in tuple(sheet_ids):
                sheet = geometry.sheets[sheet_id]
                part_ids.add(sheet.part_id)
                if sheet_id in complete_sheet_ids:
                    face_use_ids.update(sheet.face_use_ids)
                    requested_attachment_ids.update(
                        geometry.attachments_for_sheet(sheet_id)
                    )
            for face_use_id in tuple(face_use_ids):
                use = geometry.face_uses[face_use_id]
                sheet_ids.add(use.sheet_id)
                coedge_ids.update(use.coedge_ids)
                faces.add(use.face_id)
            for coedge_id in tuple(coedge_ids):
                use = geometry.coedges[coedge_id]
                face_use_ids.add(use.face_use_id)
                edges.add(use.edge_id)
            for member_id in tuple(member_ids):
                member = geometry.members[member_id]
                part_ids.add(member.part_id)
                member_use_ids.update(member.edge_use_ids)
                requested_attachment_ids.update(
                    geometry.attachments_for_member(member_id)
                )
                requested_junction_ids.update(
                    geometry._member_junctions.get(member_id, ())  # noqa: SLF001
                )
                if member.orientation_reference is not None:
                    selected.add(member.orientation_reference)
            for use_id in tuple(member_use_ids):
                use = geometry.member_edge_uses[use_id]
                member_ids.add(use.member_id)
                edges.add(use.edge_id)
            for attachment_id in tuple(requested_attachment_ids):
                attachment = geometry.attachments[attachment_id]
                for parent in (attachment.source_key, attachment.target_key):
                    selected.add(parent)
                    if parent[0] == "member":
                        member_ids.add(parent[1])
                    elif parent[0] == "sheet":
                        sheet_ids.add(parent[1])
                if attachment.member_id is not None:
                    member_ids.add(attachment.member_id)
                if attachment.part_id is not None:
                    part_ids.add(attachment.part_id)
                if attachment.sheet_id is not None:
                    sheet_ids.add(attachment.sheet_id)
                for parent in attachment.lineage:
                    include_active_lineage_parent(parent)
            for junction_id in tuple(requested_junction_ids):
                junction = geometry.junctions[junction_id]
                member_ids.update(junction.member_ids)
                sheet_ids.update(junction.sheet_ids)
                requested_attachment_ids.update(junction.attachment_ids)

            extra_vertices, extra_edges, extra_faces = _geometry_closure(
                geometry, selected | {
                    *(('vertex', value) for value in vertices),
                    *(('edge', value) for value in edges),
                    *(('face', value) for value in faces),
                }
            )
            vertices.update(extra_vertices)
            edges.update(extra_edges)
            faces.update(extra_faces)

    def attachment_complete(attachment: Attachment) -> bool:
        present = {
            "vertex": vertices,
            "edge": edges,
            "face": faces,
            "member": member_ids,
            "sheet": sheet_ids,
        }
        if attachment.source_id not in present[attachment.source_kind]:  # type: ignore[index]
            return False
        if attachment.target_id not in present[attachment.target_kind.value]:
            return False
        if attachment.part_id is not None and attachment.part_id not in part_ids:
            return False
        if attachment.sheet_id is not None and attachment.sheet_id not in sheet_ids:
            return False
        return True

    attachment_ids = {
        attachment_id
        for attachment_id in requested_attachment_ids
        if attachment_complete(geometry.attachments[attachment_id])
    }
    junction_ids = {
        junction_id
        for junction_id in requested_junction_ids
        for junction in (geometry.junctions[junction_id],)
        if set(junction.member_ids).issubset(member_ids)
        and set(junction.sheet_ids).issubset(sheet_ids)
        and set(junction.attachment_ids).issubset(attachment_ids)
    }

    # Build child lists from the local closure, rather than filtering each
    # source parent's potentially very large tuple of unrelated siblings.
    face_uses_by_sheet: dict[int, list[int]] = {}
    for face_use_id in sorted(face_use_ids):
        face_uses_by_sheet.setdefault(
            geometry.face_uses[face_use_id].sheet_id, []
        ).append(face_use_id)
    sheets_by_part: dict[int, list[int]] = {}
    for sheet_id in sorted(sheet_ids):
        sheets_by_part.setdefault(geometry.sheets[sheet_id].part_id, []).append(
            sheet_id
        )
    members_by_part: dict[int, list[int]] = {}
    for member_id in sorted(member_ids):
        members_by_part.setdefault(geometry.members[member_id].part_id, []).append(
            member_id
        )

    work = GeometryModel(tolerance=geometry.tolerance)
    work._units = geometry.units  # noqa: SLF001
    work._local_origin = geometry.local_origin.copy()  # noqa: SLF001
    work._local_origin.flags.writeable = False  # noqa: SLF001
    work._coordinate_transform = (  # noqa: SLF001
        None
        if geometry.coordinate_transform is None
        else geometry.coordinate_transform.copy()
    )
    if work._coordinate_transform is not None:  # noqa: SLF001
        work._coordinate_transform.flags.writeable = False  # noqa: SLF001
    work._crs_metadata = geometry.crs_metadata  # noqa: SLF001

    source_to_work: dict[EntityHandle, EntityHandle] = {}
    work_to_source: dict[EntityHandle, EntityHandle] = {}
    ids: dict[str, dict[int, int]] = {
        kind: {}
        for kind in (
            "vertex", "edge", "face", "part", "sheet", "face_use", "coedge",
            "member", "member_edge_use", "attachment", "junction",
        )
    }

    def mapped_active_lineage(item: Attachment) -> tuple[EntityKey, ...]:
        """Map every active lineage key; never silently discard one."""

        mapped: list[EntityKey] = []
        for kind, identifier in item.lineage:
            if not geometry._contains_entity(kind, identifier):  # noqa: SLF001
                continue
            try:
                mapped.append((kind, ids[kind][identifier]))
            except KeyError as error:
                raise GeometryError(
                    "closure is missing active attachment lineage parent "
                    f"{kind} {identifier}"
                ) from error
        return tuple(mapped)

    def remember(kind: str, source_id: int, work_id: int) -> None:
        ids[kind][source_id] = work_id
        source = geometry.handle(kind, source_id)
        made = work.handle(kind, work_id)
        source_to_work[source] = made
        work_to_source[made] = source

    with work.transaction():
        for source_id in sorted(vertices):
            position = geometry.vertices[source_id].position
            remember("vertex", source_id, work.add_point(*position))
        for source_id in sorted(edges):
            edge = geometry.edges[source_id]
            start, end = ids["vertex"][edge.start], ids["vertex"][edge.end]
            if isinstance(edge.curve, Arc):
                work_id = work.add_arc(start, ids["vertex"][edge.curve.via_vertex], end)
            elif isinstance(edge.curve, Spline):
                work_id = work.add_spline(
                    start,
                    tuple(ids["vertex"][value] for value in edge.curve.control_vertices),
                    end,
                )
            else:
                work_id = work.add_line(start, end)
            remember("edge", source_id, work_id)
        for source_id in sorted(faces):
            face = geometry.faces[source_id]

            def loop(values):
                from .entities import OrientedEdge

                return tuple(OrientedEdge(ids["edge"][value.edge], value.forward) for value in values)

            work_id = work._allocate("face")  # noqa: SLF001
            work._put_entity(  # noqa: SLF001
                "face",
                Face(
                    work_id,
                    loop(face.loop),
                    face.corners,
                    face.metadata,
                    tuple(loop(value) for value in face.holes),
                    face.surface,
                    face.parameterization,
                ),
            )
            remember("face", source_id, work_id)

        # Allocate structural keys up front so immutable cross references can
        # be materialized without temporarily invalid parent IDs.
        stores = (
            ("part", part_ids), ("sheet", sheet_ids), ("face_use", face_use_ids),
            ("coedge", coedge_ids), ("member", member_ids),
            ("member_edge_use", member_use_ids), ("attachment", attachment_ids),
            ("junction", junction_ids),
        )
        for kind, source_ids in stores:
            for source_id in sorted(source_ids):
                ids[kind][source_id] = work._allocate_structural(kind)  # noqa: SLF001

        for source_id in sorted(coedge_ids):
            item = geometry.coedges[source_id]
            work._put_structural(  # noqa: SLF001
                "coedge",
                replace(
                    item,
                    id=ids["coedge"][source_id],
                    face_use_id=ids["face_use"][item.face_use_id],
                    edge_id=ids["edge"][item.edge_id],
                ),
            )
        for source_id in sorted(face_use_ids):
            item = geometry.face_uses[source_id]
            work._put_structural(  # noqa: SLF001
                "face_use",
                replace(
                    item,
                    id=ids["face_use"][source_id],
                    sheet_id=ids["sheet"][item.sheet_id],
                    face_id=ids["face"][item.face_id],
                    loops=tuple(
                        tuple(ids["coedge"][value] for value in loop)
                        for loop in item.loops
                    ),
                ),
            )
        for source_id in sorted(member_use_ids):
            item = geometry.member_edge_uses[source_id]
            work._put_structural(  # noqa: SLF001
                "member_edge_use",
                replace(
                    item,
                    id=ids["member_edge_use"][source_id],
                    member_id=ids["member"][item.member_id],
                    edge_id=ids["edge"][item.edge_id],
                ),
            )
        for source_id in sorted(member_ids):
            item = geometry.members[source_id]
            orientation = item.orientation_reference
            if orientation is not None and orientation[1] in ids[orientation[0]]:
                orientation = (orientation[0], ids[orientation[0]][orientation[1]])
            work._put_structural(  # noqa: SLF001
                "member",
                replace(
                    item,
                    id=ids["member"][source_id],
                    part_id=ids["part"][item.part_id],
                    edge_use_ids=tuple(ids["member_edge_use"][value] for value in item.edge_use_ids),
                    orientation_reference=orientation,
                ),
            )
        for source_id in sorted(sheet_ids):
            item = geometry.sheets[source_id]
            work._put_structural(  # noqa: SLF001
                "sheet",
                replace(
                    item,
                    id=ids["sheet"][source_id],
                    part_id=ids["part"][item.part_id],
                    face_use_ids=tuple(
                        ids["face_use"][value]
                        for value in face_uses_by_sheet.get(source_id, ())
                    ),
                    declared_non_manifold_edges=tuple(
                        ids["edge"][value] for value in item.declared_non_manifold_edges
                        if value in ids["edge"]
                    ),
                ),
            )
        for source_id in sorted(part_ids):
            item = geometry.parts[source_id]
            work._put_structural(  # noqa: SLF001
                "part",
                replace(
                    item,
                    id=ids["part"][source_id],
                    sheet_ids=tuple(
                        ids["sheet"][value]
                        for value in sheets_by_part.get(source_id, ())
                    ),
                    member_ids=tuple(
                        ids["member"][value]
                        for value in members_by_part.get(source_id, ())
                    ),
                ),
            )
        for source_id in sorted(attachment_ids):
            item = geometry.attachments[source_id]
            source_kind, source_value = item.source_key
            target_kind, target_value = item.target_key
            work._put_structural(  # noqa: SLF001
                "attachment",
                replace(
                    item,
                    id=ids["attachment"][source_id],
                    member_id=(None if item.member_id is None else ids["member"][item.member_id]),
                    source_id=ids[source_kind][source_value],
                    target_id=ids[target_kind][target_value],
                    part_id=(None if item.part_id is None else ids["part"][item.part_id]),
                    sheet_id=(None if item.sheet_id is None else ids["sheet"][item.sheet_id]),
                    lineage=mapped_active_lineage(item),
                ),
            )
        for source_id in sorted(junction_ids):
            item = geometry.junctions[source_id]
            work._put_structural(  # noqa: SLF001
                "junction",
                replace(
                    item,
                    id=ids["junction"][source_id],
                    member_uses=tuple(
                        JunctionMemberUse(ids["member"][value.member_id], value.member_range)
                        for value in item.member_uses
                    ),
                    sheet_ids=tuple(ids["sheet"][value] for value in item.sheet_ids),
                    attachment_ids=tuple(ids["attachment"][value] for value in item.attachment_ids),
                ),
            )
        for kind, source_ids in stores:
            for source_id in sorted(source_ids):
                remember(kind, source_id, ids[kind][source_id])

        for source_id in sorted(vertices):
            if source_id in geometry.construction_vertices:
                part_id = geometry.construction_vertices[source_id]
                work.mark_construction_vertices(
                    (ids["vertex"][source_id],),
                    part_id=(None if part_id is None else ids["part"].get(part_id)),
                )

        # Copy only mapped neutral-geometry semantics; source-local IDs can
        # never leak into the detached model.
        for name in sorted(geometry.groups):
            mapped = []
            for reference in geometry.group(name, resolve=False):
                source = EntityHandle(geometry.model_id, reference.kind, reference.id)
                target = source_to_work.get(source)
                if target is not None and target.kind in ("vertex", "edge", "face"):
                    mapped.append(EntityRef(target.kind, target.id))  # type: ignore[arg-type]
            if mapped:
                work.add_to_group(name, mapped)
        for reference, values in sorted(
            geometry.tags.items(), key=lambda item: (item[0].kind, item[0].id)
        ):
            source = EntityHandle(geometry.model_id, reference.kind, reference.id)
            target = source_to_work.get(source)
            if target is not None and target.kind in ("vertex", "edge", "face"):
                work.tag(EntityRef(target.kind, target.id), *sorted(values))  # type: ignore[arg-type]

    return ModelClosure(
        work,
        MappingProxyType(source_to_work),
        MappingProxyType(work_to_source),
        geometry.model_id,
        geometry.revision,
        source_handles,
    )
