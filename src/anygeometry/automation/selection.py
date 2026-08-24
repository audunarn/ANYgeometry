"""Deterministic bounded discovery and selection for automation clients."""

from __future__ import annotations

from base64 import urlsafe_b64decode, urlsafe_b64encode
import json
import math
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..curves import Arc
from ..editing import measure
from ..entities import EntityRef
from ..errors import GeometryError
from ..identity import ENTITY_KINDS, EntityHandle, ResolutionStatus, validate_entity_kind
from ..model import GeometryModel
from ..operations import closest_point
from .types import (
    PROTOCOL_VERSION,
    AutomationError,
    EntitySummary,
    Quantity,
    RequestHeader,
    SelectionResult,
    SelectionSpec,
    canonical_digest,
    handle_from_dict,
)

_KIND_ALIASES = {"point": "vertex", "plate": "sheet"}
_LENGTH_FACTORS = {"m": 1.0, "mm": 1.0e-3, "cm": 1.0e-2, "in": 0.0254, "ft": 0.3048}
_AREA_FACTORS = {f"{key}^2": value * value for key, value in _LENGTH_FACTORS.items()}
_AREA_FACTORS.update({f"{key}2": value * value for key, value in _LENGTH_FACTORS.items()})
_ANGLE_FACTORS = {"rad": 1.0, "deg": math.pi / 180.0}


def _error(code: str, message: str, *, path: str = "$", details: Mapping[str, object] | None = None) -> AutomationError:
    return AutomationError(code, message, path=path, details={} if details is None else details)


def _header(model: GeometryModel, header: RequestHeader | Mapping[str, object] | None, *, request_id: str | None = None, model_id: object = None, expected_revision: int | None = None) -> RequestHeader:
    if isinstance(header, RequestHeader):
        made = header
    elif isinstance(header, Mapping):
        allowed = {"protocol_version", "request_id", "model_id", "expected_revision"}
        extra = set(header) - allowed
        if extra or set(header) != allowed:
            raise _error("MALFORMED_REQUEST", "header requires exactly protocol_version, request_id, model_id, expected_revision")
        made = RequestHeader(header["protocol_version"], header["request_id"], header["model_id"], header["expected_revision"])  # type: ignore[arg-type]
    else:
        made = RequestHeader(PROTOCOL_VERSION, request_id or "request", model.model_id if model_id is None else model_id, model.revision if expected_revision is None else expected_revision)
    if made.model_id != model.model_id:
        raise _error("WRONG_MODEL", "request model UUID does not match the active model")
    if made.expected_revision != model.revision:
        raise _error("STALE_REVISION", "request revision is stale", details={"expected": made.expected_revision, "actual": model.revision})
    return made


def _kind(value: object) -> str:
    if isinstance(value, str):
        value = _KIND_ALIASES.get(value, value)
    try:
        return validate_entity_kind(value)
    except GeometryError as error:
        raise _error("UNSUPPORTED", str(error), path="$.where.kind") from error


def _stores(model: GeometryModel) -> Mapping[str, Mapping[int, object]]:
    return {
        "vertex": model.vertices, "edge": model.edges, "face": model.faces,
        "part": model.parts, "sheet": model.sheets, "face_use": model.face_uses,
        "coedge": model.coedges, "member": model.members,
        "member_edge_use": model.member_edge_uses, "attachment": model.attachments,
        "junction": model.junctions,
    }


def _all_handles(model: GeometryModel, kinds: Iterable[str] = ENTITY_KINDS) -> tuple[EntityHandle, ...]:
    stores = _stores(model)
    return tuple(model.handle(kind, identifier) for kind in kinds for identifier in sorted(stores[kind]))


def _to_model_scalar(model: GeometryModel, quantity: Quantity, dimension: str) -> float:
    if isinstance(quantity.value, tuple):
        raise _error("MALFORMED_QUANTITY", "a scalar quantity is required")
    factors = _LENGTH_FACTORS if dimension == "length" else _AREA_FACTORS if dimension == "area" else _ANGLE_FACTORS
    if quantity.unit not in factors:
        raise _error("UNKNOWN_UNIT", f"unknown {dimension} unit {quantity.unit!r}")
    model_unit = model.units
    model_factors = _LENGTH_FACTORS if dimension == "length" else _AREA_FACTORS if dimension == "area" else _ANGLE_FACTORS
    if dimension == "area":
        model_unit = f"{model_unit}^2"
    if dimension == "angle":
        model_unit = "rad"
    if model_unit not in model_factors:
        raise _error("UNSUPPORTED_MODEL_UNITS", f"unsupported model units {model.units!r}")
    return quantity.value * factors[quantity.unit] / model_factors[model_unit]


def _to_local_point(model: GeometryModel, quantity: Quantity) -> np.ndarray:
    if not isinstance(quantity.value, tuple) or len(quantity.value) != 3:
        raise _error("MALFORMED_QUANTITY", "a position requires three values")
    scale = _to_model_scalar(model, Quantity(1.0, quantity.unit), "length")
    point = np.asarray(quantity.value, dtype=float) * scale
    if quantity.frame is None:
        raise _error("UNKNOWN_FRAME", "positions require model_local or world frame")
    if quantity.frame == "world" and model.coordinate_transform is not None:
        inverse = np.linalg.inv(np.asarray(model.coordinate_transform, dtype=float))
        homogeneous = inverse @ np.asarray((*point, 1.0))
        if abs(float(homogeneous[3])) <= np.finfo(float).eps:
            raise _error("CAPABILITY_MISSING", "world transform produced an invalid homogeneous point")
        point = homogeneous[:3] / homogeneous[3]
    return point


def _to_local_aabb(model: GeometryModel, minimum: object, maximum: object) -> tuple[np.ndarray, np.ndarray]:
    low_quantity = _quantity(minimum, path="$.where.aabb.min")
    high_quantity = _quantity(maximum, path="$.where.aabb.max")
    if low_quantity.unit != high_quantity.unit or low_quantity.frame != high_quantity.frame:
        raise _error("MALFORMED_QUANTITY", "AABB endpoints require the same unit and frame")
    if not isinstance(low_quantity.value, tuple) or not isinstance(high_quantity.value, tuple) or len(low_quantity.value) != 3 or len(high_quantity.value) != 3:
        raise _error("MALFORMED_QUANTITY", "AABB endpoints require three values")
    if any(low_quantity.value[i] > high_quantity.value[i] for i in range(3)):
        raise _error("MALFORMED_SELECTOR", "aabb min exceeds max")
    corners = np.asarray(
        [
            _to_local_point(
                model,
                Quantity(
                    (x, y, z),
                    low_quantity.unit,
                    low_quantity.frame,
                ),
            )
            for x in (low_quantity.value[0], high_quantity.value[0])
            for y in (low_quantity.value[1], high_quantity.value[1])
            for z in (low_quantity.value[2], high_quantity.value[2])
        ]
    )
    return corners.min(axis=0), corners.max(axis=0)


def _quantity(data: object, *, path: str) -> Quantity:
    if isinstance(data, Quantity):
        return data
    try:
        return Quantity.from_dict(data, path=path)
    except AutomationError:
        raise
    except Exception as error:
        raise _error("MALFORMED_QUANTITY", str(error), path=path) from error


def _entity_ref(handle: EntityHandle) -> EntityRef:
    if handle.kind not in ("vertex", "edge", "face"):
        raise _error("UNSUPPORTED", f"measurement is unavailable for {handle.kind}")
    return EntityRef(handle.kind, handle.id)  # type: ignore[arg-type]


def _bounds(model: GeometryModel, handle: EntityHandle) -> tuple[float, ...] | None:
    if handle.kind in ("vertex", "edge", "face"):
        return model.entity_bounds_many((handle.key,))[0]
    keys: list[tuple[str, int]] = []
    if handle.kind == "coedge":
        keys.append(("edge", model.coedges[handle.id].edge_id))
    elif handle.kind == "face_use":
        keys.append(("face", model.face_uses[handle.id].face_id))
    elif handle.kind == "member_edge_use":
        keys.append(("edge", model.member_edge_uses[handle.id].edge_id))
    elif handle.kind == "member":
        keys.extend(("edge", model.member_edge_uses[item].edge_id) for item in model.members[handle.id].edge_use_ids)
    elif handle.kind == "sheet":
        keys.extend(("face", model.face_uses[item].face_id) for item in model.sheets[handle.id].face_use_ids)
    elif handle.kind == "part":
        part = model.parts[handle.id]
        for sheet_id in part.sheet_ids:
            keys.extend(("face", model.face_uses[item].face_id) for item in model.sheets[sheet_id].face_use_ids)
        for member_id in part.member_ids:
            keys.extend(("edge", model.member_edge_uses[item].edge_id) for item in model.members[member_id].edge_use_ids)
    elif handle.kind == "attachment":
        item = model.attachments[handle.id]
        for kind, identifier in (item.source_key, item.target_key):
            if kind in ("vertex", "edge", "face"):
                keys.append((kind, identifier))
    elif handle.kind == "junction":
        item = model.junctions[handle.id]
        for use in item.member_uses:
            member = model.members[use.member_id]
            keys.extend(("edge", model.member_edge_uses[value].edge_id) for value in member.edge_use_ids)
    return model.bounds(keys) if keys else None


def _groups(model: GeometryModel, handle: EntityHandle) -> tuple[str, ...]:
    if handle.kind not in ("vertex", "edge", "face"):
        return ()
    ref = EntityRef(handle.kind, handle.id)  # type: ignore[arg-type]
    return tuple(sorted(name for name, members in model.groups.items() if ref in members))


def _tags(model: GeometryModel, handle: EntityHandle) -> tuple[str, ...]:
    if handle.kind not in ("vertex", "edge", "face"):
        return ()
    return model.tags_for(EntityRef(handle.kind, handle.id))  # type: ignore[arg-type]


def _adjacency(model: GeometryModel, handle: EntityHandle) -> tuple[EntityHandle, ...]:
    keys: set[tuple[str, int]] = set()
    kind, identifier = handle.key
    if kind == "vertex":
        keys.update(("edge", item) for item in model.edges_using_vertex(identifier))
        keys.update(("member", item) for item in model.members_meeting_at_vertex(identifier))
        keys.update(("sheet", item) for item in model.sheets_containing_vertex(identifier))
    elif kind == "edge":
        edge = model.edges[identifier]
        keys.update(("vertex", item) for item in (edge.start, edge.end))
        keys.update(("face", item) for item in model.faces_using_edge(identifier))
        keys.update(("coedge", item) for item in model.coedges_using_edge(identifier))
        keys.update(("member", item) for item in model.members_using_edge(identifier))
    elif kind == "face":
        face = model.faces[identifier]
        keys.update(("edge", item.edge) for loop in (face.loop,) + face.holes for item in loop)
        keys.update(("face_use", item) for item in model._face_structural_uses.get(identifier, ()))  # noqa: SLF001
        keys.update(("attachment", item) for item in model.attachments_for_face(identifier))
    elif kind == "part":
        item = model.parts[identifier]
        keys.update(("sheet", value) for value in item.sheet_ids)
        keys.update(("member", value) for value in item.member_ids)
    elif kind == "sheet":
        item = model.sheets[identifier]
        keys.add(("part", item.part_id))
        keys.update(("face_use", value) for value in item.face_use_ids)
        keys.update(("attachment", value) for value in model.attachments_for_sheet(identifier))
    elif kind == "face_use":
        item = model.face_uses[identifier]
        keys.update((("sheet", item.sheet_id), ("face", item.face_id)))
        keys.update(("coedge", value) for value in item.coedge_ids)
    elif kind == "coedge":
        item = model.coedges[identifier]
        keys.update((("face_use", item.face_use_id), ("edge", item.edge_id)))
    elif kind == "member":
        item = model.members[identifier]
        keys.add(("part", item.part_id))
        keys.update(("member_edge_use", value) for value in item.edge_use_ids)
        keys.update(("attachment", value) for value in model.attachments_for_member(identifier))
        keys.update(("junction", value) for value in model._member_junctions.get(identifier, ()))  # noqa: SLF001
    elif kind == "member_edge_use":
        item = model.member_edge_uses[identifier]
        keys.update((("member", item.member_id), ("edge", item.edge_id)))
    elif kind == "attachment":
        item = model.attachments[identifier]
        keys.update((item.source_key, item.target_key))
    elif kind == "junction":
        item = model.junctions[identifier]
        keys.update(("member", value.member_id) for value in item.member_uses)
        keys.update(("sheet", value) for value in item.sheet_ids)
        keys.update(("attachment", value) for value in item.attachment_ids)
    return tuple(model.handle(*key) for key in sorted(keys) if key[0] in _stores(model) and key[1] in _stores(model)[key[0]])


def _geometry_type(model: GeometryModel, handle: EntityHandle) -> str | None:
    if handle.kind == "edge":
        return type(model.edges[handle.id].curve).__name__
    if handle.kind == "face":
        support = model.faces[handle.id].support_surface
        return None if support is None else type(support).__name__
    return None


def _metadata(model: GeometryModel, handle: EntityHandle) -> Mapping[str, object]:
    record = _stores(model)[handle.kind][handle.id]
    return getattr(record, "metadata", {})


def _semantic_owner_part(model: GeometryModel, handle: EntityHandle) -> int | None:
    direct = model.owner_part(handle.kind, handle.id)
    if direct is not None:
        return direct
    owners: set[int] = set()
    if handle.kind == "face":
        owners.update(
            model.sheets[model.face_uses[use_id].sheet_id].part_id
            for use_id in model._face_structural_uses.get(handle.id, ())  # noqa: SLF001
        )
    elif handle.kind == "edge":
        owners.update(model.sheets[item].part_id for item in model.sheets_using_edge(handle.id))
        owners.update(model.members[item].part_id for item in model.members_using_edge(handle.id))
    elif handle.kind == "vertex":
        owners.update(model.sheets[item].part_id for item in model.sheets_containing_vertex(handle.id))
        owners.update(model.members[item].part_id for item in model.members_meeting_at_vertex(handle.id))
    return next(iter(owners)) if len(owners) == 1 else None


def _measurements(model: GeometryModel, handle: EntityHandle) -> Mapping[str, object]:
    result: dict[str, object] = {}
    try:
        if handle.kind == "vertex":
            result["position"] = Quantity(
                tuple(float(item) for item in model.vertex_position(handle.id)),
                model.units,
                "model_local",
            ).to_dict()
        elif handle.kind == "edge":
            result["length"] = Quantity(model.edge_length(handle.id), model.units).to_dict()
            if isinstance(model.edges[handle.id].curve, Arc):
                result["radius"] = Quantity(model.arc_frame(handle.id).radius, model.units).to_dict()
        elif handle.kind == "face":
            result["area"] = Quantity(
                measure(model, _entity_ref(handle), quantity="area").value,
                f"{model.units}^2",
            ).to_dict()
    except GeometryError:
        pass
    bounds = _bounds(model, handle)
    if bounds is not None:
        result["centroid"] = Quantity(
            tuple(float((bounds[i] + bounds[i + 3]) * 0.5) for i in range(3)),
            model.units,
            "model_local",
        ).to_dict()
    return result


def _topology(model: GeometryModel, handle: EntityHandle, detail: bool) -> Mapping[str, object]:
    if not detail:
        return {}
    adjacency = _adjacency(model, handle)
    if len(adjacency) > 256:
        adjacency = adjacency[:256]
    return {"adjacency": [{"model_id": str(item.model_id), "kind": item.kind, "id": item.id} for item in adjacency], "truncated": len(_adjacency(model, handle)) > len(adjacency)}


def summarize_entity(model: GeometryModel, handle: EntityHandle, *, detail: bool = False) -> EntitySummary:
    resolution = model.resolve_handle(handle)
    owner_id = _semantic_owner_part(model, handle) if resolution.status is ResolutionStatus.ACTIVE else None
    owner = None if owner_id is None else model.handle("part", owner_id)
    return EntitySummary(
        handle=handle,
        kind=handle.kind,
        groups=_groups(model, handle),
        tags=_tags(model, handle),
        owner=owner,
        adjacency=_adjacency(model, handle),
        geometry_type=_geometry_type(model, handle),
        bounds=_bounds(model, handle),
        measurements=_measurements(model, handle),
        replacement_status=resolution.status.value,
        topology=_topology(model, handle, detail),
    )


def describe_entities(
    model: GeometryModel,
    handles: Iterable[EntityHandle | Mapping[str, object]],
    *,
    request_id: str = "describe_entities",
    model_id: object = None,
    expected_revision: int | None = None,
    detail: bool = False,
    page_size: int = 100,
    cursor: str | None = None,
) -> SelectionResult:
    made_header = _header(model, None, request_id=request_id, model_id=model_id, expected_revision=expected_revision)
    canonical: list[EntityHandle] = []
    for index, raw in enumerate(handles):
        handle = raw if isinstance(raw, EntityHandle) else handle_from_dict(raw, path=f"$.handles[{index}]")
        resolution = model.resolve_handle(handle)
        if resolution.status is ResolutionStatus.WRONG_MODEL:
            raise _error("WRONG_MODEL", "entity handle belongs to another model")
        if resolution.status is not ResolutionStatus.ACTIVE:
            raise _error("INACTIVE_ENTITY", f"entity is {resolution.status.value}")
        canonical.append(handle)
    spec = SelectionSpec(
        PROTOCOL_VERSION,
        made_header.request_id,
        made_header.model_id,
        made_header.expected_revision,
        {"handles": [
            {"model_id": str(item.model_id), "kind": item.kind, "id": item.id}
            for item in canonical
        ]},
        "handle",
        False,
        page_size,
        cursor,
        None,
        detail,
    )
    return select_entities(model, spec)


def describe_model(model: GeometryModel, header: RequestHeader | Mapping[str, object] | None = None, *, request_id: str = "describe_model", model_id: object = None, expected_revision: int | None = None) -> Mapping[str, object]:
    made = _header(model, header, request_id=request_id, model_id=model_id, expected_revision=expected_revision)
    counts = {kind: len(store) for kind, store in _stores(model).items()}
    bounds = model.bounds()
    return {
        "protocol_version": made.protocol_version,
        "request_id": made.request_id,
        "model_id": str(model.model_id),
        "revision": model.revision,
        "geometry_schema_version": 4,
        "units": model.units,
        "frames": ["model_local", "world"],
        "bounds": None if bounds is None else list(bounds),
        "entity_counts": counts,
        "groups": sorted(model.groups),
        "tags": sorted({tag for values in model.tags.values() for tag in values}),
    }


def _range(value: object, model: GeometryModel, dimension: str, *, path: str) -> tuple[float | None, float | None]:
    if not isinstance(value, Mapping) or set(value) - {"min", "max"} or not value:
        raise _error("MALFORMED_SELECTOR", "range requires min and/or max", path=path)
    low = None if "min" not in value else _to_model_scalar(model, _quantity(value["min"], path=f"{path}.min"), dimension)
    high = None if "max" not in value else _to_model_scalar(model, _quantity(value["max"], path=f"{path}.max"), dimension)
    if low is not None and high is not None and low > high:
        raise _error("MALFORMED_SELECTOR", "range min exceeds max", path=path)
    return low, high


def _inside(value: float, limits: tuple[float | None, float | None]) -> bool:
    low, high = limits
    return (low is None or value >= low) and (high is None or value <= high)


def _handle_predicate(model: GeometryModel, handle: EntityHandle, node: Mapping[str, object], *, depth: int, counter: list[int]) -> bool:
    if depth > 8:
        raise _error("SELECTOR_TOO_DEEP", "Boolean selector depth exceeds 8")
    counter[0] += 1
    if counter[0] > 64:
        raise _error("SELECTOR_TOO_LARGE", "selector predicate count exceeds 64")
    if len(node) != 1:
        raise _error("MALFORMED_SELECTOR", "each selector node requires exactly one operator")
    op, value = next(iter(node.items()))
    if op in ("all", "any"):
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
            raise _error("MALFORMED_SELECTOR", f"{op} requires a non-empty array")
        results = [_handle_predicate(model, handle, item, depth=depth + 1, counter=counter) for item in value if isinstance(item, Mapping)]
        if len(results) != len(value):
            raise _error("MALFORMED_SELECTOR", f"{op} entries must be objects")
        return all(results) if op == "all" else any(results)
    if op == "not":
        if not isinstance(value, Mapping):
            raise _error("MALFORMED_SELECTOR", "not requires an object")
        return not _handle_predicate(model, handle, value, depth=depth + 1, counter=counter)
    if op == "kind":
        values = value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else (value,)
        return handle.kind in {_kind(item) for item in values}
    if op in ("handle", "handles"):
        values = (value,) if op == "handle" else value
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise _error("MALFORMED_SELECTOR", "handles requires an array")
        supplied = {handle_from_dict(item) for item in values}
        if any(item.model_id != model.model_id for item in supplied):
            raise _error("WRONG_MODEL", "selector contains a foreign handle")
        return handle in supplied
    if op == "group":
        if not isinstance(value, str):
            raise _error("MALFORMED_SELECTOR", "group requires a string")
        return value in _groups(model, handle)
    if op == "tag":
        if not isinstance(value, str):
            raise _error("MALFORMED_SELECTOR", "tag requires a string")
        return value in _tags(model, handle)
    if op == "owner":
        owner = handle_from_dict(value)
        if owner.model_id != model.model_id:
            raise _error("WRONG_MODEL", "owner selector belongs to another model")
        return owner.kind == "part" and _semantic_owner_part(model, handle) == owner.id
    if op in ("incident_to", "boundary_of", "connected_to"):
        target = handle_from_dict(value)
        if target.model_id != model.model_id:
            raise _error("WRONG_MODEL", "topology selector belongs to another model")
        if op == "incident_to":
            return target in _adjacency(model, handle)
        if op == "boundary_of":
            return handle in _adjacency(model, target)
        if handle == target:
            return True
        frontier = {handle}
        seen = set(frontier)
        for _ in range(8):
            frontier = {item for current in frontier for item in _adjacency(model, current)} - seen
            if target in frontier:
                return True
            seen.update(frontier)
            if not frontier or len(seen) > 10_000:
                break
        return False
    if op in ("curve_type", "support_type"):
        return isinstance(value, str) and _geometry_type(model, handle) == value
    if op == "metadata":
        if not isinstance(value, Mapping) or set(value) != {"key", "equals"} or not isinstance(value["key"], str) or ":" not in value["key"]:
            raise _error("MALFORMED_SELECTOR", "metadata requires namespaced key and equals")
        return _metadata(model, handle).get(value["key"], object()) == value["equals"]
    if op == "aabb":
        if not isinstance(value, Mapping) or set(value) != {"min", "max"}:
            raise _error("MALFORMED_SELECTOR", "aabb requires min and max quantities")
        low, high = _to_local_aabb(model, value["min"], value["max"])
        bounds = _bounds(model, handle)
        return bounds is not None and bool(np.all(np.asarray(bounds[:3]) <= high) and np.all(np.asarray(bounds[3:]) >= low))
    if op == "centroid_axis":
        if not isinstance(value, Mapping) or set(value) - {"axis", "min", "max"} or "axis" not in value:
            raise _error("MALFORMED_SELECTOR", "centroid_axis requires axis and bounds")
        axis = value["axis"]
        if axis not in ("x", "y", "z", 0, 1, 2):
            raise _error("MALFORMED_SELECTOR", "axis must be x, y, z, 0, 1, or 2")
        index = {"x": 0, "y": 1, "z": 2}.get(axis, axis)
        bounds = _bounds(model, handle)
        if bounds is None:
            return False
        scalar = (bounds[index] + bounds[index + 3]) * 0.5  # type: ignore[operator]
        limits = _range({key: value[key] for key in ("min", "max") if key in value}, model, "length", path="$.where.centroid_axis")
        return _inside(scalar, limits)
    if op in ("length", "area", "radius"):
        if handle.kind not in (("edge",) if op in ("length", "radius") else ("face",)):
            return False
        try:
            result = measure(model, _entity_ref(handle), quantity=op).value
        except GeometryError:
            return False
        assert isinstance(result, float)
        return _inside(result, _range(value, model, "area" if op == "area" else "length", path=f"$.where.{op}"))
    if op == "nearest":
        raise _error("MALFORMED_SELECTOR", "nearest is allowed only as the root selector")
    raise _error("UNSUPPORTED", f"unknown selector operator {op!r}")


def _nearest(model: GeometryModel, value: object) -> list[tuple[EntityHandle, float]]:
    if not isinstance(value, Mapping) or set(value) != {"point", "max_distance", "limit"}:
        raise _error("MALFORMED_SELECTOR", "nearest requires point, max_distance, and limit")
    if isinstance(value["limit"], bool) or not isinstance(value["limit"], int) or not 1 <= value["limit"] <= 1_000:
        raise _error("MALFORMED_SELECTOR", "nearest limit must be 1..1000")
    point = _to_local_point(model, _quantity(value["point"], path="$.where.nearest.point"))
    maximum = _to_model_scalar(model, _quantity(value["max_distance"], path="$.where.nearest.max_distance"), "length")
    keys = model.spatial_candidates(
        point - maximum,
        point + maximum,
        kinds=("vertex", "edge", "face"),
    )
    candidates: list[tuple[EntityHandle, float]] = []
    for handle in (model.handle(*key) for key in keys):
        ref, _witness, distance = closest_point(model, point, (_entity_ref(handle),))
        if distance <= maximum:
            candidates.append((model.handle(ref.kind, ref.id), distance))
    return sorted(candidates, key=lambda item: (item[1], item[0]))[: value["limit"]]


def _sort_key(model: GeometryModel, handle: EntityHandle, order: str, distances: Mapping[EntityHandle, float]) -> object:
    if order == "handle":
        return (False, handle)
    if order == "distance":
        value = distances.get(handle)
        return (value is None, 0.0 if value is None else value, handle)
    bounds = _bounds(model, handle)
    if order == "bounds":
        return (bounds is None, bounds or (), handle)
    if order == "centroid":
        centroid = () if bounds is None else tuple((bounds[i] + bounds[i + 3]) * 0.5 for i in range(3))
        return (bounds is None, centroid, handle)
    try:
        value = measure(model, _entity_ref(handle), quantity=order).value
    except (AutomationError, GeometryError):
        value = None
    return (value is None, 0.0 if value is None else value, handle)


def _all_conjunct(node: Mapping[str, object], operator: str) -> object | None:
    if operator in node:
        return node[operator]
    values = node.get("all")
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        for item in values:
            if isinstance(item, Mapping):
                found = _all_conjunct(item, operator)
                if found is not None:
                    return found
    return None


def _active_selector_handle(model: GeometryModel, raw: object, *, operator: str) -> EntityHandle:
    handle = handle_from_dict(raw, path=f"$.where.{operator}")
    resolution = model.resolve_handle(handle)
    if resolution.status is ResolutionStatus.WRONG_MODEL:
        raise _error("WRONG_MODEL", f"{operator} selector belongs to another model")
    if resolution.status is not ResolutionStatus.ACTIVE:
        raise _error("INACTIVE_ENTITY", f"{operator} selector target is {resolution.status.value}")
    return handle


def _connected_handles(model: GeometryModel, target: EntityHandle) -> tuple[EntityHandle, ...]:
    """Return the bounded reverse-incidence closure around one active handle."""

    frontier = {target}
    seen = set(frontier)
    for _ in range(8):
        frontier = {
            adjacent
            for current in frontier
            for adjacent in _adjacency(model, current)
        } - seen
        seen.update(frontier)
        if len(seen) > 10_000:
            raise _error("SELECTOR_TOO_LARGE", "connectivity closure exceeds 10,000 entities")
        if not frontier:
            break
    return tuple(sorted(seen))


def _candidate_handles(model: GeometryModel, where: Mapping[str, object]) -> tuple[EntityHandle, ...]:
    """Choose a conservative owner-indexed candidate set when available."""

    if set(where) in ({"handle"}, {"handles"}):
        raw = where["handles"] if "handles" in where else (where["handle"],)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            handles = tuple(handle_from_dict(item) for item in raw)
            if any(item.model_id != model.model_id for item in handles):
                raise _error("WRONG_MODEL", "selector contains a foreign handle")
            return tuple(sorted(set(handles)))
    group = _all_conjunct(where, "group")
    if isinstance(group, str):
        return tuple(model.handle(item.kind, item.id) for item in model.group(group))
    tag = _all_conjunct(where, "tag")
    if isinstance(tag, str):
        return tuple(
            model.handle(reference.kind, reference.id)
            for reference, tags in sorted(model.tags.items(), key=lambda item: (item[0].kind, item[0].id))
            if tag in tags
        )
    for operator in ("incident_to", "boundary_of"):
        raw_target = _all_conjunct(where, operator)
        if raw_target is not None:
            target = _active_selector_handle(model, raw_target, operator=operator)
            return _adjacency(model, target)
    connected = _all_conjunct(where, "connected_to")
    if connected is not None:
        return _connected_handles(
            model,
            _active_selector_handle(model, connected, operator="connected_to"),
        )
    owner = _all_conjunct(where, "owner")
    if owner is not None:
        target = _active_selector_handle(model, owner, operator="owner")
        if target.kind != "part":
            return ()
        closure = set(_connected_handles(model, target))
        closure.update(
            model.handle("vertex", vertex_id)
            for vertex_id, part_id in model.construction_vertices.items()
            if part_id == target.id
        )
        return tuple(sorted(closure))
    aabb = _all_conjunct(where, "aabb")
    kind_value = _all_conjunct(where, "kind")
    if isinstance(aabb, Mapping) and set(aabb) == {"min", "max"} and kind_value is not None:
        raw_kinds = kind_value if isinstance(kind_value, Sequence) and not isinstance(kind_value, (str, bytes)) else (kind_value,)
        kinds = tuple(_kind(item) for item in raw_kinds)
        if set(kinds) <= {"vertex", "edge", "face"}:
            low, high = _to_local_aabb(model, aabb["min"], aabb["max"])
            return tuple(model.handle(*key) for key in model.spatial_candidates(low, high, kinds=kinds))
    return _all_handles(model)


def _cursor(spec: SelectionSpec, offset: int) -> str:
    payload = {"revision": spec.expected_revision, "selector": canonical_digest(spec.where), "order": spec.order_by, "descending": spec.descending, "offset": offset}
    return urlsafe_b64encode(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).decode().rstrip("=")


def _cursor_offset(spec: SelectionSpec) -> int:
    if spec.cursor is None:
        return 0
    try:
        raw = spec.cursor + "=" * (-len(spec.cursor) % 4)
        payload = json.loads(urlsafe_b64decode(raw.encode()).decode())
        expected = {"revision": spec.expected_revision, "selector": canonical_digest(spec.where), "order": spec.order_by, "descending": spec.descending}
        if not isinstance(payload, dict) or any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError
        offset = payload["offset"]
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError
        return offset
    except Exception as error:
        raise _error("STALE_CURSOR", "cursor does not match this model revision and selector") from error


def select_entities(model: GeometryModel, spec: SelectionSpec | Mapping[str, object]) -> SelectionResult:
    made = spec if isinstance(spec, SelectionSpec) else SelectionSpec.from_dict(spec)
    _header(model, made)
    where = made.where
    distances: dict[EntityHandle, float] = {}
    if set(where) == {"nearest"}:
        nearest = _nearest(model, where["nearest"])
        handles = [item[0] for item in nearest]
        distances = dict(nearest)
    else:
        _validate_selector_shape(where, depth=1, counter=[0])
        handles = [
            handle
            for handle in _candidate_handles(model, where)
            if _handle_predicate(model, handle, where, depth=1, counter=[0])
        ]
    if made.order_by == "handle":
        handles.sort(reverse=made.descending)
    else:
        valid = [item for item in handles if not _sort_key(model, item, made.order_by, distances)[0]]
        missing = [item for item in handles if _sort_key(model, item, made.order_by, distances)[0]]
        valid.sort(
            key=lambda item: _sort_key(model, item, made.order_by, distances)[1:],
            reverse=made.descending,
        )
        missing.sort()
        handles = [*valid, *missing]
    total = len(handles)
    if total > 1_000:
        handles = handles[:1_000]
        total = 1_000
    if made.expected_cardinality is not None:
        low, high = made.expected_cardinality
        if not low <= total <= high:
            code = "AMBIGUOUS_SELECTION" if total > high else "CARDINALITY_MISMATCH"
            raise _error(code, f"selection returned {total}; expected {low}..{high}", details={"actual": total, "minimum": low, "maximum": high})
    offset = _cursor_offset(made)
    if offset > total:
        raise _error("STALE_CURSOR", "cursor is beyond the result set")
    page = handles[offset : offset + made.page_size]
    next_offset = offset + len(page)
    next_cursor = _cursor(made, next_offset) if next_offset < total else None
    return SelectionResult(
        PROTOCOL_VERSION, made.request_id, model.model_id, model.revision,
        tuple(summarize_entity(model, item, detail=made.detail) for item in page),
        total, next_cursor,
        evidence={"ordering": made.order_by, "candidate_count": len(handles), "page_offset": offset},
    )


def _validate_selector_shape(node: object, *, depth: int, counter: list[int]) -> None:
    """Validate the bounded Boolean structure even when a model is empty."""

    if depth > 8:
        raise _error("SELECTOR_TOO_DEEP", "Boolean selector depth exceeds 8")
    counter[0] += 1
    if counter[0] > 64:
        raise _error("SELECTOR_TOO_LARGE", "selector predicate count exceeds 64")
    if not isinstance(node, Mapping) or len(node) != 1:
        raise _error("MALFORMED_SELECTOR", "each selector node requires exactly one operator")
    op, value = next(iter(node.items()))
    allowed = {
        "kind", "handle", "handles", "group", "tag", "owner", "incident_to",
        "boundary_of", "connected_to", "curve_type", "support_type", "metadata",
        "aabb", "centroid_axis", "length", "area", "radius", "all", "any", "not",
    }
    if op not in allowed:
        raise _error("UNSUPPORTED", f"unknown selector operator {op!r}")
    if op in ("all", "any"):
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
            raise _error("MALFORMED_SELECTOR", f"{op} requires a non-empty array")
        for item in value:
            _validate_selector_shape(item, depth=depth + 1, counter=counter)
    elif op == "not":
        _validate_selector_shape(value, depth=depth + 1, counter=counter)


__all__ = ["describe_entities", "describe_model", "select_entities", "summarize_entity"]
