"""Revision-bound command planning and atomic application."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from threading import RLock
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence
from weakref import WeakKeyDictionary

import numpy as np

from ..editing import (
    circular_pattern,
    copy_entities,
    copy_rotated,
    copy_translated,
    linear_pattern,
    measure,
    mirror_entities,
    rectangular_pattern,
    rotate_entities,
    translate_entities,
)
from ..entities import EntityRef
from ..errors import GeometryError
from ..identity import EntityHandle, ResolutionStatus
from ..intersections import apply_imprint, plan_imprint, query_intersection
from ..model import GeometryModel
from ..policies import MutationPolicy
from .selection import (
    _header,
    _quantity,
    _semantic_owner_part,
    _to_local_point,
    _to_model_scalar,
    select_entities,
)
from .types import (
    PROTOCOL_VERSION,
    ApplyResult,
    AutomationError,
    Command,
    CommandBatch,
    EditPlan,
    Quantity,
    SelectionSpec,
    canonical_digest,
    handle_from_dict,
)

_OPERATIONS = frozenset(
    {
        "create_point", "create_edge", "create_face", "create_plate",
        "translate", "rotate", "move", "copy", "mirror", "pattern",
        "group", "tag", "delete", "imprint",
    }
)
_PORTS: Mapping[str, tuple[str, ...]] = {
    "create_point": ("vertex",),
    "create_edge": ("edge",),
    "create_face": ("face",),
    "create_plate": ("part", "sheet", "face", "face_use", "coedge", "edge"),
    "translate": ("entity",), "rotate": ("entity",), "move": ("entity",),
    "copy": ("entity",), "mirror": ("entity",), "pattern": ("entity",),
    "group": ("entity",), "tag": ("entity",), "delete": ("deleted",),
    "imprint": ("entity", "relation"),
}
_REQUIRED: Mapping[str, frozenset[str]] = {
    "create_point": frozenset(("position",)),
    "create_edge": frozenset(("start", "end")),
    "create_face": frozenset(("edges",)),
    "create_plate": frozenset(("vertices",)),
    "translate": frozenset(("targets", "vector")),
    "rotate": frozenset(("targets", "axis_point", "axis_direction", "angle")),
    "move": frozenset(("target", "to")),
    "copy": frozenset(("targets",)),
    "mirror": frozenset(("targets", "plane_point", "plane_normal")),
    "pattern": frozenset(("targets", "pattern")),
    "group": frozenset(("targets", "group")),
    "tag": frozenset(("targets", "tags")),
    "delete": frozenset(("targets",)),
    "imprint": frozenset(("first", "second", "policy")),
}
_OPTIONAL: Mapping[str, frozenset[str]] = {
    "create_face": frozenset(("corners",)),
    "copy": frozenset(("transform", "group_prefix")),
    "mirror": frozenset(("group_prefix",)),
    "pattern": frozenset(("group_prefix",)),
}

_APPLIED: "WeakKeyDictionary[GeometryModel, set[str]]" = WeakKeyDictionary()
_APPLIED_LOCK = RLock()


def _error(code: str, message: str, *, path: str = "$", details: Mapping[str, object] | None = None) -> AutomationError:
    return AutomationError(code, message, path=path, details={} if details is None else details)


def _command_shape(command: Command) -> None:
    if command.operation not in _OPERATIONS:
        raise _error("UNSUPPORTED", f"unsupported command operation {command.operation!r}")
    required = _REQUIRED[command.operation]
    optional = _OPTIONAL.get(command.operation, frozenset())
    keys = set(command.arguments)
    if required - keys:
        raise _error("MALFORMED_COMMAND", f"{command.name} is missing {sorted(required - keys)}")
    if keys - required - optional:
        raise _error("UNKNOWN_FIELD", f"{command.name} has unknown arguments {sorted(keys - required - optional)}")


def _handle(model: GeometryModel, raw: object, *, path: str) -> EntityHandle:
    if isinstance(raw, EntityHandle):
        made = raw
    elif isinstance(raw, Mapping):
        made = handle_from_dict(raw, path=path)
    else:
        raise _error("MALFORMED_HANDLE", "an EntityHandle object is required", path=path)
    resolution = model.resolve_handle(made)
    if resolution.status is ResolutionStatus.WRONG_MODEL:
        raise _error("WRONG_MODEL", "handle belongs to another model", path=path)
    if resolution.status is not ResolutionStatus.ACTIVE:
        raise _error("INACTIVE_ENTITY", f"handle is {resolution.status.value}", path=path)
    return made


def _symbol(raw: object) -> tuple[str, str] | None:
    if not isinstance(raw, str):
        return None
    pieces = raw.split(".")
    if len(pieces) != 2 or not all(pieces):
        raise _error("MALFORMED_SYMBOL", "symbol references use command.port")
    return pieces[0], pieces[1]


def _validate_symbol(raw: str, known: Mapping[str, tuple[str, ...]], *, command: str) -> None:
    name, port = _symbol(raw) or ("", "")
    if name not in known:
        raise _error("FORWARD_REFERENCE", f"{command} references unavailable command {name!r}")
    if port not in known[name]:
        raise _error("UNKNOWN_OUTPUT_PORT", f"{name!r} has no output port {port!r}")


def _selector(model: GeometryModel, batch: CommandBatch, raw: object, *, command: str) -> tuple[EntityHandle, ...]:
    if not isinstance(raw, Mapping):
        raise _error("MALFORMED_SELECTOR", "selector must be an object")
    allowed = {"where", "expected_cardinality", "order_by", "descending"}
    if set(raw) - allowed or "where" not in raw or "expected_cardinality" not in raw:
        raise _error("MALFORMED_SELECTOR", "mutation selectors require where and expected_cardinality")
    spec = SelectionSpec(
        PROTOCOL_VERSION, batch.request_id, batch.model_id, batch.expected_revision,
        raw["where"], str(raw.get("order_by", "handle")), bool(raw.get("descending", False)),
        1000, None, tuple(raw["expected_cardinality"]), False,  # type: ignore[arg-type]
    )
    return tuple(item.handle for item in select_entities(model, spec).entities)


def _existing_inputs(model: GeometryModel, batch: CommandBatch, raw: object, known: Mapping[str, tuple[str, ...]], *, command: str, path: str) -> tuple[EntityHandle, ...]:
    if isinstance(raw, str):
        _validate_symbol(raw, known, command=command)
        return ()
    if isinstance(raw, Mapping) and "where" in raw:
        return _selector(model, batch, raw, command=command)
    values = raw if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else (raw,)
    result: list[EntityHandle] = []
    for index, item in enumerate(values):
        if isinstance(item, str):
            _validate_symbol(item, known, command=command)
        else:
            result.append(_handle(model, item, path=f"{path}[{index}]"))
    return tuple(sorted(set(result)))


def _quantity_vector(model: GeometryModel, raw: object, *, path: str, direction: bool = False) -> np.ndarray:
    value = _quantity(raw, path=path)
    if not isinstance(value.value, tuple) or len(value.value) != 3:
        raise _error("MALFORMED_QUANTITY", "a three-vector is required", path=path)
    if direction:
        if value.unit != "1" or value.frame is None:
            raise _error("MALFORMED_QUANTITY", "directions require unit '1' and an explicit frame", path=path)
        vector = np.asarray(value.value, dtype=float)
        if value.frame == "world" and model.coordinate_transform is not None:
            vector = np.linalg.inv(np.asarray(model.coordinate_transform)[:3, :3]) @ vector
    else:
        origin = _to_local_point(model, Quantity((0.0, 0.0, 0.0), value.unit, value.frame))
        vector = _to_local_point(model, value) - origin
    if not np.all(np.isfinite(vector)) or (direction and np.linalg.norm(vector) <= 0.0):
        raise _error("MALFORMED_QUANTITY", "vector must be finite and non-zero")
    return vector


def _angle(model: GeometryModel, raw: object, *, path: str) -> float:
    return _to_model_scalar(model, _quantity(raw, path=path), "angle")


def _validate_arguments(model: GeometryModel, batch: CommandBatch, command: Command, known: Mapping[str, tuple[str, ...]]) -> tuple[EntityHandle, ...]:
    _command_shape(command)
    args = command.arguments
    inputs: list[EntityHandle] = []
    if command.operation == "create_point":
        _to_local_point(model, _quantity(args["position"], path=f"$.commands.{command.name}.position"))
    elif command.operation in ("create_edge",):
        for key in ("start", "end"):
            supplied = args[key]
            if isinstance(supplied, str):
                _validate_symbol(supplied, known, command=command.name)
            else:
                handle = _handle(model, supplied, path=f"$.commands.{command.name}.{key}")
                if handle.kind != "vertex":
                    raise _error("CARDINALITY_MISMATCH", "edge endpoints must be vertices")
                inputs.append(handle)
    elif command.operation in ("create_face", "create_plate"):
        key = "edges" if command.operation == "create_face" else "vertices"
        raw = args[key]
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) < 3:
            raise _error("CARDINALITY_MISMATCH", f"{key} requires at least three ordered references")
        expected_kind = "edge" if key == "edges" else "vertex"
        for item in raw:
            if isinstance(item, str):
                _validate_symbol(item, known, command=command.name)
            else:
                handle = _handle(model, item, path=f"$.commands.{command.name}.{key}")
                if handle.kind != expected_kind:
                    raise _error("CARDINALITY_MISMATCH", f"{key} must contain {expected_kind} handles")
                inputs.append(handle)
    elif command.operation in ("translate", "copy"):
        inputs.extend(_existing_inputs(model, batch, args["targets"], known, command=command.name, path=f"$.commands.{command.name}.targets"))
        if command.operation == "translate":
            vector = _quantity_vector(model, args["vector"], path=f"$.commands.{command.name}.vector")
            if np.linalg.norm(vector) <= 0.0:
                raise _error("MALFORMED_COMMAND", "translation vector must be non-zero")
        elif "transform" in args:
            _validate_transform(model, args["transform"], path=f"$.commands.{command.name}.transform")
    elif command.operation == "rotate":
        inputs.extend(_existing_inputs(model, batch, args["targets"], known, command=command.name, path=f"$.commands.{command.name}.targets"))
        _to_local_point(model, _quantity(args["axis_point"], path="$.axis_point"))
        _quantity_vector(model, args["axis_direction"], path="$.axis_direction", direction=True)
        angle = _angle(model, args["angle"], path="$.angle")
        if angle == 0.0:
            raise _error("MALFORMED_COMMAND", "rotation angle must be non-zero")
    elif command.operation == "move":
        inputs.extend(_existing_inputs(model, batch, args["target"], known, command=command.name, path=f"$.commands.{command.name}.target"))
        if inputs and (len(inputs) != 1 or inputs[0].kind != "vertex"):
            raise _error("CARDINALITY_MISMATCH", "move requires exactly one vertex")
        _to_local_point(model, _quantity(args["to"], path="$.to"))
    elif command.operation == "mirror":
        inputs.extend(_existing_inputs(model, batch, args["targets"], known, command=command.name, path="$.targets"))
        _to_local_point(model, _quantity(args["plane_point"], path="$.plane_point"))
        _quantity_vector(model, args["plane_normal"], path="$.plane_normal", direction=True)
    elif command.operation == "pattern":
        inputs.extend(_existing_inputs(model, batch, args["targets"], known, command=command.name, path="$.targets"))
        _validate_pattern(model, args["pattern"])
    elif command.operation in ("group", "tag", "delete"):
        inputs.extend(_existing_inputs(model, batch, args["targets"], known, command=command.name, path="$.targets"))
        if command.operation == "group" and (not isinstance(args["group"], str) or not args["group"]):
            raise _error("MALFORMED_COMMAND", "group requires a non-empty string")
        if command.operation == "tag":
            tags = args["tags"]
            if not isinstance(tags, Sequence) or isinstance(tags, (str, bytes)) or not tags or any(not isinstance(item, str) or not item for item in tags):
                raise _error("MALFORMED_COMMAND", "tags requires non-empty strings")
        if command.operation in ("group", "tag") and any(item.kind not in ("vertex", "edge", "face") for item in inputs):
            raise _error("UNSUPPORTED", "groups and tags currently apply to geometry handles")
    elif command.operation == "imprint":
        for key in ("first", "second"):
            item = args[key]
            if isinstance(item, str):
                raise _error("UNSUPPORTED", "imprint operands must exist during planning")
            handle = _handle(model, item, path=f"$.{key}")
            inputs.append(handle)
        try:
            MutationPolicy(args["policy"])
        except (TypeError, ValueError) as error:
            raise _error("UNSUPPORTED", f"unsupported imprint policy {args['policy']!r}") from error
        result = query_intersection(model, inputs[0], inputs[1])
        preview = plan_imprint(model, result, policy=MutationPolicy(args["policy"]))
        if preview.result.kind.value in ("unsupported", "capability_missing", "unclassified"):
            raise _error(preview.result.kind.value.upper(), "; ".join(preview.result.diagnostics))
    return tuple(sorted(set(inputs)))


def _validate_transform(model: GeometryModel, raw: object, *, path: str) -> None:
    if not isinstance(raw, Mapping) or len(raw) != 1:
        raise _error("MALFORMED_COMMAND", "transform requires exactly one of translate or rotate", path=path)
    kind, value = next(iter(raw.items()))
    if kind == "translate":
        _quantity_vector(model, value, path=f"{path}.translate")
    elif kind == "rotate":
        if not isinstance(value, Mapping) or set(value) != {"axis_point", "axis_direction", "angle"}:
            raise _error("MALFORMED_COMMAND", "rotate transform requires axis_point, axis_direction, angle")
        _to_local_point(model, _quantity(value["axis_point"], path=f"{path}.axis_point"))
        _quantity_vector(model, value["axis_direction"], path=f"{path}.axis_direction", direction=True)
        _angle(model, value["angle"], path=f"{path}.angle")
    else:
        raise _error("UNSUPPORTED", f"unsupported copy transform {kind!r}")


def _validate_pattern(model: GeometryModel, raw: object) -> None:
    if not isinstance(raw, Mapping) or "type" not in raw:
        raise _error("MALFORMED_COMMAND", "pattern requires a type")
    kind = raw["type"]
    if kind == "linear" and set(raw) == {"type", "direction", "spacing", "count"}:
        _quantity_vector(model, raw["direction"], path="$.pattern.direction", direction=True)
        _to_model_scalar(model, _quantity(raw["spacing"], path="$.pattern.spacing"), "length")
    elif kind == "circular" and set(raw) == {"type", "axis_point", "axis_direction", "angle", "count"}:
        _to_local_point(model, _quantity(raw["axis_point"], path="$.pattern.axis_point"))
        _quantity_vector(model, raw["axis_direction"], path="$.pattern.axis_direction", direction=True)
        _angle(model, raw["angle"], path="$.pattern.angle")
    elif kind == "rectangular" and set(raw) == {"type", "directions", "spacings", "counts"}:
        if not all(isinstance(raw[key], Sequence) and not isinstance(raw[key], (str, bytes)) for key in ("directions", "spacings", "counts")):
            raise _error("MALFORMED_COMMAND", "rectangular arrays require array fields")
        if not len(raw["directions"]) == len(raw["spacings"]) == len(raw["counts"]):
            raise _error("MALFORMED_COMMAND", "rectangular array dimensions disagree")
        for value in raw["directions"]:
            _quantity_vector(model, value, path="$.pattern.directions", direction=True)
        for value in raw["spacings"]:
            _to_model_scalar(model, _quantity(value, path="$.pattern.spacings"), "length")
    else:
        raise _error("UNSUPPORTED", f"unsupported or malformed pattern {kind!r}")
    count_values = (raw["count"],) if "count" in raw else tuple(raw.get("counts", ()))
    if any(isinstance(item, bool) or not isinstance(item, int) or not 1 <= item <= 1_000 for item in count_values):
        raise _error("PAYLOAD_TOO_LARGE", "pattern counts must be 1..1000")


@contextmanager
def _planning_guard(model: GeometryModel):
    revision = model.revision
    next_ids = dict(model._next_id)  # noqa: SLF001
    next_structural = dict(model._next_structural_id)  # noqa: SLF001
    spatial = model._spatial_index  # noqa: SLF001
    arc_cache = dict(model._arc_cache)  # noqa: SLF001
    length_cache = dict(model._edge_length_cache)  # noqa: SLF001
    try:
        yield
    finally:
        model._spatial_index = spatial  # noqa: SLF001
        model._arc_cache.clear()  # noqa: SLF001
        model._arc_cache.update(arc_cache)  # noqa: SLF001
        model._edge_length_cache.clear()  # noqa: SLF001
        model._edge_length_cache.update(length_cache)  # noqa: SLF001
        if model.revision != revision or model._next_id != next_ids or model._next_structural_id != next_structural:  # noqa: SLF001
            raise _error("INTERNAL_ERROR", "planning mutated model identity state")


def _counts(commands: Sequence[Command]) -> Mapping[str, int]:
    result: dict[str, int] = {}
    for command in commands:
        additions: Mapping[str, int] = {}
        if command.operation == "create_point": additions = {"vertex": 1}
        elif command.operation == "create_edge": additions = {"edge": 1}
        elif command.operation == "create_face": additions = {"face": 1}
        elif command.operation == "create_plate":
            count = len(command.arguments["vertices"])  # type: ignore[arg-type]
            additions = {"edge": count, "face": 1, "part": 1, "sheet": 1, "face_use": 1, "coedge": count}
        for kind, value in additions.items():
            result[kind] = result.get(kind, 0) + value
    return MappingProxyType(dict(sorted(result.items())))


def _count_delta(before: GeometryModel, after: GeometryModel) -> Mapping[str, int]:
    before_counts: dict[str, int] = {}
    after_counts: dict[str, int] = {}
    for kind, _identifier in before.entity_keys():
        before_counts[kind] = before_counts.get(kind, 0) + 1
    for kind, _identifier in after.entity_keys():
        after_counts[kind] = after_counts.get(kind, 0) + 1
    return MappingProxyType(
        {
            kind: after_counts.get(kind, 0) - before_counts.get(kind, 0)
            for kind in sorted(set(before_counts) | set(after_counts))
            if after_counts.get(kind, 0) != before_counts.get(kind, 0)
        }
    )


def _union_bounds(*values: tuple[float, ...] | None) -> tuple[float, ...] | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return tuple(
        min(value[index] for value in present) if index < 3 else max(value[index] for value in present)
        for index in range(6)
    )


def _preview_bounds(model: GeometryModel, fallback: tuple[float, ...] | None) -> tuple[float, ...] | None:
    result = fallback
    for change in model.last_change_set.affected_aabbs:
        result = _union_bounds(result, change.before, change.after)
    return result


def _expected_owners(
    model: GeometryModel,
    outputs: Mapping[str, Mapping[str, tuple[EntityHandle, ...]]],
) -> Mapping[str, tuple[str, ...]]:
    """Describe semantic Part ownership without exposing staged allocator IDs."""

    labels: dict[EntityHandle, str] = {}
    for name, ports in outputs.items():
        for port, handles in ports.items():
            if port == "deleted":
                continue
            for handle in handles:
                candidate = f"{name}.{port}"
                current = labels.get(handle)
                if current is None or port == handle.kind or (current.endswith(".entity") and port != "entity"):
                    labels[handle] = candidate
    result: dict[str, tuple[str, ...]] = {}
    for name, ports in outputs.items():
        for port, handles in ports.items():
            if port == "deleted":
                continue
            owners: set[str] = set()
            for handle in handles:
                if model.resolve_handle(handle).status is not ResolutionStatus.ACTIVE:
                    continue
                owner_id = _semantic_owner_part(model, handle)
                if owner_id is None or (handle.kind == "part" and handle.id == owner_id):
                    continue
                owner = model.handle("part", owner_id)
                owners.add(labels.get(owner, str(owner)))
            if owners:
                result[f"{name}.{port}"] = tuple(sorted(owners))
    return MappingProxyType(dict(sorted(result.items())))


def _staged_preview(
    model: GeometryModel,
    plan: EditPlan,
    fallback_bounds: tuple[float, ...] | None,
) -> tuple[Mapping[str, int], Mapping[str, tuple[str, ...]], tuple[float, ...] | None, tuple[str, ...]]:
    """Preview exact owner operations on an unpublished schema clone."""

    from ..serialization import from_dict, to_dict

    staged = from_dict(to_dict(model, include_features=False))
    try:
        outputs = _execute_sequence(staged, plan)
    except AutomationError as error:
        return _counts(plan.commands), MappingProxyType({}), fallback_bounds, (f"{error.code}: {error.message}",)
    except GeometryError as error:
        return _counts(plan.commands), MappingProxyType({}), fallback_bounds, (f"APPLICATION_FAILED: {error}",)
    except Exception as error:
        return _counts(plan.commands), MappingProxyType({}), fallback_bounds, (f"APPLICATION_FAILED: staged owner failure: {error}",)
    if staged.revision == plan.revision or staged.last_change_set.is_empty:
        return _count_delta(model, staged), _expected_owners(staged, outputs), _preview_bounds(staged, fallback_bounds), ("NO_EFFECT: batch would not change the model",)
    return _count_delta(model, staged), _expected_owners(staged, outputs), _preview_bounds(staged, fallback_bounds), ()


def plan_commands(model: GeometryModel, batch: CommandBatch | Mapping[str, object]) -> EditPlan:
    made = batch if isinstance(batch, CommandBatch) else CommandBatch.from_dict(batch)
    _header(model, made)
    known: dict[str, tuple[str, ...]] = {}
    resolved: dict[str, tuple[EntityHandle, ...]] = {}
    policies: dict[str, object] = {}
    capabilities: set[str] = set()
    with _planning_guard(model):
        for command in made.commands:
            values = _validate_arguments(model, made, command, known)
            resolved[command.name] = values
            known[command.name] = _PORTS[command.operation]
            capabilities.add(command.operation)
            if command.operation == "imprint":
                policies[command.name] = command.arguments["policy"]
        bounds = model.bounds(handle.key for values in resolved.values() for handle in values if handle.kind in ("vertex", "edge", "face"))
    placeholder = EditPlan(
        PROTOCOL_VERSION, made.request_id, model.model_id, model.revision, made.commands,
        resolved, known, _counts(made.commands), {}, bounds, policies, (), tuple(sorted(capabilities)), "0" * 64,
    )
    with _planning_guard(model):
        counts, owners, preview_bounds, diagnostics = _staged_preview(model, placeholder, bounds)
    preview = replace(
        placeholder,
        expected_entity_counts=counts,
        expected_owners=owners,
        affected_bounds=preview_bounds,
        diagnostics=diagnostics,
    )
    return replace(preview, digest=canonical_digest(preview.digest_payload()))


def _resolve_apply_ref(model: GeometryModel, raw: object, outputs: Mapping[str, Mapping[str, tuple[EntityHandle, ...]]], *, path: str) -> tuple[EntityHandle, ...]:
    if isinstance(raw, str):
        name, port = _symbol(raw) or ("", "")
        try:
            return outputs[name][port]
        except KeyError as error:
            raise _error("MALFORMED_SYMBOL", f"unresolved symbol {raw!r}", path=path) from error
    if isinstance(raw, Mapping) and "where" in raw:
        raise _error("INTERNAL_ERROR", "selectors must be frozen before apply", path=path)
    values = raw if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else (raw,)
    result: list[EntityHandle] = []
    for index, value in enumerate(values):
        if isinstance(value, str):
            result.extend(_resolve_apply_ref(model, value, outputs, path=f"{path}[{index}]"))
        else:
            result.append(_handle(model, value, path=f"{path}[{index}]"))
    return tuple(result)


def _ports(handles: Iterable[EntityHandle], declared: Sequence[str]) -> Mapping[str, tuple[EntityHandle, ...]]:
    values = tuple(sorted(set(handles)))
    made: dict[str, tuple[EntityHandle, ...]] = {}
    for port in declared:
        if port in ("entity", "deleted"):
            made[port] = values
        elif port == "relation":
            made[port] = tuple(
                item for item in values if item.kind in ("attachment", "junction")
            )
        else:
            made[port] = tuple(item for item in values if item.kind == port)
    return MappingProxyType(made)


def _runtime_targets(
    model: GeometryModel,
    raw: object,
    frozen: tuple[EntityHandle, ...],
    outputs: Mapping[str, Mapping[str, tuple[EntityHandle, ...]]],
    *,
    path: str,
) -> tuple[EntityHandle, ...]:
    """Resolve symbols at apply time and reuse frozen selector results."""

    if isinstance(raw, Mapping) and "where" in raw:
        return frozen
    return _resolve_apply_ref(model, raw, outputs, path=path)


def _refs(handles: Iterable[EntityHandle]) -> tuple[EntityRef, ...]:
    result = []
    for handle in handles:
        if handle.kind not in ("vertex", "edge", "face"):
            raise _error("UNSUPPORTED", f"operation requires geometry, not {handle.kind}")
        result.append(EntityRef(handle.kind, handle.id))  # type: ignore[arg-type]
    return tuple(result)


def _apply_command(model: GeometryModel, command: Command, frozen: tuple[EntityHandle, ...], outputs: Mapping[str, Mapping[str, tuple[EntityHandle, ...]]]) -> Mapping[str, tuple[EntityHandle, ...]]:
    args = command.arguments
    operation = command.operation
    made: tuple[EntityHandle, ...]
    if operation == "create_point":
        point = _to_local_point(model, _quantity(args["position"], path="$.position"))
        made = (model.handle("vertex", model.add_point(*point)),)
    elif operation == "create_edge":
        start = _resolve_apply_ref(model, args["start"], outputs, path="$.start")
        end = _resolve_apply_ref(model, args["end"], outputs, path="$.end")
        if len(start) != 1 or len(end) != 1 or start[0].kind != "vertex" or end[0].kind != "vertex":
            raise _error("CARDINALITY_MISMATCH", "edge endpoints must resolve to one vertex each")
        made = (model.handle("edge", model.add_line(start[0].id, end[0].id)),)
    elif operation in ("create_face", "create_plate"):
        key = "edges" if operation == "create_face" else "vertices"
        handles = _resolve_apply_ref(model, args[key], outputs, path=f"$.{key}")
        expected = "edge" if operation == "create_face" else "vertex"
        if len(handles) < 3 or any(item.kind != expected for item in handles):
            raise _error("CARDINALITY_MISMATCH", f"{key} must resolve to at least three {expected} handles")
        if operation == "create_face":
            corners = args.get("corners")
            face_id = model.add_face([item.id for item in handles], corners)  # type: ignore[arg-type]
            made = (model.handle("face", face_id),)
        else:
            before_edges = set(model.edges)
            face_id = model.add_plate([item.id for item in handles])
            part_id = model.add_part()
            sheet_id = model.add_sheet((face_id,), part_id=part_id)
            sheet = model.sheets[sheet_id]
            uses = tuple(model.face_uses[item] for item in sheet.face_use_ids)
            keys = [("face", face_id), ("part", part_id), ("sheet", sheet_id)]
            keys.extend(("edge", item) for item in sorted(set(model.edges) - before_edges))
            keys.extend(("face_use", item.id) for item in uses)
            keys.extend(("coedge", item) for use in uses for item in use.coedge_ids)
            made = tuple(model.handle(*item) for item in keys)
    elif operation in ("translate", "rotate", "move"):
        raw_targets = args["target"] if operation == "move" else args["targets"]
        targets = _runtime_targets(
            model, raw_targets, frozen, outputs, path="$.target" if operation == "move" else "$.targets"
        )
        refs = _refs(targets)
        if operation == "translate":
            vector = _quantity_vector(model, args["vector"], path="$.vector")
            translate_entities(model, refs, vector)
        elif operation == "rotate":
            point = _to_local_point(model, _quantity(args["axis_point"], path="$.axis_point"))
            direction = _quantity_vector(model, args["axis_direction"], path="$.axis_direction", direction=True)
            rotate_entities(model, refs, point, direction, _angle(model, args["angle"], path="$.angle"))
        else:
            if len(targets) != 1 or targets[0].kind != "vertex":
                raise _error("CARDINALITY_MISMATCH", "move requires exactly one vertex")
            destination = _to_local_point(model, _quantity(args["to"], path="$.to"))
            translate_entities(model, refs, destination - model.vertex_position(targets[0].id))
        made = targets
    elif operation == "copy":
        refs = _runtime_targets(model, args["targets"], frozen, outputs, path="$.targets")
        transform = args.get("transform")
        if transform is None:
            result = copy_entities(model, refs, group_prefix=args.get("group_prefix"))  # type: ignore[arg-type]
        else:
            kind, value = next(iter(transform.items()))  # type: ignore[union-attr]
            if kind == "translate":
                result = copy_translated(model, refs, _quantity_vector(model, value, path="$.transform.translate"), group_prefix=args.get("group_prefix"))  # type: ignore[arg-type]
            else:
                result = copy_rotated(
                    model, refs,
                    _to_local_point(model, _quantity(value["axis_point"], path="$.axis_point")),  # type: ignore[index]
                    _quantity_vector(model, value["axis_direction"], path="$.axis_direction", direction=True),  # type: ignore[index]
                    _angle(model, value["angle"], path="$.angle"),  # type: ignore[index]
                    group_prefix=args.get("group_prefix"),  # type: ignore[arg-type]
                )
        made = tuple(sorted(result.handle_map.values()))
    elif operation == "mirror":
        targets = _runtime_targets(model, args["targets"], frozen, outputs, path="$.targets")
        result = mirror_entities(
            model, targets,
            _to_local_point(model, _quantity(args["plane_point"], path="$.plane_point")),
            _quantity_vector(model, args["plane_normal"], path="$.plane_normal", direction=True),
            group_prefix=args.get("group_prefix"),  # type: ignore[arg-type]
        )
        made = tuple(sorted(result.handle_map.values()))
    elif operation == "pattern":
        targets = _runtime_targets(model, args["targets"], frozen, outputs, path="$.targets")
        raw = args["pattern"]
        kind = raw["type"]  # type: ignore[index]
        prefix = args.get("group_prefix")
        if kind == "linear":
            result = linear_pattern(model, targets, _quantity_vector(model, raw["direction"], path="$.direction", direction=True), _to_model_scalar(model, _quantity(raw["spacing"], path="$.spacing"), "length"), raw["count"], group_prefix=prefix)  # type: ignore[index,arg-type]
        elif kind == "circular":
            result = circular_pattern(model, targets, _to_local_point(model, _quantity(raw["axis_point"], path="$.axis_point")), _quantity_vector(model, raw["axis_direction"], path="$.axis_direction", direction=True), _angle(model, raw["angle"], path="$.angle"), raw["count"], group_prefix=prefix)  # type: ignore[index,arg-type]
        else:
            result = rectangular_pattern(model, targets, [_quantity_vector(model, item, path="$.directions", direction=True) for item in raw["directions"]], [_to_model_scalar(model, _quantity(item, path="$.spacings"), "length") for item in raw["spacings"]], raw["counts"], group_prefix=prefix)  # type: ignore[index,arg-type]
        made = tuple(sorted(handle for instance in result.instances for handle in instance.handle_map.values()))
    elif operation == "group":
        targets = _runtime_targets(model, args["targets"], frozen, outputs, path="$.targets")
        model.add_to_group(args["group"], _refs(targets))  # type: ignore[arg-type]
        made = targets
    elif operation == "tag":
        targets = _runtime_targets(model, args["targets"], frozen, outputs, path="$.targets")
        for reference in _refs(targets):
            model.tag(reference, *args["tags"])  # type: ignore[arg-type]
        made = targets
    elif operation == "delete":
        targets = _runtime_targets(model, args["targets"], frozen, outputs, path="$.targets")
        _delete(model, targets)
        made = targets
    elif operation == "imprint":
        first = _resolve_apply_ref(model, args["first"], outputs, path="$.first")
        second = _resolve_apply_ref(model, args["second"], outputs, path="$.second")
        if len(first) != 1 or len(second) != 1:
            raise _error("CARDINALITY_MISMATCH", "imprint operands must resolve to one handle each")
        result = query_intersection(model, first[0], second[0])
        planned = plan_imprint(model, result, policy=MutationPolicy(args["policy"]))
        application = apply_imprint(model, planned, policy=MutationPolicy(args["policy"]))
        made = tuple(sorted({*application.relations, *(model.handle(*item) for item in application.change_set.added if item[0] in ("vertex", "edge", "face", "part", "sheet", "face_use", "coedge", "member", "member_edge_use", "attachment", "junction"))}))
    else:  # pragma: no cover - guarded by planning
        raise _error("UNSUPPORTED", f"unsupported command {operation}")
    return _ports(made, _PORTS[operation])


def _delete(model: GeometryModel, handles: Sequence[EntityHandle]) -> None:
    by_kind: dict[str, list[int]] = {}
    for handle in handles:
        by_kind.setdefault(handle.kind, []).append(handle.id)
    for kind, function in (
        ("junction", model.remove_junction), ("attachment", model.remove_attachment),
        ("member", model.remove_member), ("sheet", model.remove_sheet), ("part", model.remove_part),
    ):
        for identifier in sorted(by_kind.get(kind, ()), reverse=True):
            function(identifier)
    geometry = [(kind, identifier) for kind in ("face", "edge", "vertex") for identifier in sorted(by_kind.get(kind, ()), reverse=True)]
    if geometry:
        model.remove_entities(geometry)
    unsupported = set(by_kind) - {"junction", "attachment", "member", "sheet", "part", "face", "edge", "vertex"}
    if unsupported:
        raise _error("UNSUPPORTED", f"direct deletion is unsupported for {sorted(unsupported)}")


def _audit_summary(model: GeometryModel, change_set) -> Mapping[str, object]:
    try:
        report = model.audit_changed_region(change_set)
        return {
            "scope": report.scope.value,
            "clean": report.clean,
            "certifiable": report.certifiable,
            "completed": report.completed,
            "verified": report.verified,
            "issue_counts": report.issue_counts,
            "metrics": report.metrics.to_dict(),
            "issues": [item.to_dict() for item in report.issues[:100]],
            "truncated": len(report.issues) > 100,
        }
    except Exception as error:  # audit is evidence, never a reason to hide a committed edit
        return {"scope": "changed_region", "clean": False, "certifiable": False, "completed": False, "verified": False, "diagnostic": str(error)}


def _execute_sequence(
    model: GeometryModel,
    plan: EditPlan,
) -> dict[str, Mapping[str, tuple[EntityHandle, ...]]]:
    outputs: dict[str, Mapping[str, tuple[EntityHandle, ...]]] = {}
    with model.transaction():
        for command in plan.commands:
            before_geometry = dict(model._next_id)  # noqa: SLF001
            before_structural = dict(model._next_structural_id)  # noqa: SLF001
            ports = dict(
                _apply_command(
                    model,
                    command,
                    plan.resolved_inputs.get(command.name, ()),
                    outputs,
                )
            )
            created_keys = [
                (kind, identifier)
                for kind, first in (*before_geometry.items(), *before_structural.items())
                for identifier in range(
                    first,
                    (
                        model._next_id[kind]  # noqa: SLF001
                        if kind in model._next_id  # noqa: SLF001
                        else model._next_structural_id[kind]  # noqa: SLF001
                    ),
                )
                if model._contains_entity(kind, identifier)  # noqa: SLF001
            ]
            created = tuple(model.handle(*key) for key in sorted(created_keys))
            if created:
                for port in _PORTS[command.operation]:
                    if port == "entity":
                        ports[port] = tuple(sorted(set((*ports.get(port, ()), *created))))
                    elif port == "relation":
                        relations = tuple(
                            item for item in created if item.kind in ("attachment", "junction")
                        )
                        ports[port] = tuple(sorted(set((*ports.get(port, ()), *relations))))
                    elif port not in ("deleted",):
                        typed = tuple(item for item in created if item.kind == port)
                        ports[port] = tuple(sorted(set((*ports.get(port, ()), *typed))))
            outputs[command.name] = MappingProxyType(ports)
    return outputs


def _preflight_application(model: GeometryModel, plan: EditPlan) -> None:
    """Execute on an unpublished exact clone before touching live allocators."""

    from ..serialization import from_dict, to_dict

    try:
        staged = from_dict(to_dict(model, include_features=False))
        _execute_sequence(staged, plan)
    except AutomationError:
        raise
    except GeometryError as error:
        raise _error("APPLICATION_FAILED", str(error)) from error
    except Exception as error:
        raise _error("APPLICATION_FAILED", f"staged owner failure: {error}") from error


def apply_plan(model: GeometryModel, plan: EditPlan) -> ApplyResult:
    if not isinstance(plan, EditPlan):
        raise _error("MALFORMED_PLAN", "apply_plan requires an EditPlan")
    if plan.model_id != model.model_id:
        raise _error("WRONG_MODEL", "plan belongs to another model")
    if canonical_digest(plan.digest_payload()) != plan.digest:
        raise _error("TAMPERED_PLAN", "plan digest does not match its canonical payload")
    if model.revision != plan.revision:
        raise _error("STALE_PLAN", "plan revision is stale", details={"planned": plan.revision, "actual": model.revision})
    if plan.diagnostics:
        raise _error("BLOCKED_PLAN", "plan contains blocking diagnostics", details={"diagnostics": plan.diagnostics})
    with _APPLIED_LOCK:
        if plan.digest in _APPLIED.get(model, set()):
            raise _error("STALE_PLAN", "plan was already applied")
    _preflight_application(model, plan)
    revision_before = model.revision
    try:
        outputs = _execute_sequence(model, plan)
    except AutomationError:
        raise
    except GeometryError as error:
        raise _error("APPLICATION_FAILED", str(error)) from error
    except Exception as error:
        raise _error("APPLICATION_FAILED", f"unexpected owner failure: {error}") from error
    change_set = model.last_change_set
    if change_set.revision_before != revision_before or change_set.revision_after != model.revision:
        raise _error("INTERNAL_ERROR", "batch did not publish exactly one outer ChangeSet")
    with _APPLIED_LOCK:
        _APPLIED.setdefault(model, set()).add(plan.digest)
    flat = {f"{name}.{port}": values for name, ports in outputs.items() for port, values in ports.items()}
    replacements: dict[str, tuple[EntityHandle, ...]] = {}
    for values in plan.resolved_inputs.values():
        for handle in values:
            resolution = model.resolve_handle(handle)
            if resolution.status is ResolutionStatus.REPLACED:
                replacements[str(handle)] = resolution.resolved
    return ApplyResult(
        PROTOCOL_VERSION, plan.request_id, model.model_id, revision_before, model.revision,
        plan.digest, flat, replacements, change_set, _audit_summary(model, change_set),
    )


def execute_query(model: GeometryModel, request: Mapping[str, object]) -> Mapping[str, object]:
    required = {"protocol_version", "request_id", "model_id", "expected_revision", "operation", "arguments"}
    if not isinstance(request, Mapping) or set(request) != required:
        raise _error("MALFORMED_REQUEST", "query requires protocol header, operation, and arguments")
    _header(model, {key: request[key] for key in ("protocol_version", "request_id", "model_id", "expected_revision")})
    operation = request["operation"]
    arguments = request["arguments"]
    if not isinstance(arguments, Mapping):
        raise _error("MALFORMED_REQUEST", "query arguments must be an object")
    if operation == "measure":
        if set(arguments) != {"handles", "quantity"}:
            raise _error("MALFORMED_REQUEST", "measure requires handles and quantity")
        raw = arguments["handles"]
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise _error("MALFORMED_REQUEST", "measure handles must be an array")
        handles = tuple(_handle(model, item, path="$.arguments.handles") for item in raw)
        measurement = measure(model, tuple(_refs(handles)), quantity=str(arguments["quantity"]))
        unit = (
            f"{model.units}^2"
            if measurement.kind == "area"
            else "rad"
            if measurement.kind == "angle"
            else "1"
            if measurement.kind == "normal"
            else model.units
        )
        return {"protocol_version": PROTOCOL_VERSION, "request_id": request["request_id"], "model_id": str(model.model_id), "revision": model.revision, "kind": measurement.kind, "value": measurement.value, "unit": unit, "witnesses": [list(item) for item in measurement.witnesses]}
    if operation == "intersection":
        if set(arguments) != {"first", "second"}:
            raise _error("MALFORMED_REQUEST", "intersection requires first and second")
        first = _handle(model, arguments["first"], path="$.arguments.first")
        second = _handle(model, arguments["second"], path="$.arguments.second")
        result = query_intersection(model, first, second)
        return {
            "protocol_version": PROTOCOL_VERSION, "request_id": request["request_id"], "model_id": str(model.model_id), "revision": model.revision,
            "kind": result.kind.value, "dimension": result.dimension.value, "quality": result.quality.value,
            "tolerance_used": result.tolerance_used, "max_residual": result.max_residual,
            "diagnostics": list(result.diagnostics), "witnesses": [list(item) for item in result.witnesses],
            "component_count": len(result.components),
        }
    raise _error("UNSUPPORTED", f"unsupported query operation {operation!r}")


__all__ = ["apply_plan", "execute_query", "plan_commands"]
