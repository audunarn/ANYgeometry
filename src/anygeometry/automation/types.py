"""Immutable value algebra for the provider-neutral automation protocol.

The records in this module deliberately contain only JSON-compatible values
and public :class:`~anygeometry.identity.EntityHandle` identities.  They are
safe to expose through an MCP server, a command line adapter, or an in-process
agent without giving the caller an executable escape hatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from types import MappingProxyType
from typing import Any, ClassVar, Iterable, Mapping, Sequence
from uuid import UUID

from ..entities import EntityRef
from ..identity import (
    EntityHandle,
    canonical_model_id,
    validate_entity_kind,
    validate_local_id,
)
from ..transactions import AABBChange, ChangeSet

PROTOCOL_VERSION = 1
MAX_COMMANDS = 256
MAX_PAGE_SIZE = 1_000
DEFAULT_PAGE_SIZE = 100

JsonScalar = None | bool | int | float | str
JsonValue = JsonScalar | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]


def _fail(code: str, message: str, *, path: str = "$", details: Mapping[str, object] | None = None) -> "AutomationError":
    return AutomationError(code, message, path=path, details={} if details is None else details)


def _finite(value: object, *, path: str) -> float:
    if isinstance(value, bool):
        raise _fail("MALFORMED_REQUEST", "a finite number is required", path=path)
    try:
        made = float(value)
    except (TypeError, ValueError) as error:
        raise _fail("MALFORMED_REQUEST", "a finite number is required", path=path) from error
    if not math.isfinite(made):
        raise _fail("NONFINITE_VALUE", "NaN and infinity are not permitted", path=path)
    return made


def _json_value(value: object, *, path: str = "$", depth: int = 0) -> JsonValue:
    if depth > 16:
        raise _fail("PAYLOAD_TOO_DEEP", "JSON value nesting exceeds 16", path=path)
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _fail("NONFINITE_VALUE", "NaN and infinity are not permitted", path=path)
        return value
    if isinstance(value, Mapping):
        if len(value) > 256:
            raise _fail("PAYLOAD_TOO_LARGE", "mapping exceeds 256 fields", path=path)
        made: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key or "\x00" in key:
                raise _fail("MALFORMED_REQUEST", "mapping keys must be non-empty strings", path=path)
            made[key] = _json_value(item, path=f"{path}.{key}", depth=depth + 1)
        return MappingProxyType(dict(sorted(made.items())))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > 4_096:
            raise _fail("PAYLOAD_TOO_LARGE", "array exceeds 4096 entries", path=path)
        return tuple(
            _json_value(item, path=f"{path}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        )
    raise _fail("MALFORMED_REQUEST", f"unsupported JSON value {type(value).__name__}", path=path)


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, EntityHandle):
        return handle_to_dict(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()  # type: ignore[no-any-return]
    return value


def canonical_json(value: object) -> str:
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_digest(value: object) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _strict(data: object, required: Iterable[str], optional: Iterable[str] = (), *, path: str = "$") -> Mapping[str, object]:
    if not isinstance(data, Mapping):
        raise _fail("MALFORMED_REQUEST", "an object is required", path=path)
    keys = set(data)
    required_set = set(required)
    optional_set = set(optional)
    missing = sorted(required_set - keys)
    extra = sorted(keys - required_set - optional_set)
    if missing:
        raise _fail("MALFORMED_REQUEST", f"missing fields: {', '.join(missing)}", path=path)
    if extra:
        raise _fail("UNKNOWN_FIELD", f"unknown fields: {', '.join(extra)}", path=path)
    return data


def _request_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 128 or "\x00" in value:
        raise _fail("MALFORMED_REQUEST", "request_id must be 1..128 safe characters", path="$.request_id")
    return value


def _protocol(value: object) -> int:
    if isinstance(value, bool) or value != PROTOCOL_VERSION:
        raise _fail("UNSUPPORTED_PROTOCOL", f"protocol_version must be {PROTOCOL_VERSION}", path="$.protocol_version")
    return PROTOCOL_VERSION


def handle_to_dict(handle: EntityHandle) -> dict[str, object]:
    return {"model_id": str(handle.model_id), "kind": handle.kind, "id": handle.id}


def handle_from_dict(data: object, *, path: str = "$") -> EntityHandle:
    made = _strict(data, ("model_id", "kind", "id"), path=path)
    try:
        return EntityHandle(made["model_id"], made["kind"], made["id"])  # type: ignore[arg-type]
    except Exception as error:
        raise _fail("MALFORMED_HANDLE", str(error), path=path) from error


def _string_array(value: object, *, path: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or any(not isinstance(item, str) for item in value):
        raise _fail("MALFORMED_REQUEST", "an array of strings is required", path=path)
    return tuple(value)


def _handle_array(value: object, *, path: str) -> tuple[EntityHandle, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _fail("MALFORMED_REQUEST", "an array of handles is required", path=path)
    return tuple(handle_from_dict(item, path=f"{path}[{index}]") for index, item in enumerate(value))


def _entity_key(value: object, *, path: str) -> tuple[str, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise _fail("MALFORMED_REQUEST", "entity key requires [kind,id]", path=path)
    try:
        return validate_entity_kind(value[0]), validate_local_id(value[1])
    except Exception as error:
        raise _fail("MALFORMED_REQUEST", str(error), path=path) from error


def _key_array(value: object, *, path: str) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _fail("MALFORMED_REQUEST", "an array of entity keys is required", path=path)
    return tuple(_entity_key(item, path=f"{path}[{index}]") for index, item in enumerate(value))


def _change_set_to_dict(change: ChangeSet) -> dict[str, object]:
    return {
        "revision_before": change.revision_before,
        "revision_after": change.revision_after,
        "added": [list(item) for item in change.added],
        "removed": [list(item) for item in change.removed],
        "modified": [list(item) for item in change.modified],
        "replacements": [
            {
                "source": [source.kind, source.id],
                "targets": [[target.kind, target.id] for target in targets],
            }
            for source, targets in change.replacements
        ],
        "ownership_changes": [list(item) for item in change.ownership_changes],
        "member_changes": [list(item) for item in change.member_changes],
        "attachment_changes": [list(item) for item in change.attachment_changes],
        "group_changes": list(change.group_changes),
        "tag_changes": [list(item) for item in change.tag_changes],
        "affected_aabbs": [
            {
                "entity": list(item.entity),
                "before": None if item.before is None else list(item.before),
                "after": None if item.after is None else list(item.after),
            }
            for item in change.affected_aabbs
        ],
        "invalidated_caches": [list(item) for item in change.invalidated_caches],
        "spatial_updates": [list(item) for item in change.spatial_updates],
        "feature_history_changed": change.feature_history_changed,
        "document_settings_changed": change.document_settings_changed,
    }


def _change_set_from_dict(value: object, *, path: str) -> ChangeSet:
    fields = (
        "revision_before", "revision_after", "added", "removed", "modified",
        "replacements", "ownership_changes", "member_changes",
        "attachment_changes", "group_changes", "tag_changes",
        "affected_aabbs", "invalidated_caches", "spatial_updates",
        "feature_history_changed", "document_settings_changed",
    )
    made = _strict(value, fields, path=path)
    revisions = (made["revision_before"], made["revision_after"])
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in revisions):
        raise _fail("MALFORMED_REQUEST", "ChangeSet revisions must be non-negative integers", path=path)
    replacements_raw = made["replacements"]
    if not isinstance(replacements_raw, Sequence) or isinstance(replacements_raw, (str, bytes)):
        raise _fail("MALFORMED_REQUEST", "replacements must be an array", path=f"{path}.replacements")
    replacements: list[tuple[EntityRef, tuple[EntityRef, ...]]] = []
    for index, raw in enumerate(replacements_raw):
        entry = _strict(raw, ("source", "targets"), path=f"{path}.replacements[{index}]")
        source_kind, source_id = _entity_key(entry["source"], path=f"{path}.replacements[{index}].source")
        targets = _key_array(entry["targets"], path=f"{path}.replacements[{index}].targets")
        try:
            replacements.append((EntityRef(source_kind, source_id), tuple(EntityRef(kind, identifier) for kind, identifier in targets)))  # type: ignore[arg-type]
        except Exception as error:
            raise _fail("MALFORMED_REQUEST", str(error), path=f"{path}.replacements[{index}]") from error
    aabbs_raw = made["affected_aabbs"]
    if not isinstance(aabbs_raw, Sequence) or isinstance(aabbs_raw, (str, bytes)):
        raise _fail("MALFORMED_REQUEST", "affected_aabbs must be an array", path=f"{path}.affected_aabbs")
    aabbs: list[AABBChange] = []
    for index, raw in enumerate(aabbs_raw):
        entry = _strict(raw, ("entity", "before", "after"), path=f"{path}.affected_aabbs[{index}]")
        bounds: list[tuple[float, ...] | None] = []
        for key in ("before", "after"):
            current = entry[key]
            if current is None:
                bounds.append(None)
            elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)) and len(current) == 6:
                bounds.append(tuple(_finite(item, path=f"{path}.affected_aabbs[{index}].{key}") for item in current))
            else:
                raise _fail("MALFORMED_REQUEST", "AABB bounds require six finite values or null", path=f"{path}.affected_aabbs[{index}].{key}")
        aabbs.append(AABBChange(_entity_key(entry["entity"], path=f"{path}.affected_aabbs[{index}].entity"), bounds[0], bounds[1]))  # type: ignore[arg-type]
    for key in ("feature_history_changed", "document_settings_changed"):
        if not isinstance(made[key], bool):
            raise _fail("MALFORMED_REQUEST", f"{key} must be Boolean", path=f"{path}.{key}")
    return ChangeSet(
        revisions[0], revisions[1],  # type: ignore[arg-type]
        _key_array(made["added"], path=f"{path}.added"),
        _key_array(made["removed"], path=f"{path}.removed"),
        _key_array(made["modified"], path=f"{path}.modified"),
        tuple(replacements),
        _key_array(made["ownership_changes"], path=f"{path}.ownership_changes"),
        _key_array(made["member_changes"], path=f"{path}.member_changes"),
        _key_array(made["attachment_changes"], path=f"{path}.attachment_changes"),
        _string_array(made["group_changes"], path=f"{path}.group_changes"),
        _key_array(made["tag_changes"], path=f"{path}.tag_changes"),
        tuple(aabbs),
        _key_array(made["invalidated_caches"], path=f"{path}.invalidated_caches"),
        _key_array(made["spatial_updates"], path=f"{path}.spatial_updates"),
        made["feature_history_changed"], made["document_settings_changed"],  # type: ignore[arg-type]
    )


@dataclass(frozen=True, slots=True)
class Quantity:
    """A scalar or vector value with an explicit unit and coordinate frame."""

    value: float | tuple[float, ...]
    unit: str
    frame: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.value, Sequence) and not isinstance(self.value, (str, bytes)):
            value: float | tuple[float, ...] = tuple(
                _finite(item, path="$.value") for item in self.value
            )
            if not value or len(value) > 16:
                raise _fail("MALFORMED_QUANTITY", "quantity vectors require 1..16 values")
        else:
            value = _finite(self.value, path="$.value")
        if not isinstance(self.unit, str) or not self.unit:
            raise _fail("UNKNOWN_UNIT", "quantity unit must be explicit")
        if self.frame not in (None, "model_local", "world"):
            raise _fail("UNKNOWN_FRAME", "frame must be model_local or world")
        object.__setattr__(self, "value", value)

    @classmethod
    def from_dict(cls, data: object, *, path: str = "$") -> "Quantity":
        made = _strict(data, ("value", "unit"), ("frame",), path=path)
        return cls(made["value"], made["unit"], made.get("frame"))  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"value": _plain(self.value), "unit": self.unit}
        if self.frame is not None:
            result["frame"] = self.frame
        return result


@dataclass(slots=True)
class AutomationError(Exception):
    """Stable, serializable automation failure.

    Python mutates exception traceback fields while propagating an error, so
    this one protocol record cannot use ``frozen=True``.  Its nested details
    remain immutable and adapters should serialize it rather than modify it.
    """

    code: str
    message: str
    path: str = "$"
    details: Mapping[str, JsonValue] = field(default_factory=dict)
    retryable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("automation error code must be non-empty")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("automation error message must be non-empty")
        if not isinstance(self.path, str) or not self.path.startswith("$"):
            raise ValueError("automation error path must be a JSON path")
        if not isinstance(self.retryable, bool):
            raise ValueError("automation error retryable flag must be Boolean")
        self.details = _json_value(self.details, path="$.details")  # type: ignore[assignment]
        Exception.__init__(self, f"{self.code}: {self.message}")

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "details": _plain(self.details),
            "retryable": self.retryable,
        }

    @classmethod
    def from_dict(cls, data: object, *, path: str = "$") -> "AutomationError":
        made = _strict(data, ("code", "message", "path", "details", "retryable"), path=path)
        return cls(made["code"], made["message"], made["path"], made["details"], made["retryable"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class RequestHeader:
    protocol_version: int
    request_id: str
    model_id: UUID | str
    expected_revision: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "protocol_version", _protocol(self.protocol_version))
        object.__setattr__(self, "request_id", _request_id(self.request_id))
        object.__setattr__(self, "model_id", canonical_model_id(self.model_id))
        if isinstance(self.expected_revision, bool) or not isinstance(self.expected_revision, int) or self.expected_revision < 0:
            raise _fail("MALFORMED_REQUEST", "expected_revision must be a non-negative integer")

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "model_id": str(self.model_id),
            "expected_revision": self.expected_revision,
        }


@dataclass(frozen=True, slots=True)
class EntitySummary:
    handle: EntityHandle
    kind: str
    groups: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    owner: EntityHandle | None = None
    adjacency: tuple[EntityHandle, ...] = ()
    geometry_type: str | None = None
    bounds: tuple[float, ...] | None = None
    measurements: Mapping[str, JsonValue] = field(default_factory=dict)
    replacement_status: str = "active"
    topology: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind != self.handle.kind:
            raise _fail("INTERNAL_ERROR", "summary kind and handle disagree")
        if self.owner is not None and self.owner.model_id != self.handle.model_id:
            raise _fail("INTERNAL_ERROR", "summary owner belongs to another model")
        if any(item.model_id != self.handle.model_id for item in self.adjacency):
            raise _fail("INTERNAL_ERROR", "summary adjacency belongs to another model")
        bounds = None if self.bounds is None else tuple(_finite(v, path="$.bounds") for v in self.bounds)
        if bounds is not None and len(bounds) != 6:
            raise _fail("INTERNAL_ERROR", "bounds require six values")
        object.__setattr__(self, "groups", tuple(sorted(set(self.groups))))
        object.__setattr__(self, "tags", tuple(sorted(set(self.tags))))
        object.__setattr__(self, "adjacency", tuple(sorted(set(self.adjacency))))
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "measurements", _json_value(self.measurements, path="$.measurements"))
        object.__setattr__(self, "topology", _json_value(self.topology, path="$.topology"))

    def to_dict(self) -> dict[str, object]:
        return {
            "handle": handle_to_dict(self.handle),
            "kind": self.kind,
            "groups": list(self.groups),
            "tags": list(self.tags),
            "owner": None if self.owner is None else handle_to_dict(self.owner),
            "adjacency": [handle_to_dict(item) for item in self.adjacency],
            "geometry_type": self.geometry_type,
            "bounds": None if self.bounds is None else list(self.bounds),
            "measurements": _plain(self.measurements),
            "replacement_status": self.replacement_status,
            "topology": _plain(self.topology),
        }

    @classmethod
    def from_dict(cls, data: object, *, path: str = "$") -> "EntitySummary":
        made = _strict(
            data,
            (
                "handle", "kind", "groups", "tags", "owner", "adjacency",
                "geometry_type", "bounds", "measurements",
                "replacement_status", "topology",
            ),
            path=path,
        )
        if made["geometry_type"] is not None and not isinstance(made["geometry_type"], str):
            raise _fail("MALFORMED_REQUEST", "geometry_type must be a string or null", path=f"{path}.geometry_type")
        if not isinstance(made["kind"], str) or not isinstance(made["replacement_status"], str):
            raise _fail("MALFORMED_REQUEST", "summary kind/status must be strings", path=path)
        return cls(
            handle_from_dict(made["handle"], path=f"{path}.handle"),
            made["kind"],
            _string_array(made["groups"], path=f"{path}.groups"),
            _string_array(made["tags"], path=f"{path}.tags"),
            None if made["owner"] is None else handle_from_dict(made["owner"], path=f"{path}.owner"),
            _handle_array(made["adjacency"], path=f"{path}.adjacency"),
            made["geometry_type"],
            None if made["bounds"] is None else tuple(made["bounds"]),  # type: ignore[arg-type]
            made["measurements"],
            made["replacement_status"],
            made["topology"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class SelectionSpec(RequestHeader):
    where: Mapping[str, JsonValue] = field(default_factory=dict)
    order_by: str = "handle"
    descending: bool = False
    page_size: int = DEFAULT_PAGE_SIZE
    cursor: str | None = None
    expected_cardinality: tuple[int, int] | None = None
    detail: bool = False

    def __post_init__(self) -> None:
        RequestHeader.__post_init__(self)
        object.__setattr__(self, "where", _json_value(self.where, path="$.where"))
        if not isinstance(self.descending, bool) or not isinstance(self.detail, bool):
            raise _fail("MALFORMED_REQUEST", "descending and detail must be Boolean")
        if self.order_by not in ("handle", "centroid", "bounds", "length", "area", "radius", "distance"):
            raise _fail("UNSUPPORTED", f"unsupported order_by {self.order_by!r}", path="$.order_by")
        if isinstance(self.page_size, bool) or not isinstance(self.page_size, int) or not 1 <= self.page_size <= MAX_PAGE_SIZE:
            raise _fail("PAYLOAD_TOO_LARGE", f"page_size must be 1..{MAX_PAGE_SIZE}", path="$.page_size")
        if self.cursor is not None and (not isinstance(self.cursor, str) or len(self.cursor) > 512):
            raise _fail("STALE_CURSOR", "cursor must be a bounded string", path="$.cursor")
        if self.expected_cardinality is not None:
            try:
                low, high = self.expected_cardinality
            except (TypeError, ValueError) as error:
                raise _fail("MALFORMED_REQUEST", "expected_cardinality must be [min,max]") from error
            if any(isinstance(v, bool) or not isinstance(v, int) for v in (low, high)) or low < 0 or high < low or high > MAX_PAGE_SIZE:
                raise _fail("MALFORMED_REQUEST", "invalid expected cardinality bounds")
            object.__setattr__(self, "expected_cardinality", (low, high))

    @classmethod
    def from_dict(cls, data: object) -> "SelectionSpec":
        made = _strict(
            data,
            ("protocol_version", "request_id", "model_id", "expected_revision", "where"),
            ("order_by", "descending", "page_size", "cursor", "expected_cardinality", "detail"),
        )
        cardinality = made.get("expected_cardinality")
        for key in ("descending", "detail"):
            if key in made and not isinstance(made[key], bool):
                raise _fail("MALFORMED_REQUEST", f"{key} must be Boolean", path=f"$.{key}")
        return cls(
            made["protocol_version"], made["request_id"], made["model_id"], made["expected_revision"],
            made["where"], made.get("order_by", "handle"), made.get("descending", False),
            made.get("page_size", DEFAULT_PAGE_SIZE), made.get("cursor"),
            None if cardinality is None else tuple(cardinality), made.get("detail", False),  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **RequestHeader.to_dict(self),
            "where": _plain(self.where),
            "order_by": self.order_by,
            "descending": self.descending,
            "page_size": self.page_size,
            "cursor": self.cursor,
            "expected_cardinality": None if self.expected_cardinality is None else list(self.expected_cardinality),
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class SelectionResult:
    protocol_version: int
    request_id: str
    model_id: UUID | str
    revision: int
    entities: tuple[EntitySummary, ...]
    total: int
    next_cursor: str | None = None
    evidence: Mapping[str, JsonValue] = field(default_factory=dict)
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "protocol_version", _protocol(self.protocol_version))
        object.__setattr__(self, "request_id", _request_id(self.request_id))
        model_id = canonical_model_id(self.model_id)
        object.__setattr__(self, "model_id", model_id)
        entities = tuple(self.entities)
        if any(item.handle.model_id != model_id for item in entities):
            raise _fail("INTERNAL_ERROR", "selection returned a foreign handle")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (self.revision, self.total)):
            raise _fail("INTERNAL_ERROR", "selection revision and total must be non-negative integers")
        if self.next_cursor is not None and (not isinstance(self.next_cursor, str) or len(self.next_cursor) > 512):
            raise _fail("INTERNAL_ERROR", "selection cursor must be bounded")
        if any(not isinstance(item, str) for item in self.diagnostics):
            raise _fail("INTERNAL_ERROR", "selection diagnostics must be strings")
        object.__setattr__(self, "entities", entities)
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))
        object.__setattr__(self, "evidence", _json_value(self.evidence, path="$.evidence"))

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "model_id": str(self.model_id),
            "revision": self.revision,
            "entities": [item.to_dict() for item in self.entities],
            "total": self.total,
            "next_cursor": self.next_cursor,
            "evidence": _plain(self.evidence),
            "diagnostics": list(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, data: object) -> "SelectionResult":
        made = _strict(
            data,
            (
                "protocol_version", "request_id", "model_id", "revision",
                "entities", "total", "next_cursor", "evidence", "diagnostics",
            ),
        )
        raw_entities = made["entities"]
        if not isinstance(raw_entities, Sequence) or isinstance(raw_entities, (str, bytes)):
            raise _fail("MALFORMED_REQUEST", "selection entities must be an array", path="$.entities")
        return cls(
            made["protocol_version"], made["request_id"], made["model_id"], made["revision"],  # type: ignore[arg-type]
            tuple(EntitySummary.from_dict(item, path=f"$.entities[{index}]") for index, item in enumerate(raw_entities)),
            made["total"], made["next_cursor"], made["evidence"],  # type: ignore[arg-type]
            _string_array(made["diagnostics"], path="$.diagnostics"),
        )


@dataclass(frozen=True, slots=True)
class Command:
    name: str
    operation: str
    arguments: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or len(self.name) > 64 or not self.name.replace("_", "a").isalnum() or self.name[0].isdigit():
            raise _fail("MALFORMED_COMMAND", "command name must be a bounded identifier")
        if not isinstance(self.operation, str) or not self.operation:
            raise _fail("MALFORMED_COMMAND", "command operation must be non-empty")
        object.__setattr__(self, "arguments", _json_value(self.arguments, path=f"$.commands.{self.name}.arguments"))

    @classmethod
    def from_dict(cls, data: object, *, path: str = "$") -> "Command":
        made = _strict(data, ("name", "operation", "arguments"), path=path)
        return cls(made["name"], made["operation"], made["arguments"])  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "operation": self.operation, "arguments": _plain(self.arguments)}


@dataclass(frozen=True, slots=True)
class CommandBatch(RequestHeader):
    commands: tuple[Command, ...] = ()

    def __post_init__(self) -> None:
        RequestHeader.__post_init__(self)
        commands = tuple(self.commands)
        if not commands or len(commands) > MAX_COMMANDS:
            raise _fail("PAYLOAD_TOO_LARGE", f"a batch requires 1..{MAX_COMMANDS} commands")
        if len({item.name for item in commands}) != len(commands):
            raise _fail("DUPLICATE_NAME", "command names must be unique")
        object.__setattr__(self, "commands", commands)

    @classmethod
    def from_dict(cls, data: object) -> "CommandBatch":
        made = _strict(data, ("protocol_version", "request_id", "model_id", "expected_revision", "commands"))
        raw = made["commands"]
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise _fail("MALFORMED_REQUEST", "commands must be an array", path="$.commands")
        commands = tuple(Command.from_dict(item, path=f"$.commands[{index}]") for index, item in enumerate(raw))
        return cls(made["protocol_version"], made["request_id"], made["model_id"], made["expected_revision"], commands)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return {**RequestHeader.to_dict(self), "commands": [item.to_dict() for item in self.commands]}


@dataclass(frozen=True, slots=True)
class EditPlan:
    protocol_version: int
    request_id: str
    model_id: UUID | str
    revision: int
    commands: tuple[Command, ...]
    resolved_inputs: Mapping[str, tuple[EntityHandle, ...]]
    expected_outputs: Mapping[str, tuple[str, ...]]
    expected_entity_counts: Mapping[str, int]
    expected_owners: Mapping[str, tuple[str, ...]]
    affected_bounds: tuple[float, ...] | None
    operation_policies: Mapping[str, JsonValue]
    diagnostics: tuple[str, ...]
    required_capabilities: tuple[str, ...]
    digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "protocol_version", _protocol(self.protocol_version))
        object.__setattr__(self, "request_id", _request_id(self.request_id))
        model_id = canonical_model_id(self.model_id)
        object.__setattr__(self, "model_id", model_id)
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 0:
            raise _fail("MALFORMED_PLAN", "plan revision must be a non-negative integer")
        commands = tuple(self.commands)
        if not commands or len(commands) > MAX_COMMANDS or any(not isinstance(item, Command) for item in commands):
            raise _fail("MALFORMED_PLAN", f"plan requires 1..{MAX_COMMANDS} commands")
        object.__setattr__(self, "commands", commands)
        resolved = {str(key): tuple(value) for key, value in self.resolved_inputs.items()}
        if any(not isinstance(handle, EntityHandle) or handle.model_id != model_id for values in resolved.values() for handle in values):
            raise _fail("INTERNAL_ERROR", "plan contains a foreign handle")
        object.__setattr__(self, "resolved_inputs", MappingProxyType(dict(sorted(resolved.items()))))
        if any(not isinstance(value, Sequence) or isinstance(value, (str, bytes)) for value in self.expected_outputs.values()):
            raise _fail("MALFORMED_PLAN", "expected output values must be arrays")
        if any(not isinstance(value, Sequence) or isinstance(value, (str, bytes)) for value in self.expected_owners.values()):
            raise _fail("MALFORMED_PLAN", "expected owner values must be arrays")
        expected_outputs = {str(k): tuple(v) for k, v in sorted(self.expected_outputs.items())}
        expected_owners = {str(k): tuple(v) for k, v in sorted(self.expected_owners.items())}
        if any(not isinstance(item, str) or not item for values in expected_outputs.values() for item in values):
            raise _fail("MALFORMED_PLAN", "expected output ports must be non-empty strings")
        if any(not isinstance(item, str) or not item for values in expected_owners.values() for item in values):
            raise _fail("MALFORMED_PLAN", "expected owners must be non-empty symbolic references")
        counts: dict[str, int] = {}
        for key, value in self.expected_entity_counts.items():
            if not isinstance(key, str) or not key or isinstance(value, bool) or not isinstance(value, int):
                raise _fail("MALFORMED_PLAN", "expected entity count deltas must be integers")
            counts[key] = value
        bounds = None if self.affected_bounds is None else tuple(_finite(item, path="$.affected_bounds") for item in self.affected_bounds)
        if bounds is not None and len(bounds) != 6:
            raise _fail("MALFORMED_PLAN", "affected_bounds requires six finite values")
        diagnostics = tuple(self.diagnostics)
        capabilities = tuple(self.required_capabilities)
        if any(not isinstance(item, str) for item in (*diagnostics, *capabilities)):
            raise _fail("MALFORMED_PLAN", "plan diagnostics and capabilities must be strings")
        object.__setattr__(self, "expected_outputs", MappingProxyType(expected_outputs))
        object.__setattr__(self, "expected_entity_counts", MappingProxyType(dict(sorted(counts.items()))))
        object.__setattr__(self, "expected_owners", MappingProxyType(expected_owners))
        object.__setattr__(self, "affected_bounds", bounds)
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "required_capabilities", capabilities)
        object.__setattr__(self, "operation_policies", _json_value(self.operation_policies, path="$.operation_policies"))
        if len(self.digest) != 64 or any(ch not in "0123456789abcdef" for ch in self.digest):
            raise _fail("MALFORMED_PLAN", "plan digest must be canonical SHA-256")

    def digest_payload(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "model_id": str(self.model_id),
            "revision": self.revision,
            "commands": [item.to_dict() for item in self.commands],
            "resolved_inputs": {key: [handle_to_dict(item) for item in value] for key, value in self.resolved_inputs.items()},
            "expected_outputs": {key: list(value) for key, value in self.expected_outputs.items()},
            "expected_entity_counts": dict(self.expected_entity_counts),
            "expected_owners": {key: list(value) for key, value in self.expected_owners.items()},
            "affected_bounds": None if self.affected_bounds is None else list(self.affected_bounds),
            "operation_policies": _plain(self.operation_policies),
            "diagnostics": list(self.diagnostics),
            "required_capabilities": list(self.required_capabilities),
        }

    @classmethod
    def from_dict(cls, data: object) -> "EditPlan":
        made = _strict(
            data,
            (
                "protocol_version", "request_id", "model_id", "revision", "commands",
                "resolved_inputs", "expected_outputs", "expected_entity_counts",
                "expected_owners", "affected_bounds", "operation_policies", "diagnostics",
                "required_capabilities", "digest",
            ),
        )
        commands_raw = made["commands"]
        if not isinstance(commands_raw, Sequence) or isinstance(commands_raw, (str, bytes)):
            raise _fail("MALFORMED_PLAN", "plan commands must be an array")
        resolved_raw = made["resolved_inputs"]
        if not isinstance(resolved_raw, Mapping):
            raise _fail("MALFORMED_PLAN", "resolved_inputs must be an object")
        resolved: dict[str, tuple[EntityHandle, ...]] = {}
        for key, values in resolved_raw.items():
            if not isinstance(key, str) or not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise _fail("MALFORMED_PLAN", "resolved_inputs entries must be handle arrays")
            resolved[key] = tuple(handle_from_dict(item, path=f"$.resolved_inputs.{key}") for item in values)
        expected_outputs = made["expected_outputs"]
        expected_counts = made["expected_entity_counts"]
        expected_owners = made["expected_owners"]
        if not isinstance(expected_outputs, Mapping) or not isinstance(expected_counts, Mapping) or not isinstance(expected_owners, Mapping):
            raise _fail("MALFORMED_PLAN", "plan output/count/owner fields must be objects")
        return cls(
            made["protocol_version"], made["request_id"], made["model_id"], made["revision"],  # type: ignore[arg-type]
            tuple(Command.from_dict(item, path=f"$.commands[{index}]") for index, item in enumerate(commands_raw)),
            resolved,
            {str(key): tuple(value) for key, value in expected_outputs.items()},  # type: ignore[arg-type]
            dict(expected_counts),
            {str(key): tuple(value) for key, value in expected_owners.items()},  # type: ignore[arg-type]
            None if made["affected_bounds"] is None else tuple(made["affected_bounds"]),  # type: ignore[arg-type]
            made["operation_policies"], tuple(made["diagnostics"]), tuple(made["required_capabilities"]), made["digest"],  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, object]:
        return {**self.digest_payload(), "digest": self.digest}


@dataclass(frozen=True, slots=True)
class ApplyResult:
    protocol_version: int
    request_id: str
    model_id: UUID | str
    revision_before: int
    revision_after: int
    plan_digest: str
    outputs: Mapping[str, tuple[EntityHandle, ...]]
    replacements: Mapping[str, tuple[EntityHandle, ...]]
    change_set: ChangeSet
    changed_region_audit: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "protocol_version", _protocol(self.protocol_version))
        object.__setattr__(self, "request_id", _request_id(self.request_id))
        model_id = canonical_model_id(self.model_id)
        object.__setattr__(self, "model_id", model_id)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (self.revision_before, self.revision_after)):
            raise _fail("INTERNAL_ERROR", "apply revisions must be non-negative integers")
        if self.revision_after <= self.revision_before:
            raise _fail("INTERNAL_ERROR", "successful apply must advance the revision")
        if len(self.plan_digest) != 64 or any(ch not in "0123456789abcdef" for ch in self.plan_digest):
            raise _fail("INTERNAL_ERROR", "apply result plan digest is invalid")
        outputs = {str(k): tuple(v) for k, v in self.outputs.items()}
        replacements = {str(k): tuple(v) for k, v in self.replacements.items()}
        if any(h.model_id != model_id for values in (*outputs.values(), *replacements.values()) for h in values):
            raise _fail("INTERNAL_ERROR", "apply result contains a foreign handle")
        object.__setattr__(self, "outputs", MappingProxyType(dict(sorted(outputs.items()))))
        object.__setattr__(self, "replacements", MappingProxyType(dict(sorted(replacements.items()))))
        object.__setattr__(self, "changed_region_audit", _json_value(self.changed_region_audit, path="$.changed_region_audit"))

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "model_id": str(self.model_id),
            "revision_before": self.revision_before,
            "revision_after": self.revision_after,
            "plan_digest": self.plan_digest,
            "outputs": {key: [handle_to_dict(item) for item in value] for key, value in self.outputs.items()},
            "replacements": {key: [handle_to_dict(item) for item in value] for key, value in self.replacements.items()},
            "change_set": _change_set_to_dict(self.change_set),
            "changed_region_audit": _plain(self.changed_region_audit),
        }

    @classmethod
    def from_dict(cls, data: object) -> "ApplyResult":
        made = _strict(
            data,
            (
                "protocol_version", "request_id", "model_id", "revision_before",
                "revision_after", "plan_digest", "outputs", "replacements",
                "change_set", "changed_region_audit",
            ),
        )
        mappings: list[dict[str, tuple[EntityHandle, ...]]] = []
        for key in ("outputs", "replacements"):
            raw = made[key]
            if not isinstance(raw, Mapping):
                raise _fail("MALFORMED_REQUEST", f"{key} must be an object", path=f"$.{key}")
            mappings.append({
                str(name): _handle_array(values, path=f"$.{key}.{name}")
                for name, values in raw.items()
            })
        return cls(
            made["protocol_version"], made["request_id"], made["model_id"],  # type: ignore[arg-type]
            made["revision_before"], made["revision_after"], made["plan_digest"],  # type: ignore[arg-type]
            mappings[0], mappings[1],
            _change_set_from_dict(made["change_set"], path="$.change_set"),
            made["changed_region_audit"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class AutomationResponse:
    protocol_version: int
    request_id: str
    ok: bool
    result: JsonValue = None
    error: AutomationError | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "protocol_version", _protocol(self.protocol_version))
        object.__setattr__(self, "request_id", _request_id(self.request_id))
        if self.ok == (self.error is not None):
            raise ValueError("successful responses cannot contain errors and failures require one")
        object.__setattr__(self, "result", _json_value(self.result, path="$.result"))

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "ok": self.ok,
            "result": _plain(self.result) if self.ok else None,
            "error": None if self.error is None else self.error.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: object) -> "AutomationResponse":
        made = _strict(data, ("protocol_version", "request_id", "ok", "result", "error"))
        if not isinstance(made["ok"], bool):
            raise _fail("MALFORMED_REQUEST", "response ok must be Boolean", path="$.ok")
        error = None if made["error"] is None else AutomationError.from_dict(made["error"], path="$.error")
        return cls(made["protocol_version"], made["request_id"], made["ok"], made["result"], error)  # type: ignore[arg-type]


__all__ = [
    "ApplyResult", "AutomationError", "AutomationResponse", "Command", "CommandBatch",
    "DEFAULT_PAGE_SIZE", "EditPlan", "EntitySummary", "MAX_COMMANDS", "MAX_PAGE_SIZE",
    "PROTOCOL_VERSION", "Quantity", "RequestHeader", "SelectionResult", "SelectionSpec",
    "canonical_digest", "canonical_json", "handle_from_dict", "handle_to_dict",
]
