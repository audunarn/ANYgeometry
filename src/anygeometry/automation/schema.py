"""Machine-readable discovery for automation protocol version 1."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Mapping

from .types import (
    AutomationError,
    CommandBatch,
    PROTOCOL_VERSION,
    RequestHeader,
    SelectionSpec,
    canonical_json,
)


_QUANTITY = {
    "type": "object",
    "additionalProperties": False,
    "required": ["value", "unit"],
    "properties": {
        "value": {"oneOf": [{"type": "number"}, {"type": "array", "items": {"type": "number"}, "minItems": 1, "maxItems": 16}]},
        "unit": {"enum": ["1", "m", "mm", "cm", "in", "ft", "m^2", "mm^2", "cm^2", "in^2", "ft^2", "rad", "deg"]},
        "frame": {"enum": ["model_local", "world"]},
    },
}


def automation_json_schema() -> Mapping[str, object]:
    """Return a dependency-free JSON Schema for transport validation."""

    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://github.com/audunarn/ANYgeometry/schemas/automation-v1.json",
        "title": "ANYgeometry Automation Protocol",
        "type": "object",
        "oneOf": [{"$ref": "#/$defs/selection"}, {"$ref": "#/$defs/batch"}, {"$ref": "#/$defs/query"}],
        "$defs": {
            "quantity": _QUANTITY,
            "handle": {
                "type": "object", "additionalProperties": False,
                "required": ["model_id", "kind", "id"],
                "properties": {
                    "model_id": {"type": "string", "format": "uuid"},
                    "kind": {"enum": ["vertex", "edge", "face", "part", "sheet", "face_use", "coedge", "member", "member_edge_use", "attachment", "junction"]},
                    "id": {"type": "integer", "minimum": 1},
                },
            },
            "header": {
                "type": "object",
                "properties": {
                    "protocol_version": {"const": PROTOCOL_VERSION},
                    "request_id": {"type": "string", "minLength": 1, "maxLength": 128},
                    "model_id": {"type": "string", "format": "uuid"},
                    "expected_revision": {"type": "integer", "minimum": 0},
                },
                "required": ["protocol_version", "request_id", "model_id", "expected_revision"],
            },
            "selection": {
                "allOf": [{"$ref": "#/$defs/header"}],
                "type": "object", "additionalProperties": False,
                "required": ["protocol_version", "request_id", "model_id", "expected_revision", "where"],
                "properties": {
                    "protocol_version": {"const": PROTOCOL_VERSION}, "request_id": {"type": "string"}, "model_id": {"type": "string"}, "expected_revision": {"type": "integer"},
                    "where": {"type": "object", "minProperties": 1, "maxProperties": 1},
                    "order_by": {"enum": ["handle", "centroid", "bounds", "length", "area", "radius", "distance"]},
                    "descending": {"type": "boolean"}, "page_size": {"type": "integer", "minimum": 1, "maximum": 1000},
                    "cursor": {"type": ["string", "null"], "maxLength": 512},
                    "expected_cardinality": {"type": ["array", "null"], "prefixItems": [{"type": "integer", "minimum": 0}, {"type": "integer", "minimum": 0}], "minItems": 2, "maxItems": 2},
                    "detail": {"type": "boolean"},
                },
            },
            "command": {
                "type": "object", "additionalProperties": False,
                "required": ["name", "operation", "arguments"],
                "properties": {
                    "name": {"type": "string", "pattern": "^[A-Za-z_][A-Za-z0-9_]{0,63}$"},
                    "operation": {"enum": ["create_point", "create_edge", "create_face", "create_plate", "translate", "rotate", "move", "copy", "mirror", "pattern", "group", "tag", "delete", "imprint"]},
                    "arguments": {"type": "object", "maxProperties": 16},
                },
            },
            "batch": {
                "allOf": [{"$ref": "#/$defs/header"}], "type": "object", "additionalProperties": False,
                "required": ["protocol_version", "request_id", "model_id", "expected_revision", "commands"],
                "properties": {
                    "protocol_version": {"const": PROTOCOL_VERSION}, "request_id": {"type": "string"}, "model_id": {"type": "string"}, "expected_revision": {"type": "integer"},
                    "commands": {"type": "array", "items": {"$ref": "#/$defs/command"}, "minItems": 1, "maxItems": 256},
                },
            },
            "query": {
                "allOf": [{"$ref": "#/$defs/header"}], "type": "object", "additionalProperties": False,
                "required": ["protocol_version", "request_id", "model_id", "expected_revision", "operation", "arguments"],
                "properties": {
                    "protocol_version": {"const": PROTOCOL_VERSION}, "request_id": {"type": "string"}, "model_id": {"type": "string"}, "expected_revision": {"type": "integer"},
                    "operation": {"enum": ["measure", "intersection"]}, "arguments": {"type": "object"},
                },
            },
        },
    }
    return deepcopy(schema)


def tool_catalog() -> tuple[Mapping[str, object], ...]:
    """Return stable provider-neutral tool descriptions."""

    common = {"protocol_version": PROTOCOL_VERSION, "strict": True}
    return (
        {**common, "name": "kernel_capabilities", "mutating": False, "description": "Discover protocol limits, command kinds, units, and typed errors."},
        {**common, "name": "model_summary", "mutating": False, "description": "Describe the bound model UUID, revision, bounds, units, groups, and counts."},
        {**common, "name": "select_entities", "mutating": False, "description": "Run one bounded deterministic selector and return canonical handles."},
        {**common, "name": "describe_entities", "mutating": False, "description": "Inspect active canonical handles with bounded topology detail."},
        {**common, "name": "query_geometry", "mutating": False, "description": "Measure or run a qualified intersection query."},
        {**common, "name": "plan_edit", "mutating": False, "description": "Freeze a revision-bound atomic command batch without allocating IDs."},
        {**common, "name": "apply_edit", "mutating": True, "description": "Apply one verified plan exactly once through owner transactions."},
    )


def describe_capabilities() -> Mapping[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "geometry_schema_version": 4,
        "provider_neutral": True,
        "dependency_free": True,
        "entity_aliases": {"point": "vertex", "plate": "sheet"},
        "units": {"length": ["m", "mm", "cm", "in", "ft"], "area": ["m^2", "mm^2", "cm^2", "in^2", "ft^2"], "angle": ["rad", "deg"]},
        "frames": ["model_local", "world"],
        "limits": {"boolean_depth": 8, "predicate_count": 64, "default_page_size": 100, "maximum_results": 1000, "maximum_commands": 256},
        "selection_operators": ["kind", "handle", "handles", "group", "tag", "owner", "incident_to", "boundary_of", "connected_to", "curve_type", "support_type", "metadata", "aabb", "centroid_axis", "nearest", "length", "area", "radius", "all", "any", "not"],
        "query_operations": ["measure", "intersection"],
        "command_operations": ["create_point", "create_edge", "create_face", "create_plate", "translate", "rotate", "move", "copy", "mirror", "pattern", "group", "tag", "delete", "imprint"],
        "error_codes": ["WRONG_MODEL", "STALE_REVISION", "AMBIGUOUS_SELECTION", "CARDINALITY_MISMATCH", "UNSUPPORTED", "CAPABILITY_MISSING", "STALE_PLAN", "TAMPERED_PLAN", "UNKNOWN_UNIT", "UNKNOWN_FRAME", "STALE_CURSOR"],
        "tools": list(tool_catalog()),
    }


def automation_dumps(value: object) -> str:
    """Encode a protocol value with canonical deterministic JSON."""

    return canonical_json(value)


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def automation_loads(kind: str, payload: str) -> SelectionSpec | CommandBatch | Mapping[str, object]:
    """Decode strict JSON into a request record without executing it."""

    if not isinstance(payload, str) or len(payload.encode("utf-8")) > 1_048_576:
        raise ValueError("automation JSON must be a string no larger than 1 MiB")
    data = json.loads(payload, object_pairs_hook=_reject_duplicates, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON number {value}")))
    if kind == "selection":
        return SelectionSpec.from_dict(data)
    if kind == "batch":
        return CommandBatch.from_dict(data)
    if kind == "query":
        required = {"protocol_version", "request_id", "model_id", "expected_revision", "operation", "arguments"}
        if not isinstance(data, Mapping) or set(data) != required:
            extra = sorted(set(data) - required) if isinstance(data, Mapping) else []
            code = "UNKNOWN_FIELD" if extra else "MALFORMED_REQUEST"
            raise AutomationError(code, "query requires exactly protocol header, operation, and arguments", details={"extra": extra})
        RequestHeader(data["protocol_version"], data["request_id"], data["model_id"], data["expected_revision"])  # type: ignore[arg-type]
        if data["operation"] not in ("measure", "intersection") or not isinstance(data["arguments"], Mapping):
            raise AutomationError("UNSUPPORTED", "query operation must be measure or intersection with object arguments")
        return data
    raise ValueError(f"unknown automation JSON kind {kind!r}")


__all__ = ["automation_dumps", "automation_json_schema", "automation_loads", "describe_capabilities", "tool_catalog"]
