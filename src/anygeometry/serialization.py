"""Versioned JSON-ready serialization owned by the geometry package."""

from __future__ import annotations

import gzip
import hashlib
import hmac
import json
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping
from uuid import UUID

import numpy as np

from .curves import Arc, Spline, Straight
from .entities import Edge, EntityRef, Face, OrientedEdge, Vertex
from .errors import GeometryError
from .features import (
    FeatureHistory,
    FeatureOutputRef,
    FeatureRecord,
    FeatureStatus,
    _frozen_feature_diagnostic,
    builtin_feature_registry,
)
from .model import GeometryModel
from .structural import (
    Attachment,
    AttachmentKind,
    AttachmentTargetKind,
    BoundaryPolicy,
    Coedge,
    ConnectivityPolicy,
    FaceUse,
    Junction,
    JunctionKind,
    JunctionMemberUse,
    Member,
    MemberEdgeUse,
    NonManifoldPolicy,
    Orientation,
    ParameterRange,
    Part,
    Sheet,
    SheetTopologyPolicy,
)
from .surfaces import CoonsSurface, Cone, Cylinder, Plane, RuledSurface
from .tolerance import TolerancePolicy

SCHEMA = "anygeometry"
VERSION = 3

_GEOMETRY_KINDS = ("vertex", "edge", "face")
_STRUCTURAL_KINDS = (
    "part",
    "sheet",
    "face_use",
    "coedge",
    "member",
    "member_edge_use",
    "attachment",
    "junction",
)
_ID_KINDS = _GEOMETRY_KINDS + _STRUCTURAL_KINDS
_CURRENT_REQUIRED_FIELDS = {
    "schema",
    "version",
    "model_id",
    "revision",
    "coordinates",
    "tolerance",
    "id_state",
    "vertices",
    "edges",
    "faces",
    "structural",
    "groups",
    "tags",
    "replacement_history",
    "extensions",
    "checksum",
}
_CURRENT_OPTIONAL_FIELDS = {"features"}

__all__ = ["SCHEMA", "VERSION", "from_dict", "read_geometry", "to_dict", "write_geometry"]


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise GeometryError(f"{name} must be an integer")
    return int(value)


def _non_negative_integer(value: object, name: str) -> int:
    made = _integer(value, name)
    if made < 0:
        raise GeometryError(f"{name} must be non-negative")
    return made


def _positive_integer(value: object, name: str) -> int:
    made = _integer(value, name)
    if made <= 0:
        raise GeometryError(f"{name} must be positive")
    return made


def _object(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise GeometryError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise GeometryError(f"{name} keys must be strings")
    return value


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise GeometryError(f"{name} must be a list")
    return value


def _exact_fields(
    value: Mapping[str, object],
    *,
    required: set[str],
    optional: set[str] | None = None,
    name: str,
) -> None:
    optional = set() if optional is None else optional
    present = set(value)
    missing = required - present
    unexpected = present - required - optional
    if missing:
        raise GeometryError(f"{name} is missing required field(s): {', '.join(sorted(missing))}")
    if unexpected:
        raise GeometryError(f"{name} has unexpected field(s): {', '.join(sorted(unexpected))}")


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise GeometryError(f"{name} must be a boolean")
    return value


def _finite_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        made = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise GeometryError(f"{name} must be a finite array with shape {shape}") from error
    if made.shape != shape or not np.all(np.isfinite(made)):
        raise GeometryError(f"{name} must be a finite array with shape {shape}")
    return made


def _canonical_payload(document: Mapping[str, object]) -> bytes:
    payload = {key: value for key, value in document.items() if key != "checksum"}
    try:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise GeometryError(f"geometry document is not canonical JSON: {error}") from error
    return text.encode("utf-8")


def _checksum(document: Mapping[str, object]) -> dict[str, str]:
    return {
        "algorithm": "sha256",
        "value": hashlib.sha256(_canonical_payload(document)).hexdigest(),
    }


def _verify_checksum(document: Mapping[str, object]) -> None:
    value = _object(document.get("checksum"), "checksum")
    _exact_fields(value, required={"algorithm", "value"}, name="checksum")
    if value["algorithm"] != "sha256":
        raise GeometryError("checksum algorithm must be 'sha256'")
    digest = value["value"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise GeometryError("checksum value must be a lowercase SHA-256 digest")
    expected = hashlib.sha256(_canonical_payload(document)).hexdigest()
    if not hmac.compare_digest(digest, expected):
        raise GeometryError("geometry document checksum mismatch")


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    made: dict[str, object] = {}
    for key, value in pairs:
        if key in made:
            raise GeometryError(f"duplicate JSON object key {key!r}")
        made[key] = value
    return made


def _unique_ids(records: list[object], name: str) -> None:
    found: set[int] = set()
    for position, raw in enumerate(records):
        item = _object(raw, f"{name} record {position}")
        identifier = _positive_integer(item.get("id"), f"{name} ID")
        if identifier in found:
            raise GeometryError(f"duplicate {name} ID {identifier}")
        found.add(identifier)


def _ref(reference: EntityRef) -> list[object]:
    return [reference.kind, int(reference.id)]


def _loop(loop: tuple[OrientedEdge, ...]) -> list[list[object]]:
    return [[item.edge, item.forward] for item in loop]


def _json_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise GeometryError("JSON object keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise GeometryError("metadata floats must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise GeometryError(
        f"metadata value of type {type(value).__name__} is not JSON serializable"
    )


def _metadata(value: object, name: str = "metadata") -> dict[str, object]:
    data = _object(value, name)
    if any(not key for key in data):
        raise GeometryError(f"{name} keys cannot be empty")
    made = _json_value(data)
    assert isinstance(made, dict)
    return made


def _parameter_range_value(value: ParameterRange) -> list[float]:
    return [value.start, value.end]


def _orientation_value(value: Orientation) -> str:
    return "forward" if value is Orientation.FORWARD else "reversed"


def _decode_orientation(value: object, name: str) -> Orientation:
    if value == "forward":
        return Orientation.FORWARD
    if value == "reversed":
        return Orientation.REVERSED
    raise GeometryError(f"{name} must be 'forward' or 'reversed'")


def _decode_parameter_range(value: object, name: str) -> ParameterRange:
    raw = _list(value, name)
    if len(raw) != 2:
        raise GeometryError(f"{name} must contain exactly two parameters")
    return ParameterRange(raw[0], raw[1])  # type: ignore[arg-type]


def _ids(value: object, name: str) -> tuple[int, ...]:
    return tuple(_positive_integer(item, name) for item in _list(value, f"{name}s"))


def _structural_document(geometry: GeometryModel) -> dict[str, object]:
    return {
        "parts": [
            {
                "id": item.id,
                "sheet_ids": list(item.sheet_ids),
                "member_ids": list(item.member_ids),
                "name": item.name,
                "metadata": item.metadata.to_dict(),
            }
            for item in sorted(geometry.parts.values(), key=lambda record: record.id)
        ],
        "sheets": [
            {
                "id": item.id,
                "part_id": item.part_id,
                "face_use_ids": list(item.face_use_ids),
                "policy": {
                    "boundary": item.policy.boundary.value,
                    "non_manifold": item.policy.non_manifold.value,
                    "connectivity": item.policy.connectivity.value,
                },
                "declared_non_manifold_edges": list(item.declared_non_manifold_edges),
                "name": item.name,
                "metadata": item.metadata.to_dict(),
            }
            for item in sorted(geometry.sheets.values(), key=lambda record: record.id)
        ],
        "face_uses": [
            {
                "id": item.id,
                "sheet_id": item.sheet_id,
                "face_id": item.face_id,
                "loops": [list(loop) for loop in item.loops],
                "orientation": _orientation_value(item.orientation),
                "metadata": item.metadata.to_dict(),
            }
            for item in sorted(geometry.face_uses.values(), key=lambda record: record.id)
        ],
        "coedges": [
            {
                "id": item.id,
                "face_use_id": item.face_use_id,
                "edge_id": item.edge_id,
                "orientation": _orientation_value(item.orientation),
                "metadata": item.metadata.to_dict(),
            }
            for item in sorted(geometry.coedges.values(), key=lambda record: record.id)
        ],
        "members": [
            {
                "id": item.id,
                "part_id": item.part_id,
                "edge_use_ids": list(item.edge_use_ids),
                "name": item.name,
                "metadata": item.metadata.to_dict(),
            }
            for item in sorted(geometry.members.values(), key=lambda record: record.id)
        ],
        "member_edge_uses": [
            {
                "id": item.id,
                "member_id": item.member_id,
                "edge_id": item.edge_id,
                "parent_range": _parameter_range_value(item.parent_range),
                "orientation": _orientation_value(item.orientation),
                "metadata": item.metadata.to_dict(),
            }
            for item in sorted(
                geometry.member_edge_uses.values(), key=lambda record: record.id
            )
        ],
        "attachments": [
            {
                "id": item.id,
                "member_id": item.member_id,
                "kind": item.kind.value,
                "target_kind": item.target_kind.value,
                "target_id": item.target_id,
                "member_range": _parameter_range_value(item.member_range),
                "target_parameters": [
                    _parameter_range_value(value) for value in item.target_parameters
                ],
                "metadata": item.metadata.to_dict(),
            }
            for item in sorted(
                geometry.attachments.values(), key=lambda record: record.id
            )
        ],
        "junctions": [
            {
                "id": item.id,
                "kind": item.kind.value,
                "member_uses": [
                    {
                        "member_id": value.member_id,
                        "member_range": _parameter_range_value(value.member_range),
                    }
                    for value in item.member_uses
                ],
                "sheet_ids": list(item.sheet_ids),
                "attachment_ids": list(item.attachment_ids),
                "metadata": item.metadata.to_dict(),
            }
            for item in sorted(geometry.junctions.values(), key=lambda record: record.id)
        ],
    }


def _surface(surface: object) -> dict[str, object] | None:
    if surface is None:
        return None
    if isinstance(surface, CoonsSurface):
        if not surface.has_boundaries:
            return {"type": "coons"}
        assert surface.bottom is not None and surface.right is not None and surface.top is not None and surface.left is not None
        return {
            "type": "coons",
            "bottom": surface.bottom.tolist(),
            "right": surface.right.tolist(),
            "top": surface.top.tolist(),
            "left": surface.left.tolist(),
        }
    if isinstance(surface, Plane):
        return {"type": "plane", "origin": surface.origin.tolist(), "u_vector": surface.u_vector.tolist(), "v_vector": surface.v_vector.tolist()}
    if isinstance(surface, Cylinder):
        return {"type": "cylinder", "origin": surface.origin.tolist(), "axis": surface.axis.tolist(), "radial_direction": surface.radial_direction.tolist(), "radius": surface.radius, "height": surface.height, "start_angle": surface.start_angle, "sweep_angle": surface.sweep_angle}
    if isinstance(surface, Cone):
        return {"type": "cone", "origin": surface.origin.tolist(), "axis": surface.axis.tolist(), "radial_direction": surface.radial_direction.tolist(), "radius_start": surface.radius_start, "radius_end": surface.radius_end, "height": surface.height, "start_angle": surface.start_angle, "sweep_angle": surface.sweep_angle}
    if isinstance(surface, RuledSurface):
        return {"type": "ruled", "first_boundary": surface.first_boundary.tolist(), "second_boundary": surface.second_boundary.tolist()}
    raise GeometryError(f"unsupported surface type {type(surface).__name__}")


def _feature_input(reference: EntityRef | FeatureOutputRef) -> dict[str, object]:
    if isinstance(reference, EntityRef):
        return {"entity": _ref(reference)}
    return {
        "feature": int(reference.feature_id),
        "output": reference.output_key,
        "kind": reference.kind,
    }


def _feature_history(
    history: FeatureHistory, geometry: GeometryModel
) -> dict[str, object]:
    registry = builtin_feature_registry()
    baseline = history.baseline
    records: list[dict[str, object]] = []
    for record in history.records:
        unavailable = not record.suppressed and not registry.has(record.kind)
        state = FeatureStatus.FROZEN.value if unavailable else record.state
        diagnostic = (
            _frozen_feature_diagnostic(record.kind)
            if unavailable
            else record.diagnostic
        )
        records.append(
            {
                "id": int(record.feature_id),
                "kind": record.kind,
                "kind_version": int(record.kind_version),
                "name": record.name,
                "parameters": _json_value(record.parameters),
                "inputs": {
                    port: [_feature_input(reference) for reference in references]
                    for port, references in sorted(record.inputs.items())
                },
                "dependencies": [int(item) for item in record.dependencies],
                "suppressed": record.suppressed,
                "outputs": {
                    key: _ref(reference)
                    for key, reference in sorted(record.outputs.items())
                },
                "state": state.value if isinstance(state, FeatureStatus) else state,
                "diagnostic": diagnostic,
                "materialization_checksum": (
                    record.materialization_checksum
                    or (
                        history.materialization_checksum(record, geometry)
                        if state in ("ok", "active", "frozen")
                        and record.outputs
                        else None
                    )
                ),
            }
        )
    return {
        "version": history.VERSION,
        "next_id": int(history.next_id),
        "baseline": (
            None
            if baseline is None
            else _json_value(baseline)
        ),
        "records": records,
    }


def to_dict(
    geometry: GeometryModel,
    *,
    include_features: bool = True,
    certified: bool = False,
) -> dict[str, object]:
    """Return a deterministic, validated and checksummed geometry document.

    Ordinary output requires valid geometry and structural topology.  Certified
    output additionally delegates to ``GeometryModel.strict_audit`` when that
    qualification API is available and refuses a non-certifiable report.
    """

    if include_features:
        geometry.features.validate_persistence(geometry)
    errors = geometry.validate_topology()
    structural_errors = geometry._validate_structural()  # noqa: SLF001
    if errors or structural_errors:
        raise GeometryError(
            "cannot serialize invalid topology: "
            + "; ".join((*errors, *structural_errors))
        )
    if certified:
        audit = getattr(geometry, "strict_audit", None)
        if audit is None:
            raise GeometryError("certified serialization requires strict_audit support")
        report = audit()
        certifiable = getattr(report, "certifiable", getattr(report, "is_certifiable", False))
        if not certifiable:
            raise GeometryError("certified serialization requires a clean strict audit")

    try:
        model_id = str(UUID(str(geometry.model_id)))
    except (TypeError, ValueError, AttributeError) as error:
        raise GeometryError("model_id must be a valid UUID") from error
    if UUID(model_id).int == 0:
        raise GeometryError("model_id cannot be the nil UUID")
    revision = _non_negative_integer(geometry.revision, "model revision")
    if not isinstance(geometry.units, str) or not geometry.units or "\x00" in geometry.units:
        raise GeometryError("model units must be a non-empty string without NUL")
    local_origin = _finite_array(geometry.local_origin, (3,), "local origin")
    transform = geometry.coordinate_transform
    if transform is not None:
        transform = _finite_array(transform, (4, 4), "coordinate transform")
        if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1.0e-14):
            raise GeometryError("coordinate transform must be affine")
        if abs(float(np.linalg.det(transform[:3, :3]))) <= np.finfo(float).eps:
            raise GeometryError("coordinate transform must be nonsingular")
    if not isinstance(geometry.tolerance, TolerancePolicy):
        raise GeometryError("model tolerance must be a TolerancePolicy")

    extensions = deepcopy(getattr(geometry, "_serialization_extensions", {}))
    extensions = _object(extensions, "extensions")
    for namespace, value in extensions.items():
        if not namespace or ":" not in namespace:
            raise GeometryError("extension keys must be non-empty namespaces containing ':'")
        _json_value(value)

    curves = []
    for edge in sorted(geometry.edges.values(), key=lambda item: item.id):
        if isinstance(edge.curve, Straight):
            curve: dict[str, object] = {"type": "straight"}
        elif isinstance(edge.curve, Arc):
            curve = {"type": "arc", "via_vertex": edge.curve.via_vertex}
        elif isinstance(edge.curve, Spline):
            curve = {"type": "spline", "control_vertices": list(edge.curve.control_vertices)}
        else:  # pragma: no cover - closed public union
            raise GeometryError(f"unsupported curve type {type(edge.curve).__name__}")
        curves.append({"id": edge.id, "start": edge.start, "end": edge.end, "curve": curve})
    id_state = geometry.id_state()
    id_state.update(geometry._next_structural_id)  # noqa: SLF001
    document: dict[str, object] = {
        "schema": SCHEMA,
        "version": VERSION,
        "model_id": model_id,
        "revision": revision,
        "coordinates": {
            "units": geometry.units,
            "local_origin": local_origin.tolist(),
            "coordinate_transform": None if transform is None else transform.tolist(),
        },
        "tolerance": {
            item.name: getattr(geometry.tolerance, item.name)
            for item in fields(TolerancePolicy)
        },
        "id_state": {kind: id_state[kind] for kind in _ID_KINDS},
        "vertices": [
            {"id": vertex.id, "position": vertex.position.tolist()}
            for vertex in sorted(geometry.vertices.values(), key=lambda item: item.id)
        ],
        "edges": curves,
        "faces": [
            {
                "id": face.id,
                "loop": _loop(face.loop),
                "corners": list(face.corners),
                "holes": [_loop(loop) for loop in face.holes],
                "surface": _surface(face.surface),
                "metadata": _json_value(face.metadata),
            }
            for face in sorted(geometry.faces.values(), key=lambda item: item.id)
        ],
        "structural": _structural_document(geometry),
        "groups": {
            name: [_ref(reference) for reference in geometry.group(name, resolve=False)]
            for name in sorted(geometry.groups)
        },
        "tags": [
            {"entity": _ref(reference), "values": sorted(values)}
            for reference, values in sorted(
                geometry.tags.items(), key=lambda item: (item[0].kind, item[0].id)
            )
        ],
        "replacement_history": [
            {"old": _ref(old), "new": [_ref(item) for item in new]}
            for old, new in sorted(
                geometry.replacement_history().items(),
                key=lambda item: (item[0].kind, item[0].id),
            )
        ],
        "extensions": dict(extensions),
    }
    if include_features:
        document["features"] = _feature_history(geometry.features, geometry)
    document["checksum"] = _checksum(document)
    return document


def _entity_ref(value: object) -> EntityRef:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise GeometryError("entity reference must be [kind, id]")
    kind, identifier = value
    if kind not in ("vertex", "edge", "face") or isinstance(identifier, bool):
        raise GeometryError("invalid entity reference")
    return EntityRef(kind, _integer(identifier, "entity ID"))  # type: ignore[arg-type]


def _feature_input_ref(value: object) -> EntityRef | FeatureOutputRef:
    if not isinstance(value, Mapping):
        raise GeometryError("feature input reference must be an object")
    if "entity" in value:
        if set(value) != {"entity"}:
            raise GeometryError("entity feature input has unexpected fields")
        return _entity_ref(value["entity"])
    required = {"feature", "output", "kind"}
    if set(value) != required:
        raise GeometryError("feature output reference is malformed")
    kind = value["kind"]
    if kind not in ("vertex", "edge", "face"):
        raise GeometryError("invalid feature output entity kind")
    return FeatureOutputRef(
        _integer(value["feature"], "feature ID"),
        str(value["output"]),
        kind,  # type: ignore[arg-type]
    )


def _unique_string_keys(values: Mapping[object, object], name: str) -> None:
    seen: set[str] = set()
    for key in values:
        if not isinstance(key, str) or not key:
            raise GeometryError(f"{name} keys must be non-empty strings")
        if key in seen:
            raise GeometryError(f"duplicate {name} key {key!r}")
        seen.add(key)


def _decode_feature_history(value: object, *, strict: bool = False) -> FeatureHistory:
    if not isinstance(value, Mapping):
        raise GeometryError("features must be an object")
    if strict:
        _exact_fields(
            value,
            required={"version", "next_id", "baseline", "records"},
            name="feature history",
        )
    version = _integer(value.get("version", 1), "feature-history version")
    if version != FeatureHistory.VERSION:
        raise GeometryError(f"unsupported feature-history version {version}")
    baseline = value.get("baseline")
    if baseline is not None and not isinstance(baseline, Mapping):
        raise GeometryError("feature-history baseline must be a geometry object")
    if isinstance(baseline, Mapping) and "features" in baseline:
        raise GeometryError("feature-history baseline cannot contain another history")
    records: list[FeatureRecord] = []
    raw_records = value.get("records", ())
    if strict:
        raw_records = _list(raw_records, "feature records")
    for item in raw_records:
        if not isinstance(item, Mapping):
            raise GeometryError("feature record must be an object")
        if strict:
            _exact_fields(
                item,
                required={
                    "id",
                    "kind",
                    "kind_version",
                    "name",
                    "parameters",
                    "inputs",
                    "dependencies",
                    "suppressed",
                    "outputs",
                    "state",
                    "diagnostic",
                    "materialization_checksum",
                },
                name="feature record",
            )
        raw_inputs = item.get("inputs", {})
        raw_outputs = item.get("outputs", {})
        if not isinstance(raw_inputs, Mapping) or not isinstance(raw_outputs, Mapping):
            raise GeometryError("feature inputs and outputs must be objects")
        _unique_string_keys(raw_inputs, "feature input port")
        _unique_string_keys(raw_outputs, "feature output")
        if strict:
            if not isinstance(item["kind"], str) or not item["kind"].strip():
                raise GeometryError("feature kind must be a non-empty string")
            if not isinstance(item["name"], str) or not item["name"].strip():
                raise GeometryError("feature name must be a non-empty string")
            _metadata(item["parameters"], "feature parameters")
            _list(item["dependencies"], "feature dependencies")
            _bool(item["suppressed"], "feature suppressed")
            try:
                FeatureStatus(item["state"])
            except (TypeError, ValueError) as error:
                raise GeometryError("feature state contains an unknown enum value") from error
            if item["diagnostic"] is not None and not isinstance(item["diagnostic"], str):
                raise GeometryError("feature diagnostic must be a string or null")
            if item["materialization_checksum"] is not None and not isinstance(
                item["materialization_checksum"], str
            ):
                raise GeometryError(
                    "feature materialization checksum must be a string or null"
                )
        made_inputs: dict[str, tuple[EntityRef | FeatureOutputRef, ...]] = {}
        for port, references in raw_inputs.items():
            raw_references = (
                _list(references, f"feature input port {port!r}")
                if strict
                else references
            )
            decoded = tuple(
                _feature_input_ref(reference) for reference in raw_references
            )
            if strict and len(set(decoded)) != len(decoded):
                raise GeometryError(
                    f"feature input port {port!r} has duplicate references"
                )
            made_inputs[str(port)] = decoded
        raw_dependencies = item.get("dependencies", ())
        dependencies = tuple(
            _integer(dependency, "feature dependency")
            for dependency in raw_dependencies
        )
        if strict and len(set(dependencies)) != len(dependencies):
            raise GeometryError("feature record has duplicate dependencies")
        records.append(
            FeatureRecord(
                feature_id=_integer(item["id"], "feature ID"),
                kind=str(item["kind"]),
                kind_version=_integer(
                    item.get("kind_version", 1), "feature kind version"
                ),
                name=(item["name"] if strict else str(item.get("name", item["kind"]))),
                parameters=dict(item.get("parameters", {})),
                inputs=made_inputs,
                dependencies=dependencies,
                suppressed=(
                    _bool(item["suppressed"], "feature suppressed")
                    if strict
                    else bool(item.get("suppressed", False))
                ),
                outputs={
                    str(key): _entity_ref(reference)
                    for key, reference in raw_outputs.items()
                },
                state=(item["state"] if strict else str(item.get("state", "pending"))),
                diagnostic=(
                    None
                    if item.get("diagnostic") is None
                    else str(item.get("diagnostic"))
                ),
                materialization_checksum=(
                    None
                    if item.get("materialization_checksum") is None
                    else str(item.get("materialization_checksum"))
                ),
            )
        )
    next_id = _integer(value.get("next_id", 1), "next feature ID")
    if strict:
        if next_id <= max((record.feature_id for record in records), default=0):
            raise GeometryError("next feature ID would reuse an existing feature ID")
        if next_id <= 0:
            raise GeometryError("next feature ID must be positive")
    else:
        # Older documents historically normalized a stale/missing allocator
        # counter on load.  Keep that migration behavior while schema 3 and
        # public FeatureHistory construction remain fail-closed.
        next_id = max(
            next_id,
            max((record.feature_id for record in records), default=0) + 1,
        )
    history = FeatureHistory(
        baseline=None if baseline is None else dict(baseline),
        records=records,
        next_id=next_id,
    )
    history.validate()
    return history


def _oriented_loop(value: object) -> tuple[OrientedEdge, ...]:
    if not isinstance(value, list):
        raise GeometryError("face loop must be a list")
    made = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2 or not isinstance(item[1], bool):
            raise GeometryError("oriented edge must be [edge_id, forward]")
        made.append(OrientedEdge(_integer(item[0], "edge ID"), item[1]))
    return tuple(made)


def _decode_surface(value: object, *, strict: bool = False) -> object:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise GeometryError("surface must be an object")
    data = dict(_object(value, "surface"))
    kind = data.pop("type", None)
    constructors = {
        "coons": CoonsSurface,
        "plane": Plane,
        "cylinder": Cylinder,
        "cone": Cone,
        "ruled": RuledSurface,
    }
    if kind not in constructors:
        raise GeometryError(f"unsupported surface type {kind!r}")
    if strict:
        fields_by_kind = {
            "coons": (set() if not data else {"bottom", "right", "top", "left"}),
            "plane": {"origin", "u_vector", "v_vector"},
            "cylinder": {
                "origin",
                "axis",
                "radial_direction",
                "radius",
                "height",
                "start_angle",
                "sweep_angle",
            },
            "cone": {
                "origin",
                "axis",
                "radial_direction",
                "radius_start",
                "radius_end",
                "height",
                "start_angle",
                "sweep_angle",
            },
            "ruled": {"first_boundary", "second_boundary"},
        }
        expected = fields_by_kind[str(kind)]
        if set(data) != expected:
            missing = expected - set(data)
            unexpected = set(data) - expected
            detail = []
            if missing:
                detail.append("missing " + ", ".join(sorted(missing)))
            if unexpected:
                detail.append("unexpected " + ", ".join(sorted(unexpected)))
            raise GeometryError(f"malformed {kind} surface: {'; '.join(detail)}")
    return constructors[kind](**data)  # type: ignore[index,operator]


def _decode_structural(value: object, geometry: GeometryModel) -> None:
    data = _object(value, "structural topology")
    fields_by_store = {
        "parts",
        "sheets",
        "face_uses",
        "coedges",
        "members",
        "member_edge_uses",
        "attachments",
        "junctions",
    }
    _exact_fields(data, required=fields_by_store, name="structural topology")
    records = {
        name: _list(data[name], f"structural {name}") for name in fields_by_store
    }
    for name, values in records.items():
        _unique_ids(values, name.replace("_", "-"))

    for raw in records["parts"]:
        item = _object(raw, "part record")
        _exact_fields(
            item,
            required={"id", "sheet_ids", "member_ids", "name", "metadata"},
            name="part record",
        )
        made = Part(
            id=_positive_integer(item["id"], "part ID"),
            sheet_ids=_ids(item["sheet_ids"], "sheet ID"),
            member_ids=_ids(item["member_ids"], "member ID"),
            name=item["name"],  # type: ignore[arg-type]
            metadata=_metadata(item["metadata"], "part metadata"),
        )
        geometry._put_structural("part", made)  # noqa: SLF001

    for raw in records["sheets"]:
        item = _object(raw, "sheet record")
        _exact_fields(
            item,
            required={
                "id",
                "part_id",
                "face_use_ids",
                "policy",
                "declared_non_manifold_edges",
                "name",
                "metadata",
            },
            name="sheet record",
        )
        policy = _object(item["policy"], "sheet policy")
        _exact_fields(
            policy,
            required={"boundary", "non_manifold", "connectivity"},
            name="sheet policy",
        )
        try:
            made_policy = SheetTopologyPolicy(
                boundary=BoundaryPolicy(policy["boundary"]),
                non_manifold=NonManifoldPolicy(policy["non_manifold"]),
                connectivity=ConnectivityPolicy(policy["connectivity"]),
            )
        except (TypeError, ValueError) as error:
            raise GeometryError("sheet policy contains an unknown enum value") from error
        made = Sheet(
            id=_positive_integer(item["id"], "sheet ID"),
            part_id=_positive_integer(item["part_id"], "part ID"),
            face_use_ids=_ids(item["face_use_ids"], "face-use ID"),
            policy=made_policy,
            declared_non_manifold_edges=_ids(
                item["declared_non_manifold_edges"],
                "declared non-manifold edge ID",
            ),
            name=item["name"],  # type: ignore[arg-type]
            metadata=_metadata(item["metadata"], "sheet metadata"),
        )
        geometry._put_structural("sheet", made)  # noqa: SLF001

    for raw in records["face_uses"]:
        item = _object(raw, "face-use record")
        _exact_fields(
            item,
            required={"id", "sheet_id", "face_id", "loops", "orientation", "metadata"},
            name="face-use record",
        )
        loops = tuple(
            _ids(loop, "coedge ID")
            for loop in _list(item["loops"], "face-use loops")
        )
        made = FaceUse(
            id=_positive_integer(item["id"], "face-use ID"),
            sheet_id=_positive_integer(item["sheet_id"], "sheet ID"),
            face_id=_positive_integer(item["face_id"], "face ID"),
            loops=loops,
            orientation=_decode_orientation(item["orientation"], "face-use orientation"),
            metadata=_metadata(item["metadata"], "face-use metadata"),
        )
        geometry._put_structural("face_use", made)  # noqa: SLF001

    for raw in records["coedges"]:
        item = _object(raw, "coedge record")
        _exact_fields(
            item,
            required={"id", "face_use_id", "edge_id", "orientation", "metadata"},
            name="coedge record",
        )
        made = Coedge(
            id=_positive_integer(item["id"], "coedge ID"),
            face_use_id=_positive_integer(item["face_use_id"], "face-use ID"),
            edge_id=_positive_integer(item["edge_id"], "edge ID"),
            orientation=_decode_orientation(item["orientation"], "coedge orientation"),
            metadata=_metadata(item["metadata"], "coedge metadata"),
        )
        geometry._put_structural("coedge", made)  # noqa: SLF001

    for raw in records["members"]:
        item = _object(raw, "member record")
        _exact_fields(
            item,
            required={"id", "part_id", "edge_use_ids", "name", "metadata"},
            name="member record",
        )
        made = Member(
            id=_positive_integer(item["id"], "member ID"),
            part_id=_positive_integer(item["part_id"], "part ID"),
            edge_use_ids=_ids(item["edge_use_ids"], "member-edge-use ID"),
            name=item["name"],  # type: ignore[arg-type]
            metadata=_metadata(item["metadata"], "member metadata"),
        )
        geometry._put_structural("member", made)  # noqa: SLF001

    for raw in records["member_edge_uses"]:
        item = _object(raw, "member-edge-use record")
        _exact_fields(
            item,
            required={
                "id",
                "member_id",
                "edge_id",
                "parent_range",
                "orientation",
                "metadata",
            },
            name="member-edge-use record",
        )
        made = MemberEdgeUse(
            id=_positive_integer(item["id"], "member-edge-use ID"),
            member_id=_positive_integer(item["member_id"], "member ID"),
            edge_id=_positive_integer(item["edge_id"], "edge ID"),
            parent_range=_decode_parameter_range(item["parent_range"], "parent range"),
            orientation=_decode_orientation(
                item["orientation"], "member-edge-use orientation"
            ),
            metadata=_metadata(item["metadata"], "member-edge-use metadata"),
        )
        geometry._put_structural("member_edge_use", made)  # noqa: SLF001

    for raw in records["attachments"]:
        item = _object(raw, "attachment record")
        _exact_fields(
            item,
            required={
                "id",
                "member_id",
                "kind",
                "target_kind",
                "target_id",
                "member_range",
                "target_parameters",
                "metadata",
            },
            name="attachment record",
        )
        try:
            attachment_kind = AttachmentKind(item["kind"])
            target_kind = AttachmentTargetKind(item["target_kind"])
        except (TypeError, ValueError) as error:
            raise GeometryError("attachment contains an unknown enum value") from error
        made = Attachment(
            id=_positive_integer(item["id"], "attachment ID"),
            member_id=_positive_integer(item["member_id"], "member ID"),
            kind=attachment_kind,
            target_kind=target_kind,
            target_id=_positive_integer(item["target_id"], "attachment target ID"),
            member_range=_decode_parameter_range(item["member_range"], "member range"),
            target_parameters=tuple(
                _decode_parameter_range(value, "target parameter range")
                for value in _list(item["target_parameters"], "target parameters")
            ),
            metadata=_metadata(item["metadata"], "attachment metadata"),
        )
        geometry._put_structural("attachment", made)  # noqa: SLF001

    for raw in records["junctions"]:
        item = _object(raw, "junction record")
        _exact_fields(
            item,
            required={
                "id",
                "kind",
                "member_uses",
                "sheet_ids",
                "attachment_ids",
                "metadata",
            },
            name="junction record",
        )
        try:
            junction_kind = JunctionKind(item["kind"])
        except (TypeError, ValueError) as error:
            raise GeometryError("junction contains an unknown enum value") from error
        member_uses: list[JunctionMemberUse] = []
        for raw_use in _list(item["member_uses"], "junction member uses"):
            use = _object(raw_use, "junction member-use record")
            _exact_fields(
                use,
                required={"member_id", "member_range"},
                name="junction member-use record",
            )
            member_uses.append(
                JunctionMemberUse(
                    member_id=_positive_integer(use["member_id"], "member ID"),
                    member_range=_decode_parameter_range(
                        use["member_range"], "member range"
                    ),
                )
            )
        made = Junction(
            id=_positive_integer(item["id"], "junction ID"),
            kind=junction_kind,
            member_uses=tuple(member_uses),
            sheet_ids=_ids(item["sheet_ids"], "sheet ID"),
            attachment_ids=_ids(item["attachment_ids"], "attachment ID"),
            metadata=_metadata(item["metadata"], "junction metadata"),
        )
        geometry._put_structural("junction", made)  # noqa: SLF001


def _migrate_legacy_structural(geometry: GeometryModel, source_version: int) -> None:
    """Infer only face ownership from schema-1/2 materialized topology."""

    if geometry.faces:
        sheet_ids: list[int] = []
        next_coedge = 1
        for position, face in enumerate(
            sorted(geometry.faces.values(), key=lambda item: item.id), start=1
        ):
            # Legacy documents carried no cross-face orientation, manifold, or
            # connectivity intent.  A separate inferred sheet per face retains
            # ownership without rejecting valid legacy faces or inventing that
            # they form one consistently oriented structural sheet.
            sheet_id = position
            face_use_id = position
            sheet_ids.append(sheet_id)
            loops: list[tuple[int, ...]] = []
            for loop in (face.loop,) + tuple(face.holes):
                loop_ids: list[int] = []
                for oriented in loop:
                    coedge = Coedge(
                        next_coedge,
                        face_use_id,
                        oriented.edge,
                        Orientation.FORWARD if oriented.forward else Orientation.REVERSED,
                    )
                    geometry._put_structural("coedge", coedge)  # noqa: SLF001
                    loop_ids.append(coedge.id)
                    next_coedge += 1
                loops.append(tuple(loop_ids))
            geometry._put_structural(  # noqa: SLF001
                "face_use",
                FaceUse(face_use_id, sheet_id, face.id, tuple(loops)),
            )
            geometry._put_structural(  # noqa: SLF001
                "sheet",
                Sheet(
                    sheet_id,
                    1,
                    (face_use_id,),
                    name=f"Migrated face {face.id}",
                ),
            )
        geometry._put_structural(  # noqa: SLF001
            "part", Part(1, sheet_ids=tuple(sheet_ids), name="Migrated geometry")
        )
    for kind in _STRUCTURAL_KINDS:
        store = geometry._structural_store(kind)  # noqa: SLF001
        geometry._next_structural_id[kind] = max(store, default=0) + 1  # noqa: SLF001
    geometry._serialization_extensions = {  # noqa: SLF001
        "anygeometry:migration": {
            "source_version": source_version,
            "target_version": VERSION,
            "inferred": "face ownership only; no members inferred",
        }
    }


def _decode_geometry_records(
    document: Mapping[str, object], geometry: GeometryModel, *, strict: bool
) -> None:
    vertices = _list(document.get("vertices", []), "vertices")
    edges = _list(document.get("edges", []), "edges")
    faces = _list(document.get("faces", []), "faces")
    _unique_ids(vertices, "vertex")
    _unique_ids(edges, "edge")
    _unique_ids(faces, "face")

    for raw in vertices:
        item = _object(raw, "vertex record")
        if strict:
            _exact_fields(item, required={"id", "position"}, name="vertex record")
        identifier = _positive_integer(item["id"], "vertex ID")
        position = _finite_array(item["position"], (3,), "vertex position")
        geometry._put_entity("vertex", Vertex(identifier, position))  # noqa: SLF001

    for raw in edges:
        item = _object(raw, "edge record")
        if strict:
            _exact_fields(
                item,
                required={"id", "start", "end", "curve"},
                name="edge record",
            )
        identifier = _positive_integer(item["id"], "edge ID")
        curve_data = _object(item["curve"], "curve")
        curve_kind = curve_data.get("type")
        if curve_kind == "straight":
            if strict:
                _exact_fields(curve_data, required={"type"}, name="straight curve")
            curve = Straight()
        elif curve_kind == "arc":
            if strict:
                _exact_fields(
                    curve_data,
                    required={"type", "via_vertex"},
                    name="arc curve",
                )
            curve = Arc(_positive_integer(curve_data["via_vertex"], "arc via vertex"))
        elif curve_kind == "spline":
            if strict:
                _exact_fields(
                    curve_data,
                    required={"type", "control_vertices"},
                    name="spline curve",
                )
            curve = Spline(_ids(curve_data["control_vertices"], "spline control vertex"))
        else:
            raise GeometryError(f"unsupported curve type {curve_kind!r}")
        geometry._put_entity(  # noqa: SLF001
            "edge",
            Edge(
                identifier,
                _positive_integer(item["start"], "edge start vertex"),
                _positive_integer(item["end"], "edge end vertex"),
                curve,
            ),
        )

    for raw in faces:
        item = _object(raw, "face record")
        if strict:
            _exact_fields(
                item,
                required={"id", "loop", "corners", "holes", "surface", "metadata"},
                name="face record",
            )
        identifier = _positive_integer(item["id"], "face ID")
        metadata = _metadata(item.get("metadata", {}), "face metadata")
        geometry._put_entity(  # noqa: SLF001
            "face",
            Face(
                identifier,
                _oriented_loop(item["loop"]),
                tuple(
                    _non_negative_integer(value, "face corner")
                    for value in item.get("corners", [])  # type: ignore[union-attr]
                ),
                metadata,
                tuple(
                    _oriented_loop(loop)
                    for loop in _list(item.get("holes", []), "face holes")
                ),
                _decode_surface(item.get("surface"), strict=strict),
            ),
        )


def _restore_counters(
    document: Mapping[str, object], geometry: GeometryModel, *, strict: bool
) -> None:
    raw_state = document.get("id_state")
    if raw_state is None and not strict:
        raw_state = {
            kind: max(geometry._entity_store(kind), default=0) + 1  # noqa: SLF001
            for kind in _GEOMETRY_KINDS
        }
    state = _object(raw_state, "id_state")
    expected = set(_ID_KINDS if strict else _GEOMETRY_KINDS)
    if strict:
        _exact_fields(state, required=expected, name="id_state")
    geometry_state: dict[str, int] = {}
    for kind in _GEOMETRY_KINDS:
        value = _positive_integer(state[kind], f"{kind} ID counter")
        if value <= max(geometry._entity_store(kind), default=0):  # noqa: SLF001
            raise GeometryError(f"{kind} ID counter would reuse an existing ID")
        geometry_state[kind] = value
    geometry.restore_id_state(geometry_state)
    if strict:
        for kind in _STRUCTURAL_KINDS:
            value = _positive_integer(state[kind], f"{kind} ID counter")
            store = geometry._structural_store(kind)  # noqa: SLF001
            if value <= max(store, default=0):
                raise GeometryError(f"{kind} ID counter would reuse an existing ID")
            geometry._next_structural_id[kind] = value  # noqa: SLF001


def _decode_semantics(
    document: Mapping[str, object], geometry: GeometryModel, *, strict: bool
) -> None:
    raw_groups = _object(document.get("groups", {}), "groups")
    for name, values in raw_groups.items():
        if not name or "\x00" in name:
            raise GeometryError("group name must be a non-empty string without NUL")
        references = tuple(
            _entity_ref(value) for value in _list(values, f"group {name!r}")
        )
        if len(set(references)) != len(references):
            raise GeometryError(f"group {name!r} contains duplicate entity references")
        # Loading materializes an isolated, not-yet-published model.  Populate
        # the owner's private semantic stores directly, then validate the
        # complete document before returning it.
        geometry._groups[name] = set(references)  # noqa: SLF001

    seen_tags: set[EntityRef] = set()
    for raw in _list(document.get("tags", []), "tags"):
        item = _object(raw, "tag record")
        if strict:
            _exact_fields(item, required={"entity", "values"}, name="tag record")
        reference = _entity_ref(item["entity"])
        if reference in seen_tags:
            raise GeometryError(f"duplicate tag record for {reference}")
        seen_tags.add(reference)
        raw_values = _list(item["values"], "tag values")
        if any(not isinstance(value, str) or not value for value in raw_values):
            raise GeometryError("tag values must be non-empty strings")
        if len(set(raw_values)) != len(raw_values):
            raise GeometryError(f"duplicate tag value for {reference}")
        geometry._tags[reference] = set(raw_values)  # type: ignore[arg-type]  # noqa: SLF001

    seen_history: set[EntityRef] = set()
    for raw in _list(
        document.get("replacement_history", []), "replacement history"
    ):
        item = _object(raw, "replacement-history record")
        if strict:
            _exact_fields(item, required={"old", "new"}, name="replacement-history record")
        old = _entity_ref(item["old"])
        if old in seen_history:
            raise GeometryError(f"duplicate replacement-history record for {old}")
        seen_history.add(old)
        replacements = tuple(
            _entity_ref(value)
            for value in _list(item["new"], "replacement descendants")
        )
        if len(set(replacements)) != len(replacements):
            raise GeometryError(f"duplicate replacement descendant for {old}")
        geometry._replacement_history[old] = replacements  # noqa: SLF001


def from_dict(document: Mapping[str, Any]) -> GeometryModel:
    """Restore a complete geometry document and validate all references."""

    if not isinstance(document, Mapping):
        raise GeometryError("geometry document must be an object")
    if document.get("schema", SCHEMA) != SCHEMA:
        raise GeometryError("not an ANYgeometry document")
    version = _integer(document.get("version", 1), "version")
    if version not in (1, 2, VERSION):
        raise GeometryError(f"unsupported ANYgeometry version {version}")
    strict = version == VERSION
    if strict:
        _exact_fields(
            document,
            required=_CURRENT_REQUIRED_FIELDS,
            optional=_CURRENT_OPTIONAL_FIELDS,
            name="schema-3 geometry document",
        )
        try:
            model_id = UUID(str(document["model_id"]))
        except (ValueError, TypeError, AttributeError) as error:
            raise GeometryError("model_id must be a valid UUID") from error
        if model_id.int == 0:
            raise GeometryError("model_id cannot be the nil UUID")
        revision = _non_negative_integer(document["revision"], "model revision")
        raw_tolerance = _object(document["tolerance"], "tolerance policy")
        tolerance_fields = {item.name for item in fields(TolerancePolicy)}
        _exact_fields(
            raw_tolerance,
            required=tolerance_fields,
            name="tolerance policy",
        )
        try:
            tolerance = TolerancePolicy(**raw_tolerance)  # type: ignore[arg-type]
        except TypeError as error:
            raise GeometryError(f"malformed tolerance policy: {error}") from error
        geometry = GeometryModel(model_id=model_id, tolerance=tolerance)
        geometry._revision = revision  # noqa: SLF001
        coordinates = _object(document["coordinates"], "coordinates")
        _exact_fields(
            coordinates,
            required={"units", "local_origin", "coordinate_transform"},
            name="coordinates",
        )
        units = coordinates["units"]
        if not isinstance(units, str) or not units or "\x00" in units:
            raise GeometryError("coordinates units must be a non-empty string without NUL")
        geometry._units = units  # noqa: SLF001
        geometry._local_origin = _finite_array(  # noqa: SLF001
            coordinates["local_origin"], (3,), "local origin"
        )
        geometry._local_origin.flags.writeable = False  # noqa: SLF001
        raw_transform = coordinates["coordinate_transform"]
        if raw_transform is None:
            geometry._coordinate_transform = None  # noqa: SLF001
        else:
            transform = _finite_array(raw_transform, (4, 4), "coordinate transform")
            if not np.allclose(
                transform[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1.0e-14
            ):
                raise GeometryError("coordinate transform must be affine")
            if abs(float(np.linalg.det(transform[:3, :3]))) <= np.finfo(float).eps:
                raise GeometryError("coordinate transform must be nonsingular")
            transform.flags.writeable = False
            geometry._coordinate_transform = transform  # noqa: SLF001
        extensions = _object(document["extensions"], "extensions")
        for namespace, value in extensions.items():
            if not namespace or ":" not in namespace:
                raise GeometryError(
                    "extension keys must be non-empty namespaces containing ':'"
                )
            _json_value(value)
        geometry._serialization_extensions = deepcopy(dict(extensions))  # noqa: SLF001
    else:
        geometry = GeometryModel()

    try:
        _decode_geometry_records(document, geometry, strict=strict)
        if strict:
            _decode_structural(document["structural"], geometry)
        else:
            _migrate_legacy_structural(geometry, version)
        _restore_counters(document, geometry, strict=strict)
        _decode_semantics(document, geometry, strict=strict)
        if version >= 2 and "features" in document:
            geometry._install_feature_history(  # noqa: SLF001
                _decode_feature_history(document["features"], strict=strict)
            )
    except (KeyError, TypeError, ValueError) as error:
        raise GeometryError(f"malformed geometry document: {error}") from error
    geometry._rebuild_incidence()  # noqa: SLF001
    geometry._rebuild_member_incidence()  # noqa: SLF001
    geometry._spatial_index = None  # noqa: SLF001
    errors = geometry.validate_topology()
    errors = (*errors, *geometry._validate_structural())  # noqa: SLF001
    if errors:
        raise GeometryError("invalid geometry topology: " + "; ".join(errors))
    for kind, store in (("vertex", geometry.vertices), ("edge", geometry.edges), ("face", geometry.faces)):
        if geometry.id_state()[kind] <= max(store, default=0):
            raise GeometryError(f"{kind} ID counter would reuse an existing ID")
    history = geometry.replacement_history()
    keys = geometry.entity_keys()
    counters = geometry.id_state()
    for old, replacements in history.items():
        if old.id <= 0 or old.id >= counters[old.kind]:
            raise GeometryError(
                f"replacement history references missing entity {old}"
            )
        for replacement in replacements:
            if replacement.kind != old.kind:
                raise GeometryError(
                    f"replacement history changes entity kind from {old} "
                    f"to {replacement}"
                )
            if replacement.id <= 0 or replacement.id >= counters[replacement.kind]:
                raise GeometryError(
                    f"replacement history references missing entity {replacement}"
                )
            if (
                (replacement.kind, replacement.id) not in keys
                and replacement not in history
            ):
                raise GeometryError(
                    "replacement history has an unresolved descendant "
                    f"{replacement}"
                )

    visiting: set[EntityRef] = set()
    visited: set[EntityRef] = set()

    def check_history(reference: EntityRef) -> None:
        if reference in visited or reference not in history:
            return
        if reference in visiting:
            raise GeometryError(
                f"replacement history contains a cycle at {reference}"
            )
        visiting.add(reference)
        for replacement in history[reference]:
            check_history(replacement)
        visiting.remove(reference)
        visited.add(reference)

    for reference in history:
        check_history(reference)
    for members in geometry.groups.values():
        for reference in members:
            if (reference.kind, reference.id) not in geometry.entity_keys() and reference not in geometry.replacement_history():
                raise GeometryError(f"group references missing entity {reference}")
    if "features" not in document:
        # A materialized legacy document is valid design input, but its
        # construction intent cannot be inferred safely.  Preserve it as the
        # immutable base for all features added after migration.
        geometry.features._capture_baseline_unchecked(  # noqa: SLF001
            geometry, force=True
        )
    else:
        keys = geometry.entity_keys()
        history_keys = geometry.replacement_history()
        registry = builtin_feature_registry()
        for record in geometry.features._records:  # noqa: SLF001
            for key, reference in record.outputs.items():
                if (
                    (reference.kind, reference.id) not in keys
                    and reference not in history_keys
                ):
                    diagnostic = (
                        f"feature {record.feature_id} output {key!r} references "
                        f"missing entity {reference}"
                    )
                    if registry.has(record.kind):
                        raise GeometryError(diagnostic)
                    record.state = "invalid"
                    record.diagnostic = diagnostic
        for record in geometry.features._records:  # noqa: SLF001
            if record.suppressed or registry.has(record.kind):
                continue
            if record.state == "invalid":
                continue
            if not record.outputs:
                record.state = "invalid"
                record.diagnostic = (
                    f"feature {record.feature_id} has no last-good outputs"
                )
                continue
            if record.materialization_checksum is None:
                # Documents written before checksummed frozen features remain
                # viewable; the next write persists the derived checksum.
                if strict:
                    record.state = "invalid"
                    record.diagnostic = (
                        f"feature {record.feature_id} has no materialization checksum"
                    )
                    continue
                record.materialization_checksum = (
                    geometry.features.materialization_checksum(record, geometry)
                )
                record.state = "frozen"
                record.diagnostic = _frozen_feature_diagnostic(record.kind)
                continue
            diagnostic = geometry.features.validate_materialization(record, geometry)
            record.state = "frozen" if diagnostic is None else "invalid"
            record.diagnostic = (
                _frozen_feature_diagnostic(record.kind)
                if diagnostic is None
                else diagnostic
            )
    if strict:
        _verify_checksum(document)
    return geometry


def write_geometry(
    path: str | Path, geometry: GeometryModel, *, certified: bool = False
) -> None:
    """Write deterministic JSON, gzip-compressed when the suffix is ``.gz``."""

    target = Path(path)
    payload = json.dumps(
        to_dict(geometry, certified=certified),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    if target.suffix.lower() == ".gz":
        # Suppress the wall-clock timestamp and source filename from the gzip
        # header so identical geometry produces byte-identical artifacts.
        with target.open("wb") as raw_stream:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_stream,
                mtime=0,
            ) as stream:
                stream.write(payload.encode("utf-8"))
    else:
        target.write_text(payload, encoding="utf-8", newline="\n")


def read_geometry(path: str | Path) -> GeometryModel:
    target = Path(path)
    try:
        if target.suffix.lower() == ".gz":
            with gzip.open(target, "rt", encoding="utf-8") as stream:
                document = json.load(
                    stream, object_pairs_hook=_reject_duplicate_json_keys
                )
        else:
            document = json.loads(
                target.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
            )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise GeometryError(f"malformed JSON geometry document: {error}") from error
    return from_dict(document)
