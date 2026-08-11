"""Versioned JSON-ready serialization owned by the geometry package."""

from __future__ import annotations

import gzip
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .curves import Arc, Spline, Straight
from .entities import Edge, EntityRef, Face, OrientedEdge, Vertex
from .errors import GeometryError
from .features import (
    FeatureHistory,
    FeatureOutputRef,
    FeatureRecord,
    builtin_feature_registry,
)
from .model import GeometryModel
from .surfaces import CoonsSurface, Cone, Cylinder, Plane, RuledSurface

SCHEMA = "anygeometry"
VERSION = 2

__all__ = ["SCHEMA", "VERSION", "from_dict", "read_geometry", "to_dict", "write_geometry"]


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise GeometryError(f"{name} must be an integer")
    return int(value)


def _ref(reference: EntityRef) -> list[object]:
    return [reference.kind, reference.id]


def _loop(loop: tuple[OrientedEdge, ...]) -> list[list[object]]:
    return [[item.edge, item.forward] for item in loop]


def _json_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise GeometryError("metadata floats must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise GeometryError(
        f"metadata value of type {type(value).__name__} is not JSON serializable"
    )


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
        "feature": reference.feature_id,
        "output": reference.output_key,
        "kind": reference.kind,
    }


def _feature_history(
    history: FeatureHistory, geometry: GeometryModel
) -> dict[str, object]:
    return {
        "version": history.VERSION,
        "next_id": history.next_id,
        "baseline": deepcopy(history.baseline),
        "records": [
            {
                "id": record.feature_id,
                "kind": record.kind,
                "kind_version": record.kind_version,
                "name": record.name,
                "parameters": _json_value(record.parameters),
                "inputs": {
                    port: [_feature_input(reference) for reference in references]
                    for port, references in sorted(record.inputs.items())
                },
                "dependencies": list(record.dependencies),
                "suppressed": record.suppressed,
                "outputs": {
                    key: _ref(reference)
                    for key, reference in sorted(record.outputs.items())
                },
                "state": record.state,
                "diagnostic": record.diagnostic,
                "materialization_checksum": (
                    record.materialization_checksum
                    or (
                        history.materialization_checksum(record, geometry)
                        if record.state in ("ok", "active", "frozen")
                        and record.outputs
                        else None
                    )
                ),
            }
            for record in history.records
        ],
    }


def to_dict(
    geometry: GeometryModel, *, include_features: bool = True
) -> dict[str, object]:
    """Return a deterministic, JSON-ready complete geometry document."""

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
    document: dict[str, object] = {
        "schema": SCHEMA,
        "version": VERSION,
        "id_state": geometry.id_state(),
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
    }
    if include_features:
        document["features"] = _feature_history(geometry.features, geometry)
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


def _decode_feature_history(value: object) -> FeatureHistory:
    if not isinstance(value, Mapping):
        raise GeometryError("features must be an object")
    version = _integer(value.get("version", 1), "feature-history version")
    if version != FeatureHistory.VERSION:
        raise GeometryError(f"unsupported feature-history version {version}")
    baseline = value.get("baseline")
    if baseline is not None and not isinstance(baseline, Mapping):
        raise GeometryError("feature-history baseline must be a geometry object")
    if isinstance(baseline, Mapping) and "features" in baseline:
        raise GeometryError("feature-history baseline cannot contain another history")
    records: list[FeatureRecord] = []
    for item in value.get("records", ()):
        if not isinstance(item, Mapping):
            raise GeometryError("feature record must be an object")
        raw_inputs = item.get("inputs", {})
        raw_outputs = item.get("outputs", {})
        if not isinstance(raw_inputs, Mapping) or not isinstance(raw_outputs, Mapping):
            raise GeometryError("feature inputs and outputs must be objects")
        records.append(
            FeatureRecord(
                feature_id=_integer(item["id"], "feature ID"),
                kind=str(item["kind"]),
                kind_version=_integer(
                    item.get("kind_version", 1), "feature kind version"
                ),
                name=str(item.get("name", item["kind"])),
                parameters=dict(item.get("parameters", {})),
                inputs={
                    str(port): tuple(_feature_input_ref(reference) for reference in references)
                    for port, references in raw_inputs.items()
                },
                dependencies=tuple(
                    _integer(dependency, "feature dependency")
                    for dependency in item.get("dependencies", ())
                ),
                suppressed=bool(item.get("suppressed", False)),
                outputs={
                    str(key): _entity_ref(reference)
                    for key, reference in raw_outputs.items()
                },
                state=str(item.get("state", "pending")),
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
    history = FeatureHistory(
        baseline=None if baseline is None else dict(baseline),
        records=records,
        next_id=_integer(value.get("next_id", 1), "next feature ID"),
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


def _decode_surface(value: object) -> object:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise GeometryError("surface must be an object")
    data = dict(value)
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
    return constructors[kind](**data)  # type: ignore[index,operator]


def from_dict(document: Mapping[str, Any]) -> GeometryModel:
    """Restore a complete geometry document and validate all references."""

    if document.get("schema", SCHEMA) != SCHEMA:
        raise GeometryError("not an ANYgeometry document")
    version = _integer(document.get("version", 1), "version")
    if version not in (1, VERSION):
        raise GeometryError(f"unsupported ANYgeometry version {version}")
    geometry = GeometryModel()
    try:
        for item in document.get("vertices", []):
            identifier = _integer(item["id"], "vertex ID")
            position = np.asarray(item["position"], dtype=float)
            if identifier <= 0 or identifier in geometry.vertices or position.shape != (3,) or not np.all(np.isfinite(position)):
                raise GeometryError("invalid or duplicate vertex")
            geometry.vertices[identifier] = Vertex(identifier, position)
        for item in document.get("edges", []):
            identifier = _integer(item["id"], "edge ID")
            curve_data = item["curve"]
            curve_kind = curve_data["type"]
            if curve_kind == "straight":
                curve = Straight()
            elif curve_kind == "arc":
                curve = Arc(_integer(curve_data["via_vertex"], "arc via vertex"))
            elif curve_kind == "spline":
                curve = Spline(
                    tuple(
                        _integer(value, "spline control vertex")
                        for value in curve_data["control_vertices"]
                    )
                )
            else:
                raise GeometryError(f"unsupported curve type {curve_kind!r}")
            if identifier <= 0 or identifier in geometry.edges:
                raise GeometryError("invalid or duplicate edge")
            geometry.edges[identifier] = Edge(
                identifier,
                _integer(item["start"], "edge start vertex"),
                _integer(item["end"], "edge end vertex"),
                curve,
            )
        for item in document.get("faces", []):
            identifier = _integer(item["id"], "face ID")
            if identifier <= 0 or identifier in geometry.faces:
                raise GeometryError("invalid or duplicate face")
            geometry.faces[identifier] = Face(
                identifier,
                _oriented_loop(item["loop"]),
                tuple(_integer(value, "face corner") for value in item.get("corners", [])),
                dict(item.get("metadata", {})),
                tuple(_oriented_loop(loop) for loop in item.get("holes", [])),
                _decode_surface(item.get("surface")),
            )
        state = document.get("id_state") or {
            "vertex": max(geometry.vertices, default=0) + 1,
            "edge": max(geometry.edges, default=0) + 1,
            "face": max(geometry.faces, default=0) + 1,
        }
        geometry.restore_id_state(
            {
                kind: _integer(state[kind], f"{kind} ID counter")
                for kind in ("vertex", "edge", "face")
            }
        )
        for name, values in document.get("groups", {}).items():
            geometry.groups[str(name)] = {_entity_ref(value) for value in values}
        for item in document.get("tags", []):
            geometry.tags[_entity_ref(item["entity"])] = {str(value) for value in item["values"]}
        for item in document.get("replacement_history", []):
            geometry._replacement_history[_entity_ref(item["old"])] = tuple(  # noqa: SLF001
                _entity_ref(value) for value in item["new"]
            )
        if version >= 2 and "features" in document:
            geometry.features = _decode_feature_history(document["features"])
    except (KeyError, TypeError, ValueError) as error:
        raise GeometryError(f"malformed geometry document: {error}") from error
    errors = geometry.validate_topology()
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
    if version == 1 or "features" not in document:
        # A materialized legacy document is valid design input, but its
        # construction intent cannot be inferred safely.  Preserve it as the
        # immutable base for all features added after migration.
        geometry.features.capture_baseline(geometry, force=True)
    else:
        keys = geometry.entity_keys()
        history_keys = geometry.replacement_history()
        registry = builtin_feature_registry()
        for record in geometry.features.records:
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
        for record in geometry.features.records:
            if record.suppressed or registry.has(record.kind):
                continue
            if record.state == "invalid":
                continue
            if record.materialization_checksum is None:
                # Documents written before checksummed frozen features remain
                # viewable; the next write persists the derived checksum.
                record.materialization_checksum = (
                    geometry.features.materialization_checksum(record, geometry)
                )
                record.state = "frozen"
                record.diagnostic = (
                    f"executor for feature kind {record.kind!r} is unavailable; "
                    "using its verified last-good materialization"
                )
                continue
            diagnostic = geometry.features.validate_materialization(record, geometry)
            record.state = "frozen" if diagnostic is None else "invalid"
            record.diagnostic = (
                f"executor for feature kind {record.kind!r} is unavailable; "
                "using its verified last-good materialization"
                if diagnostic is None
                else diagnostic
            )
    return geometry


def write_geometry(path: str | Path, geometry: GeometryModel) -> None:
    """Write deterministic JSON, gzip-compressed when the suffix is ``.gz``."""

    target = Path(path)
    payload = json.dumps(to_dict(geometry), indent=2, sort_keys=True) + "\n"
    if target.suffix.lower() == ".gz":
        with gzip.open(target, "wt", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    else:
        target.write_text(payload, encoding="utf-8", newline="\n")


def read_geometry(path: str | Path) -> GeometryModel:
    target = Path(path)
    if target.suffix.lower() == ".gz":
        with gzip.open(target, "rt", encoding="utf-8") as stream:
            document = json.load(stream)
    else:
        document = json.loads(target.read_text(encoding="utf-8"))
    return from_dict(document)
