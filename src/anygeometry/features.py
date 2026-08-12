"""Persistent, dependency-aware geometry feature history.

``EntityRef`` remains the identity of materialized topology.  A
``FeatureOutputRef`` is the stable design-time address of an output slot and
is resolved to materialized entities whenever the history is regenerated.

The history is intentionally executor-agnostic.  ANYgeometry registers its
own neutral operations; downstream packages may register namespaced feature
kinds without creating a reverse dependency from this package.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any, Callable, Dict, Iterable, Mapping, Protocol, Sequence, TYPE_CHECKING

from .entities import EntityKind, EntityRef
from .errors import GeometryError

if TYPE_CHECKING:
    from .model import GeometryModel

__all__ = [
    "FeatureExecution",
    "FeatureExecutor",
    "FeatureHistory",
    "FeatureInputRef",
    "FeatureOutputRef",
    "FeatureRecord",
    "FeatureRegistry",
    "FeatureResult",
    "FeatureStatus",
    "RegenerationReport",
    "builtin_feature_registry",
]


_KINDS = ("vertex", "edge", "face")


class FeatureStatus(str, Enum):
    """Portable feature state used by history, UI badges, and validation."""

    PENDING = "pending"
    OK = "ok"
    ACTIVE = "active"
    SUPPRESSED = "suppressed"
    BLOCKED = "blocked"
    FAILED = "failed"
    FROZEN = "frozen"
    INVALID = "invalid"


@dataclass(frozen=True)
class FeatureOutputRef:
    """Stable address of one named output of a modelling feature."""

    feature_id: int
    output_key: str
    kind: EntityKind

    def __post_init__(self) -> None:
        if isinstance(self.feature_id, bool) or int(self.feature_id) <= 0:
            raise ValueError("a feature output needs a positive feature ID")
        if not isinstance(self.output_key, str) or not self.output_key.strip():
            raise ValueError("a feature output needs a non-empty key")
        if self.kind not in _KINDS:
            raise ValueError(f"unknown feature-output entity kind {self.kind!r}")


FeatureInputRef = EntityRef | FeatureOutputRef


@dataclass
class FeatureRecord:
    """Serializable definition and current materialization of one feature."""

    feature_id: int
    kind: str
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, tuple[FeatureInputRef, ...]] = field(default_factory=dict)
    suppressed: bool = False
    kind_version: int = 1
    dependencies: tuple[int, ...] = ()
    outputs: Dict[str, EntityRef] = field(default_factory=dict)
    state: str = "pending"
    diagnostic: str | None = None
    materialization_checksum: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.feature_id, bool) or int(self.feature_id) <= 0:
            raise ValueError("a feature needs a positive integer ID")
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("a feature needs a non-empty kind")
        if not isinstance(self.name, str) or not self.name.strip():
            self.name = self.kind
        if isinstance(self.kind_version, bool) or int(self.kind_version) <= 0:
            raise ValueError("a feature kind version must be a positive integer")
        self.parameters = deepcopy(dict(self.parameters))
        self.inputs = {
            str(port): tuple(references)
            for port, references in dict(self.inputs).items()
        }
        self.dependencies = tuple(dict.fromkeys(int(item) for item in self.dependencies))
        self.outputs = dict(self.outputs)
        if self.materialization_checksum is not None:
            checksum = str(self.materialization_checksum).lower()
            if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
                raise ValueError("a feature materialization checksum must be SHA-256")
            self.materialization_checksum = checksum

    def output_ref(self, key: str) -> FeatureOutputRef:
        reference = self.outputs.get(str(key))
        if reference is None:
            raise KeyError(f"feature {self.feature_id} has no output {key!r}")
        return FeatureOutputRef(self.feature_id, str(key), reference.kind)


@dataclass(frozen=True)
class FeatureExecution:
    """Outputs returned by a feature executor."""

    outputs: Mapping[str, EntityRef]


class FeatureExecutor(Protocol):
    def execute(
        self,
        geometry: "GeometryModel",
        feature: FeatureRecord,
        inputs: Mapping[str, tuple[EntityRef, ...]],
    ) -> FeatureExecution | Mapping[str, EntityRef]: ...


ExecutorCallable = Callable[
    ["GeometryModel", FeatureRecord, Mapping[str, tuple[EntityRef, ...]]],
    FeatureExecution | Mapping[str, EntityRef],
]


class FeatureRegistry:
    """Executor lookup keyed by a serializable namespaced feature kind."""

    def __init__(self) -> None:
        self._executors: Dict[str, FeatureExecutor | ExecutorCallable] = {}

    def register(
        self,
        kind: str,
        executor: FeatureExecutor | ExecutorCallable,
        *,
        replace: bool = False,
    ) -> None:
        key = str(kind).strip()
        if not key:
            raise ValueError("a feature executor needs a non-empty kind")
        if key in self._executors and not replace:
            raise ValueError(f"a feature executor is already registered for {key!r}")
        self._executors[key] = executor

    def unregister(self, kind: str) -> None:
        self._executors.pop(str(kind), None)

    def has(self, kind: str) -> bool:
        return str(kind) in self._executors

    def execute(
        self,
        geometry: "GeometryModel",
        feature: FeatureRecord,
        inputs: Mapping[str, tuple[EntityRef, ...]],
    ) -> FeatureExecution:
        try:
            executor = self._executors[feature.kind]
        except KeyError:
            raise GeometryError(
                f"no executor is available for feature kind {feature.kind!r}"
            ) from None
        execute = getattr(executor, "execute", executor)
        made = execute(geometry, feature, inputs)  # type: ignore[misc,operator]
        if isinstance(made, FeatureExecution):
            return made
        return FeatureExecution(dict(made))

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(sorted(self._executors))


@dataclass(frozen=True)
class FeatureResult:
    feature_id: int
    state: str
    outputs: Mapping[str, EntityRef] = field(default_factory=dict)
    diagnostic: str | None = None


@dataclass(frozen=True)
class RegenerationReport:
    success: bool
    features: tuple[FeatureResult, ...]
    replacements: tuple[tuple[EntityRef, tuple[EntityRef, ...]], ...] = ()
    diagnostic: str | None = None


class FeatureHistory:
    """Ordered persistent feature definitions plus an immutable base model."""

    VERSION = 1

    def __init__(
        self,
        *,
        baseline: Mapping[str, Any] | None = None,
        records: Iterable[FeatureRecord] = (),
        next_id: int = 1,
    ) -> None:
        self.baseline: Dict[str, Any] | None = (
            None if baseline is None else deepcopy(dict(baseline))
        )
        self.records: list[FeatureRecord] = [deepcopy(item) for item in records]
        self._next_id = int(next_id)
        self._normalize_next_id()

    def _normalize_next_id(self) -> None:
        minimum = max((item.feature_id for item in self.records), default=0) + 1
        if self._next_id < minimum:
            self._next_id = minimum

    @property
    def next_id(self) -> int:
        return self._next_id

    def snapshot(self) -> Dict[str, Any]:
        return {
            "baseline": deepcopy(self.baseline),
            "records": deepcopy(self.records),
            "next_id": self._next_id,
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        self.baseline = deepcopy(snapshot.get("baseline"))
        self.records = deepcopy(list(snapshot.get("records", ())))
        self._next_id = int(snapshot.get("next_id", 1))
        self._normalize_next_id()

    def capture_baseline(self, geometry: "GeometryModel", *, force: bool = False) -> None:
        """Capture current materialized topology before the first feature."""

        if self.baseline is not None and not force:
            return
        from .serialization import to_dict

        self.baseline = to_dict(geometry, include_features=False)

    def append(
        self,
        kind: str,
        *,
        name: str | None = None,
        parameters: Mapping[str, Any] | None = None,
        inputs: Mapping[str, Sequence[FeatureInputRef]] | None = None,
        suppressed: bool = False,
        kind_version: int = 1,
        dependencies: Sequence[int] = (),
    ) -> FeatureRecord:
        record = FeatureRecord(
            feature_id=self._next_id,
            kind=str(kind),
            name=name or str(kind),
            parameters={} if parameters is None else dict(parameters),
            inputs={
                str(port): tuple(items)
                for port, items in ({} if inputs is None else inputs).items()
            },
            suppressed=bool(suppressed),
            kind_version=kind_version,
            dependencies=tuple(dependencies),
        )
        self.records.append(record)
        self._next_id += 1
        try:
            self.validate()
        except Exception:
            self.records.pop()
            self._next_id -= 1
            raise
        return record

    def get(self, feature_id: int) -> FeatureRecord:
        for record in self.records:
            if record.feature_id == int(feature_id):
                return record
        raise KeyError(f"no feature {feature_id}")

    def remove(self, feature_id: int, *, cascade: bool = False) -> tuple[int, ...]:
        wanted = int(feature_id)
        dependents = self.dependents(wanted, transitive=True)
        if dependents and not cascade:
            raise GeometryError(
                f"feature {wanted} is used by feature(s) {list(dependents)}"
            )
        removed = {wanted, *(dependents if cascade else ())}
        before = len(self.records)
        self.records[:] = [item for item in self.records if item.feature_id not in removed]
        if len(self.records) == before:
            raise KeyError(f"no feature {wanted}")
        return tuple(sorted(removed))

    def set_suppressed(self, feature_id: int, suppressed: bool = True) -> None:
        self.get(feature_id).suppressed = bool(suppressed)

    def update(
        self,
        feature_id: int,
        *,
        name: str | None = None,
        parameters: Mapping[str, Any] | None = None,
        inputs: Mapping[str, Sequence[FeatureInputRef]] | None = None,
        dependencies: Sequence[int] | None = None,
    ) -> FeatureRecord:
        """Atomically replace editable fields and revalidate dependencies."""

        record = self.get(feature_id)
        previous = deepcopy(record)
        if name is not None:
            if not str(name).strip():
                raise ValueError("a feature name cannot be empty")
            record.name = str(name)
        if parameters is not None:
            record.parameters = deepcopy(dict(parameters))
        if inputs is not None:
            record.inputs = {
                str(port): tuple(references)
                for port, references in inputs.items()
            }
        if dependencies is not None:
            record.dependencies = tuple(
                dict.fromkeys(int(item) for item in dependencies)
            )
        try:
            self.validate()
        except Exception:
            index = self.records.index(record)
            self.records[index] = previous
            raise
        return record

    def move(self, feature_id: int, index: int) -> None:
        record = self.get(feature_id)
        old = self.records.index(record)
        self.records.pop(old)
        self.records.insert(max(0, min(int(index), len(self.records))), record)
        try:
            self.validate()
        except Exception:
            self.records.remove(record)
            self.records.insert(old, record)
            raise

    def dependencies_of(self, record: FeatureRecord | int) -> tuple[int, ...]:
        made = self.get(record) if isinstance(record, int) else record
        dependencies = list(made.dependencies)
        for references in made.inputs.values():
            dependencies.extend(
                item.feature_id
                for item in references
                if isinstance(item, FeatureOutputRef)
            )
        return tuple(dict.fromkeys(dependencies))

    def dependents(self, feature_id: int, *, transitive: bool = False) -> tuple[int, ...]:
        found: list[int] = []
        pending = [int(feature_id)]
        while pending:
            parent = pending.pop(0)
            direct = [
                item.feature_id
                for item in self.records
                if parent in self.dependencies_of(item) and item.feature_id not in found
            ]
            found.extend(direct)
            if transitive:
                pending.extend(direct)
        return tuple(found)

    def validate(self) -> None:
        seen: set[int] = set()
        for record in self.records:
            if record.feature_id in seen:
                raise GeometryError(f"duplicate feature ID {record.feature_id}")
            for dependency in self.dependencies_of(record):
                if dependency not in seen:
                    raise GeometryError(
                        f"feature {record.feature_id} depends on later or missing "
                        f"feature {dependency}"
                    )
            for references in record.inputs.values():
                for reference in references:
                    if not isinstance(reference, (EntityRef, FeatureOutputRef)):
                        raise GeometryError(
                            f"feature {record.feature_id} has an invalid input reference"
                        )
            seen.add(record.feature_id)

    def resolve(
        self, reference: FeatureInputRef, geometry: "GeometryModel"
    ) -> tuple[EntityRef, ...]:
        if isinstance(reference, EntityRef):
            return geometry.resolve_ref(reference)
        record = self.get(reference.feature_id)
        if record.suppressed or record.state not in ("ok", "active", "frozen"):
            return ()
        materialized = record.outputs.get(reference.output_key)
        if materialized is None or materialized.kind != reference.kind:
            return ()
        return geometry.resolve_ref(materialized)

    def materialization_checksum(
        self, record: FeatureRecord | int, geometry: "GeometryModel"
    ) -> str:
        """Hash the resolved topology owned by one feature.

        The payload contains output-slot identity plus the complete face/edge/
        vertex closure needed to view or mesh those outputs.  It deliberately
        excludes labels and unrelated downstream topology.
        """

        made = self.get(record) if isinstance(record, int) else record
        from .serialization import to_dict

        document = to_dict(geometry, include_features=False)
        vertices = {int(item["id"]): item for item in document["vertices"]}
        edges = {int(item["id"]): item for item in document["edges"]}
        faces = {int(item["id"]): item for item in document["faces"]}
        selected: dict[str, set[int]] = {kind: set() for kind in _KINDS}
        resolved_outputs: dict[str, list[list[object]]] = {}

        for key, reference in sorted(made.outputs.items()):
            current = geometry.resolve_ref(reference)
            resolved_outputs[key] = [[item.kind, item.id] for item in current]
            for item in current:
                selected[item.kind].add(item.id)

        for face_id in tuple(selected["face"]):
            face = faces.get(face_id)
            if face is None:
                continue
            for edge_id, _forward in face["loop"]:
                selected["edge"].add(int(edge_id))
            for loop in face.get("holes", ()):
                for edge_id, _forward in loop:
                    selected["edge"].add(int(edge_id))
            selected["vertex"].update(int(item) for item in face.get("corners", ()))

        for edge_id in tuple(selected["edge"]):
            edge = edges.get(edge_id)
            if edge is None:
                continue
            selected["vertex"].update((int(edge["start"]), int(edge["end"])))
            curve = edge.get("curve", {})
            if "via_vertex" in curve:
                selected["vertex"].add(int(curve["via_vertex"]))
            selected["vertex"].update(
                int(item) for item in curve.get("control_vertices", ())
            )

        selected_refs = {
            (kind, identifier)
            for kind, identifiers in selected.items()
            for identifier in identifiers
        }
        payload = {
            "outputs": resolved_outputs,
            "vertices": [vertices[item] for item in sorted(selected["vertex"]) if item in vertices],
            "edges": [edges[item] for item in sorted(selected["edge"]) if item in edges],
            "faces": [faces[item] for item in sorted(selected["face"]) if item in faces],
            "groups": {
                name: [reference for reference in members if tuple(reference) in selected_refs]
                for name, members in sorted(document.get("groups", {}).items())
                if any(tuple(reference) in selected_refs for reference in members)
            },
            "tags": [
                item
                for item in document.get("tags", ())
                if tuple(item["entity"]) in selected_refs
            ],
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def validate_materialization(
        self, record: FeatureRecord | int, geometry: "GeometryModel"
    ) -> str | None:
        """Return a blocking diagnostic when a frozen materialization is bad."""

        made = self.get(record) if isinstance(record, int) else record
        if made.suppressed:
            return None
        if not made.outputs:
            return f"feature {made.feature_id} has no last-good outputs"
        for key, reference in made.outputs.items():
            if not geometry.resolve_ref(reference):
                return f"feature {made.feature_id} output {key!r} is unresolved"
        if made.materialization_checksum is None:
            return f"feature {made.feature_id} has no materialization checksum"
        try:
            actual = self.materialization_checksum(made, geometry)
        except (TypeError, ValueError) as error:
            return f"feature {made.feature_id} materialization is invalid: {error}"
        if actual != made.materialization_checksum:
            return f"feature {made.feature_id} materialization checksum does not match"
        return None

    def regenerate(
        self,
        geometry: "GeometryModel",
        registry: FeatureRegistry,
    ) -> RegenerationReport:
        """Atomically replay the history from its base materialization."""

        self.validate()
        unknown = [
            record
            for record in self.records
            if not record.suppressed and not registry.has(record.kind)
        ]
        if unknown:
            results = []
            for record in unknown:
                diagnostic = self.validate_materialization(record, geometry)
                state = "frozen" if diagnostic is None else "invalid"
                results.append(
                    FeatureResult(
                        record.feature_id,
                        state,
                        dict(record.outputs),
                        diagnostic,
                    )
                )
            identifiers = ", ".join(str(record.feature_id) for record in unknown)
            detail = next(
                (item.diagnostic for item in results if item.diagnostic),
                f"no executor is available for frozen feature(s) {identifiers}",
            )
            return RegenerationReport(
                False,
                tuple(results),
                diagnostic=f"regeneration is disabled: {detail}",
            )
        if self.baseline is None:
            self.capture_baseline(geometry)
        assert self.baseline is not None
        from .serialization import from_dict

        previous_history = geometry.replacement_history()
        previous_outputs = {
            (record.feature_id, key): geometry.resolve_ref(reference)
            for record in self.records
            for key, reference in record.outputs.items()
        }
        working = from_dict(deepcopy(self.baseline))
        working.reserve_id_state(geometry.id_state())
        replayed = deepcopy(self.records)
        by_id = {record.feature_id: record for record in replayed}
        results: list[FeatureResult] = []

        for record in replayed:
            record.outputs = {}
            record.diagnostic = None
            if record.suppressed:
                record.state = "suppressed"
                results.append(FeatureResult(record.feature_id, record.state))
                continue
            resolved: Dict[str, tuple[EntityRef, ...]] = {}
            blocked = next(
                (
                    f"dependency {dependency} is not active"
                    for dependency in record.dependencies
                    if by_id[dependency].state != "ok"
                ),
                None,
            )
            for port, references in record.inputs.items():
                if blocked:
                    break
                made: list[EntityRef] = []
                for reference in references:
                    if isinstance(reference, EntityRef):
                        current = working.resolve_ref(reference)
                    else:
                        source = by_id[reference.feature_id]
                        materialized = source.outputs.get(reference.output_key)
                        current = (
                            ()
                            if source.state != "ok" or materialized is None
                            else working.resolve_ref(materialized)
                        )
                    if not current:
                        blocked = f"input {port!r} cannot be resolved"
                        break
                    made.extend(current)
                if blocked:
                    break
                resolved[port] = tuple(dict.fromkeys(made))
            if blocked:
                record.state = "blocked"
                record.diagnostic = blocked
                results.append(
                    FeatureResult(record.feature_id, record.state, diagnostic=blocked)
                )
                continue

            snapshot = working.topology_snapshot()
            working.begin_replacement_log()
            try:
                execution = registry.execute(working, record, resolved)
                outputs = {str(key): value for key, value in execution.outputs.items()}
                for key, reference in outputs.items():
                    if not key or not isinstance(reference, EntityRef):
                        raise GeometryError(
                            f"feature {record.feature_id} returned an invalid output"
                        )
                    if not working.resolve_ref(reference):
                        raise GeometryError(
                            f"feature {record.feature_id} output {key!r} is missing"
                        )
                errors = working.validate_topology()
                if errors:
                    raise GeometryError("; ".join(errors))
            except Exception as error:
                working.restore_topology(snapshot)
                detail = str(error) or type(error).__name__
                return RegenerationReport(
                    False,
                    tuple(results)
                    + (FeatureResult(record.feature_id, "failed", diagnostic=detail),),
                    diagnostic=f"feature {record.feature_id} failed: {detail}",
                )
            record.outputs = outputs
            record.state = "ok"
            results.append(FeatureResult(record.feature_id, "ok", dict(outputs)))

        transition_by_old: Dict[EntityRef, tuple[EntityRef, ...]] = {}
        new_by_id = {record.feature_id: record for record in replayed}
        # A downstream topology feature and its upstream producer can resolve
        # to the same old terminal entity.  Iterate in history order and let
        # the latest producer own that transition; otherwise an upstream
        # aggregate output could conflict with the child's more precise slot.
        for (feature_id, key), old_items in previous_outputs.items():
            new_binding = new_by_id[feature_id].outputs.get(key)
            descendants = (
                () if new_binding is None else working.resolve_ref(new_binding)
            )
            for old in old_items:
                same_kind = tuple(item for item in descendants if item.kind == old.kind)
                if old not in same_kind:
                    transition_by_old[old] = same_kind
        transitions = list(transition_by_old.items())

        historical = list(previous_history.items())
        try:
            working.record_replacements_atomic((*historical, *transitions))
        except GeometryError as error:
            return RegenerationReport(
                False,
                tuple(results),
                diagnostic=f"cannot preserve feature lineage: {error}",
            )

        geometry.restore_topology(working.topology_snapshot())
        self.records = replayed
        for record in self.records:
            record.materialization_checksum = (
                self.materialization_checksum(record, geometry)
                if record.state == "ok" and record.outputs
                else None
            )
        return RegenerationReport(True, tuple(results), tuple(transitions))


def _one(
    inputs: Mapping[str, tuple[EntityRef, ...]], port: str, kind: str
) -> EntityRef:
    values = inputs.get(port, ())
    if len(values) != 1 or values[0].kind != kind:
        raise GeometryError(f"feature input {port!r} needs one {kind}")
    return values[0]


def _created_outputs(
    geometry: "GeometryModel", before: Mapping[str, set[int]]
) -> Dict[str, EntityRef]:
    stores = {
        "vertex": geometry.vertices,
        "edge": geometry.edges,
        "face": geometry.faces,
    }
    outputs: Dict[str, EntityRef] = {}
    for kind in _KINDS:
        made = sorted(set(stores[kind]) - before[kind])
        for index, identifier in enumerate(made):
            outputs[f"{kind}/{index}"] = EntityRef(kind, identifier)  # type: ignore[arg-type]
    return outputs


def _before(geometry: "GeometryModel") -> Dict[str, set[int]]:
    return {
        "vertex": set(geometry.vertices),
        "edge": set(geometry.edges),
        "face": set(geometry.faces),
    }


def builtin_feature_registry() -> FeatureRegistry:
    """Return the neutral executors shipped by ANYgeometry."""

    registry = FeatureRegistry()

    def point(geometry, feature, inputs):
        del inputs
        position = feature.parameters.get("position", (0.0, 0.0, 0.0))
        return {"point": EntityRef("vertex", geometry.add_point(*position))}

    def line(geometry, feature, inputs):
        del feature
        start = _one(inputs, "start", "vertex")
        end = _one(inputs, "end", "vertex")
        return {"edge": EntityRef("edge", geometry.add_line(start.id, end.id))}

    def arc(geometry, feature, inputs):
        del feature
        start = _one(inputs, "start", "vertex")
        via = _one(inputs, "via", "vertex")
        end = _one(inputs, "end", "vertex")
        return {"edge": EntityRef("edge", geometry.add_arc(start.id, via.id, end.id))}

    def spline(geometry, feature, inputs):
        del feature
        start = _one(inputs, "start", "vertex")
        end = _one(inputs, "end", "vertex")
        controls = inputs.get("controls", ())
        if any(item.kind != "vertex" for item in controls):
            raise GeometryError("spline controls must be vertices")
        edge = geometry.add_spline(start.id, [item.id for item in controls], end.id)
        return {"edge": EntityRef("edge", edge)}

    def polyline(geometry, feature, inputs):
        vertices = inputs.get("vertices", ())
        if len(vertices) < 2 or any(item.kind != "vertex" for item in vertices):
            raise GeometryError("a polyline feature needs at least two vertices")
        edges = geometry.add_polyline(
            [item.id for item in vertices], bool(feature.parameters.get("close", False))
        )
        return {f"edge/{index}": EntityRef("edge", item) for index, item in enumerate(edges)}

    def face(geometry, feature, inputs):
        edges = inputs.get("edges", ())
        if len(edges) < 3 or any(item.kind != "edge" for item in edges):
            raise GeometryError("a face feature needs at least three edges")
        identifier = geometry.add_face(
            [item.id for item in edges], corners=feature.parameters.get("corners")
        )
        return {"face": EntityRef("face", identifier)}

    def plate(geometry, feature, inputs):
        del feature
        vertices = inputs.get("vertices", ())
        if len(vertices) < 3 or any(item.kind != "vertex" for item in vertices):
            raise GeometryError("a plate feature needs at least three vertices")
        before = _before(geometry)
        identifier = geometry.add_plate([item.id for item in vertices])
        outputs = _created_outputs(geometry, before)
        outputs["face"] = EntityRef("face", identifier)
        return outputs

    def extrude(geometry, feature, inputs):
        edges = inputs.get("edges", ())
        if not edges or any(item.kind != "edge" for item in edges):
            raise GeometryError("an extrude feature needs edges")
        before = _before(geometry)
        geometry.extrude([item.id for item in edges], feature.parameters["vector"])
        return _created_outputs(geometry, before)

    def sketch_extrude(geometry, feature, inputs):
        from .sketch import SketchDefinition, materialize_sketch

        support = _one(inputs, "support_face", "face")
        definition = SketchDefinition.from_parameters(feature.parameters)
        return materialize_sketch(geometry, support.id, definition)

    def fragment_overlaps(geometry, feature, inputs):
        del feature
        from .overlaps import fragment_coplanar_overlaps

        faces = inputs.get("faces", ())
        if len(faces) < 2 or any(item.kind != "face" for item in faces):
            raise GeometryError(
                "plate overlap fragmentation needs at least two ordered faces"
            )
        return fragment_coplanar_overlaps(
            geometry, [item.id for item in faces]
        ).outputs

    def revolve(geometry, feature, inputs):
        edges = inputs.get("edges", ())
        if not edges or any(item.kind != "edge" for item in edges):
            raise GeometryError("a revolve feature needs edges")
        before = _before(geometry)
        geometry.revolve(
            [item.id for item in edges],
            feature.parameters.get("axis_point", (0.0, 0.0, 0.0)),
            feature.parameters.get("axis_direction", (0.0, 0.0, 1.0)),
            float(feature.parameters["angle"]),
            feature.parameters.get("segments"),
        )
        return _created_outputs(geometry, before)

    def transform_feature(geometry, feature, inputs):
        from .operations import transform

        entities = inputs.get("entities", ())
        if not entities:
            raise GeometryError("a transform feature needs entities")
        transform(geometry, feature.parameters["matrix"], entities)
        return {f"entity/{index}": item for index, item in enumerate(entities)}

    def split_edge(geometry, feature, inputs):
        edge = _one(inputs, "edge", "edge")
        point_id, halves = geometry.split_edge(
            edge.id, float(feature.parameters.get("fraction", 0.5))
        )
        return {
            "point": EntityRef("vertex", point_id),
            "edge/0": EntityRef("edge", halves[0]),
            "edge/1": EntityRef("edge", halves[1]),
        }

    def split_face(geometry, feature, inputs):
        from .operations import split_face_at, split_face_between

        face_ref = _one(inputs, "face", "face")
        if "start" in inputs or "end" in inputs:
            start = _one(inputs, "start", "vertex")
            end = _one(inputs, "end", "vertex")
            divider, faces = split_face_between(
                geometry, face_ref.id, start.id, end.id
            )
        else:
            divider, faces = split_face_at(
                geometry,
                face_ref.id,
                int(feature.parameters.get("axis", 0)),
                float(feature.parameters.get("fraction", 0.5)),
            )
        return {
            "divider": EntityRef("edge", divider),
            "face/0": EntityRef("face", faces[0]),
            "face/1": EntityRef("face", faces[1]),
        }

    def strip_face_feature(geometry, feature, inputs):
        from .operations import strip_face

        face_ref = _one(inputs, "face", "face")
        faces, dividers = strip_face(
            geometry,
            face_ref.id,
            int(feature.parameters.get("axis", 0)),
            int(feature.parameters.get("count", 2)),
        )
        return {
            **{
                f"face/{index}": EntityRef("face", identifier)
                for index, identifier in enumerate(faces)
            },
            **{
                f"divider/{index}": EntityRef("edge", identifier)
                for index, identifier in enumerate(dividers)
            },
        }

    def trim_hole_feature(geometry, feature, inputs):
        from .operations import punch_hole

        face_ref = _one(inputs, "face", "face")
        face_id, boundary = punch_hole(
            geometry,
            face_ref.id,
            feature.parameters["centre"],
            float(feature.parameters["radius"]),
        )
        return {
            "face": EntityRef("face", face_id),
            **{
                f"boundary/{index}": EntityRef("edge", identifier)
                for index, identifier in enumerate(boundary)
            },
        }

    def set_corners(geometry, feature, inputs):
        face_ref = _one(inputs, "face", "face")
        geometry.set_face_corners(face_ref.id, feature.parameters.get("corners", ()))
        return {"face": face_ref}

    def copy_feature(geometry, feature, inputs):
        from .editing import copy_entities

        entities = inputs.get("entities", ())
        return copy_entities(
            geometry, entities, matrix=feature.parameters.get("matrix")
        ).outputs

    def mirror_feature(geometry, feature, inputs):
        from .editing import mirror_entities

        return mirror_entities(
            geometry,
            inputs.get("entities", ()),
            feature.parameters.get("plane_point", (0.0, 0.0, 0.0)),
            feature.parameters.get("plane_normal", (1.0, 0.0, 0.0)),
        ).outputs

    def linear_pattern_feature(geometry, feature, inputs):
        from .editing import linear_pattern

        result = linear_pattern(
            geometry,
            inputs.get("entities", ()),
            feature.parameters["direction"],
            float(feature.parameters["spacing"]),
            int(feature.parameters["count"]),
        )
        return {
            f"instance/{index}/{key}": reference
            for index, instance in enumerate(result.instances)
            for key, reference in instance.outputs.items()
        }

    def circular_pattern_feature(geometry, feature, inputs):
        from .editing import circular_pattern

        result = circular_pattern(
            geometry,
            inputs.get("entities", ()),
            feature.parameters.get("axis_point", (0.0, 0.0, 0.0)),
            feature.parameters.get("axis_direction", (0.0, 0.0, 1.0)),
            float(feature.parameters["angle_step"]),
            int(feature.parameters["count"]),
        )
        return {
            f"instance/{index}/{key}": reference
            for index, instance in enumerate(result.instances)
            for key, reference in instance.outputs.items()
        }

    def reverse_feature(geometry, feature, inputs):
        from .editing import reverse_edge, reverse_face

        values = inputs.get("entity", ())
        if len(values) != 1:
            raise GeometryError("reverse needs one edge or face")
        entity = values[0]
        if entity.kind == "edge":
            reverse_edge(geometry, entity.id)
        elif entity.kind == "face":
            reverse_face(geometry, entity.id)
        else:
            raise GeometryError("reverse supports edges and faces")
        return {"entity": entity}

    def generator(name: str):
        def execute(geometry, feature, inputs):
            del inputs
            from . import generators

            build = getattr(generators, name)
            source = build(**feature.parameters)
            inserted = geometry.insert_model(source)
            return inserted.outputs

        return execute

    for kind, executor in {
        "geometry.point": point,
        "geometry.line": line,
        "geometry.arc": arc,
        "geometry.spline": spline,
        "geometry.polyline": polyline,
        "geometry.face": face,
        "geometry.plate": plate,
        "geometry.extrude": extrude,
        "geometry.sketch.extrude": sketch_extrude,
        "geometry.fragment.overlaps": fragment_overlaps,
        "geometry.revolve": revolve,
        "geometry.transform": transform_feature,
        "geometry.split_edge": split_edge,
        "geometry.split_face": split_face,
        "geometry.strip_face": strip_face_feature,
        "geometry.trim_hole": trim_hole_feature,
        "geometry.set_face_corners": set_corners,
        "geometry.copy": copy_feature,
        "geometry.mirror": mirror_feature,
        "geometry.pattern.linear": linear_pattern_feature,
        "geometry.pattern.circular": circular_pattern_feature,
        "geometry.reverse": reverse_feature,
    }.items():
        registry.register(kind, executor)
    for name in (
        "plate",
        "stiffened_panel",
        "cylinder",
        "cone",
        "shell",
        "bulkhead",
        "frame",
        "girder",
        "stiffener",
    ):
        registry.register(f"generator.{name}", generator(name))
    return registry
