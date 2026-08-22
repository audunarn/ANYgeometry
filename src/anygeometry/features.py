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
from numbers import Integral, Real
from typing import Any, Callable, Dict, Iterable, Mapping, Protocol, Sequence, TYPE_CHECKING

import numpy as np

from .curves import Arc, Spline, Straight
from .entities import Edge, EntityKind, EntityRef, Face, Vertex
from .errors import GeometryError
from .surfaces import CoonsSurface, Cone, Cylinder, Plane, RuledSurface

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
_FULL_REPLAY_DIAGNOSTIC = "feature-history structure changed; full replay required"
_INCREMENTAL_ADDITIVE_FEATURES = frozenset(
    {
        "geometry.point",
        "geometry.line",
        "geometry.arc",
        "geometry.spline",
        "geometry.polyline",
        "geometry.face",
        "geometry.plate",
        "geometry.extrude",
        "geometry.sketch.extrude",
        "geometry.revolve",
        "geometry.copy",
        "geometry.mirror",
        "geometry.pattern.linear",
        "geometry.pattern.circular",
        "geometry.pattern.rectangular",
        "geometry.pattern.transforms",
    }
)
_INCREMENTAL_MUTATING_FEATURES = frozenset(
    {
        "geometry.fragment.overlaps",
        "geometry.transform",
        "geometry.split_edge",
        "geometry.split_face",
        "geometry.strip_face",
        "geometry.trim_hole",
        "geometry.set_face_corners",
        "geometry.reverse",
    }
)


def _frozen_feature_diagnostic(kind: str) -> str:
    """Canonical loader-visible diagnostic for an unavailable executor."""

    return (
        f"executor for feature kind {kind!r} is unavailable; "
        "using its verified last-good materialization"
    )


def _values_equal(first: object, second: object) -> bool:
    """Deep equality for JSON-like feature state, including NumPy values."""

    if first is second:
        return True
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        try:
            return bool(np.array_equal(first, second))
        except (TypeError, ValueError):
            return False
    if type(first) is not type(second):
        return False
    if isinstance(first, Mapping):
        return (
            first.keys() == second.keys()  # type: ignore[union-attr]
            and all(_values_equal(first[key], second[key]) for key in first)  # type: ignore[index]
        )
    if isinstance(first, (tuple, list)):
        return len(first) == len(second) and all(  # type: ignore[arg-type]
            _values_equal(left, right)
            for left, right in zip(first, second)  # type: ignore[arg-type]
        )
    if isinstance(first, FeatureRecord):
        return all(
            _values_equal(getattr(first, name), getattr(second, name))
            for name in (
                "feature_id",
                "kind",
                "name",
                "parameters",
                "inputs",
                "suppressed",
                "kind_version",
                "dependencies",
                "outputs",
                "state",
                "diagnostic",
                "materialization_checksum",
            )
        )
    try:
        return bool(first == second)
    except (TypeError, ValueError):
        return False


def _validate_json_value(value: object, path: str) -> None:
    """Reject values that cannot survive the strict JSON feature codec."""

    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, Integral):
        return
    if isinstance(value, Real):
        if not np.isfinite(float(value)):
            raise GeometryError(f"{path} must be finite")
        return
    if isinstance(value, np.ndarray):
        _validate_json_value(value.tolist(), path)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise GeometryError(f"{path} keys must be non-empty strings")
            _validate_json_value(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for position, item in enumerate(value):
            _validate_json_value(item, f"{path}[{position}]")
        return
    raise GeometryError(
        f"{path} value of type {type(value).__name__} is not JSON serializable"
    )


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise GeometryError(f"{name} must be a positive integer")
    return int(value)


def _validate_entity_ref(reference: object, name: str) -> None:
    if not isinstance(reference, EntityRef):
        raise GeometryError(f"{name} must be an EntityRef")
    if reference.kind not in _KINDS:
        raise GeometryError(f"{name} has unknown entity kind {reference.kind!r}")
    _positive_int(reference.id, f"{name} entity ID")


def _validate_feature_output_ref(reference: object, name: str) -> None:
    if not isinstance(reference, FeatureOutputRef):
        raise GeometryError(f"{name} must be a FeatureOutputRef")
    _positive_int(reference.feature_id, f"{name} feature ID")
    if not isinstance(reference.output_key, str) or not reference.output_key.strip():
        raise GeometryError(f"{name} needs a non-empty output key")
    if reference.kind not in _KINDS:
        raise GeometryError(f"{name} has unknown entity kind {reference.kind!r}")


def _closure_surface(surface: object) -> dict[str, object] | None:
    """Encode one support surface exactly as the geometry document does."""

    if surface is None:
        return None
    if isinstance(surface, CoonsSurface):
        if not surface.has_boundaries:
            return {"type": "coons"}
        assert (
            surface.bottom is not None
            and surface.right is not None
            and surface.top is not None
            and surface.left is not None
        )
        return {
            "type": "coons",
            "bottom": surface.bottom.tolist(),
            "right": surface.right.tolist(),
            "top": surface.top.tolist(),
            "left": surface.left.tolist(),
        }
    if isinstance(surface, Plane):
        return {
            "type": "plane",
            "origin": surface.origin.tolist(),
            "u_vector": surface.u_vector.tolist(),
            "v_vector": surface.v_vector.tolist(),
        }
    if isinstance(surface, Cylinder):
        return {
            "type": "cylinder",
            "origin": surface.origin.tolist(),
            "axis": surface.axis.tolist(),
            "radial_direction": surface.radial_direction.tolist(),
            "radius": surface.radius,
            "height": surface.height,
            "start_angle": surface.start_angle,
            "sweep_angle": surface.sweep_angle,
        }
    if isinstance(surface, Cone):
        return {
            "type": "cone",
            "origin": surface.origin.tolist(),
            "axis": surface.axis.tolist(),
            "radial_direction": surface.radial_direction.tolist(),
            "radius_start": surface.radius_start,
            "radius_end": surface.radius_end,
            "height": surface.height,
            "start_angle": surface.start_angle,
            "sweep_angle": surface.sweep_angle,
        }
    if isinstance(surface, RuledSurface):
        return {
            "type": "ruled",
            "first_boundary": surface.first_boundary.tolist(),
            "second_boundary": surface.second_boundary.tolist(),
        }
    raise GeometryError(f"unsupported surface type {type(surface).__name__}")


def _closure_vertex(vertex: Vertex) -> dict[str, object]:
    return {"id": vertex.id, "position": vertex.position.tolist()}


def _closure_edge(edge: Edge) -> dict[str, object]:
    if isinstance(edge.curve, Straight):
        curve: dict[str, object] = {"type": "straight"}
    elif isinstance(edge.curve, Arc):
        curve = {"type": "arc", "via_vertex": edge.curve.via_vertex}
    elif isinstance(edge.curve, Spline):
        curve = {
            "type": "spline",
            "control_vertices": list(edge.curve.control_vertices),
        }
    else:  # pragma: no cover - closed public union
        raise GeometryError(f"unsupported curve type {type(edge.curve).__name__}")
    return {
        "id": edge.id,
        "start": edge.start,
        "end": edge.end,
        "curve": curve,
    }


def _closure_face(face: Face) -> dict[str, object]:
    return {
        "id": face.id,
        "loop": [[item.edge, item.forward] for item in face.loop],
        "corners": list(face.corners),
        "holes": [
            [[item.edge, item.forward] for item in loop] for loop in face.holes
        ],
        "surface": _closure_surface(face.surface),
        "metadata": face.metadata.to_dict(),  # type: ignore[union-attr]
    }


def _exact_entity_payload(
    reference: EntityRef,
    geometry: "GeometryModel",
    memo: dict[EntityRef, object],
) -> object:
    """Return an ID-independent, exact topology payload for one active entity."""

    cached = memo.get(reference)
    if cached is not None:
        return cached
    if reference.kind == "vertex":
        vertex = geometry.vertices[reference.id]
        payload: object = {
            "kind": "vertex",
            "position": vertex.position.tolist(),
        }
    elif reference.kind == "edge":
        edge = geometry.edges[reference.id]
        if isinstance(edge.curve, Straight):
            curve: object = {"type": "straight"}
        elif isinstance(edge.curve, Arc):
            curve = {
                "type": "arc",
                "via": _exact_entity_payload(
                    EntityRef("vertex", edge.curve.via_vertex), geometry, memo
                ),
            }
        elif isinstance(edge.curve, Spline):
            curve = {
                "type": "spline",
                "controls": [
                    _exact_entity_payload(EntityRef("vertex", item), geometry, memo)
                    for item in edge.curve.control_vertices
                ],
            }
        else:  # pragma: no cover - closed public union
            raise GeometryError(f"unsupported curve type {type(edge.curve).__name__}")
        payload = {
            "kind": "edge",
            "start": _exact_entity_payload(
                EntityRef("vertex", edge.start), geometry, memo
            ),
            "end": _exact_entity_payload(
                EntityRef("vertex", edge.end), geometry, memo
            ),
            "curve": curve,
        }
    else:
        face = geometry.faces[reference.id]

        def loop_payload(loop: Sequence[object]) -> list[object]:
            return [
                {
                    "edge": _exact_entity_payload(
                        EntityRef("edge", item.edge), geometry, memo  # type: ignore[attr-defined]
                    ),
                    "forward": bool(item.forward),  # type: ignore[attr-defined]
                }
                for item in loop
            ]

        payload = {
            "kind": "face",
            "loop": loop_payload(face.loop),
            "corners": list(face.corners),
            "holes": [loop_payload(loop) for loop in face.holes],
            "surface": _closure_surface(face.surface),
            "parameterization": _closure_surface(face.parameterization),
            "metadata": face.metadata.to_dict(),  # type: ignore[union-attr]
        }
    memo[reference] = payload
    return payload


def _exact_outputs_equal(
    old_outputs: Mapping[str, EntityRef],
    old_geometry: "GeometryModel",
    new_outputs: Mapping[str, EntityRef],
    new_geometry: "GeometryModel",
) -> bool:
    """Compare output slots and complete closures without ID/proximity matching."""

    if old_outputs.keys() != new_outputs.keys():
        return False
    old_memo: dict[EntityRef, object] = {}
    new_memo: dict[EntityRef, object] = {}
    for key in sorted(old_outputs):
        old_resolved = old_geometry.resolve_ref(old_outputs[key])
        new_resolved = new_geometry.resolve_ref(new_outputs[key])
        if len(old_resolved) != len(new_resolved):
            return False
        old_payloads = [
            _exact_entity_payload(item, old_geometry, old_memo)
            for item in old_resolved
        ]
        new_payloads = [
            _exact_entity_payload(item, new_geometry, new_memo)
            for item in new_resolved
        ]
        encode = lambda value: json.dumps(  # noqa: E731
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        if sorted(map(encode, old_payloads)) != sorted(map(encode, new_payloads)):
            return False
    return True


def _entity_closure_refs(
    references: Iterable[EntityRef], geometry: "GeometryModel"
) -> set[EntityRef]:
    """Return exact active topology closure without crossing replacement by proximity."""

    selected: set[EntityRef] = set()
    pending = list(references)
    while pending:
        reference = pending.pop()
        for active in geometry.resolve_ref(reference):
            if active in selected:
                continue
            selected.add(active)
            if active.kind == "face":
                face = geometry.faces[active.id]
                pending.extend(
                    EntityRef("edge", item.edge)
                    for loop in (face.loop,) + tuple(face.holes)
                    for item in loop
                )
            elif active.kind == "edge":
                edge = geometry.edges[active.id]
                pending.extend(
                    (EntityRef("vertex", edge.start), EntityRef("vertex", edge.end))
                )
                if isinstance(edge.curve, Arc):
                    pending.append(EntityRef("vertex", edge.curve.via_vertex))
                elif isinstance(edge.curve, Spline):
                    pending.extend(
                        EntityRef("vertex", item)
                        for item in edge.curve.control_vertices
                    )
    return selected


def _face_corner_vertex_ids(face: Face, geometry: "GeometryModel") -> tuple[int, ...]:
    """Resolve mapped corner loop positions to their oriented start vertices."""

    made: list[int] = []
    for position in face.corners:
        if position < 0 or position >= len(face.loop):
            raise GeometryError(
                f"face {face.id} mapped corner position {position} is outside its loop"
            )
        oriented = face.loop[position]
        edge = geometry.edges.get(oriented.edge)
        if edge is None:
            raise GeometryError(
                f"face {face.id} corner references missing edge {oriented.edge}"
            )
        made.append(edge.start if oriented.forward else edge.end)
    return tuple(made)


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
        self._baseline: Dict[str, Any] | None = (
            None if baseline is None else deepcopy(dict(baseline))
        )
        self._records: list[FeatureRecord] = [deepcopy(item) for item in records]
        if isinstance(next_id, bool) or not isinstance(next_id, Integral):
            raise GeometryError("next feature ID must be a positive integer")
        self._next_id = int(next_id)
        self._owner: GeometryModel | None = None
        self.validate()

    def _bind_owner(self, owner: "GeometryModel") -> None:
        """Bind this history to its sole document owner without publishing."""

        if self._owner is not None and self._owner is not owner:
            raise GeometryError("a feature history cannot belong to two models")
        self._owner = owner

    @property
    def baseline(self) -> Dict[str, Any] | None:
        """Detached baseline copy; use ``capture_baseline`` to replace it."""

        return deepcopy(self._baseline)

    @property
    def records(self) -> list[FeatureRecord]:
        """Detached records; persistent edits go through owner-aware methods."""

        return deepcopy(self._records)

    @property
    def next_id(self) -> int:
        return self._next_id

    def snapshot(self) -> Dict[str, Any]:
        return {
            "baseline": deepcopy(self._baseline),
            "records": deepcopy(self._records),
            "next_id": self._next_id,
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        """Atomically restore feature state and notify an attached owner once."""

        self._apply_mutation(lambda: self._restore_unchecked(snapshot))

    def _restore_unchecked(self, snapshot: Mapping[str, Any]) -> None:
        self._baseline = deepcopy(snapshot.get("baseline"))
        self._records = deepcopy(list(snapshot.get("records", ())))
        raw_next = snapshot.get("next_id", 1)
        if isinstance(raw_next, bool) or not isinstance(raw_next, Integral):
            raise GeometryError("next feature ID must be a positive integer")
        self._next_id = int(raw_next)

    def _apply_mutation(self, operation: Callable[[], Any]) -> Any:
        """Run one history edit with rollback and one owner publication."""

        before = self.snapshot()
        if self._owner is not None:
            self._owner._feature_history_will_change(self)  # noqa: SLF001
        try:
            result = operation()
            self.validate()
        except Exception:
            self._restore_unchecked(before)
            raise
        if not _values_equal(before, self.snapshot()) and self._owner is not None:
            self._owner._feature_history_did_change(self)  # noqa: SLF001
        return result

    def capture_baseline(self, geometry: "GeometryModel", *, force: bool = False) -> None:
        """Capture current materialized topology before the first feature."""

        if self._owner is not None and geometry is not self._owner:
            raise GeometryError("a feature baseline must be captured from its owner model")
        if self._baseline is not None and not force:
            return
        from .serialization import to_dict

        baseline = to_dict(geometry, include_features=False)
        self._apply_mutation(lambda: setattr(self, "_baseline", deepcopy(baseline)))

    def _capture_baseline_unchecked(
        self, geometry: "GeometryModel", *, force: bool = False
    ) -> None:
        """Loader-only baseline capture without revision publication."""

        if self._baseline is not None and not force:
            return
        from .serialization import to_dict

        self._baseline = to_dict(geometry, include_features=False)

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
        def apply() -> None:
            self._records.append(record)
            self._next_id += 1

        self._apply_mutation(apply)
        return deepcopy(record)

    def get(self, feature_id: int) -> FeatureRecord:
        """Return a detached feature record safe for inspection."""

        return deepcopy(self._get_record(feature_id))

    def _get_record(self, feature_id: int) -> FeatureRecord:
        for record in self._records:
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
        if not any(item.feature_id == wanted for item in self._records):
            raise KeyError(f"no feature {wanted}")
        earliest = min(
            index
            for index, item in enumerate(self._records)
            if item.feature_id in removed
        )

        def apply() -> None:
            self._records[:] = [
                item for item in self._records if item.feature_id not in removed
            ]
            if self._records:
                marker = self._records[min(earliest, len(self._records) - 1)]
                marker.state = FeatureStatus.PENDING.value
                marker.diagnostic = _FULL_REPLAY_DIAGNOSTIC

        self._apply_mutation(apply)
        return tuple(sorted(removed))

    def set_suppressed(self, feature_id: int, suppressed: bool = True) -> None:
        record = self._get_record(feature_id)
        made = bool(suppressed)
        if record.suppressed == made:
            return

        def apply() -> None:
            record.suppressed = made
            record.state = FeatureStatus.PENDING.value
            record.diagnostic = None

        self._apply_mutation(apply)

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

        record = self._get_record(feature_id)

        def apply() -> None:
            before = (
                record.name,
                deepcopy(record.parameters),
                deepcopy(record.inputs),
                tuple(record.dependencies),
            )
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
                record.dependencies = tuple(int(item) for item in dependencies)
            after = (
                record.name,
                record.parameters,
                record.inputs,
                record.dependencies,
            )
            if not _values_equal(before, after):
                record.state = FeatureStatus.PENDING.value
                record.diagnostic = None

        self._apply_mutation(apply)
        return deepcopy(record)

    def adopt_frozen(
        self,
        geometry: "GeometryModel",
        *,
        kind: str,
        outputs: Mapping[str, EntityRef],
        name: str | None = None,
        parameters: Mapping[str, Any] | None = None,
        inputs: Mapping[str, Sequence[FeatureInputRef]] | None = None,
        kind_version: int = 1,
        dependencies: Sequence[int] = (),
        expected_checksum: str | None = None,
        diagnostic: str | None = None,
    ) -> FeatureRecord:
        """Append an exact, already-materialized feature as frozen.

        This is the public bridge for trusted importers or script features that
        materialize topology through owner-controlled geometry operations but
        have no replay executor in ANYgeometry.  Output references must name
        active entities in ``geometry`` exactly; replacement lineage is never
        followed while adopting a binding.  The canonical closure checksum is
        computed by ANYgeometry and persisted with the record.

        When this history belongs to a model, ``geometry`` must be that owner.
        Feature-ID allocation, record insertion, checksum verification,
        validation, rollback, model revision, and change notification are one
        owner-aware atomic edit.
        ``expected_checksum`` is optional, but lets an importer fail closed if
        the materialization changed between its own validation and adoption.
        """

        from .model import GeometryModel

        if not isinstance(geometry, GeometryModel):
            raise TypeError("frozen feature adoption needs a GeometryModel")
        if self._owner is not None and geometry is not self._owner:
            raise GeometryError(
                "a frozen feature materialization must belong to its history owner"
            )
        if not isinstance(outputs, Mapping):
            raise GeometryError("frozen feature outputs must be an object")

        made: Dict[str, EntityRef] = {}
        for key, reference in outputs.items():
            if not isinstance(key, str) or not key or "\x00" in key:
                raise GeometryError(
                    "frozen feature output keys must be non-empty strings without NUL"
                )
            _validate_entity_ref(reference, f"frozen feature output {key!r}")
            store = {
                "vertex": geometry.vertices,
                "edge": geometry.edges,
                "face": geometry.faces,
            }[reference.kind]
            if reference.id not in store:
                raise GeometryError(
                    f"frozen feature output {key!r} does not reference an active "
                    f"{reference.kind} {reference.id}; adoption never follows lineage"
                )
            made[key] = reference
        if not made:
            raise GeometryError("a frozen feature needs at least one output")

        if expected_checksum is not None:
            if (
                not isinstance(expected_checksum, str)
                or len(expected_checksum) != 64
                or expected_checksum.lower() != expected_checksum
                or any(character not in "0123456789abcdef" for character in expected_checksum)
            ):
                raise GeometryError(
                    "expected frozen materialization checksum must be lowercase SHA-256"
                )
        if diagnostic is not None and (
            not isinstance(diagnostic, str) or not diagnostic.strip() or "\x00" in diagnostic
        ):
            raise GeometryError(
                "frozen feature diagnostic must be a non-empty string without NUL"
            )

        integrity_errors = (
            *geometry.validate_topology(),
            *geometry._validate_structural(),  # noqa: SLF001
        )
        if integrity_errors:
            raise GeometryError(
                "cannot adopt a frozen materialization from an invalid owner model: "
                + "; ".join(integrity_errors)
            )

        record = FeatureRecord(
            feature_id=self._next_id,
            kind=str(kind),
            name=name or str(kind),
            parameters={} if parameters is None else dict(parameters),
            inputs={
                str(port): tuple(references)
                for port, references in ({} if inputs is None else inputs).items()
            },
            kind_version=kind_version,
            dependencies=tuple(dependencies),
            outputs=dict(made),
            state=FeatureStatus.FROZEN.value,
        )
        frozen_diagnostic = diagnostic or (
            f"feature kind {record.kind!r} uses an explicitly adopted, verified "
            "frozen materialization; regeneration requires a compatible executor"
        )
        record.diagnostic = frozen_diagnostic

        def apply() -> None:
            self._records.append(record)
            self._next_id += 1
            record.materialization_checksum = self.materialization_checksum(
                record, geometry
            )
            if (
                expected_checksum is not None
                and record.materialization_checksum != expected_checksum
            ):
                raise GeometryError(
                    f"feature {record.feature_id} frozen materialization checksum "
                    "does not match the expected checksum"
                )
            invalid = self.validate_materialization(record, geometry)
            if invalid is not None:
                raise GeometryError(
                    f"cannot adopt feature {record.feature_id} materialization: {invalid}"
                )

        self._apply_mutation(apply)
        return deepcopy(record)

    def move(self, feature_id: int, index: int) -> None:
        record = self._get_record(feature_id)
        old = self._records.index(record)
        target = max(0, min(int(index), len(self._records) - 1))
        if target == old:
            return

        def apply() -> None:
            self._records.pop(old)
            self._records.insert(target, record)
            record.state = FeatureStatus.PENDING.value
            record.diagnostic = None

        self._apply_mutation(apply)

    def dependencies_of(self, record: FeatureRecord | int) -> tuple[int, ...]:
        made = self._get_record(record) if isinstance(record, int) else record
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
                for item in self._records
                if parent in self.dependencies_of(item) and item.feature_id not in found
            ]
            found.extend(direct)
            if transitive:
                pending.extend(direct)
        return tuple(found)

    def validate(self) -> None:
        """Validate every persisted field after possible hostile tampering."""

        _positive_int(self._next_id, "next feature ID")
        if self._baseline is not None:
            if not isinstance(self._baseline, Mapping):
                raise GeometryError("feature-history baseline must be a geometry object")
            if "features" in self._baseline:
                raise GeometryError("feature-history baseline cannot contain another history")
            _validate_json_value(self._baseline, "feature-history baseline")

        seen: set[int] = set()
        for position, record in enumerate(self._records):
            if not isinstance(record, FeatureRecord):
                raise GeometryError(
                    f"feature record {position} has type {type(record).__name__}, "
                    "expected FeatureRecord"
                )
            feature_id = _positive_int(record.feature_id, "feature ID")
            if feature_id in seen:
                raise GeometryError(f"duplicate feature ID {feature_id}")
            if not isinstance(record.kind, str) or not record.kind.strip():
                raise GeometryError(f"feature {feature_id} kind must be a non-empty string")
            if not isinstance(record.name, str) or not record.name.strip():
                raise GeometryError(f"feature {feature_id} name must be a non-empty string")
            if "\x00" in record.kind or "\x00" in record.name:
                raise GeometryError(f"feature {feature_id} kind/name cannot contain NUL")
            _positive_int(record.kind_version, f"feature {feature_id} kind version")
            if not isinstance(record.suppressed, bool):
                raise GeometryError(f"feature {feature_id} suppressed must be boolean")
            if not isinstance(record.parameters, Mapping):
                raise GeometryError(f"feature {feature_id} parameters must be an object")
            _validate_json_value(record.parameters, f"feature {feature_id} parameters")

            if not isinstance(record.dependencies, (tuple, list)):
                raise GeometryError(f"feature {feature_id} dependencies must be ordered")
            dependencies = tuple(
                _positive_int(item, f"feature {feature_id} dependency")
                for item in record.dependencies
            )
            if len(set(dependencies)) != len(dependencies):
                raise GeometryError(f"feature {feature_id} has duplicate dependencies")

            if not isinstance(record.inputs, Mapping):
                raise GeometryError(f"feature {feature_id} inputs must be an object")
            for port, references in record.inputs.items():
                if not isinstance(port, str) or not port:
                    raise GeometryError(
                        f"feature {feature_id} input ports must be non-empty strings"
                    )
                if not isinstance(references, (tuple, list)):
                    raise GeometryError(
                        f"feature {feature_id} input port {port!r} must be ordered"
                    )
                for reference in references:
                    if isinstance(reference, EntityRef):
                        _validate_entity_ref(
                            reference, f"feature {feature_id} input port {port!r}"
                        )
                    elif isinstance(reference, FeatureOutputRef):
                        _validate_feature_output_ref(
                            reference, f"feature {feature_id} input port {port!r}"
                        )
                    else:
                        raise GeometryError(
                            f"feature {feature_id} has an invalid input reference"
                        )
                try:
                    unique_references = set(references)
                except TypeError as error:
                    raise GeometryError(
                        f"feature {feature_id} input port {port!r} is not hashable"
                    ) from error
                if len(unique_references) != len(references):
                    raise GeometryError(
                        f"feature {feature_id} input port {port!r} has duplicate references"
                    )

            if not isinstance(record.outputs, Mapping):
                raise GeometryError(f"feature {feature_id} outputs must be an object")
            for key, reference in record.outputs.items():
                if not isinstance(key, str) or not key:
                    raise GeometryError(
                        f"feature {feature_id} output keys must be non-empty strings"
                    )
                _validate_entity_ref(reference, f"feature {feature_id} output {key!r}")

            try:
                FeatureStatus(record.state)
            except (TypeError, ValueError) as error:
                raise GeometryError(
                    f"feature {feature_id} state contains an unknown enum value"
                ) from error
            if record.diagnostic is not None and not isinstance(record.diagnostic, str):
                raise GeometryError(
                    f"feature {feature_id} diagnostic must be a string or null"
                )
            checksum = record.materialization_checksum
            if checksum is not None and (
                not isinstance(checksum, str)
                or len(checksum) != 64
                or checksum.lower() != checksum
                or any(character not in "0123456789abcdef" for character in checksum)
            ):
                raise GeometryError(
                    f"feature {feature_id} materialization checksum must be lowercase SHA-256"
                )

            for dependency in self.dependencies_of(record):
                if dependency not in seen:
                    raise GeometryError(
                        f"feature {feature_id} depends on later or missing "
                        f"feature {dependency}"
                    )
            seen.add(feature_id)
        if self._next_id <= max(seen, default=0):
            raise GeometryError("next feature ID would reuse an existing feature ID")

    def validate_persistence(self, geometry: "GeometryModel") -> None:
        """Reject history that the strict loader could not restore faithfully."""

        self.validate()
        registry = builtin_feature_registry()
        for record in self._records:
            known = registry.has(record.kind)
            if known and record.state in ("ok", "active") and not record.outputs:
                raise GeometryError(
                    f"active feature {record.feature_id} has no materialized outputs"
                )
            for key, reference in record.outputs.items():
                if not geometry.resolve_ref(reference):
                    raise GeometryError(
                        f"feature {record.feature_id} output {key!r} references "
                        f"missing entity {reference}"
                    )
            if known or record.suppressed:
                continue

            # The strict loader can preserve an unavailable executor only as
            # an explicitly frozen, checksummed last-good materialization.
            # Requiring its canonical state here makes every accepted current-schema
            # document stable under a to_dict -> from_dict -> to_dict cycle.
            if not record.outputs or record.materialization_checksum is None:
                raise GeometryError(
                    f"unknown feature {record.feature_id} requires verified "
                    "last-good outputs and a materialization checksum"
                )
            diagnostic = self.validate_materialization(record, geometry)
            if diagnostic is not None:
                raise GeometryError(
                    f"unknown feature {record.feature_id} has no verified "
                    f"last-good materialization: {diagnostic}"
                )

    def resolve(
        self, reference: FeatureInputRef, geometry: "GeometryModel"
    ) -> tuple[EntityRef, ...]:
        if isinstance(reference, EntityRef):
            return geometry.resolve_ref(reference)
        record = self._get_record(reference.feature_id)
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

        made = self._get_record(record) if isinstance(record, int) else record
        selected: dict[str, set[int]] = {kind: set() for kind in _KINDS}
        resolved_outputs: dict[str, list[list[object]]] = {}

        for key, reference in sorted(made.outputs.items()):
            current = geometry.resolve_ref(reference)
            resolved_outputs[key] = [[item.kind, item.id] for item in current]
            for item in current:
                selected[item.kind].add(item.id)

        for face_id in tuple(selected["face"]):
            face = geometry.faces.get(face_id)
            if face is None:
                continue
            for loop in (face.loop,) + tuple(face.holes):
                selected["edge"].update(item.edge for item in loop)
            # ``Face.corners`` contains loop positions, not entity IDs.  The
            # oriented-start translation is explicit even though the complete
            # edge closure below normally contributes the same vertices.
            selected["vertex"].update(_face_corner_vertex_ids(face, geometry))

        for edge_id in tuple(selected["edge"]):
            edge = geometry.edges.get(edge_id)
            if edge is None:
                continue
            selected["vertex"].update((edge.start, edge.end))
            if isinstance(edge.curve, Arc):
                selected["vertex"].add(edge.curve.via_vertex)
            elif isinstance(edge.curve, Spline):
                selected["vertex"].update(edge.curve.control_vertices)

        selected_refs = {
            (kind, identifier)
            for kind, identifiers in selected.items()
            for identifier in identifiers
        }
        payload = {
            "outputs": resolved_outputs,
            "vertices": [
                _closure_vertex(geometry.vertices[item])
                for item in sorted(selected["vertex"])
                if item in geometry.vertices
            ],
            "edges": [
                _closure_edge(geometry.edges[item])
                for item in sorted(selected["edge"])
                if item in geometry.edges
            ],
            "faces": [
                _closure_face(geometry.faces[item])
                for item in sorted(selected["face"])
                if item in geometry.faces
            ],
            "groups": {
                name: [
                    [reference.kind, reference.id]
                    for reference in geometry.group(name, resolve=False)
                    if (reference.kind, reference.id) in selected_refs
                ]
                for name in sorted(geometry.groups)
                if any(
                    (reference.kind, reference.id) in selected_refs
                    for reference in geometry.group(name, resolve=False)
                )
            },
            "tags": [
                {
                    "entity": [reference.kind, reference.id],
                    "values": sorted(values),
                }
                for reference, values in sorted(
                    geometry.tags.items(),
                    key=lambda item: (item[0].kind, item[0].id),
                )
                if (reference.kind, reference.id) in selected_refs
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

        made = self._get_record(record) if isinstance(record, int) else record
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

    @staticmethod
    def _incremental_additive(kind: str) -> bool:
        return kind in _INCREMENTAL_ADDITIVE_FEATURES or kind.startswith("generator.")

    def _incremental_start(self, dirty_index: int) -> tuple[int, bool] | None:
        """Return a safe replay start and whether the suffix needs fresh closure."""

        start = dirty_index
        force_new = False
        positions = {
            record.feature_id: index for index, record in enumerate(self._records)
        }
        while True:
            earlier = start
            for record in self._records[start:]:
                if self._incremental_additive(record.kind) or record.suppressed:
                    continue
                if record.kind not in _INCREMENTAL_MUTATING_FEATURES:
                    return None
                force_new = True
                for references in record.inputs.values():
                    for reference in references:
                        if isinstance(reference, FeatureOutputRef):
                            earlier = min(earlier, positions[reference.feature_id])
                        elif record.outputs:
                            # A materialized modifier with a raw topology input
                            # has no exact pre-feature checkpoint to replay.
                            return None
            if earlier == start:
                return start, force_new
            start = earlier

    def _earliest_dirty_index(self) -> int | None:
        for index, record in enumerate(self._records):
            clean_state = (
                FeatureStatus.SUPPRESSED.value
                if record.suppressed
                else FeatureStatus.OK.value
            )
            if record.state not in (clean_state, FeatureStatus.ACTIVE.value):
                return index
        return None

    def _regenerate_incremental(
        self,
        geometry: "GeometryModel",
        registry: FeatureRegistry,
        dirty_index: int,
        *,
        force_new: bool = False,
    ) -> RegenerationReport:
        """Replay one additive dirty suffix on a clone of live materialization.

        Clean prefix entities stay in the clone with their active IDs.  Each
        dirty feature executes above the owner's allocator high-water mark.
        Stable output keys and exact ID-independent closure payloads decide
        whether the old binding is retained; no spatial or tolerance match is
        involved.
        """

        from .serialization import from_dict, to_dict

        working = from_dict(to_dict(geometry, include_features=False))
        working.reserve_id_state(geometry.id_state())
        replayed = deepcopy(self._records)
        by_id = {record.feature_id: record for record in replayed}
        previous_outputs = {
            (record.feature_id, key): geometry.resolve_ref(reference)
            for record in self._records
            for key, reference in record.outputs.items()
        }
        results = [
            FeatureResult(
                record.feature_id,
                record.state,
                dict(record.outputs),
                record.diagnostic,
            )
            for record in replayed[:dirty_index]
        ]
        discard_new: set[EntityRef] = set()
        retire_old: set[EntityRef] = set()

        for record, previous in zip(
            replayed[dirty_index:], self._records[dirty_index:]
        ):
            old_outputs = dict(previous.outputs)
            record.diagnostic = None
            if record.suppressed:
                record.outputs = {}
                record.state = FeatureStatus.SUPPRESSED.value
                for reference in old_outputs.values():
                    retire_old.update(_entity_closure_refs((reference,), geometry))
                results.append(FeatureResult(record.feature_id, record.state))
                continue

            resolved: Dict[str, tuple[EntityRef, ...]] = {}
            blocked = next(
                (
                    f"dependency {dependency} is not active"
                    for dependency in record.dependencies
                    if by_id[dependency].state != FeatureStatus.OK.value
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
                            if source.state != FeatureStatus.OK.value
                            or materialized is None
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
                record.outputs = {}
                record.state = FeatureStatus.BLOCKED.value
                record.diagnostic = blocked
                for reference in old_outputs.values():
                    retire_old.update(_entity_closure_refs((reference,), geometry))
                results.append(
                    FeatureResult(record.feature_id, record.state, diagnostic=blocked)
                )
                continue

            snapshot = working.topology_snapshot()
            try:
                execution = registry.execute(working, record, resolved)
                outputs = {str(key): value for key, value in execution.outputs.items()}
                for key, reference in outputs.items():
                    if not key or not isinstance(reference, EntityRef):
                        raise GeometryError(
                            f"feature {record.feature_id} returned an invalid output"
                        )
                    store = {
                        "vertex": working.vertices,
                        "edge": working.edges,
                        "face": working.faces,
                    }[reference.kind]
                    if reference.id not in store:
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

            if not force_new and old_outputs and _exact_outputs_equal(
                old_outputs, geometry, outputs, working
            ):
                record.outputs = old_outputs
                for reference in outputs.values():
                    discard_new.update(_entity_closure_refs((reference,), working))
            else:
                record.outputs = outputs
                for reference in old_outputs.values():
                    retire_old.update(_entity_closure_refs((reference,), geometry))
            record.state = FeatureStatus.OK.value
            results.append(
                FeatureResult(record.feature_id, record.state, dict(record.outputs))
            )

        transition_by_old: Dict[EntityRef, tuple[EntityRef, ...]] = {}
        new_by_id = {record.feature_id: record for record in replayed}
        for (feature_id, key), old_items in previous_outputs.items():
            new_binding = new_by_id[feature_id].outputs.get(key)
            descendants = (
                () if new_binding is None else working.resolve_ref(new_binding)
            )
            for old in old_items:
                same_kind = tuple(item for item in descendants if item.kind == old.kind)
                if old not in same_kind:
                    transition_by_old[old] = same_kind
        transitions = tuple(transition_by_old.items())

        kept = _entity_closure_refs(
            (
                reference
                for record in replayed
                for reference in record.outputs.values()
            ),
            working,
        )
        protected_inputs: set[EntityRef] = set()
        for record in replayed:
            for references in record.inputs.values():
                for reference in references:
                    if isinstance(reference, EntityRef):
                        current = working.resolve_ref(reference)
                    else:
                        source = new_by_id[reference.feature_id]
                        binding = source.outputs.get(reference.output_key)
                        current = () if binding is None else working.resolve_ref(binding)
                    protected_inputs.update(_entity_closure_refs(current, working))
        kept.update(protected_inputs)
        conflicting = [
            old
            for old, descendants in transitions
            if old in kept and old not in descendants
        ]
        if conflicting:
            return RegenerationReport(
                False,
                tuple(results),
                diagnostic=(
                    "cannot replace feature output shared by a preserved output: "
                    + ", ".join(map(str, conflicting))
                ),
            )
        discard_new.difference_update(kept)
        retire_old.difference_update(kept)

        def remove_exact(references: Iterable[EntityRef]) -> None:
            selected = set(references)
            for reference in sorted(
                selected,
                key=lambda item: (
                    {"face": 0, "edge": 1, "vertex": 2}[item.kind],
                    -item.id,
                ),
            ):
                store = {
                    "vertex": working.vertices,
                    "edge": working.edges,
                    "face": working.faces,
                }[reference.kind]
                if reference.id not in store:
                    continue
                if reference.kind == "face":
                    working.remove_face(reference.id, record=False)
                elif reference.kind == "edge":
                    working.remove_edge(reference.id, record=False)
                else:
                    working.remove_vertex(reference.id, record=False)

        try:
            with working.transaction():
                remove_exact(discard_new)
                remove_exact(retire_old)
                working.record_replacements_atomic(transitions)
        except GeometryError as error:
            return RegenerationReport(
                False,
                tuple(results),
                diagnostic=f"cannot preserve feature lineage: {error}",
            )

        for record in replayed:
            record.materialization_checksum = (
                self.materialization_checksum(record, working)
                if record.state == FeatureStatus.OK.value and record.outputs
                else None
            )
        staged = FeatureHistory(
            baseline=self._baseline,
            records=replayed,
            next_id=self._next_id,
        )
        geometry.restore_design(
            {
                "topology": working.topology_snapshot(),
                "features": staged.snapshot(),
            }
        )
        return RegenerationReport(True, tuple(results), transitions)

    def regenerate(
        self,
        geometry: "GeometryModel",
        registry: FeatureRegistry,
    ) -> RegenerationReport:
        """Atomically replay the history from its base materialization."""

        self.validate()
        unknown = [
            record
            for record in self._records
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

        dirty_index = self._earliest_dirty_index()
        force_full_replay = any(
            record.diagnostic == _FULL_REPLAY_DIAGNOSTIC
            for record in self._records
        )
        if dirty_index is None and self._records:
            return RegenerationReport(
                True,
                tuple(
                    FeatureResult(
                        record.feature_id,
                        record.state,
                        dict(record.outputs),
                        record.diagnostic,
                    )
                    for record in self._records
                ),
            )
        dirty_index = 0 if dirty_index is None else dirty_index
        incremental = None if force_full_replay else self._incremental_start(dirty_index)
        if incremental is not None:
            start, force_new = incremental
            if not (start == 0 and force_new):
                incremental_report = self._regenerate_incremental(
                    geometry, registry, start, force_new=force_new
                )
                if (
                    incremental_report.success
                    or "cannot transfer face ownership: replacement face(s) are already owned"
                    not in (incremental_report.diagnostic or "")
                ):
                    return incremental_report
                # Structural generators materialize their own fresh Sheet.
                # Incremental replay temporarily contains both that owner and
                # the old generated owner, so lineage transfer is ambiguous.
                # Retry the already-detached operation from the exact feature
                # baseline, where only the new generator owner exists.  This
                # is identity/history replay, never a proximity repair.
                force_full_replay = True
                dirty_index = 0
        if dirty_index > 0 and not force_full_replay:
            return RegenerationReport(
                False,
                tuple(
                    FeatureResult(
                        record.feature_id,
                        record.state,
                        dict(record.outputs),
                        record.diagnostic,
                    )
                    for record in self._records
                ),
                diagnostic=(
                    "regeneration requires an exact pre-feature checkpoint: "
                    "a dirty mutating feature uses raw topology after a clean prefix; "
                    "bind feature-owned inputs with FeatureOutputRef"
                ),
            )
        from .serialization import from_dict, to_dict

        baseline = (
            to_dict(geometry, include_features=False)
            if self._baseline is None
            else deepcopy(self._baseline)
        )

        previous_history = geometry.replacement_history()
        previous_outputs = {
            (record.feature_id, key): geometry.resolve_ref(reference)
            for record in self._records
            for key, reference in record.outputs.items()
        }
        working = from_dict(deepcopy(baseline))
        working.reserve_id_state(geometry.id_state())
        replayed = deepcopy(self._records)
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

        for record in replayed:
            record.materialization_checksum = (
                self.materialization_checksum(record, working)
                if record.state == "ok" and record.outputs
                else None
            )
        staged = FeatureHistory(
            baseline=baseline,
            records=replayed,
            next_id=self._next_id,
        )
        geometry.restore_design(
            {
                "topology": working.topology_snapshot(),
                "features": staged.snapshot(),
            }
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

    def rectangular_pattern_feature(geometry, feature, inputs):
        from .editing import rectangular_pattern

        result = rectangular_pattern(
            geometry,
            inputs.get("entities", ()),
            feature.parameters["directions"],
            feature.parameters["spacings"],
            feature.parameters["counts"],
        )
        return {
            f"instance/{index}/{key}": reference
            for index, instance in enumerate(result.instances)
            for key, reference in instance.outputs.items()
        }

    def transform_pattern_feature(geometry, feature, inputs):
        from .editing import pattern_entities

        result = pattern_entities(
            geometry,
            inputs.get("entities", ()),
            feature.parameters["matrices"],
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
        "geometry.pattern.rectangular": rectangular_pattern_feature,
        "geometry.pattern.transforms": transform_pattern_feature,
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
