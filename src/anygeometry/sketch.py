"""Persistent planar sketches with a small, audited constraint solver."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Mapping, Sequence, TYPE_CHECKING
from uuid import uuid4

import numpy as np

from .errors import GeometryError
from .entities import EntityRef

if TYPE_CHECKING:  # pragma: no cover
    from .model import GeometryModel

__all__ = [
    "SketchConstraint", "SketchDefinition", "SketchPlane",
    "face_sketch_plane", "plane_sketch_plane", "materialize_sketch",
    "materialize_sketch_on_plane", "solve_sketch",
]

ConstraintKind = Literal["distance", "coincident", "on_edge", "on_vertex", "horizontal", "vertical", "fixed"]


def _finite_pair(value: Sequence[float], label: str) -> tuple[float, float]:
    values = np.asarray(value, dtype=float)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise GeometryError(f"{label} needs two finite plane coordinates")
    return float(values[0]), float(values[1])


@dataclass(frozen=True)
class SketchConstraint:
    """A distance, point coincidence, or supporting-boundary coincidence."""

    kind: ConstraintKind
    first: str
    second: str | None = None
    value: float | None = None
    boundary_index: int | None = None
    id: str | None = None
    position: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("distance", "coincident", "on_edge", "on_vertex", "horizontal", "vertical", "fixed"):
            raise GeometryError(f"unknown sketch constraint {self.kind!r}")
        if not str(self.first).strip():
            raise GeometryError("a sketch constraint needs a first point")
        if self.kind in ("distance", "coincident", "horizontal", "vertical"):
            if not self.second or self.second == self.first:
                raise GeometryError(f"sketch {self.kind} needs two different points")
        if self.kind == "distance":
            distance = float(self.value) if self.value is not None else -1.0
            if not np.isfinite(distance) or distance <= 0.0:
                raise GeometryError("a sketch distance must be finite and positive")
            object.__setattr__(self, "value", distance)
        if self.kind in ("on_edge", "on_vertex"):
            index = int(self.boundary_index) if self.boundary_index is not None else -1
            if index < 0:
                raise GeometryError(f"sketch {self.kind} needs a boundary index")
            object.__setattr__(self, "boundary_index", index)
        if self.position is not None:
            object.__setattr__(self, 'position', _finite_pair(self.position, 'fixed position'))

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"kind": self.kind, "first": self.first}
        if self.id is not None:
            result["id"] = self.id
        if self.position is not None:
            result['position'] = list(self.position)
        if self.second is not None:
            result["second"] = self.second
        if self.value is not None:
            result["value"] = float(self.value)
        if self.boundary_index is not None:
            result["boundary_index"] = int(self.boundary_index)
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SketchConstraint":
        return cls(
            kind=str(data.get("kind", "")),  # type: ignore[arg-type]
            first=str(data.get("first", "")),
            second=None if data.get("second") is None else str(data["second"]),
            value=None if data.get("value") is None else float(data["value"]),
            boundary_index=None if data.get("boundary_index") is None else int(data["boundary_index"]),
            id=None if data.get("id") is None else str(data["id"]),
            position=None if data.get('position') is None else tuple(data['position']),
        )


@dataclass(frozen=True)
class SketchDefinition:
    """JSON-safe editable sketch intent."""

    points: Mapping[str, tuple[float, float]]
    path: tuple[str, ...] = ()
    constraints: tuple[SketchConstraint, ...] = ()
    closed: bool = True
    extrusion: float = 0.0
    version: int = 1
    entities: tuple = ()
    wires: tuple = ()
    references: tuple = ()
    # Transient: identities of solve-time reference anchors in a bound definition.
    # Never persisted; legacy user points that merely start with "ref:" are not listed.
    bound_reference_keys: frozenset = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bound_reference_keys", frozenset(str(key) for key in self.bound_reference_keys))
        points = {str(key): _finite_pair(value, f"sketch point {key!r}") for key, value in self.points.items()}
        if self.version not in (1, 2):
            raise GeometryError("unsupported sketch version")
        if self.version == 1 and len(points) < 2:
            raise GeometryError("a sketch needs at least two points")
        path = tuple(str(item) for item in self.path)
        minimum = 3 if self.closed else 2
        if self.version == 1 and len(path) < minimum:
            raise GeometryError(f"a sketch path needs at least {minimum} points")
        missing = [item for item in path if item not in points]
        if missing:
            raise GeometryError(f"sketch path references missing point {missing[0]!r}")
        if len(set(path)) != len(path):
            raise GeometryError("sketch path cannot repeat a point key")
        constraints = tuple(replace(c, position=points[c.first]) if c.kind=='fixed' and c.position is None and c.first in points else c for c in self.constraints)
        if self.version == 2:
            constraints = tuple(replace(c,id=str(uuid4())) if c.id is None else c for c in constraints)
            if len({c.id for c in constraints}) != len(constraints):
                raise GeometryError('sketch constraint IDs must be unique')
        from .sketch_references import SketchReference
        references = tuple(SketchReference.from_dict(item) for item in self.references)
        if references and self.version != 2:
            raise GeometryError("sketch references need a version 2 sketch")
        if len({item.id for item in references}) != len(references):
            raise GeometryError("sketch reference IDs must be unique")
        reference_keys = {key for item in references for key in item.keys}
        if reference_keys & set(points):
            raise GeometryError("sketch points cannot reuse a reference point key")
        for constraint in constraints:
            names = (constraint.first,) + (() if constraint.second is None else (constraint.second,))
            if constraint.kind == "fixed" and constraint.first in reference_keys:
                raise GeometryError("a reference point is already fixed; it cannot be constrained again")
            if any(name not in points and name not in reference_keys for name in names):
                raise GeometryError("sketch constraint references an unavailable point")
        object.__setattr__(self, "references", references)
        extrusion = float(self.extrusion)
        if not np.isfinite(extrusion):
            raise GeometryError("sketch extrusion must be finite")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "extrusion", extrusion)
        from .sketch_entities import SketchEntity, SketchWire
        entities = tuple(item if isinstance(item, SketchEntity) else SketchEntity.from_dict(item) for item in self.entities)
        wires = tuple(item if isinstance(item, SketchWire) else SketchWire.from_dict(item) for item in self.wires)
        if len({item.id for item in entities}) != len(entities) or len({item.id for item in wires}) != len(wires):
            raise GeometryError("sketch entity and contour IDs must be unique")
        if any(key not in points for item in entities for key in item.points):
            raise GeometryError("sketch entity references unavailable point")
        if any(key not in {item.id for item in entities} for wire in wires for key in wire.entities):
            raise GeometryError("sketch contour references unavailable entity")
        object.__setattr__(self, "entities", entities)
        object.__setattr__(self, "wires", wires)

    def to_parameters(self) -> dict[str, object]:
        result = {
            "points": {key: list(value) for key, value in self.points.items()},
            "path": list(self.path),
            "constraints": [item.to_dict() for item in self.constraints],
            "closed": bool(self.closed),
            "extrusion": float(self.extrusion),
        }
        if self.version == 2:
            result.update(version=2, entities=[item.to_dict() for item in self.entities], wires=[item.to_dict() for item in self.wires])
            if self.references:
                result["references"] = [item.to_dict() for item in self.references]
        return result

    @classmethod
    def from_parameters(cls, data: Mapping[str, object]) -> "SketchDefinition":
        raw_points = data.get("points", {})
        raw_constraints = data.get("constraints", ())
        if not isinstance(raw_points, Mapping):
            raise GeometryError("sketch points must be a mapping")
        if not isinstance(raw_constraints, Sequence) or isinstance(raw_constraints, (str, bytes)):
            raise GeometryError("sketch constraints must be a sequence")
        constraints: list[SketchConstraint] = []
        for item in raw_constraints:
            if not isinstance(item, Mapping):
                raise GeometryError("each sketch constraint must be a mapping")
            constraints.append(SketchConstraint.from_dict(item))
        return cls(
            points={str(key): _finite_pair(value, f"sketch point {key!r}") for key, value in raw_points.items()},
            path=tuple(str(item) for item in data.get("path", ())),
            constraints=tuple(constraints),
            closed=bool(data.get("closed", True)),
            extrusion=float(data.get("extrusion", 0.0)),
            version=int(data.get("version", 1)),
            entities=tuple(data.get("entities", ())),
            wires=tuple(data.get("wires", ())),
            references=tuple(data.get("references", ())),
        )


@dataclass(frozen=True)
class SketchPlane:
    origin: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    normal: np.ndarray
    boundary_vertices: tuple[np.ndarray, ...]
    boundary_edges: tuple[tuple[np.ndarray, np.ndarray], ...]

    def world(self, point: Sequence[float]) -> np.ndarray:
        u, v = _finite_pair(point, "sketch point")
        return self.origin + u * self.x_axis + v * self.y_axis

    def local(self, point: Sequence[float]) -> np.ndarray:
        value = np.asarray(point, dtype=float)
        delta = value - self.origin
        return np.asarray((delta @ self.x_axis, delta @ self.y_axis), dtype=float)


def plane_sketch_plane(
    origin: Sequence[float], normal: Sequence[float], x_axis: Sequence[float],
) -> SketchPlane:
    """Build a detached, right-handed sketch plane from persisted vectors."""

    centre = np.asarray(origin, dtype=float)
    axis = np.asarray(normal, dtype=float)
    reference = np.asarray(x_axis, dtype=float)
    if centre.shape != (3,) or not np.all(np.isfinite(centre)):
        raise GeometryError("sketch plane origin needs three finite components")
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise GeometryError("sketch plane normal needs three finite components")
    if reference.shape != (3,) or not np.all(np.isfinite(reference)):
        raise GeometryError("sketch plane x axis needs three finite components")
    axis_length = float(np.linalg.norm(axis))
    if axis_length <= 1.0e-12:
        raise GeometryError("sketch plane normal must be non-zero")
    axis = axis / axis_length
    reference = reference - float(reference @ axis) * axis
    reference_length = float(np.linalg.norm(reference))
    if reference_length <= 1.0e-12:
        raise GeometryError("sketch plane x axis must not be parallel to its normal")
    reference = reference / reference_length
    y_axis = np.cross(axis, reference)
    y_axis /= np.linalg.norm(y_axis)
    return SketchPlane(centre, reference, y_axis, axis, (), ())


def face_sketch_plane(geometry: "GeometryModel", face_id: int) -> SketchPlane:
    """Return a deterministic local frame and reject non-flat support faces."""

    face = geometry.faces[int(face_id)]
    samples = np.vstack([geometry.sample_edge(item.edge, np.linspace(0.0, 1.0, 9)) for item in face.loop])
    centre = samples.mean(axis=0)
    _left, singular, _vectors = np.linalg.svd(samples - centre)
    if float(singular[-1]) > 1.0e-9 * max(float(singular[0]), 1.0):
        raise GeometryError("a sketch support must be a flat plate")
    normal = np.asarray(geometry.face_normal(face_id, 0.5, 0.5), dtype=float)
    normal /= np.linalg.norm(normal)
    first = face.loop[0]
    origin = geometry.vertex_position(geometry.oriented_start_vertex(first))
    tangent = np.asarray(geometry.oriented_start_tangent(first), dtype=float)
    tangent -= float(tangent @ normal) * normal
    length = float(np.linalg.norm(tangent))
    if length <= 1.0e-12:
        raise GeometryError("the sketch support has a degenerate first edge")
    x_axis = tangent / length
    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    vertices = tuple(geometry.vertex_position(geometry.oriented_start_vertex(item)) for item in face.loop)
    local_vertices = tuple(np.asarray(((point - origin) @ x_axis, (point - origin) @ y_axis), dtype=float) for point in vertices)
    local_edges = tuple((local_vertices[index], local_vertices[(index + 1) % len(local_vertices)]) for index in range(len(local_vertices)))
    return SketchPlane(origin, x_axis, y_axis, normal, local_vertices, local_edges)


def _constraint_system(coordinates: np.ndarray, keys: tuple[str, ...], constraints: Sequence[SketchConstraint], plane: SketchPlane) -> tuple[np.ndarray, np.ndarray]:
    index = {key: position for position, key in enumerate(keys)}
    residuals: list[float] = []
    rows: list[np.ndarray] = []
    columns = 2 * len(keys)
    for item in constraints:
        first_index = index[item.first]
        first = coordinates[first_index]
        if item.kind in ("horizontal", "vertical"):
            axis = 1 if item.kind == "horizontal" else 0
            second_index = index[item.second or ""]
            row = np.zeros(columns); row[2 * first_index + axis] = 1.; row[2 * second_index + axis] = -1.
            rows.append(row); residuals.append(float(first[axis] - coordinates[second_index, axis]))
        elif item.kind == "coincident":
            second_index = index[item.second or ""]
            for axis in (0, 1):
                row = np.zeros(columns); row[2 * first_index + axis] = 1.0; row[2 * second_index + axis] = -1.0
                rows.append(row); residuals.append(float(first[axis] - coordinates[second_index, axis]))
        elif item.kind == "distance":
            second_index = index[item.second or ""]
            delta = first - coordinates[second_index]
            length = float(np.linalg.norm(delta))
            if length <= 1.0e-14:
                raise GeometryError(f"distance constraint {item.first}-{item.second} has no direction")
            direction = delta / length
            row = np.zeros(columns); row[2 * first_index:2 * first_index + 2] = direction; row[2 * second_index:2 * second_index + 2] = -direction
            rows.append(row); residuals.append(length - float(item.value))
        elif item.kind == "on_vertex":
            target = plane.boundary_vertices[int(item.boundary_index)]
            for axis in (0, 1):
                row = np.zeros(columns); row[2 * first_index + axis] = 1.0
                rows.append(row); residuals.append(float(first[axis] - target[axis]))
        elif item.kind == "on_edge":
            start, end = plane.boundary_edges[int(item.boundary_index)]
            direction = end - start; length = float(np.linalg.norm(direction))
            if length <= 1.0e-14:
                raise GeometryError("the referenced sketch boundary edge is degenerate")
            normal = np.asarray((-direction[1], direction[0]), dtype=float) / length
            row = np.zeros(columns); row[2 * first_index:2 * first_index + 2] = normal
            rows.append(row); residuals.append(float((first - start) @ normal))
    return np.asarray(residuals, dtype=float), np.asarray(rows, dtype=float)


def solve_sketch(definition: SketchDefinition, plane: SketchPlane, *, tolerance: float = 1.0e-9, maximum_iterations: int = 30) -> dict[str, tuple[float, float]]:
    """Solve constraints with minimum-norm Newton corrections."""

    if definition.references:
        raise GeometryError("sketch references must be resolved and bound before solving")

    if definition.version == 2 or any(c.kind == 'fixed' for c in definition.constraints):
        from .sketch_entities import solve_sketch_edit
        result = solve_sketch_edit(definition, plane, tolerance=tolerance, maximum_iterations=maximum_iterations)
        if not result.converged:
            raise GeometryError('sketch constraints are inconsistent: ' + ', '.join(result.conflicts))
        return dict(result.points)
    keys = tuple(definition.points)
    coordinates = np.asarray([definition.points[key] for key in keys], dtype=float)
    if not definition.constraints:
        return {key: tuple(coordinates[index]) for index, key in enumerate(keys)}
    for item in definition.constraints:
        if item.boundary_index is not None:
            limit = len(plane.boundary_edges if item.kind == "on_edge" else plane.boundary_vertices)
            if item.boundary_index >= limit:
                raise GeometryError(f"sketch {item.kind} boundary index {item.boundary_index} is unavailable")
    target = max(float(tolerance) * max(float(np.ptp(coordinates, axis=0).max()), 1.0), 1.0e-12)
    for _iteration in range(int(maximum_iterations)):
        residual, jacobian = _constraint_system(coordinates, keys, definition.constraints, plane)
        if not len(residual) or float(np.max(np.abs(residual))) <= target:
            break
        correction, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)
        if not np.all(np.isfinite(correction)):
            raise GeometryError("sketch constraint solution became non-finite")
        coordinates += correction.reshape((-1, 2))
    residual, _jacobian = _constraint_system(coordinates, keys, definition.constraints, plane)
    maximum = 0.0 if not len(residual) else float(np.max(np.abs(residual)))
    if maximum > target:
        raise GeometryError(f"sketch constraints are inconsistent; residual {maximum:.6g} m")
    return {key: (float(coordinates[index, 0]), float(coordinates[index, 1])) for index, key in enumerate(keys)}


def materialize_sketch(geometry: "GeometryModel", face_id: int, definition: SketchDefinition) -> dict[str, EntityRef]:
    """Create named sketch topology and optional normal shell extrusion."""

    plane = face_sketch_plane(geometry, face_id)
    return materialize_sketch_on_plane(geometry, plane, definition)


def materialize_sketch_on_plane(
    geometry: "GeometryModel", plane: SketchPlane, definition: SketchDefinition,
) -> dict[str, EntityRef]:
    """Create named sketch topology on an independent workplane."""

    if definition.references:
        raise GeometryError("sketch references must be resolved and bound before materialization")
    if definition.version == 2:
        from .sketch_entities import materialize_entities
        return materialize_entities(geometry, plane, definition)
    solved = solve_sketch(definition, plane)
    parent = {key: key for key in solved}

    def root(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for constraint in definition.constraints:
        if constraint.kind == "coincident" and constraint.second is not None:
            first_root, second_root = root(constraint.first), root(constraint.second)
            if first_root != second_root:
                parent[second_root] = first_root
    made: dict[str, int] = {}
    vertices: dict[str, int] = {}
    for key, position in solved.items():
        representative = root(key)
        if representative not in made:
            made[representative] = geometry.add_point(*plane.world(position))
        vertices[key] = made[representative]
    pairs = list(zip(definition.path, definition.path[1:]))
    if definition.closed:
        pairs.append((definition.path[-1], definition.path[0]))
    if any(vertices[first] == vertices[second] for first, second in pairs):
        raise GeometryError("a coincidence constraint collapsed a sketch segment")
    edges = [geometry.add_line(vertices[first], vertices[second]) for first, second in pairs]
    outputs: dict[str, EntityRef] = {
        **{f"point/{key}": EntityRef("vertex", identifier) for key, identifier in vertices.items()},
        **{f"profile/edge/{index}": EntityRef("edge", identifier) for index, identifier in enumerate(edges)},
    }
    if abs(definition.extrusion) > 1.0e-15:
        before = set(geometry.faces)
        geometry.extrude(edges, tuple(float(item) for item in plane.normal * definition.extrusion))
        outputs.update({f"extrusion/face/{index}": EntityRef("face", identifier) for index, identifier in enumerate(sorted(set(geometry.faces) - before))})
    return outputs
