"""Analytical structural-surface intersections and lightweight fallback."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Sequence
from uuid import UUID

import numpy as np

from .curves import Arc, Spline, Straight, arc_frame
from .curved_intersections import (
    qualified_curve_curve,
    qualified_curve_face,
    qualified_face_face,
)
from .entities import EntityRef, OrientedEdge
from .errors import GeometryError
from .identity import (
    EntityHandle,
    ResolutionStatus,
    validate_entity_kind,
    validate_local_id,
)
from .model import GeometryModel
from .policies import MutationPolicy
from .predicates import (
    DEFAULT_INTERSECTION_QUALIFICATION_POLICY,
    CertifiedCurveTrace,
    IntersectionComponent,
    IntersectionDimension,
    IntersectionKind,
    IntersectionQualificationPolicy,
    IntersectionQuality,
    IntersectionResult,
    ParameterRange as IntersectionParameterRange,
    qualified_line_plane,
    qualified_plane_plane,
    qualified_segment_segment,
)
from .transactions import ChangeSet
from .surfaces import CoonsSurface, Cylinder, Plane, RuledSurface, SurfaceProtocol
from .structural import ParameterRange as StructuralParameterRange

__all__ = [
    "ExpectedImprintChange", "FaceIntersection", "ImprintApplication",
    "ImprintOperation", "ImprintPlan", "apply_imprint", "clip_line_to_face",
    "intersect_faces", "intersect_surfaces", "line_cylinder", "line_line",
    "line_plane", "numerical_surface_intersection", "plane_cylinder",
    "plane_plane", "plan_imprint", "query_intersection",
]


def clip_line_to_face(
    geometry: GeometryModel,
    face_id: int,
    line_point_value: Sequence[float],
    line_direction: Sequence[float],
) -> IntersectionResult:
    """Return every material interval of a line clipped by a planar face.

    Holes are subtracted by the polygon backend, so concave faces and multiple
    disconnected material intervals are preserved.  Curved trim edges and
    non-planar supports fail closed instead of being sampled into topology.
    Line parameters are signed physical distances.
    """

    face_id = validate_local_id(face_id, name="face ID")
    face = geometry.faces.get(face_id)
    if face is None:
        raise GeometryError(f"no face {face_id}")
    face_handle = geometry.handle("face", face_id)
    tolerance_used = geometry.tolerance.effective_length(
        _face_length_scale(geometry, face_id)
    )
    try:
        from shapely.geometry import LineString, Point, Polygon
    except ImportError:  # pragma: no cover - optional dependency
        return IntersectionResult(
            IntersectionKind.CAPABILITY_MISSING,
            diagnostics=("planar_backend_unavailable",),
            second_parent=face_handle,
            tolerance_used=tolerance_used,
        )
    try:
        plane = _qualified_face_plane(geometry, face_id)
    except GeometryError:
        return IntersectionResult(
            IntersectionKind.UNSUPPORTED,
            diagnostics=("face_line_clipping_requires_a_planar_support",),
            second_parent=face_handle,
            tolerance_used=tolerance_used,
        )
    for loop in (face.loop,) + face.holes:
        if any(not isinstance(geometry.edges[item.edge].curve, Straight) for item in loop):
            return IntersectionResult(
                IntersectionKind.UNSUPPORTED,
                diagnostics=("curved_planar_trim_requires_a_qualified_curve_backend",),
                second_parent=face_handle,
                tolerance_used=tolerance_used,
            )

    point = _point(line_point_value, "line_point")
    raw_direction = _point(line_direction, "line_direction")
    direction_length = float(np.linalg.norm(raw_direction))
    if direction_length <= 0.0:
        return IntersectionResult(
            IntersectionKind.UNCLASSIFIED,
            diagnostics=("degenerate_line_direction",),
            second_parent=face_handle,
            tolerance_used=tolerance_used,
        )
    direction = raw_direction / direction_length
    support_result = qualified_line_plane(point, direction, plane).with_context(
        second_parent=face_handle, tolerance_used=tolerance_used
    )
    if not support_result.classified:
        return support_result
    if support_result.kind is IntersectionKind.DISJOINT:
        return support_result

    def loop_uv(loop: Sequence[OrientedEdge]) -> list[tuple[float, float]]:
        made = []
        for item in loop:
            vertex = geometry.oriented_start_vertex(item)
            made.append(tuple(float(value) for value in plane.local_uv(
                geometry.vertices[vertex].position
            )))
        return made

    polygon = Polygon(loop_uv(face.loop), [loop_uv(loop) for loop in face.holes])
    if polygon.is_empty or not polygon.is_valid:
        return IntersectionResult(
            IntersectionKind.UNCLASSIFIED,
            diagnostics=("invalid_planar_face_polygon",),
            second_parent=face_handle,
            tolerance_used=tolerance_used,
        )

    def component_for_point(world: np.ndarray) -> IntersectionComponent:
        uv = plane.local_uv(world)
        parameter = float((world - point) @ direction)
        return IntersectionComponent(
            (tuple(float(value) for value in world),),
            IntersectionQuality.VERIFIED_APPROXIMATE,
            first_parameter=(parameter,),
            second_parameter=uv,
            first_parameter_path=((parameter,),),
            second_parameter_path=(tuple(float(value) for value in uv),),
            second_subparent=face_handle,
        )

    if support_result.kind is IntersectionKind.CROSS:
        world = np.asarray(support_result.witnesses[0], dtype=float)
        uv = plane.local_uv(world)
        candidate = Point(uv)
        if not polygon.covers(candidate):
            return IntersectionResult(
                IntersectionKind.DISJOINT,
                diagnostics=("line_plane_hit_outside_face_material",),
                second_parent=face_handle,
                tolerance_used=tolerance_used,
            )
        kind = (
            IntersectionKind.TOUCH_POINT
            if polygon.boundary.covers(candidate)
            else IntersectionKind.CROSS
        )
        return IntersectionResult(
            kind,
            (component_for_point(world),),
            second_parent=face_handle,
            tolerance_used=tolerance_used,
        )

    # The line lies in the support plane.  Span it beyond every trim vertex;
    # polygon intersection then returns all disjoint line strings with holes
    # already removed.
    world_vertices = np.asarray(
        [
            geometry.vertices[geometry.oriented_start_vertex(item)].position
            for loop in (face.loop,) + face.holes
            for item in loop
        ],
        dtype=float,
    )
    parameters = (world_vertices - point) @ direction
    extent = max(float(np.ptp(parameters)), 1.0)
    lower = float(np.min(parameters) - extent)
    upper = float(np.max(parameters) + extent)
    endpoints_uv = [
        plane.local_uv(point + value * direction)
        for value in (lower, upper)
    ]
    clipped = polygon.intersection(LineString(endpoints_uv))

    line_components: list[IntersectionComponent] = []
    point_components: list[IntersectionComponent] = []

    def collect(value) -> None:
        if value.is_empty:
            return
        if value.geom_type == "LineString":
            coordinates = list(value.coords)
            if len(coordinates) < 2:
                return
            worlds = tuple(
                np.asarray(plane.evaluate(float(uv[0]), float(uv[1])), dtype=float)
                for uv in (coordinates[0], coordinates[-1])
            )
            line_parameters = tuple(float((world - point) @ direction) for world in worlds)
            if line_parameters[1] < line_parameters[0]:
                worlds = (worlds[1], worlds[0])
                line_parameters = (line_parameters[1], line_parameters[0])
            line_components.append(
                IntersectionComponent(
                    tuple(tuple(float(item) for item in world) for world in worlds),
                    IntersectionQuality.VERIFIED_APPROXIMATE,
                    first_parameter_range=IntersectionParameterRange(*line_parameters),
                    first_parameter_path=tuple((value,) for value in line_parameters),
                    second_parameter_path=tuple(
                        (float(uv[0]), float(uv[1]))
                        for uv in (coordinates[0], coordinates[-1])
                    ),
                    direction=tuple(float(item) for item in direction),
                    second_subparent=face_handle,
                )
            )
            return
        if value.geom_type == "Point":
            uv = tuple(value.coords)[0]
            world = np.asarray(plane.evaluate(float(uv[0]), float(uv[1])), dtype=float)
            point_components.append(component_for_point(world))
            return
        for child in getattr(value, "geoms", ()):
            collect(child)

    collect(clipped)
    if line_components:
        line_components.sort(
            key=lambda item: item.first_parameter_range.lower  # type: ignore[union-attr]
        )
        return IntersectionResult(
            IntersectionKind.OVERLAP_CURVE,
            tuple(line_components),
            diagnostics=("planar_material_intervals",),
            second_parent=face_handle,
            tolerance_used=tolerance_used,
        )
    if point_components:
        point_components.sort(key=lambda item: item.first_parameter or ())
        return IntersectionResult(
            IntersectionKind.TOUCH_POINT,
            tuple(point_components),
            second_parent=face_handle,
            tolerance_used=tolerance_used,
        )
    return IntersectionResult(
        IntersectionKind.DISJOINT,
        diagnostics=("coplanar_line_misses_face_material",),
        second_parent=face_handle,
        tolerance_used=tolerance_used,
    )


def _point(value: Sequence[float], name: str) -> np.ndarray:
    made = np.asarray(value, dtype=float)
    if made.shape != (3,) or not np.all(np.isfinite(made)):
        raise GeometryError(f"{name} must be a finite 3-vector")
    return made


def line_line(
    first_point: Sequence[float], first_direction: Sequence[float],
    second_point: Sequence[float], second_direction: Sequence[float], *, tolerance: float = 1e-9,
) -> np.ndarray | None:
    """Intersection of two 3D lines, or ``None`` for skew/parallel lines."""

    p, q = _point(first_point, "first_point"), _point(second_point, "second_point")
    a, b = _point(first_direction, "first_direction"), _point(second_direction, "second_direction")
    if float(np.linalg.norm(a)) <= 0.0 or float(np.linalg.norm(b)) <= 0.0:
        raise GeometryError("line directions must be non-zero")
    if float(np.linalg.norm(np.cross(a, b))) <= tolerance * float(np.linalg.norm(a)) * float(np.linalg.norm(b)):
        return None
    matrix = np.column_stack((a, -b))
    parameters, *_ = np.linalg.lstsq(matrix, q - p, rcond=None)
    first = p + parameters[0] * a
    second = q + parameters[1] * b
    # Residual scaling is local to the participating lines.  Translating both
    # lines by a large world offset must not change the classification.
    scale = max(
        float(np.linalg.norm(q - p)),
        float(np.linalg.norm(first - p)),
        float(np.linalg.norm(second - q)),
        1.0,
    )
    return 0.5 * (first + second) if float(np.linalg.norm(first-second)) <= tolerance*scale else None


def line_plane(
    line_point_value: Sequence[float], line_direction: Sequence[float], plane: Plane,
    *, tolerance: float = 1e-12,
) -> np.ndarray | None:
    point, direction = _point(line_point_value, "line_point"), _point(line_direction, "line_direction")
    if float(np.linalg.norm(direction)) <= 0.0:
        raise GeometryError("line direction must be non-zero")
    denominator = float(plane.normal @ direction)
    if abs(denominator) <= tolerance * max(float(np.linalg.norm(direction)), 1.0):
        return None
    parameter = float(plane.normal @ (plane.origin - point)) / denominator
    return point + parameter * direction


def plane_plane(first: Plane, second: Plane, *, tolerance: float = 1e-12) -> tuple[np.ndarray, np.ndarray] | None:
    """Return a point and unit direction for two non-parallel planes."""

    direction = np.cross(first.normal, second.normal)
    squared = float(direction @ direction)
    if squared <= tolerance:
        return None
    offsets = np.array([first.normal @ first.origin, second.normal @ second.origin, 0.0])
    matrix = np.vstack((first.normal, second.normal, direction))
    point = np.linalg.solve(matrix, offsets)
    return point, direction / np.sqrt(squared)


def line_cylinder(
    line_point_value: Sequence[float], line_direction: Sequence[float], cylinder: Cylinder,
    *, bounded: bool = True, tolerance: float = 1e-12,
) -> tuple[np.ndarray, ...]:
    """Analytical line/cylinder intersections, optionally clipped axially."""

    point, direction = _point(line_point_value, "line_point"), _point(line_direction, "line_direction")
    if float(np.linalg.norm(direction)) <= 0.0:
        raise GeometryError("line direction must be non-zero")
    offset = point - cylinder.origin
    radial_direction = direction - float(direction @ cylinder.axis)*cylinder.axis
    radial_offset = offset - float(offset @ cylinder.axis)*cylinder.axis
    coefficients = (
        float(radial_direction @ radial_direction),
        2.0*float(radial_offset @ radial_direction),
        float(radial_offset @ radial_offset) - cylinder.radius**2,
    )
    a, b, c = coefficients
    if abs(a) <= tolerance:
        return ()
    discriminant = b*b - 4*a*c
    if discriminant < -tolerance:
        return ()
    roots = [-b/(2*a)] if abs(discriminant) <= tolerance else [(-b-np.sqrt(discriminant))/(2*a), (-b+np.sqrt(discriminant))/(2*a)]
    made = []
    for root in roots:
        candidate = point + root*direction
        if bounded:
            _u, v = cylinder.local_uv(candidate)
            if not -tolerance <= v <= 1.0+tolerance:
                continue
        made.append(candidate)
    return tuple(made)


def plane_cylinder(
    plane: Plane, cylinder: Cylinder, *, samples: int = 129, tolerance: float = 1e-12,
) -> tuple[np.ndarray, ...]:
    """Analytical sampled curve(s) for a bounded plane/cylinder intersection."""

    denominator = float(plane.normal @ cylinder.axis)
    angles = cylinder.start_angle + np.linspace(0.0, cylinder.sweep_angle, max(int(samples), 3))
    radial = np.cos(angles)[:,None]*cylinder.radial_direction + np.sin(angles)[:,None]*cylinder.circumferential_direction
    bases = cylinder.origin + cylinder.radius*radial
    if abs(denominator) > tolerance:
        axial = ((plane.origin-bases) @ plane.normal) / denominator
        keep = (axial >= min(0.0, cylinder.height)-tolerance) & (axial <= max(0.0, cylinder.height)+tolerance)
        points = bases[keep] + axial[keep,None]*cylinder.axis
        return (points,) if len(points) else ()
    signed = (bases-plane.origin) @ plane.normal
    crossings = []
    def signed_at(angle: float) -> float:
        radial_direction = (
            np.cos(angle) * cylinder.radial_direction
            + np.sin(angle) * cylinder.circumferential_direction
        )
        return float(
            (cylinder.origin + cylinder.radius * radial_direction - plane.origin)
            @ plane.normal
        )

    for index in range(len(signed)-1):
        if abs(float(signed[index])) <= tolerance:
            angle = angles[index]
        elif signed[index]*signed[index+1] > 0.0:
            continue
        else:
            lower, upper = float(angles[index]), float(angles[index + 1])
            lower_value = float(signed[index])
            for _ in range(64):
                angle = 0.5 * (lower + upper)
                value = signed_at(angle)
                if abs(value) <= tolerance:
                    break
                if lower_value * value <= 0.0:
                    upper = angle
                else:
                    lower = angle
                    lower_value = value
        base = cylinder.origin + cylinder.radius*(np.cos(angle)*cylinder.radial_direction + np.sin(angle)*cylinder.circumferential_direction)
        crossings.append(np.vstack((base, base+cylinder.height*cylinder.axis)))
    return tuple(crossings)


def numerical_surface_intersection(
    first: SurfaceProtocol, second: SurfaceProtocol, *, samples: int = 41, tolerance: float = 1e-4,
) -> np.ndarray:
    """Deterministic lightweight fallback for supported parametric patches.

    This is intentionally a qualification-friendly sampling fallback, not a
    replacement for a CAD kernel.  Returned points are ordered by the first
    surface parameters and are suitable for preview or subsequent refinement.
    """

    parameters = np.linspace(0.0, 1.0, max(int(samples), 3))
    second_points = np.asarray([second.evaluate(float(u), float(v)) for u in parameters for v in parameters])
    matches = []
    scale = max(float(np.ptp(second_points, axis=0).max()), 1.0)
    for u in parameters:
        for v in parameters:
            point = np.asarray(first.evaluate(float(u), float(v)))
            distances = np.linalg.norm(second_points-point, axis=1)
            if float(distances.min()) <= tolerance*scale:
                matches.append(point)
    if not matches:
        return np.empty((0,3), dtype=float)
    unique = []
    for point in matches:
        if not unique or min(float(np.linalg.norm(point-other)) for other in unique) > tolerance*scale:
            unique.append(point)
    return np.asarray(unique)


def intersect_surfaces(
    first: SurfaceProtocol,
    second: SurfaceProtocol,
    *,
    samples: int = 129,
    tolerance: float = 1.0e-5,
) -> tuple[np.ndarray, ...]:
    """Dispatch qualified analytical pairs, then use the numerical fallback."""

    if isinstance(first, Plane) and isinstance(second, Cylinder):
        return plane_cylinder(first, second, samples=samples)
    if isinstance(first, Cylinder) and isinstance(second, Plane):
        return plane_cylinder(second, first, samples=samples)
    if isinstance(first, Plane) and isinstance(second, Plane):
        line = plane_plane(first, second)
        if line is None:
            return ()
        point, direction = line
        intersections = []
        for surface in (first, second):
            corners = [
                surface.evaluate(u, v)
                for u, v in ((0.0,0.0),(1.0,0.0),(1.0,1.0),(0.0,1.0))
            ]
            parameters = []
            for start, end in zip(corners, corners[1:]+corners[:1]):
                hit = line_line(point, direction, start, end-start)
                if hit is not None:
                    along = float((hit-start) @ (end-start)) / max(float((end-start) @ (end-start)), 1e-30)
                    if -tolerance <= along <= 1.0+tolerance:
                        parameters.append(float((hit-point) @ direction))
            if len(parameters) < 2:
                return ()
            interval = (min(parameters), max(parameters))
            intersections.append(interval)
        lower = max(item[0] for item in intersections)
        upper = min(item[1] for item in intersections)
        return (np.vstack((point+lower*direction, point+upper*direction)),) if upper > lower+tolerance else ()
    points = numerical_surface_intersection(
        first, second, samples=max(3, int(np.sqrt(samples))), tolerance=tolerance
    )
    return (points,) if len(points) else ()


@dataclass(frozen=True)
class FaceIntersection:
    """Topology produced by imprinting a shared face intersection."""

    edge: EntityRef
    first_faces: tuple[EntityRef, ...]
    second_faces: tuple[EntityRef, ...]
    edges: tuple[EntityRef, ...] = ()

    def __post_init__(self) -> None:
        """Keep ``edge`` as the compatible primary-edge accessor."""

        shared = tuple(self.edges) if self.edges else (self.edge,)
        if not shared or shared[0] != self.edge:
            raise GeometryError("the primary intersection edge must be first")
        if len(set(shared)) != len(shared) or any(
            reference.kind != "edge" for reference in shared
        ):
            raise GeometryError("intersection edges must be unique edge references")
        object.__setattr__(self, "edges", shared)


class ImprintOperation(StrEnum):
    """Closed set of deterministic actions represented by an imprint plan."""

    FACE_IMPRINT = "face_imprint"
    MEMBER_CONNECTION = "member_connection"
    MEMBER_SHEET_RELATION = "member_sheet_relation"
    NO_TOPOLOGY = "no_topology"


@dataclass(frozen=True, slots=True)
class ExpectedImprintChange:
    """One deterministic, declarative change expected from an imprint plan."""

    action: str
    entity_kind: str
    parents: tuple[EntityHandle, ...] = ()

    def __post_init__(self) -> None:
        action = str(self.action).strip()
        entity_kind = str(self.entity_kind).strip()
        parents = tuple(self.parents)
        if not action or not entity_kind:
            raise GeometryError("expected imprint changes need an action and entity kind")
        if any(not isinstance(item, EntityHandle) for item in parents):
            raise GeometryError("expected imprint change parents must be EntityHandle values")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "entity_kind", entity_kind)
        object.__setattr__(self, "parents", tuple(sorted(parents)))


@dataclass(frozen=True, slots=True)
class ImprintPlan:
    """Immutable non-mutating plan tied to one model revision and query."""

    model_id: UUID
    revision: int
    first_parent: EntityHandle
    second_parent: EntityHandle
    result: IntersectionResult
    policy: object
    operation: ImprintOperation | str
    expected_changes: tuple[ExpectedImprintChange, ...] = ()
    affected: tuple[EntityHandle, ...] = ()
    qualification: IntersectionQualificationPolicy = (
        DEFAULT_INTERSECTION_QUALIFICATION_POLICY
    )

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, UUID) or self.model_id.int == 0:
            raise GeometryError("imprint plan model_id must be a non-nil UUID")
        if isinstance(self.revision, bool) or int(self.revision) < 0:
            raise GeometryError("imprint plan revision must be non-negative")
        if self.first_parent.model_id != self.model_id or self.second_parent.model_id != self.model_id:
            raise GeometryError("imprint plan parents must belong to its model")
        if self.result.first_parent != self.first_parent or self.result.second_parent != self.second_parent:
            raise GeometryError("imprint plan query parents do not match the plan")
        try:
            operation = ImprintOperation(self.operation)
        except (TypeError, ValueError) as error:
            raise GeometryError("invalid imprint operation") from error
        changes = tuple(self.expected_changes)
        if any(not isinstance(item, ExpectedImprintChange) for item in changes):
            raise GeometryError("expected changes must be ExpectedImprintChange values")
        affected = tuple(sorted(set(self.affected)))
        if any(item.model_id != self.model_id for item in affected):
            raise GeometryError("affected handles must belong to the imprint model")
        if not isinstance(self.qualification, IntersectionQualificationPolicy):
            raise GeometryError(
                "imprint qualification must be IntersectionQualificationPolicy"
            )
        object.__setattr__(self, "revision", int(self.revision))
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "expected_changes", changes)
        object.__setattr__(self, "affected", affected)


@dataclass(frozen=True, slots=True)
class ImprintApplication:
    """Committed outcome of applying a verified :class:`ImprintPlan`."""

    plan: ImprintPlan
    result: IntersectionResult
    change_set: ChangeSet
    relations: tuple[EntityHandle, ...] = ()
    face_intersection: FaceIntersection | None = None
    reused: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ImprintPlan):
            raise GeometryError("imprint application needs an ImprintPlan")
        if not isinstance(self.result, IntersectionResult):
            raise GeometryError("imprint application needs an IntersectionResult")
        if not isinstance(self.change_set, ChangeSet):
            raise GeometryError("imprint application needs a ChangeSet")
        relations = tuple(sorted(set(self.relations)))
        if any(item.model_id != self.plan.model_id for item in relations):
            raise GeometryError("imprint relations must belong to the plan model")
        object.__setattr__(self, "relations", relations)
        object.__setattr__(self, "reused", bool(self.reused))


def _face_plane(geometry: GeometryModel, face_id: int) -> Plane:
    face = geometry.faces[face_id]
    if isinstance(face.surface, Plane):
        return face.surface
    points = np.asarray([geometry.vertex_position(geometry.oriented_start_vertex(item)) for item in face.loop])
    origin = points.mean(axis=0)
    _u, singular, vectors = np.linalg.svd(points-origin)
    extent = float(np.linalg.norm(np.ptp(points, axis=0)))
    if singular[-1] > geometry.tolerance.effective_surface_residual(extent):
        raise GeometryError(f"face {face_id} is not planar")
    u_vector = vectors[0]
    v_vector = np.cross(vectors[-1], u_vector)
    return Plane(origin, u_vector, v_vector)


def _qualified_face_plane(geometry: GeometryModel, face_id: int) -> Plane:
    """Return an exact planar support without reinterpreting curved faces.

    Partitioning a planar plate produces topology-backed Coons patches.  Four
    straight coplanar boundaries define an exactly planar Coons surface, even
    though its serialization intentionally retains the generic support kind.
    Recognising that identity lets structural line clipping remain on the
    public query/plan/apply path.  Explicit curved Coons boundaries and every
    other curved support continue to fail closed.
    """

    face = geometry.faces[face_id]
    if isinstance(face.surface, Plane):
        return face.surface
    if face.surface is None:
        return _face_plane(geometry, face_id)
    if not isinstance(face.surface, CoonsSurface):
        raise GeometryError(f"face {face_id} has a non-planar support")
    if any(
        not isinstance(geometry.edges[item.edge].curve, Straight)
        for loop in (face.loop,) + face.holes
        for item in loop
    ):
        raise GeometryError(f"face {face_id} has curved Coons boundaries")

    plane = _face_plane(geometry, face_id)
    if face.surface.has_boundaries:
        samples = np.vstack(
            (
                face.surface.bottom,
                face.surface.right,
                face.surface.top,
                face.surface.left,
            )
        )
        residual = np.abs((samples - plane.origin) @ plane.normal)
        extent = float(np.linalg.norm(np.ptp(samples, axis=0)))
        if float(np.max(residual)) > geometry.tolerance.effective_surface_residual(
            extent
        ):
            raise GeometryError(f"face {face_id} has a non-planar Coons support")
    return plane


def _face_length_scale(geometry: GeometryModel, face_id: int) -> float:
    """Translation-invariant characteristic extent for one bounded face."""

    bounds = geometry._entity_bounds(("face", int(face_id)))  # noqa: SLF001
    extent = 0.0
    if bounds is not None:
        extent = float(
            np.linalg.norm(np.asarray(bounds[3:], dtype=float) - bounds[:3])
        )
    surface = geometry.faces[int(face_id)].surface
    if isinstance(surface, Cylinder):
        extent = max(extent, surface.radius, abs(surface.height))
    return max(extent, 1.0)


def _boundary_vertex(geometry: GeometryModel, face_id: int, point: np.ndarray) -> int:
    face = geometry.faces[face_id]
    scale = _face_length_scale(geometry, face_id)
    length_tolerance = geometry.tolerance.effective_length(scale)
    parameter_tolerance = geometry.tolerance.effective_parameter(scale, scale)
    for loop in (face.loop,) + face.holes:
        for item in loop:
            for vertex_id in (
                geometry.oriented_start_vertex(item),
                geometry.oriented_end_vertex(item),
            ):
                if (
                    float(
                        np.linalg.norm(
                            geometry.vertex_position(vertex_id) - point
                        )
                    )
                    <= length_tolerance
                ):
                    return vertex_id
    best = None
    for loop in (face.loop,) + face.holes:
        for item in loop:
            _candidate, parameter, distance = geometry.closest_edge_point(
                item.edge, point
            )
            if best is None or distance < best[0]:
                best = (distance, item.edge, parameter)
    assert best is not None
    if best[0] > geometry.tolerance.effective_surface_residual(scale):
        raise GeometryError("intersection endpoint is not on the face boundary")
    vertex, _edges = geometry.split_edge(
        best[1],
        min(max(best[2], parameter_tolerance), 1.0 - parameter_tolerance),
    )
    geometry.move_point(vertex, *point)
    return vertex


def _merge_vertex(geometry: GeometryModel, old: int, new: int) -> None:
    if old == new:
        return
    # Reverse incidence gives the exact changed closure.  Replace records via
    # the journal owner so vertex-edge incidence is detached and reattached
    # before the obsolete vertex is retired.
    for edge_id in tuple(geometry.edges_using_vertex(old)):
        edge = geometry.edges[edge_id]
        curve = edge.curve
        if isinstance(curve, Arc) and curve.via_vertex == old:
            curve = Arc(new)
        elif isinstance(curve, Spline) and old in curve.control_vertices:
            curve = Spline(
                tuple(
                    new if vertex == old else vertex
                    for vertex in curve.control_vertices
                )
            )
        geometry._put_entity(  # noqa: SLF001
            "edge",
            replace(
                edge,
                start=new if edge.start == old else edge.start,
                end=new if edge.end == old else edge.end,
                curve=curve,
            ),
        )
    geometry.remove_vertex(old, record=False)
    geometry.record_replacement(EntityRef("vertex", old), (EntityRef("vertex", new),))


def _capture_face_attachments(geometry: GeometryModel, face_id: int):
    snapshots = []
    for attachment_id in sorted(
        geometry._target_attachments.get(("face", face_id), ())  # noqa: SLF001
    ):
        attachment = geometry.attachments[attachment_id]
        if len(attachment.target_parameters) != 2:
            raise GeometryError(
                f"attachment {attachment_id} has no face parameter rectangle"
            )
        first_range, second_range = attachment.target_parameters
        # Include the centre and edge midpoints.  A split can place the four
        # corners in different descendants, and retaining only corners would
        # lose a thin but valid attachment fragment.
        first_values = tuple(
            sorted(
                set(
                    (
                        first_range.start,
                        0.5 * (first_range.start + first_range.end),
                        first_range.end,
                    )
                )
            )
        )
        second_values = tuple(
            sorted(
                set(
                    (
                        second_range.start,
                        0.5 * (second_range.start + second_range.end),
                        second_range.end,
                    )
                )
            )
        )
        samples_uv = tuple(
            (first, second)
            for first in first_values
            for second in second_values
        )
        world_samples = tuple(
            geometry.face_point(face_id, first, second)
            for first, second in samples_uv
        )
        snapshots.append(("target", attachment, world_samples))
    for attachment_id in sorted(
        geometry._source_attachments.get(("face", face_id), ())  # noqa: SLF001
    ):
        if attachment_id not in {
            item[1].id for item in snapshots
        }:
            snapshots.append(("source", geometry.attachments[attachment_id], ()))
    return tuple(snapshots)


def _remap_face_attachments(
    geometry: GeometryModel,
    old_face,
    descendants: Sequence[int],
    snapshots,
) -> None:
    """Retarget or split qualified face attachments across descendants."""

    for role, attachment, world_samples in snapshots:
        attachment_id = attachment.id
        if role == "source":
            made_ids: list[int] = []
            lineage = tuple(
                dict.fromkeys((*attachment.lineage, ("face", old_face.id)))
            )
            for index, face_id in enumerate(sorted(descendants)):
                if index == 0:
                    geometry._put_structural(  # noqa: SLF001
                        "attachment",
                        replace(
                            attachment,
                            source_id=face_id,
                            lineage=lineage,
                        ),
                    )
                    made_ids.append(attachment_id)
                else:
                    made_ids.append(
                        geometry.add_attachment(
                            attachment.member_id,
                            attachment.kind,
                            attachment.target_kind,
                            attachment.target_id,
                            attachment.member_range,
                            attachment.target_parameters,
                            connection_intent=attachment.connection_intent,
                            evidence=attachment.evidence,
                            max_residual=attachment.max_residual,
                            tolerance_used=attachment.tolerance_used,
                            part_id=attachment.part_id,
                            sheet_id=attachment.sheet_id,
                            provenance=attachment.provenance,
                            lineage=lineage,
                            source_kind="face",
                            source_id=face_id,
                            metadata=attachment.metadata,
                        )
                    )
            if len(made_ids) > 1:
                for junction_id in tuple(
                    sorted(geometry._attachment_junctions.get(attachment_id, ()))  # noqa: SLF001
                ):
                    junction = geometry.junctions[junction_id]
                    expanded: list[int] = []
                    for identifier in junction.attachment_ids:
                        expanded.extend(
                            made_ids if identifier == attachment_id else (identifier,)
                        )
                    geometry._put_structural(  # noqa: SLF001
                        "junction",
                        replace(
                            junction,
                            attachment_ids=tuple(dict.fromkeys(expanded)),
                        ),
                    )
            continue
        candidates: list[tuple[int, tuple[tuple[float, float], ...]]] = []
        for face_id in sorted(descendants):
            mapped = tuple(
                tuple(float(value) for value in geometry.face_local_uv(face_id, point))
                for point in world_samples
            )
            retained = tuple(
                uv for uv in mapped if geometry.face_contains_uv(face_id, uv)
            )
            if retained:
                candidates.append((face_id, retained))
        if not candidates:
            raise GeometryError(
                f"attachment {attachment_id} cannot be mapped to a face fragment"
            )
        made_ids: list[int] = []
        lineage = tuple(
            dict.fromkeys((*attachment.lineage, ("face", old_face.id)))
        )
        for index, (face_id, mapped) in enumerate(candidates):
            u_values = [item[0] for item in mapped]
            v_values = [item[1] for item in mapped]
            parameters = (
                StructuralParameterRange(min(u_values), max(u_values)),
                StructuralParameterRange(min(v_values), max(v_values)),
            )
            if index == 0:
                geometry._put_structural(  # noqa: SLF001
                    "attachment",
                    replace(
                        attachment,
                        target_id=face_id,
                        target_parameters=parameters,
                        lineage=lineage,
                    ),
                )
                made_ids.append(attachment_id)
                continue
            made_ids.append(
                geometry.add_attachment(
                    attachment.member_id,
                    attachment.kind,
                    attachment.target_kind,
                    face_id,
                    attachment.member_range,
                    parameters,
                    connection_intent=attachment.connection_intent,
                    evidence=attachment.evidence,
                    max_residual=attachment.max_residual,
                    tolerance_used=attachment.tolerance_used,
                    part_id=attachment.part_id,
                    sheet_id=attachment.sheet_id,
                    provenance=attachment.provenance,
                    lineage=lineage,
                    source_kind=attachment.source_kind,
                    source_id=attachment.source_id,
                    metadata=attachment.metadata,
                )
            )
        if len(made_ids) > 1:
            for junction_id in tuple(
                sorted(geometry._attachment_junctions.get(attachment_id, ()))  # noqa: SLF001
            ):
                junction = geometry.junctions[junction_id]
                expanded: list[int] = []
                for identifier in junction.attachment_ids:
                    expanded.extend(made_ids if identifier == attachment_id else (identifier,))
                geometry._put_structural(  # noqa: SLF001
                    "junction",
                    replace(junction, attachment_ids=tuple(dict.fromkeys(expanded))),
                )


def _fragment_with_edge(geometry: GeometryModel, face_id: int, start: int, end: int, edge_id: int) -> tuple[int,int]:
    from .operations import _partition_face_holes, _split_loop  # internal topology primitives

    face = geometry.faces[face_id]
    first_chain, second_chain = _split_loop(face, start, end, geometry)
    first_holes, second_holes = _partition_face_holes(
        geometry, face, first_chain, second_chain, start, end
    )
    surface = face.surface
    parameterization = face.parameterization
    metadata, tags = dict(face.metadata), geometry.tags_for(face.ref)
    attachment_snapshots = _capture_face_attachments(geometry, face_id)
    geometry._delete_entity("face", face_id)  # noqa: SLF001
    made = []
    for loop, holes in zip(
        (
            tuple(first_chain) + (OrientedEdge(edge_id, False),),
            tuple(second_chain) + (OrientedEdge(edge_id, True),),
        ),
        (first_holes, second_holes),
    ):
        corners = geometry._detect_corners(loop) if len(loop) >= 4 else None  # noqa: SLF001
        identifier = geometry.add_face_from_loop(loop, corners, surface=surface)
        geometry._put_entity(  # noqa: SLF001
            "face",
            replace(
                geometry.faces[identifier],
                holes=holes,
                metadata=dict(metadata),
                surface=surface,
                parameterization=parameterization,
            ),
        )
        geometry.tag(EntityRef("face",identifier), *tags)
        made.append(identifier)
    _remap_face_attachments(geometry, face, made, attachment_snapshots)
    geometry.record_replacement(EntityRef("face",face_id), tuple(EntityRef("face",item) for item in made))
    return made[0], made[1]


def _fragment_with_edge_chain(
    geometry: GeometryModel,
    face_id: int,
    start: int,
    end: int,
    edge_ids: Sequence[int],
) -> tuple[int, int]:
    """Split one face by an oriented, shared multi-edge curve chain."""

    from .operations import _partition_face_holes, _split_loop

    made_edges = tuple(validate_local_id(item, name="imprint edge ID") for item in edge_ids)
    if not made_edges:
        raise GeometryError("an imprint chain needs at least one edge")
    current = start
    for edge_id in made_edges:
        edge = geometry.edges[edge_id]
        if edge.start != current:
            raise GeometryError("imprint edge chain is not continuously oriented")
        current = edge.end
    if current != end:
        raise GeometryError("imprint edge chain does not reach its end vertex")
    face = geometry.faces[face_id]
    first_chain, second_chain = _split_loop(face, start, end, geometry)
    first_holes, second_holes = _partition_face_holes(
        geometry, face, first_chain, second_chain, start, end
    )
    reverse_path = tuple(OrientedEdge(edge_id, False) for edge_id in reversed(made_edges))
    forward_path = tuple(OrientedEdge(edge_id, True) for edge_id in made_edges)
    surface = face.surface
    parameterization = face.parameterization
    metadata, tags = dict(face.metadata), geometry.tags_for(face.ref)
    attachment_snapshots = _capture_face_attachments(geometry, face_id)
    geometry._delete_entity("face", face_id)  # noqa: SLF001
    made = []
    for loop, holes in zip(
        (tuple(first_chain) + reverse_path, tuple(second_chain) + forward_path),
        (first_holes, second_holes),
    ):
        corners = geometry._detect_corners(loop) if len(loop) >= 4 else None  # noqa: SLF001
        identifier = geometry.add_face_from_loop(loop, corners, surface=surface)
        geometry._put_entity(  # noqa: SLF001
            "face",
            replace(
                geometry.faces[identifier],
                holes=holes,
                metadata=dict(metadata),
                surface=surface,
                parameterization=parameterization,
            ),
        )
        geometry.tag(EntityRef("face", identifier), *tags)
        made.append(identifier)
    _remap_face_attachments(geometry, face, made, attachment_snapshots)
    geometry.record_replacement(
        EntityRef("face", face_id),
        tuple(EntityRef("face", item) for item in made),
    )
    return made[0], made[1]


def _fragment_with_closed_chain(
    geometry: GeometryModel,
    face_id: int,
    edge_ids: Sequence[int | OrientedEdge],
) -> tuple[int, int]:
    """Split a face into an outer remainder and an inner curved region."""

    ring = tuple(
        item
        if isinstance(item, OrientedEdge)
        else OrientedEdge(validate_local_id(item, name="imprint edge ID"), True)
        for item in edge_ids
    )
    if len(ring) < 3:
        raise GeometryError("a closed imprint chain needs at least three edges")
    current = geometry.oriented_start_vertex(ring[0])
    start = current
    for oriented in ring:
        if geometry.oriented_start_vertex(oriented) != current:
            raise GeometryError("closed imprint chain is discontinuous")
        current = geometry.oriented_end_vertex(oriented)
    if current != start:
        raise GeometryError("closed imprint chain is not closed")
    face = geometry.faces[face_id]
    surface = face.surface
    parameterization = face.parameterization
    metadata, tags = dict(face.metadata), geometry.tags_for(face.ref)
    if _loop_signed_area(geometry, face_id, face.loop) * _loop_signed_area(
        geometry, face_id, ring
    ) < 0.0:
        ring = _reverse_loop(ring)
    attachment_snapshots = _capture_face_attachments(geometry, face_id)
    geometry._delete_entity("face", face_id)  # noqa: SLF001
    outer = geometry.add_face_from_loop(face.loop, face.corners, surface=surface)
    geometry._put_entity(  # noqa: SLF001
        "face",
        replace(
            geometry.faces[outer],
            holes=tuple(face.holes) + (_reverse_loop(ring),),
            metadata=dict(metadata),
            surface=surface,
            parameterization=parameterization,
        ),
    )
    inner = geometry.add_face_from_loop(ring, surface=surface)
    geometry._put_entity(  # noqa: SLF001
        "face",
        replace(
            geometry.faces[inner],
            metadata=dict(metadata),
            surface=surface,
            parameterization=parameterization,
        ),
    )
    for identifier in (outer, inner):
        geometry.tag(EntityRef("face", identifier), *tags)
    _remap_face_attachments(
        geometry, face, (outer, inner), attachment_snapshots
    )
    geometry.record_replacement(
        EntityRef("face", face_id),
        (EntityRef("face", outer), EntityRef("face", inner)),
    )
    return outer, inner


def _point_segment_distance_value(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> float:
    direction = end - start
    length_squared = float(direction @ direction)
    if length_squared <= np.finfo(float).tiny:
        return float(np.linalg.norm(point - start))
    parameter = float(np.clip((point - start) @ direction / length_squared, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + parameter * direction)))


def _region_boundary_matches_loop(
    geometry: GeometryModel,
    loop: Sequence[OrientedEdge],
    boundary: Sequence[Sequence[float]],
    tolerance: float,
) -> bool:
    """Whether one certified path is exactly one model trim loop."""

    points = tuple(np.asarray(point, dtype=float) for point in boundary)
    if len(points) < 3:
        return False
    for point in points:
        distance = min(
            geometry.closest_edge_point(item.edge, point)[2]
            for item in loop
        )
        if distance > tolerance:
            return False
    for item in loop:
        point = geometry.vertex_position(geometry.oriented_start_vertex(item))
        if min(float(np.linalg.norm(point - candidate)) for candidate in points) > tolerance:
            return False
    return True


def _region_boundary_matches_face(
    geometry: GeometryModel,
    face_id: int,
    boundaries: Sequence[Sequence[Sequence[float]]],
    tolerance: float,
) -> bool:
    face = geometry.faces[face_id]
    if len(boundaries) != 1 + len(face.holes):
        return False
    if not _region_boundary_matches_loop(
        geometry, face.loop, boundaries[0], tolerance
    ):
        return False
    unused = set(range(1, len(boundaries)))
    for hole in face.holes:
        matches = [
            index
            for index in sorted(unused)
            if _region_boundary_matches_loop(
                geometry, hole, boundaries[index], tolerance
            )
        ]
        if len(matches) != 1:
            return False
        unused.remove(matches[0])
    return not unused


def _connect_contained_curved_region(
    geometry: GeometryModel,
    first_face: int,
    second_face: int,
    component: IntersectionComponent,
    tolerance: float,
) -> tuple[EntityHandle, ...]:
    """Connect a certified coincident region equal to one parent's material."""

    if not component.boundary_paths:
        raise GeometryError("coincident region has no certified boundary path")
    boundaries = component.boundary_paths
    first_matches = _region_boundary_matches_face(
        geometry, first_face, boundaries, tolerance
    )
    second_matches = _region_boundary_matches_face(
        geometry, second_face, boundaries, tolerance
    )
    if first_matches == second_matches:
        raise GeometryError(
            "partial curved coincident region is not a single contained face"
        )
    contained = first_face if first_matches else second_face
    containing = second_face if first_matches else first_face
    contained_face = geometry.faces[contained]
    containing_face = geometry.faces[containing]
    if _region_boundary_matches_loop(
        geometry, containing_face.loop, boundaries[0], tolerance
    ):
        overlap = containing
    else:
        _outside, overlap = _fragment_with_closed_chain(
            geometry,
            containing,
            contained_face.loop,
        )
    for hole in contained_face.holes:
        overlap, _hole_material = _fragment_with_closed_chain(
            geometry, overlap, hole
        )
    return _connect_fully_coincident_faces(geometry, contained, overlap)


def _face_accepts_component_endpoints(
    geometry: GeometryModel,
    face_id: int,
    endpoints: tuple[np.ndarray, np.ndarray],
) -> bool:
    scale = _face_length_scale(geometry, face_id)
    tolerance = geometry.tolerance.effective_surface_residual(scale)
    face = geometry.faces[face_id]
    for point in endpoints:
        best = min(
            (
                geometry.closest_edge_point(oriented.edge, point)[2]
                for loop in (face.loop,) + face.holes
                for oriented in loop
            ),
            default=float("inf"),
        )
        if best > tolerance:
            return False
    return True


def _component_boundary_edge(
    geometry: GeometryModel,
    face_id: int,
    endpoints: tuple[np.ndarray, np.ndarray],
    tolerance: float,
) -> int | None:
    candidates: list[int] = []
    face = geometry.faces[face_id]
    for oriented in tuple(item for loop in (face.loop,) + face.holes for item in loop):
        edge = geometry.edges[oriented.edge]
        start = geometry.vertex_position(edge.start)
        end = geometry.vertex_position(edge.end)
        direct = (
            float(np.linalg.norm(start - endpoints[0])) <= tolerance
            and float(np.linalg.norm(end - endpoints[1])) <= tolerance
        )
        reversed_match = (
            float(np.linalg.norm(start - endpoints[1])) <= tolerance
            and float(np.linalg.norm(end - endpoints[0])) <= tolerance
        )
        if direct or reversed_match:
            candidates.append(edge.id)
    return min(candidates) if candidates else None


def _trace_vertices_and_edges(
    geometry: GeometryModel,
    first_face: int,
    second_face: int,
    witnesses: Sequence[Sequence[float]],
    *,
    closed: bool,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    points = [np.asarray(point, dtype=float) for point in witnesses]
    tolerance = max(
        geometry.tolerance.effective_curve_fit_residual(
            max(_face_length_scale(geometry, first_face), _face_length_scale(geometry, second_face))
        ),
        geometry.tolerance.length,
    )
    compact: list[np.ndarray] = []
    for point in points:
        if not compact or float(np.linalg.norm(point - compact[-1])) > tolerance:
            compact.append(point)
    if closed and len(compact) > 1 and float(np.linalg.norm(compact[0] - compact[-1])) <= tolerance:
        compact.pop()
    minimum = 3 if closed else 2
    if len(compact) < minimum:
        raise GeometryError("qualified imprint trace has too few distinct points")
    # Preserve exact primitive topology whenever the certified trace lies on
    # one line or circle.  This covers the analytical and common
    # Cylinder/Cone ring cases without replacing them by tolerance-sensitive
    # sampled chords.
    if not closed:
        chord = compact[-1] - compact[0]
        chord_length = float(np.linalg.norm(chord))
        if chord_length > 0.0:
            line_error = max(
                float(
                    np.linalg.norm(
                        (point - compact[0])
                        - float((point - compact[0]) @ chord / (chord_length * chord_length)) * chord
                    )
                )
                for point in compact
            )
            if line_error <= tolerance:
                first_vertices = [
                    _boundary_vertex(geometry, first_face, compact[0]),
                    _boundary_vertex(geometry, first_face, compact[-1]),
                ]
                second_vertices = [
                    _boundary_vertex(geometry, second_face, compact[0]),
                    _boundary_vertex(geometry, second_face, compact[-1]),
                ]
                for old, new in zip(second_vertices, first_vertices):
                    _merge_vertex(geometry, old, new)
                edge_id = geometry.add_line(*first_vertices)
                return tuple(first_vertices), (edge_id,)
        if len(compact) >= 3:
            middle_index = len(compact) // 2
            try:
                frame = arc_frame(compact[0], compact[middle_index], compact[-1])
            except (ValueError, GeometryError):
                frame = None
            if frame is not None:
                circle_error = max(
                    max(
                        abs(float(np.linalg.norm(point - frame.center)) - frame.radius),
                        abs(float((point - frame.center) @ frame.normal)),
                    )
                    for point in compact
                )
                if circle_error <= tolerance:
                    first_vertices = [
                        _boundary_vertex(geometry, first_face, compact[0]),
                        _boundary_vertex(geometry, first_face, compact[-1]),
                    ]
                    second_vertices = [
                        _boundary_vertex(geometry, second_face, compact[0]),
                        _boundary_vertex(geometry, second_face, compact[-1]),
                    ]
                    for old, new in zip(second_vertices, first_vertices):
                        _merge_vertex(geometry, old, new)
                    via = geometry.add_point(*compact[middle_index])
                    edge_id = geometry.add_arc(first_vertices[0], via, first_vertices[1])
                    return (first_vertices[0], via, first_vertices[1]), (edge_id,)
    elif len(compact) >= 6:
        half = len(compact) // 2
        quarter = max(1, half // 2)
        three_quarter = min(len(compact) - 1, half + max(1, (len(compact) - half) // 2))
        try:
            first_frame = arc_frame(compact[0], compact[quarter], compact[half])
            second_frame = arc_frame(compact[half], compact[three_quarter], compact[0])
            circle_error = max(
                max(
                    abs(float(np.linalg.norm(point - first_frame.center)) - first_frame.radius),
                    abs(float((point - first_frame.center) @ first_frame.normal)),
                )
                for point in compact
            )
        except (ValueError, GeometryError):
            first_frame = second_frame = None
            circle_error = float("inf")
        if first_frame is not None and second_frame is not None and circle_error <= tolerance:
            start_vertex = geometry.add_point(*compact[0])
            middle_vertex = geometry.add_point(*compact[half])
            first_via = geometry.add_point(*compact[quarter])
            second_via = geometry.add_point(*compact[three_quarter])
            edges = (
                geometry.add_arc(start_vertex, first_via, middle_vertex),
                geometry.add_arc(middle_vertex, second_via, start_vertex),
            )
            return (start_vertex, first_via, middle_vertex, second_via), edges
    if closed:
        endpoint_indices = list(range(0, len(compact), 2))
        vertices = [geometry.add_point(*compact[index]) for index in endpoint_indices]
        edges: list[int] = []
        for position, start_index in enumerate(endpoint_indices):
            end_position = (position + 1) % len(endpoint_indices)
            end_index = endpoint_indices[end_position]
            middle_index = (start_index + 1) % len(compact)
            if end_position == 0 and middle_index == 0:
                control_point = 0.5 * (compact[start_index] + compact[end_index])
            else:
                middle_point = compact[middle_index]
                control_point = (
                    2.0 * middle_point
                    - 0.5 * (compact[start_index] + compact[end_index])
                )
            control = geometry.add_point(*control_point)
            edges.append(
                geometry.add_spline(
                    vertices[position],
                    (control,),
                    vertices[end_position],
                )
            )
        return tuple(vertices), tuple(edges)
    else:
        first_vertices = [
            _boundary_vertex(geometry, first_face, compact[0]),
            _boundary_vertex(geometry, first_face, compact[-1]),
        ]
        second_vertices = [
            _boundary_vertex(geometry, second_face, compact[0]),
            _boundary_vertex(geometry, second_face, compact[-1]),
        ]
        for old, new in zip(second_vertices, first_vertices):
            _merge_vertex(geometry, old, new)
        endpoint_indices = list(range(0, len(compact), 2))
        if endpoint_indices[-1] != len(compact) - 1:
            endpoint_indices.append(len(compact) - 1)
        vertices = [first_vertices[0]]
        vertices.extend(
            geometry.add_point(*compact[index])
            for index in endpoint_indices[1:-1]
        )
        vertices.append(first_vertices[1])
        edges: list[int] = []
        for position, (start_index, end_index) in enumerate(
            zip(endpoint_indices, endpoint_indices[1:])
        ):
            if end_index - start_index == 2:
                middle_point = compact[start_index + 1]
                control_point = (
                    2.0 * middle_point
                    - 0.5 * (compact[start_index] + compact[end_index])
                )
            else:
                control_point = 0.5 * (
                    compact[start_index] + compact[end_index]
                )
            control = geometry.add_point(*control_point)
            edges.append(
                geometry.add_spline(
                    vertices[position],
                    (control,),
                    vertices[position + 1],
                )
            )
        return tuple(vertices), tuple(edges)


def _imprint_components(
    geometry: GeometryModel,
    first_face: int,
    second_face: int,
    components: Sequence[IntersectionComponent],
) -> FaceIntersection:
    """Atomically imprint every certified open/closed curved trace."""

    with geometry.transaction():
        first_active = [first_face]
        second_active = [second_face]
        relation_edges: list[int] = []
        for component in components:
            witnesses = tuple(np.asarray(point, dtype=float) for point in component.witnesses)
            if len(witnesses) < 2:
                raise GeometryError("curved face imprint needs complete trace witnesses")
            scale = max(
                _face_length_scale(geometry, first_active[0]),
                _face_length_scale(geometry, second_active[0]),
            )
            tolerance = geometry.tolerance.effective_curve_fit_residual(scale)
            closed = len(witnesses) >= 3 and float(np.linalg.norm(witnesses[0] - witnesses[-1])) <= tolerance
            if closed:
                first_target = next(
                    (
                        face_id
                        for face_id in first_active
                        if geometry.face_contains_uv(
                            face_id,
                            geometry.face_local_uv(face_id, witnesses[0]),
                        )
                    ),
                    None,
                )
                second_target = next(
                    (
                        face_id
                        for face_id in second_active
                        if geometry.face_contains_uv(
                            face_id,
                            geometry.face_local_uv(face_id, witnesses[0]),
                        )
                    ),
                    None,
                )
            else:
                endpoints = (witnesses[0], witnesses[-1])
                first_target = next(
                    (face_id for face_id in first_active if _face_accepts_component_endpoints(geometry, face_id, endpoints)),
                    None,
                )
                second_target = next(
                    (face_id for face_id in second_active if _face_accepts_component_endpoints(geometry, face_id, endpoints)),
                    None,
                )
            if first_target is None or second_target is None:
                raise GeometryError("imprint component does not map to one active face descendant")
            if not closed:
                endpoints = (witnesses[0], witnesses[-1])
                first_boundary = _component_boundary_edge(
                    geometry, first_target, endpoints, tolerance
                )
                second_boundary = _component_boundary_edge(
                    geometry, second_target, endpoints, tolerance
                )
                boundary_edge = first_boundary or second_boundary
                if boundary_edge is not None:
                    edge = geometry.edges[boundary_edge]
                    edge_points = (
                        geometry.vertex_position(edge.start),
                        geometry.vertex_position(edge.end),
                    )
                    if first_boundary is None:
                        made_vertices = tuple(
                            _boundary_vertex(geometry, first_target, point)
                            for point in edge_points
                        )
                        for old, new in zip(made_vertices, (edge.start, edge.end)):
                            _merge_vertex(geometry, old, new)
                        first_children = _fragment_with_edge_chain(
                            geometry,
                            first_target,
                            edge.start,
                            edge.end,
                            (boundary_edge,),
                        )
                        first_active.remove(first_target)
                        first_active.extend(first_children)
                    if second_boundary is None:
                        made_vertices = tuple(
                            _boundary_vertex(geometry, second_target, point)
                            for point in edge_points
                        )
                        for old, new in zip(made_vertices, (edge.start, edge.end)):
                            _merge_vertex(geometry, old, new)
                        second_children = _fragment_with_edge_chain(
                            geometry,
                            second_target,
                            edge.start,
                            edge.end,
                            (boundary_edge,),
                        )
                        second_active.remove(second_target)
                        second_active.extend(second_children)
                    relation_edges.append(boundary_edge)
                    continue
            _vertices, edges = _trace_vertices_and_edges(
                geometry,
                first_target,
                second_target,
                witnesses,
                closed=closed,
            )
            relation_edges.extend(edges)
            if closed:
                first_children = _fragment_with_closed_chain(geometry, first_target, edges)
                second_children = _fragment_with_closed_chain(geometry, second_target, edges)
            else:
                first_children = _fragment_with_edge_chain(
                    geometry,
                    first_target,
                    geometry.edges[edges[0]].start,
                    geometry.edges[edges[-1]].end,
                    edges,
                )
                second_children = _fragment_with_edge_chain(
                    geometry,
                    second_target,
                    geometry.edges[edges[0]].start,
                    geometry.edges[edges[-1]].end,
                    edges,
                )
            first_active.remove(first_target)
            first_active.extend(first_children)
            second_active.remove(second_target)
            second_active.extend(second_children)
        errors = geometry.validate_topology()
        if errors:
            raise GeometryError(
                "multi-component curved imprint produced invalid topology: "
                + "; ".join(errors)
            )
        # A component can already be a boundary edge of both participating
        # descendants.  Keep the first deterministic occurrence while
        # avoiding duplicate references in the public result.
        references = tuple(
            EntityRef("edge", edge_id)
            for edge_id in dict.fromkeys(relation_edges)
        )
        if not references:
            raise GeometryError("curved imprint produced no relation edges")
        return FaceIntersection(
            references[0],
            tuple(EntityRef("face", item) for item in sorted(first_active)),
            tuple(EntityRef("face", item) for item in sorted(second_active)),
            references,
        )


def _imprint_segment(
    geometry: GeometryModel,
    first_face: int,
    second_face: int,
    endpoints: tuple[np.ndarray, np.ndarray],
) -> FaceIntersection:
    with geometry.transaction():
        first_vertices = [_boundary_vertex(geometry, first_face, endpoint) for endpoint in endpoints]
        second_vertices = [_boundary_vertex(geometry, second_face, endpoint) for endpoint in endpoints]
        for old, new in zip(second_vertices, first_vertices):
            _merge_vertex(geometry, old, new)
        edge = geometry.add_line(*first_vertices)
        first_made = _fragment_with_edge(geometry, first_face, *first_vertices, edge)
        second_made = _fragment_with_edge(geometry, second_face, *first_vertices, edge)
        errors = geometry.validate_topology()
        if errors:
            raise GeometryError(
                "intersection imprint produced invalid topology: "
                + "; ".join(errors)
            )
        return FaceIntersection(
            EntityRef("edge", edge),
            tuple(EntityRef("face", item) for item in first_made),
            tuple(EntityRef("face", item) for item in second_made),
        )


def _axial_plane_cylinder_segment(
    geometry: GeometryModel,
    plane_face: int,
    cylinder_face: int,
) -> tuple[np.ndarray, np.ndarray]:
    plane = geometry.faces[plane_face].surface
    cylinder = geometry.faces[cylinder_face].surface
    assert isinstance(plane, Plane) and isinstance(cylinder, Cylinder)
    if abs(float(plane.normal @ cylinder.axis)) > geometry.tolerance.angular:
        raise GeometryError(
            "topology imprint supports axial plane-cylinder lines and exactly "
            "transverse closed rings; use intersect_surfaces to query an "
            "oblique intersection"
        )
    candidates = []
    for curve in plane_cylinder(plane, cylinder):
        if curve.shape != (2,3):
            continue
        inside = True
        for face_id in (plane_face, cylinder_face):
            for point in curve:
                projected, uv, distance = geometry.project_to_face(face_id, point)
                scale = max(
                    _face_length_scale(geometry, plane_face),
                    _face_length_scale(geometry, cylinder_face),
                )
                if (
                    distance
                    > geometry.tolerance.effective_surface_residual(scale)
                    or not geometry.face_contains_uv(face_id, uv)
                ):
                    inside = False
                    break
            if not inside:
                break
        if inside:
            candidates.append((curve[0],curve[1]))
    if len(candidates) != 1:
        raise GeometryError(
            "the bounded plane/cylinder faces must share exactly one axial "
            f"intersection segment, found {len(candidates)}"
        )
    return candidates[0]


def _reverse_loop(loop: Sequence[OrientedEdge]) -> tuple[OrientedEdge, ...]:
    return tuple(
        OrientedEdge(item.edge, not item.forward) for item in reversed(loop)
    )


def _loop_signed_area(
    geometry: GeometryModel, face_id: int, loop: Sequence[OrientedEdge]
) -> float:
    points = np.asarray(
        [
            geometry.face_support_local_uv(
                face_id,
                geometry.vertex_position(geometry.oriented_start_vertex(item))
            )
            for item in loop
        ],
        dtype=float,
    )
    following = np.roll(points, -1, axis=0)
    return 0.5 * float(
        np.sum(points[:, 0] * following[:, 1] - following[:, 0] * points[:, 1])
    )


def _same_cylinder_band(
    candidate: Cylinder, reference: Cylinder, *, tolerance: float
) -> bool:
    scale = max(abs(reference.height), reference.radius, 1.0)
    return (
        float(np.linalg.norm(candidate.origin - reference.origin))
        <= tolerance * scale
        and float(np.linalg.norm(candidate.axis - reference.axis)) <= tolerance
        and float(
            np.linalg.norm(
                candidate.radial_direction - reference.radial_direction
            )
        )
        <= tolerance
        and abs(candidate.radius - reference.radius) <= tolerance * scale
        and abs(candidate.height - reference.height) <= tolerance * scale
    )


def _angular_distance(first: float, second: float) -> float:
    period = 2.0 * np.pi
    return abs((first - second + np.pi) % period - np.pi)


def _boundary_edge_at_point(
    geometry: GeometryModel, face_id: int, point: np.ndarray
) -> int:
    face = geometry.faces[face_id]
    ranked = []
    for item in face.loop:
        _candidate, _parameter, distance = geometry.closest_edge_point(
            item.edge, point
        )
        ranked.append((distance, item.edge))
    distance, edge_id = min(ranked)
    scale = _face_length_scale(geometry, face_id)
    if distance > geometry.tolerance.effective_surface_residual(scale):
        raise GeometryError(
            f"cylinder face {face_id} is not bounded by its surface patch"
        )
    return edge_id


def _transverse_cylinder_band(
    geometry: GeometryModel,
    plane_face: int,
    cylinder_face: int,
    *,
    tolerance: float | None = None,
) -> tuple[tuple[int, ...], float, np.ndarray]:
    """Preflight one complete conformal cylindrical band without mutation."""

    plane = geometry.faces[plane_face].surface
    reference = geometry.faces[cylinder_face].surface
    assert isinstance(plane, Plane) and isinstance(reference, Cylinder)
    scale = max(reference.radius, abs(reference.height), 1.0)
    length_tolerance = (
        geometry.tolerance.effective_length(scale)
        if tolerance is None
        else float(tolerance)
    )
    parameter_tolerance = geometry.tolerance.effective_parameter(
        max(abs(reference.height), reference.radius), scale
    )
    angular_tolerance = geometry.tolerance.angular
    alignment = abs(float(plane.normal @ reference.axis))
    if abs(alignment - 1.0) > angular_tolerance:
        raise GeometryError(
            "closed plane-cylinder imprint requires a transverse plane whose "
            "normal is parallel to the cylinder axis"
        )
    if reference.height <= 0.0:
        raise GeometryError(
            "closed plane-cylinder imprint currently requires positive "
            "cylinder height"
        )
    denominator = float(plane.normal @ reference.axis)
    axial = float(plane.normal @ (plane.origin - reference.origin)) / denominator
    fraction = axial / reference.height
    if not parameter_tolerance < fraction < 1.0 - parameter_tolerance:
        raise GeometryError(
            "the transverse plane must cut strictly inside the cylinder height"
        )

    candidates: list[tuple[float, int, Cylinder]] = []
    period = 2.0 * np.pi
    for face_id, face in geometry.faces.items():
        surface = face.surface
        if not isinstance(surface, Cylinder) or not _same_cylinder_band(
            surface, reference, tolerance=max(angular_tolerance, length_tolerance / scale)
        ):
            continue
        if face.holes:
            raise GeometryError(
                "closed plane-cylinder imprint requires cylinder faces "
                "without existing holes"
            )
        if not angular_tolerance < surface.sweep_angle < period - angular_tolerance:
            raise GeometryError(
                "closed plane-cylinder imprint requires open cylindrical "
                "patches with positive sweeps below 2*pi"
            )
        candidates.append((surface.start_angle % period, face_id, surface))
    candidates.sort(key=lambda item: (item[0], item[1]))
    if len(candidates) < 3:
        raise GeometryError(
            "a closed cylinder band needs at least three conformal surface patches"
        )
    if abs(sum(item[2].sweep_angle for item in candidates) - period) > angular_tolerance:
        raise GeometryError(
            "the selected cylinder face does not belong to one complete band"
        )
    for current, following in zip(candidates, candidates[1:] + candidates[:1]):
        end_angle = current[0] + current[2].sweep_angle
        if _angular_distance(end_angle, following[0]) > angular_tolerance:
            raise GeometryError(
                "the selected cylinder band has an angular gap or overlap"
            )
        current_end = current[2].evaluate(1.0, fraction)
        following_start = following[2].evaluate(0.0, fraction)
        if float(np.linalg.norm(current_end - following_start)) > length_tolerance:
            raise GeometryError("adjacent cylinder patches do not meet exactly")
        if _boundary_edge_at_point(
            geometry, current[1], current_end
        ) != _boundary_edge_at_point(geometry, following[1], following_start):
            raise GeometryError(
                "the cylinder band is geometrically coincident but not "
                "topologically conformal"
            )

    plane_data = geometry.faces[plane_face]
    if plane_data.holes:
        raise GeometryError(
            "closed plane-cylinder imprint currently requires a plane face "
            "without existing holes"
        )
    if any(
        not isinstance(geometry.edges[item.edge].curve, Straight)
        for item in plane_data.loop
    ):
        raise GeometryError(
            "closed plane-cylinder imprint currently requires a straight-edged "
            "plane boundary"
        )
    polygon = np.asarray(
        [
            plane.local_uv(
                geometry.vertex_position(geometry.oriented_start_vertex(item))
            )
            for item in plane_data.loop
        ],
        dtype=float,
    )
    turns = []
    for previous, current, following in zip(
        np.roll(polygon, 1, axis=0), polygon, np.roll(polygon, -1, axis=0)
    ):
        first = current - previous
        second = following - current
        cross = float(first[0] * second[1] - first[1] * second[0])
        if abs(cross) > angular_tolerance:
            turns.append(np.sign(cross))
    if not turns or min(turns) != max(turns):
        raise GeometryError(
            "closed plane-cylinder imprint currently requires a convex plane face"
        )
    centre = reference.origin + axial * reference.axis
    centre_uv = plane.local_uv(centre)
    if not geometry.face_contains_uv(plane_face, centre_uv):
        raise GeometryError("the cylinder axis does not pass inside the plane face")
    boundary_distance = min(
        geometry.closest_edge_point(item.edge, centre)[2]
        for item in plane_data.loop
    )
    if boundary_distance <= reference.radius + length_tolerance:
        raise GeometryError(
            "the complete intersection ring must lie strictly inside the plane face"
        )
    return tuple(item[1] for item in candidates), fraction, centre


def _transverse_ring_curve(
    geometry: GeometryModel, face_ids: Sequence[int], fraction: float
) -> np.ndarray:
    pieces = []
    for face_id in face_ids:
        surface = geometry.faces[face_id].surface
        assert isinstance(surface, Cylinder)
        points = np.asarray(
            [surface.evaluate(float(u), fraction) for u in np.linspace(0.0, 1.0, 17)]
        )
        pieces.append(points[:-1])
    return np.vstack((*pieces, pieces[0][0][None, :]))


def _cylinder_fragment_surface(
    surface: Cylinder, fraction: float, *, upper: bool
) -> Cylinder:
    return Cylinder(
        surface.origin + (fraction * surface.height * surface.axis if upper else 0.0),
        surface.axis,
        surface.radial_direction,
        surface.radius,
        surface.height * ((1.0 - fraction) if upper else fraction),
        surface.start_angle,
        surface.sweep_angle,
    )


def _fragment_cylinder_transverse(
    geometry: GeometryModel,
    face_id: int,
    start: int,
    end: int,
    edge_id: int,
    fraction: float,
) -> tuple[int, int]:
    from .operations import _split_loop  # internal topology primitive

    face = geometry.faces[face_id]
    surface = face.surface
    assert isinstance(surface, Cylinder)
    first_chain, second_chain = _split_loop(face, start, end, geometry)
    metadata, tags = dict(face.metadata), geometry.tags_for(face.ref)
    parameterization = face.parameterization
    attachment_snapshots = _capture_face_attachments(geometry, face_id)
    geometry._delete_entity("face", face_id)  # noqa: SLF001
    made = []
    for chain, oriented_ring in (
        (first_chain, OrientedEdge(edge_id, False)),
        (second_chain, OrientedEdge(edge_id, True)),
    ):
        loop = tuple(chain) + (oriented_ring,)
        corners = geometry._detect_corners(loop) if len(loop) >= 4 else None  # noqa: SLF001
        identifier = geometry.add_face_from_loop(loop, corners)
        axial_values = [
            float(
                (
                    geometry.vertex_position(geometry.oriented_start_vertex(item))
                    - surface.origin
                )
                @ surface.axis
            )
            / surface.height
            for item in chain
        ]
        geometry._put_entity(  # noqa: SLF001
            "face",
            replace(
                geometry.faces[identifier],
                metadata=dict(metadata),
                surface=_cylinder_fragment_surface(
                    surface,
                    fraction,
                    upper=float(np.mean(axial_values)) > fraction,
                ),
                parameterization=parameterization,
            ),
        )
        geometry.tag(EntityRef("face", identifier), *tags)
        made.append(identifier)
    _remap_face_attachments(geometry, face, made, attachment_snapshots)
    geometry.record_replacement(
        EntityRef("face", face_id),
        tuple(EntityRef("face", item) for item in made),
    )
    return made[0], made[1]


def _fragment_plane_with_ring(
    geometry: GeometryModel, plane_face: int, edge_ids: Sequence[int]
) -> tuple[int, int]:
    face = geometry.faces[plane_face]
    surface = face.surface
    assert isinstance(surface, Plane)
    metadata, tags = dict(face.metadata), geometry.tags_for(face.ref)
    parameterization = face.parameterization
    outer_loop, corners = face.loop, face.corners
    ring_loop = tuple(OrientedEdge(edge_id, True) for edge_id in edge_ids)
    if _loop_signed_area(geometry, plane_face, outer_loop) * _loop_signed_area(
        geometry, plane_face, ring_loop
    ) < 0.0:
        ring_loop = _reverse_loop(ring_loop)
    attachment_snapshots = _capture_face_attachments(geometry, plane_face)
    geometry._delete_entity("face", plane_face)  # noqa: SLF001
    annulus = geometry.add_face_from_loop(outer_loop, corners, surface=surface)
    geometry._put_entity(  # noqa: SLF001
        "face",
        replace(
            geometry.faces[annulus],
            holes=(_reverse_loop(ring_loop),),
            metadata=dict(metadata),
            parameterization=parameterization,
        ),
    )
    disk = geometry.add_face_from_loop(ring_loop, surface=surface)
    geometry._put_entity(  # noqa: SLF001
        "face",
        replace(
            geometry.faces[disk],
            metadata=dict(metadata),
            parameterization=parameterization,
        ),
    )
    for identifier in (annulus, disk):
        geometry.tag(EntityRef("face", identifier), *tags)
    _remap_face_attachments(
        geometry, face, (annulus, disk), attachment_snapshots
    )
    geometry.record_replacement(
        EntityRef("face", plane_face),
        (EntityRef("face", annulus), EntityRef("face", disk)),
    )
    return annulus, disk


def _imprint_transverse_plane_cylinder(
    geometry: GeometryModel,
    plane_face: int,
    cylinder_face: int,
    *,
    fragment: bool,
) -> FaceIntersection | np.ndarray:
    face_ids, fraction, _centre = _transverse_cylinder_band(
        geometry, plane_face, cylinder_face
    )
    if not fragment:
        return _transverse_ring_curve(geometry, face_ids, fraction)

    with geometry.transaction():
        vertices: list[tuple[int, int]] = []
        edges = []
        for face_id in face_ids:
            surface = geometry.faces[face_id].surface
            assert isinstance(surface, Cylinder)
            start = _boundary_vertex(
                geometry, face_id, surface.evaluate(0.0, fraction)
            )
            end = _boundary_vertex(
                geometry, face_id, surface.evaluate(1.0, fraction)
            )
            via = geometry.add_point(*surface.evaluate(0.5, fraction))
            edge_id = geometry.add_arc(start, via, end)
            vertices.append((start, end))
            edges.append(edge_id)
        for current, following in zip(vertices, vertices[1:] + vertices[:1]):
            if current[1] != following[0]:
                raise GeometryError(
                    "the imprinted cylinder ring is not a continuous topology loop"
                )
        cylinder_faces = []
        for face_id, (start, end), edge_id in zip(face_ids, vertices, edges):
            cylinder_faces.extend(
                _fragment_cylinder_transverse(
                    geometry, face_id, start, end, edge_id, fraction
                )
            )
        plane_faces = _fragment_plane_with_ring(geometry, plane_face, edges)
        errors = geometry.validate_topology()
        if errors:
            raise GeometryError(
                "closed plane-cylinder imprint produced invalid topology: "
                + "; ".join(errors)
            )
        references = tuple(EntityRef("edge", edge_id) for edge_id in edges)
        return FaceIntersection(
            references[0],
            tuple(EntityRef("face", item) for item in plane_faces),
            tuple(EntityRef("face", item) for item in cylinder_faces),
            references,
        )


def intersect_faces(
    geometry: GeometryModel,
    first_face: int,
    second_face: int,
    *,
    fragment: bool = True,
    policy: MutationPolicy | str | None = None,
) -> FaceIntersection | tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Compatibility adapter over the query/plan/apply workflow.

    New code should call those stages directly.  ``fragment=False`` keeps the
    historical array/endpoint return shape while using the same trim-aware
    typed query and therefore never collapses concave or holed material to one
    artificial min/max interval.
    """

    if first_face == second_face:
        raise GeometryError("two distinct faces are required")
    result = query_intersection(
        geometry,
        geometry.handle("face", int(first_face)),
        geometry.handle("face", int(second_face)),
    )
    if not result.classified:
        raise GeometryError(
            "face intersection is not qualified: " + "; ".join(result.diagnostics)
        )
    if not fragment:
        if result.kind is IntersectionKind.DISJOINT:
            raise GeometryError("faces do not intersect")
        first_surface = geometry.faces[first_face].surface
        second_surface = geometry.faces[second_face].surface
        if (
            isinstance(first_surface, Plane)
            and isinstance(second_surface, Cylinder)
            and abs(abs(float(first_surface.normal @ second_surface.axis)) - 1.0)
            <= geometry.tolerance.angular
        ):
            return _transverse_ring_curve(
                geometry,
                *_transverse_cylinder_band(
                    geometry, first_face, second_face
                )[:2],
            )
        if (
            isinstance(first_surface, Cylinder)
            and isinstance(second_surface, Plane)
            and abs(abs(float(second_surface.normal @ first_surface.axis)) - 1.0)
            <= geometry.tolerance.angular
        ):
            face_ids, fraction, _centre = _transverse_cylinder_band(
                geometry, second_face, first_face
            )
            return _transverse_ring_curve(geometry, face_ids, fraction)
        if len(result.components) != 1 or len(result.components[0].witnesses) < 2:
            raise GeometryError(
                "legacy face query cannot represent multiple intersection components; "
                "use query_intersection"
            )
        return (
            np.asarray(result.components[0].witnesses[0], dtype=float),
            np.asarray(result.components[0].witnesses[-1], dtype=float),
        )
    if policy is None:
        raise GeometryError(
            "topology-changing face intersection requires an explicit mutation policy"
        )
    normalized = _normalize_intersection_policy(policy)
    policy_value = _policy_name(normalized)
    if policy_value == "reject":
        raise GeometryError("intersection mutation rejected by policy")
    if policy_value in (
        "keep_separate_part",
        "keep_disconnected",
        "contact_only",
    ):
        return intersect_faces(
            geometry, first_face, second_face, fragment=False
        )
    if policy_value != "imprint":
        raise GeometryError(
            f"{policy_value} is not qualified for face imprinting"
        )
    plan = plan_imprint(geometry, result, policy=policy)
    application = apply_imprint(geometry, plan, policy=policy)
    if application.face_intersection is None:
        raise GeometryError("face imprint did not produce shared topology")
    return application.face_intersection


# ---------------------------------------------------------------------------
# Qualified public query / plan / apply workflow
# ---------------------------------------------------------------------------


def _normalize_operand(
    geometry: GeometryModel,
    value: EntityHandle | EntityRef | tuple[str, int] | int,
) -> EntityHandle:
    if isinstance(value, EntityHandle):
        resolution = geometry.resolve_handle(value)
        if resolution.status is ResolutionStatus.ACTIVE:
            return value
        if resolution.status is ResolutionStatus.REPLACED and len(resolution.resolved) == 1:
            return resolution.resolved[0]
        raise GeometryError(
            f"intersection operand {value} is not one active entity: "
            f"{resolution.status.value}"
        )
    if isinstance(value, EntityRef):
        return geometry.handle(
            validate_entity_kind(value.kind),
            validate_local_id(value.id, name="intersection operand ID"),
        )
    if isinstance(value, tuple) and len(value) == 2:
        return geometry.handle(
            validate_entity_kind(value[0]),
            validate_local_id(value[1], name="intersection operand ID"),
        )
    try:
        identifier = validate_local_id(value, name="intersection face ID")
    except GeometryError as error:
        raise GeometryError(
            "intersection operands must be handles, local refs, keys, or "
            "positive integer face IDs"
        ) from error
    # Compatibility with the historical face/face API.  New cross-kind
    # callers should always pass handles, which cannot be ambiguous.
    return geometry.handle("face", identifier)


def _qualified_result(
    geometry: GeometryModel,
    first: EntityHandle,
    second: EntityHandle,
    kind: IntersectionKind,
    components: Sequence[IntersectionComponent] = (),
    *,
    diagnostics: Sequence[str] = (),
    dimension: IntersectionDimension | None = None,
    tolerance_used: float | None = None,
) -> IntersectionResult:
    return IntersectionResult(
        kind,
        tuple(components),
        tuple(diagnostics),
        dimension,
        first,
        second,
        geometry.tolerance.length if tolerance_used is None else tolerance_used,
    )


def _face_polygon_in_plane(
    geometry: GeometryModel,
    face_id: int,
    plane: Plane,
):
    try:
        from shapely.geometry import Polygon
    except ImportError:  # pragma: no cover - optional dependency
        return None
    face = geometry.faces[face_id]
    if any(
        not isinstance(geometry.edges[item.edge].curve, Straight)
        for loop in (face.loop,) + face.holes
        for item in loop
    ):
        raise GeometryError("curved planar trim overlay is unsupported")

    def points(loop: Sequence[OrientedEdge]) -> list[tuple[float, float]]:
        return [
            tuple(
                float(item)
                for item in plane.local_uv(
                    geometry.vertex_position(geometry.oriented_start_vertex(edge_use))
                )
            )
            for edge_use in loop
        ]

    return Polygon(points(face.loop), [points(loop) for loop in face.holes])


def _overlay_components(
    value,
    first_plane: Plane,
    second_plane: Plane,
    first: EntityHandle,
    second: EntityHandle,
) -> tuple[IntersectionComponent, ...]:
    components: list[IntersectionComponent] = []

    def collect(item) -> None:
        if item.is_empty:
            return
        geometry_type = item.geom_type
        if geometry_type == "Polygon":
            coordinates = tuple(item.exterior.coords)[:-1]
        elif geometry_type == "LineString":
            coordinates = tuple(item.coords)
        elif geometry_type == "Point":
            coordinates = (tuple(item.coords)[0],)
        else:
            for child in getattr(item, "geoms", ()):  # deterministic backend order
                collect(child)
            return
        if not coordinates:
            return
        witnesses = tuple(
            tuple(
                float(component)
                for component in first_plane.evaluate(float(uv[0]), float(uv[1]))
            )
            for uv in coordinates
        )
        second_parameters = tuple(
            tuple(
                float(component)
                for component in second_plane.local_uv(np.asarray(point, dtype=float))
            )
            for point in witnesses
        )
        components.append(
            IntersectionComponent(
                witnesses,
                IntersectionQuality.VERIFIED_APPROXIMATE,
                first_parameter=(float(coordinates[0][0]), float(coordinates[0][1])),
                second_parameter=second_parameters[0],
                first_parameter_path=tuple(
                    (float(uv[0]), float(uv[1])) for uv in coordinates
                ),
                second_parameter_path=second_parameters,
                first_subparent=first,
                second_subparent=second,
            )
        )

    collect(value)
    components.sort(key=lambda item: (item.witnesses[0], len(item.witnesses)))
    return tuple(components)


def _query_planar_faces(
    geometry: GeometryModel,
    first: EntityHandle,
    second: EntityHandle,
    first_plane: Plane,
    second_plane: Plane,
) -> IntersectionResult:
    scale = max(
        _face_length_scale(geometry, first.id),
        _face_length_scale(geometry, second.id),
    )
    tolerance = geometry.tolerance.effective_length(scale)
    supports = qualified_plane_plane(
        first_plane,
        second_plane,
        policy=geometry.tolerance,
        characteristic_length=scale,
    )
    if supports.kind is IntersectionKind.DISJOINT:
        return _qualified_result(
            geometry,
            first,
            second,
            IntersectionKind.DISJOINT,
            diagnostics=("parallel_face_supports",),
            tolerance_used=tolerance,
        )
    if supports.kind is IntersectionKind.COINCIDENT:
        try:
            first_polygon = _face_polygon_in_plane(
                geometry, first.id, first_plane
            )
            second_polygon = _face_polygon_in_plane(
                geometry, second.id, first_plane
            )
        except GeometryError as error:
            return _qualified_result(
                geometry,
                first,
                second,
                IntersectionKind.UNSUPPORTED,
                diagnostics=(str(error),),
                tolerance_used=tolerance,
            )
        if first_polygon is None or second_polygon is None:
            return _qualified_result(
                geometry,
                first,
                second,
                IntersectionKind.CAPABILITY_MISSING,
                diagnostics=("planar_backend_unavailable",),
                tolerance_used=tolerance,
            )
        if (
            first_polygon.is_empty
            or second_polygon.is_empty
            or not first_polygon.is_valid
            or not second_polygon.is_valid
        ):
            return _qualified_result(
                geometry,
                first,
                second,
                IntersectionKind.UNCLASSIFIED,
                diagnostics=("invalid_planar_face_polygon",),
                tolerance_used=tolerance,
            )
        overlap = first_polygon.intersection(second_polygon)
        if overlap.is_empty:
            return _qualified_result(
                geometry,
                first,
                second,
                IntersectionKind.DISJOINT,
                diagnostics=("coplanar_faces_disjoint",),
                tolerance_used=tolerance,
            )
        components = _overlay_components(
            overlap, first_plane, second_plane, first, second
        )
        area_tolerance = geometry.tolerance.effective_area(scale)
        if float(overlap.area) > area_tolerance:
            symmetric_area = float(first_polygon.symmetric_difference(second_polygon).area)
            first_remainder = float(first_polygon.difference(second_polygon).area)
            second_remainder = float(second_polygon.difference(first_polygon).area)
            if symmetric_area <= area_tolerance:
                kind = IntersectionKind.COINCIDENT
            elif first_remainder <= area_tolerance or second_remainder <= area_tolerance:
                kind = IntersectionKind.CONTAINED
            else:
                kind = IntersectionKind.OVERLAP_REGION
            return _qualified_result(
                geometry,
                first,
                second,
                kind,
                components,
                diagnostics=("coplanar_face_overlay",),
                dimension=IntersectionDimension.REGION,
                tolerance_used=tolerance,
            )
        length = float(getattr(overlap, "length", 0.0))
        if length > tolerance:
            return _qualified_result(
                geometry,
                first,
                second,
                IntersectionKind.OVERLAP_CURVE,
                components,
                diagnostics=("coplanar_boundary_overlap",),
                dimension=IntersectionDimension.CURVE,
                tolerance_used=tolerance,
            )
        return _qualified_result(
            geometry,
            first,
            second,
            IntersectionKind.TOUCH_POINT,
            components,
            diagnostics=("coplanar_point_touch",),
            tolerance_used=tolerance,
        )
    if supports.kind is not IntersectionKind.CROSS or not supports.components:
        return _qualified_result(
            geometry,
            first,
            second,
            supports.kind,
            diagnostics=supports.diagnostics,
            tolerance_used=tolerance,
        )

    support_component = supports.components[0]
    point = np.asarray(support_component.witnesses[0], dtype=float)
    direction = np.asarray(support_component.direction, dtype=float)
    first_clip = clip_line_to_face(geometry, first.id, point, direction)
    second_clip = clip_line_to_face(geometry, second.id, point, direction)
    for result in (first_clip, second_clip):
        if result.kind in (
            IntersectionKind.UNCLASSIFIED,
            IntersectionKind.UNSUPPORTED,
            IntersectionKind.CAPABILITY_MISSING,
        ):
            return _qualified_result(
                geometry,
                first,
                second,
                result.kind,
                diagnostics=result.diagnostics,
                tolerance_used=tolerance,
            )
    first_intervals = tuple(
        component.first_parameter_range
        for component in first_clip.components
        if component.first_parameter_range is not None
    )
    second_intervals = tuple(
        component.first_parameter_range
        for component in second_clip.components
        if component.first_parameter_range is not None
    )
    first_points = tuple(
        component.first_parameter[0]
        for component in first_clip.components
        if component.first_parameter is not None
    )
    second_points = tuple(
        component.first_parameter[0]
        for component in second_clip.components
        if component.first_parameter is not None
    )
    intersections: list[IntersectionComponent] = []
    touches: list[IntersectionComponent] = []

    def append_touch(parameter: float) -> None:
        witness = point + parameter * direction
        if any(
            float(
                np.linalg.norm(
                    np.asarray(existing.witnesses[0], dtype=float) - witness
                )
            )
            <= tolerance
            for existing in touches
        ):
            return
        touches.append(
            IntersectionComponent(
                (tuple(float(item) for item in witness),),
                IntersectionQuality.VERIFIED_APPROXIMATE,
                first_parameter=(parameter,),
                second_parameter=tuple(
                    float(item) for item in second_plane.local_uv(witness)
                ),
                first_subparent=first,
                second_subparent=second,
            )
        )

    for first_range in first_intervals:
        for second_range in second_intervals:
            lower = max(first_range.lower, second_range.lower)
            upper = min(first_range.upper, second_range.upper)
            if upper < lower - tolerance:
                continue
            if upper - lower <= tolerance:
                append_touch(0.5 * (lower + upper))
                continue
            witnesses = tuple(point + value * direction for value in (lower, upper))
            intersections.append(
                IntersectionComponent(
                    tuple(tuple(float(item) for item in witness) for witness in witnesses),
                    IntersectionQuality.VERIFIED_APPROXIMATE,
                    first_parameter_range=IntersectionParameterRange(lower, upper),
                    first_parameter_path=((lower,), (upper,)),
                    second_parameter_path=tuple(
                        tuple(float(item) for item in second_plane.local_uv(witness))
                        for witness in witnesses
                    ),
                    direction=tuple(float(item) for item in direction),
                    first_subparent=first,
                    second_subparent=second,
                )
            )
    for parameter in first_points:
        if any(
            interval.lower - tolerance <= parameter <= interval.upper + tolerance
            for interval in second_intervals
        ) or any(abs(parameter - other) <= tolerance for other in second_points):
            append_touch(parameter)
    for parameter in second_points:
        if any(
            interval.lower - tolerance <= parameter <= interval.upper + tolerance
            for interval in first_intervals
        ):
            append_touch(parameter)
    if intersections:
        intersections.sort(key=lambda item: item.first_parameter_range.lower)  # type: ignore[union-attr]
        return _qualified_result(
            geometry,
            first,
            second,
            IntersectionKind.CROSS,
            intersections,
            diagnostics=("bounded_face_crossing",),
            dimension=IntersectionDimension.CURVE,
            tolerance_used=tolerance,
        )
    if touches:
        touches.sort(key=lambda item: item.witnesses)
        return _qualified_result(
            geometry,
            first,
            second,
            IntersectionKind.TOUCH_POINT,
            touches,
            diagnostics=("bounded_face_touch",),
            tolerance_used=tolerance,
        )
    return _qualified_result(
        geometry,
        first,
        second,
        IntersectionKind.DISJOINT,
        diagnostics=("face_material_intervals_disjoint",),
        tolerance_used=tolerance,
    )


def _curve_definition_points(
    geometry: GeometryModel, edge_id: int
) -> tuple[np.ndarray, ...]:
    edge = geometry.edges[edge_id]
    vertex_ids = [edge.start]
    if isinstance(edge.curve, Arc):
        vertex_ids.append(edge.curve.via_vertex)
    elif isinstance(edge.curve, Spline):
        vertex_ids.extend(edge.curve.control_vertices)
    vertex_ids.append(edge.end)
    return tuple(geometry.vertex_position(vertex_id) for vertex_id in vertex_ids)


def _angle_on_arc_sweep(angle: float, sweep: float, angular_tolerance: float) -> bool:
    for offset in range(-2, 3):
        candidate = angle + offset * 2.0 * np.pi
        if sweep >= 0.0:
            if -angular_tolerance <= candidate <= sweep + angular_tolerance:
                return True
        elif sweep - angular_tolerance <= candidate <= angular_tolerance:
            return True
    return False


def _certified_curve_inside_convex_support(
    geometry: GeometryModel,
    support_face_id: int,
    edge_id: int,
    *,
    tolerance: float,
) -> tuple[bool, str]:
    """Certify a complete existing curve strictly inside a planar face.

    Straight segments and Bezier curves use their defining-point convex hull;
    circular arcs use analytical half-space extrema over their actual sweep.
    The certificate is deliberately sufficient rather than permissive, and no
    sampled curve is ever promoted into persistent topology.
    """

    face = geometry.faces[support_face_id]
    surface = face.surface
    if not isinstance(surface, Plane):
        return False, "boundary_curve_support_requires_plane"
    if face.holes:
        return False, "boundary_curve_support_holes_are_unsupported"
    if len(face.loop) < 3 or any(
        not isinstance(geometry.edges[item.edge].curve, Straight)
        for item in face.loop
    ):
        return False, "boundary_curve_support_requires_straight_convex_trim"

    boundary = np.asarray(
        [
            surface.local_uv(
                geometry.vertex_position(geometry.oriented_start_vertex(item))
            )
            for item in face.loop
        ],
        dtype=float,
    )
    vectors = np.roll(boundary, -1, axis=0) - boundary
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= tolerance):
        return False, "boundary_curve_support_has_degenerate_trim"
    area_twice = float(
        np.sum(
            boundary[:, 0] * np.roll(boundary[:, 1], -1)
            - boundary[:, 1] * np.roll(boundary[:, 0], -1)
        )
    )
    area_tolerance = geometry.tolerance.effective_area(
        _face_length_scale(geometry, support_face_id)
    )
    if abs(area_twice) <= area_tolerance:
        return False, "boundary_curve_support_has_degenerate_trim"
    orientation = 1.0 if area_twice > 0.0 else -1.0
    turns = orientation * np.asarray(
        [
            vectors[index, 0] * vectors[(index + 1) % len(vectors), 1]
            - vectors[index, 1] * vectors[(index + 1) % len(vectors), 0]
            for index in range(len(vectors))
        ],
        dtype=float,
    )
    if np.any(turns < -area_tolerance):
        return False, "boundary_curve_support_must_be_convex"
    inward = orientation * np.column_stack((-vectors[:, 1], vectors[:, 0]))
    inward /= lengths[:, None]

    definition_points = _curve_definition_points(geometry, edge_id)
    residual_tolerance = geometry.tolerance.effective_surface_residual(
        _face_length_scale(geometry, support_face_id)
    )
    if any(
        abs(float((point - surface.origin) @ surface.normal)) > residual_tolerance
        for point in definition_points
    ):
        return False, "boundary_curve_is_not_coplanar_with_support"

    edge = geometry.edges[edge_id]
    if isinstance(edge.curve, Arc):
        frame = arc_frame(
            definition_points[0], definition_points[1], definition_points[2]
        )
        center_uv = np.asarray(surface.local_uv(frame.center), dtype=float)
        e1_uv = (
            np.asarray(surface.local_uv(frame.center + frame.e1), dtype=float)
            - center_uv
        )
        e2_uv = (
            np.asarray(surface.local_uv(frame.center + frame.e2), dtype=float)
            - center_uv
        )
        for origin, normal in zip(boundary, inward):
            constant = float(normal @ (center_uv - origin))
            cosine = float(frame.radius * (normal @ e1_uv))
            sine = float(frame.radius * (normal @ e2_uv))
            angles = [0.0, frame.sweep]
            minimum_angle = float(np.arctan2(sine, cosine) + np.pi)
            for offset in range(-2, 3):
                candidate = minimum_angle + offset * 2.0 * np.pi
                if _angle_on_arc_sweep(
                    candidate, frame.sweep, geometry.tolerance.angular
                ):
                    angles.append(candidate)
            minimum = min(
                constant + cosine * np.cos(angle) + sine * np.sin(angle)
                for angle in angles
            )
            if minimum <= tolerance:
                return False, "boundary_curve_touches_or_leaves_support_trim"
        return True, "arc_sweep_halfspace_certificate"

    definition_uv = np.asarray(
        [surface.local_uv(point) for point in definition_points], dtype=float
    )
    margins = np.asarray(
        [
            inward[index] @ (definition_uv - boundary[index]).T
            for index in range(len(boundary))
        ]
    )
    if np.any(margins <= tolerance):
        return False, "boundary_curve_control_hull_touches_or_leaves_support_trim"
    certificate = (
        "bezier_control_hull_certificate"
        if isinstance(edge.curve, Spline)
        else "straight_endpoint_certificate"
    )
    return True, certificate


def _boundary_curve_component(
    geometry: GeometryModel,
    first: EntityHandle,
    second: EntityHandle,
    *,
    plane_face_id: int,
    edge_id: int,
    first_is_plane: bool,
) -> IntersectionComponent:
    edge = geometry.edges[edge_id]
    endpoints = tuple(
        tuple(float(value) for value in geometry.vertex_position(vertex_id))
        for vertex_id in (edge.start, edge.end)
    )
    plane = geometry.faces[plane_face_id].surface
    assert isinstance(plane, Plane)
    plane_path = tuple(
        tuple(float(value) for value in plane.local_uv(point)) for point in endpoints
    )
    edge_path = ((0.0,), (1.0,))
    edge_handle = geometry.handle("edge", edge_id)
    return IntersectionComponent(
        endpoints,
        IntersectionQuality.EXACT,
        first_parameter_path=plane_path if first_is_plane else edge_path,
        second_parameter_path=edge_path if first_is_plane else plane_path,
        first_subparent=first if first_is_plane else edge_handle,
        second_subparent=edge_handle if first_is_plane else second,
    )


def _query_planar_support_boundary_curve(
    geometry: GeometryModel,
    first: EntityHandle,
    second: EntityHandle,
    *,
    plane_face_id: int,
    curved_face_id: int,
    first_is_plane: bool,
) -> IntersectionResult:
    scale = max(
        _face_length_scale(geometry, plane_face_id),
        _face_length_scale(geometry, curved_face_id),
    )
    tolerance = geometry.tolerance.effective_length(scale)
    plane_edges = {
        item.edge
        for loop in (geometry.faces[plane_face_id].loop,)
        + geometry.faces[plane_face_id].holes
        for item in loop
    }
    curved_edges = tuple(
        sorted(
            {
                item.edge
                for loop in (geometry.faces[curved_face_id].loop,)
                + geometry.faces[curved_face_id].holes
                for item in loop
            }
        )
    )
    shared = tuple(edge_id for edge_id in curved_edges if edge_id in plane_edges)
    if len(shared) == 1:
        component = _boundary_curve_component(
            geometry,
            first,
            second,
            plane_face_id=plane_face_id,
            edge_id=shared[0],
            first_is_plane=first_is_plane,
        )
        return _qualified_result(
            geometry,
            first,
            second,
            IntersectionKind.OVERLAP_CURVE,
            (component,),
            diagnostics=("existing_shared_boundary_curve",),
            dimension=IntersectionDimension.CURVE,
            tolerance_used=tolerance,
        )
    if len(shared) > 1:
        return _qualified_result(
            geometry,
            first,
            second,
            IntersectionKind.UNSUPPORTED,
            diagnostics=("multiple_shared_boundary_curves_are_unsupported",),
            tolerance_used=tolerance,
        )

    certified: list[tuple[int, str]] = []
    rejections: list[str] = []
    for edge_id in curved_edges:
        accepted, diagnostic = _certified_curve_inside_convex_support(
            geometry,
            plane_face_id,
            edge_id,
            tolerance=tolerance,
        )
        if accepted:
            certified.append((edge_id, diagnostic))
        else:
            rejections.append(f"edge_{edge_id}:{diagnostic}")
    if len(certified) != 1:
        diagnostic = (
            "multiple_complete_boundary_curves_on_planar_support"
            if len(certified) > 1
            else "no_complete_boundary_curve_on_planar_support"
        )
        return _qualified_result(
            geometry,
            first,
            second,
            IntersectionKind.UNSUPPORTED,
            diagnostics=(diagnostic, *rejections),
            tolerance_used=tolerance,
        )
    edge_id, certificate = certified[0]
    component = _boundary_curve_component(
        geometry,
        first,
        second,
        plane_face_id=plane_face_id,
        edge_id=edge_id,
        first_is_plane=first_is_plane,
    )
    return _qualified_result(
        geometry,
        first,
        second,
        IntersectionKind.CONTAINED,
        (component,),
        diagnostics=(
            "certified_nonplanar_boundary_curve_on_planar_support",
            certificate,
            f"boundary_edge:{edge_id}",
        ),
        dimension=IntersectionDimension.CURVE,
        tolerance_used=tolerance,
    )


def _query_plane_cylinder_faces(
    geometry: GeometryModel,
    first: EntityHandle,
    second: EntityHandle,
    plane_face: int,
    cylinder_face: int,
    *,
    first_is_plane: bool,
    plane: Plane | None = None,
) -> IntersectionResult:
    if plane is None:
        candidate = geometry.faces[plane_face].surface
        plane = candidate if isinstance(candidate, Plane) else None
    cylinder = geometry.faces[cylinder_face].surface
    assert plane is not None and isinstance(cylinder, Cylinder)
    scale = max(
        _face_length_scale(geometry, plane_face),
        _face_length_scale(geometry, cylinder_face),
    )
    tolerance = geometry.tolerance.effective_surface_residual(scale)
    raw_curves = plane_cylinder(plane, cylinder, samples=257)
    components: list[IntersectionComponent] = []
    for raw_curve in raw_curves:
        points = np.asarray(raw_curve, dtype=float)
        if points.ndim != 2 or points.shape[1:] != (3,) or len(points) == 0:
            continue
        material: list[np.ndarray] = []
        for point in points:
            keep = True
            for face_id in (plane_face, cylinder_face):
                _projected, uv, distance = geometry.project_to_face(face_id, point)
                if distance > tolerance or not geometry.face_contains_uv(face_id, uv):
                    keep = False
                    break
            if keep:
                material.append(point)
        if not material:
            continue
        # The analytical plane/cylinder generator is parameter ordered.  Split
        # holes/gaps instead of reconnecting non-adjacent retained samples.
        runs: list[list[np.ndarray]] = []
        current: list[np.ndarray] = []
        maximum_step = max(
            4.0 * scale / max(len(points) - 1, 1),
            16.0 * tolerance,
        )
        for point in material:
            if current and float(np.linalg.norm(point - current[-1])) > maximum_step:
                runs.append(current)
                current = []
            current.append(point)
        if current:
            runs.append(current)
        for run in runs:
            witnesses_array = (run[0],) if len(run) == 1 else (run[0], run[-1])
            plane_path = tuple(
                tuple(float(item) for item in plane.local_uv(point))
                for point in witnesses_array
            )
            cylinder_path = tuple(
                tuple(float(item) for item in cylinder.local_uv(point))
                for point in witnesses_array
            )
            components.append(
                IntersectionComponent(
                    tuple(
                        tuple(float(item) for item in point)
                        for point in witnesses_array
                    ),
                    IntersectionQuality.VERIFIED_APPROXIMATE,
                    first_parameter_path=(plane_path if first_is_plane else cylinder_path),
                    second_parameter_path=(cylinder_path if first_is_plane else plane_path),
                    max_residual=tolerance,
                    first_subparent=first,
                    second_subparent=second,
                )
            )
    if not components:
        return _qualified_result(
            geometry,
            first,
            second,
            IntersectionKind.DISJOINT,
            diagnostics=("bounded_plane_cylinder_material_disjoint",),
            tolerance_used=tolerance,
        )
    components.sort(key=lambda item: item.witnesses)
    point_only = all(len(component.witnesses) == 1 for component in components)
    return _qualified_result(
        geometry,
        first,
        second,
        IntersectionKind.TOUCH_POINT if point_only else IntersectionKind.CROSS,
        components,
        diagnostics=("analytical_plane_cylinder",),
        dimension=(
            IntersectionDimension.POINT
            if point_only
            else IntersectionDimension.CURVE
        ),
        tolerance_used=tolerance,
    )


def _query_face_face(
    geometry: GeometryModel,
    first: EntityHandle,
    second: EntityHandle,
    qualification: IntersectionQualificationPolicy = DEFAULT_INTERSECTION_QUALIFICATION_POLICY,
) -> IntersectionResult:
    if first.id == second.id:
        component = IntersectionComponent(
            (
                tuple(
                    float(item)
                    for item in geometry.vertex_position(
                        geometry.oriented_start_vertex(
                            geometry.faces[first.id].loop[0]
                        )
                    )
                ),
            ),
            IntersectionQuality.EXACT,
            first_subparent=first,
            second_subparent=second,
        )
        return _qualified_result(
            geometry,
            first,
            second,
            IntersectionKind.COINCIDENT,
            (component,),
            diagnostics=("same_face_identity",),
            dimension=IntersectionDimension.REGION,
        )
    first_surface = geometry.faces[first.id].surface
    second_surface = geometry.faces[second.id].surface
    try:
        first_plane = _qualified_face_plane(geometry, first.id)
    except GeometryError:
        first_plane = None
    try:
        second_plane = _qualified_face_plane(geometry, second.id)
    except GeometryError:
        second_plane = None
    if first_plane is not None and second_plane is not None:
        return _query_planar_faces(
            geometry, first, second, first_plane, second_plane
        )
    if first_plane is not None and second_surface is not None:
        boundary = _query_planar_support_boundary_curve(
            geometry,
            first,
            second,
            plane_face_id=first.id,
            curved_face_id=second.id,
            first_is_plane=True,
        )
        if boundary.kind not in (
            IntersectionKind.UNSUPPORTED,
            IntersectionKind.UNCLASSIFIED,
            IntersectionKind.CAPABILITY_MISSING,
        ):
            return boundary
        if isinstance(second_surface, (RuledSurface, CoonsSurface)) and any(
            marker in diagnostic
            for diagnostic in boundary.diagnostics
            for marker in (
                "must_be_convex",
                "holes_are_unsupported",
                "curved_trim",
                "touches_or_leaves_support_trim",
            )
        ):
            return _qualified_result(
                geometry,
                first,
                second,
                IntersectionKind.UNCLASSIFIED,
                diagnostics=(
                    "planar_boundary_curve_material_qualification_unresolved",
                    *boundary.diagnostics,
                ),
            )
    elif second_plane is not None and first_surface is not None:
        boundary = _query_planar_support_boundary_curve(
            geometry,
            first,
            second,
            plane_face_id=second.id,
            curved_face_id=first.id,
            first_is_plane=False,
        )
        if boundary.kind not in (
            IntersectionKind.UNSUPPORTED,
            IntersectionKind.UNCLASSIFIED,
            IntersectionKind.CAPABILITY_MISSING,
        ):
            return boundary
        if isinstance(first_surface, (RuledSurface, CoonsSurface)) and any(
            marker in diagnostic
            for diagnostic in boundary.diagnostics
            for marker in (
                "must_be_convex",
                "holes_are_unsupported",
                "curved_trim",
                "touches_or_leaves_support_trim",
            )
        ):
            return _qualified_result(
                geometry,
                first,
                second,
                IntersectionKind.UNCLASSIFIED,
                diagnostics=(
                    "planar_boundary_curve_material_qualification_unresolved",
                    *boundary.diagnostics,
                ),
            )
    # Every built-in curved support pair, including Plane/curved pairs, is
    # routed through the shared certified engine used by strict and local
    # audit.  Legacy sampled special cases remain private compatibility
    # helpers; they no longer classify the public query workflow.
    return qualified_face_face(geometry, first, second, qualification)


def _member_parameter(use, edge_parameter: float) -> float:
    from .structural import Orientation

    local = (
        edge_parameter
        if use.orientation is Orientation.FORWARD
        else 1.0 - edge_parameter
    )
    return use.parent_range.start + local * use.parent_range.length


def _component_on_members(
    geometry: GeometryModel,
    component: IntersectionComponent,
    first_use,
    second_use,
    first_edge: EntityHandle,
    second_edge: EntityHandle,
) -> IntersectionComponent:
    first_parameter = (
        None
        if component.first_parameter is None
        else (_member_parameter(first_use, component.first_parameter[0]),)
    )
    second_parameter = (
        None
        if component.second_parameter is None
        else (_member_parameter(second_use, component.second_parameter[0]),)
    )
    first_range = None
    if component.first_parameter_range is not None:
        first_range = IntersectionParameterRange(
            _member_parameter(first_use, component.first_parameter_range.start),
            _member_parameter(first_use, component.first_parameter_range.end),
        )
    second_range = None
    if component.second_parameter_range is not None:
        second_range = IntersectionParameterRange(
            _member_parameter(second_use, component.second_parameter_range.start),
            _member_parameter(second_use, component.second_parameter_range.end),
        )
    return replace(
        component,
        first_parameter=first_parameter,
        second_parameter=second_parameter,
        first_parameter_range=first_range,
        second_parameter_range=second_range,
        first_parameter_path=(
            () if first_parameter is None else (first_parameter,)
        ),
        second_parameter_path=(
            () if second_parameter is None else (second_parameter,)
        ),
        first_subparent=first_edge,
        second_subparent=second_edge,
    )


def _query_same_circle_arcs(
    geometry: GeometryModel,
    first_edge_id: int,
    second_edge_id: int,
) -> IntersectionResult:
    first_frame = geometry.arc_frame(first_edge_id)
    second_frame = geometry.arc_frame(second_edge_id)
    scale = max(first_frame.radius, second_frame.radius, 1.0)
    tolerance = geometry.tolerance.effective_length(scale)
    if (
        float(np.linalg.norm(first_frame.center - second_frame.center)) > tolerance
        or abs(first_frame.radius - second_frame.radius) > tolerance
        or abs(float(first_frame.normal @ second_frame.normal))
        < 1.0 - geometry.tolerance.angular
    ):
        return IntersectionResult(
            IntersectionKind.UNSUPPORTED,
            diagnostics=("non_coincident_arc_supports",),
            tolerance_used=tolerance,
        )

    samples = geometry.sample_edge(second_edge_id, np.asarray((0.0, 0.5, 1.0)))
    angles = np.unwrap(
        np.asarray(
            [
                np.arctan2(
                    float((point - first_frame.center) @ first_frame.e2),
                    float((point - first_frame.center) @ first_frame.e1),
                )
                for point in samples
            ]
        )
    )
    best: tuple[float, float, float] | None = None
    first_lower, first_upper = sorted((0.0, first_frame.sweep))
    for shift in range(-2, 3):
        shifted = angles + shift * 2.0 * np.pi
        lower = max(first_lower, float(min(shifted[0], shifted[-1])))
        upper = min(first_upper, float(max(shifted[0], shifted[-1])))
        overlap = upper - lower
        if best is None or overlap > best[0]:
            best = (overlap, lower, upper)
    assert best is not None
    angular_tolerance = tolerance / max(first_frame.radius, tolerance)
    if best[0] < -angular_tolerance:
        return IntersectionResult(
            IntersectionKind.DISJOINT,
            diagnostics=("same_circle_arcs_disjoint",),
            tolerance_used=tolerance,
        )
    lower, upper = best[1], best[2]
    if upper - lower <= angular_tolerance:
        first_parameter = float(np.clip(0.5 * (lower + upper) / first_frame.sweep, 0.0, 1.0))
        witness = geometry.sample_edge(first_edge_id, np.asarray((first_parameter,)))[0]
        _other, second_parameter, residual = geometry.closest_edge_point(
            second_edge_id, witness
        )
        return IntersectionResult(
            IntersectionKind.TOUCH_POINT,
            (
                IntersectionComponent(
                    (tuple(float(item) for item in witness),),
                    IntersectionQuality.VERIFIED_APPROXIMATE,
                    first_parameter=(first_parameter,),
                    second_parameter=(second_parameter,),
                    max_residual=residual,
                ),
            ),
            tolerance_used=tolerance,
        )
    first_parameters = tuple(
        float(np.clip(value / first_frame.sweep, 0.0, 1.0))
        for value in (lower, upper)
    )
    witnesses = tuple(
        geometry.sample_edge(first_edge_id, np.asarray((parameter,)))[0]
        for parameter in first_parameters
    )
    second_parameters = tuple(
        geometry.closest_edge_point(second_edge_id, witness)[1]
        for witness in witnesses
    )
    first_full = min(first_parameters) <= geometry.tolerance.parameter and max(first_parameters) >= 1.0 - geometry.tolerance.parameter
    second_full = min(second_parameters) <= geometry.tolerance.parameter and max(second_parameters) >= 1.0 - geometry.tolerance.parameter
    kind = (
        IntersectionKind.COINCIDENT
        if first_full and second_full
        else (
            IntersectionKind.CONTAINED
            if first_full != second_full
            else IntersectionKind.OVERLAP_CURVE
        )
    )
    return IntersectionResult(
        kind,
        (
            IntersectionComponent(
                tuple(tuple(float(item) for item in witness) for witness in witnesses),
                IntersectionQuality.VERIFIED_APPROXIMATE,
                first_parameter_range=IntersectionParameterRange(*first_parameters),
                second_parameter_range=IntersectionParameterRange(*second_parameters),
                first_parameter_path=tuple((value,) for value in first_parameters),
                second_parameter_path=tuple((value,) for value in second_parameters),
                max_residual=max(
                    geometry.closest_edge_point(second_edge_id, witness)[2]
                    for witness in witnesses
                ),
            ),
        ),
        tolerance_used=tolerance,
    )


def _query_member_member(
    geometry: GeometryModel,
    first: EntityHandle,
    second: EntityHandle,
    qualification: IntersectionQualificationPolicy = DEFAULT_INTERSECTION_QUALIFICATION_POLICY,
) -> IntersectionResult:
    first_member = geometry.members[first.id]
    second_member = geometry.members[second.id]
    components: list[tuple[IntersectionKind, IntersectionComponent]] = []
    unsupported: list[str] = []
    tolerance = geometry.tolerance.length
    for first_use_id in first_member.edge_use_ids:
        first_use = geometry.member_edge_uses[first_use_id]
        first_edge = geometry.edges[first_use.edge_id]
        for second_use_id in second_member.edge_use_ids:
            second_use = geometry.member_edge_uses[second_use_id]
            second_edge = geometry.edges[second_use.edge_id]
            local = _query_edge_edge(
                geometry,
                geometry.handle("edge", first_edge.id),
                geometry.handle("edge", second_edge.id),
                qualification,
            )
            tolerance = max(tolerance, local.tolerance_used or 0.0)
            if local.kind in (
                IntersectionKind.UNSUPPORTED,
                IntersectionKind.CAPABILITY_MISSING,
                IntersectionKind.UNCLASSIFIED,
            ):
                unsupported.extend(local.diagnostics)
                continue
            for component in local.components:
                components.append(
                    (
                        local.kind,
                        _component_on_members(
                            geometry,
                            component,
                            first_use,
                            second_use,
                            geometry.handle("edge", first_edge.id),
                            geometry.handle("edge", second_edge.id),
                        ),
                    )
                )
    if not components:
        if unsupported:
            return _qualified_result(
                geometry,
                first,
                second,
                IntersectionKind.UNSUPPORTED,
                diagnostics=tuple(sorted(set(unsupported))),
                tolerance_used=tolerance,
            )
        return _qualified_result(
            geometry,
            first,
            second,
            IntersectionKind.DISJOINT,
            diagnostics=("member_axes_disjoint",),
            tolerance_used=tolerance,
        )
    components.sort(
        key=lambda item: (
            (item[1].first_parameter_range.lower,)
            if item[1].first_parameter_range is not None
            else item[1].first_parameter or (),
            (item[1].second_parameter_range.lower,)
            if item[1].second_parameter_range is not None
            else item[1].second_parameter or (),
        )
    )
    kinds = {item[0] for item in components}
    precedence = (
        IntersectionKind.COINCIDENT,
        IntersectionKind.CONTAINED,
        IntersectionKind.OVERLAP_CURVE,
        IntersectionKind.CROSS,
        IntersectionKind.TOUCH_POINT,
        IntersectionKind.TANGENT,
    )
    kind = next(item for item in precedence if item in kinds)
    selected = tuple(component for component_kind, component in components if component_kind is kind)
    return _qualified_result(
        geometry,
        first,
        second,
        kind,
        selected,
        diagnostics=("member_axis_components",),
        tolerance_used=tolerance,
    )


def _sheet_face_ids(geometry: GeometryModel, sheet_id: int) -> tuple[int, ...]:
    sheet = geometry.sheets[sheet_id]
    return tuple(
        sorted(geometry.face_uses[use_id].face_id for use_id in sheet.face_use_ids)
    )


def _query_member_material(
    geometry: GeometryModel,
    member: EntityHandle,
    target: EntityHandle,
    face_ids: Sequence[int],
    qualification: IntersectionQualificationPolicy = DEFAULT_INTERSECTION_QUALIFICATION_POLICY,
) -> IntersectionResult:
    member_record = geometry.members[member.id]
    components: list[IntersectionComponent] = []
    point_kinds: list[IntersectionKind] = []
    unsupported: list[str] = []
    tolerance = geometry.tolerance.length
    for use_id in member_record.edge_use_ids:
        use = geometry.member_edge_uses[use_id]
        edge = geometry.edges[use.edge_id]
        if not isinstance(edge.curve, Straight):
            edge_handle = geometry.handle("edge", edge.id)
            for face_id in face_ids:
                face_handle = geometry.handle("face", face_id)
                local = _query_edge_face(
                    geometry, edge_handle, face_handle, qualification
                )
                tolerance = max(tolerance, local.tolerance_used or 0.0)
                if local.kind in (
                    IntersectionKind.UNSUPPORTED,
                    IntersectionKind.CAPABILITY_MISSING,
                    IntersectionKind.UNCLASSIFIED,
                ):
                    unsupported.extend(local.diagnostics)
                    continue
                for component in local.components:
                    first_parameter = (
                        None
                        if component.first_parameter is None
                        else (_member_parameter(use, component.first_parameter[0]),)
                    )
                    first_range = None
                    if component.first_parameter_range is not None:
                        mapped_start = _member_parameter(
                            use, component.first_parameter_range.start
                        )
                        mapped_end = _member_parameter(
                            use, component.first_parameter_range.end
                        )
                        first_range = IntersectionParameterRange(
                            mapped_start, mapped_end
                        )
                    mapped_path = tuple(
                        (_member_parameter(use, value[0]),)
                        for value in component.first_parameter_path
                        if value
                    )
                    components.append(
                        replace(
                            component,
                            first_parameter=first_parameter,
                            first_parameter_range=first_range,
                            first_parameter_path=mapped_path,
                            first_subparent=edge_handle,
                            second_subparent=face_handle,
                        )
                    )
                    if first_range is None:
                        point_kinds.append(local.kind)
            continue
        edge_start = geometry.vertex_position(edge.start)
        edge_end = geometry.vertex_position(edge.end)
        vector = edge_end - edge_start
        length = float(np.linalg.norm(vector))
        if length <= 0.0:
            unsupported.append(f"member_edge_{edge.id}_degenerate")
            continue
        direction = vector / length
        edge_handle = geometry.handle("edge", edge.id)
        for face_id in face_ids:
            face_handle = geometry.handle("face", face_id)
            clipped = clip_line_to_face(
                geometry, face_id, edge_start, direction
            )
            tolerance = max(tolerance, clipped.tolerance_used or 0.0)
            if clipped.kind in (
                IntersectionKind.UNSUPPORTED,
                IntersectionKind.CAPABILITY_MISSING,
                IntersectionKind.UNCLASSIFIED,
            ):
                unsupported.extend(clipped.diagnostics)
                continue
            for component in clipped.components:
                if component.first_parameter_range is not None:
                    lower = max(0.0, component.first_parameter_range.lower)
                    upper = min(length, component.first_parameter_range.upper)
                    if upper < lower - tolerance:
                        continue
                    first_edge_parameter = float(np.clip(lower / length, 0.0, 1.0))
                    second_edge_parameter = float(np.clip(upper / length, 0.0, 1.0))
                    first_member_parameter = _member_parameter(
                        use, first_edge_parameter
                    )
                    second_member_parameter = _member_parameter(
                        use, second_edge_parameter
                    )
                    witnesses = tuple(
                        edge_start + parameter * vector
                        for parameter in (first_edge_parameter, second_edge_parameter)
                    )
                    face = geometry.faces[face_id]
                    if face.surface is None:
                        unsupported.append(f"face_{face_id}_missing_support")
                        continue
                    uv_path = tuple(
                        tuple(
                            float(item)
                            for item in geometry.face_local_uv(face_id, point)
                        )
                        for point in witnesses
                    )
                    components.append(
                        IntersectionComponent(
                            tuple(
                                tuple(float(item) for item in point)
                                for point in witnesses
                            ),
                            IntersectionQuality.VERIFIED_APPROXIMATE,
                            first_parameter_range=IntersectionParameterRange(
                                first_member_parameter, second_member_parameter
                            ),
                            first_parameter_path=(
                                (first_member_parameter,),
                                (second_member_parameter,),
                            ),
                            second_parameter_path=uv_path,
                            direction=tuple(float(item) for item in direction),
                            max_residual=component.max_residual,
                            first_subparent=edge_handle,
                            second_subparent=face_handle,
                        )
                    )
                    continue
                if component.first_parameter is None:
                    continue
                physical = component.first_parameter[0]
                if physical < -tolerance or physical > length + tolerance:
                    continue
                edge_parameter = float(np.clip(physical / length, 0.0, 1.0))
                member_parameter = _member_parameter(use, edge_parameter)
                witness = edge_start + edge_parameter * vector
                face = geometry.faces[face_id]
                if face.surface is None:
                    unsupported.append(f"face_{face_id}_missing_support")
                    continue
                uv = tuple(
                    float(item)
                    for item in geometry.face_local_uv(face_id, witness)
                )
                components.append(
                    IntersectionComponent(
                        (tuple(float(item) for item in witness),),
                        IntersectionQuality.VERIFIED_APPROXIMATE,
                        first_parameter=(member_parameter,),
                        second_parameter=uv,
                        first_parameter_path=((member_parameter,),),
                        second_parameter_path=(uv,),
                        max_residual=component.max_residual,
                        first_subparent=edge_handle,
                        second_subparent=face_handle,
                    )
                )
                point_kinds.append(clipped.kind)
    if not components:
        if unsupported:
            missing = "planar_backend_unavailable" in unsupported
            return _qualified_result(
                geometry,
                member,
                target,
                (
                    IntersectionKind.CAPABILITY_MISSING
                    if missing
                    else IntersectionKind.UNSUPPORTED
                ),
                diagnostics=tuple(sorted(set(unsupported))),
                tolerance_used=tolerance,
            )
        return _qualified_result(
            geometry,
            member,
            target,
            IntersectionKind.DISJOINT,
            diagnostics=("member_outside_target_material",),
            tolerance_used=tolerance,
        )
    components.sort(
        key=lambda item: (
            (item.first_parameter_range.lower,)
            if item.first_parameter_range is not None
            else item.first_parameter or (),
            item.second_subparent.sort_key if item.second_subparent else (),
        )
    )
    interval_components = tuple(
        item for item in components if item.first_parameter_range is not None
    )
    if interval_components:
        ranges = sorted(
            (
                item.first_parameter_range.lower,
                item.first_parameter_range.upper,
            )
            for item in interval_components
        )
        cursor = 0.0
        complete = True
        parameter_tolerance = geometry.tolerance.parameter
        for lower, upper in ranges:
            if lower > cursor + parameter_tolerance:
                complete = False
                break
            cursor = max(cursor, upper)
        complete = complete and cursor >= 1.0 - parameter_tolerance
        return _qualified_result(
            geometry,
            member,
            target,
            IntersectionKind.CONTAINED if complete else IntersectionKind.OVERLAP_CURVE,
            interval_components,
            diagnostics=("member_material_intervals",),
            dimension=IntersectionDimension.CURVE,
            tolerance_used=tolerance,
        )
    kind = (
        IntersectionKind.TOUCH_POINT
        if point_kinds and all(item is IntersectionKind.TOUCH_POINT for item in point_kinds)
        else IntersectionKind.CROSS
    )
    return _qualified_result(
        geometry,
        member,
        target,
        kind,
        components,
        diagnostics=("member_material_points",),
        tolerance_used=tolerance,
    )


def _query_edge_edge(
    geometry: GeometryModel,
    first: EntityHandle,
    second: EntityHandle,
    qualification: IntersectionQualificationPolicy = DEFAULT_INTERSECTION_QUALIFICATION_POLICY,
) -> IntersectionResult:
    first_edge = geometry.edges[first.id]
    second_edge = geometry.edges[second.id]
    if isinstance(first_edge.curve, Straight) and isinstance(second_edge.curve, Straight):
        result = qualified_segment_segment(
            geometry.vertex_position(first_edge.start),
            geometry.vertex_position(first_edge.end),
            geometry.vertex_position(second_edge.start),
            geometry.vertex_position(second_edge.end),
            policy=geometry.tolerance,
        )
    elif isinstance(first_edge.curve, Arc) and isinstance(second_edge.curve, Arc):
        result = _query_same_circle_arcs(geometry, first.id, second.id)
        if result.kind is IntersectionKind.UNSUPPORTED:
            result = qualified_curve_curve(geometry, first, second, qualification)
    else:
        result = qualified_curve_curve(geometry, first, second, qualification)
    scale = max(geometry.edge_length(first.id), geometry.edge_length(second.id))
    return result.with_context(
        first_parent=first,
        second_parent=second,
        tolerance_used=(
            result.tolerance_used
            if result.tolerance_used is not None
            else geometry.tolerance.effective_length(scale)
        ),
    )


def _query_edge_face(
    geometry: GeometryModel,
    edge: EntityHandle,
    face: EntityHandle,
    qualification: IntersectionQualificationPolicy = DEFAULT_INTERSECTION_QUALIFICATION_POLICY,
) -> IntersectionResult:
    edge_record = geometry.edges[edge.id]
    if not isinstance(edge_record.curve, Straight):
        return qualified_curve_face(geometry, edge, face, qualification)
    start = geometry.vertex_position(edge_record.start)
    end = geometry.vertex_position(edge_record.end)
    vector = end - start
    length = float(np.linalg.norm(vector))
    tolerance = geometry.tolerance.effective_length(length)
    if length <= tolerance:
        return _qualified_result(
            geometry,
            edge,
            face,
            IntersectionKind.UNCLASSIFIED,
            diagnostics=("degenerate_edge",),
            tolerance_used=tolerance,
        )
    clipped = clip_line_to_face(geometry, face.id, start, vector / length)
    if clipped.kind in (
        IntersectionKind.UNSUPPORTED,
        IntersectionKind.CAPABILITY_MISSING,
        IntersectionKind.UNCLASSIFIED,
    ):
        if clipped.kind is IntersectionKind.CAPABILITY_MISSING:
            return _qualified_result(
                geometry,
                edge,
                face,
                clipped.kind,
                diagnostics=clipped.diagnostics,
                tolerance_used=clipped.tolerance_used,
            )
        # Curved supports and trims are handled by the shared built-in engine;
        # a missing optional planar backend must not make core curved queries
        # unavailable.
        return qualified_curve_face(geometry, edge, face, qualification)
    components: list[IntersectionComponent] = []
    for component in clipped.components:
        if component.first_parameter_range is not None:
            lower = max(0.0, component.first_parameter_range.lower)
            upper = min(length, component.first_parameter_range.upper)
            if upper < lower - tolerance:
                continue
            parameters = (lower / length, upper / length)
            witnesses = tuple(start + parameter * vector for parameter in parameters)
            surface = geometry.faces[face.id].surface
            if surface is None:
                return _qualified_result(
                    geometry,
                    edge,
                    face,
                    IntersectionKind.UNSUPPORTED,
                    diagnostics=("face_missing_support",),
                    tolerance_used=tolerance,
                )
            components.append(
                IntersectionComponent(
                    tuple(tuple(float(item) for item in point) for point in witnesses),
                    IntersectionQuality.VERIFIED_APPROXIMATE,
                    first_parameter_range=IntersectionParameterRange(*parameters),
                    first_parameter_path=tuple((value,) for value in parameters),
                    second_parameter_path=tuple(
                        tuple(float(item) for item in surface.local_uv(point))
                        for point in witnesses
                    ),
                    direction=tuple(float(item) for item in vector / length),
                    max_residual=component.max_residual,
                    first_subparent=edge,
                    second_subparent=face,
                )
            )
            continue
        if component.first_parameter is None:
            continue
        physical = component.first_parameter[0]
        if physical < -tolerance or physical > length + tolerance:
            continue
        parameter = float(np.clip(physical / length, 0.0, 1.0))
        witness = start + parameter * vector
        surface = geometry.faces[face.id].surface
        if surface is None:
            return _qualified_result(
                geometry,
                edge,
                face,
                IntersectionKind.UNSUPPORTED,
                diagnostics=("face_missing_support",),
                tolerance_used=tolerance,
            )
        uv = tuple(float(item) for item in surface.local_uv(witness))
        components.append(
            IntersectionComponent(
                (tuple(float(item) for item in witness),),
                IntersectionQuality.VERIFIED_APPROXIMATE,
                first_parameter=(parameter,),
                second_parameter=uv,
                first_parameter_path=((parameter,),),
                second_parameter_path=(uv,),
                max_residual=component.max_residual,
                first_subparent=edge,
                second_subparent=face,
            )
        )
    if not components:
        return _qualified_result(
            geometry,
            edge,
            face,
            IntersectionKind.DISJOINT,
            diagnostics=("bounded_edge_outside_face_material",),
            tolerance_used=tolerance,
        )
    components.sort(
        key=lambda item: (
            (item.first_parameter_range.lower,)
            if item.first_parameter_range is not None
            else item.first_parameter or ()
        )
    )
    intervals = tuple(
        item for item in components if item.first_parameter_range is not None
    )
    if intervals:
        complete = any(
            item.first_parameter_range.lower <= geometry.tolerance.parameter
            and item.first_parameter_range.upper >= 1.0 - geometry.tolerance.parameter
            for item in intervals
        )
        return _qualified_result(
            geometry,
            edge,
            face,
            IntersectionKind.CONTAINED if complete else IntersectionKind.OVERLAP_CURVE,
            intervals,
            diagnostics=("edge_material_intervals",),
            dimension=IntersectionDimension.CURVE,
            tolerance_used=tolerance,
        )
    kind = (
        IntersectionKind.TOUCH_POINT
        if clipped.kind is IntersectionKind.TOUCH_POINT
        else IntersectionKind.CROSS
    )
    return _qualified_result(
        geometry,
        edge,
        face,
        kind,
        components,
        diagnostics=("bounded_edge_face_point",),
        tolerance_used=tolerance,
    )


def _swap_component_parents(component: IntersectionComponent) -> IntersectionComponent:
    return replace(
        component,
        first_parameter=component.second_parameter,
        second_parameter=component.first_parameter,
        first_parameter_range=component.second_parameter_range,
        second_parameter_range=component.first_parameter_range,
        first_parameter_path=component.second_parameter_path,
        second_parameter_path=component.first_parameter_path,
        first_region=component.second_region,
        second_region=component.first_region,
        first_subparent=component.second_subparent,
        second_subparent=component.first_subparent,
        curve_traces=tuple(
            CertifiedCurveTrace(
                trace.points,
                trace.second_parameter_path,
                trace.first_parameter_path,
                trace.certificate,
                closed=trace.closed,
            )
            for trace in component.curve_traces
        ),
    )


def query_intersection(
    geometry: GeometryModel,
    first: EntityHandle | EntityRef | tuple[str, int] | int,
    second: EntityHandle | EntityRef | tuple[str, int] | int,
    *,
    qualification: IntersectionQualificationPolicy | None = None,
) -> IntersectionResult:
    """Query a qualified intersection without mutating model state.

    New callers should pass model-bound handles.  Local refs/keys and bare
    face IDs remain accepted compatibility forms and are normalized before
    dispatch.  Unsupported operand pairs return a typed fail-closed result.
    """

    if not isinstance(geometry, GeometryModel):
        raise TypeError("query_intersection needs a GeometryModel")
    qualified_policy = (
        DEFAULT_INTERSECTION_QUALIFICATION_POLICY
        if qualification is None
        else qualification
    )
    if not isinstance(qualified_policy, IntersectionQualificationPolicy):
        raise GeometryError(
            "qualification must be IntersectionQualificationPolicy"
        )
    revision = geometry.revision
    first_handle = _normalize_operand(geometry, first)
    second_handle = _normalize_operand(geometry, second)
    pair = (first_handle.kind, second_handle.kind)
    if pair == ("face", "face"):
        result = _query_face_face(
            geometry, first_handle, second_handle, qualified_policy
        )
    elif pair == ("edge", "edge"):
        result = _query_edge_edge(
            geometry, first_handle, second_handle, qualified_policy
        )
    elif pair == ("edge", "face"):
        result = _query_edge_face(
            geometry, first_handle, second_handle, qualified_policy
        )
    elif pair == ("face", "edge"):
        forward = _query_edge_face(
            geometry, second_handle, first_handle, qualified_policy
        )
        components = tuple(_swap_component_parents(item) for item in forward.components)
        result = _qualified_result(
            geometry,
            first_handle,
            second_handle,
            forward.kind,
            components,
            diagnostics=forward.diagnostics,
            dimension=forward.dimension,
            tolerance_used=forward.tolerance_used,
        )
    elif pair == ("member", "member"):
        result = _query_member_member(
            geometry, first_handle, second_handle, qualified_policy
        )
    elif pair in (("member", "face"), ("member", "sheet")):
        face_ids = (
            (second_handle.id,)
            if second_handle.kind == "face"
            else _sheet_face_ids(geometry, second_handle.id)
        )
        result = _query_member_material(
            geometry,
            first_handle,
            second_handle,
            face_ids,
            qualified_policy,
        )
    elif pair in (("face", "member"), ("sheet", "member")):
        face_ids = (
            (first_handle.id,)
            if first_handle.kind == "face"
            else _sheet_face_ids(geometry, first_handle.id)
        )
        forward = _query_member_material(
            geometry,
            second_handle,
            first_handle,
            face_ids,
            qualified_policy,
        )
        components = tuple(_swap_component_parents(item) for item in forward.components)
        result = _qualified_result(
            geometry,
            first_handle,
            second_handle,
            forward.kind,
            components,
            diagnostics=forward.diagnostics,
            dimension=forward.dimension,
            tolerance_used=forward.tolerance_used,
        )
    else:
        result = _qualified_result(
            geometry,
            first_handle,
            second_handle,
            IntersectionKind.UNSUPPORTED,
            diagnostics=(
                f"intersection operand pair {first_handle.kind}/{second_handle.kind} "
                "is unsupported",
            ),
        )
    if geometry.revision != revision:
        raise RuntimeError("intersection query unexpectedly mutated the geometry model")
    return result


def _normalize_intersection_policy(policy: object) -> object:
    if policy is None:
        raise GeometryError("imprint planning requires an explicit policy")
    from .structural import ConnectionIntent

    if isinstance(policy, (MutationPolicy, ConnectionIntent)):
        return policy
    for enum_type in (MutationPolicy, ConnectionIntent):
        try:
            return enum_type(policy)  # type: ignore[call-arg]
        except (TypeError, ValueError):
            continue
    raise GeometryError(f"invalid intersection policy {policy!r}")


def _policy_name(policy: object) -> str:
    value = getattr(policy, "value", policy)
    return str(value)


def _intersection_pair_family(
    first_parent: EntityHandle, second_parent: EntityHandle
) -> str:
    pair = (first_parent.kind, second_parent.kind)
    if pair == ("face", "face"):
        return "face_face"
    if pair == ("member", "member"):
        return "member_member"
    if set(pair) == {"member", "face"} or set(pair) == {"member", "sheet"}:
        return "member_material"
    raise GeometryError(
        f"intersection planning does not support operand pair "
        f"{first_parent.kind}/{second_parent.kind}"
    )


def _preflight_intersection_policy(
    first_parent: EntityHandle,
    second_parent: EntityHandle,
    policy: object,
) -> str:
    """Validate a policy against the persistent operation for one pair."""

    family = _intersection_pair_family(first_parent, second_parent)
    policy_value = _policy_name(policy)
    allowed = {
        "face_face": frozenset(("reject", "reuse_existing", "imprint", "connect")),
        "member_member": frozenset(
            (
                "reject",
                "reuse_existing",
                "connect",
                "keep_disconnected",
                "contact_only",
                "imprint",
            )
        ),
        "member_material": frozenset(
            (
                "reject",
                "reuse_existing",
                "connect",
                "keep_disconnected",
                "contact_only",
                "imprint",
            )
        ),
    }[family]
    if policy_value not in allowed:
        raise GeometryError(
            f"policy {policy_value!r} is not valid for intersection pair "
            f"{first_parent.kind}/{second_parent.kind}"
        )
    return family


def _unsupported_imprint_result(
    result: IntersectionResult, diagnostic: str
) -> IntersectionResult:
    """Preserve a valid query while making a mutation limitation explicit."""

    return IntersectionResult(
        IntersectionKind.UNSUPPORTED,
        diagnostics=(*result.diagnostics, str(diagnostic)),
        first_parent=result.first_parent,
        second_parent=result.second_parent,
        tolerance_used=result.tolerance_used,
    )


def _face_imprint_limitation(
    geometry: GeometryModel, result: IntersectionResult
) -> str | None:
    """Return a typed pre-plan limitation for unsupported semantic remaps."""

    assert result.first_parent is not None and result.second_parent is not None
    face_ids = (result.first_parent.id, result.second_parent.id)
    for face_id in face_ids:
        attachments = geometry.attachments_for_face(face_id)
        if attachments and result.dimension is IntersectionDimension.REGION:
            return (
                f"face {face_id} has attachments {list(attachments)} whose "
                "region-fragment parameters require an explicit remap"
            )
        face = geometry.faces[face_id]
        for edge_id in sorted(
            {item.edge for loop in (face.loop,) + face.holes for item in loop}
        ):
            member_ids = geometry.members_using_edge(edge_id)
            edge_attachments = tuple(
                sorted(geometry._target_attachments.get(("edge", edge_id), ()))  # noqa: SLF001
            )
            if edge_attachments or (
                result.dimension is IntersectionDimension.REGION and member_ids
            ):
                return (
                    f"face {face_id} boundary edge {edge_id} has member owners "
                    f"{list(member_ids)} or attachments {list(edge_attachments)} "
                    "whose parameters require an explicit remap"
                )
    return None


def plan_imprint(
    geometry: GeometryModel,
    result_or_first: IntersectionResult | EntityHandle | EntityRef | tuple[str, int] | int,
    second: EntityHandle | EntityRef | tuple[str, int] | int | None = None,
    *,
    policy: object,
    qualification: IntersectionQualificationPolicy | None = None,
) -> ImprintPlan:
    """Create an immutable deterministic plan without changing the model."""

    revision = geometry.revision
    qualified_policy = (
        DEFAULT_INTERSECTION_QUALIFICATION_POLICY
        if qualification is None
        else qualification
    )
    if not isinstance(qualified_policy, IntersectionQualificationPolicy):
        raise GeometryError(
            "qualification must be IntersectionQualificationPolicy"
        )
    normalized_policy = _normalize_intersection_policy(policy)
    if isinstance(result_or_first, IntersectionResult):
        if second is not None:
            raise GeometryError("second operand cannot accompany an IntersectionResult")
        result = result_or_first
        if result.first_parent is None or result.second_parent is None:
            raise GeometryError("an imprint query result needs model-bound parents")
        first_parent, second_parent = result.first_parent, result.second_parent
        for parent in (first_parent, second_parent):
            resolution = geometry.resolve_handle(parent)
            if resolution.status is not ResolutionStatus.ACTIVE:
                raise GeometryError("imprint plan parents must still be active")
    else:
        if second is None:
            raise GeometryError("plan_imprint needs two operands or a qualified result")
        result = query_intersection(
            geometry, result_or_first, second, qualification=qualified_policy
        )
        assert result.first_parent is not None and result.second_parent is not None
        first_parent, second_parent = result.first_parent, result.second_parent

    policy_value = _policy_name(normalized_policy)
    family = _preflight_intersection_policy(
        first_parent, second_parent, normalized_policy
    )
    same_parent = first_parent == second_parent
    no_geometric_intersection = result.kind is IntersectionKind.DISJOINT
    pair = {first_parent.kind, second_parent.kind}
    if same_parent or no_geometric_intersection:
        operation = ImprintOperation.NO_TOPOLOGY
        changes = ()
    elif first_parent.kind == second_parent.kind == "face":
        # A qualified face/face curve with CONNECT intent is the persistent
        # shell/sheet T-junction operation.  It uses the same B-rep imprint as
        # explicit IMPRINT, including FaceUse/Coedge replacement, so callers
        # never need to infer shell coupling from coincident coordinates.
        connect_geometry = (
            policy_value == "connect"
            and result.classified
            and result.dimension in (
                IntersectionDimension.CURVE,
                IntersectionDimension.REGION,
            )
        )
        if (
            policy_value == "connect"
            and result.classified
            and result.dimension not in (
                IntersectionDimension.CURVE,
                IntersectionDimension.REGION,
            )
        ):
            result = _unsupported_imprint_result(
                result,
                "face-face CONNECT requires a qualified curve or coincident "
                "region intersection",
            )
        if (
            policy_value == "imprint"
            and result.classified
            and result.dimension not in (
                IntersectionDimension.CURVE,
                IntersectionDimension.REGION,
            )
        ):
            result = _unsupported_imprint_result(
                result,
                "face-face IMPRINT requires a qualified curve or region; "
                "point-only contact creates no persistent face topology",
            )
        reuse_components = (
            _shared_edges_for_components(
                geometry,
                first_parent.id,
                second_parent.id,
                result.components,
                result.tolerance_used or geometry.tolerance.length,
            )
            if policy_value == "reuse_existing"
            and result.classified
            and result.dimension is IntersectionDimension.CURVE
            else ()
        )
        wants_face_imprint = (
            policy_value == "imprint"
            or connect_geometry
            or len(reuse_components) == len(result.components) > 0
        )
        if wants_face_imprint and result.classified:
            limitation = _face_imprint_limitation(geometry, result)
            if limitation is not None:
                result = _unsupported_imprint_result(result, limitation)
        operation = (
            ImprintOperation.FACE_IMPRINT
            if wants_face_imprint and result.classified
            else ImprintOperation.NO_TOPOLOGY
        )
        changes = (
            (
                ExpectedImprintChange(
                    "create_or_reuse", "edge", (first_parent, second_parent)
                ),
                ExpectedImprintChange("fragment", "face", (first_parent,)),
                ExpectedImprintChange("fragment", "face", (second_parent,)),
            )
            if operation is ImprintOperation.FACE_IMPRINT
            else ()
        )
    elif pair == {"member"}:
        existing_relation = None
        if policy_value != "reject" and result.classified:
            existing_relation = (
                _reuse_existing_member_junction(
                    geometry, first_parent.id, second_parent.id, result
                )
                if policy_value == "reuse_existing"
                else _existing_member_junction(
                    geometry,
                    first_parent.id,
                    second_parent.id,
                    normalized_policy,
                    _junction_kind_for_result(result),
                )
            )
        if policy_value == "reuse_existing" and existing_relation is None:
            operation = ImprintOperation.NO_TOPOLOGY
            changes = ()
        else:
            if (
                existing_relation is None
                and policy_value != "reject"
                and result.classified
                and len(result.components) != 1
            ):
                result = _unsupported_imprint_result(
                    result,
                    "member connection planning currently requires exactly one "
                    "qualified intersection component",
                )
            elif (
                existing_relation is None
                and policy_value in ("connect", "imprint")
                and result.classified
                and result.dimension is not IntersectionDimension.POINT
            ):
                result = _unsupported_imprint_result(
                    result,
                    "connected member overlap requires an explicit qualified "
                    "overlap-topology operation",
                )
            operation = (
                ImprintOperation.NO_TOPOLOGY
                if policy_value == "reject" or not result.classified
                else ImprintOperation.MEMBER_CONNECTION
            )
            changes = (
                ExpectedImprintChange(
                    "create_or_reuse", "junction", (first_parent, second_parent)
                ),
            )
            if policy_value in ("connect", "imprint"):
                changes = (
                    ExpectedImprintChange("split_or_reuse", "edge", (first_parent, second_parent)),
                    *changes,
                )
    elif "member" in pair and pair & {"face", "sheet"}:
        classification_limitation = (
            _member_sheet_preflight_supported(
                geometry, first_parent, second_parent, result
            )
            if result.classified and result.components
            else None
        )
        if classification_limitation is not None:
            result = _unsupported_imprint_result(
                result, classification_limitation
            )
            operation = ImprintOperation.NO_TOPOLOGY
            changes = ()
        elif not result.classified:
            operation = ImprintOperation.NO_TOPOLOGY
            changes = ()
        elif len(result.components) == 0:
            operation = ImprintOperation.NO_TOPOLOGY
            changes = ()
        elif policy_value == "reuse_existing":
            provisional = ImprintPlan(
                geometry.model_id,
                revision,
                first_parent,
                second_parent,
                result,
                normalized_policy,
                ImprintOperation.MEMBER_SHEET_RELATION,
            )
            existing_relations = _existing_member_sheet_relations(
                geometry, provisional, result
            )
            operation = (
                ImprintOperation.MEMBER_SHEET_RELATION
                if existing_relations is not None
                else ImprintOperation.NO_TOPOLOGY
            )
            changes = ()
        else:
            operation = (
                ImprintOperation.NO_TOPOLOGY
                if policy_value == "reject"
                else ImprintOperation.MEMBER_SHEET_RELATION
            )
            changes = (
                ExpectedImprintChange(
                    "create_or_reuse", "attachment", (first_parent, second_parent)
                ),
            )
    else:  # pragma: no cover - pair family is closed by preflight
        raise GeometryError(f"unsupported intersection planning family {family}")
    affected = {
        first_parent,
        second_parent,
        *(
            parent
            for component in result.components
            for parent in (component.first_subparent, component.second_subparent)
            if parent is not None
        ),
    }
    plan = ImprintPlan(
        geometry.model_id,
        revision,
        first_parent,
        second_parent,
        result,
        normalized_policy,
        operation,
        tuple(changes),
        tuple(sorted(affected)),
        qualified_policy,
    )
    if geometry.revision != revision:
        raise RuntimeError("imprint planning unexpectedly mutated the geometry model")
    return plan


def _resolve_plan_parent(
    geometry: GeometryModel, handle: EntityHandle
) -> EntityHandle:
    resolution = geometry.resolve_handle(handle)
    if resolution.status is not ResolutionStatus.ACTIVE:
        raise GeometryError(
            f"imprint plan is stale: {handle.kind}{handle.id} is "
            f"{resolution.status.value}"
        )
    return handle


def _active_handle_for_subparent(
    geometry: GeometryModel, handle: EntityHandle | None
) -> EntityHandle | None:
    if handle is None:
        return None
    resolution = geometry.resolve_handle(handle)
    if resolution.status is ResolutionStatus.ACTIVE:
        return handle
    if resolution.status is ResolutionStatus.REPLACED:
        return next(
            (item for item in resolution.resolved if item.kind == handle.kind),
            None,
        )
    return None


def _junction_kind_for_result(result: IntersectionResult):
    from .structural import JunctionKind

    if result.kind in (
        IntersectionKind.COINCIDENT,
        IntersectionKind.CONTAINED,
        IntersectionKind.OVERLAP_CURVE,
    ):
        return JunctionKind.OVERLAP
    if (
        result.dimension is IntersectionDimension.POINT
        and len(result.components) == 1
        and _member_component_has_endpoint(result.components[0])
    ):
        return JunctionKind.ENDPOINT
    return JunctionKind.CROSSING


def _parameter_is_endpoint(value: float) -> bool:
    return value <= 0.0 or value >= 1.0


def _member_component_has_endpoint(component: IntersectionComponent) -> bool:
    """Whether either participating Member meets at one of its endpoints."""

    for value in (component.first_parameter, component.second_parameter):
        if value is not None and len(value) == 1 and _parameter_is_endpoint(
            float(value[0])
        ):
            return True
    return False


def _reuse_existing_member_junction(
    geometry: GeometryModel,
    first_member: int,
    second_member: int,
    result: IntersectionResult,
):
    """Return any compatible pre-existing member relation, regardless intent."""

    from .structural import JunctionKind

    required_kind = _junction_kind_for_result(result)
    candidate_ids = set(
        geometry._member_junctions.get(first_member, ())  # noqa: SLF001
    )
    candidate_ids.intersection_update(
        geometry._member_junctions.get(second_member, ())  # noqa: SLF001
    )
    compatible = {
        JunctionKind.ENDPOINT: frozenset((JunctionKind.ENDPOINT,)),
        JunctionKind.CROSSING: frozenset((JunctionKind.CROSSING,)),
        JunctionKind.OVERLAP: frozenset((JunctionKind.OVERLAP,)),
    }[required_kind]
    return next(
        (
            geometry.junctions[junction_id]
            for junction_id in sorted(candidate_ids)
            if set(geometry.junctions[junction_id].member_ids)
            == {first_member, second_member}
            and geometry.junctions[junction_id].kind in compatible
        ),
        None,
    )


def _range_from_component(
    component: IntersectionComponent,
    *,
    first: bool,
):
    from .structural import ParameterRange

    interval = (
        component.first_parameter_range
        if first
        else component.second_parameter_range
    )
    parameter = (
        component.first_parameter if first else component.second_parameter
    )
    if interval is not None:
        lower = float(np.clip(interval.lower, 0.0, 1.0))
        upper = float(np.clip(interval.upper, 0.0, 1.0))
        return ParameterRange(lower, upper)
    if parameter is None:
        raise GeometryError("intersection component is missing parent parameters")
    return ParameterRange.point(float(np.clip(parameter[0], 0.0, 1.0)))


def _existing_member_junction(
    geometry: GeometryModel,
    first_member: int,
    second_member: int,
    intent: object,
    kind: object,
):
    intent_value = _policy_name(intent)
    candidate_ids = set(geometry._member_junctions.get(first_member, ()))  # noqa: SLF001
    candidate_ids.intersection_update(
        geometry._member_junctions.get(second_member, ())  # noqa: SLF001
    )
    for junction_id in sorted(candidate_ids):
        junction = geometry.junctions[junction_id]
        if set(junction.member_ids) != {first_member, second_member}:
            continue
        if junction.kind != kind:
            continue
        existing_intent = getattr(junction, "connection_intent", None)
        if existing_intent is None:
            existing_intent = junction.metadata.to_dict().get("connection_intent")
        if _policy_name(existing_intent) == intent_value:
            return junction
    return None


def _member_connection_application(
    geometry: GeometryModel,
    plan: ImprintPlan,
    revalidated: IntersectionResult,
) -> tuple[tuple[EntityHandle, ...], bool]:
    from .structural import (
        AttachmentEvidence,
        AttachmentKind,
        AttachmentTargetKind,
        ConnectionIntent,
        JunctionKind,
        JunctionMemberUse,
    )

    def relation_handles(junction) -> tuple[EntityHandle, ...]:
        return tuple(
            (
                *(geometry.handle("attachment", item) for item in junction.attachment_ids),
                geometry.handle("junction", junction.id),
            )
        )

    first, second = plan.first_parent, plan.second_parent
    if first.kind != "member" or second.kind != "member":
        raise GeometryError("member connection plan requires two Members")
    if not revalidated.components:
        raise GeometryError("member connection needs a qualified component")
    intent = ConnectionIntent(_policy_name(plan.policy))
    if intent is ConnectionIntent.REJECT:
        raise GeometryError("intersection mutation rejected by policy")
    if intent is ConnectionIntent.REUSE_EXISTING:
        existing = _reuse_existing_member_junction(
            geometry, first.id, second.id, revalidated
        )
        if existing is None:
            raise GeometryError(
                "REUSE_EXISTING requires a compatible existing member junction"
            )
        return relation_handles(existing), True
    junction_kind = _junction_kind_for_result(revalidated)
    existing = _existing_member_junction(
        geometry, first.id, second.id, intent, junction_kind
    )
    if existing is not None:
        return relation_handles(existing), True
    component = revalidated.components[0]
    if intent in (ConnectionIntent.CONNECT, ConnectionIntent.IMPRINT):
        if revalidated.dimension is not IntersectionDimension.POINT:
            raise GeometryError(
                "connected member overlap is unsupported; use an explicit "
                "disconnected/contact policy or qualified overlap operation"
            )
        edge_handles = (
            _active_handle_for_subparent(geometry, component.first_subparent),
            _active_handle_for_subparent(geometry, component.second_subparent),
        )
        parameters = (component.first_parameter, component.second_parameter)
        if any(item is None for item in edge_handles) or any(
            item is None for item in parameters
        ):
            raise GeometryError("connected member crossing lacks edge parameters")
        shared_vertex: int | None = None
        for edge_handle, parameter in zip(edge_handles, parameters):
            assert edge_handle is not None and parameter is not None
            edge = geometry.edges[edge_handle.id]
            edge_parameter = float(parameter[0])
            # Component parent parameters are member parameters.  Convert by
            # locating the participating use preserved in the subparent.
            use = next(
                item
                for item in geometry.member_edge_uses.values()
                if item.edge_id == edge_handle.id
                and item.member_id in (first.id, second.id)
                and item.parent_range.contains(edge_parameter, tolerance=geometry.tolerance.parameter)
            )
            local = (
                (edge_parameter - use.parent_range.start)
                / use.parent_range.length
            )
            from .structural import Orientation

            if use.orientation is Orientation.REVERSED:
                local = 1.0 - local
            if local <= geometry.tolerance.parameter:
                vertex_id = edge.start
            elif local >= 1.0 - geometry.tolerance.parameter:
                vertex_id = edge.end
            else:
                vertex_id, _children = geometry.split_edge(edge.id, local)
            if shared_vertex is None:
                shared_vertex = vertex_id
            elif vertex_id != shared_vertex:
                _merge_vertex(geometry, vertex_id, shared_vertex)
        assert shared_vertex is not None
    uses = (
        JunctionMemberUse(first.id, _range_from_component(component, first=True)),
        JunctionMemberUse(second.id, _range_from_component(component, first=False)),
    )
    attachment_ids: tuple[int, ...] = ()
    if junction_kind is JunctionKind.ENDPOINT:
        first_range, second_range = uses[0].member_range, uses[1].member_range
        if first_range.is_point and _parameter_is_endpoint(first_range.start):
            source, target = first, second
            source_range, target_range = first_range, second_range
        elif second_range.is_point and _parameter_is_endpoint(second_range.start):
            source, target = second, first
            source_range, target_range = second_range, first_range
        else:  # pragma: no cover - guarded by endpoint junction classification
            raise GeometryError("endpoint junction lacks an endpoint parameter")
        attachment_id = geometry.ensure_attachment(
            source.id,
            AttachmentKind.MEMBER_ENDPOINT_ON_MEMBER,
            AttachmentTargetKind.MEMBER,
            target.id,
            source_range,
            (target_range,),
            connection_intent=intent,
            evidence=AttachmentEvidence(component.quality.value),
            max_residual=component.max_residual,
            tolerance_used=revalidated.tolerance_used or 0.0,
            part_id=geometry.members[source.id].part_id,
            provenance={
                "classification": revalidated.kind.value,
                "source_member": source.id,
                "target_member": target.id,
            },
            lineage=(("member", source.id), ("member", target.id)),
            metadata={
                "connection_intent": intent.value,
                "classification": revalidated.kind.value,
            },
        )
        attachment_ids = (attachment_id,)
    metadata = {
        "connection_intent": intent.value,
        "classification": revalidated.kind.value,
        "tolerance_used": revalidated.tolerance_used or 0.0,
        "max_residual": revalidated.max_residual,
    }
    junction_id = geometry.ensure_junction(
        junction_kind,
        uses,
        attachment_ids=attachment_ids,
        connection_intent=intent,
        metadata=metadata,
        provenance={
            "classification": revalidated.kind.value,
            "first_member": first.id,
            "second_member": second.id,
        },
    )
    return relation_handles(geometry.junctions[junction_id]), False


def _sheet_for_face(geometry: GeometryModel, face_id: int) -> int | None:
    matches = sorted(
        geometry.face_uses[use_id].sheet_id
        for use_id in geometry._face_structural_uses.get(face_id, ())  # noqa: SLF001
    )
    return matches[0] if matches else None


def _member_and_target(plan: ImprintPlan) -> tuple[EntityHandle, EntityHandle, bool]:
    if plan.first_parent.kind == "member":
        return plan.first_parent, plan.second_parent, True
    if plan.second_parent.kind == "member":
        return plan.second_parent, plan.first_parent, False
    raise GeometryError("member-sheet plan has no Member")


def _component_face_handle(
    geometry: GeometryModel,
    component: IntersectionComponent,
    *,
    member_is_first: bool,
    face_ids: Sequence[int],
) -> EntityHandle:
    face_handle = (
        component.second_subparent
        if member_is_first
        else component.first_subparent
    )
    if face_handle is not None and face_handle.kind == "face":
        return face_handle
    if len(face_ids) != 1:
        raise GeometryError("member-sheet component lacks face sequence")
    return geometry.handle("face", face_ids[0])


def _member_sheet_component_kind(
    geometry: GeometryModel,
    member_id: int,
    face_id: int,
    member_range,
    component: IntersectionComponent,
    *,
    member_is_first: bool = True,
    target_is_sheet: bool = False,
):
    """Return a precise persistent member/face relation or fail closed."""

    from .structural import AttachmentKind, JunctionKind

    member_subparent = (
        component.first_subparent
        if member_is_first
        else component.second_subparent
    )

    if member_range.is_point:
        at_endpoint = _parameter_is_endpoint(member_range.start)
        if at_endpoint:
            return (
                AttachmentKind.MEMBER_ENDPOINT_ON_SHEET
                if target_is_sheet
                else AttachmentKind.MEMBER_THROUGH_FACE,
                JunctionKind.ENDPOINT,
            )
        if component.witnesses:
            witness = np.asarray(component.witnesses[0], dtype=float)
            face = geometry.faces[face_id]
            boundary_distance = min(
                (
                    geometry.closest_edge_point(item.edge, witness)[2]
                    for loop in (face.loop,) + face.holes
                    for item in loop
                ),
                default=float("inf"),
            )
            member_record = geometry.members[member_id]
            member_extent = sum(
                geometry.edge_length(
                    geometry.member_edge_uses[use_id].edge_id
                )
                for use_id in member_record.edge_use_ids
            )
            tolerance = geometry.tolerance.effective_length(member_extent)
            if boundary_distance <= tolerance:
                return (
                    AttachmentKind.MEMBER_ON_FACE_BOUNDARY,
                    JunctionKind.CROSSING,
                )
        return (
            AttachmentKind.MEMBER_CROSS_SHEET
            if target_is_sheet
            else AttachmentKind.MEMBER_THROUGH_FACE,
            JunctionKind.CROSSING,
        )

    face_edges = {
        item.edge
        for loop in (geometry.faces[face_id].loop,) + geometry.faces[face_id].holes
        for item in loop
    }
    if member_subparent is not None and member_subparent.kind == "edge":
        edge_id = member_subparent.id
        if edge_id in face_edges:
            return AttachmentKind.MEMBER_ON_FACE_BOUNDARY, JunctionKind.OVERLAP
        edge_sheets = set(geometry.sheets_using_edge(edge_id))
        target_sheets = {
            geometry.face_uses[face_use_id].sheet_id
            for face_use_id in geometry._face_structural_uses.get(face_id, ())  # noqa: SLF001
        }
        if edge_sheets and (
            not target_sheets or not edge_sheets.isdisjoint(target_sheets)
        ):
            return AttachmentKind.MEMBER_ON_SHEET_INTERSECTION, JunctionKind.OVERLAP
    return (
        AttachmentKind.MEMBER_ON_SHEET
        if target_is_sheet
        else AttachmentKind.MEMBER_ON_FACE,
        JunctionKind.OVERLAP,
    )


def _existing_member_sheet_relations(
    geometry: GeometryModel,
    plan: ImprintPlan,
    result: IntersectionResult,
) -> tuple[EntityHandle, ...] | None:
    """Resolve a complete compatible relation set without creating records."""

    from .structural import AttachmentKind, AttachmentTargetKind

    member, target, member_is_first = _member_and_target(plan)
    face_ids = (
        (target.id,)
        if target.kind == "face"
        else _sheet_face_ids(geometry, target.id)
    )
    relations: list[EntityHandle] = []
    for component in result.components:
        member_range = _range_from_component(component, first=member_is_first)
        face_handle = _component_face_handle(
            geometry,
            component,
            member_is_first=member_is_first,
            face_ids=face_ids,
        )
        attachment_kind, _junction_kind = _member_sheet_component_kind(
            geometry,
            member.id,
            face_handle.id,
            member_range,
            component,
            member_is_first=member_is_first,
            target_is_sheet=target.kind == "sheet",
        )
        attachment_target_kind = (
            AttachmentTargetKind.SHEET
            if attachment_kind
            in {
                AttachmentKind.MEMBER_ENDPOINT_ON_SHEET,
                AttachmentKind.MEMBER_CROSS_SHEET,
                AttachmentKind.MEMBER_ON_SHEET,
            }
            else AttachmentTargetKind.EDGE
            if attachment_kind
            in {
                AttachmentKind.MEMBER_ON_FACE_BOUNDARY,
                AttachmentKind.MEMBER_ON_SHEET_INTERSECTION,
            }
            else AttachmentTargetKind.FACE
        )
        member_subparent = (
            component.first_subparent
            if member_is_first
            else component.second_subparent
        )
        attachment_target_id = (
            target.id
            if attachment_target_kind is AttachmentTargetKind.SHEET
            else member_subparent.id
            if attachment_target_kind is AttachmentTargetKind.EDGE
            and member_subparent is not None
            else face_handle.id
        )
        attachment = next(
            (
                item
                for item in geometry.attachments.values()
                if item.member_id == member.id
                and item.target_kind is attachment_target_kind
                and item.target_id == attachment_target_id
                and item.member_range == member_range
                and item.kind is attachment_kind
            ),
            None,
        )
        if attachment is None:
            return None
        relations.append(geometry.handle("attachment", attachment.id))
        sheet_id = (
            target.id
            if target.kind == "sheet"
            else _sheet_for_face(geometry, face_handle.id)
        )
        if sheet_id is not None:
            junction = next(
                (
                    item
                    for item in geometry.junctions.values()
                    if item.member_ids == (member.id,)
                    and item.sheet_ids == (sheet_id,)
                    and attachment.id in item.attachment_ids
                ),
                None,
            )
            if junction is not None:
                relations.append(geometry.handle("junction", junction.id))
    return tuple(sorted(set(relations)))


def _member_sheet_preflight_supported(
    geometry: GeometryModel,
    first_parent: EntityHandle,
    second_parent: EntityHandle,
    result: IntersectionResult,
) -> str | None:
    """Return why a member/material result cannot be persistently classified."""

    from .structural import ConnectionIntent

    synthetic = ImprintPlan(
        geometry.model_id,
        geometry.revision,
        first_parent,
        second_parent,
        result,
        ConnectionIntent.CONTACT_ONLY,
        ImprintOperation.MEMBER_SHEET_RELATION,
    )
    member, target, member_is_first = _member_and_target(synthetic)
    face_ids = (
        (target.id,)
        if target.kind == "face"
        else _sheet_face_ids(geometry, target.id)
    )
    try:
        for component in result.components:
            member_range = _range_from_component(
                component, first=member_is_first
            )
            face_handle = _component_face_handle(
                geometry,
                component,
                member_is_first=member_is_first,
                face_ids=face_ids,
            )
            _member_sheet_component_kind(
                geometry,
                member.id,
                face_handle.id,
                member_range,
                component,
                member_is_first=member_is_first,
                target_is_sheet=target.kind == "sheet",
            )
    except GeometryError as error:
        return str(error)
    return None


def _member_sheet_application(
    geometry: GeometryModel,
    plan: ImprintPlan,
    revalidated: IntersectionResult,
) -> tuple[tuple[EntityHandle, ...], bool]:
    from .structural import (
        AttachmentKind,
        AttachmentTargetKind,
        ConnectionIntent,
        JunctionKind,
        JunctionMemberUse,
        ParameterRange,
    )

    member, target, member_is_first = _member_and_target(plan)
    intent = ConnectionIntent(_policy_name(plan.policy))
    if intent is ConnectionIntent.REJECT:
        raise GeometryError("intersection mutation rejected by policy")
    if not revalidated.components:
        raise GeometryError("member-sheet relation needs a qualified component")
    if intent is ConnectionIntent.REUSE_EXISTING:
        existing = _existing_member_sheet_relations(
            geometry, plan, revalidated
        )
        if existing is None:
            raise GeometryError(
                "REUSE_EXISTING requires compatible existing member-sheet relations"
            )
        return existing, True
    relations: list[EntityHandle] = []
    reused = True
    face_ids = (
        (target.id,) if target.kind == "face" else _sheet_face_ids(geometry, target.id)
    )
    for component in revalidated.components:
        member_range = _range_from_component(
            component, first=member_is_first
        )
        face_handle = _component_face_handle(
            geometry,
            component,
            member_is_first=member_is_first,
            face_ids=face_ids,
        )
        parameter_path = (
            component.second_parameter_path
            if member_is_first
            else component.first_parameter_path
        )
        if not parameter_path:
            point = (
                component.second_parameter
                if member_is_first
                else component.first_parameter
            )
            if point is None:
                raise GeometryError("member-sheet component lacks face parameters")
            parameter_path = (point,)
        u_values = tuple(float(item[0]) for item in parameter_path)
        v_values = tuple(float(item[1]) for item in parameter_path)
        target_parameters = (
            ParameterRange(min(u_values), max(u_values)),
            ParameterRange(min(v_values), max(v_values)),
        )
        attachment_kind, relation_junction_kind = _member_sheet_component_kind(
            geometry,
            member.id,
            face_handle.id,
            member_range,
            component,
            member_is_first=member_is_first,
            target_is_sheet=target.kind == "sheet",
        )
        attachment_target_kind = (
            AttachmentTargetKind.SHEET
            if attachment_kind
            in {
                AttachmentKind.MEMBER_ENDPOINT_ON_SHEET,
                AttachmentKind.MEMBER_CROSS_SHEET,
                AttachmentKind.MEMBER_ON_SHEET,
            }
            else AttachmentTargetKind.EDGE
            if attachment_kind
            in {
                AttachmentKind.MEMBER_ON_FACE_BOUNDARY,
                AttachmentKind.MEMBER_ON_SHEET_INTERSECTION,
            }
            else AttachmentTargetKind.FACE
        )
        member_subparent = (
            component.first_subparent
            if member_is_first
            else component.second_subparent
        )
        attachment_target_id = (
            target.id
            if attachment_target_kind is AttachmentTargetKind.SHEET
            else member_subparent.id
            if attachment_target_kind is AttachmentTargetKind.EDGE
            and member_subparent is not None
            else face_handle.id
        )
        existing = next(
            (
                item
                for item in geometry.attachments.values()
                if item.member_id == member.id
                and item.target_kind is attachment_target_kind
                and item.target_id == attachment_target_id
                and item.member_range == member_range
                and _policy_name(
                    getattr(item, "connection_intent", item.metadata.to_dict().get("connection_intent"))
                )
                == intent.value
            ),
            None,
        )
        if existing is not None:
            attachment_id = existing.id
        else:
            reused = False
            metadata = {
                "connection_intent": intent.value,
                "classification": revalidated.kind.value,
                "tolerance_used": revalidated.tolerance_used or 0.0,
                "max_residual": component.max_residual,
                "face_sequence": [face_handle.id],
            }
            attachment_id = geometry.ensure_attachment(
                member.id,
                attachment_kind,
                attachment_target_kind,
                attachment_target_id,
                member_range,
                target_parameters,
                connection_intent=intent,
                evidence=component.quality.value,
                max_residual=component.max_residual,
                tolerance_used=revalidated.tolerance_used or 0.0,
                part_id=geometry.members[member.id].part_id,
                sheet_id=(target.id if target.kind == "sheet" else None),
                provenance={
                    "classification": revalidated.kind.value,
                    "face_id": face_handle.id,
                },
                lineage=(("member", member.id), ("face", face_handle.id)),
                metadata=metadata,
            )
        relations.append(geometry.handle("attachment", attachment_id))

        sheet_id = target.id if target.kind == "sheet" else _sheet_for_face(
            geometry, face_handle.id
        )
        if sheet_id is not None:
            existing_junction = next(
                (
                    item
                    for item in geometry.junctions.values()
                    if item.member_ids == (member.id,)
                    and item.sheet_ids == (sheet_id,)
                    and attachment_id in item.attachment_ids
                    and _policy_name(
                        getattr(item, "connection_intent", item.metadata.to_dict().get("connection_intent"))
                    )
                    == intent.value
                ),
                None,
            )
            needs_junction = member_range.is_point or intent in (
                ConnectionIntent.CONNECT,
                ConnectionIntent.IMPRINT,
            )
            if existing_junction is None and needs_junction:
                reused = False
                junction_id = geometry.ensure_junction(
                    relation_junction_kind,
                    (JunctionMemberUse(member.id, member_range),),
                    sheet_ids=(sheet_id,),
                    attachment_ids=(attachment_id,),
                    connection_intent=intent,
                    metadata={"connection_intent": intent.value},
                    provenance={
                        "classification": revalidated.kind.value,
                        "face_id": face_handle.id,
                    },
                )
            elif existing_junction is not None:
                junction_id = existing_junction.id
            else:
                junction_id = None
            if junction_id is not None:
                relations.append(geometry.handle("junction", junction_id))
    return tuple(relations), reused


def _world_key(point: Sequence[float], quantum: float) -> tuple[int, int, int]:
    return tuple(
        int(round(float(component) / quantum)) for component in point
    )  # type: ignore[return-value]


def _point_on_segment(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    tolerance: float,
) -> bool:
    vector = end - start
    squared = float(vector @ vector)
    if squared <= tolerance * tolerance:
        return float(np.linalg.norm(point - start)) <= tolerance
    parameter = float((point - start) @ vector / squared)
    projected = start + np.clip(parameter, 0.0, 1.0) * vector
    return (
        -tolerance <= parameter <= 1.0 + tolerance
        and float(np.linalg.norm(point - projected)) <= tolerance
    )


def _shared_edges_for_components(
    geometry: GeometryModel,
    first_face: int,
    second_face: int,
    components: Sequence[IntersectionComponent],
    tolerance: float,
) -> tuple[int, ...]:
    first_edges = {
        item.edge
        for loop in (geometry.faces[first_face].loop,) + geometry.faces[first_face].holes
        for item in loop
    }
    second_edges = {
        item.edge
        for loop in (geometry.faces[second_face].loop,) + geometry.faces[second_face].holes
        for item in loop
    }
    shared = tuple(sorted(first_edges & second_edges))
    matched: list[int] = []
    for component in components:
        if len(component.witnesses) < 2:
            return ()
        exact_subparents = tuple(
            parent.id
            for parent in (component.first_subparent, component.second_subparent)
            if parent is not None and parent.kind == "edge" and parent.id in shared
        )
        if len(exact_subparents) == 1:
            matched.append(exact_subparents[0])
            continue
        endpoints = tuple(
            np.asarray(point, dtype=float)
            for point in (component.witnesses[0], component.witnesses[-1])
        )
        match = next(
            (
                edge_id
                for edge_id in shared
                if all(
                    min(
                        float(np.linalg.norm(sample - endpoint))
                        for sample in geometry.sample_edge(
                            edge_id, np.asarray((0.0, 1.0))
                        )
                    )
                    <= tolerance
                    for endpoint in endpoints
                )
            ),
            None,
        )
        if match is None:
            return ()
        matched.append(match)
    return tuple(dict.fromkeys(matched))


def _boundary_edge_for_curved_face_component(
    geometry: GeometryModel,
    result: IntersectionResult,
    curved_face_id: int,
) -> int | None:
    if len(result.components) != 1:
        return None
    curved_edges = {
        item.edge
        for loop in (geometry.faces[curved_face_id].loop,)
        + geometry.faces[curved_face_id].holes
        for item in loop
    }
    candidates = tuple(
        parent.id
        for parent in (
            result.components[0].first_subparent,
            result.components[0].second_subparent,
        )
        if parent is not None and parent.kind == "edge" and parent.id in curved_edges
    )
    return candidates[0] if len(candidates) == 1 else None


def _imprint_planar_support_with_boundary_curve(
    geometry: GeometryModel,
    *,
    plane_face_id: int,
    curved_face_id: int,
    edge_id: int,
    plane_is_first: bool,
    tolerance: float,
) -> FaceIntersection:
    edge = geometry.edges[edge_id]
    endpoints = tuple(
        geometry.vertex_position(vertex_id) for vertex_id in (edge.start, edge.end)
    )
    vertex_by_key = {
        _world_key(point, tolerance): vertex_id
        for point, vertex_id in zip(endpoints, (edge.start, edge.end))
    }
    edge_by_vertices = {tuple(sorted((edge.start, edge.end))): edge_id}
    plane_children, relation_edges = _fragment_planar_face_by_segments(
        geometry,
        plane_face_id,
        ((endpoints[0], endpoints[1]),),
        vertex_by_key,
        edge_by_vertices,
        tolerance=tolerance,
    )
    if edge_id not in relation_edges:
        raise GeometryError(
            "planar boundary-curve imprint did not reuse the certified edge"
        )
    errors = geometry.validate_topology()
    if errors:
        raise GeometryError(
            "boundary-curve imprint produced invalid topology: " + "; ".join(errors)
        )
    edge_reference = EntityRef("edge", edge_id)
    plane_faces = tuple(EntityRef("face", item) for item in plane_children)
    curved_faces = (EntityRef("face", curved_face_id),)
    return FaceIntersection(
        edge_reference,
        plane_faces if plane_is_first else curved_faces,
        curved_faces if plane_is_first else plane_faces,
        (edge_reference,),
    )


def _fragment_planar_face_by_segments(
    geometry: GeometryModel,
    face_id: int,
    segments: Sequence[tuple[np.ndarray, np.ndarray]],
    vertex_by_key: dict[tuple[int, int, int], int],
    edge_by_vertices: dict[tuple[int, int], int],
    *,
    tolerance: float,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Fragment one straight-trimmed planar face by every material segment.

    Interior segment endpoints receive deterministic nearest-boundary seams so
    the existing loop topology can represent a conforming T-junction without
    extending the physical intersection relation itself.
    """

    try:
        from shapely.geometry import LineString, Point
        from shapely.ops import nearest_points, polygonize, unary_union
    except ImportError as error:  # pragma: no cover - preflight guards this
        raise GeometryError("planar_backend_unavailable") from error

    face = geometry.faces[face_id]
    surface = face.surface
    if not isinstance(surface, Plane):
        raise GeometryError("multi-component topology fragmentation requires Plane support")

    # A T-junction commonly terminates with the complete boundary edge of one
    # participating shell face already representing the qualified curve. If
    # that edge was selected as the canonical relation by the caller, retain
    # the face and its FaceUse/Coedge identity instead of needlessly rebuilding
    # an identical polygon (which could disturb an otherwise connected Sheet).
    boundary_edges = {
        item.edge for loop in (face.loop,) + face.holes for item in loop
    }
    existing_relation_edges: list[int] = []
    for start, end in segments:
        start_vertex = vertex_by_key.get(_world_key(start, tolerance))
        end_vertex = vertex_by_key.get(_world_key(end, tolerance))
        if start_vertex is None or end_vertex is None:
            break
        edge_id = edge_by_vertices.get(
            tuple(sorted((start_vertex, end_vertex)))
        )
        if edge_id is None or edge_id not in boundary_edges:
            break
        edge = geometry.edges.get(edge_id)
        if edge is None or not isinstance(edge.curve, Straight):
            break
        endpoints = geometry.sample_edge(edge_id, np.asarray((0.0, 1.0)))
        if not (
            (
                float(np.linalg.norm(endpoints[0] - start)) <= tolerance
                and float(np.linalg.norm(endpoints[1] - end)) <= tolerance
            )
            or (
                float(np.linalg.norm(endpoints[1] - start)) <= tolerance
                and float(np.linalg.norm(endpoints[0] - end)) <= tolerance
            )
        ):
            break
        existing_relation_edges.append(edge_id)
    if len(existing_relation_edges) == len(segments):
        return (face_id,), tuple(dict.fromkeys(existing_relation_edges))

    polygon = _face_polygon_in_plane(geometry, face_id, surface)
    if polygon is None:
        raise GeometryError("planar_backend_unavailable")
    if polygon.is_empty or not polygon.is_valid:
        raise GeometryError("invalid planar face polygon")

    # Certified predicates may return a witness a few ulps inside an existing
    # trim corner (for example 1.9999999999999996 for an exact coordinate of
    # 2.0).  Passing that value to polygonize leaves a microscopic dangling
    # segment instead of noding the cut into the boundary, so the valid split
    # is silently reconstructed as one cell.  Snap only to existing trim
    # vertices within the participating model tolerance, before any UV
    # linework is created.  Distance then vertex ID provides a deterministic
    # choice if a tolerance-sized model contains more than one candidate.
    trim_vertex_ids = tuple(
        sorted(
            {
                vertex_id
                for loop in (face.loop,) + face.holes
                for item in loop
                for vertex_id in (
                    geometry.edges[item.edge].start,
                    geometry.edges[item.edge].end,
                )
            }
        )
    )
    trim_vertices = tuple(
        (vertex_id, geometry.vertex_position(vertex_id))
        for vertex_id in trim_vertex_ids
    )

    def snap_endpoint(point: np.ndarray) -> np.ndarray:
        raw = np.asarray(point, dtype=float)
        if not trim_vertices:
            return raw
        distance, vertex_id, exact = min(
            (
                float(np.linalg.norm(raw - position)),
                candidate_id,
                position,
            )
            for candidate_id, position in trim_vertices
        )
        if distance > tolerance:
            return raw
        vertex_by_key[_world_key(raw, tolerance)] = vertex_id
        vertex_by_key[_world_key(exact, tolerance)] = vertex_id
        return exact

    segments = tuple(
        (snap_endpoint(start), snap_endpoint(end)) for start, end in segments
    )

    direction = segments[0][1] - segments[0][0]
    direction_length = float(np.linalg.norm(direction))
    if direction_length <= tolerance:
        raise GeometryError("planned face cut contains a degenerate segment")
    direction /= direction_length
    canonical_index = int(np.argmax(np.abs(direction)))
    if direction[canonical_index] < 0.0:
        direction *= -1.0
    ordered_segments = []
    for raw_start, raw_end in segments:
        start, end = raw_start, raw_end
        if float(start @ direction) > float(end @ direction):
            start, end = end, start
        ordered_segments.append((start, end))
    ordered_segments.sort(
        key=lambda item: (
            float(item[0] @ direction),
            float(item[1] @ direction),
        )
    )

    cut_lines = []
    seam_lines = []
    boundary_world: list[np.ndarray] = []
    ordered_uv: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for start, end in ordered_segments:
        start_uv = tuple(float(item) for item in surface.local_uv(start))
        end_uv = tuple(float(item) for item in surface.local_uv(end))
        ordered_uv.append((start_uv, end_uv))
        cut = LineString((start_uv, end_uv))
        material = cut.intersection(polygon)
        if material.is_empty:
            raise GeometryError("planned face cut no longer lies in face material")
        cut_lines.append(cut)
        for point_uv, point_world in ((start_uv, start), (end_uv, end)):
            if float(Point(point_uv).distance(polygon.boundary)) <= tolerance:
                boundary_world.append(point_world)

    # Gaps between disjoint material components are construction seams.  On a
    # face with a hole they lie outside material; on the other participating
    # face they make the actual intersection stubs part of a complete split
    # line instead of leaving unrepresentable dangling interior edges.
    for (_start, previous_end), (following_start, _end) in zip(
        ordered_uv, ordered_uv[1:]
    ):
        seam_lines.append(LineString((previous_end, following_start)))
    for point_uv in (ordered_uv[0][0], ordered_uv[-1][1]):
        point = Point(point_uv)
        if float(point.distance(polygon.boundary)) <= tolerance:
            continue
        _source, boundary = nearest_points(point, polygon.boundary)
        boundary_uv = tuple(boundary.coords)[0]
        seam_lines.append(LineString((point_uv, boundary_uv)))
        boundary_world.append(
            np.asarray(
                surface.evaluate(float(boundary_uv[0]), float(boundary_uv[1])),
                dtype=float,
            )
        )

    # Split every touched trim edge before replacing the face.  Its resulting
    # subedges can then be reused exactly by the polygonized cells.
    for point in boundary_world:
        _boundary_vertex(geometry, face_id, point)
    face = geometry.faces[face_id]
    for loop in (face.loop,) + face.holes:
        for item in loop:
            edge = geometry.edges[item.edge]
            for vertex_id in (edge.start, edge.end):
                vertex_by_key.setdefault(
                    _world_key(geometry.vertex_position(vertex_id), tolerance),
                    vertex_id,
                )
            if isinstance(edge.curve, Straight):
                edge_by_vertices.setdefault(
                    tuple(sorted((edge.start, edge.end))), edge.id
                )

    linework = unary_union((polygon.boundary, *cut_lines, *seam_lines))
    cells = tuple(
        sorted(
            (
                cell
                for cell in polygonize(linework)
                if polygon.covers(cell.representative_point())
                and float(cell.area) > geometry.tolerance.effective_area(
                    _face_length_scale(geometry, face_id)
                )
            ),
            key=lambda cell: (
                round(float(cell.centroid.x), 14),
                round(float(cell.centroid.y), 14),
                round(float(cell.area), 14),
            ),
        )
    )
    if not cells:
        raise GeometryError("planar imprint produced no material cells")
    metadata = face.metadata.to_dict()
    parameterization = face.parameterization
    tags = geometry.tags_for(face.ref)
    attachment_snapshots = _capture_face_attachments(geometry, face_id)
    geometry._delete_entity("face", face_id)  # noqa: SLF001
    cut_edge_ids: set[int] = set()

    def vertex(uv: Sequence[float]) -> int:
        world = np.asarray(surface.evaluate(float(uv[0]), float(uv[1])), dtype=float)
        key = _world_key(world, tolerance)
        existing = vertex_by_key.get(key)
        if existing is not None and existing in geometry.vertices:
            return existing
        made = geometry.add_point(*world)
        vertex_by_key[key] = made
        return made

    def loop(coordinates) -> tuple[OrientedEdge, ...]:
        points = list(coordinates)[:-1]
        vertices = [vertex(point) for point in points]
        made: list[OrientedEdge] = []
        for start_vertex, end_vertex in zip(
            vertices, vertices[1:] + vertices[:1]
        ):
            key = tuple(sorted((start_vertex, end_vertex)))
            edge_id = edge_by_vertices.get(key)
            if edge_id is None or edge_id not in geometry.edges:
                edge_id = geometry.add_line(start_vertex, end_vertex)
                edge_by_vertices[key] = edge_id
            edge = geometry.edges[edge_id]
            made.append(OrientedEdge(edge_id, edge.start == start_vertex))
            start_world = geometry.vertex_position(start_vertex)
            end_world = geometry.vertex_position(end_vertex)
            if any(
                _point_on_segment(start_world, cut_start, cut_end, tolerance)
                and _point_on_segment(end_world, cut_start, cut_end, tolerance)
                for cut_start, cut_end in segments
            ):
                cut_edge_ids.add(edge_id)
        return tuple(made)

    children: list[int] = []
    for cell in cells:
        outer = loop(cell.exterior.coords)
        holes = tuple(loop(ring.coords) for ring in cell.interiors)
        corners = geometry._detect_corners(outer) if len(outer) == 4 else None  # noqa: SLF001
        child = geometry.add_face_from_loop(outer, corners, surface=surface)
        geometry._put_entity(  # noqa: SLF001
            "face",
            replace(
                geometry.faces[child],
                holes=holes,
                metadata=metadata,
                surface=surface,
                parameterization=parameterization,
            ),
        )
        geometry.tag(EntityRef("face", child), *tags)
        children.append(child)
    _remap_face_attachments(
        geometry, face, children, attachment_snapshots
    )
    geometry.record_replacement(
        EntityRef("face", face_id),
        tuple(EntityRef("face", child) for child in children),
    )
    return tuple(children), tuple(sorted(cut_edge_ids))


def _sheet_ids_for_face(geometry: GeometryModel, face_id: int) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                geometry.face_uses[use_id].sheet_id
                for use_id in geometry._face_structural_uses.get(face_id, ())  # noqa: SLF001
            }
        )
    )


def _ensure_face_used_by_sheet(
    geometry: GeometryModel, face_id: int, sheet_id: int
) -> EntityHandle | None:
    """Add one persistent FaceUse when a shared region gains another Sheet owner."""

    existing_use_ids = tuple(
        sorted(geometry._face_structural_uses.get(face_id, ()))  # noqa: SLF001
    )
    for use_id in existing_use_ids:
        if geometry.face_uses[use_id].sheet_id == sheet_id:
            return None
    for use_id in existing_use_ids:
        existing = geometry.face_uses[use_id]
        geometry._put_structural(  # noqa: SLF001
            "face_use",
            replace(
                existing,
                metadata={
                    **dict(existing.metadata),
                    "anygeometry.shared_region": True,
                },
            ),
        )
    sheet = geometry.sheets[sheet_id]
    use_id = geometry._allocate_structural("face_use")  # noqa: SLF001
    made = geometry._new_face_use(  # noqa: SLF001
        use_id,
        sheet_id,
        face_id,
        metadata={"anygeometry.shared_region": True},
    )
    geometry._put_structural("face_use", made)  # noqa: SLF001
    geometry._put_structural(  # noqa: SLF001
        "sheet", replace(sheet, face_use_ids=(*sheet.face_use_ids, use_id))
    )
    return geometry.handle("face_use", use_id)


def _share_planar_overlap_ownership(
    geometry: GeometryModel,
    overlap_faces: Sequence[EntityRef],
    sheet_ids: Sequence[int],
) -> tuple[EntityHandle, ...]:
    relations: list[EntityHandle] = []
    for face_ref in sorted(overlap_faces, key=lambda item: item.id):
        relations.append(geometry.handle("face", face_ref.id))
        for sheet_id in sorted(set(sheet_ids)):
            made = _ensure_face_used_by_sheet(geometry, face_ref.id, sheet_id)
            if made is not None:
                relations.append(made)
    return tuple(relations)


def _connect_fully_coincident_faces(
    geometry: GeometryModel,
    first_face: int,
    second_face: int,
) -> tuple[EntityHandle, ...]:
    """Represent full curved coincidence as one face used by every owner Sheet."""

    if first_face == second_face:
        return (geometry.handle("face", first_face),)
    canonical, retired = sorted((first_face, second_face))
    canonical_face = geometry.faces[canonical]
    retired_face = geometry.faces[retired]
    if canonical_face.holes and retired_face.holes:
        oriented_holes: list[tuple[OrientedEdge, ...]] = []
        changed_holes = False
        for canonical_hole in canonical_face.holes:
            matching = next(
                (
                    retired_hole
                    for retired_hole in retired_face.holes
                    if {item.edge for item in retired_hole}
                    == {item.edge for item in canonical_hole}
                ),
                None,
            )
            if matching is not None:
                oriented_holes.append(tuple(matching))
                changed_holes = changed_holes or tuple(matching) != tuple(canonical_hole)
            else:
                oriented_holes.append(tuple(canonical_hole))
        if changed_holes:
            geometry._put_entity(  # noqa: SLF001
                "face",
                replace(canonical_face, holes=tuple(oriented_holes)),
            )
    retired_uses = tuple(
        geometry.face_uses[use_id]
        for use_id in sorted(geometry._face_structural_uses.get(retired, ()))  # noqa: SLF001
    )
    snapshots = _capture_face_attachments(geometry, retired)
    geometry._delete_entity("face", retired)  # noqa: SLF001
    relation_handles: list[EntityHandle] = [geometry.handle("face", canonical)]
    for old_use in retired_uses:
        made = _ensure_face_used_by_sheet(geometry, canonical, old_use.sheet_id)
        if made is not None:
            relation_handles.append(made)
        for coedge_id in old_use.coedge_ids:
            if coedge_id in geometry.coedges:
                geometry._delete_structural("coedge", coedge_id)  # noqa: SLF001
        geometry._delete_structural("face_use", old_use.id)  # noqa: SLF001
        sheet = geometry.sheets[old_use.sheet_id]
        retained = tuple(item for item in sheet.face_use_ids if item != old_use.id)
        geometry._put_structural(  # noqa: SLF001
            "sheet", replace(sheet, face_use_ids=retained)
        )
    _remap_face_attachments(
        geometry, retired_face, (canonical,), snapshots
    )
    geometry.record_replacement(
        EntityRef("face", retired), (EntityRef("face", canonical),)
    )
    errors = geometry.validate_topology()
    if errors:
        raise GeometryError(
            "coincident-region connection produced invalid topology: "
            + "; ".join(errors)
        )
    return tuple(relation_handles)


def _face_imprint_application(
    geometry: GeometryModel,
    plan: ImprintPlan,
    revalidated: IntersectionResult,
) -> tuple[FaceIntersection | None, tuple[EntityHandle, ...], bool]:
    if _policy_name(plan.policy) not in (
        MutationPolicy.IMPRINT.value,
        "connect",
        "reuse_existing",
    ):
        raise GeometryError(
            "face topology operation requires IMPRINT, CONNECT, or REUSE_EXISTING policy"
        )
    if revalidated.kind in (
        IntersectionKind.UNSUPPORTED,
        IntersectionKind.CAPABILITY_MISSING,
        IntersectionKind.UNCLASSIFIED,
    ):
        raise GeometryError(
            "face imprint cannot apply an unqualified intersection: "
            + "; ".join(revalidated.diagnostics)
        )
    if revalidated.dimension is IntersectionDimension.REGION:
        first_face, second_face = plan.first_parent.id, plan.second_parent.id
        first_surface = geometry.faces[first_face].surface
        second_surface = geometry.faces[second_face].surface
        if (
            revalidated.kind is IntersectionKind.COINCIDENT
            and not (
                isinstance(first_surface, Plane)
                and isinstance(second_surface, Plane)
            )
        ):
            with geometry.transaction():
                relations = _connect_fully_coincident_faces(
                    geometry, first_face, second_face
                )
            return None, relations, False
        if not (
            isinstance(first_surface, Plane)
            and isinstance(second_surface, Plane)
        ):
            if len(revalidated.components) != 1:
                raise GeometryError(
                    "curved coincident-region connection needs one certified cell"
                )
            with geometry.transaction():
                relations = _connect_contained_curved_region(
                    geometry,
                    first_face,
                    second_face,
                    revalidated.components[0],
                    revalidated.tolerance_used or geometry.tolerance.length,
                )
            return None, relations, False
        from .overlaps import fragment_coplanar_overlaps

        sheet_ids = tuple(
            sorted(
                set(
                    (*_sheet_ids_for_face(geometry, first_face),
                     *_sheet_ids_for_face(geometry, second_face))
                )
            )
        )
        fragmented = fragment_coplanar_overlaps(
            geometry, (first_face, second_face)
        )
        relation_handles = _share_planar_overlap_ownership(
            geometry,
            tuple(EntityRef("face", item) for item in fragmented.overlap_faces),
            sheet_ids,
        )
        return None, relation_handles, False
    if revalidated.kind is IntersectionKind.DISJOINT:
        detail = "; ".join(revalidated.diagnostics)
        if "parallel_face_supports" in revalidated.diagnostics:
            raise GeometryError("faces are parallel or coplanar; no unique intersection line")
        raise GeometryError(
            "disjoint faces cannot be imprinted" + (f": {detail}" if detail else "")
        )
    if revalidated.dimension is not IntersectionDimension.CURVE:
        raise GeometryError("point-only face contact cannot be imprinted as an edge")

    first_face, second_face = plan.first_parent.id, plan.second_parent.id
    first_surface = geometry.faces[first_face].surface
    second_surface = geometry.faces[second_face].surface
    if isinstance(first_surface, Plane) and isinstance(second_surface, Cylinder):
        alignment = abs(float(first_surface.normal @ second_surface.axis))
        if abs(alignment - 1.0) <= geometry.tolerance.angular:
            made = _imprint_transverse_plane_cylinder(
                geometry, first_face, second_face, fragment=True
            )
            assert isinstance(made, FaceIntersection)
            return made, (), False
    if isinstance(first_surface, Cylinder) and isinstance(second_surface, Plane):
        alignment = abs(float(second_surface.normal @ first_surface.axis))
        if abs(alignment - 1.0) <= geometry.tolerance.angular:
            made = _imprint_transverse_plane_cylinder(
                geometry, second_face, first_face, fragment=True
            )
            assert isinstance(made, FaceIntersection)
            return (
                FaceIntersection(
                    made.edge,
                    made.second_faces,
                    made.first_faces,
                    made.edges,
                ),
                (),
                False,
            )
    tolerance = revalidated.tolerance_used or geometry.tolerance.length
    shared_edges = _shared_edges_for_components(
        geometry,
        first_face,
        second_face,
        revalidated.components,
        tolerance,
    )
    if len(shared_edges) == len(revalidated.components):
        references = tuple(EntityRef("edge", edge_id) for edge_id in shared_edges)
        return (
            FaceIntersection(
                references[0],
                (EntityRef("face", first_face),),
                (EntityRef("face", second_face),),
                references,
            ),
            (),
            True,
        )
    plane_is_first = isinstance(first_surface, Plane)
    plane_is_second = isinstance(second_surface, Plane)
    if plane_is_first != plane_is_second:
        plane_face = first_face if plane_is_first else second_face
        curved_face = second_face if plane_is_first else first_face
        curved_surface = second_surface if plane_is_first else first_surface
        if curved_surface is not None and not isinstance(curved_surface, Cylinder):
            boundary_edge = _boundary_edge_for_curved_face_component(
                geometry, revalidated, curved_face
            )
            if boundary_edge is not None:
                made = _imprint_planar_support_with_boundary_curve(
                    geometry,
                    plane_face_id=plane_face,
                    curved_face_id=curved_face,
                    edge_id=boundary_edge,
                    plane_is_first=plane_is_first,
                    tolerance=tolerance,
                )
                return made, (), False
    if _policy_name(plan.policy) == "reuse_existing":
        raise GeometryError(
            "REUSE_EXISTING requires a compatible shared face-intersection edge"
        )

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for component in revalidated.components:
        if len(component.witnesses) < 2:
            raise GeometryError("face imprint curve needs two endpoint witnesses")
        segments.append(
            (
                np.asarray(component.witnesses[0], dtype=float),
                np.asarray(component.witnesses[-1], dtype=float),
            )
        )
    if isinstance(first_surface, Plane) and isinstance(second_surface, Plane):
        vertex_by_key: dict[tuple[int, int, int], int] = {}
        edge_by_vertices: dict[tuple[int, int], int] = {}

        # Prefer an already-existing full boundary edge as the canonical
        # shared relation. This is the exact shell-wall case produced by a
        # straight sketch extrusion: the wall owns the curve already, while
        # the supporting sheet needs to be imprinted with it.
        for start, end in segments:
            candidates: list[int] = []
            for face_id in (first_face, second_face):
                face = geometry.faces[face_id]
                for loop in (face.loop,) + face.holes:
                    for item in loop:
                        edge = geometry.edges[item.edge]
                        if not isinstance(edge.curve, Straight):
                            continue
                        endpoints = geometry.sample_edge(
                            edge.id, np.asarray((0.0, 1.0))
                        )
                        if (
                            float(np.linalg.norm(endpoints[0] - start)) <= tolerance
                            and float(np.linalg.norm(endpoints[1] - end)) <= tolerance
                        ) or (
                            float(np.linalg.norm(endpoints[1] - start)) <= tolerance
                            and float(np.linalg.norm(endpoints[0] - end)) <= tolerance
                        ):
                            candidates.append(edge.id)
            if candidates:
                edge_id = min(candidates)
                edge = geometry.edges[edge_id]
                for vertex_id in (edge.start, edge.end):
                    vertex_by_key[_world_key(
                        geometry.vertex_position(vertex_id), tolerance
                    )] = vertex_id
                edge_by_vertices[
                    tuple(sorted((edge.start, edge.end)))
                ] = edge_id
        first_children, first_edges = _fragment_planar_face_by_segments(
            geometry,
            first_face,
            segments,
            vertex_by_key,
            edge_by_vertices,
            tolerance=tolerance,
        )
        second_children, second_edges = _fragment_planar_face_by_segments(
            geometry,
            second_face,
            segments,
            vertex_by_key,
            edge_by_vertices,
            tolerance=tolerance,
        )
        relation_edges = tuple(sorted(set(first_edges) & set(second_edges)))
        if not relation_edges:
            raise GeometryError("planar imprint did not create a shared relation edge")
        references = tuple(EntityRef("edge", edge_id) for edge_id in relation_edges)
        return (
            FaceIntersection(
                references[0],
                tuple(EntityRef("face", item) for item in first_children),
                tuple(EntityRef("face", item) for item in second_children),
                references,
            ),
            (),
            False,
        )
    return (
        _imprint_components(
            geometry, first_face, second_face, revalidated.components
        ),
        (),
        False,
    )


def apply_imprint(
    geometry: GeometryModel,
    plan: ImprintPlan,
    *,
    policy: object,
) -> ImprintApplication:
    """Atomically apply a fresh verified plan with repeated explicit intent."""

    if not isinstance(plan, ImprintPlan):
        raise TypeError("apply_imprint needs an ImprintPlan")
    normalized_policy = _normalize_intersection_policy(policy)
    if _policy_name(normalized_policy) != _policy_name(plan.policy):
        raise GeometryError("apply policy does not match the planned policy")
    if geometry.model_id != plan.model_id:
        raise GeometryError("imprint plan belongs to another geometry model")
    if geometry.revision != plan.revision:
        reused_application = _reuse_stale_face_plan(geometry, plan)
        if reused_application is not None:
            return reused_application
        raise GeometryError(
            f"imprint plan is stale: planned revision {plan.revision}, "
            f"current revision {geometry.revision}"
        )
    _resolve_plan_parent(geometry, plan.first_parent)
    _resolve_plan_parent(geometry, plan.second_parent)
    if not plan.result.classified:
        raise GeometryError(
            "unqualified intersection plan cannot be applied: "
            + "; ".join(plan.result.diagnostics)
        )
    revalidated = query_intersection(
        geometry,
        plan.first_parent,
        plan.second_parent,
        qualification=plan.qualification,
    )
    if (
        revalidated.kind != plan.result.kind
        or revalidated.dimension != plan.result.dimension
        or len(revalidated.components) != len(plan.result.components)
    ):
        raise GeometryError("intersection changed since imprint planning")
    if not revalidated.classified:
        raise GeometryError(
            "unqualified intersection cannot be applied: "
            + "; ".join(revalidated.diagnostics)
        )
    policy_value = _policy_name(normalized_policy)
    if policy_value == "reject":
        raise GeometryError("intersection mutation rejected by policy")
    if policy_value == "reuse_existing" and plan.operation is ImprintOperation.NO_TOPOLOGY:
        raise GeometryError(
            "REUSE_EXISTING requires compatible existing topology or relations"
        )
    if plan.operation is ImprintOperation.NO_TOPOLOGY:
        if (
            plan.result.kind is IntersectionKind.DISJOINT
            and "parallel_face_supports" in plan.result.diagnostics
        ):
            raise GeometryError(
                "faces are parallel or coplanar; no unique intersection line"
            )
        raise GeometryError(
            f"policy {policy_value} has no applicable persistent operation for "
            f"{plan.first_parent.kind}/{plan.second_parent.kind}"
        )

    relations: tuple[EntityHandle, ...] = ()
    face_intersection: FaceIntersection | None = None
    reused = False
    revision_before = geometry.revision
    with geometry.transaction():
        if plan.operation is ImprintOperation.FACE_IMPRINT:
            face_intersection, relations, reused = _face_imprint_application(
                geometry, plan, revalidated
            )
        elif plan.operation is ImprintOperation.MEMBER_CONNECTION:
            relations, reused = _member_connection_application(
                geometry, plan, revalidated
            )
        elif plan.operation is ImprintOperation.MEMBER_SHEET_RELATION:
            relations, reused = _member_sheet_application(
                geometry, plan, revalidated
            )
        else:  # pragma: no cover - closed enum guard
            raise GeometryError(f"unsupported imprint operation {plan.operation.value}")
    change_set = geometry.last_change_set
    if geometry.revision == revision_before:
        change_set = ChangeSet(revision_before, revision_before)
        reused = True
    return ImprintApplication(
        plan,
        revalidated,
        change_set,
        relations,
        face_intersection,
        reused,
    )


def _reuse_stale_face_plan(
    geometry: GeometryModel, plan: ImprintPlan
) -> ImprintApplication | None:
    """Recognize an already-applied face plan without weakening stale checks."""

    if plan.operation is not ImprintOperation.FACE_IMPRINT:
        return None
    first_resolution = geometry.resolve_handle(plan.first_parent)
    second_resolution = geometry.resolve_handle(plan.second_parent)
    if (
        first_resolution.status is not ResolutionStatus.REPLACED
        or second_resolution.status is not ResolutionStatus.REPLACED
    ):
        return None
    first_faces = tuple(
        item for item in first_resolution.resolved if item.kind == "face"
    )
    second_faces = tuple(
        item for item in second_resolution.resolved if item.kind == "face"
    )
    if not first_faces or not second_faces:
        return None
    current_revision = geometry.revision
    empty_change = ChangeSet(current_revision, current_revision)
    if plan.result.dimension is IntersectionDimension.REGION:
        source_ids = {plan.first_parent.id, plan.second_parent.id}
        relations = tuple(
            sorted(
                {
                    handle
                    for handle in (*first_faces, *second_faces)
                    if source_ids.issubset(
                        set(
                            geometry.faces[handle.id].metadata.to_dict().get(
                                "source_faces", ()
                            )
                        )
                    )
                }
            )
        )
        if not relations:
            return None
        return ImprintApplication(
            plan,
            plan.result,
            empty_change,
            relations=relations,
            reused=True,
        )

    first_edges = {
        item.edge
        for handle in first_faces
        for loop in (geometry.faces[handle.id].loop,) + geometry.faces[handle.id].holes
        for item in loop
    }
    second_edges = {
        item.edge
        for handle in second_faces
        for loop in (geometry.faces[handle.id].loop,) + geometry.faces[handle.id].holes
        for item in loop
    }
    shared = tuple(sorted(first_edges & second_edges))
    tolerance = plan.result.tolerance_used or geometry.tolerance.length
    matches: list[int] = []
    for component in plan.result.components:
        if len(component.witnesses) < 2:
            return None
        endpoints = tuple(
            np.asarray(point, dtype=float)
            for point in (component.witnesses[0], component.witnesses[-1])
        )
        edge_id = next(
            (
                candidate
                for candidate in shared
                if all(
                    min(
                        float(np.linalg.norm(sample - endpoint))
                        for sample in geometry.sample_edge(
                            candidate, np.asarray((0.0, 1.0))
                        )
                    )
                    <= tolerance
                    for endpoint in endpoints
                )
            ),
            None,
        )
        if edge_id is None:
            return None
        matches.append(edge_id)
    references = tuple(
        EntityRef("edge", edge_id) for edge_id in dict.fromkeys(matches)
    )
    if not references:
        return None
    return ImprintApplication(
        plan,
        plan.result,
        empty_change,
        face_intersection=FaceIntersection(
            references[0],
            tuple(EntityRef("face", item.id) for item in first_faces),
            tuple(EntityRef("face", item.id) for item in second_faces),
            references,
        ),
        reused=True,
    )
