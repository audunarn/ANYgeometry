"""Analytical structural-surface intersections and lightweight fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .curves import Arc, Spline, Straight
from .entities import EntityRef, OrientedEdge
from .errors import GeometryError
from .model import GeometryModel
from .surfaces import CoonsSurface, Cylinder, Plane, SurfaceProtocol

__all__ = [
    "FaceIntersection", "intersect_faces", "intersect_surfaces", "line_cylinder", "line_line",
    "line_plane", "numerical_surface_intersection", "plane_cylinder",
    "plane_plane",
]


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
    scale = max(float(np.linalg.norm(p)), float(np.linalg.norm(q)), 1.0)
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
    for index in range(len(signed)-1):
        if signed[index] == 0.0:
            angle = angles[index]
        elif signed[index]*signed[index+1] > 0.0:
            continue
        else:
            ratio = signed[index]/(signed[index]-signed[index+1])
            angle = angles[index] + ratio*(angles[index+1]-angles[index])
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


def _face_plane(geometry: GeometryModel, face_id: int) -> Plane:
    face = geometry.faces[face_id]
    if isinstance(face.surface, Plane):
        return face.surface
    points = np.asarray([geometry.vertex_position(geometry.oriented_start_vertex(item)) for item in face.loop])
    origin = points.mean(axis=0)
    _u, singular, vectors = np.linalg.svd(points-origin)
    if singular[-1] > 1e-8*max(singular[0], 1.0):
        raise GeometryError(f"face {face_id} is not planar")
    u_vector = vectors[0]
    v_vector = np.cross(vectors[-1], u_vector)
    return Plane(origin, u_vector, v_vector)


def _line_face_interval(geometry: GeometryModel, face_id: int, point: np.ndarray, direction: np.ndarray) -> tuple[float,float] | None:
    face = geometry.faces[face_id]
    values = []
    for item in face.loop:
        samples = geometry.sample_edge(item.edge, np.linspace(0.0,1.0,33))
        if not item.forward:
            samples = samples[::-1]
        for first, second in zip(samples[:-1], samples[1:]):
            hit = line_line(point, direction, first, second-first, tolerance=1e-7)
            if hit is None:
                continue
            segment = second-first
            along = float((hit-first) @ segment) / max(float(segment@segment), 1e-30)
            if -1e-7 <= along <= 1.0+1e-7:
                values.append(float((hit-point) @ direction))
    if len(values) < 2:
        return None
    return min(values), max(values)


def _boundary_vertex(geometry: GeometryModel, face_id: int, point: np.ndarray) -> int:
    face = geometry.faces[face_id]
    scale = max(float(np.linalg.norm(point)), 1.0)
    for item in face.loop:
        for vertex_id in (geometry.oriented_start_vertex(item), geometry.oriented_end_vertex(item)):
            if float(np.linalg.norm(geometry.vertex_position(vertex_id)-point)) <= 1e-7*scale:
                return vertex_id
    best = None
    for item in face.loop:
        _candidate, parameter, distance = geometry.closest_edge_point(
            item.edge, point
        )
        if best is None or distance < best[0]:
            best = (distance, item.edge, parameter)
    assert best is not None
    if best[0] > 1e-5*scale:
        raise GeometryError("intersection endpoint is not on the face boundary")
    vertex, _edges = geometry.split_edge(best[1], min(max(best[2],1e-9),1-1e-9))
    geometry.move_point(vertex, *point)
    return vertex


def _merge_vertex(geometry: GeometryModel, old: int, new: int) -> None:
    if old == new:
        return
    for edge in geometry.edges.values():
        if edge.start == old:
            edge.start = new
        if edge.end == old:
            edge.end = new
        if isinstance(edge.curve, Arc) and edge.curve.via_vertex == old:
            edge.curve = Arc(new)
        elif isinstance(edge.curve, Spline) and old in edge.curve.control_vertices:
            edge.curve = Spline(
                tuple(
                    new if vertex == old else vertex
                    for vertex in edge.curve.control_vertices
                )
            )
    if not geometry.edges_using_vertex(old):
        geometry.remove_vertex(old, record=False)
    geometry.record_replacement(EntityRef("vertex", old), (EntityRef("vertex", new),))


def _fragment_with_edge(geometry: GeometryModel, face_id: int, start: int, end: int, edge_id: int) -> tuple[int,int]:
    from .operations import _partition_face_holes, _split_loop  # internal topology primitives

    face = geometry.faces[face_id]
    first_chain, second_chain = _split_loop(face, start, end, geometry)
    first_holes, second_holes = _partition_face_holes(
        geometry, face, first_chain, second_chain, start, end
    )
    surface = face.surface
    metadata, tags = dict(face.metadata), geometry.tags_for(face.ref)
    geometry.remove_face(face_id, record=False)
    made = []
    for loop, holes in zip(
        (
            tuple(first_chain) + (OrientedEdge(edge_id, False),),
            tuple(second_chain) + (OrientedEdge(edge_id, True),),
        ),
        (first_holes, second_holes),
    ):
        corners = geometry._detect_corners(loop) if len(loop) >= 4 else None  # noqa: SLF001
        identifier = geometry.add_face_from_loop(loop, corners)
        child = geometry.faces[identifier]
        child.holes = holes
        child.metadata.update(metadata)
        if isinstance(surface, Plane) and corners is None:
            child.surface = surface
        elif corners is not None:
            child.surface = CoonsSurface()
        else:
            child.surface = surface
        geometry.tag(EntityRef("face",identifier), *tags)
        made.append(identifier)
    geometry.record_replacement(EntityRef("face",face_id), tuple(EntityRef("face",item) for item in made))
    return made[0], made[1]


def _imprint_segment(
    geometry: GeometryModel,
    first_face: int,
    second_face: int,
    endpoints: tuple[np.ndarray, np.ndarray],
) -> FaceIntersection:
    snapshot = geometry.topology_snapshot()
    try:
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
    except Exception:
        geometry.restore_topology(snapshot)
        raise
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
    if abs(float(plane.normal @ cylinder.axis)) > 1.0e-10:
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
                scale = max(float(np.linalg.norm(point)), 1.0)
                if distance > 1.0e-7*scale or not geometry.face_contains_uv(face_id, uv):
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
    plane = geometry.faces[face_id].surface
    assert isinstance(plane, Plane)
    points = np.asarray(
        [
            plane.local_uv(
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
    scale = max(
        float(np.linalg.norm(reference.origin)),
        abs(reference.height),
        reference.radius,
        1.0,
    )
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
    scale = max(float(np.linalg.norm(point)), 1.0)
    if distance > 1.0e-7 * scale:
        raise GeometryError(
            f"cylinder face {face_id} is not bounded by its surface patch"
        )
    return edge_id


def _transverse_cylinder_band(
    geometry: GeometryModel,
    plane_face: int,
    cylinder_face: int,
    *,
    tolerance: float = 1.0e-9,
) -> tuple[tuple[int, ...], float, np.ndarray]:
    """Preflight one complete conformal cylindrical band without mutation."""

    plane = geometry.faces[plane_face].surface
    reference = geometry.faces[cylinder_face].surface
    assert isinstance(plane, Plane) and isinstance(reference, Cylinder)
    alignment = abs(float(plane.normal @ reference.axis))
    if abs(alignment - 1.0) > 1.0e-10:
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
    if not tolerance < fraction < 1.0 - tolerance:
        raise GeometryError(
            "the transverse plane must cut strictly inside the cylinder height"
        )

    candidates: list[tuple[float, int, Cylinder]] = []
    period = 2.0 * np.pi
    for face_id, face in geometry.faces.items():
        surface = face.surface
        if not isinstance(surface, Cylinder) or not _same_cylinder_band(
            surface, reference, tolerance=tolerance
        ):
            continue
        if face.holes:
            raise GeometryError(
                "closed plane-cylinder imprint requires cylinder faces "
                "without existing holes"
            )
        if not tolerance < surface.sweep_angle < period - tolerance:
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
    if abs(sum(item[2].sweep_angle for item in candidates) - period) > tolerance:
        raise GeometryError(
            "the selected cylinder face does not belong to one complete band"
        )
    for current, following in zip(candidates, candidates[1:] + candidates[:1]):
        end_angle = current[0] + current[2].sweep_angle
        if _angular_distance(end_angle, following[0]) > tolerance:
            raise GeometryError(
                "the selected cylinder band has an angular gap or overlap"
            )
        current_end = current[2].evaluate(1.0, fraction)
        following_start = following[2].evaluate(0.0, fraction)
        if float(np.linalg.norm(current_end - following_start)) > tolerance * max(
            float(np.linalg.norm(current_end)), 1.0
        ):
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
        if abs(cross) > tolerance:
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
    scale = max(float(np.linalg.norm(centre)), reference.radius, 1.0)
    if boundary_distance <= reference.radius + tolerance * scale:
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
    geometry.remove_face(face_id, record=False)
    made = []
    for chain, oriented_ring in (
        (first_chain, OrientedEdge(edge_id, False)),
        (second_chain, OrientedEdge(edge_id, True)),
    ):
        loop = tuple(chain) + (oriented_ring,)
        corners = geometry._detect_corners(loop) if len(loop) >= 4 else None  # noqa: SLF001
        identifier = geometry.add_face_from_loop(loop, corners)
        child = geometry.faces[identifier]
        child.metadata.update(metadata)
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
        child.surface = _cylinder_fragment_surface(
            surface, fraction, upper=float(np.mean(axial_values)) > fraction
        )
        geometry.tag(child.ref, *tags)
        made.append(identifier)
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
    outer_loop, corners = face.loop, face.corners
    ring_loop = tuple(OrientedEdge(edge_id, True) for edge_id in edge_ids)
    if _loop_signed_area(geometry, plane_face, outer_loop) * _loop_signed_area(
        geometry, plane_face, ring_loop
    ) < 0.0:
        ring_loop = _reverse_loop(ring_loop)
    geometry.remove_face(plane_face, record=False)
    annulus = geometry.add_face_from_loop(outer_loop, corners, surface=surface)
    geometry.faces[annulus].holes = (_reverse_loop(ring_loop),)
    disk = geometry.add_face_from_loop(ring_loop, surface=surface)
    for identifier in (annulus, disk):
        geometry.faces[identifier].metadata.update(metadata)
        geometry.tag(EntityRef("face", identifier), *tags)
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

    snapshot = geometry.topology_snapshot()
    try:
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
    except Exception:
        geometry.restore_topology(snapshot)
        raise
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
) -> FaceIntersection | tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Imprint shared topology for qualified structural-surface crossings.

    A transverse plane/cylinder cut is applied atomically to one complete,
    conformal cylinder band.  It creates an exact arc loop shared by the two
    plane regions and by both axial fragments of every cylinder patch.
    """

    if first_face == second_face:
        raise GeometryError("two distinct faces are required")
    first_surface = geometry.faces[first_face].surface
    second_surface = geometry.faces[second_face].surface
    if isinstance(first_surface, Plane) and isinstance(second_surface, Cylinder):
        alignment = abs(float(first_surface.normal @ second_surface.axis))
        if abs(alignment - 1.0) <= 1.0e-10:
            return _imprint_transverse_plane_cylinder(
                geometry, first_face, second_face, fragment=fragment
            )
        endpoints = _axial_plane_cylinder_segment(
            geometry, first_face, second_face
        )
        return _imprint_segment(geometry, first_face, second_face, endpoints) if fragment else endpoints
    if isinstance(first_surface, Cylinder) and isinstance(second_surface, Plane):
        alignment = abs(float(second_surface.normal @ first_surface.axis))
        if abs(alignment - 1.0) <= 1.0e-10:
            made = _imprint_transverse_plane_cylinder(
                geometry, second_face, first_face, fragment=fragment
            )
            if not isinstance(made, FaceIntersection):
                return made
            return FaceIntersection(
                made.edge,
                made.second_faces,
                made.first_faces,
                made.edges,
            )
        endpoints = _axial_plane_cylinder_segment(
            geometry, second_face, first_face
        )
        if not fragment:
            return endpoints
        made = _imprint_segment(geometry, first_face, second_face, endpoints)
        return made
    first_plane, second_plane = _face_plane(geometry, first_face), _face_plane(geometry, second_face)
    line = plane_plane(first_plane, second_plane)
    if line is None:
        raise GeometryError("faces are parallel or coplanar; no unique intersection line")
    point, direction = line
    first_interval = _line_face_interval(geometry, first_face, point, direction)
    second_interval = _line_face_interval(geometry, second_face, point, direction)
    if first_interval is None or second_interval is None:
        raise GeometryError("intersection line does not cross both face boundaries")
    lower, upper = max(first_interval[0],second_interval[0]), min(first_interval[1],second_interval[1])
    if upper-lower <= 1e-9:
        raise GeometryError("faces do not overlap along an intersection segment")
    endpoints = (point+lower*direction, point+upper*direction)
    if not fragment:
        return endpoints
    return _imprint_segment(geometry, first_face, second_face, endpoints)
