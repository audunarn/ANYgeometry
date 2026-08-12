"""General topology and geometric operations.

Nothing in this module exists solely to create mapped quadrilateral regions;
those decomposition policies belong to :mod:`anymesher`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .curves import Arc, Spline
from .entities import EntityRef, Face, OrientedEdge
from .errors import GeometryError
from .model import GeometryModel
from .surfaces import CoonsSurface, Cone, Cylinder, Plane, RuledSurface

__all__ = [
    "closest_point",
    "fragment_face",
    "project",
    "punch_hole",
    "split_face",
    "split_face_at",
    "split_face_between",
    "strip_face",
    "surface_point",
    "transform",
    "trim_face",
]


def surface_point(
    geometry: GeometryModel, face: Face | int, u: float, v: float
) -> np.ndarray:
    """Evaluate the authoritative surface carried by a face."""

    face_id = face if isinstance(face, int) else face.id
    return geometry.face_point(face_id, float(u), float(v))


def closest_point(
    geometry: GeometryModel,
    point: Sequence[float],
    references: Iterable[EntityRef] | None = None,
) -> Tuple[EntityRef, np.ndarray, float]:
    """Return the closest point on selected geometry entities."""

    target = np.asarray(point, dtype=float)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise GeometryError("point must be a finite 3-vector")
    refs = list(references) if references is not None else [
        *(vertex.ref for vertex in geometry.vertices.values()),
        *(edge.ref for edge in geometry.edges.values()),
        *(face.ref for face in geometry.faces.values()),
    ]
    if not refs:
        raise GeometryError("closest_point needs at least one entity")
    candidates: List[Tuple[float, str, int, EntityRef, np.ndarray]] = []
    for reference in refs:
        if reference.kind == "vertex":
            made = geometry.vertex_position(reference.id).copy()
        elif reference.kind == "edge":
            made = _closest_on_edge(geometry, reference.id, target)
        else:
            made, _uv, _distance = geometry.project_to_face(reference.id, target)
        distance = float(np.linalg.norm(made - target))
        candidates.append((distance, reference.kind, reference.id, reference, made))
    distance, _kind, _id, reference, made = min(
        candidates, key=lambda item: (item[0], item[1], item[2])
    )
    return reference, made, distance


def project(
    geometry: GeometryModel, face: int | EntityRef, point: Sequence[float]
) -> Tuple[np.ndarray, Tuple[float, float], float]:
    """Project a point onto a face."""

    face_id = face.id if isinstance(face, EntityRef) else int(face)
    if isinstance(face, EntityRef) and face.kind != "face":
        raise GeometryError("projection target must be a face")
    return geometry.project_to_face(face_id, point)


def _closest_on_edge(
    geometry: GeometryModel, edge_id: int, target: np.ndarray
) -> np.ndarray:
    made, _parameter, _distance = geometry.closest_edge_point(edge_id, target)
    return made


def _split_face_between_impl(
    geometry: GeometryModel,
    face_id: int,
    start_vertex: int,
    end_vertex: int,
    *,
    tolerance: float = 1.0e-8,
) -> Tuple[int, Tuple[int, int]]:
    """Fragment a face between any two non-adjacent boundary vertices.

    The new edge is shared by both replacement faces.  Straight and circular
    cuts are retained exactly; a lightweight spline represents a more general
    supported surface curve.
    """

    face = geometry.faces.get(int(face_id))
    if face is None:
        raise GeometryError(f"no face {face_id}")
    first_chain, second_chain = _split_loop(face, start_vertex, end_vertex, geometry)
    if len(first_chain) == 1 or len(second_chain) == 1:
        raise GeometryError("a cut cannot duplicate an existing boundary edge")
    first_holes, second_holes = _partition_face_holes(
        geometry,
        face,
        first_chain,
        second_chain,
        start_vertex,
        end_vertex,
    )
    dividing_edge = _surface_divider(
        geometry, face, start_vertex, end_vertex, tolerance=tolerance
    )
    surface = face.surface
    metadata = dict(face.metadata)
    groups = [name for name, members in geometry.groups.items() if face.ref in members]
    tags = geometry.tags_for(face.ref)
    geometry.remove_face(face.id, record=False)
    first_loop = tuple(first_chain) + (OrientedEdge(dividing_edge, False),)
    second_loop = tuple(second_chain) + (OrientedEdge(dividing_edge, True),)
    first_corners = geometry._detect_corners(first_loop) if len(first_loop) >= 4 else None  # noqa: SLF001
    second_corners = geometry._detect_corners(second_loop) if len(second_loop) >= 4 else None  # noqa: SLF001
    first = geometry.add_face_from_loop(first_loop, first_corners, surface=surface)
    second = geometry.add_face_from_loop(second_loop, second_corners, surface=surface)
    for made, holes in ((first, first_holes), (second, second_holes)):
        # A split changes trims, not the authoritative support.  Retaining the
        # verified parent surface is particularly important for cylindrical,
        # conical, and ruled faces; a Coons patch is only a mapping and must
        # not silently replace that support.
        geometry._put_entity(  # noqa: SLF001
            "face",
            replace(
                geometry.faces[made],
                holes=holes,
                metadata=dict(metadata),
                surface=surface,
            ),
        )
        geometry.tag(EntityRef("face", made), *tags)
    geometry.record_replacement(
        EntityRef("face", face.id),
        (EntityRef("face", first), EntityRef("face", second)),
    )
    errors = geometry.validate_topology()
    if errors:
        raise GeometryError(
            "face split produced invalid topology: " + "; ".join(errors)
        )
    return dividing_edge, (first, second)


def split_face_between(
    geometry: GeometryModel,
    face_id: int,
    start_vertex: int,
    end_vertex: int,
    *,
    tolerance: float = 1.0e-8,
) -> Tuple[int, Tuple[int, int]]:
    """Atomically fragment a face between boundary vertices."""

    with geometry.transaction():
        return _split_face_between_impl(
            geometry,
            face_id,
            start_vertex,
            end_vertex,
            tolerance=tolerance,
        )


split_face = split_face_between


def _split_loop(
    face: Face, start_vertex: int, end_vertex: int, geometry: GeometryModel
) -> Tuple[List[OrientedEdge], List[OrientedEdge]]:
    positions = {
        geometry.oriented_start_vertex(item): index
        for index, item in enumerate(face.loop)
    }
    if start_vertex not in positions or end_vertex not in positions:
        raise GeometryError("both ends of a cut must be boundary vertices")
    if start_vertex == end_vertex:
        raise GeometryError("a cut needs two distinct boundary vertices")
    start, end, count = positions[start_vertex], positions[end_vertex], len(face.loop)
    first = [face.loop[(start + step) % count] for step in range((end - start) % count)]
    second = [face.loop[(end + step) % count] for step in range((start - end) % count)]
    return first, second


def _sample_oriented_chain(
    geometry: GeometryModel, chain: Sequence[OrientedEdge], *, samples: int = 33
) -> np.ndarray:
    points: List[np.ndarray] = []
    for item in chain:
        edge = geometry.edges[item.edge]
        count = 2 if not isinstance(edge.curve, (Arc, Spline)) else samples
        made = geometry.sample_edge(item.edge, np.linspace(0.0, 1.0, count))
        if not item.forward:
            made = made[::-1]
        points.extend(made[:-1])
    if chain:
        points.append(
            geometry.vertex_position(geometry.oriented_end_vertex(chain[-1]))
        )
    return np.asarray(points, dtype=float)


def _partition_face_holes(
    geometry: GeometryModel,
    face: Face,
    first_chain: Sequence[OrientedEdge],
    second_chain: Sequence[OrientedEdge],
    start_vertex: int,
    end_vertex: int,
) -> Tuple[Tuple[Tuple[OrientedEdge, ...], ...], Tuple[Tuple[OrientedEdge, ...], ...]]:
    """Assign intact trims to one child, rejecting cuts that touch a trim."""

    if not face.holes:
        return (), ()
    try:
        first_polygon = np.asarray(
            [
                geometry.face_local_uv(face.id, point)
                for point in _sample_oriented_chain(geometry, first_chain)
            ],
            dtype=float,
        )
        second_polygon = np.asarray(
            [
                geometry.face_local_uv(face.id, point)
                for point in _sample_oriented_chain(geometry, second_chain)
            ],
            dtype=float,
        )
        start_uv = np.asarray(
            geometry.face_local_uv(face.id, geometry.vertex_position(start_vertex))
        )
        end_uv = np.asarray(
            geometry.face_local_uv(face.id, geometry.vertex_position(end_vertex))
        )
    except (ValueError, GeometryError, np.linalg.LinAlgError) as error:
        raise GeometryError(
            f"cannot classify face holes for this cut: {error}"
        ) from error
    divider = end_uv - start_uv
    divider_length = float(np.linalg.norm(divider))
    if divider_length <= 1.0e-14:
        raise GeometryError("face cut collapses in local surface coordinates")
    scale = max(
        float(np.ptp(np.vstack((first_polygon, second_polygon)), axis=0).max()),
        1.0,
    )
    tolerance = 1.0e-9 * scale
    assigned: List[List[Tuple[OrientedEdge, ...]]] = [[], []]
    for index, hole in enumerate(face.holes, start=1):
        points = _sample_oriented_chain(geometry, hole, samples=65)[:-1]
        hole_uv = np.asarray(
            [geometry.face_local_uv(face.id, point) for point in points],
            dtype=float,
        )
        signed = (
            divider[0] * (hole_uv[:, 1] - start_uv[1])
            - divider[1] * (hole_uv[:, 0] - start_uv[0])
        )
        if (
            np.any(np.abs(signed) <= tolerance * divider_length)
            or (float(signed.min()) < 0.0 < float(signed.max()))
        ):
            raise GeometryError(
                f"face cut intersects or touches hole {index}; split the trim first"
            )
        owners = [
            all(
                geometry._point_in_polygon(  # noqa: SLF001
                    point, polygon, include_boundary=False
                )
                for point in hole_uv
            )
            for polygon in (first_polygon, second_polygon)
        ]
        if owners.count(True) != 1:
            raise GeometryError(
                f"face cut cannot assign hole {index} unambiguously"
            )
        assigned[owners.index(True)].append(hole)
    return tuple(assigned[0]), tuple(assigned[1])


def _surface_divider(
    geometry: GeometryModel,
    face: Face,
    start_vertex: int,
    end_vertex: int,
    *,
    tolerance: float,
) -> int:
    start = geometry.vertex_position(start_vertex)
    end = geometry.vertex_position(end_vertex)
    if isinstance(face.surface, Plane) or face.surface is None:
        return geometry.add_line(start_vertex, end_vertex)
    start_uv = geometry.face_local_uv(face.id, start)
    end_uv = geometry.face_local_uv(face.id, end)
    parameters = np.linspace(0.0, 1.0, 17)
    samples = np.asarray(
        [
            geometry.face_point(
                face.id,
                (1.0 - step) * start_uv[0] + step * end_uv[0],
                (1.0 - step) * start_uv[1] + step * end_uv[1],
            )
            for step in parameters
        ]
    )
    straight = start + parameters[:, None] * (end - start)
    scale = max(float(np.linalg.norm(end - start)), 1.0)
    if float(np.linalg.norm(samples - straight, axis=1).max()) <= tolerance * scale:
        return geometry.add_line(start_vertex, end_vertex)
    via = geometry.add_point(*samples[len(samples) // 2])
    try:
        arc = geometry.add_arc(start_vertex, via, end_vertex)
    except (GeometryError, ValueError):
        geometry.remove_vertex(via, record=False)
    else:
        fitted = geometry.sample_edge(arc, parameters)
        if float(np.linalg.norm(samples - fitted, axis=1).max()) <= tolerance * scale:
            return arc
        geometry.remove_edge(arc, record=False)
        geometry.remove_vertex(via, record=False)
    control_ids = [geometry.add_point(*sample) for sample in samples[1:-1:4]]
    return geometry.add_spline(start_vertex, control_ids, end_vertex)


def _split_face_at_impl(
    geometry: GeometryModel,
    face_id: int,
    axis: int,
    fraction: float,
    *,
    tolerance: float = 1.0e-8,
) -> Tuple[int, Tuple[int, int]]:
    """Split a four-side face across a local coordinate fraction."""

    if axis not in (0, 1):
        raise GeometryError("axis must be 0 or 1")
    if not 0.0 < float(fraction) < 1.0:
        raise GeometryError("fraction must be strictly between 0 and 1")
    face = geometry.faces.get(face_id)
    if face is None:
        raise GeometryError(f"no face {face_id}")
    if len(face.corners) != 4:
        raise GeometryError("split_face_at requires a four-side parameterization")
    start = _split_side_at(geometry, face_id, axis, float(fraction))
    end = _split_side_at(geometry, face_id, axis + 2, 1.0 - float(fraction))
    return split_face_between(
        geometry, face_id, start, end, tolerance=tolerance
    )


def split_face_at(
    geometry: GeometryModel,
    face_id: int,
    axis: int,
    fraction: float,
    *,
    tolerance: float = 1.0e-8,
) -> Tuple[int, Tuple[int, int]]:
    """Atomically split a four-side face across a local coordinate fraction."""

    with geometry.transaction():
        return _split_face_at_impl(
            geometry, face_id, axis, fraction, tolerance=tolerance
        )


def _split_side_at(
    geometry: GeometryModel, face_id: int, side_index: int, fraction: float
) -> int:
    side = geometry.faces[face_id].sides()[side_index]
    lengths = np.asarray([geometry.edge_length(item.edge) for item in side])
    breaks = np.concatenate(([0.0], np.cumsum(lengths) / lengths.sum()))
    segment = min(
        max(int(np.searchsorted(breaks, fraction, side="right") - 1), 0),
        len(side) - 1,
    )
    local = (fraction - breaks[segment]) / (breaks[segment + 1] - breaks[segment])
    item = side[segment]
    if local <= 1.0e-10:
        return geometry.oriented_start_vertex(item)
    if local >= 1.0 - 1.0e-10:
        return geometry.oriented_end_vertex(item)
    parameter = local if item.forward else 1.0 - local
    vertex, _halves = geometry.split_edge(item.edge, parameter)
    return vertex


def _strip_face_impl(
    geometry: GeometryModel, face_id: int, axis: int, count: int
) -> Tuple[List[int], List[int]]:
    """Fragment a four-side structural face into conforming strips."""

    if isinstance(count, bool) or int(count) != count or int(count) < 2:
        raise GeometryError("a strip count needs to be at least 2")
    if axis not in (0, 1):
        raise GeometryError("axis must be 0 or 1")
    if face_id not in geometry.faces:
        raise GeometryError(f"no face {face_id}")
    cuts = []
    for fraction in (index / int(count) for index in range(1, int(count))):
        start = _split_side_at(geometry, face_id, axis, fraction)
        end = _split_side_at(geometry, face_id, axis + 2, 1.0 - fraction)
        cuts.append((start, end))
    strips: List[int] = []
    dividers: List[int] = []
    remainder = face_id
    for index, (start, end) in enumerate(cuts):
        divider, (first, second) = split_face_between(
            geometry, remainder, start, end
        )
        dividers.append(divider)
        if index + 1 < len(cuts):
            next_cut = cuts[index + 1]
            following = _face_holding(geometry, (first, second), *next_cut)
        else:
            following = second
        strips.append(first if following == second else second)
        remainder = following
    strips.append(remainder)
    return strips, dividers


def strip_face(
    geometry: GeometryModel, face_id: int, axis: int, count: int
) -> Tuple[List[int], List[int]]:
    """Atomically fragment a four-side structural face into strips."""

    with geometry.transaction():
        return _strip_face_impl(geometry, face_id, axis, count)


def _face_holding(
    geometry: GeometryModel,
    candidates: Sequence[int],
    start_vertex: int,
    end_vertex: int,
) -> int:
    for face_id in candidates:
        vertices = {
            vertex
            for item in geometry.faces[face_id].loop
            for vertex in (
                geometry.oriented_start_vertex(item),
                geometry.oriented_end_vertex(item),
            )
        }
        if {start_vertex, end_vertex} <= vertices:
            return face_id
    raise GeometryError("no face fragment contains the next strip cut")


def _trim_face_impl(
    geometry: GeometryModel,
    face_id: int,
    inner_loops: Iterable[Sequence[OrientedEdge]],
) -> int:
    """Attach checked inner trim loops without replacing face identity."""

    face = geometry.faces.get(face_id)
    if face is None:
        raise GeometryError(f"no face {face_id}")
    loops = []
    for supplied in inner_loops:
        loop = tuple(supplied)
        if len(loop) < 2:
            raise GeometryError("an inner trim loop needs at least two edges")
        for item in loop:
            if item.edge not in geometry.edges:
                raise GeometryError(f"trim references missing edge {item.edge}")
        for current, following in zip(loop, loop[1:] + loop[:1]):
            if geometry.oriented_end_vertex(current) != geometry.oriented_start_vertex(following):
                raise GeometryError("inner trim loop is not continuous")
        loops.append(loop)
    geometry._put_entity(  # noqa: SLF001
        "face", replace(face, holes=tuple(loops))
    )
    errors = geometry.validate_topology()
    if errors:
        raise GeometryError("invalid face trim: " + "; ".join(errors))
    return face_id


def trim_face(
    geometry: GeometryModel,
    face_id: int,
    inner_loops: Iterable[Sequence[OrientedEdge]],
) -> int:
    """Atomically attach checked inner trim loops."""

    with geometry.transaction():
        return _trim_face_impl(geometry, face_id, inner_loops)


def _punch_hole_impl(
    geometry: GeometryModel,
    face_id: int,
    centre: Sequence[float],
    radius: float,
) -> Tuple[int, Tuple[int, ...]]:
    """Create a circular trim loop; no meshing decomposition is implied."""

    face = geometry.faces.get(face_id)
    if face is None:
        raise GeometryError(f"no face {face_id}")
    radius = float(radius)
    if not np.isfinite(radius) or radius <= 0.0:
        raise GeometryError("hole radius must be finite and positive")
    centre_point = np.asarray(centre, dtype=float)
    if centre_point.shape != (3,) or not np.all(np.isfinite(centre_point)):
        raise GeometryError("hole centre must be a finite 3-vector")
    _projected, uv, distance = geometry.project_to_face(face_id, centre_point)
    if distance > 1.0e-7 * max(radius, 1.0):
        raise GeometryError("hole centre does not lie on the face")
    normal = geometry.face_normal(face_id, *uv)
    planarity_samples = np.asarray(
        [geometry.face_point(face_id, u, v) for u in (0.0, 0.5, 1.0) for v in (0.0, 0.5, 1.0)]
    )
    offsets = planarity_samples - planarity_samples.mean(axis=0)
    _left, singular, _right = np.linalg.svd(offsets, full_matrices=False)
    if singular[-1] > 1.0e-7 * max(singular[0], 1.0):
        raise GeometryError(
            "punch_hole currently requires a planar structural face; "
            "use trim_face with an explicit curved trim loop otherwise"
        )
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(reference @ normal)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    basis_u = reference - float(reference @ normal) * normal
    basis_u /= float(np.linalg.norm(basis_u))
    basis_v = np.cross(normal, basis_u)
    ring_positions = [
        centre_point + radius * (np.cos(angle) * basis_u + np.sin(angle) * basis_v)
        for angle in np.linspace(0.0, 2.0 * np.pi, 5)[:-1]
    ]
    for position in ring_positions:
        _on_face, _local, boundary_distance = geometry.project_to_face(face_id, position)
        if boundary_distance > 1.0e-7 * max(radius, 1.0):
            raise GeometryError("hole does not fit inside the face parameter domain")
    ring = [geometry.add_point(*position) for position in ring_positions]
    arcs = []
    for index in range(4):
        angle = (index + 0.5) * 0.5 * np.pi
        via = geometry.add_point(
            *(centre_point + radius * (np.cos(angle) * basis_u + np.sin(angle) * basis_v))
        )
        arcs.append(geometry.add_arc(ring[index], via, ring[(index + 1) % 4]))
    loop = tuple(OrientedEdge(edge, True) for edge in arcs)
    geometry._put_entity(  # noqa: SLF001
        "face", replace(face, holes=face.holes + (loop,))
    )
    errors = geometry.validate_topology()
    if errors:
        raise GeometryError("invalid punched hole: " + "; ".join(errors))
    return face_id, tuple(arcs)


def punch_hole(
    geometry: GeometryModel,
    face_id: int,
    centre: Sequence[float],
    radius: float,
) -> Tuple[int, Tuple[int, ...]]:
    """Atomically create a circular trim loop."""

    with geometry.transaction():
        return _punch_hole_impl(geometry, face_id, centre, radius)


def _fragment_face_impl(
    geometry: GeometryModel,
    face_id: int,
    cuts: Iterable[Tuple[int, int]],
) -> Tuple[int, ...]:
    """Apply boundary-vertex cuts and return all current descendants."""

    current = [face_id]
    for start, end in cuts:
        target = next(
            (
                candidate
                for candidate in current
                if {start, end}
                <= {
                    geometry.oriented_start_vertex(item)
                    for item in geometry.faces[candidate].loop
                }
            ),
            None,
        )
        if target is None:
            raise GeometryError("no current fragment contains both cut vertices")
        _edge, made = split_face_between(geometry, target, start, end)
        current.remove(target)
        current.extend(made)
    return tuple(sorted(current))


def fragment_face(
    geometry: GeometryModel,
    face_id: int,
    cuts: Iterable[Tuple[int, int]],
) -> Tuple[int, ...]:
    """Atomically apply a sequence of boundary-vertex cuts."""

    with geometry.transaction():
        return _fragment_face_impl(geometry, face_id, cuts)


def _transform_impl(
    geometry: GeometryModel,
    matrix: Sequence[Sequence[float]],
    references: Iterable[EntityRef] | None = None,
) -> Tuple[EntityRef, ...]:
    """Apply a finite homogeneous transform while preserving entity IDs."""

    transform_matrix = np.asarray(matrix, dtype=float)
    if transform_matrix.shape != (4, 4) or not np.all(np.isfinite(transform_matrix)):
        raise GeometryError("transform must be a finite 4x4 matrix")
    if not np.allclose(transform_matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-14):
        raise GeometryError("only affine homogeneous transforms are supported")
    linear = transform_matrix[:3, :3]
    singular_values = np.linalg.svd(linear, compute_uv=False)
    scale = float(singular_values.max())
    if scale <= 0.0 or float(singular_values.min()) <= 1.0e-12 * scale:
        raise GeometryError("singular affine transforms are not supported")
    refs = list(references) if references is not None else [
        vertex.ref for vertex in geometry.vertices.values()
    ]
    vertex_ids = set()
    for reference in refs:
        if reference.kind == "vertex":
            vertex_ids.add(reference.id)
        elif reference.kind == "edge":
            edge = geometry.edges[reference.id]
            vertex_ids.update((edge.start, edge.end))
            if isinstance(edge.curve, Arc):
                vertex_ids.add(edge.curve.via_vertex)
            elif isinstance(edge.curve, Spline):
                vertex_ids.update(edge.curve.control_vertices)
        else:
            face = geometry.faces[reference.id]
            for loop in (face.loop,) + face.holes:
                for item in loop:
                    edge = geometry.edges[item.edge]
                    vertex_ids.update((edge.start, edge.end))
                    if isinstance(edge.curve, Arc):
                        vertex_ids.add(edge.curve.via_vertex)
                    elif isinstance(edge.curve, Spline):
                        vertex_ids.update(edge.curve.control_vertices)
    affected_faces = {
        face.id
        for face in geometry.faces.values()
        if any(
            geometry.oriented_start_vertex(item) in vertex_ids
            or geometry.oriented_end_vertex(item) in vertex_ids
            for loop in (face.loop,) + face.holes
            for item in loop
        )
    }
    complete_faces = {
        face.id
        for face in geometry.faces.values()
        if {
            vertex
            for loop in (face.loop,) + face.holes
            for item in loop
            for vertex in (
                geometry.oriented_start_vertex(item),
                geometry.oriented_end_vertex(item),
            )
        } <= vertex_ids
    }
    affected_edges = {
        edge.id
        for edge in geometry.edges.values()
        if {
            edge.start,
            edge.end,
            *(
                (edge.curve.via_vertex,)
                if isinstance(edge.curve, Arc)
                else edge.curve.control_vertices
                if isinstance(edge.curve, Spline)
                else ()
            ),
        }
        & vertex_ids
    }
    is_uniform = np.allclose(
        singular_values,
        singular_values[0],
        rtol=1.0e-10,
        atol=1.0e-12 * max(float(singular_values[0]), 1.0),
    )
    has_circular_geometry = any(
        isinstance(geometry.edges[edge_id].curve, Arc)
        for edge_id in affected_edges
    ) or any(
        isinstance(geometry.faces[face_id].surface, (Cylinder, Cone))
        for face_id in affected_faces
    )
    if not is_uniform and has_circular_geometry:
        raise GeometryError(
            "anisotropic scale or shear is not supported for arcs, "
            "cylinders, or cones"
        )
    transformed_surfaces = {
        face_id: _transform_surface(geometry.faces[face_id].surface, transform_matrix)
        for face_id in complete_faces
    }
    # Dependency bounds must be captured before any defining vertex moves.
    # The immutable edge records themselves do not change, and faces are
    # replaced only after their boundary has moved, so relying on the later
    # owner writes would leave an already-materialized spatial index at the
    # old edge/face bounds.
    journal = geometry._transaction_journal  # noqa: SLF001
    assert journal is not None
    for face_id in sorted(affected_faces):
        geometry._capture_entity("face", face_id)  # noqa: SLF001
        journal.spatial_updates.add(("face", face_id))
    for edge_id in sorted(affected_edges):
        edge_key = ("edge", edge_id)
        journal.bounds_before.setdefault(  # noqa: SLF001
            edge_key, geometry._entity_bounds(edge_key)  # noqa: SLF001
        )
        journal.spatial_updates.add(edge_key)
    for vertex_id in sorted(vertex_ids):
        point = np.append(geometry.vertex_position(vertex_id), 1.0)
        made = transform_matrix @ point
        if abs(float(made[3])) <= 1.0e-14:
            raise GeometryError("transform maps a point to infinity")
        geometry._put_entity(  # noqa: SLF001
            "vertex",
            replace(geometry.vertices[vertex_id], position=made[:3] / made[3]),
        )
    # Explicit surfaces are invalidated when their defining topology moves;
    # boundary-backed Coons evaluation remains authoritative and exact for
    # affine transforms of the supported structural patches.
    for face_id in affected_faces:
        if face_id in transformed_surfaces:
            geometry._put_entity(  # noqa: SLF001
                "face",
                replace(
                    geometry.faces[face_id],
                    surface=transformed_surfaces[face_id],
                ),
            )
        else:
            geometry._put_entity(  # noqa: SLF001
                "face",
                replace(
                    geometry.faces[face_id],
                    surface=(
                        CoonsSurface()
                        if len(geometry.faces[face_id].corners) == 4
                        else None
                    ),
                ),
            )
    return tuple(EntityRef("vertex", item) for item in sorted(vertex_ids))


def transform(
    geometry: GeometryModel,
    matrix: Sequence[Sequence[float]],
    references: Iterable[EntityRef] | None = None,
) -> Tuple[EntityRef, ...]:
    """Atomically apply a finite homogeneous transform."""

    with geometry.transaction():
        return _transform_impl(geometry, matrix, references)


def _transform_surface(surface: object, matrix: np.ndarray) -> object:
    if surface is None:
        return surface
    linear, translation = matrix[:3, :3], matrix[:3, 3]

    def point(value: np.ndarray) -> np.ndarray:
        return linear @ value + translation

    if isinstance(surface, CoonsSurface):
        if not surface.has_boundaries:
            return surface
        assert surface.bottom is not None and surface.right is not None and surface.top is not None and surface.left is not None
        mapped = [array @ linear.T + translation for array in (surface.bottom, surface.right, surface.top, surface.left)]
        return CoonsSurface(*mapped)

    if isinstance(surface, Plane):
        return Plane(point(surface.origin), linear @ surface.u_vector, linear @ surface.v_vector)
    if isinstance(surface, RuledSurface):
        first = surface.first_boundary @ linear.T + translation
        second = surface.second_boundary @ linear.T + translation
        return RuledSurface(first, second)
    if isinstance(surface, (Cylinder, Cone)):
        singular = np.linalg.svd(linear, compute_uv=False)
        if not np.allclose(singular, singular[0], rtol=1e-10, atol=1e-12):
            raise GeometryError(
                "non-uniform transforms of cylinders/cones are not supported"
            )
        scale = float(singular[0])
        if scale <= 0.0:
            raise GeometryError("singular transforms of cylinders/cones are not supported")
        # Rebase the angular coordinate on the transformed u=0 direction.
        # Under a reflection cross(axis, radial) changes handedness, so the
        # signed sweep must change with it to preserve every (u, v) point.
        start_radial = (
            np.cos(surface.start_angle) * surface.radial_direction
            + np.sin(surface.start_angle) * surface.circumferential_direction
        )
        handedness = -1.0 if float(np.linalg.det(linear)) < 0.0 else 1.0
        common = dict(
            origin=point(surface.origin),
            axis=linear @ surface.axis,
            radial_direction=linear @ start_radial,
            height=surface.height * scale,
            start_angle=0.0,
            sweep_angle=surface.sweep_angle * handedness,
        )
        if isinstance(surface, Cylinder):
            return Cylinder(radius=surface.radius * scale, **common)
        return Cone(
            radius_start=surface.radius_start * scale,
            radius_end=surface.radius_end * scale,
            **common,
        )
    raise GeometryError(f"unsupported surface transform {type(surface).__name__}")
