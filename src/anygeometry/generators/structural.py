"""Neutral structural-surface generators with semantic geometry groups."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..entities import EntityRef, OrientedEdge
from ..errors import GeometryError
from ..model import GeometryModel
from ..surfaces import Cone, Cylinder, Plane
from .layout import centered_member_positions, cleanup_axis, closed_loop_member_count


def _real_scalar(value: object, name: str) -> float:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GeometryError(f"{name} must be a finite real numeric scalar") from exc
    if raw.shape != () or raw.dtype.kind in {"b", "c"}:
        raise GeometryError(f"{name} must be a finite real numeric scalar")
    item = raw.item()
    if isinstance(item, (bool, np.bool_, complex, np.complexfloating)):
        raise GeometryError(f"{name} must be a finite real numeric scalar")
    try:
        made = float(item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GeometryError(f"{name} must be a finite real numeric scalar") from exc
    if not np.isfinite(made):
        raise GeometryError(f"{name} must be a finite real numeric scalar")
    return made


def _positive(value: float, name: str, *, zero: bool = False) -> float:
    made = _real_scalar(value, name)
    if not np.isfinite(made) or (made < 0.0 if zero else made <= 0.0):
        raise GeometryError(f"{name} must be finite and {'non-negative' if zero else 'positive'}")
    return made


def _vector3(value: Sequence[float], name: str) -> np.ndarray:
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise GeometryError(f"{name} must be a finite numeric 3-vector") from exc
    if raw.shape != (3,) or any(
        isinstance(component, (bool, np.bool_, complex, np.complexfloating))
        for component in raw.flat
    ):
        raise GeometryError(
            f"{name} must be a finite real non-boolean numeric 3-vector"
        )
    try:
        made = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GeometryError(f"{name} must be a finite numeric 3-vector") from exc
    if not np.all(np.isfinite(made)):
        raise GeometryError(f"{name} must be a finite numeric 3-vector")
    return made


def _unit(value: Sequence[float], name: str) -> np.ndarray:
    made = _vector3(value, name)
    length = float(np.linalg.norm(made))
    if length <= 0.0:
        raise GeometryError(f"{name} must be a finite non-zero 3-vector")
    return made / length


def _integer_at_least(value: object, minimum: int, name: str) -> int:
    if isinstance(value, (str, bytes)):
        raise GeometryError(f"{name} must be an integer of at least {minimum}")
    try:
        numeric = _real_scalar(value, name)
    except GeometryError as exc:
        raise GeometryError(f"{name} must be an integer of at least {minimum}") from exc
    if (
        not np.isfinite(numeric)
        or not numeric.is_integer()
        or numeric < minimum
    ):
        raise GeometryError(f"{name} must be an integer of at least {minimum}")
    return int(numeric)


def _planar_grid_impl(
    x_values: Sequence[float], y_values: Sequence[float], *, origin: Sequence[float],
    u_direction: Sequence[float], v_direction: Sequence[float], semantic_group: str,
    geometry: GeometryModel,
) -> GeometryModel:
    base = _vector3(origin, "origin")
    u_axis, v_axis = _unit(u_direction, "u_direction"), _unit(v_direction, "v_direction")
    if float(np.linalg.norm(np.cross(u_axis, v_axis))) <= 1.0e-12:
        raise GeometryError("plate directions must be independent")
    if not isinstance(semantic_group, str) or not semantic_group.strip():
        raise GeometryError("semantic_group must be a non-empty string")
    vertices = {(i, j): geometry.add_point(*(base + x*u_axis + y*v_axis)) for j, y in enumerate(y_values) for i, x in enumerate(x_values)}
    vertex_indices = {vertex: index for index, vertex in vertices.items()}
    edge_cache: dict[tuple[int, int], int] = {}

    def oriented(first: int, second: int) -> OrientedEdge:
        key = tuple(sorted((first, second)))
        if key not in edge_cache:
            edge_cache[key] = geometry.add_line(first, second)
        edge = geometry.edges[edge_cache[key]]
        return OrientedEdge(edge.id, edge.start == first)

    faces: list[EntityRef] = []
    for j in range(len(y_values) - 1):
        for i in range(len(x_values) - 1):
            ring = [vertices[i, j], vertices[i+1, j], vertices[i+1, j+1], vertices[i, j+1]]
            loop = tuple(oriented(a, b) for a, b in zip(ring, ring[1:] + ring[:1]))
            face_id = geometry.add_face_from_loop(
                loop,
                (0, 1, 2, 3),
                surface=Plane(
                    geometry.vertex_position(ring[0]),
                    geometry.vertex_position(ring[1])
                    - geometry.vertex_position(ring[0]),
                    geometry.vertex_position(ring[3])
                    - geometry.vertex_position(ring[0]),
                ),
            )
            faces.append(EntityRef("face", face_id))
    part = geometry.add_part(name=semantic_group)
    geometry.add_sheet(
        tuple(reference.id for reference in faces),
        part_id=part,
        name=semantic_group,
    )
    geometry.add_to_group("shell", faces)
    geometry.add_to_group("plate", faces)
    if semantic_group not in ("shell", "plate"):
        geometry.add_to_group(semantic_group, faces)
    boundaries = [edge.ref for edge in geometry.edges.values() if len(geometry.faces_using_edge(edge.id)) == 1]
    geometry.add_to_group("boundaries", boundaries)
    transverse: list[EntityRef] = []
    longitudinal: list[EntityRef] = []
    for edge in geometry.edges.values():
        (i0, j0), (i1, j1) = vertex_indices[edge.start], vertex_indices[edge.end]
        if i0 == i1 and 0 < i0 < len(x_values) - 1:
            transverse.append(edge.ref)
        if j0 == j1 and 0 < j0 < len(y_values) - 1:
            longitudinal.append(edge.ref)
    geometry.add_to_group("longitudinal_stiffeners", longitudinal)
    geometry.add_to_group("transverse_stiffeners", transverse)
    member_chains: list[tuple[OrientedEdge, ...]] = []
    member_names: list[str] = []
    for j in range(1, len(y_values) - 1):
        axis = []
        for i in range(len(x_values) - 1):
            first, second = vertices[i, j], vertices[i + 1, j]
            edge = geometry.edges[edge_cache[tuple(sorted((first, second)))]]
            axis.append(OrientedEdge(edge.id, edge.start == first))
        member_chains.append(tuple(axis))
        member_names.append(f"longitudinal_stiffener_{j}")
    for i in range(1, len(x_values) - 1):
        axis = []
        for j in range(len(y_values) - 1):
            first, second = vertices[i, j], vertices[i, j + 1]
            edge = geometry.edges[edge_cache[tuple(sorted((first, second)))]]
            axis.append(OrientedEdge(edge.id, edge.start == first))
        member_chains.append(tuple(axis))
        member_names.append(f"transverse_stiffener_{i}")
    geometry.add_members(member_chains, part_id=part, names=member_names)
    return geometry


def _planar_grid(
    x_values: Sequence[float], y_values: Sequence[float], *, origin: Sequence[float],
    u_direction: Sequence[float], v_direction: Sequence[float], semantic_group: str,
) -> GeometryModel:
    """Build one compound panel as a single validated transaction."""

    geometry = GeometryModel()
    with geometry.transaction():
        return _planar_grid_impl(
            x_values,
            y_values,
            origin=origin,
            u_direction=u_direction,
            v_direction=v_direction,
            semantic_group=semantic_group,
            geometry=geometry,
        )


def plate(
    length: float, width: float, *, origin: Sequence[float] = (0, 0, 0),
    u_direction: Sequence[float] = (1, 0, 0), v_direction: Sequence[float] = (0, 1, 0),
    semantic_group: str = "shell",
) -> GeometryModel:
    length, width = _positive(length, "length"), _positive(width, "width")
    return _planar_grid((0.0, length), (0.0, width), origin=origin, u_direction=u_direction, v_direction=v_direction, semantic_group=semantic_group)


def stiffened_panel(
    length: float, width: float, *, longitudinal_spacing: float,
    transverse_spacing: float | None = None, origin: Sequence[float] = (0, 0, 0),
    u_direction: Sequence[float] = (1, 0, 0), v_direction: Sequence[float] = (0, 1, 0),
    semantic_group: str = "shell",
) -> GeometryModel:
    length, width = _positive(length, "length"), _positive(width, "width")
    longitudinal_spacing = _positive(longitudinal_spacing, "longitudinal_spacing")
    longitudinal = centered_member_positions(width, longitudinal_spacing, fallback_midpoint=False)
    transverse = (
        ()
        if transverse_spacing is None
        else centered_member_positions(
            length,
            _positive(transverse_spacing, "transverse_spacing"),
            fallback_midpoint=False,
        )
    )
    return _planar_grid(
        cleanup_axis((0.0, *transverse, length), length),
        cleanup_axis((0.0, *longitudinal, width), width),
        origin=origin, u_direction=u_direction, v_direction=v_direction, semantic_group=semantic_group,
    )


def _revolved_impl(
    radius_start: float, radius_end: float, height: float, *, circumferential_segments: int,
    longitudinal_spacing: float | None, ring_spacing: float | None,
    origin: Sequence[float], axis: Sequence[float], radial_direction: Sequence[float], is_cone: bool,
    geometry: GeometryModel,
) -> GeometryModel:
    r0, r1, height = _positive(radius_start, "radius_start", zero=True), _positive(radius_end, "radius_end", zero=True), _positive(height, "height")
    if max(r0, r1) <= 0.0:
        raise GeometryError("at least one radius must be positive")
    segments = _integer_at_least(
        circumferential_segments, 3, "circumferential_segments"
    )
    if longitudinal_spacing is not None:
        longitudinal_spacing = _positive(
            longitudinal_spacing, "longitudinal_spacing"
        )
        segments = max(segments, closed_loop_member_count(2*np.pi*max(r0, r1), longitudinal_spacing))
    base, axial = _vector3(origin, "origin"), _unit(axis, "axis")
    radial = _unit(radial_direction, "radial_direction")
    radial = radial - float(radial @ axial) * axial
    radial_length = float(np.linalg.norm(radial))
    if radial_length <= 1e-12:
        raise GeometryError("origin/radial_direction do not define a finite radial basis")
    radial /= radial_length
    tangent = np.cross(axial, radial)
    stations = (
        ()
        if ring_spacing is None
        else centered_member_positions(
            height,
            _positive(ring_spacing, "ring_spacing"),
            fallback_midpoint=False,
        )
    )
    levels = cleanup_axis((0.0, *stations, height), height)
    rings: list[list[int]] = []
    level_radii: list[float] = []
    for z in levels:
        radius = r0 + (r1-r0)*z/height
        level_radii.append(radius)
        if radius <= 1.0e-12:
            apex = geometry.add_point(*(base + z*axial))
            rings.append([apex] * segments)
        else:
            rings.append([geometry.add_point(*(base + z*axial + radius*(np.cos(2*np.pi*i/segments)*radial + np.sin(2*np.pi*i/segments)*tangent))) for i in range(segments)])
    arcs: list[list[int | None]] = []
    for level, vertices in enumerate(rings):
        z = levels[level]
        radius = level_radii[level]
        row: list[int | None] = []
        if radius <= 1.0e-12:
            arcs.append([None] * segments)
            continue
        for i in range(segments):
            angle = 2*np.pi*(i+0.5)/segments
            via = geometry.add_point(*(base + z*axial + radius*(np.cos(angle)*radial + np.sin(angle)*tangent)))
            row.append(geometry.add_arc(vertices[i], via, vertices[(i+1)%segments]))
        arcs.append(row)
    generators = [[geometry.add_line(rings[j][i], rings[j+1][i]) for i in range(segments)] for j in range(len(levels)-1)]
    faces: list[EntityRef] = []
    for j in range(len(levels)-1):
        for i in range(segments):
            lower_arc, upper_arc = arcs[j][i], arcs[j+1][i]
            common = dict(origin=base+levels[j]*axial, axis=axial, radial_direction=radial, height=levels[j+1]-levels[j], start_angle=2*np.pi*i/segments, sweep_angle=2*np.pi/segments)
            face_surface = (
                Cone(
                    radius_start=r0+(r1-r0)*levels[j]/height,
                    radius_end=r0+(r1-r0)*levels[j+1]/height,
                    **common,
                )
                if is_cone
                else Cylinder(radius=r0, **common)
            )
            if lower_arc is None:
                assert upper_arc is not None
                loop = (
                    OrientedEdge(generators[j][i], True),
                    OrientedEdge(upper_arc, True),
                    OrientedEdge(generators[j][(i+1)%segments], False),
                )
                face_id = geometry.add_face_from_loop(loop, surface=face_surface)
            elif upper_arc is None:
                loop = (
                    OrientedEdge(lower_arc, True),
                    OrientedEdge(generators[j][(i+1)%segments], True),
                    OrientedEdge(generators[j][i], False),
                )
                face_id = geometry.add_face_from_loop(loop, surface=face_surface)
            else:
                loop = (OrientedEdge(lower_arc, True), OrientedEdge(generators[j][(i+1)%segments], True), OrientedEdge(upper_arc, False), OrientedEdge(generators[j][i], False))
                face_id = geometry.add_face_from_loop(
                    loop, (0,1,2,3), surface=face_surface
                )
            faces.append(EntityRef("face", face_id))
    part = geometry.add_part(name="cone" if is_cone else "cylinder")
    geometry.add_sheet(
        tuple(reference.id for reference in faces),
        part_id=part,
        name="shell",
    )
    geometry.add_to_group("shell", faces)
    bottom = (
        [EntityRef("vertex", rings[0][0])]
        if arcs[0][0] is None
        else [EntityRef("edge", int(edge)) for edge in arcs[0]]
    )
    top = (
        [EntityRef("vertex", rings[-1][0])]
        if arcs[-1][0] is None
        else [EntityRef("edge", int(edge)) for edge in arcs[-1]]
    )
    geometry.add_to_group("bottom", bottom)
    geometry.add_to_group("top", top)
    geometry.add_to_group("boundaries", (*bottom, *top))
    geometry.add_to_group("longitudinal_stiffeners", [EntityRef("edge", edge) for row in generators for edge in row])
    geometry.add_to_group("ring_stiffeners", [EntityRef("edge", int(edge)) for row in arcs[1:-1] for edge in row if edge is not None])
    member_chains = [
        tuple(OrientedEdge(row[i], True) for row in generators)
        for i in range(segments)
    ]
    member_names = [f"longitudinal_stiffener_{i}" for i in range(segments)]
    for level, row in enumerate(arcs[1:-1], start=1):
        ring = tuple(
            OrientedEdge(int(edge), True) for edge in row if edge is not None
        )
        if ring:
            member_chains.append(ring)
            member_names.append(f"ring_stiffener_{level}")
    geometry.add_members(member_chains, part_id=part, names=member_names)
    return geometry


def _revolved(
    radius_start: float, radius_end: float, height: float, *, circumferential_segments: int,
    longitudinal_spacing: float | None, ring_spacing: float | None,
    origin: Sequence[float], axis: Sequence[float], radial_direction: Sequence[float], is_cone: bool,
) -> GeometryModel:
    """Build one compound revolved shell as a single validated transaction."""

    geometry = GeometryModel()
    with geometry.transaction():
        return _revolved_impl(
            radius_start,
            radius_end,
            height,
            circumferential_segments=circumferential_segments,
            longitudinal_spacing=longitudinal_spacing,
            ring_spacing=ring_spacing,
            origin=origin,
            axis=axis,
            radial_direction=radial_direction,
            is_cone=is_cone,
            geometry=geometry,
        )


def cylinder(radius: float, height: float, *, circumferential_segments: int = 12, longitudinal_spacing: float | None = None, ring_spacing: float | None = None, origin: Sequence[float] = (0,0,0), axis: Sequence[float] = (0,0,1), radial_direction: Sequence[float] = (1,0,0)) -> GeometryModel:
    radius = _positive(radius, "radius")
    return _revolved(radius, radius, height, circumferential_segments=circumferential_segments, longitudinal_spacing=longitudinal_spacing, ring_spacing=ring_spacing, origin=origin, axis=axis, radial_direction=radial_direction, is_cone=False)


def cone(radius_start: float, radius_end: float, height: float, *, circumferential_segments: int = 12, longitudinal_spacing: float | None = None, ring_spacing: float | None = None, origin: Sequence[float] = (0,0,0), axis: Sequence[float] = (0,0,1), radial_direction: Sequence[float] = (1,0,0)) -> GeometryModel:
    return _revolved(radius_start, radius_end, height, circumferential_segments=circumferential_segments, longitudinal_spacing=longitudinal_spacing, ring_spacing=ring_spacing, origin=origin, axis=axis, radial_direction=radial_direction, is_cone=True)


def shell(*args: object, **kwargs: object) -> GeometryModel:
    return plate(*args, **kwargs)  # type: ignore[arg-type]


def bulkhead(*args: object, **kwargs: object) -> GeometryModel:
    kwargs.setdefault("semantic_group", "bulkhead")
    return plate(*args, **kwargs)  # type: ignore[arg-type]


def frame(*args: object, **kwargs: object) -> GeometryModel:
    kwargs.setdefault("semantic_group", "frame")
    return plate(*args, **kwargs)  # type: ignore[arg-type]


def girder(length: float, *, origin: Sequence[float] = (0,0,0), direction: Sequence[float] = (1,0,0)) -> GeometryModel:
    geometry = GeometryModel()
    start_point = _vector3(origin, "origin")
    start = geometry.add_point(*start_point)
    end = geometry.add_point(*(start_point + _positive(length, "length")*_unit(direction, "direction")))
    edge = geometry.add_line(start, end)
    part = geometry.add_part(name="girder")
    geometry.add_member((edge,), part_id=part, name="girder")
    reference = EntityRef("edge", edge)
    geometry.add_to_group("girder", [reference])
    geometry.add_to_group("girders", [reference])
    return geometry


def stiffener(*args: object, **kwargs: object) -> GeometryModel:
    geometry = girder(*args, **kwargs)
    references = geometry.group("girder", resolve=False)
    geometry.remove_group("girder")
    geometry.remove_group("girders")
    geometry.add_to_group("stiffener", references)
    geometry.add_to_group("stiffeners", references)
    return geometry
