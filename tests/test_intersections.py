"""Analytical intersections, fallback curves, and shell imprinting."""

from __future__ import annotations

import numpy as np
import pytest

from anygeometry import (
    Cylinder,
    EntityRef,
    FaceIntersection,
    GeometryError,
    GeometryModel,
    IntersectionKind,
    MutationPolicy,
    OrientedEdge,
    Plane,
    clip_line_to_face,
    intersect_faces,
    intersect_surfaces,
    line_cylinder,
    line_line,
    line_plane,
    numerical_surface_intersection,
    plane_cylinder,
    plane_plane,
    from_dict,
    to_dict,
    trim_face,
)
from anygeometry.generators import cylinder


def _without_document_identity(document: dict) -> dict:
    """Canonical geometry payload for comparing independently created models."""

    return {
        key: value
        for key, value in document.items()
        if key not in {"model_id", "checksum"}
    }


def _without_allocator_state(document: dict) -> dict:
    """Committed state payload unaffected by deliberate failed-edit ID gaps."""

    return {
        key: value
        for key, value in document.items()
        if key not in {"id_state", "checksum"}
    }


def _plane(
    origin: tuple[float, float, float],
    u: tuple[float, float, float],
    v: tuple[float, float, float],
) -> Plane:
    return Plane(np.asarray(origin), np.asarray(u), np.asarray(v))


def test_line_line_handles_crossing_skew_parallel_and_invalid_lines() -> None:
    crossing = line_line(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.5, -1.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    assert crossing == pytest.approx((0.5, 0.0, 0.0))
    offset = np.asarray((1.0e12, -2.0e12, 3.0e12))
    translated = line_line(
        offset + (0.0, 0.0, 0.0),
        (1.0e9, 0.0, 0.0),
        offset + (0.5, -1.0, 0.0),
        (0.0, 1.0e-6, 0.0),
    )
    assert translated is not None
    assert translated - offset == pytest.approx(crossing)
    assert (
        line_line(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 1.0),
            (0.0, 1.0, 0.0),
        )
        is None
    )
    assert (
        line_line(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
        )
        is None
    )
    with pytest.raises(GeometryError, match="non-zero"):
        line_line(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
        )


def test_line_plane_and_plane_plane_intersections() -> None:
    horizontal = _plane((0.0, 0.0, 2.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    vertical = _plane((0.5, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

    assert line_plane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), horizontal) == pytest.approx(
        (0.0, 0.0, 2.0)
    )
    assert line_plane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), horizontal) is None
    with pytest.raises(GeometryError, match="non-zero"):
        line_plane((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), horizontal)

    intersection = plane_plane(horizontal, vertical)
    assert intersection is not None
    point, direction = intersection
    assert horizontal.normal @ (point - horizontal.origin) == pytest.approx(0.0)
    assert vertical.normal @ (point - vertical.origin) == pytest.approx(0.0)
    assert abs(float(direction @ horizontal.normal)) <= 1.0e-12
    assert abs(float(direction @ vertical.normal)) <= 1.0e-12
    assert plane_plane(horizontal, horizontal) is None


def test_line_cylinder_returns_bounded_secant_and_tangent_hits() -> None:
    cylinder = Cylinder(
        np.zeros(3),
        np.asarray((0.0, 0.0, 1.0)),
        np.asarray((1.0, 0.0, 0.0)),
        2.0,
        4.0,
    )

    hits = line_cylinder((-3.0, 0.0, 2.0), (1.0, 0.0, 0.0), cylinder)
    tangent = line_cylinder((-3.0, 2.0, 2.0), (1.0, 0.0, 0.0), cylinder)

    assert len(hits) == 2
    assert hits[0] == pytest.approx((-2.0, 0.0, 2.0))
    assert hits[1] == pytest.approx((2.0, 0.0, 2.0))
    assert len(tangent) == 1
    assert tangent[0] == pytest.approx((0.0, 2.0, 2.0))
    assert not line_cylinder((-3.0, 0.0, 6.0), (1.0, 0.0, 0.0), cylinder)
    assert len(
        line_cylinder(
            (-3.0, 0.0, 6.0),
            (1.0, 0.0, 0.0),
            cylinder,
            bounded=False,
        )
    ) == 2
    with pytest.raises(GeometryError, match="non-zero"):
        line_cylinder((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), cylinder)


def test_plane_cylinder_covers_cross_section_and_axial_generators() -> None:
    cylinder = Cylinder(
        np.zeros(3),
        np.asarray((0.0, 0.0, 1.0)),
        np.asarray((1.0, 0.0, 0.0)),
        2.0,
        4.0,
    )
    horizontal = _plane((0.0, 0.0, 2.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    vertical = _plane((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

    circles = plane_cylinder(horizontal, cylinder, samples=65)
    generators = plane_cylinder(vertical, cylinder, samples=65)

    assert len(circles) == 1
    assert circles[0].shape == (65, 3)
    assert np.linalg.norm(circles[0][:, :2], axis=1) == pytest.approx(2.0)
    assert circles[0][:, 2] == pytest.approx(2.0)
    assert len(generators) == 2
    assert all(curve.shape == (2, 3) for curve in generators)
    assert all(np.linalg.norm(curve[:, :2], axis=1) == pytest.approx(2.0) for curve in generators)


def test_numerical_surface_fallback_is_deterministic() -> None:
    horizontal = _plane((0.0, -1.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0))
    vertical = _plane((0.0, 0.0, -1.0), (2.0, 0.0, 0.0), (0.0, 0.0, 2.0))

    first = numerical_surface_intersection(horizontal, vertical, samples=21)
    second = numerical_surface_intersection(horizontal, vertical, samples=21)

    assert first.shape == (21, 3)
    assert first == pytest.approx(second)
    assert first[:, 1:] == pytest.approx(0.0)
    assert first[:, 0] == pytest.approx(np.linspace(0.0, 2.0, 21))


def test_surface_dispatch_uses_analytical_plane_cylinder_path() -> None:
    cylinder = Cylinder(
        np.zeros(3),
        np.asarray((0.0, 0.0, 1.0)),
        np.asarray((1.0, 0.0, 0.0)),
        2.0,
        4.0,
    )
    plane = _plane((0.0, 0.0, 2.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))

    curves = intersect_surfaces(plane, cylinder, samples=65)

    assert len(curves) == 1
    assert curves[0].shape == (65, 3)
    assert np.linalg.norm(curves[0][:, :2], axis=1) == pytest.approx(2.0)


def _crossing_shells() -> tuple[GeometryModel, int, int]:
    geometry = GeometryModel()
    horizontal = geometry.add_plate(
        geometry.add_points(
            (
                (-1.0, -1.0, 0.0),
                (1.0, -1.0, 0.0),
                (1.0, 1.0, 0.0),
                (-1.0, 1.0, 0.0),
            )
        )
    )
    vertical = geometry.add_plate(
        geometry.add_points(
            (
                (-1.0, 0.0, -1.0),
                (1.0, 0.0, -1.0),
                (1.0, 0.0, 1.0),
                (-1.0, 0.0, 1.0),
            )
        )
    )
    return geometry, horizontal, vertical


def test_shell_intersection_query_is_non_mutating() -> None:
    geometry, horizontal, vertical = _crossing_shells()
    counts = (len(geometry.vertices), len(geometry.edges), len(geometry.faces))

    endpoints = intersect_faces(geometry, horizontal, vertical, fragment=False)

    assert isinstance(endpoints, tuple)
    assert np.asarray(endpoints) == pytest.approx(
        np.asarray(((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    )
    assert (len(geometry.vertices), len(geometry.edges), len(geometry.faces)) == counts


def test_shell_intersection_imprints_one_shared_edge_and_fragments_both_faces() -> None:
    geometry, horizontal, vertical = _crossing_shells()
    old_refs = (EntityRef("face", horizontal), EntityRef("face", vertical))
    geometry.add_to_group("shell", old_refs)
    geometry.tag(old_refs[0], "deck")
    geometry.tag(old_refs[1], "bulkhead")

    result = intersect_faces(
        geometry, horizontal, vertical, policy=MutationPolicy.IMPRINT
    )

    assert isinstance(result, FaceIntersection)
    assert result.edges == (result.edge,)
    assert result.edge in (edge.ref for edge in geometry.edges.values())
    assert len(result.first_faces) == len(result.second_faces) == 2
    assert len(geometry.faces_using_edge(result.edge.id)) == 4
    assert set(geometry.resolve_ref(old_refs[0])) == set(result.first_faces)
    assert set(geometry.resolve_ref(old_refs[1])) == set(result.second_faces)
    assert set(geometry.group("shell")) == set(result.first_faces + result.second_faces)
    assert all(geometry.tags_for(reference) == ("deck",) for reference in result.first_faces)
    assert all(geometry.tags_for(reference) == ("bulkhead",) for reference in result.second_faces)
    assert geometry.validate_topology() == ()


def test_axial_plane_cylinder_intersection_imprints_one_shared_edge() -> None:
    geometry = GeometryModel()
    root_two = np.sqrt(2.0)
    arc_points = geometry.add_points(
        ((2.0, 0.0, 0.0), (root_two, root_two, 0.0), (0.0, 2.0, 0.0))
    )
    arc = geometry.add_arc(*arc_points)
    cylinder_face = geometry.extrude((arc,), (0.0, 0.0, 3.0))[0]
    geometry.set_face_surface(
        cylinder_face,
        Cylinder(
            np.zeros(3),
            np.asarray((0.0, 0.0, 1.0)),
            np.asarray((1.0, 0.0, 0.0)),
            2.0,
            3.0,
            0.0,
            0.5 * np.pi,
        ),
    )
    radial = np.asarray((1.0, 1.0, 0.0)) / root_two
    plane_face = geometry.add_plate(
        geometry.add_points(
            (
                tuple(-3.0 * radial),
                tuple(3.0 * radial),
                tuple(3.0 * radial + (0.0, 0.0, 3.0)),
                tuple(-3.0 * radial + (0.0, 0.0, 3.0)),
            )
        )
    )

    result = intersect_faces(
        geometry, plane_face, cylinder_face, policy=MutationPolicy.IMPRINT
    )

    assert isinstance(result, FaceIntersection)
    assert result.edges == (result.edge,)
    assert len(geometry.faces_using_edge(result.edge.id)) == 4
    assert np.allclose(
        geometry.sample_edge(result.edge.id, np.asarray((0.0, 1.0))),
        np.asarray(((root_two, root_two, 0.0), (root_two, root_two, 3.0))),
        rtol=0.0,
        atol=1.0e-10,
    )
    assert geometry.validate_topology() == ()


def _transverse_shell(
    *, plane_half_width: float = 2.0
) -> tuple[GeometryModel, int, tuple[int, ...]]:
    geometry = cylinder(
        1.0, 2.0, circumferential_segments=4
    )
    cylinder_faces = tuple(reference.id for reference in geometry.group("shell"))
    plane_face = geometry.add_plate(
        geometry.add_points(
            (
                (-plane_half_width, -plane_half_width, 1.0),
                (plane_half_width, -plane_half_width, 1.0),
                (plane_half_width, plane_half_width, 1.0),
                (-plane_half_width, plane_half_width, 1.0),
            )
        )
    )
    plane_ref = EntityRef("face", plane_face)
    geometry.add_to_group("deck", (plane_ref,))
    geometry.tag(plane_ref, "bulkhead")
    for face_id in cylinder_faces:
        geometry.tag(EntityRef("face", face_id), "shell")
    return geometry, plane_face, cylinder_faces


def test_transverse_plane_cylinder_query_returns_complete_closed_ring() -> None:
    geometry, plane_face, cylinder_faces = _transverse_shell()
    document = to_dict(geometry)

    ring = intersect_faces(
        geometry, plane_face, cylinder_faces[2], fragment=False
    )

    assert isinstance(ring, np.ndarray)
    assert ring.shape == (65, 3)
    assert ring[0] == pytest.approx(ring[-1])
    assert np.linalg.norm(ring[:, :2], axis=1) == pytest.approx(1.0)
    assert ring[:, 2] == pytest.approx(1.0)
    assert to_dict(geometry) == document


def test_transverse_plane_cylinder_imprints_atomic_shared_ring() -> None:
    geometry, plane_face, cylinder_faces = _transverse_shell()
    old_plane = EntityRef("face", plane_face)
    old_cylinders = tuple(EntityRef("face", item) for item in cylinder_faces)

    result = intersect_faces(
        geometry,
        plane_face,
        cylinder_faces[0],
        policy=MutationPolicy.IMPRINT,
    )

    assert isinstance(result, FaceIntersection)
    assert result.edge == result.edges[0]
    assert len(result.edges) == 4
    assert len(result.first_faces) == 2
    assert len(result.second_faces) == 8
    assert all(
        len(geometry.faces_using_edge(reference.id)) == 4
        for reference in result.edges
    )
    ring_samples = np.vstack(
        [
            geometry.sample_edge(reference.id, np.linspace(0.0, 1.0, 9))
            for reference in result.edges
        ]
    )
    assert np.linalg.norm(ring_samples[:, :2], axis=1) == pytest.approx(1.0)
    assert ring_samples[:, 2] == pytest.approx(1.0)
    assert set(geometry.resolve_ref(old_plane)) == set(result.first_faces)
    assert all(len(geometry.resolve_ref(reference)) == 2 for reference in old_cylinders)
    assert set(geometry.group("deck")) == set(result.first_faces)
    assert set(geometry.group("shell")) == set(result.second_faces)
    assert all(
        geometry.tags_for(reference) == ("bulkhead",)
        for reference in result.first_faces
    )
    assert all(
        geometry.tags_for(reference) == ("shell",)
        for reference in result.second_faces
    )
    assert sorted(len(geometry.faces[reference.id].holes) for reference in result.first_faces) == [0, 1]
    assert all(
        isinstance(geometry.faces[reference.id].surface, Plane)
        for reference in result.first_faces
    )
    assert all(
        isinstance(geometry.faces[reference.id].surface, Cylinder)
        for reference in result.second_faces
    )
    assert geometry.validate_topology() == ()

    document = to_dict(geometry)
    restored = from_dict(document)
    assert to_dict(restored) == document
    assert restored.validate_topology() == ()


def test_transverse_ring_is_deterministic_and_preserves_argument_order() -> None:
    first, first_plane, first_cylinders = _transverse_shell()
    second, second_plane, second_cylinders = _transverse_shell()

    forward = intersect_faces(
        first, first_plane, first_cylinders[1], policy=MutationPolicy.IMPRINT
    )
    reverse = intersect_faces(
        second, second_cylinders[1], second_plane, policy=MutationPolicy.IMPRINT
    )

    assert isinstance(forward, FaceIntersection)
    assert isinstance(reverse, FaceIntersection)
    assert len(forward.first_faces) == len(reverse.second_faces) == 2
    assert len(forward.second_faces) == len(reverse.first_faces) == 8
    assert _without_document_identity(to_dict(first)) == _without_document_identity(
        to_dict(second)
    )


def test_transverse_ring_preflight_failure_does_not_mutate_geometry() -> None:
    geometry, plane_face, cylinder_faces = _transverse_shell(
        plane_half_width=0.75
    )
    document = to_dict(geometry)

    with pytest.raises(GeometryError, match="strictly inside"):
        intersect_faces(
            geometry,
            plane_face,
            cylinder_faces[0],
            policy=MutationPolicy.IMPRINT,
        )

    assert to_dict(geometry) == document
    assert geometry.validate_topology() == ()


def test_transverse_ring_rolls_back_if_fragmentation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry, plane_face, cylinder_faces = _transverse_shell()
    document = to_dict(geometry)
    original_add_arc = geometry.add_arc
    calls = 0

    def interrupted_add_arc(start: int, via: int, end: int) -> int:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected ring failure")
        return original_add_arc(start, via, end)

    monkeypatch.setattr(geometry, "add_arc", interrupted_add_arc)

    with pytest.raises(RuntimeError, match="injected ring failure"):
        intersect_faces(
            geometry,
            plane_face,
            cylinder_faces[0],
            policy=MutationPolicy.IMPRINT,
        )

    after = to_dict(geometry)
    assert _without_allocator_state(after) == _without_allocator_state(document)
    assert all(
        after["id_state"][kind] >= value
        for kind, value in document["id_state"].items()
    )
    assert any(
        after["id_state"][kind] > value
        for kind, value in document["id_state"].items()
    )
    assert geometry.validate_topology() == ()


def test_shell_intersection_rejects_parallel_or_nonoverlapping_faces() -> None:
    geometry = GeometryModel()
    first = geometry.add_plate(
        geometry.add_points(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
        )
    )
    second = geometry.add_plate(
        geometry.add_points(
            ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0))
        )
    )

    with pytest.raises(GeometryError, match="parallel or coplanar"):
        intersect_faces(geometry, first, second, policy=MutationPolicy.IMPRINT)


def test_planar_line_clipping_returns_all_material_intervals_minus_holes() -> None:
    geometry = GeometryModel()
    outer = geometry.add_points(((0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0)))
    face = geometry.add_plate(outer)
    inner = geometry.add_points(((1, 1, 0), (3, 1, 0), (3, 3, 0), (1, 3, 0)))
    inner_edges = tuple(
        geometry.add_line(inner[index], inner[(index + 1) % 4])
        for index in range(4)
    )
    trim_face(
        geometry,
        face,
        ((OrientedEdge(inner_edges[index], True) for index in range(4)),),
    )

    result = clip_line_to_face(geometry, face, (-1, 2, 0), (5, 0, 0))

    assert result.kind is IntersectionKind.OVERLAP_CURVE
    assert len(result.components) == 2
    assert [component.first_parameter_range.start for component in result.components] == pytest.approx((1.0, 4.0))
    assert [component.first_parameter_range.end for component in result.components] == pytest.approx((2.0, 5.0))


def test_face_intersection_requires_explicit_supported_mutation_policy() -> None:
    geometry = GeometryModel()
    first = geometry.add_plate(
        geometry.add_points(((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)))
    )
    second = geometry.add_plate(
        geometry.add_points(((1, -1, -1), (1, 3, -1), (1, 3, 1), (1, -1, 1)))
    )

    with pytest.raises(GeometryError, match="requires an explicit mutation policy"):
        intersect_faces(geometry, first, second)
    with pytest.raises(GeometryError, match="rejected by policy"):
        intersect_faces(geometry, first, second, policy=MutationPolicy.REJECT)
    result = intersect_faces(
        geometry, first, second, policy=MutationPolicy.KEEP_SEPARATE_PART
    )
    assert isinstance(result, tuple)
    assert len(geometry.faces) == 2
