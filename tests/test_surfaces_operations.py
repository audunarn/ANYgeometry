"""Surface evaluation, projection, transforms, trims, and fragmentation."""

from __future__ import annotations

import numpy as np
import pytest

from anygeometry import (
    CoonsSurface,
    Cone,
    Cylinder,
    EntityRef,
    GeometryError,
    GeometryModel,
    OrientedEdge,
    Plane,
    RuledSurface,
    closest_point,
    fragment_face,
    project,
    punch_hole,
    split_face_at,
    split_face_between,
    strip_face,
    surface_point,
    transform,
    trim_face,
    to_dict,
)
from anygeometry.surfaces import closest_uv, surface_normal


def test_explicit_surface_evaluation_uv_and_normals() -> None:
    plane = Plane(
        np.asarray((1.0, 2.0, 3.0)),
        np.asarray((4.0, 0.0, 0.0)),
        np.asarray((0.0, 2.0, 0.0)),
    )
    cylinder = Cylinder(
        np.zeros(3),
        np.asarray((0.0, 0.0, 1.0)),
        np.asarray((1.0, 0.0, 0.0)),
        2.0,
        5.0,
    )
    cone = Cone(
        np.zeros(3),
        np.asarray((0.0, 0.0, 1.0)),
        np.asarray((1.0, 0.0, 0.0)),
        2.0,
        1.0,
        5.0,
    )
    ruled = RuledSurface(
        np.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))),
        np.asarray(((0.0, 1.0, 1.0), (2.0, 1.0, 1.0))),
    )

    assert plane.evaluate(0.25, 0.5) == pytest.approx((2.0, 3.0, 3.0))
    assert plane.local_uv((2.0, 3.0, 3.0)) == pytest.approx((0.25, 0.5))
    assert plane.normal == pytest.approx((0.0, 0.0, 1.0))

    cylinder_point = cylinder.evaluate(0.25, 0.4)
    assert cylinder_point == pytest.approx((0.0, 2.0, 2.0), abs=1.0e-12)
    assert cylinder.local_uv(cylinder_point) == pytest.approx((0.25, 0.4))
    assert surface_normal(cylinder, 0.0, 0.5) == pytest.approx((1.0, 0.0, 0.0), abs=1.0e-5)

    cone_point = cone.evaluate(0.5, 0.5)
    assert cone_point == pytest.approx((-1.5, 0.0, 2.5), abs=1.0e-12)
    assert cone.local_uv(cone_point) == pytest.approx((0.5, 0.5))
    assert ruled.evaluate(0.25, 0.5) == pytest.approx((0.5, 0.5, 0.5))
    assert closest_uv(ruled, (0.5, 0.5, 0.5)) == pytest.approx((0.25, 0.5), abs=1.0e-5)


def test_coons_face_uses_the_topology_boundary_convention(
    rectangle: tuple[GeometryModel, int, tuple[int, ...], tuple[int, ...]],
) -> None:
    geometry, face, _vertices, _edges = rectangle

    assert isinstance(geometry.faces[face].surface, CoonsSurface)
    assert surface_point(geometry, face, 0.0, 0.0) == pytest.approx((0.0, 0.0, 0.0))
    assert surface_point(geometry, face, 1.0, 0.0) == pytest.approx((4.0, 0.0, 0.0))
    assert surface_point(geometry, face, 1.0, 1.0) == pytest.approx((4.0, 2.0, 0.0))
    assert surface_point(geometry, face, 0.5, 0.5) == pytest.approx((2.0, 1.0, 0.0))
    assert geometry.face_normal(face, 0.5, 0.5) == pytest.approx((0.0, 0.0, 1.0))


def test_explicit_coons_surface_evaluates_and_inverts_without_topology() -> None:
    bottom = np.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
    right = np.asarray(((2.0, 0.0, 0.0), (2.0, 1.0, 0.5)))
    top = np.asarray(((0.0, 1.0, 0.5), (2.0, 1.0, 0.5)))
    left = np.asarray(((0.0, 0.0, 0.0), (0.0, 1.0, 0.5)))
    surface = CoonsSurface(bottom, right, top, left)

    point = surface.evaluate(0.25, 0.75)

    assert point == pytest.approx((0.5, 0.75, 0.375))
    assert surface.local_uv(point) == pytest.approx((0.25, 0.75), abs=1.0e-6)


def test_explicit_coons_surface_validates_scale_aware_oriented_corners() -> None:
    scale = 1.0e8
    bottom = np.asarray(((0.0, 0.0, 0.0), (2.0 * scale, 0.0, 0.0)))
    right = np.asarray(
        ((2.0 * scale + 1.0e-3, 0.0, 0.0), (2.0 * scale, scale, 0.0))
    )
    top = np.asarray(((0.0, scale, 0.0), (2.0 * scale, scale, 0.0)))
    left = np.asarray(((0.0, 0.0, 0.0), (0.0, scale, 0.0)))

    # The 1e-11-relative endpoint difference is only construction roundoff.
    surface = CoonsSurface(bottom, right, top, left)
    assert surface.evaluate(0.5, 0.5) == pytest.approx((scale, 0.5 * scale, 0.0))

    incompatible = right.copy()
    incompatible[0, 0] += 1.0
    with pytest.raises(
        GeometryError, match=r"bottom\[-1\].*right\[0\]"
    ):
        CoonsSurface(bottom, incompatible, top, left)

    with pytest.raises(GeometryError, match=r"top\[0\].*left\[-1\]"):
        CoonsSurface(bottom, right, top[::-1], left)

    tiny = 1.0e-9
    tiny_bottom = np.asarray(((0.0, 0.0, 0.0), (2.0 * tiny, 0.0, 0.0)))
    tiny_right = np.asarray(
        ((2.0 * tiny + 5.0e-14, 0.0, 0.0), (2.0 * tiny, tiny, 0.0))
    )
    tiny_top = np.asarray(((0.0, tiny, 0.0), (2.0 * tiny, tiny, 0.0)))
    tiny_left = np.asarray(((0.0, 0.0, 0.0), (0.0, tiny, 0.0)))
    CoonsSurface(tiny_bottom, tiny_right, tiny_top, tiny_left)

    tiny_incompatible = tiny_right.copy()
    tiny_incompatible[0, 0] += 1.0e-11
    with pytest.raises(
        GeometryError, match=r"bottom\[-1\].*right\[0\]"
    ):
        CoonsSurface(tiny_bottom, tiny_incompatible, tiny_top, tiny_left)


def test_surface_normals_are_defined_at_every_parameter_boundary() -> None:
    surfaces = (
        Plane(
            np.zeros(3),
            np.asarray((2.0, 0.0, 0.0)),
            np.asarray((0.0, 1.0, 0.5)),
        ),
        Cylinder(
            np.zeros(3),
            np.asarray((0.0, 0.0, 1.0)),
            np.asarray((1.0, 0.0, 0.0)),
            2.0,
            5.0,
            start_angle=0.4,
            sweep_angle=1.2,
        ),
        Cone(
            np.zeros(3),
            np.asarray((0.0, 0.0, 1.0)),
            np.asarray((1.0, 0.0, 0.0)),
            2.0,
            1.0,
            5.0,
            start_angle=-0.3,
            sweep_angle=1.5,
        ),
        RuledSurface(
            np.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))),
            np.asarray(((0.0, 1.0, 1.0), (2.0, 1.0, 1.0))),
        ),
        CoonsSurface(
            np.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))),
            np.asarray(((2.0, 0.0, 0.0), (2.0, 1.0, 0.5))),
            np.asarray(((0.0, 1.0, 0.5), (2.0, 1.0, 0.5))),
            np.asarray(((0.0, 0.0, 0.0), (0.0, 1.0, 0.5))),
        ),
    )

    for surface in surfaces:
        for u, v in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)):
            normal = surface_normal(surface, u, v)
            assert np.all(np.isfinite(normal))
            assert np.linalg.norm(normal) == pytest.approx(1.0)


@pytest.mark.parametrize("surface_type", (Cylinder, Cone))
@pytest.mark.parametrize("sweep", (-1.37, 1.37))
def test_reflection_preserves_curved_surface_uv_and_topology(
    surface_type: type[Cylinder] | type[Cone], sweep: float
) -> None:
    common = dict(
        origin=np.asarray((0.3, -0.4, 0.2)),
        axis=np.asarray((0.0, 0.0, 1.0)),
        radial_direction=np.asarray((1.0, 0.0, 0.0)),
        height=3.25,
        start_angle=1.23,
        sweep_angle=sweep,
    )
    surface = (
        Cylinder(radius=2.0, **common)
        if surface_type is Cylinder
        else Cone(radius_start=2.0, radius_end=1.25, **common)
    )
    geometry = GeometryModel()
    lower = tuple(
        geometry.add_point(*surface.evaluate(u, 0.0)) for u in (0.0, 0.5, 1.0)
    )
    upper = tuple(
        geometry.add_point(*surface.evaluate(u, 1.0)) for u in (0.0, 0.5, 1.0)
    )
    bottom = geometry.add_arc(lower[0], lower[1], lower[2])
    end = geometry.add_line(lower[2], upper[2])
    top = geometry.add_arc(upper[0], upper[1], upper[2])
    start = geometry.add_line(lower[0], upper[0])
    face = geometry.add_face_from_loop(
        (
            OrientedEdge(bottom, True),
            OrientedEdge(end, True),
            OrientedEdge(top, False),
            OrientedEdge(start, False),
        ),
        (0, 1, 2, 3),
        surface=surface,
    )
    samples = tuple(
        (u, v, geometry.face_point(face, u, v))
        for u in (0.0, 0.17, 0.5, 0.83, 1.0)
        for v in (0.0, 0.4, 1.0)
    )
    normal_before = surface_normal(surface, 0.37, 0.61)
    matrix = np.asarray(
        (
            (-1.75, 0.0, 0.0, 4.0),
            (0.0, 1.75, 0.0, -2.0),
            (0.0, 0.0, 1.75, 0.5),
            (0.0, 0.0, 0.0, 1.0),
        )
    )

    transform(geometry, matrix)

    transformed = geometry.faces[face].surface
    assert isinstance(transformed, surface_type)
    assert transformed.start_angle == 0.0
    assert transformed.sweep_angle == pytest.approx(-sweep)
    orthogonal = matrix[:3, :3] / 1.75
    expected_normal = np.linalg.det(orthogonal) * orthogonal @ normal_before
    assert surface_normal(transformed, 0.37, 0.61) == pytest.approx(
        expected_normal, abs=1.0e-9
    )
    for u, v, original in samples:
        expected = matrix[:3, :3] @ original + matrix[:3, 3]
        actual = geometry.face_point(face, u, v)
        assert actual == pytest.approx(expected, abs=1.0e-11)
        assert transformed.local_uv(actual) == pytest.approx((u, v), abs=1.0e-11)

    parameters = np.linspace(0.0, 1.0, 9)
    assert geometry.sample_edge(bottom, parameters) == pytest.approx(
        np.vstack([geometry.face_point(face, u, 0.0) for u in parameters]),
        abs=1.0e-11,
    )
    assert geometry.sample_edge(top, parameters) == pytest.approx(
        np.vstack([geometry.face_point(face, u, 1.0) for u in parameters]),
        abs=1.0e-11,
    )
    corner_uv = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    for vertex, (u, v) in zip(geometry.face_corner_vertices(face), corner_uv):
        assert geometry.face_point(face, u, v) == pytest.approx(
            geometry.vertex_position(vertex), abs=1.0e-11
        )
    assert geometry.validate_topology() == ()


def test_projection_is_bounded_to_the_finite_face() -> None:
    geometry = GeometryModel()
    face = geometry.add_plate(
        geometry.add_points(
            ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 2.0, 0.0), (0.0, 2.0, 0.0))
        )
    )

    projected, uv, distance = project(geometry, face, (6.0, -1.0, 2.0))

    assert uv == pytest.approx((1.0, 0.0))
    assert projected == pytest.approx((4.0, 0.0, 0.0))
    assert distance == pytest.approx(3.0)


def test_projection_and_closest_point_on_curved_and_selected_entities(
    quarter_cylinder: tuple[GeometryModel, int, int],
) -> None:
    geometry, arc, face = quarter_cylinder
    target = np.asarray((3.0 / np.sqrt(2.0), 3.0 / np.sqrt(2.0), 1.5))

    projected, uv, distance = project(geometry, face, target)

    assert np.linalg.norm(projected[:2]) == pytest.approx(2.0)
    assert uv == pytest.approx((0.5, 0.5), abs=1.0e-5)
    assert distance == pytest.approx(1.0)

    reference, edge_point, edge_distance = closest_point(
        geometry,
        (2.1, 0.0, 0.0),
        (EntityRef("edge", arc),),
    )
    assert reference == EntityRef("edge", arc)
    assert edge_point == pytest.approx((2.0, 0.0, 0.0), abs=1.0e-10)
    assert edge_distance == pytest.approx(0.1)


def test_transform_preserves_ids_and_moves_all_curve_definition_vertices() -> None:
    geometry = GeometryModel()
    start, control_a, control_b, end = geometry.add_points(
        ((0.0, 0.0, 0.0), (1.0, 2.0, 0.0), (2.0, 2.0, 0.0), (3.0, 0.0, 0.0))
    )
    spline = geometry.add_spline(start, (control_a, control_b), end)
    reference = EntityRef("edge", spline)
    geometry.add_to_group("beam_axis", (reference,))
    before = geometry.sample_edge(spline, np.linspace(0.0, 1.0, 9))
    matrix = np.eye(4)
    matrix[:3, 3] = (5.0, -2.0, 3.0)

    moved = transform(geometry, matrix, (reference,))

    assert {item.id for item in moved} == {start, control_a, control_b, end}
    assert geometry.group("beam_axis") == (reference,)
    assert geometry.sample_edge(spline, np.linspace(0.0, 1.0, 9)) == pytest.approx(
        before + np.asarray((5.0, -2.0, 3.0))
    )


def test_split_and_strip_faces_preserve_lineage_groups_and_exact_plane() -> None:
    geometry = GeometryModel()
    face = geometry.add_plate(
        geometry.add_points(
            ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 2.0, 0.0), (0.0, 2.0, 0.0))
        )
    )
    old_ref = EntityRef("face", face)
    geometry.add_to_group("deck", (old_ref,))
    geometry.tag(old_ref, "loaded")

    divider, made = split_face_at(geometry, face, axis=0, fraction=0.5)

    made_refs = tuple(EntityRef("face", item) for item in made)
    assert set(geometry.resolve_ref(old_ref)) == set(made_refs)
    assert set(geometry.group("deck")) == set(made_refs)
    assert all(geometry.tags_for(reference) == ("loaded",) for reference in made_refs)
    assert geometry.sample_edge(divider, np.linspace(0.0, 1.0, 5))[:, 0] == pytest.approx(2.0)
    assert all(isinstance(geometry.faces[item].surface, Plane) for item in made)

    strips, dividers = strip_face(geometry, made[0], axis=1, count=2)
    assert len(strips) == 2
    assert len(dividers) == 1
    assert geometry.validate_topology() == ()


def test_fragment_face_applies_multiple_boundary_vertex_cuts() -> None:
    geometry = GeometryModel()
    vertices = tuple(
        geometry.add_points(
            (
                (0.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (4.0, 0.0, 0.0),
                (4.0, 2.0, 0.0),
                (2.0, 2.0, 0.0),
                (0.0, 2.0, 0.0),
            )
        )
    )
    face = geometry.add_face(geometry.add_polyline(vertices, close=True))

    descendants = fragment_face(geometry, face, ((vertices[1], vertices[4]),))

    assert len(descendants) == 2
    assert len(geometry.resolve_ref(EntityRef("face", face))) == 2
    assert geometry.validate_topology() == ()


def test_trim_and_hole_keep_face_identity_and_valid_inner_loops(
    rectangle: tuple[GeometryModel, int, tuple[int, ...], tuple[int, ...]],
) -> None:
    geometry, face, _vertices, _edges = rectangle
    original = EntityRef("face", face)

    returned, arcs = punch_hole(geometry, face, (2.0, 1.0, 0.0), 0.25)

    assert returned == face
    assert geometry.entity_ref("face", face) == original
    assert len(arcs) == 4
    assert len(geometry.faces[face].holes) == 1
    assert geometry.validate_topology() == ()

    with pytest.raises(GeometryError, match="continuous"):
        trim_face(
            geometry,
            face,
            (
                (
                    OrientedEdge(arcs[0], True),
                    OrientedEdge(arcs[2], True),
                    OrientedEdge(arcs[3], True),
                ),
            ),
        )


def test_splitting_a_hole_edge_rewrites_the_inner_loop_and_lineage() -> None:
    geometry = GeometryModel()
    face = geometry.add_plate(
        geometry.add_points(
            ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 4.0, 0.0), (0.0, 4.0, 0.0))
        )
    )
    _face, arcs = punch_hole(geometry, face, (2.0, 2.0, 0.0), 0.5)
    original = EntityRef("edge", arcs[0])

    _point, halves = geometry.split_edge(original.id, 0.5)

    inner_edges = tuple(item.edge for item in geometry.faces[face].holes[0])
    assert original.id not in geometry.edges
    assert set(halves) <= set(inner_edges)
    assert original.id not in inner_edges
    assert set(geometry.resolve_ref(original)) == {
        EntityRef("edge", halves[0]),
        EntityRef("edge", halves[1]),
    }
    assert geometry.validate_topology() == ()


def test_diagonal_face_split_supports_triangles_with_evaluable_plane() -> None:
    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 4.0, 0.0), (0.0, 4.0, 0.0))
    )
    face = geometry.add_plate(vertices)

    divider, made = split_face_between(geometry, face, vertices[0], vertices[2])

    assert len(made) == 2
    assert all(len(geometry.faces[item].loop) == 3 for item in made)
    assert all(isinstance(geometry.faces[item].surface, Plane) for item in made)
    assert len(geometry.faces_using_edge(divider)) == 2
    assert all(np.all(np.isfinite(geometry.face_point(item, 0.2, 0.2))) for item in made)
    assert geometry.validate_topology() == ()


def test_face_split_assigns_intact_hole_and_rolls_back_crossing_cut() -> None:
    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 4.0, 0.0), (0.0, 4.0, 0.0))
    )
    face = geometry.add_plate(vertices)
    punch_hole(geometry, face, (1.0, 2.0, 0.0), 0.25)

    _divider, made = split_face_at(geometry, face, 0, 0.5)

    assert sorted(len(geometry.faces[item].holes) for item in made) == [0, 1]
    assert geometry.validate_topology() == ()

    crossing = GeometryModel()
    crossing_vertices = crossing.add_points(
        ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 4.0, 0.0), (0.0, 4.0, 0.0))
    )
    crossing_face = crossing.add_plate(crossing_vertices)
    punch_hole(crossing, crossing_face, (2.0, 2.0, 0.0), 0.25)
    before = to_dict(crossing)

    with pytest.raises(GeometryError, match="intersects or touches hole"):
        split_face_at(crossing, crossing_face, 0, 0.5)

    assert to_dict(crossing) == before
    assert crossing.validate_topology() == ()


def test_invalid_trim_and_overlapping_punch_are_atomic() -> None:
    geometry = GeometryModel()
    face = geometry.add_plate(
        geometry.add_points(
            ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 4.0, 0.0), (0.0, 4.0, 0.0))
        )
    )
    outside = geometry.add_points(
        ((5.0, 5.0, 0.0), (6.0, 5.0, 0.0), (5.5, 6.0, 0.0))
    )
    outside_loop = geometry.order_loop(geometry.add_polyline(outside, close=True))

    with pytest.raises(GeometryError, match="not strictly inside"):
        trim_face(geometry, face, (outside_loop,))

    assert geometry.faces[face].holes == ()
    punch_hole(geometry, face, (2.0, 2.0, 0.0), 0.4)
    before = to_dict(geometry)

    with pytest.raises(GeometryError, match="overlap|does not fit"):
        punch_hole(geometry, face, (2.6, 2.0, 0.0), 0.4)

    assert to_dict(geometry) == before
    assert geometry.validate_topology() == ()


def test_operations_reject_invalid_targets_and_parameters(
    rectangle: tuple[GeometryModel, int, tuple[int, ...], tuple[int, ...]],
) -> None:
    geometry, face, _vertices, _edges = rectangle
    with pytest.raises(GeometryError, match="axis must be"):
        split_face_at(geometry, face, 3, 0.5)
    with pytest.raises(GeometryError, match="strictly between"):
        split_face_at(geometry, face, 0, 0.0)
    with pytest.raises(GeometryError, match="finite 4x4"):
        transform(geometry, np.eye(3))
    with pytest.raises(GeometryError, match="at least one entity"):
        closest_point(GeometryModel(), (0.0, 0.0, 0.0))
