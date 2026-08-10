"""Neutral engineering generators and deterministic layout helpers."""

from __future__ import annotations

import numpy as np
import pytest

from anygeometry import Cone, Cylinder, EntityRef, GeometryError, GeometryModel, Plane
from anygeometry.generators import (
    bulkhead,
    cone,
    cylinder,
    frame,
    girder,
    plate,
    shell,
    stiffened_panel,
    stiffener,
)
from anygeometry.generators.layout import (
    bay_ranges,
    centered_bay_breaks,
    centered_member_positions,
    cleanup_axis,
    closed_loop_member_count,
    positive_spacing,
    symmetric_samples,
)


def _assert_valid_groups(geometry: GeometryModel) -> None:
    assert geometry.validate_topology() == ()
    keys = geometry.entity_keys()
    for name in geometry.groups:
        assert all((reference.kind, reference.id) in keys for reference in geometry.group(name))


def test_layout_preserves_spacing_and_symmetric_remainders() -> None:
    assert positive_spacing(None) == 0.0
    assert positive_spacing(True) == 0.0
    assert positive_spacing(np.bool_(True)) == 0.0
    assert positive_spacing(float("inf")) == 0.0
    assert centered_member_positions(10.0, 3.0) == (0.5, 3.5, 6.5, 9.5)
    assert centered_member_positions(12.0, 3.0, include_ends=True) == (
        0.0,
        3.0,
        6.0,
        9.0,
        12.0,
    )
    assert centered_member_positions(2.0, 3.0) == (1.0,)
    assert centered_member_positions(2.0, 3.0, fallback_midpoint=False) == ()
    assert centered_bay_breaks(10.0, 3.0) == (0.0, 0.5, 3.5, 6.5, 9.5, 10.0)
    assert cleanup_axis((-1.0, 0.0, 2.0, 2.0, 5.0), 4.0) == (0.0, 2.0, 4.0)
    assert bay_ranges(10.0, (3.0, 7.0), 0.2) == (
        (0.0, 2.9),
        (3.1, 6.9),
        (7.1, 10.0),
    )
    assert closed_loop_member_count(2.0 * np.pi, np.pi / 2.0) == 4
    assert closed_loop_member_count(True, 1.0) == 0
    assert closed_loop_member_count(1.0, False) == 0
    assert closed_loop_member_count(np.bool_(True), 1.0) == 0
    assert symmetric_samples((0.0, 1.0, 2.0, 3.0, 4.0), 3) == (0.0, 2.0, 4.0)


def test_plate_has_exact_surface_boundaries_and_semantic_group() -> None:
    geometry = plate(
        4.0,
        3.0,
        origin=(1.0, 2.0, 3.0),
        u_direction=(0.0, 1.0, 0.0),
        v_direction=(0.0, 0.0, 1.0),
        semantic_group="bulkhead",
    )

    assert (len(geometry.vertices), len(geometry.edges), len(geometry.faces)) == (4, 4, 1)
    assert geometry.group("shell") == geometry.group("plate") == geometry.group("bulkhead")
    assert len(geometry.group("boundaries")) == 4
    face = geometry.faces[geometry.group("plate")[0].id]
    assert isinstance(face.surface, Plane)
    assert face.surface.evaluate(1.0, 1.0) == pytest.approx((1.0, 6.0, 6.0))
    _assert_valid_groups(geometry)


def test_stiffened_panel_is_fragmented_on_shared_semantic_member_edges() -> None:
    geometry = stiffened_panel(
        4.0,
        3.0,
        longitudinal_spacing=1.0,
        transverse_spacing=2.0,
        semantic_group="deck",
    )

    assert (len(geometry.vertices), len(geometry.edges), len(geometry.faces)) == (12, 17, 6)
    assert len(geometry.group("longitudinal_stiffeners")) == 4
    assert len(geometry.group("transverse_stiffeners")) == 3
    assert len(geometry.group("boundaries")) == 10
    assert geometry.group("deck") == geometry.group("shell")
    assert all(
        len(geometry.faces_using_edge(reference.id)) == 2
        for reference in geometry.group("longitudinal_stiffeners")
    )
    _assert_valid_groups(geometry)


def test_skew_panel_classifies_internal_members_by_topology() -> None:
    u_direction = np.asarray((1.0, 0.0, 0.0))
    v_direction = np.asarray((1.0, 1.0, 0.0))
    geometry = stiffened_panel(
        4.0,
        3.0,
        longitudinal_spacing=1.0,
        transverse_spacing=2.0,
        origin=(2.0, -1.0, 4.0),
        u_direction=u_direction,
        v_direction=v_direction,
        semantic_group="deck",
    )

    assert len(geometry.group("longitudinal_stiffeners")) == 4
    assert len(geometry.group("transverse_stiffeners")) == 3
    assert geometry.group("deck") == geometry.group("shell")
    for name, expected_direction in (
        ("longitudinal_stiffeners", u_direction),
        ("transverse_stiffeners", v_direction),
    ):
        expected_direction /= np.linalg.norm(expected_direction)
        for reference in geometry.group(name):
            edge = geometry.edges[reference.id]
            delta = geometry.vertex_position(edge.end) - geometry.vertex_position(
                edge.start
            )
            delta /= np.linalg.norm(delta)
            assert abs(float(delta @ expected_direction)) == pytest.approx(1.0)
            assert len(geometry.faces_using_edge(reference.id)) == 2
    _assert_valid_groups(geometry)


@pytest.mark.parametrize(
    ("builder", "args", "surface_type"),
    (
        (cylinder, (2.0, 5.0), Cylinder),
        (cone, (2.0, 1.5, 5.0), Cone),
    ),
)
def test_revolved_generators_are_deterministic_and_group_real_shared_edges(
    builder: object,
    args: tuple[float, ...],
    surface_type: type[object],
) -> None:
    geometry = builder(  # type: ignore[operator]
        *args,
        circumferential_segments=8,
        longitudinal_spacing=2.0,
        ring_spacing=2.5,
    )

    assert (len(geometry.vertices), len(geometry.edges), len(geometry.faces)) == (48, 40, 16)
    assert len(geometry.group("shell")) == 16
    assert len(geometry.group("bottom")) == 8
    assert len(geometry.group("top")) == 8
    assert len(geometry.group("boundaries")) == 16
    assert len(geometry.group("longitudinal_stiffeners")) == 16
    assert len(geometry.group("ring_stiffeners")) == 8
    assert all(isinstance(face.surface, surface_type) for face in geometry.faces.values())
    assert all(
        len(geometry.faces_using_edge(reference.id)) == 2
        for reference in geometry.group("ring_stiffeners")
    )
    assert all(
        np.linalg.norm(geometry.face_point(face.id, 0.5, 0.5)[:2]) > 1.4
        for face in geometry.faces.values()
    )
    _assert_valid_groups(geometry)


def test_true_cone_apex_is_supported_without_degenerate_topology() -> None:
    geometry = cone(2.0, 0.0, 3.0, circumferential_segments=8)

    assert geometry.group("shell")
    top = geometry.group("top")
    assert len(top) == 1
    assert top[0].kind == "vertex"
    assert geometry.validate_topology() == ()
    assert geometry.vertex_position(top[0].id) == pytest.approx(
        (0.0, 0.0, 3.0)
    )


def test_revolved_generator_uses_projected_transformed_basis_at_apex() -> None:
    geometry = cone(
        0.0,
        2.0,
        3.0,
        circumferential_segments=6,
        origin=(1.0, 2.0, 3.0),
        axis=(0.0, 2.0, 0.0),
        radial_direction=(2.0, 1.0, 0.0),
    )

    bottom = geometry.group("bottom")
    assert bottom == (EntityRef("vertex", 1),)
    assert geometry.vertex_position(bottom[0].id) == pytest.approx((1.0, 2.0, 3.0))
    first_surface = geometry.faces[geometry.group("shell")[0].id].surface
    assert isinstance(first_surface, Cone)
    assert first_surface.evaluate(0.0, 0.0) == pytest.approx((1.0, 2.0, 3.0))
    assert first_surface.evaluate(0.0, 1.0) == pytest.approx((3.0, 5.0, 3.0))
    assert all(np.all(np.isfinite(vertex.position)) for vertex in geometry.vertices.values())
    _assert_valid_groups(geometry)


def test_member_and_ring_entity_counts_are_repeatable_at_spacing_thresholds() -> None:
    arguments = dict(
        radius=3.0,
        height=10.0,
        circumferential_segments=3,
        longitudinal_spacing=np.pi,
        ring_spacing=3.0,
    )
    first = cylinder(**arguments)
    second = cylinder(**arguments)

    assert (len(first.vertices), len(first.edges), len(first.faces)) == (72, 66, 30)
    assert len(first.group("longitudinal_stiffeners")) == 30
    assert len(first.group("ring_stiffeners")) == 24
    assert len(first.group("bottom")) == len(first.group("top")) == 6
    assert first.groups == second.groups
    assert np.asarray(
        [vertex.position for vertex in first.vertices.values()]
    ) == pytest.approx(
        np.asarray([vertex.position for vertex in second.vertices.values()])
    )
    _assert_valid_groups(first)


def test_named_alias_generators_only_add_geometry_meaning() -> None:
    shell_model = shell(2.0, 1.0)
    bulkhead_model = bulkhead(2.0, 1.0)
    frame_model = frame(2.0, 1.0)
    girder_model = girder(3.0, origin=(1.0, 0.0, 0.0), direction=(0.0, 1.0, 0.0))
    stiffener_model = stiffener(3.0)

    assert shell_model.group("shell")
    assert bulkhead_model.group("bulkhead") == bulkhead_model.group("shell")
    assert frame_model.group("frame") == frame_model.group("shell")
    assert girder_model.group("girder") == girder_model.group("girders")
    assert girder_model.group("girders") == (EntityRef("edge", 1),)
    assert stiffener_model.group("stiffener") == stiffener_model.group("stiffeners")
    assert stiffener_model.group("stiffeners") == (EntityRef("edge", 1),)
    assert not stiffener_model.group("girder")
    assert not stiffener_model.group("girders")
    assert girder_model.vertex_position(2) == pytest.approx((1.0, 3.0, 0.0))


@pytest.mark.parametrize(
    ("call", "message"),
    (
        (lambda: plate(0.0, 1.0), "length must"),
        (lambda: plate(1.0, 1.0, u_direction=(1.0, 0.0, 0.0), v_direction=(2.0, 0.0, 0.0)), "independent"),
        (lambda: cylinder(-1.0, 2.0), "radius must"),
        (lambda: cylinder(1.0, 2.0, circumferential_segments=2), "at least 3"),
        (lambda: cone(0.0, 0.0, 2.0), "at least one radius"),
        (lambda: girder(1.0, direction=(0.0, 0.0, 0.0)), "non-zero"),
    ),
)
def test_generators_reject_invalid_engineering_dimensions(
    call: object,
    message: str,
) -> None:
    with pytest.raises(GeometryError, match=message):
        call()  # type: ignore[operator]


@pytest.mark.parametrize(
    "call",
    (
        lambda: plate(True, 1.0),
        lambda: plate(np.asarray(True), 1.0),
        lambda: plate(1.0 + 0.0j, 1.0),
        lambda: plate(np.asarray([1.0]), 1.0),
        lambda: plate(1.0, float("nan")),
        lambda: plate(1.0, 1.0, origin=(False, 0.0, 0.0)),
        lambda: plate(1.0, 1.0, u_direction=(1.0, np.inf, 0.0)),
        lambda: plate(1.0, 1.0, semantic_group=""),
        lambda: stiffened_panel(1.0, 1.0, longitudinal_spacing=True),
        lambda: stiffened_panel(
            1.0,
            1.0,
            longitudinal_spacing=0.5,
            transverse_spacing=np.inf,
        ),
        lambda: cylinder(True, 2.0),
        lambda: cylinder(1.0, np.nan),
        lambda: cylinder(1.0, 2.0, circumferential_segments=np.inf),
        lambda: cylinder(1.0, 2.0, circumferential_segments="8"),
        lambda: cylinder(1.0, 2.0, circumferential_segments=np.bool_(True)),
        lambda: cylinder(1.0, 2.0, longitudinal_spacing=False),
        lambda: cylinder(1.0, 2.0, ring_spacing=np.nan),
        lambda: cylinder(1.0, 2.0, axis=(0.0, True, 1.0)),
        lambda: cone(np.bool_(False), 1.0, 2.0),
        lambda: girder(False),
        lambda: girder(1.0, origin=(0.0, np.bool_(False), 0.0)),
    ),
)
def test_generators_reject_nonfinite_and_boolean_numeric_inputs(call: object) -> None:
    with pytest.raises(GeometryError):
        call()  # type: ignore[operator]
