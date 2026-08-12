"""Focused radial ordering and curved-support split contracts."""

from __future__ import annotations

import numpy as np
import pytest

from anygeometry import (
    Cone,
    GeometryError,
    GeometryModel,
    OrientedEdge,
    Plane,
    split_face_at,
)


_RADIAL_POINTS = {
    "positive_y": (0.5, 1.0, 0.0),
    "positive_z": (0.5, 0.0, 1.0),
    "negative_y": (0.5, -1.0, 0.0),
    "negative_z": (0.5, 0.0, -1.0),
}


def _radial_model(order: tuple[str, ...]) -> tuple[GeometryModel, int]:
    model = GeometryModel()
    axis_start, axis_end = model.add_points(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    axis = model.add_line(axis_start, axis_end)
    part = model.add_part(name="radial-order")
    for label in order:
        third = model.add_point(*_RADIAL_POINTS[label])
        second = model.add_line(axis_end, third)
        third_edge = model.add_line(third, axis_start)
        surface = Plane(
            model.vertex_position(axis_start),
            model.vertex_position(axis_end) - model.vertex_position(axis_start),
            model.vertex_position(third) - model.vertex_position(axis_start),
        )
        face = model.add_face_from_loop(
            (
                OrientedEdge(axis, True),
                OrientedEdge(second, True),
                OrientedEdge(third_edge, True),
            ),
            surface=surface,
        )
        model.update_face_metadata(face, radial_label=label)
        model.add_sheet((face,), part_id=part)
    return model, axis


def _radial_labels(model: GeometryModel, edge_id: int) -> tuple[str, ...]:
    return tuple(
        str(model.faces[model.face_uses[use_id].face_id].metadata["radial_label"])
        for use_id in model.radial_face_uses(edge_id)
    )


def test_radial_face_use_order_is_geometric_and_insertion_independent() -> None:
    labels = tuple(_RADIAL_POINTS)
    first, first_axis = _radial_model(labels)
    second, second_axis = _radial_model(tuple(reversed(labels)))

    first_order = _radial_labels(first, first_axis)
    second_order = _radial_labels(second, second_axis)

    assert first_order == second_order
    assert first_order == _radial_labels(first, first_axis)
    assert set(first_order) == set(labels)
    assert set(first.radial_face_uses(first_axis)) == set(
        first.nonmanifold_face_uses(first_axis)
    )
    assert first.validate_topology() == ()
    assert second.validate_topology() == ()


def test_split_cone_face_preserves_authoritative_support() -> None:
    model = GeometryModel()
    surface = Cone(
        np.zeros(3),
        np.asarray((0.0, 0.0, 1.0)),
        np.asarray((1.0, 0.0, 0.0)),
        2.0,
        1.0,
        3.0,
        0.0,
        0.5 * np.pi,
    )
    lower = model.add_points(tuple(surface.evaluate(u, 0.0) for u in (0.0, 0.5, 1.0)))
    upper = model.add_points(tuple(surface.evaluate(u, 1.0) for u in (0.0, 0.5, 1.0)))
    bottom = model.add_arc(*lower)
    far_side = model.add_line(lower[-1], upper[-1])
    top = model.add_arc(*upper)
    near_side = model.add_line(lower[0], upper[0])
    face = model.add_face_from_loop(
        (
            OrientedEdge(bottom, True),
            OrientedEdge(far_side, True),
            OrientedEdge(top, False),
            OrientedEdge(near_side, False),
        ),
        (0, 1, 2, 3),
        surface=surface,
    )

    divider, children = split_face_at(model, face, axis=0, fraction=0.5)

    assert all(model.faces[item].surface is surface for item in children)
    assert len(model.faces_using_edge(divider)) == 2
    sampled = model.sample_edge(divider, np.linspace(0.0, 1.0, 9))
    expected = np.asarray(
        [surface.evaluate(0.5, value) for value in np.linspace(0.0, 1.0, 9)]
    )
    assert min(
        float(np.linalg.norm(sampled - expected, axis=1).max()),
        float(np.linalg.norm(sampled[::-1] - expected, axis=1).max()),
    ) <= 1.0e-12
    assert model.validate_topology() == ()


def test_unsupported_general_curved_split_is_atomic() -> None:
    class WarpedSurface:
        def evaluate(self, u: float, v: float) -> np.ndarray:
            u = float(u)
            v = float(v)
            return np.asarray((u, v, u * (1.0 - u) * v * (1.0 - v)))

        def local_uv(self, point: object) -> tuple[float, float]:
            value = np.asarray(point, dtype=float)
            return float(value[0]), float(value[1])

    model = GeometryModel()
    face = model.add_plate(
        model.add_points(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
        )
    )
    model.set_face_surface(face, WarpedSurface())  # type: ignore[arg-type]
    keys_before = model.entity_keys()
    vertex_ids_before = set(model.vertices)
    edge_ids_before = set(model.edges)
    face_before = model.faces[face]
    support_before = np.asarray(
        [model.face_support_point(face, 0.37, value) for value in (0.0, 0.25, 0.5, 1.0)]
    )
    history_before = model.replacement_history()
    revision_before = model.revision
    change_before = model.last_change_set

    with pytest.raises(GeometryError, match="unsupported support divider"):
        split_face_at(model, face, axis=0, fraction=0.37, tolerance=1.0e-12)

    assert model.entity_keys() == keys_before
    assert set(model.vertices) == vertex_ids_before
    assert set(model.edges) == edge_ids_before
    assert model.faces[face].loop == face_before.loop
    assert model.faces[face].corners == face_before.corners
    assert np.array_equal(
        np.asarray(
            [
                model.face_support_point(face, 0.37, value)
                for value in (0.0, 0.25, 0.5, 1.0)
            ]
        ),
        support_before,
    )
    assert model.replacement_history() == history_before
    assert model.revision == revision_before
    assert model.last_change_set == change_before
    assert model.validate_topology() == ()
