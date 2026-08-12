"""Versioned, deterministic geometry serialization qualification."""

from __future__ import annotations

from copy import deepcopy
import json

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
    from_dict,
    punch_hole,
    read_geometry,
    to_dict,
    write_geometry,
)
from anygeometry.curves import Arc, Spline
from anygeometry.generators import plate


def _geometry_with_surface(surface: object) -> GeometryModel:
    """Build topology that follows the surface used by a codec fixture."""

    if isinstance(surface, CoonsSurface) and not surface.has_boundaries:
        geometry = plate(2.0, 1.0)
        geometry.set_face_surface(next(iter(geometry.faces)), surface)
        return geometry

    assert isinstance(surface, (CoonsSurface, Plane, Cylinder, Cone, RuledSurface))
    geometry = GeometryModel()
    if isinstance(surface, (Cylinder, Cone)):
        lower = tuple(
            geometry.add_point(*surface.evaluate(u, 0.0))
            for u in (0.0, 0.5, 1.0)
        )
        upper = tuple(
            geometry.add_point(*surface.evaluate(u, 1.0))
            for u in (0.0, 0.5, 1.0)
        )
        bottom = geometry.add_arc(lower[0], lower[1], lower[2])
        end = geometry.add_line(lower[2], upper[2])
        top = geometry.add_arc(upper[0], upper[1], upper[2])
        start = geometry.add_line(lower[0], upper[0])
        geometry.add_face_from_loop(
            (
                OrientedEdge(bottom, True),
                OrientedEdge(end, True),
                OrientedEdge(top, False),
                OrientedEdge(start, False),
            ),
            (0, 1, 2, 3),
            surface=surface,
        )
        return geometry

    corners = tuple(
        geometry.add_point(*surface.evaluate(u, v))
        for u, v in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    )
    face = geometry.add_plate(corners)
    geometry.set_face_surface(face, surface)
    return geometry


def _complex_geometry() -> GeometryModel:
    geometry = plate(4.0, 3.0, semantic_group="deck")
    face_ref = geometry.group("deck")[0]
    geometry.update_face_metadata(
        face_ref.id, label="main deck", revision=2
    )
    geometry.tag(face_ref, "selected", "structural")
    punch_hole(geometry, face_ref.id, (2.0, 1.5, 0.0), 0.25)

    boundary = geometry.group("boundaries")[0]
    geometry.add_to_group("loaded_boundary", (boundary,))
    geometry.tag(boundary, "pressure")
    geometry.split_edge(boundary.id, 0.4)

    start, control_a, control_b, end = geometry.add_points(
        ((0.0, 0.0, 2.0), (1.0, 1.0, 2.0), (2.0, 1.0, 2.0), (3.0, 0.0, 2.0))
    )
    spline = geometry.add_spline(start, (control_a, control_b), end)
    geometry.add_to_group("beam_axes", (EntityRef("edge", spline),))
    return geometry


def test_round_trip_preserves_complete_document_and_id_allocation() -> None:
    geometry = _complex_geometry()
    document = to_dict(geometry)

    restored = from_dict(deepcopy(document))

    assert to_dict(restored) == document
    assert restored.validate_topology() == ()
    assert restored.group("deck") == geometry.group("deck")
    assert restored.group("loaded_boundary") == geometry.group("loaded_boundary")
    assert restored.replacement_history() == geometry.replacement_history()
    assert any(isinstance(edge.curve, Arc) for edge in restored.edges.values())
    assert any(isinstance(edge.curve, Spline) for edge in restored.edges.values())
    next_vertex = restored.id_state()["vertex"]
    assert restored.add_point(9.0, 9.0, 9.0) == next_vertex


def test_json_and_gzip_files_are_deterministic_and_equivalent(tmp_path) -> None:
    geometry = _complex_geometry()
    json_path = tmp_path / "geometry.json"
    gzip_path = tmp_path / "geometry.json.gz"

    write_geometry(json_path, geometry)
    first_payload = json_path.read_text(encoding="utf-8")
    write_geometry(json_path, geometry)
    write_geometry(gzip_path, geometry)

    assert json_path.read_text(encoding="utf-8") == first_payload
    assert first_payload.endswith("\n")
    assert json.loads(first_payload)["schema"] == "anygeometry"
    assert to_dict(read_geometry(json_path)) == to_dict(geometry)
    assert to_dict(read_geometry(gzip_path)) == to_dict(geometry)


@pytest.mark.parametrize(
    "surface",
    (
        CoonsSurface(),
        CoonsSurface(
            np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
            np.asarray(((1.0, 0.0, 0.0), (1.0, 1.0, 0.0))),
            np.asarray(((0.0, 1.0, 0.0), (1.0, 1.0, 0.0))),
            np.asarray(((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        ),
        Plane(np.zeros(3), np.asarray((2.0, 0.0, 0.0)), np.asarray((0.0, 1.0, 0.0))),
        Cylinder(
            np.zeros(3),
            np.asarray((0.0, 0.0, 1.0)),
            np.asarray((1.0, 0.0, 0.0)),
            2.0,
            3.0,
            0.2,
            1.5,
        ),
        Cone(
            np.zeros(3),
            np.asarray((0.0, 0.0, 1.0)),
            np.asarray((1.0, 0.0, 0.0)),
            2.0,
            1.0,
            3.0,
            0.2,
            1.5,
        ),
        RuledSurface(
            np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
            np.asarray(((0.0, 1.0, 1.0), (1.0, 1.0, 1.0))),
        ),
    ),
)
def test_every_surface_type_round_trips(surface: object) -> None:
    geometry = _geometry_with_surface(surface)

    document = to_dict(geometry)
    restored = from_dict(document)

    assert to_dict(restored) == document
    assert type(next(iter(restored.faces.values())).surface) is type(surface)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda data: data.update(schema="another-package"), "not an ANYgeometry"),
        (lambda data: data.update(version=999), "unsupported ANYgeometry version"),
        (lambda data: data["id_state"].update(vertex=1), "counter would reuse"),
        (lambda data: data["edges"][0].update(start=999), "missing vertex"),
        (
            lambda data: data["groups"].update(orphan=[["face", 999]]),
            r"reference[s]? missing entity",
        ),
        (
            lambda data: data["tags"].append(
                {"entity": ["face", 999], "values": ["orphan"]}
            ),
            r"reference[s]? missing entity",
        ),
        (
            lambda data: data["replacement_history"].append(
                {"old": ["edge", 999], "new": [["edge", 1000]]}
            ),
            "replacement history references missing entity",
        ),
    ),
)
def test_malformed_or_dangling_documents_fail_closed(
    mutation: object,
    message: str,
) -> None:
    document = to_dict(_complex_geometry())
    mutation(document)  # type: ignore[operator]

    with pytest.raises(GeometryError, match=message):
        from_dict(document)


def test_replacement_history_cycles_fail_closed() -> None:
    document = to_dict(_complex_geometry())
    first_replacement = document["replacement_history"][0]
    old_reference = first_replacement["old"]
    replacement_reference = first_replacement["new"][0]
    document["replacement_history"].append(
        {"old": replacement_reference, "new": [old_reference]}
    )

    with pytest.raises(GeometryError, match="replacement history contains a cycle"):
        from_dict(document)


def test_fractional_ids_and_two_edge_outer_faces_fail_closed() -> None:
    fractional = to_dict(plate(2.0, 1.0))
    fractional["vertices"][0]["id"] = 1.5
    with pytest.raises(GeometryError, match="vertex ID must be an integer"):
        from_dict(fractional)

    collapsed = to_dict(plate(2.0, 1.0))
    collapsed["faces"][0]["loop"] = [[1, True], [1, False]]
    collapsed["faces"][0]["corners"] = []
    with pytest.raises(GeometryError, match="needs at least 3 edges"):
        from_dict(collapsed)


def test_deserialization_rejects_boundary_surface_divergence() -> None:
    document = to_dict(plate(2.0, 1.0))
    document["vertices"][2]["position"] = [3.0, 2.0, 1.0]

    with pytest.raises(GeometryError, match="inconsistent with its explicit surface"):
        from_dict(document)


def test_construction_rejects_a_self_intersecting_face() -> None:
    geometry = GeometryModel()
    vertices = geometry.add_points(
        (
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
        )
    )
    edges = tuple(
        geometry.add_line(vertices[index], vertices[(index + 1) % 4])
        for index in range(4)
    )
    with pytest.raises(GeometryError, match="self-intersects"):
        geometry.add_face_from_loop(
            tuple(OrientedEdge(edge, True) for edge in edges),
            surface=Plane(
                np.zeros(3),
                np.asarray((1.0, 0.0, 0.0)),
                np.asarray((0.0, 1.0, 0.0)),
            ),
        )


def test_construction_rejects_a_face_inconsistent_with_its_surface() -> None:
    geometry = GeometryModel()
    vertices = geometry.add_points(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        )
    )
    edges = tuple(
        geometry.add_line(vertices[index], vertices[(index + 1) % 4])
        for index in range(4)
    )
    with pytest.raises(GeometryError, match="inconsistent with its explicit surface"):
        geometry.add_face_from_loop(
            tuple(OrientedEdge(edge, True) for edge in edges),
            surface=Plane(
                np.asarray((0.0, 0.0, 1.0)),
                np.asarray((1.0, 0.0, 0.0)),
                np.asarray((0.0, 1.0, 0.0)),
            ),
        )
