"""Bounded owner-level checks for explicit multi-face sheet orientation."""

from __future__ import annotations

from copy import deepcopy

import pytest

from anygeometry import GeometryError, GeometryModel, Orientation, OrientedEdge
from anygeometry.serialization import to_dict


def _same_sense_adjacent_faces() -> tuple[GeometryModel, int, int, int]:
    geometry = GeometryModel()
    p0, p1, p2, p3, p4, p5 = geometry.add_points(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
        )
    )
    lower_left = geometry.add_line(p0, p1)
    shared = geometry.add_line(p1, p2)
    upper_left = geometry.add_line(p2, p3)
    left = geometry.add_line(p3, p0)
    upper_right = geometry.add_line(p2, p5)
    right = geometry.add_line(p5, p4)
    lower_right = geometry.add_line(p4, p1)
    first = geometry.add_face_from_loop(
        tuple(
            OrientedEdge(edge_id, True)
            for edge_id in (lower_left, shared, upper_left, left)
        )
    )
    # This loop is geometrically valid but traverses the common edge in the
    # same direction as the first face.  Face-use orientation, rather than a
    # geometric-normal rewrite, makes the structural Sheet coherent.
    second = geometry.add_face_from_loop(
        tuple(
            OrientedEdge(edge_id, True)
            for edge_id in (shared, upper_right, right, lower_right)
        )
    )
    return geometry, first, second, shared


def test_add_sheet_accepts_explicit_face_use_orientations_without_geometry_edits() -> None:
    geometry, first, second, shared = _same_sense_adjacent_faces()
    faces_before = deepcopy(to_dict(geometry)["faces"])

    sheet_id = geometry.add_sheet(
        (first, second),
        orientations=(Orientation.FORWARD, Orientation.REVERSED),
        name="joined",
    )

    sheet = geometry.sheets[sheet_id]
    uses = [geometry.face_uses[use_id] for use_id in sheet.face_use_ids]
    assert [(use.face_id, use.orientation) for use in uses] == [
        (first, Orientation.FORWARD),
        (second, Orientation.REVERSED),
    ]
    effective = []
    for use in uses:
        coedge = next(
            geometry.coedges[coedge_id]
            for coedge_id in use.coedge_ids
            if geometry.coedges[coedge_id].edge_id == shared
        )
        effective.append(int(use.orientation) * int(coedge.orientation))
    assert effective[0] == -effective[1]
    assert to_dict(geometry)["faces"] == faces_before
    assert geometry.validate_topology() == ()
    assert geometry._validate_structural() == ()  # noqa: SLF001


@pytest.mark.parametrize(
    ("orientations", "message"),
    [
        ((Orientation.FORWARD,), "one entry per face"),
        ((Orientation.FORWARD, 0), "FORWARD or REVERSED"),
        ((Orientation.FORWARD, True), "FORWARD or REVERSED"),
    ],
)
def test_add_sheet_rejects_invalid_orientation_vectors_before_owner_allocation(
    orientations, message: str
) -> None:
    geometry, first, second, _shared = _same_sense_adjacent_faces()
    before = to_dict(geometry)

    with pytest.raises(GeometryError, match=message):
        geometry.add_sheet((first, second), orientations=orientations)

    assert to_dict(geometry) == before
