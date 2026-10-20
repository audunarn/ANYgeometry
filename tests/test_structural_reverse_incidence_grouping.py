"""Reverse-incidence grouping for shared structural edges and faces."""

from __future__ import annotations

import pytest

from anygeometry import GeometryError, GeometryModel
from anygeometry.entities import OrientedEdge


def _multi_owner_model() -> tuple[GeometryModel, int, int, int, int, int]:
    """Rectangle plate with one owned sheet and one owned boundary member."""
    model = GeometryModel()
    points = model.add_points(((0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)))
    face = model.add_plate(points)
    part = model.add_part(name="multi-owner")
    sheet = model.add_sheet((face,), part_id=part)
    edge = model.faces[face].loop[0].edge
    member = model.add_member((edge,), part_id=part)
    return model, part, sheet, member, face, edge


def _expected_member_uses(model: GeometryModel, edge: int) -> set[int]:
    return {use.id for use in model.member_edge_uses.values() if use.edge_id == edge}


def _expected_face_uses(model: GeometryModel, face: int) -> set[int]:
    return {use.id for use in model.face_uses.values() if use.face_id == face}


def _expected_coedges(model: GeometryModel, edge: int) -> set[int]:
    return {coedge.id for coedge in model.coedges.values() if coedge.edge_id == edge}


def _snapshot(model: GeometryModel, part: int) -> tuple[int, int, tuple[int, ...]]:
    return (model.revision, len(model.members), tuple(model.parts[part].member_ids))


def _corrupt(
    model: GeometryModel, index: str, mode: str, edge: int, face: int
) -> str:
    if index == "member":
        bucket = model._edge_member_uses[edge]
        expected = _expected_member_uses(model, edge)
        message = rf"edge {edge} member reverse incidence is stale"
    elif index == "face":
        bucket = model._face_structural_uses[face]
        expected = _expected_face_uses(model, face)
        message = rf"face {face} structural reverse incidence is stale"
    else:
        bucket = model._edge_coedges[edge]
        expected = _expected_coedges(model, edge)
        message = rf"edge {edge} coedge reverse incidence is stale"
    if mode == "missing":
        bucket.discard(min(expected))
    else:
        bucket.add(max(expected) + 1000)
    return message


def _adjacent_sheet(model: GeometryModel, part: int, edge: int) -> tuple[int, int]:
    axis = model.edges[edge]
    lower = model.add_points(((0, -1, 0), (2, -1, 0)))
    vertices = (axis.end, axis.start, *lower)
    lines = model.add_lines(tuple(zip(vertices[1:], vertices[2:] + vertices[:1])))
    face = model.add_face_from_loop(
        (OrientedEdge(edge, False), *(OrientedEdge(line, True) for line in lines)),
        (0, 1, 2, 3),
    )
    return model.add_sheet((face,), part_id=part), face


def test_shared_edge_groups_member_and_coedge_owners() -> None:
    model, part, sheet, member, face, edge = _multi_owner_model()
    second_member = model.add_member((edge,), part_id=part)
    second_sheet, second_face = _adjacent_sheet(model, part, edge)

    assert model._edge_member_uses[edge] == _expected_member_uses(model, edge)
    assert len(model._edge_member_uses[edge]) == 2
    assert model._face_structural_uses[face] == _expected_face_uses(model, face)
    assert len(model._face_structural_uses[face]) == 1
    assert model._face_structural_uses[second_face] == _expected_face_uses(model, second_face)
    assert model._edge_coedges[edge] == _expected_coedges(model, edge)
    assert len(model._edge_coedges[edge]) == 2
    assert second_member in model.parts[part].member_ids
    assert second_sheet in model.parts[part].sheet_ids


def test_multi_owner_closure_accepts_further_member_without_false_stale() -> None:
    model, part, sheet, member, face, edge = _multi_owner_model()
    model.add_member((edge,), part_id=part)
    _adjacent_sheet(model, part, edge)
    third_member = model.add_member((edge,), part_id=part)

    assert third_member in model.parts[part].member_ids
    assert len(model._edge_member_uses[edge]) == 3


@pytest.mark.parametrize("index", ["member", "face", "coedge"])
@pytest.mark.parametrize("mode", ["missing", "extra"])
def test_corrupt_reverse_incidence_entry_is_detected_and_rolled_back(
    index: str, mode: str
) -> None:
    model, part, sheet, member, face, edge = _multi_owner_model()
    before = _snapshot(model, part)
    message = _corrupt(model, index, mode, edge, face)

    with pytest.raises(GeometryError, match=message):
        model.add_member((edge,), part_id=part)

    assert _snapshot(model, part) == before
