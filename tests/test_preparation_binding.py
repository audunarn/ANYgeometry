from dataclasses import replace

import pytest

from anygeometry import (
    GeometryModel, GeometryError, IntersectionBatchPolicy, ConnectionIntent,
    plan_intersections, apply_intersections, has_current_intersection_preparation,
    to_dict, from_dict,
    clone_prepared_geometry,
)


def crossing():
    model = GeometryModel()
    for points in (
        ((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)),
        ((-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1)),
    ):
        model.add_plate(model.add_points(points))
    return model


def prepare(model, operands=None, policy=ConnectionIntent.CONNECT):
    plan = plan_intersections(model, tuple(model.faces) if operands is None else operands, policy=policy)
    return apply_intersections(model, plan, policy=policy)


def test_complete_preparation_receipt_is_read_only_and_covers_descendants():
    model = crossing()
    assert not has_current_intersection_preparation(model)
    application = prepare(model)
    assert len(model.faces) == 4
    before = to_dict(model)
    assert has_current_intersection_preparation(model)
    assert has_current_intersection_preparation(model, face_ids=(next(iter(model.faces)),))
    assert to_dict(model) == before
    assert apply_intersections(model, application.plan, policy=ConnectionIntent.CONNECT).reused
    assert has_current_intersection_preparation(model)


@pytest.mark.parametrize("edit", ["revision", "raw_state", "clone", "load", "restore"])
def test_edit_or_document_reconstruction_requires_new_classification(edit):
    model = crossing()
    original = model.topology_snapshot()
    prepare(model)
    if edit == "revision":
        model.add_point(4, 4, 4)
    elif edit == "raw_state":
        vertex = next(iter(model.vertices))
        model._vertices[vertex] = replace(model.vertices[vertex], position=(9., 9., 9.))
        with pytest.raises(GeometryError, match="invalid topology"):
            has_current_intersection_preparation(model)
        return
    elif edit == "clone":
        model = model.clone(preserve_identity=True)
    elif edit == "load":
        model = from_dict(to_dict(model))
    else:
        model.restore_topology(original)
    assert not has_current_intersection_preparation(model)


def test_partial_or_disabled_face_batch_does_not_certify_material_ownership():
    model = crossing()
    prepare(model, operands=(1,))
    assert not has_current_intersection_preparation(model, face_ids=(1,))
    model = crossing()
    prepare(model, policy=IntersectionBatchPolicy(face_connections=False))
    assert not has_current_intersection_preparation(model)


def test_unchanged_complete_batch_receives_binding_but_invalid_faces_fail():
    model = crossing()
    model.remove_face(2)
    prepare(model)
    assert has_current_intersection_preparation(model)
    with pytest.raises(GeometryError):
        has_current_intersection_preparation(model, face_ids=(999,))


def test_explicit_prepared_copy_carries_only_an_exact_current_binding():
    model = crossing()
    prepare(model)
    before = to_dict(model)
    made = clone_prepared_geometry(model)
    assert made is not model and to_dict(made) == before
    assert has_current_intersection_preparation(made)
    made.add_point(7, 7, 7)
    assert not has_current_intersection_preparation(made)
    assert has_current_intersection_preparation(model) and to_dict(model) == before
    unprepared = crossing()
    other = clone_prepared_geometry(unprepared)
    assert other.model_id != unprepared.model_id
    assert not has_current_intersection_preparation(other)
