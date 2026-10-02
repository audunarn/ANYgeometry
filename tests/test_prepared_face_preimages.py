"""Transient authored identities follow only qualified owner applications."""
from dataclasses import FrozenInstanceError, replace

import pytest

from anygeometry import (EntityRef, GeometryError, GeometryModel, apply_intersections,
    clone_prepared_geometry, from_dict, has_current_intersection_preparation,
    plan_intersections, query_prepared_face_preimages, set_prepared_face_corners,
    to_dict, validate_prepared_face_preimages_binding)


def crossing(third=False):
    model = GeometryModel()
    rows = [((-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0)),
            ((-1,0,-1),(1,0,-1),(1,0,1),(-1,0,1))]
    if third:
        rows.append(((0,-1,-1),(0,1,-1),(0,1,1),(0,-1,1)))
    for points in rows:
        model.add_plate(model.add_points(points))
    return model


def prepare(model, faces=None, **policy):
    from anygeometry import IntersectionBatchPolicy
    controls = IntersectionBatchPolicy(**policy)
    plan = plan_intersections(model, tuple(model.faces) if faces is None else faces, policy=controls)
    return apply_intersections(model,plan,policy=controls)


def assert_exact_map(model, binding, originals):
    assert binding.authored_face_ids == tuple(originals)
    assert dict(binding.source_to_current_faces) == {
        face: tuple(sorted(ref.id for ref in model.resolve_ref(EntityRef('face',face))))
        for face in originals}
    descendants = [face for _, faces in binding.face_descendants for face in faces]
    assert len(descendants) == len(set(descendants))
    assert set(descendants) == set(model.faces)
    validate_prepared_face_preimages_binding(model,binding)


def test_first_application_captures_authored_snapshot_and_query_is_immutable():
    model = crossing()
    original = to_dict(model)
    with pytest.raises(GeometryError,match='complete current preparation'):
        query_prepared_face_preimages(model)
    prepare(model)
    committed = to_dict(model)
    binding = query_prepared_face_preimages(model,face_ids=(next(iter(model.faces)),),
                                            expected_revision=model.revision)
    assert binding.authored_model_id == model.model_id
    assert binding.authored_revision == original['revision']
    assert binding.authored_checksum == original['checksum']['value']
    assert binding.revision == model.revision
    assert binding.source_checksum == committed['checksum']['value']
    assert_exact_map(model,binding,(1,2))
    with pytest.raises(TypeError):
        binding.source_to_current_faces[1] = ()
    with pytest.raises(FrozenInstanceError):
        binding.revision = 0
    assert len(model._intersection_preparation_receipt) == 4
    assert to_dict(model) == committed
    with pytest.raises(GeometryError,match='revision is stale'):
        query_prepared_face_preimages(model,expected_revision=model.revision-1)
    with pytest.raises(GeometryError):
        query_prepared_face_preimages(model,face_ids=(999,))


def test_exact_replay_and_reclassification_keep_original_anchor():
    model = crossing()
    application = prepare(model)
    first = query_prepared_face_preimages(model)
    assert apply_intersections(model,application.plan,policy='connect').reused
    assert query_prepared_face_preimages(model) == first
    assert prepare(model).reused
    assert query_prepared_face_preimages(model) == first


def test_partial_then_complete_application_composes_only_new_replacements():
    model = crossing(third=True)
    original = to_dict(model)
    first = prepare(model,(1,2))
    assert not has_current_intersection_preparation(model)
    with pytest.raises(GeometryError,match='complete current preparation'):
        query_prepared_face_preimages(model)
    assert first.change_set.replacements
    partial_map = model._prepared_face_preimages_receipt[0]
    assert partial_map.authored_face_ids == (1,2,3)
    assert not partial_map.coverage
    second = prepare(model)
    assert second.change_set.replacements
    complete = query_prepared_face_preimages(model)
    assert complete.authored_checksum == original['checksum']['value']
    assert_exact_map(model,complete,(1,2,3))


@pytest.mark.parametrize('edit',('point','raw_metadata','load','clone','restore'))
def test_unknown_or_stale_provenance_never_relabels_fragments_after_new_batch(edit):
    model = crossing()
    prepare(model)
    binding = query_prepared_face_preimages(model)
    if edit == 'point':
        model.add_point(5,5,5)
    elif edit == 'raw_metadata':
        face = next(iter(model.faces))
        model._faces[face] = replace(model.faces[face],metadata={'edited':'outside proof'})
    elif edit == 'load':
        model = from_dict(to_dict(model))
    elif edit == 'clone':
        model = model.clone(preserve_identity=True)
    else:
        snapshot = model.topology_snapshot()
        model.add_point(5,5,5)
        model.restore_topology(snapshot)
    with pytest.raises(GeometryError):
        validate_prepared_face_preimages_binding(model,binding)
    prepare(model)
    assert has_current_intersection_preparation(model)
    with pytest.raises(GeometryError):
        query_prepared_face_preimages(model)


def test_stale_unfragmented_proof_is_not_renewed_by_arbitrary_edits():
    model = GeometryModel()
    model.add_plate(model.add_points(((0,0,0),(1,0,0),(1,1,0),(0,1,0))))
    prepare(model)
    model.add_point(5,5,5)
    prepare(model)
    assert has_current_intersection_preparation(model)
    with pytest.raises(GeometryError,match='stale'):
        query_prepared_face_preimages(model)


def test_qualified_clone_and_corner_edit_preserve_authored_mapping():
    model = GeometryModel()
    face = model.add_plate(model.add_points(((0,0,0),(2,0,0),(2,2,0),(0,2,0))))
    model.split_edge(model.faces[face].loop[0].edge,.5)
    prepare(model)
    source = to_dict(model)
    binding = query_prepared_face_preimages(model)
    copied = clone_prepared_geometry(model)
    validate_prepared_face_preimages_binding(copied,binding)
    observed = []
    def hook(*_):
        try:
            query_prepared_face_preimages(copied)
        except GeometryError:
            observed.append(False)
        else:
            observed.append(True)
    copied.add_change_hook(hook)
    face = next(iter(copied.faces))
    requested = next(corners for corners in ((0,1,2,3),(0,1,2,4))
                     if corners != copied.faces[face].corners)
    set_prepared_face_corners(copied,{face:requested})
    current = query_prepared_face_preimages(copied)
    assert current.face_descendants == binding.face_descendants
    assert current.authored_checksum == binding.authored_checksum
    assert current.authored_revision == binding.authored_revision
    assert observed and not any(observed)
    with pytest.raises(GeometryError,match='stale'):
        validate_prepared_face_preimages_binding(copied,binding)
    assert to_dict(model) == source


def test_mapping_tampering_wrong_model_and_unknown_history_fail_closed():
    model = crossing()
    prepare(model)
    binding = query_prepared_face_preimages(model)
    swapped = replace(binding,face_descendants=tuple(reversed(binding.face_descendants)))
    with pytest.raises(GeometryError,match='definition binding changed'):
        validate_prepared_face_preimages_binding(model,swapped)
    with pytest.raises(GeometryError,match='definition binding changed'):
        validate_prepared_face_preimages_binding(model,replace(binding,face_descendants=object()))
    with pytest.raises(GeometryError,match='another model'):
        validate_prepared_face_preimages_binding(model.clone(),binding)
    unknown = from_dict(to_dict(model))
    prepare(unknown)
    with pytest.raises(GeometryError,match='unavailable'):
        query_prepared_face_preimages(unknown)


def test_failed_or_cancelled_candidate_preserves_receipts_and_document():
    model = crossing()
    prepare(model)
    before = to_dict(model)
    binding = query_prepared_face_preimages(model)
    with pytest.raises(GeometryError,match='cancelled'):
        prepare(model,cancellation_check=lambda:True)
    assert to_dict(model) == before
    assert query_prepared_face_preimages(model) == binding
    with model.transaction():
        with pytest.raises(GeometryError,match='committed model'):
            query_prepared_face_preimages(model)


def test_missing_candidate_replacement_delta_fails_before_commit(monkeypatch):
    import anygeometry.batch_intersections as batch
    model = crossing()
    before = to_dict(model)
    original = batch._apply_intersections_in_place
    def omit_delta(*args, **kwargs):
        outcome = original(*args, **kwargs)
        return replace(outcome,change_set=replace(outcome.change_set,replacements=()))
    monkeypatch.setattr(batch,'_apply_intersections_in_place',omit_delta)
    with pytest.raises(GeometryError,match='missing or conflicting preimages'):
        prepare(model)
    assert to_dict(model) == before
    assert not hasattr(model,'_intersection_preparation_receipt')
    assert not hasattr(model,'_prepared_face_preimages_receipt')


def test_first_fragmentation_inside_outer_transaction_has_no_authored_provenance():
    model = crossing()
    source = to_dict(model)
    plan = plan_intersections(model,tuple(model.faces),policy='connect')
    with model.transaction():
        application = apply_intersections(model,plan,policy='connect')
    assert application.plan is plan
    assert application.plan.source_checksum == source['checksum']['value']
    assert model.model_id == application.plan.model_id
    assert len(model.faces) == 4
    assert model.last_change_set.replacements
    assert application.joint_edges
    assert all(edge.id in model.edges for edge in application.joint_edges)
    assert model.validate_topology() == ()
    assert not hasattr(model,'_prepared_face_preimages_receipt')
    with pytest.raises(GeometryError,match='complete current preparation'):
        query_prepared_face_preimages(model)
    # A fresh full classification remains valid, but cannot recover the missing
    # original snapshot anchor from persistent face replacement history.
    assert prepare(model).reused
    assert has_current_intersection_preparation(model)
    committed = to_dict(model)
    with pytest.raises(GeometryError,match='unavailable'):
        query_prepared_face_preimages(model)
    assert to_dict(model) == committed
