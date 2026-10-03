"""Complete owner input visibility is not an implicit remapping certificate."""
from dataclasses import FrozenInstanceError, replace
import json

import pytest

from anygeometry import (
    EntityRef, GeometryError, GeometryModel, apply_intersections,
    clone_prepared_geometry, from_dict, plan_intersections, query_prepared_face_preimages,
    query_prepared_model_scope, to_dict, validate_prepared_model_scope_binding,
    set_prepared_face_corners,
)
from anygeometry.prepared_face_preimages import _binding_checksum
from anygeometry.structural import ParameterRange


def fixture():
    model = GeometryModel()
    faces = []
    for rows in (
            ((-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0)),
            ((-1,0,-1),(1,0,-1),(1,0,1),(-1,0,1)),
            ((10,10,0),(12,10,0),(12,12,0),(10,12,0))):
        faces.append(model.add_plate(model.add_points(rows)))
    sheet = model.add_sheet((faces[0],), name='original-sheet')
    edge = model.faces[faces[0]].loop[0].edge
    vertex = model.add_point(0,-1,0)
    tip = model.add_point(0,-1,1)
    member = model.add_member((model.add_line(vertex, tip),),
                             metadata={'section': 'beam'}, orientation_reference=('edge', edge))
    point = model.add_attachment(None, 'vertex_on_edge', 'edge', edge, ParameterRange.point(0),
        (ParameterRange.point(.5),), source_kind='vertex', source_id=vertex,
        evidence='exact', tolerance_used=1e-9, metadata={'purpose': 'point-load'})
    connection = model.add_attachment(member, 'member_on_boundary', 'edge', edge,
        ParameterRange.point(0), (ParameterRange.point(.5),), evidence='exact', tolerance_used=1e-9)
    isolated = model.add_point(30,40,50)
    model.add_to_group('isolated loads', (EntityRef('vertex', isolated),))
    model.tag(EntityRef('face', faces[2]), 'unrelated-neighbour')
    model.set_face_metadata(faces[1], {'child-property': {'value': [1,2]}})
    model._serialization_extensions = {'test:external': {'reference': isolated}}
    return model, (faces, sheet, member, (point, connection), isolated)


def prepare(model, faces=None):
    plan = plan_intersections(model, tuple(model.faces) if faces is None else faces, policy='connect')
    apply_intersections(model, plan, policy='connect')


def test_complete_snapshots_include_incoming_unrelated_and_isolated_records():
    model, (faces, sheet, member, attachments, isolated) = fixture()
    original = to_dict(model)
    prepare(model)
    current = to_dict(model)
    scope = query_prepared_model_scope(model, expected_revision=model.revision)
    assert scope.authored_document == original
    assert scope.current_document == current
    assert len(current['faces']) > len(original['faces'])
    assert any(row['id'] == isolated for row in scope.authored_document['vertices'])
    structural = scope.authored_document['structural']
    assert {row['id'] for row in structural['attachments']} == set(attachments)
    assert any(row['id'] == member for row in structural['members'])
    assert any(row['sheet_id'] == sheet for row in structural['face_uses'])
    assert any(row['id'] == faces[2] for row in scope.authored_document['faces'])
    assert scope.authored_document['extensions'] == original['extensions']
    validate_prepared_model_scope_binding(model, scope)
    # Both ordinary documents can be consumed through the PUBLIC owner decoder.
    assert to_dict(from_dict(scope.authored_document)) == original
    assert to_dict(from_dict(scope.current_document)) == current
    assert to_dict(model) == current


def test_scope_is_immutable_and_decoded_views_are_detached():
    model, _ = fixture(); prepare(model)
    scope = query_prepared_model_scope(model)
    original = scope.authored_document
    original['structural']['attachments'].clear()
    original['vertices'].clear()
    current = scope.current_document
    current['faces'][0]['metadata']['forged'] = True
    assert scope.authored_document['structural']['attachments']
    assert scope.authored_document['vertices']
    assert 'forged' not in scope.current_document['faces'][0]['metadata']
    with pytest.raises(FrozenInstanceError):
        scope.authored_document_json = '{}'
    validate_prepared_model_scope_binding(model, scope)


def test_clone_preserves_extension_content_without_mutable_aliases():
    model, _ = fixture()
    original = to_dict(model)
    copied = model.clone(preserve_identity=True)
    assert to_dict(copied) == original
    copied._serialization_extensions['test:external']['reference'] = -3
    assert to_dict(model) == original


@pytest.mark.parametrize('field', ['authored_document_json', 'current_document_json'])
def test_deleted_or_changed_snapshot_record_is_not_a_negative_proof(field):
    model, _ = fixture(); prepare(model)
    scope = query_prepared_model_scope(model)
    payload = json.loads(getattr(scope, field))
    payload['structural']['attachments'].clear()
    tampered = replace(scope, **{field: json.dumps(payload, sort_keys=True, separators=(',', ':'))})
    with pytest.raises(GeometryError, match='binding changed'):
        validate_prepared_model_scope_binding(model, tampered)


def test_forged_ancestry_with_missing_original_references_refuses():
    model, _ = fixture(); prepare(model)
    scope = query_prepared_model_scope(model)
    altered = replace(scope.face_preimages, authored_document_json='{}')
    with pytest.raises(GeometryError, match='definition binding changed'):
        validate_prepared_model_scope_binding(model, replace(scope, face_preimages=altered,
                                                              authored_document_json='{}'))


@pytest.mark.parametrize('field', ['authored_document_json', 'current_document_json'])
def test_string_subclass_cannot_forge_missing_scope_records(field):
    class ForgedText(str):
        def __ne__(self, _):
            return False
    model, _ = fixture(); prepare(model)
    scope = query_prepared_model_scope(model)
    forged = replace(scope, **{field: ForgedText('{}')})
    with pytest.raises(GeometryError, match='plain immutable snapshot fields'):
        validate_prepared_model_scope_binding(model, forged)


@pytest.mark.parametrize('kind', ['revision', 'vertex', 'face_use', 'member', 'attachment', 'extension'])
def test_all_current_persisted_content_is_bound_including_raw_same_revision_edits(kind):
    model, (_, _, member, attachments, isolated) = fixture(); prepare(model)
    scope = query_prepared_model_scope(model)
    if kind == 'revision':
        model.add_point(80,80,80)
    elif kind == 'vertex':
        model._vertices[isolated] = replace(model.vertices[isolated], position=(31,40,50))
    elif kind == 'face_use':
        key = next(iter(model.face_uses))
        model._face_uses[key] = replace(model.face_uses[key], metadata={'child-local': 'new'})
    elif kind == 'member':
        model._members[member] = replace(model.members[member], metadata={'section': 'changed'})
    elif kind == 'attachment':
        key = attachments[0]
        model._attachments[key] = replace(model.attachments[key], metadata={'load': 'changed'})
    else:
        model._serialization_extensions['test:external']['reference'] = -1
    with pytest.raises(GeometryError):
        validate_prepared_model_scope_binding(model, scope)
    with pytest.raises(GeometryError):
        query_prepared_model_scope(model)


def test_qualified_copy_retains_scope_but_load_plain_clone_and_foreign_model_refuse():
    model, _ = fixture(); prepare(model)
    scope = query_prepared_model_scope(model)
    copied = clone_prepared_geometry(model)
    assert query_prepared_model_scope(copied) == scope
    validate_prepared_model_scope_binding(copied, scope)
    for unknown in (model.clone(), from_dict(to_dict(model)), GeometryModel()):
        with pytest.raises(GeometryError):
            query_prepared_model_scope(unknown)
        with pytest.raises(GeometryError):
            validate_prepared_model_scope_binding(unknown, scope)


def test_old_partial_receipts_do_not_invent_complete_original_inputs():
    model, (faces, *_) = fixture()
    with pytest.raises(GeometryError):
        query_prepared_model_scope(model)
    prepare(model, (faces[0],))
    with pytest.raises(GeometryError):
        query_prepared_model_scope(model)
    model, _ = fixture(); prepare(model)
    old = replace(query_prepared_face_preimages(model), authored_document_json=None)
    model._prepared_face_preimages_receipt = old, _binding_checksum(old)
    with pytest.raises(GeometryError, match='snapshot is unavailable'):
        query_prepared_model_scope(model)


@pytest.mark.parametrize('operation', ['query', 'validate'])
@pytest.mark.parametrize('effect', ['cancel', 'edit'])
def test_callbacks_cannot_return_stale_scope_and_cancellation_preserves_state(operation, effect):
    model, _ = fixture(); prepare(model)
    scope = query_prepared_model_scope(model)
    before = to_dict(model)
    def callback(_):
        if effect == 'edit':
            model._serialization_extensions['test:external']['reference'] = -2
            return False
        return True
    with pytest.raises(GeometryError):
        if operation == 'query':
            query_prepared_model_scope(model, cancellation_check=callback)
        else:
            validate_prepared_model_scope_binding(model, scope, cancellation_check=callback)
    if effect == 'cancel':
        assert to_dict(model) == before


def test_stale_expected_revision_and_uncommitted_reads_refuse():
    model, _ = fixture(); prepare(model)
    with pytest.raises(GeometryError):
        query_prepared_model_scope(model, expected_revision=model.revision-1)
    with model.transaction():
        with pytest.raises(GeometryError):
            query_prepared_model_scope(model)


def test_qualified_corner_update_retains_original_and_changes_current_scope():
    model = GeometryModel()
    face = model.add_plate(model.add_points(((0,0,0),(2,0,0),(2,2,0),(0,2,0))))
    model.split_edge(model.faces[face].loop[0].edge, .5)
    original = to_dict(model)
    prepare(model)
    scope = query_prepared_model_scope(model)
    requested = next(value for value in ((0,1,2,3),(0,1,2,4))
                     if value != model.faces[face].corners)
    set_prepared_face_corners(model, {face: requested})
    with pytest.raises(GeometryError):
        validate_prepared_model_scope_binding(model, scope)
    new = query_prepared_model_scope(model)
    assert new.authored_document == original
    assert new.current_document == to_dict(model)
    assert new.current_document != scope.current_document


def test_idempotent_application_and_portable_fixture_preserve_scope():
    model, _ = fixture(); prepare(model)
    scope = query_prepared_model_scope(model)
    prepare(model)
    assert query_prepared_model_scope(model) == scope
    from examples.prepared_model_scope_handoff import build_prepared_scope
    prepared, portable, original = build_prepared_scope()
    assert portable.authored_document == original
    assert len(original['faces']) == 3
    assert len(portable.current_document['faces']) > len(original['faces'])
    validate_prepared_model_scope_binding(prepared, portable)
