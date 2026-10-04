"""Typed visibility must not be confused with semantic or meshing permission."""
from dataclasses import FrozenInstanceError, replace
import json

import numpy as np
import pytest

import anygeometry
from anygeometry import (
    GeometryError, GeometryModel, clone_prepared_geometry, to_dict,
    query_prepared_authored_constraint_scope as query,
    validate_prepared_authored_constraint_scope_binding as validate,
)
from anygeometry.authored_constraint_scope import _dispositions, _isolated_vertices
from anygeometry.structural import JunctionMemberUse, ParameterRange
from examples.prepared_sheet_joint_component_handoff import build
from test_prepared_model_scope import fixture as reference_fixture, prepare


@pytest.fixture(scope='module')
def seed():
    return build(unrelated=True)


@pytest.fixture
def prepared(seed):
    model, roots, sheets, edge = seed
    return clone_prepared_geometry(model), roots, sheets, edge


def assert_unqualified(receipt):
    assert receipt.typed_inventory_complete
    for field in ('semantic_mapping_qualified', 'parameter_remapping_qualified',
                  'material_qualified', 'publication_qualified',
                  'external_reference_scope_qualified'):
        assert getattr(receipt, field) is False
    for row in receipt.inventory['record_dispositions'] + receipt.inventory['traces']:
        assert row['semantic_mapping_qualified'] is False
        assert row['parameter_remapping_qualified'] is False


def test_complete_pair_and_partial_root_preserve_all_literal_neighbours(prepared):
    model, roots, sheets, edge = prepared
    before = to_dict(model)
    full = query(model, (np.int64(roots[1]), np.int64(roots[0])))
    partial = query(model, (roots[0],))
    assert full.selected_root_ids == roots[:2]
    assert len(full.current_face_ids) == 8
    assert full.outside_root_ids == ()
    assert partial.outside_root_ids == (roots[1],)
    assert len(partial.current_face_ids) == 2
    traces = [row for row in partial.inventory['traces'] if row['current_edge_id'] == edge]
    assert len(traces) == 1
    row = traces[0]
    assert row['trace_kind'] == 'paired_interior'
    assert len(row['adjacent_face_ids']) == 4
    assert len(row['adjacent_face_use_ids']) == 4
    assert row['adjacent_sheet_ids'] == list(sheets[:2])
    assert len(row['adjacent_coedge_ids']) == 4
    assert row['adjacent_authored_root_ids'] == list(roots[:2])
    assert row['outside_authored_root_ids'] == [roots[1]]
    assert roots[2] not in full.outside_root_ids
    assert roots[2] in full.scope.face_preimages.authored_face_ids
    assert row['edge_definition'] == next(item for item in before['edges'] if item['id'] == edge)
    assert {item['id'] for item in row['endpoint_records']} == set(row['endpoint_ids'])
    for receipt in (full, partial):
        assert_unqualified(receipt)
        validate(model, receipt)
    assert to_dict(model) == before


def test_full_reference_tables_retain_incoming_and_transitive_original_records():
    model, (faces, sheet, member, attachments, isolated) = reference_fixture()
    model.add_junction('crossing', (JunctionMemberUse(member, ParameterRange.point(0)),),
                       sheet_ids=(sheet,), attachment_ids=(attachments[1],),
                       metadata={'opaque-junction': 'no semantic equivalence'})
    model._attachments[attachments[0]] = replace(model._attachments[attachments[0]],
        lineage=(('member', member), ('attachment', attachments[1])))
    original = to_dict(model)
    prepare(model)
    current = to_dict(model)
    receipt = query(model, (faces[0],))
    inventory = receipt.inventory
    for name, document in (('original', original), ('current', current)):
        for kind in ('members', 'member_edge_uses', 'attachments', 'junctions'):
            assert inventory[name]['records'][kind] == document['structural'][kind]
        opaque = inventory[name]['opaque_unqualified']
        assert opaque['groups'] == document['groups']
        assert opaque['tags'] == document['tags']
        assert opaque['extensions'] == document['extensions']
        assert opaque['features'] == document.get('features', {})
        assert isolated in inventory[name]['isolated_vertex_ids']
    assert inventory['original']['records']['members'][0]['orientation_reference']
    assert inventory['original']['records']['attachments'][0]['lineage']
    assert inventory['original']['records']['junctions'][0]['member_uses']
    assert inventory['record_dispositions'] == _dispositions(original, current)
    assert any(row['literal_disposition'] != 'literal_unchanged'
               for row in inventory['record_dispositions'])
    assert_unqualified(receipt)
    validate(model, receipt)
    assert to_dict(model) == current


def test_all_dispositions_are_literal_and_unqualified():
    kinds = ('members', 'member_edge_uses', 'attachments', 'junctions')
    before = {'structural': {kind: [] for kind in kinds}}
    after = {'structural': {kind: [] for kind in kinds}}
    before['structural']['members'] = [{'id': 1}, {'id': 2, 'v': -0.0}, {'id': 3}]
    after['structural']['members'] = [{'id': 1}, {'id': 2, 'v': 0.0}, {'id': 4}]
    rows = _dispositions(before, after)
    assert [row['literal_disposition'] for row in rows] == [
        'literal_unchanged', 'literal_changed', 'source_only', 'current_only']
    assert all(row['semantic_mapping_qualified'] is False for row in rows)


def test_corner_offsets_cannot_hide_an_isolated_vertex():
    model = GeometryModel()
    isolated = model.add_point(20, 30, 40)
    root = model.add_plate(model.add_points(((0,0,0),(2,0,0),(2,2,0),(0,2,0))))
    assert isolated in model.faces[root].corners  # Same integer; different namespace.
    original = to_dict(model)
    prepare(model)
    receipt = query(model, (root,))
    assert receipt.inventory['original']['isolated_vertex_ids'] == [isolated]
    assert receipt.inventory['current']['isolated_vertex_ids'] == [isolated]
    assert receipt.scope.authored_document == original
    # Curve control/via references really ARE vertex identifiers.
    synthetic = {'vertices': [{'id': key} for key in range(1, 7)], 'edges': [
        {'start': 1, 'end': 2, 'curve': {'control_vertices': [3, 4]}},
        {'start': 1, 'end': 2, 'curve': {'via_vertex': 5}}]}
    assert _isolated_vertices(synthetic) == [6]


def test_detached_inventory_and_frozen_receipt(prepared):
    model, roots, _, _ = prepared
    receipt = query(model, roots[:2])
    value = receipt.inventory
    value['current']['records']['attachments'].clear()
    value['traces'][0]['endpoint_records'].clear()
    assert receipt.inventory['current']['records']['attachments']
    assert receipt.inventory['traces'][0]['endpoint_records']
    with pytest.raises(FrozenInstanceError):
        receipt.inventory_json = '{}'
    validate(model, receipt)


@pytest.mark.parametrize('change', ['inventory', 'neighbours', 'faces', 'boundaries', 'flags', 'scope'])
def test_forged_omissions_and_claims_refuse(prepared, change):
    model, roots, _, _ = prepared
    receipt = query(model, roots[:1])
    if change == 'inventory':
        value = receipt.inventory
        value['original']['records']['attachments'].clear()
        value['current']['records']['attachments'].clear()
        forged = replace(receipt, inventory_json=json.dumps(value))
    else:
        field, value = {'neighbours': ('outside_root_ids', ()), 'faces': ('current_face_ids', ()),
            'boundaries': ('boundary_correspondences', ()),
            'flags': ('semantic_mapping_qualified', True), 'scope': ('scope', None)}[change]
        forged = replace(receipt, **{field: value})
    with pytest.raises(GeometryError):
        validate(model, forged)


@pytest.mark.parametrize('ids', [(), (True,), (np.bool_(False),), (1.0,), ('1',), (1,1), (9999,), None])
def test_bad_root_selection_refuses(prepared, ids):
    model, _, _, _ = prepared
    with pytest.raises(GeometryError):
        query(model, ids)


@pytest.mark.parametrize('mutation', ['same_revision', 'revision'])
def test_stale_content_and_revision_refuse(prepared, mutation):
    model, roots, _, _ = prepared
    receipt = query(model, roots[:2])
    face = next(iter(model.faces))
    if mutation == 'same_revision':
        revision = model.revision
        model._faces[face] = replace(model.faces[face], metadata={'new': True})
        assert model.revision == revision
    else:
        model.set_face_metadata(face, {'new': True})
    with pytest.raises(GeometryError):
        validate(model, receipt)
    with pytest.raises(GeometryError):
        query(model, roots[:2])


def test_wrong_model_and_expected_revision_refuse(prepared):
    model, roots, _, _ = prepared
    receipt = query(model, roots[:2])
    other, *_ = build()
    with pytest.raises(GeometryError):
        validate(other, receipt)
    with pytest.raises(GeometryError):
        query(model, roots[:2], expected_revision=model.revision-1)


def test_caller_input_coercion_cannot_reanchor_owner_scope(prepared):
    model, roots, _, _ = prepared
    class MutatingRoots:
        def __iter__(self):
            model._serialization_extensions['input-edit'] = True
            return iter(roots[:2])
    with pytest.raises(GeometryError):
        query(model, MutatingRoots())


@pytest.mark.parametrize('operation', ['query', 'validate'])
@pytest.mark.parametrize('kind', ['truthy', 'exception', 'false_mutation'])
def test_callback_cancellation_and_false_mutation_cannot_publish(prepared, operation, kind):
    model, roots, _, _ = prepared
    receipt = query(model, roots[:2])
    before = to_dict(model)
    error = RuntimeError('exact cancellation exception')
    def callback(_):
        if kind == 'exception':
            raise error
        if kind == 'false_mutation':
            model._serialization_extensions['callback-edit'] = {'x': 1}
            return False
        return True
    with pytest.raises(RuntimeError if kind == 'exception' else GeometryError) as caught:
        if operation == 'query':
            query(model, roots[:2], cancellation_check=callback)
        else:
            validate(model, receipt, cancellation_check=callback)
    if kind == 'exception':
        assert caught.value is error
    if kind != 'false_mutation':
        assert to_dict(model) == before


def test_last_callback_mutation_refuses(prepared):
    model, roots, _, _ = prepared
    phases = []
    query(model, roots[:2], cancellation_check=lambda phase: phases.append(phase) or False)
    count = 0
    def callback(_):
        nonlocal count
        count += 1
        if count == len(phases):
            model._serialization_extensions['last-edit'] = True
        return False
    with pytest.raises(GeometryError):
        query(model, roots[:2], cancellation_check=callback)
    assert count == len(phases)


def test_transient_callback_inputs_and_caller_list_are_detached(prepared):
    model, roots, _, _ = prepared
    before = to_dict(model)
    selected = [roots[0]]
    count = 0
    def callback(_):
        nonlocal count
        count += 1
        if count == 1:
            selected.append(roots[1])
            model._serialization_extensions['temporary-edit'] = True
        elif count == 2:
            del model._serialization_extensions['temporary-edit']
        return False
    receipt = query(model, selected, cancellation_check=callback)
    assert receipt.selected_root_ids == (roots[0],)
    assert receipt.outside_root_ids == (roots[1],)
    assert receipt.scope.current_document == before
    assert to_dict(model) == before


def test_public_exports_and_portable_curved_wall_inventory():
    names = ('PreparedAuthoredConstraintScope', 'query_prepared_authored_constraint_scope',
             'validate_prepared_authored_constraint_scope_binding')
    assert all(name in anygeometry.__all__ for name in names)
    from examples.authored_constraint_scope_handoff import build as portable
    model, roots, joints, full, partial = portable()
    assert joints
    assert full.selected_root_ids == roots
    assert partial.outside_root_ids == (roots[1],)
    assert_unqualified(full)
    assert all(row['edge_definition'] for row in full.inventory['traces'])
    assert full.scope.current_document == to_dict(model)


def test_callback_cannot_rebase_model_and_receipt_together():
    model = GeometryModel()
    root = model.add_plate(model.add_points(((0,0,0),(2,0,0),(2,2,0),(0,2,0))))
    prepare(model)
    receipt = query(model, (root,))
    changed = False

    def callback(_):
        nonlocal changed
        if not changed:
            changed = True
            # Build valid fresh preparation on the SAME object: transferred
            # receipts would instead hit the earlier weak-owner binding guard.
            identity = model.model_id
            model.__dict__.clear()
            GeometryModel.__init__(model, model_id=identity)
            new_root = model.add_plate(model.add_points(
                ((0,0,0),(2,0,0),(2,2,0),(0,2,0))))
            assert new_root == root
            model.set_face_metadata(root, {'resealed': True})
            prepare(model)
            fresh = query(model, (root,))
            validate(model, fresh)
            for field in receipt.__dataclass_fields__:
                object.__setattr__(receipt, field, getattr(fresh, field))
        return False

    with pytest.raises(GeometryError):
        validate(model, receipt, cancellation_check=callback)
    assert changed
    # The replacement genuinely binds the new preparation; the old validation
    # still had to reject the coordinated mutation during its own invocation.
    validate(model, receipt)
