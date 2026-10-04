"""Fresh source-child scope preserves qualified literal joint declarations."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from anygeometry import GeometryError, to_dict
from anygeometry.prepared_sheet_joint_component import (
    _dependencies, _index, _qualify_preserved_source_joints, _source_relations,
    query_prepared_sheet_joint_component as query,
    validate_prepared_sheet_joint_component_binding as validate,
)
from examples.reopened_sheet_joint_component_handoff import build, verify


def test_reopened_child_design_is_a_fresh_complete_scope():
    result = verify()
    assert result['authored_roots'] == tuple(range(1, 9))
    assert result['current_joint_edge'] == 17
    assert result['preserved_joint_attachment_ids'] == (1, 2)
    assert result['preserved_joint_junction_ids'] == (1,)
    assert result['occurrence_count'] == 8
    assert not result['semantic_mapping_qualified'] and not result['publication_qualified']


@pytest.mark.parametrize('field', ('preserved_joint_attachment_ids', 'preserved_joint_junction_ids'))
def test_preserved_relation_receipt_cannot_be_forged(field):
    _, closure, _, joint, _ = build()
    model = closure.working_model
    receipt = query(model, joint)
    before = to_dict(model)
    with pytest.raises(GeometryError):
        validate(model, replace(receipt, **{field: ()}))
    assert to_dict(model) == before


@pytest.mark.parametrize('change', (
    'attachment_payload', 'attachment_parameters', 'attachment_missing',
    'junction_payload', 'junction_missing', 'edge_definition',
    'endpoint_position', 'endpoint_missing', 'unsupported_relation',
))
def test_literal_source_relation_proof_does_not_infer_remapping(change):
    _, closure, _, joint, _ = build()
    receipt = query(closure.working_model, joint)
    source = _index(receipt.scope.authored_document)
    current = deepcopy(_index(receipt.scope.current_document))
    attachments, junctions = set(receipt.attachment_ids), set(receipt.junction_ids)
    original_attachments, original_junctions = set(source['attachments']), set(source['junctions'])
    link = min(original_attachments)
    edge = source['attachments'][link]['target_id']
    vertex = source['edges'][edge]['start']
    if change == 'attachment_payload':
        current['attachments'][link]['metadata'] = {'changed': True}
    elif change == 'attachment_parameters':
        source['attachments'][link]['target_parameters'] = [[0.0, .5]]
    elif change == 'attachment_missing':
        attachments.remove(link)
    elif change == 'junction_payload':
        current['junctions'][min(junctions)]['metadata'] = {'changed': True}
    elif change == 'junction_missing':
        junctions.clear()
    elif change == 'edge_definition':
        current['edges'][edge]['metadata'] = {'changed': True}
    elif change == 'endpoint_position':
        current['vertices'][vertex]['position'][0] = float(np.nextafter(
            current['vertices'][vertex]['position'][0], np.inf))
    elif change == 'endpoint_missing':
        source['vertices'].pop(vertex)
    else:
        source['attachments'][99] = dict(source['attachments'][link], id=99,
                                         kind='vertex_on_face', target_kind='face')
        original_attachments.add(99)
    with pytest.raises(GeometryError):
        _qualify_preserved_source_joints(source, current, original_attachments,
                                        original_junctions, attachments, junctions, lambda: None)


def test_preserved_relation_proof_propagates_cancellation_exception():
    _, closure, _, joint, _ = build()
    receipt = query(closure.working_model, joint)
    failure = RuntimeError('preserved source relation cancellation')
    def check():
        raise failure
    with pytest.raises(RuntimeError) as caught:
        _qualify_preserved_source_joints(
            _index(receipt.scope.authored_document), _index(receipt.scope.current_document),
            set(receipt.preserved_joint_attachment_ids), set(receipt.preserved_joint_junction_ids),
            set(receipt.attachment_ids), set(receipt.junction_ids), check)
    assert caught.value is failure


def test_original_incoming_relation_closure_cannot_disappear_from_source_proof():
    _, closure, _, joint, _ = build(unrelated=True)
    receipt = query(closure.working_model, joint)
    source = _index(receipt.scope.authored_document)
    current = _index(receipt.scope.current_document)
    deps = _dependencies(source, set(receipt.sheet_ids))
    outside_edge = next(key for key, row in source['edges'].items()
                        if key not in deps['edges'] and row['start'] not in deps['vertices']
                        and row['end'] not in deps['vertices'])
    prototype = dict(source['attachments'][1], kind='vertex_on_edge', source_kind='vertex',
                     source_id=source['edges'][outside_edge]['start'], target_id=outside_edge,
                     member_range=[0., 0.], target_parameters=[[0., 0.]])
    # Reverse dependency order requires a genuine fixed point. All geometric
    # parents are in the unrelated component; lineage is the only incoming link.
    source['attachments'][100] = dict(prototype, id=100, lineage=[['attachment', 99]])
    source['attachments'][99] = dict(prototype, id=99, lineage=[['attachment', 1]])
    source['attachments'][101] = dict(prototype, id=101, lineage=[['junction', 1]])
    source['junctions'][99] = dict(source['junctions'][1], id=99, sheet_ids=[], attachment_ids=[100])
    links, joints = _source_relations(source, deps, deps['edges'], lambda: None)
    assert links == {1, 2, 99, 100, 101} and joints == {1, 99}
    with pytest.raises(GeometryError, match='original Attachment remapping'):
        _qualify_preserved_source_joints(source, current, links, joints,
                                        set(receipt.attachment_ids), set(receipt.junction_ids), lambda: None)
