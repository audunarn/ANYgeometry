"""Bounded original-member eligibility, independent of consumer meshing."""
from dataclasses import replace
import json

import numpy as np
import pytest

from anygeometry import (
    GeometryError, GeometryModel, Plane, apply_intersections, plan_intersections,
    query_prepared_member_sheet_joint_component as query,
    query_prepared_sheet_joint_component as legacy_query,
    validate_prepared_member_sheet_joint_component_binding as validate,
    validate_prepared_sheet_joint_component_binding as legacy_validate, to_dict,
)
from anygeometry.structural import ParameterRange


def build(reverse=False, extra=False):
    model = GeometryModel()
    for points, plane in (
        (((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)),
         Plane((0, 0, 0), (1, 0, 0), (0, 1, 0))),
        (((-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1)),
         Plane((0, 0, 0), (1, 0, 0), (0, 0, 1))),
    ):
        face = model.add_plate(model.add_points(points))
        model.set_face_surface(face, plane)
        model.add_sheet((face,))
    edge = next(e.id for e in model.edges.values()
                if all(model.vertex_position(v)[0] == -1 and model.vertex_position(v)[2] == 0
                       for v in (e.start, e.end)))
    member = model.add_member((edge,))
    if reverse:
        model.reverse_member(member)
    vertex = model.add_point(*model.sample_edge(edge, np.array([.75]))[0])
    for _ in range(2 if extra else 1):
        model.add_attachment(None, 'vertex_on_edge', 'edge', edge, ParameterRange.point(0.),
            (ParameterRange.point(.75),), source_kind='vertex', source_id=vertex,
            evidence='exact', tolerance_used=1e-9)
    apply_intersections(model, plan_intersections(model, tuple(model.faces), policy='connect'),
                        policy='connect')
    joint = next(a.target_id for a in model.attachments.values() if a.kind == 'sheet_on_joint')
    return model, joint


@pytest.mark.parametrize('reverse', [False, True])
def test_bounded_member_point_receipt_preserves_occurrence_and_station(reverse):
    model, joint = build(reverse)
    before = to_dict(model)
    receipt = query(model, joint)
    relation = receipt.member_relation
    assert len(relation['current_member_uses']) == 2
    assert relation['current_attachment']['target_parameters'] == [[.5, .5]]
    assert relation['point_squared_distance_bound'] == [0, 1]
    assert receipt.bounded_relation_mapping_qualified
    assert receipt.occurrence_mapping_qualified
    assert not receipt.semantic_mapping_qualified
    assert not receipt.beam_discretization_qualified
    assert not receipt.external_reference_transfer_qualified
    assert not receipt.publication_qualified
    assert len(receipt.occurrence_correspondence) == 2
    assert len(receipt.source_records['members']) == 1
    assert len(receipt.current_records['member_edge_uses']) == 2
    point = relation['current_attachment']
    assert point['id'] in receipt.attachment_ids
    assert point in receipt.current_records['attachments']
    assert relation['source_attachment'] in receipt.source_records['attachments']
    assert relation['current_vertex'] in receipt.current_records['vertices']
    assert set(receipt.current_face_ids) == set(model.faces)
    validate(model, receipt)
    with pytest.raises(GeometryError, match='unsupported Member semantics'):
        legacy_query(model, joint)
    with pytest.raises(GeometryError, match='requires an owner receipt'):
        legacy_validate(model, receipt)
    relation['current_member_uses'].clear()
    assert len(receipt.member_relation['current_member_uses']) == 2
    assert to_dict(model) == before


def test_extra_original_relation_is_explicitly_refused():
    model, joint = build(extra=True)
    with pytest.raises(GeometryError, match='one original point Attachment'):
        query(model, joint)


@pytest.mark.parametrize('change', ['omit', 'station', 'orientation', 'flag'])
def test_forged_relation_or_qualification_flag_is_not_live_authority(change):
    model, joint = build()
    receipt = query(model, joint)
    payload = receipt.member_relation
    if change == 'omit':
        payload['current_member_uses'].pop()
    elif change == 'station':
        payload['current_attachment']['target_parameters'] = [[0., 0.]]
    elif change == 'orientation':
        payload['current_member_uses'][0]['orientation'] = 'reversed'
    forged = (replace(receipt, publication_qualified=True) if change == 'flag' else
              replace(receipt, member_relation_json=json.dumps(payload)))
    with pytest.raises(GeometryError, match='definition binding changed'):
        validate(model, forged)


def test_wrong_owner_and_stale_model_are_refused():
    model, joint = build()
    receipt = query(model, joint)
    with pytest.raises(GeometryError):
        validate(model.clone(preserve_identity=True), receipt)
    model.add_point(9, 9, 9)
    with pytest.raises(GeometryError):
        validate(model, receipt)


@pytest.mark.parametrize('revision_change', [False, True])
def test_callback_edits_cannot_publish_a_receipt(revision_change):
    model, joint = build()
    invoked = []
    def edit(_phase):
        if not invoked:
            invoked.append(True)
            if revision_change:
                model.add_point(9, 9, 9)
            else:
                key = next(iter(model.members))
                model._members[key] = replace(model.members[key], name='changed')
        return False
    with pytest.raises(GeometryError):
        query(model, joint, cancellation_check=edit)
    assert invoked


def test_cancellation_preserves_model():
    model, joint = build()
    before = to_dict(model)
    with pytest.raises(GeometryError, match='cancelled'):
        query(model, joint, cancellation_check=lambda _phase: True)
    assert to_dict(model) == before


def test_required_carrier_ancestry_cannot_be_omitted(monkeypatch):
    import anygeometry.prepared_member_sheet_component as component
    model, joint = build()
    real_query = component.query_prepared_edge_subcurve_preimages
    def omit(*args, **kwargs):
        result = real_query(*args, **kwargs)
        return replace(result, records=())
    monkeypatch.setattr(component, 'query_prepared_edge_subcurve_preimages', omit)
    with pytest.raises(GeometryError, match='source-carrier ancestry is unavailable'):
        query(model, joint)


@pytest.mark.parametrize('field', ['member_relation_json', 'edge_preimages'])
def test_unusable_receipt_fields_raise_typed_errors(field):
    model, joint = build()
    receipt = query(model, joint)
    with pytest.raises(GeometryError, match='plain immutable'):
        validate(model, replace(receipt, **{field: object()}))
