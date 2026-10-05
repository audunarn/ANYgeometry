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


def test_source_serialization_cannot_repair_initially_forged_evidence():
    model, joint = build()
    valid = query(model, joint)
    forged = replace(valid, member_relation_json='{}')
    triggered = []
    class Repair(dict):
        def __deepcopy__(self, memo):
            triggered.append(True)
            object.__setattr__(forged, 'member_relation_json', valid.member_relation_json)
            model._serialization_extensions = {}
            return {}
    model._serialization_extensions = Repair()
    with pytest.raises(GeometryError, match='definition binding changed'):
        validate(model, forged)
    assert triggered


def test_last_source_guard_cannot_change_validated_receipt(monkeypatch):
    import anygeometry.prepared_member_sheet_component as component
    model, joint = build()
    receipt = query(model, joint)
    real_validate = component.validate_prepared_model_scope_binding
    scope_calls = []
    active = []
    triggered = []
    class Change(dict):
        def __deepcopy__(self, memo):
            if active:
                triggered.append(True)
                object.__setattr__(receipt, 'member_relation_json', '{}')
                model._serialization_extensions = {}
            return {}
    def closing_guard(geometry, scope, **kwargs):
        if scope is receipt.scope:
            scope_calls.append(True)
            if len(scope_calls) == 2:
                active.append(True)
        return real_validate(geometry, scope, **kwargs)
    model._serialization_extensions = Change()
    monkeypatch.setattr(component, 'validate_prepared_model_scope_binding', closing_guard)
    with pytest.raises(GeometryError, match='definition binding changed'):
        validate(model, receipt)
    assert triggered


def test_query_last_serialization_cannot_mutate_generated_ancestry(monkeypatch):
    import anygeometry.prepared_member_sheet_component as component
    model, joint = build()
    record = query(model, joint).edge_preimages.records[0]
    real_validate = component.validate_prepared_model_scope_binding
    calls = []
    active = []
    triggered = []
    class Change(dict):
        def __deepcopy__(self, memo):
            if active:
                triggered.append(True)
                object.__setattr__(record, 'interval', ((0, 1), (0, 1)))
                model._serialization_extensions = {}
            return {}
    def final_relation_guard(geometry, scope, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            active.append(True)
        return real_validate(geometry, scope, **kwargs)
    model._serialization_extensions = Change()
    monkeypatch.setattr(component, 'validate_prepared_model_scope_binding', final_relation_guard)
    with pytest.raises(GeometryError, match='output definition changed'):
        query(model, joint)
    assert triggered


@pytest.mark.parametrize('field', ['scope', 'joint_edge_id', 'ancestry_record'])
def test_signature_rejects_behavioral_graph_nodes_before_hooks(field):
    model, joint = build()
    receipt = query(model, joint)
    triggered = []
    class Behavioral(dict):
        def items(self):
            triggered.append(True)
            return super().items()
    if field == 'ancestry_record':
        forged = replace(receipt, edge_preimages=replace(receipt.edge_preimages,
                          records=(Behavioral(),)))
    else:
        forged = replace(receipt, **{field: Behavioral()})
    with pytest.raises(GeometryError, match='plain immutable owner fields'):
        validate(model, forged)
    assert not triggered


def test_signature_type_whitelist_does_not_execute_custom_metaclass():
    model, joint = build()
    receipt = query(model, joint)
    triggered = []
    class Meta(type):
        def __hash__(cls):
            triggered.append('hash')
            return type.__hash__(cls)
        def __eq__(cls, other):
            triggered.append('eq')
            return type.__eq__(cls, other)
    class Behavioral(dict, metaclass=Meta):
        pass
    with pytest.raises(GeometryError, match='plain immutable owner fields'):
        validate(model, replace(receipt, scope=Behavioral()))
    assert not triggered


def test_cyclic_receipt_graph_raises_typed_error():
    model, joint = build()
    receipt = query(model, joint)
    scope = replace(receipt.scope)
    object.__setattr__(scope, 'face_preimages', scope)
    with pytest.raises(GeometryError, match='cyclic receipt fields'):
        validate(model, replace(receipt, scope=scope))


@pytest.mark.parametrize('depth', (600, 1600, 4000))
def test_malformed_deep_receipt_raises_typed_error_without_operand_cap(depth):
    model, joint = build()
    receipt = query(model, joint)
    nested = ()
    for _ in range(depth):
        nested = (nested,)
    with pytest.raises(GeometryError, match='invalid receipt nesting|definition binding changed'):
        validate(model, replace(receipt, current_face_ids=nested))
