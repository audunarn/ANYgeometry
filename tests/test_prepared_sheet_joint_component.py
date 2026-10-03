"""Declared current structural closure is distinct from general remapping."""
from dataclasses import FrozenInstanceError, replace
import json

import numpy as np
import pytest

from anygeometry import EntityRef, GeometryError, to_dict
from anygeometry.prepared_sheet_joint_component import (
    query_prepared_sheet_joint_component as query,
    validate_prepared_sheet_joint_component_binding as validate,
    validate_prepared_sheet_joint_component_selection as select,
)
from anygeometry.structural import ParameterRange
from examples.prepared_sheet_joint_component_handoff import build, verify


def test_portable_two_sheet_joint_complete_siblings_and_selection():
    model, roots, sheets, edge = build()
    before=to_dict(model)
    receipt=query(model,edge,expected_revision=model.revision)
    assert edge==25
    assert receipt.authored_face_ids==roots
    assert receipt.sheet_ids==sheets
    assert receipt.current_face_ids==tuple(range(3,11))
    assert len(receipt.attachment_ids)==2 and len(receipt.junction_ids)==1
    assert [len(row[3]) for row in receipt.occurrence_correspondence]==[2,6]
    assert receipt.occurrence_mapping_qualified
    assert not receipt.semantic_mapping_qualified and not receipt.publication_qualified
    assert len(receipt.current_records['face_uses'])==8
    assert len(receipt.source_records['face_uses'])==2
    assert all(row['orientation'] in ('forward','reversed') for row in receipt.current_records['coedges'])
    with pytest.raises(GeometryError,match='omits connected'):
        select(model,receipt,(roots[0],))
    select(model,receipt,roots)
    validate(model,receipt)
    assert to_dict(model)==before
    assert verify()['joint_edge']==25


def test_closure_reaches_second_joint_only_through_sheet_without_owner_cap():
    model,roots,sheets,edge=build(third=True,unrelated=True)
    receipt=query(model,edge)
    assert len(model.sheets_using_edge(edge))==2
    assert receipt.sheet_ids==sheets[:3]
    assert receipt.authored_face_ids==roots[:3]
    assert len(receipt.joint_edge_ids)==2
    assert len(receipt.junction_ids)==2 and len(receipt.attachment_ids)==4
    assert sheets[-1] not in receipt.sheet_ids and roots[-1] not in receipt.authored_face_ids
    assert all(row['sheet_id'] in sheets[:3] for row in receipt.current_records['face_uses'])
    with pytest.raises(GeometryError,match='omits connected'):
        select(model,receipt,roots[:2])
    select(model,receipt,roots[:3])


def test_generated_structural_owners_are_not_original_owners():
    model,_,_,edge=build(explicit_sheets=False)
    with pytest.raises(GeometryError,match='source-less Sheet'):
        query(model,edge)


def test_scope_retains_complete_raw_references_without_semantic_mapping_claim():
    def add_refs(model,roots,sheets):
        model.add_to_group('original property',(EntityRef('face',roots[0]),))
        model.tag(EntityRef('face',roots[0]),'original-tag')
        model.set_face_metadata(roots[0],{'load':'not-remapped-by-this-receipt'})
        model._serialization_extensions={'test:external':{'owner':roots[0]}}
    model,roots,_,edge=build(unrelated=True,before_prepare=add_refs)
    receipt=query(model,edge)
    assert receipt.source_records['groups']['original property']==[['face',roots[0]]]
    assert receipt.scope.authored_document['extensions']=={'test:external':{'owner':roots[0]}}
    assert any(row['face_id']==roots[-1] for row in receipt.scope.authored_document['structural']['face_uses'])
    assert not receipt.semantic_mapping_qualified
    with pytest.raises(FrozenInstanceError):
        receipt.sheet_ids=()
    decoded=receipt.current_records
    decoded['face_uses'].clear()
    assert receipt.current_records['face_uses']


def test_touching_member_is_refused_without_pruning():
    def add_member(model,roots,sheets):
        edge=model.faces[roots[0]].loop[0].edge
        model.add_member((edge,))
    model,_,_,edge=build(before_prepare=add_member)
    with pytest.raises(GeometryError,match='Member'):
        query(model,edge)


def test_touching_unsupported_point_attachment_is_refused():
    def add_attachment(model,roots,sheets):
        edge=model.faces[roots[0]].loop[-1].edge
        vertex=model.add_point(0,2,0)
        model.add_attachment(None,'vertex_on_edge','edge',edge,ParameterRange.point(0),
            (ParameterRange.point(.5),),source_kind='vertex',source_id=vertex,
            evidence='exact',tolerance_used=1e-9)
    model,_,_,edge=build(before_prepare=add_attachment)
    with pytest.raises(GeometryError,match='unsupported/uncontained Attachment'):
        query(model,edge)


@pytest.mark.parametrize('kind',('records','occurrence','qualification','scope'))
def test_tampered_receipt_refuses(kind):
    model,_,_,edge=build()
    receipt=query(model,edge)
    if kind=='records':
        records=receipt.current_records
        records['attachments'].clear()
        forged=replace(receipt,current_records_json=json.dumps(records))
    elif kind=='occurrence':
        forged=replace(receipt,occurrence_correspondence=receipt.occurrence_correspondence[:1])
    elif kind=='qualification':
        forged=replace(receipt,semantic_mapping_qualified=True)
    else:
        forged=replace(receipt,scope=object())
    with pytest.raises(GeometryError):
        validate(model,forged)


def test_stale_wrong_owner_and_revision_refuse():
    model,_,_,edge=build()
    receipt=query(model,edge)
    other,_,_,_=build()
    with pytest.raises(GeometryError):
        validate(other,receipt)
    with pytest.raises(GeometryError,match='revision'):
        query(model,edge,expected_revision=model.revision+1)
    key=receipt.current_face_ids[0]
    model._faces[key]=replace(model.faces[key],metadata={'late':'same revision'})
    with pytest.raises(GeometryError):
        validate(model,receipt)


@pytest.mark.parametrize('value',(True,np.bool_(False),1.5,'25'))
def test_bad_identifier_refuses(value):
    model,_,_,_=build()
    with pytest.raises(GeometryError):
        query(model,value)


def test_coercion_is_once_and_cannot_change_bound_model():
    class Identifier(int):
        calls=0
        def __int__(self):
            self.calls+=1
            key=next(iter(model.faces))
            model._faces[key]=replace(model.faces[key],metadata={'coercion':True})
            return super().__int__()
    model,_,_,edge=build()
    value=Identifier(edge)
    with pytest.raises(GeometryError):
        query(model,value)
    assert value.calls==1


def test_cancellation_identity_and_source_nonmutation():
    model,_,_,edge=build()
    before=to_dict(model)
    error=RuntimeError('cancel component')
    def check(*_):
        raise error
    with pytest.raises(RuntimeError) as caught:
        query(model,edge,cancellation_check=check)
    assert caught.value is error
    assert to_dict(model)==before


def test_final_callback_mutation_is_not_published():
    model,_,_,edge=build()
    receipt=query(model,edge)
    count=0
    def count_check(*_):
        nonlocal count
        count+=1
    query(model,edge,cancellation_check=count_check)
    calls=0
    def mutate_last(*_):
        nonlocal calls
        calls+=1
        if calls==count:
            key=receipt.current_face_ids[0]
            model._faces[key]=replace(model.faces[key],metadata={'last-callback':True})
    with pytest.raises(GeometryError):
        query(model,edge,cancellation_check=mutate_last)
    assert calls==count


def test_derived_index_mismatch_even_without_persisted_mutation():
    model,_,_,edge=build()
    model._edge_coedges[edge]=()
    with pytest.raises(GeometryError,match='index mismatch'):
        query(model,edge)


@pytest.mark.parametrize('selection',((True,),(),(1,1),(999,)))
def test_invalid_or_incomplete_selection_refuses(selection):
    model,_,_,edge=build()
    with pytest.raises(GeometryError):
        select(model,query(model,edge),selection)


def test_external_member_orientation_vertex_is_a_touching_dependency():
    def add_member(model,roots,sheets):
        vertex=model.edges[model.faces[roots[0]].loop[0].edge].start
        line=model.add_line(*model.add_points(((20,0,0),(20,0,1))))
        model.add_member((line,),orientation_reference=('vertex',vertex))
    model,_,_,edge=build(before_prepare=add_member)
    with pytest.raises(GeometryError,match='Member'):
        query(model,edge)


def test_external_attachment_lineage_alone_touches_component():
    def add_attachment(model,roots,sheets):
        points=model.add_points(((20,0,0),(20,0,1)))
        line=model.add_line(*points)
        model.add_attachment(None,'vertex_on_edge','edge',line,ParameterRange.point(0),
            (ParameterRange.point(0),),source_kind='vertex',source_id=points[0],
            evidence='exact',tolerance_used=1e-9,lineage=(('sheet',sheets[0]),))
    model,_,_,edge=build(before_prepare=add_attachment)
    with pytest.raises(GeometryError,match='unsupported/uncontained Attachment'):
        query(model,edge)


def test_selected_attachment_outgoing_lineage_refuses(monkeypatch):
    import anygeometry.joint_edges as joints
    original=joints.declare_joint_edge
    def declare(model,*args,**kwargs):
        result=original(model,*args,**kwargs)
        joint=next(row for row in model.junctions.values()
                   if any(model.attachments[key].target_id==args[0] for key in row.attachment_ids))
        key=joint.attachment_ids[0]
        model._put_structural('attachment',replace(model.attachments[key],lineage=(('vertex',1),)))
        return result
    monkeypatch.setattr(joints,'declare_joint_edge',declare)
    model,_,_,edge=build()
    with pytest.raises(GeometryError,match='Attachment lineage'):
        query(model,edge)


def test_none_scope_is_a_typed_refusal():
    model,_,_,edge=build()
    receipt=replace(query(model,edge),scope=None)
    with pytest.raises(GeometryError,match='owner scope'):
        validate(model,receipt)


def test_final_callback_hidden_index_mutation_is_rejected():
    model,_,_,edge=build()
    count=0
    def count_check(*_):
        nonlocal count
        count+=1
    query(model,edge,cancellation_check=count_check)
    calls=0
    def mutate_last(*_):
        nonlocal calls
        calls+=1
        if calls==count:
            model._edge_coedges[edge]=()
    with pytest.raises(GeometryError,match='declaration changed'):
        query(model,edge,cancellation_check=mutate_last)


def test_transient_callback_edits_do_not_contaminate_detached_definitions():
    model,_,_,edge=build()
    baseline=query(model,edge)
    key=baseline.current_face_ids[0]
    original=model.faces[key]
    calls=0
    def transient(*_):
        nonlocal calls
        calls+=1
        if calls==1:
            model._faces[key]=replace(original,metadata={'transient':'not-the-snapshot'})
        elif calls==2:
            model._faces[key]=original
    assert query(model,edge,cancellation_check=transient)==baseline


def test_undeclared_second_shared_edge_cannot_omit_its_third_sheet(monkeypatch):
    import anygeometry.joint_edges as joints
    original=joints.declare_joint_edge
    def declare(model,edge,*args,**kwargs):
        if all(model.vertex_position(vertex)[0]==1 for vertex in
               (model.edges[edge].start,model.edges[edge].end)):
            return
        return original(model,edge,*args,**kwargs)
    monkeypatch.setattr(joints,'declare_joint_edge',declare)
    model,_,sheets,edge=build(third=True)
    # Public serialization performs the existing full model qualification.
    assert to_dict(model)['faces']
    assert len(model.sheets_using_edge(edge))==2
    second=[key for key in model.edges if len(model.sheets_using_edge(key))==2 and
            sheets[-1] in model.sheets_using_edge(key)]
    assert len(second)==1
    assert not joints.query_joint_edge(model,second[0]).declared
    with pytest.raises(GeometryError,match='declared multi-Sheet joint'):
        query(model,edge)


def test_final_callback_single_owner_edge_index_mutation_is_rejected():
    model,_,_,edge=build()
    receipt=query(model,edge)
    boundary=next(row['id'] for row in receipt.current_records['edges']
                  if len(model.sheets_using_edge(row['id']))==1)
    count=0
    def count_check(*_):
        nonlocal count
        count+=1
    query(model,edge,cancellation_check=count_check)
    calls=0
    def mutate_last(*_):
        nonlocal calls
        calls+=1
        if calls==count:
            model._edge_coedges[boundary]=()
    with pytest.raises(GeometryError,match='derived occurrence changed'):
        query(model,edge,cancellation_check=mutate_last)
