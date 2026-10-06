"""Original spatial relations survive genuine public preparation and repeated cuts."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as F
import json
import os
from pathlib import Path
import sys
from importlib.metadata import version

import pytest
import anygeometry
from anygeometry import (GeometryModel,GeometryError,ParameterRange,plan_intersections,apply_intersections,
    query_prepared_native_material_reference_scope as query,
    validate_prepared_native_material_reference_scope_binding as validate)
from anygeometry.structural import JunctionMemberUse
from anygeometry.native_arc_parameter_maps import NativeArcParameterMapPolicy


def test_source_runtime():
    origin=Path(anygeometry.__file__).resolve()
    assert origin==Path(__file__).resolve().parents[1]/'src/anygeometry/__init__.py'
    assert all(os.environ[k]=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'))
    if os.environ.get('NATIVE_ATTACHMENT_RUNTIME_EVIDENCE'):
        Path(os.environ['NATIVE_ATTACHMENT_RUNTIME_EVIDENCE']).write_text(json.dumps(dict(executable=sys.executable,
            python_version=sys.version,source_origin=str(origin),numpy_version=version('numpy'),pytest_version=version('pytest'),
            threads={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}),indent=2))


def _cut(model,x):
    return model.add_plate(model.add_points(((x,0,-1),(x,4,-1),(x,4,1),(x,0,1))))


def _prepare(model,faces=None):
    faces=tuple(model.faces) if faces is None else tuple(faces)
    apply_intersections(model,plan_intersections(model,faces,policy='connect'),policy='connect')


def _axis_fixture(cuts=(1.,3.),duplicates=False):
    model=GeometryModel();root=model.add_plate(model.add_points(((0,0,0),(4,0,0),(4,4,0),(0,4,0))))
    sheet=model.add_sheet((root,),name='authored owner')
    member=model.add_member((model.add_line(*model.add_points(((0,2,0),(4,2,0)))),))
    attachments=[model.add_attachment(member,'member_on_face','face',root,ParameterRange(0,1),
        (ParameterRange(0,1),ParameterRange(.5,.5)),evidence='exact',tolerance_used=1e-8,sheet_id=sheet)]
    if duplicates:
        attachments.append(model.add_attachment(member,'member_on_face','face',root,ParameterRange(0,1),
            (ParameterRange(0,1),ParameterRange(.5,.5)),evidence='exact',tolerance_used=1e-8,sheet_id=sheet))
    junction=model.ensure_junction('overlap',(JunctionMemberUse(member,ParameterRange(0,1)),),
                                  attachment_ids=tuple(attachments),sheet_ids=(sheet,))
    added=tuple(_cut(model,x) for x in cuts);_prepare(model,(root,*added))
    return model,root,member,tuple(attachments),junction


def test_original_axis_attachment_and_junction_qualify_through_public_cuts():
    model,root,member,attachments,junction=_axis_fixture()
    receipt=query(model,[root]);inventory=receipt.inventory
    assert receipt.document_material_qualified
    assert inventory['attachment_native_maps'][0]['classification']!='refused',inventory['attachment_native_maps']
    assert inventory['junction_native_maps'][0]['classification']!='refused',inventory['junction_native_maps']
    assert len(inventory['attachment_native_maps'][0]['current_attachment_ids'])==3
    for identifier in inventory['attachment_native_maps'][0]['current_attachment_ids']:
        assert ('attachment',attachments[0]) in model.attachments[identifier].lineage
    validate(model,receipt)


def test_repeated_child_cuts_and_idempotent_preparation_preserve_original_identity():
    model,root,member,attachments,junction=_axis_fixture(cuts=(1.,3.))
    first=query(model,[root]);source=first.scope.authored_document
    _prepare(model)
    receipt=query(model,[root]);rows=receipt.inventory['attachment_native_maps']
    row=next(r for r in rows if r['source_attachment_id']==attachments[0])
    assert row['classification']!='refused',row
    assert row['source_checksum']==source['checksum']['value']
    assert row['source_revision']==source['revision']
    assert len(row['current_attachment_ids'])==3
    assert receipt.inventory['junction_native_maps'][0]['classification']!='refused'
    validate(model,receipt)


def test_new_authorship_after_preparation_retains_existing_stale_provenance_refusal():
    model,root,*_=_axis_fixture(cuts=(1.,))
    receipt=query(model,[root]);_cut(model,3.);_prepare(model)
    with pytest.raises(GeometryError,match='stale'):query(model,[root])
    with pytest.raises(GeometryError):validate(model,receipt)


def test_duplicate_authored_relations_have_distinct_prospective_source_identity():
    model,root,member,attachments,junction=_axis_fixture(duplicates=True)
    rows=query(model,[root]).inventory['attachment_native_maps']
    assert {r['source_attachment_id'] for r in rows}==set(attachments)
    assert all(r['classification']!='refused' for r in rows),rows
    assert not set(rows[0]['current_attachment_ids']) & set(rows[1]['current_attachment_ids'])


def test_boundary_point_preserves_every_incident_face_owner():
    model=GeometryModel();root=model.add_plate(model.add_points(((0,0,0),(4,0,0),(4,4,0),(0,4,0))))
    member=model.add_member((model.add_line(*model.add_points(((2,2,-1),(2,2,1)))),))
    source=model.add_attachment(member,'member_through_face','face',root,ParameterRange(.5,.5),
        (ParameterRange(.5,.5),ParameterRange(.5,.5)),evidence='exact',tolerance_used=1e-8)
    cut=_cut(model,2.);_prepare(model,(root,cut))
    receipt=query(model,[root]);row=receipt.inventory['attachment_native_maps'][0]
    assert row['classification']!='refused',row
    assert len(row['current_attachment_ids'])==2
    assert {model.attachments[key].target_id for key in row['current_attachment_ids']}==set(receipt.material_rows[0].current_face_ids)


def test_wrong_original_spatial_relation_cannot_qualify_from_exact_label():
    model=GeometryModel();root=model.add_plate(model.add_points(((0,0,0),(4,0,0),(4,4,0),(0,4,0))))
    member=model.add_member((model.add_line(*model.add_points(((20,20,0),(24,20,0)))),))
    model.add_attachment(member,'member_on_face','face',root,ParameterRange(0,1),
        (ParameterRange(0,1),ParameterRange(.5,.5)),evidence='exact',max_residual=0.,tolerance_used=1e-8)
    _prepare(model)
    receipt=query(model,[root]);row=receipt.inventory['attachment_native_maps'][0]
    assert row['classification']=='refused'
    assert not receipt.geometry_native_reference_maps_qualified


def _map_inputs(receipt):
    p=receipt.scope.face_preimages;i=receipt.inventory
    from anygeometry.cylinder_charts import _Proof
    return [receipt.scope.authored_document,receipt.scope.current_document,p.face_descendants,i['native_edge_maps'],
        i['member_native_maps'],{r['id']:True for r in receipt.scope.authored_document['structural']['members']},
        {r['id']:True for r in receipt.scope.authored_document['structural']['sheets']},
        {key:(kind,data) for key,kind,data in p.authored_native_supports},
        {key:(kind,data) for key,kind,data in p.current_native_supports},
        {f for row in receipt.material_rows if row.document_material_qualified for f in row.current_face_ids},
        _Proof(NativeArcParameterMapPolicy(),None),None,
        tuple((kind,key,tuple(map(tuple,children))) for kind,key,children in i['replacement_history_snapshot'])]


@pytest.mark.parametrize('adversary',('missing','wrong_range','wrong_rectangle','payload','source_identity','extra'))
def test_whole_relation_census_refuses_mutated_record_maps(adversary):
    from anygeometry.native_attachment_maps import map_native_attachment_references
    model,root,*_=_axis_fixture(duplicates=adversary=='source_identity');receipt=query(model,[root]);args=_map_inputs(receipt)
    current=args[1]['structural']['attachments'];row=next(r for r in current if r['kind']=='member_on_face')
    if adversary=='missing':current.remove(row)
    elif adversary=='wrong_range':row['member_range'][1]=row['member_range'][0]
    elif adversary=='wrong_rectangle':row['target_parameters']=[[0.,0.],[0.,0.]]
    elif adversary=='payload':row['metadata']={'different':True}
    elif adversary=='source_identity':row['lineage'].append(['attachment',args[0]['structural']['attachments'][1]['id']])
    else:
        extra=deepcopy(row);extra['id']=999999;current.append(extra)
    rows,_,_,_=map_native_attachment_references(*args)
    assert any(r['classification']=='refused' for r in rows),rows


def test_cancellation_budget_and_binding_remain_atomic_after_attachment_maps():
    model,root,*_=_axis_fixture();receipt=query(model,[root])
    with pytest.raises(GeometryError,match='cancelled'):query(model,[root],cancellation_check=lambda _:True)
    with pytest.raises(GeometryError,match='budget'):query(model,[root],policy=NativeArcParameterMapPolicy(1))
    with pytest.raises(GeometryError,match='not issued'):validate(model,replace(receipt))
    validate(model,receipt)


def test_composed_world_bound_matches_independent_affine_endpoint_oracle():
    from anygeometry.native_attachment_maps import map_native_attachment_references
    model,root,member,_,_=_axis_fixture(cuts=());receipt=query(model,[root]);args=_map_inputs(receipt)
    use=args[1]['structural']['member_edge_uses'][0]
    edge=next(e for e in args[1]['edges'] if e['id']==use['edge_id'])
    vertices={v['id']:v for v in args[1]['vertices']};delta=2.**-35
    vertices[edge['start']]['position'][0]+=delta
    vertices[edge['end']]['position'][0]-=delta
    rows,_,_,_=map_native_attachment_references(*args)
    assert rows[0]['classification']!='refused',rows
    bound=F(*rows[0]['maps'][0]['squared_world_residual_bound'])
    assert bound==F(delta)**2
    for t in (F(0),F(1,3),F(1,2),F(2,3),F(1)):
        assert (F(delta)*(1-2*t))**2<=bound


def test_original_vertex_and_member_edge_points_survive_boundary_edge_cuts():
    from test_prepared_model_scope import fixture,prepare
    model,(roots,sheet,member,attachments,isolated)=fixture()
    prepare(model);receipt=query(model,[roots[0]])
    rows={r['source_attachment_id']:r for r in receipt.inventory['attachment_native_maps']}
    assert all(rows[key]['classification']!='refused' for key in attachments),rows
    validate(model,receipt)


def test_incoming_original_member_station_and_junction_survive_member_cut():
    model=GeometryModel();root=model.add_plate(model.add_points(((0,0,0),(4,0,0),(4,4,0),(0,4,0))))
    edge=model.add_line(*model.add_points(((0,2,0),(4,2,0))));first=model.add_member((edge,))
    second=model.add_member((model.add_line(*model.add_points(((3,2,0),(3,2,1)))),))
    attachment=model.add_attachment(second,'member_endpoint_on_member','member',first,ParameterRange.point(0.),
        (ParameterRange.point(.75),),evidence='exact',tolerance_used=1e-8)
    junction=model.ensure_junction('crossing',(JunctionMemberUse(first,ParameterRange.point(.75)),
        JunctionMemberUse(second,ParameterRange.point(0.))),attachment_ids=(attachment,))
    cut=_cut(model,2.);_prepare(model,(root,cut))
    receipt=query(model,[root]);rows=receipt.inventory['attachment_native_maps']
    assert rows[0]['classification']!='refused',rows
    assert receipt.inventory['junction_native_maps'][0]['classification']!='refused'


@pytest.mark.parametrize('adversary',('missing_attachment','extra_attachment','member_range','sheet_owner'))
def test_original_junction_complete_relation_inventory_refuses_adversaries(adversary):
    from anygeometry.native_attachment_maps import map_native_attachment_references,map_native_junction_references
    model,root,*_=_axis_fixture();receipt=query(model,[root]);args=_map_inputs(receipt)
    target=args[1]['structural']['junctions'][0]
    if adversary=='missing_attachment':target['attachment_ids'].pop()
    elif adversary=='extra_attachment':target['attachment_ids'].append(999999)
    elif adversary=='member_range':target['member_uses'][0]['member_range']=[0.,.5]
    else:target['sheet_ids']=[999999]
    maps,_,_,context=map_native_attachment_references(*args)
    rows=map_native_junction_references(args[0],args[1],maps,args[5],args[6],context)
    assert rows[0]['classification']=='refused'


def test_reversed_authenticated_edge_interval_is_composed_once():
    from anygeometry.native_attachment_maps import map_native_attachment_references
    model=GeometryModel();root=model.add_plate(model.add_points(((0,0,0),(4,0,0),(4,4,0),(0,4,0))))
    edge=model.faces[root].loop[0].edge;member=model.add_member((edge,));model.reverse_member(member)
    model.add_attachment(member,'member_on_boundary','edge',edge,ParameterRange(0,1),(ParameterRange(0,1),),
                         evidence='exact',tolerance_used=1e-8)
    _prepare(model);receipt=query(model,[root]);args=_map_inputs(receipt)
    target=next(e for e in args[1]['edges'] if e['id']==edge)
    target['start'],target['end']=target['end'],target['start']
    use=next(u for u in args[1]['structural']['member_edge_uses'] if u['member_id']==member)
    use['orientation']='forward'
    for mapping in args[3]:
        if mapping['edge_id']==edge:mapping['interval']=[[1,1],[0,1]]
    for mapping in args[4]:
        if mapping['current_edge_id']==edge:mapping['native_map']['interval']=[[1,1],[0,1]]
    rows,_,_,_=map_native_attachment_references(*args)
    assert rows[0]['classification']!='refused',rows
    assert F(*rows[0]['maps'][0]['squared_world_residual_bound'])==0


@pytest.mark.parametrize('kind',('face_singleton','edge_point_duplicate','unknown_face_history','unknown_edge_history','unknown_attachment_history'))
def test_review_relation_multiplicity_and_history_public_caller_refuses(kind,monkeypatch):
    import anygeometry.native_attachment_maps as mapping
    if kind=='edge_point_duplicate':
        model,root=_edge_point_fixture()
    else:
        model,root,*_=_axis_fixture()
    original=mapping.map_native_attachment_references
    def altered(before,current,*args,**kwargs):
        records=current['structural']['attachments']
        row=next(r for r in records if r['kind']=='vertex_on_edge') if kind=='edge_point_duplicate' else next(r for r in records if r['kind']=='member_on_face')
        if kind in ('face_singleton','edge_point_duplicate'):
            extra=deepcopy(row);extra['id']=999999
            if not any(k=='attachment' for k,_ in extra['lineage']):extra['lineage'].append(['attachment',row['id']])
            if kind=='face_singleton':extra['member_range']=[row['member_range'][1]]*2
            records.append(extra)
        else:
            row['lineage'].append([kind.removeprefix('unknown_').removesuffix('_history'),999999])
        return original(before,current,*args,**kwargs)
    monkeypatch.setattr(mapping,'map_native_attachment_references',altered)
    receipt=query(model,[root])
    assert not receipt.geometry_native_reference_maps_qualified
    assert any(row['classification']=='refused' for row in receipt.inventory['attachment_native_maps'])


def test_review_edge_point_at_split_authenticates_first_child_owner():
    from anygeometry.native_attachment_maps import map_native_attachment_references
    model,root=_edge_point_fixture();receipt=query(model,[root])
    args=_map_inputs(receipt);records=args[1]['structural']['attachments']
    row=next(r for r in records if r['kind']=='vertex_on_edge')
    before=next(r for r in args[0]['structural']['attachments'] if r['id']==row['id'])
    point=F(before['target_parameters'][0][0])
    alternatives=[]
    for native in args[3]:
        if native['ancestor_edge_id']!=before['target_id'] or native['edge_id']==row['target_id']:continue
        a,b=(F(*p) for p in native['interval'])
        if min(a,b)<=point<=max(a,b):alternatives.append((native['edge_id'],(point-a)/(b-a)))
    assert alternatives
    row['target_id']=alternatives[0][0];row['target_parameters']=[[float(alternatives[0][1])]*2]
    rows,_,_,_=map_native_attachment_references(*args)
    assert next(r for r in rows if r['source_attachment_id']==row['id'])['classification']=='refused'


def _edge_point_fixture():
    model=GeometryModel();root=model.add_plate(model.add_points(((0,0,0),(4,0,0),(4,4,0),(0,4,0))))
    edge=model.faces[root].loop[0].edge;vertex=model.add_point(2.,0.,0.)
    model.add_attachment(None,'vertex_on_edge','edge',edge,ParameterRange.point(0.),(ParameterRange.point(.5),),
        source_kind='vertex',source_id=vertex,evidence='exact',tolerance_used=1e-8)
    cut=_cut(model,2.);_prepare(model,(root,cut))
    return model,root


def test_review_untracked_source_lineage_cannot_inherit_derived_owner_qualification(monkeypatch):
    import anygeometry.native_attachment_maps as mapping
    model,root,*_=_axis_fixture()
    roots=query(model,[root]).scope.face_preimages.authored_face_ids
    assert query(model,roots).geometry_native_reference_maps_qualified
    original=mapping.map_native_attachment_references
    def ambiguous(*args,**kwargs):
        rows,sources,untracked,context=original(*args,**kwargs)
        derived=next(r['id'] for r in args[1]['structural']['attachments'] if r['kind']=='sheet_on_joint')
        return rows,sources,[*untracked,derived],context
    monkeypatch.setattr(mapping,'map_native_attachment_references',ambiguous)
    receipt=query(model,roots)
    assert receipt.inventory['untracked_attachment_ids']
    assert not receipt.geometry_native_reference_maps_qualified


def test_review_reversed_alias_point_owner_follows_ordered_history():
    from anygeometry.native_attachment_maps import map_native_attachment_references
    model,root=_edge_point_fixture();receipt=query(model,[root]);args=_map_inputs(receipt)
    row=next(r for r in args[1]['structural']['attachments'] if r['kind']=='vertex_on_edge')
    source=next(r for r in args[0]['structural']['attachments'] if r['id']==row['id'])
    left=row['target_id'];point=F(source['target_parameters'][0][0])
    right=next(native for native in args[3] if native['ancestor_edge_id']==source['target_id'] and
        native['edge_id']!=left and min(F(*x) for x in native['interval'])<=point<=max(F(*x) for x in native['interval']))
    right_id=right['edge_id'];a,b=(F(*x) for x in right['interval'])
    history=[r for r in args[12] if r[:2]!=('edge',source['target_id'])]
    history.extend((('edge',source['target_id'],(('edge',999999),)),
                    ('edge',999999,(('edge',right_id),('edge',left)))))
    args[12]=tuple(history)
    saved=deepcopy(row)
    row['target_id']=right_id;row['target_parameters']=[[float((point-a)/(b-a))]*2]
    rows,_,_,_=map_native_attachment_references(*args)
    assert next(r for r in rows if r['source_attachment_id']==row['id'])['classification']!='refused'
    row.update(saved)
    rows,_,_,_=map_native_attachment_references(*args)
    assert next(r for r in rows if r['source_attachment_id']==row['id'])['classification']=='refused'
