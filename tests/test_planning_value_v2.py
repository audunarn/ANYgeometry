"""Finite production wire grammar; diagnostic v1 stays the default."""
import json
from dataclasses import fields
import numpy as np
import pytest
from anygeometry import _planning_values as values
from anygeometry._planning_qualification import QualificationDecline


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


@pytest.mark.parametrize('value',[None,False,True,2**90,-3,0.,-0.,float.fromhex('0x0.0000000000001p-1022'),
    'AAAAAAAAAAA=','\u03c0',np.float64(-0.),np.int64(-9),np.bool_(True)])
def test_v2_exact_scalar_types_and_bits_keep_v1_default(value):
    old=values.encode_value(value);packet=values.encode_value(value,codec='v2')
    assert packet[0]=='v2'
    out=values.decode_value(packet,codec='v2')
    assert type(out) is type(value) and values.encode_value(out)==old
    assert values.encode_value(value)==old and values.encode_value(out,codec='v2')==packet


def records():
    v=values
    plane=v.Plane(np.zeros(3),np.array([1.,0.,0.]),np.array([0.,1.,0.]))
    cylinder=v.Cylinder(np.zeros(3),np.array([0.,0.,1.]),np.array([1.,0.,0.]),1.,1.)
    other=v.Cylinder(np.zeros(3),np.array([1.,0.,0.]),np.array([0.,1.,0.]),1.,1.)
    line=v.LinePath((0.,0.,0.),(1.,0.,0.));path=v.ArrangementPath(line)
    point=v.ArrangementPoint((0.,0.,0.));cell=v.ArrangementCell(((0,True),),())
    material=v.MaterialArrangement(2,(path,),(cell,),1.,support=plane)
    ruled=v.PlaneRuledSupport((0.,0.,0.),(1.,0.,0.),(0.,1.,0.),0.,1.)
    quadric=v.QuadricSupport('plane',axis=(0.,1.,0.))
    return (plane,cylinder,line,path,point,cell,material,
        v.EllipticArc((0.,0.,0.),(1.,0.,0.),(0.,1.,0.),0.,1.),
        v.CylinderIntersectionCurve(cylinder,other,0.,.5,1),v._CylinderSupport.from_surface(cylinder),
        v.BezierQuadricCurve(ruled,quadric,0.,1.),quadric,ruled)


@pytest.mark.parametrize('index',range(13))
def test_v2_every_registered_record_preserves_constructor_wire(index):
    value=records()[index]
    assert type(value) is values.registry()[index]
    old=values.encode_value(value);packet=values.encode_value(value,codec='v2')
    assert values.expand_value_v2(packet)==old
    assert values.encode_value(values.decode_value(packet,codec='v2'))==old


def test_v2_fixed_schema_matches_all_source_owned_fields():
    assert values._V2_RECORD_FIELDS==tuple((cls.__name__,tuple(f.name for f in fields(cls) if f.init)) for cls in values.registry())
    assert values.value_codec_contract()['version']=='v2'


@pytest.mark.parametrize('bad',[
    ['v1',None],['v2'],['v2',0.],['v2','AAAAAAAAAAB='],['v2','AAAAAAAAAAA'],['v2',' AAAAAAAAAAA='],
    ['v2',[0]],['v2',[2,1]],['v2',[3,True]],['v2',[4,'AAAAAAAAAAA=']],['v2','f/AAAAAAAAA='],
    ['v2',[5,2,'Ag==']],['v2',[1,3]],['v2',[5,True,'AA==']],['v2',[5,3,'AA==']],
    ['v2',[5,0,'AA==']],['v2',[11,0]],['v2',[6,True,[]]],['v2',[9,0,[1],'AAAAAAAAAAA=',1]],
    ['v2',[10,0,99,[]]],['v2',[10,0,0,[]]],['v2',[10,0,0,[None,None,None],[]]],
    ['v2',[8,0,[[1,None],[1,None]]]]])
def test_v2_malformed_or_alternative_encodings_refuse(bad):
    with pytest.raises(QualificationDecline):values.decode_value(bad,codec='v2')


def test_v2_flags_shared_identity_and_signed_zero():
    array=np.array([-0.,1.]);array.flags.writeable=False
    packet=values.encode_value((array,array),codec='v2')
    out=values.decode_value(packet,codec='v2')
    assert out[0] is out[1] and out[0].tobytes()==array.tobytes() and not out[0].flags.writeable
    with pytest.raises(QualificationDecline):values.encode_value((array,array.view()),codec='v2')


def test_v2_version_is_explicit_and_canonical_string_required():
    packet=values.encode_value((1.,),codec='v2')
    with pytest.raises(QualificationDecline):values.decode_value(packet)
    with pytest.raises(QualificationDecline):values.decode_value(values.encode_value((1.,)),codec='v2')
    with pytest.raises(QualificationDecline):values.decode_stage_values([json.dumps(packet)],codec='v2')
    with pytest.raises(QualificationDecline):values.encode_value(None,codec='unknown')


def test_v2_compact_aggregate_expanded_value_and_node_bounds():
    packet=values.encode_value(np.zeros(64),codec='v2');wire=canonical(packet)
    expanded=len(canonical(values.encode_value(np.zeros(64))).encode())
    with pytest.raises(QualificationDecline):values.decode_value(packet,codec='v2',payload_limit=expanded-1)
    assert values.decode_value(packet,codec='v2',payload_limit=expanded).shape==(64,)
    with pytest.raises(QualificationDecline):values.decode_stage_values([wire,wire],codec='v2',payload_limit=len(canonical([wire,wire]).encode())-1)
    with pytest.raises(QualificationDecline):values.encode_value((1,2),codec='v2',limit=2)


def test_v2_late_malformed_packet_never_commits_deferred_audit(monkeypatch):
    packet=values.encode_value(records()[1],codec='v2');calls=[]
    original=values.audit
    def observe(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(values,'audit',observe)
    with pytest.raises(QualificationDecline):values.decode_stage_values([canonical(packet),'["v2",[11,0]]'],codec='v2')
    assert calls==[]
    assert len(values.decode_stage_values([canonical(packet)],codec='v2'))==1 and calls==[1]


def task_rows():
    return [dict(kind='pair',first=1,second=2,tolerance=1e-8,face_tolerances=[1e-8,1e-8],cost=3.),
        dict(kind='face',face=1,inputs=values.encode_value(([],[])),support=values.encode_value(records()[0]),
            tolerance=1e-8,area_tolerance=1e-12,seam='baseline',cost=1.)]


def test_v2_transport_copy_preserves_original_tasks_and_exact_metadata():
    original=task_rows();before=canonical(original)
    transport=values.transport_tasks_v2(original)
    assert canonical(original)==before and transport[0] is not original[0]
    assert transport[0]['face_tolerances'] is not original[0]['face_tolerances']
    assert transport[1]['inputs'][0]=='v2' and transport[1]['support'][0]=='v2'
    values.verify_transport_tasks_v2(original,transport)


@pytest.mark.parametrize('mutation',['order','id','cost','bits'])
def test_v2_transport_metadata_rebinding_refuses(mutation):
    original=task_rows();transport=list(values.transport_tasks_v2(original))
    if mutation=='order':transport.reverse()
    elif mutation=='id':transport[0]['first']=2
    elif mutation=='cost':transport[0]['cost']=4.
    else:
        original[0]['tolerance']=0.;transport[0]['tolerance']=-0.
    with pytest.raises(QualificationDecline):values.verify_transport_tasks_v2(original,transport)


@pytest.mark.parametrize('bad',[None,{'version':'v1','schema_sha256':'a'*64},{'version':'v2','schema_sha256':'a'*64},
    {'version':'v2','schema_sha256':'a'*64,'extra':0}])
def test_v2_role_contract_cannot_downgrade_or_rebind(bad):
    with pytest.raises(QualificationDecline):values.verify_value_codec_contract(bad)


def test_v2_worker_context_and_role_receipts_bind_version():
    from anygeometry._planning_canaries import _fixture
    from anygeometry._planning_executor import freeze_model
    from anygeometry._planning_stages import WorkerModel
    from anygeometry._planning_session import _verify_worker_resources
    model,_=_fixture();context=WorkerModel(freeze_model(model),value_codec='v2')
    context.verify();context.value_codec=None
    with pytest.raises(QualificationDecline,match='codec'):context.verify()
    identity={'binary_sha256':'a'*64};codec=values.value_codec_contract()
    report=dict(resource_guard=dict(controller_loaded=False,native_adapter_loaded=False,native_core_loaded=False,
        binary_sha256=identity['binary_sha256']),value_codec=codec)
    _verify_worker_resources(report,identity,codec)
    report.pop('value_codec')
    with pytest.raises(QualificationDecline,match='codec'):_verify_worker_resources(report,identity,codec)


def test_v2_decode_uses_independent_expansion_before_constructor(monkeypatch):
    array=np.array([-0.,1.]);old=values.encode_value(array);packet=values.encode_value(array,codec='v2')
    decode=values._decode_value
    def mutate_input(expanded,**kwargs):
        packet[1][3]='AAAAAAAAAAAAAAAAAAAAAA=='
        return decode(expanded,**kwargs)
    monkeypatch.setattr(values,'_decode_value',mutate_input)
    restored=values.decode_value(packet,codec='v2')
    assert values.encode_value(restored)==old


def test_v2_later_constructor_alias_is_still_refused(monkeypatch):
    wire=canonical(values.encode_value(records()[1],codec='v2'));owners=[];aliases=[]
    original=values.record_manifest
    def introduce_alias(value,fresh,**kwargs):
        result=original(value,fresh,**kwargs)
        if type(value) is values.Cylinder:
            owners.append(value)
            if len(owners)==2:aliases.append(owners[0]._circumferential)
        return result
    monkeypatch.setattr(values,'record_manifest',introduce_alias)
    with pytest.raises(QualificationDecline,match='alias'):values.decode_stage_values([wire,wire],codec='v2')
    monkeypatch.setattr(values,'record_manifest',original);owners.clear();aliases.clear()
    assert len(values.decode_stage_values([wire,wire],codec='v2'))==2


def test_v2_declared_field_schema_change_refuses_initial_contract(monkeypatch):
    table=list(values._V2_RECORD_FIELDS);name,names=table[0];table[0]=(name,tuple(reversed(names)))
    monkeypatch.setattr(values,'_V2_RECORD_FIELDS',tuple(table))
    with pytest.raises(QualificationDecline,match='schema'):values.value_codec_contract()
