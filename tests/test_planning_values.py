"""Exact typed stage value transport; separate from cold source-model admission."""
import importlib,math,struct
from dataclasses import replace
import numpy as np
import pytest
from anygeometry._planning_qualification import QualificationDecline
from anygeometry.surfaces import Plane,Cylinder
from anygeometry.material_arrangement import ArrangementPath,ArrangementPoint,MaterialArrangement,ArrangementCell
from anygeometry.arrangement_geometry import LinePath

def api():return importlib.import_module('anygeometry._planning_values')

def test_actual_material_records_roundtrip_and_signed_zero():
    p=Plane(np.array([-0.,0.,0.]),np.array([1.,0.,0.]),np.array([0.,1.,0.]))
    _=p.normal
    value=MaterialArrangement(2,(ArrangementPath(LinePath((0.,0.,0.),(1.,0.,0.))),),(ArrangementCell(((0,True),),()),),1.,support=p)
    packet=api().encode_value(value);restored=api().decode_value(packet)
    assert type(restored) is MaterialArrangement and api().encode_value(restored)==packet
    assert restored.support.origin.tobytes()==p.origin.tobytes()
    assert restored.support.normal.tobytes()==p.normal.tobytes()
    assert restored.support is not p

@pytest.mark.parametrize('readonly',[False,True])
def test_cylinder_extra_rebuilt_not_omitted(readonly):
    c=Cylinder(np.zeros(3),np.array([0.,0.,1.]),np.array([1.,0.,0.]),1.,1.)
    if readonly:c._circumferential.flags.writeable=False
    restored=api().decode_value(api().encode_value(c))
    assert restored._circumferential.tobytes()==c._circumferential.tobytes()
    assert restored._circumferential.flags.writeable==c._circumferential.flags.writeable
    assert not np.shares_memory(restored._circumferential,c._circumferential)

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),object(),np.array([float('nan')]),np.array([object()]),lambda:None])
def test_unknown_nonfinite_values_refuse(bad):
    with pytest.raises(QualificationDecline):api().encode_value(bad)

def test_plane_edited_cache_refuses():
    p=Plane(np.zeros(3),np.array([1.,0.,0.]),np.array([0.,1.,0.]))
    object.__setattr__(p,'normal',np.array([1.,0.,0.]))
    with pytest.raises(QualificationDecline):api().encode_value(p)

@pytest.mark.parametrize('reverse',[False,True])
def test_sample_comparison_private_plane_cache_presence(reverse):
    cold=Plane(np.zeros(3),np.array([1.,0.,0.]),np.array([0.,1.,0.]))
    warm=replace(cold);_=warm.normal
    first,second=(warm,cold) if reverse else (cold,warm)
    left,right=api().encode_value(first),api().encode_value(second)
    assert left!=right
    assert api().comparison_packet(left)==api().comparison_packet(right)
    assert 'normal' not in vars(cold) and left!=right

@pytest.mark.parametrize('difference',['normal_writeable','signed_zero','corrupt_normal','cylinder'])
def test_sample_comparison_preserves_exact_differences(difference):
    cold=Plane(np.zeros(3),np.array([1.,0.,0.]),np.array([0.,1.,0.]))
    other=replace(cold)
    if difference=='normal_writeable':other.normal.flags.writeable=True
    elif difference=='signed_zero':
        origin=np.zeros(3);origin[0]=-0.
        other=Plane(origin,cold.u_vector,cold.v_vector)
    elif difference=='corrupt_normal':_=other.normal
    else:
        cold=Cylinder(np.zeros(3),np.array([0.,0.,1.]),np.array([1.,0.,0.]),1.,1.)
        other=replace(cold);other._circumferential.flags.writeable=False
    left,right=api().encode_value(cold),api().encode_value(other)
    if difference=='corrupt_normal':
        right[4][0][1][2]='0'*64
        with pytest.raises(QualificationDecline):api().comparison_packet(right)
    else:assert api().comparison_packet(left)!=api().comparison_packet(right)

def test_unknown_record_extra_refuses():
    p=Plane(np.zeros(3),np.array([1.,0.,0.]),np.array([0.,1.,0.]))
    object.__setattr__(p,'unknown',3)
    with pytest.raises(QualificationDecline):api().encode_value(p)

def test_shared_record_and_array_identity_roundtrips():
    p=ArrangementPoint((0.,0.,0.));array=np.array([-0.,1.])
    value=(p,p,array,array)
    out=api().decode_value(api().encode_value(value))
    assert out[0] is out[1] and out[2] is out[3]
    assert out[2].tobytes()==array.tobytes()

@pytest.mark.parametrize('bad',[['record','os.system',[]],['ref',0],['float','7ff0000000000000'],['array','|O',[1],'0000000000000000',True]])
def test_malformed_wire_refuses(bad):
    with pytest.raises(QualificationDecline):api().decode_value(bad)

@pytest.mark.parametrize('value',[np.float64(-0.),np.int64(3),np.bool_(True)])
def test_exact_numpy_scalar_bits_and_type(value):
    out=api().decode_value(api().encode_value(value))
    assert type(out) is type(value) and out.tobytes()==value.tobytes()

def test_distinct_arrays_sharing_memory_refuse():
    array=np.arange(3,dtype=float)
    with pytest.raises(QualificationDecline,match='alias'):
        api().encode_value((array,array.view()))

def test_derived_shape_metadata_requires_exact_integer():
    p=Plane(np.zeros(3),np.array([1.,0.,0.]),np.array([0.,1.,0.]));_=p.normal
    packet=api().encode_value(p);packet[4][0][1][1]=[3.0]
    with pytest.raises(QualificationDecline):api().decode_value(packet)


def _stage_wire(value):
    import json
    return json.dumps(api().encode_value(value),sort_keys=True,separators=(',',':'))


def _cylinder():
    return Cylinder(np.zeros(3),np.array([0.,0.,1.]),np.array([1.,0.,0.]),1.,1.)


def test_closed_stage_decode_exact_mixed_and_single_final_audit(monkeypatch):
    module=api();c=_cylinder();c._circumferential.flags.writeable=False
    shared=np.array([-0.,1.]);first=(c,c,shared,shared)
    wires=[_stage_wire(first),_stage_wire(Plane(np.zeros(3),np.array([1.,0.,0.]),np.array([0.,1.,0.])))]
    expected=[module.decode_value(__import__('json').loads(wire)) for wire in wires]
    original=module.audit;calls=[]
    def observe(state,**kwargs):
        calls.append(state)
        return original(state,**kwargs)
    monkeypatch.setattr(module,'audit',observe)
    actual=module.decode_stage_values(wires)
    assert len(calls)==1
    assert actual[0][0] is actual[0][1] and actual[0][2] is actual[0][3]
    assert actual[0][2].tobytes()==shared.tobytes()
    assert actual[0][0]._circumferential.flags.writeable is False
    assert [module.encode_value(item) for item in actual]==[module.encode_value(item) for item in expected]


def test_closed_stage_decode_empty_and_immutable_input_snapshot(monkeypatch):
    module=api();wires=[_stage_wire(1),_stage_wire(-0.)]
    assert module.decode_stage_values([])==()
    original=module._decode_value;calls=[]
    def mutate(packet,**kwargs):
        calls.append(1);wires[1]=_stage_wire(9.)
        return original(packet,**kwargs)
    monkeypatch.setattr(module,'_decode_value',mutate)
    result=module.decode_stage_values(wires)
    assert result[0]==1 and struct.pack('!d',result[1])==struct.pack('!d',-0.)
    assert len(calls)==2


def test_closed_stage_decode_later_alias_refuses_and_next_call_is_clean(monkeypatch):
    module=api();wires=[_stage_wire(_cylinder()),_stage_wire(_cylinder())]
    original=module.record_manifest;owners=[];aliases=[]
    def introduce_alias(value,fresh,**kwargs):
        result=original(value,fresh,**kwargs)
        if type(value) is Cylinder:
            owners.append(value)
            if len(owners)==2:aliases.append(owners[0]._circumferential)
        return result
    monkeypatch.setattr(module,'record_manifest',introduce_alias)
    with pytest.raises(QualificationDecline,match='alias'):module.decode_stage_values(wires)
    monkeypatch.setattr(module,'record_manifest',original)
    aliases.clear();owners.clear()
    assert len(module.decode_stage_values(wires))==2


def test_closed_stage_decode_malformed_later_packet_does_not_commit(monkeypatch):
    module=api();wire=_stage_wire(_cylinder());original=module.audit;calls=[]
    def observe(state,**kwargs):calls.append(state);return original(state,**kwargs)
    monkeypatch.setattr(module,'audit',observe)
    with pytest.raises(QualificationDecline):module.decode_stage_values([wire,'["ref",0]'])
    assert calls==[]
    assert len(module.decode_stage_values([wire]))==1 and len(calls)==1


@pytest.mark.parametrize('bad',[['not json'],[3],object()])
def test_closed_stage_decode_input_schema_refuses(bad):
    with pytest.raises(QualificationDecline):api().decode_stage_values(bad)


def test_closed_stage_decode_packet_and_per_value_node_limits():
    wire=_stage_wire((1,2,3))
    with pytest.raises(QualificationDecline,match='payload'):api().decode_stage_values([wire],payload_limit=4)
    with pytest.raises(QualificationDecline,match='node'):api().decode_stage_values([wire],limit=2)


def test_closed_stage_decode_lone_surrogate_is_eligible_wire_refusal():
    with pytest.raises(QualificationDecline,match='UTF-8'):
        api().decode_stage_values(['["str","\ud800"]'])


def test_closed_stage_decode_keeps_aggregate_audit_bound(monkeypatch):
    module=api();wire=_stage_wire(_cylinder());original=module.audit;limits=[]
    def observe(state,**kwargs):
        limits.append(kwargs.get('limit',100000))
        return original(state,**kwargs)
    monkeypatch.setattr(module,'audit',observe)
    assert len(module.decode_stage_values([wire]))==1
    assert limits==[100000]
