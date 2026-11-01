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
