"""PP5c bounded alias qualification, explicitly not a purity/aliasing proof."""
import ctypes
from dataclasses import fields
import sys
import numpy as np
import pytest
from anygeometry import _planning_qualification as q
from anygeometry._planning_executor import freeze_model, thaw_model, unpack, pack
from anygeometry.surfaces import Cylinder
from anygeometry.serialization import from_dict,to_dict
from functools import cache
from pathlib import Path


def cylinder():
    return Cylinder((0.,0.,0.),(0.,0.,1.),(1.,0.,0.),2.,3.,.2,1.5)


@cache
def document():
    # Captured real Cylinder face closure; full project admission runs separately.
    import json
    return json.loads((Path(__file__).parent/'data/pp5c-cylinder-input.json').read_text(encoding='utf-8'))


def model():
    return from_dict(document())


def support(value):
    return next(face.surface for face in value.faces.values() if type(face.surface) is Cylinder)


def test_pristine_record_is_admitted_without_mutation():
    value=cylinder(); before={key:id(part) for key,part in vars(value).items()}
    q._check_model_record_state({'surface':value})
    assert set(vars(value))==set(before)
    assert all(id(vars(value)[key]) == old for key,old in before.items())


@pytest.mark.parametrize('readonly',[False,True])
def test_transport_binds_derived_bytes_and_reconstructs_fresh_arrays(readonly):
    value=model()
    support(value)._circumferential.flags.writeable=not readonly
    packet=freeze_model(value)
    assert 'constructor_state' in unpack(packet,'model')
    rebuilt=thaw_model(packet)
    assert to_dict(rebuilt)==to_dict(value)
    assert not np.shares_memory(support(value)._circumferential,support(rebuilt)._circumferential)
    assert support(value)._circumferential.tobytes()==support(rebuilt)._circumferential.tobytes()
    assert support(rebuilt)._circumferential.flags.writeable is (not readonly)


@pytest.mark.parametrize('kind',['edited-extra','edited-fields','shape','dtype','nonfinite','signed-zero',
    'unknown-extra','unregistered-class','other-record','external-reference','external-view','field-alias','noncontiguous'])
def test_unqualified_state_refuses(kind):
    value=cylinder();other=None;outside=None
    if kind=='edited-extra': value._circumferential[1]=2.
    elif kind=='edited-fields': object.__setattr__(value,'radial_direction',np.array((0.,1.,0.)))
    elif kind=='shape': object.__setattr__(value,'_circumferential',value._circumferential.reshape(1,3))
    elif kind=='dtype': object.__setattr__(value,'_circumferential',value._circumferential.astype(np.float32))
    elif kind=='nonfinite': value._circumferential[1]=np.inf
    elif kind=='signed-zero': value._circumferential[0]=-0.
    elif kind=='unknown-extra': object.__setattr__(value,'unknown',1)
    elif kind=='unregistered-class':
        class Unknown(Cylinder): pass
        value=Unknown((0.,0.,0.),(0.,0.,1.),(1.,0.,0.),2.,3.,.2,1.5)
    elif kind=='other-record':
        other=cylinder();object.__setattr__(other,'_circumferential',value._circumferential)
    elif kind=='external-reference': outside=value._circumferential
    elif kind=='external-view': outside=value._circumferential.view()
    elif kind=='field-alias': object.__setattr__(value,'origin',value._circumferential)
    elif kind=='noncontiguous': object.__setattr__(value,'_circumferential',np.array([0.,9.,1.,9.,0.,9.])[::2])
    with pytest.raises(q.QualificationDecline):
        q._check_model_record_state({'surface':value,'other':other})
    if outside is not None: assert outside.size==3


def test_documented_limitation_raw_native_view_is_not_detected():
    value=cylinder()
    refs=sys.getrefcount(value._circumferential)
    raw=(ctypes.c_double*3).from_address(value._circumferential.ctypes.data)
    external=np.ctypeslib.as_array(raw)
    assert np.shares_memory(external,value._circumferential)
    observed_refs=sys.getrefcount(value._circumferential)
    assert refs==observed_refs
    q._check_model_record_state({'surface':value})
    # Explicit ACCEPTED RESIDUAL RISK, not a pass claim for absence of aliases.
    external[1]=2.
    with pytest.raises(q.QualificationDecline): q._check_model_record_state({'surface':value})


def test_worker_rechecks_constructor_state_after_decode(monkeypatch):
    import anygeometry.serialization as codec
    value=model();packet=freeze_model(value)
    original=codec.from_dict
    def altered(document):
        fresh=original(document);support(fresh)._circumferential[1]=2.;return fresh
    monkeypatch.setattr(codec,'from_dict',altered)
    with pytest.raises(q.QualificationDecline): thaw_model(packet)


def test_constructor_state_cannot_be_omitted_from_transport():
    packet=freeze_model(model());body=unpack(packet,'model');body.pop('constructor_state')
    with pytest.raises(q.QualificationDecline): thaw_model(pack(body,'model'))

def test_gc_tracked_subclass_view_is_in_bounded_census():
    import gc
    class Tracked(np.ndarray): pass
    value=cylinder()
    raw=(ctypes.c_double*3).from_address(value._circumferential.ctypes.data)
    outside=np.ctypeslib.as_array(raw).view(Tracked)
    assert gc.is_tracked(outside)
    with pytest.raises(q.QualificationDecline,match='GC-census alias'):
        q._check_model_record_state({'surface':value})


def test_external_weak_array_reference_declines():
    import weakref
    value=cylinder();outside=weakref.ref(value._circumferential)
    with pytest.raises(q.QualificationDecline,match='weak'):
        q._check_model_record_state({'surface':value})
    assert outside() is value._circumferential


@pytest.mark.parametrize('field,value',[('c_contiguous',1),('shape',[3.0])])
def test_constructor_manifest_requires_exact_metadata_types(field,value):
    packet=freeze_model(model());body=unpack(packet,'model')
    body['constructor_state'][0][field]=value
    with pytest.raises(q.QualificationDecline,match='constructor-state'):
        thaw_model(pack(body,'model'))


def test_missing_registered_derived_attribute_declines_at_source():
    value=cylinder();object.__delattr__(value,'_circumferential')
    with pytest.raises(q.QualificationDecline,match='constructor schema'):
        q._check_model_record_state({'surface':value})


def test_unqualified_field_cannot_execute_array_conversion():
    calls=[]
    class Foreign:
        def __array__(self,dtype=None,copy=None):
            calls.append('executed');return np.zeros(3,dtype=dtype)
    value=cylinder();object.__setattr__(value,'origin',Foreign())
    with pytest.raises(q.QualificationDecline):
        q._check_model_record_state({'surface':value})
    assert calls==[]
