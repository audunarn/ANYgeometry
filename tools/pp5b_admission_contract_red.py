"""PP5b baseline contracts and outside-alias counterexample; no planning."""
import ctypes
import dataclasses
import gc
import sys
import numpy as np
import pytest
from anygeometry import _planning_qualification as q
from anygeometry.definition_binding import definition_checksum
from anygeometry.surfaces import Cylinder

def cylinder():
    return Cylinder((0.,0.,0.),(0.,0.,1.),(1.,0.,0.),1.,1.)

def fingerprint(value):
    a=value._circumferential
    return (a.dtype.str,a.shape,a.flags.c_contiguous,a.flags.owndata,a.flags.writeable,
            a.base is None,bool(np.all(np.isfinite(a))),a.tobytes(),definition_checksum(value))

def test_pristine_registered_constructor_state_admits():
    q._check_model_record_state({'surface':cylinder()})

@pytest.mark.parametrize('kind', ['edited-extra','edited-fields','shape','dtype','nonfinite',
    'other-record-alias','external-reference','external-view','signed-zero','unknown-extra','unregistered-class'])
def test_required_negative_remains_refused(kind):
    value=cylinder();other=None;outside=None
    if kind=='edited-extra': value._circumferential[1]=2.
    elif kind=='edited-fields': object.__setattr__(value,'radial_direction',np.array((0.,1.,0.)))
    elif kind=='shape': object.__setattr__(value,'_circumferential',value._circumferential.reshape(1,3))
    elif kind=='dtype': object.__setattr__(value,'_circumferential',value._circumferential.astype(np.float32))
    elif kind=='nonfinite': value._circumferential[1]=np.inf
    elif kind=='other-record-alias':
        other=cylinder();object.__setattr__(other,'_circumferential',value._circumferential)
    elif kind=='external-reference': outside=value._circumferential
    elif kind=='external-view': outside=value._circumferential.view()
    elif kind=='signed-zero': value._circumferential[0]=-0.
    elif kind=='unknown-extra': object.__setattr__(value,'unregistered',1)
    elif kind=='unregistered-class':
        class OtherCylinder(Cylinder): pass
        value=OtherCylinder((0.,0.,0.),(0.,0.,1.),(1.,0.,0.),1.,1.)
    with pytest.raises(q.QualificationDecline):
        q._check_model_record_state({'surface':value,'other':other})
    # Keep aliases alive across the gate. No negative test's state is stripped.
    if outside is not None: assert outside.size==3

def test_constructor_equality_and_owner_refcount_cannot_prove_no_outside_alias():
    value=cylinder()
    fresh=type(value)(**{f.name:getattr(value,f.name) for f in dataclasses.fields(value)})
    assert fingerprint(value)==fingerprint(fresh)
    before=fingerprint(value)
    before_refs=sys.getrefcount(value._circumferential)
    # A valid view of memory owned by this live, in-scope ndarray. No foreign memory access.
    raw=(ctypes.c_double*3).from_address(value._circumferential.ctypes.data)
    outside=np.ctypeslib.as_array(raw)
    assert np.shares_memory(value._circumferential,outside)
    assert sys.getrefcount(value._circumferential)==before_refs
    assert fingerprint(value)==before
    assert not gc.is_tracked(value._circumferential) and not gc.is_tracked(outside)
    assert not any(item is outside for item in gc.get_objects())
    original=value.evaluate(.25,0.)
    outside[1]=2.
    assert value.evaluate(.25,0.)[1]==2. and original[1]==1.
    assert definition_checksum(value)==before[-1]