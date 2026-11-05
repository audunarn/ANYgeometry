"""Pure data comparison controls, never public geometry/plan qualification."""
from dataclasses import dataclass,field
from enum import Enum
from fractions import Fraction
from uuid import UUID
import struct
import numpy as np
import pytest

def project(value):
    from _exact_plan_comparison import exact_projection
    return exact_projection(value)

@dataclass
class Carrier:
    array:object
    budget:int=3
    metadata:object=field(default_factory=dict)
    cache:object=field(default=1,compare=False,metadata={'definition':False})

def fresh():return Carrier(np.array([[0.,1.],[2.,3.]],dtype=np.float64),metadata={'purpose':'budget'})

def test_old_generated_equality_is_undefined_for_independent_arrays():
    a,b=fresh(),fresh()
    with pytest.raises(ValueError,match='truth value'):assert a==b

def test_independent_same_bits_compare_equal_without_alias_requirement():
    a,b=fresh(),fresh();assert a.array is not b.array
    assert project(a)==project(b)

@pytest.mark.parametrize('fault',['coordinate','shape','dtype','signedzero','writeability','strides','metadata','budget','cache','class'])
def test_full_typed_projection_rejects_each_real_distinction(fault):
    a,b=fresh(),fresh()
    if fault=='coordinate':b.array[0,1]=np.nextafter(1.,2.)
    elif fault=='shape':b.array=b.array.reshape(4)
    elif fault=='dtype':b.array=b.array.astype(np.float32)
    elif fault=='signedzero':b.array[0,0]=-0.
    elif fault=='writeability':b.array.flags.writeable=False
    elif fault=='strides':
        a.array=np.zeros((2,2));b.array=np.zeros((2,2)).T
    elif fault=='metadata':b.metadata={'purpose':'other'}
    elif fault=='budget':b.budget=4
    elif fault=='cache':b.cache=2
    else:
        @dataclass
        class Other(Carrier):pass
        b=Other(b.array,b.budget,b.metadata,b.cache)
    assert project(a)!=project(b)


def test_scalar_type_and_float_bits_are_exact():
    assert project(True)!=project(1) and project(1)!=project(1.)
    assert project(0.)!=project(-0.)
    a=struct.unpack('>d',bytes.fromhex('7ff8000000000001'))[0]
    b=struct.unpack('>d',bytes.fromhex('7ff8000000000002'))[0]
    assert project(a)!=project(b) and project(a)==project(a)
    assert project((1,2))!=project([1,2])


def test_uuid_enum_fraction_preserve_full_type_and_value():
    class E(Enum):FIRST='one'
    class F(Enum):FIRST='one'
    assert project(E.FIRST)!=project(F.FIRST)
    assert project(E.FIRST)!=project('one')
    assert project(Fraction(1,2))!=project(.5)
    assert project(UUID(int=1))==project(UUID(int=1))
    assert project(UUID(int=1))!=project(UUID(int=2))


def test_array_dtype_metadata_and_scalar_dtype_are_preserved():
    a=np.array([1,2],dtype=np.dtype('i4',metadata={'unit':'x'}))
    b=np.array([1,2],dtype=np.dtype('i4',metadata={'unit':'y'}))
    assert project(a)!=project(b)
    assert project(np.float32(1.))!=project(np.float64(1.))


def test_all_declared_fields_including_implementation_caches_are_compared():
    a,b=fresh(),fresh();b.cache={'opaque_data':(2,)}
    assert project(a)!=project(b)

@pytest.mark.parametrize('value',[object(),lambda:None,np.array([object()],dtype=object)])
def test_opaque_values_fail_closed(value):
    with pytest.raises((TypeError,ValueError)):project(value)


def test_cycles_fail_closed_but_repeated_alias_values_do_not():
    a=[];a.append(a)
    with pytest.raises((TypeError,ValueError)):project(a)
    common=[1,2]
    assert project((common,common))==project(([1,2],[1,2]))


@pytest.mark.parametrize('family',['array','integer'])
def test_dataclass_cannot_hide_inherited_payload(family):
    if family=='array':
        @dataclass
        class Hidden(np.ndarray):marker:int=0
        a=np.array([1.]).view(Hidden);b=np.array([2.]).view(Hidden)
    else:
        @dataclass
        class Hidden(int):marker:int=0
        a=int.__new__(Hidden,1);b=int.__new__(Hidden,2)
    # Neither ignored C/base payload may be accepted as declared-field-only data.
    for value in (a,b):
        with pytest.raises(TypeError,match='unsupported inherited payload'):project(value)


def test_supported_string_enum_and_plain_dataclass_inheritance_remain_values():
    from enum import StrEnum
    class Choice(StrEnum):FIRST='first'
    assert project(Choice.FIRST)==project(Choice.FIRST)
    assert project(Choice.FIRST)!=project('first')
    @dataclass
    class Derived(Carrier):extra:int=7
    a,b=fresh(),fresh()
    one=Derived(a.array,a.budget,a.metadata,a.cache,7)
    two=Derived(b.array,b.budget,b.metadata,b.cache,7)
    assert project(one)==project(two)
    two.extra=8
    assert project(one)!=project(two)
