"""Exact typed test projection; deliberately not a public plan equality API."""
from dataclasses import Field,fields
from enum import Enum,EnumType
from fractions import Fraction
from types import MappingProxyType
from uuid import UUID
import struct
import numpy as np


def exact_projection(value):
    """Compare complete declared data, never aliasing, approximate values or opaque providers.

    Every dataclass field is retained, including compare=False/definition=False
    fields. Arrays retain dtype metadata, shape, strides, C-order element bytes
    and writeability. Cycles, object arrays and unknown values are refused.
    """
    active=set()
    names=('bool_','int8','int16','int32','int64','uint8','uint16','uint32','uint64',
           'intp','uintp','longlong','ulonglong','float16','float32','float64','longdouble',
           'complex64','complex128','clongdouble','str_','bytes_','datetime64','timedelta64','void')
    scalar_types=tuple(vars(np)[name] for name in names if name in vars(np))

    def dtype(data):
        if data.hasobject:raise TypeError('object dtype is not exact comparison data')
        members=None if data.fields is None else tuple((key,dtype(row[0]),row[1],walk(row[2]) if len(row)>2 else None)
                                                        for key,row in data.fields.items())
        sub=None if data.subdtype is None else (dtype(data.subdtype[0]),tuple(data.subdtype[1]))
        return ('dtype',data.str,data.byteorder,data.itemsize,data.alignment,data.isalignedstruct,
                data.names,members,sub,walk(data.metadata))

    def walk(item):
        kind=type(item)
        # Identity precedes the strong class reference: no foreign metaclass
        # equality, and the retained class prevents id reuse between projections.
        tag=(id(kind),kind)
        if item is None:return ('none',tag)
        if kind is bool or kind is int or kind is str or kind is bytes:
            return ('scalar',tag,item)
        if kind is float:return ('float-bits',tag,struct.pack('>d',item))
        if kind is UUID:return ('uuid',tag,item.int)
        if kind is Fraction:return ('fraction',tag,item.numerator,item.denominator)
        identity=id(item)
        if identity in active:raise ValueError('cyclic comparison data')
        active.add(identity)
        try:
            if kind is np.ndarray:
                return ('array',tag,dtype(item.dtype),tuple(item.shape),tuple(item.strides),
                        bool(item.flags.writeable),item.tobytes(order='C'))
            if any(kind is scalar for scalar in scalar_types):
                return ('numpy-scalar',tag,dtype(item.dtype),item.tobytes())
            if kind is tuple or kind is list:
                return ('sequence',tag,tuple(walk(part) for part in item))
            if kind is dict or kind is MappingProxyType:
                return ('mapping',tag,frozenset((walk(key),walk(part)) for key,part in item.items()))
            if type(kind) is EnumType and Enum in type.__getattribute__(kind,'__mro__'):
                return ('enum',tag,object.__getattribute__(item,'_name_'),walk(object.__getattribute__(item,'_value_')))
            if type(kind) is type:
                mro=type.__getattribute__(kind,'__mro__')
                payload_bases=(np.ndarray,np.generic,bool,int,float,complex,str,bytes,bytearray,
                               tuple,list,dict,set,frozenset,memoryview,range,slice,UUID,Fraction,BaseException)
                if any(base is payload for base in mro for payload in payload_bases):
                    raise TypeError('unsupported inherited payload')
                declared=None
                for base in mro:
                    namespace=type.__getattribute__(base,'__dict__')
                    if '__dataclass_fields__' in namespace:
                        declared=namespace['__dataclass_fields__'];break
                if type(declared) is dict and all(type(field) is Field for field in declared.values()):
                    return ('dataclass',tag,tuple((field.name,walk(field.metadata),
                        walk(object.__getattribute__(item,field.name))) for field in fields(kind)))
            raise TypeError('unsupported opaque comparison value')
        finally:active.remove(identity)
    return walk(value)
