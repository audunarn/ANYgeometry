"""Named runtime adapters; no provider-prefix or mutable-store exemption."""
import abc
import gc
import hashlib
import importlib
import json
import marshal
import numbers
import decimal
import types
from fractions import Fraction

import numpy as np
from ._planning_qualification import QualificationDecline,pin_module

# Explicit operations used by the admitted geometry/constructor path. This is
# deliberately a finite name list, not every member of a provider namespace.
NUMPY_OPERATIONS=tuple('abs all allclose angle any append arange arccos arcsin arctan2 argmax argmin argsort array array_equal asarray ascontiguousarray atleast_1d atleast_2d bincount broadcast_to ceil clip column_stack concatenate copysign cos count_nonzero cross cumsum diag diff divide dot einsum empty exp eye finfo flatnonzero floor frexp frombuffer fromiter full full_like geterr geterrcall hstack hypot iscomplexobj isfinite ldexp linspace max maximum mean meshgrid min minimum mod ndim nextafter nonzero ones ones_like outer ptp ravel repeat require roll roots round searchsorted shape shares_memory sign sin sinc size split sqrt stack subtract sum tan tile triu_indices unique unravel_index unwrap vstack where zeros zeros_like'.split())
NUMPY_LINALG=tuple('det eigh inv lstsq matrix_rank norm pinv solve svd'.split())
NUMPY_OPERATIONS+=('getbufsize',)
_ABC_TYPES=(numbers.Number,numbers.Complex,numbers.Real,numbers.Rational,numbers.Integral,Fraction)
_REGISTRATIONS=({np.number,decimal.Decimal},{complex,np.inexact},{float,np.floating},set(),{int,np.integer},set())
_CONCRETE=(bool,int,float,complex,Fraction,np.float64,np.int64,np.bool_,np.ndarray)
# Columns: Number, Complex, Real, Rational, Integral. These expectations come
# from numbers.py/Fraction inheritance and numpy.numerictypes._register_types,
# not observed H answers. ndarray/bool_ never opt into numbers.Number.
_DECISIONS=((1,1,1,1,1,0),(1,1,1,1,1,0),(1,1,1,0,0,0),(1,1,0,0,0,0),(1,1,1,1,0,1),
            (1,1,1,0,0,0),(1,1,1,1,1,0),(0,0,0,0,0,0),(0,0,0,0,0,0))


def verify_numeric_abcs():
    """Finite predicate check without clearing or rewriting any H ABC cache."""
    for abstract,expected in zip(_ABC_TYPES,_REGISTRATIONS):
        registry,_,_,_=abc._get_dump(abstract)
        actual={reference() for reference in registry}
        if actual!=expected or '__subclasshook__' in vars(abstract):
            raise QualificationDecline('numeric ABC registrations/hook outside contract')
    instances=(False,0,0.,0j,Fraction(0),np.float64(0),np.int64(0),np.bool_(False),np.empty(0))
    for concrete,instance,row in zip(_CONCRETE,instances,_DECISIONS):
        for abstract,expected in zip(_ABC_TYPES,row):
            if issubclass(concrete,abstract) is not bool(expected) or isinstance(instance,abstract) is not bool(expected):
                raise QualificationDecline('numeric ABC selected predicate differs from audited table')


def numeric_abcs():return _ABC_TYPES


def operation(provider,name):
    allowed=NUMPY_OPERATIONS if provider=='numpy' else NUMPY_LINALG if provider=='numpy.linalg' else ()
    if name not in allowed:raise QualificationDecline('unregistered runtime operation: '+provider+'.'+name)
    module=importlib.import_module(provider)
    if name not in vars(module):raise QualificationDecline('runtime operation missing')
    from ._planning_ledger import runtime_infrastructure
    handle,_=runtime_infrastructure()
    anchor=vars(handle[0])['ANCHOR']
    if not any(row[0]==provider and row[1]==name and row[3] is vars(module)[name] for row in anchor):
        raise QualificationDecline('runtime operation has no owner-load anchor')
    return module,vars(module)[name]


def dispatcher_edges(value):
    from numpy._core._multiarray_umath import _ArrayFunctionDispatcher
    if type(value) is not _ArrayFunctionDispatcher:
        raise QualificationDecline('unknown array dispatch adapter')
    from ._planning_ledger import runtime_infrastructure
    handle,_=runtime_infrastructure()
    anchor=vars(handle[0])['ANCHOR']
    if not any(row[3] is value for row in anchor):raise QualificationDecline('unanchored array dispatcher')
    implementation=value._implementation
    if implementation is not value.__wrapped__:
        raise QualificationDecline('array exposed implementation edges differ')
    # The hidden dispatcher pointer is not exposed by this NumPy build. Its
    # binding to this exact owner-load object is explicitly native-image risk,
    # never a GC-derived pointer equality claim. Keep this limit in the receipt.
    return (implementation,)


def _fingerprint(function,refs,identities):
    refs.append(function);identities.append(id(function))
    if type(function) is types.FunctionType:
        refs.extend((function.__code__,function.__defaults__,function.__kwdefaults__))
        identities.extend((id(function.__code__),id(function.__defaults__),id(function.__kwdefaults__)))
        # Unit-level adapter binding. The loaded manifest traverses these actual
        # functions and their globals/defaults/closures before admission.
        from ._planning_code import code_digest,passive_data
        return [function.__module__,function.__qualname__,code_digest(function.__code__),
                passive_data(function.__defaults__).hex(),passive_data(function.__kwdefaults__).hex()]
    if type(function) is types.BuiltinFunctionType:
        return [function.__module__,function.__qualname__]
    raise QualificationDecline('unknown runtime implementation')


class OperationBinding:
    def __init__(self,provider,name):
        self.provider,self.name=provider,name
        self.references=[]
        self.record,self.identities=self.capture(self.references)
    def capture(self,refs):
        module,value=operation(self.provider,self.name);refs.extend((module,value))
        ids=[id(module),id(value)]
        functions=dispatcher_edges(value) if type(value).__name__=='_ArrayFunctionDispatcher' else (value,)
        return dict(provider=pin_module(module),functions=[_fingerprint(f,refs,ids) for f in functions]),ids
    def verify(self):
        from ._planning_executor import PlannerSourceChanged
        try:
            current,ids=self.capture([])
            if current!=self.record or ids!=self.identities:raise QualificationDecline('runtime adapter rebound')
        except Exception as error:raise PlannerSourceChanged('runtime operation changed: '+str(error)) from error


def capture_operation(provider,name):return OperationBinding(provider,name)
