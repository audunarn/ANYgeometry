"""Definition/load-point anchors for named NumPy operations.

No optional provider is imported. Hidden C dispatcher links retain the same
explicit image-provenance limitation as backend._RUNTIME_POINT_IMPLEMENTATIONS.
Exposed Python implementations and bindings are anchored, not re-baselined by
the late validator. This module does not itself admit a numerical workload.
"""
if 'ANCHOR' in globals():
    raise RuntimeError('runtime anchor reload is forbidden; restart the owning application')

import sys
import types
import numpy as np

NUMPY_NAMES=tuple('abs all allclose angle any append arange arccos arcsin arctan2 argmax argmin argsort array array_equal asarray ascontiguousarray atleast_1d atleast_2d bincount broadcast_to ceil clip column_stack concatenate copysign cos count_nonzero cross cumsum diag diff divide dot einsum empty exp eye finfo flatnonzero floor frexp frombuffer fromiter full full_like geterr geterrcall hstack hypot iscomplexobj isfinite ldexp linspace max maximum mean meshgrid min minimum mod ndim nextafter nonzero ones ones_like outer ptp ravel repeat require roll roots round searchsorted shape shares_memory sign sin sinc size split sqrt stack subtract sum tan tile triu_indices unique unravel_index unwrap vstack where zeros zeros_like'.split())
LINALG_NAMES=tuple('det eigh inv lstsq matrix_rank norm pinv solve svd'.split())
NUMPY_NAMES+=('getbufsize',)


def _metadata(value,remaining=512):
    if remaining<=0:return ('unqualified-bound',)
    if value is None or type(value) in (str,bytes,int,bool):return ('scalar',type(value),value)
    if type(value) is float:return ('float',value.hex())
    if type(value) in (tuple,list):
        if len(value)>remaining:return ('unqualified-bound',)
        return (type(value),tuple(_metadata(item,remaining//(len(value)+1)) for item in value))
    if type(value) is dict:
        if len(value)>remaining or any(type(key) is not str for key in value):return ('unqualified-mapping',)
        return ('dict',tuple((key,_metadata(item,remaining//(len(value)+1))) for key,item in sorted(value.items())))
    # This is a binding record only. Closed mutable metadata qualification is
    # separately required by the loaded manifest; it is not an opaque exemption.
    return ('binding-only',id(type(value)),id(value))


def _functions(value):
    from numpy._core._multiarray_umath import _ArrayFunctionDispatcher
    pending=[value];seen=set();result=[];edges=[]
    while pending:
        function=pending.pop()
        if id(function) in seen:continue
        seen.add(id(function))
        if len(seen)>256:return (),(),False
        if type(function) not in (types.FunctionType,types.BuiltinFunctionType,np.ufunc,_ArrayFunctionDispatcher,type):
            return (),(),False
        if type(function) is type:
            namespace=type.__getattribute__(function,'__dict__')
            if not namespace.get('__module__','').startswith('numpy'):
                return (),(),False
            for name in ('__new__','__init__'):
                child=namespace.get(name)
                if type(child) is staticmethod:child=child.__func__
                if type(child) is types.FunctionType:pending.append(child)
            continue
        if type(function) is types.FunctionType:
            namespace=function.__globals__
            owner=sys.modules.get(namespace.get('__name__'))
            if owner is None or vars(owner) is not namespace:return (),(),False
            global_bindings=tuple((key,namespace[key],namespace[key].__code__ if type(namespace[key]) is types.FunctionType else None)
                for key in sorted(set(function.__code__.co_names)) if key in namespace)
            try:cells=tuple(cell.cell_contents for cell in function.__closure__ or ())
            except ValueError:return (),(),False
            result.append((function,function.__code__,function.__defaults__,_metadata(function.__defaults__),
                function.__kwdefaults__,_metadata(function.__kwdefaults__),cells,_metadata(cells),namespace,global_bindings))
        for name in ('_implementation','__wrapped__'):
            child=getattr(function,name,None)
            edges.append((function,name,child))
            if child is not None:pending.append(child)
    return tuple(result),tuple(edges),True


def _capture():
    rows=[]
    # Only already-loaded modules required by the owner are eligible here.
    for provider,names in (('numpy',NUMPY_NAMES),('numpy.linalg',LINALG_NAMES)):
        module=sys.modules.get(provider)
        for name in names:
            value=None if module is None else vars(module).get(name)
            functions,edges,complete=((),(),False) if value is None else _functions(value)
            # A custom Python function substituted before owner import is not a
            # NumPy public adapter merely because it is now stored under its name.
            if type(value) is types.FunctionType and not value.__globals__.get('__name__','').startswith('numpy.'):
                complete=False
            rows.append((provider,name,module,value,functions,edges,complete))
    return tuple(rows)


ANCHOR=_capture()


def _capture_json():
    native=sys.modules.get('_json');encoder=sys.modules.get('json.encoder')
    if type(native) is not types.ModuleType or type(encoder) is not types.ModuleType:
        return native,encoder,(),False
    rows=[]
    for source,target in (('make_encoder','c_make_encoder'),('encode_basestring','encode_basestring'),
                          ('encode_basestring_ascii','encode_basestring_ascii')):
        value=vars(native).get(source)
        metadata=None
        if source=='make_encoder':
            valid=(type(value) is type and type.__getattribute__(value,'__module__')=='_json' and
                   type.__getattribute__(value,'__name__')=='Encoder' and
                   type.__getattribute__(value,'__bases__')==(object,))
            if valid:
                namespace=type.__getattribute__(value,'__dict__')
                fields=('markers','default','encoder','indent','key_separator','item_separator','sort_keys','skipkeys')
                valid=(set(namespace)=={'__new__','__call__','__doc__','__module__',*fields} and
                       type(namespace['__new__']) is types.BuiltinFunctionType and
                       type(namespace['__call__']) is types.WrapperDescriptorType and
                       namespace['__call__'].__objclass__ is value and
                       all(type(namespace[name]) is types.MemberDescriptorType and namespace[name].__objclass__ is value for name in fields))
                metadata=(value.__name__,value.__qualname__,value.__bases__,value.__mro__)
        else:
            valid=(type(value) is types.BuiltinFunctionType and value.__module__=='_json' and value.__name__==source)
        if not valid or vars(encoder).get(target) is not value:return native,encoder,(),False
        rows.append((source,target,value,tuple(vars(value).items()) if type(value) is type else None,metadata))
    return native,encoder,tuple(rows),True


JSON_ANCHOR=_capture_json()


def verify_owner(owner,_expected=ANCHOR,_namespace=globals(),_module=sys.modules[__name__],
                 _names=(NUMPY_NAMES,LINALG_NAMES),_json=JSON_ANCHOR):
    from ._planning_qualification import QualificationDecline
    handle=vars(owner).get('_RUNTIME_OPERATION_OWNER')
    if (type(handle) is not tuple or len(handle)!=4 or handle[0] is not _module or
            _namespace.get('OWNER') is not handle or
            sys.modules.get(__name__) is not _module or
            vars(sys.modules['anygeometry']).get('_planning_runtime_anchor') is not _module or
            _namespace.get('ANCHOR') is not _expected or
            _namespace.get('NUMPY_NAMES') is not _names[0] or
            _namespace.get('LINALG_NAMES') is not _names[1] or _namespace.get('JSON_ANCHOR') is not _json):
        raise QualificationDecline('owner runtime anchor missing/rebound; restart required')
    native,encoder,json_rows,complete=_json
    if (not complete or sys.modules.get('_json') is not native or sys.modules.get('json.encoder') is not encoder):
        raise QualificationDecline('JSON owner-load anchor absent or rebound')
    for source,target,value,namespace,metadata in json_rows:
        if vars(native).get(source) is not value or vars(encoder).get(target) is not value:
            raise QualificationDecline('JSON native owner binding changed')
        if namespace is not None and (len(vars(value))!=len(namespace) or any(vars(value).get(key) is not item for key,item in namespace)):
            raise QualificationDecline('JSON native descriptor binding changed')
        if metadata is not None and (value.__name__!=metadata[0] or value.__qualname__!=metadata[1] or
                                      value.__bases__ is not metadata[2] or value.__mro__ is not metadata[3]):
            raise QualificationDecline('JSON native class metadata changed')
    for name,function,code,defaults,kwdefaults in handle[3]:
        if (_namespace.get(name) is not function or function.__code__ is not code or
                function.__defaults__ is not defaults or function.__kwdefaults__ is not kwdefaults):
            raise QualificationDecline('runtime anchor verifier/registry definition changed')
    for provider,name,module,value,functions,edges,complete in _expected:
        if not complete:raise QualificationDecline('named runtime anchor unavailable: '+provider+'.'+name)
        if sys.modules.get(provider) is not module or vars(module).get(name) is not value:
            raise QualificationDecline('anchored runtime operation rebound: '+provider+'.'+name)
        for parent,attribute,child in edges:
            if getattr(parent,attribute,None) is not child:
                raise QualificationDecline('anchored runtime implementation edge rebound')
        for function,code,defaults,default_value,kwdefaults,kw_value,cells,cell_value,namespace,bindings in functions:
            if (function.__code__ is not code or function.__defaults__ is not defaults or
                    _metadata(function.__defaults__)!=default_value or function.__kwdefaults__ is not kwdefaults or
                    _metadata(function.__kwdefaults__)!=kw_value or function.__globals__ is not namespace or
                    _metadata(tuple(cell.cell_contents for cell in function.__closure__ or ()))!=cell_value or
                    any(namespace.get(key) is not bound or (bound.__code__ if type(bound) is types.FunctionType else None) is not bound_code
                        for key,bound,bound_code in bindings)):
                raise QualificationDecline('anchored runtime Python implementation changed')
    return _expected


# Immutable owner-load handle. The caller uses the stored verifier, first
# checking its original code; replacing the public verifier cannot self-certify.
OWNER=(sys.modules[__name__],verify_owner,verify_owner.__code__,
       tuple((name,value,value.__code__,value.__defaults__,value.__kwdefaults__)
             for name,value in tuple(globals().items()) if type(value) is types.FunctionType))
