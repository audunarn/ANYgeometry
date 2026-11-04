"""Exact initial legacy generated-wrapper refusal; never parallel qualification."""
import dataclasses
import hashlib
from pathlib import Path
import sys
import types

import pytest
from _planning_contract_helpers import (capture_with_dispatcher_boundary,
    dispatcher_boundary_refusal, require_qualified_planning_lane,classify_initial_runtime_refusal)
from anygeometry import _planning_manifest as manifest
from anygeometry._planning_qualification import QualificationDecline


def _codes(code):
    rows={code.co_qualname:code}
    for part in code.co_consts:
        if type(part) is types.CodeType:rows.update(_codes(part))
    return rows


def _same_code(one,two):
    return manifest.code_digest(one)==manifest.code_digest(two)


def legacy_wrapper_evidence(roots):
    """Validate only two real owner classes and their exact legacy producer."""
    if sys.version_info[:2] not in ((3,11),(3,12)):return None
    from anygeometry import tolerance, _canonical_curve_runtime
    candidates=((tolerance,tolerance.TolerancePolicy,'@pre-slots'),
                (_canonical_curve_runtime,_canonical_curve_runtime._Bank,''))
    found=None
    for owner,cls,suffix in candidates:
        if roots==(cls,) or roots==(tolerance.TolerancePolicy.effective_length,cls) and cls is tolerance.TolerancePolicy:
            found=(owner,cls,suffix);break
    if found is None:return None
    owner,cls,suffix=found
    assert vars(owner).get(cls.__name__) is cls
    source=Path(owner.__file__).resolve()
    assert source.is_relative_to(Path(manifest.__file__).resolve().parent)
    owned_codes=_codes(compile(source.read_bytes(),str(source),'exec',dont_inherit=True))
    wrapper=vars(cls)['__repr__'];factory=vars(dataclasses).get('_recursive_repr')
    assert type(factory) is types.FunctionType and factory.__globals__ is vars(dataclasses)
    provider=Path(dataclasses.__file__).resolve()
    provider_codes=_codes(compile(provider.read_bytes(),str(provider),'exec',dont_inherit=True))
    for name in ('_recursive_repr','_repr_fn','_create_fn'):
        actual=vars(dataclasses)[name]
        assert type(actual) is types.FunctionType and actual.__globals__ is vars(dataclasses)
        assert _same_code(actual.__code__,provider_codes[name]),'legacy producer differs from source'
    assert type(wrapper) is types.FunctionType and wrapper.__globals__ is vars(dataclasses)
    assert wrapper.__code__ in factory.__code__.co_consts
    assert _same_code(wrapper.__code__,provider_codes['_recursive_repr.<locals>.wrapper'])
    assert Path(wrapper.__code__.co_filename).resolve()==provider
    assert wrapper.__module__==owner.__name__ and wrapper.__qualname__==cls.__name__+'.__repr__'
    cells=dict(zip(wrapper.__code__.co_freevars,(c.cell_contents for c in wrapper.__closure__)))
    assert set(cells)=={'repr_running','user_function'}
    assert type(cells['repr_running']) is set and not cells['repr_running']
    body=vars(wrapper).get('__wrapped__')
    assert type(body) is types.FunctionType and cells['user_function'] is body
    assert body.__globals__ is vars(owner)
    expected=dataclasses._repr_fn([f for f in dataclasses.fields(cls) if f.repr],vars(owner)).__wrapped__
    assert _same_code(body.__code__,expected.__code__),'legacy wrapped body differs from producer'
    # Real source-authored methods cannot hide an unrelated replacement behind
    # the first unsupported generated repr. No owner/numerical bodies are invoked
    # and no code is installed; the stdlib repr factory only builds test expectations.
    for name,value in vars(cls).items():
        if type(value) is not types.FunctionType or value is wrapper:continue
        key=cls.__name__+'.'+name
        if key in owned_codes:
            assert value.__globals__ is vars(owner),'owned method globals changed'
            assert _same_code(value.__code__,owned_codes[key]),'owned method differs from source'
        elif value.__code__.co_filename!='<string>':
            assert value is vars(dataclasses).get('_dataclass_getstate') or value is vars(dataclasses).get('_dataclass_setstate')
            assert _same_code(value.__code__,provider_codes[value.__qualname__])
    if suffix:
        setter=vars(cls)['__setattr__']
        original=dict(zip(setter.__code__.co_freevars,(c.cell_contents for c in setter.__closure__)))['cls']
        assert original is not cls and vars(original)['__repr__'] is wrapper
        assert vars(original)['__dataclass_fields__'] is vars(cls)['__dataclass_fields__']
        assert vars(original)['__dataclass_params__'] is vars(cls)['__dataclass_params__']
    path=owner.__name__+'.'+cls.__name__+suffix+'.__repr__'
    reason="callable closure refused: ['"+path+": owned function namespace/origin mismatch']"
    return dict(reason=reason,owner=owner.__name__,provider=str(provider),
                provider_sha256=hashlib.sha256(provider.read_bytes()).hexdigest(),
                owner_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())


def capture_with_legacy_dataclass_boundary(roots,*,request=None):
    if dispatcher_boundary_refusal() is not None:
        return capture_with_dispatcher_boundary(roots,request=request)
    evidence=legacy_wrapper_evidence(roots)
    if evidence is None:return capture_with_dispatcher_boundary(roots,request=request)
    reason=evidence['reason'];require_qualified_planning_lane(reason)
    with pytest.raises(QualificationDecline) as caught:manifest.capture_callable_manifest(roots)
    if classify_initial_runtime_refusal(caught.value,request=request):return None
    assert type(caught.value) is QualificationDecline and str(caught.value)==reason, (
        'exact legacy refusal differs',reason,type(caught.value).__name__,str(caught.value),evidence)
    if request is not None:
        request.node.user_properties.extend((('execution_scope','initial-legacy-dataclass-wrapper-refusal'),
            ('runtime_refusal',reason),('original_test_body_executed',False),
            ('postcapture_qualification',False),('legacy_provider',evidence['provider']),
            ('legacy_provider_sha256',evidence['provider_sha256'])))
    return None
