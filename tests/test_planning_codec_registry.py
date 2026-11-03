"""Exact-runtime passive codec observations and independent packaging contracts."""
import os
from pathlib import Path
import runpy
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]


def configuration(monkeypatch,**flags):
    import setuptools
    result={}
    for suffix in ('BUILD_NATIVE','REQUIRE_NATIVE','DISABLE_NATIVE','BUILD_PLANNING_NATIVE','REQUIRE_PLANNING_NATIVE','DISABLE_PLANNING_NATIVE','BUILD_RUNTIME_INSPECTOR','REQUIRE_RUNTIME_INSPECTOR','DISABLE_RUNTIME_INSPECTOR'):
        monkeypatch.delenv('ANYGEOMETRY_'+suffix,raising=False)
    for suffix,value in flags.items():monkeypatch.setenv('ANYGEOMETRY_'+suffix,value)
    monkeypatch.setattr(setuptools,'setup',lambda **kwargs:result.update(kwargs))
    runpy.run_path(str(ROOT/'setup.py'))
    return result


def test_inspector_only_is_nonlimited_and_keeps_numerical_core_off(monkeypatch):
    extensions=configuration(monkeypatch,BUILD_RUNTIME_INSPECTOR='1')['ext_modules']
    assert [item.name for item in extensions]==['anygeometry._planning_runtime_inspector']
    assert not extensions[0].py_limited_api
    assert not any(name=='Py_LIMITED_API' for name,value in extensions[0].define_macros)


def test_combined_helper_build_preserves_old_abi_and_uses_nonlimited_wheel(monkeypatch):
    extensions=configuration(monkeypatch,BUILD_RUNTIME_INSPECTOR='1',BUILD_PLANNING_NATIVE='1')['ext_modules']
    by_name={item.name:item for item in extensions}
    assert set(by_name)=={'anygeometry._planning_native_core','anygeometry._planning_runtime_inspector'}
    assert by_name['anygeometry._planning_native_core'].py_limited_api
    assert ('Py_LIMITED_API','0x030B0000') in by_name['anygeometry._planning_native_core'].define_macros
    assert not by_name['anygeometry._planning_runtime_inspector'].py_limited_api


def test_default_off_keeps_extensions_absent(monkeypatch):
    assert configuration(monkeypatch)['ext_modules']==[]


def test_passive_inspector_api_returns_immutable_container_copies():
    from anygeometry._planning_codec_registry import snapshot
    value=snapshot()
    assert type(value.search_path) is tuple
    assert type(value.codecs) is tuple and type(value.error_handlers) is tuple
    assert tuple(row[0] for row in value.codecs)==('utf_8','cp1252')
    assert tuple(row[0] for row in value.error_handlers)==('strict','backslashreplace')
    assert value.profile['runtime_sha256']=='7ffe3ab11342db03cc18b026f27485c0f74f5bf10f239f1f6573394900596394'


def test_actual_mixed_wheel_tag_is_not_abi3(monkeypatch):
    from setuptools import Distribution
    config=configuration(monkeypatch,BUILD_RUNTIME_INSPECTOR='1',BUILD_PLANNING_NATIVE='1')
    wheel=config['cmdclass']['bdist_wheel'](Distribution(config))
    wheel.ensure_finalized()
    assert wheel.get_tag()==('cp314','cp314','win_amd64')


def test_observer_does_not_search_or_fill_missing_selected_cache():
    import codecs
    from anygeometry._planning_codec_registry import snapshot
    calls=[]
    def search(name):
        calls.append(name)
        return None
    codecs.register(search)
    try:
        # unregister clears CPython's cache; this is a disposable test mutation.
        codecs.unregister(search)
        codecs.register(search)
        first=snapshot()
        assert all(not present for _,present,_ in first.codecs)
        for _ in range(4):
            current=snapshot()
            assert current.codecs==first.codecs
            assert current.matches() and first.matches()
        assert calls==[]
    finally:
        codecs.unregister(search)


def test_cached_entries_and_search_mutation_are_reported_by_identity():
    import codecs
    from anygeometry._planning_codec_registry import snapshot
    utf8=codecs.lookup('utf-8')
    cp1252=codecs.lookup('cp1252')
    before=snapshot()
    assert before.codecs[0][1] and before.codecs[0][2] is utf8
    assert before.codecs[1][1] and before.codecs[1][2] is cp1252
    def search(name):
        raise AssertionError('observer called search')
    codecs.register(search)
    try:
        after=snapshot()
        assert not before.matches()
        assert after.search_path[-1] is search and after.matches()
    finally:
        codecs.unregister(search)


def test_cache_warming_is_drift_without_policy_exception():
    import codecs
    from anygeometry._planning_codec_registry import snapshot
    def search(name): return None
    codecs.register(search)
    codecs.unregister(search)
    before=snapshot()
    assert not before.codecs[1][1]
    codec=codecs.lookup('cp1252')
    assert not before.matches()
    assert snapshot().codecs[1][2] is codec


def test_error_handler_identity_change_and_strong_lifetime():
    import codecs
    import weakref
    import gc
    from anygeometry._planning_codec_registry import snapshot
    old=codecs.lookup_error('strict')
    before=snapshot()
    class Handler:
        def __call__(self,error): raise AssertionError('observer called handler')
    handler=Handler()
    ref=weakref.ref(handler)
    try:
        codecs.register_error('strict',handler)
        after=snapshot()
        assert not before.matches() and after.error_handlers[0][2] is handler
        codecs.register_error('strict',old)
        del handler
        gc.collect()
        assert ref() is not None and not after.matches()
        del after
        gc.collect()
        assert ref() is None
    finally:
        codecs.register_error('strict',old)


@pytest.mark.parametrize('name',['snapshot','matches','identity'])
def test_altered_exports_refused_before_call(monkeypatch,name):
    from anygeometry import _planning_codec_registry as adapter
    module=sys.modules[adapter._NAME]
    calls=[]
    monkeypatch.setattr(module,name,lambda *args:calls.append(args))
    with pytest.raises(adapter.InspectionUnavailable): adapter.snapshot()
    assert not calls


def test_missing_helper_does_not_rebaseline(monkeypatch):
    from anygeometry import _planning_codec_registry as adapter
    monkeypatch.setattr(adapter,'_ANCHOR',None)
    monkeypatch.setattr(adapter,'_FAILURE','missing at initial load')
    with pytest.raises(adapter.InspectionUnavailable,match='missing'): adapter.snapshot()


def test_replaced_module_and_loader_decline(monkeypatch):
    from anygeometry import _planning_codec_registry as adapter
    module=sys.modules[adapter._NAME]
    monkeypatch.setitem(sys.modules,adapter._NAME,type(sys)(adapter._NAME))
    with pytest.raises(adapter.InspectionUnavailable): adapter.identity()
    monkeypatch.setitem(sys.modules,adapter._NAME,module)
    monkeypatch.setattr(module,'__loader__',object())
    with pytest.raises(adapter.InspectionUnavailable): adapter.identity()


def test_profile_layout_and_compiler_identities_are_distinct():
    from anygeometry._planning_codec_registry import identity
    value=identity()
    assert value['runtime_msc_ver']==1944 and value['helper_msc_ver']==1950
    assert value['codecs_size']==32
    assert 0 < value['interpreter_codecs_offset'] < value['interpreter_size']
    assert len(value['binary_sha256'])==64
    assert all(row['image_offset']>0 for row in value['exports'])


def test_native_rejects_foreign_capsule_and_guard_has_no_override():
    from anygeometry import _planning_runtime_inspector as native
    with pytest.raises(ValueError): native.matches(object())
    with pytest.raises(TypeError): native.snapshot(False)
    with pytest.raises(TypeError): native.identity(False)


def test_initial_capture_refuses_altered_export_without_invoking(monkeypatch):
    from anygeometry import _planning_codec_registry as adapter
    calls=[]
    monkeypatch.setattr(sys.modules[adapter._NAME],'identity',lambda:calls.append(True))
    fresh=runpy.run_path(adapter.__file__)
    assert fresh['_ANCHOR'] is None
    with pytest.raises(fresh['InspectionUnavailable']): fresh['snapshot']()
    assert calls==[]


def test_hostile_unselected_key_never_dispatches_equality_and_present_none_is_distinct():
    import codecs,gc
    from anygeometry._planning_codec_registry import snapshot
    old=codecs.lookup_error('strict')
    calls=[]
    def handler(error): raise AssertionError('unexpected handler')
    codecs.register_error('strict',handler)
    # Test-only access in this disposable interpreter; production never exports
    # these containers. A Python callable makes the private dict GC-tracked.
    registry=next(item for item in gc.get_referrers(handler)
                  if type(item) is dict and item.get('strict') is handler
                  and item.get('backslashreplace') is codecs.lookup_error('backslashreplace'))
    class Key:
        def __hash__(self): return hash('strict')
        def __eq__(self,other): calls.append(other); return False
    key=Key()
    try:
        registry[key]=handler
        calls.clear()
        with pytest.raises(RuntimeError,match='non-exact'): snapshot()
        assert not calls
        del registry[key]
        registry['strict']=None
        value=snapshot()
        assert value.error_handlers[0]==('strict',True,None)
        del registry['strict']
        assert snapshot().error_handlers[0]==('strict',False,None)
        assert not value.matches()
    finally:
        registry.pop(key,None)
        codecs.register_error('strict',old)


@pytest.mark.parametrize('fault',['version','headers','dll','flags'])
def test_build_profile_rejects_mismatched_inputs_before_compilation(monkeypatch,fault):
    config=configuration(monkeypatch,BUILD_RUNTIME_INSPECTOR='1')
    validate=config['cmdclass']['build_ext'].build_extension.__globals__['_validate_runtime_inspector_build']
    if fault=='version': monkeypatch.setattr(sys,'version','unsupported')
    elif fault=='flags': monkeypatch.setenv('CL','/DPy_GIL_DISABLED=1')
    else:
        original=Path.read_bytes
        def altered(path):
            if (fault=='headers' and path.name=='Python.h') or (fault=='dll' and path.suffix=='.dll'):
                return b'unsupported'
            return original(path)
        monkeypatch.setattr(Path,'read_bytes',altered)
    with pytest.raises(RuntimeError): validate()


def test_matches_rechecks_after_allocating_wrapper_validation(monkeypatch):
    import codecs
    from anygeometry import _planning_codec_registry as adapter
    before=adapter.snapshot()
    old=codecs.lookup_error('strict')
    original=adapter._validated
    calls=[]
    def validation():
        result=original()
        calls.append(True)
        if len(calls)==2: codecs.register_error('strict',lambda error:None)
        return result
    monkeypatch.setattr(adapter,'_validated',validation)
    try:
        assert before.matches() is False
    finally:
        codecs.register_error('strict',old)
