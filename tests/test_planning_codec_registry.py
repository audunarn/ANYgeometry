"""Optional diagnostic packaging remains separate from the production value codec."""
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
    extensions = configuration(monkeypatch, BUILD_RUNTIME_INSPECTOR='1')['ext_modules']
    if sys.platform != 'win32':
        assert extensions == []
        with pytest.raises(RuntimeError, match='exact Windows profile'):
            configuration(monkeypatch, REQUIRE_RUNTIME_INSPECTOR='1')
        return
    assert [item.name for item in extensions] == ['anygeometry._planning_runtime_inspector']
    assert not extensions[0].py_limited_api
    assert not any(name == 'Py_LIMITED_API' for name, value in extensions[0].define_macros)


def test_combined_helper_build_preserves_old_abi_and_uses_nonlimited_wheel(monkeypatch):
    extensions = configuration(monkeypatch, BUILD_RUNTIME_INSPECTOR='1', BUILD_PLANNING_NATIVE='1')['ext_modules']
    if sys.platform != 'win32':
        assert extensions == []
        with pytest.raises(RuntimeError, match='requires Windows'):
            configuration(monkeypatch, REQUIRE_PLANNING_NATIVE='1')
        return
    by_name = {item.name: item for item in extensions}
    assert set(by_name) == {'anygeometry._planning_native_core', 'anygeometry._planning_runtime_inspector'}
    assert by_name['anygeometry._planning_native_core'].py_limited_api
    assert ('Py_LIMITED_API', '0x030B0000') in by_name['anygeometry._planning_native_core'].define_macros
    assert not by_name['anygeometry._planning_runtime_inspector'].py_limited_api


def test_default_off_keeps_extensions_absent(monkeypatch):
    assert configuration(monkeypatch)['ext_modules'] == []


def test_actual_mixed_wheel_tag_is_not_abi3(monkeypatch):
    from setuptools import Distribution
    config = configuration(monkeypatch, BUILD_RUNTIME_INSPECTOR='1', BUILD_PLANNING_NATIVE='1')
    wheel = config['cmdclass']['bdist_wheel'](Distribution(config))
    wheel.ensure_finalized()
    python, abi, platform = wheel.get_tag()
    if sys.platform == 'win32':
        assert python == 'cp' + str(sys.version_info.major) + str(sys.version_info.minor)
        assert abi != 'abi3' and platform == 'win_amd64'
    else:
        assert (python, abi, platform) == ('py3', 'none', 'any')


@pytest.mark.parametrize('name', ['snapshot', 'matches', 'identity'])
def test_altered_diagnostic_exports_refuse_before_call(monkeypatch, name):
    from anygeometry import _planning_codec_registry as adapter
    module = sys.modules.get(adapter._NAME)
    if module is None:
        with pytest.raises(adapter.InspectionUnavailable):
            adapter.snapshot()
        return  # No native export executed or mutation coverage claimed here.
    calls = []
    monkeypatch.setattr(module, name, lambda *args: calls.append(args))
    with pytest.raises(adapter.InspectionUnavailable):
        adapter.snapshot()
    assert calls == []


def test_missing_helper_does_not_rebaseline(monkeypatch):
    from anygeometry import _planning_codec_registry as adapter
    monkeypatch.setattr(adapter, '_ANCHOR', None)
    monkeypatch.setattr(adapter, '_FAILURE', 'missing at initial load')
    with pytest.raises(adapter.InspectionUnavailable, match='missing'):
        adapter.snapshot()


def test_native_rejects_foreign_capsule_and_has_no_override():
    import importlib
    from anygeometry import _planning_codec_registry as adapter
    try:
        native = importlib.import_module(adapter._NAME)
    except ModuleNotFoundError as error:
        assert error.name == adapter._NAME
        with pytest.raises(adapter.InspectionUnavailable):
            adapter.snapshot()
    else:
        with pytest.raises(ValueError):
            native.matches(object())
        with pytest.raises(TypeError):
            native.snapshot(False)
        with pytest.raises(TypeError):
            native.identity(False)


@pytest.mark.parametrize('fault', ['version', 'headers', 'dll', 'flags'])
def test_build_profile_rejects_mismatched_inputs_before_compilation(monkeypatch, fault):
    import json
    config = configuration(monkeypatch, BUILD_RUNTIME_INSPECTOR='1')
    validate = config['cmdclass']['build_ext'].build_extension.__globals__['_validate_runtime_inspector_build']
    profile = json.loads((ROOT / 'native/runtime_inspector_profile.json').read_text())
    if sys.platform != 'win32' or sys.version != profile['runtime_version']:
        with pytest.raises(RuntimeError, match='runtime is outside the exact profile'):
            validate()
        return  # The unsupported runtime is refused before any build; no compile claim.
    if fault == 'version':
        monkeypatch.setattr(sys, 'version', 'unsupported')
    elif fault == 'flags':
        monkeypatch.setenv('CL', '/DPy_GIL_DISABLED=1')
    else:
        original = Path.read_bytes
        def altered(path):
            if (fault == 'headers' and path.name == 'Python.h') or (fault == 'dll' and path.suffix == '.dll'):
                return b'unsupported'
            return original(path)
        monkeypatch.setattr(Path, 'read_bytes', altered)
    with pytest.raises(RuntimeError):
        validate()


def test_current_value_codec_contract_does_not_require_private_inspector(monkeypatch):
    import builtins
    from anygeometry import _planning_values as values
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if '_planning_codec_registry' in name or '_planning_runtime_inspector' in name:
            pytest.fail('production value codec imported private interpreter inspector')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    contract = values.value_codec_contract()
    assert contract['version'] == 'v2' and len(contract['schema_sha256']) == 64
    values.verify_value_codec_contract(contract)
    value = ('wire', 1, 2.5)
    assert values.decode_value(values.encode_value(value, codec='v2'), codec='v2') == value


@pytest.mark.parametrize('change', ['version', 'schema_sha256', 'extra'])
def test_current_value_codec_request_binding_cannot_be_forged(change):
    from anygeometry import _planning_values as values
    from anygeometry._planning_qualification import QualificationDecline
    contract = values.value_codec_contract()
    contract[change] = 'foreign'
    with pytest.raises(QualificationDecline, match='request binding'):
        values.verify_value_codec_contract(contract)


def test_replaced_diagnostic_module_and_loader_decline(monkeypatch):
    from anygeometry import _planning_codec_registry as adapter
    module = sys.modules.get(adapter._NAME)
    if module is None:
        with pytest.raises(adapter.InspectionUnavailable):
            adapter.identity()
        return
    monkeypatch.setitem(sys.modules, adapter._NAME, type(sys)(adapter._NAME))
    with pytest.raises(adapter.InspectionUnavailable):
        adapter.identity()
    monkeypatch.setitem(sys.modules, adapter._NAME, module)
    monkeypatch.setattr(module, '__loader__', object())
    with pytest.raises(adapter.InspectionUnavailable):
        adapter.identity()


def test_initial_diagnostic_capture_refuses_altered_export_without_invoking(monkeypatch):
    from anygeometry import _planning_codec_registry as adapter
    module = sys.modules.get(adapter._NAME)
    if module is None:
        with pytest.raises(adapter.InspectionUnavailable):
            adapter.snapshot()
        return
    calls = []
    monkeypatch.setattr(module, 'identity', lambda: calls.append(True))
    fresh = runpy.run_path(adapter.__file__)
    assert fresh['_ANCHOR'] is None
    with pytest.raises(fresh['InspectionUnavailable']):
        fresh['snapshot']()
    assert calls == []
