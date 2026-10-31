import contextvars
import json
from pathlib import Path
import sys
import types
import pytest

from anygeometry import _planning_qualification as q
from anygeometry import _planning_purity as p


def ledger_api():
    from anygeometry import _planning_ledger
    return _planning_ledger


def test_native_exports_do_not_define_task_input_types(monkeypatch):
    import threadpoolctl
    class NativeExport:
        __module__ = property(lambda self: 'not a static module string')
    m = types.ModuleType('pp1c_explicit_module')
    m.__file__ = __file__
    m.native = NativeExport
    monkeypatch.setattr(q, 'sys', types.SimpleNamespace(modules={m.__name__: m},
        executable=sys.executable, version=sys.version, platform=sys.platform,
        implementation=sys.implementation))
    monkeypatch.setattr(threadpoolctl, 'threadpool_info', lambda: [dict(
        filepath=__file__, num_threads=1, internal_api='controlled-test')])
    observed = q.capture_environment()
    assert all('NativeExport' not in x for x in observed['types'])
    with pytest.raises(q.QualificationDecline, match='type'):
        q.check_types(NativeExport(), observed)


def test_generated_native_pin_survives_json_transport(monkeypatch, tmp_path):
    import threadpoolctl
    path = tmp_path/'provider.pyd'
    path.write_bytes(b'controlled test file; not loaded as executable')
    provider = types.ModuleType('provider')
    provider.__file__ = str(path)
    runtime = types.ModuleType('cython_runtime')
    monkeypatch.setattr(q, 'sys', types.SimpleNamespace(modules={'provider': provider, 'cython_runtime': runtime},
        executable=sys.executable, version=sys.version, platform=sys.platform,
        implementation=sys.implementation))
    monkeypatch.setattr(threadpoolctl, 'threadpool_info', lambda: [dict(
        filepath=str(path), num_threads=1, internal_api='controlled-test')])
    observed = q.capture_environment(generated_modules=('cython_runtime',))
    q.verify_environment(json.loads(json.dumps(observed)), observed)


def test_explicit_model_and_domain_input_adapters():
    from anygeometry._planning_canaries import _fixture
    model, domains = _fixture()
    allowed = q.input_type_names()
    q.check_types((model, domains), {'types': allowed})
    class Intruder:
        pass
    with pytest.raises(q.QualificationDecline, match='type'):
        q.check_types((model, Intruder()), {'types': allowed})


def test_metaclass_equality_cannot_impersonate_an_allowed_input_type():
    calls = []
    class SpoofMeta(type):
        def __eq__(self, other):
            calls.append(other)
            return True
        __hash__ = type.__hash__
    class Spoof(metaclass=SpoofMeta):
        pass
    Spoof.__module__, Spoof.__qualname__ = 'builtins', 'int'
    with pytest.raises(q.QualificationDecline, match='type'):
        q.check_types(Spoof(), {'types': ['builtins.int']})
    assert calls == []


def test_inventory_handles_are_complete_and_ledger_omission_refuses():
    module = types.ModuleType('anygeometry.synthetic_ledger')
    module._MEMO = {}
    module.mystery = []
    objects = {}
    report = p.inventory((module,), store_objects=objects)
    assert set(objects) == {r.path for r in report.stores}
    ledger = ledger_api().build_ledger(report, objects)
    assert set(ledger.rows) == set(objects)
    with pytest.raises(q.QualificationDecline, match='declined'):
        ledger.prepare(owned_worker=True)
    rows = dict(ledger.rows)
    rows.pop(next(iter(rows)))
    with pytest.raises(q.QualificationDecline, match='coverage'):
        ledger_api().check_coverage(report, rows)


def test_reviewed_cache_reset_preserves_definition_metadata():
    module = types.ModuleType('anygeometry.quadric_events')
    module._MEMO = {'warm': 1}
    module.__all__ = ['public_name']
    objects = {}
    report = p.inventory((module,), store_objects=objects)
    ledger = ledger_api().build_ledger(report, objects)
    ledger.prepare(owned_worker=True)
    assert module._MEMO == {} and module.__all__ == ['public_name']


def test_active_context_declines_before_any_cache_is_reset():
    module = types.ModuleType('anygeometry.analytic_roots')
    module._ROOT_MEMO = contextvars.ContextVar('pp1c-test', default=None)
    module._MEMO = {'preserved': 1}
    token = module._ROOT_MEMO.set({'active_budget': 1})
    try:
        objects = {}
        report = p.inventory((module,), store_objects=objects)
        ledger = ledger_api().build_ledger(report, objects)
        with pytest.raises(q.QualificationDecline, match='context|declined'):
            ledger.prepare(owned_worker=True)
        assert module._MEMO == {'preserved': 1}
    finally:
        module._ROOT_MEMO.reset(token)


@pytest.mark.parametrize('enabled', [False, True])
def test_junction_switch_uses_its_declared_inactive_value(enabled):
    module = types.ModuleType('anygeometry.arrangement_geometry')
    module._JUNCTION_BOUNDS = contextvars.ContextVar('junction-switch', default=enabled)
    objects = {}
    report = p.inventory((module,), store_objects=objects)
    ledger = ledger_api().build_ledger(report, objects)
    if enabled:
        with pytest.raises(q.QualificationDecline, match='context'):
            ledger.prepare(owned_worker=True)
    else:
        ledger.prepare(owned_worker=True)
