"""PP1d regressions: each negative exercises the reviewed parent hole."""
import hashlib
import sys
import types

import pytest

from anygeometry import _planning_ledger as l
from anygeometry import _planning_purity as p
from anygeometry import _planning_qualification as q


def bound_canary():
    return 'ok'


def bound_task(value=None):
    return value


def gate(registry=None):
    stores = {} if registry is None else registry.resets
    pin = dict(schema='PP1b-v1', modules={__name__: q.pin_module(sys.modules[__name__])},
               files={}, python={}, versions={}, cpu={}, settings={},
               blas=[dict(num_threads=1)], types=q.input_type_names(),
               reset_registry={name: q._type_name(type(value)) for name, value in stores.items()})
    result = q.WorkerGate(pin, stores=stores, owned_worker=True, observe=lambda: pin, ledger=registry)
    result.start(dict(pair=bound_canary, face=bound_canary), dict(pair='ok', face='ok'))
    return result


def ledger(monkeypatch, source):
    module = types.ModuleType('anygeometry.pp1d_fixture')
    exec(source, module.__dict__)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    objects = {}
    report = p.inventory((module,), store_objects=objects)
    return module, l.build_ledger(report, objects)


def test_rebound_cache_is_reset_or_declined(monkeypatch):
    module, registry = ledger(monkeypatch, '_MEMO = {}')
    module._MEMO = {'warm': 1}
    try:
        registry.prepare(owned_worker=True)
    except q.QualificationDecline:
        return
    assert module._MEMO == {}, 'live rebound cache escaped reset'


def test_new_store_declines_before_reset(monkeypatch):
    module, registry = ledger(monkeypatch, '_MEMO = {"preserved": 1}')
    module.new_store = {'warm': 2}
    with pytest.raises(q.QualificationDecline):
        registry.prepare(owned_worker=True)
    assert module._MEMO == {'preserved': 1}


def test_reset_descendant_with_external_alias_declines(monkeypatch):
    module, registry = ledger(monkeypatch, '_MEMO = {"nested": {}}')
    module.external = module._MEMO['nested']
    module.external['warm'] = 1
    with pytest.raises(q.QualificationDecline, match='alias'):
        registry.prepare(owned_worker=True)
    assert module.external == {'warm': 1}


@pytest.mark.parametrize('source', [
    'def f(*, _memo={}): pass',
    'def f(*, _memo={}): pass\nalias = f.__kwdefaults__["_memo"]',
])
def test_default_memo_is_not_a_definition(monkeypatch, source):
    module, registry = ledger(monkeypatch, source)
    module.f.__kwdefaults__['_memo'][-0.0] = 'warm'
    with pytest.raises(q.QualificationDecline):
        registry.prepare(owned_worker=True)


def test_protected_content_change_declines(monkeypatch):
    module, registry = ledger(monkeypatch, '__all__ = ["original"]\n_MEMO = {"keep": 1}')
    module.__all__.append('changed')
    with pytest.raises(q.QualificationDecline):
        registry.prepare(owned_worker=True)
    assert module._MEMO == {'keep': 1}


def test_task_module_spoof_declines():
    worker = gate()
    def spoof():
        return 'must not run'
    spoof.__module__ = __name__
    with pytest.raises(q.QualificationDecline):
        worker.run(spoof)
    assert worker.poisoned


def test_task_foreign_globals_declines(monkeypatch):
    worker = gate()
    fake = types.FunctionType(bound_task.__code__, {'__name__': __name__}, 'bound_task', (None,))
    fake.__module__ = __name__
    monkeypatch.setitem(globals(), 'bound_task', fake)
    with pytest.raises(q.QualificationDecline):
        worker.run(fake)


@pytest.mark.parametrize('name', ['_warm_cache', 'normal'])
def test_plane_extra_state_declines(name):
    from anygeometry.surfaces import Plane
    plane = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    object.__setattr__(plane, name, {'warm': object()})
    with pytest.raises(q.QualificationDecline):
        q.check_types(plane, {'types': q.input_type_names()})


def test_model_extra_state_declines():
    from anygeometry.model import GeometryModel
    model = GeometryModel()
    model._fake_cache = {'warm': object()}
    with pytest.raises(q.QualificationDecline):
        q.check_types(model, {'types': q.input_type_names()})


def test_model_nested_surface_extra_state_declines():
    from anygeometry._planning_canaries import _fixture
    model, domains = _fixture()
    surface = next(iter(model.faces.values())).surface
    assert surface is not None
    object.__setattr__(surface, '_fake_cache', {'warm': object()})
    with pytest.raises(q.QualificationDecline):
        q.check_types(model, {'types': q.input_type_names()})


def test_live_input_is_not_accepted_before_reset():
    worker = gate()
    with pytest.raises(q.QualificationDecline):
        worker.run(bound_task, [1, 2])


def test_real_gate_requires_complete_live_ledger():
    with pytest.raises(q.QualificationDecline, match='ledger'):
        q.WorkerGate({'reset_registry': {}}, stores={}, owned_worker=True)


def test_builtin_pin_is_runtime_not_launcher():
    if sys.platform != 'win32':
        with pytest.raises(q.QualificationDecline, match='^interpreter runtime adapter is Windows-only$'):
            q.pin_module(sys)
        return
    pin = q.pin_module(sys)
    assert 'runtime_files' in pin, 'builtin was pinned to launcher only'
    names = {__import__('pathlib').Path(path).name.lower() for path in pin['runtime_files']}
    if sys.platform == 'win32':
        runtime_name = f'python{sys.version_info.major}{sys.version_info.minor}{getattr(sys, "abiflags", "")}.dll'
        assert runtime_name in names
        assert any(name.startswith('vcruntime140') for name in names)
    assert pin.get('sha256') != hashlib.sha256(open(sys.executable, 'rb').read()).hexdigest()


def test_payload_decoded_after_reset_and_checksum_failure_poisoned(monkeypatch):
    worker = gate()
    cache = {'old': 1}
    worker.stores = {'cache': cache}
    original = q.decode_task_payload
    calls = []
    def decode(payload, digest):
        assert cache == {}
        calls.append('decoded')
        return original(payload, digest)
    monkeypatch.setattr(q, 'decode_task_payload', decode)
    payload = b'{"args":[[1,2]],"kwargs":{}}'
    result = worker.run_json(bound_task, payload, hashlib.sha256(payload).hexdigest())
    assert result == [1, 2] and calls == ['decoded']
    with pytest.raises(q.QualificationDecline, match='checksum'):
        worker.run_json(bound_task, payload, 'wrong')
    assert worker.poisoned and not worker.ready


def bound_mutation():
    _mutation_target.__all__.append('changed')
    return 'result must not escape'


def test_posttask_protected_mutation_discards_result(monkeypatch):
    module, registry = ledger(monkeypatch, '__all__ = ["original"]\n_MEMO = {}')
    worker = gate(registry)
    monkeypatch.setitem(globals(), '_mutation_target', module)
    with pytest.raises(q.QualificationDecline, match='protected'):
        worker.run(bound_mutation)
    assert worker.poisoned and not worker.ready


@pytest.mark.parametrize('source,attribute', [
    ('def f(): pass', 'f'), ('class C: pass', 'C')])
def test_new_function_or_class_store_declines(monkeypatch, source, attribute):
    module, registry = ledger(monkeypatch, source + '\n_MEMO = {}')
    getattr(module, attribute).cache = {'new': 1}
    with pytest.raises(q.QualificationDecline, match='live store'):
        registry.prepare(owned_worker=True)


@pytest.mark.parametrize('first,second', [(0.0, -0.0), (1, True), (1, 1.0)])
def test_protected_digest_preserves_typed_bits(first, second):
    assert l.content_digest([first]) != l.content_digest([second])
