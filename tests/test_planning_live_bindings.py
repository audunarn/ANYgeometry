import hashlib
import marshal
import sys
import types

import pytest
from anygeometry import _planning_ledger as l
from anygeometry import _planning_purity as p
from anygeometry import _planning_qualification as q


def fixture(monkeypatch):
    owner = types.ModuleType('pp1f_runtime_fixture')
    owner.__file__ = __file__
    owner.__dict__['__builtins__'] = {}
    exec('def implementation():\n    return 1', owner.__dict__)
    monkeypatch.setitem(sys.modules, owner.__name__, owner)
    module = types.ModuleType('anygeometry.backend')
    module._RUNTIME_PYTHON_GLOBALS = owner.__dict__
    objects = {}
    report = p.inventory((module,), store_objects=objects)
    ledger = l.build_ledger(report, objects, modules=(module,))
    return owner, ledger


def test_registered_namespace_is_live_and_never_reset(monkeypatch):
    owner, ledger = fixture(monkeypatch)
    path = 'anygeometry.backend._RUNTIME_PYTHON_GLOBALS'
    assert ledger.rows[path]['action'] == 'require-live-binding'
    before = dict(owner.__dict__)
    ledger.prepare(owned_worker=True)
    assert owner.__dict__ == before
    assert path not in ledger.resets


def test_live_code_pin_contains_actual_code_hash(monkeypatch):
    owner, ledger = fixture(monkeypatch)
    pins = ledger.live_code_pins()
    pin = pins['anygeometry.backend._RUNTIME_PYTHON_GLOBALS']['implementation']
    assert pin['sha256'] == hashlib.sha256(marshal.dumps(owner.implementation.__code__)).hexdigest()
    assert pin['module'] == owner.__name__


@pytest.mark.parametrize('mutation', ['rebind', 'code', 'add_name', 'module'])
def test_live_binding_mutation_refuses_before_reset(monkeypatch, mutation):
    owner, ledger = fixture(monkeypatch)
    assert ledger.rows['anygeometry.backend._RUNTIME_PYTHON_GLOBALS']['action'] == 'require-live-binding'
    if mutation == 'rebind':
        owner.implementation = types.FunctionType(owner.implementation.__code__, owner.__dict__)
    elif mutation == 'code':
        owner.implementation.__code__ = owner.implementation.__code__.replace(co_consts=(None, 2))
    elif mutation == 'add_name':
        owner.new_binding = 1
    else:
        monkeypatch.setitem(sys.modules, owner.__name__, types.ModuleType(owner.__name__))
    with pytest.raises(q.QualificationDecline):
        ledger.prepare(owned_worker=True)


def test_named_plane_state_is_preserved_and_guarded(monkeypatch):
    from anygeometry.nurbs_surface_intersection import _topology
    from pathlib import Path
    assert sys.modules[_topology.__name__] is _topology
    assert Path(_topology.__file__).resolve()==Path(l.__file__).resolve().parent/'nurbs_surface_intersection/_topology.py'
    ledger = l.loaded_geometry_ledger(modules=(_topology,))
    assert not ledger.report.truncated
    key = 'anygeometry.nurbs_surface_intersection._topology._PLANE.__dict__'
    assert ledger.rows[key]['action'] == 'require-live-binding'
    value = ledger.objects[key]
    old = l.content_digest(value)
    ledger.validate()
    assert l.content_digest(value) == old
    monkeypatch.setitem(value, 'origin', (1., 0., 0.))
    with pytest.raises(q.QualificationDecline):
        ledger.validate()


def test_gate_requires_reference_live_code_pins(monkeypatch):
    owner, ledger = fixture(monkeypatch)
    assert hasattr(ledger, 'live_code_pins')
    with pytest.raises(q.QualificationDecline, match='live code'):
        q.WorkerGate({'reset_registry': {}}, stores={}, owned_worker=True,
                     ledger=ledger, observe=lambda: {})


@pytest.mark.parametrize('mutation', ['function', 'code'])
def test_attested_function_and_code_references_stay_alive(monkeypatch, mutation):
    import gc
    import weakref
    owner, ledger = fixture(monkeypatch)
    function_ref = weakref.ref(owner.implementation)
    code_ref = weakref.ref(owner.implementation.__code__)
    if mutation == 'function':
        owner.implementation = None
    else:
        owner.implementation.__code__ = owner.implementation.__code__.replace(co_consts=(None, 2))
    gc.collect()
    assert function_ref() is not None, 'recorded function identity could be recycled'
    assert code_ref() is not None, 'recorded code identity could be recycled'
    with pytest.raises(q.QualificationDecline):
        ledger.validate()
