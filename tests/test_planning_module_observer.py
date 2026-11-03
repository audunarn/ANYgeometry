"""Standalone module pinning and the real external write-observer boundary.

Genuine observer mutation coverage requires assigned caller imports. Standalone
CI proves native NumPy module pins and fail-closed unknown-subclass handling;
JUnit properties distinguish it from the caller-integration contract.
"""
import hashlib
from pathlib import Path
import sys,types
import pytest
import numpy as np
from anygeometry._planning_qualification import ModuleObserver,QualificationDecline,is_package_module,pin_module


def _standalone_contract(record_property):
    """Exercise the standalone boundary when no real provider is installed."""
    if 'anysolver._qualified_authority_epoch' in sys.modules:
        record_property('observer_contract', 'genuine_provider')
        return False
    record_property('observer_contract', 'standalone')
    assert type(np) is types.ModuleType
    assert is_package_module(np)
    expected_path = Path(np.__file__).resolve()
    expected_pin = dict(kind='python', path=str(expected_path),
                        sha256=hashlib.sha256(expected_path.read_bytes()).hexdigest())
    assert pin_module(np) == expected_pin
    assert pin_module(np, package_only=True) == expected_pin

    class Foreign(types.ModuleType):
        @property
        def __dict__(self):
            pytest.fail('unadmitted module read hook ran')

    module = types.ModuleType('external')
    module.__class__ = Foreign
    with pytest.raises(QualificationDecline, match='unregistered external module observer'):
        is_package_module(module)
    with pytest.raises(QualificationDecline, match='unregistered external module observer'):
        ModuleObserver(module)
    with pytest.raises(QualificationDecline, match='unpinned custom module type'):
        pin_module(module)
    with pytest.raises(QualificationDecline, match='unregistered external module observer'):
        pin_module(module, package_only=True)
    assert 'anysolver._qualified_authority_epoch' not in sys.modules
    return True


def test_real_observer_keeps_portable_pin_and_legacy_refusal(record_property):
    if _standalone_contract(record_property):
        return
    assert type(np) is not types.ModuleType
    captured=ModuleObserver(np);captured.verify()
    assert pin_module(np,package_only=True,observer=captured)['path']==np.__file__
    with pytest.raises(QualificationDecline):pin_module(np)


def test_observer_foreign_registry_collision_refuses_without_equality(monkeypatch,record_property):
    if _standalone_contract(record_property):
        return
    owner=sys.modules['anysolver._qualified_authority_epoch'];calls=[]
    cell=next(cell for name,cell in zip(owner._WATCH_MODULE.__code__.co_freevars,owner._WATCH_MODULE.__closure__)
              if name=='tracked_module_types')
    class Collision:
        def __hash__(self):return hash(types.ModuleType)
        def __eq__(self,other):calls.append('eq');raise AssertionError('foreign key equality')
    monkeypatch.setattr(cell,'cell_contents',{Collision():type(np)})
    with pytest.raises(QualificationDecline):ModuleObserver(np)
    assert calls==[]


@pytest.mark.parametrize('hook',[False,True])
def test_unknown_external_module_subclass_refuses_without_read_hook(hook):
    class Foreign(types.ModuleType):
        if hook:
            @property
            def __dict__(self):pytest.fail('unadmitted module read hook ran')
    module=types.ModuleType('external');module.__class__=Foreign
    with pytest.raises(QualificationDecline):is_package_module(module)


@pytest.mark.parametrize('target',['factory','setter','read_hook'])
def test_observer_initial_source_or_namespace_skew_refuses(monkeypatch,target,record_property):
    if _standalone_contract(record_property):
        return
    owner=sys.modules['anysolver._qualified_authority_epoch'];cls=type(np)
    if target=='read_hook':monkeypatch.setattr(cls,'__getattribute__',lambda *a:pytest.fail('changed read hook invoked'),raising=False)
    else:
        function=owner._make_attribute_registry if target=='factory' else vars(cls)['__setattr__']
        monkeypatch.setattr(function,'__code__',function.__code__.replace(co_consts=(*function.__code__.co_consts,'forged')))
    try:
        with pytest.raises(QualificationDecline):ModuleObserver(np)
    finally:monkeypatch.undo()


@pytest.mark.parametrize('target',['factory','setter','registry','registry_namespace','module','read_hook'])
def test_observer_retained_changes_are_terminal(monkeypatch,target,record_property):
    if _standalone_contract(record_property):
        return
    owner=sys.modules['anysolver._qualified_authority_epoch'];cls=type(np);captured=ModuleObserver(np)
    if target in ('factory','setter'):
        function=owner._make_attribute_registry if target=='factory' else vars(cls)['__setattr__']
        monkeypatch.setattr(function,'__code__',function.__code__.replace())
    elif target=='registry':
        cells=dict(zip(owner._WATCH_MODULE.__code__.co_freevars,(cell.cell_contents for cell in owner._WATCH_MODULE.__closure__)))
        monkeypatch.setitem(cells['tracked_module_types'],types.ModuleType,types.ModuleType)
    elif target=='registry_namespace':
        cell=next(cell for name,cell in zip(owner._WATCH_MODULE.__code__.co_freevars,owner._WATCH_MODULE.__closure__)
                  if name=='tracked_module_types')
        monkeypatch.setattr(cell,'cell_contents',dict(cell.cell_contents))
    elif target=='module':monkeypatch.setitem(sys.modules,'numpy',types.ModuleType('numpy'))
    else:monkeypatch.setattr(cls,'__getattribute__',lambda *a:pytest.fail('changed read hook invoked'),raising=False)
    try:
        with pytest.raises(QualificationDecline):captured.verify()
    finally:monkeypatch.undo()
