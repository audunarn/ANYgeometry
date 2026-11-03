"""Run with assigned caller imports: real external write-observer boundary."""
import sys,types
import pytest
import numpy as np
from anygeometry._planning_qualification import ModuleObserver,QualificationDecline,is_package_module,pin_module


def test_real_observer_keeps_portable_pin_and_legacy_refusal():
    assert type(np) is not types.ModuleType
    captured=ModuleObserver(np);captured.verify()
    assert pin_module(np,package_only=True,observer=captured)['path']==np.__file__
    with pytest.raises(QualificationDecline):pin_module(np)


def test_observer_foreign_registry_collision_refuses_without_equality(monkeypatch):
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
def test_observer_initial_source_or_namespace_skew_refuses(monkeypatch,target):
    owner=sys.modules['anysolver._qualified_authority_epoch'];cls=type(np)
    if target=='read_hook':monkeypatch.setattr(cls,'__getattribute__',lambda *a:pytest.fail('changed read hook invoked'),raising=False)
    else:
        function=owner._make_attribute_registry if target=='factory' else vars(cls)['__setattr__']
        monkeypatch.setattr(function,'__code__',function.__code__.replace(co_consts=(*function.__code__.co_consts,'forged')))
    try:
        with pytest.raises(QualificationDecline):ModuleObserver(np)
    finally:monkeypatch.undo()


@pytest.mark.parametrize('target',['factory','setter','registry','registry_namespace','module','read_hook'])
def test_observer_retained_changes_are_terminal(monkeypatch,target):
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
