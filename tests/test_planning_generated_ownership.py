"""R5 generated edges derive authority from a live registered owner."""
import types
import pytest
from anygeometry._planning_manifest import capture_callable_manifest
from anygeometry._planning_qualification import QualificationDecline
from anygeometry.tolerance import TolerancePolicy


def test_registered_generated_slots_and_repr_closure_are_attested():
    capture_callable_manifest((TolerancePolicy,)).verify()


def test_module_spelling_does_not_register_a_class():
    class Foreign:
        pass
    Foreign.__module__='anygeometry.tolerance'
    with pytest.raises(QualificationDecline,match='unregistered'):
        capture_callable_manifest((Foreign,))


def test_foreign_generated_closure_class_refuses(monkeypatch):
    original=vars(TolerancePolicy)['__setattr__']
    class Foreign:
        pass
    Foreign.__module__=TolerancePolicy.__module__
    cells=tuple(types.CellType(Foreign) if name=='cls' else cell
        for name,cell in zip(original.__code__.co_freevars,original.__closure__))
    forged=types.FunctionType(original.__code__,original.__globals__,original.__name__,
        original.__defaults__,cells)
    forged.__module__=original.__module__
    forged.__qualname__=original.__qualname__
    monkeypatch.setattr(TolerancePolicy,'__setattr__',forged)
    with pytest.raises(QualificationDecline):
        capture_callable_manifest((TolerancePolicy,))


def test_forged_provider_origin_refuses(monkeypatch):
    original=vars(TolerancePolicy)['__repr__']
    forged=types.FunctionType(original.__code__.replace(co_filename='<forged-provider>'),
        original.__globals__,original.__name__,original.__defaults__,original.__closure__)
    forged.__module__=original.__module__
    forged.__qualname__=original.__qualname__
    forged.__dict__.update(vars(original))
    monkeypatch.setattr(TolerancePolicy,'__repr__',forged)
    with pytest.raises(QualificationDecline):
        capture_callable_manifest((TolerancePolicy,))
