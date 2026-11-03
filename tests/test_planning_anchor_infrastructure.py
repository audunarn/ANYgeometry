"""Exact owner-held provenance infrastructure remains bound at ledger boundaries."""
import importlib
import sys
import types
import pytest
from anygeometry import model, _planning_runtime_anchor as anchor
from anygeometry._planning_qualification import QualificationDecline


def test_owner_keeps_original_module_not_nested_anchor_payload():
    assert model._RUNTIME_OPERATION_OWNER is anchor.OWNER
    assert '_RUNTIME_OPERATION_ANCHOR' not in vars(model)


def test_inventory_classifies_only_verified_anchor_infrastructure():
    from anygeometry._planning_ledger import _census
    objects={}
    report,_=_census((model,anchor),objects)
    assert not any(path.startswith('anygeometry._planning_runtime_anchor.') for path in objects)
    assert not any('_RUNTIME_OPERATION_OWNER' in path for path in objects)
    assert not report.truncated


def test_delete_and_reimport_cannot_rebaseline_owner():
    del sys.modules[anchor.__name__]
    replacement=importlib.import_module(anchor.__name__)
    try:
        assert replacement is not anchor
        assert model._RUNTIME_OPERATION_OWNER is anchor.OWNER
        with pytest.raises(QualificationDecline):replacement.verify_owner(model)
        with pytest.raises(QualificationDecline):anchor.verify_owner(model)
    finally:
        sys.modules[anchor.__name__]=anchor
        vars(sys.modules['anygeometry'])['_planning_runtime_anchor']=anchor


def test_unknown_infrastructure_alias_remains_unknown():
    from anygeometry._planning_ledger import _census
    module=types.ModuleType('anygeometry.ppq_fake_infrastructure')
    module.alias=types.ModuleType(anchor.__name__)
    report,_=_census((module,),{})
    assert any(path.endswith('.alias') for path,reason in report.unknowns)


def test_ledger_boundary_rejects_anchor_function_replacement(monkeypatch):
    from anygeometry._planning_ledger import _census
    monkeypatch.setattr(anchor,'_metadata',lambda value,remaining=512:None)
    with pytest.raises(QualificationDecline):_census((model,),{})


@pytest.mark.parametrize('change',['verifier_code','verifier_symbol','owner_registry','operation_registry'])
def test_exact_infrastructure_seal_refuses_tamper(monkeypatch,change):
    from anygeometry._planning_ledger import runtime_infrastructure
    if change=='verifier_code':monkeypatch.setattr(anchor.verify_owner,'__code__',(lambda *a,**k:()).__code__)
    elif change=='verifier_symbol':monkeypatch.setattr(anchor,'verify_owner',lambda *a:())
    elif change=='owner_registry':monkeypatch.setattr(anchor,'OWNER',tuple(list(anchor.OWNER)))
    else:monkeypatch.setattr(anchor,'NUMPY_NAMES',())
    with pytest.raises(QualificationDecline):runtime_infrastructure()


def test_ledger_rechecks_infrastructure_source(monkeypatch):
    from anygeometry import _planning_ledger as ledger, _planning_qualification as qualification
    module=types.ModuleType('ppq_empty_ledger')
    before=ledger.loaded_geometry_ledger(modules=(module,))
    pin=qualification.pin_module
    monkeypatch.setattr(qualification,'pin_module',lambda module,**kw:dict(pin(module,**kw),sha256='0'*64))
    with pytest.raises(QualificationDecline,match='source/binding'):before.validate()
