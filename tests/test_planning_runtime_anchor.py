"""Owner-load anchors cannot be silently reconstructed by a later validator."""
import importlib
import pytest
import numpy as np
from anygeometry import model
from anygeometry._planning_qualification import QualificationDecline


def anchor():
    from anygeometry import _planning_runtime_anchor
    return _planning_runtime_anchor


def test_owner_import_created_immutable_anchor():
    module=anchor()
    assert type(module.ANCHOR) is tuple
    assert model._RUNTIME_OPERATION_OWNER is module.OWNER
    assert module.verify_owner(model) is module.ANCHOR


def test_initial_foreign_wrapper_refuses_without_rebase(monkeypatch):
    module=anchor();original=module.ANCHOR
    monkeypatch.setattr(np.linalg,'norm',lambda *a,**k:0)
    with pytest.raises(QualificationDecline):module.verify_owner(model)
    assert module.ANCHOR is original


@pytest.mark.parametrize('where',['module','owner','delete'])
def test_anchor_rebinding_or_deletion_refuses(monkeypatch,where):
    module=anchor()
    if where=='module':monkeypatch.setattr(module,'ANCHOR',tuple(list(module.ANCHOR)))
    elif where=='owner':monkeypatch.setattr(model,'_RUNTIME_OPERATION_OWNER',())
    else:monkeypatch.delattr(model,'_RUNTIME_OPERATION_OWNER')
    with pytest.raises(QualificationDecline):module.verify_owner(model)


def test_anchor_reload_cannot_refresh_trust():
    module=anchor();original=module.ANCHOR
    with pytest.raises(RuntimeError,match='reload'):importlib.reload(module)
    assert module.ANCHOR is original
    module.verify_owner(model)
