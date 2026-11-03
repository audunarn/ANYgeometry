"""JSON native exports are anchored before late profile construction."""
import json
import _json
import pytest
from anygeometry import model,_planning_runtime_anchor as anchor
from anygeometry._planning_qualification import QualificationDecline


def test_json_owner_anchor_complete_and_names_explicit():
    assert anchor.JSON_ANCHOR[-1] is True
    assert tuple(row[0] for row in anchor.JSON_ANCHOR[2])==('make_encoder','encode_basestring','encode_basestring_ascii')
    anchor.verify_owner(model)


def test_paired_late_native_replacement_cannot_rebase(monkeypatch):
    from anygeometry._planning_profiles import BranchProfiles
    replacement=lambda *args:None
    monkeypatch.setattr(_json,'make_encoder',replacement)
    monkeypatch.setattr(json.encoder,'c_make_encoder',replacement)
    with pytest.raises(QualificationDecline):BranchProfiles()


@pytest.mark.parametrize('name',['markers','__call__','__name__'])
def test_mutable_native_heap_type_changes_are_detected(name):
    cls=_json.make_encoder
    original=getattr(cls,name)
    try:
        setattr(cls,name,'changed')
        with pytest.raises(QualificationDecline):anchor.verify_owner(model)
    finally:setattr(cls,name,original)
