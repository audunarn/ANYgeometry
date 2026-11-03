"""Selected ABC decisions are checked against a fixed table without resetting H."""
import abc
import numbers
import weakref
import pytest
from anygeometry._planning_runtime import verify_numeric_abcs
from anygeometry._planning_qualification import QualificationDecline


def test_warm_irrelevant_cache_history_preserves_selected_decisions(monkeypatch):
    class Unrelated:pass
    assert not isinstance(Unrelated(),numbers.Rational)
    monkeypatch.setattr(abc.ABCMeta,'_abc_caches_clear',lambda *a:pytest.fail('host cache cleared'))
    verify_numeric_abcs()


@pytest.mark.parametrize('predicate',['__subclasscheck__','__instancecheck__'])
def test_poisoned_selected_predicate_refuses(monkeypatch,predicate):
    original=getattr(abc.ABCMeta,predicate)
    def poisoned(cls,value):
        if cls is numbers.Rational:return False
        return original(cls,value)
    monkeypatch.setattr(abc.ABCMeta,predicate,poisoned)
    with pytest.raises(QualificationDecline,match='predicate'):verify_numeric_abcs()


def test_unknown_virtual_registration_refuses(monkeypatch):
    class Foreign:pass
    original=abc._get_dump
    def altered(cls):
        registry,positive,negative,version=original(cls)
        return (registry|{weakref.ref(Foreign)},positive,negative,version) if cls is numbers.Real else original(cls)
    monkeypatch.setattr(abc,'_get_dump',altered)
    with pytest.raises(QualificationDecline,match='registration'):verify_numeric_abcs()
