"""Exact generated-wrapper scope and genuine plain-owned binding units."""
def test_legacy_helper_is_available():
    from _planning_legacy_contract import capture_with_legacy_dataclass_boundary
    assert callable(capture_with_legacy_dataclass_boundary)


import sys
import pytest
from _planning_contract_helpers import capture_with_dispatcher_boundary
from _planning_legacy_contract import legacy_wrapper_evidence
from anygeometry._planning_executor import PlannerSourceChanged


@pytest.mark.parametrize('change',['selected','unselected','class','code'])
def test_plain_owned_slot_class_mutations_have_admitted_baseline(monkeypatch,request,change):
    from anygeometry import _planning_native as native
    cls=native._SlotHandle
    method=cls.__int__
    captured=capture_with_dispatcher_boundary((method,cls),request=request)
    if captured is None:return # Only exact missing dispatcher, not mutation coverage.
    captured.verify();captured.verify(full=True)  # cold and retained warm baseline
    assert native._SlotHandle is cls and cls.__int__ is method
    if change=='selected':monkeypatch.setattr(cls,'__int__',lambda self:9)
    elif change=='unselected':monkeypatch.setattr(cls,'__hash__',lambda self:9)
    elif change=='class':monkeypatch.setattr(native,'_SlotHandle',type('Other',(),{}))
    else:monkeypatch.setattr(method,'__code__',method.__code__.replace())
    with pytest.raises(PlannerSourceChanged):captured.verify()
    with pytest.raises(PlannerSourceChanged):captured.verify(full=True)
    request.node.user_properties.append(('plain_owned_mutation_coverage',change))


@pytest.mark.parametrize('change',['wrapper','owned-method'])
def test_legacy_provider_source_skew_is_not_classified(monkeypatch,request,change):
    from anygeometry import tolerance
    cls=tolerance.TolerancePolicy
    if sys.version_info[:2] not in ((3,11),(3,12)):
        assert legacy_wrapper_evidence((cls,)) is None
        request.node.user_properties.append(('legacy_source_unit','legacy-provider-absent'))
        return
    assert legacy_wrapper_evidence((cls,)) is not None
    target=cls.__repr__ if change=='wrapper' else cls.effective_area
    monkeypatch.setattr(target,'__code__',target.__code__.replace(co_filename='<forged-provider>'))
    with pytest.raises(AssertionError):legacy_wrapper_evidence((cls,))
    request.node.user_properties.append(('legacy_source_unit','actual-provider-skew-rejected'))


def test_legacy_refusal_does_not_cover_unrelated_roots():
    from anygeometry.tolerance import TolerancePolicy
    assert legacy_wrapper_evidence((TolerancePolicy,object())) is None
