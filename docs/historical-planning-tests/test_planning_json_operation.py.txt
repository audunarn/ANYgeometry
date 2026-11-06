"""Completed closed-tree JSON operation connects to real guarded invocation."""
import json
import pytest
from anygeometry._planning_profiles import BranchProfiles
from anygeometry._planning_qualification import QualificationDecline
from anygeometry._planning_executor import PlannerSourceChanged


@pytest.mark.parametrize('ascii_only',[True,False])
def test_closed_json_operation_capture_and_actual_call(ascii_only):
    value={'text':'\u03a9','rows':[1,2.5,True,None]}
    kwargs=dict(sort_keys=True,separators=(',',':'),allow_nan=False,ensure_ascii=ascii_only)
    captured=BranchProfiles().capture(json.dumps,value,**kwargs)
    assert captured.manifest.nodes<100
    assert captured()==json.dumps(value,**kwargs)


def test_guarded_json_rechecks_tree_before_invocation():
    value={'x':[1]}
    captured=BranchProfiles().capture(json.dumps,value,sort_keys=True,separators=(',',':'),allow_nan=False)
    value['x'].append(object())
    with pytest.raises(PlannerSourceChanged):captured()


def test_guarded_json_rechecks_guard_definition(monkeypatch):
    captured=BranchProfiles().capture(json.dumps,{'x':1},sort_keys=True,separators=(',',':'),allow_nan=False)
    monkeypatch.setattr(BranchProfiles,'json',lambda *args:None)
    with pytest.raises(PlannerSourceChanged):captured()


def test_json_operation_declines_integer_conversion_context_dependence():
    with pytest.raises(QualificationDecline):BranchProfiles().json({'x':1<<2048},dict(sort_keys=True,separators=(',',':'),allow_nan=False))


@pytest.mark.parametrize('field',['__setattr__','indent'])
def test_initial_encoder_assignment_hooks_refuse_without_execution(monkeypatch,field):
    def foreign(*args):pytest.fail('encoder assignment hook executed')
    replacement=foreign if field=='__setattr__' else property(lambda self:None,foreign)
    monkeypatch.setattr(json.JSONEncoder,field,replacement,raising=False)
    with pytest.raises(QualificationDecline):BranchProfiles()


def test_json_operation_keeps_four_megabyte_bound():
    with pytest.raises(QualificationDecline):BranchProfiles().json({'x':'a'*4_000_001},dict(sort_keys=True,separators=(',',':'),allow_nan=False))


@pytest.mark.parametrize('where',['value','key'])
def test_ascii_surrogate_pairs_are_counted_as_twelve_bytes(where):
    text='\U0001f600'*333334
    value={'x':text} if where=='value' else {text:0}
    with pytest.raises(QualificationDecline):BranchProfiles().json(value,dict(sort_keys=True,separators=(',',':'),allow_nan=False))


def test_non_ascii_profile_uses_utf8_bound_for_supplementary_characters():
    value={'x':'\U0001f600'*1000}
    captured=BranchProfiles().capture(json.dumps,value,sort_keys=True,separators=(',',':'),allow_nan=False,ensure_ascii=False)
    assert len(captured().encode('utf-8'))==4008
