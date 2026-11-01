"""Public-route PP5 controls must compare actual plan and applied-model content."""
import importlib
import pytest
from anygeometry import batch_intersections as bi
from anygeometry._planning_canaries import _fixture
from anygeometry.serialization import from_dict,to_dict

def test_public_real_pair_and_face_stages_and_apply_ci(monkeypatch,tmp_path):
    module=importlib.import_module('anygeometry._planning_host')
    model,_=_fixture();doc=to_dict(model);source=from_dict(doc);ref=from_dict(doc)
    faces=tuple(source.faces)
    monkeypatch.delenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING',raising=False)
    serial=bi.plan_intersections(ref,faces,policy='connect')
    bi.apply_intersections(ref,serial,policy='connect')
    expected=(serial.content_checksum,to_dict(ref)['checksum']['value'])
    def observed(plan):
        candidate=from_dict(doc);bi.apply_intersections(candidate,plan,policy='connect')
        return (plan.content_checksum,to_dict(candidate)['checksum']['value'])
    control=module.StageControl(tmp_path/'parallel',expected_ci=expected,observed_ci=observed,workers=2)
    monkeypatch.setenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING','1')
    plan=bi.plan_intersections(source,faces,policy='connect',_parallel_control=control)
    assert control.mode=='parallel',control.decline_reason
    assert plan.content_checksum==serial.content_checksum
    assert [r['stage'] for r in control.reports]==['pair','face']
    assert all(r['actual_workers']==2 for r in control.reports)
    assert control.ci_verified and to_dict(source)==doc

def test_ci_refusal_cannot_replay_after_source_mutation(tmp_path):
    from anygeometry._planning_host import StageControl,qualified_plan
    from anygeometry._planning_executor import PlannerSourceChanged
    model,_=_fixture();model=from_dict(to_dict(model));events=[]
    def observed(plan):model._revision+=1;return ('wrong','apply')
    control=StageControl(tmp_path/'never',expected_ci=('plan','apply'),observed_ci=observed)
    with pytest.raises(PlannerSourceChanged):
        qualified_plan(model,control,lambda:events.append('replay'),lambda bridge:object(),policy=bi.IntersectionBatchPolicy(),mode='exact')
    assert events==[]
