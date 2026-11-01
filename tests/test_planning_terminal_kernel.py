"""Terminal ownership errors must cross actual numerical retry handlers intact."""
import pytest

from anygeometry import _planning_host as host, _planning_executor as executor
from anygeometry import batch_intersections as bi, material_arrangement as ma
from anygeometry._planning_canaries import _fixture


@pytest.mark.parametrize('reuse',[False,True])
@pytest.mark.parametrize('kind',['PlannerCancelled','PlannerDeadline','PlannerSourceChanged','PlannerCleanupError'])
def test_actual_ring_split_does_not_reissue_or_replay_terminal(tmp_path,monkeypatch,reuse,kind):
    model,_=_fixture();events=[];hits=[];owner=[];error=getattr(executor,kind)('injected terminal boundary')
    control=host.StageControl(tmp_path,expected_ci=('p','a'),observed_ci=lambda _:pytest.fail('CI after terminal'),reuse_workers=reuse)
    def run(self,stage,tasks,serial_task):
        events.append(stage)
        return tuple(serial_task(task) for task in tasks)
    monkeypatch.setattr(host.Bridge,'run',run);monkeypatch.setattr(host._SessionBridge,'run',run)
    def fail():hits.append(error);raise error
    def once(*a,cancellation_check,**k):
        events.append('inside-ring-split');owner[0].poll=fail;cancellation_check()
        pytest.fail('terminal callback returned')
    monkeypatch.setattr(ma,'_arrange_material_once',once)
    policy=bi.IntersectionBatchPolicy()
    def parallel(bridge):
        owner.append(bridge)
        return bi._plan_intersections(model,tuple(model.faces),policy=policy,_planning_mode='exact',_seam_policy='ring_split',_stages=bridge)
    with pytest.raises(type(error)) as caught:
        host.qualified_plan(model,control,lambda:pytest.fail('serial fallback after terminal'),parallel,policy=policy,mode='exact')
    assert caught.value is error and hits==[error]
    assert events==['pair','face','inside-ring-split']
    assert control.mode=='terminal' and not control.ci_verified and control.decline_reason is None


def test_actual_kernel_cannot_convert_owned_cleanup_failure_to_replay(tmp_path,monkeypatch):
    model,_=_fixture();events=[];failure=OSError('owned close failure')
    class Planner:
        def __init__(self,*a,**k):events.append('start')
        def submit_stage(self,*a,**k):events.append('submit');return {}
        def close(self):events.append('close');raise failure
    monkeypatch.setattr(host,'OwnedPlanner',Planner)
    control=host.StageControl(tmp_path,expected_ci=('p','a'),observed_ci=lambda _:pytest.fail('CI after failed cleanup'))
    policy=bi.IntersectionBatchPolicy()
    def parallel(bridge):
        return bi._plan_intersections(model,tuple(model.faces),policy=policy,_planning_mode='exact',_seam_policy='ring_split',_stages=bridge)
    with pytest.raises(executor.PlannerCleanupError) as caught:
        host.qualified_plan(model,control,lambda:pytest.fail('serial fallback after failed cleanup'),parallel,policy=policy,mode='exact')
    assert caught.value.__cause__ is failure and events==['start','submit','close']
    assert control.mode=='terminal' and not control.ci_verified and control.decline_reason is None
