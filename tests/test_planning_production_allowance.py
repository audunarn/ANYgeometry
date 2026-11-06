"""Pure fake-clock production allowance contracts; no geometry/native execution."""
import json
from types import SimpleNamespace
import pytest
from anygeometry import _planning_controller as controller, _planning_host as host
from anygeometry import _planning_executor as executor
from anygeometry._planning_values import encode_value
from anygeometry.runtime_diagnostics import collect_runtime_records
from test_planning_r8_controller import _finalization_fixture


def _fixture(monkeypatch,tmp_path):
    bridge_type=getattr(controller,'_ProductionBridge',host._SessionBridge)
    work,plan,policy=_finalization_fixture(monkeypatch,tmp_path)
    monkeypatch.setattr(controller,'_SessionBridge',host._SessionBridge)
    if hasattr(controller,'_ProductionBridge'):
        monkeypatch.setattr(controller,'_ProductionBridge',bridge_type)
    clock=[0.];events=[];seen=[]
    monkeypatch.setattr(controller.time,'monotonic',lambda:clock[0])
    original_close=bridge_type.close
    def close(self):
        original_close(self);events.append('retired')
    monkeypatch.setattr(bridge_type,'close',close)
    def parallel(bridge):
        seen.append(bridge)
        def kernel(task):
            assert bridge.retired and bridge.session is None
            events.append('sample');clock[0]=100.
            return None
        candidate=dict(sampling_sha256='c'*64,task_layout=((0,0,'pair'),),
            results=[json.dumps(encode_value(None,codec='v2'))])
        bridge.pending.append(('pair',({'kind':'pair','cost':1.},),kernel,candidate))
        return plan
    return work,plan,policy,clock,events,seen,parallel


def test_allowance_expired_during_required_sample_discards_candidate_and_replays_once(monkeypatch,tmp_path):
    work,plan,policy,clock,events,seen,parallel=_fixture(monkeypatch,tmp_path)
    replay=object();polls=[]
    def serial(poll):
        assert seen[0].retired and seen[0].session is None
        assert 'retired' in events and events.index('retired')<events.index('sample')
        events.append('serial');poll();polls.append(True);clock[0]=300.;poll()
        return replay
    with collect_runtime_records() as records:
        actual=controller.run(None,lambda:pytest.fail('wrong serial entry'),parallel,
            policy=policy,mode='exact',serial_replay=serial)
    assert actual is replay and actual is not plan
    assert events.count('serial')==1 and polls==[True]
    receipt=records['parallel_planning'][-1]
    assert receipt['disposition']=='serial' and receipt['retired'] is True
    assert receipt['stages']==[] and 'allowance' in receipt['reason']
    assert work.is_dir()


@pytest.mark.parametrize('name',['PlannerDeadline','PlannerCancelled','PlannerSourceChanged','PlannerMemoryExceeded','PlannerCleanupError'])
def test_explicit_terminal_after_retirement_never_replays(monkeypatch,tmp_path,name):
    work,plan,policy,clock,events,seen,parallel=_fixture(monkeypatch,tmp_path)
    failure=getattr(executor,name)('explicit terminal')
    def callback():
        if 'sample' in events:raise failure
        return False
    policy.cancellation_check=callback
    # Keep internal clock unexpired so the actual explicit callback is observed.
    original=parallel
    def active(bridge):
        result=original(bridge)
        kind,tasks,kernel,candidate=bridge.pending[-1]
        def task(value):
            result=kernel(value);clock[0]=1.;return result
        bridge.pending[-1]=(kind,tasks,task,candidate)
        return result
    with collect_runtime_records() as records,pytest.raises(type(failure)) as caught:
        host.qualified_plan(None,None,lambda:pytest.fail('terminal replay'),active,policy=policy,mode='exact')
    assert caught.value is failure and seen[0].retired is True
    assert records['parallel_planning'][-1]['disposition']=='terminal'


@pytest.mark.parametrize('fault',['source','memory','cleanup','caller_deadline'])
def test_allowance_fallback_does_not_bypass_terminal_boundaries(monkeypatch,tmp_path,fault):
    work,plan,policy,clock,events,seen,parallel=_fixture(monkeypatch,tmp_path)
    errors=dict(source=executor.PlannerSourceChanged,memory=executor.PlannerMemoryExceeded,
        cleanup=executor.PlannerCleanupError,caller_deadline=executor.PlannerDeadline)
    failure=errors[fault]('fallback boundary')
    if fault=='source':
        def verify(source):
            if 'sample' in events:raise failure
        monkeypatch.setattr(controller,'verify_sources',verify)
    elif fault=='memory':
        lease=controller.memory_profile().begin()
        def check():
            if 'sample' in events:raise failure
        lease.check=check
    elif fault=='cleanup':
        bridge_type=getattr(controller,'_ProductionBridge',host._SessionBridge)
        close=bridge_type.close
        def cleanup(self):
            if 'sample' in events:
                self._retirement_error=failure;raise failure
            close(self)
        monkeypatch.setattr(bridge_type,'close',cleanup)
    else:
        def callback():
            if 'sample' in events:raise failure
            return False
        policy.cancellation_check=callback
    with collect_runtime_records() as records,pytest.raises(type(failure)) as caught:
        host.qualified_plan(None,None,lambda:pytest.fail('terminal replay'),parallel,policy=policy,mode='exact')
    assert caught.value is failure
    assert records['parallel_planning'][-1]['disposition']=='terminal'


def test_stage_control_and_owned_deadlines_stay_terminal(monkeypatch,tmp_path):
    clock=[0.];monkeypatch.setattr(host.time,'monotonic',lambda:clock[0])
    control=SimpleNamespace(timeout=100,tree_options=lambda:{})
    bridge=host._SessionBridge({},control)
    clock[0]=100.
    with pytest.raises(executor.PlannerDeadline,match='stage planning deadline consumed'):
        bridge.check_clock()


def test_owned_process_deadline_is_not_converted(monkeypatch,tmp_path):
    work,plan,policy,clock,events,seen,parallel=_fixture(monkeypatch,tmp_path)
    failure=executor.PlannerDeadline('owned planner deadline consumed')
    def owned(bridge):raise failure
    with pytest.raises(executor.PlannerDeadline) as caught:
        controller.run(None,lambda:pytest.fail('owned timeout replay'),owned,policy=policy,mode='exact')
    assert caught.value is failure


def test_unexpired_production_candidate_is_still_parallel(monkeypatch,tmp_path):
    work,plan,policy,clock,events,seen,parallel=_fixture(monkeypatch,tmp_path)
    def active(bridge):
        seen.append(bridge);return plan
    with collect_runtime_records() as records:
        result=controller.run(None,lambda:pytest.fail('unexpired replay'),active,policy=policy,mode='exact')
    assert result is plan
    assert records['parallel_planning'][-1]['disposition']=='parallel'


def test_production_allowance_cannot_be_swallowed_by_geometry_retry(monkeypatch,tmp_path):
    from anygeometry.errors import GeometryError
    clock=[0.];monkeypatch.setattr(controller.time,'monotonic',lambda:clock[0])
    bridge=controller._ProductionBridge({},SimpleNamespace(timeout=100,tree_options=lambda:{}))
    clock[0]=100.
    with pytest.raises(controller._ProductionAllowanceExpired):
        try:bridge.check_clock()
        except GeometryError:pytest.fail('numerical retry swallowed internal allowance')


@pytest.mark.parametrize('phase',['before-retirement','retired'])
@pytest.mark.parametrize('original',['allowance','decline','callback'])
def test_retirement_accounting_failure_is_sticky_and_blocks_replay(monkeypatch,tmp_path,phase,original):
    work,plan,policy,clock,events,seen,unused=_fixture(monkeypatch,tmp_path)
    failure=executor.PlannerMemoryExceeded('retirement accounting refused')
    callback_error=RuntimeError('original callback failure')
    armed=[False];observations=[];closed=[];serial_calls=[]
    class FakeOwned:
        def __init__(self):
            self.tree=SimpleNamespace(memory_snapshot=lambda:{},_memory_observation={})
        def close(self):closed.append(True)
    def observe(name,*args):
        observations.append(name)
        if name==phase:raise failure
    def callback():
        if armed[0]:raise callback_error
        return False
    if original=='callback':policy.cancellation_check=callback
    def parallel(bridge):
        seen.append(bridge);bridge.session=FakeOwned();bridge.retired=False
        bridge.control.lease.observe_phase=observe
        if original=='callback':
            armed[0]=True;bridge.poll()
            pytest.fail('callback did not fail')
        if original=='allowance':raise controller._ProductionAllowanceExpired('eligible expiry')
        raise controller.QualificationDecline('eligible decline')
    def serial():serial_calls.append(True);return plan
    expected=callback_error if original=='callback' else failure
    with collect_runtime_records() as records,pytest.raises(type(expected)) as caught:
        host.qualified_plan(None,None,serial,parallel,policy=policy,mode='exact')
    assert caught.value is expected
    assert serial_calls==[] and closed==[True]
    assert seen[0].retired is True and seen[0].session is None
    assert seen[0]._retirement_error is failure
    assert records['parallel_planning'][-1]['disposition']=='terminal'
    observed=list(observations)
    with pytest.raises(executor.PlannerMemoryExceeded) as repeated:
        seen[0].close()
    assert repeated.value is failure
    assert closed==[True] and observations==observed
