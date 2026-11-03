"""Absolute caller budgets outlive the bounded parallel attempt; no real sleeps."""
import pytest
from anygeometry import _planning_host as host, _planning_executor as executor
from anygeometry import batch_intersections as bi
from anygeometry._planning_canaries import _fixture
from anygeometry._planning_qualification import QualificationDecline
from anygeometry.cancellation import CancellationSignal, _event
from anygeometry.serialization import from_dict,to_dict


def setup(tmp_path,monkeypatch,*,overall=500.,timeout=10.):
    original,_=_fixture();model=from_dict(to_dict(original));clock=[0.]
    monkeypatch.setattr(host.time,'monotonic',lambda:clock[0])
    signal=CancellationSignal();policy=bi.IntersectionBatchPolicy(cancellation_check=signal)
    control=host.StageControl(tmp_path/'owned',expected_ci=('p','a'),observed_ci=lambda _:('p','a'),timeout=timeout)
    # Set the intended field directly so baseline failures exercise behavior,
    # independently of the new constructor keyword's separate regression.
    control.overall_deadline=overall
    return model,clock,signal,policy,control


def test_control_accepts_absolute_caller_deadline(tmp_path):
    control=host.StageControl(tmp_path/'owned',expected_ci=('p','a'),observed_ci=lambda _:('p','a'),overall_deadline=500.)
    assert control.overall_deadline==500.


@pytest.mark.parametrize('deadline',[True,float('nan')])
def test_constructor_rejects_invalid_explicit_deadline(tmp_path,deadline):
    with pytest.raises(bi.GeometryError):
        host.StageControl(tmp_path/'owned',expected_ci=('p','a'),observed_ci=lambda _:('p','a'),overall_deadline=deadline)


@pytest.mark.parametrize('overall',[0.,-1.])
def test_exhausted_caller_budget_never_launches_or_replays(tmp_path,monkeypatch,overall):
    model,clock,signal,policy,control=setup(tmp_path,monkeypatch,overall=overall)
    with pytest.raises(executor.PlannerDeadline):
        host.qualified_plan(model,control,lambda:pytest.fail('replayed'),lambda _:pytest.fail('launched'),policy=policy,mode='exact')
    assert control.mode=='terminal'


@pytest.mark.parametrize('now',[110.,111.])
def test_parallel_clock_remains_bounded_with_longer_caller_budget(tmp_path,monkeypatch,now):
    model,clock,signal,policy,control=setup(tmp_path,monkeypatch,timeout=110.)
    def parallel(bridge):clock[0]=now;bridge.check_deadline();return 'late'
    with pytest.raises(executor.PlannerDeadline):host.qualified_plan(model,control,lambda:pytest.fail('replayed'),parallel,policy=policy,mode='exact')
    assert control.mode=='terminal' and not control.ci_verified


@pytest.mark.parametrize('deadline',[True,'500',float('nan'),float('inf'),float('-inf'),object()])
def test_invalid_explicit_budget_is_terminal_not_serial(tmp_path,monkeypatch,deadline):
    model,clock,signal,policy,control=setup(tmp_path,monkeypatch,overall=deadline)
    with pytest.raises(bi.GeometryError):
        host.qualified_plan(model,control,lambda:pytest.fail('invalid deadline replayed'),lambda _:pytest.fail('launched'),policy=policy,mode='exact')
    assert control.mode=='terminal'


def test_deadline_cannot_be_renewed_by_admission(tmp_path,monkeypatch):
    model,clock,signal,policy,control=setup(tmp_path,monkeypatch,overall=5.)
    def admission(*args):control.overall_deadline=500.;return True
    control.admission=admission
    def parallel(bridge):clock[0]=5.;bridge.check_deadline();return 'late'
    with pytest.raises(executor.PlannerDeadline):host.qualified_plan(model,control,lambda:pytest.fail('replayed'),parallel,policy=policy,mode='exact')
    assert control.mode=='terminal'


def test_unsupported_policy_stays_original_serial_but_honors_explicit_budget(tmp_path,monkeypatch):
    model,clock,signal,policy,control=setup(tmp_path,monkeypatch,overall=5.)
    policy=bi.IntersectionBatchPolicy(max_predicates=1)
    def serial():clock[0]=5.;return 'late'
    with pytest.raises(executor.PlannerDeadline):
        host.qualified_plan(model,control,serial,lambda _:pytest.fail('launched'),policy=policy,mode='exact',serial_replay=lambda _:pytest.fail('unsupported adapter'))
    assert control.mode=='terminal'


@pytest.mark.parametrize('overall',[None,500.])
def test_replay_can_run_beyond_110_with_remaining_caller_budget(tmp_path,monkeypatch,overall):
    model,clock,signal,policy,control=setup(tmp_path,monkeypatch,overall=overall)
    def decline(_):raise QualificationDecline('refused')
    def replay(poll):clock[0]=150.;poll();return 'original'
    assert host.qualified_plan(model,control,lambda:pytest.fail('wrong replay'),decline,policy=policy,mode='exact',serial_replay=replay)=='original'
    assert control.mode=='serial' and not control.ci_verified


def test_elapsed_parallel_clock_does_not_reclassify_decline(tmp_path,monkeypatch):
    model,clock,signal,policy,control=setup(tmp_path,monkeypatch)
    def decline(_):clock[0]=11.;raise QualificationDecline('already refused')
    assert host.qualified_plan(model,control,lambda:'original',decline,policy=policy,mode='exact')=='original'
    assert control.mode=='serial'


@pytest.mark.parametrize('now',[5.,6.])
def test_overall_expiry_is_enforced_during_parallel_poll(tmp_path,monkeypatch,now):
    model,clock,signal,policy,control=setup(tmp_path,monkeypatch,overall=5.)
    def parallel(bridge):clock[0]=now;bridge.check_deadline();return 'candidate'
    with pytest.raises(executor.PlannerDeadline):host.qualified_plan(model,control,lambda:pytest.fail('replayed'),parallel,policy=policy,mode='exact')
    assert control.mode=='terminal' and not control.ci_verified


@pytest.mark.parametrize('at_boundary',[True,False])
def test_replay_final_serialization_cannot_outlive_overall_budget(tmp_path,monkeypatch,at_boundary):
    model,clock,signal,policy,control=setup(tmp_path,monkeypatch,overall=5.)
    finished=[False];original=host.to_dict
    def serialize(value):
        result=original(value)
        if finished[0]:clock[0]=5. if at_boundary else 6.
        return result
    monkeypatch.setattr(host,'to_dict',serialize)
    def decline(_):raise QualificationDecline('refused')
    def serial():finished[0]=True;return 'late'
    with pytest.raises(executor.PlannerDeadline):host.qualified_plan(model,control,serial,decline,policy=policy,mode='exact')
    assert control.mode=='terminal'


@pytest.mark.parametrize('reuse',[False,True])
def test_decline_retires_before_long_replay_without_reusing_parallel_clock(tmp_path,monkeypatch,reuse):
    from anygeometry import _planning_session as session
    model,clock,signal,policy,control=setup(tmp_path,monkeypatch);events=[]
    control.reuse_workers=reuse
    class Refuse:
        def __init__(self,*args,**kwargs):events.append('start')
        def submit_stage(self,*args,**kwargs):raise QualificationDecline('owned refusal')
        def close(self):events.append('retired');clock[0]=11.
    monkeypatch.setattr(host,'OwnedPlanner',Refuse);monkeypatch.setattr(session,'OwnedSession',Refuse)
    def parallel(bridge):return bridge.run('pair',[dict(kind='pair',cost=1.)],lambda _:pytest.fail('sampled'))
    def replay(poll):
        assert events==['start','retired'];events.append('replay')
        clock[0]=150.;poll();return 'original'
    assert host.qualified_plan(model,control,lambda:pytest.fail('wrong replay'),parallel,policy=policy,mode='exact',serial_replay=replay)=='original'
    assert events==['start','retired','replay'] and control.mode=='serial'


def test_callback_consuming_overall_budget_is_terminal(tmp_path,monkeypatch):
    model,clock,signal,policy,control=setup(tmp_path,monkeypatch,overall=5.)
    def parallel(bridge):
        def is_set():clock[0]=5.;return False
        monkeypatch.setattr(_event(signal),'is_set',is_set)
        bridge.check_deadline();return 'late'
    with pytest.raises(executor.PlannerDeadline):host.qualified_plan(model,control,lambda:pytest.fail('replayed'),parallel,policy=policy,mode='exact')
    assert control.mode=='terminal'


@pytest.mark.parametrize('path',['serial','parallel'])
def test_no_publication_after_post_snapshot_poll_mutates_source(tmp_path,monkeypatch,path):
    model,clock,signal,policy,control=setup(tmp_path,monkeypatch)
    finished=[False];serialized=[False];original=host.to_dict;revision=model._revision
    def serialize(value):
        result=original(value)
        if finished[0]:serialized[0]=True
        return result
    def is_set():
        if serialized[0]:model._revision+=1
        return False
    monkeypatch.setattr(host,'to_dict',serialize);monkeypatch.setattr(_event(signal),'is_set',is_set)
    def parallel(_):
        if path=='serial':raise QualificationDecline('refused')
        finished[0]=True;return 'result'
    def serial():finished[0]=True;return 'result'
    try:result=host.qualified_plan(model,control,serial,parallel,policy=policy,mode='exact')
    except executor.PlannerSourceChanged:
        assert control.mode=='terminal'
    else:
        assert result=='result' and model._revision==revision, 'published after an unchecked final callback mutation'


@pytest.mark.parametrize('path',['serial','parallel'])
def test_final_serialization_memory_pressure_stays_terminal(tmp_path,monkeypatch,path):
    from anygeometry import _planning_memory as memory
    model,clock,signal,policy,control=setup(tmp_path,monkeypatch)
    observed=dict(host_peak_bytes=100,host_private_bytes=90,available_bytes=2000)
    monkeypatch.setattr(memory,'observe_memory',lambda:observed)
    control.memory_budget=memory.MemoryBudget(100,400,1000,(1,2))
    finished=[False];original=host.to_dict
    def serialize(value):
        result=original(value)
        if finished[0]:observed['host_peak_bytes']=101
        return result
    monkeypatch.setattr(host,'to_dict',serialize)
    def parallel(_):
        if path=='serial':raise QualificationDecline('refused')
        finished[0]=True;return 'late'
    def serial():finished[0]=True;return 'late'
    with pytest.raises(executor.PlannerMemoryExceeded):host.qualified_plan(model,control,serial,parallel,policy=policy,mode='exact')
    assert control.mode=='terminal' and not control.ci_verified
    assert control.memory_report['terminal_reason']


def test_explicit_parallel_deadline_remains_terminal_with_caller_time_left(tmp_path,monkeypatch):
    model,clock,signal,policy,control=setup(tmp_path,monkeypatch)
    error=executor.PlannerDeadline('actual parallel timeout')
    def parallel(_):raise error
    with pytest.raises(executor.PlannerDeadline) as caught:host.qualified_plan(model,control,lambda:pytest.fail('replayed'),parallel,policy=policy,mode='exact')
    assert caught.value is error and control.mode=='terminal'
