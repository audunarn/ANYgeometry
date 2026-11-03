"""Replay units: fake clocks and injected kernels isolate package admission.

These do not qualify mutated kernels. Actual unmodified-kernel counterparts are
test_planning_cancellation_process, test_planning_session_cancellation_process,
and the public canary in test_planning_r8_integration; source/native mutation
and retirement guards also remain covered by their dedicated test files.
"""
import pytest
from anygeometry import _planning_host as host
from anygeometry import _planning_executor as executor
from anygeometry import _planning_session as session
from anygeometry import batch_intersections as bi
from anygeometry._planning_canaries import _fixture
from anygeometry._planning_qualification import QualificationDecline
from anygeometry.cancellation import CancellationSignal
from anygeometry.serialization import from_dict,to_dict


def setup(tmp_path,monkeypatch,*,reuse=False):
    original,_=_fixture();model=from_dict(to_dict(original));clock=[0.]
    # Deliberate fault-injection units must reach the replay lifecycle rather
    # than decline earlier because the test replaced a protected kernel body.
    monkeypatch.setattr(host,'_admit_stage_package',lambda *a,**k:lambda:None)
    monkeypatch.setattr(host.time,'monotonic',lambda:clock[0])
    signal=CancellationSignal();policy=bi.IntersectionBatchPolicy(cancellation_check=signal)
    control=host.StageControl(tmp_path/'owned',expected_ci=('p','a'),observed_ci=lambda _:pytest.fail('refusal ran CI'),timeout=10,reuse_workers=reuse,overall_deadline=10.)
    return model,clock,signal,policy,control


@pytest.mark.parametrize('fault',['deadline','cancel','source'])
def test_late_serial_return_is_not_published(tmp_path,monkeypatch,fault):
    model,clock,signal,policy,control=setup(tmp_path,monkeypatch)
    def decline(_):raise QualificationDecline('worker refused')
    def serial():
        if fault=='deadline':clock[0]=11.
        elif fault=='cancel':signal.cancel()
        else:model._revision+=1
        return 'late result'
    error={'deadline':executor.PlannerDeadline,'cancel':executor.PlannerCancelled,'source':executor.PlannerSourceChanged}[fault]
    with pytest.raises(error):host.qualified_plan(model,control,serial,decline,policy=policy,mode='exact')
    assert control.mode=='terminal' and not control.ci_verified


@pytest.mark.parametrize('reuse',[False,True])
@pytest.mark.parametrize('fault',['deadline','cancel'])
def test_public_fallback_polls_inside_original_kernel_after_cleanup(tmp_path,monkeypatch,reuse,fault):
    model,clock,signal,policy,control=setup(tmp_path,monkeypatch,reuse=reuse);events=[]
    class Refuse:
        def __init__(self,*a,**k):events.append('start')
        def submit_stage(self,*a,**k):raise QualificationDecline('owned refusal')
        def prepare_plan(self,*a,**k):raise QualificationDecline('owned refusal')
        def close(self):events.append('retired')
    monkeypatch.setattr(host,'OwnedPlanner',Refuse);monkeypatch.setattr(session,'OwnedSession',Refuse)
    original=bi._pair_paths
    def pair(*a,**k):
        assert events==['start','retired'];events.append('serial pair')
        if fault=='deadline':clock[0]=11.
        else:signal.cancel()
        return original(*a,**k)
    monkeypatch.setattr(bi,'_pair_paths',pair);monkeypatch.setenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING','1')
    with pytest.raises(executor.PlannerDeadline if fault=='deadline' else executor.PlannerCancelled):
        bi.plan_intersections(model,tuple(model.faces),policy=policy,_parallel_control=control)
    assert events==['start','retired','serial pair'] and control.mode=='terminal'


@pytest.mark.parametrize('error',[bi.GeometryError('original bridge attribution'),OSError('original caller'),QualificationDecline('original kernel')])
def test_original_serial_error_identity_survives(tmp_path,monkeypatch,error):
    model,clock,signal,policy,control=setup(tmp_path,monkeypatch)
    def decline(_):raise QualificationDecline('owned refusal')
    def serial():raise error
    with pytest.raises(type(error)) as caught:host.qualified_plan(model,control,serial,decline,policy=policy,mode='exact')
    assert caught.value is error and control.mode=='serial'


def test_public_unexpired_fallback_preserves_plan_and_apply(tmp_path,monkeypatch):
    model,clock,signal,policy,control=setup(tmp_path,monkeypatch);doc=to_dict(model);events=[]
    monkeypatch.delenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING',raising=False)
    original=bi.plan_intersections(model,tuple(model.faces),policy=policy)
    bi.apply_intersections(model,original,policy=policy);expected=to_dict(model)
    model=from_dict(doc)
    class Refuse:
        def __init__(self,*a,**k):pass
        def submit_stage(self,*a,**k):raise QualificationDecline('refusal')
        def close(self):events.append('retired')
    monkeypatch.setattr(host,'OwnedPlanner',Refuse);monkeypatch.setattr(session,'OwnedSession',Refuse)
    monkeypatch.setenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING','1')
    actual=bi.plan_intersections(model,tuple(model.faces),policy=policy,_parallel_control=control)
    assert actual.content_checksum==original.content_checksum and to_dict(model)==doc
    bi.apply_intersections(model,actual,policy=policy)
    assert to_dict(model)==expected and control.mode=='serial' and events==['retired']

@pytest.mark.parametrize('reason',['callback','predicate_budget','pair_budget'])
def test_unadmitted_policy_keeps_original_serial_route(tmp_path,monkeypatch,reason):
    model,clock,signal,policy,control=setup(tmp_path,monkeypatch)
    kwargs={'cancellation_check':lambda:pytest.fail('arbitrary callback polled')} if reason=='callback' else {('max_predicates' if reason=='predicate_budget' else 'max_candidate_pairs'):1}
    policy=bi.IntersectionBatchPolicy(**kwargs)
    assert host.qualified_plan(model,control,lambda:'original',lambda _:pytest.fail('delegated'),policy=policy,mode='exact',serial_replay=lambda _:pytest.fail('unsupported replay adapter'))=='original'
    assert control.mode=='serial'


@pytest.mark.parametrize('error',[OSError('signal read'),QualificationDecline('signal read'),KeyboardInterrupt('signal read')])
def test_public_replay_poll_error_keeps_identity(tmp_path,monkeypatch,error):
    from anygeometry.cancellation import _event
    model,clock,signal,policy,control=setup(tmp_path,monkeypatch);events=[]
    class Refuse:
        def __init__(self,*a,**k):pass
        def submit_stage(self,*a,**k):raise QualificationDecline('refusal')
        def close(self):events.append('retired')
    original=bi._pair_paths
    def pair(*a,**k):
        assert events==['retired']
        def fail():raise error
        _event(signal).is_set=fail
        return original(*a,**k)
    monkeypatch.setattr(host,'OwnedPlanner',Refuse);monkeypatch.setattr(session,'OwnedSession',Refuse)
    monkeypatch.setattr(bi,'_pair_paths',pair)
    monkeypatch.setenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING','1')
    with pytest.raises(type(error)) as caught:bi.plan_intersections(model,tuple(model.faces),policy=policy,_parallel_control=control)
    assert caught.value is error and control.mode=='terminal'


@pytest.mark.parametrize('error',[bi.GeometryError('original support error'),OSError('original support error')])
def test_public_serial_error_wrapper_is_identical(tmp_path,monkeypatch,error):
    model,clock,signal,policy,control=setup(tmp_path,monkeypatch)
    def fail(*a,**k):raise error
    monkeypatch.setattr(bi,'_pair_paths',fail)
    monkeypatch.delenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING',raising=False)
    with pytest.raises(type(error)) as original:bi.plan_intersections(model,tuple(model.faces),policy=policy)
    class Refuse:
        def __init__(self,*a,**k):pass
        def submit_stage(self,*a,**k):raise QualificationDecline('refusal')
        def close(self):pass
    monkeypatch.setattr(host,'OwnedPlanner',Refuse);monkeypatch.setattr(session,'OwnedSession',Refuse)
    monkeypatch.setenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING','1')
    with pytest.raises(type(error)) as replay:bi.plan_intersections(model,tuple(model.faces),policy=policy,_parallel_control=control)
    assert str(replay.value)==str(original.value) and replay.value.__cause__ is original.value.__cause__
    assert control.mode=='serial'


def test_exact_deadline_boundary_refuses_final_return(tmp_path,monkeypatch):
    model,clock,signal,policy,control=setup(tmp_path,monkeypatch)
    def decline(_):raise QualificationDecline('refused')
    def serial():clock[0]=10.;return 'late'
    with pytest.raises(executor.PlannerDeadline):host.qualified_plan(model,control,serial,decline,policy=policy,mode='exact')
    assert control.mode=='terminal'
