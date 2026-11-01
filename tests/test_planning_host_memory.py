"""Exact admission arithmetic and connected host pressure/refusal behavior."""
import importlib
from types import SimpleNamespace
import pytest
from anygeometry import _planning_host as host, _planning_executor as executor
from anygeometry._planning_qualification import QualificationDecline
from anygeometry._planning_canaries import _fixture
from anygeometry.serialization import to_dict


def api():return importlib.import_module('anygeometry._planning_memory')
def budget(**changes):
    values=dict(host_peak_estimate_bytes=100,tree_limit_bytes=400,total_limit_bytes=1000,free_fraction=(1,2))
    values.update(changes);return api().MemoryBudget(**values)
def sample(peak=100,available=2000):return dict(host_peak_bytes=peak,host_private_bytes=min(90,peak),available_bytes=available)


@pytest.mark.parametrize('total,available,accept',[(5,10,True),(4,10,False),(5,9,False),(5,11,True)])
def test_exact_headroom_and_available_ram_boundary(monkeypatch,total,available,accept):
    m=api();monkeypatch.setattr(m,'observe_memory',lambda:sample(1,available))
    b=budget(host_peak_estimate_bytes=1,tree_limit_bytes=3,total_limit_bytes=total)
    if not accept:
        with pytest.raises(QualificationDecline):b.begin()
    else:
        lease=b.begin();report=lease.report()
        assert report['required_with_headroom_bytes']==5 and report['tree_limit_bytes']==3
        assert report['effective_limit_bytes']==5


@pytest.mark.parametrize('field,value',[
 ('host_peak_estimate_bytes',None),('host_peak_estimate_bytes',False),('host_peak_estimate_bytes',0),
 ('tree_limit_bytes',-1),('tree_limit_bytes',1.0),('total_limit_bytes','1000'),
 ('free_fraction',(0,2)),('free_fraction',(3,2)),('free_fraction',(1,0)),
 ('free_fraction',(True,2)),('free_fraction',[1,2]),('total_limit_bytes',2**64),
])
def test_unknown_or_untyped_budget_is_refused(field,value):
    with pytest.raises(QualificationDecline):budget(**{field:value})


@pytest.mark.parametrize('bad',[None,{},sample(0),dict(sample(),host_private_bytes=101),dict(sample(),host_peak_bytes=True)])
def test_unknown_or_inconsistent_native_accounting_refuses(monkeypatch,bad):
    m=api();monkeypatch.setattr(m,'observe_memory',lambda:bad)
    with pytest.raises(QualificationDecline):budget().begin()


def test_budget_is_frozen_for_an_admitted_lease(monkeypatch):
    m=api();value=sample();monkeypatch.setattr(m,'observe_memory',lambda:value)
    b=budget();lease=b.begin();object.__setattr__(b,'host_peak_estimate_bytes',1000)
    value['host_peak_bytes']=101
    with pytest.raises(executor.PlannerMemoryExceeded):lease.check()
    r=lease.report();assert r['host_peak_estimate_bytes']==100 and r['observed_host_peak_bytes']==101 and r['terminal_reason']


def test_post_admission_query_error_is_terminal(monkeypatch):
    m=api();monkeypatch.setattr(m,'observe_memory',lambda:sample());lease=budget().begin()
    def fail():raise OSError('lost native accounting')
    monkeypatch.setattr(m,'observe_memory',fail)
    with pytest.raises(executor.PlannerMemoryExceeded):lease.check()
    assert lease.report()['terminal_reason']


def test_lifetime_peak_cannot_decrease(monkeypatch):
    m=api();value=sample();monkeypatch.setattr(m,'observe_memory',lambda:value);lease=budget().begin()
    value['host_peak_bytes']=99
    with pytest.raises(executor.PlannerMemoryExceeded):lease.check()


def call(model,control,serial,parallel,policy=None):
    if policy is None:policy=SimpleNamespace(cancellation_check=None,max_predicates=None,max_candidate_pairs=None)
    return host.qualified_plan(model,control,serial,parallel,policy=policy,mode='exact')


def test_refusal_precedes_snapshot_and_spawn(tmp_path,monkeypatch):
    m=api();monkeypatch.setattr(m,'observe_memory',lambda:sample());model,_=_fixture();calls=[]
    control=host.StageControl(tmp_path,expected_ci=('p','a'),observed_ci=lambda _:('p','a'),memory_budget=budget(total_limit_bytes=500))
    monkeypatch.setattr(host,'freeze_model',lambda *a,**k:pytest.fail('refused budget allocated snapshot'))
    assert call(model,control,lambda:calls.append('serial') or 'serial',lambda _:pytest.fail('spawn'))=='serial'
    assert calls==['serial'] and control.mode=='serial' and control.decline_reason


def test_accepted_budget_supplies_exact_tree_cap(tmp_path,monkeypatch):
    m=api();monkeypatch.setattr(m,'observe_memory',lambda:sample());model,_=_fixture();before=to_dict(model);seen=[]
    control=host.StageControl(tmp_path,expected_ci=('p','a'),observed_ci=lambda _:('p','a'),memory_budget=budget())
    def parallel(bridge):seen.append(bridge.tree_options);return 'candidate'
    assert call(model,control,lambda:pytest.fail('fallback'),parallel)=='candidate'
    assert seen==[{'tree_memory_limit':400}] and to_dict(model)==before
    assert control.mode=='parallel' and control.memory_report['outcome']=='parallel' and control.memory_report['observed_host_peak_bytes']==100


@pytest.mark.parametrize('where',['snapshot','candidate','ci'])
def test_pressure_is_terminal_without_serial_replay(tmp_path,monkeypatch,where):
    m=api();value=sample();monkeypatch.setattr(m,'observe_memory',lambda:value);model,_=_fixture();events=[]
    freeze=host.freeze_model
    def frozen(*a,**k):
        result=freeze(*a,**k)
        if where=='snapshot':value['host_peak_bytes']=101
        return result
    monkeypatch.setattr(host,'freeze_model',frozen)
    def ci(_):
        events.append('ci')
        if where=='ci':value['host_peak_bytes']=101
        return ('p','a')
    control=host.StageControl(tmp_path,expected_ci=('p','a'),observed_ci=ci,memory_budget=budget())
    def parallel(bridge):
        events.append('parallel')
        if where=='candidate':value['host_peak_bytes']=101
        return 'candidate'
    with pytest.raises(executor.PlannerMemoryExceeded):call(model,control,lambda:pytest.fail('memory pressure replayed serial'),parallel)
    assert control.mode=='terminal' and not control.ci_verified and control.memory_report['terminal_reason']
    if where=='snapshot':assert events==[]


def test_cancellation_wins_over_simultaneous_host_pressure(tmp_path,monkeypatch):
    from anygeometry.cancellation import CancellationSignal
    m=api();value=sample();monkeypatch.setattr(m,'observe_memory',lambda:value);model,_=_fixture();signal=CancellationSignal()
    control=host.StageControl(tmp_path,expected_ci=('p','a'),observed_ci=lambda _:('p','a'),memory_budget=budget())
    policy=SimpleNamespace(cancellation_check=signal,max_predicates=None,max_candidate_pairs=None)
    def parallel(bridge):value['host_peak_bytes']=101;signal.cancel();return 'candidate'
    with pytest.raises(executor.PlannerCancelled):call(model,control,lambda:pytest.fail('cancel replay'),parallel,policy)


def test_tree_cap_conflict_refuses_before_snapshot(tmp_path,monkeypatch):
    m=api();monkeypatch.setattr(m,'observe_memory',lambda:sample());model,_=_fixture()
    control=host.StageControl(tmp_path,expected_ci=('p','a'),observed_ci=lambda _:('p','a'),tree_memory_limit=500,memory_budget=budget())
    monkeypatch.setattr(host,'freeze_model',lambda *a,**k:pytest.fail('cap conflict allocated snapshot'))
    assert call(model,control,lambda:'serial',lambda _:pytest.fail('spawn'))=='serial'
    assert control.mode=='serial'


def test_snapshot_refusal_with_admitted_budget_keeps_serial_result(tmp_path,monkeypatch):
    m=api();monkeypatch.setattr(m,'observe_memory',lambda:sample());model,_=_fixture();events=[]
    control=host.StageControl(tmp_path,expected_ci=('p','a'),observed_ci=lambda _:('p','a'),memory_budget=budget())
    def refuse(*a,**k):raise QualificationDecline('snapshot refused')
    monkeypatch.setattr(host,'freeze_model',refuse)
    def replay(poll):poll();events.append('serial');return 'serial'
    policy=SimpleNamespace(cancellation_check=None,max_predicates=None,max_candidate_pairs=None)
    result=host.qualified_plan(model,control,lambda:pytest.fail('unpolled replay'),lambda _:pytest.fail('spawn'),
                               policy=policy,mode='exact',serial_replay=replay)
    assert result=='serial' and events==['serial'] and control.memory_report['outcome']=='serial'


@pytest.mark.parametrize('before_bridge',[False,True])
def test_memory_pressure_during_serial_replay_is_terminal(tmp_path,monkeypatch,before_bridge):
    m=api();value=sample();monkeypatch.setattr(m,'observe_memory',lambda:value);model,_=_fixture();events=[]
    control=host.StageControl(tmp_path,expected_ci=('p','a'),observed_ci=lambda _:('p','a'),memory_budget=budget())
    def refuse(*a,**k):raise QualificationDecline('deliberate refusal')
    if before_bridge:monkeypatch.setattr(host,'freeze_model',refuse)
    def replay(poll):
        value['host_peak_bytes']=101;events.append('serial-start');poll();pytest.fail('pressure ignored')
    policy=SimpleNamespace(cancellation_check=None,max_predicates=None,max_candidate_pairs=None)
    with pytest.raises(executor.PlannerMemoryExceeded):
        host.qualified_plan(model,control,lambda:pytest.fail('unpolled replay'),refuse,
                            policy=policy,mode='exact',serial_replay=replay)
    assert control.mode=='terminal' and not control.ci_verified and events==['serial-start']


def test_pressure_during_final_serialization_is_terminal(tmp_path,monkeypatch):
    m=api();value=sample();monkeypatch.setattr(m,'observe_memory',lambda:value);model,_=_fixture();count=0
    original=host.to_dict
    def serialize(*a,**k):
        nonlocal count
        count+=1;result=original(*a,**k)
        if count==3:value['host_peak_bytes']=101
        return result
    monkeypatch.setattr(host,'to_dict',serialize)
    control=host.StageControl(tmp_path,expected_ci=('p','a'),observed_ci=lambda _:('p','a'),memory_budget=budget())
    with pytest.raises(executor.PlannerMemoryExceeded):call(model,control,lambda:pytest.fail('replay'),lambda _:'candidate')
    assert count==3 and control.mode=='terminal' and not control.ci_verified


def test_reused_session_does_not_convert_memory_error_to_refusal(tmp_path,monkeypatch):
    m=api();value=sample();monkeypatch.setattr(m,'observe_memory',lambda:value);model,_=_fixture()
    control=host.StageControl(tmp_path,expected_ci=('p','a'),observed_ci=lambda _:('p','a'),memory_budget=budget(),reuse_workers=True)
    def parallel(bridge):value['host_peak_bytes']=101;bridge.check_deadline()
    with pytest.raises(executor.PlannerMemoryExceeded):call(model,control,lambda:pytest.fail('replay'),parallel)
    assert control.mode=='terminal' and not control.ci_verified


def test_tree_cap_stays_frozen_after_admission(tmp_path,monkeypatch):
    m=api();monkeypatch.setattr(m,'observe_memory',lambda:sample());model,_=_fixture();b=budget()
    control=host.StageControl(tmp_path,expected_ci=('p','a'),observed_ci=lambda _:('p','a'),memory_budget=b)
    def parallel(bridge):
        control.tree_memory_limit=999;object.__setattr__(b,'tree_limit_bytes',999)
        assert bridge.tree_options=={'tree_memory_limit':400} and control.tree_options()=={'tree_memory_limit':400}
        return 'candidate'
    assert call(model,control,lambda:pytest.fail('replay'),parallel)=='candidate'


def test_memory_failure_is_sticky_and_report_cannot_reset_it(monkeypatch):
    m=api();value=sample();monkeypatch.setattr(m,'observe_memory',lambda:value);lease=budget().begin()
    value['host_peak_bytes']=101
    with pytest.raises(executor.PlannerMemoryExceeded):lease.check()
    lease.report()['terminal_reason']=None;value['host_peak_bytes']=100
    with pytest.raises(executor.PlannerMemoryExceeded):lease.check()


def test_native_observation_exception_before_admission_refuses(monkeypatch):
    m=api()
    def fail():raise OSError('unknown accounting')
    monkeypatch.setattr(m,'observe_memory',fail)
    with pytest.raises(QualificationDecline):budget().begin()


def test_peak_already_above_estimate_refuses_before_snapshot(tmp_path,monkeypatch):
    m=api();monkeypatch.setattr(m,'observe_memory',lambda:sample(101));model,_=_fixture()
    control=host.StageControl(tmp_path,expected_ci=('p','a'),observed_ci=lambda _:('p','a'),memory_budget=budget())
    monkeypatch.setattr(host,'freeze_model',lambda *a,**k:pytest.fail('over-budget snapshot'))
    assert call(model,control,lambda:'serial',lambda _:pytest.fail('spawn'))=='serial'
    assert control._memory_lease is None and control.mode=='serial'


@pytest.mark.parametrize('reuse',[False,True])
def test_actual_kernel_ring_split_cannot_intercept_memory_terminal(tmp_path,monkeypatch,reuse):
    from anygeometry import batch_intersections as bi, material_arrangement as ma
    m=api();value=sample();monkeypatch.setattr(m,'observe_memory',lambda:value);model,_=_fixture();errors=[];events=[]
    original_check=m.MemoryLease.check
    def check(self):
        try:return original_check(self)
        except executor.PlannerMemoryExceeded as error:errors.append(error);raise
    monkeypatch.setattr(m.MemoryLease,'check',check)
    control=host.StageControl(tmp_path,expected_ci=('p','a'),observed_ci=lambda _:pytest.fail('CI after pressure'),
                              memory_budget=budget(),reuse_workers=reuse)
    # Only worker execution is replaced. Actual kernel constructs tasks and the
    # actual face sampler calls arrange_material/_ring_split_attempt below.
    def run(self,kind,tasks,serial_task):
        events.append(kind)
        return tuple(serial_task(task) for task in tasks)
    monkeypatch.setattr(host.Bridge,'run',run);monkeypatch.setattr(host._SessionBridge,'run',run)
    def once(*a,cancellation_check,**k):
        events.append('inside-ring-split');value['host_peak_bytes']=101;cancellation_check()
        pytest.fail('terminal memory poll returned')
    monkeypatch.setattr(ma,'_arrange_material_once',once)
    policy=bi.IntersectionBatchPolicy()
    def parallel(bridge):return bi._plan_intersections(model,tuple(model.faces),policy=policy,_planning_mode='exact',_seam_policy='ring_split',_stages=bridge)
    with pytest.raises(executor.PlannerMemoryExceeded) as caught:
        host.qualified_plan(model,control,lambda:pytest.fail('serial fallback'),parallel,policy=policy,mode='exact')
    assert events==['pair','face','inside-ring-split']
    assert len(errors)==1 and caught.value is errors[0]
    assert control.mode=='terminal' and control.decline_reason is None and control.memory_report['terminal_reason']


@pytest.mark.parametrize('phase',['prepare','sample-pair'])
def test_preparation_and_transcript_memory_failures_retire_without_replay(tmp_path,monkeypatch,phase):
    from test_planning_preparation_host import fixture
    invoke,control,events=fixture(tmp_path,monkeypatch);m=api();control.memory_budget=budget()
    monkeypatch.setattr(m,'observe_memory',lambda:sample(101 if phase in events else 100))
    with pytest.raises(executor.PlannerMemoryExceeded):invoke()
    assert events==(['assembly','start','prepare','retired'] if phase=='prepare' else
                    ['assembly','start','prepare','retired','assembly','sample-pair'])
    assert control.mode=='terminal' and not control.ci_verified and control.memory_report['terminal_reason']
