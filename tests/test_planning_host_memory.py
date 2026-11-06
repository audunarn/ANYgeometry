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

def _native_memory_case(monkeypatch, record_property):
    """Exercise the real initial refusal when this platform has no Windows ABI."""
    import sys
    m=api()
    if sys.platform=='win32':
        record_property('execution_scope','windows-native-memory-contract')
        return m
    assert m._WIN_DLL_TYPE is None
    assert all(getattr(m,name) is None for name in (
        '_AvailablePrototype','_ResolvePrototype','_ContainingPrototype',
        '_FilenamePrototype','_DirectoryPrototype'))
    def forbidden_kernel():
        pytest.fail('unsupported platform entered native memory kernel')
    monkeypatch.setattr(m,'_memory_kernel',forbidden_kernel)
    for query in (m._available_bytes,m.available_memory_identity):
        with pytest.raises(QualificationDecline,match='^available memory ABI/schema changed$'):
            query()
    record_property('execution_scope','unsupported-native-memory-initial-refusal')
    record_property('native_memory_mutation_qualification',False)
    return None


def _native_memory_profile_test(function):
    """Keep the native ABI refusal ahead of the host image profile on non-Windows."""
    from functools import wraps
    import inspect
    import sys
    from _planning_numpy_profile_contract import numpy_profile_test
    positive=numpy_profile_test(function)
    @wraps(function)
    def invoke(*args,**kwargs):
        request=kwargs['request']
        if sys.platform!='win32':
            record=lambda key,value:request.node.user_properties.append((key,value))
            assert _native_memory_case(kwargs['monkeypatch'],record) is None
            request.node.user_properties.append(('original_test_body_executed',False))
            return
        return positive(*args,**kwargs)
    invoke.__signature__=inspect.signature(positive)
    return invoke


def test_nonwindows_native_memory_contract_checks_before_kernel(monkeypatch):
    # Simulate only the source-declared unavailable initialization metadata;
    # actual _available_bytes/_memory_layout still decide the refusal.
    m=api();properties=[]
    with monkeypatch.context() as unavailable:
        unavailable.setattr(m.sys,'platform','unsupported-test-platform')
        for name in ('_WIN_DLL_TYPE','_AvailablePrototype','_ResolvePrototype',
                     '_ContainingPrototype','_FilenamePrototype','_DirectoryPrototype'):
            unavailable.setattr(m,name,None)
        assert _native_memory_case(unavailable,lambda *row:properties.append(row)) is None
        assert ('native_memory_mutation_qualification',False) in properties
        unavailable.setattr(m,'_available_bytes',lambda:m._memory_kernel())
        with pytest.raises(pytest.fail.Exception,match='unsupported platform entered native memory kernel'):
            _native_memory_case(unavailable,lambda *row:None)



def test_available_native_schema_export_and_fresh_observations(monkeypatch,record_property):
    import ctypes
    m=_native_memory_case(monkeypatch,record_property)
    if m is None:return
    identity=m.available_memory_identity()
    assert identity['abi']=='MEMORYSTATUSEX-v1' and identity['size']==64
    assert identity['offsets']==[0,4,8,16,24,32,40,48,56]
    assert identity['export']=='GlobalMemoryStatusEx' and identity['image_offset']>=0
    assert identity['image_path'].lower().endswith(('kernel32.dll','kernelbase.dll'))
    assert len(identity['image_sha256'])==64
    values=[2000,0];calls=[]
    def available():calls.append('available');return values.pop(0)
    class Process:
        def __init__(self):calls.append('process')
        def memory_info(self):calls.append('private');return SimpleNamespace(peak_pagefile=100,private=90)
    monkeypatch.setattr(m,'_available_bytes',available)
    monkeypatch.setattr(m.psutil,'Process',Process)
    monkeypatch.setattr(m.psutil,'virtual_memory',lambda:pytest.fail('broad aggregate query'))
    assert m._sample()==(100,90,2000) and m._sample()==(100,90,0)
    assert calls==['process','private','available']*2


@pytest.mark.parametrize('field',['_fields_','class'])
def test_available_native_initial_layout_skew_refuses(monkeypatch,field,record_property):
    m=_native_memory_case(monkeypatch,record_property)
    if m is None:return
    if field=='class':
        monkeypatch.setattr(m,'_MemoryStatus',object)
        with pytest.raises(QualificationDecline):m.available_memory_identity()
    else:
        fields=m._MemoryStatus._fields_;original=fields[:]
        try:
            fields.clear()
            with pytest.raises(QualificationDecline):m.available_memory_identity()
        finally:fields[:]=original


@pytest.mark.parametrize('change',['fields','descriptor','provider','image','signature','library','handle'])
@_native_memory_profile_test
def test_available_native_host_binding_mutation_is_terminal(monkeypatch,change,record_property):
    import ctypes
    from pathlib import Path
    from anygeometry._planning_code import InfrastructurePin
    m=_native_memory_case(monkeypatch,record_property)
    if m is None:return
    captured=InfrastructurePin('host');captured.verify()
    original_fields=m._MemoryStatus._fields_[:]
    try:
        if change=='fields':m._MemoryStatus._fields_.reverse()
        elif change=='descriptor':monkeypatch.setattr(m._MemoryStatus,'ullAvailPhys',None)
        elif change=='provider':
            monkeypatch.setattr(ctypes.CDLL,'__getattr__',lambda *args:pytest.fail('changed provider invoked'))
        elif change=='signature':monkeypatch.setattr(m._AvailablePrototype,'_argtypes_',())
        elif change=='library':monkeypatch.setattr(ctypes.windll,'kernel32',object())
        elif change=='handle':monkeypatch.setattr(m._memory_kernel(),'_handle',1)
        else:
            target=Path(captured.available_memory['image_path']);original=Path.read_bytes
            monkeypatch.setattr(Path,'read_bytes',lambda path:b'changed native image' if path==target else original(path))
        with pytest.raises(executor.PlannerSourceChanged):captured.verify()
    finally:m._MemoryStatus._fields_[:]=original_fields


@pytest.mark.parametrize('field',['errcheck','argtypes','restype'])
def test_available_native_initial_signature_hook_refuses_without_call(monkeypatch,field,record_property):
    import ctypes
    m=_native_memory_case(monkeypatch,record_property)
    if m is None:return
    called=[]
    if field=='errcheck':monkeypatch.setattr(m._AvailablePrototype,'__call__',lambda *args:called.append('invoked'))
    elif field=='argtypes':monkeypatch.setattr(m._AvailablePrototype,'_argtypes_',(ctypes.c_void_p,))
    else:monkeypatch.setattr(m._AvailablePrototype,'_restype_',ctypes.c_void_p)
    with pytest.raises(QualificationDecline):m._available_bytes()
    assert called==[]


def test_available_native_reuses_only_library_and_signature_types(monkeypatch,record_property):
    import ctypes,weakref,gc
    m=_native_memory_case(monkeypatch,record_property)
    if m is None:return
    m.available_memory_identity();kernel=m._memory_kernel()
    pointers=ctypes.POINTER(m._MemoryStatus);prototype=m._AvailablePrototype;refs=[]
    original=m._memory_binding
    def binding():
        library,query,address=original()
        assert library is kernel and type(query) is prototype
        refs.append(weakref.ref(query));return library,query,address
    monkeypatch.setattr(m,'_memory_binding',binding)
    # CPython 3.13 calls _dlopen; 3.14's Windows loader calls _LoadLibrary.
    # Poison the real factory alias only: do not create an unused ctypes API.
    loader_alias='_LoadLibrary' if hasattr(ctypes,'_LoadLibrary') else '_dlopen'
    assert callable(getattr(ctypes,loader_alias))
    monkeypatch.setattr(ctypes,loader_alias,lambda *args:pytest.fail('extra native load reference'))
    monkeypatch.setattr(ctypes,'WINFUNCTYPE',lambda *args,**kwargs:pytest.fail('new signature type'))
    for _ in range(100):assert type(m._available_bytes()) is int
    # CPython3.14 PyCFuncPtr_FromDll deliberately retains self->callable=self.
    # Prove collectability in this isolated test, never force GC in production.
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert ctypes.POINTER(m._MemoryStatus) is pointers


def test_available_native_does_not_use_or_mutate_cached_export_signature(monkeypatch,record_property):
    import ctypes
    m=_native_memory_case(monkeypatch,record_property)
    if m is None:return
    kernel=m._memory_kernel();cached=kernel.GlobalMemoryStatusEx;called=[]
    monkeypatch.setattr(cached,'argtypes',(ctypes.c_void_p,))
    original=cached.errcheck;cached.errcheck=lambda *args:called.append('cached hook')
    try:
        before=(cached.argtypes,cached.restype,cached.errcheck)
        assert type(m._available_bytes()) is int
        assert (cached.argtypes,cached.restype,cached.errcheck)==before and called==[]
    finally:
        if original is None:del cached.errcheck
        else:cached.errcheck=original


@pytest.mark.parametrize('phase',['initial','admitted'])
def test_available_native_failure_keeps_admission_classification(monkeypatch,phase,request):
    m=api();failure=OSError('original available query failed')
    def observe():
        value=sample();value['available_bytes']=m._available_bytes();return value
    monkeypatch.setattr(m,'observe_memory',observe)
    monkeypatch.setattr(m,'_available_bytes',lambda:2000)
    valid=budget().begin();valid.check()
    lease=valid if phase=='admitted' else None
    request.node.user_properties.extend((('execution_scope','injected-native-observation-component'),
                                       ('native_memory_qualification',False)))
    def fail():raise failure
    monkeypatch.setattr(m,'_available_bytes',fail)
    if lease is None:
        with pytest.raises(QualificationDecline) as caught:budget().begin()
    else:
        with pytest.raises(executor.PlannerMemoryExceeded) as caught:lease.check()
        monkeypatch.setattr(m,'_available_bytes',lambda:2000)
        with pytest.raises(executor.PlannerMemoryExceeded):lease.check()
    assert caught.value.__cause__ is failure


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


def test_phase_memory_records_initial_host_and_whole_job_upper_bound(monkeypatch):
    m=api();value=sample();monkeypatch.setattr(m,'observe_memory',lambda:value)
    lease=budget().begin();value['host_private_bytes']=80
    job=dict(requested_bytes=400,limit_bytes=400,flags=0x2200,peak_job_bytes=250,peak_process_bytes=200)
    lease.observe_phase('pair-returned',job);job['peak_job_bytes']=1
    report=lease.report();phase=report['phases'][-1]
    assert report['initial_host_private_bytes']==90 and report['initial_host_peak_bytes']==100
    assert phase['host_private_bytes']==80
    assert phase['job']['startup_peak_upper_bound_bytes']==250
    assert phase['job']['peak_process_bytes']==200
    assert phase['job']['isolated_startup_peak_measured'] is False
    with pytest.raises(executor.PlannerMemoryExceeded):lease.observe_phase('pair-returned')


def test_phase_memory_cannot_accept_unenforced_job(monkeypatch):
    m=api();monkeypatch.setattr(m,'observe_memory',lambda:sample());lease=budget().begin()
    with pytest.raises(executor.PlannerMemoryExceeded):
        lease.observe_phase('pair-returned',dict(requested_bytes=400,limit_bytes=400,flags=0,peak_job_bytes=250,peak_process_bytes=200))


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
    # This callback checks host pressure without dispatching a worker. Package
    # qualification belongs to the separate real admission/integration tests.
    monkeypatch.setattr(host,'_admit_stage_package',lambda *args:lambda:None)
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
    import sys
    from anygeometry import batch_intersections as bi, material_arrangement as ma
    m=api();value=sample();model,_=_fixture();errors=[];events=[]
    ring_code=ma._ring_split_attempt.__code__
    once_code=ma._arrange_material_once.__code__
    def observe():
        # Inject pressure through the memory observation, not through a replaced
        # numerical function that package admission must correctly refuse.
        frame=sys._getframe(1);inside_once=False;inside_ring=False
        try:
            while frame is not None:
                inside_once |= frame.f_code is once_code
                inside_ring |= frame.f_code is ring_code
                frame=frame.f_back
        finally:del frame
        if inside_once and inside_ring and value['host_peak_bytes']==100:
            events.append('inside-ring-split');value['host_peak_bytes']=101
        return value
    monkeypatch.setattr(m,'observe_memory',observe)
    original_check=m.MemoryLease.check
    def check(self):
        try:return original_check(self)
        except executor.PlannerMemoryExceeded as error:errors.append(error);raise
    monkeypatch.setattr(m.MemoryLease,'check',check)
    control=host.StageControl(tmp_path,expected_ci=('p','a'),observed_ci=lambda _:pytest.fail('CI after pressure'),
                              memory_budget=budget(),reuse_workers=reuse)
    # Worker execution and its package-admission prerequisite are isolated.
    # The actual kernel still constructs tasks and the actual face sampler calls
    # arrange_material/_ring_split_attempt below, with real pressure checks.
    monkeypatch.setattr(host,'_admit_stage_package',lambda *args:lambda:None)
    def run(self,kind,tasks,serial_task):
        events.append(kind)
        return tuple(serial_task(task) for task in tasks)
    monkeypatch.setattr(host.Bridge,'run',run);monkeypatch.setattr(host._SessionBridge,'run',run)
    policy=bi.IntersectionBatchPolicy()
    def parallel(bridge):return bi._plan_intersections(model,tuple(model.faces),policy=policy,_planning_mode='exact',_seam_policy='ring_split',_stages=bridge)
    with pytest.raises(executor.PlannerMemoryExceeded) as caught:
        host.qualified_plan(model,control,lambda:pytest.fail('serial fallback'),parallel,policy=policy,mode='exact')
    assert events==['pair','face','inside-ring-split']
    assert len(errors)==1 and caught.value is errors[0]
    assert ma._ring_split_attempt.__code__ is ring_code and ma._arrange_material_once.__code__ is once_code
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
