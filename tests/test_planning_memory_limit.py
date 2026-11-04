"""PP4a explicit committed-memory cap; no RAM/performance qualification claim."""
import ctypes,json
import pytest
from _planning_contract_helpers import require_qualified_planning_lane
from _planning_numpy_profile_contract import numpy_profile_test
from anygeometry._planning_process import OwnedTree
from anygeometry._planning_qualification import QualificationDecline
from anygeometry import _planning_host as host
from anygeometry import _planning_executor as executor
from anygeometry import _planning_session as session
from anygeometry import batch_intersections as bi
from anygeometry._planning_canaries import _fixture
from anygeometry.serialization import from_dict,to_dict


def _windows_job_initial_refusal(directory,request,memory_limit_bytes=128*1024**2):
    from anygeometry import _planning_process as process
    reason=None if process.os.name=='nt' else 'owned planner requires Windows Job containment'
    require_qualified_planning_lane(reason)
    if reason is None:
        request.node.user_properties.append(('execution_scope','original-Windows-Job-assertions'))
        request.node.user_properties.append(('original_test_body_executed',True))
        return False
    with pytest.raises(QualificationDecline,match='^owned planner requires Windows Job containment$') as caught:
        OwnedTree(directory,memory_limit_bytes=memory_limit_bytes)
    assert type(caught.value) is QualificationDecline
    assert not (directory/'cleanup.json').exists()
    request.node.user_properties.extend((('execution_scope','initial-Windows-Job-refusal'),
        ('postcapture_qualification',False),('original_test_body_executed',False)))
    return True


@pytest.mark.parametrize('value',[True,False,0,-1,1.5,'1024',2**(8*ctypes.sizeof(ctypes.c_size_t))])
def test_invalid_memory_cap_refuses_before_job_creation(tmp_path,monkeypatch,value):
    from anygeometry import _planning_native as native,_planning_process as process
    for valid in (None,1,2**(8*ctypes.sizeof(ctypes.c_size_t))-1):
        assert process.validate_memory_limit(valid) is None
    monkeypatch.setattr(native,'NativeAPI',lambda:pytest.fail('invalid cap created native Job backend'))
    with pytest.raises(QualificationDecline,match='^invalid owned-tree committed-memory limit$'):
        OwnedTree(tmp_path,memory_limit_bytes=value)


def test_explicit_cap_readback_and_retirement_receipt(tmp_path,request):
    if _windows_job_initial_refusal(tmp_path,request):return
    cap=128*1024**2;tree=OwnedTree(tmp_path,memory_limit_bytes=cap)
    try:
        report=tree.memory_snapshot()
        assert report['limit_bytes']==cap and report['flags']==0x2200
        assert 0<=report['peak_job_bytes']<=cap
    finally:tree.close()
    final=json.loads((tmp_path/'cleanup.json').read_text())
    assert final['verified'] and final['active_processes']==0 and final['memory']['limit_bytes']==cap


def test_none_preserves_unqualified_original_cleanup_schema(tmp_path,request):
    if _windows_job_initial_refusal(tmp_path,request,memory_limit_bytes=None):return
    tree=OwnedTree(tmp_path,memory_limit_bytes=None)
    assert tree.memory_snapshot() is None
    tree.close();final=json.loads((tmp_path/'cleanup.json').read_text())
    assert 'memory' not in final


@pytest.mark.parametrize('fault',['larger','disabled','breakaway'])
def test_memory_readback_drift_refuses_and_still_cleans(tmp_path,fault,request):
    if _windows_job_initial_refusal(tmp_path,request):return
    cap=128*1024**2;tree=OwnedTree(tmp_path,memory_limit_bytes=cap)
    value=tree._limits_type();value.basic.flags=0x2200;value.jm=cap
    if fault=='larger':value.jm=cap*2
    elif fault=='disabled':value.basic.flags=0x2000
    else:value.basic.flags|=0x800
    assert tree.api.SetInformationJobObject(tree.handle,9,ctypes.byref(value),ctypes.sizeof(value))
    with pytest.raises(QualificationDecline):tree.memory_snapshot()
    with pytest.raises(QualificationDecline):tree.close()
    assert tree.handle is None
    final=json.loads((tmp_path/'cleanup.json').read_text())
    assert final['verified'] and final['active_processes']==0 and 'memory_error' in final


def test_memory_query_failure_does_not_skip_cleanup(tmp_path,monkeypatch,request):
    if _windows_job_initial_refusal(tmp_path,request):return
    tree=OwnedTree(tmp_path,memory_limit_bytes=128*1024**2)
    def fail():raise OSError('query failed')
    monkeypatch.setattr(tree,'memory_snapshot',fail)
    with pytest.raises(OSError,match='query failed'):tree.close()
    assert tree.handle is None and json.loads((tmp_path/'cleanup.json').read_text())['verified']


@pytest.mark.parametrize('reuse',[False,True])
@numpy_profile_test
def test_public_control_propagates_tree_cap(tmp_path,monkeypatch,reuse):
    original,_=_fixture();model=from_dict(to_dict(original));seen=[];cap=512*1024**2
    class Capture:
        def __init__(self,*a,**kw):seen.append(kw['tree_memory_limit']);raise QualificationDecline('capture refusal')
    monkeypatch.setattr(host,'OwnedPlanner',Capture);monkeypatch.setattr(session,'OwnedSession',Capture)
    monkeypatch.setenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING','1')
    control=host.StageControl(tmp_path/'p',expected_ci=('p','a'),observed_ci=lambda _:pytest.fail('CI after refusal'),workers=2,reuse_workers=reuse,tree_memory_limit=cap)
    bi.plan_intersections(model,tuple(model.faces),policy='connect',_parallel_control=control)
    assert seen==[cap] and control.mode=='serial'


def test_executor_installs_cap_before_spawn(tmp_path,monkeypatch):
    events=[];cap=512*1024**2
    class Tree:
        def __init__(self,directory,*,memory_limit_bytes):events.append(('install',memory_limit_bytes))
        def spawn(self,*args):events.append(('spawn',));return object()
        def close(self):events.append(('close',))
    from anygeometry import _planning_process as process
    monkeypatch.setattr(process,'OwnedTree',Tree)
    owned=executor.OwnedPlanner(tmp_path/'owned',tree_memory_limit=cap)
    owned.close();assert events==[('install',cap),('spawn',),('close',)]

def test_constructor_query_failure_closes_created_handle(tmp_path,monkeypatch,request):
    if _windows_job_initial_refusal(tmp_path,request):return
    seen=[]
    def fail(self):seen.append(self);raise OSError('readback failed')
    monkeypatch.setattr(OwnedTree,'memory_snapshot',fail)
    with pytest.raises(OSError,match='readback failed'):OwnedTree(tmp_path,memory_limit_bytes=128*1024**2)
    assert len(seen)==1 and seen[0].handle is None


def test_interrupt_during_memory_query_still_verifies_cleanup(tmp_path,monkeypatch,request):
    if _windows_job_initial_refusal(tmp_path,request):return
    tree=OwnedTree(tmp_path,memory_limit_bytes=128*1024**2)
    def fail():raise KeyboardInterrupt('query interrupted')
    monkeypatch.setattr(tree,'memory_snapshot',fail)
    with pytest.raises(KeyboardInterrupt):tree.close()
    assert tree.handle is None and json.loads((tmp_path/'cleanup.json').read_text())['verified']


def test_bridge_freezes_requested_cap(tmp_path):
    control=host.StageControl(tmp_path,expected_ci=('p','a'),observed_ci=lambda _:('p','a'),tree_memory_limit=128*1024**2)
    bridge=host.Bridge({},control)
    control.tree_memory_limit=512*1024**2
    assert bridge.tree_options=={'tree_memory_limit':128*1024**2}

def test_retirement_observation_survives_unverified_cleanup(tmp_path,monkeypatch,request):
    if _windows_job_initial_refusal(tmp_path,request):return
    tree=OwnedTree(tmp_path,memory_limit_bytes=128*1024**2)
    def fail():raise RuntimeError('PID accounting failed')
    monkeypatch.setattr(tree,'_pids',fail)
    with pytest.raises(RuntimeError,match='PID accounting failed'):tree.close()
    assert tree.handle is None and not (tmp_path/'cleanup.json').exists()
    record=json.loads((tmp_path/'memory-retirement.json').read_text())
    assert record['memory_observation']['limit_bytes']==128*1024**2 and 'verified' not in record


def test_missing_capped_directory_refuses_before_handle(tmp_path,monkeypatch,request):
    if _windows_job_initial_refusal(tmp_path/'missing',request):return
    from anygeometry import _planning_native as native
    monkeypatch.setattr(native,'NativeAPI',lambda:pytest.fail('missing directory created native Job backend'))
    with pytest.raises(QualificationDecline,match='^capped owned-tree directory must already exist$'):
        OwnedTree(tmp_path/'missing',memory_limit_bytes=128*1024**2)

def test_readback_error_uses_immediate_ctypes_error_copy(tmp_path,monkeypatch,request):
    if _windows_job_initial_refusal(tmp_path,request):return
    tree=OwnedTree(tmp_path,memory_limit_bytes=128*1024**2)
    original=tree.api.QueryInformationJobObject
    def refuse(*args):ctypes.set_last_error(87);return 0
    try:
        monkeypatch.setattr(tree.api,'QueryInformationJobObject',refuse)
        with pytest.raises(OSError) as caught:tree.memory_snapshot()
        assert caught.value.winerror==87 and caught.value.job_operation=='QueryInformationJobObject'
    finally:
        monkeypatch.setattr(tree.api,'QueryInformationJobObject',original);tree.close()


@pytest.mark.parametrize('fault',['late_peak','late_limit_drift','late_query_failure'])
def test_retirement_rechecks_native_peak_and_flags(tmp_path,monkeypatch,fault,request):
    if _windows_job_initial_refusal(tmp_path,request):return
    cap=128*1024**2;tree=OwnedTree(tmp_path,memory_limit_bytes=cap)
    query=tree.api.QueryInformationJobObject;terminate=tree.api.TerminateJobObject
    retired=False
    def retire(*args):
        nonlocal retired
        result=terminate(*args)
        if result:retired=True
        return result
    def observed(handle,kind,buffer,size,returned):
        result=query(handle,kind,buffer,size,returned)
        if kind==9 and retired and result:
            if fault=='late_query_failure':ctypes.set_last_error(5);return 0
            limits=ctypes.cast(buffer,ctypes.POINTER(tree._limits_type)).contents
            if fault=='late_peak':limits.pjm=cap+1
            else:limits.jm=cap+1
        return result
    monkeypatch.setattr(tree.api,'TerminateJobObject',retire)
    monkeypatch.setattr(tree.api,'QueryInformationJobObject',observed)
    with pytest.raises((QualificationDecline,OSError)):tree.close()
    report=json.loads((tmp_path/'cleanup.json').read_text())
    assert tree.handle is None and report['verified'] and report['active_processes']==0
    assert 'memory_error' in report and 'memory' not in report
    if fault=='late_query_failure':
        assert report['memory_observation'] is None
        assert report['memory_failure']['operation']=='QueryInformationJobObject'
        assert report['memory_failure']['winerror']==5
    else:
        observation=report['memory_observation']
        assert observation['peak_job_bytes']>cap if fault=='late_peak' else observation['limit_bytes']!=cap
