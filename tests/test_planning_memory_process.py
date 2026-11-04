"""Small real Windows enforcement and nested geometry; no whole-model runs."""
from _planning_numpy_profile_contract import numpy_profile_test
from _planning_platform_contract import windows_owned_test, observe_plan_calls, assert_unsupported512_terminal
import ctypes,json,subprocess,sys
from pathlib import Path
from ctypes import wintypes as W
import psutil,pytest
from anygeometry._planning_process import OwnedTree
from anygeometry import _planning_executor as executor
from anygeometry import _planning_session as session
from anygeometry import _planning_host as host
from anygeometry import batch_intersections as bi
from anygeometry._planning_canaries import _fixture
from anygeometry.serialization import from_dict,to_dict
from anygeometry._planning_qualification import QualificationDecline


def test_required_qualified_lane_forbids_platform_refusal(monkeypatch,tmp_path,request):
    from types import SimpleNamespace
    import _planning_platform_contract as platform
    from anygeometry import _planning_process as process
    monkeypatch.setattr(platform,'os',SimpleNamespace(name='posix',environ={
        'ANYGEOMETRY_REQUIRE_QUALIFIED_PLANNING_TESTS':'1'}))
    # This unit verifies a boundary before ownership; it never substitutes WinAPI.
    def forbidden(*args,**kwargs):pytest.fail('native ownership boundary was reached')
    monkeypatch.setattr(process,'OwnedTree',forbidden)
    entered=[]
    @platform.windows_owned_test
    def original(tmp_path):entered.append(True)
    with pytest.raises(pytest.fail.Exception,match='qualified planning test lane requires original Windows assertions'):
        original(tmp_path=tmp_path,request=request)
    assert entered==[] and tuple(tmp_path.iterdir())==()


@windows_owned_test
def test_real_job_rejects_over_cap_virtual_commit(tmp_path):
    cap=64*1024**2;tree=OwnedTree(tmp_path,memory_limit_bytes=cap);child=None
    code="""import ctypes,json
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.VirtualAlloc.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_ulong,ctypes.c_ulong];k.VirtualAlloc.restype=ctypes.c_void_p
k.VirtualFree.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_ulong];k.VirtualFree.restype=ctypes.c_int
p=k.VirtualAlloc(None,1024**2,0x3000,4);assert p
assert k.VirtualFree(p,0,0x8000)
p=k.VirtualAlloc(None,128*1024**2,0x3000,4)
print(json.dumps(dict(over_cap_commit_denied=not bool(p))),flush=True)
if p:assert k.VirtualFree(p,0,0x8000)
"""
    command=[sys.executable,'-B','-c',code]
    try:
        child=subprocess.Popen(command,cwd=tmp_path,stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW|4)
        assert tree.api.AssignProcessToJobObject(tree.handle,W.HANDLE(int(child._handle)))
        native=ctypes.WinDLL('ntdll');native.NtResumeProcess.argtypes=[W.HANDLE];native.NtResumeProcess.restype=W.LONG
        assert native.NtResumeProcess(W.HANDLE(int(child._handle)))==0
        stdout,stderr=child.communicate(timeout=10)
        assert child.returncode==0,stderr.decode(errors='replace')
        assert json.loads(stdout)['over_cap_commit_denied']
        (tmp_path/'pressure-result.json').write_bytes(stdout)
    finally:
        if child is not None and child.poll() is None:
            assert psutil.Process(child.pid).cmdline()==command
            child.kill();child.wait(timeout=5)
        # A denied oversized request can still appear in Windows peak accounting.
        # The production peak guard MUST refuse, not certify a compliant peak.
        with pytest.raises(QualificationDecline):tree.close()
    report=json.loads((tmp_path/'cleanup.json').read_text())
    admitted=json.loads((tmp_path/'memory-limit.json').read_text())
    assert admitted['limit_bytes']==cap and admitted['flags']==0x2200
    assert report['verified'] and report['active_processes']==0 and 'memory_error' in report
    bad=report['memory_observation']
    assert bad['limit_bytes']==cap and bad['flags']==0x2200 and bad['peak_job_bytes']>cap
    assert tree.handle is None


@windows_owned_test
def test_tiny_cap_refuses_real_planner_and_cleans(tmp_path,monkeypatch):
    original,_=_fixture();model=from_dict(to_dict(original));owned=None
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**k:pytest.fail('unknown limit spawned child'))
    try:
        with pytest.raises((OSError,QualificationDecline)):
            owned=executor.OwnedPlanner(tmp_path/'tiny',workers=2,timeout=10,tree_memory_limit=1)
            owned.submit(model,('pair',))
    finally:
        if owned is not None:owned.close()
    report=json.loads((tmp_path/'tiny/cleanup.json').read_text())
    assert report['verified'] and report['active_processes']==0 and report['started_children']==0
    assert 'memory_error' in report and report['memory_observation'] is None
    assert report['memory_failure']['operation']=='SetInformationJobObject'
    assert report['memory_failure']['requested_bytes']==1
    assert type(report['memory_failure']['winerror']) is int and report['memory_failure']['winerror']>0


@pytest.mark.parametrize('reuse',[False,True])
@windows_owned_test
@numpy_profile_test
def test_two_worker_geometry_under_explicit_tree_cap(tmp_path,monkeypatch,reuse,request):
    original,_=_fixture();doc=to_dict(original);reference=from_dict(doc)
    monkeypatch.setenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING','0')
    serial=bi.plan_intersections(reference,tuple(reference.faces),policy='connect');bi.apply_intersections(reference,serial,policy='connect')
    expected=(serial.content_checksum,to_dict(reference)['checksum']['value']);members=[]
    def observed(plan):
        value=from_dict(doc);bi.apply_intersections(value,plan,policy='connect');return plan.content_checksum,to_dict(value)['checksum']['value']
    # Enumerate only this owned Windows Job, never machine-wide processes.
    def membership(owned):
        seen=set()
        def check():
            for pid in owned.tree._pids():
                if pid==owned.child.pid or pid in seen:continue
                try:process=psutil.Process(pid);command=process.cmdline()
                except psutil.NoSuchProcess:continue
                # Startup can temporarily own platform/version helpers. They
                # are not workers; final report PID equality below requires
                # every requested worker, without broadening cleanup rules.
                if len(command)!=7 or command[-3] not in ('worker','session-worker'):continue
                assert Path(command[3]).resolve()==Path(executor.__file__).with_name('_planning_child.py').resolve()
                directory=Path(command[-2]).resolve();assert directory.is_relative_to(owned.directory)
                api=owned.tree.api;api.OpenProcess.argtypes=[W.DWORD,W.BOOL,W.DWORD];api.OpenProcess.restype=W.HANDLE
                api.IsProcessInJob.argtypes=[W.HANDLE,W.HANDLE,ctypes.POINTER(W.BOOL)];api.IsProcessInJob.restype=W.BOOL
                handle=api.OpenProcess(0x1000,False,pid);assert handle
                try:
                    result=W.BOOL();assert api.IsProcessInJob(handle,owned.tree.handle,ctypes.byref(result)) and result.value
                finally:api.CloseHandle(handle)
                members.append(dict(pid=pid,directory=str(directory),in_outer_capped_job=True));seen.add(pid)
        return check
    if reuse:
        base=session.OwnedSession
        class Observed(base):
            def prepare_plan(self,*a,poll,**kw):
                check=membership(self)
                def observe():poll();check()
                return super().prepare_plan(*a,poll=observe,**kw)
        monkeypatch.setattr(session,'OwnedSession',Observed)
    else:
        base=host.OwnedPlanner
        class Observed(base):
            def submit_stage(self,*a,poll,**kw):
                check=membership(self)
                def observe():poll();check()
                return super().submit_stage(*a,poll=observe,**kw)
        monkeypatch.setattr(host,'OwnedPlanner',Observed)
    monkeypatch.setenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING','1');source=from_dict(doc);cap=512*1024**2
    control=host.StageControl(tmp_path/'geometry',expected_ci=expected,observed_ci=observed,workers=2,timeout=80,reuse_workers=reuse,tree_memory_limit=cap)
    from anygeometry._planning_executor import PlannerCleanupError
    with observe_plan_calls(enabled=True) as calls:
        try:
            plan=bi.plan_intersections(source,tuple(source.faces),policy='connect',_parallel_control=control)
        except PlannerCleanupError as error:
            assert_unsupported512_terminal(error, reuse=reuse, control=control, source=source,
                                           document=doc, calls=calls, request=request)
            return
    assert control.mode=='parallel' and control.ci_verified and plan.content_checksum==serial.content_checksum
    assert to_dict(source)==doc and len(members)==(2 if reuse else 4)
    if reuse:
        reported={w['pid'] for report in control.reports for w in report['worker_reports']}
        assert reported=={item['pid'] for item in members}
    else:
        # Legacy reports intentionally have no PID. Bind their ordered values to
        # checksum-validated files in the independently observed worker folders.
        expected_directories=set()
        for report in control.reports:
            for index,worker in enumerate(report['worker_reports']):
                directory=(control.directory/report['stage']/('worker-'+str(index))).resolve()
                assert executor.read_packet(directory/'result.json','worker-result')==worker
                expected_directories.add(str(directory))
        assert expected_directories=={item['directory'] for item in members}
        assert len(expected_directories)==len(members)==4
    (tmp_path/'membership.json').write_text(json.dumps(members,indent=2))
    for name in (['session'] if reuse else ['pair','face']):
        report=json.loads((control.directory/name/'cleanup.json').read_text())
        assert report['verified'] and report['active_processes']==0 and report['memory']['limit_bytes']==cap
        assert 0<report['memory']['peak_job_bytes']<=cap

    request.node.user_properties[:]=[(key,value) for key,value in request.node.user_properties if key!='execution_scope']
    request.node.user_properties.extend((('execution_scope','unsupported512-single-fit-observation'),
        ('resource_configuration','unsupported512-safe-fit-observation'),('reuse_workers',reuse),
        ('parallel_qualified',False),('memory_capacity_qualified',False)))


@pytest.mark.parametrize('reuse', [False, True])
@pytest.mark.parametrize('fault', ['valid', 'source', 'callback', 'cleanup', 'context', 'flags', 'limit', 'peak', 'active', 'verified'])
def test_strict_unsupported512_evidence_rejects_nonresource_failures(fault, request, reuse):
    # Portable validation-unit fixtures, not native execution or cap enforcement.
    from copy import deepcopy
    from _planning_platform_contract import _strict_memory_failure
    from anygeometry._planning_executor import PlannerCleanupError, PlannerSourceChanged
    request.node.user_properties.append(('execution_scope', 'resource-evidence-validation-unit'))
    cap = 512 * 1024**2
    observation = dict(requested_bytes=cap, limit_bytes=cap, flags=0x2200,
                       peak_job_bytes=cap + 4096, peak_process_bytes=cap // 2)
    cause = QualificationDecline('owned-tree memory enforcement differs from request: ' + str(observation))
    error = PlannerCleanupError('preparation cleanup failed' if reuse else 'session planner cleanup failed')
    error.__cause__ = cause
    cleanup = dict(verified=True, active_processes=0, memory_observation=deepcopy(observation),
                   memory_phase='after_cleanup', memory_error='QualificationDecline: ' + str(cause),
                   memory_failure=dict(operation=None, winerror=None, requested_bytes=cap))
    admission = dict(observation, peak_job_bytes=0, peak_process_bytes=0)
    if fault == 'source': error.__cause__ = PlannerSourceChanged('source changed')
    elif fault == 'callback': error.__cause__ = OSError('callback failed')
    elif fault == 'cleanup': error.__cause__ = OSError('CloseHandle failed')
    elif fault == 'context': cause.__context__ = RuntimeError('unrelated cleanup defect')
    elif fault == 'flags': cleanup['memory_observation']['flags'] = 0x2000
    elif fault == 'limit': admission['limit_bytes'] = cap * 2
    elif fault == 'peak': cleanup['memory_observation']['peak_job_bytes'] = cap
    elif fault == 'active': cleanup['active_processes'] = 1
    elif fault == 'verified': cleanup['verified'] = False
    if fault == 'valid':
        assert _strict_memory_failure(error, cleanup, admission, reuse=reuse) == observation
    else:
        with pytest.raises(AssertionError):
            _strict_memory_failure(error, cleanup, admission, reuse=reuse)


def _controlled_terminal512(tmp_path, monkeypatch, reuse):
    """Checksum-valid metadata fixtures only, not measured native resource evidence."""
    from types import SimpleNamespace
    monkeypatch.delenv('CI', raising=False)
    cap=512*1024**2;source=__import__('anygeometry').GeometryModel()
    document=to_dict(source);directory=tmp_path/'geometry';folder=directory/('session' if reuse else 'pair');folder.mkdir(parents=True)
    observation=dict(requested_bytes=cap,limit_bytes=cap,flags=0x2200,peak_job_bytes=cap+4096,peak_process_bytes=cap//2)
    cause=QualificationDecline('owned-tree memory enforcement differs from request: '+str(observation))
    error=executor.PlannerCleanupError('preparation cleanup failed' if reuse else 'session planner cleanup failed');error.__cause__=cause
    cause.__context__=QualificationDecline('QualificationDecline: MemoryError: ')
    cleanup=dict(verified=True,active_processes=0,memory_observation=observation,memory_phase='after_cleanup',memory_error='QualificationDecline: '+str(cause),memory_failure=dict(operation=None,winerror=None,requested_bytes=cap))
    for name,value in [('cleanup.json',cleanup),('workers-cleanup.json',dict(verified=True,active_processes=0)),('memory-limit.json',dict(observation,peak_job_bytes=0,peak_process_bytes=0))]:(folder/name).write_text(json.dumps(value))
    setup=dict(model=executor.freeze_model(source),sources={},workers=2,batch_size=None,limits=dict(payload_limit=4_000_000,walk_limit=100000))
    session_packet=executor.pack(setup,'session');executor.write_packet(folder/'session.json',session_packet)
    body=dict(session=session_packet['sha256'],sequence=0,previous=session_packet['sha256'],stage='pair',tasks=[dict(kind='pair')])
    command=executor.pack(body,'session-command');executor.write_packet(folder/'command-0.json',command)
    result=dict(status='declined',reason='QualificationDecline: MemoryError: ',session_sha256=session_packet['sha256'],request_sha256=command['sha256'])
    executor.write_packet(folder/'result.json',executor.pack(result,'result'))
    control=SimpleNamespace(directory=directory,reuse_workers=reuse,tree_memory_limit=cap,workers=2,mode='terminal',ci_verified=False)
    def fixture(name):
        assert name=='tmp_path'
        return tmp_path
    request=SimpleNamespace(node=SimpleNamespace(user_properties=[],nodeid='controlled-validation'),
        config=SimpleNamespace(option=SimpleNamespace(basetemp=None)),getfixturevalue=fixture)
    return error,control,source,document,request


def test_reuse512_terminal_lifecycle_has_exact_resource_proof(tmp_path,monkeypatch,request):
    error,control,source,document,isolated=_controlled_terminal512(tmp_path,monkeypatch,True)
    assert_unsupported512_terminal(error,reuse=True,control=control,source=source,document=document,calls=['candidate'],request=isolated)
    summary=json.loads((control.directory/'unsupported512-terminal.json').read_text())
    assert summary['reuse_workers'] is True and summary['parallel_qualified'] is False
    request.node.user_properties.append(('execution_scope','resource-evidence-validation-unit'))


@pytest.mark.parametrize('reuse',[False,True])
@pytest.mark.parametrize('route',['direct','budget','consistent','unknown','wrong-budget-cap','conflict','missing','bool-budget-cap'])
def test_unsupported512_budget_cap_is_exact(tmp_path,monkeypatch,request,reuse,route):
    from types import SimpleNamespace
    from anygeometry._planning_memory import MemoryBudget
    error,control,source,document,isolated=_controlled_terminal512(tmp_path,monkeypatch,reuse)
    cap=512*1024**2
    if route!='direct':
        control.memory_budget=MemoryBudget(cap,cap,cap*3,(1,1))
    if route in ('budget','unknown','wrong-budget-cap','missing','bool-budget-cap'):
        control.tree_memory_limit=None
    if route=='unknown':control.memory_budget=SimpleNamespace(tree_limit_bytes=cap)
    elif route=='wrong-budget-cap':control.memory_budget=MemoryBudget(cap,cap*2,cap*4,(1,1))
    elif route=='conflict':control.tree_memory_limit=cap*2
    elif route=='missing':control.memory_budget=None
    elif route=='bool-budget-cap':object.__setattr__(control.memory_budget,'tree_limit_bytes',True)
    if route in ('direct','budget','consistent'):
        assert_unsupported512_terminal(error,reuse=reuse,control=control,source=source,document=document,calls=['candidate'],request=isolated)
    else:
        with pytest.raises(AssertionError):
            assert_unsupported512_terminal(error,reuse=reuse,control=control,source=source,document=document,calls=['candidate'],request=isolated)
    request.node.user_properties.append(('execution_scope','resource-evidence-validation-unit'))


@pytest.mark.parametrize('reuse',[False,True])
@pytest.mark.parametrize('fault',['valid','folder','flag','reuse-type','message','context','serial','session','command','result','handle','workers'])
def test_unsupported512_lifecycle_binding_is_strict(tmp_path,monkeypatch,request,reuse,fault):
    error,control,source,document,isolated=_controlled_terminal512(tmp_path,monkeypatch,reuse)
    folder=control.directory/('session' if reuse else 'pair');calls=['candidate']
    if fault=='folder':folder.rename(control.directory/('pair' if reuse else 'session'))
    elif fault=='flag':control.reuse_workers=not reuse
    elif fault=='reuse-type':control.reuse_workers=int(reuse)
    elif fault=='message':error.args=('session planner cleanup failed' if reuse else 'preparation cleanup failed',)
    elif fault=='context':error.__context__=executor.PlannerCancelled('cancelled')
    elif fault=='serial':calls.append('serial')
    elif fault=='session':
        body=executor.read_packet(folder/'session.json','session');body['workers']=3;executor.write_packet(folder/'session.json',executor.pack(body,'session'))
    elif fault=='command':
        body=executor.read_packet(folder/'command-0.json','session-command');body['sequence']=1;executor.write_packet(folder/'command-0.json',executor.pack(body,'session-command'))
    elif fault=='result':
        body=executor.read_packet(folder/'result.json','result');body['request_sha256']='0'*64;executor.write_packet(folder/'result.json',executor.pack(body,'result'))
    elif fault=='handle':(folder/'handle-cleanup-error-1.json').write_text('{}')
    elif fault=='workers':(folder/'workers-cleanup.json').write_text(json.dumps(dict(verified=True,active_processes=1)))
    if fault=='valid':
        assert_unsupported512_terminal(error,reuse=reuse,control=control,source=source,document=document,calls=calls,request=isolated)
    else:
        with pytest.raises((AssertionError,QualificationDecline)):
            assert_unsupported512_terminal(error,reuse=reuse,control=control,source=source,document=document,calls=calls,request=isolated)
    request.node.user_properties.append(('execution_scope','resource-evidence-validation-unit'))


@pytest.mark.parametrize('wrong_base',[False,True])
def test_unsupported512_ci_copy_is_fixed_bounded_and_private_free(tmp_path,monkeypatch,request,wrong_base):
    import _planning_platform_contract as contract
    error,control,source,document,isolated=_controlled_terminal512(tmp_path,monkeypatch,True)
    assert_unsupported512_terminal(error,reuse=True,control=control,source=source,document=document,calls=['candidate'],request=isolated)
    summary=json.loads((control.directory/'unsupported512-terminal.json').read_text())
    root=tmp_path/'checkout';(root/'tests').mkdir(parents=True)
    monkeypatch.setattr(contract,'__file__',str(root/'tests/_planning_platform_contract.py'))
    monkeypatch.setenv('CI','true')
    folder=control.directory/'session';(folder/'private-unrelated.txt').write_text('never retained')
    assert isolated.config.option.basetemp is None
    if wrong_base:
        isolated.getfixturevalue=lambda name:tmp_path/'another-test'
        with pytest.raises(AssertionError):
            contract._retain_unsupported512_evidence(summary,[folder],isolated)
        assert not list((root/'ci-isolation').rglob('cleanup.json'))
        return
    target=contract._retain_unsupported512_evidence(summary,[folder],isolated)
    assert target.is_relative_to(root/'ci-isolation/unsupported512')
    assert {p.name for p in target.iterdir()}=={'session','summary.json'}
    assert {p.name for p in (target/'session').iterdir()}=={'cleanup.json','workers-cleanup.json','memory-limit.json','session.json','command-0.json','result.json'}
    retained=json.loads((target/'summary.json').read_text());assert len(retained['retained_files'])==6
    request.node.user_properties.append(('execution_scope','resource-evidence-validation-unit'))
