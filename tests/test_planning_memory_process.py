"""Small real Windows enforcement and nested geometry; no whole-model runs."""
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
def test_two_worker_geometry_under_explicit_tree_cap(tmp_path,monkeypatch,reuse):
    original,_=_fixture();doc=to_dict(original);reference=from_dict(doc)
    monkeypatch.delenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING',raising=False)
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
    plan=bi.plan_intersections(source,tuple(source.faces),policy='connect',_parallel_control=control)
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
