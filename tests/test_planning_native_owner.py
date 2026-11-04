"""Native ownership regression and optional packaging contracts."""
import os
from pathlib import Path
import runpy
import sys
import pytest
from _planning_platform_contract import windows_owned_test

ROOT = Path(__file__).resolve().parents[1]

def setup_config(monkeypatch, **flags):
    import setuptools
    captured = {}
    for name in ('BUILD_NATIVE','REQUIRE_NATIVE','DISABLE_NATIVE','BUILD_PLANNING_NATIVE','REQUIRE_PLANNING_NATIVE','DISABLE_PLANNING_NATIVE'):
        monkeypatch.delenv('ANYGEOMETRY_'+name, raising=False)
    for name, value in flags.items(): monkeypatch.setenv('ANYGEOMETRY_'+name, value)
    monkeypatch.setattr(setuptools, 'setup', lambda **kw: captured.update(kw))
    runpy.run_path(str(ROOT/'setup.py'))
    return captured

def test_helper_only_is_stable_abi_and_does_not_enable_numerics(monkeypatch):
    config=setup_config(monkeypatch, BUILD_PLANNING_NATIVE='1')
    extensions=config['ext_modules']
    if sys.platform != 'win32':
        assert extensions == []
        with pytest.raises(RuntimeError, match='^the planning resource helper requires Windows$'):
            setup_config(monkeypatch, REQUIRE_PLANNING_NATIVE='1')
        return
    assert [e.name for e in extensions]==['anygeometry._planning_native_core']
    assert extensions[0].py_limited_api
    assert ('Py_LIMITED_API','0x030B0000') in extensions[0].define_macros

def test_helper_off_keeps_pure_build(monkeypatch):
    assert setup_config(monkeypatch)['ext_modules']==[]

@windows_owned_test
def test_native_loader_exports_binding_identity(tmp_path):
    from anygeometry._planning_native import identity
    result=identity()
    assert result['abi']=='cp311-abi3'
    assert result['resource_abi']==1
    assert len(result['binary_sha256'])==64
    assert result['module']=='anygeometry._planning_native_core'

@windows_owned_test
def test_missing_helper_refuses_before_resource_acquisition(tmp_path, monkeypatch):
    from anygeometry import _planning_native as native
    from anygeometry._planning_process import OwnedTree
    from anygeometry._planning_qualification import QualificationDecline
    def missing(*args): raise ImportError('not installed')
    monkeypatch.setattr(native.importlib, 'import_module', missing)
    with pytest.raises(QualificationDecline, match='helper unavailable'):
        OwnedTree(tmp_path)
    assert list(tmp_path.iterdir()) == []

@pytest.mark.parametrize('operation', ['CreateFileW stdin','CreateFileW stdout','CreateFileW stderr'])
@windows_owned_test
def test_partial_native_stdio_acquisition_retires(tmp_path, operation):
    from anygeometry._planning_process import OwnedTree
    tree=OwnedTree(tmp_path)
    tree.api.inject_fault(operation)
    with pytest.raises(OSError): tree.spawn('worker',tmp_path/'child',5)
    assert tree.handle is None
    assert not tree.children[0]._duplicates
    assert tree.children[0].info.process is None


@windows_owned_test
def test_signal_after_native_close_cannot_restore_handle(tmp_path, monkeypatch):
    from anygeometry._planning_process import OwnedTree
    tree=OwnedTree(tmp_path);actual=tree.api.CloseHandle;target=tree.handle
    def interrupted(handle):
        result=actual(handle)
        if handle==target: raise KeyboardInterrupt('after committed native close')
        return result
    monkeypatch.setattr(tree.api,'CloseHandle',interrupted)
    with pytest.raises(KeyboardInterrupt): tree.close()
    assert tree.handle is None
    monkeypatch.setattr(tree.api,'CloseHandle',actual)
    tree.close()


@windows_owned_test
def test_uncertain_native_close_is_poisoned_and_never_retried(tmp_path):
    import ctypes
    from anygeometry._planning_process import OwnedTree
    tree=OwnedTree(tmp_path);old=tree.handle
    tree.api.inject_fault('CloseHandle',6,after=True)
    with pytest.raises(OSError): tree.close()
    assert tree.api.state()['poisoned'] and tree.handle==old
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateEventW.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_wchar_p]
    kernel.CreateEventW.restype=ctypes.c_void_p
    kernel.SetEvent.argtypes=[ctypes.c_void_p];kernel.SetEvent.restype=ctypes.c_int
    kernel.CloseHandle.argtypes=[ctypes.c_void_p];kernel.CloseHandle.restype=ctypes.c_int
    event=kernel.CreateEventW(None,False,False,None)
    assert event
    try:
        with pytest.raises(OSError): tree.close()
        assert kernel.SetEvent(event), 'poisoned raw value must never close a reused handle'
    finally: assert kernel.CloseHandle(event)
    assert len(list(tmp_path.glob('handle-cleanup-error-*.json')))==2


def _sleeping_child(tmp_path,monkeypatch,seconds='.15'):
    from anygeometry import _planning_process as process
    fake=tmp_path/'fake';fake.mkdir()
    (fake/'_planning_child.py').write_text('import time\ntime.sleep('+seconds+')\n')
    monkeypatch.setattr(process,'__file__',str(fake/'_planning_process.py'))
    tree=process.OwnedTree(tmp_path)
    return tree,tree.spawn('worker',tmp_path/'child',5)


@windows_owned_test
def test_native_wait_close_synchronizes_without_gil_deadlock(tmp_path,monkeypatch):
    import threading,time
    tree,child=_sleeping_child(tmp_path,monkeypatch)
    result=[]
    def waiter():
        try: result.append(child.wait(timeout=3))
        except BaseException as error: result.append(error)
    thread=threading.Thread(target=waiter);thread.start()
    try:
        deadline=time.monotonic()+2
        while not tree.api.state(child.index)['borrowers'] and time.monotonic()<deadline: time.sleep(.001)
        assert tree.api.state(child.index)['borrowers']==1
        child.close_handles()
        thread.join(timeout=3)
        assert not thread.is_alive() and result==[0]
        assert tree.api.state(child.index)['borrowers']==0 and child.info.process is None
    finally:
        thread.join(timeout=3);tree.close()


@windows_owned_test
def test_wait_releases_borrower_before_delivering_python_signal(tmp_path,monkeypatch):
    import _thread,threading,time
    tree,child=_sleeping_child(tmp_path,monkeypatch)
    trigger=threading.Thread(target=lambda:(time.sleep(.03),_thread.interrupt_main()))
    trigger.start()
    try:
        with pytest.raises(KeyboardInterrupt): child.wait(timeout=3)
        assert tree.api.state(child.index)['borrowers']==0
        child.close_handles()
    finally:
        trigger.join(timeout=3);tree.close()

def test_mixed_abi_extensions_are_not_relabelled_abi3(monkeypatch):
    from setuptools import Extension, Distribution
    import setuptools.command.bdist_wheel
    config=setup_config(monkeypatch)
    monkeypatch.setattr(setuptools.command.bdist_wheel.bdist_wheel,'get_tag',lambda self:('cp314','cp314','win_amd64'))
    extension=Extension('not_limited', ['not_built.cpp'], py_limited_api=False)
    command=config['cmdclass']['bdist_wheel'](Distribution({'ext_modules':[extension]}))
    assert command.get_tag()==('cp314','cp314','win_amd64')


def test_planning_helper_disable_overrides_request(monkeypatch):
    config=setup_config(monkeypatch,BUILD_PLANNING_NATIVE='1',DISABLE_PLANNING_NATIVE='1')
    assert config['ext_modules']==[]


def test_both_extensions_are_independent_limited_abi(monkeypatch):
    config=setup_config(monkeypatch,BUILD_NATIVE='1',BUILD_PLANNING_NATIVE='1')
    if sys.platform != 'win32':
        # Setup intentionally omits the Windows resource helper on POSIX.
        assert {e.name for e in config['ext_modules']} == {'anygeometry._core'}
        assert all(e.py_limited_api for e in config['ext_modules'])
        return
    assert {e.name for e in config['ext_modules']}=={'anygeometry._core','anygeometry._planning_native_core'}
    assert all(e.py_limited_api for e in config['ext_modules'])


@windows_owned_test
def test_binary_identity_matches_loaded_resource_helper(tmp_path):
    import hashlib
    from anygeometry._planning_native import identity,load
    info=identity();native=load()
    assert info['binary_sha256']==hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
    assert 'limited=0x030B0000' in info['build'] and 'gil=required' in info['build']

@windows_owned_test
def test_poisoned_owner_refuses_new_native_acquisitions(tmp_path):
    from anygeometry._planning_process import OwnedTree
    tree=OwnedTree(tmp_path)
    child_index=tree.api.reserve()
    # Successful close followed by injected uncertain return models a reused raw
    # value: this native owner must never make another acquiring attempt.
    tree.api.inject_fault('CloseHandle',6,after=True)
    with pytest.raises(OSError): tree.close()
    native=tree.api.native;owner=tree.api.owner
    with pytest.raises(OSError): native.reserve_child(owner)
    with pytest.raises(OSError): native.job_create(owner)
    with pytest.raises(OSError):
        native.create(owner,child_index,sys.executable,sys.executable,'\0\0',str(tmp_path),str(tmp_path/'never-out'),str(tmp_path/'never-err'))
    assert not (tmp_path/'never-out').exists() and not (tmp_path/'never-err').exists()


@windows_owned_test
def test_owned_member_observation_handle_is_registered(tmp_path,monkeypatch):
    import ctypes
    from ctypes import wintypes as W
    tree,child=_sleeping_child(tmp_path,monkeypatch,'.3')
    handle=tree.api.OpenProcess(0x1000,False,child.pid)
    assert handle
    try:
        member=W.BOOL()
        assert tree.api.IsProcessInJob(handle,tree.handle,ctypes.byref(member)) and member.value
        assert tree.api.CloseHandle(handle)
        handle=None
    finally:
        if handle: tree.api.CloseHandle(handle)
        tree.close()

@windows_owned_test
def test_child_poison_is_sticky_on_native_owner(tmp_path,monkeypatch):
    tree,child=_sleeping_child(tmp_path,monkeypatch,'.3')
    tree.api.inject_fault('CloseHandle',6,after=True)
    assert not tree.api.close_child(child.index,0)
    assert tree.api.state()['poisoned']
    with pytest.raises(OSError): tree.api.reserve()
    with pytest.raises(OSError):
        tree.api.native.create(tree.api.owner,child.index,sys.executable,sys.executable,'\0\0',str(tmp_path),str(tmp_path/'no-out'),str(tmp_path/'no-err'))
    with pytest.raises(OSError): tree.close()
    assert tree.handle is None and not (tmp_path/'no-out').exists()


@windows_owned_test
def test_closed_slot_key_cannot_close_another_childs_reused_value(tmp_path,monkeypatch):
    tree,first=_sleeping_child(tmp_path,monkeypatch,'.3')
    second=tree.spawn('worker',tmp_path/'second',5)
    try:
        assert tree.api.close_child(first.index,0)
        other=second.info.process
        # A raw value can name a new resource after close. The original slot key
        # remains retired and cannot redirect the operation to that new resource.
        tree.api.native.close(tree.api.owner,other,first.index,0)
        assert second.info.process==other and second.wait(timeout=3)==0
    finally: tree.close()

@pytest.mark.parametrize('phase',['io','thread'])
@windows_owned_test
def test_reentrant_retirement_during_child_close_has_no_ambient_slot(tmp_path,monkeypatch,phase):
    from anygeometry._planning_process import OwnedTree
    tree=OwnedTree(tmp_path);actual=tree.api.CloseHandle;triggered=[]
    def interrupted(handle):
        target=None
        if tree.children:
            child=tree.children[0]
            target=(child._duplicates[0] if child._duplicates else None) if phase=='io' else child.info.thread
        result=actual(handle)
        if not triggered and target and handle==target:
            triggered.append(int(handle))
            # Signal/reentrant retirement must not inherit the child operation's
            # slot selection when it reaches the Job close.
            tree.close()
            raise KeyboardInterrupt('interrupt at completed child operation')
        return result
    monkeypatch.setattr(tree.api,'CloseHandle',interrupted)
    try:
        with pytest.raises(KeyboardInterrupt):tree.spawn('worker',tmp_path/'child',5)
        assert triggered and tree.handle is None
        for child in tree.children:
            assert child.info.process is None and child.info.thread is None and not child._duplicates
    finally:
        monkeypatch.setattr(tree.api,'CloseHandle',actual)
        # Baseline may have retained ambient routing; clear only the obsolete
        # test context so its tracked resources are retired after the assertion.
        if hasattr(tree.api,'_local'):tree.api._local.slot=None
        tree.close()


@pytest.mark.parametrize('method,attribute',[('close_child','slot'),('resume_child','resume_index')])
@windows_owned_test
def test_interrupt_at_previous_routing_assignment_does_not_taint_retirement(tmp_path,monkeypatch,method,attribute):
    import inspect,sys
    from anygeometry._planning_process import OwnedTree
    from anygeometry._planning_windows_launch import WindowsChild
    tree=OwnedTree(tmp_path)
    # An allocated observation gives close_child a real native slot without
    # executing a planner. Use suspended native creation through the adapter.
    directory=tmp_path/'child';directory.mkdir()
    child=WindowsChild(tree.api,[sys.executable,'-B','-c','pass'])
    tree.children.append(child)
    child.create(tree.handle,directory,dict(os.environ),directory/'stdout',directory/'stderr')
    source,start=inspect.getsourcelines(getattr(type(tree.api),method))
    # Old code: interrupt immediately after ambient assignment and before try.
    # New code: interrupt immediately before its single argument seam call.
    boundary=next((start+i+1 for i,line in enumerate(source) if 'self._local.'+attribute+' =' in line),None)
    if boundary is None:
        boundary=next(start+i for i,line in enumerate(source) if 'return self.CloseHandle(' in line or 'return self.ResumeThread(' in line)
    function=getattr(type(tree.api),method).__code__
    def interrupt(frame,event,arg):
        if frame.f_code is function and event=='line' and frame.f_lineno==boundary:
            sys.settrace(None);raise KeyboardInterrupt('routing publication boundary')
        return interrupt
    try:
        sys.settrace(interrupt)
        with pytest.raises(KeyboardInterrupt):
            tree.api.close_child(child.index,1) if method=='close_child' else tree.api.resume_child(child.index)
        sys.settrace(None)
        # Avoid exact planner command preflight: the test child has never run;
        # native termination is explicit and scoped to this Job before close.
        assert tree.api.TerminateJobObject(tree.handle,1)
        child.wait(timeout=3)
        tree.close()
        assert tree.handle is None and child.info.process is None and child.info.thread is None
    finally:
        sys.settrace(None)
        if hasattr(tree.api,'_local'):
            tree.api._local.slot=None;tree.api._local.resume_index=-1
        tree.close()

def test_slot_token_is_immutable_and_preserves_native_identity():
    from anygeometry._planning_native import _SlotHandle
    token=_SlotHandle(1234,7,1)
    assert int(token)==1234 and token==1234 and not token!=1234
    assert token.slot==(7,1) and hash(token)==hash(1234)
    assert token!=_SlotHandle(1234,8,1)
    assert not hasattr(token,'__dict__')
    with pytest.raises(AttributeError):token.slot=(8,1)
    with pytest.raises(TypeError):token[1]=8


@windows_owned_test
def test_reentrant_retirement_from_resume_preserves_job_identity(tmp_path,monkeypatch):
    from anygeometry._planning_process import OwnedTree
    tree=OwnedTree(tmp_path);actual=tree.api.ResumeThread;seen=[]
    def interrupted(handle):
        result=actual(handle)
        assert result==1
        seen.append((int(handle),handle.slot))
        tree.close()
        raise KeyboardInterrupt('after native resume')
    monkeypatch.setattr(tree.api,'ResumeThread',interrupted)
    try:
        with pytest.raises(KeyboardInterrupt):tree.spawn('worker',tmp_path/'child',5)
        assert seen and seen[0][1]==(tree.children[0].index,1)
        assert tree.handle is None and tree.children[0].info.process is None
    finally:
        monkeypatch.setattr(tree.api,'ResumeThread',actual);tree.close()


def test_platform_adapter_initial_refusal_is_real_and_resource_free(tmp_path, monkeypatch, request):
    """Scoped POSIX-branch unit; this does not claim an actual POSIX run."""
    from types import SimpleNamespace
    import _planning_platform_contract as contract
    from anygeometry import _planning_process as process
    shim = SimpleNamespace(name='posix', environ={})
    monkeypatch.setattr(contract, 'os', shim)
    monkeypatch.setattr(process, 'os', shim)
    calls = []
    @windows_owned_test
    def original(tmp_path):
        calls.append('original')
        pytest.fail('unsupported host entered the native positive body')
    node = SimpleNamespace(user_properties=[])
    original(tmp_path=tmp_path, request=SimpleNamespace(node=node))
    assert calls == [] and tuple(tmp_path.iterdir()) == ()
    assert node.user_properties == [('execution_scope', 'unsupported-platform-refusal')]
    request.node.user_properties.append(('execution_scope', 'simulated-POSIX-adapter-unit'))


def test_platform_adapter_required_lane_cannot_classify_refusal(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import _planning_platform_contract as contract
    monkeypatch.setattr(contract, 'os', SimpleNamespace(name='posix', environ={
        'ANYGEOMETRY_REQUIRE_QUALIFIED_PLANNING_TESTS': '1'}))
    @windows_owned_test
    def original(tmp_path):
        pytest.fail('unsupported host entered the native positive body')
    node = SimpleNamespace(user_properties=[])
    with pytest.raises(pytest.fail.Exception, match='^qualified planning test lane requires original Windows assertions$'):
        original(tmp_path=tmp_path, request=SimpleNamespace(node=node))
    assert node.user_properties == [] and tuple(tmp_path.iterdir()) == ()
