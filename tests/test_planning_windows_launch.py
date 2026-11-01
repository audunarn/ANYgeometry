"""W0 real Windows transaction negatives; no model execution."""
import ctypes
from ctypes import wintypes as W
import json
import subprocess

import pytest

from anygeometry._planning_process import OwnedTree
from anygeometry._planning_qualification import QualificationDecline
from anygeometry._planning_executor import PlannerCleanupError


def zero(tree, folder):
    tree.close()
    report=json.loads((folder/'cleanup.json').read_text())
    assert report['verified'] and report['active_processes']==0 and tree.handle is None
    for child in tree.children:
        assert child.info.process is None and child.info.thread is None and not child._duplicates


def test_launch_uses_explicit_win32_transaction_not_popen(tmp_path,monkeypatch):
    tree=OwnedTree(tmp_path)
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**k:pytest.fail('plain Popen launch'))
    try:
        child=tree.spawn('worker',tmp_path/'worker',10)
        assert child.pid in tree._pids()
    finally:zero(tree,tmp_path)


def test_owned_job_membership_precedes_primary_thread_resume(tmp_path,monkeypatch):
    tree=OwnedTree(tmp_path);events=[];resume=tree.api.ResumeThread
    resume.argtypes=[W.HANDLE];resume.restype=W.DWORD
    def check(thread):
        assert len(tree.children)==1
        child=tree.children[0]
        assert child.pid in tree._pids() and child.poll() is None
        events.append(child.pid)
        result=resume(thread)
        assert result==1
        return result
    monkeypatch.setattr(tree.api,'ResumeThread',check)
    try:
        child=tree.spawn('worker',tmp_path/'worker',10)
        assert events==[child.pid]
    finally:zero(tree,tmp_path)


def test_create_failure_cannot_publish_or_leave_child(tmp_path,monkeypatch):
    tree=OwnedTree(tmp_path)
    def fail(*a):ctypes.set_last_error(5);return 0
    monkeypatch.setattr(tree.api,'CreateProcessW',fail)
    try:
        with pytest.raises(OSError) as caught:tree.spawn('worker',tmp_path/'worker',10)
        assert caught.value.winerror==5
    finally:zero(tree,tmp_path)


@pytest.mark.parametrize('returned',[0xffffffff,0,2])
def test_resume_failure_or_unexpected_suspend_count_retires(tmp_path,monkeypatch,returned):
    tree=OwnedTree(tmp_path)
    def fail(*a):ctypes.set_last_error(5);return returned
    monkeypatch.setattr(tree.api,'ResumeThread',fail)
    try:
        with pytest.raises((OSError,QualificationDecline)):
            tree.spawn('worker',tmp_path/'worker',10)
    finally:zero(tree,tmp_path)


def test_failed_job_handle_close_is_not_silently_accepted(tmp_path,monkeypatch):
    tree=OwnedTree(tmp_path);real=tree.api.CloseHandle;handle=tree.handle
    retried=False
    def fail(value):
        number=value.value if hasattr(value,'value') else value
        if number==handle:ctypes.set_last_error(5);return 0
        return real(value)
    monkeypatch.setattr(tree.api,'CloseHandle',fail)
    try:
        with pytest.raises(OSError):tree.close()
        assert tree.handle==handle
        failure=json.loads((tmp_path/'handle-cleanup-error.json').read_text())
        assert failure['verified'] is False and failure['job_handle_retained'] is True
        # The successful receipt certifies process retirement only. The handle
        # failure is terminal and remains as a separate preserved error record.
        retired=json.loads((tmp_path/'cleanup.json').read_text())
        assert retired['verified'] is True and retired['active_processes']==0
        before=(tmp_path/'handle-cleanup-error.json').read_bytes()
        monkeypatch.setattr(tree.api,'CloseHandle',real)
        zero(tree,tmp_path)
        retried=True
        assert tree.handle is None and (tmp_path/'handle-cleanup-error.json').read_bytes()==before
    finally:
        monkeypatch.setattr(tree.api,'CloseHandle',real)
        # Base incorrectly drops the handle even when native CloseHandle fails.
        # This test owns that exact recorded handle and restores only for cleanup.
        if not retried and tree.handle is None:tree.handle=handle
        zero(tree,tmp_path)


@pytest.mark.parametrize('operation',['duplicate','size','initialize','job-attribute','handle-attribute','membership-query','membership-false'])
def test_native_creation_faults_fail_closed(tmp_path,monkeypatch,operation):
    tree=OwnedTree(tmp_path)
    def fail(*a):ctypes.set_last_error(5);return 0
    if operation=='duplicate':monkeypatch.setattr(tree.api,'DuplicateHandle',fail)
    elif operation in ('size','initialize'):
        actual=tree.api.InitializeProcThreadAttributeList
        actual.argtypes=[ctypes.c_void_p,W.DWORD,W.DWORD,ctypes.POINTER(ctypes.c_size_t)];actual.restype=W.BOOL
        def initialize(buffer,*a):
            if (operation=='size' and buffer is None) or (operation=='initialize' and buffer is not None):return fail()
            return actual(buffer,*a)
        monkeypatch.setattr(tree.api,'InitializeProcThreadAttributeList',initialize)
    elif operation in ('job-attribute','handle-attribute'):
        actual=tree.api.UpdateProcThreadAttribute
        actual.argtypes=[ctypes.c_void_p,W.DWORD,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_void_p];actual.restype=W.BOOL
        def update(buffer,flags,kind,*a):
            if kind==(0x2000d if operation=='job-attribute' else 0x20002):return fail()
            return actual(buffer,flags,kind,*a)
        monkeypatch.setattr(tree.api,'UpdateProcThreadAttribute',update)
    elif operation=='membership-query':monkeypatch.setattr(tree.api,'IsProcessInJob',fail)
    else:
        def absent(process,job,member):ctypes.cast(member,ctypes.POINTER(W.BOOL))[0]=False;return 1
        monkeypatch.setattr(tree.api,'IsProcessInJob',absent)
    try:
        with pytest.raises((OSError,QualificationDecline)):tree.spawn('worker',tmp_path/'worker',10)
    finally:zero(tree,tmp_path)


@pytest.mark.parametrize('which',['role','directory'])
def test_unknown_role_or_directory_refuses_before_creation(tmp_path,monkeypatch,which):
    root=tmp_path/'owned';root.mkdir();tree=OwnedTree(root)
    monkeypatch.setattr(tree.api,'CreateProcessW',lambda *a:pytest.fail('invalid request created a child'))
    destination=tmp_path/'outside' if which=='directory' else root/'inside'
    try:
        with pytest.raises(QualificationDecline):tree.spawn('unknown' if which=='role' else 'worker',destination,10)
        assert not destination.exists()
    finally:zero(tree,root)


@pytest.mark.parametrize('kind',['io','thread','process'])
def test_failed_child_handle_cleanup_retains_handle_and_error_evidence(tmp_path,monkeypatch,kind):
    tree=OwnedTree(tmp_path);real=tree.api.CloseHandle;failed=[]
    def close(value):
        handle=value.value if hasattr(value,'value') else value
        if tree.children:
            child=tree.children[0]
            target=(child._duplicates[0] if child._duplicates else None) if kind=='io' else getattr(child.info,kind)
            if target and handle==target:
                failed.append(handle);ctypes.set_last_error(5);return 0
        return real(value)
    monkeypatch.setattr(tree.api,'CloseHandle',close)
    try:
        with pytest.raises((OSError,PlannerCleanupError)):
            tree.spawn('worker',tmp_path/'worker',10)
            tree.close()
        assert failed and tree.handle is None
        child=tree.children[0]
        assert child._duplicates if kind=='io' else getattr(child.info,kind)
        failure=json.loads((tmp_path/'handle-cleanup-error.json').read_text())
        assert failure['verified'] is False and failure['errors']
        before=(tmp_path/'handle-cleanup-error.json').read_bytes()
        monkeypatch.setattr(tree.api,'CloseHandle',real)
        zero(tree,tmp_path)
        assert not child._duplicates and child.info.thread is None and child.info.process is None
        assert (tmp_path/'handle-cleanup-error.json').read_bytes()==before
    finally:
        monkeypatch.setattr(tree.api,'CloseHandle',real)
        zero(tree,tmp_path)


@pytest.mark.parametrize('environment',[{1:'x'},{'X':1},{'X':'a\0b'},{'a=b':'x'},{'a\0b':'x'}])
def test_invalid_environment_cannot_reach_native_creation(tmp_path,monkeypatch,environment):
    from anygeometry._planning_windows_launch import WindowsChild
    tree=OwnedTree(tmp_path);actual=WindowsChild.create
    monkeypatch.setattr(tree.api,'CreateProcessW',lambda *a:pytest.fail('invalid environment reached native create'))
    def create(self,job,directory,original,out,err):return actual(self,job,directory,environment,out,err)
    monkeypatch.setattr(WindowsChild,'create',create)
    try:
        with pytest.raises(QualificationDecline):tree.spawn('worker',tmp_path/'worker',10)
    finally:zero(tree,tmp_path)


@pytest.mark.parametrize('kind',['all-zero','missing-pid'])
def test_incomplete_native_creation_record_refuses_and_retires(tmp_path,monkeypatch,kind):
    from anygeometry._planning_windows_launch import _bind,ProcessInfo
    tree=OwnedTree(tmp_path);_bind(tree.api);actual=tree.api.CreateProcessW
    def create(*a):
        if kind=='all-zero':return 1
        result=actual(*a)
        if result:ctypes.cast(a[-1],ctypes.POINTER(ProcessInfo)).contents.pid=0
        return result
    monkeypatch.setattr(tree.api,'CreateProcessW',create)
    try:
        with pytest.raises(QualificationDecline,match='incomplete owned process creation record'):
            tree.spawn('worker',tmp_path/'worker',10)
    finally:zero(tree,tmp_path)


def test_never_created_child_poll_wait_and_close_are_safe(tmp_path):
    from anygeometry._planning_windows_launch import WindowsChild
    tree=OwnedTree(tmp_path);child=WindowsChild(tree.api,['never-created'])
    tree.children.append(child)
    try:
        assert child.pid is None and child.poll() is None and child.wait(timeout=0) is None
    finally:zero(tree,tmp_path)
