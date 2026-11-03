"""Opus W0 follow-ups: fail before native side effects, retain failed evidence."""
from _planning_platform_contract import windows_owned_test
import ctypes
import json
from pathlib import Path

import pytest

from anygeometry._planning_process import OwnedTree
from anygeometry._planning_windows_launch import WindowsChild
from anygeometry._planning_qualification import QualificationDecline


@windows_owned_test
def test_closed_uncapped_tree_refuses_before_directory_or_native_create(tmp_path,monkeypatch):
    tree=OwnedTree(tmp_path);tree.close()
    monkeypatch.setattr(tree.api,'CreateProcessW',lambda *a:pytest.fail('closed Job reached native creation'))
    with pytest.raises(QualificationDecline):tree.spawn('worker',tmp_path/'late',5)
    assert not (tmp_path/'late').exists()


@pytest.mark.parametrize('job',[None,0])
@windows_owned_test
def test_null_job_refuses_before_membership_or_resume(tmp_path,monkeypatch,job):
    tree=OwnedTree(tmp_path);child=WindowsChild(tree.api,['uncreated'])
    monkeypatch.setattr(tree.api,'IsProcessInJob',lambda *a:pytest.fail('NULL Job queried any-Job membership'))
    try:
        with pytest.raises(QualificationDecline):child.resume(job)
    finally:tree.close()


@windows_owned_test
def test_empty_environment_key_refuses_before_native_create(tmp_path,monkeypatch):
    tree=OwnedTree(tmp_path);actual=WindowsChild.create
    monkeypatch.setattr(tree.api,'CreateProcessW',lambda *a:pytest.fail('empty environment key reached native creation'))
    monkeypatch.setattr(WindowsChild,'create',lambda self,job,directory,env,out,err:actual(self,job,directory,{'':'value'},out,err))
    try:
        with pytest.raises(QualificationDecline):tree.spawn('worker',tmp_path/'worker',5)
    finally:tree.close()


@windows_owned_test
def test_closed_job_retry_attempts_all_retained_children_and_records_error(tmp_path):
    tree=OwnedTree(tmp_path);tree.close();seen=[]
    class Child:
        def __init__(self,name):self.name=name
        def close_handles(self):
            seen.append(self.name)
            if self.name=='first':raise OSError('retained first child failed')
    tree.children=[Child('first'),Child('second')]
    with pytest.raises(OSError,match='retained first'):tree.close()
    assert seen==['first','second']
    error=json.loads((tmp_path/'handle-cleanup-error.json').read_text())
    assert error['verified'] is False and error['job_handle_retained'] is False
    tree.children=[]


@windows_owned_test
def test_repeated_job_close_failures_preserve_first_record(tmp_path,monkeypatch):
    tree=OwnedTree(tmp_path);real=tree.api.CloseHandle;target=tree.handle;code=[5]
    def fail(value):
        handle=value.value if hasattr(value,'value') else value
        if handle==target:ctypes.set_last_error(code[0]);return 0
        return real(value)
    monkeypatch.setattr(tree.api,'CloseHandle',fail)
    try:
        with pytest.raises(OSError):tree.close()
        first=(tmp_path/'handle-cleanup-error.json').read_bytes()
        code[0]=6
        with pytest.raises(OSError):tree.close()
        assert (tmp_path/'handle-cleanup-error.json').read_bytes()==first
        records=sorted(tmp_path.glob('handle-cleanup-error-*.json'))
        assert len(records)==2
        assert [json.loads(p.read_text())['errors'][0]['winerror'] for p in records]==[5,6]
    finally:
        monkeypatch.setattr(tree.api,'CloseHandle',real);tree.close()


@windows_owned_test
def test_sidecar_write_failure_still_raises_the_handle_error(tmp_path,monkeypatch):
    tree=OwnedTree(tmp_path);real=tree.api.CloseHandle;open_file=Path.open;target=tree.handle
    def fail(value):
        handle=value.value if hasattr(value,'value') else value
        if handle==target:ctypes.set_last_error(5);return 0
        return real(value)
    def no_receipt(self,mode='r',*a,**k):
        if self.name.startswith('handle-cleanup-error') and any(flag in mode for flag in 'wax'):
            raise OSError('evidence disk unavailable')
        return open_file(self,mode,*a,**k)
    monkeypatch.setattr(tree.api,'CloseHandle',fail);monkeypatch.setattr(Path,'open',no_receipt)
    try:
        with pytest.raises(OSError) as caught:tree.close()
        assert caught.value.winerror==5 and tree.handle==target
        assert caught.value.__cause__.args==('evidence disk unavailable',)
        assert json.loads((tmp_path/'cleanup.json').read_text())['active_processes']==0
    finally:
        monkeypatch.setattr(tree.api,'CloseHandle',real);monkeypatch.setattr(Path,'open',open_file);tree.close()
