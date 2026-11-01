"""Retirement failure permits cleanup retry, never a new owned launch."""
import ctypes
import json

import pytest

from anygeometry._planning_process import OwnedTree
from anygeometry._planning_qualification import QualificationDecline


@pytest.mark.parametrize('terminate_fails',[False,True])
def test_retained_job_after_failed_close_cannot_launch(tmp_path,monkeypatch,terminate_fails):
    tree=OwnedTree(tmp_path)
    close=tree.api.CloseHandle;terminate=tree.api.TerminateJobObject;target=tree.handle
    def failed_close(value):
        handle=value.value if hasattr(value,'value') else value
        if handle==target:ctypes.set_last_error(5);return 0
        return close(value)
    def failed_terminate(*args):ctypes.set_last_error(5);return 0
    monkeypatch.setattr(tree.api,'CloseHandle',failed_close)
    if terminate_fails:monkeypatch.setattr(tree.api,'TerminateJobObject',failed_terminate)
    try:
        with pytest.raises(OSError):tree.close()
        assert tree.handle==target
        monkeypatch.setattr(tree.api,'CreateProcessW',lambda *a:pytest.fail('retiring Job reached native creation'))
        with pytest.raises(QualificationDecline):tree.spawn('worker',tmp_path/'late',5)
        assert not (tmp_path/'late').exists()
    finally:
        monkeypatch.setattr(tree.api,'CloseHandle',close)
        monkeypatch.setattr(tree.api,'TerminateJobObject',terminate)
        tree.close()
    assert tree.handle is None
    assert json.loads((tmp_path/'cleanup.json').read_text())['active_processes']==0


def test_successfully_closed_child_handles_are_not_reclosed_on_job_retry(tmp_path,monkeypatch):
    tree=OwnedTree(tmp_path);child=tree.spawn('worker',tmp_path/'worker',10)
    process_handle=child.info.process;job_handle=tree.handle;real=tree.api.CloseHandle
    calls=[];fail_job=[True]
    def close(value):
        handle=value.value if hasattr(value,'value') else value
        calls.append(handle)
        if handle==job_handle and fail_job[0]:ctypes.set_last_error(5);return 0
        return real(value)
    monkeypatch.setattr(tree.api,'CloseHandle',close)
    try:
        with pytest.raises(OSError):tree.close()
        assert tree.handle==job_handle and child.info.process is None and child.info.thread is None
        assert not child._duplicates and calls.count(process_handle)==1
        before=len(calls);fail_job[0]=False
        tree.close()
        assert calls[before:]==[job_handle]
        assert tree.handle is None and calls.count(process_handle)==1
    finally:
        fail_job[0]=False;tree.close()
