"""Bounded diagnostic copies only; no native/resource qualification."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import _planning_platform_contract as contract


@pytest.fixture
def attempt(tmp_path, monkeypatch):
    root=tmp_path/'checkout';(root/'tests').mkdir(parents=True)
    monkeypatch.setattr(contract,'__file__',str(root/'tests/_planning_platform_contract.py'))
    monkeypatch.setenv('CI','true')
    directory=tmp_path/'owned'/'geometry';folder=directory/'session';folder.mkdir(parents=True)
    names=('cleanup.json','workers-cleanup.json','memory-limit.json','session.json',
           'command-0.json','result.json','plan.json','command-1.json','command-2.json')
    for name in names:(folder/name).write_bytes(('opaque unvalidated '+name).encode())
    request=SimpleNamespace(getfixturevalue=lambda name:tmp_path/'owned')
    return root,folder,SimpleNamespace(directory=directory),request,names


def test_fixed_attempt_bytes_are_unvalidated_and_unknown_files_not_exported(attempt):
    root,folder,control,request,names=attempt
    (folder/'private-unrelated.txt').write_text('not exported')
    (folder/'nested').mkdir();(folder/'nested'/'result.json').write_text('not exported')
    target=contract._retain_unvalidated512_attempt(control,reuse=True,request=request)
    assert target.is_relative_to(root/'ci-isolation/unsupported512')
    summary=json.loads((target/'summary.json').read_text())
    assert summary['status']=='UNVALIDATED'
    assert summary['parallel_qualified'] is False and summary['memory_capacity_qualified'] is False
    assert summary['cleanup_verified'] is None
    assert {p.name for p in (target/'session').iterdir()}==set(names)
    assert len(summary['retained_files'])==9
    for entry in summary['retained_files']:
        raw=(folder/Path(entry['path']).name).read_bytes()
        assert (target/entry['path']).read_bytes()==raw
        assert entry['bytes']==len(raw) and entry['sha256']==hashlib.sha256(raw).hexdigest()


def test_missing_optional_files_are_reported_without_fabrication(attempt):
    _,folder,control,request,_=attempt
    (folder/'command-1.json').unlink();(folder/'command-2.json').unlink()
    target=contract._retain_unvalidated512_attempt(control,reuse=True,request=request)
    summary=json.loads((target/'summary.json').read_text())
    assert summary['missing_files']==['session/command-1.json','session/command-2.json']
    assert len(summary['retained_files'])==7


def test_non_ci_does_not_create_an_attempt(attempt,monkeypatch):
    root,_,control,request,_=attempt
    monkeypatch.delenv('CI',raising=False)
    assert contract._retain_unvalidated512_attempt(control,reuse=True,request=request) is None
    assert not (root/'ci-isolation').exists()


@pytest.mark.parametrize('fault',('outside','private','folder-symlink','file-symlink','empty','file-bound','total-bound'))
def test_unvalidated_attempt_guards_remain_strict(attempt,monkeypatch,fault,tmp_path):
    _,folder,control,request,names=attempt
    if fault=='outside':request.getfixturevalue=lambda name:tmp_path/'unrelated'
    elif fault=='private':
        private=tmp_path/'owned'/'.pp1-evidence'/'geometry';(private/'session').mkdir(parents=True)
        control.directory=private
    elif fault in ('folder-symlink','file-symlink'):
        # Guard-unit facet only, not a claim of OS symlink/native qualification.
        selected=folder if fault=='folder-symlink' else folder/'session.json'
        original=Path.is_symlink
        monkeypatch.setattr(Path,'is_symlink',lambda path:path==selected or original(path))
    elif fault=='empty':(folder/'session.json').write_bytes(b'')
    elif fault=='file-bound':(folder/'session.json').write_bytes(b'x'*8_100_001)
    else:
        for name in names[:5]:(folder/name).write_bytes(b'x'*6_500_000)
    with pytest.raises(AssertionError):
        contract._retain_unvalidated512_attempt(control,reuse=True,request=request)


@pytest.mark.parametrize('strict_failure',(False,True))
def test_retention_failure_never_masks_strict_failure_or_creates_a_pass(monkeypatch,strict_failure):
    retention=OSError('owned snapshot failed');original=AssertionError('strict digest mismatch')
    def retain(*args,**kwargs):raise retention
    def strict(*args,**kwargs):
        if strict_failure:raise original
    monkeypatch.setattr(contract,'_retain_unvalidated512_attempt',retain)
    monkeypatch.setattr(contract,'_assert_unsupported512_terminal',strict)
    with pytest.raises(AssertionError if strict_failure else OSError) as caught:
        contract.assert_unsupported512_terminal(None,reuse=True,control=None,source=None,
            document=None,calls=None,request=None)
    assert caught.value is (original if strict_failure else retention)
    if strict_failure:assert 'UNVALIDATED attempt retention failed' in original.__notes__[0]
