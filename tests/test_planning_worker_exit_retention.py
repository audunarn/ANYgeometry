"""UNVALIDATED owned worker file copies; no resource qualification."""
import hashlib,json
from pathlib import Path
from types import SimpleNamespace
import pytest
import _planning_platform_contract as contract

WORKER_FILES=('stdout.txt','stderr.txt','result.json','session.json','request.json','limits.json',
              'command-0.json','command-1.json','command-2.json')

@pytest.fixture
def attempt(tmp_path,monkeypatch):
    root=tmp_path/'checkout';(root/'tests').mkdir(parents=True)
    monkeypatch.setattr(contract,'__file__',str(root/'tests/_planning_platform_contract.py'))
    monkeypatch.setenv('CI','true')
    base=tmp_path/'owned';parent=base/'geometry/session';parent.mkdir(parents=True)
    (parent/'session.json').write_bytes(b'opaque parent setup')
    for index in range(2):
        worker=parent/f'worker-{index}';worker.mkdir()
        for name in WORKER_FILES:
            (worker/name).write_bytes(b'' if name=='stdout.txt' else ('unvalidated '+name).encode())
    control=SimpleNamespace(directory=parent.parent)
    request=SimpleNamespace(getfixturevalue=lambda name:base)
    return root,parent,control,request


def test_fixed_two_worker_bytes_and_empty_logs_are_supporting_only(attempt):
    root,parent,control,request=attempt
    (parent/'worker-2').mkdir();(parent/'worker-2/stderr.txt').write_bytes(b'not owned by fixed two-worker export')
    (parent/'worker-0/private.txt').write_bytes(b'private')
    (parent/'worker-0/nested').mkdir();(parent/'worker-0/nested/result.json').write_bytes(b'not exported')
    target=contract._retain_unvalidated512_attempt(control,reuse=True,request=request)
    summary=json.loads((target/'summary.json').read_text())
    assert summary['status']=='UNVALIDATED' and summary['parallel_qualified'] is False
    assert summary['memory_capacity_qualified'] is False and summary['cleanup_verified'] is None
    assert summary['worker_missing_files']==[]
    for index in range(2):
        assert {p.name for p in (target/'session'/f'worker-{index}').iterdir()}==set(WORKER_FILES)
        for name in WORKER_FILES:
            assert (target/'session'/f'worker-{index}'/name).read_bytes()==(parent/f'worker-{index}'/name).read_bytes()
    assert not (target/'session/worker-2').exists()
    assert len(summary['retained_files'])==1+2*len(WORKER_FILES)
    for row in summary['retained_files']:
        raw=(target/row['path']).read_bytes()
        assert row['bytes']==len(raw) and row['sha256']==hashlib.sha256(raw).hexdigest()


def test_worker_missing_is_separate_from_parent_and_never_fabricated(attempt):
    _,parent,control,request=attempt
    (parent/'worker-0/command-2.json').unlink()
    for name in WORKER_FILES:(parent/'worker-1'/name).unlink()
    (parent/'worker-1').rmdir()
    target=contract._retain_unvalidated512_attempt(control,reuse=True,request=request)
    summary=json.loads((target/'summary.json').read_text())
    assert summary['worker_missing_files']==['session/worker-0/command-2.json','session/worker-1/']
    assert 'session/command-2.json' in summary['missing_files']
    assert not (target/'session/worker-0/command-2.json').exists()
    assert not (target/'session/worker-1').exists()


@pytest.mark.parametrize('fault',('worker-symlink','file-symlink','wrong-folder','empty-packet','file-bound','total-bound'))
def test_worker_copy_guards(attempt,monkeypatch,tmp_path,fault):
    _,parent,control,request=attempt
    if fault in ('worker-symlink','file-symlink'):
        # Guard-unit facet, not an OS/native symlink qualification.
        selected=parent/'worker-0' if fault=='worker-symlink' else parent/'worker-0/stderr.txt'
        original=Path.is_symlink
        monkeypatch.setattr(Path,'is_symlink',lambda path:path==selected or original(path))
    elif fault=='wrong-folder':request.getfixturevalue=lambda name:tmp_path/'unrelated'
    elif fault=='empty-packet':(parent/'worker-0/result.json').write_bytes(b'')
    elif fault=='file-bound':(parent/'worker-0/stderr.txt').write_bytes(b'x'*8_100_001)
    else:
        for index in range(2):
            for name in ('stdout.txt','stderr.txt'):(parent/f'worker-{index}'/name).write_bytes(b'x'*8_000_000)
    with pytest.raises(AssertionError):contract._retain_unvalidated512_attempt(control,reuse=True,request=request)


def test_unknown_exit1_is_still_refused_even_with_memoryerror_log(tmp_path,monkeypatch):
    from test_planning_memory_process import _controlled_terminal512
    from anygeometry import _planning_executor as executor
    from anygeometry._planning_qualification import QualificationDecline
    error,control,source,document,request=_controlled_terminal512(tmp_path,monkeypatch,True)
    root=tmp_path/'checkout';(root/'tests').mkdir(parents=True)
    monkeypatch.setattr(contract,'__file__',str(root/'tests/_planning_platform_contract.py'));monkeypatch.setenv('CI','true')
    folder=control.directory/'session';worker=folder/'worker-0';worker.mkdir()
    (worker/'stderr.txt').write_bytes(b'MemoryError: diagnostic text is not authenticated cause')
    error.__cause__.__context__=QualificationDecline('QualificationDecline: owned process exit: 1')
    result=executor.read_packet(folder/'result.json','result');result['reason']='QualificationDecline: owned process exit: 1'
    executor.write_packet(folder/'result.json',executor.pack(result,'result'))
    with pytest.raises(AssertionError):
        contract.assert_unsupported512_terminal(error,reuse=True,control=control,source=source,document=document,calls=['candidate'],request=request)
    attempts=list((root/'ci-isolation/unsupported512').iterdir());assert len(attempts)==1
    assert (attempts[0]/'session/worker-0/stderr.txt').read_bytes()==(worker/'stderr.txt').read_bytes()
    assert json.loads((attempts[0]/'summary.json').read_text())['memory_capacity_qualified'] is False
