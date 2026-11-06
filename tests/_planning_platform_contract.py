"""Run original Windows assertions or the real unsupported-host ownership refusal.

This adapter neither qualifies a numerical profile nor claims that an initial
refusal exercised native creation, workers, memory, or postcapture cancellation.
"""
from functools import wraps
import inspect
import os

import pytest


def windows_owned_test(function):
    """Preserve the test body/parametrization and add a request fixture for scope."""
    original_signature = inspect.signature(function)
    if 'tmp_path' not in original_signature.parameters:
        raise TypeError('native-platform contract needs the original tmp_path fixture')
    has_request = 'request' in original_signature.parameters
    parameters = list(original_signature.parameters.values())
    if not has_request:
        parameters.append(inspect.Parameter('request', inspect.Parameter.KEYWORD_ONLY))

    @wraps(function)
    def invoke(*args, **kwargs):
        request = kwargs['request'] if has_request else kwargs.pop('request')
        if os.name == 'nt':
            request.node.user_properties.append(('execution_scope', 'original-Windows-assertions'))
            return function(*args, **kwargs)

        if os.environ.get('ANYGEOMETRY_REQUIRE_QUALIFIED_PLANNING_TESTS') == '1':
            pytest.fail('qualified planning test lane requires original Windows assertions')
        request.node.user_properties.append(('execution_scope', 'unsupported-platform-refusal'))
        bound = original_signature.bind(*args, **kwargs)
        directory = bound.arguments['tmp_path']
        assert tuple(directory.iterdir()) == (), 'unsupported-host check needs an untouched fixture directory'
        from anygeometry._planning_process import OwnedTree
        from anygeometry._planning_qualification import QualificationDecline
        with pytest.raises(QualificationDecline) as caught:
            OwnedTree(directory)
        assert type(caught.value) is QualificationDecline
        assert str(caught.value) == 'owned planner requires Windows Job containment'
        assert tuple(directory.iterdir()) == (), 'initial refusal created owned-tree evidence or resources'

    invoke.__signature__ = original_signature.replace(parameters=parameters)
    return invoke


from contextlib import contextmanager
import ast
import json
import sys


@contextmanager
def observe_plan_calls(*, enabled):
    """Observe candidate/serial entry without replacing any production binding."""
    calls = []
    if not enabled:
        yield calls
        return
    from anygeometry import batch_intersections as batch
    code = batch._plan_intersections.__code__
    previous = sys.getprofile()
    def observe(frame, event, argument):
        if previous is not None:
            previous(frame, event, argument)
        if event == 'call' and frame.f_code is code:
            calls.append('serial' if frame.f_locals['_stages'] is None else 'candidate')
    sys.setprofile(observe)
    try:
        yield calls
    finally:
        sys.setprofile(previous)


def _strict_memory_failure(error, cleanup, admission, *, reuse=False):
    """Validate a measured cap breach, never an arbitrary cleanup exception."""
    from anygeometry._planning_executor import PlannerCleanupError
    from anygeometry._planning_qualification import QualificationDecline
    assert type(reuse) is bool
    message = 'preparation cleanup failed' if reuse else 'session planner cleanup failed'
    cap = 512 * 1024**2
    def measured(row):
        assert type(row) is dict and set(row) == {'requested_bytes', 'limit_bytes', 'flags', 'peak_job_bytes', 'peak_process_bytes'}
        assert all(type(value) is int and value >= 0 for value in row.values())
        assert row['requested_bytes'] == row['limit_bytes'] == cap and row['flags'] == 0x2200
        return row
    measured(admission)
    assert admission['peak_job_bytes'] <= cap
    assert type(cleanup['active_processes']) is int
    assert cleanup['verified'] is True and cleanup['active_processes'] == 0
    observed = measured(cleanup['memory_observation'])
    assert observed['peak_job_bytes'] > cap
    assert cleanup['memory_phase'] == 'after_cleanup'
    assert cleanup['memory_failure'] == dict(operation=None, winerror=None, requested_bytes=cap)
    assert type(error) is PlannerCleanupError and str(error) == message
    cause = error.__cause__
    assert type(cause) is QualificationDecline
    prefix = 'owned-tree memory enforcement differs from request: '
    assert str(cause).startswith(prefix)
    cause_observation = measured(ast.literal_eval(str(cause)[len(prefix):]))
    assert cap < cause_observation['peak_job_bytes'] <= observed['peak_job_bytes']
    assert cleanup['memory_error'] == 'QualificationDecline: ' + str(cause)
    pending, seen = [error], set()
    allowed = {str(cause), 'QualificationDecline: MemoryError: ', 'MemoryError: '}
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        assert (type(current) is PlannerCleanupError and str(current) == message or
                type(current) is QualificationDecline and str(current) in allowed)
        pending.extend(value for value in (current.__cause__, current.__context__) if value is not None)
    return observed


def _retain_unsupported512_evidence(summary, folders, request):
    """CI-only fixed-file copies, never a recursive scratch/evidence export."""
    if os.environ.get('CI') != 'true':return None
    import hashlib
    from pathlib import Path
    import uuid
    root=Path(__file__).resolve().parents[1]
    destination=root/'ci-isolation'/'unsupported512'
    assert not any(part in root.parts for part in ('.pp1-evidence','.ppq-evidence'))
    for path in (root/'ci-isolation',destination):
        assert not path.is_symlink()
        path.mkdir(exist_ok=True)
        assert path.resolve().is_relative_to(root)
    destination=destination/(('reuse-' if summary['reuse_workers'] else 'nonreuse-')+uuid.uuid4().hex)
    destination.mkdir(exist_ok=False)
    base=Path(request.getfixturevalue('tmp_path')).resolve()
    copied=[];total=0
    fixed=('cleanup.json','workers-cleanup.json','memory-limit.json','session.json','command-0.json','result.json')
    optional=('plan.json','command-1.json','command-2.json')
    for folder in folders:
        assert not folder.is_symlink() and folder.resolve().is_relative_to(base)
        assert not any(part in folder.parts for part in ('.pp1-evidence','.ppq-evidence'))
        target=destination/folder.name;target.mkdir()
        for name in fixed+optional:
            path=folder/name
            assert not path.is_symlink()
            if name in optional and not path.exists():continue
            assert path.is_file() and not path.is_symlink()
            assert path.resolve().parent==folder.resolve()
            size=path.stat().st_size
            assert 0<size<=2*4_000_000+100000
            assert total+size<=32_000_000
            raw=path.read_bytes();assert len(raw)==size
            (target/name).write_bytes(raw);total+=size
            copied.append(dict(path=folder.name+'/'+name,bytes=size,sha256=hashlib.sha256(raw).hexdigest()))
    (destination/'summary.json').write_text(json.dumps(dict(summary,retained_files=copied),indent=2),encoding='utf-8')
    return destination


def _retain_unvalidated512_attempt(control, *, reuse, request):
    """Fixed owned CI bytes only; none of their safety/protocol claims are accepted."""
    if os.environ.get('CI') != 'true':return None
    import hashlib
    from pathlib import Path
    import uuid
    assert type(reuse) is bool
    root=Path(__file__).resolve().parents[1]
    assert not any(part in root.parts for part in ('.pp1-evidence','.ppq-evidence'))
    raw_base=Path(request.getfixturevalue('tmp_path'))
    assert not raw_base.is_symlink()
    base=raw_base.resolve()
    destination=root/'ci-isolation'/'unsupported512'
    for path in (root/'ci-isolation',destination):
        assert not path.is_symlink()
        path.mkdir(exist_ok=True)
        assert path.resolve().is_relative_to(root)
    destination=destination/('UNVALIDATED-'+('reuse-' if reuse else 'nonreuse-')+uuid.uuid4().hex)
    destination.mkdir(exist_ok=False)
    summary=dict(schema='unsupported512-unvalidated-attempt-v1',status='UNVALIDATED',
        parallel_qualified=False,memory_capacity_qualified=False,cleanup_verified=None,
        requested_reuse_workers=reuse,copy_complete=False,retained_files=[],missing_files=[])
    (destination/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    names=('cleanup.json','workers-cleanup.json','memory-limit.json','session.json',
           'command-0.json','result.json','plan.json','command-1.json','command-2.json')
    copied=[];missing=[];worker_missing=[];total=0
    for name in (('session',) if reuse else ('pair','face')):
        folder=control.directory/name
        assert not folder.is_symlink()
        assert folder.resolve().is_relative_to(base)
        assert not any(part in folder.parts for part in ('.pp1-evidence','.ppq-evidence'))
        parent=folder
        while parent!=raw_base:
            assert not parent.is_symlink()
            assert parent.parent!=parent
            parent=parent.parent
        if not folder.exists():
            missing.append(name+'/');continue
        assert folder.is_dir()
        target=destination/name;target.mkdir()
        for filename in names:
            path=folder/filename
            assert not path.is_symlink()
            if not path.exists():
                missing.append(name+'/'+filename);continue
            assert path.is_file() and path.resolve().parent==folder.resolve()
            size=path.stat().st_size
            assert 0<size<=2*4_000_000+100000 and total+size<=32_000_000
            raw=path.read_bytes();assert len(raw)==size
            (target/filename).write_bytes(raw);total+=size
            copied.append(dict(path=name+'/'+filename,bytes=size,sha256=hashlib.sha256(raw).hexdigest()))
        # Fixed two-worker supporting bytes only: never interpret logs as cause.
        worker_names=('stdout.txt','stderr.txt','result.json','session.json','request.json','limits.json',
                      'command-0.json','command-1.json','command-2.json')
        for index in range(2):
            worker=folder/('worker-'+str(index));relative=name+'/'+worker.name
            assert not worker.is_symlink()
            assert worker.resolve().is_relative_to(base) and worker.resolve().parent==folder.resolve()
            if not worker.exists():
                worker_missing.append(relative+'/');continue
            assert worker.is_dir()
            worker_target=target/worker.name;worker_target.mkdir()
            for filename in worker_names:
                path=worker/filename
                assert not path.is_symlink()
                if not path.exists():
                    worker_missing.append(relative+'/'+filename);continue
                assert path.is_file() and path.resolve().parent==worker.resolve()
                size=path.stat().st_size
                # Empty log files are actual evidence; packets remain nonempty.
                assert (size>=0 if filename in ('stdout.txt','stderr.txt') else size>0)
                assert size<=2*4_000_000+100000 and total+size<=32_000_000
                raw=path.read_bytes();assert len(raw)==size
                (worker_target/filename).write_bytes(raw);total+=size
                copied.append(dict(path=relative+'/'+filename,bytes=size,sha256=hashlib.sha256(raw).hexdigest()))
    summary.update(copy_complete=True,retained_files=copied,missing_files=missing,worker_missing_files=worker_missing)
    (destination/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return destination


def assert_unsupported512_terminal(error, *, reuse, control, source, document, calls, request):
    """Preserve first-failure bytes without changing any strict acceptance assertion."""
    retention_error=None
    try:
        _retain_unvalidated512_attempt(control,reuse=reuse,request=request)
    except Exception as failure:
        retention_error=failure
    try:
        _assert_unsupported512_terminal(error,reuse=reuse,control=control,source=source,
                                       document=document,calls=calls,request=request)
    except BaseException as failure:
        if retention_error is not None:
            BaseException.add_note(failure,'UNVALIDATED attempt retention failed; original strict failure preserved')
        raise
    if retention_error is not None:
        raise retention_error


def _unsupported512_command_chain(folder, *, session, reuse):
    """Validate the entire finite published prefix, never any matching old digest."""
    from anygeometry._planning_executor import read_packet
    from anygeometry._planning_session import validate_command
    canonical=tuple(f'command-{i}.json' for i in range(3))
    present={}
    for path in folder.iterdir():
        if not path.name.casefold().startswith('command'):continue
        assert path.name in canonical, 'noncanonical or excess command filename'
        assert not path.is_symlink() and path.is_file()
        assert path.resolve().parent==folder.resolve()
        present[path.name]=path
    indices=[i for i,name in enumerate(canonical) if name in present]
    assert indices and indices==list(range(len(indices))), 'missing command prefix'
    first=None;seen=();previous=session;chain=[]
    for sequence in indices:
        path=present[canonical[sequence]]
        assert 0<path.stat().st_size<=2*4_000_000+100000
        body,digest=read_packet(path,'session-command',with_digest=True)
        kind=validate_command(body,session=session,sequence=sequence,previous=previous,seen=seen)
        if sequence==0:first=body
        elif not reuse:assert kind=='finish', 'nonreuse lifecycle has only one data stage'
        if kind=='finish':
            assert sequence==indices[-1], 'finish must be the last published command'
        else:
            assert body['tasks'], 'published data stage must be nonempty'
            seen+=kind,
        chain.append(dict(sequence=sequence,stage=kind,sha256=digest))
        previous=digest
    return first,previous,chain


def _assert_unsupported512_terminal(error, *, reuse, control, source, document, calls, request):
    """Measured safe terminal for either withdrawn diagnostic512 capacity claim."""
    from anygeometry._planning_executor import read_packet
    from anygeometry._planning_session import validate_command
    from anygeometry._planning_memory import MemoryBudget
    from anygeometry.serialization import to_dict
    cap=512*1024**2
    assert type(reuse) is bool and type(control.reuse_workers) is bool and control.reuse_workers is reuse
    budget=getattr(control,'memory_budget',None)
    if budget is not None:
        assert type(budget) is MemoryBudget
        assert type(budget.tree_limit_bytes) is int and budget.tree_limit_bytes==cap
    if control.tree_memory_limit is None:
        assert budget is not None
    else:
        assert type(control.tree_memory_limit) is int and control.tree_memory_limit==cap
    assert type(control.workers) is int and control.workers==2
    assert calls == ['candidate'], 'no serial fallback or candidate retry is allowed'
    assert control.mode == 'terminal' and control.ci_verified is False
    assert to_dict(source) == document
    allowed=('session',) if reuse else ('pair','face')
    names={path.name for path in control.directory.iterdir() if path.is_dir()}
    assert names and names.issubset(allowed)
    if reuse:assert names=={'session'}
    folders=[control.directory/name for name in allowed if name in names]
    assert not list(control.directory.rglob('handle-cleanup-error*.json'))
    failures=[]
    for folder in folders:
        assert not folder.is_symlink()
        cleanup=json.loads((folder/'cleanup.json').read_text())
        workers=json.loads((folder/'workers-cleanup.json').read_text())
        for record in (cleanup,workers):
            assert record['verified'] is True and type(record['active_processes']) is int and record['active_processes']==0
        assert not any(name in workers for name in ('memory_error','cleanup_error','error','errors'))
        if 'memory_error' not in cleanup:
            assert not reuse
            assert cleanup['memory']['requested_bytes']==cleanup['memory']['limit_bytes']==cap
            assert cleanup['memory']['flags']==0x2200
            assert cleanup['memory']['peak_job_bytes']<=cap
            continue
        admission=json.loads((folder/'memory-limit.json').read_text())
        observed=_strict_memory_failure(error,cleanup,admission,reuse=reuse)
        result,result_digest=read_packet(folder/'result.json','result',with_digest=True)
        setup,session=read_packet(folder/'session.json','session',with_digest=True)
        command_body,command,chain=_unsupported512_command_chain(folder,session=session,reuse=reuse)
        assert type(setup) is dict and type(setup.get('workers')) is int and setup['workers']==control.workers
        expected_stage='pair' if reuse else folder.name
        assert validate_command(command_body,session=session,sequence=0,previous=session,seen=())==expected_stage
        assert command_body['tasks']
        assert result['status']=='declined' and result['reason']=='QualificationDecline: MemoryError: '
        assert result['session_sha256']==session and result['request_sha256']==command
        failures.append(dict(stage=folder.name,observation=observed,session_sha256=session,
                             command_sha256=command,result_sha256=result_digest,command_chain=chain))
    assert len(failures)==1
    request.node.user_properties[:]=[(key,value) for key,value in request.node.user_properties if key!='execution_scope']
    request.node.user_properties.extend((('execution_scope','unsupported512-resource-safety'),
        ('resource_configuration','unsupported512-diagnostic-lifecycle'),('reuse_workers',reuse),
        ('parallel_qualified',False),('memory_capacity_qualified',False)))
    summary=dict(schema='unsupported512-resource-safety-v2',execution_scope='unsupported512-resource-safety',
        reuse_workers=reuse,memory_bound_accepted=False,resource_cleanup_verified=True,active_processes=0,
        parallel_qualified=False,calls=calls,input_unchanged=True,failures=failures)
    (control.directory/'unsupported512-terminal.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    _retain_unsupported512_evidence(summary,folders,request)
