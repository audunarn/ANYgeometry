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


def _strict_memory_failure(error, cleanup, admission):
    """Validate a measured cap breach, never an arbitrary cleanup exception."""
    from anygeometry._planning_executor import PlannerCleanupError
    from anygeometry._planning_qualification import QualificationDecline
    cap = 512 * 1024**2
    def measured(row):
        assert type(row) is dict and set(row) == {'requested_bytes', 'limit_bytes', 'flags', 'peak_job_bytes', 'peak_process_bytes'}
        assert all(type(value) is int and value >= 0 for value in row.values())
        assert row['requested_bytes'] == row['limit_bytes'] == cap and row['flags'] == 0x2200
        return row
    measured(admission)
    assert cleanup['verified'] is True and cleanup['active_processes'] == 0
    observed = measured(cleanup['memory_observation'])
    assert observed['peak_job_bytes'] > cap
    assert cleanup['memory_phase'] == 'after_cleanup'
    assert cleanup['memory_failure'] == dict(operation=None, winerror=None, requested_bytes=cap)
    assert type(error) is PlannerCleanupError and str(error) == 'session planner cleanup failed'
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
        assert (type(current) is PlannerCleanupError and str(current) == 'session planner cleanup failed' or
                type(current) is QualificationDecline and str(current) in allowed)
        pending.extend(value for value in (current.__cause__, current.__context__) if value is not None)
    return observed


def assert_unsupported512_terminal(error, *, reuse, control, source, document, calls, request):
    """Only a withdrawn nonreuse cap can record a measured resource terminal."""
    from anygeometry._planning_executor import read_packet
    from anygeometry.serialization import to_dict
    assert reuse is False and control.reuse_workers is False
    assert calls == ['candidate'], 'no serial fallback or candidate retry is allowed'
    assert control.mode == 'terminal' and control.ci_verified is False
    assert to_dict(source) == document
    folders = [control.directory / name for name in ('pair', 'face') if (control.directory / name).exists()]
    assert folders and not list(control.directory.rglob('handle-cleanup-error*.json'))
    failures = []
    for folder in folders:
        cleanup = json.loads((folder / 'cleanup.json').read_text())
        workers = json.loads((folder / 'workers-cleanup.json').read_text())
        assert cleanup['verified'] is True and cleanup['active_processes'] == 0
        assert workers['verified'] is True and workers['active_processes'] == 0
        if 'memory_error' not in cleanup:
            assert cleanup['memory']['requested_bytes'] == cleanup['memory']['limit_bytes'] == 512 * 1024**2
            assert cleanup['memory']['flags'] == 0x2200
            assert cleanup['memory']['peak_job_bytes'] <= 512 * 1024**2
            continue
        admission = json.loads((folder / 'memory-limit.json').read_text())
        observed = _strict_memory_failure(error, cleanup, admission)
        result = read_packet(folder / 'result.json', 'result')
        _, session = read_packet(folder / 'session.json', 'session', with_digest=True)
        _, command = read_packet(folder / 'command-0.json', 'session-command', with_digest=True)
        assert result['status'] == 'declined' and result['reason'] == 'QualificationDecline: MemoryError: '
        assert result['session_sha256'] == session and result['request_sha256'] == command
        failures.append(dict(stage=folder.name, observation=observed))
    assert len(failures) == 1
    request.node.user_properties[:] = [(key, value) for key, value in request.node.user_properties if key != 'execution_scope']
    request.node.user_properties.append(('execution_scope', 'unsupported512-terminal'))
    (control.directory / 'unsupported512-terminal.json').write_text(json.dumps(dict(
        schema='unsupported512-resource-safety-v1', execution_scope='unsupported512-terminal',
        memory_bound_accepted=False, resource_cleanup_verified=True, active_processes=0,
        parallel_qualified=False, calls=calls, input_unchanged=True, failures=failures), indent=2))
