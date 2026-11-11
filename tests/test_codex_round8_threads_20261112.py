"""Codex review round 8 of the thread route (2026-11-11): nothing runs in a worker before the marker.

Finding: the copied task context was entered before ``_IN_TASK`` was set, so a trace callback firing on entry to
the task body could start a stage from the worker. Class: "something runs in a worker of the route before the
marker that makes a stage there refuse". Closed with two markers that exist BEFORE the code they cover runs:
* the worker THREAD: every pool thread of the route carries the name prefix ``anygeometry-plan`` from its
  construction, i.e. before it starts; a stage refuses in such a thread (covers the worker's own frames - execute,
  the context copy, C-call profile events - from the first hook ``threading`` installs in the thread);
* the task CONTEXT: ``_IN_TASK`` is set in the copied context before the task body is entered (covers the body
  from its call event on, and any thread or callback that inherits or copies that context).
Out of contract (stated in the module docstring): a hook that runs in a worker thread before ``threading`` has
registered the thread, a worker thread that was renamed, and deliberate bypass of the guards.

Tests marked FAILS-ON-OLD fail on bb3d7ba. Bounded like the round-6 tests.
"""
import faulthandler
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from anygeometry import GeometryError
from anygeometry import _planning_threads as pt
from anygeometry.batch_intersections import IntersectionBatchPolicy
from test_planning_threads import ENV, _clear, joint

WAIT = 30
BOUND = 240
POOL = 'anygeometry-plan'
MODULE = pt.__file__


@pytest.fixture(autouse=True)
def _bounded(monkeypatch):
    _clear(monkeypatch)
    faulthandler.dump_traceback_later(BOUND, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


def _two():
    return [dict(kind='face', face=0, cost=1.), dict(kind='face', face=1, cost=2.)]


def _fine(item, check=None):
    return item['face']


def _counted_pools(monkeypatch):
    pools = []

    class Pool(ThreadPoolExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            pools.append(self)
    monkeypatch.setattr(pt, 'ThreadPoolExecutor', Pool)
    return pools


def _reenter(seen, where):
    """What a hook does once: try to run a stage right here, and to plan with the thread route."""
    if seen.get(where):
        return
    seen[where] = 'entered'
    try:
        pt.ThreadStages(2).run('face', _two(), _fine)
        seen[where] = 'a nested stage ran'
    except GeometryError as exc:
        seen[where] = str(exc)


REFUSED = 'thread planning stage inside a task of another stage is not supported'


@pytest.mark.parametrize('function', ('execute', 'body'))
def test_trace_hook_on_entry_to_a_worker_function_cannot_start_a_stage(function, monkeypatch):   # FAILS-ON-OLD
    pools, seen = _counted_pools(monkeypatch), {}

    def trace(frame, event, argument):
        if event == 'call' and frame.f_code.co_name == function and frame.f_code.co_filename == MODULE:
            _reenter(seen, function)                  # the call event: before the first line of the function
        return None
    threading.settrace(trace)
    try:
        result = pt.plan_with_threads(lambda: pt.ThreadStages(2).run('face', _two(), _fine),
                                      lambda: seen.setdefault('serial', True) and 'serial')
    finally:
        threading.settrace(None)
    assert result == [0, 1]                           # the stage itself is undisturbed, and nothing was repeated
    assert seen == {function: REFUSED} and len(pools) == 1


def test_profile_hook_on_the_first_event_of_a_worker_cannot_start_a_stage(monkeypatch):       # FAILS-ON-OLD
    pools, seen = _counted_pools(monkeypatch), {}

    def profile(frame, event, argument):
        if threading.current_thread().name.startswith(POOL):
            _reenter(seen, 'first event')             # whatever the first profiled event in a pool thread is
    threading.setprofile(profile)
    try:
        result = pt.ThreadStages(2).run('face', _two(), _fine)
    finally:
        threading.setprofile(None)
    assert result == [0, 1]
    assert seen == {'first event': REFUSED} and len(pools) == 1


def test_c_call_profile_event_of_the_context_entry_cannot_start_a_stage(monkeypatch):          # FAILS-ON-OLD
    # The worker enters the copied context through C calls (Context.run); a profile hook sees them as c_call
    # events in the worker's own context, where the context marker is not set: the thread marker refuses.
    pools, seen = _counted_pools(monkeypatch), {}

    def profile(frame, event, argument):
        if (event == 'c_call' and frame.f_code.co_name == 'execute' and frame.f_code.co_filename == MODULE
                and getattr(argument, '__name__', '') == 'run'):
            _reenter(seen, 'context entry')
    threading.setprofile(profile)
    try:
        result = pt.ThreadStages(2).run('face', _two(), _fine)
    finally:
        threading.setprofile(None)
    assert result == [0, 1]
    assert seen == {'context entry': REFUSED} and len(pools) == 1


def test_marker_is_set_before_the_body_and_in_every_worker_frame(monkeypatch):               # FAILS-ON-OLD
    seen = []

    def trace(frame, event, argument):
        if event == 'call' and frame.f_code.co_filename == MODULE and frame.f_code.co_name in ('execute', 'body'):
            seen.append((frame.f_code.co_name, pt._IN_TASK.get(), pt._inside_task()))
        return None
    threading.settrace(trace)
    try:
        assert pt.ThreadStages(1).run('face', _two()[:1], _fine) == [0]
    finally:
        threading.settrace(None)
    assert seen == [('execute', False, True), ('body', True, True)]
    assert pt._IN_TASK.get() is False and pt._inside_task() is False        # the dispatcher is not marked


def test_planning_is_not_eligible_anywhere_in_a_worker(monkeypatch):                         # FAILS-ON-OLD
    monkeypatch.setenv(ENV, '4')
    model, _ = joint()
    policy = IntersectionBatchPolicy()
    seen = {}

    def trace(frame, event, argument):
        if event == 'call' and frame.f_code.co_name == 'execute' and frame.f_code.co_filename == MODULE:
            seen.setdefault('in execute', pt.thread_stages(model, policy, 'exact', None))
        return None
    threading.settrace(trace)
    try:
        assert pt.ThreadStages(1).run('face', _two()[:1], _fine) == [0]
    finally:
        threading.settrace(None)
    assert seen == {'in execute': None}
    assert isinstance(pt.thread_stages(model, policy, 'exact', None), pt.ThreadStages)


def test_every_pool_of_the_route_names_its_threads_before_they_start():
    """Source rule: the one place that creates a pool passes the marker prefix, so a pool thread is marked from
    its construction."""
    import ast
    from pathlib import Path
    tree = ast.parse(Path(MODULE).read_text(encoding='utf-8'))
    pools = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'ThreadPoolExecutor']
    assert len(pools) == 1
    prefix = {keyword.arg: keyword.value for keyword in pools[0].keywords}['thread_name_prefix']
    assert isinstance(prefix, ast.Name) and prefix.id == '_WORKER_PREFIX' and pt._WORKER_PREFIX == POOL
    assert not any(isinstance(node, ast.Call) and getattr(node.func, 'attr', None) == 'Thread' for node in ast.walk(tree))
