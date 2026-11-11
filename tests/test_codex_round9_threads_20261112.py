"""Codex review round 9 of the thread route (2026-11-11): a worker stays marked until its thread is really gone.

Finding: after ``threading`` has unregistered a worker and its task context has exited, thread-local cleanup still
runs in that thread; a finalizer there could start a stage, because both markers (thread name through the
``threading`` registry, task context) read state that is torn down before the thread is gone.
Class: "the guard reads state that the interpreter tears down before the thread is really gone, or that does not
exist yet when the thread already runs". Closed with a third marker that depends on neither: the module-level
registry ``_WORKERS`` of OS thread idents. A pool thread adds its ident as its first statement (the pool's
initializer, and again first in ``execute``); only the dispatcher removes it, after ``pool.shutdown(wait=True)``
has joined the thread. A stage refuses when the calling thread's ident is registered.

Tests marked FAILS-ON-OLD fail on 2f55dc1. Bounded, and no thread object may be left behind.
"""
import ast
import faulthandler
import threading
from anygeometry._planning_threads import _Crew as ThreadPoolExecutor   # the crew of a stage (since round 10)
from pathlib import Path

import pytest

from anygeometry import GeometryError
from anygeometry import _planning_threads as pt
from test_planning_threads import _clear

BOUND = 240
MODULE = pt.__file__
REFUSED = 'thread planning stage inside a task of another stage is not supported'


def _registry():
    return set(getattr(pt, '_WORKERS', ()))


@pytest.fixture(autouse=True)
def _bounded(monkeypatch):
    _clear(monkeypatch)
    before, registered = threading.active_count(), _registry()
    faulthandler.dump_traceback_later(BOUND, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()
    assert threading.active_count() == before
    assert _registry() == registered


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
    monkeypatch.setattr(pt, '_Crew', Pool)
    return pools


def _reenter(seen, where):
    try:
        pt.ThreadStages(2).run('face', _two(), _fine)
        seen[where] = 'a nested stage ran'
    except GeometryError as exc:
        seen[where] = str(exc)
    except BaseException as exc:
        seen[where] = f'{type(exc).__name__}: {exc}'


LOCAL = threading.local()


class _AtTeardown:
    """Kept in a thread-local of the worker: finalized when the worker's thread state is torn down."""

    def __init__(self, seen):
        self.seen = seen

    def __del__(self):
        self.seen['registered with threading'] = threading.get_ident() in threading._active
        self.seen['context marker'] = pt._IN_TASK.get()
        self.seen['ident marker'] = threading.get_ident() in _registry()
        _reenter(self.seen, 'teardown')


@pytest.mark.parametrize('fails', (False, True))
def test_finalizer_during_worker_teardown_cannot_start_a_stage(fails, monkeypatch):           # FAILS-ON-OLD
    pools, seen = _counted_pools(monkeypatch), {}

    def task(item, check=None):
        if item['face'] == 1:
            LOCAL.kept = _AtTeardown(seen)            # lives until this worker thread is torn down
            if fails:
                raise GeometryError('ordinary refusal')
        return item['face']

    def serial():
        seen['serial'] = True
        return 'serial'
    result = pt.plan_with_threads(lambda: pt.ThreadStages(2).run('face', _two(), task), serial)
    # The finalizer ran before the pool was joined, i.e. before the verdict; it saw neither of the older markers.
    assert seen.pop('registered with threading') is False and seen.pop('context marker') is False
    assert seen.pop('ident marker') is True
    assert seen.pop('teardown') == REFUSED and len(pools) == 1
    assert (result, seen) == (('serial', dict(serial=True)) if fails else ([0, 1], {}))


def test_worker_idents_are_registered_from_the_first_statement_until_the_join(monkeypatch):   # FAILS-ON-OLD
    seen = []

    def trace(frame, event, argument):
        if event == 'call' and frame.f_code.co_filename == MODULE and frame.f_code.co_name in ('execute', 'body'):
            seen.append((frame.f_code.co_name, threading.get_ident() in pt._WORKERS))
        return None

    def task(item, check=None):
        seen.append(('task', threading.get_ident() in pt._WORKERS))
        return item['face']
    assert threading.get_ident() not in pt._WORKERS
    before = set(pt._WORKERS)
    threading.settrace(trace)
    try:
        assert pt.ThreadStages(1).run('face', _two()[:1], task) == [0]
    finally:
        threading.settrace(None)
    assert seen == [('execute', True), ('body', True), ('task', True)]      # already at the call event of execute
    assert set(pt._WORKERS) == before and threading.get_ident() not in pt._WORKERS


def test_idents_stay_registered_when_a_thread_was_not_joined(monkeypatch):                     # FAILS-ON-OLD
    # Round 10 form: the join of ONE crew thread fails (after really joining, so the test can end cleanly); that
    # thread stays registered and the stage is not finished; the other thread is unmarked.
    failed = []

    class Thread(threading.Thread):
        calls = 0

        def join(self, timeout=None):
            super().join(timeout)
            self.calls += 1                           # first join: the dispatcher's wait; second: its shutdown
            if self.name.endswith('_1') and self.calls == 2:
                failed.append(self.ident)
                raise MemoryError('join')

    class Threading:
        def __getattr__(self, name):
            return Thread if name == 'Thread' else getattr(threading, name)
    monkeypatch.setattr(pt, 'threading', Threading())
    before = set(pt._WORKERS)
    barrier = threading.Barrier(2, timeout=30)

    def task(item, check=None):
        barrier.wait()                                # both crew threads take a task
        return item['face']
    try:
        with pytest.raises(GeometryError, match='did not finish') as error:
            pt.ThreadStages(2).run('face', _two(), task)
        assert isinstance(error.value.__cause__, MemoryError)
        assert set(pt._WORKERS) - before == set(failed) and len(failed) == 1
    finally:
        for ident in failed:
            pt._WORKERS.pop(ident, None)


def test_first_profiled_event_of_every_pool_thread_is_already_inside(monkeypatch):
    # Named test limit of round 9: the first profiled event anywhere in worker start-up, not only in this module.
    first, lock = {}, threading.Lock()
    main = threading.get_ident()

    def profile(frame, event, argument):
        ident = threading.get_ident()
        if ident != main and ident not in first:      # the first event only: the thread is registered with threading
            with lock:
                first.setdefault(ident, (pt._inside_task(), frame.f_code.co_filename == MODULE))
    threading.setprofile(profile)
    try:
        assert pt.ThreadStages(2).run('face', _two(), _fine) == [0, 1]
    finally:
        threading.setprofile(None)
    assert first and all(inside for inside, _ in first.values())
    assert not any(ours for _, ours in first.values())                     # it is start-up code, not this module


def test_the_module_creates_threads_only_in_its_crew():
    """Source rule (round 10): exactly one ``threading.Thread(...)`` call, inside ``_Crew.__init__``; no executor,
    no Timer, no ``_thread``; and no thread is started anywhere but in ``_Crew.start``."""
    tree = ast.parse(Path(MODULE).read_text(encoding='utf-8'))
    nodes = list(ast.walk(tree))
    called = [getattr(node.func, 'id', getattr(node.func, 'attr', None)) for node in nodes if isinstance(node, ast.Call)]
    imported = [alias.name for node in nodes if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names]
    assert called.count('Thread') == 1
    starts = sorted(node.func.value.id for node in nodes if isinstance(node, ast.Call)
                    and getattr(node.func, 'attr', None) == 'start')
    assert starts == ['crew', 'thread']             # crew.start() in the stage, thread.start() in the crew
    assert not {'ThreadPoolExecutor', 'ProcessPoolExecutor', 'Timer', 'start_new_thread', 'start_joinable_thread',
                'submit'} & set(called)
    assert not {'Thread', 'Timer', '_thread', 'ThreadPoolExecutor', 'ProcessPoolExecutor', 'multiprocessing',
                'concurrent.futures', 'futures'} & set(imported)
    assert not any(isinstance(node, ast.ImportFrom) and (node.module or '').startswith('concurrent') for node in nodes)
    crew = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == '_Crew')
    inside = {id(node) for node in ast.walk(crew)}
    for node in nodes:
        if isinstance(node, ast.Call) and (getattr(node.func, 'attr', None) == 'Thread' or (
                getattr(node.func, 'attr', None) == 'start' and node.func.value.id == 'thread')):
            assert id(node) in inside
