"""Codex review round 7 of the thread route (2026-11-11): the classes behind its two findings.

A. No stage inside a stage task. A stage that is asked to run inside a task of another stage (in the worker, or in
   a thread that inherited the task's context) refuses with a typed GeometryError BEFORE it creates a pool, and a
   planning call inside a task is not eligible for the thread route. So no worker of the route ever belongs to a
   record other than the one whose verdict decides: the evidence "workers may still run" cannot be lost in an
   ordinary task failure.
B. Classifying an exception as an interrupt never runs code of the exception object: the module uses the object's
   real type (``type(exc)`` and the C-level subtype test), so an exception with a hostile ``__class__`` cannot make
   the recording fail before it has recorded.

Tests marked FAILS-ON-OLD fail on 05fcc7d. Bounded like the round-6 tests.
"""
import contextvars
import faulthandler
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


def _serial(seen):
    def serial():
        seen['serial'] = True
        return 'serial'
    return serial


def _counted_pools(monkeypatch):
    pools = []

    class Pool(ThreadPoolExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            pools.append(self)
    monkeypatch.setattr(pt, 'ThreadPoolExecutor', Pool)
    return pools


# ---- A. no stage inside a stage task ----

def test_stage_inside_a_task_is_refused_before_any_pool_exists(monkeypatch):               # FAILS-ON-OLD
    pools, seen = _counted_pools(monkeypatch), {}

    def inner(item, check=None):
        seen['inner task ran'] = True
        return item['face']

    def task(item, check=None):
        if item['face'] == 1:
            pt.ThreadStages(2).run('face', _two(), inner)
        return item['face']
    with pytest.raises(GeometryError) as error:
        pt.ThreadStages(2).run('face', _two(), task)
    assert 'inside a task of another stage' in str(error.value) and 'face:1' in str(error.value)
    assert len(pools) == 1 and seen == {}


def test_unfinished_inner_workers_cannot_exist_beside_the_serial_repeat(monkeypatch):      # FAILS-ON-OLD
    # Round-7 scenario: an inner stage whose worker blocks, whose wait() fails ordinarily and whose shutdown fails
    # without draining. Its refusal used to become an ordinary task failure of the outer stage, and the outer
    # boundary repeated serially beside the inner worker. Now the inner stage never starts.
    # The doubles are installed before any thread exists and decide per instance / per calling thread (nothing is
    # patched from inside a task): the SECOND pool ever created is the inner one and cannot be shut down; wait()
    # fails only when a pool worker (the inner dispatcher) calls it.
    pools, seen = [], {}
    release = threading.Event()
    real_wait = pt.wait

    class Pool(ThreadPoolExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            pools.append(self)

        def shutdown(self, wait=True, **options):
            if len(pools) > 1 and self is pools[1] and not release.is_set():
                raise MemoryError('inner shutdown')
            super().shutdown(wait=wait, **options)
    monkeypatch.setattr(pt, 'ThreadPoolExecutor', Pool)

    def waiting(futures, *args, **kwargs):
        if threading.current_thread().name.startswith(POOL):
            raise RuntimeError('inner wait')
        return real_wait(futures, *args, **kwargs)
    monkeypatch.setattr(pt, 'wait', waiting)

    def inner(item, check=None):
        seen['inner worker started'] = True
        release.wait(WAIT)
        return item['face']

    def task(item, check=None):
        if item['face'] == 0:
            pt.ThreadStages(2).run('face', _two(), inner)
        return item['face']
    try:
        result = pt.plan_with_threads(lambda: pt.ThreadStages(2).run('face', _two(), task), _serial(seen))
        assert result == 'serial'
        assert seen == dict(serial=True)              # the repeat is safe: no inner worker was ever started
        assert len(pools) == 1
    finally:
        release.set()
        for pool in pools:
            pool.shutdown(wait=True)


def test_thread_that_inherited_a_task_context_is_refused_as_well(monkeypatch):             # FAILS-ON-OLD
    pools, seen = _counted_pools(monkeypatch), {}

    def task(item, check=None):
        def foreign():
            try:
                pt.ThreadStages(2).run('face', _two(), _fine)
            except GeometryError as exc:
                seen['refused'] = str(exc)
        thread = threading.Thread(target=contextvars.copy_context().run, args=(foreign,))
        thread.start()
        thread.join(WAIT)
        return item['face']
    assert pt.ThreadStages(1).run('face', _two()[:1], task) == [0]
    assert 'inside a task of another stage' in seen['refused'] and len(pools) == 1


def test_planning_inside_a_task_is_not_eligible_for_the_thread_route(monkeypatch):         # FAILS-ON-OLD
    monkeypatch.setenv(ENV, '4')
    model, _ = joint()
    policy = IntersectionBatchPolicy()
    assert isinstance(pt.thread_stages(model, policy, 'exact', None), pt.ThreadStages)
    seen = {}

    def task(item, check=None):
        seen['stages'] = pt.thread_stages(model, policy, 'exact', None)
        return item['face']
    assert pt.ThreadStages(1).run('face', _two()[:1], task) == [0]
    assert seen == dict(stages=None)


def test_unfinished_workers_of_the_one_stage_block_the_serial_repeat_while_they_live(monkeypatch):
    # The worker is verifiably still running when the verdict is given (no timing assumption: it is released only
    # after the assertions).
    pools, seen = _counted_pools(monkeypatch), {}
    started, release = threading.Event(), threading.Event()
    real_wait = pt.wait

    def waiting(futures, *args, **kwargs):
        assert started.wait(WAIT)
        raise RuntimeError('ordinary failure of wait')
    monkeypatch.setattr(pt, 'wait', waiting)
    monkeypatch.setattr(ThreadPoolExecutor, 'shutdown',
                        lambda self, wait=True, **k: (_ for _ in ()).throw(MemoryError('shutdown')))

    def task(item, check=None):
        started.set()
        release.wait()                                # unbounded on purpose: only the test releases it
        seen['worker finished'] = True
        return item['face']
    try:
        with pytest.raises(GeometryError, match='did not finish') as error:
            pt.plan_with_threads(lambda: pt.ThreadStages(1).run('face', _two()[:1], task), _serial(seen))
        assert isinstance(error.value.__cause__, MemoryError)
        assert seen == {}                             # no serial repeat, and the worker has not finished
        assert any(thread.name.startswith(POOL) and thread.is_alive() for thread in threading.enumerate())
    finally:
        monkeypatch.undo()
        release.set()
        for pool in pools:
            pool.shutdown(wait=True)
    assert seen == {'worker finished': True}
    assert real_wait is pt.wait


# ---- B. classification never runs code of the exception ----

class Hostile(KeyboardInterrupt):
    """An interrupt whose ``__class__`` cannot be read: ``isinstance`` on it raises."""
    @property
    def __class__(self):
        raise RuntimeError('classification failed')


class HostileOrdinary(ValueError):
    @property
    def __class__(self):
        raise RuntimeError('classification failed')


def _outcome(call):
    try:
        return ('returned', call())
    except BaseException as exc:                      # the real type, without isinstance
        return ('raised', type(exc), exc.args)


def test_interrupt_with_a_hostile_class_leaving_the_attempt_is_still_an_interrupt():       # FAILS-ON-OLD
    seen = {}

    def threaded():
        raise Hostile('stop')
    assert _outcome(lambda: pt.plan_with_threads(threaded, _serial(seen))) == ('raised', Hostile, ('stop',))
    assert seen == {}


@pytest.mark.parametrize('where', ('task', 'wait', 'shutdown', 'future'))
def test_interrupt_with_a_hostile_class_anywhere_in_a_stage_is_still_an_interrupt(where, monkeypatch):   # FAILS-ON-OLD (not 'task')
    seen = {}
    if where == 'wait':
        real = pt.wait

        def waiting(futures, *args, **kwargs):
            real(futures, *args, **kwargs)
            raise Hostile('stop')
        monkeypatch.setattr(pt, 'wait', waiting)
    if where in ('shutdown', 'future'):
        class Pool(ThreadPoolExecutor):
            def shutdown(self, wait=True, **options):
                super().shutdown(wait=wait, **options)
                if where == 'shutdown':
                    raise Hostile('stop')

            def submit(self, *args, **kwargs):
                future = super().submit(*args, **kwargs)
                if where == 'future':
                    real = future.result

                    def result(timeout=None):
                        real(timeout)
                        raise Hostile('stop')
                    future.result = result
                return future
        monkeypatch.setattr(pt, 'ThreadPoolExecutor', Pool)

    def task(item, check=None):
        if where == 'task':
            raise Hostile('stop')
        return item['face']
    expected = ('raised', Hostile, ('stop',))
    assert _outcome(lambda: pt.ThreadStages(2).run('face', _two(), task)) == expected

    def swallowing():
        try:
            pt.ThreadStages(2).run('face', _two(), task)
        except BaseException:
            seen['swallowed'] = True
        raise RuntimeError('later ordinary failure')
    assert _outcome(lambda: pt.plan_with_threads(swallowing, _serial(seen))) == expected
    assert seen == dict(swallowed=True)


def test_ordinary_exception_with_a_hostile_class_is_still_ordinary():                      # FAILS-ON-OLD
    seen = {}

    def threaded():
        raise HostileOrdinary('refused')
    assert _outcome(lambda: pt.plan_with_threads(threaded, _serial(seen))) == ('returned', 'serial')

    def task(item, check=None):
        raise HostileOrdinary('refused')
    assert _outcome(lambda: pt.ThreadStages(2).run('face', _two(), task)) == ('raised', HostileOrdinary, ('refused',))
    assert _outcome(lambda: pt.plan_with_threads(lambda: pt.ThreadStages(2).run('face', _two(), task),
                                                 _serial(seen))) == ('returned', 'serial')


def test_the_module_never_uses_isinstance_on_a_caught_exception():
    """Source rule behind B: no ``isinstance(<name bound by except or holding one>, ...)`` in the thread module;
    exception kinds are decided by ``_ordinary`` / ``_is`` on the real type."""
    import ast
    from pathlib import Path
    tree = ast.parse(Path(pt.__file__).read_text(encoding='utf-8'))
    names = {'exc', 'failure', 'held', 'error', 'interrupt', 'instead'}
    hits = [node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, 'id', None) in ('isinstance', 'issubclass')
            and node.args and getattr(node.args[0], 'id', None) in names]
    assert hits == []
    assert pt._ordinary(ValueError('x')) and not pt._ordinary(KeyboardInterrupt()) and not pt._ordinary(Hostile())
    assert pt._ordinary(HostileOrdinary('x')) and not pt._ordinary(SystemExit())


def test_future_loop_records_on_its_own(monkeypatch):
    # White box: the interrupt met while reading the futures is noted there, before the stage's own guard notes it.
    notes = []
    real = pt._Attempt.note

    def note(self, exc):
        notes.append(type(exc))
        return real(self, exc)
    monkeypatch.setattr(pt._Attempt, 'note', note)

    class Pool(ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            future = super().submit(*args, **kwargs)
            real_result = future.result

            def result(timeout=None):
                real_result(timeout)
                raise SystemExit('reading futures')
            future.result = result
            return future
    monkeypatch.setattr(pt, 'ThreadPoolExecutor', Pool)
    with pytest.raises(SystemExit):
        pt.ThreadStages(1).run('face', _two()[:1], _fine)
    assert notes == [SystemExit, SystemExit]          # the future loop, then the guard around the stage
