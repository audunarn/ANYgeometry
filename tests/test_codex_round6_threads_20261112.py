"""Codex review round 6 of the thread route (2026-11-11): the classes behind its three findings.

A. Every statement of the module that has an interrupt in hand records it in the attempt before anything else
   (reading the futures, whatever leaves a stage, whatever leaves the threaded attempt), so an interrupt that a
   caller swallowed, or that a later ordinary failure replaced, still decides the verdict.
B. The attempt record belongs to the thread that created it: a stage run from any other thread (a thread that
   inherited the context) keeps its own record, so ``running`` of the boundary's attempt is written by one thread.
   (Since round 7 a stage inside a task of another stage is refused outright; the nested tests here say so.)
C. An ordinary failure of the context reset cannot bypass the verdict.

Tests marked FAILS-ON-OLD fail on 380b389. Every test here is bounded: a hang (for example in an executor
shutdown) dumps all stacks and ends the process after ``BOUND`` seconds instead of blocking the suite.
"""
import contextvars
import faulthandler
import threading
from anygeometry._planning_threads import _Crew as ThreadPoolExecutor   # the crew of a stage (since round 10)

import pytest

from anygeometry import GeometryError
from anygeometry import _planning_threads as pt
from test_planning_threads import _clear

WAIT = 30
BOUND = 240
FATAL = (KeyboardInterrupt, SystemExit)
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


def _swallowing(stage, kinds, seen):
    """A caller of the stage that (wrongly) swallows its interrupt and then fails ordinarily."""
    def threaded():
        try:
            stage()
        except kinds:
            seen['swallowed'] = True
        raise RuntimeError('later ordinary failure')
    return threaded


# ---- A. an interrupt in hand is always recorded ----

# (Round 10: the stage owns its threads; there are no futures to read. The interrupt "met while reading the
# futures" is now the interrupt met while joining, covered by the round-5 wait/shutdown tests and by
# test_codex_round10_threads_20261112.py.)


def test_interrupt_of_a_worker_outside_its_task_slot_decides_the_verdict(monkeypatch):   # FAILS-ON-OLD
    # A worker's machinery fails ordinarily, and its stop signal is then interrupted: the interrupt is not in the
    # task's slot; since round 10 it is recorded in the slot of the crew thread.
    seen, real, doomed = {}, pt.perf_counter_ns, set()

    def clock():
        if threading.get_ident() in doomed:
            doomed.discard(threading.get_ident())
            seen['machinery'] = True
            raise MemoryError('bookkeeping')
        return real()
    monkeypatch.setattr(pt, 'perf_counter_ns', clock)

    class Event(threading.Event):
        def set(self):
            if threading.current_thread().name.startswith(POOL) and seen.get('machinery') and not seen.get('stop'):
                seen['stop'] = True
                raise SystemExit('stop signal in worker')
            super().set()

    class Threading:
        def __getattr__(self, name):
            return Event if name == 'Event' else getattr(threading, name)
    monkeypatch.setattr(pt, 'threading', Threading())

    def task(item, check=None):
        if item['face'] == 1:
            doomed.add(threading.get_ident())
        return item['face']
    with pytest.raises(SystemExit, match='stop signal in worker'):
        pt.plan_with_threads(_swallowing(lambda: pt.ThreadStages(1).run('face', _two(), task), SystemExit, seen),
                             _serial(seen))
    assert seen == dict(machinery=True, stop=True, swallowed=True)


@pytest.mark.parametrize('fatal', FATAL)
def test_interrupt_outside_any_stage_survives_a_failing_context_reset(fatal, monkeypatch):     # FAILS-ON-OLD
    seen = {}
    _failing_reset(monkeypatch, seen)

    def threaded():
        raise fatal('between the stages')
    with pytest.raises(fatal, match='between the stages'):
        pt.plan_with_threads(threaded, _serial(seen))
    assert seen == dict(reset=True)


# ---- C. the context reset ----

def _failing_reset(monkeypatch, seen, really_reset=True):
    real = pt._ATTEMPT

    class Variable:
        def get(self, *default):
            return real.get(*default)

        def set(self, value):
            seen.pop('token', None)
            token = real.set(value)
            if not really_reset:
                seen['token'] = token
            return token

        def reset(self, token):
            if really_reset:
                real.reset(token)
            seen['reset'] = True
            raise MemoryError('context reset')
    monkeypatch.setattr(pt, '_ATTEMPT', Variable())
    return real


@pytest.mark.parametrize('fatal', FATAL)
def test_recorded_interrupt_survives_a_failing_context_reset(fatal, monkeypatch):             # FAILS-ON-OLD
    seen = {}
    _failing_reset(monkeypatch, seen)
    real = pt.wait

    def waiting(futures, *args, **kwargs):
        real(futures, *args, **kwargs)
        raise fatal('during wait')
    monkeypatch.setattr(pt, 'wait', waiting)
    with pytest.raises(fatal, match='during wait'):
        pt.plan_with_threads(lambda: pt.ThreadStages(2).run('face', _two(), _fine), _serial(seen))
    assert seen == dict(reset=True)


def test_failing_context_reset_alone_is_ordinary(monkeypatch):
    seen = {}
    _failing_reset(monkeypatch, seen)

    def task(item, check=None):
        raise GeometryError('ordinary refusal')
    assert pt.plan_with_threads(lambda: pt.ThreadStages(2).run('face', _two(), task), _serial(seen)) == 'serial'
    assert seen == dict(reset=True, serial=True)
    seen.clear()                                      # also after a successful attempt: discarded, repeated serially
    assert pt.plan_with_threads(lambda: pt.ThreadStages(2).run('face', _two(), _fine), _serial(seen)) == 'serial'
    assert seen == dict(reset=True, serial=True)


def test_an_attempt_left_in_the_context_by_a_failed_reset_is_not_used_again(monkeypatch):     # FAILS-ON-OLD
    seen = {}
    real = _failing_reset(monkeypatch, seen, really_reset=False)
    try:
        assert pt.plan_with_threads(lambda: pt.ThreadStages(2).run('face', _two(), _fine), _serial(seen)) == 'serial'
        stale = real.get()
        assert stale is not None and stale.closed
        before = len(stale.worker_slots)
        # A stage called on its own afterwards applies the rule itself instead of leaving it to the closed attempt.

        class Pool(ThreadPoolExecutor):
            def shutdown(self, wait=True, **options):
                super().shutdown(wait=wait, **options)
                raise MemoryError('shutdown')
        monkeypatch.setattr(pt, '_Crew', Pool)
        with pytest.raises(GeometryError, match='did not finish'):
            pt.ThreadStages(2).run('face', _two(), _fine)
        assert len(stale.worker_slots) == before == 2                 # nothing added by the later stage
    finally:
        real.reset(seen['token'])
    assert real.get() is None


# ---- B. the attempt record belongs to its thread ----

def test_nested_stage_in_a_worker_cannot_clear_the_running_state_of_the_outer_stage(monkeypatch):   # FAILS-ON-OLD
    seen, pools = {}, []
    main = threading.get_ident()
    nested_done, release = threading.Event(), threading.Event()
    real_wait = pt.wait

    class Pool(ThreadPoolExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            pools.append(self)

        def shutdown(self, wait=True, **options):
            if threading.get_ident() == main and not seen.get('shutdown'):
                seen['shutdown'] = True               # the OUTER pool: fails before draining its blocked worker
                raise MemoryError('shutdown')
            super().shutdown(wait=wait, **options)
    monkeypatch.setattr(pt, '_Crew', Pool)

    def waiting(futures, *args, **kwargs):
        if threading.get_ident() != main:
            return real_wait(futures, *args, **kwargs)
        assert nested_done.wait(WAIT)                 # the nested stage has finished; one outer worker still runs
        seen['wait'] = True
        raise RuntimeError('ordinary failure of wait')
    monkeypatch.setattr(pt, 'wait', waiting)

    def task(item, check=None):
        if item['face'] == 0:
            with pytest.raises(GeometryError, match='inside a task of another stage'):
                pt.ThreadStages(2).run('face', _two(), _fine)                   # since round 7: refused, no pool
            seen['nested'] = True
            nested_done.set()
            return 0
        assert release.wait(WAIT)                     # this outer worker is still running at the verdict
        return 1
    try:
        with pytest.raises(GeometryError, match='did not finish') as error:
            pt.plan_with_threads(lambda: pt.ThreadStages(2).run('face', _two(), task), _serial(seen))
        assert isinstance(error.value.__cause__, MemoryError)
        assert seen == dict(nested=True, wait=True, shutdown=True)       # and no serial repeat beside the worker
    finally:
        release.set()
        for pool in pools:
            ThreadPoolExecutor.shutdown(pool, wait=True)


def test_stage_in_a_thread_that_inherited_the_context_keeps_its_own_record():                 # FAILS-ON-OLD
    seen = {}

    def threaded():
        attempt = pt._ATTEMPT.get()
        assert attempt is not None and attempt.owner == threading.get_ident()

        def foreign():
            assert pt._ATTEMPT.get() is attempt       # inherited, but not this thread's
            seen['foreign'] = pt.ThreadStages(2).run('face', _two(), _fine)
        context = contextvars.copy_context()
        thread = threading.Thread(target=context.run, args=(foreign,))
        thread.start()
        thread.join(WAIT)
        assert not thread.is_alive()
        assert attempt.worker_slots == [] and not attempt.running
        result = pt.ThreadStages(2).run('face', _two(), _fine)
        assert len(attempt.worker_slots) == 2 and not attempt.running     # task slots and crew slots
        return result
    assert pt.plan_with_threads(threaded, _serial(seen)) == [0, 1]
    assert seen == dict(foreign=[0, 1])


def test_nested_stage_refusal_is_an_ordinary_task_failure():
    # Since round 7 a stage inside a task is refused before it has a pool; for the outer stage that is an ordinary
    # failure of that task (serial wrapping, serial repeat at the boundary).
    def task(item, check=None):
        if item['face'] == 1:
            pt.ThreadStages(2).run('face', _two(), _fine)
        return item['face']
    with pytest.raises(GeometryError, match='face:1: thread planning stage inside a task of another stage'):
        pt.ThreadStages(2).run('face', _two(), task)
    assert pt.plan_with_threads(lambda: pt.ThreadStages(2).run('face', _two(), task), lambda: 'serial') == 'serial'
