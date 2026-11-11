"""Owner decision after Codex round 12 (2026-11-11): the serial repeat only after a plain refusal.

The one decision (``_Attempt.verdict``) when a threaded attempt ended with an ordinary exception:
(1) an observed interrupt is raised;
(2) the serial repeat ONLY IF the exception is a plain refusal - a typed GeometryError (a task refusal of a stage,
    or a refusal of the planner's own code in the dispatching thread) or ``_NotAudited`` - AND every started thread
    of every stage was joined;
(3) otherwise a typed GeometryError with the original exception as its cause; nothing is planned.
So a failure of the route's own machinery never becomes a silent serial answer, and an interrupt that was lost
(round 12: a hook raising inside the recording step) can at worst become a typed refusal.

Tests marked FAILS-ON-OLD fail on d48e6b0. Bounded; no thread object and no registry entry may be left behind.
"""
import faulthandler
import sys
import threading

import pytest

from anygeometry import GeometryError
from anygeometry import _planning_threads as pt
from test_planning_threads import ENV, _clear

BOUND = 240
MODULE = pt.__file__
FAILED = 'thread planning failed outside a geometry refusal'


@pytest.fixture(autouse=True)
def _bounded(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setattr(threading, 'excepthook', lambda args: None)
    before, registered = threading.active_count(), set(pt._WORKERS)
    faulthandler.dump_traceback_later(BOUND, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()
    assert threading.active_count() == before
    assert set(pt._WORKERS) == registered


def _two():
    return [dict(kind='face', face=0, cost=1.), dict(kind='face', face=1, cost=2.)]


def _fine(item, check=None):
    return item['face']


def _attempt(threaded):
    """(outcome, whether serial ran) of one boundary call."""
    seen = {}

    def serial():
        seen['serial'] = True
        return 'serial'
    try:
        outcome = ('returned', pt.plan_with_threads(threaded, serial))
    except BaseException as exc:
        outcome = ('raised', type(exc), str(exc), type(exc.__cause__))
    return outcome, bool(seen)


def _stage(task, threads=2):
    return lambda: pt.ThreadStages(threads).run('face', _two(), task)


# ---- (2) the plain refusals: serial decides ----

def test_task_refusal_is_repeated_serially():
    def task(item, check=None):
        raise GeometryError(f"refused {item['face']}")
    assert _attempt(_stage(task)) == (('returned', 'serial'), True)


def test_unaudited_input_is_repeated_serially():
    def threaded():
        pt.ThreadStages(2).admit_face_inputs({1: [object()]}, {1: []})
    assert _attempt(threaded) == (('returned', 'serial'), True)


def test_refusal_of_the_planner_between_the_stages_is_repeated_serially():
    def threaded():
        assert pt.ThreadStages(2).run('face', _two(), _fine) == [0, 1]
        raise GeometryError('geometry changed during intersection planning')
    assert _attempt(threaded) == (('returned', 'serial'), True)


def test_success_is_returned_and_nothing_is_repeated():
    assert _attempt(_stage(_fine)) == (('returned', [0, 1]), False)


# ---- (3) everything else: a typed refusal, never a silent serial answer ----

def test_machinery_failure_is_a_typed_refusal(monkeypatch):                                   # FAILS-ON-OLD
    real, doomed = pt.perf_counter_ns, set()

    def clock():
        if threading.get_ident() in doomed:
            doomed.discard(threading.get_ident())
            raise MemoryError('bookkeeping')
        return real()
    monkeypatch.setattr(pt, 'perf_counter_ns', clock)

    def task(item, check=None):
        if item['face'] == 1:
            doomed.add(threading.get_ident())
        return item['face']
    outcome, serial = _attempt(_stage(task))
    assert outcome[:2] == ('raised', GeometryError) and FAILED in outcome[2] and '(MemoryError)' in outcome[2]
    assert ENV in outcome[2] and outcome[3] is MemoryError and not serial
    with pytest.raises(GeometryError, match='MemoryError') as error:      # the same from a stage called on its own
        doomed.clear()
        pt.ThreadStages(2).run('face', _two(), task)
    assert type(error.value.__cause__) is MemoryError


@pytest.mark.parametrize('kind', (ValueError, RuntimeError, ZeroDivisionError))
def test_task_failure_that_is_not_a_refusal_is_a_typed_refusal(kind):                         # FAILS-ON-OLD
    def task(item, check=None):
        raise kind('a defect, not a geometry answer')
    outcome, serial = _attempt(_stage(task))
    assert outcome[:2] == ('raised', GeometryError) and FAILED in outcome[2] and outcome[3] is kind and not serial


def test_failure_of_the_planner_between_the_stages_that_is_not_a_refusal_is_a_typed_refusal():   # FAILS-ON-OLD
    def threaded():
        assert pt.ThreadStages(2).run('face', _two(), _fine) == [0, 1]
        raise KeyError('a defect')
    outcome, serial = _attempt(threaded)
    assert outcome[:2] == ('raised', GeometryError) and outcome[3] is KeyError and not serial


def test_task_refusal_beside_an_unaccounted_thread_is_not_repeated(monkeypatch):
    # A plain task refusal, but one crew thread was killed on entry (before its marker): not every started
    # thread is accounted for, so the refusal does not lead to the serial repeat.
    seen = {}

    def trace(frame, event, argument):
        if (event == 'call' and frame.f_code.co_filename == MODULE and frame.f_code.co_name == '_main'
                and not seen.get('hook')):
            seen['hook'] = True
            raise RuntimeError('before the marker')
        return None

    def task(item, check=None):
        raise GeometryError('ordinary refusal')
    threading.settrace(trace)
    try:
        outcome, serial = _attempt(_stage(task))
    finally:
        threading.settrace(None)
    assert outcome[:2] == ('raised', GeometryError) and 'did not finish' in outcome[2] and not serial
    assert seen == dict(hook=True)


def test_interrupt_lost_inside_the_recording_step_is_at_worst_a_typed_refusal():              # FAILS-ON-OLD
    # Round 12: a trace hook in the dispatcher raises an ordinary exception inside ``note`` before the interrupt is
    # stored. Out of contract for the interrupt rule - but the outcome is a typed refusal, not serial planning.
    armed = {}

    def trace(frame, event, argument):
        if frame.f_code.co_name == 'note' and frame.f_code.co_filename == MODULE and not armed.get('fired'):
            def local(frame, event, argument):
                if event == 'line' and not armed.get('fired'):
                    armed['fired'] = True
                    raise MemoryError('hook inside the recording step')
                return local
            return local
        return None

    def threaded():
        raise KeyboardInterrupt('between the stages')
    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        outcome, serial = _attempt(threaded)
    finally:
        sys.settrace(previous)
    assert armed == dict(fired=True) and not serial
    assert outcome[:2] == ('raised', GeometryError) and outcome[3] is MemoryError


def test_observed_interrupt_is_still_raised():
    def task(item, check=None):
        raise SystemExit('in worker')
    outcome, serial = _attempt(_stage(task))
    assert outcome[:3] == ('raised', SystemExit, 'in worker') and not serial

    def swallowing():
        try:
            pt.ThreadStages(2).run('face', _two(), task)
        except SystemExit:
            pass
        raise GeometryError('a later refusal')            # even a plain refusal does not hide the interrupt
    outcome, serial = _attempt(swallowing)
    assert outcome[:3] == ('raised', SystemExit, 'in worker') and not serial


def test_verdict_table():                                                                    # FAILS-ON-OLD
    def verdict(failure, running=0, interrupt=None):
        attempt = pt._Attempt()
        attempt.running = running
        if interrupt is not None:
            attempt.note(interrupt)
        return attempt.verdict(failure)
    refusal, stop = GeometryError('refused'), KeyboardInterrupt()

    class Refusal(GeometryError):
        pass
    assert verdict(refusal) is None and verdict(Refusal('x')) is None and verdict(pt._NotAudited('x')) is None
    for failure in (MemoryError(), RuntimeError('x'), ValueError('x'), OSError('x'), AssertionError()):
        error = verdict(failure)
        assert type(error) is GeometryError and error.__cause__ is failure and 'thread planning failed' in str(error)
    for failure in (refusal, pt._NotAudited('x'), MemoryError()):
        error = verdict(failure, running=1)
        assert type(error) is GeometryError and error.__cause__ is failure and 'did not finish' in str(error)
        assert verdict(failure, interrupt=stop) is stop and verdict(failure, running=1, interrupt=stop) is stop
