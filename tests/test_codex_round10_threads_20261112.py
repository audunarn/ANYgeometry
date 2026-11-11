"""Codex review round 10 of the thread route (2026-11-11): the stage owns its threads.

Findings: (1) one dispatcher's cleanup could erase the marker of a concurrent stage's thread that reused the
ident; (2) the worker initializer ran outside the module's handlers, so an interrupt there was lost and a
failure before the marker left a thread unmarked; (3) a returned executor shutdown did not prove that every
started worker was joined. Class: "a statement or state between pool creation, first worker statement, last
worker statement and join whose failure or overlap leaves workers unaccounted".

Closed with one state machine, ``_Crew`` (created -> started -> joined), that replaces the executor:
* the dispatcher creates every thread object first, sets ``attempted`` BEFORE ``start()``, and joins each
  thread itself; a thread registers itself (ident -> its crew) as its first statement, inside the handler that
  records what it meets, and sets ``done`` as its last;
* the registry is owner-keyed and changed only under its lock: a crew removes an entry only after it has joined
  that thread and only while the entry still names that crew;
* a stage is finished - and the serial repeat possible - only if the crew is accounted: every thread whose
  start was attempted was joined and is done.

Tests marked FAILS-ON-OLD fail on 19f912f. Bounded; no thread object and no registry entry may be left behind.
"""
import faulthandler
import threading

import pytest

from anygeometry import GeometryError
from anygeometry import _planning_threads as pt
from test_planning_threads import _clear

WAIT = 30
BOUND = 240
POOL = 'anygeometry-plan'
MODULE = pt.__file__
FATAL = (KeyboardInterrupt, SystemExit)


def _registry():
    return set(getattr(pt, '_WORKERS', getattr(pt, '_WORKER_IDENTS', ())))


@pytest.fixture(autouse=True)
def _bounded(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setattr(threading, 'excepthook', lambda args: None)      # a deliberately killed thread stays quiet
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


def _serial(seen):
    def serial():
        seen['serial'] = True
        return 'serial'
    return serial


def _outcome(call):
    try:
        return ('returned', call())
    except BaseException as exc:
        return ('raised', type(exc), str(exc))


def _threads(monkeypatch, cls):
    """Make the crew build its threads from ``cls`` (a threading.Thread subclass)."""
    class Threading:
        def __getattr__(self, name):
            return cls if name == 'Thread' else getattr(threading, name)
    monkeypatch.setattr(pt, 'threading', Threading())


# ---- (1) the registry is owner-keyed ----

def test_a_crew_never_removes_an_entry_that_names_another_owner():                           # FAILS-ON-OLD
    crew = pt._Crew(2, lambda: None, threading.Event())
    crew.start()
    pt.wait(crew)
    idents = list(crew.entered)
    assert all(pt._WORKERS[ident] is crew for ident in idents) and all(crew.done)
    other = object()                                  # a later thread of another stage reused the first ident
    pt._WORKERS[idents[0]] = other
    try:
        crew.shutdown()
        assert pt._WORKERS.get(idents[0]) is other and idents[1] not in pt._WORKERS
        assert crew.accounted()
    finally:
        pt._WORKERS.pop(idents[0], None)


def test_concurrent_dispatchers_keep_their_own_markers_and_leave_none():                     # FAILS-ON-OLD
    # Two dispatcher threads run stages at the same time, again and again: every task sees its own thread
    # registered to a crew, results are right, and the registry is empty afterwards.
    errors, rounds = [], 40

    def task(item, check=None):
        if not isinstance(pt._WORKERS.get(threading.get_ident()), pt._Crew):
            errors.append('worker not registered')
        return item['face']

    def dispatcher():
        try:
            for _ in range(rounds):
                if pt.ThreadStages(2).run('face', _two(), task) != [0, 1]:
                    errors.append('wrong result')
        except BaseException as exc:
            errors.append(repr(exc))
    threads = [threading.Thread(target=dispatcher, name=f'dispatcher-{n}') for n in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(BOUND)
    assert errors == [] and not any(thread.is_alive() for thread in threads)


# ---- (2) registration inside the recording handler; a thread without a marker is never forgotten ----

@pytest.mark.parametrize('fatal', FATAL)
def test_interrupt_while_a_worker_starts_up_is_recorded(fatal, monkeypatch):                 # FAILS-ON-OLD
    seen = {}

    def trace(frame, event, argument):
        # The first statement region of a crew thread: the call event of the work loop (old tip: of the
        # executor initializer).
        if (event == 'call' and frame.f_code.co_filename == MODULE
                and frame.f_code.co_name in ('work', '_enter_worker') and not seen.get('hook')):
            seen['hook'] = True
            raise fatal('during start-up')
        return None
    threading.settrace(trace)
    try:
        outcome = _outcome(lambda: pt.plan_with_threads(lambda: pt.ThreadStages(2).run('face', _two(), _fine),
                                                        _serial(seen)))
    finally:
        threading.settrace(None)
    assert outcome == ('raised', fatal, 'during start-up') and seen == dict(hook=True)


@pytest.mark.parametrize('error', (RuntimeError, KeyboardInterrupt))
def test_thread_that_dies_before_its_marker_exists_blocks_the_serial_repeat(error, monkeypatch):   # FAILS-ON-OLD
    # The hook kills one crew thread on entry to its main function, before the registration statement: nothing
    # of ours records what it met, and at its teardown it has no ident marker. The crew is not accounted.
    seen = {}

    def trace(frame, event, argument):
        if (event == 'call' and frame.f_code.co_filename == MODULE
                and frame.f_code.co_name in ('_main', '_enter_worker') and not seen.get('hook')):
            seen['hook'] = True
            raise error('before the marker')
        return None
    threading.settrace(trace)
    try:
        outcome = _outcome(lambda: pt.plan_with_threads(lambda: pt.ThreadStages(2).run('face', _two(), _fine),
                                                        _serial(seen)))
    finally:
        threading.settrace(None)
    assert outcome[:2] == ('raised', GeometryError) and 'did not finish' in outcome[2]
    assert seen == dict(hook=True)                    # no serial repeat


def test_thread_that_dies_after_its_tasks_but_before_its_last_statement_is_not_accounted(monkeypatch):   # FAILS-ON-OLD
    import inspect
    crew = pt._Crew(1, lambda: None, threading.Event())
    lines, first = inspect.getsourcelines(pt._Crew._main)
    last = first + next(n for n, line in enumerate(lines) if line.strip().startswith('self.done[slot] = True'))

    def trace(frame, event, argument):
        if frame.f_code.co_name == '_main' and frame.f_code.co_filename == MODULE:
            def local(frame, event, argument):
                if event == 'line' and frame.f_lineno == last:
                    raise RuntimeError('killed before the last statement')      # outside the handlers of _main
                return local
            return local
        return None
    threading.settrace(trace)
    try:
        crew.start()
        pt.wait(crew)
    finally:
        threading.settrace(None)
    crew.shutdown()
    assert crew.joined == [True] and crew.done == [False] and not crew.accounted()
    assert crew.entered[0] not in pt._WORKERS         # the thread is gone (joined), so its marker is removed


# ---- (3) partial start-up: every thread whose start was attempted is accounted for ----

def test_start_that_raises_although_the_thread_runs_blocks_the_serial_repeat(monkeypatch):   # FAILS-ON-OLD
    seen, made = {}, []
    release, second_in = threading.Event(), threading.Event()

    class Thread(threading.Thread):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made.append(self)

        def start(self):
            super().start()                           # the thread IS running
            if self.name.endswith('_1'):
                assert second_in.wait(WAIT)           # ... and inside a task
                raise MemoryError('start bookkeeping')

        def join(self, timeout=None):
            if self.name.endswith('_1') and not release.is_set():
                raise RuntimeError('cannot join')     # as if the caller no longer knew this thread
            super().join(timeout)
    _threads(monkeypatch, Thread)

    def task(item, check=None):
        if threading.current_thread().name.endswith('_1'):
            seen['second worker running'] = True
            second_in.set()
            release.wait()                            # only the test releases it
            seen['second worker finished'] = True
        else:
            second_in.wait(WAIT)                      # leave the other task to the second thread
        return item['face']
    try:
        outcome = _outcome(lambda: pt.plan_with_threads(lambda: pt.ThreadStages(2).run('face', _two(), task),
                                                        _serial(seen)))
        assert outcome[:2] == ('raised', GeometryError) and 'did not finish' in outcome[2]
        alive = [thread for thread in made if thread.is_alive()]
        assert [thread.name for thread in alive] == [f'{POOL}_1']
        assert 'serial' not in seen and 'second worker finished' not in seen
        assert alive[0].ident in pt._WORKERS          # still marked: it was not joined
    finally:
        release.set()
        for thread in made:
            threading.Thread.join(thread, BOUND)
        for thread in made:
            pt._WORKERS.pop(thread.ident, None)


def test_start_that_fails_before_the_thread_exists_is_not_treated_as_joined(monkeypatch):    # FAILS-ON-OLD
    # Conservative by design: after a failed start() the module cannot prove that the thread never ran, so the
    # crew stays unaccounted (typed refusal, no serial repeat) - also when, as here, it really never ran.
    seen = {}

    class Thread(threading.Thread):
        def start(self):
            if self.name.endswith('_1'):
                raise RuntimeError("can't start new thread")
            super().start()
    _threads(monkeypatch, Thread)
    outcome = _outcome(lambda: pt.plan_with_threads(lambda: pt.ThreadStages(2).run('face', _two(), _fine),
                                                    _serial(seen)))
    assert outcome[:2] == ('raised', GeometryError) and 'did not finish' in outcome[2] and seen == {}


def test_crew_state_machine():                                                              # FAILS-ON-OLD
    stopped, ran = threading.Event(), []
    crew = pt._Crew(3, lambda: ran.append(threading.get_ident()), stopped)
    assert crew.attempted == [False] * 3 and crew.accounted()        # created: nothing to account for
    assert [thread.is_alive() for thread in crew.threads] == [False] * 3
    crew.start()
    assert crew.attempted == [True] * 3
    pt.wait(crew)
    assert crew.done == [True] * 3 and crew.joined == [False] * 3 and not crew.accounted()   # started, not joined
    assert sorted(crew.entered) == sorted(ran) and all(pt._WORKERS[ident] is crew for ident in crew.entered)
    crew.shutdown()
    assert crew.joined == [True] * 3 and crew.accounted() and not set(crew.entered) & set(pt._WORKERS)
    crew.shutdown()                                   # idempotent
    assert crew.accounted()


def test_ordinary_stage_is_unchanged_by_the_crew():
    tasks = [dict(kind='face', face=index, cost=float(index % 3)) for index in range(9)]
    names = set()

    def task(item, check=None):
        names.add(threading.current_thread().name)
        return item['face'] * 3
    assert pt.ThreadStages(4).run('face', tasks, task) == [index * 3 for index in range(9)]
    assert names and all(name.startswith(POOL) for name in names) and len(names) <= 4

    def refusing(item, check=None):
        if item['face'] in (3, 5):
            raise GeometryError(f"refused {item['face']}")
        return item['face']
    with pytest.raises(GeometryError, match='face:3: refused 3'):
        pt.ThreadStages(4).run('face', tasks, refusing)
    assert pt.plan_with_threads(lambda: pt.ThreadStages(4).run('face', tasks, refusing), lambda: 'serial') == 'serial'
