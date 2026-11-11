"""Codex review round 11 of the thread route (2026-11-11).

1. An interrupt caught by the crew's own join loop was only kept in a local and could be replaced by an ordinary
   failure of the cleanup that follows. Class: "a handler has an interrupt in hand and does something else before
   recording it". Closed: the join loop notes it in the attempt first; and a source rule over the whole module
   requires that EVERY handler that can hold an interrupt starts with the recording statement.
2. A hook raising on the return event of a worker's main function, after its last statement, is seen by no
   statement of the module. That is written into the contract limit now (the thread is still joined before the
   stage counts as finished). The earlier docstring claim that such a death leaves the crew unaccounted was wrong
   and is corrected; there is deliberately no test that pins the unobserved behaviour.
Also: the registry is keyed by crew AND slot (a reused ident inside one crew), the ident is published before the
registry entry is written, and crew threads are never daemon threads.

Tests marked FAILS-ON-OLD fail on 73242f5. Bounded; no thread object and no registry entry may be left behind.
"""
import ast
import faulthandler
import threading
from pathlib import Path

import pytest

from anygeometry import GeometryError
from anygeometry import _planning_threads as pt
from test_planning_threads import _clear

BOUND = 240
MODULE = pt.__file__
FATAL = (KeyboardInterrupt, SystemExit)


@pytest.fixture(autouse=True)
def _bounded(monkeypatch):
    _clear(monkeypatch)
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


def _outcome(call):
    try:
        return ('returned', call())
    except BaseException as exc:
        return ('raised', type(exc), str(exc))


@pytest.mark.parametrize('fatal', FATAL)
@pytest.mark.parametrize('direct', (False, True))
def test_interrupt_of_a_join_survives_an_ordinary_failure_of_the_cleanup_after_it(fatal, direct, monkeypatch):   # FAILS-ON-OLD
    seen = {}
    main = threading.get_ident()

    class Thread(threading.Thread):
        calls = 0

        def join(self, timeout=None):
            super().join(timeout)
            self.calls += 1                           # first join: the dispatcher's wait; second: the crew's shutdown
            if self.name.endswith('_0') and self.calls == 2:
                seen['join'] = True
                raise fatal('during join')

    class Threading:
        def __getattr__(self, name):
            return Thread if name == 'Thread' else getattr(threading, name)
    monkeypatch.setattr(pt, 'threading', Threading())
    real = pt._WORKERS_LOCK

    class Lock:                                       # the registry cleanup after the joins fails ordinarily
        def __enter__(self):
            if threading.get_ident() == main and seen.get('join') and not seen.get('cleanup'):
                seen['cleanup'] = True
                raise MemoryError('registry cleanup')
            return real.__enter__()

        def __exit__(self, *args):
            return real.__exit__(*args)
    monkeypatch.setattr(pt, '_WORKERS_LOCK', Lock())

    def serial():
        seen['serial'] = True
        return 'serial'
    try:
        if direct:
            outcome = _outcome(lambda: pt.ThreadStages(2).run('face', _two(), _fine))
        else:
            outcome = _outcome(lambda: pt.plan_with_threads(lambda: pt.ThreadStages(2).run('face', _two(), _fine), serial))
        assert outcome == ('raised', fatal, 'during join')
        assert seen == dict(join=True, cleanup=True)
    finally:
        monkeypatch.undo()
        for ident, key in list(pt._WORKERS.items()):   # the failed cleanup left the entries of the joined threads
            if not any(thread.ident == ident for thread in threading.enumerate()):
                pt._WORKERS.pop(ident, None)


def test_every_handler_that_can_hold_an_interrupt_records_it_first():
    """Source rule for the class: in the thread module, every ``except BaseException`` (or bare ``except``) starts
    with ``<something>.note(exc)`` or with the store of ``exc`` into a preallocated slot (``xs[i] = exc``)."""
    tree = ast.parse(Path(MODULE).read_text(encoding='utf-8'))
    handlers = [node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)
                and (node.type is None or getattr(node.type, 'id', None) == 'BaseException')]
    assert len(handlers) == 7                         # boundary, stage guard, task, crew thread, wait, cleanup, join
    for handler in handlers:
        assert handler.name == 'exc'
        first = handler.body[0]
        noted = (isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
                 and getattr(first.value.func, 'attr', None) == 'note'
                 and [getattr(argument, 'id', None) for argument in first.value.args] == ['exc'])
        stored = (isinstance(first, ast.Assign) and len(first.targets) == 1
                  and isinstance(first.targets[0], ast.Subscript) and getattr(first.value, 'id', None) == 'exc')
        assert noted or stored, ast.unparse(handler)[:200]
    # Tuples and other spellings that could catch an interrupt are not used at all.
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is not None:
            assert isinstance(node.type, ast.Name) and node.type.id in ('BaseException', 'Exception', '_Abandoned',
                                                                        '_NotAudited', '_RouteFailure')


def test_crew_notes_a_join_interrupt_before_anything_else(monkeypatch):                     # FAILS-ON-OLD
    notes = []

    class Thread(threading.Thread):
        def join(self, timeout=None):
            super().join(timeout)
            if not notes:
                raise SystemExit('during join')

    class Threading:
        def __getattr__(self, name):
            return Thread if name == 'Thread' else getattr(threading, name)
    monkeypatch.setattr(pt, 'threading', Threading())
    crew = pt._Crew(2, lambda: None, threading.Event(), notes.append)
    crew.start()
    with pytest.raises(SystemExit):
        crew.shutdown()
    assert [type(exc) for exc in notes] == [SystemExit]
    assert crew.joined == [False, True] and not crew.accounted()        # the first thread's join did not return
    crew.shutdown()
    assert crew.accounted()


def test_registry_is_keyed_by_crew_and_slot():                                               # FAILS-ON-OLD
    # Both workers are alive at the same time (barrier), so their idents are distinct by construction - the test
    # does not rely on scheduling (round 12: two trivial workers could run one after the other and reuse an ident).
    barrier, release = threading.Barrier(2, timeout=30), threading.Event()

    def work():
        barrier.wait()
        release.wait(30)
    crew = pt._Crew(2, work, threading.Event())
    crew.start()
    try:
        deadline = 300
        while (None in crew.entered or not all(ident in pt._WORKERS for ident in crew.entered)) and deadline:
            threading.Event().wait(.01)
            deadline -= 1
        first, second = crew.entered
        assert first != second and all(thread.is_alive() for thread in crew.threads)
        assert pt._WORKERS[first] is crew.keys[0] and pt._WORKERS[second] is crew.keys[1]
        assert crew.keys[0] == (crew, 0) and crew.keys[0] is not crew.keys[1]
    finally:
        release.set()
    pt.wait(crew)
    # As if the first thread's ident had been reused by the (not yet joined) thread of the second slot.
    pt._WORKERS[first] = crew.keys[1]
    crew.joined[0] = True                             # only the first slot counts as joined
    crew.attempted[1] = False                         # (keep shutdown from joining the second slot here)
    try:
        crew.shutdown()
        assert pt._WORKERS.get(first) is crew.keys[1]            # not removed: it is no longer slot 0's entry
    finally:
        crew.attempted[1] = True
        pt._WORKERS.pop(first, None)
        crew.shutdown()
        pt._WORKERS.pop(second, None)


def test_crew_threads_are_never_daemon_threads():                                            # FAILS-ON-OLD
    made = []

    def dispatcher():
        made.append(pt._Crew(2, lambda: None, threading.Event()))
    thread = threading.Thread(target=dispatcher, daemon=True)
    thread.start()
    thread.join(BOUND)
    assert made and [worker.daemon for worker in made[0].threads] == [False, False]


def test_ident_is_published_before_the_registry_entry(monkeypatch):
    # If writing the registry entry fails, the thread has published its ident and holds no entry: nothing is
    # left that could not be removed. (The failure is an ordinary machinery failure of that thread.)
    main = threading.get_ident()
    real = pt._WORKERS_LOCK

    class Lock:
        def __enter__(self):
            if threading.get_ident() != main:
                raise MemoryError('registration')
            return real.__enter__()

        def __exit__(self, *args):
            return real.__exit__(*args)
    monkeypatch.setattr(pt, '_WORKERS_LOCK', Lock())
    stopped = threading.Event()
    crew = pt._Crew(1, lambda: None, stopped)
    crew.start()
    pt.wait(crew)
    crew.shutdown()
    assert crew.entered[0] is not None and crew.entered[0] not in pt._WORKERS
    assert type(crew.machinery[0]) is MemoryError and crew.done == [True] and crew.accounted() and stopped.is_set()
    # (Owner decision after round 12: a machinery failure is a typed refusal, not a serial repeat.)
    with pytest.raises(GeometryError, match='thread planning failed') as error:
        pt.ThreadStages(1).run('face', _two()[:1], _fine)
    assert type(error.value.__cause__) is MemoryError
    with pytest.raises(GeometryError, match='thread planning failed'):
        pt.plan_with_threads(lambda: pt.ThreadStages(1).run('face', _two()[:1], _fine), lambda: pytest.fail('serial'))
