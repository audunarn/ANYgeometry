"""Reused OS thread idents in the opt-in thread route (0.4.7 gate class B, 2026-11-24).

The worker registry ``_planning_threads._WORKERS`` maps an OS thread ident to the crew slot that owns the
thread; a dispatcher removes the entry only after it has joined that thread. Between the end of a worker and
that removal the OS may give the same ident to a new thread (glibc reuses thread idents at once: hosted
Linux failed test_concurrent_dispatchers_keep_their_own_markers_and_leave_none 30 times out of 30, Intel
macOS 3 of 30, Windows never). ``_inside_task`` then took the new thread - another dispatcher - for a worker
and refused its stage ("thread planning stage inside a task of another stage is not supported").

Contract (after the Codex review of 43f7810, which found the is_alive() probe unsafe in teardown): an entry
marks the thread unless ``threading._active`` proves that a different live Thread now owns the ident; a worker
in teardown, a native thread, a present None and any non-crew value keep the mark (fail closed). No lock is
taken and no thread is joined or probed.
"""
import threading

import pytest

from anygeometry import _planning_threads as pt


def _two():
    return [dict(face=0), dict(face=1)]


def _finished_crew():
    crew = pt._Crew(1, lambda: None, threading.Event())
    crew.start()
    pt.wait(crew)
    return crew


def test_an_entry_of_a_finished_worker_does_not_mark_the_thread_that_reuses_its_ident():
    crew = _finished_crew()
    me = threading.get_ident()
    assert me not in pt._WORKERS
    with pt._WORKERS_LOCK:
        pt._WORKERS[me] = crew.keys[0]        # as if the finished worker had had this thread's ident
    try:
        assert not crew.threads[0].is_alive()
        assert not pt._inside_task()
        assert pt.ThreadStages(2).run('face', _two(), lambda item, check=None: item['face']) == [0, 1]
    finally:
        with pt._WORKERS_LOCK:
            pt._WORKERS.pop(me, None)
        crew.shutdown()


def test_a_live_worker_is_still_inside_a_task():
    seen = []
    ready, release = threading.Event(), threading.Event()

    def work():
        seen.append(pt._inside_task())
        ready.set()
        release.wait(30)
    crew = pt._Crew(1, work, threading.Event())
    crew.start()
    try:
        assert ready.wait(30)
    finally:
        release.set()
        crew.shutdown()
    assert seen == [True]


def test_an_entry_that_is_not_a_crew_slot_still_marks_the_thread():
    """Anything else in the registry is treated as a live worker (fail closed)."""
    me = threading.get_ident()
    with pt._WORKERS_LOCK:
        pt._WORKERS[me] = object()
    try:
        assert pt._inside_task()
    finally:
        with pt._WORKERS_LOCK:
            pt._WORKERS.pop(me, None)


# ---- every branch of _registered, through an injected ``threading._active`` view ----

def _entry_for(crew, ident):
    with pt._WORKERS_LOCK:
        pt._WORKERS[ident] = crew.keys[0]


IDENT = -20261124                                  # never a real ident


@pytest.fixture
def stale():
    crew = _finished_crew()
    _entry_for(crew, IDENT)
    yield crew
    with pt._WORKERS_LOCK:
        pt._WORKERS.pop(IDENT, None)
    crew.shutdown()


def test_a_different_live_thread_on_the_ident_proves_the_entry_stale(stale):
    other = threading.Thread(target=lambda: None)
    assert not pt._registered(IDENT, active={IDENT: other})


def test_the_crew_thread_itself_on_the_ident_keeps_the_mark(stale):
    assert pt._registered(IDENT, active={IDENT: stale.threads[0]})


def test_no_thread_on_the_ident_keeps_the_mark(stale):
    """A worker in teardown has left threading._active; a native thread was never in it."""
    assert pt._registered(IDENT, active={})


def test_something_that_is_not_a_thread_on_the_ident_keeps_the_mark(stale):
    assert pt._registered(IDENT, active={IDENT: object()})


def test_a_present_none_entry_keeps_the_mark():
    with pt._WORKERS_LOCK:
        pt._WORKERS[IDENT] = None
    try:
        assert pt._registered(IDENT, active={IDENT: threading.Thread(target=lambda: None)})
    finally:
        with pt._WORKERS_LOCK:
            pt._WORKERS.pop(IDENT, None)


def test_a_missing_entry_is_not_a_mark():
    assert IDENT not in pt._WORKERS and not pt._registered(IDENT, active={})


def test_the_real_registry_check_never_probes_or_joins_a_thread(monkeypatch):
    """Codex (43f7810): is_alive() can join natively in teardown; the check must not call it."""
    crew = _finished_crew()
    me = threading.get_ident()

    def forbidden(self, *args, **kwargs):
        raise AssertionError('thread probed or joined by the registry check')
    with pt._WORKERS_LOCK:
        pt._WORKERS[me] = crew.keys[0]
    monkeypatch.setattr(threading.Thread, 'is_alive', forbidden)
    monkeypatch.setattr(threading.Thread, 'join', forbidden)
    try:
        assert not pt._inside_task()
    finally:
        monkeypatch.undo()
        with pt._WORKERS_LOCK:
            pt._WORKERS.pop(me, None)
        crew.shutdown()
