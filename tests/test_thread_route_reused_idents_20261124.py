"""Reused OS thread idents in the opt-in thread route (0.4.7 gate class B, 2026-11-24).

The worker registry ``_planning_threads._WORKERS`` maps an OS thread ident to the crew slot that owns the
thread; a dispatcher removes the entry only after it has joined that thread. Between the end of a worker and
that removal the OS may give the same ident to a new thread (glibc reuses thread idents at once: hosted
Linux failed test_concurrent_dispatchers_keep_their_own_markers_and_leave_none 30 times out of 30, Intel
macOS 3 of 30, Windows never). ``_inside_task`` then took the new thread - another dispatcher - for a worker
and refused its stage ("thread planning stage inside a task of another stage is not supported").

Contract: a registry entry marks the current thread only while the worker thread it names is alive (an
alive thread has not exited, so its ident cannot have been reused).
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
