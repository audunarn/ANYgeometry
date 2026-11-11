"""Codex review round 16 of the thread route (2026-11-11).

A route failure of ANY task that ran is the outcome of the stage, not only one of the lowest-index failed task:
after the join the outcomes of all tasks are scanned for a route failure (as the interrupt slots are), and the
lowest-index one is raised before any task failure. Otherwise a plain refusal of a lower-index task would hide it
and the boundary would repeat planning serially after the route has failed.

What is NOT changed (pinned here): an ordinary task failure of a higher index stays hidden behind the refusal of a
lower-index task, because the serial route would never have reached that task.

Tests marked FAILS-ON-OLD fail on 23f9c4f. Bounded; no thread object and no registry entry may be left behind.
"""
import faulthandler
import threading

import pytest

from anygeometry import GeometryError
from anygeometry import batch_intersections as bi
from anygeometry import _planning_threads as pt
from test_planning_threads import _clear

BOUND = 240


@pytest.fixture(autouse=True)
def _bounded(monkeypatch):
    _clear(monkeypatch)
    before, registered = threading.active_count(), set(pt._WORKERS)
    faulthandler.dump_traceback_later(BOUND, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()
    assert threading.active_count() == before
    assert set(pt._WORKERS) == registered


def _tasks(count):
    return [dict(kind='face', face=index, cost=1.) for index in range(count)]


def _call(call):
    try:
        return ('returned', call()), None
    except BaseException as exc:
        return ('raised', type(exc), str(exc)), exc


def _attempt(threaded):
    seen = {}

    def serial():
        seen['serial'] = True
        return 'serial'
    outcome, error = _call(lambda: pt.plan_with_threads(threaded, serial))
    return outcome, error, bool(seen)


def _loader_failure():
    bi._thread_route('no_such_function')


def _nested_stage():
    pt.ThreadStages(2).run('face', _tasks(2), lambda item, check=None: item['face'])


ROUTE_FAILURES = {'loader': (_loader_failure, 'could not be loaded'),
                  'nested stage': (_nested_stage, 'inside a task of another stage')}


def _barrier_task(count, late, late_failure, early_failure):
    """Task function for ``count`` tasks on ``count`` threads. Task ``late`` fails with ``late_failure`` once
    every task is inside its body; only then does task 0 fail with ``early_failure``. So the higher-index task
    is never abandoned: its outcome exists when the stage decides."""
    inside = threading.Barrier(count)
    failed = threading.Event()

    def task(item, check=None):
        index = item['face']
        inside.wait(BOUND)
        if index == late:
            try:
                late_failure()
            finally:
                failed.set()
        if index == 0:
            assert failed.wait(BOUND)
            early_failure()
        return index
    return task


def _refuse():
    raise GeometryError('a refusal of geometry')


def _crash():
    raise RuntimeError('an ordinary task failure')


@pytest.mark.parametrize('kind', sorted(ROUTE_FAILURES))
@pytest.mark.parametrize('count,late', ((2, 1), (3, 2)))
def test_route_failure_of_a_higher_index_task_is_not_hidden_by_a_refusal(kind, count, late):   # FAILS-ON-OLD
    failure, text = ROUTE_FAILURES[kind]
    stage = lambda: pt.ThreadStages(count).run('face', _tasks(count), _barrier_task(count, late, failure, _refuse))
    outcome, error, serial = _attempt(stage)                 # through the boundary: no serial repeat
    assert outcome[:2] == ('raised', pt._RouteFailure) and text in outcome[2] and not serial
    direct, direct_error = _call(stage)                      # a stage called on its own
    assert direct[:2] == ('raised', pt._RouteFailure) and text in direct[2]


@pytest.mark.parametrize('kind', sorted(ROUTE_FAILURES))
def test_route_failure_of_a_higher_index_task_comes_before_an_ordinary_task_failure(kind):    # FAILS-ON-OLD
    failure, text = ROUTE_FAILURES[kind]
    stage = lambda: pt.ThreadStages(2).run('face', _tasks(2), _barrier_task(2, 1, failure, _crash))
    outcome, error, serial = _attempt(stage)
    assert outcome[:2] == ('raised', pt._RouteFailure) and text in outcome[2] and not serial


def test_lowest_index_route_failure_is_the_one_raised():
    def nested():
        try:
            _nested_stage()
        except pt._RouteFailure as exc:
            raise pt._refusal('the route failure of task 0', exc)
    stage = lambda: pt.ThreadStages(2).run('face', _tasks(2), _barrier_task(2, 1, _loader_failure, nested))
    outcome, error, serial = _attempt(stage)
    assert outcome == ('raised', pt._RouteFailure, 'the route failure of task 0') and not serial


def test_ordinary_failure_of_a_higher_index_task_stays_behind_a_refusal():
    # Unchanged: serial planning stops at task 0's refusal and never reaches task 1, so the repeat is serial's.
    stage = lambda: pt.ThreadStages(2).run('face', _tasks(2), _barrier_task(2, 1, _crash, _refuse))
    outcome, error, serial = _attempt(stage)
    assert outcome == ('returned', 'serial') and serial
    direct, direct_error = _call(stage)
    assert direct[:2] == ('raised', GeometryError) and type(direct_error) is GeometryError
    assert 'face:0' in direct[2] and 'a refusal of geometry' in direct[2]


def test_a_refusal_alone_still_leads_to_the_serial_repeat():
    stage = lambda: pt.ThreadStages(2).run('face', _tasks(2), _barrier_task(2, 1, lambda: None, _refuse))
    outcome, error, serial = _attempt(stage)
    assert outcome == ('returned', 'serial') and serial
