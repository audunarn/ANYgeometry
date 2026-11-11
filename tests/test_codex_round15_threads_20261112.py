"""Codex review round 15 of the thread route (2026-11-11).

1. The entry of a stage (its guards, the task list, the attempt record) is the route's own code: a failure there
   is a route failure, not a refusal that could lead to a serial repeat, and not a raw exception.
2. "The route could not be loaded" carries the route's provenance: the class lives in batch_intersections
   (``_ThreadRouteFailure``), and the route module uses that very class.
3. Every statement of the route is inside a handler: the reads of the switches (also in batch_intersections),
   ``route``'s preparation, the eligibility test. Only the serial call is outside, so serial planning raises what
   it raises without the switch.

Tests marked FAILS-ON-OLD fail on 4ed7259. Bounded; no thread object and no registry entry may be left behind.
"""
import faulthandler
import os
import sys
import threading

import pytest

from anygeometry import GeometryError, plan_intersections
from anygeometry import batch_intersections as bi
from anygeometry import _planning_threads as pt
from anygeometry.batch_intersections import IntersectionBatchPolicy
from test_planning_threads import ENV, PROCESS, _clear, joint

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


def _two():
    return [dict(kind='face', face=0, cost=1.), dict(kind='face', face=1, cost=2.)]


def _fine(item, check=None):
    return item['face']


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


class _FailingEnvironment:
    """os as a module sees it, with a read of the named switch that fails."""
    def __init__(self, key, error):
        self.environ = self
        self.key, self.error = key, error

    def get(self, key, default=None):
        if key == self.key:
            raise self.error('environment')
        return os.environ.get(key, default)

    def __getattr__(self, name):
        return getattr(os, name)


# ---- 1. the entry of a stage ----

@pytest.mark.parametrize('error', (GeometryError, RuntimeError))
def test_failure_of_a_stage_guard_is_a_route_failure(error):                                 # FAILS-ON-OLD
    # The dispatching thread's name cannot be read when the stage asks whether it is inside a worker.
    results = []

    class Dispatcher(threading.Thread):
        armed = False

        @property
        def name(self):
            if self.armed:
                raise error('the thread name cannot be read')
            return 'dispatcher'

        @name.setter
        def name(self, value):
            pass

        def run(self):
            stage = lambda: pt.ThreadStages(2).run('face', _two(), _fine)
            self.armed = True
            try:
                results.append(_attempt(stage))       # through the boundary
                results.append(_call(stage))          # a stage called on its own
            finally:
                self.armed = False
    thread = Dispatcher()
    thread.start()
    thread.join(BOUND)
    (outcome, cause, serial), (direct, direct_cause) = results
    assert outcome[:2] == ('raised', pt._RouteFailure) and 'could not be entered' in outcome[2] and not serial
    assert type(cause.__cause__) is error
    assert direct[:2] == ('raised', pt._RouteFailure) and type(direct_cause.__cause__) is error


def test_failure_to_create_the_record_of_a_stage_called_on_its_own_is_a_route_failure(monkeypatch):   # FAILS-ON-OLD
    monkeypatch.setattr(pt, '_Attempt', lambda: (_ for _ in ()).throw(MemoryError('attempt record')))
    outcome, error = _call(lambda: pt.ThreadStages(2).run('face', _two(), _fine))
    assert outcome[:2] == ('raised', pt._RouteFailure) and 'could not be entered' in outcome[2]
    assert type(error.__cause__) is MemoryError


# ---- 2. provenance of "could not be loaded" ----

def test_loader_refusal_has_the_provenance_of_the_route(monkeypatch):                         # FAILS-ON-OLD
    assert pt._RouteFailure is bi._ThreadRouteFailure and issubclass(bi._ThreadRouteFailure, GeometryError)
    with pytest.raises(bi._ThreadRouteFailure, match='could not be loaded'):
        bi._thread_route('no_such_function')

    # A task whose nested planning call cannot load the route: not a task refusal, no serial repeat.
    def task(item, check=None):
        if item['face'] == 1:
            bi._thread_route('no_such_function')
        return item['face']
    outcome, error, serial = _attempt(lambda: pt.ThreadStages(2).run('face', _two(), task))
    assert outcome[:2] == ('raised', pt._RouteFailure) and 'could not be loaded' in outcome[2] and not serial
    assert isinstance(error.__cause__, AttributeError)

    monkeypatch.setenv(ENV, '4')
    monkeypatch.setitem(sys.modules, 'anygeometry._planning_threads', None)
    model, faces = joint()
    with pytest.raises(bi._ThreadRouteFailure, match='could not be loaded'):
        plan_intersections(model, faces, policy='connect')


# ---- 3. every statement of the route is inside a handler ----

@pytest.mark.parametrize('error', (MemoryError, OSError))
def test_failure_to_read_the_switch_in_the_dispatch_is_a_route_failure(error, monkeypatch):   # FAILS-ON-OLD
    monkeypatch.setattr(bi, 'os', _FailingEnvironment('ANYGEOMETRY_PLANNING_THREADS', error))
    model, faces = joint()
    with pytest.raises(bi._ThreadRouteFailure, match='could not be read') as raised:
        plan_intersections(model, faces, policy='connect')
    assert type(raised.value.__cause__) is error


@pytest.mark.parametrize('key', (ENV, PROCESS))
def test_failure_to_read_a_switch_in_the_route_is_a_route_failure(key, monkeypatch):          # FAILS-ON-OLD
    monkeypatch.setenv(ENV, '4')
    monkeypatch.setattr(pt, 'os', _FailingEnvironment(key, MemoryError))
    model, _ = joint()
    for call in (pt.validate_switches, lambda: pt.thread_stages(model, IntersectionBatchPolicy(), 'exact', None),
                 lambda: pt.route(model, IntersectionBatchPolicy(), 'exact', None, lambda stages: 'planned')):
        outcome, error = _call(call)
        assert outcome[:2] == ('raised', pt._RouteFailure) and 'could not be read' in outcome[2]
        assert type(error.__cause__) is MemoryError


def test_failure_while_the_route_prepares_is_a_route_failure(monkeypatch):                   # FAILS-ON-OLD
    monkeypatch.setenv(ENV, '4')
    model, _ = joint()
    planned = []
    monkeypatch.setattr(pt, 'thread_stages', lambda *args: (_ for _ in ()).throw(KeyError('preparation')))
    outcome, error = _call(lambda: pt.route(model, IntersectionBatchPolicy(), 'exact', None, planned.append))
    assert outcome[:2] == ('raised', pt._RouteFailure) and 'could not be prepared' in outcome[2] and planned == []
    assert type(error.__cause__) is KeyError


def test_the_serial_route_raises_what_it_raises_without_the_switch(monkeypatch):
    # Not eligible: plan(None) is the serial call and is outside every handler of the route.
    monkeypatch.setenv(ENV, '4')
    model, _ = joint()

    def plan(stages):
        assert stages is None
        raise ValueError("serial planning's own exception")
    outcome, error = _call(lambda: pt.route(model, IntersectionBatchPolicy(), 'topology', None, plan))
    assert outcome == ('raised', ValueError, "serial planning's own exception") and error.__cause__ is None

    # After a plain refusal of the attempt: the serial repeat raises serial's own exception as well.
    def refusing(stages):
        if stages is not None:
            raise GeometryError('refused')
        raise ZeroDivisionError("serial planning's own exception")
    outcome, error = _call(lambda: pt.route(model, IntersectionBatchPolicy(), 'exact', None, refusing))
    assert outcome[:2] == ('raised', ZeroDivisionError)


def test_switch_unset_reads_the_switch_and_nothing_else_of_the_route(monkeypatch):
    calls = []
    monkeypatch.setattr(bi, '_thread_route', lambda name: calls.append(name))
    model, faces = joint()
    assert plan_intersections(model, faces, policy='connect').content_checksum and calls == []
    assert bi._thread_switch_set() is False
    monkeypatch.setenv(ENV, '0')
    assert bi._thread_switch_set() is False
    monkeypatch.setenv(ENV, '3')
    assert bi._thread_switch_set() is True
