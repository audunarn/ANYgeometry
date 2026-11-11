"""Codex review round 14 of the thread route (2026-11-11), owner decision "single outer boundary".

All route code runs inside ONE outer boundary:
* ``batch_intersections`` reaches the route only through ``_thread_route``, which loads the route module inside a
  handler and calls exactly two of its functions, ``validate_switches`` (public entry) and ``route`` (dispatch);
* every refusal the route itself raises is a ``_RouteFailure`` (a GeometryError subclass); it is never a plain
  refusal for any boundary, so a nested boundary cannot turn it into a serial repeat;
* stage entry and admission failures are route failures, not refusals;
* the refusal texts are built from literals and an exact-``str`` class name: building them cannot raise.

Tests marked FAILS-ON-OLD fail on e396f3f. Bounded; no thread object and no registry entry may be left behind.
"""
import ast
import faulthandler
import sys
import threading
from pathlib import Path

import pytest

from anygeometry import GeometryError, plan_intersections
from anygeometry import batch_intersections as bi
from anygeometry import _planning_threads as pt
from anygeometry.batch_intersections import IntersectionBatchPolicy
from test_planning_threads import ENV, PROCESS, _clear, joint

BOUND = 240
FAILED = 'thread planning failed outside a geometry refusal'


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


def _stage(task, threads=2):
    return lambda: pt.ThreadStages(threads).run('face', _two(), task)


# ---- provenance by class: a route failure is never a plain refusal, for any boundary ----

def test_nested_boundary_cannot_turn_a_route_failure_into_a_serial_repeat():                  # FAILS-ON-OLD
    seen = []

    def defect(item, check=None):
        raise ValueError('a defect')

    def inner():
        return pt.plan_with_threads(_stage(defect), lambda: seen.append('inner serial'))
    outcome, error, serial = _attempt(inner)
    assert outcome[:2] == ('raised', pt._RouteFailure) and FAILED in outcome[2] and not serial and seen == []
    assert type(error.__cause__) is ValueError          # the inner refusal itself, passed through unchanged


def test_route_failure_of_a_task_is_not_wrapped_into_a_task_refusal():                        # FAILS-ON-OLD
    # A stage inside a task refuses with a route failure; the outer stage does not make a task refusal of it.
    def task(item, check=None):
        if item['face'] == 1:
            pt.ThreadStages(2).run('face', _two(), _fine)
        return item['face']
    outcome, error = _call(_stage(task))
    assert outcome == ('raised', pt._RouteFailure, 'thread planning stage inside a task of another stage is not supported')
    outcome, error, serial = _attempt(_stage(task))
    assert outcome[:2] == ('raised', pt._RouteFailure) and 'inside a task' in outcome[2] and not serial


def test_every_refusal_of_the_route_is_a_route_failure(monkeypatch):                          # FAILS-ON-OLD
    assert issubclass(pt._RouteFailure, GeometryError)
    monkeypatch.setenv(ENV, '9')
    with pytest.raises(pt._RouteFailure, match=ENV):
        pt.validate_switches()
    monkeypatch.setenv(ENV, '2')
    monkeypatch.setenv(PROCESS, '1')
    with pytest.raises(pt._RouteFailure, match='mutually exclusive'):
        pt.validate_switches()
    monkeypatch.delenv(PROCESS)
    stages, raised = pt.ThreadStages(2), []
    thread = threading.Thread(target=lambda: raised.append(_call(lambda: stages.run('face', _two(), _fine))[0]))
    thread.start()
    thread.join(BOUND)
    assert raised == [('raised', pt._RouteFailure, 'thread planning stage entered from a foreign thread')]
    attempt = pt._Attempt()
    for failure in (MemoryError(), ValueError('x')):
        assert type(attempt.verdict(failure)) is pt._RouteFailure
    attempt.running = 1
    assert type(attempt.verdict(GeometryError('refused'))) is pt._RouteFailure
    already = pt._RouteFailure('already decided')
    assert pt._Attempt().verdict(already) is already


# ---- stage entry and admission are inside the boundary ----

class Unreadable:
    def __iter__(self):
        raise GeometryError('raised while the task list was read')


class Unlistable(dict):
    def values(self):
        raise GeometryError('raised while the inputs were listed')


def test_failure_while_a_stage_reads_its_tasks_is_a_route_failure():                          # FAILS-ON-OLD
    outcome, error, serial = _attempt(lambda: pt.ThreadStages(2).run('face', Unreadable(), _fine))
    assert outcome[:2] == ('raised', pt._RouteFailure) and 'could not read its tasks' in outcome[2] and not serial
    assert str(error.__cause__) == 'raised while the task list was read'


@pytest.mark.parametrize('method', ('admit_domains', 'admit_face_inputs'))
def test_failure_while_a_stage_admits_its_inputs_is_a_route_failure(method):                  # FAILS-ON-OLD
    def threaded():
        getattr(pt.ThreadStages(2), method)(Unlistable(), {})
    outcome, error, serial = _attempt(threaded)
    assert outcome[:2] == ('raised', pt._RouteFailure) and 'could not admit' in outcome[2] and not serial
    assert str(error.__cause__) == 'raised while the inputs were listed'


def test_unaudited_inputs_are_still_plain(monkeypatch):
    def threaded():
        pt.ThreadStages(2).admit_face_inputs({1: [object()]}, {1: []})
    assert _attempt(threaded)[::2] == (('returned', 'serial'), True)

    def faces():
        pt.ThreadStages(2).admit_domains({}, {7: object()})
    assert _attempt(faces)[::2] == (('returned', 'serial'), True)


# ---- the refusal text cannot raise ----

class BadName(str):
    def __format__(self, spec):
        raise RuntimeError('name formatting')

    def __str__(self):
        raise RuntimeError('name text')

    def __add__(self, other):
        raise RuntimeError('name concatenation')

    def __radd__(self, other):
        raise RuntimeError('name concatenation')


class Failure(Exception):
    pass


Failure.__name__ = BadName('Failure')


def test_refusal_text_does_not_depend_on_the_name_of_the_failure(monkeypatch):               # FAILS-ON-OLD
    assert pt._type_name(Failure()) == 'exception' and type(pt._type_name(Failure())) is str
    assert pt._type_name(ValueError()) == 'ValueError'

    def task(item, check=None):
        raise Failure('a defect')
    outcome, error = _call(_stage(task))                                  # a stage called on its own
    assert outcome[:2] == ('raised', pt._RouteFailure) and '(exception)' in outcome[2] and type(error.__cause__) is Failure
    outcome, error, serial = _attempt(_stage(task))                       # the boundary: the ORIGINAL failure is the cause
    assert outcome[:2] == ('raised', pt._RouteFailure) and type(error.__cause__) is Failure and not serial
    monkeypatch.setenv(ENV, '4')                                          # the eligibility test
    model, _ = joint()
    monkeypatch.setattr(pt, '_audited_model', lambda model: (_ for _ in ()).throw(Failure('eligibility')))
    outcome, error = _call(lambda: pt.thread_stages(model, IntersectionBatchPolicy(), 'exact', None))
    assert outcome[:2] == ('raised', pt._RouteFailure) and type(error.__cause__) is Failure


# ---- one outer boundary in the dispatch code ----

def test_failure_to_load_the_route_module_is_a_typed_refusal(monkeypatch):                    # FAILS-ON-OLD
    monkeypatch.setenv(ENV, '4')
    monkeypatch.setitem(sys.modules, 'anygeometry._planning_threads', None)       # importing it fails
    model, faces = joint()
    with pytest.raises(GeometryError, match='could not be loaded') as error:
        plan_intersections(model, faces, policy='connect')
    assert isinstance(error.value.__cause__, ImportError)
    with pytest.raises(GeometryError, match='could not be loaded'):               # also a missing function
        monkeypatch.undo()
        bi._thread_route('no_such_function')


def test_dispatch_reaches_the_route_only_through_its_one_helper():
    """Source rule: in batch_intersections the route module is named only inside ``_thread_route``, and the helper
    is called with exactly the two names ``validate_switches`` and ``route``."""
    tree = ast.parse(Path(bi.__file__).read_text(encoding='utf-8'))
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_thread_route')
    inside = {id(node) for node in ast.walk(helper)}
    for node in ast.walk(tree):
        text = None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
        if isinstance(node, ast.ImportFrom):
            text = node.module or ''
        if isinstance(node, ast.alias):
            text = node.name
        if isinstance(node, (ast.Name, ast.Attribute)):
            text = getattr(node, 'id', getattr(node, 'attr', ''))
        if text and '_planning_threads' in text and not isinstance(node, ast.Constant):
            assert id(node) in inside, ast.dump(node)[:120]
        if text and text.strip('.') == '_planning_threads':
            assert id(node) in inside
    calls = sorted(node.args[0].value for node in ast.walk(tree)
                   if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == '_thread_route')
    assert calls == ['route', 'validate_switches']


def test_route_is_the_single_entry(monkeypatch):                                              # FAILS-ON-OLD
    model, _ = joint()
    policy = IntersectionBatchPolicy()
    seen = []

    def plan(stages):
        seen.append(type(stages).__name__)
        return 'planned'
    monkeypatch.setenv(ENV, '4')
    assert pt.route(model, policy, 'exact', None, plan) == 'planned' and seen == ['ThreadStages']
    del seen[:]
    assert pt.route(model, policy, 'topology', None, plan) == 'planned' and seen == ['NoneType']      # not eligible
    del seen[:]

    def refusing(stages):
        seen.append(type(stages).__name__)
        if stages is not None:
            raise GeometryError('refused')
        return 'serial plan'
    assert pt.route(model, policy, 'exact', None, refusing) == 'serial plan' and seen == ['ThreadStages', 'NoneType']
    del seen[:]

    def defect(stages):
        seen.append(type(stages).__name__)
        raise KeyError('a defect')
    with pytest.raises(pt._RouteFailure, match=FAILED):
        pt.route(model, policy, 'exact', None, defect)
    assert seen == ['ThreadStages']                   # no serial repeat
    monkeypatch.setenv(ENV, '9')
    with pytest.raises(pt._RouteFailure, match=ENV):
        pt.route(model, policy, 'exact', None, plan)
