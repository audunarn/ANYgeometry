"""Codex review round 13 of the thread route (2026-11-11; the complete review of the module).

1. A plain refusal is decided by provenance, not only by type: the GeometryError a stage builds from its
   lowest-index refused task is the only ordinary exception that may leave a stage without marking the attempt
   as failed. A GeometryError of the route's own machinery (for example raised while that refusal is formatted)
   is a typed refusal, not a serial repeat.
2. ``plan_with_threads`` lets no ordinary exception of the attempt out except through the decision: start-up,
   publication after success, the decision itself, the discard clock. ``thread_stages`` answers a failure of the
   eligibility test with a typed GeometryError, and a scope module that is still being imported is not a failure.
3. The decision never raises (a failure whose class cannot even be named).
4. A stage called on its own has no serial route: a plain refusal is re-raised there (contract wording).

Tests marked FAILS-ON-OLD fail on c3647d8. Bounded; no thread object and no registry entry may be left behind.
"""
import faulthandler
import sys
import threading
import types

import pytest

from anygeometry import GeometryError
from anygeometry import _planning_threads as pt
from anygeometry.batch_intersections import IntersectionBatchPolicy
from anygeometry.runtime_diagnostics import _increment, collect_runtime_diagnostics
from test_planning_threads import ENV, _clear, joint

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


def _attempt(threaded):
    """(outcome, the exception object or None, whether serial ran) of one boundary call."""
    seen = {}

    def serial():
        seen['serial'] = True
        return 'serial'
    try:
        return ('returned', pt.plan_with_threads(threaded, serial)), None, bool(seen)
    except BaseException as exc:
        return ('raised', type(exc), str(exc)), exc, bool(seen)


def _stage(task, threads=2):
    return lambda: pt.ThreadStages(threads).run('face', _two(), task)


# ---- 1. provenance of a plain refusal ----

class Unprintable(GeometryError):
    """A refusal whose text cannot be built: formatting it raises another GeometryError."""
    def __str__(self):
        raise GeometryError('raised while the refusal was formatted')


def test_geometry_error_of_the_wrapping_machinery_is_not_a_plain_refusal():                   # FAILS-ON-OLD
    def task(item, check=None):
        raise Unprintable()
    outcome, error, serial = _attempt(_stage(task))
    assert outcome[:2] == ('raised', GeometryError) and FAILED in outcome[2] and not serial
    assert type(error.__cause__) is GeometryError and str(error.__cause__) == 'raised while the refusal was formatted'
    with pytest.raises(GeometryError, match=FAILED):          # the same from a stage called on its own
        pt.ThreadStages(2).run('face', _two(), task)


def test_geometry_error_of_the_worker_machinery_is_not_a_plain_refusal(monkeypatch):          # FAILS-ON-OLD
    real, doomed = pt.perf_counter_ns, set()

    def clock():
        if threading.get_ident() in doomed:
            doomed.discard(threading.get_ident())
            raise GeometryError('raised by the bookkeeping, not by geometry')
        return real()
    monkeypatch.setattr(pt, 'perf_counter_ns', clock)

    def task(item, check=None):
        if item['face'] == 1:
            doomed.add(threading.get_ident())
        return item['face']
    outcome, error, serial = _attempt(_stage(task))
    assert outcome[:2] == ('raised', GeometryError) and FAILED in outcome[2] and not serial
    assert str(error.__cause__) == 'raised by the bookkeeping, not by geometry'


def test_stage_failure_that_a_caller_swallowed_still_forbids_the_serial_repeat():             # FAILS-ON-OLD
    def task(item, check=None):
        raise ValueError('a defect')

    def threaded():
        try:
            pt.ThreadStages(2).run('face', _two(), task)
        except ValueError:
            pass
        raise GeometryError('a later refusal of the planner')
    outcome, error, serial = _attempt(threaded)
    assert outcome[:2] == ('raised', GeometryError) and FAILED in outcome[2] and not serial
    assert str(error.__cause__) == 'a later refusal of the planner'


def test_the_plain_refusals_keep_their_serial_repeat():
    def task(item, check=None):
        raise GeometryError(f"refused {item['face']}")
    assert _attempt(_stage(task))[::2] == (('returned', 'serial'), True)

    def between():
        assert pt.ThreadStages(2).run('face', _two(), _fine) == [0, 1]
        raise GeometryError('geometry changed during intersection planning')
    assert _attempt(between)[::2] == (('returned', 'serial'), True)

    def unaudited():
        pt.ThreadStages(2).admit_face_inputs({1: [object()]}, {1: []})
    assert _attempt(unaudited)[::2] == (('returned', 'serial'), True)

    def refused_then_refused():                       # a stage's own refusal, swallowed, then a planner refusal
        try:
            pt.ThreadStages(2).run('face', _two(), task)
        except GeometryError:
            pass
        raise GeometryError('a later refusal of the planner')
    assert _attempt(refused_then_refused)[::2] == (('returned', 'serial'), True)


def test_stage_marks_only_its_own_refusal_object():                                           # FAILS-ON-OLD
    def task(item, check=None):
        raise GeometryError(f"refused {item['face']}")

    def threaded():
        attempt = pt._ATTEMPT.get()
        with pytest.raises(GeometryError) as error:
            pt.ThreadStages(2).run('face', _two(), task)
        assert attempt.refusal is error.value and not attempt.failed
        assert str(error.value) == 'intersection material arrangement face:0: refused 0'
        assert str(error.value.__cause__) == 'refused 0' and error.value.__suppress_context__
        raise error.value
    assert _attempt(threaded)[::2] == (('returned', 'serial'), True)


# ---- 4. a stage called on its own ----

def test_stage_called_on_its_own_re_raises_a_plain_refusal():
    def task(item, check=None):
        raise GeometryError(f"refused {item['face']}")
    with pytest.raises(GeometryError) as error:
        pt.ThreadStages(2).run('face', _two(), task)
    assert str(error.value) == 'intersection material arrangement face:0: refused 0'


# ---- 2. no ordinary exception leaves the boundary undecided ----

def test_failure_while_publishing_a_successful_attempt_is_a_typed_refusal():                 # FAILS-ON-OLD
    def threaded():
        _increment('test_published_counter', 3)
        return pt.ThreadStages(2).run('face', _two(), _fine)
    with collect_runtime_diagnostics() as counters:
        counters['test_published_counter'] = 'not a number'      # publication cannot add to this
        outcome, error, serial = _attempt(threaded)
    assert outcome[:2] == ('raised', GeometryError) and FAILED in outcome[2] and not serial
    assert type(error.__cause__) is TypeError


def test_failure_of_the_discard_clock_is_a_typed_refusal(monkeypatch):                        # FAILS-ON-OLD
    real = pt._diagnostics._increment

    def increment(key, amount=1):
        if key == 'intersection_thread_discarded_attempt_time_ns':
            raise OverflowError('clock')
        return real(key, amount)
    monkeypatch.setattr(pt._diagnostics, '_increment', increment)

    def task(item, check=None):
        raise GeometryError('ordinary refusal')
    outcome, error, serial = _attempt(_stage(task))
    assert outcome[:2] == ('raised', GeometryError) and 'while its outcome was decided' in outcome[2] and not serial
    assert type(error.__cause__) is OverflowError


def test_failure_before_the_attempt_exists_is_a_typed_refusal(monkeypatch):                   # FAILS-ON-OLD
    monkeypatch.setattr(pt, '_Attempt', lambda: (_ for _ in ()).throw(MemoryError('attempt record')))
    ran = []
    outcome, error, serial = _attempt(lambda: ran.append(1))
    assert outcome[:2] == ('raised', GeometryError) and 'could not start' in outcome[2] and not serial and not ran
    assert type(error.__cause__) is MemoryError


def test_eligibility_survives_a_scope_module_that_is_still_being_imported(monkeypatch):      # FAILS-ON-OLD
    monkeypatch.setenv(ENV, '4')
    model, _ = joint()
    policy = IntersectionBatchPolicy()
    name = pt._SCOPE_VARIABLES[0][0]
    monkeypatch.setitem(sys.modules, name, types.ModuleType(name))       # in sys.modules, variable not yet set
    assert isinstance(pt.thread_stages(model, policy, 'exact', None), pt.ThreadStages)


def test_failure_of_the_eligibility_test_is_a_typed_refusal(monkeypatch):                    # FAILS-ON-OLD
    monkeypatch.setenv(ENV, '4')
    model, _ = joint()
    monkeypatch.setattr(pt, '_audited_model', lambda model: (_ for _ in ()).throw(KeyError('model')))
    with pytest.raises(GeometryError, match='eligibility could not be decided') as error:
        pt.thread_stages(model, IntersectionBatchPolicy(), 'exact', None)
    assert type(error.value.__cause__) is KeyError
    monkeypatch.setenv(ENV, '9')                      # the switch refusal itself is unchanged
    with pytest.raises(GeometryError, match=ENV):
        pt.thread_stages(model, IntersectionBatchPolicy(), 'exact', None)


# ---- 3. the decision never raises ----

class Nameless(type):
    @property
    def __name__(cls):
        raise RuntimeError('the class cannot be named')


class Anonymous(Exception, metaclass=Nameless):
    pass


def test_decision_does_not_depend_on_naming_the_failure():                                   # FAILS-ON-OLD
    def task(item, check=None):
        raise Anonymous('a defect')
    outcome, error, serial = _attempt(_stage(task))
    assert outcome[:2] == ('raised', GeometryError) and FAILED in outcome[2] and not serial
    assert type(error.__cause__) is Anonymous
    assert pt._type_name(Anonymous('x')) == 'Anonymous' and pt._type_name(ValueError()) == 'ValueError'
    attempt = pt._Attempt()
    assert type(attempt.verdict(Anonymous('x'))) is GeometryError
