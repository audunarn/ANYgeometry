"""Codex review round 5 of the thread route (2026-11-11): the interrupt rule is enforced at ONE boundary.

Rule (``_planning_threads._Attempt.verdict``, applied by ``plan_with_threads`` and by a directly called
``ThreadStages.run``): when the threaded attempt ends with an ordinary exception,
* an interrupt (KeyboardInterrupt / SystemExit) seen anywhere in the stage - by ``wait()``, recorded by a worker,
  or met in the dispatcher's cleanup - is raised instead, whatever ordinary exception arrived;
* otherwise, if the workers are not known to have finished (``pool.shutdown`` did not return), a typed
  GeometryError is raised: no serial planning beside workers that may still run;
* only otherwise planning is repeated on the serial route.

Tests marked FAILS-ON-OLD fail on 5d2d3ae. Also closed: the four test limits Codex named in round 5.
"""
import ast
import threading
from anygeometry._planning_threads import _Crew as ThreadPoolExecutor   # the crew of a stage (since round 10)

import pytest

from anygeometry import GeometryError
from anygeometry import _planning_threads as pt
from test_codex_boundaries_threads_20261112 import plain_reexport, plan_store_uses
from test_codex_round4_threads_20261112 import clocked_record_fields, record_calls
from test_planning_threads import _clear, comparable_records

WAIT = 30
FATAL = (KeyboardInterrupt, SystemExit)


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    # Bounded: a hang (for example in an executor shutdown) dumps all stacks and ends the process.
    import faulthandler
    _clear(monkeypatch)
    faulthandler.dump_traceback_later(240, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


def _two():
    return [dict(kind='face', face=0, cost=1.), dict(kind='face', face=1, cost=2.)]


def _failing_shutdown(monkeypatch, error, seen):
    """The pool really drains, then its shutdown reports ``error`` (once)."""
    class Pool(ThreadPoolExecutor):
        def shutdown(self, wait=True, **options):
            super().shutdown(wait=wait, **options)
            if not seen.get('shutdown'):
                seen['shutdown'] = True
                raise error('shutdown')
    monkeypatch.setattr(pt, '_Crew', Pool)


def _interrupted_wait(monkeypatch, fatal, seen):
    """``wait`` returns only after every worker is done, then the dispatching thread is interrupted."""
    real = pt.wait

    def waiting(futures, *args, **kwargs):
        real(futures, *args, **kwargs)
        seen['wait'] = True
        raise fatal('during wait')
    monkeypatch.setattr(pt, 'wait', waiting)


def _serial(seen):
    def serial():
        seen['serial'] = True
        return 'serial'
    return serial


def _fine(item, check=None):
    return item['face']


@pytest.mark.parametrize('fatal', FATAL)
@pytest.mark.parametrize('direct', (False, True))
def test_interrupt_of_wait_survives_an_ordinary_shutdown_failure(fatal, direct, monkeypatch):      # FAILS-ON-OLD
    seen = {}
    _interrupted_wait(monkeypatch, fatal, seen)
    _failing_shutdown(monkeypatch, MemoryError, seen)
    with pytest.raises(fatal, match='during wait'):
        if direct:
            pt.ThreadStages(2).run('face', _two(), _fine)
        else:
            pt.plan_with_threads(lambda: pt.ThreadStages(2).run('face', _two(), _fine), _serial(seen))
    assert seen == dict(wait=True, shutdown=True)


@pytest.mark.parametrize('fatal', FATAL)
@pytest.mark.parametrize('direct', (False, True))
def test_interrupt_of_wait_survives_an_ordinary_failure_of_the_stop_signal(fatal, direct, monkeypatch):   # FAILS-ON-OLD
    seen = {}
    _interrupted_wait(monkeypatch, fatal, seen)
    main = threading.get_ident()

    class Event(threading.Event):
        def set(self):
            if threading.get_ident() == main and seen.get('wait') and not seen.get('stop'):
                seen['stop'] = True
                raise MemoryError('stop signal')
            super().set()

    class Threading:                                  # the module as the dispatcher sees it, with this Event
        def __getattr__(self, name):
            return Event if name == 'Event' else getattr(threading, name)
    monkeypatch.setattr(pt, 'threading', Threading())
    with pytest.raises(fatal, match='during wait'):
        if direct:
            pt.ThreadStages(2).run('face', _two(), _fine)
        else:
            pt.plan_with_threads(lambda: pt.ThreadStages(2).run('face', _two(), _fine), _serial(seen))
    assert seen == dict(wait=True, stop=True)


@pytest.mark.parametrize('fatal', FATAL)
@pytest.mark.parametrize('direct', (False, True))
def test_interrupt_recorded_by_a_worker_survives_an_ordinary_shutdown_failure(fatal, direct, monkeypatch):   # FAILS-ON-OLD
    seen = {}
    _failing_shutdown(monkeypatch, MemoryError, seen)

    def task(item, check=None):
        if item['face'] == 1:
            seen['interrupt'] = True
            raise fatal('in worker')
        return item['face']
    with pytest.raises(fatal, match='in worker'):
        if direct:
            pt.ThreadStages(2).run('face', _two(), task)
        else:
            pt.plan_with_threads(lambda: pt.ThreadStages(2).run('face', _two(), task), _serial(seen))
    assert seen == dict(interrupt=True, shutdown=True)


@pytest.mark.parametrize('direct', (False, True))
def test_no_serial_repeat_while_workers_are_not_known_to_have_finished(direct, monkeypatch):   # FAILS-ON-OLD
    seen = {}
    _failing_shutdown(monkeypatch, MemoryError, seen)
    with pytest.raises(GeometryError, match='did not finish') as error:
        if direct:
            pt.ThreadStages(2).run('face', _two(), _fine)
        else:
            pt.plan_with_threads(lambda: pt.ThreadStages(2).run('face', _two(), _fine), _serial(seen))
    assert seen == dict(shutdown=True) and isinstance(error.value.__cause__, MemoryError)


@pytest.mark.parametrize('fatal', FATAL)
def test_interrupt_during_cleanup_after_an_ordinary_task_failure_is_raised(fatal, monkeypatch):
    seen = {}
    _failing_shutdown(monkeypatch, fatal, seen)

    def task(item, check=None):
        raise GeometryError('ordinary refusal')
    with pytest.raises(fatal, match='shutdown'):
        pt.plan_with_threads(lambda: pt.ThreadStages(2).run('face', _two(), task), _serial(seen))
    assert seen == dict(shutdown=True)


def test_interrupt_of_an_earlier_stage_is_not_forgotten_by_a_later_ordinary_failure(monkeypatch):   # FAILS-ON-OLD
    # The attempt, not one stage, owns the rule: a caller that (wrongly) swallowed a stage interrupt and then
    # failed ordinarily still ends in the interrupt, never in the serial repeat.
    seen = {}

    def task(item, check=None):
        raise SystemExit('in worker')

    def threaded():
        try:
            pt.ThreadStages(2).run('pair', [dict(kind='pair', first=0, second=1, cost=1.)], task)
        except SystemExit:
            pass
        raise RuntimeError('later ordinary failure')
    with pytest.raises(SystemExit, match='in worker'):
        pt.plan_with_threads(threaded, _serial(seen))
    assert seen == {}


def test_ordinary_failures_still_lead_to_the_serial_repeat_and_success_is_unchanged():
    seen = {}

    def task(item, check=None):
        raise GeometryError('ordinary refusal')
    assert pt.plan_with_threads(lambda: pt.ThreadStages(2).run('face', _two(), task), _serial(seen)) == 'serial'
    assert seen == dict(serial=True)
    assert pt.plan_with_threads(lambda: pt.ThreadStages(2).run('face', _two(), _fine), _serial(seen)) == [0, 1]
    assert pt.plan_with_threads(lambda: (_ for _ in ()).throw(pt._NotAudited('x')), lambda: 'serial') == 'serial'
    with pytest.raises(GeometryError, match='face:0: ordinary refusal'):
        pt.ThreadStages(2).run('face', _two(), task)


def test_attempts_do_not_leak_into_each_other():
    def task(item, check=None):
        raise SystemExit('in worker')
    with pytest.raises(SystemExit):
        pt.plan_with_threads(lambda: pt.ThreadStages(2).run('face', _two(), task), lambda: 'serial')
    assert pt._ATTEMPT.get() is None
    assert pt.plan_with_threads(lambda: (_ for _ in ()).throw(RuntimeError('x')), lambda: 'serial') == 'serial'


# ---- the four test limits named in round 5 ----

def test_record_comparison_keeps_the_row_kind_of_a_clocked_record():                    # FAILS-ON-OLD
    class Odd(dict):
        pass
    snap = dict(face_id=3, moved_edges=2, seconds=.25)
    assert comparable_records({'arrangement_snap_fallback': [snap]}) != comparable_records(
        {'arrangement_snap_fallback': [Odd(snap)]})
    assert comparable_records({'arrangement_snap_fallback': [snap]}) == comparable_records(
        {'arrangement_snap_fallback': [dict(snap, seconds=7.)]})
    other = type('Odd', (dict,), {})                  # another class with the same name
    assert comparable_records({'x': [Odd(a=1)]}) != comparable_records({'x': [other(a=1)]})


def test_clocked_record_census_sees_qualified_calls():                                  # FAILS-ON-OLD
    source = ("from . import runtime_diagnostics as diagnostics\n"
              "diagnostics._record('arrangement_snap_fallback', {'face_id': 1, 'extra': 2})\n"
              "_record('arrangement_snap_fallback', {'seconds': 1})\n"
              "_record('ring_split', {'other': 1})\n")
    assert clocked_record_fields(ast.parse(source), 'arrangement_snap_fallback') == [['extra', 'face_id'], ['seconds']]
    assert clocked_record_fields(ast.parse("_record('arrangement_snap_fallback', row)\n"), 'arrangement_snap_fallback') == [None]


def test_record_scan_is_not_suppressed_by_a_nested_function_or_a_wildcard_import():     # FAILS-ON-OLD
    nested = "def outer():\n    def _record(a, b):\n        pass\n_record(key, 1)\n"
    assert record_calls(ast.parse(nested))[1]
    assert record_calls(ast.parse("from .runtime_diagnostics import *\n_record('x', 1)\n"))[1]
    assert record_calls(ast.parse("class A:\n    def _record(self, a):\n        pass\n_record(key, 1)\n"))[1]
    assert record_calls(ast.parse('def _record(a, b):\n    pass\n_record(x, y)\n')) == (set(), [])


def test_reexport_check_resolves_aliases():                                             # FAILS-ON-OLD
    assert plain_reexport(ast.parse('from .plane_split import PlaneSplitPlan, plan_plane_split, apply_plane_split\n'
                                    '__all__ = ["plan_plane_split", "apply_plane_split"]\n')) == []
    for source in ('from .plane_split import plan_plane_split as go\ngo()\n',
                   'from .plane_split import plan_plane_split\ngo = plan_plane_split\ngo()\n',
                   'from .plane_split import plan_plane_split\nplan_plane_split()\n',
                   'from .plane_split import apply_plane_split\nhooks = [apply_plane_split]\n',
                   'from . import plane_split\n',
                   'from .plane_split import *\n'):
        assert plain_reexport(ast.parse(source)), source
    assert plan_store_uses(ast.parse('from .plane_split import plan_plane_split as go\n'))
