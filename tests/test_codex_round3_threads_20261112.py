"""Codex review round 3 of the thread route (2026-11-11): one reproduction per finding.

Tests marked FAILS-ON-OLD fail on c145d49 (the route before these fixes).
1. The switches are validated at the public entry before the existing-material and bounded early returns.
2. Records are compared exactly; the only excluded values are the clock fields named in
   ``_planning_threads.RECORD_CLOCK_FIELDS`` (and the ``*_time_ns`` counters).
3. A worker's KeyboardInterrupt / SystemExit propagates even when a lower-index task failed ordinarily.
4. The plane_split plan-store scan sees attribute access (``plane_split._PLANS.get``).
5. The allow-list memo shares an answer between equal curves of the same exact types (as on main) and never
   between equal curves whose exact types differ.
"""
import ast
from pathlib import Path

import pytest

import anygeometry
from anygeometry import GeometryError, plan_intersections
from anygeometry import arrangement_geometry as ag
from anygeometry import bounded_nurbs
from anygeometry import existing_material_preparation
from anygeometry import _planning_threads as pt
from anygeometry.arrangement_geometry import LinePath
from test_codex_findings_threads_20261111 import _curves
from test_planning_threads import ENV, PROCESS, _clear, comparable_records, joint

PLANNED = object()


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    _clear(monkeypatch)


# ---- 1. switch validation on every path of the public entry ----

def _existing(monkeypatch):
    monkeypatch.setattr(existing_material_preparation, 'try_plan_existing', lambda *args, **kwargs: PLANNED)


def _bounded(monkeypatch):
    monkeypatch.setattr(bounded_nurbs, 'handles_bounded', lambda *args, **kwargs: True)
    monkeypatch.setattr(bounded_nurbs, 'plan_bounded', lambda *args, **kwargs: PLANNED)


def _general(monkeypatch):
    pass


ROUTES = (_existing, _bounded, _general)


@pytest.mark.parametrize('route', ROUTES)
def test_both_switches_refuse_on_every_route(route, monkeypatch):                       # FAILS-ON-OLD (2 of 3)
    route(monkeypatch)
    monkeypatch.setenv(ENV, '2')
    monkeypatch.setenv(PROCESS, '1')
    model, faces = joint()
    with pytest.raises(GeometryError) as error:
        plan_intersections(model, faces, policy='connect')
    assert str(error.value) == f'{ENV} and {PROCESS} are mutually exclusive'


@pytest.mark.parametrize('route', ROUTES)
@pytest.mark.parametrize('value', ('9', '-1', 'four', '1' * 5000, '01', '08', '00', ' 4', '4 '))
def test_unsupported_thread_count_refuses_identically_on_every_route(route, value, monkeypatch):   # FAILS-ON-OLD
    monkeypatch.setenv(ENV, value)
    model, faces = joint()
    with pytest.raises(GeometryError) as general:
        plan_intersections(model, faces, policy='connect')
    route(monkeypatch)
    with pytest.raises(GeometryError) as error:
        plan_intersections(model, faces, policy='connect')
    assert str(error.value) == str(general.value) and ENV in str(error.value)
    monkeypatch.setenv(PROCESS, '1')                 # the same refusal also when the process route is asked for
    with pytest.raises(GeometryError) as both:
        plan_intersections(model, faces, policy='connect')
    assert str(both.value) == str(general.value)


@pytest.mark.parametrize('route', (_existing, _bounded))
@pytest.mark.parametrize('value', (None, '', '0', '1', '8'))
def test_a_valid_switch_leaves_the_early_routes_alone(route, value, monkeypatch):
    route(monkeypatch)
    if value is not None:
        monkeypatch.setenv(ENV, value)
    model, faces = joint()
    assert plan_intersections(model, faces, policy='connect') is PLANNED


def test_switch_unset_never_loads_the_thread_module_check(monkeypatch):
    # Default off: the public entry does not consult the thread module at all.
    monkeypatch.setattr(pt, 'validate_switches', lambda: (_ for _ in ()).throw(AssertionError('consulted')))
    monkeypatch.setattr(pt, 'thread_stages', lambda *args: (_ for _ in ()).throw(AssertionError('consulted')))
    for value in (None, '', '0'):
        if value is not None:
            monkeypatch.setenv(ENV, value)
        model, faces = joint()
        assert plan_intersections(model, faces, policy='connect').content_checksum


# ---- 2. exact record comparison ----

def test_only_the_named_clock_fields_are_excluded_from_record_identity():               # FAILS-ON-OLD
    assert pt.RECORD_CLOCK_FIELDS == {'arrangement_snap_fallback': ('seconds',)}
    snap = dict(face_id=3, moved_edges=2, seconds=.25)
    assert comparable_records({'arrangement_snap_fallback': [snap]}) == comparable_records(
        {'arrangement_snap_fallback': [dict(snap, seconds=9.)]})
    assert comparable_records({'arrangement_snap_fallback': [snap]}) != comparable_records(
        {'arrangement_snap_fallback': [dict(snap, moved_edges=3)]})
    # A field that merely contains 'seconds', or 'seconds' under another record key, is content.
    assert comparable_records({'ring_split': [dict(seconds=1.)]}) != comparable_records({'ring_split': [dict(seconds=2.)]})
    assert comparable_records({'arrangement_snap_fallback': [dict(snap, seconds_budget=1)]}) != comparable_records(
        {'arrangement_snap_fallback': [dict(snap, seconds_budget=2)]})
    # No lossy conversion: container kind and number type are content, nested values are not flattened.
    assert comparable_records({'x': [dict(a=(1, 2))]}) != comparable_records({'x': [dict(a=[1, 2])]})
    assert comparable_records({'x': [dict(a='1')]}) != comparable_records({'x': [dict(a=1)]})
    assert comparable_records({'x': [dict(a=dict(seconds=1))]}) != comparable_records({'x': [dict(a=dict(seconds=2))]})
    assert comparable_records({'x': [1, 2]}) != comparable_records({'x': [2, 1]})


def test_every_record_clock_field_in_the_package_is_named():
    """The only record item with a clock value is the snap fallback's ``seconds`` (source scan of every
    ``_record(...)`` call: a new record key must be judged here before it can pass)."""
    root = Path(anygeometry.__file__).parent
    keys = set()
    for path in sorted(root.rglob('*.py')):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if (isinstance(node, ast.Call) and getattr(node.func, 'id', getattr(node.func, 'attr', None)) == '_record'
                    and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                keys.add(node.args[0].value)
    assert keys == {'arrangement_snap_fallback', 'ring_split', 'parallel_planning'}
    assert set(pt.RECORD_CLOCK_FIELDS) <= keys


# ---- 3. interrupts of a worker ----

@pytest.mark.parametrize('fatal', (KeyboardInterrupt, SystemExit))
@pytest.mark.parametrize('threads', (1, 4))
def test_worker_interrupt_wins_over_an_ordinary_failure_of_a_lower_task(fatal, threads):   # FAILS-ON-OLD
    tasks = [dict(kind='face', face=0, cost=1.), dict(kind='face', face=1, cost=9.), dict(kind='face', face=2, cost=5.)]

    def task(item, check=None):
        if item['face'] == 1:                         # highest cost: runs first
            raise fatal('stop')
        raise GeometryError('ordinary refusal')
    with pytest.raises(fatal):
        pt.ThreadStages(threads).run('face', tasks, task)
    with pytest.raises(fatal):                        # and it is not turned into a serial repeat
        pt.plan_with_threads(lambda: pt.ThreadStages(threads).run('face', tasks, task), lambda: 'serial')


def test_ordinary_failures_still_report_the_lowest_task():
    tasks = [dict(kind='face', face=index, cost=float(index)) for index in range(4)]

    def task(item, check=None):
        if item['face'] in (1, 3):
            raise GeometryError(f"refused {item['face']}")
        return item['face']
    with pytest.raises(GeometryError, match='face:1: refused 1'):
        pt.ThreadStages(2).run('face', tasks, task)


# ---- 4. the plan-store scan ----

def test_plan_store_scan_sees_attribute_access_and_module_binding():                    # FAILS-ON-OLD
    from test_codex_boundaries_threads_20261112 import plan_store_uses
    for source in ('from . import plane_split\nplane_split._PLANS.get(1)\n',
                   'import anygeometry.plane_split as ps\nps._PLANS[1] = 2\n',
                   'from .plane_split import _PLANS\n',
                   'from .plane_split import *\n',
                   'from . import plane_split as other\n',
                   'from .plane_split import plan_plane_split\nplan_plane_split()\n',
                   "import sys\ngetattr(sys.modules['anygeometry.plane_split'], '_PLANS')\n",
                   'from .plane_split import apply_plane_split as go\n'):
        assert plan_store_uses(ast.parse(source)), source
    for source in ('from .plane_split import _stage_plane_split\n_stage_plane_split()\n',
                   '_PLANS = {}\n_PLANS[1] = 2\n',                  # another module's own store of the same name
                   'from .plane_split import _plane_sign, _qualify_sheet_ownership\n'):
        assert not plan_store_uses(ast.parse(source)), source


# ---- 5. allow-list memo key ----

def test_equal_curves_of_the_same_exact_types_share_one_memo_entry_as_on_main():        # FAILS-ON-OLD
    first, _ = _curves()
    second, _ = _curves()
    assert first is not second and first == second and hash(first) == hash(second)
    ag._leaves_moderate_cached.cache_clear()
    assert ag._inputs_moderate(first) is True and ag._inputs_moderate(second) is True
    info = ag._leaves_moderate_cached.cache_info()
    assert (info.misses, info.hits, info.currsize) == (1, 1, 1)
    ag._leaves_moderate_cached.cache_clear()


@pytest.mark.parametrize('order', ('plain_first', 'odd_first'))
def test_equal_curves_of_other_exact_types_never_share_an_answer(order):
    plain, odd = _curves()                            # equal and same hash; one field is a str subclass
    ag._leaves_moderate_cached.cache_clear()
    for _ in range(3):
        for curve in ((plain, odd) if order == 'plain_first' else (odd, plain)):
            assert ag._inputs_moderate(curve) is (curve is plain)
    assert ag._leaves_moderate_cached.cache_info().currsize == 2
    ag._leaves_moderate_cached.cache_clear()


def test_memo_key_never_changes_the_decision():
    plain, odd = _curves()
    again, odd_again = _curves()
    curves = [plain, again, odd, odd_again, plain, LinePath((0., 0., 0.), (1., 0., 0.))]
    ag._leaves_moderate_cached.cache_clear()
    for _ in range(2):
        assert [ag._inputs_moderate(curve) for curve in curves] == [ag._leaves_moderate(curve) for curve in curves]
    assert ag._leaves_moderate_cached.cache_info().currsize == 2
    ag._leaves_moderate_cached.cache_clear()
