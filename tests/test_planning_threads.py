"""Opt-in thread dispatch of the pair and face stages (``ANYGEOMETRY_PLANNING_THREADS``).

The threaded route must give the plan, the prepared model, the deterministic counters and the
refusals of the serial route. These tests run on any interpreter (on a GIL build the threads only
interleave); on a free-threaded build they also exercise real parallel execution, and the repeated
runs make races visible as differences. The module does not import on the base a27ee2c (no
``_planning_threads``), so every test fails there.
"""
import json
import math
import threading
from uuid import UUID

import pytest

from anygeometry import GeometryError, GeometryModel, apply_intersections, plan_intersections, to_dict
from anygeometry import batch_intersections as bi
from anygeometry._planning_threads import RECORD_CLOCK_FIELDS
from anygeometry import _planning_threads as pt
from anygeometry.batch_intersections import IntersectionBatchPolicy
from anygeometry.generators import cylinder
from anygeometry.runtime_diagnostics import (_increment, _record, collect_runtime_diagnostics,
                                             collect_runtime_records)
from anygeometry.surfaces import Cylinder
from _exact_plan_comparison import exact_projection

ENV = pt.THREADS_ENVIRONMENT
PROCESS = 'ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING'


def _clear(patch):
    for name in (ENV, PROCESS, bi.PLANNING_MODE_ENVIRONMENT, bi.SEAM_POLICY_ENVIRONMENT):
        patch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    _clear(monkeypatch)


SEGMENTS = 4


def joint():
    """A chord tube, two braces through it and three plates: plane/plane, plane/cylinder and
    cylinder/cylinder pairs."""
    model = GeometryModel(model_id=UUID(int=20261110))     # fixed: plans and documents compare whole
    model.insert_model(cylinder(1., 6., origin=(-3., 0., 0.), axis=(1., 0., 0.), radial_direction=(0., 1., 0.),
                                circumferential_segments=SEGMENTS))
    model.insert_model(cylinder(.45, 4., origin=(-1.2, 0., -2.), axis=(0., 0., 1.), radial_direction=(1., 0., 0.),
                                circumferential_segments=SEGMENTS))
    model.insert_model(cylinder(.4, 4., origin=(1.1, -1.9, -.7), axis=(0., math.cos(.35), math.sin(.35)),
                                radial_direction=(1., 0., 0.), circumferential_segments=SEGMENTS))
    for points in (((-2.5, -2., .2), (2.5, -2., .2), (2.5, 2., .2), (-2.5, 2., .2)),
                   ((.3, -2., -2.), (.3, 2., -2.), (.3, 2., 2.), (.3, -2., 2.))):
        model.add_plate(model.add_points(points))
    return model, sorted(model.faces)


def comparable_records(records):
    """The records exactly as collected (keys, order, values and their types), without the clock fields
    named in the contract (``_planning_threads.RECORD_CLOCK_FIELDS``): nothing else is dropped or converted."""
    def exact(value):
        if isinstance(value, dict):
            return ('dict', tuple((key, exact(item)) for key, item in value.items()))
        if isinstance(value, (list, tuple)):
            return (type(value).__name__, tuple(exact(item) for item in value))
        return (type(value).__name__, repr(value))

    def row_of(key, row):
        clocks = RECORD_CLOCK_FIELDS.get(key, ())
        if isinstance(row, dict) and clocks:
            row = {name: item for name, item in row.items() if name not in clocks}
        return exact(row)
    return {key: [row_of(key, row) for row in rows] for key, rows in records.items()}


def run(threads, policy, patch, apply=True):
    """Plan (and apply) on a fresh model; returns everything that must be identical between routes."""
    if threads:
        patch.setenv(ENV, str(threads))
    else:
        patch.delenv(ENV, raising=False)
    model, faces = joint()
    with collect_runtime_diagnostics() as counters, collect_runtime_records() as records:
        plan = plan_intersections(model, faces, policy=policy)
        if apply:
            apply_intersections(model, plan, policy=policy)
    stable = {key: value for key, value in counters.items() if not key.endswith('_time_ns')}
    result = dict(plan=exact_projection(plan), checksum=plan.content_checksum,
                  document=json.dumps(to_dict(model), sort_keys=True, default=repr),
                  counters=stable, records=comparable_records(records))
    return result, {key for key in counters if key.endswith('_time_ns')}


POLICIES = {'baseline': IntersectionBatchPolicy(), 'ring_split': IntersectionBatchPolicy(seam_policy='ring_split')}


@pytest.fixture(scope='module')
def serial():
    with pytest.MonkeyPatch.context() as patch:
        _clear(patch)
        return {name: run(0, policy, patch) for name, policy in POLICIES.items()}


@pytest.mark.parametrize('name', sorted(POLICIES))
@pytest.mark.parametrize('threads', (1, 4, 8))
def test_threaded_plan_and_prepared_model_equal_serial(name, threads, serial, monkeypatch):
    reference, reference_timed = serial[name]
    result, timed = run(threads, POLICIES[name], monkeypatch)
    assert result['counters'] == reference['counters']
    assert result['plan'] == reference['plan']
    assert result['document'] == reference['document']
    assert result == reference
    # The route really was taken: its stage clocks exist only on the threaded route.
    assert {'intersection_thread_pair_stage_time_ns', 'intersection_thread_face_stage_time_ns'} <= timed
    assert not any(key.startswith('intersection_thread_') for key in reference_timed)
    assert {'intersection_support_and_clipping_time_ns', 'intersection_arrangements_time_ns'} <= timed


def test_repeated_threaded_runs_are_identical(serial, monkeypatch):
    reference, _ = serial['ring_split']
    for _ in range(8):
        assert run(8, POLICIES['ring_split'], monkeypatch, apply=False)[0]['plan'] == reference['plan']
    assert run(8, POLICIES['ring_split'], monkeypatch)[0] == reference


def test_tasks_run_on_worker_threads(monkeypatch):
    seen = set()
    original = bi._pair_paths

    def observed(*args, **kwargs):
        seen.add(threading.get_ident())
        return original(*args, **kwargs)
    monkeypatch.setattr(bi, '_pair_paths', observed)
    run(4, POLICIES['baseline'], monkeypatch, apply=False)
    assert seen and threading.get_ident() not in seen


def test_planner_refusal_equals_the_serial_refusal(monkeypatch):
    original = bi._pair_paths

    def refusing(first, second, *args, **kwargs):
        a, b = first.support, second.support
        if isinstance(a, Cylinder) and isinstance(b, Cylinder) and a.radius != b.radius:
            raise GeometryError(f'refused radii {a.radius}/{b.radius}')
        return original(first, second, *args, **kwargs)
    monkeypatch.setattr(bi, '_pair_paths', refusing)
    messages = []
    for threads in (0, 4, 8, 8):
        monkeypatch.setenv(ENV, str(threads))
        model, faces = joint()
        before = to_dict(model)
        with pytest.raises(GeometryError) as error:
            plan_intersections(model, faces, policy='connect')
        assert to_dict(model) == before
        messages.append(str(error.value))
    assert messages[0].startswith('intersection batch face:') and 'refused radii' in messages[0]
    assert messages == [messages[0]] * 4


def test_first_failure_in_task_order_wins_and_later_counters_are_dropped():
    stages = pt.ThreadStages(4)
    tasks = [dict(kind='pair', first=i, second=100 + i, cost=float(i + 1)) for i in range(12)]
    later_failed = threading.Event()

    def task(item, check):
        index = item['first']
        _increment('ran')
        _record('rows', index)
        if index == 9:
            later_failed.set()
            raise GeometryError('refused 9')
        if index == 3:
            assert later_failed.wait(10.)     # the later task fails first in time
            raise GeometryError('refused 3')
        if index == 6:
            raise ValueError('not a geometry refusal')
        check()
        return index
    with collect_runtime_diagnostics() as counters, collect_runtime_records() as records:
        with pytest.raises(GeometryError) as error:
            stages.run('pair', tasks, task)
    assert str(error.value) == 'intersection batch face:3/face:103: refused 3'
    assert isinstance(error.value.__cause__, GeometryError)
    assert counters['ran'] == 4 and records['rows'] == [0, 1, 2, 3]
    # A foreign exception is passed through unwrapped, as the serial loop does.
    with pytest.raises(ValueError, match='not a geometry refusal'):
        stages.run('face', [dict(kind='face', face=i, cost=1.) for i in range(4, 8)],
                   lambda item, check: task(dict(first=item['face']), check))


def test_results_and_published_counters_follow_task_order():
    stages = pt.ThreadStages(8)
    tasks = [dict(kind='face', face=i, cost=float((i * 7) % 5 + 1)) for i in range(200)]

    def task(item, check):
        _record('order', item['face'])
        _increment('total', item['face'])
        return item['face'] * 2
    with collect_runtime_diagnostics() as counters, collect_runtime_records() as records:
        assert stages.run('face', tasks, task) == [2 * i for i in range(200)]
    assert records['order'] == list(range(200)) and counters['total'] == sum(range(200))
    assert stages.run('pair', [], task) == []


def test_budgets_and_other_modes_stay_on_the_serial_route(monkeypatch):
    monkeypatch.setenv(ENV, '4')
    model, faces = joint()
    monkeypatch.setattr(pt.ThreadStages, 'run', lambda *a, **k: pytest.fail('thread route used'))
    with pytest.raises(GeometryError, match='budget'):
        plan_intersections(model, faces, policy=IntersectionBatchPolicy(max_predicates=5))
    with pytest.raises(GeometryError, match='maximum_candidate_pairs=1'):
        plan_intersections(model, faces, policy=IntersectionBatchPolicy(max_candidate_pairs=1))
    plan_intersections(model, faces, policy=IntersectionBatchPolicy(max_candidate_pairs=10 ** 9))


@pytest.mark.parametrize('value', ('9', '-1', 'two', '1.5', ' 2', '00x'))
def test_invalid_thread_count_is_refused(value, monkeypatch):
    monkeypatch.setenv(ENV, value)
    model, faces = joint()
    with pytest.raises(GeometryError, match=ENV):
        plan_intersections(model, faces, policy='connect')


def test_thread_and_process_routes_are_mutually_exclusive(monkeypatch):
    monkeypatch.setenv(ENV, '2')
    assert pt.requested_threads() == 2
    monkeypatch.setenv(PROCESS, '1')
    with pytest.raises(GeometryError, match='mutually exclusive'):
        pt.thread_stages(joint()[0], IntersectionBatchPolicy(), 'exact', None)


@pytest.mark.parametrize('value', (None, '', '0'))
def test_switch_off_never_consults_the_thread_route(value, monkeypatch):
    if value is not None:
        monkeypatch.setenv(ENV, value)
    monkeypatch.setattr(pt, 'thread_stages', lambda *a, **k: pytest.fail('thread route consulted'))
    model, faces = joint()
    plan_intersections(model, faces, policy='connect')
