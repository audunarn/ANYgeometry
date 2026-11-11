"""Codex review round 1 of the thread route (2026-11-11): one reproduction per finding.

Every test marked FAILS-ON-OLD fails on 5466409 (the thread route before the fixes); the others
guard a boundary Codex named and already held. Contract after the fixes: the threaded route gives
serial's value, message, counters and records, or planning runs on the serial route.
"""
import threading
from types import SimpleNamespace

import pytest

from anygeometry import GeometryError, plan_intersections, to_dict
from anygeometry import arrangement_geometry as ag
from anygeometry import batch_intersections as bi
from anygeometry import _planning_threads as pt
from anygeometry import quadric_events as qe
from anygeometry import quadric_supports as qs
from anygeometry.batch_intersections import IntersectionBatchPolicy
from anygeometry.branch_algebra import BezierRuledSupport
from anygeometry.branch_curves import BezierQuadricCurve
from anygeometry.quadric_algebra import QuadricSupport
from anygeometry.runtime_diagnostics import collect_runtime_diagnostics, collect_runtime_records
from anygeometry.surfaces import Cylinder
from test_planning_threads import ENV, PROCESS, _clear, comparable_records, joint


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    _clear(monkeypatch)


def outcome(threads, patch, policy='connect'):
    """Message (or checksum), deterministic counters and record contents of one planning call."""
    if threads:
        patch.setenv(ENV, str(threads))
    else:
        patch.delenv(ENV, raising=False)
    model, faces = joint()
    before = to_dict(model)
    with collect_runtime_diagnostics() as counters, collect_runtime_records() as records:
        try:
            value = plan_intersections(model, faces, policy=policy).content_checksum
        except Exception as exc:
            value = (type(exc).__name__, str(exc), type(exc.__cause__).__name__)
    assert to_dict(model) == before
    return dict(value=value, counters={k: v for k, v in counters.items() if not k.endswith('_time_ns')},
                records=comparable_records(records))


# ---- finding 1: a refusal must carry serial's counters, not those of work serial never reached ----

def _refuse_unequal_cylinders(monkeypatch):
    original = bi._pair_paths

    def refusing(first, second, *args, **kwargs):
        a, b = first.support, second.support
        if isinstance(a, Cylinder) and isinstance(b, Cylinder) and a.radius != b.radius:
            raise GeometryError(f'refused radii {a.radius}/{b.radius}')
        return original(first, second, *args, **kwargs)
    monkeypatch.setattr(bi, '_pair_paths', refusing)


@pytest.mark.parametrize('threads', (1, 4, 8))
def test_f1_pair_refusal_has_the_serial_message_and_counters(threads, monkeypatch):     # FAILS-ON-OLD
    _refuse_unequal_cylinders(monkeypatch)
    serial = outcome(0, monkeypatch)
    assert serial['value'][0] == 'GeometryError' and 'refused radii' in serial['value'][1]
    threaded = outcome(threads, monkeypatch)
    assert threaded['value'] == serial['value']
    assert threaded['counters'] == serial['counters']
    assert threaded == serial


def _arranged_faces(monkeypatch):
    seen = []
    original = bi.arrange_material
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bi, 'arrange_material', lambda domain, *a, **k: seen.append(domain.face_id) or original(domain, *a, **k))
        outcome(0, patch)
    return seen


@pytest.mark.parametrize('threads', (1, 8))
def test_f1_face_refusal_has_the_serial_message_and_counters(threads, monkeypatch):
    faces = _arranged_faces(monkeypatch)
    target = faces[len(faces) // 2]
    original = bi.arrange_material

    def refusing(domain, *args, **kwargs):
        if domain.face_id >= target:
            raise GeometryError(f'refused face {domain.face_id}')
        return original(domain, *args, **kwargs)
    monkeypatch.setattr(bi, 'arrange_material', refusing)
    serial = outcome(0, monkeypatch)
    assert serial['value'][1] == f'intersection material arrangement face:{target}: refused face {target}'
    assert outcome(threads, monkeypatch) == serial


@pytest.mark.parametrize('threads', (1, 8))
def test_f1_normaliser_failure_of_an_earlier_face_wins_as_in_serial(threads, monkeypatch):   # FAILS-ON-OLD
    # Boundary named by Codex: serial normalises each arrangement before the next face is arranged.
    faces = _arranged_faces(monkeypatch)
    original = bi.arrange_material

    def refusing(domain, *args, **kwargs):
        if domain.face_id == faces[-1]:
            raise GeometryError('late arrangement refusal')
        return original(domain, *args, **kwargs)
    monkeypatch.setattr(bi, 'arrange_material', refusing)
    monkeypatch.setattr(bi, '_normalise_cells', lambda value: (_ for _ in ()).throw(GeometryError('normaliser refused')))
    serial = outcome(0, monkeypatch)
    assert serial['value'][1] == 'normaliser refused'
    assert outcome(threads, monkeypatch) == serial


def test_f1_refused_threaded_attempt_is_visible_only_as_a_clock(monkeypatch):            # FAILS-ON-OLD
    _refuse_unequal_cylinders(monkeypatch)
    monkeypatch.setenv(ENV, '4')
    model, faces = joint()
    with collect_runtime_diagnostics() as counters:
        with pytest.raises(GeometryError):
            plan_intersections(model, faces, policy='connect')
    assert 'intersection_thread_discarded_attempt_time_ns' in counters
    assert not any(key.startswith('intersection_thread_') and key.endswith('_stage_time_ns') for key in counters)


# ---- finding 2: a cancellation callback keeps planning on the serial route ----

def _cancelling(limit):
    polls, callers = [], set()

    def cancel():
        polls.append(1)
        callers.add(threading.get_ident())
        return len(polls) > limit
    return cancel, polls, callers


@pytest.mark.parametrize('limit', (0, 7, 400, 10 ** 9))
def test_f2_cancellation_callback_declines_the_thread_route(limit, monkeypatch):          # FAILS-ON-OLD
    results = []
    for threads in (0, 4):
        cancel, polls, callers = _cancelling(limit)
        if threads:
            monkeypatch.setattr(pt.ThreadStages, 'run', lambda *a, **k: pytest.fail('thread route used'))
        result = outcome(threads, monkeypatch, IntersectionBatchPolicy(cancellation_check=cancel))
        assert callers <= {threading.get_ident()}
        results.append((result, len(polls)))
    assert results[0] == results[1]
    if limit < 10 ** 9:
        assert results[0][0]['value'][1].endswith('intersection planning cancelled')


def test_f2_dispatcher_has_no_cancellation_path():                                        # FAILS-ON-OLD
    with pytest.raises(TypeError):
        pt.ThreadStages(2, lambda: False)
    assert pt.thread_stages(joint()[0], IntersectionBatchPolicy(), 'exact', None) is None      # switch unset


# ---- finding 3: memo size bounds hold when two threads evict at the same time ----

class Colliding(dict):
    """A memo whose first eviction on the main thread lets another thread store an entry first."""
    other = thread = None

    def pop(self, *args):
        if self.other is not None and threading.current_thread() is threading.main_thread():
            other, self.other = self.other, None
            self.thread = threading.Thread(target=other)
            self.thread.start()
            self.thread.join(1.)        # old code: the other thread finishes here; fixed: it waits for the lock
        return super().pop(*args)


def _finish(memo):
    assert memo.thread is not None
    memo.thread.join(20.)
    assert not memo.thread.is_alive()


def test_f3_root_memo_bound_survives_an_eviction_collision(monkeypatch):                  # FAILS-ON-OLD
    memo = Colliding((('old', i), ()) for i in range(4))
    monkeypatch.setattr(qe, '_MEMO', memo)
    monkeypatch.setattr(qe, '_MEMO_SIZE', 4)
    monkeypatch.setattr(qe, 'QuadricSupport', SimpleNamespace(from_surface=lambda value: value))
    monkeypatch.setattr(qe, '_curve_quadric_roots', lambda curve, *a: (float(curve),))
    memo.other = lambda: qe.curve_quadric_roots(200, 'q')
    assert qe.curve_quadric_roots(100, 'q') == (100.,)
    _finish(memo)
    assert len(memo) <= 4
    assert qe.curve_quadric_roots(100, 'q') == (100.,) and qe.curve_quadric_roots(200, 'q') == (200.,)
    for value in range(300, 340):                       # the ordinary serial eviction still holds the bound
        qe.curve_quadric_roots(value, 'q')
        assert len(memo) == 4


def test_f3_first_choice_bound_survives_an_eviction_collision(monkeypatch):               # FAILS-ON-OLD
    memo = Colliding((('old', i), True) for i in range(4096))
    monkeypatch.setattr(qs, '_FIRST_CHOICE', memo)
    monkeypatch.setattr(qs, 'ruled_support', lambda value: value)
    monkeypatch.setattr(qs, 'QuadricSupport', SimpleNamespace(from_surface=lambda value: value))
    plan = SimpleNamespace(discriminant_roots=lambda: (), pole_roots=lambda: ())
    monkeypatch.setattr(qs, 'get_plan', lambda first, second: plan)
    memo.other = lambda: qs._prefer_first(3, 4)
    assert qs._prefer_first(1, 2) is True
    _finish(memo)
    assert len(memo) <= 4096


# ---- finding 4: the allow-list memo must not hand one curve's answer to an equal curve of other types ----

class Text(str):
    pass


def _curves():
    first = BezierRuledSupport(((0., 0., 0.), (.5, 0., 0.), (1., 1., 0.)), (0., 0., 1.))
    second = QuadricSupport("plane", (0., 0., 0.), (0., 0., 1.))
    plain = BezierQuadricCurve(first, second, 0., 1.)
    odd = BezierQuadricCurve(first, second, 0., 1., parameterization=Text(plain.parameterization))
    assert plain == odd and hash(plain) == hash(odd)
    assert ag._leaves_moderate(plain) is True and ag._leaves_moderate(odd) is False
    return plain, odd


@pytest.mark.parametrize('order', ('plain_first', 'odd_first'))
def test_f4_allow_list_memo_answer_does_not_depend_on_history(order):                     # FAILS-ON-OLD
    plain, odd = _curves()
    ag._leaves_moderate_cached.cache_clear()
    sequence = (plain, odd) if order == 'plain_first' else (odd, plain)
    for _ in range(3):
        for curve in sequence:
            assert ag._inputs_moderate(curve) is (curve is plain)
    ag._leaves_moderate_cached.cache_clear()


def test_f4_allow_list_memo_still_remembers_a_curve():
    plain, _ = _curves()
    ag._leaves_moderate_cached.cache_clear()
    for _ in range(5):
        assert ag._inputs_moderate(plain) is True
    info = ag._leaves_moderate_cached.cache_info()
    assert (info.misses, info.hits) == (1, 4)
    ag._leaves_moderate_cached.cache_clear()


# ---- findings 5 and 6: the switch itself ----

def test_f5_both_switches_are_refused_by_the_public_entry(monkeypatch):                   # FAILS-ON-OLD
    monkeypatch.setenv(ENV, '2')
    monkeypatch.setenv(PROCESS, '1')
    model, faces = joint()
    with pytest.raises(GeometryError, match='mutually exclusive'):
        plan_intersections(model, faces, policy='connect')


@pytest.mark.parametrize('value', ('1' * 5000, '0' * 5000 + '4', '4' + ' ' * 10, '٤'))
def test_f6_any_unsupported_switch_value_is_a_typed_refusal(value, monkeypatch):          # FAILS-ON-OLD (first value)
    monkeypatch.setenv(ENV, value)
    model, faces = joint()
    with pytest.raises(GeometryError, match=ENV) as error:
        plan_intersections(model, faces, policy='connect')
    assert len(str(error.value)) < 300


# ---- boundary named by Codex: only audited boundary curves take the thread route ----

def test_boundary_curve_whitelist_keeps_other_models_serial(monkeypatch):                 # FAILS-ON-OLD
    monkeypatch.setenv(ENV, '4')
    model, _ = joint()
    assert isinstance(pt.thread_stages(model, IntersectionBatchPolicy(), 'exact', None), pt.ThreadStages)
    kinds = {type(edge.curve) for edge in model.edges.values()}
    assert kinds <= set(pt._AUDITED_EDGE_CURVES)
    for kind in kinds:
        monkeypatch.setattr(pt, '_AUDITED_EDGE_CURVES', tuple(k for k in pt._AUDITED_EDGE_CURVES if k is not kind))
        assert pt.thread_stages(model, IntersectionBatchPolicy(), 'exact', None) is None
        monkeypatch.undo()
        _clear(monkeypatch)
        monkeypatch.setenv(ENV, '4')
