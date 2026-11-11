"""Open boundaries of the thread route named by Codex round 2 (2026-11-11), closed 2026-11-12.

1. Whitelist: the thread stages run only on EFFECTIVE inputs of audited exact types - the material
   domains (support and boundary curves) before the pair stage, and every trace, member input and
   point before the face stage. Anything else declines to the serial route.
2. Cell normalisation runs inside the face task, directly after that face's arrangement, as in serial.
3. ``plane_split._PLANS`` is not reachable from a stage task.

Tests marked FAILS-ON-OLD fail on d8f8b82 (the route before this slice); the others are proofs of a
property that already held, or guard that the whitelist is not vacuous.
"""
import ast
import threading
from dataclasses import replace
from pathlib import Path

import pytest

import anygeometry
from anygeometry import GeometryError, plan_intersections, to_dict
from anygeometry import batch_intersections as bi
from anygeometry import _planning_threads as pt
from anygeometry import plane_split
from anygeometry.arrangement_geometry import LinePath
from anygeometry.material_arrangement import ArrangementPath, ArrangementPoint, MaterialDomain
from anygeometry.runtime_diagnostics import collect_runtime_diagnostics, collect_runtime_records
from test_planning_threads import ENV, _clear, comparable_records, joint

POOL = 'anygeometry-plan'


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    _clear(monkeypatch)


class OddLine(LinePath):
    """A line the audit never saw (a subclass may carry state of its own)."""
    __slots__ = ()


def _odd(curve):
    return OddLine(curve.start, curve.end) if type(curve) is LinePath else curve


def _chain(exc):
    rows = []
    while exc is not None:
        rows.append((type(exc).__name__, str(exc)))
        exc = exc.__cause__
    return rows


def planned(threads, patch):
    """Everything that must be equal between the routes, and the raw counters of the call."""
    if threads:
        patch.setenv(ENV, str(threads))
    else:
        patch.delenv(ENV, raising=False)
    model, faces = joint()
    before = to_dict(model)
    with collect_runtime_diagnostics() as counters, collect_runtime_records() as records:
        try:
            value = plan_intersections(model, faces, policy='connect').content_checksum
        except Exception as exc:
            value = _chain(exc)
    assert to_dict(model) == before
    same = dict(value=value, counters={k: v for k, v in counters.items() if not k.endswith('_time_ns')},
                records=comparable_records(records))
    return same, dict(counters)


def _stage_clocks(counters):
    return sorted(key for key in counters if key.startswith('intersection_thread_') and key.endswith('_stage_time_ns'))


def _thread_log(monkeypatch, name):
    """Wrap a planner function so that every call records the thread it ran on."""
    original, seen, lock = getattr(bi, name), [], threading.Lock()

    def logged(*args, **kwargs):
        with lock:
            seen.append(threading.current_thread().name)
        return original(*args, **kwargs)
    monkeypatch.setattr(bi, name, logged)
    return seen


# ---- 1. whitelist on the effective stage inputs ----

def test_audited_model_takes_both_thread_stages(monkeypatch):
    pair, face = _thread_log(monkeypatch, '_pair_paths'), _thread_log(monkeypatch, 'arrange_material')
    serial, _ = planned(0, monkeypatch)
    del pair[:], face[:]
    threaded, counters = planned(4, monkeypatch)
    assert threaded == serial and isinstance(serial['value'], str)
    assert _stage_clocks(counters) == ['intersection_thread_face_stage_time_ns', 'intersection_thread_pair_stage_time_ns']
    assert 'intersection_thread_discarded_attempt_time_ns' not in counters
    assert pair and face and all(name.startswith(POOL) for name in pair + face)


def test_unaudited_effective_boundary_curve_declines_before_the_pair_stage(monkeypatch):   # FAILS-ON-OLD
    original = MaterialDomain.from_model.__func__

    def substituted(cls, model, face_id):
        domain = original(cls, model, face_id)
        return replace(domain, boundaries=tuple(
            tuple(replace(path, curve=_odd(path.curve)) for path in loop) for loop in domain.boundaries))
    monkeypatch.setattr(MaterialDomain, 'from_model', classmethod(substituted))
    pair = _thread_log(monkeypatch, '_pair_paths')
    serial, _ = planned(0, monkeypatch)
    del pair[:]
    threaded, counters = planned(4, monkeypatch)
    assert threaded == serial
    assert pair and not any(name.startswith(POOL) for name in pair)
    assert _stage_clocks(counters) == [] and 'intersection_thread_discarded_attempt_time_ns' in counters


def test_unaudited_generated_trace_declines_before_the_face_stage(monkeypatch):            # FAILS-ON-OLD
    original = bi._pair_paths

    def generating(*args, **kwargs):
        return tuple(_odd(curve) for curve in original(*args, **kwargs))
    monkeypatch.setattr(bi, '_pair_paths', generating)
    face = _thread_log(monkeypatch, 'arrange_material')
    serial, _ = planned(0, monkeypatch)
    del face[:]
    threaded, counters = planned(4, monkeypatch)
    assert threaded == serial
    assert face and not any(name.startswith(POOL) for name in face)
    assert _stage_clocks(counters) == [] and 'intersection_thread_discarded_attempt_time_ns' in counters


def _domains():
    model, faces = joint()
    return {face_id: MaterialDomain.from_model(model, face_id) for face_id in faces}


def test_admission_of_domains_is_by_exact_type(monkeypatch):                               # FAILS-ON-OLD
    stages, domains = pt.ThreadStages(2), _domains()
    stages.admit_domains(domains, {})
    with pytest.raises(pt._NotAudited):
        stages.admit_domains(domains, {7: object()})                 # a face without a material domain
    first = next(iter(domains))
    for kind in {type(domain.support) for domain in domains.values()}:
        monkeypatch.setattr(pt, '_AUDITED_SUPPORTS', tuple(k for k in pt._AUDITED_SUPPORTS if k is not kind))
        with pytest.raises(pt._NotAudited):
            stages.admit_domains(domains, {})
        monkeypatch.undo()
    curves = {type(path.curve) for domain in domains.values() for loop in domain.boundaries for path in loop}
    assert curves <= set(pt._AUDITED_PATH_CURVES)
    for kind in curves:
        monkeypatch.setattr(pt, '_AUDITED_PATH_CURVES', tuple(k for k in pt._AUDITED_PATH_CURVES if k is not kind))
        with pytest.raises(pt._NotAudited):
            stages.admit_domains(domains, {})
        monkeypatch.undo()
    odd = replace(domains[first], boundaries=((object(),),))
    with pytest.raises(pt._NotAudited):
        stages.admit_domains({**domains, first: odd}, {})


def test_admission_of_face_inputs_covers_traces_member_inputs_and_points():                # FAILS-ON-OLD
    stages = pt.ThreadStages(2)
    line = LinePath((0., 0., 0.), (1., 0., 0.))
    good = ArrangementPath(line, owners=(1, 2))
    member = ArrangementPath(line, member_ids=(5,))
    point = ArrangementPoint((0., 0., 0.), member_ids=(5,))
    stages.admit_face_inputs({1: [good, member], 2: []}, {1: [point], 2: []})
    for traces, points in (({1: [ArrangementPath(OddLine(line.start, line.end), member_ids=(5,))]}, {1: []}),
                           ({1: [line]}, {1: []}),
                           ({1: [good]}, {1: [(0., 0., 0.)]})):
        with pytest.raises(pt._NotAudited):
            stages.admit_face_inputs(traces, points)


def test_not_audited_is_not_a_geometry_refusal():
    # It must never reach a caller as a typed geometry answer; plan_with_threads turns it into the serial route.
    assert issubclass(pt._NotAudited, Exception) and not issubclass(pt._NotAudited, GeometryError)
    assert pt.plan_with_threads(lambda: (_ for _ in ()).throw(pt._NotAudited('x')), lambda: 'serial') == 'serial'


# ---- 2. normalisation inside the face task ----

def test_cells_are_normalised_inside_the_face_task_directly_after_its_arrangement(monkeypatch):   # FAILS-ON-OLD
    events, lock = [], threading.Lock()
    arrange, normalise = bi.arrange_material, bi._normalise_cells

    def arranging(domain, *args, **kwargs):
        value = arrange(domain, *args, **kwargs)
        with lock:
            events.append((threading.get_ident(), threading.current_thread().name, 'A', id(value)))
        return value

    def normalising(value):
        with lock:
            events.append((threading.get_ident(), threading.current_thread().name, 'N', id(value)))
        return normalise(value)
    monkeypatch.setattr(bi, 'arrange_material', arranging)
    monkeypatch.setattr(bi, '_normalise_cells', normalising)
    serial, _ = planned(0, monkeypatch)
    count = len(events) // 2
    assert count and [kind for *_, kind, _ in events] == ['A', 'N'] * count      # the serial interleaving
    del events[:]
    threaded, counters = planned(4, monkeypatch)
    assert threaded == serial and 'intersection_thread_face_stage_time_ns' in counters
    assert len(events) == 2 * count
    last = {}
    for ident, name, kind, value in events:
        assert name.startswith(POOL)
        if kind == 'N':
            assert last.get(ident) == ('A', value)       # same thread, straight after this face's arrangement
        last[ident] = (kind, value)


def test_diagnostics_of_normalisation_are_published_in_the_serial_order(monkeypatch):      # FAILS-ON-OLD
    from anygeometry.runtime_diagnostics import _record
    arrange, normalise = bi.arrange_material, bi._normalise_cells

    def arranging(domain, *args, **kwargs):
        _record('test_face_order', ('arranged', domain.face_id))
        return arrange(domain, *args, **kwargs)

    def normalising(value):
        _record('test_face_order', ('normalised', len(value.paths)))
        return normalise(value)
    monkeypatch.setattr(bi, 'arrange_material', arranging)
    monkeypatch.setattr(bi, '_normalise_cells', normalising)
    serial, _ = planned(0, monkeypatch)
    rows = serial['records']['test_face_order']
    assert len(rows) >= 4 and [row[0] for row in rows[:2]] == ["'arranged'", "'normalised'"]
    for threads in (1, 4, 8):
        assert planned(threads, monkeypatch)[0] == serial


# ---- 3. plane_split._PLANS is not reachable from a stage task ----

_STORE_FUNCTIONS = frozenset(('plan_plane_split', 'apply_plane_split', '_plan_plane_split'))


def plan_store_uses(tree):
    """What in a module could reach ``plane_split._PLANS``: an attribute ``._PLANS`` on anything, the store
    or ``*`` imported from plane_split, the plane_split module object bound by an import or looked up by
    name, or one of the functions that use the store. A module's own global called ``_PLANS`` is not one."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == '_PLANS':
            found.add('attribute _PLANS')
        if isinstance(node, (ast.Name, ast.Attribute)) and getattr(node, 'id', getattr(node, 'attr', None)) in _STORE_FUNCTIONS:
            found.add(getattr(node, 'id', getattr(node, 'attr', None)))
        if isinstance(node, ast.ImportFrom):
            from_split = (node.module or '').split('.')[-1] == 'plane_split'
            for alias in node.names:
                if from_split and (alias.name in ('_PLANS', '*') or alias.name in _STORE_FUNCTIONS):
                    found.add(f'import {alias.name}')
                if not from_split and alias.name == 'plane_split':
                    found.add('module plane_split')
        if isinstance(node, ast.Import) and any(alias.name.split('.')[-1] == 'plane_split' for alias in node.names):
            found.add('module plane_split')
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and (
                node.value == '_PLANS' or node.value.split('.')[-1] == 'plane_split'):
            found.add(f'string {node.value}')
    return sorted(found)


def test_plane_split_plan_store_is_reached_only_through_the_public_split_entries():
    """Static proof: inside the package only plane_split itself can reach the store, and __init__ only
    re-exports the two public functions without calling them. A planner task has no path to the store."""
    root = Path(anygeometry.__file__).parent
    users = {}
    for path in sorted(root.rglob('*.py')):
        uses = plan_store_uses(ast.parse(path.read_text(encoding='utf-8')))
        if uses:
            users[path.relative_to(root).as_posix()] = uses
    assert sorted(users) == ['__init__.py', 'plane_split.py'], users
    assert users['__init__.py'] == ['import apply_plane_split', 'import plan_plane_split'], users
    init = ast.parse((root / '__init__.py').read_text(encoding='utf-8'))
    calls = {getattr(node.func, 'id', getattr(node.func, 'attr', None))
             for node in ast.walk(init) if isinstance(node, ast.Call)}
    assert not calls & _STORE_FUNCTIONS


def test_thread_stages_never_touch_the_plane_split_plan_store(monkeypatch):
    class Guard:
        def __getattr__(self, name):
            raise AssertionError(f'plane_split._PLANS.{name} used during planning')

        def __getitem__(self, key):
            raise AssertionError('plane_split._PLANS read during planning')

        def __setitem__(self, key, value):
            raise AssertionError('plane_split._PLANS written during planning')
    monkeypatch.setattr(plane_split, '_PLANS', Guard())
    serial, _ = planned(0, monkeypatch)
    threaded, counters = planned(4, monkeypatch)
    assert threaded == serial and isinstance(serial['value'], str)
    assert len(_stage_clocks(counters)) == 2


# ---- round-2 minors: whole cause chain of a refusal, and records that are really there ----

@pytest.mark.parametrize('threads', (1, 8))
def test_refusal_has_the_serial_cause_chain(threads, monkeypatch):
    original = bi._pair_paths

    def refusing(first, second, *args, **kwargs):
        if first.face_id != second.face_id and type(first.support) is type(second.support) is bi.Cylinder:
            try:
                raise ValueError(f'inner {first.face_id}/{second.face_id}')
            except ValueError as inner:
                raise GeometryError(f'refused {first.face_id}/{second.face_id}') from inner
        return original(first, second, *args, **kwargs)
    monkeypatch.setattr(bi, '_pair_paths', refusing)
    serial, _ = planned(0, monkeypatch)
    assert [kind for kind, _ in serial['value']] == ['GeometryError', 'GeometryError', 'ValueError']
    assert planned(threads, monkeypatch)[0] == serial


def test_records_of_both_stages_are_compared_by_content_and_order(monkeypatch):
    # The joint model emits no runtime records of its own (measured: the serial record set is empty), so
    # record parity is tested on records emitted from inside the pair and the face task functions.
    from anygeometry.runtime_diagnostics import _record
    pair, arrange = bi._pair_paths, bi.arrange_material

    def pairing(first, second, *args, **kwargs):
        _record('test_pair_order', dict(pair=(first.face_id, second.face_id)))
        return pair(first, second, *args, **kwargs)

    def arranging(domain, *args, **kwargs):
        _record('test_face_order', dict(face=domain.face_id))
        return arrange(domain, *args, **kwargs)
    monkeypatch.setattr(bi, '_pair_paths', pairing)
    monkeypatch.setattr(bi, 'arrange_material', arranging)
    serial, _ = planned(0, monkeypatch)
    assert len(serial['records']['test_pair_order']) >= 4 and len(serial['records']['test_face_order']) >= 4
    for threads in (1, 4, 8):
        threaded, counters = planned(threads, monkeypatch)
        assert threaded == serial and len(_stage_clocks(counters)) == 2
