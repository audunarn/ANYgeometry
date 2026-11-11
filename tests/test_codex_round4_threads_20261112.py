"""Codex review round 4 of the thread route (2026-11-11).

Blocking finding: a worker's KeyboardInterrupt / SystemExit could still be lost when ANOTHER worker failed with an
ordinary exception outside the task function (setup, bookkeeping, outcome storage). Contract now: the dispatcher
drains all workers, then raises the lowest-index interrupt seen anywhere in a worker, only then an ordinary
machinery failure, only then the lowest-index task failure. Every interrupt test here makes the two events
verifiably coexist (barrier / events / a fixed one-worker sequence), not by scheduling luck.

Tests marked FAILS-ON-OLD fail on 9326eba. Also closed here: the test weaknesses Codex named (dict key types in
the record comparison, non-literal record keys, the plan-store scan and the re-exported split functions, and
"default off does not import the thread module" shown in a fresh interpreter).
"""
import ast
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

import anygeometry
from anygeometry import GeometryError
from anygeometry import _planning_threads as pt
from test_codex_boundaries_threads_20261112 import plan_store_uses
from test_planning_threads import ENV, PROCESS, _clear, comparable_records

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


def _both_ways(make, expected):
    """The dispatcher raises ``expected``; so does plan_with_threads (no serial repeat for an interrupt)."""
    task, seen = make()
    with pytest.raises(expected):
        pt.ThreadStages(2).run('face', _two(), task)
    assert seen == dict(interrupt=True, machinery=True), seen
    task, seen = make()
    with pytest.raises(expected):
        pt.plan_with_threads(lambda: pt.ThreadStages(2).run('face', _two(), task), lambda: 'serial')
    assert seen == dict(interrupt=True, machinery=True), seen


# ---- interrupt of one worker, ordinary machinery failure of another ----

@pytest.mark.parametrize('fatal', FATAL)
@pytest.mark.parametrize('interrupt_face', (0, 1))
def test_interrupt_wins_over_a_bookkeeping_failure_of_another_worker(fatal, interrupt_face, monkeypatch):   # FAILS-ON-OLD
    real, doomed = pt.perf_counter_ns, set()

    def make():
        barrier, interrupted, seen = threading.Barrier(2, timeout=WAIT), threading.Event(), {}

        def clock():                                  # the first clock read after the task function returned
            if threading.get_ident() in doomed:
                doomed.discard(threading.get_ident())
                seen['machinery'] = True
                raise MemoryError('bookkeeping')
            return real()
        monkeypatch.setattr(pt, 'perf_counter_ns', clock)

        def task(item, check=None):
            barrier.wait()                            # both tasks are inside their task function
            if item['face'] == interrupt_face:
                seen['interrupt'] = True
                interrupted.set()
                raise fatal('stop')
            assert interrupted.wait(WAIT)
            doomed.add(threading.get_ident())
            return item['face']
        return task, seen
    _both_ways(make, fatal)


@pytest.mark.parametrize('fatal', FATAL)
def test_interrupt_wins_over_a_setup_failure_of_another_worker(fatal, monkeypatch):       # FAILS-ON-OLD
    real = pt._ROOT_MEMO

    def make():
        owner, lock, in_body, failed, seen = [], threading.Lock(), threading.Event(), threading.Event(), {}

        class Memo:                                   # worker setup: the first worker passes, the second fails
            def set(self, value):
                with lock:
                    first = not owner
                    if first:
                        owner.append(threading.get_ident())
                if first:
                    return real.set(value)
                assert in_body.wait(WAIT)             # the other worker is inside its task function
                seen['machinery'] = True
                failed.set()
                raise RuntimeError('setup')
        monkeypatch.setattr(pt, '_ROOT_MEMO', Memo())

        def task(item, check=None):
            seen['interrupt'] = True
            in_body.set()
            assert failed.wait(WAIT)                  # the other worker's setup has failed
            raise fatal('stop')
        return task, seen
    _both_ways(make, fatal)


@pytest.mark.parametrize('fatal', FATAL)
def test_interrupt_inside_bookkeeping_wins_over_an_ordinary_task_failure(fatal, monkeypatch):
    real, doomed = pt.perf_counter_ns, set()

    def make():
        barrier, refused, seen = threading.Barrier(2, timeout=WAIT), threading.Event(), {}

        def clock():
            if threading.get_ident() in doomed:
                doomed.discard(threading.get_ident())
                seen['interrupt'] = True
                raise fatal('during bookkeeping')
            return real()
        monkeypatch.setattr(pt, 'perf_counter_ns', clock)

        def task(item, check=None):
            barrier.wait()
            if item['face'] == 0:
                seen['machinery'] = True              # here: the ordinary failure of the lower task
                refused.set()
                raise GeometryError('ordinary refusal')
            assert refused.wait(WAIT)
            doomed.add(threading.get_ident())
            return item['face']
        return task, seen
    _both_ways(make, fatal)


@pytest.mark.parametrize('fatal', FATAL)
def test_one_worker_ordinary_failure_first_then_interrupt_of_a_lower_task(fatal):
    # One worker, cost order: task 1 fails ordinarily first, then task 0 (lower index, still due) is interrupted.
    tasks = [dict(kind='face', face=0, cost=1.), dict(kind='face', face=1, cost=9.)]

    def make():
        order = []

        def task(item, check=None):
            order.append(item['face'])
            if item['face'] == 1:
                raise GeometryError('ordinary refusal')
            raise fatal('stop')
        return task, order
    task, order = make()
    with pytest.raises(fatal):
        pt.ThreadStages(1).run('face', tasks, task)
    assert order == [1, 0]
    task, order = make()
    with pytest.raises(fatal):
        pt.plan_with_threads(lambda: pt.ThreadStages(1).run('face', tasks, task), lambda: 'serial')
    assert order == [1, 0]


def test_the_lowest_index_interrupt_is_the_one_raised():
    barrier = threading.Barrier(2, timeout=WAIT)

    def task(item, check=None):
        barrier.wait()
        raise (KeyboardInterrupt if item['face'] == 0 else SystemExit)('stop')
    with pytest.raises(KeyboardInterrupt):
        pt.ThreadStages(2).run('face', _two(), task)


def test_machinery_failure_alone_is_a_typed_refusal(monkeypatch):
    # (Owner decision after round 12: only a plain refusal is repeated serially; this is a typed refusal now.)
    real, doomed = pt.perf_counter_ns, set()

    def clock():
        if threading.get_ident() in doomed:
            doomed.discard(threading.get_ident())
            raise MemoryError('bookkeeping')
        return real()
    monkeypatch.setattr(pt, 'perf_counter_ns', clock)

    def task(item, check=None):
        if item['face'] == 1:
            doomed.add(threading.get_ident())
        return item['face']
    with pytest.raises(GeometryError, match='thread planning failed') as error:
        pt.ThreadStages(2).run('face', _two(), task)
    assert type(error.value.__cause__) is MemoryError
    with pytest.raises(GeometryError, match='thread planning failed') as error:
        pt.plan_with_threads(lambda: pt.ThreadStages(2).run('face', _two(), task), lambda: pytest.fail('serial repeat'))
    assert type(error.value.__cause__) is MemoryError


def test_task_failures_and_results_are_unchanged():
    tasks = [dict(kind='face', face=index, cost=float(index)) for index in range(6)]
    assert pt.ThreadStages(3).run('face', tasks, lambda item, check=None: item['face'] * 2) == [0, 2, 4, 6, 8, 10]

    def task(item, check=None):
        if item['face'] in (2, 4):
            raise GeometryError(f"refused {item['face']}")
        return item['face']
    with pytest.raises(GeometryError, match='face:2: refused 2') as error:
        pt.ThreadStages(3).run('face', tasks, task)
    assert str(error.value.__cause__) == 'refused 2'


# ---- test weaknesses named in round 4 ----

def test_record_comparison_encodes_dictionary_key_types_and_dictionary_kinds():         # FAILS-ON-OLD
    class Odd(dict):
        pass
    assert comparable_records({'x': [{1: 'a'}]}) != comparable_records({'x': [{1.0: 'a'}]})
    assert comparable_records({'x': [{1: 'a'}]}) != comparable_records({'x': [{True: 'a'}]})
    assert comparable_records({'x': [dict(a=1)]}) != comparable_records({'x': [Odd(a=1)]})
    assert comparable_records({'x': [dict(a=dict(b=1))]}) != comparable_records({'x': [dict(a=Odd(b=1))]})
    assert comparable_records({'x': [dict(a=1, b=2)]}) == comparable_records({'x': [dict(a=1, b=2)]})


def record_calls(tree):
    """(literal record keys, what the scan cannot judge) of one module: a ``_record`` call whose key is not a
    string literal, ``_record`` imported under another name, or ``_record`` used as a value (aliased, passed on).
    ``self._record`` is an object's own attribute or method (two unrelated classes have one), not the diagnostics
    function, and is left out. A module that defines a function ``_record`` of its own and does not import the
    diagnostics one (exact_predicates: its predicate trace) uses the bare name for that function, which is left
    out as well; a module that does both is not judged."""
    keys, unjudged, called = set(), [], set()
    nodes = list(ast.walk(tree))
    # Only a MODULE-LEVEL function owns the bare name; a nested function or a method of that name does not.
    defines = any(isinstance(node, ast.FunctionDef) and node.name == '_record' for node in tree.body)
    imports = any(isinstance(node, ast.ImportFrom) and any(alias.name == '_record' for alias in node.names)
                  for node in nodes)
    if defines and imports:
        unjudged.append('own _record and imported _record in one module')
    if any(isinstance(node, ast.ImportFrom) and (node.module or '').split('.')[-1] == 'runtime_diagnostics'
           and any(alias.name == '*' for alias in node.names) for node in nodes):
        unjudged.append('wildcard import of the diagnostics module')

    def own(node):
        if isinstance(node, ast.Name):
            return defines and not imports
        return isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'self'
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, 'id', getattr(node.func, 'attr', None)) == '_record'
                and not own(node.func)):
            called.add(id(node.func))
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                keys.add(node.args[0].value)
            else:
                unjudged.append('non-literal key')
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(alias.name == '_record' and alias.asname for alias in node.names):
            unjudged.append('imported under another name')
        if (isinstance(node, (ast.Name, ast.Attribute)) and getattr(node, 'id', getattr(node, 'attr', None)) == '_record'
                and id(node) not in called and not own(node)):
            unjudged.append('used as a value')
        if isinstance(node, ast.Constant) and node.value == '_record':
            unjudged.append('looked up by name')
    return keys, unjudged


def clocked_record_fields(tree, key):
    """The sorted field names of every ``_record(key, {...})`` call (bare or qualified, as ``record_calls`` counts
    them); ``None`` for a call whose row is not a dict literal with literal field names."""
    found = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, 'id', getattr(node.func, 'attr', None)) == '_record'
                and not (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                         and node.func.value.id == 'self')
                and node.args and getattr(node.args[0], 'value', None) == key):
            row = node.args[1] if len(node.args) > 1 else None
            literal = isinstance(row, ast.Dict) and all(
                isinstance(name, ast.Constant) and isinstance(name.value, str) for name in row.keys)
            found.append(sorted(name.value for name in row.keys) if literal else None)
    return found


def test_record_scan_declines_what_it_cannot_judge():
    for source in ('_record(key, row)\n', "_record('a' + suffix, row)\n",
                   'from .runtime_diagnostics import _record as emit\nemit("x", 1)\n',
                   'emit = _record\n', 'emit = diagnostics._record\n', "getattr(diagnostics, '_record')('x', 1)\n",
                   'for key in keys:\n    _record(key, 1)\n'):
        assert record_calls(ast.parse(source))[1], source
    assert record_calls(ast.parse("from .runtime_diagnostics import _record\n_record('x', 1)\nd._record('y', 2)\n")) == (
        {'x', 'y'}, [])
    assert record_calls(ast.parse('self._record = snapshot()\nself._record(first)\n')) == (set(), [])
    assert record_calls(ast.parse('def _record(a, b):\n    pass\n_record(x, y)\n')) == (set(), [])
    assert record_calls(ast.parse('from .runtime_diagnostics import _record\ndef _record(a, b):\n    pass\n'))[1]


def test_every_record_key_and_every_field_of_the_clocked_record_is_judged():
    root = Path(anygeometry.__file__).parent
    keys, snap_fields = set(), []
    for path in sorted(root.rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        found, unjudged = record_calls(tree)
        assert not unjudged, (path.name, unjudged)
        keys |= found
        snap_fields += clocked_record_fields(tree, 'arrangement_snap_fallback')
    assert keys == {'arrangement_snap_fallback', 'ring_split', 'parallel_planning'}
    assert snap_fields == [sorted(('face_id', 'moved_edges', 'snapped_vertices', 'open_curved_vertices', 'max_move',
                                   'max_curved_disagreement', 'max_cycle_gap', 'refused_ratio', 'accepted_ratio',
                                   'seconds'))]
    assert pt.RECORD_CLOCK_FIELDS == {'arrangement_snap_fallback': ('seconds',)}


def test_plan_store_scan_sees_the_re_exported_split_functions():                         # FAILS-ON-OLD
    for source in ('from . import plan_plane_split as go\ngo()\n',
                   'from anygeometry import apply_plane_split\n',
                   'from . import plan_plane_split\n',
                   'import anygeometry\nanygeometry.plan_plane_split()\n',
                   "getattr(anygeometry, 'apply_plane_split')()\n",
                   'from . import *\n'):
        assert plan_store_uses(ast.parse(source)), source
    assert not plan_store_uses(ast.parse('from . import surfaces\nfrom .plane_split import _stage_plane_split\n'))


SCRIPT = '''
import sys
from uuid import UUID
from anygeometry import GeometryModel, plan_intersections
model = GeometryModel(model_id=UUID(int=7))
for points in (((-1., -1., 0.), (1., -1., 0.), (1., 1., 0.), (-1., 1., 0.)),
               ((0., -1., -1.), (0., 1., -1.), (0., 1., 1.), (0., -1., 1.))):
    model.add_plate(model.add_points(points))
plan = plan_intersections(model, sorted(model.faces), policy='connect')
assert plan.content_checksum
print('thread module loaded:', 'anygeometry._planning_threads' in sys.modules)
'''


@pytest.mark.parametrize('value', (None, '', '0', '2'))
def test_default_off_does_not_import_the_thread_module(value):
    environment = {key: item for key, item in os.environ.items() if key not in (ENV, PROCESS)}
    if value is not None:
        environment[ENV] = value
    done = subprocess.run([sys.executable, '-c', SCRIPT], env=environment, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip() == f"thread module loaded: {value == '2'}"
