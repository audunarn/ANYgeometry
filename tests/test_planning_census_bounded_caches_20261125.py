"""Bounded stores in the planner's module-state census (0.4.7 gate class H, 2026-11-25).

``exact_curves._BRANCH_CONSTANTS`` and ``exact_curves._DOUBLE_NODES`` are reset roots bounded at
``_BRANCH_CONSTANTS_LIMIT`` (4096) entries each. Warm, they hold about 50 walk nodes per entry, so a process
that had evaluated many cylinder pairs exceeded the census bound (160,000) and the interpreter ledger declined
('ledger creation live coverage mismatch'); the package worker ledger's alias walk (80,000 identities)
declined the same way ('worker cache alias walk bound exceeded').

Owner decision 2026-11-25 (bound kept at 160,000). Codex round 1 showed that skipping the payload loses the
parent's alias/identity protection, so the stores are walked exactly as before, with their entries counted
against a reserved budget derived from their own bound (limit entries x 64 nodes per entry). The walk is the
check: every state the parent refused below the bound is still refused, and a removed, changed or exceeded
bound declines.
"""
import json
import math
import os
import pathlib
import subprocess
import sys
import types
from collections import OrderedDict

import pytest

from _planning_numpy_profile_contract import numpy_profile_test

from anygeometry import _planning_ledger as ledger
from anygeometry import _planning_purity
from anygeometry import exact_curves as ec
from anygeometry._planning_qualification import QualificationDecline

BOUNDED = ('_BRANCH_CONSTANTS', '_DOUBLE_NODES')


def _pair(i):
    t = .3 + i*1e-4
    first = ec._CylinderSupport((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), 1., 6., 0., 2*math.pi)
    second = ec._CylinderSupport((0., 0., -1.), (0., math.sin(t), math.cos(t)), (1., 0., 0.), .5, 4., 0.,
                                 2*math.pi)
    return types.SimpleNamespace(first=first, second=second)


@pytest.fixture
def caches():
    """Both caches cleared for the test and restored afterwards (no other module state is touched)."""
    saved = {name: dict(getattr(ec, name)) for name in BOUNDED}
    for name in saved:
        getattr(ec, name).clear()
    yield
    for name, content in saved.items():
        getattr(ec, name).clear()
        getattr(ec, name).update(content)


def _fill(count=4096):
    for i in range(count):
        pair = _pair(i)
        ec._double_nodes(pair)
        ec._branch_constants(pair)
    assert len(ec._BRANCH_CONSTANTS) == len(ec._DOUBLE_NODES) == count


def _rows(result):
    return sorted((path, row['action']) for path, row in result.rows.items())


def _one_more(name='_BRANCH_CONSTANTS'):
    """A 4097th entry of the writer's own shape."""
    key, (first, second, payload) = next(iter(getattr(ec, name).items()))
    extra = _pair(10**6)
    getattr(ec, name)[(id(extra.first), id(extra.second))] = (extra.first, extra.second, payload)
    return extra


# The class H regression: full caches truncated both planning censuses.

def _fresh_process_check(call):
    """Run ``call`` (a statement on this module, imported as ``t``) in a fresh interpreter: the census asserts
    below assume a clean process, and a long test process leaves ordinary state of its own (past the shared
    census cap, or objects aliased into a cache entry) that the census correctly declines. Reports JSON."""
    src = pathlib.Path(ledger.__file__).resolve().parents[1]
    # No inherited PYTHONOPTIMIZE: the child's assertions are the checks (Codex review, tests round).
    env = {key: value for key, value in os.environ.items() if key != 'PYTHONOPTIMIZE'}
    env['PYTHONPATH'] = str(src)
    probe = _FRESH_PROBE.replace('@@MODULE@@', pathlib.Path(__file__).stem).replace('@@CALL@@', call)
    try:
        result = subprocess.run([sys.executable, '-c', probe, str(pathlib.Path(__file__).parent)],
                                capture_output=True, text=True, env=env, timeout=900, cwd=str(src.parent))
    except subprocess.TimeoutExpired as exc:
        pytest.fail('fresh-process check timed out: ' + call + ' ' + str(exc.stdout)[-500:])
    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report['ok'], report['error']


_FRESH_PROBE = r'''
import json, sys, traceback
if sys.flags.optimize:
    raise SystemExit('optimized interpreter: the checks are assertions')
sys.path.insert(0, sys.argv[1])
import @@MODULE@@ as t
try:
    @@CALL@@
    report = {'ok': True, 'error': ''}
except BaseException:
    report = {'ok': False, 'error': traceback.format_exc()[-3000:]}
print(json.dumps(report))
'''


@numpy_profile_test
def test_full_bounded_caches_do_not_truncate_the_planning_census():
    _fresh_process_check('t._check_full_bounded_caches()')


def _check_full_bounded_caches():
    _fill()
    modules = tuple(m for n, m in sorted(sys.modules.items()) if n == 'anygeometry' or n.startswith('anygeometry.'))
    assert _planning_purity.inventory(modules).truncated           # the shared budget alone would be exceeded
    result = ledger.loaded_geometry_ledger(trusted_image=True)     # the interpreter ledger's own call
    assert not result.report.truncated
    result.validate()
    for name in BOUNDED:
        assert result.rows['anygeometry.exact_curves.' + name]['action'] == 'reset'
    assert len(ec._BRANCH_CONSTANTS) == len(ec._DOUBLE_NODES) == 4096       # admitted without clearing


_PACKAGE_PROBE = r'''
import math, types
from anygeometry import exact_curves as ec, _planning_ledger as ledger
from anygeometry._planning_canaries import _fixture
from anygeometry._planning_executor import freeze_model, thaw_model
from anygeometry._planning_qualification import QualificationDecline
owned = ledger.package_worker_ledger(thaw_model(freeze_model(_fixture()[0])))
assert {'anygeometry.exact_curves._BRANCH_CONSTANTS', 'anygeometry.exact_curves._DOUBLE_NODES'} <= set(owned.resets)
for i in range(4096):
    t = .3 + i*1e-4
    a = ec._CylinderSupport((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), 1., 6., 0., 2*math.pi)
    b = ec._CylinderSupport((0., 0., -1.), (0., math.sin(t), math.cos(t)), (1., 0., 0.), .5, 4., 0., 2*math.pi)
    pair = types.SimpleNamespace(first=a, second=b); ec._double_nodes(pair); ec._branch_constants(pair)
owned.validate()                     # the manifest route's alias walk with both caches full
key, (first, second, payload) = next(iter(ec._BRANCH_CONSTANTS.items()))
ec._BRANCH_CONSTANTS[(id(b), id(a))] = (b, a, payload)
try:
    owned.validate()
except QualificationDecline as exc:
    print('DECLINED', exc)
'''


@numpy_profile_test
def test_full_bounded_caches_pass_the_package_worker_ledger():
    # A fresh process: the package ledger refuses unrelated state a long test process accumulates.
    import os
    import pathlib
    import subprocess
    src = pathlib.Path(ledger.__file__).resolve().parents[1]
    # No inherited PYTHONOPTIMIZE: the child's assertions are the checks (Codex review, tests round).
    env = {key: value for key, value in os.environ.items() if key != 'PYTHONOPTIMIZE'}
    env['PYTHONPATH'] = str(src)
    result = subprocess.run([sys.executable, '-c', _PACKAGE_PROBE], capture_output=True, text=True, env=env,
                            timeout=600)
    assert result.returncode == 0, result.stderr[-2000:]
    assert 'DECLINED bounded store exceeds its limit' in result.stdout, result.stdout


def test_exact_curves_ledger_rows_are_independent_of_cache_fill(caches):
    empty = ledger.loaded_geometry_ledger(trusted_image=True, modules=(ec,))
    _fill()
    full = ledger.loaded_geometry_ledger(trusted_image=True, modules=(ec,))
    full.validate()
    assert _rows(full) == _rows(empty)
    rebuilt = ledger.build_ledger(full.report, dict(full.objects), modules=(ec,), trusted_image=True)
    assert _rows(rebuilt) == _rows(full)


def test_owned_worker_reset_still_clears_the_bounded_caches(caches):
    _fill(8)
    result = ledger.loaded_geometry_ledger(trusted_image=True, modules=(ec,))
    result.validate()
    # prepare() also refuses unrelated declined rows a long test process may have created (e.g. copyreg's
    # __slotnames__ entries), so the registered reset is exercised directly.
    from anygeometry._planning_qualification import reset_declined
    paths = ['anygeometry.exact_curves.' + name for name in BOUNDED]
    assert all(path in result.resets for path in paths)
    reset_declined({path: result.resets[path] for path in paths}, owned_worker=True)
    assert not ec._BRANCH_CONSTANTS and not ec._DOUBLE_NODES


# Fail closed: the bound itself.

def test_cache_past_its_limit_declines(caches):
    _fill()
    _one_more()
    with pytest.raises(QualificationDecline, match='exceeds its limit'):
        ledger.loaded_geometry_ledger(trusted_image=True, modules=(ec,))


@pytest.mark.parametrize('limit', [8192, 4096.0, True, None])
def test_changed_limit_declines(caches, monkeypatch, limit):
    monkeypatch.setattr(ec, '_BRANCH_CONSTANTS_LIMIT', limit)
    with pytest.raises(QualificationDecline, match='limit changed or removed'):
        ledger.loaded_geometry_ledger(trusted_image=True, modules=(ec,))


def test_removed_limit_declines(caches, monkeypatch):
    monkeypatch.delattr(ec, '_BRANCH_CONSTANTS_LIMIT')
    with pytest.raises(QualificationDecline, match='limit changed or removed'):
        ledger.loaded_geometry_ledger(trusted_image=True, modules=(ec,))


class _Unbounded(dict):
    pass


@pytest.mark.parametrize('name', BOUNDED)
@pytest.mark.parametrize('replacement', [OrderedDict, _Unbounded, list])
def test_replaced_store_type_declines(caches, monkeypatch, name, replacement):
    monkeypatch.setattr(ec, name, replacement())
    with pytest.raises(QualificationDecline, match='exact dict'):
        ledger.loaded_geometry_ledger(trusted_image=True, modules=(ec,))


def test_store_reached_first_through_another_binding_declines(caches, monkeypatch):
    other = types.ModuleType('anygeometry._bounded_alias_fixture')
    other.held = ec._DOUBLE_NODES
    monkeypatch.setitem(sys.modules, other.__name__, other)
    with pytest.raises(QualificationDecline, match='registered binding'):
        ledger.loaded_geometry_ledger(trusted_image=True, modules=(other, ec))


def test_foreign_module_of_the_same_name_declines():
    impostor = types.ModuleType('anygeometry.exact_curves')
    impostor._BRANCH_CONSTANTS, impostor._DOUBLE_NODES, impostor._BRANCH_CONSTANTS_LIMIT = {}, {}, 4096
    with pytest.raises(QualificationDecline, match='unregistered bounded store'):
        ledger.bounded_store_check(impostor, '_DOUBLE_NODES')
    with pytest.raises(QualificationDecline, match='unregistered bounded store'):
        ledger.bounded_store_check(ec, '_PI_CACHE')


# Codex round 1 (23ad978): states the parent 1ac4352 refused must still be refused.

def test_h1_cached_support_aliased_elsewhere_declines(caches, monkeypatch):
    pair = _pair(0)
    ec._branch_constants(pair)
    other = types.ModuleType('anygeometry.zz_bounded_alias_fixture')
    other.held = [pair.first]
    monkeypatch.setitem(sys.modules, other.__name__, other)
    result = ledger.loaded_geometry_ledger(trusted_image=True, modules=(ec, other))
    with pytest.raises(QualificationDecline, match='external alias'):
        result.validate()


def _lookalike(extra_slot):
    slots = ('origin', 'axis', 'radial_direction', 'radius', 'height', 'start_angle', 'sweep_angle')
    return type('_CylinderSupport', (), {'__slots__': slots + (('extra',) if extra_slot else ()),
                                         '__module__': ec.__name__})


def _lookalike_entry(cls, **extra):
    supports = []
    for _ in range(2):
        value = object.__new__(cls)
        for name, item in zip(('origin', 'axis', 'radial_direction', 'radius', 'height', 'start_angle',
                               'sweep_angle'), ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), 1., 6., 0., 1.)):
            if name not in extra.get('skip', ()):
                object.__setattr__(value, name, item)
        for name, item in extra.get('set', {}).items():
            object.__setattr__(value, name, item)
        supports.append(value)
    ec._DOUBLE_NODES[(id(supports[0]), id(supports[1]))] = (supports[0], supports[1], ())
    return supports


def test_h2_rebound_support_class_hiding_nested_mutable_state_declines(caches, monkeypatch):
    nested = []
    for _ in range(40):
        nested = [nested]
    cls = _lookalike(True)
    monkeypatch.setattr(ec, '_CylinderSupport', cls)
    _lookalike_entry(cls, set={'extra': nested})
    with pytest.raises(QualificationDecline):
        ledger.loaded_geometry_ledger(trusted_image=True, modules=(ec,))


def test_h2_census_never_executes_a_field_descriptor(caches, monkeypatch):
    calls = []
    slots = ('origin', 'axis', 'radial_direction', 'height', 'start_angle', 'sweep_angle')
    cls = type('_CylinderSupport', (), {'__slots__': slots, '__module__': ec.__name__,
                                        'radius': property(lambda self: calls.append(1) or 1.)})
    monkeypatch.setattr(ec, '_CylinderSupport', cls)
    _lookalike_entry(cls, skip=('radius',))
    try:
        ledger.loaded_geometry_ledger(trusted_image=True, modules=(ec,)).validate()
    except QualificationDecline:
        pass
    assert calls == []


def test_l5_missing_support_slot_is_never_an_attribute_error(caches):
    # The parent walk skips an unset slot (native member descriptor); never an uncaught AttributeError.
    _lookalike_entry(ec._CylinderSupport, skip=('radius',))
    result = ledger.loaded_geometry_ledger(trusted_image=True, modules=(ec,))
    result.validate()


def test_m3_one_oversized_entry_declines(caches):
    pair = _pair(0)
    nodes = tuple(ec._DoubleNode(1, 1, float(i)) for i in range(160001))
    ec._DOUBLE_NODES[(id(pair.first), id(pair.second))] = (pair.first, pair.second, nodes)
    with pytest.raises(QualificationDecline):
        ledger.loaded_geometry_ledger(trusted_image=True, modules=(ec,))


def test_m3_entry_budget_covers_what_the_writer_produces(caches):
    _fill(64)
    # The writer's largest entry (two nodes) fits the per-entry budget with room to spare.
    first, second, _ = next(iter(ec._DOUBLE_NODES.values()))
    ec._DOUBLE_NODES[(id(first), id(second))] = (first, second, (ec._DoubleNode(1, 1, 0.), ec._DoubleNode(-1, 1, 3.)))
    ledger.loaded_geometry_ledger(trusted_image=True, modules=(ec,)).validate()


def test_m4_growth_between_bound_check_and_walk_declines(caches, monkeypatch):
    _fill()
    result = ledger.loaded_geometry_ledger(trusted_image=True, modules=(ec,))
    real, raced = ledger.inventory, []

    def racing(*args, **kwargs):
        if not raced:
            raced.append(_one_more())   # inside one census: after the bound check, before the walk
        return real(*args, **kwargs)

    monkeypatch.setattr(ledger, 'inventory', racing)
    with pytest.raises(QualificationDecline):
        result.validate()
    assert raced


def test_m4_growth_after_capture_declines_at_validation(caches):
    _fill()
    result = ledger.loaded_geometry_ledger(trusted_image=True, modules=(ec,))
    _one_more()
    with pytest.raises(QualificationDecline, match='exceeds its limit'):
        result.validate()
