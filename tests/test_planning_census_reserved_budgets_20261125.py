"""Reserved census budgets for four limit-bounded caches (0.4.7 gate class H, owner decision 2026-11-25).

Four module caches bound their own size by a limit constant and, warm, hold more walk nodes than the shared
census budget (160,000) leaves free, so the census truncated and the process route declined ('ledger creation
live coverage mismatch'; on the package worker route 'worker cache alias walk bound exceeded'). Each cache is
now walked exactly as before, but its entries count against a reserved budget of its own (limit entries x a
measured per-entry node and identity budget); a removed or changed limit, a store past its limit and an entry
past its budget still decline. A fifth such cache, _TRIG_CACHE, is deliberately not reserved (owner decision):
it stays a declined store. The per-cache fill tests below use only the caches' own writers and limit
constants, so they run unchanged on the parent and fail there (census truncated). The Round 2 tests at the
end reproduce the Codex round-1 probes; each is refused on the parent as well.
"""
import importlib
import json
import math
import os
import pathlib
import subprocess
import sys
import types
from collections import OrderedDict
from fractions import Fraction

import numpy as np
import pytest

from anygeometry import _planning_ledger as ledger
from anygeometry import _planning_purity
from anygeometry._planning_qualification import QualificationDecline

# path -> (module, store, limit constant): the caches' own names, nothing taken from the ledger under test.
CACHES = {
    'anygeometry.exact_curves._BRANCH_CONSTANTS': ('anygeometry.exact_curves', '_BRANCH_CONSTANTS',
                                                   '_BRANCH_CONSTANTS_LIMIT'),
    'anygeometry.exact_curves._DOUBLE_NODES': ('anygeometry.exact_curves', '_DOUBLE_NODES', '_BRANCH_CONSTANTS_LIMIT'),
    'anygeometry.quadric_curve_events._BRANCH_POLYNOMIALS': ('anygeometry.quadric_curve_events', '_BRANCH_POLYNOMIALS',
                                                             '_BRANCH_POLYNOMIALS_LIMIT'),
    'anygeometry.quadric_events._MEMO': ('anygeometry.quadric_events', '_MEMO', '_MEMO_SIZE'),
}
PATHS = tuple(CACHES)
RESET_PATHS = PATHS
# Owner decision 2026-11-25: _TRIG_CACHE is not in the reserved-budget table; it stays a declined store.
TRIG_PATH = 'anygeometry.separating_plane_certificate._TRIG_CACHE'


def _module(path):
    return importlib.import_module(CACHES[path][0])


def _store(path):
    return getattr(_module(path), CACHES[path][1])


def _limit(path):
    return getattr(_module(path), CACHES[path][2])


@pytest.fixture
def caches():
    """All four caches cleared for the test and restored afterwards (no other module state is touched)."""
    saved = {path: dict(_store(path)) for path in PATHS}
    for path in PATHS:
        _store(path).clear()
    yield
    for path, content in saved.items():
        _store(path).clear()
        _store(path).update(content)


# The caches' own writers.

def _cylinder_pair(i):
    from anygeometry import exact_curves as ec
    t = .3 + i*1e-4
    first = ec._CylinderSupport((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), 1., 6., 0., 2*math.pi)
    second = ec._CylinderSupport((0., 0., -1.), (0., math.sin(t), math.cos(t)), (1., 0., 0.), .5, 4., 0.,
                                 2*math.pi)
    return types.SimpleNamespace(first=first, second=second)


def _branch_curve():
    from anygeometry.analytic_supports import cylinder_cylinder_support
    from anygeometry.exact_curves import CylinderIntersectionCurve
    from anygeometry.surfaces import Cylinder
    leg = Cylinder((0., 0., -3.), (0., 0., 1.), (1., 0., 0.), 2., 6.)
    unit = np.asarray((.3, 1., .2))/np.linalg.norm((.3, 1., .2))
    brace = Cylinder(-3*unit, unit, (1., 0., 0.), .5, 6.)
    return next(c for c in cylinder_cylinder_support(leg, brace).curves if isinstance(c, CylinderIntersectionCurve))


def _fill_branch_constants(count, start=0):
    from anygeometry import exact_curves as ec
    for i in range(start, start + count):
        ec._branch_constants(_cylinder_pair(i))


def _fill_double_nodes(count, start=0):
    from anygeometry import exact_curves as ec
    for i in range(start, start + count):
        ec._double_nodes(_cylinder_pair(i))


def _fill_branch_polynomials(count, start=0):
    from anygeometry import quadric_curve_events as qce
    from anygeometry.cylinder_curve_events import _multiply
    base = _branch_curve()
    for i in range(start, start + count):
        curve = base.transformed(((1., 0., 0., i*1e-3), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.)))
        _, numerator, _, _, _, products, _ = qce._branch_polynomials(curve)
        # A general quadric forms all nine numerator products (quadric_roots' own expression): the largest entry.
        for a in range(3):
            for b in range(3):
                products[(a, b)] = _multiply(numerator[a], numerator[b])


def _fill_memo(count, start=0):
    from anygeometry import quadric_events as qe
    from anygeometry.arrangement_geometry import LinePath
    from anygeometry.surfaces import Cylinder
    quadric = Cylinder((0., 0., -3.), (0., 0., 1.), (1., 0., 0.), 1., 6.)
    for i in range(start, start + count):
        z = i*1e-3
        qe.curve_quadric_roots(LinePath((-2., .1, z), (2., .3, z + .5)), quadric)


def _fill_trig(count, start=0):
    from anygeometry import separating_plane_certificate as spc
    for i in range(start, start + count):
        x = Fraction(i*7919 + 1, 1 << 20)
        hit = spc._sincos_fixed(x, 64)
        # SPCertifier.sincos's own insertion (no _Policy/_Budget object needed for the value itself).
        if len(spc._TRIG_CACHE) >= spc._TRIG_CACHE_LIMIT:
            spc._TRIG_CACHE.clear()
        spc._TRIG_CACHE[x] = hit


FILL = dict(zip(PATHS, (_fill_branch_constants, _fill_double_nodes, _fill_branch_polynomials, _fill_memo)))


# Fill order for several full caches: the branch-polynomial and memo writers also reach _branch_constants.
FILL_ORDER = (PATHS[2], PATHS[3], PATHS[1], PATHS[0])


def _fill(path):
    FILL[path](_limit(path))
    # A writer clears its store at the limit, so entries another writer put there first leave it short.
    missing = _limit(path) - len(_store(path))
    if missing:
        FILL[path](missing, start=_limit(path))
    assert len(_store(path)) == _limit(path)


def _all_modules():
    for path in PATHS:
        _module(path)
    return tuple(m for n, m in sorted(sys.modules.items()) if n == 'anygeometry' or n.startswith('anygeometry.'))


def _actions(result):
    return {path: result.rows[path]['action'] for path in PATHS if path in result.rows}


# Criterion 1: each cache full to its limit truncated the census on the parent; it now passes.

def _fresh_process_check(call):
    """Run ``call`` (a statement on this module, imported as ``t``) in a fresh interpreter: the census asserts
    below assume a clean process, and a long test process leaves ordinary state of its own (past the shared
    census cap, or objects aliased into a cache entry) that the census correctly declines. Reports JSON."""
    src = pathlib.Path(ledger.__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(src))
    probe = _FRESH_PROBE.replace('@@MODULE@@', pathlib.Path(__file__).stem).replace('@@CALL@@', call)
    try:
        result = subprocess.run([sys.executable, '-c', probe, str(pathlib.Path(__file__).parent)],
                                capture_output=True, text=True, env=env, timeout=1800, cwd=str(src.parent))
    except subprocess.TimeoutExpired as exc:
        pytest.fail('fresh-process check timed out: ' + call + ' ' + str(exc.stdout)[-500:])
    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report['ok'], report['error']


_FRESH_PROBE = r'''
import json, sys, traceback
sys.path.insert(0, sys.argv[1])
import @@MODULE@@ as t
try:
    @@CALL@@
    report = {'ok': True, 'error': ''}
except BaseException:
    report = {'ok': False, 'error': traceback.format_exc()[-3000:]}
print(json.dumps(report))
'''


def _check_one_full_cache(path):
    modules = _all_modules()
    empty = _actions(ledger.loaded_geometry_ledger(trusted_image=True))
    _fill(path)
    assert _planning_purity.inventory(modules).truncated       # the shared budget alone would be exceeded
    result = ledger.loaded_geometry_ledger(trusted_image=True)
    assert not result.report.truncated
    result.validate()
    assert _actions(result) == empty                           # the same decision as with the cache empty
    assert result.rows[path]['action'] == ('reset' if path in RESET_PATHS else 'decline')
    assert len(_store(path)) == _limit(path)                   # validated without clearing


@pytest.mark.parametrize('path', PATHS)
def test_one_full_cache_does_not_truncate_the_planning_census(path):
    _fresh_process_check('t._check_one_full_cache(%r)' % path)


def test_all_four_full_caches_do_not_truncate_the_planning_census():
    _fresh_process_check('t._check_all_four_full_caches()')


def _check_all_four_full_caches():
    _all_modules()
    empty = _actions(ledger.loaded_geometry_ledger(trusted_image=True))
    for path in FILL_ORDER:
        _fill(path)
    result = ledger.loaded_geometry_ledger(trusted_image=True)
    assert not result.report.truncated
    result.validate()             # the all-module census validates; other declined rows still refuse the host
    assert _actions(result) == empty
    # The ledger of the four caches' own modules: full, each a 'reset' row and no other declined row except
    # copyreg's __slotnames__ entries a long test process may create (host admission itself, validate_host,
    # is shown in a fresh process by the package probe below). _TRIG_CACHE is not among them.
    owners = tuple(dict.fromkeys(_module(path) for path in PATHS))
    host = ledger.loaded_geometry_ledger(trusted_image=True, modules=owners)
    host.validate()
    assert all(host.rows[path]['action'] == 'reset' for path in PATHS)
    assert not [path for path, row in host.rows.items()
                if row['action'] == 'decline' and not path.endswith('.__slotnames__')]
    assert all(len(_store(path)) == _limit(path) for path in PATHS)


# The table: every registered store, its reviewed limit and its measured per-entry budget.

def test_the_table_registers_exactly_the_four_caches_at_their_reviewed_limits():
    rows = {module + '.' + store: (name, limit) for module, store, name, limit, _, _ in ledger._BOUNDED_STORES}
    assert set(rows) == set(PATHS) and TRIG_PATH not in rows
    for path in PATHS:
        assert rows[path][0] == CACHES[path][2] and rows[path][1] == _limit(path)


def test_trig_cache_stays_a_declined_store_and_a_full_one_still_declines():
    _fresh_process_check('t._check_trig_cache_declined_store()')


def _check_trig_cache_declined_store():
    # Owner decision 2026-11-25: no reserved budget for _TRIG_CACHE. Small, it is a declined row as before;
    # full to its limit, the shared census truncates and the ledger declines (serial planning), as on main.
    from anygeometry import separating_plane_certificate as spc
    saved = dict(spc._TRIG_CACHE)
    try:
        spc._TRIG_CACHE.clear()
        _fill_trig(8)
        small = ledger.loaded_geometry_ledger(trusted_image=True)
        assert not small.report.truncated and small.rows[TRIG_PATH]['action'] == 'decline'
        _fill_trig(spc._TRIG_CACHE_LIMIT - len(spc._TRIG_CACHE), start=8)
        assert len(spc._TRIG_CACHE) == spc._TRIG_CACHE_LIMIT
        with pytest.raises(QualificationDecline):
            ledger.loaded_geometry_ledger(trusted_image=True)
    finally:
        spc._TRIG_CACHE.clear()
        spc._TRIG_CACHE.update(saved)


def _entry_cost(key, value):
    """(walk nodes, first-reached identities) of one entry measured alone on the interpreter census."""
    module = types.ModuleType('zz_entry_measure')
    module.store = {key: value}
    usage = {}
    _planning_purity.inventory((module,), max_nodes=10**7, bounded_usage=usage,
                               bounded={id(module.store): (module.store, module, 'store', 1, 10**7, 10**7)})
    _, _, nodes, identities, _, _ = usage[id(module.store)]['walked']
    return nodes, identities


def _entry_nodes(key, value):
    return _entry_cost(key, value)[0]


@pytest.mark.parametrize('path', PATHS)
def test_entry_budget_covers_the_writers_largest_entry(caches, path):
    # Each entry measured alone (no descendant seen before): the most a writer's entry can cost.
    FILL[path](64)
    budget, identities = ledger.bounded_store_row(path)[4:6]
    costs = [_entry_cost(key, value) for key, value in _store(path).items()]
    assert max(nodes for nodes, _ in costs) <= budget
    assert max(ids for _, ids in costs) <= identities
    if path == 'anygeometry.exact_curves._DOUBLE_NODES':
        # The writer's largest entry holds two nodes; the fill's pairs hold none.
        from anygeometry import exact_curves as ec
        first, second, _ = next(iter(ec._DOUBLE_NODES.values()))
        nodes = (ec._DoubleNode(1, 1, 0.), ec._DoubleNode(-1, 1, 3.))
        cost = _entry_cost((id(first), id(second)), (first, second, nodes))
        assert cost[0] <= budget and cost[1] <= identities


def test_the_ledger_table_itself_is_no_declined_census_store():
    # A module-level dict in the ledger would itself be a declined store and refuse every route.
    result = ledger.loaded_geometry_ledger(trusted_image=True, modules=(ledger,))
    # copyreg's __slotnames__ entries are unrelated state a long test process may have created.
    assert not [path for path, row in result.rows.items()
                if row['action'] == 'decline' and not path.endswith('.__slotnames__')]


def test_a_foreign_module_of_a_registered_name_gets_no_reserved_budget(caches):
    # Not the imported module: walked as on the parent, against the shared budget only (never declined
    # for lacking the limit, never granted the reserved budget).
    for path in PATHS:
        impostor = types.ModuleType(CACHES[path][0])
        setattr(impostor, CACHES[path][1], {('warm', 0): 1})
        assert ledger._bounded_registry((impostor,)) == {}
        result = ledger.loaded_geometry_ledger(trusted_image=True, modules=(impostor,))
        assert not result.report.truncated
        setattr(impostor, CACHES[path][1], {('warm', i): tuple(float(j) for j in range(30)) for i in range(6000)})
        with pytest.raises(QualificationDecline, match='coverage mismatch'):
            ledger.loaded_geometry_ledger(trusted_image=True, modules=(impostor,))


# Criterion 3: fail closed for each cache.

@pytest.mark.parametrize('path', PATHS)
def test_cache_past_its_limit_declines(caches, path):
    store = _store(path)
    store.update((('pad', i), None) for i in range(_limit(path) + 1))
    with pytest.raises(QualificationDecline, match='exceeds its limit'):
        ledger.loaded_geometry_ledger(trusted_image=True, modules=(_module(path),))


@pytest.mark.parametrize('path', PATHS)
def test_cache_at_its_limit_of_trivial_entries_is_admitted(caches, path):
    _store(path).update((('pad', i), None) for i in range(_limit(path)))
    ledger.loaded_geometry_ledger(trusted_image=True, modules=(_module(path),)).validate()


@pytest.mark.parametrize('path', PATHS)
def test_entry_past_its_budget_declines(caches, path):
    budget = ledger.bounded_store_row(path)[4]
    store = _store(path)
    store[1] = (0.,)*(budget - 2)                    # key 1 + tuple 1 + budget-2 floats: exactly the budget
    ledger.loaded_geometry_ledger(trusted_image=True, modules=(_module(path),)).validate()
    store[1] = (0.,)*(budget - 1)                    # one node more
    # Truncated inside the store: the census stops there, so a later bounded store is not reached either.
    with pytest.raises(QualificationDecline, match='coverage mismatch|registered binding'):
        ledger.loaded_geometry_ledger(trusted_image=True, modules=(_module(path),))


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('limit', ['double', 'float', True, None])
def test_changed_limit_declines(caches, monkeypatch, path, limit):
    value = {'double': 2*_limit(path), 'float': float(_limit(path))}.get(limit, limit)
    monkeypatch.setattr(_module(path), CACHES[path][2], value)
    with pytest.raises(QualificationDecline, match='limit changed or removed'):
        ledger.loaded_geometry_ledger(trusted_image=True, modules=(_module(path),))


@pytest.mark.parametrize('path', PATHS)
def test_removed_limit_declines(caches, monkeypatch, path):
    monkeypatch.delattr(_module(path), CACHES[path][2])
    with pytest.raises(QualificationDecline, match='limit changed or removed'):
        ledger.loaded_geometry_ledger(trusted_image=True, modules=(_module(path),))


class _Unbounded(dict):
    pass


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('replacement', [OrderedDict, _Unbounded, list])
def test_replaced_store_type_declines(caches, monkeypatch, path, replacement):
    monkeypatch.setattr(_module(path), CACHES[path][1], replacement())
    with pytest.raises(QualificationDecline, match='exact dict'):
        ledger.loaded_geometry_ledger(trusted_image=True, modules=(_module(path),))


@pytest.mark.parametrize('path', PATHS)
def test_store_reached_first_through_another_binding_declines(caches, monkeypatch, path):
    other = types.ModuleType('anygeometry._bounded_alias_fixture')
    other.held = _store(path)
    monkeypatch.setitem(sys.modules, other.__name__, other)
    with pytest.raises(QualificationDecline, match='registered binding'):
        ledger.loaded_geometry_ledger(trusted_image=True, modules=(other, _module(path)))


# Criterion 2: the Codex round-1 High/Medium findings, for every cache.

def _descendant(path):
    """A mutable or identity-bearing descendant of one real entry."""
    FILL[path](2)
    key, value = next(iter(_store(path).items()))
    if path == 'anygeometry.quadric_curve_events._BRANCH_POLYNOMIALS':
        return value[2][5]                             # the entry's mutable products dict
    if path == 'anygeometry.quadric_events._MEMO':
        return key[0]                                  # the memoized curve
    return value[0] if type(value[0]) is not tuple else value


@pytest.mark.parametrize('path', RESET_PATHS)
def test_h1_cache_descendant_aliased_elsewhere_declines(caches, monkeypatch, path):
    other = types.ModuleType('anygeometry.zz_bounded_alias_fixture')
    other.held = [_descendant(path)]
    monkeypatch.setitem(sys.modules, other.__name__, other)
    result = ledger.loaded_geometry_ledger(trusted_image=True, modules=(_module(path), other))
    with pytest.raises(QualificationDecline, match='external alias'):
        result.validate()


def test_h1_entry_object_aliased_into_the_standalone_store_declines(caches, monkeypatch):
    # The 'reset descendant has external alias: ..._STANDALONE.key[0]' decline seen in full-suite shards:
    # an earlier test left an object of a cache entry (here the memoized curve of a _MEMO key) also
    # referenced from _canonical_curve_runtime._STANDALONE, so resetting the cache would not drop it. The
    # census declines fail-closed; it is why the full-cache census checks run in a fresh process.
    from anygeometry import _canonical_curve_runtime as runtime
    path = 'anygeometry.quadric_events._MEMO'
    _fill_memo(1)
    curve = next(iter(_store(path)))[0]
    monkeypatch.setitem(runtime._STANDALONE, (curve,), None)
    result = ledger.loaded_geometry_ledger(trusted_image=True, modules=(_module(path), runtime))
    with pytest.raises(QualificationDecline, match=r'external alias: .*_STANDALONE'):
        result.validate()


def _hiding_class(extra):
    return type('_Hidden', (), {'__slots__': ('value',) + (('extra',) if extra else ()), '__module__': __name__})


@pytest.mark.parametrize('path', PATHS)
def test_h2_entry_object_hiding_nested_mutable_state_declines(caches, path):
    nested = []
    for _ in range(40):
        nested = [nested]
    holder = object.__new__(_hiding_class(True))
    object.__setattr__(holder, 'value', 1.)
    object.__setattr__(holder, 'extra', nested)
    _store(path)[('hidden', 0)] = (holder,)
    with pytest.raises(QualificationDecline):
        ledger.loaded_geometry_ledger(trusted_image=True, modules=(_module(path),))


@pytest.mark.parametrize('path', PATHS)
def test_h2_census_never_executes_an_entry_descriptor(caches, path):
    calls = []
    cls = type('_Hidden', (), {'__slots__': (), '__module__': __name__,
                               'value': property(lambda self: calls.append(1) or 1.),
                               '__getattribute__': lambda self, name: calls.append(name) or 1.})
    _store(path)[('hidden', 0)] = (cls(),)
    try:
        ledger.loaded_geometry_ledger(trusted_image=True, modules=(_module(path),)).validate()
    except QualificationDecline:
        pass
    assert calls == []


@pytest.mark.parametrize('path', PATHS)
def test_m3_one_oversized_entry_declines(caches, path):
    _store(path)[('big', 0)] = tuple(float(i) for i in range(160001))
    with pytest.raises(QualificationDecline):
        ledger.loaded_geometry_ledger(trusted_image=True, modules=(_module(path),))


@pytest.mark.parametrize('path', PATHS)
def test_m4_growth_inside_one_census_declines(caches, monkeypatch, path):
    store = _store(path)
    store.update((('pad', i), None) for i in range(_limit(path)))
    result = ledger.loaded_geometry_ledger(trusted_image=True, modules=(_module(path),))
    real, raced = ledger.inventory, []

    def racing(*args, **kwargs):
        if not raced:
            store[('pad', -1)] = None              # inside one census: after the bound check, before the walk
            raced.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(ledger, 'inventory', racing)
    with pytest.raises(QualificationDecline):
        result.validate()
    assert raced


@pytest.mark.parametrize('path', PATHS)
def test_m4_growth_after_capture_declines_at_validation(caches, path):
    store = _store(path)
    store.update((('pad', i), None) for i in range(_limit(path)))
    result = ledger.loaded_geometry_ledger(trusted_image=True, modules=(_module(path),))
    store[('pad', -1)] = None
    with pytest.raises(QualificationDecline, match='exceeds its limit'):
        result.validate()


# The package worker ledger (manifest route): the same budgets on its alias walk.

_PACKAGE_PROBE = r'''
import sys
sys.path.insert(0, sys.argv[1])
import test_planning_census_reserved_budgets_20261125 as t
from anygeometry import _planning_ledger as ledger
from anygeometry._planning_canaries import _fixture
from anygeometry._planning_executor import freeze_model, thaw_model
from anygeometry._planning_qualification import QualificationDecline
owned = ledger.package_worker_ledger(thaw_model(freeze_model(_fixture()[0])))
assert set(t.RESET_PATHS) <= set(owned.resets), sorted(owned.resets)
for path in t.FILL_ORDER:
    if path in t.RESET_PATHS:
        t._fill(path)
assert all(len(t._store(path)) == t._limit(path) for path in t.RESET_PATHS)
owned.validate()                     # the manifest route's alias walk with the four reset caches full
print('FULL OK')
for path in t.RESET_PATHS:
    store = t._store(path)
    saved = dict(store)
    _, _, _, limit, _, identities = next(row for row in ledger._BOUNDED_STORES if row[0] + '.' + row[1] == path)
    store.clear()
    store[('big', 0)] = tuple(float(i) for i in range(limit*identities))         # one identity per float
    try:
        owned.validate()
        print('ADMITTED', path)
    except QualificationDecline as exc:
        print('DECLINED', path, exc)
    store.clear(); store.update(saved)
    store.update((('pad', i), None) for i in range(limit + 1 - len(store)))     # one entry past the limit
    try:
        owned.validate()
        print('ADMITTED', path)
    except QualificationDecline as exc:
        print('DECLINED', path, exc)
    store.clear(); store.update(saved)
owned.validate()
print('RESTORED OK')
owned.prepare(owned_worker=True)     # admission and reset of the four full caches on the manifest route
assert all(len(t._store(path)) == 0 for path in t.RESET_PATHS)
print('PREPARED OK')
'''


def test_full_caches_pass_the_package_worker_ledger_and_budgets_hold():
    # A fresh process: the package ledger refuses unrelated state a long test process accumulates.
    src = pathlib.Path(ledger.__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(src))
    result = subprocess.run([sys.executable, '-c', _PACKAGE_PROBE, str(pathlib.Path(__file__).parent)],
                            capture_output=True, text=True, env=env, timeout=900)
    assert result.returncode == 0, result.stderr[-2000:]
    lines = result.stdout.splitlines()
    assert lines[0] == 'FULL OK' and lines[-2:] == ['RESTORED OK', 'PREPARED OK'], result.stdout
    for path in RESET_PATHS:
        assert 'DECLINED ' + path + ' worker cache alias walk bound exceeded: ' + path in lines, result.stdout
        assert 'DECLINED ' + path + ' bounded store exceeds its limit: ' + path in lines, result.stdout


_HOST_PROBE = r'''
import sys
sys.path.insert(0, sys.argv[1])
import test_planning_census_reserved_budgets_20261125 as t
from anygeometry import _planning_ledger as ledger
for path in t.FILL_ORDER:
    t._fill(path)
owners = tuple(dict.fromkeys(t._module(path) for path in t.PATHS))
host = ledger.loaded_geometry_ledger(trusted_image=True, modules=owners)
host.validate_host()                 # host admission of the four full caches, nothing cleared
assert all(host.rows[path]['action'] == 'reset' for path in t.PATHS)
assert all(len(t._store(path)) == t._limit(path) for path in t.PATHS)
print('HOST OK')
'''


def test_four_full_caches_are_admitted_by_the_host_ledger_of_their_modules():
    # A fresh process (a long test process creates unrelated copyreg __slotnames__ stores, declined rows).
    src = pathlib.Path(ledger.__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(src))
    result = subprocess.run([sys.executable, '-c', _HOST_PROBE, str(pathlib.Path(__file__).parent)],
                            capture_output=True, text=True, env=env, timeout=900)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.splitlines() == ['HOST OK'], result.stdout


# The slot read never hashes a type and reads every instance against its type's current layout.

class _HashingMeta(type):
    hashed = []

    def __hash__(cls):
        _HashingMeta.hashed.append(cls)
        return 0


def test_slot_read_never_hashes_a_type_and_walks_every_instance():
    cls = _HashingMeta('_Slotted', (), {'__slots__': ('first', 'second'), '__module__': __name__})
    module = types.ModuleType('zz_layout_fixture')
    first, second = cls(), cls()
    first.first, first.second = 1., [1.]
    second.first = {}                                   # 'second' unset: skipped, as before
    module.items = (first, second)
    report = _planning_purity.inventory((module,))
    assert _HashingMeta.hashed == []
    assert [s.path for s in report.stores] == ['zz_layout_fixture.items[0].second', 'zz_layout_fixture.items[1].first']
    assert [path for path, _ in report.unknowns] == ['zz_layout_fixture.items[0]', 'zz_layout_fixture.items[1]']


# Round 2: the Codex round-1 probes on 4b91b68 (findings 1-4). Each test passes here and on the parent 1ac4352,
# which refuses every probe (there because the warm cache alone exceeds its shared budget); 4b91b68 and
# fb04f3b admitted each one. On the parent the reserved-budget table does not exist: only the refusal is checked.

PARENT = not hasattr(ledger, '_BOUNDED_STORES')
BC = 'anygeometry.exact_curves._BRANCH_CONSTANTS'


def _pad_full(path, floats=58):
    """The cache full to its limit, each entry an int key and a tuple of distinct floats (60 walk nodes)."""
    store = _store(path)
    store.clear()
    store.update((i, tuple(float(i*1000 + j) + .5 for j in range(floats))) for i in range(_limit(path)))


def _refused(build, then=None):
    """The decline text of build() (and then(result)), or None when both pass."""
    try:
        result = build()
        if then is not None:
            then(result)
    except QualificationDecline as exc:
        return str(exc)
    return None


@pytest.mark.parametrize('order', ['cache first', 'cache last'])
def test_r2_f1_ordinary_state_aliased_into_a_cache_stays_on_the_shared_budget(caches, order):
    # 3,000 lists of 60 floats in the unreserved _ISO_CACHE exceed the shared budget. Aliasing the same lists
    # into _BRANCH_CONSTANTS must not move them to its reserve, whichever module the census reaches first.
    from anygeometry import exact_curves as ec, nurbs
    modules = (ec, nurbs) if order == 'cache first' else (nurbs, ec)
    lists = [[float(i*100 + j) + .5 for j in range(60)] for i in range(3000)]
    saved = dict(nurbs._ISO_CACHE)
    try:
        nurbs._ISO_CACHE.clear()
        nurbs._ISO_CACHE.update((('iso', i), item) for i, item in enumerate(lists))
        build = lambda: ledger.loaded_geometry_ledger(trusted_image=True, modules=modules)
        assert 'coverage mismatch' in _refused(build)                 # the shared budget alone
        ec._BRANCH_CONSTANTS.update(enumerate(lists))                 # each entry 62 nodes: within its budget
        assert 'coverage mismatch' in _refused(build)
    finally:
        nurbs._ISO_CACHE.clear()
        nurbs._ISO_CACHE.update(saved)


def test_r2_f1_entry_identity_budget_holds_on_the_interpreter_census(caches):
    # Within the node budget (402 of 512) but past the identity budget (320): 400 distinct lists in one entry.
    path = 'anygeometry.quadric_curve_events._BRANCH_POLYNOMIALS'
    _pad_full(path)
    store = _store(path)
    store[0] = tuple([] for _ in range(400))
    message = _refused(lambda: ledger.loaded_geometry_ledger(trusted_image=True, modules=(_module(path),)))
    assert message is not None
    if not PARENT:
        assert 'coverage mismatch' in message
        store[0] = tuple([] for _ in range(300))                      # within both budgets: admitted
        ledger.loaded_geometry_ledger(trusted_image=True, modules=(_module(path),)).validate()


def test_r2_f2_slot_layout_is_read_again_when_a_descriptor_returns_mid_walk(caches, monkeypatch):
    # The walk reads the first instance's layout without the removed native slot descriptor; it comes back
    # before the second instance is walked, whose slot holds a list also bound outside the cache.
    from anygeometry import exact_curves as ec
    cls = type('_Slotted', (), {'__slots__': ('held',), '__module__': __name__})
    first, second, held = cls(), cls(), [1.]
    second.held = held
    descriptor = cls.__dict__['held']
    other = types.ModuleType('anygeometry.zz_layout_alias_fixture')
    other.exposed = held
    monkeypatch.setitem(sys.modules, other.__name__, other)
    ec._BRANCH_CONSTANTS[('slots', 0)] = (first, second)
    del cls.held
    result = ledger.loaded_geometry_ledger(trusted_image=True, modules=(ec, other))
    real = ledger.inventory

    def racing(*args, **kwargs):
        boundary = kwargs['infrastructure_boundary']

        def hooked(value, path):
            if value is second and 'held' not in cls.__dict__:
                cls.held = descriptor
            return boundary(value, path)
        return real(*args, **dict(kwargs, infrastructure_boundary=hooked))

    monkeypatch.setattr(ledger, 'inventory', racing)
    with pytest.raises(QualificationDecline, match='external alias'):
        result.validate()
    assert 'held' in cls.__dict__


@pytest.mark.parametrize('change', ['limit', 'entry'])
def test_r2_f3_a_change_after_the_registry_check_declines(caches, monkeypatch, change):
    # 'limit': the limit constant doubled after the registry check, before the walk (Codex's probe).
    # 'entry': an already walked entry replaced with a large tuple while the walk is still in the store.
    from anygeometry import exact_curves as ec
    _pad_full(BC)
    store, limit = ec._BRANCH_CONSTANTS, ec._BRANCH_CONSTANTS_LIMIT
    real, last = ledger.inventory, store[limit - 1]

    def racing(*args, **kwargs):
        if change == 'limit':
            monkeypatch.setattr(ec, '_BRANCH_CONSTANTS_LIMIT', 2*limit)
            return real(*args, **kwargs)
        boundary = kwargs['infrastructure_boundary']

        def hooked(value, path):
            if value is last:
                store[0] = tuple(float(i) + .5 for i in range(200000))
            return boundary(value, path)
        return real(*args, **dict(kwargs, infrastructure_boundary=hooked))

    def then(result):
        monkeypatch.setattr(ledger, 'inventory', racing)
        result.validate()

    message = _refused(lambda: ledger.loaded_geometry_ledger(trusted_image=True, modules=(ec,)), then)
    assert message is not None
    if not PARENT:
        assert ('limit changed or removed' if change == 'limit' else 'changed during the census') in message


@pytest.mark.parametrize('order', ['impostor first', 'impostor last'])
def test_r2_f4_an_impostor_binding_the_real_store_gets_no_reserve(caches, order):
    # A module object of the same name binding the real _BRANCH_CONSTANTS under the same name, in either order.
    from anygeometry import exact_curves as ec
    _pad_full(BC)
    impostor = types.ModuleType(ec.__name__)
    impostor._BRANCH_CONSTANTS = ec._BRANCH_CONSTANTS
    modules = (impostor, ec) if order == 'impostor first' else (ec, impostor)
    if not PARENT:
        ledger.loaded_geometry_ledger(trusted_image=True, modules=(ec,)).validate()   # control: admitted alone
    message = _refused(lambda: ledger.loaded_geometry_ledger(trusted_image=True, modules=modules))
    assert message is not None
    if not PARENT:
        assert 'not reached only at its registered binding' in message


# The package worker route (a fresh process): findings 1 and 3.

_PACKAGE_R2_PROBE = r'''
import gc, sys
sys.path.insert(0, sys.argv[1])
import test_planning_census_reserved_budgets_20261125 as t
from anygeometry import _planning_ledger as ledger, exact_curves as ec, branch_predicates as bp, quadric_events as qe
from anygeometry._planning_canaries import _fixture
from anygeometry._planning_executor import freeze_model, thaw_model
from anygeometry._planning_qualification import QualificationDecline
owned = ledger.package_worker_ledger(thaw_model(freeze_model(_fixture()[0])))
owned.validate()
real_referents, limit = gc.get_referents, ec._BRANCH_CONSTANTS_LIMIT
sentinel = None

def case(name, setup, hook=None):
    global sentinel
    def hooked(*objects):
        # Fires once, when the walk asks for the sentinel's referents, and puts the real function back first:
        # the manifest check after the walk refuses a rebound gc.get_referents.
        if any(o is sentinel for o in objects):
            gc.get_referents = real_referents
            hook()
        return real_referents(*objects)
    sentinel = setup()
    if hook is not None:
        gc.get_referents = hooked
    try:
        owned.validate()
        print(name, 'ADMITTED' if gc.get_referents is real_referents else 'NOT-REACHED')
    except QualificationDecline as exc:
        print(name, 'DECLINED', exc)
    except Exception as exc:
        print(name, 'ERROR', type(exc).__name__, exc)
    finally:
        gc.get_referents = real_referents
        ec._BRANCH_CONSTANTS_LIMIT = limit
        for store in (ec._BRANCH_CONSTANTS, bp._PATCHES, qe._MEMO):
            store.clear()

def big():
    return tuple(float(i) + .5 for i in range(80001))

def patches(alias):
    value = big()
    bp._PATCHES[('big', 0)] = value
    if alias:
        qe._MEMO[('alias', 0)] = value

def padded():
    t._pad_full(t.BC)
    return ec._BRANCH_CONSTANTS[7]

case('f1-patches', lambda: patches(False))
case('f1-patches-aliased-from-memo', lambda: patches(True))
case('f1-tuple-in-entry', lambda: ec._BRANCH_CONSTANTS.__setitem__(('big', 0), big()))
case('f3-padded-full', padded)
case('f3-entry-after-capture', padded, lambda: ec._BRANCH_CONSTANTS.__setitem__(('pad', -1), None))
case('f3-limit-mid-walk', padded, lambda: setattr(ec, '_BRANCH_CONSTANTS_LIMIT', 2*limit))
owned.validate()
print('DONE')
'''


def _package_r2_lines():
    src = pathlib.Path(ledger.__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(src))
    result = subprocess.run([sys.executable, '-c', _PACKAGE_R2_PROBE, str(pathlib.Path(__file__).parent)],
                            capture_output=True, text=True, env=env, timeout=900)
    assert result.returncode == 0, result.stderr[-2000:]
    lines = result.stdout.splitlines()
    assert lines[-1] == 'DONE', result.stdout
    return dict(line.split(' ', 1) for line in lines[:-1])


def test_r2_package_route_probes():
    out = _package_r2_lines()
    shared = 'DECLINED worker cache alias walk bound exceeded'
    # Finding 1: what another reset store reaches is charged to the shared 80,000, whichever store comes first.
    assert out['f1-patches'] == shared and out['f1-patches-aliased-from-memo'] == shared, out
    if PARENT:
        assert all(value.startswith('DECLINED') for value in out.values()), out
        return
    # Finding 1: one entry's identities are bounded by its own budget, not by the store's.
    assert out['f1-tuple-in-entry'] == shared + ': ' + BC, out
    # Finding 3: the walked store is checked again after the walk.
    assert out['f3-padded-full'] == 'ADMITTED', out
    assert out['f3-entry-after-capture'] == 'DECLINED bounded store exceeds its limit: ' + BC, out
    assert out['f3-limit-mid-walk'] == 'DECLINED bounded store limit changed or removed: ' + BC, out
