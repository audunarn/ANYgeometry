"""Reserved census budgets for four limit-bounded caches (0.4.7 gate class H, owner decision 2026-11-25).

Five module caches bound their own size by a limit constant and, warm, hold more walk nodes than the shared
census budget (160,000) leaves free, so the census truncated and the process route declined ('ledger creation
live coverage mismatch'; on the package worker route 'worker cache alias walk bound exceeded'). Each cache is
now walked exactly as before, but its entries count against a reserved budget of its own (limit entries x a
measured per-entry node budget); a removed or changed limit, a store past its limit and an entry past its
budget still decline. The per-cache fill tests below use only the caches' own writers and limit constants, so
they run unchanged on the parent and fail there (census truncated).
"""
import importlib
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

@pytest.mark.parametrize('path', PATHS)
def test_one_full_cache_does_not_truncate_the_planning_census(caches, path):
    modules = _all_modules()
    empty = _actions(ledger.loaded_geometry_ledger(trusted_image=True))
    _fill(path)
    assert _planning_purity.inventory(modules).truncated       # the shared budget alone would be exceeded
    result = ledger.loaded_geometry_ledger(trusted_image=True)
    assert not result.report.truncated
    result.validate()
    assert _actions(result) == empty                           # the same decision as with the cache empty
    assert result.rows[path]['action'] == ('reset' if path in RESET_PATHS else 'decline')
    assert len(_store(path)) == _limit(path)                   # admitted without clearing


def test_all_four_full_caches_do_not_truncate_the_planning_census(caches):
    _all_modules()
    empty = _actions(ledger.loaded_geometry_ledger(trusted_image=True))
    for path in FILL_ORDER:
        _fill(path)
    result = ledger.loaded_geometry_ledger(trusted_image=True)
    assert not result.report.truncated
    result.validate()
    assert _actions(result) == empty
    assert all(len(_store(path)) == _limit(path) for path in PATHS)


# The table: every registered store, its reviewed limit and its measured per-entry budget.

def test_the_table_registers_exactly_the_four_caches_at_their_reviewed_limits():
    rows = {module + '.' + store: (name, limit) for module, store, name, limit, _, _ in ledger._BOUNDED_STORES}
    assert set(rows) == set(PATHS) and TRIG_PATH not in rows
    for path in PATHS:
        assert rows[path][0] == CACHES[path][2] and rows[path][1] == _limit(path)


def test_trig_cache_stays_a_declined_store_and_a_full_one_still_declines():
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


def _entry_nodes(key, value):
    module = types.ModuleType('zz_entry_measure')
    module.store = {key: value}
    usage = {}
    _planning_purity.inventory((module,), max_nodes=10**7, bounded_usage=usage,
                               bounded_store=lambda v, p: (1, 10**7) if v is module.store else None)
    return usage['zz_entry_measure.store'][1]


@pytest.mark.parametrize('path', PATHS)
def test_entry_budget_covers_the_writers_largest_entry(caches, path):
    # Each entry measured alone (no descendant seen before): the most a writer's entry can cost.
    FILL[path](64)
    budget = ledger.bounded_store_row(path)[4]
    largest = max(_entry_nodes(key, value) for key, value in _store(path).items())
    assert largest <= budget
    if path == 'anygeometry.exact_curves._DOUBLE_NODES':
        # The writer's largest entry holds two nodes; the fill's pairs hold none.
        from anygeometry import exact_curves as ec
        first, second, _ = next(iter(ec._DOUBLE_NODES.values()))
        nodes = (ec._DoubleNode(1, 1, 0.), ec._DoubleNode(-1, 1, 3.))
        assert _entry_nodes((id(first), id(second)), (first, second, nodes)) <= budget


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
'''


def test_full_caches_pass_the_package_worker_ledger_and_budgets_hold():
    # A fresh process: the package ledger refuses unrelated state a long test process accumulates.
    src = pathlib.Path(ledger.__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(src))
    result = subprocess.run([sys.executable, '-c', _PACKAGE_PROBE, str(pathlib.Path(__file__).parent)],
                            capture_output=True, text=True, env=env, timeout=900)
    assert result.returncode == 0, result.stderr[-2000:]
    lines = result.stdout.splitlines()
    assert lines[0] == 'FULL OK' and lines[-1] == 'RESTORED OK', result.stdout
    for path in RESET_PATHS:
        assert 'DECLINED ' + path + ' worker cache alias walk bound exceeded: ' + path in lines, result.stdout
        assert 'DECLINED ' + path + ' bounded store exceeds its limit: ' + path in lines, result.stdout


# The census reads each type's slot layout once (census time with full caches); the walk is unchanged.

class _HashingMeta(type):
    hashed = []

    def __hash__(cls):
        _HashingMeta.hashed.append(cls)
        return 0


def test_slot_layout_cache_never_hashes_a_type_and_walks_every_instance():
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
