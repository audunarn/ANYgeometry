"""Opaque bounded stores in the planner's module-state census (0.4.7 gate class H, 2026-11-25).

``exact_curves._BRANCH_CONSTANTS`` and ``exact_curves._DOUBLE_NODES`` are reset roots bounded at
``_BRANCH_CONSTANTS_LIMIT`` (4096) entries each. Warm, they hold about 25 census nodes per entry, so a process
that had evaluated many cylinder pairs needed more than the census bound (160,000) and the ledger declined
('ledger creation live coverage mismatch'): the process route fell back to serial in warmed processes.

Owner decision 2026-11-25 ("Opaque bounded stores", bound kept at 160,000): the two caches are opaque like
LRU entries, behind an exact structural check that declines when the bound is removed, changed or
exceeded, the binding is replaced, or the entry shape cannot be established.
"""
import math
import sys
import types
from collections import OrderedDict

import numpy as np
import pytest

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


def test_full_bounded_caches_do_not_truncate_the_planning_census(caches):
    # Raw inventory of all loaded anygeometry modules with both caches full needs ~224,500 nodes.
    _fill()
    modules = tuple(m for n, m in sorted(sys.modules.items()) if n == 'anygeometry' or n.startswith('anygeometry.'))
    assert _planning_purity.inventory(modules).truncated           # the raw walk alone would exceed the bound
    result = ledger.loaded_geometry_ledger(trusted_image=True)     # the planner's own call
    assert not result.report.truncated
    result.validate()                                              # the boundary census again
    for name in BOUNDED:
        assert result.rows['anygeometry.exact_curves.' + name]['action'] == 'reset'
    assert len(ec._BRANCH_CONSTANTS) == len(ec._DOUBLE_NODES) == 4096       # admitted without clearing


def test_exact_curves_census_is_independent_of_cache_fill(caches):
    empty = ledger.loaded_geometry_ledger(trusted_image=True, modules=(ec,))
    _fill()
    full = ledger.loaded_geometry_ledger(trusted_image=True, modules=(ec,))
    full.validate()
    assert _rows(full) == _rows(empty)
    assert full.report.unknowns == empty.report.unknowns and full.report.aliases == empty.report.aliases
    report, objects = full.report, dict(full.objects)
    rebuilt = ledger.build_ledger(report, objects, modules=(ec,), trusted_image=True)
    assert _rows(rebuilt) == _rows(full)


def test_owned_worker_reset_still_clears_the_bounded_caches(caches):
    _fill(8)
    result = ledger.loaded_geometry_ledger(trusted_image=True, modules=(ec,))
    result.validate()
    # The opaque stores stay reset roots; only the registered reset (not a census walk) touches them.
    # (prepare() itself also refuses unrelated declined rows a long test process may have created, e.g.
    # copyreg's __slotnames__ entries, so the reset registry is exercised directly.)
    from anygeometry._planning_qualification import reset_declined
    paths = ['anygeometry.exact_curves.' + name for name in BOUNDED]
    assert all(path in result.resets for path in paths)
    reset_declined({path: result.resets[path] for path in paths}, owned_worker=True)
    assert not ec._BRANCH_CONSTANTS and not ec._DOUBLE_NODES


# Fail closed: each broken premise of the opaque treatment declines.

def test_cache_past_its_limit_declines(caches):
    _fill()
    key, (first, second, constants) = next(iter(ec._BRANCH_CONSTANTS.items()))
    extra = _pair(10**6)
    ec._BRANCH_CONSTANTS[(id(extra.first), id(extra.second))] = (extra.first, extra.second, constants)
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


def _corrupt_entries():
    pair = _pair(0)
    ec._branch_constants(pair)
    ec._double_nodes(pair)
    (key, (first, second, constants)), = ec._BRANCH_CONSTANTS.items()
    writeable = constants._replace(axis=np.array(constants.axis))
    other = _pair(1)
    return dict(
        mutable_payload=('_BRANCH_CONSTANTS', key, (first, second, [1.])),
        writeable_array=('_BRANCH_CONSTANTS', key, (first, second, writeable)),
        foreign_support=('_BRANCH_CONSTANTS', key, (types.SimpleNamespace(), second, constants)),
        key_mismatch=('_BRANCH_CONSTANTS', (id(other.first), id(other.second)), (first, second, constants)),
        mutable_node=('_DOUBLE_NODES', key, (first, second, ([1, 1, 0.],))),
        foreign_node_value=('_DOUBLE_NODES', key, (first, second, (ec._DoubleNode(1, 1, {}),))),
        long_entry=('_DOUBLE_NODES', key, (first, second, (), None)),
        string_key=('_DOUBLE_NODES', 'key', (first, second, ())),
    )


@pytest.mark.parametrize('case', ['mutable_payload', 'writeable_array', 'foreign_support', 'key_mismatch',
                                  'mutable_node', 'foreign_node_value', 'long_entry', 'string_key'])
def test_unestablished_entry_shape_declines(caches, case):
    name, key, entry = _corrupt_entries()[case]
    getattr(ec, name)[key] = entry
    with pytest.raises(QualificationDecline, match='content check failed'):
        ledger.loaded_geometry_ledger(trusted_image=True, modules=(ec,))


def test_mutable_support_field_declines(caches):
    pair = _pair(0)
    ec._branch_constants(pair)
    object.__setattr__(pair.first, 'origin', [0., 0., 0.])
    with pytest.raises(QualificationDecline, match='content check failed'):
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


def test_growth_past_the_limit_after_capture_declines_at_validation(caches):
    _fill()
    result = ledger.loaded_geometry_ledger(trusted_image=True, modules=(ec,))
    key, (first, second, constants) = next(iter(ec._BRANCH_CONSTANTS.items()))
    extra = _pair(10**6)
    ec._BRANCH_CONSTANTS[(id(extra.first), id(extra.second))] = (extra.first, extra.second, constants)
    with pytest.raises(QualificationDecline, match='exceeds its limit'):
        result.validate()
