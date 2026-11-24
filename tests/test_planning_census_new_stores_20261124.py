"""Module state that the planner's census refuses (0.4.7 gate class C, 2026-11-24).

With the census bound raised (test_planning_census_headroom_20261124), the census of a planning worker
completed and refused three stores added after c3b2704, hidden until then by the truncated census:
``arrangement_geometry._SWEEP_REJECTION`` (a context switch whose default was True),
``exact_curves._DoubleNode.__annotate_func__.cell[0]`` (the Python 3.14 annotation cell of a
typing.NamedTuple) and ``source_tensor_chart._METHODS`` (a dict built at import).
Owner decision (2026-11-24): change the code so these stores do not exist, rather than registering them
as trusted stores. Behaviour is unchanged.
"""
import math

import pytest

import importlib

from anygeometry import arrangement_geometry, exact_curves
from anygeometry._planning_purity import inventory

# the package exports a function of the same name, so take the module itself
source_tensor_chart = importlib.import_module('anygeometry.source_tensor_chart')


def _store_paths(module):
    report = inventory((module,))
    assert not report.truncated
    return [store.path for store in report.stores]


def test_the_sweep_rejection_switch_is_empty_by_default_and_off_only_inside_the_policy():
    assert arrangement_geometry._SWEEP_REJECTION.get() is None          # None: rejection on (default)
    with arrangement_geometry._junction_bounds_policy(False):
        assert arrangement_geometry._SWEEP_REJECTION.get() is False     # the unfiltered exact-accounting route
    with arrangement_geometry._junction_bounds_policy(True):
        assert arrangement_geometry._SWEEP_REJECTION.get() is None
    assert arrangement_geometry._SWEEP_REJECTION.get() is None


def test_the_double_node_type_has_no_annotation_store():
    assert not [p for p in _store_paths(exact_curves) if p.startswith('anygeometry.exact_curves._DoubleNode.__annotate')]
    node = exact_curves._DoubleNode(1, -1, math.pi)
    assert (node.sign, node.tau, node.angle) == (1, -1, math.pi) and tuple(node) == (1, -1, math.pi)
    assert exact_curves._DoubleNode._fields == ('sign', 'tau', 'angle')


def test_the_canonical_evaluator_table_is_not_a_mutable_store():
    assert not [p for p in _store_paths(source_tensor_chart) if '_METHODS' in p]


def test_a_replaced_evaluator_is_still_refused(monkeypatch):
    from anygeometry import GeometryError
    from anygeometry.surfaces import CoonsSurface
    original = CoonsSurface.evaluate

    def replaced(self, *args, **kwargs):
        return original(self, *args, **kwargs)
    monkeypatch.setattr(CoonsSurface, 'evaluate', replaced)
    probe = object.__new__(CoonsSurface)
    with pytest.raises(GeometryError, match='canonical authored evaluators'):
        source_tensor_chart._canonical_evaluators(probe)
