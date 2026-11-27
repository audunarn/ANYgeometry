"""Module state that the planner's census refuses (0.4.7 gate class C, 2026-11-24).

With the census bound raised (test_planning_census_headroom_20261124), the census of a planning worker
completed and refused three stores added after c3b2704, hidden until then by the truncated census:
``arrangement_geometry._SWEEP_REJECTION`` (a context switch whose default was True),
``exact_curves._DoubleNode.__annotate_func__.cell[0]`` (the Python 3.14 annotation cell of a
typing.NamedTuple) and ``source_tensor_chart._METHODS`` (a dict built at import).
Owner decision (2026-11-24): change the code so these stores do not exist, rather than registering them
as trusted stores. Behaviour is unchanged.
"""
import json
import math
import os
import pathlib
import subprocess
import sys

import pytest

import importlib

from anygeometry import arrangement_geometry, exact_curves
from anygeometry._planning_purity import inventory

# the package exports a function of the same name, so take the module itself
source_tensor_chart = importlib.import_module('anygeometry.source_tensor_chart')


def _store_paths(module):
    # In a fresh interpreter: a long test process warms module caches whose entries reach state other tests
    # left behind, and the raw census of one module then truncates (0.4.7 gate run 45663432941, shard 8).
    # The stores a module defines at import are what this file is about.
    src = pathlib.Path(module.__file__).resolve().parents[1]
    env = {key: value for key, value in os.environ.items() if key != 'PYTHONOPTIMIZE'}
    env['PYTHONPATH'] = str(src)
    probe = ('import importlib, json\n'
             'from anygeometry._planning_purity import inventory\n'
             'report = inventory((importlib.import_module(%r),))\n'
             'print(json.dumps({"truncated": report.truncated, "paths": [s.path for s in report.stores]}))\n'
             % module.__name__)
    result = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True, env=env,
                            cwd=str(src.parent), timeout=600)
    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert not report['truncated']
    return report['paths']


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


def test_a_double_node_copies_and_pickles_like_the_former_named_tuple():
    """Codex review of d03e595 (Low): tuple.__reduce_ex__ rebuilt the class with one argument."""
    import copy
    import pickle
    node = exact_curves._DoubleNode(1, -1, math.pi)
    for clone in (copy.copy(node), copy.deepcopy(node), pickle.loads(pickle.dumps(node))):
        assert type(clone) is exact_curves._DoubleNode and clone == node and clone.angle == math.pi
