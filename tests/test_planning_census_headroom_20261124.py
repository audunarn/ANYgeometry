"""Headroom of the planner's module-state census (0.4.7 gate class C, 2026-11-24).

The process planning route admits a worker only after ``_planning_ledger`` has taken a complete census
of the loaded ``anygeometry`` module state (``_planning_purity.inventory``). A census that reaches its
node bound is truncated and the route declines to the serial planner. On d71c5bc the census inside a
worker needed about 82,400 nodes against a bound of 80,000 (c3b2704: about 79,400), so the process
route declined on every qualified Windows job of the weekly Tests run.

Owner decision (2026-11-24): the bound is 160,000, and this test fails in ordinary CI once a fresh
process that has planned needs more than 75% of it, so growth is seen in the change that causes it.
"""
import json
import pathlib
import subprocess
import sys

import pytest

from anygeometry import _planning_purity

SRC = pathlib.Path(_planning_purity.__file__).resolve().parents[1]

PROBE = r'''
import json, math, sys
from uuid import UUID
from anygeometry import GeometryModel, plan_intersections
from anygeometry.generators import cylinder
from anygeometry._planning_purity import inventory, INVENTORY_MAX_NODES
model = GeometryModel(model_id=UUID(int=20261124))
model.insert_model(cylinder(1., 6., origin=(-3., 0., 0.), axis=(1., 0., 0.), radial_direction=(0., 1., 0.),
                            circumferential_segments=4))
model.add_plate(model.add_points(((-2.5, -2., .2), (2.5, -2., .2), (2.5, 2., .2), (-2.5, 2., .2))))
from anygeometry.batch_intersections import IntersectionBatchPolicy
plan_intersections(model, sorted(model.faces), policy=IntersectionBatchPolicy())
import anygeometry._planning_ledger                  # the census owner itself is loaded in a worker
modules = tuple(m for n, m in sorted(sys.modules.items()) if n == 'anygeometry' or n.startswith('anygeometry.'))
lo, hi = 1, 10**7
while lo < hi:
    mid = (lo + hi) // 2
    if inventory(modules, max_nodes=mid, max_depth=10**6).truncated:
        lo = mid + 1
    else:
        hi = mid
print(json.dumps(dict(modules=len(modules), needed=lo, bound=INVENTORY_MAX_NODES,
                      default_truncated=inventory(modules).truncated)))
'''


def _probe():
    env = dict(__import__('os').environ, PYTHONPATH=str(SRC))
    result = subprocess.run([sys.executable, '-c', PROBE], capture_output=True, text=True, env=env, timeout=600)
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_the_census_bound_is_the_owner_value():
    assert _planning_purity.INVENTORY_MAX_NODES == 160000
    import inspect
    assert inspect.signature(_planning_purity.inventory).parameters['max_nodes'].default == 160000


def test_a_planning_process_census_is_complete_with_a_quarter_of_the_bound_left():
    facts = _probe()
    assert not facts['default_truncated'], facts
    assert facts['needed'] <= .75*facts['bound'], facts
