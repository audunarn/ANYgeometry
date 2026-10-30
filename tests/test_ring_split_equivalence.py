"""RING-SPLIT RS-2: equivalence of the ring_split and baseline arrangements of one face (production brief RS-2).

Both policies cut the same material: the non-decomposition edges are equal as point sets (pieces may merge),
the cell areas sum to the same material area, every edge ring_split adds is a decomposition edge, and dissolving
the decomposition edges gives the same regions. The model-scale oracle (sector60, r7, face200, centre, the six
caps, leg1, plan and apply) is the harness redesign/rs_oracle.py outside the repository.
"""
import collections
import math

import numpy as np
import pytest

from anygeometry import EllipticArc
from anygeometry.arrangement_geometry import LinePath, point_parameters
from anygeometry.material_arrangement import ArrangementPath, ArrangementPoint, MaterialDomain, arrange_material
from anygeometry.surfaces import Plane

from test_ring_split_seams import AREA_TOL, N, TOL, cap


def _edge_area(curve):
    """0.5 * integral (x dy - y dx) in the world xy plane (the test supports lie in z = 0)."""
    if type(curve) is EllipticArc:
        cx, cy = curve.center[0], curve.center[1]
        ux, uy, vx, vy = curve.u_vector[0], curve.u_vector[1], curve.v_vector[0], curve.v_vector[1]
        t0, t1 = curve.start_angle, curve.start_angle + curve.sweep_angle
        dc, ds = math.cos(t1) - math.cos(t0), math.sin(t1) - math.sin(t0)
        return .5 * (cx * (uy * dc + vy * ds) - cy * (ux * dc + vx * ds) + (ux * vy - uy * vx) * (t1 - t0))
    (x0, y0, _), (x1, y1, _) = curve.evaluate(0.), curve.evaluate(1.)
    return .5 * (x0 * y1 - x1 * y0)


def _cell_area(arrangement, cell):
    loop = lambda edges: sum((1 if forward else -1) * _edge_area(arrangement.paths[i].curve) for i, forward in edges)
    return abs(loop(cell.outer)) - sum(abs(loop(hole)) for hole in cell.holes)


def _regions(arrangement):
    parent = list(range(len(arrangement.cells)))

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    users = collections.defaultdict(list)
    for c, cell in enumerate(arrangement.cells):
        for loop in (cell.outer, *cell.holes):
            for i, _forward in loop:
                users[i].append(c)
    for i, cells in users.items():
        if arrangement.paths[i].decomposition:
            for c in cells[1:]:
                parent[find(c)] = find(cells[0])
    area = collections.defaultdict(float)
    for c, cell in enumerate(arrangement.cells):
        area[find(c)] += _cell_area(arrangement, cell)
    return sorted(area.values())


def _length(curve):
    if type(curve) is EllipticArc:
        return abs(curve.sweep_angle) * float(np.linalg.norm(curve.u_vector))
    return float(np.linalg.norm(np.asarray(curve.evaluate(1.)) - np.asarray(curve.evaluate(0.))))


def _assert_equivalent(baseline, ring_split):
    group = lambda p: (type(p.curve).__name__, p.owners, p.source_edge)
    sides = []
    for arrangement in (baseline, ring_split):
        groups = collections.defaultdict(list)
        for p in arrangement.paths:
            if not p.decomposition:
                groups[group(p)].append(p.curve)
        sides.append(groups)
    a, b = sides
    assert set(a) == set(b)
    for key in a:
        assert abs(sum(map(_length, a[key])) - sum(map(_length, b[key]))) <= TOL * (len(a[key]) + len(b[key]))
        for mine, other in ((a[key], b[key]), (b[key], a[key])):
            for curve in mine:
                for t in (0., .25, .5, .75, 1.):
                    assert any(point_parameters(c, curve.evaluate(t), tolerance=TOL) for c in other), (key, t)
    total = [sum(_cell_area(x, c) for c in x.cells) for x in (baseline, ring_split)]
    assert abs(total[0] - total[1]) <= 1e-9 * total[0]
    assert abs(baseline.area - ring_split.area) <= 2 * baseline.native_area_tolerance
    ra, rb = _regions(baseline), _regions(ring_split)
    assert len(ra) == len(rb) and all(abs(p - q) <= 1e-9 * max(1., p) for p, q in zip(ra, rb))
    return ra


@pytest.mark.parametrize("angle", [0.0, 2 * math.pi / 3, 0.6435011087932844])
def test_cap_arrangements_are_equivalent(angle):
    domain, traces = cap(angle, frame=0.0)
    baseline = arrange_material(domain, traces, tolerance=TOL, area_tolerance=AREA_TOL, seam_policy="baseline")
    ring_split = arrange_material(domain, traces, tolerance=TOL, area_tolerance=AREA_TOL, seam_policy="ring_split")
    assert sum(p.decomposition for p in ring_split.paths) == N + 2
    regions = _assert_equivalent(baseline, ring_split)
    assert len(regions) == 2                                 # inside and outside the rim circle
    assert abs(sum(regions) - 400.) <= 1e-9 * 400            # the whole 20 x 20 plate


def test_point_end_and_crossing_variants_are_equivalent():
    domain, traces = cap(skip=(0,))
    points = (ArrangementPoint((5., 0., 0.)),)
    kw = dict(tolerance=TOL, area_tolerance=AREA_TOL, points=points)
    _assert_equivalent(arrange_material(domain, traces, seam_policy="baseline", **kw),
                       arrange_material(domain, traces, seam_policy="ring_split", **kw))
    domain, traces = cap()
    traces = (*traces, ArrangementPath(LinePath((10., .4, 0.), (5.5, 0., 0.)), owners=(99,)))
    _assert_equivalent(arrange_material(domain, traces, tolerance=TOL, area_tolerance=AREA_TOL, seam_policy="baseline"),
                       arrange_material(domain, traces, tolerance=TOL, area_tolerance=AREA_TOL,
                                        seam_policy="ring_split"))


def test_the_regions_check_detects_a_missing_material_edge():
    # the double: an oracle that ignores one non-decomposition edge must fail
    domain, traces = cap()
    baseline = arrange_material(domain, traces, tolerance=TOL, area_tolerance=AREA_TOL, seam_policy="baseline")
    ring_split = arrange_material(domain, traces, tolerance=TOL, area_tolerance=AREA_TOL, seam_policy="ring_split")
    arc = next(i for i, p in enumerate(ring_split.paths) if type(p.curve) is EllipticArc)
    from dataclasses import replace
    broken = replace(ring_split, paths=tuple(replace(p, decomposition=True) if i == arc else p
                                             for i, p in enumerate(ring_split.paths)))
    with pytest.raises(AssertionError):
        _assert_equivalent(baseline, broken)
