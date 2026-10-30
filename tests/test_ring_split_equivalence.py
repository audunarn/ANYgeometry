"""RING-SPLIT RS-2: equivalence of the ring_split and baseline arrangements of one face (production brief RS-2).

Both policies cut the same material: the non-decomposition edges are equal as point sets (pieces may merge),
the cell areas sum to the same material area, every edge ring_split adds is a decomposition edge, and dissolving
the decomposition edges gives the same regions. The model-scale oracle (sector60, r7, face200, centre, the six
caps, leg1, plan and apply) is the harness redesign/rs_oracle.py outside the repository.
"""
import collections
import math
from fractions import Fraction

import numpy as np
import pytest

from anygeometry import EllipticArc
from anygeometry.arrangement_geometry import LinePath, point_parameters
from anygeometry.material_arrangement import ArrangementPath, ArrangementPoint, MaterialDomain, arrange_material
from anygeometry.surfaces import Plane
from anygeometry import planar_stations as stations

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


def _region_components(arrangement):
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
    components = collections.defaultdict(list)
    for c in range(len(arrangement.cells)):
        components[find(c)].append(c)
    return list(components.values())


def _regions(arrangement):
    return sorted(sum(_cell_area(arrangement, arrangement.cells[c]) for c in component)
                  for component in _region_components(arrangement))


def _region_boundaries(arrangement):
    """Physical boundary point sets and area of each dissolved region, retaining owner identity."""
    regions = []
    for component in _region_components(arrangement):
        groups = collections.defaultdict(list)
        for c in component:
            cell = arrangement.cells[c]
            for loop in (cell.outer, *cell.holes):
                for i, _forward in loop:
                    path = arrangement.paths[i]
                    if not path.decomposition:
                        groups[(type(path.curve).__name__, path.owners, path.source_edge)].append(path.curve)
        area = sum(_cell_area(arrangement, arrangement.cells[c]) for c in component)
        regions.append((area, groups))
    return regions


def _covered(curve, others):
    """Complete interval coverage, rather than a finite sampling of the physical point set.

    Straight curves use exact station projections. Arc pieces in these tests retain their input
    ellipse frame; their angle intervals are compared modulo a turn, with a conservative speed bound.
    Unsupported reparameterisations fail closed.
    """
    intervals = []
    if type(curve) is LinePath:
        segment = stations.segment_of(curve)
        direction = tuple(b - a for a, b in zip(*segment))
        length2 = segment.length2()
        if length2 == 0:
            return any(point_parameters(other, curve.evaluate(0.), tolerance=TOL) for other in others)
        for other in others:
            if type(other) is not LinePath:
                continue
            parameters = []
            for point in stations.segment_of(other):
                t = sum((x - a) * d for x, a, d in zip(point, segment[0], direction)) / length2
                if stations.distance2(point, stations.at(segment, t)) > Fraction(TOL) ** 2:
                    break
                parameters.append(t)
            if len(parameters) == 2:
                a, b = max(Fraction(0), min(parameters)), min(Fraction(1), max(parameters))
                if a <= b:
                    intervals.append((a, b))
        lo, hi = Fraction(0), Fraction(1)
        gap_small = lambda gap: gap * gap * length2 <= Fraction(TOL) ** 2
    elif type(curve) is EllipticArc:
        lo, hi = sorted((curve.start_angle, curve.start_angle + curve.sweep_angle))
        speed = float(np.linalg.norm(curve.u_vector) + np.linalg.norm(curve.v_vector))
        assert math.isfinite(speed) and speed > 0
        for other in others:
            if type(other) is not EllipticArc or not all(np.array_equal(getattr(curve, name), getattr(other, name))
                    for name in ("center", "u_vector", "v_vector")):
                continue
            a, b = sorted((other.start_angle, other.start_angle + other.sweep_angle))
            for k in range(math.floor((lo - b) / (2 * math.pi)), math.ceil((hi - a) / (2 * math.pi)) + 1):
                left, right = max(lo, a + k * 2 * math.pi), min(hi, b + k * 2 * math.pi)
                if left <= right:
                    intervals.append((left, right))
        gap_small = lambda gap: gap * speed <= TOL
    else:
        raise AssertionError("unsupported curve in region point-set oracle")
    cursor = lo
    for a, b in sorted(intervals):
        if a > cursor and not gap_small(a - cursor):
            return False
        cursor = max(cursor, b)
    return cursor >= hi or gap_small(hi - cursor)


def _same_point_sets(first, second):
    return (set(first) == set(second) and all(
        all(_covered(curve, second[key]) for curve in first[key])
        and all(_covered(curve, first[key]) for curve in second[key]) for key in first))


def _length(curve):
    if type(curve) is EllipticArc:
        return abs(curve.sweep_angle) * float(np.linalg.norm(curve.u_vector))
    return float(np.linalg.norm(np.asarray(curve.evaluate(1.)) - np.asarray(curve.evaluate(0.))))


def _assert_equivalent(baseline, ring_split, *, traces=(), points=()):
    assert baseline.face_id == ring_split.face_id
    group = lambda p: (type(p.curve).__name__, p.owners, p.source_edge)
    sides = []
    for arrangement in (baseline, ring_split):
        groups = collections.defaultdict(list)
        for p in arrangement.paths:
            if not p.decomposition:
                groups[group(p)].append(p.curve)
        sides.append(groups)
    a, b = sides
    assert _same_point_sets(a, b), "physical edge point sets differ"
    expected_traces = collections.defaultdict(list)
    for path in traces:
        expected_traces[group(path)].append(path.curve)
    for side in sides:
        assert set(expected_traces) <= set(side)
        assert _same_point_sets(expected_traces, {key: side[key] for key in expected_traces})
    for arrangement in (baseline, ring_split):
        for point in points:
            assert any(np.linalg.norm(np.asarray(path.curve.evaluate(t)) - point.position) <= TOL
                       for path in arrangement.paths for t in (0., 1.)), "input point contact was lost"
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
    remaining = _region_boundaries(ring_split)
    for area, boundary in _region_boundaries(baseline):
        matches = [i for i, (other_area, other_boundary) in enumerate(remaining)
                   if abs(area - other_area) <= 1e-9 * max(1., area)
                   and _same_point_sets(boundary, other_boundary)]
        assert matches, "dissolved region boundary or owner identity differs"
        remaining.pop(matches[0])
    assert not remaining
    return ra


@pytest.mark.parametrize("angle", [0.0, 2 * math.pi / 3, 0.6435011087932844])
def test_cap_arrangements_are_equivalent(angle):
    domain, traces = cap(angle, frame=0.0)
    baseline = arrange_material(domain, traces, tolerance=TOL, area_tolerance=AREA_TOL, seam_policy="baseline")
    ring_split = arrange_material(domain, traces, tolerance=TOL, area_tolerance=AREA_TOL, seam_policy="ring_split")
    assert sum(p.decomposition for p in ring_split.paths) == N + 2
    regions = _assert_equivalent(baseline, ring_split, traces=traces)
    assert len(regions) == 2                                 # inside and outside the rim circle
    assert abs(sum(regions) - 400.) <= 1e-9 * 400            # the whole 20 x 20 plate


def test_point_end_and_crossing_variants_are_equivalent():
    domain, traces = cap(skip=(0,))
    points = (ArrangementPoint((5., 0., 0.)),)
    kw = dict(tolerance=TOL, area_tolerance=AREA_TOL, points=points)
    _assert_equivalent(arrange_material(domain, traces, seam_policy="baseline", **kw),
                       arrange_material(domain, traces, seam_policy="ring_split", **kw), traces=traces, points=points)
    domain, traces = cap()
    traces = (*traces, ArrangementPath(LinePath((10., .4, 0.), (5.5, 0., 0.)), owners=(99,)))
    _assert_equivalent(arrange_material(domain, traces, tolerance=TOL, area_tolerance=AREA_TOL, seam_policy="baseline"),
                       arrange_material(domain, traces, tolerance=TOL, area_tolerance=AREA_TOL,
                                        seam_policy="ring_split"), traces=traces)


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


def test_regions_reject_equal_areas_with_different_boundaries():
    from dataclasses import replace
    from anygeometry.material_arrangement import ArrangementCell, MaterialArrangement

    paths, indices, cells = [], {}, []
    for y in (0, 1):
        for x in (0, 1):
            corners = ((x, y, 0), (x + 1, y, 0), (x + 1, y + 1, 0), (x, y + 1, 0))
            loop = []
            for a, b in zip(corners, corners[1:] + corners[:1]):
                key = tuple(sorted((a, b)))
                if key not in indices:
                    i = indices[key] = len(paths)
                    decomposition = a[1] == b[1] == 1
                    paths.append(ArrangementPath(LinePath(*key), owners=() if decomposition else (1,),
                                                 source_edge=None if decomposition else i + 1,
                                                 decomposition=decomposition))
                loop.append((indices[key], a == key[0]))
            cells.append(ArrangementCell(tuple(loop), ()))
    baseline = MaterialArrangement(7, tuple(paths), tuple(cells), 4., support=Plane(
        (0., 0., 0.), (1., 0., 0.), (0., 1., 0.)))
    # Correct dissolution joins left/right columns. Corrupt the cell adjacency so it
    # instead joins bottom/top rows, retaining the same physical edges and all areas.
    broken_cells = []
    for k, cell in enumerate(cells):
        loop = []
        for i, forward in cell.outer:
            if k >= 2 and paths[i].decomposition:
                paths.append(paths[i])
                i = len(paths) - 1
            loop.append((i, forward))
        broken_cells.append(replace(cell, outer=tuple(loop)))
    for pair, x in (((0, 1), 0), ((2, 3), 2)):
        i = len(paths)
        paths.append(ArrangementPath(LinePath((x, 0, 0), (x, 2, 0)), decomposition=True))
        for k in pair:
            broken_cells[k] = replace(broken_cells[k], outer=broken_cells[k].outer + ((i, True), (i, False)))
    broken = replace(baseline, paths=tuple(paths), cells=tuple(broken_cells))
    assert _regions(baseline) == _regions(broken) == [2., 2.]
    with pytest.raises(AssertionError):
        _assert_equivalent(baseline, broken)
