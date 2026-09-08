"""Bounded orthogonal trim proof over owner-certified whole-curve tubes.

No samples certify topology. The caller has already established actual shared
Vertices, strict axial/circular monotonicity and disjoint derivative cones at
orthogonal corners. Nonadjacent tubes must be separated. Seeds are chosen from
open cells outside every endpoint uncertainty band, making ray counts exact.
"""

from .cylinder_charts import _Refusal, _hull, _q


def _scales(geometry):
    return (abs(geometry.radius * _q(geometry.surface.sweep_angle)), abs(geometry.height))


def _box(row):
    return tuple(_hull((row['first'][axis], row['last'][axis])) for axis in (0, 1))


def _separated(a, b, geometry, proof):
    proof.step('pair_tests', proof.policy.max_pair_tests)
    for left, right, scale in zip(_box(a), _box(b), _scales(geometry)):
        gap = max(right.lo - left.hi, left.lo - right.hi)
        if gap > geometry.parameter and gap * scale > 2 * geometry.tau:
            return True
    return False


def certify_loop(rows, identifiers, index, geometry, proof):
    """Certify a simple orthogonal loop, including concave corners."""
    proof.cancel('patch orthogonal loop')
    proof.counts['orthogonal_loops'] = proof.counts.get('orthogonal_loops', 0) + 1
    if len(rows) < 4:
        raise _Refusal('patch_orthogonal_loop_size')
    for i, row in enumerate(rows):
        following = rows[(i + 1) % len(rows)]
        # Equal-axis forward subdivisions remain monotone; reversal overlaps.
        if row['axis'] == following['axis'] and row['sign'] != following['sign']:
            raise _Refusal('patch_adjacent_carrier_reversal')
        for j in range(i + 1, len(rows)):
            if j == i + 1 or (i == 0 and j == len(rows) - 1):
                continue
            if not _separated(row, rows[j], geometry, proof):
                raise _Refusal('patch_nonadjacent_carriers_touch')
    area = proof.i(0)
    for row in rows:
        a, b = row['first'], row['last']
        area = proof.add(area, proof.sub(proof.mul(a[0], b[1]), proof.mul(a[1], b[0])))
    if area.lo > 0:
        winding = 1
    elif area.hi < 0:
        winding = -1
    else:
        raise _Refusal('patch_loop_orientation_ambiguous')
    bounds = []
    for axis in (0, 1):
        boxes = [endpoint[axis] for row in rows for endpoint in (row['first'], row['last'])]
        bounds.extend((proof.i(min(b.lo for b in boxes), min(b.hi for b in boxes)),
                       proof.i(max(b.lo for b in boxes), max(b.hi for b in boxes))))
    return dict(index=index, coedges=tuple(identifiers), bounds=tuple(bounds),
                winding=winding, rows=tuple(rows), orthogonal=True)


def _inside(seed, rows, proof):
    """Ray parity on a cell disjoint from every curve endpoint enclosure."""
    crossings = 0
    x, y = seed
    for row in rows:
        proof.step('pair_tests', proof.policy.max_pair_tests)
        if row['axis'] != 1:
            continue
        a, b = row['first'][1], row['last'][1]
        low, high = (a, b) if row['sign'] == 1 else (b, a)
        if y < low.lo or y > high.hi:
            continue
        if not low.hi < y < high.lo:
            raise _Refusal('patch_seed_ray_ambiguous')
        band = row['band']
        if x < band.lo:
            crossings += 1
        elif x <= band.hi:
            raise _Refusal('patch_seed_on_carrier')
    return crossings % 2 == 1


def _stations(rows, axis, geometry, proof):
    boxes = sorted((endpoint[axis] for row in rows for endpoint in (row['first'], row['last'])),
                   key=lambda box: (box.lo, box.hi))
    merged = []
    for box in boxes:
        proof.step('pair_tests', proof.policy.max_pair_tests)
        if merged and box.lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], box.hi))
        else:
            merged.append((box.lo, box.hi))
    scale = _scales(geometry)[axis]
    # Each midpoint is strictly separated from the adjacent uncertainty bands.
    return tuple((a[1] + b[0]) / 2 for a, b in zip(merged, merged[1:])
                 if b[0] - a[1] > 2 * geometry.parameter
                 and (b[0] - a[1]) * scale > 4 * geometry.tau)


def certify_material(loops, geometry, proof):
    """Separate all loops and certify material/void witnesses, without clipping."""
    proof.cancel('patch orthogonal material')
    outer = loops[0]
    if any(b.lo < -geometry.parameter or b.hi > 1 + geometry.parameter for b in outer['bounds']):
        raise _Refusal('patch_material_outside_support')
    if any(hole['winding'] != -outer['winding'] for hole in loops[1:]):
        raise _Refusal('patch_hole_winding')
    for i, left in enumerate(loops):
        for right in loops[i + 1:]:
            for a in left['rows']:
                for b in right['rows']:
                    if not _separated(a, b, geometry, proof):
                        raise _Refusal('patch_hole_clearance')
    rows = tuple(row for loop in loops for row in loop['rows'])
    xs, ys = (_stations(rows, axis, geometry, proof) for axis in (0, 1))
    seeds = {}
    for x in xs:
        for y in ys:
            proof.step('pair_tests', proof.policy.max_pair_tests)
            seed = (x, y)
            inside = tuple(_inside(seed, loop['rows'], proof) for loop in loops)
            if sum(inside[1:]) > 1:
                raise _Refusal('patch_nested_holes')
            if inside[0] and not any(inside[1:]):
                seeds.setdefault(0, seed)
            for i in range(1, len(loops)):
                if inside[i]:
                    if not inside[0]:
                        raise _Refusal('patch_hole_outside_material')
                    seeds.setdefault(i, seed)
            # All boundary tubes are disjoint and loops are simple. A hole
            # witness inside the outer loop plus an outer witness outside it
            # excludes both disjointness and reverse nesting of their interiors.
            if len(seeds) == len(loops):
                for i, loop in enumerate(loops):
                    loop['seed'] = seeds[i]
                return loops
    raise _Refusal('patch_material_seed_unresolved')
