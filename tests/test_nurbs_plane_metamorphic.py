"""Light metamorphic backbone for the plane x NURBS section kernel (KERNEL-STRATEGY-REVIEW proposal 6).

No oracle: every relation compares the kernel with itself on a transformed input.
Seeded ``random`` only; tiny 3x3 degree-2 patches; public entry ``plane_nurbs_section``
(the budget relation sets the audit's ``CURRENT_WORK`` exactly as ``plan_plane_split`` does).
"""
from __future__ import annotations

import random
from fractions import Fraction

import pytest

from anygeometry import GeometryError, Plane
from anygeometry.nurbs import NurbsSurface
from anygeometry.plane_nurbs import CURRENT_WORK, PlaneNurbsNotCertified, plane_nurbs_section
from anygeometry.section_audit import Work

KN3 = (0., 0., 0., 1., 1., 1.)
SEEDS = range(12)


# ------------------------------------------------------------------ case builders (dyadic data)

def dyadic(rng, lo, hi, den):
    return rng.randint(lo * den, hi * den) / den


def make_case(seed):
    """3x3 patch with dyadic controls (a graph-like wavy sheet) and a generic dyadic plane."""
    rng = random.Random(seed)
    net = tuple(tuple((i / 2 + dyadic(rng, -1, 1, 16) / 4, j / 2 + dyadic(rng, -1, 1, 16) / 4,
                       dyadic(rng, -1, 1, 8)) for j in range(3)) for i in range(3))
    tilt = (dyadic(rng, -1, 1, 8) / 2, dyadic(rng, -1, 1, 8) / 2)
    level = dyadic(rng, -1, 1, 64) / 2 + 1 / 128
    return net, (level, tilt)


def plane_of(level_tilt):
    level, (tx, ty) = level_tilt
    return Plane((0., 0., level), (1., 0., tx), (0., 1., ty))


def surface_of(net):
    return NurbsSurface(2, 2, KN3, KN3, net)


# ------------------------------------------------------------------ outcome signature

def outcome(plane, surface, work=None, detail=True):
    """('ok', signature) or ('refused', exception class name); anything else escapes (a failure)."""
    token = CURRENT_WORK.set(work) if work is not None else None
    try:
        section = plane_nurbs_section(plane, surface)
    except GeometryError as error:
        return ('refused', type(error).__name__)
    finally:
        if token is not None:
            CURRENT_WORK.reset(token)
    paths = []
    for path in section.paths:
        if path.closed:
            paths.append((True, ()))
            continue
        ends = sorted(tuple(round(c, 9) for c in section.chart_point(i, surface))
                      for i in (path.roots[0], path.roots[-1]))
        paths.append((False, tuple(ends), len(path.arcs) if detail else 0))
    return ('ok', section.coincident, tuple(sorted(paths)))


# ------------------------------------------------------------------ exact rigid maps

def apply_map(surface, plane, f_point, f_vector):
    net = tuple(tuple(f_point(tuple(float(c) for c in p)) for p in row) for row in surface.controls)
    return (NurbsSurface(2, 2, KN3, KN3, net),
            Plane(f_point(tuple(float(c) for c in plane.origin)), f_vector(tuple(float(c) for c in plane.u_vector)),
                  f_vector(tuple(float(c) for c in plane.v_vector))))


def rigid_cases():
    """Generic sections plus contact/coincident/missing planes (those refuse or degenerate)."""
    cases = [(plane_of(make_case(s)[1]), surface_of(make_case(s)[0])) for s in SEEDS]
    rng = random.Random(7)
    return cases + [fuzz_case(rng) for _ in range(16)]


RIGID = {
    'permute': (lambda p: (p[1], p[2], p[0]), lambda v: (v[1], v[2], v[0])),
    'swap_xy': (lambda p: (p[1], p[0], p[2]), lambda v: (v[1], v[0], v[2])),
    'flip_z': (lambda p: (p[0], p[1], -p[2]), lambda v: (v[0], v[1], -v[2])),
    'flip_x': (lambda p: (-p[0], p[1], p[2]), lambda v: (-v[0], v[1], v[2])),
    'scale_8': (lambda p: tuple(8. * c for c in p), lambda v: tuple(8. * c for c in v)),
    'scale_eighth': (lambda p: tuple(c / 8. for c in p), lambda v: tuple(c / 8. for c in v)),
    'shift': (lambda p: (p[0] + 4., p[1] - 2.5, p[2] + 8.), lambda v: v),
}


def test_operand_order_swap():
    """Swapping the plane's two parameter vectors is the same plane (reversed normal)."""
    for index, (plane, surface) in enumerate(rigid_cases()):
        swapped = Plane(plane.origin, plane.v_vector, plane.u_vector)
        assert outcome(plane, surface) == outcome(swapped, surface), index


@pytest.mark.parametrize('name', sorted(RIGID))
def test_exact_rigid_change_keeps_topology_and_refusal_class(name):
    f_point, f_vector = RIGID[name]
    kinds = set()
    for index, (plane, surface) in enumerate(rigid_cases()):
        base = outcome(plane, surface)
        moved_surface, moved_plane = apply_map(surface, plane, f_point, f_vector)
        assert outcome(moved_plane, moved_surface) == base, (name, index)
        kinds.add(base[0])
    assert kinds == {'ok', 'refused'}, kinds


# ------------------------------------------------------------------ same surface, finer representation

def insert_mid(net, knots_u, knots_v, direction):
    """Insert the knot 1/2 once (Boehm, exact in dyadic data) in direction 'u' or 'v'."""
    rows = [list(r) for r in net]          # rows run along v
    if direction == 'u':
        rows = [list(col) for col in zip(*rows)]

    # single Bezier span of degree 2: P'_0 = P0, P'_1 = (P0+P1)/2, P'_2 = (P1+P2)/2, P'_3 = P2
    out = []
    for line in rows:
        p0, p1, p2 = line
        mid = lambda a, b: tuple((x + y) / 2 for x, y in zip(a, b))
        out.append([p0, mid(p0, p1), mid(p1, p2), p2])
    new_knots = (0., 0., 0., .5, 1., 1., 1.)
    if direction == 'u':
        out = [list(r) for r in zip(*out)]
        return tuple(tuple(r) for r in out), new_knots, knots_v
    return tuple(tuple(r) for r in out), knots_u, new_knots


def elevate_u(net):
    """Degree 2 -> 3 in u on one Bezier span: Q_i = i/3 P_{i-1} + (1 - i/3) P_i."""
    cols = []
    for j in range(3):
        p = [net[i][j] for i in range(3)]
        q = [p[0]]
        for i in (1, 2):
            q.append(tuple(i / 3 * a + (1 - i / 3) * b for a, b in zip(p[i - 1], p[i])))
        q.append(p[2])
        cols.append(q)
    return tuple(tuple(cols[j][i] for j in range(3)) for i in range(4))


def coarse(plane, surface):
    kind = outcome(plane, surface, detail=False)
    if kind[0] != 'ok':
        return kind
    # same certified topology: path count, closedness and end points (chart positions, 1e-6)
    paths = tuple(sorted((c, tuple(tuple(round(x, 6) for x in e) for e in ends) if not c else ())
                         for c, *rest in kind[2] for ends in [rest[0] if rest else ()]))
    return ('ok', kind[1], paths)


def test_knot_insertion_keeps_certified_topology():
    oks = 0
    for seed in SEEDS:
        net, lt = make_case(seed)
        plane = plane_of(lt)
        base = coarse(plane, surface_of(net))
        n1, ku, kv = insert_mid(net, KN3, KN3, 'u')
        refined = NurbsSurface(2, 2, ku, kv, n1)
        assert coarse(plane, refined) == base, ('u', seed)
        n3, ku3, kv3 = insert_mid(net, KN3, KN3, 'v')
        assert coarse(plane, NurbsSurface(2, 2, ku3, kv3, n3)) == base, ('v', seed)
        oks += base[0] == 'ok'
    assert oks >= 3


def test_degree_elevation_keeps_certified_topology():
    oks = 0
    for seed in SEEDS:
        net, lt = make_case(seed)
        plane = plane_of(lt)
        base = coarse(plane, surface_of(net))
        elevated = NurbsSurface(3, 2, (0., 0., 0., 0., 1., 1., 1., 1.), KN3, elevate_u(net))
        assert coarse(plane, elevated) == base, seed
        oks += base[0] == 'ok'
    assert oks >= 3


# ------------------------------------------------------------------ budget monotonicity

def test_budget_monotonicity():
    certified = 0
    for seed in SEEDS:
        net, lt = make_case(seed)
        plane, surface = plane_of(lt), surface_of(net)
        reference = outcome(plane, surface)
        cheap = outcome(plane, surface, Work(1, 32))
        if cheap[0] == 'ok':
            assert cheap == reference
        else:
            assert reference[0] != 'ok' or cheap == ('refused', 'PlaneNurbsNotCertified')
        previous = None
        for n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
            current = outcome(plane, surface, Work(n, 32))
            if previous is not None and previous[0] == 'ok':
                assert current == previous, (seed, n)      # certified at N: identical at 2N
            if current[0] != 'ok':
                if reference[0] == 'ok':                   # a refusal below the need is the typed class
                    assert current == ('refused', 'PlaneNurbsNotCertified'), (seed, n, current)
            previous = current
        assert previous == reference or reference[0] != 'ok'
        certified += reference[0] == 'ok'
    assert certified >= 3


# ------------------------------------------------------------------ refusal fuzzing

def fuzz_case(rng):
    net = tuple(tuple((i / 2 + dyadic(rng, -2, 2, 4), j / 2 + dyadic(rng, -2, 2, 4), dyadic(rng, -2, 2, 4))
                      for j in range(3)) for i in range(3))
    kind = rng.randrange(5)
    if kind == 0:                      # plane exactly through a control level (contacts, corners)
        level = net[rng.randrange(3)][rng.randrange(3)][2]
        plane = Plane((0., 0., level), (1., 0., 0.), (0., 1., 0.))
    elif kind == 1:                    # plane containing the patch (coincident) when flat
        net = tuple(tuple((x, y, 0.) for x, y, _ in row) for row in net)
        plane = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    elif kind == 2:                    # vertical plane on a control column
        x = net[rng.randrange(3)][0][0]
        plane = Plane((x, 0., 0.), (0., 1., 0.), (0., 0., 1.))
    elif kind == 3:                    # plane missing the patch
        plane = Plane((0., 0., 100.), (1., 0., 0.), (0., 1., 0.))
    else:
        plane = Plane((0., 0., dyadic(rng, -2, 2, 8)), (1., dyadic(rng, -1, 1, 4), 0.),
                      (0., 1., dyadic(rng, -1, 1, 4)))
    return plane, surface_of(net)


def test_every_exception_is_a_typed_refusal():
    rng = random.Random(20261113)
    seen = set()
    for _ in range(60):
        plane, surface = fuzz_case(rng)
        try:
            plane_nurbs_section(plane, surface)
            seen.add('ok')
        except GeometryError as error:
            seen.add(type(error).__name__)
            if isinstance(error, PlaneNurbsNotCertified):
                assert str(error)
        # any non-GeometryError exception propagates and fails the test
    assert 'ok' in seen
