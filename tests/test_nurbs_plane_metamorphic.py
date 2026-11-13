"""Light metamorphic backbone for the plane x NURBS section kernel (KERNEL-STRATEGY-REVIEW proposal 6).

No oracle: every relation compares the kernel with itself on a transformed input.
Seeded ``random`` only; tiny 3x3 degree-2 patches; public entry ``plane_nurbs_section``
(the budget relation goes through ``plan_plane_split`` and its public ``qualification`` policy).
"""
from __future__ import annotations

import random
from fractions import Fraction

import pytest

from anygeometry import GeometryError, Plane
from anygeometry.nurbs import NurbsSurface
from anygeometry import GeometryModel, apply_plane_split, plan_plane_split
from anygeometry.nurbs import CurveOnSurface, NurbsCurve
from anygeometry.plane_nurbs import PlaneNurbsNotCertified, plane_nurbs_section
from anygeometry.predicates import IntersectionQualificationPolicy
from anygeometry.serialization import to_dict

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

def outcome(plane, surface, detail=True):
    """('ok', signature) or ('refused', exception class name); anything else escapes (a failure)."""
    try:
        section = plane_nurbs_section(plane, surface)
    except GeometryError as error:
        return ('refused', type(error).__name__)
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


def elevation_case(seed):
    """Dyadic controls whose degree elevation is exactly representable in doubles.

    Q_i = (i P_{i-1} + (3 - i) P_i) / 3 is exact when the three u-controls of every column
    are congruent modulo 3 in units of 1/64, per coordinate (so no thirds appear)."""
    rng = random.Random(1000 + seed)
    net = [[None] * 3 for _ in range(3)]
    for j in range(3):
        coords = []
        for axis in range(3):
            target = rng.randrange(3)
            column = []
            for i in range(3):
                grid = (32 * i, 32 * j, 0)[axis]
                column.append(grid + rng.choice([r for r in range(-6, 7) if (grid + r) % 3 == target]) * (1 + 3 * (axis == 2)))
            coords.append(column)
        for i in range(3):
            net[i][j] = tuple(coords[axis][i] / 64 for axis in range(3))
    return tuple(tuple(r) for r in net)


def elevate_u(net):
    """Degree 2 -> 3 in u on one Bezier span, in exact rationals; asserted exact as doubles."""
    out = [[None] * 3 for _ in range(4)]
    for j in range(3):
        p = [tuple(Fraction(c) for c in net[i][j]) for i in range(3)]
        q = [p[0]] + [tuple(Fraction(i, 3) * a + (1 - Fraction(i, 3)) * b for a, b in zip(p[i - 1], p[i]))
                      for i in (1, 2)] + [p[2]]
        for i in range(4):
            floats = tuple(float(c) for c in q[i])
            assert tuple(Fraction(c) for c in floats) == q[i], 'elevated control is not a double'
            out[i][j] = floats
    return tuple(tuple(r) for r in out)


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
        net = elevation_case(seed)
        plane = plane_of(make_case(seed)[1])
        base = coarse(plane, surface_of(net))
        elevated = NurbsSurface(3, 2, (0., 0., 0., 0., 1., 1., 1., 1.), KN3, elevate_u(net))
        assert coarse(plane, elevated) == base, seed
        oks += base[0] == 'ok'
    assert oks >= 3


# ------------------------------------------------------------------ budget monotonicity (public policy)

RIDGE = NurbsSurface(2, 1, KN3, (0., 0., 1., 1.),
                     tuple(((i / 2, 0., z), (i / 2, 1., z)) for i, z in enumerate((.25, -.25, .25))))


def ridge_model():
    model = GeometryModel()
    points = [(0., 0.), (1., 0.), (1., 1.), (0., 1.)]
    vertices = [model.add_point(*RIDGE.evaluate(*p)) for p in points]
    edges = [model.add_curve(vertices[i], vertices[(i + 1) % 4], CurveOnSurface(
        RIDGE, NurbsCurve(1, (0., 0., 1., 1.), (p, points[(i + 1) % 4])))) for i, p in enumerate(points)]
    return model, model.add_face(edges, surface=RIDGE)


def split_under(level, boxes):
    """('ok', published model) or ('refused', error) for one plan+apply under a box budget."""
    model, face = ridge_model()
    plane = Plane((0., 0., level), (1., 0., 0.), (0., 1., 0.))
    try:
        plan = plan_plane_split(model, [face], plane,
                                qualification=IntersectionQualificationPolicy(max_boxes_per_pair=boxes))
    except GeometryError as error:
        return 'refused', error
    apply_plane_split(model, plan, policy='imprint')
    return 'ok', to_dict(model)


CAPS = [4 ** k for k in range(0, 7)] + [200_000]


def published(result):
    kind, payload = result
    if kind == 'ok':
        payload = {key: value for key, value in payload.items() if key not in ('model_id', 'checksum')}
    return kind, payload


@pytest.mark.parametrize('level', (.0625, -.125))
def test_budget_monotonicity(level):
    results = [split_under(level, cap) for cap in CAPS]
    reference = published(results[-1])
    assert reference[0] == 'ok'
    first = next(i for i, r in enumerate(results) if r[0] == 'ok')
    assert first > 0, 'the cheapest cap must demonstrate a budget refusal'
    for cap, result in zip(CAPS[first:], results[first:]):
        assert published(result) == reference, cap            # certified at N: identical at larger caps
    kind, error = results[0]                                  # the cheapest cap: the typed budget refusal
    assert type(error) is PlaneNurbsNotCertified and 'budget' in str(error), error
    assert error.budget_limit == CAPS[0] and error.budget_denied >= 1


@pytest.mark.xfail(strict=True, reason='FINDING (gap audit G1): between the cheapest cap and the certifying cap '
                   'the refusal can be an untyped GeometryError (not certified disjoint / invalid topology)')
def test_every_budget_refusal_is_the_typed_budget_class():
    for cap in CAPS:
        kind, error = split_under(.0625, cap)
        if kind == 'refused':
            assert type(error) is PlaneNurbsNotCertified and 'budget' in str(error), (cap, error)


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
            assert type(error) is PlaneNurbsNotCertified, (type(error).__name__, str(error))
            assert str(error)
            seen.add('refused')
        # any non-GeometryError exception propagates and fails the test
    assert seen == {'ok', 'refused'}, seen
