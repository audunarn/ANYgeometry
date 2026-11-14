"""Optional exact plane x NURBS pre-pass (proposal 10, narrow slice).

Cell certificates are unit-tested on hand grids so that flipping a sign test or a strictness comparison is
killed; the five acceptance sections must decide with the stated cell counts; the apex-tangent plane must be
DECLINED (never decided); decided sections are compared with the unchanged atlas route.
"""
import dataclasses
import random
from fractions import Fraction as F
from types import SimpleNamespace

import pytest

from anygeometry.nurbs import NurbsSurface
from anygeometry.tolerance import TolerancePolicy
from anygeometry.nurbs_surface_intersection import IntersectionStatus, query_nurbs_surface_intersection
from anygeometry.nurbs_surface_intersection._plane_prepass import (
    PrepassDecline, PrepassStatus, _Sturm, _Decline, _as_plane, _power, excludes_cell, monotone_direction,
    plane_nurbs_prepass)

K1 = (0., 0., 1., 1.)
K2 = (0., 0., 0., 1., 1., 1.)
K3 = (0.,) * 4 + (1.,) * 4
KH = (0., 0., 0., 0., .5, 1., 1., 1., 1.)


def plane(c):
    return NurbsSurface(1, 1, K1, K1, tuple(tuple((float(i), float(j), c) for j in range(2)) for i in range(2)))


def bump(h=0.8, s=1.0):
    z = ((0, 0, 0), (0, h, 0), (0, 0, 0))
    return NurbsSurface(2, 2, K2, K2, tuple(tuple((s * i / 2, s * j / 2, z[i][j]) for j in range(3)) for i in range(3)))


def humps(row, scale=1.0):
    z = [[0] * 5, row, [0] * 5]
    return NurbsSurface(2, 3, K2, KH, tuple(tuple((i / 2, j / 4, scale * z[i][j]) for j in range(5)) for i in range(3)))


def grid(rows):
    return tuple(tuple((F(x),) for x in row) for row in rows)


# ---------------------------------------------------------------- cell certificates (mutation targets)
def test_exclusion_needs_strict_sign():
    assert excludes_cell([F(1), F(2)]) and excludes_cell([F(-1), F(-3)])
    assert not excludes_cell([F(0), F(1)])            # touching: never excluded
    assert not excludes_cell([F(0), F(-1)])
    assert not excludes_cell([F(-1), F(1)])


def test_monotone_direction_strictness_and_sign():
    # F = u*v: F_v = u and F_u = v each have a zero coefficient, mean direction touches 0: refused
    assert monotone_direction(grid(((0, 0), (0, 1)))) is None
    # F = u - 1/2: t = (0, -1), strictly one-signed along (0, 1)
    assert monotone_direction(grid(((F(-1, 2), F(-1, 2)), (F(1, 2), F(1, 2))))) == (0, 1)
    # F = 1/2 - u: the opposite strict sign is accepted too
    assert monotone_direction(grid(((F(1, 2), F(1, 2)), (F(-1, 2), F(-1, 2))))) == (0, 1)
    # F = v - 1/2: t = (1, 0)
    assert monotone_direction(grid(((F(-1, 2), F(1, 2)), (F(-1, 2), F(1, 2))))) == (1, 0)
    # biquadratic bowl: the gradient vanishes inside, never certified
    bowl = grid(((F(1, 2), F(-1, 2), F(1, 2)), (F(-1, 2), F(-3, 2), F(-1, 2)), (F(1, 2), F(-1, 2), F(1, 2))))
    assert monotone_direction(bowl) is None


def test_mean_direction_is_strict_and_used():
    # F = u*v + u (bilinear): F_u = v + 1 > 0 strictly -> an axis direction already works
    assert monotone_direction(grid(((0, 0), (1, 2)))) == (0, 1)
    # F = u + v - 3/4 * u*v*0 : F_u = F_v = 1; (1,0) is returned first (axis order is part of the contract)
    assert monotone_direction(grid(((-1, 0), (0, 1)))) == (1, 0)
    # only the mixed (mean) direction is strict: F_v = u - 1/2 changes sign, F_u = v - 1/2 changes sign,
    # F = (u - 1/2)(v - 1/2) has a saddle critical point at (1/2, 1/2) -> refused
    assert monotone_direction(grid(((F(1, 4), F(-1, 4)), (F(-1, 4), F(1, 4))))) is None


# ---------------------------------------------------------------- exact univariate helpers
def test_sturm_refuses_multiple_root_and_endpoint_root():
    with pytest.raises(_Decline) as e:
        _Sturm(_power([F(1, 4), F(-1, 4), F(1, 4)]))                 # (t - 1/2)^2 / ... double root
    assert e.value.code is PrepassDecline.MULTIPLE_EDGE_ROOT
    s = _Sturm(_power([F(-1, 2), F(1, 2)]))                          # t - 1/2, simple root at 1/2
    assert s.count(F(0), F(1)) == 1 and s.count(F(0), F(1, 4)) == 0
    with pytest.raises(_Decline) as e:
        s.count(F(0), F(1, 2))                                       # root exactly at a segment end
    assert e.value.code is PrepassDecline.CORNER_OR_DYADIC_ROOT
    with pytest.raises(_Decline):
        _Sturm([F(0)])


# ---------------------------------------------------------------- acceptance sections
CASES = (
    ("bump z=.1", lambda: (plane(.1), bump()), PrepassStatus.DECIDED, (True,), 37),
    ("2-hump z=.03", lambda: (plane(.03), humps([0, 1, 0, 1, 0], .5)), PrepassStatus.DECIDED, (True,), 74),
    ("2-hump z=.06", lambda: (plane(.06), humps([0, 1, 0, 1, 0], .5)), PrepassStatus.DECIDED, (True,), 58),
    ("2 real humps z=.1", lambda: (plane(.1), humps([0, 1.2, -1.5, 1.2, 0])), PrepassStatus.DECIDED, (True, True), 58),
    ("disjoint z=.5", lambda: (plane(.5), bump()), PrepassStatus.DISJOINT, (), 5),
)


@pytest.mark.parametrize("name,make,status,closed,cells", CASES, ids=[c[0] for c in CASES])
def test_acceptance_sections_decided(name, make, status, closed, cells):
    a, b = make()
    result = plane_nurbs_prepass(a, b)
    assert result.status is status and result.reason is None
    assert sorted(c.closed for c in result.components) == sorted(closed)
    assert result.cells_tested == cells
    assert plane_nurbs_prepass(b, a).components == result.components      # operand order is irrelevant


def test_apex_tangent_is_declined_not_decided():
    result = plane_nurbs_prepass(plane(.2), bump())
    assert result.status is PrepassStatus.DECLINED and result.reason is PrepassDecline.DEPTH_CAP
    assert result.components == () and result.max_depth == 10      # the documented depth cap, not deeper


def test_declines_outside_the_narrow_class():
    assert plane_nurbs_prepass(bump(), bump(.5)).reason is PrepassDecline.NOT_PLANE_NURBS
    assert plane_nurbs_prepass(plane(.1), bump(.8, 2.0)).reason is PrepassDecline.PLANE_FOOTPRINT
    trimmed = dataclasses.replace(bump(), u_range=(0.25, 1.0))
    assert plane_nurbs_prepass(plane(.1), trimmed).reason is PrepassDecline.RANGES_NOT_FULL
    # coplanar: F is identically zero, never a decision
    assert plane_nurbs_prepass(plane(0.0), bump(0.0)).status is PrepassStatus.DECLINED


# ---------------------------------------------------------------- the opt-in hook and the unchanged route
def _query(a, b, **extra):
    q = SimpleNamespace(max_boxes_per_pair=3000, **extra)
    return query_nurbs_surface_intersection(a, b, tolerances=TolerancePolicy(), qualification=q)


def test_hook_only_decides_disjoint_and_is_off_by_default():
    a, b = plane(.5), bump()
    off, on = _query(a, b), _query(a, b, plane_prepass=True)
    assert off.status is on.status is IntersectionStatus.DISJOINT and off.cover_complete and on.cover_complete
    assert off.algorithm == "generic-cover" and on.algorithm == "plane-prepass"
    assert on.stats.boxes_visited == 0 < off.stats.boxes_visited
    assert _query(a, b, plane_prepass=1).algorithm == "generic-cover"      # only an exact True enables it


def test_hook_falls_through_unchanged_for_sections_and_tangent():
    a, b = plane(.1), bump()
    off, on = _query(a, b), _query(a, b, plane_prepass=True)
    assert on.status is off.status and on.algorithm == off.algorithm == "generic-atlas"
    assert on.components == off.components and on.stats.boxes_visited == off.stats.boxes_visited
    capped = SimpleNamespace(max_boxes_per_pair=20, plane_prepass=True)
    tangent = query_nurbs_surface_intersection(plane(.2), bump(), tolerances=TolerancePolicy(), qualification=capped)
    assert tangent.status is IntersectionStatus.UNCLASSIFIED and tangent.algorithm == "generic-cover"


def _overlap(box_a, box_b):
    return all(lo1 <= hi2 and lo2 <= hi1 for (lo1, hi1), (lo2, hi2) in zip(box_a, box_b))


def _compare_with_atlas(a, b):
    pre = plane_nurbs_prepass(a, b)
    old = _query(a, b)
    assert old.status is IntersectionStatus.CERTIFIED and old.algorithm == "generic-atlas"
    assert pre.status is PrepassStatus.DECIDED
    assert sorted(c.closed for c in old.components) == sorted(c.closed for c in pre.components)
    free = list(pre.components)
    for comp in old.components:        # one-to-one pairing: the atlas chart box (NURBS operand) meets a pre-pass arc
        hits = [k for k, p in enumerate(free) if p.closed == comp.closed and any(
            _overlap(comp.second_chart, tuple((float(lo), float(hi)) for lo, hi in arc.native_box))
            for arc in p.arcs)]
        assert hits, "an atlas component has no pre-pass counterpart"
        free.pop(hits[0])
    assert not free
    return pre, old


def test_single_bump_matches_atlas_with_fewer_boxes():
    pre, old = _compare_with_atlas(plane(.1), bump())
    assert pre.cells_tested * 2 <= old.stats.boxes_visited


@pytest.mark.slow
@pytest.mark.parametrize("make", [CASES[1][1], CASES[3][1]], ids=["2-hump z=.03", "2 real humps z=.1"])
def test_multi_hump_matches_atlas_with_a_fifth_of_the_boxes(make):
    pre, old = _compare_with_atlas(*make())
    assert pre.cells_tested * 5 <= old.stats.boxes_visited


# ---------------------------------------------------------------- seeded property check (no hypothesis)
def _sampled_signs(a, b, n=33):
    plane_data = _as_plane(a)
    nrm, off = [float(x) for x in plane_data[0]], float(plane_data[1])
    signs = set()
    for i in range(n):
        for j in range(n):
            x, y, z = b.evaluate(i / (n - 1), j / (n - 1))
            value = nrm[0] * x + nrm[1] * y + nrm[2] * z - off
            if abs(value) > 1e-9:
                signs.add(value > 0)
    return signs


def test_random_smooth_surfaces_decide_consistently_or_decline():
    rng = random.Random(20261114)
    decided = 0
    for _ in range(40):
        zs = tuple(tuple(round(rng.uniform(-.6, .6), 3) for _ in range(4)) for _ in range(4))
        surf = NurbsSurface(3, 3, K3, K3, tuple(tuple((i / 3, j / 3, zs[i][j]) for j in range(4)) for i in range(4)))
        c = round(rng.uniform(-.5, .5), 3)
        res = plane_nurbs_prepass(plane(c), surf)
        if res.status is PrepassStatus.DECLINED:
            assert res.reason in (PrepassDecline.DEPTH_CAP, PrepassDecline.CORNER_OR_DYADIC_ROOT,
                                  PrepassDecline.MULTIPLE_EDGE_ROOT, PrepassDecline.ROOT_DEGREE,
                                  PrepassDecline.CELL_CAP)
            continue
        decided += 1
        signs = _sampled_signs(plane(c), surf)
        if res.status is PrepassStatus.DISJOINT:
            assert len(signs) == 1
        else:
            assert res.components and len(signs) == 2          # a decided section implies a sign change
            for comp in res.components:
                assert (comp.roots[0] == comp.roots[-1]) == comp.closed
                assert len(comp.arcs) == len(comp.roots) - 1
    assert decided >= 20                                       # the generator is not vacuous


@pytest.mark.slow
def test_random_surfaces_components_equal_atlas():
    """Random single-hump surfaces (centre control height h, plane height c): loops, disjoint, same as the atlas."""
    rng = random.Random(7)
    closed = disjoint = 0
    for _ in range(6):
        h = round(rng.uniform(.5, 1.4), 3)
        c = round(rng.uniform(-.3, h / 4 * 1.25), 3)               # h/4 is the apex height: beyond it, disjoint
        a, b = plane(c), bump(h)
        pre = plane_nurbs_prepass(a, b)
        if pre.status is PrepassStatus.DECLINED:
            continue
        old = _query(a, b)
        if old.status is IntersectionStatus.UNCLASSIFIED:
            continue
        if pre.status is PrepassStatus.DISJOINT:
            assert old.status is IntersectionStatus.DISJOINT
            disjoint += 1
        else:
            assert old.status is IntersectionStatus.CERTIFIED
            assert sorted(x.closed for x in old.components) == sorted(x.closed for x in pre.components)
            closed += 1
    assert closed + disjoint >= 4
