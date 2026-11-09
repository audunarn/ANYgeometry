"""Early exact coincidence / overlap / shared-boundary refusal on the generic route (gap audit 2026-11-09, gap 1).

Positive cases must return UNCLASSIFIED with a typed sub-reason before any atlas box is visited.  A small box
budget keeps the old code (which has no detector) finite; the detector never depends on it.
"""
import dataclasses
import time
from types import SimpleNamespace

import pytest

from anygeometry.nurbs import NurbsSurface
from anygeometry.tolerance import TolerancePolicy
from anygeometry.nurbs_surface_intersection import IntersectionStatus, query_nurbs_surface_intersection

K1 = (0., 0., 1., 1.)
K2 = (0., 0., 0., 1., 1., 1.)
BUDGET = SimpleNamespace(max_boxes_per_pair=100)


def plane(f, n=2, knots=K1):
    return NurbsSurface(1, 1, knots, K1, tuple(tuple(f(i / (n - 1), j) for j in range(2)) for i in range(n)))


def bump(h=0.8):
    z = ((0, 0, 0), (0, h, 0), (0, 0, 0))
    return NurbsSurface(2, 2, K2, K2, tuple(tuple((i / 2, j / 2, z[i][j]) for j in range(3)) for i in range(3)))


def bump_inserted_half_u(h=0.8):
    """The same biquadratic Bezier with u=1/2 inserted (hand-derived: Q0=P0, Q1=(P0+P1)/2, Q2=(P1+P2)/2, Q3=P2)."""
    p = bump(h).controls
    rows = (p[0], tuple(tuple((a + b) / 2 for a, b in zip(x, y)) for x, y in zip(p[0], p[1])),
            tuple(tuple((a + b) / 2 for a, b in zip(x, y)) for x, y in zip(p[1], p[2])), p[2])
    return NurbsSurface(2, 2, (0., 0., 0., .5, 1., 1., 1.), K2, rows)


def query(a, b, qualification=BUDGET):
    started = time.perf_counter()
    result = query_nurbs_surface_intersection(a, b, tolerances=TolerancePolicy(), qualification=qualification)
    return result, time.perf_counter() - started


def flat(u, j):
    return (u, float(j), 0.)


POSITIVE = {
    "identical": (lambda: (bump(), bump()), "coincident"),
    "same_object": (lambda: (lambda a: (a, a))(bump()), "coincident"),
    "knot_inserted_copy": (lambda: (bump(), bump_inserted_half_u()), "coincident"),
    "sub_range_of_same_surface": (lambda: (bump(), dataclasses.replace(bump(), u_range=(.25, .75))), "coincident"),
    "coplanar_different_knots": (lambda: (plane(flat), plane(flat, n=3, knots=(0., 0., .5, 1., 1.))),
                                 "coincident"),
    "coplanar_shifted_overlap": (lambda: (plane(flat), plane(lambda u, j: (u + .5, float(j), 0.))),
                                 "coplanar-overlap"),
    "coplanar_transposed_sub_patch": (lambda: (plane(flat), plane(lambda u, j: (.25 + .5 * j, .25 + .5 * u, 0.))),
                                      "coplanar-overlap"),
    "shared_edge_perpendicular": (lambda: (plane(flat), plane(lambda u, j: (u, 1., float(j)))),
                                  "shared-boundary-curve"),
    "coplanar_edge_touch": (lambda: (plane(flat), plane(lambda u, j: (u + 1., float(j), 0.))),
                            "shared-boundary-curve"),
}


@pytest.mark.parametrize("name", sorted(POSITIVE))
def test_proven_degenerate_pair_is_typed_and_refused_before_the_atlas(name):
    make, expected = POSITIVE[name]
    result, elapsed = query(*make())
    assert result.status is IntersectionStatus.UNCLASSIFIED
    assert result.components == () and not result.cover_complete
    assert getattr(result, "unclassified_reason", None) == expected
    assert result.stats.boxes_visited == 0
    assert elapsed < 2.0


def test_reason_is_a_typed_enum_and_reversed_shared_edge_is_found():
    from anygeometry.nurbs_surface_intersection import UnclassifiedReason
    # second plane's shared edge runs in the opposite direction
    result, _ = query(plane(flat), plane(lambda u, j: (1. - u, 1., float(j))))
    assert result.unclassified_reason is UnclassifiedReason.SHARED_BOUNDARY_CURVE
    assert result.stats.boxes_visited == 0


NEGATIVE = {
    # coplanar squares touching only at a corner: zero overlap area and no common edge curve
    "coplanar_corner_touch_only": lambda: (plane(flat), plane(lambda u, j: (u + 1., j + 1., 0.))),
    "coplanar_disjoint": lambda: (plane(flat), plane(lambda u, j: (u + 3., float(j), 0.))),
    "same_surface_ranges_touch_on_a_line": lambda: (dataclasses.replace(bump(), u_range=(0., .5)),
                                                    dataclasses.replace(bump(), u_range=(.5, 1.))),
    "parallel_offset_planes_other_knots": lambda: (plane(flat), plane(lambda u, j: (u, float(j), 1e-9), n=3,
                                                                      knots=(0., 0., .5, 1., 1.))),
    "shared_edge_but_trimmed_support": lambda: (plane(flat), dataclasses.replace(
        plane(lambda u, j: (u, 1., float(j))), u_range=(0., .5))),
}


@pytest.mark.parametrize("name", sorted(NEGATIVE))
def test_unproven_pairs_stay_silent(name):
    result, _ = query(*NEGATIVE[name](), qualification=SimpleNamespace(max_boxes_per_pair=40))
    assert getattr(result, "unclassified_reason", None) is None


def test_transversal_generic_pair_is_still_certified_without_a_reason():
    result, _ = query(plane(lambda u, j: (u, float(j), .1)), bump(), qualification=None)
    assert result.status is IntersectionStatus.CERTIFIED
    assert result.unclassified_reason is None
