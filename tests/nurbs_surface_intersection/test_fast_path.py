"""Public-query tests of the shared-affine-projection scalar fast path (author-unrun; root executes).

All fixtures are exact polynomial graphs z = P(x, y) on x, y in [-1, 1]; component counts are analytic
(circle r = 1/2: one closed; r = 1/4 and 3/4: two nested closed; y = 0: one open).
"""
from __future__ import annotations

import dataclasses
from fractions import Fraction
from math import comb
from types import SimpleNamespace

import pytest

from anygeometry.errors import GeometryError
from anygeometry.nurbs import NurbsSurface
from anygeometry.nurbs_surface_intersection import (
    IntersectionStatus, NurbsSurfaceIntersectionCancelled, NurbsSurfaceIntersectionIntegrityError,
    NurbsSurfaceIntersectionRefused, evaluate_arc, query_nurbs_surface_intersection)
from anygeometry.nurbs_surface_intersection._evaluate import result_digest
from anygeometry.tolerance import TolerancePolicy

TOL = TolerancePolicy()
F = Fraction


def _bernstein(coefficients, n):
    """Bernstein coefficients in t of sum c_k x^k with x = -1 + 2 t."""
    a = [F(0)] * (n + 1)
    for k, c in enumerate(coefficients):
        for j in range(k + 1):
            a[j] += c * comb(k, j) * 2 ** j * (-1) ** (k - j)
    return [sum(F(comb(i, j), comb(n, j)) * a[j] for j in range(i + 1)) for i in range(n + 1)]


def _exact(value: Fraction) -> float:
    made = float(value)
    assert F(made) == value, "fixture coefficient is not an exact double"
    return made


def graph(poly, n, *, weight=None, u_range=None, v_range=None, perm=(0, 1, 2), shift=(0.0, 0.0, 0.0)):
    poly = {key: F(value) for key, value in poly.items()}
    along_x = [_bernstein([poly.get((k, l), F(0)) for k in range(n + 1)], n) for l in range(n + 1)]
    z = [_bernstein([along_x[l][i] for l in range(n + 1)], n) for i in range(n + 1)]
    controls = []
    for i in range(n + 1):
        row = []
        for j in range(n + 1):
            point = (-1.0 + 2.0 * i / n, -1.0 + 2.0 * j / n, _exact(z[i][j]))
            row.append(tuple(point[p] + shift[k] for k, p in enumerate(perm)))
        controls.append(tuple(row))
    knots = (0.0,) * (n + 1) + (1.0,) * (n + 1)
    weights = None if weight is None else tuple((weight,) * (n + 1) for _ in range(n + 1))
    return NurbsSurface(n, n, knots, knots, tuple(controls), weights, u_range, v_range)


R2 = {(2, 0): 1, (0, 2): 1}
CIRCLE = {(2, 0): 2, (0, 2): 2, (0, 0): F(-1, 4)}
LINE = {(2, 0): 1, (0, 2): 1, (0, 1): 1}


def circle_pair(**options):
    return graph(R2, 2, **options), graph(CIRCLE, 2, **options)


def nested_pair(**options):
    base = {(2, 0): 3, (0, 2): 3}
    # base + 9 (r^2 - 1/16)(r^2 - 9/16) = 3 r^2 + 9 r^4 - 45/8 r^2 + 81/256
    other = {(2, 0): F(-21, 8), (0, 2): F(-21, 8), (4, 0): 9, (0, 4): 9, (2, 2): 18, (0, 0): F(81, 256)}
    return graph(base, 4, **options), graph(other, 4, **options)


def certified(first, second, **kwargs):
    result = query_nurbs_surface_intersection(first, second, tolerances=TOL, **kwargs)
    return result


def test_closed_circle_is_one_complete_closed_component_and_inputs_unchanged():
    first, second = circle_pair()
    before = (dataclasses.replace(first), dataclasses.replace(second))
    result = certified(first, second)
    assert result.status is IntersectionStatus.CERTIFIED and result.cover_complete
    assert result.algorithm == "shared-affine-projection-scalar"
    assert len(result.components) == 1 and result.components[0].closed
    assert result.components[0].root_ids[0] == result.components[0].root_ids[-1]
    assert first == before[0] and second == before[1]
    assert result.first_source.matches(first) and result.second_source.matches(second)
    assert result.stats.components == 1 and result.stats.roots >= 1 and result.stats.engine_box_charges > 0


def test_open_curved_line_has_two_distinct_boundary_endpoints():
    result = certified(graph(R2, 2), graph(LINE, 2))
    assert result.status is IntersectionStatus.CERTIFIED and len(result.components) == 1
    component = result.components[0]
    assert not component.closed and component.root_ids[0] != component.root_ids[-1]
    low, high = component.world_lo, component.world_hi
    assert low[1] <= 0.0 <= high[1] and low[0] <= -1.0 + 1e-9 and high[0] >= 1.0 - 1e-9


def test_two_nested_loops_are_two_closed_components():
    result = certified(*nested_pair())
    assert result.status is IntersectionStatus.CERTIFIED
    assert [c.closed for c in result.components] == [True, True]


@pytest.mark.parametrize("perm,shift", [((0, 1, 2), (0.0, 0.0, 0.0)), ((2, 0, 1), (3.0, -5.0, 7.0)),
                                        ((1, 2, 0), (1e6, 0.0, -1e6))])
def test_permuted_and_translated_axes_keep_the_component_structure(perm, shift):
    result = certified(*circle_pair(perm=perm, shift=shift))
    assert result.status is IntersectionStatus.CERTIFIED
    assert [c.closed for c in result.components] == [True]


@pytest.mark.parametrize("first_weight,second_weight", [(1e300, 1e-300), (1e-300, 1e300), (3.0, 3.0)])
def test_uniform_weights_cancel_exactly(first_weight, second_weight):
    first = graph(R2, 2, weight=first_weight)
    second = graph(CIRCLE, 2, weight=second_weight)
    result = certified(first, second)
    assert result.status is IntersectionStatus.CERTIFIED and [c.closed for c in result.components] == [True]


def test_reversed_charts_and_cropped_rectangle():
    first, second = circle_pair(u_range=(1.0, 0.0), v_range=(1.0, 0.0))
    assert [c.closed for c in certified(first, second).components] == [True]
    half = graph(R2, 2, u_range=(0.5, 1.0)), graph(CIRCLE, 2, u_range=(0.5, 1.0))
    result = certified(*half)
    assert result.status is IntersectionStatus.CERTIFIED and [c.closed for c in result.components] == [False]
    disjoint = certified(graph(R2, 2, u_range=(0.0, 0.25)), graph(CIRCLE, 2, u_range=(0.75, 1.0)))
    assert disjoint.status is IntersectionStatus.DISJOINT and disjoint.cover_complete


def test_non_admitted_pairs_keep_the_generic_unclassified_fallback():
    weights = tuple((1.0, 1.0, 1.0) if i != 1 else (1.0, 2.0, 1.0) for i in range(3))
    knots = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    base = graph(R2, 2)
    rational = NurbsSurface(2, 2, knots, knots, base.controls, weights)
    result = certified(rational, graph(CIRCLE, 2), qualification=SimpleNamespace(max_subdivision_depth=4))
    assert result.status is IntersectionStatus.UNCLASSIFIED and result.algorithm == "generic-cover"
    assert not result.cover_complete and result.components == ()


def test_tangency_coincidence_and_budget_refuse_explicitly():
    tangent = certified(graph(R2, 2), graph({(2, 0): 2, (0, 2): 2}, 2))      # touch only at the origin
    assert tangent.status is IntersectionStatus.UNCLASSIFIED and tangent.components == ()
    same = certified(graph(R2, 2), graph(R2, 2), qualification=SimpleNamespace(max_subdivision_depth=4))
    assert same.status is IntersectionStatus.UNCLASSIFIED and same.components == ()
    first, second = circle_pair()
    starved = certified(first, second, qualification=SimpleNamespace(max_boxes_per_pair=3))
    assert starved.status is IntersectionStatus.UNCLASSIFIED and starved.components == ()
    assert starved.stats.engine_box_charges <= 3
    capped = certified(*nested_pair(), qualification=SimpleNamespace(max_components=1))
    assert capped.status is IntersectionStatus.UNCLASSIFIED and capped.components == ()


def test_cancellation_and_mutation_during_the_fast_path():
    first, second = circle_pair()
    calls = []

    def cancel_later():
        calls.append(1)
        return len(calls) > 3

    with pytest.raises(NurbsSurfaceIntersectionCancelled):
        certified(first, second, cancellation_check=cancel_later)

    def mutate():
        object.__setattr__(first, "controls", graph(R2, 2, shift=(0.0, 0.0, 1.0)).controls)
        return False

    with pytest.raises(NurbsSurfaceIntersectionIntegrityError):
        certified(first, second, cancellation_check=mutate)


def test_evaluation_gives_certified_enclosures_on_the_circle_and_rejects_forgery():
    first, second = circle_pair()
    result = certified(first, second)
    for station in (0, F(1, 4), F(1, 2), F(3, 4), 1):
        evaluation = evaluate_arc(result, 0, 0, station, first=first, second=second, tolerances=TOL)
        x, y, z = evaluation.witness
        assert abs(x * x + y * y - 0.25) < 1e-6 and not evaluation.witness_is_exact
        assert evaluation.world_diameter <= TOL.length and evaluation.validation_seconds >= 0.0
        assert all(l <= w <= h for l, w, h in zip(evaluation.world_lo, evaluation.witness, evaluation.world_hi))
        (u0, u1), (v0, v1) = evaluation.native_box
        assert 0 <= u0 <= u1 <= 1 and 0 <= v0 <= v1 <= 1
        # exact independent identity: both originals agree at the enclosure centre up to the enclosure size
        centre = (float((u0 + u1) / 2), float((v0 + v1) / 2))
        assert abs(first.evaluate(*centre)[2] - second.evaluate(*centre)[2]) < 1e-6
    with pytest.raises(GeometryError):
        evaluate_arc(result, 0, 0, F(3, 2), first=first, second=second, tolerances=TOL)
    forged = dataclasses.replace(result, boxes=result.boxes[:-1] if len(result.boxes) > 1 else ())
    with pytest.raises(NurbsSurfaceIntersectionIntegrityError):
        evaluate_arc(forged, 0, 0, F(1, 2), first=first, second=second, tolerances=TOL)
    with pytest.raises(NurbsSurfaceIntersectionIntegrityError):
        evaluate_arc(result, 0, 0, F(1, 2), first=graph(R2, 2, shift=(0.0, 0.0, 1.0)), second=second,
                     tolerances=TOL)
    with pytest.raises(NurbsSurfaceIntersectionCancelled):
        evaluate_arc(result, 0, 0, F(1, 2), first=first, second=second, tolerances=TOL,
                     cancellation_check=lambda: True)


def _resign(result):
    return dataclasses.replace(result, descriptor_digest=result_digest(
        result.structural, result.roots, result.boxes, result.components))


def test_forged_root_with_recomputed_digest_is_rejected_by_canonical_rederivation():
    first, second = circle_pair()
    result = certified(first, second)
    root = result.roots[result.components[0].arcs[0].first_root]
    fake = dataclasses.replace(root, axis=0, value=F(1, 2), lo=F(1, 2), hi=F(1, 2))
    forged = _resign(dataclasses.replace(result, roots=tuple(fake if r is root else r for r in result.roots)))
    assert forged.descriptor_digest != result.descriptor_digest
    with pytest.raises(NurbsSurfaceIntersectionIntegrityError):
        evaluate_arc(forged, 0, 0, 0, first=first, second=second, tolerances=TOL)
    stale = dataclasses.replace(result, roots=forged.roots)
    with pytest.raises(NurbsSurfaceIntersectionIntegrityError):
        evaluate_arc(stale, 0, 0, F(1, 2), first=first, second=second, tolerances=TOL)
    box = result.boxes[0]
    forged_box = _resign(dataclasses.replace(result, boxes=(dataclasses.replace(box, kind="u" if box.kind == "v" else "v"),)
                                             + result.boxes[1:]))
    with pytest.raises(NurbsSurfaceIntersectionIntegrityError):
        evaluate_arc(forged_box, 0, 0, F(1, 2), first=first, second=second, tolerances=TOL)


IRRATIONAL = {(2, 0): 2, (0, 2): 2, (0, 0): F(-1, 2)}        # r^2 = 1/2: edge roots y = sqrt(1/2) are irrational


def test_stations_use_the_actual_endpoints_irrational_roots_and_shared_endpoints():
    first, second = graph(R2, 2), graph(IRRATIONAL, 2)
    result = certified(first, second)
    assert result.status is IntersectionStatus.CERTIFIED and result.components[0].closed
    arcs = result.components[0].arcs
    for index, arc in enumerate(arcs):
        for station in (0, F(1, 1000), F(1, 3), F(999, 1000), 1):
            evaluation = evaluate_arc(result, 0, index, station, first=first, second=second, tolerances=TOL)
            x, y, _z = evaluation.witness
            assert abs(x * x + y * y - 0.5) < 1e-6
            assert evaluation.world_diameter <= min(TOL.length, TOL.surface_residual)
    for index, arc in enumerate(arcs):
        following = arcs[(index + 1) % len(arcs)]
        assert arc.second_root == following.first_root
        end = evaluate_arc(result, 0, index, 1, first=first, second=second, tolerances=TOL)
        start = evaluate_arc(result, 0, (index + 1) % len(arcs), 0, first=first, second=second, tolerances=TOL)
        assert all(a <= d and c <= b for (a, b), (c, d) in zip(end.native_box, start.native_box))
    try:                                   # extremely close to an endpoint: success or an explicit refusal only
        evaluate_arc(result, 0, 0, F(1, 10 ** 9), first=first, second=second, tolerances=TOL)
    except NurbsSurfaceIntersectionRefused:
        pass


def test_tiny_tolerance_with_an_exact_endpoint_refuses_instead_of_widening():
    first, second = graph(R2, 2), graph(LINE, 2)
    result = certified(first, second)
    tiny = TolerancePolicy(length=1e-30, relative_length=1e-30, surface_residual=1e-30)
    for station in (0, 1, F(1, 2)):
        with pytest.raises(NurbsSurfaceIntersectionRefused):
            evaluate_arc(result, 0, 0, station, first=first, second=second, tolerances=tiny)


def test_callback_context_mutation_cannot_bypass_the_private_budget_and_caller_context_is_preserved():
    from anygeometry.plane_nurbs import CURRENT_WORK
    first, second = circle_pair()
    sentinel = object()
    token = CURRENT_WORK.set(sentinel)
    try:
        seen = []

        def poison():
            seen.append(CURRENT_WORK.get())
            CURRENT_WORK.set(None)
            return False

        result = certified(first, second, qualification=SimpleNamespace(max_boxes_per_pair=1),
                           cancellation_check=poison)
        assert result.status is IntersectionStatus.UNCLASSIFIED and result.components == ()
        assert result.stats.engine_box_charges <= 1
        assert CURRENT_WORK.get() is sentinel and seen and all(item is sentinel for item in seen)
        with pytest.raises(NurbsSurfaceIntersectionCancelled):
            certified(first, second, cancellation_check=lambda: True)
        assert CURRENT_WORK.get() is sentinel
    finally:
        CURRENT_WORK.reset(token)


def test_reentrant_callback_query_does_not_disturb_the_outer_query():
    first, second = circle_pair()
    far = (graph(R2, 2), graph({(2, 0): 2, (0, 2): 2, (0, 0): -10}, 2))
    inner = []

    def reenter():
        inner.append(certified(*far).status)
        return False

    result = certified(first, second, cancellation_check=reenter)
    assert result.status is IntersectionStatus.CERTIFIED and len(result.components) == 1
    assert inner and all(status is IntersectionStatus.DISJOINT for status in inner)


def test_last_callback_mutation_is_detected_by_evaluation():
    first, second = circle_pair()
    result = certified(first, second)
    counter = [0]

    def count():
        counter[0] += 1
        return False

    evaluate_arc(result, 0, 0, F(1, 2), first=first, second=second, tolerances=TOL, cancellation_check=count)
    total = counter[0]
    assert total >= 2
    fresh_first, fresh_second = circle_pair()
    seen = [0]

    def mutate_on_last():
        seen[0] += 1
        if seen[0] == total:
            object.__setattr__(fresh_first, "controls", graph(R2, 2, shift=(0.0, 0.0, 1.0)).controls)
        return False

    with pytest.raises(NurbsSurfaceIntersectionIntegrityError):
        evaluate_arc(result, 0, 0, F(1, 2), first=fresh_first, second=fresh_second, tolerances=TOL,
                     cancellation_check=mutate_on_last)
    assert seen[0] == total


def test_reported_diameter_is_a_true_upper_bound_even_when_subnormal():
    from anygeometry.nurbs_surface_intersection._intervals import diameter_upper
    for squared in (F(1, 10 ** 700), F(1, 4 ** 200), F(3, 7), F(0)):
        reported = diameter_upper(squared)
        assert F(reported) ** 2 >= squared and (squared == 0 or reported > 0)


def test_malformed_roots_and_impossible_hulls_are_integrity_errors():
    from anygeometry.nurbs_surface_intersection._evaluate import _RootCopy
    from anygeometry.nurbs_surface_intersection._records import FrozenRoot
    from anygeometry.nurbs_surface_intersection._topology import Enclosures
    edge = (F(0), F(1))
    good = FrozenRoot(0, F(0), F(0), F(1), (F(-1), F(2)), edge, 0, 0)
    assert not _RootCopy(good).exact
    for bad in (FrozenRoot(0, F(0), F(1, 2), F(1, 2), (F(1),), edge, 0, 0),             # does not vanish
                FrozenRoot(0, F(0), F(0), F(1), (F(1), F(1)), edge, 0, 0),              # no sign change
                FrozenRoot(0, F(0), F(0), F(2), (F(-1), F(2)), edge, 0, 0),             # outside its edge
                FrozenRoot(2, F(0), F(0), F(1), (F(-1), F(2)), edge, 0, 0)):            # bad axis
        with pytest.raises(NurbsSurfaceIntersectionIntegrityError):
            _RootCopy(bad)
    enclosures = Enclosures(graph(R2, 2), graph(R2, 2, shift=(0.0, 0.0, 10.0)))
    with pytest.raises(NurbsSurfaceIntersectionIntegrityError):
        enclosures.world((((0.0, 1.0), (0.0, 1.0)), ((0.0, 1.0), (0.0, 1.0))))


def rational_pair(offset=F(0)):
    """Nonuniform rational (weights 1,2,1 across u) pair; B_z - A_z = C / w_i with C = (2u-1)^2 + (2v-1)^2 - 1/4
    (+ offset), so the weighted height difference is exactly the circle of radius 1/4 around (1/2, 1/2) in
    native (u, v).  The shared projection identity is an ANALYTIC ORACLE only; the kernel must not use it."""
    f = (F(1), F(-1), F(1))
    w = (1.0, 2.0, 1.0)
    knots = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    rows_a, rows_b = [], []
    for i in range(3):
        row_a, row_b = [], []
        for j in range(3):
            z_a = f[i] + f[j]
            c = f[i] + f[j] - F(1, 4) + offset
            z_b = z_a + c / F(int(w[i]))
            row_a.append((-1.0 + i, -1.0 + j, _exact(z_a)))
            row_b.append((-1.0 + i, -1.0 + j, _exact(z_b)))
        rows_a.append(tuple(row_a))
        rows_b.append(tuple(row_b))
    weights = tuple((w[i],) * 3 for i in range(3))
    return (NurbsSurface(2, 2, knots, knots, tuple(rows_a), weights),
            NurbsSurface(2, 2, knots, knots, tuple(rows_b), weights))


def test_generic_route_is_taken_and_tiny_budget_preserves_frontier():
    first, second = rational_pair()
    result = certified(first, second, qualification=SimpleNamespace(max_boxes_per_pair=48))
    assert result.algorithm == "generic-cover" and result.components == () and not result.cover_complete
    assert result.status is IntersectionStatus.UNCLASSIFIED
    report = result.atlas
    assert report is not None and not report.stitching_implemented
    assert report.excluded_by_hull + report.excluded_by_f + report.excluded_by_k > 0
    # Full generic positive acceptance is in test_generic; a deliberately tiny
    # discovery budget must keep its residual support visible.
    assert not report.complete_local_cover and report.frontier_remaining > 0
    assert result.stats.pieces_prepared >= 2 and result.stats.certificates == report.covered


def test_generic_rational_pair_without_intersection_is_disjoint_and_budget_leaves_a_counted_frontier():
    far = certified(*rational_pair(offset=F(10)))
    assert far.status is IntersectionStatus.DISJOINT and far.cover_complete and far.atlas.covered == 0
    first, second = rational_pair()
    starved = certified(first, second, qualification=SimpleNamespace(max_boxes_per_pair=5))
    assert starved.status is IntersectionStatus.UNCLASSIFIED and starved.components == ()
    assert starved.stats.boxes_visited <= 5 and starved.atlas.frontier_remaining > 0
    assert not starved.atlas.complete_local_cover


def test_generic_route_cancellation_and_mutation():
    first, second = rational_pair()
    with pytest.raises(NurbsSurfaceIntersectionCancelled):
        certified(first, second, cancellation_check=lambda: True)

    def mutate():
        object.__setattr__(first, "controls", graph(R2, 2, shift=(0.0,0.0,1.0)).controls)
        return False

    with pytest.raises(NurbsSurfaceIntersectionIntegrityError):
        certified(first, second, cancellation_check=mutate)


def test_evaluation_respects_a_reversed_chart_map():
    first, second = circle_pair(u_range=(1.0, 0.0), v_range=(1.0, 0.0))
    result = certified(first, second)
    evaluation = evaluate_arc(result, 0, 0, F(1, 2), first=first, second=second, tolerances=TOL)
    (u0, u1), (v0, v1) = evaluation.native_box
    (cu0, cu1), (cv0, cv1) = evaluation.first_chart
    assert cu0 <= 1 - float((u0 + u1) / 2) <= cu1 and cv0 <= 1 - float((v0 + v1) / 2) <= cv1


def test_evaluation_limit_refuses_instead_of_pretending():
    first, second = circle_pair()
    result = certified(first, second)
    with pytest.raises(NurbsSurfaceIntersectionRefused):
        evaluate_arc(result, 0, 0, F(1, 2), first=first, second=second,
                     tolerances=TolerancePolicy(length=1e-300), qualification=SimpleNamespace(max_newton_iterations=1))
