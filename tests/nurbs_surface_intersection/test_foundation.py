"""Foundation tests: intervals vs exact reference, disjoint/unclassified, budget, cancel, integrity,
and a generic local Krawczyk certificate.  Not executed by the author; root runs them."""
from __future__ import annotations

import itertools
import math
from fractions import Fraction
from types import SimpleNamespace

import pytest

from anygeometry.errors import GeometryError
from anygeometry.nurbs import NurbsSurface
from anygeometry.nurbs_surface_intersection import (
    IntersectionStatus, NurbsSurfaceIntersectionCancelled, NurbsSurfaceIntersectionIntegrityError,
    certify_local_arc, query_nurbs_surface_intersection)
from anygeometry.nurbs_surface_intersection._intervals import Interval, float_down, float_up
from anygeometry.tolerance import TolerancePolicy

KNOTS = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
B = (0.0, 0.0, 1.0)      # Bernstein coefficients of t**2 on [0, 1]


def graph(a: float, c: float) -> NurbsSurface:
    """(x, y, a (x^2 + y^2) + c) as an exact bidegree-2 Bezier patch."""
    return NurbsSurface(2, 2, KNOTS, KNOTS,
                        tuple(tuple((i / 2, j / 2, a * (B[i] + B[j]) + c) for j in range(3)) for i in range(3)))


def test_interval_ops_match_exact_enumeration():
    values = [Fraction(n, d) for n in (-3, -1, 0, 2, 5) for d in (1, 3)]
    for a0, a1, b0, b1 in itertools.product(values[:6], repeat=4):
        if a0 > a1 or b0 > b1:
            continue
        a, b = Interval(a0, a1), Interval(b0, b1)
        for op, f in ((a + b, lambda x, y: x + y), (a - b, lambda x, y: x - y), (a * b, lambda x, y: x * y)):
            corners = [f(x, y) for x in (a0, a1) for y in (b0, b1)]
            assert op.lo == min(corners) and op.hi == max(corners)


def test_float_conversion_is_outward():
    for value in (Fraction(1, 3), Fraction(-1, 3), Fraction(10 ** 30, 7), Fraction(1, 10 ** 30)):
        assert Fraction(float_down(value)) <= value <= Fraction(float_up(value))
    assert float_down(Fraction(10 ** 400)) < math.inf and float_up(Fraction(10 ** 400)) == math.inf


def test_disjoint_requires_complete_exclusion():
    result = query_nurbs_surface_intersection(graph(1.0, 0.0), graph(2.0, -10.0), tolerances=TolerancePolicy())
    assert result.status is IntersectionStatus.DISJOINT and result.cover_complete and result.components == ()
    assert result.stats.boxes_visited + result.stats.engine_box_charges >= 1


def test_intersecting_pair_is_unclassified_never_disjoint():
    from test_generic import rational_circle
    first,second=rational_circle()
    limits = SimpleNamespace(max_subdivision_depth=3)
    result = query_nurbs_surface_intersection(first,second, tolerances=TolerancePolicy(),
                                              qualification=limits)
    assert result.status is IntersectionStatus.UNCLASSIFIED and not result.cover_complete
    assert result.components == ()


def test_box_budget_exhaustion_is_unclassified():
    limits = SimpleNamespace(max_boxes_per_pair=1)
    result = query_nurbs_surface_intersection(graph(1.0, 0.0), graph(2.0, -0.5), tolerances=TolerancePolicy(),
                                              qualification=limits)
    assert result.status is IntersectionStatus.UNCLASSIFIED and "budget" in result.reason


def test_cancellation_truthy_raises_and_callback_errors_propagate():
    first, second = graph(1.0, 0.0), graph(2.0, -10.0)
    with pytest.raises(NurbsSurfaceIntersectionCancelled):
        query_nurbs_surface_intersection(first, second, tolerances=TolerancePolicy(), cancellation_check=lambda: 1)

    def boom():
        raise RuntimeError("callback failure")

    with pytest.raises(RuntimeError):
        query_nurbs_surface_intersection(first, second, tolerances=TolerancePolicy(), cancellation_check=boom)


def test_live_input_mutation_during_callback_is_detected_and_input_not_written():
    first, second = graph(1.0, 0.0), graph(2.0, -10.0)

    def mutate():
        object.__setattr__(first, "controls", graph(1.0, 5.0).controls)
        return False

    with pytest.raises(NurbsSurfaceIntersectionIntegrityError):
        query_nurbs_surface_intersection(first, second, tolerances=TolerancePolicy(), cancellation_check=mutate)
    clean = graph(2.0, -10.0)
    query_nurbs_surface_intersection(graph(1.0, 0.0), clean, tolerances=TolerancePolicy())
    assert clean == graph(2.0, -10.0)


def test_rejects_non_surface_inputs():
    with pytest.raises(GeometryError):
        query_nurbs_surface_intersection(object(), graph(1.0, 0.0), tolerances=TolerancePolicy())


def _box(centre: Fraction, half: Fraction):
    return tuple((centre - half, centre + half) for _ in range(4))


def graph_weighted(a: float, c: float, weight: float, u_range=None) -> NurbsSurface:
    """``graph`` with a uniform weight (ratio 1: the same geometry at any scale)."""
    rows = tuple(tuple((i / 2, j / 2, a * (B[i] + B[j]) + c) for j in range(3)) for i in range(3))
    return NurbsSurface(2, 2, KNOTS, KNOTS, rows, tuple((weight,) * 3 for _ in range(3)), u_range, None)


# The dependent coordinates need a larger domain than the free coordinate.
# A small geometric box also bounds the curvature variation in the minor;
# the certificate must still prove strict inclusion over a NONZERO interval.
_REGION = dict(unknown=[Interval(0, 1)] * 3, free_interval=(Fraction(3, 8), Fraction(5, 8)))
_REGULAR_BOX = _box(Fraction(1, 2), Fraction(1, 128))


def test_local_certificate_on_regular_graph_pair():
    # S1 = (x, y, x^2 + y^2) and S2 = (x, y, 2 (x^2 + y^2) - 1/2) meet in x^2 + y^2 = 1/2; (1/2, 1/2) is on it.
    first, second = graph(1.0, 0.0), graph(2.0, -0.5)
    assert first.exact_point(0.5, 0.5) == second.exact_point(0.5, 0.5)       # independent exact identity
    cert = certify_local_arc(first, second, _REGULAR_BOX, 0, **_REGION)
    assert cert.certified and cert.regular and cert.immersed and cert.weights_positive
    assert not cert.uses_dense_4d_tensor
    assert all(e.strictly_inside(Interval(0, 1)) for e in cert.unknown_enclosure)


def test_local_certificate_reversed_chart_uses_knot_coordinates():
    cert = certify_local_arc(graph_weighted(1.0, 0.0, 1.0, (1.0, 0.0)), graph(2.0, -0.5),
                             _REGULAR_BOX, 0, **_REGION)
    assert cert.certified


def test_local_certificate_rejects_box_outside_cropped_chart_and_bad_arguments():
    cropped = graph_weighted(1.0, 0.0, 1.0, (0.0, 0.5))
    with pytest.raises(GeometryError):
        certify_local_arc(cropped, graph(2.0, -0.5), _box(Fraction(1, 2), Fraction(1, 16)), 0, **_REGION)
    first, second = graph(1.0, 0.0), graph(2.0, -0.5)
    box = _box(Fraction(1, 2), Fraction(1, 16))
    for bad in (dict(free_coordinate=4), dict(free_coordinate=True), dict(knot_box=box[:3]),
                dict(unknown=[Interval(0, 1)] * 2), dict(free_interval=(0, 2))):
        arguments = dict(knot_box=box, free_coordinate=0)
        arguments.update(bad)
        with pytest.raises(GeometryError):
            certify_local_arc(first, second, arguments.pop("knot_box"), arguments.pop("free_coordinate"),
                              **arguments)
    with pytest.raises(GeometryError):
        certify_local_arc(object(), second, box, 0)


@pytest.mark.parametrize("weight", [1e300, 1e-300])
def test_local_certificate_is_scale_equivalent_for_uniform_weights(weight):
    reference = certify_local_arc(graph(1.0, 0.0), graph(2.0, -0.5), _REGULAR_BOX, 0,
                                  **_REGION)
    scaled = certify_local_arc(graph_weighted(1.0, 0.0, weight), graph_weighted(2.0, -0.5, weight),
                               _REGULAR_BOX, 0, **_REGION)
    assert reference.certified and scaled.certified


def test_forged_list_containers_cannot_alias_the_snapshot():
    first, second = graph(1.0, 0.0), graph(2.0, -10.0)
    forged = [[list(point) for point in row] for row in first.controls]
    object.__setattr__(first, "controls", forged)

    def mutate_in_place():
        forged[0][0][2] = 5.0
        return False

    with pytest.raises(NurbsSurfaceIntersectionIntegrityError):
        query_nurbs_surface_intersection(first, second, tolerances=TolerancePolicy(),
                                         cancellation_check=mutate_in_place)


def test_truthy_initial_callback_does_no_expensive_exact_work(monkeypatch):
    calls = []
    monkeypatch.setattr(NurbsSurface, "exact_net", lambda self: calls.append("net"))
    monkeypatch.setattr(NurbsSurface, "patch_bounds", lambda self, *a, **k: calls.append("bounds"))
    with pytest.raises(NurbsSurfaceIntersectionCancelled):
        query_nurbs_surface_intersection(graph(1.0, 0.0), graph(2.0, -10.0), tolerances=TolerancePolicy(),
                                         cancellation_check=lambda: True)
    assert calls == []


def test_issued_records_are_deeply_immutable_and_depth_is_a_total_limit():
    from test_generic import rational_circle
    first,second=rational_circle()
    result = query_nurbs_surface_intersection(first,second, tolerances=TolerancePolicy(),
                                              qualification=SimpleNamespace(max_subdivision_depth=2))
    with pytest.raises(Exception):
        result.stats.boxes_visited = 99
    with pytest.raises(Exception):
        Interval(0, 1).lo = Fraction(5)
    assert result.status is IntersectionStatus.UNCLASSIFIED and result.stats.max_depth <= 2
    tiny = query_nurbs_surface_intersection(graph(1.0, 0.0), graph(2.0, -0.5), tolerances=TolerancePolicy(),
                                            qualification=SimpleNamespace(max_boxes_per_pair=2))
    assert tiny.stats.boxes_visited <= 2 and tiny.status is IntersectionStatus.UNCLASSIFIED


def test_local_certificate_refuses_tangent_pair_and_far_box():
    # x^2 + y^2 = 2 (x^2 + y^2) at the origin: tangent surfaces touching at (0, 0, 0); no regular certificate
    tangent = certify_local_arc(graph(1.0, 0.0), graph(2.0, 0.0), _box(Fraction(1, 16), Fraction(1, 16)), 0)
    assert not tangent.certified
    far = certify_local_arc(graph(1.0, 0.0), graph(2.0, -10.0), _box(Fraction(1, 2), Fraction(1, 16)), 0)
    assert not far.certified
