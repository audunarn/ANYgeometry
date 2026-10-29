"""Near-parallel Plane/Cylinder support intersection: exact routing, plane-chart traces, refusals, serialization.

The oracle is independent of the code under test: residuals of returned points against the STORED plane and
cylinder are evaluated in exact rational arithmetic on the float coordinates.
"""
from fractions import Fraction
import json
import math

import numpy as np
import pytest

from anygeometry import BezierQuadricCurve, EllipticArc
from anygeometry.analytic_supports import plane_cylinder_support, cylinder_cylinder_support
from anygeometry.errors import GeometryError
from anygeometry.plane_ruled import (
    PlaneRuledSupport, PlaneRuledSupportError, build_plane_ruled_support, choose_ruling,
    elliptic_route_bound, exact_axial_determinant, exact_parallel,
)
from anygeometry.runtime_diagnostics import collect_runtime_diagnostics
from anygeometry.serialization import _bezier_quadric_curve_record, _decode_bezier_quadric_curve
from anygeometry.surfaces import Cylinder, Plane

TOL = 1e-9


def _fvec(v):
    return [Fraction(float(x)) for x in v]


def plane_distance(point, plane):
    p, o = _fvec(point), _fvec(plane.origin)
    u, v = _fvec(plane.u_vector), _fvec(plane.v_vector)
    n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
    nn = sum(c * c for c in n)
    return float(abs(sum((p[i] - o[i]) * n[i] for i in range(3))) / Fraction(math.sqrt(float(nn))))


def cylinder_distance(point, cylinder):
    w = [a - b for a, b in zip(_fvec(point), _fvec(cylinder.origin))]
    a = _fvec(cylinder.axis)
    along = sum(w[i] * a[i] for i in range(3)) / sum(c * c for c in a)
    perp = [w[i] - along * a[i] for i in range(3)]
    g = sum(c * c for c in perp) - Fraction(cylinder.radius) ** 2
    return float(abs(g) / (2 * Fraction(cylinder.radius)))


def web_and_skin(axial=2.45e-16, radius=5.0, offset=1.0, height=4.0):
    """The Face 200 situation: a vertical web plane x = 45 whose cylinder axis tilts by ``axial`` (rounded parallel)."""
    plane = Plane((45.0, -10.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    cylinder = Cylinder((45.0 - offset, 0.0, 0.0), (axial, 0.0, 1.0), (1.0, 0.0, 0.0), radius, height)
    return plane, cylinder


def test_exact_classification_uses_stored_doubles_not_the_rounded_normal():
    plane, cylinder = web_and_skin(axial=2.45e-16)
    assert exact_axial_determinant(plane, cylinder) != 0
    assert not exact_parallel(plane, cylinder)
    plane, cylinder = web_and_skin(axial=0.0)
    assert exact_axial_determinant(plane, cylinder) == 0
    assert exact_parallel(plane, cylinder)


@pytest.mark.parametrize("axial", [1e-18, 2.45e-16, 1e-12, 1e-8, 1e-4, -2.45e-16, -1e-8])
def test_near_parallel_traces_lie_on_both_stored_supports(axial):
    plane, cylinder = web_and_skin(axial=axial)
    result = plane_cylinder_support(plane, cylinder, tolerance=TOL)
    assert result.curves, "the real traces must be restored"
    ys = set()
    zs = []
    for curve in result.curves:
        points = curve.evaluate(np.linspace(0.0, 1.0, 33))
        for point in points:
            assert plane_distance(point, plane) <= TOL
            assert cylinder_distance(point, cylinder) <= TOL
            ys.add(round(float(point[1]), 3))
            # TRUE axial coordinate along the (tilted) cylinder axis, not world z
            zs.append(float((point - np.asarray(cylinder.origin)) @ np.asarray(cylinder.axis)))
    # the two generator-like traces y = +-sqrt(r^2 - offset^2) span the whole native axial extent [0, height]
    assert min(zs) == pytest.approx(0.0, abs=1e-6) and max(zs) == pytest.approx(cylinder.height, abs=1e-6)
    assert min(zs) >= -TOL and max(zs) <= cylinder.height + TOL
    assert any(abs(y - math.sqrt(24.0)) < 1e-2 for y in ys) and any(abs(y + math.sqrt(24.0)) < 1e-2 for y in ys)
    assert not result.coincident


def test_face_200_regression_no_point_far_from_support():
    plane, cylinder = web_and_skin(axial=2.453887774e-16)
    result = plane_cylinder_support(plane, cylinder, tolerance=TOL)
    candidates = [np.asarray(p) for p in result.points]
    for curve in result.curves:
        candidates.extend(curve.evaluate(np.linspace(0.0, 1.0, 9)))
    for segment in result.segments:
        candidates.extend(np.asarray(end) for end in segment)
    for point in candidates:
        assert abs(point[0] - 45.0) <= TOL, "rawpoint off the plane (0.4 m failure)"


def test_plane_chart_curves_are_retained_bezier_quadric_family_on_the_plane_support():
    plane, cylinder = web_and_skin()
    result = plane_cylinder_support(plane, cylinder, tolerance=TOL)
    assert all(isinstance(curve, BezierQuadricCurve) for curve in result.curves)
    assert all(isinstance(curve.first, PlaneRuledSupport) for curve in result.curves)
    for curve in result.curves:
        lo, hi = curve.bounds()
        assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
        assert np.all(lo <= curve.evaluate(0.5) + 1e-9) and np.all(curve.evaluate(0.5) - 1e-9 <= hi)
        # chart inversion through the plane coordinates
        point = curve.evaluate(0.3)
        parameters = curve.parameters_of(point, tolerance=TOL)
        assert parameters and abs(parameters[0] - 0.3) < 1e-6


def test_well_conditioned_pairs_keep_the_elliptic_route():
    plane = Plane((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 1.0))
    cylinder = Cylinder((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), 1.0, 3.0)
    result = plane_cylinder_support(plane, cylinder, tolerance=TOL)
    assert result.curves and all(isinstance(curve, EllipticArc) for curve in result.curves)


def test_elliptic_bound_refuses_the_ill_conditioned_ellipse_and_accepts_the_good_one():
    plane, cylinder = web_and_skin(axial=2.45e-16)
    bound = elliptic_route_bound(plane, cylinder, np.array([1e16, 0.0, 0.0]), np.array([0.0, 5.0, 2e16]),
                                 np.array([5.0, 0.0, 0.0]), limit=.25 * TOL)
    assert bound > .25 * TOL


def test_window_contains_exact_projection_of_the_whole_patch():
    plane, cylinder = web_and_skin()
    support, s_lo, s_hi = build_plane_ruled_support(plane, cylinder, tolerance=TOL)
    assert support.length > 0
    w1, w2, fraction = choose_ruling(plane, cylinder)
    assert fraction > 0.99                                  # the y vector rules; the z (axis-like) vector is the directrix
    base = np.asarray(plane.origin)
    for angle in np.linspace(0.0, 2.0 * math.pi, 25):
        for z in (0.0, 4.0):
            p = cylinder.origin + cylinder.radius * (math.cos(angle) * cylinder.radial_direction
                                                       + math.sin(angle) * cylinder.circumferential_direction) \
                + z * cylinder.axis
            p = p + (45.0 - p[0]) * np.array([1.0, 0.0, 0.0])      # onto the plane
            t = (p - base) @ np.asarray(w1) / (np.asarray(w1) @ np.asarray(w1))
            s = (p - base) @ np.asarray(w2) / (np.asarray(w2) @ np.asarray(w2))
            assert support.t0 <= t <= support.t0 + support.length
            assert s_lo <= s <= s_hi


def test_ruling_conditioning_refuses_with_a_typed_error():
    axis = np.array([0.0, 1e-6, 1.0])
    plane = Plane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1e-6, 0.0, 1.0))
    cylinder = Cylinder((0.0, 0.0, 0.0), axis, (1.0, 0.0, 0.0), 1.0, 2.0)
    assert not exact_parallel(plane, cylinder)
    with pytest.raises(PlaneRuledSupportError) as caught:
        plane_cylinder_support(plane, cylinder, tolerance=TOL)
    assert caught.value.reason == "ruling_conditioning"
    assert isinstance(caught.value, GeometryError)
    diagnostic = caught.value.to_diagnostic()
    assert diagnostic["reason"] == "ruling_conditioning" and "perpendicular_fraction" in diagnostic


def test_plane_chart_folds_enumerate_the_complete_closed_ellipse():
    """Forced plane-chart route on a well-conditioned plane: folds (extreme chart points) and the seam are events."""
    from anygeometry.plane_ruled_support import plane_ruled_cylinder_support
    plane = Plane((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 1.0))
    cylinder = Cylinder((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), 1.0, 3.0)
    result = plane_ruled_cylinder_support(plane, cylinder, tolerance=TOL)
    assert result.curves and all(isinstance(curve, BezierQuadricCurve) for curve in result.curves)
    ys, xs = [], []
    for curve in result.curves:
        for point in curve.evaluate(np.linspace(0.0, 1.0, 41)):
            assert plane_distance(point, plane) <= TOL
            assert cylinder_distance(point, cylinder) <= TOL
            xs.append(float(point[0]))
            ys.append(float(point[1]))
    assert min(ys) == pytest.approx(-1.0, abs=1e-6) and max(ys) == pytest.approx(1.0, abs=1e-6)
    assert min(xs) == pytest.approx(-1.0, abs=1e-6) and max(xs) == pytest.approx(1.0, abs=1e-6)
    # the chart ends of the pieces chain into a closed loop: every end meets another end
    ends = [curve.evaluate(u) for curve in result.curves for u in (0.0, 1.0)]
    for index, point in enumerate(ends):
        assert any(np.linalg.norm(point - other) <= 1e-9 for j, other in enumerate(ends) if j != index)


def test_exactly_parallel_plane_keeps_the_generator_route():
    plane, cylinder = web_and_skin(axial=0.0)
    result = plane_cylinder_support(plane, cylinder, tolerance=TOL)
    assert not result.curves and len(result.segments) == 2


def test_cancellation_is_cooperative_in_the_plane_chart_route():
    plane, cylinder = web_and_skin()
    with pytest.raises(GeometryError):
        plane_cylinder_support(plane, cylinder, tolerance=TOL, cancellation_check=lambda: True)


def test_runtime_counters_record_the_route_without_changing_results():
    plane, cylinder = web_and_skin()
    with collect_runtime_diagnostics() as counters:
        first = plane_cylinder_support(plane, cylinder, tolerance=TOL)
    assert counters["plane_cylinder_plane_chart_routes"] == 1
    assert counters["plane_chart_curves"] == len(first.curves)
    again = plane_cylinder_support(plane, cylinder, tolerance=TOL)
    assert len(again.curves) == len(first.curves)


def test_plane_ruled_serialization_round_trip_and_unknown_kind_is_refused():
    plane, cylinder = web_and_skin()
    curve = plane_cylinder_support(plane, cylinder, tolerance=TOL).curves[0]
    record = json.loads(json.dumps(_bezier_quadric_curve_record(curve)))
    assert record["first"]["kind"] == "plane_ruled"
    decoded = _decode_bezier_quadric_curve(record)
    taus = np.linspace(0.0, 1.0, 11)
    assert np.array_equal(decoded.evaluate(taus), curve.evaluate(taus))
    record["first"]["kind"] = "plane_ruled_v2"
    with pytest.raises(GeometryError):
        _decode_bezier_quadric_curve(record)


def test_cylinder_cylinder_near_parallel_axes_never_return_off_support_points():
    """Risk inventory (the cap-helper denominators): either a typed refusal or points on both supports."""
    first = Cylinder((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), 2.0, 3.0)
    for tilt in (1e-18, 1e-16, 1e-12, 1e-8):
        second = Cylinder((1.0, 0.0, 0.0), (tilt, 0.0, 1.0), (1.0, 0.0, 0.0), 2.0, 3.0)
        try:
            result = cylinder_cylinder_support(first, second, tolerance=TOL)
        except GeometryError:
            continue                                         # fail closed is acceptable
        points = [np.asarray(p) for p in result.points]
        for curve in result.curves:
            points.extend(curve.evaluate(np.linspace(0.0, 1.0, 9)))
        for segment in result.segments:
            points.extend(np.asarray(end) for end in segment)
        for point in points:
            assert cylinder_distance(point, first) <= 1e-7, f"tilt {tilt}: off first support"
            assert cylinder_distance(point, second) <= 1e-7, f"tilt {tilt}: off second support"
