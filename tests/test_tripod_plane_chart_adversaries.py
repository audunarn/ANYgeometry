"""Adversaries of the plane-chart route: folds, boundary events, coincident angular edge, bounds, pair qualification.

Every outcome is either an exact-oracle-consistent result or a TYPED refusal naming the pair; nothing may vanish.
"""
from fractions import Fraction
import math

import numpy as np
import pytest

from anygeometry import BezierQuadricCurve
from anygeometry.analytic_supports import SupportIntersection
from anygeometry.plane_ruled import PlaneRuledSupportError, qualify_plane_cylinder_result, support_residuals
from anygeometry.plane_ruled_support import (
    cluster_extent_metres, coincidence_bound, curve_error_bound, patch_margins, plane_ruled_cylinder_support,
)
from anygeometry.surfaces import Cylinder, Plane

TOL = 1e-9


def _residuals(result, plane, cylinder):
    stations = [np.asarray(p) for p in result.points]
    for curve in result.curves:
        stations.extend(curve.evaluate(np.linspace(0.0, 1.0, 25)))
    return [support_residuals(p, plane, cylinder) for p in stations], stations


def tilted_plane(eps, c):
    """x + eps z = c (the stored vectors give the exact normal (1, 0, eps))."""
    return Plane((c, 0.0, 0.0), (0.0, 1.0, 0.0), (-eps, 0.0, 1.0))


def unit_cylinder(height=4.0, **kwargs):
    return Cylinder((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), 1.0, height, **kwargs)


def test_interior_fold_is_enumerated_and_both_branches_meet_at_the_tangent_point():
    eps, zstar = 1e-3, 2.0
    plane, cylinder = tilted_plane(eps, 1.0 + eps * zstar), unit_cylinder()
    result = plane_ruled_cylinder_support(plane, cylinder, tolerance=TOL)
    assert result.curves and all(isinstance(c, BezierQuadricCurve) for c in result.curves)
    residuals, stations = _residuals(result, plane, cylinder)
    assert all(p <= TOL and c <= TOL for p, c in residuals)
    z = [float(p[2]) for p in stations]
    assert min(z) == pytest.approx(zstar, abs=1e-6) and max(z) == pytest.approx(4.0, abs=1e-6)
    ends = [curve.evaluate(u) for curve in result.curves for u in (0.0, 1.0)]
    assert any(np.linalg.norm(p - np.array([1.0, 0.0, zstar])) <= 1e-6 for p in ends), "tangent point is a chart end"


@pytest.mark.parametrize("gap", [2e-10, 1e-9, 1e-7])
def test_fold_close_to_the_height_boundary_is_kept_or_typed_never_silently_dropped(gap):
    eps, height = 1e-3, 4.0
    zstar = height - gap
    plane, cylinder = tilted_plane(eps, 1.0 + eps * zstar), unit_cylinder(height)
    try:
        result = plane_ruled_cylinder_support(plane, cylinder, tolerance=TOL, face_pair=(1, 2))
    except PlaneRuledSupportError as refusal:
        assert refusal.details["face_pair"] == (1, 2) and refusal.reason
        return
    near = [np.asarray(p) for p in result.points]
    for curve in result.curves:
        near.extend(curve.evaluate(np.linspace(0.0, 1.0, 5)))
    assert any(np.linalg.norm(p - np.array([1.0, 0.0, zstar])) <= 1e-6 for p in near), \
        "the real sliver of plane-meet-patch must not disappear"


def test_fold_beyond_the_finite_patch_is_never_reported_as_a_contact():
    """The true tangency lies 5e-7 above the patch; with a 1e-10 tolerance nothing of the patch may be reported."""
    from anygeometry.analytic_supports import plane_cylinder_support
    eps, height, tol = 1e-6, 2.0, 1e-10
    plane = tilted_plane(eps, 1.0 + eps * (height + 5e-7))
    cylinder = unit_cylinder(height)
    try:
        result = plane_cylinder_support(plane, cylinder, tolerance=tol, face_pair=(3, 4))
    except PlaneRuledSupportError as refusal:
        assert refusal.reason                                 # an actionable typed refusal is acceptable
        return
    reported = [np.asarray(p) for p in result.points]
    for curve in result.curves:
        reported.extend(curve.evaluate(np.linspace(0.0, 1.0, 9)))
    for point in reported:
        assert min(patch_margins(cylinder, point)) >= -tol, "a contact outside the true finite patch was emitted"


def test_coincident_angular_boundary_trace_is_retained():
    """A web plane through the seam generator of a partial-sweep skin: the trace IS the patch's angular edge."""
    axis = (0.0, 1e-14, 1.0)
    cylinder = Cylinder((0.0, 0.0, 0.0), axis, (1.0, 0.0, 0.0), 1.0, 4.0, 0.0, 0.5 * math.pi)
    plane = Plane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    result = plane_ruled_cylinder_support(plane, cylinder, tolerance=TOL, face_pair=(7, 8))
    assert result.curves, "the coincident angular edge must be retained, not dropped"
    residuals, stations = _residuals(result, plane, cylinder)
    assert all(p <= TOL and c <= TOL for p, c in residuals)
    axial = [float((p - np.asarray(cylinder.origin)) @ np.asarray(cylinder.axis)) for p in stations]
    assert min(axial) == pytest.approx(0.0, abs=1e-6) and max(axial) == pytest.approx(4.0, abs=1e-6)
    for point in stations:
        assert min(patch_margins(cylinder, point)) >= -TOL
        assert point[0] == pytest.approx(1.0, abs=1e-6)             # the opposite trace (angle pi) is outside the sweep


def test_curve_error_bound_is_derived_finite_and_below_a_quarter_tolerance():
    plane = Plane((45.0, -10.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    cylinder = Cylinder((44.0, 0.0, 0.0), (2.45e-16, 0.0, 1.0), (1.0, 0.0, 0.0), 5.0, 4.0)
    result = plane_ruled_cylinder_support(plane, cylinder, tolerance=TOL)
    assert result.curves
    for curve in result.curves:
        bound, details = curve_error_bound(curve.plan(), curve.first, curve, cylinder)
        assert 0.0 < bound <= 0.25 * TOL, details


def test_cluster_extent_covers_the_whole_chart_not_a_linear_t_estimate():
    plane = Plane((45.0, -10.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    cylinder = Cylinder((44.0, 0.0, 0.0), (2.45e-16, 0.0, 1.0), (1.0, 0.0, 0.0), 5.0, 4.0)
    curve = plane_ruled_cylinder_support(plane, cylinder, tolerance=TOL).curves[0]
    fs = curve.plan().floats()
    full = cluster_extent_metres(curve.first, fs, 0.0, 1.0)
    assert full >= curve.first.length * 0.99
    assert cluster_extent_metres(curve.first, fs, 0.5, 0.5) < 1e-6          # a degenerate cluster stays tiny


def test_coincidence_bound_is_small_only_for_a_piece_lying_in_the_boundary_plane():
    from anygeometry.quadric_supports import _boundary_planes
    axis = (0.0, 1e-14, 1.0)
    cylinder = Cylinder((0.0, 0.0, 0.0), axis, (1.0, 0.0, 0.0), 1.0, 4.0, 0.0, 0.5 * math.pi)
    plane = Plane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    result = plane_ruled_cylinder_support(plane, cylinder, tolerance=TOL)
    curve = result.curves[0]
    lo, hi = sorted((curve.start, curve.start + curve.sweep))
    boundaries = _boundary_planes(cylinder)
    bounds = [coincidence_bound(curve.first, curve.plan(), plane_, lo, hi, mid_world_distance=1e-12)
              for plane_ in boundaries]
    assert min(bounds[2:]) <= 0.25 * TOL                       # the angular edge it lies in
    assert bounds[0] > 0.25 * TOL or bounds[0] > 1.0            # the height planes it only crosses


def test_pair_qualification_refuses_the_historic_off_support_point_with_pair_residual_tolerance():
    plane = Plane((45.0, -10.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    cylinder = Cylinder((44.0, 0.0, 0.0), (2.45e-16, 0.0, 1.0), (1.0, 0.0, 0.0), 5.0, 4.0)
    bad = SupportIntersection(points=((45.0, -5.258610463437158, 4.0),))
    with pytest.raises(PlaneRuledSupportError) as caught:
        qualify_plane_cylinder_result(bad, plane, cylinder, TOL, (199, 200))
    diagnostic = caught.value.to_diagnostic()
    assert diagnostic["reason"] == "support_residual" and diagnostic["face_pair"] == (199, 200)
    assert diagnostic["tolerance"] == TOL and max(diagnostic["plane_residual"], diagnostic["cylinder_residual"]) > TOL
    good = plane_ruled_cylinder_support(plane, cylinder, tolerance=TOL, face_pair=(199, 200))
    worst = qualify_plane_cylinder_result(good, plane, cylinder, TOL, (199, 200))
    assert max(worst) <= TOL


try:                                  # the fixed focused-check list names this file: also run the proof regressions here
    import test_tripod_plane_chart_proofs as _proofs
except ImportError:
    from tests import test_tripod_plane_chart_proofs as _proofs
# only the test functions: the proof module's helpers (e.g. its own unit_cylinder) must not shadow this file's
globals().update({name: value for name, value in vars(_proofs).items() if name.startswith("test_")})


def test_residual_oracle_is_exact_on_a_float_point():
    plane = Plane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    cylinder = unit_cylinder()
    plane_distance, cylinder_distance = support_residuals((1.0, 0.0, 0.0), plane, cylinder)
    assert plane_distance == 0.0 and cylinder_distance == 0.0
    assert support_residuals((1.0, 0.0, 0.5), plane, cylinder)[0] == pytest.approx(0.5)
