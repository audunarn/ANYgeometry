"""NumPy-only fit-point interpolation of sketch splines (``_nurbs(fit=True)``)."""
import numpy as np
import pytest

from anygeometry import GeometryError
from anygeometry.sketch_entities import _nurbs


def _chord(points):
    points = np.asarray(points, dtype=float)
    t = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    return t / t[-1]


POINTS = ((0., 0.), (1., 2.), (2.5, 2.2), (4., 0.5), (5., -1.), (7., 0.))


def test_cubic_not_a_knot_knots_and_interpolation_at_chord_parameters():
    curve = _nurbs(POINTS, 3, fit=True)
    t = _chord(POINTS)
    assert curve.degree == 3 and curve.domain == (0., 1.)
    assert curve.knots == pytest.approx((0., 0., 0., 0., t[2], t[3], 1., 1., 1., 1.), abs=0., rel=0.)
    assert curve.evaluate(t) == pytest.approx(np.asarray(POINTS), abs=1e-12)


def test_cubic_reproduces_uniformly_spaced_line():
    points = [(1. + 2. * i, -3. + i) for i in range(6)]
    curve = _nurbs(points, 3, fit=True)
    s = np.linspace(0., 1., 23)
    assert curve.evaluate(s) == pytest.approx(np.c_[1. + 10. * s, -3. + 5. * s], abs=1e-12)


def test_three_point_quadratic_matches_closed_form_bezier():
    points = np.asarray(((0., 0.), (1., 1.5), (3., 0.)))
    curve = _nurbs(points, 2, fit=True)
    u = _chord(points)[1]
    middle = (points[1] - (1 - u) ** 2 * points[0] - u ** 2 * points[2]) / (2 * u * (1 - u))
    assert curve.degree == 2 and curve.knots == (0., 0., 0., 1., 1., 1.)
    assert np.asarray(curve.controls) == pytest.approx(np.asarray((points[0], middle, points[2])), abs=1e-12)


def test_quadratic_request_with_more_points_stays_cubic_and_two_points_are_linear():
    assert _nurbs(POINTS, 2, fit=True).degree == 3
    line = _nurbs(((0., 0.), (2., 1.)), 3, fit=True)
    assert line.degree == 1 and line.knots == (0., 0., 1., 1.)
    assert np.asarray(line.controls) == pytest.approx(np.asarray(((0., 0.), (2., 1.))), abs=0.)


def test_degree_one_fit_is_the_polyline_through_the_points():
    curve = _nurbs(POINTS, 1, fit=True)
    t = _chord(POINTS)
    assert curve.knots == pytest.approx(np.r_[0., 0., t[1:-1], 1., 1.], abs=0., rel=0.)
    assert np.asarray(curve.controls) == pytest.approx(np.asarray(POINTS), abs=1e-12)


def test_closed_fit_duplicates_first_point_and_closes():
    square = ((3., 3.), (7., 3.), (7., 7.), (3., 7.))
    curve = _nurbs(square, 3, fit=True, closed=True)
    closed = np.vstack((square, square[:1]))
    assert curve.evaluate(_chord(closed)) == pytest.approx(closed, abs=1e-12)
    assert len(curve.controls) == 5


@pytest.mark.parametrize('points, degree, message', [
    (((0., 0.),), 3, 'at least two points'),
    (((0., 0.), (1., 1.), (1., 1.), (2., 0.)), 3, 'distinct neighbours'),
    (((0., 0.), (1., np.nan), (2., 0.)), 3, 'finite'),
    (POINTS, 4, 'odd degree'),
    (((0., 0.), (1., 1.)), 0, 'positive'),
])
def test_fit_refusals_are_typed(points, degree, message):
    with pytest.raises(GeometryError, match=message):
        _nurbs(points, degree, fit=True)


def test_matches_previous_scipy_construction_when_scipy_is_available():
    interpolate = pytest.importorskip('scipy.interpolate')
    for points, degree, knots in ((POINTS, 3, None), (POINTS[:3], 2, (0., 0., 0., 1., 1., 1.)),
                                  (POINTS, 1, None), (POINTS + ((8., 2.), (9., 1.)), 5, None)):
        reference = interpolate.make_interp_spline(_chord(points), np.asarray(points), k=degree, t=knots)
        curve = _nurbs(points, degree, fit=True)
        assert curve.knots == pytest.approx(reference.t, abs=0., rel=0.)
        assert np.asarray(curve.controls) == pytest.approx(reference.c, abs=1e-10)
