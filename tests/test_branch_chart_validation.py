"""Analytical regressions for public polynomial-chart endpoint validation."""

import numpy as np
import pytest

from anygeometry.branch_algebra import BezierRuledSupport
from anygeometry.branch_curves import BezierQuadricCurve
from anygeometry.errors import GeometryError
from anygeometry.quadric_algebra import QuadricSupport


DIRECTRIX = BezierRuledSupport(((0., 0., 0.), (.5, 0., 0.), (1., 0., 0.)), (0., 0., 1.))


def quadric(quadratic=0., linear=0., constant=0.):
    """z^2 + quadratic*x^2 + linear*x + constant = 0; x=t, z=s."""
    return QuadricSupport("general", matrix=(quadratic, 0., 0., 0., 0., 0., 0., 0., 1.),
                          linear=(.5 * linear, 0., 0.), constant=constant)


@pytest.mark.parametrize("curvature", [-1., 0., 1.])
@pytest.mark.parametrize("start,sweep", [(0., .5), (.5, -.5)])
def test_simple_fold_requires_regularized_endpoint(curvature, start, sweep):
    # Delta = 4(t + curvature*t^2), a simple fold for every sign of Delta''.
    with pytest.raises(GeometryError, match="simple fold"):
        BezierQuadricCurve(DIRECTRIX, quadric(-curvature, -1.), start, sweep)


@pytest.mark.parametrize("curvature", [-1., 0., 1.])
@pytest.mark.parametrize("start,sweep,mode,end", [(0., .5, "left_square", 0),
                                                 (.5, -.5, "right_square", 1)])
def test_regularized_simple_fold_keeps_analytical_jet(curvature, start, sweep, mode, end):
    curve = BezierQuadricCurve(DIRECTRIX, quadric(-curvature, -1.), start, sweep, parameterization=mode)
    # t=.5*u^2 and s=sqrt(.5)*u*sqrt(1+curvature*.5*u^2), u=tau or 1-tau.
    sign = 1. if end == 0 else -1.
    np.testing.assert_allclose(curve.derivative(end), (0., 0., sign * np.sqrt(.5)), atol=1e-14)
    np.testing.assert_allclose(curve.second_derivative(end), (1., 0., 0.), atol=1e-14)


@pytest.mark.parametrize("branch", [-1, 1])
@pytest.mark.parametrize("start,sweep,end", [(0., .5, 0), (.5, -.5, 1)])
def test_double_contact_keeps_one_sided_linear_jet(branch, start, sweep, end):
    # z^2-x^2=0 gives the exact straight branches (t, 0, branch*t).
    curve = BezierQuadricCurve(DIRECTRIX, quadric(-1.), start, sweep, branch)
    np.testing.assert_allclose(curve.derivative(end), (sweep, 0., branch * sweep), atol=1e-14)
    np.testing.assert_allclose(curve.second_derivative(end), (0., 0., 0.), atol=1e-14)


@pytest.mark.parametrize("start,end", [(-5e-13, .5), (.5, -5e-13),
                                       (.5, 1. + 5e-13), (1. + 5e-13, .5)])
def test_tolerated_domain_roundoff_is_canonicalized_and_enclosed(start, end):
    curve = BezierQuadricCurve(DIRECTRIX, quadric(constant=-1.), start, end - start)
    canonical = np.clip((start, end), 0., 1.)
    assert curve.start == canonical[0]
    assert curve.start + curve.sweep == canonical[1]
    points = curve.evaluate(np.linspace(0., 1., 9))
    lo, hi = curve.bounds()
    assert np.all(points >= lo) and np.all(points <= hi)


@pytest.mark.parametrize("start,sweep", [(-5e-13, 1e-13), (1. + 5e-13, -1e-13)])
def test_domain_canonicalization_cannot_make_a_zero_extent_chart(start, sweep):
    with pytest.raises(GeometryError, match="extent"):
        BezierQuadricCurve(DIRECTRIX, quadric(constant=-1.), start, sweep)


def test_regular_endpoint_near_simple_fold_does_not_use_node_jet():
    # Delta=4(t+offset): the radical is small enough for the old residual-only
    # node test, but the actual fold lies outside the directrix domain.
    offset, sweep = 1e-16, .5
    curve = BezierQuadricCurve(DIRECTRIX, quadric(linear=-1., constant=-offset), 0., sweep)
    np.testing.assert_allclose(curve.derivative(0.), (sweep, 0., sweep / (2 * np.sqrt(offset))), rtol=1e-14)
