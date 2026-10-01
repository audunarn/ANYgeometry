"""Independent analytic checks for planar material Green integrals."""
from fractions import Fraction
import math

import numpy as np
import pytest
from anygeometry import EllipticArc
from anygeometry.arrangement_geometry import LinePath,BezierPath
from anygeometry.material_arrangement import MaterialDomain,ArrangementPath
from anygeometry.surfaces import Plane


def area(curves,support=None,tolerance=1e-25):
    support=support or Plane((0.,0.,0.),(1.,0.,0.),(0.,1.,0.))
    return MaterialDomain(1,support,()).area_loop(tuple(ArrangementPath(curve) for curve in curves),tolerance)


@pytest.mark.parametrize('sweep',(.125,1e-4,-.125,-1e-4))
def test_small_circular_caps_match_an_independent_rational_series(sweep):
    arc=EllipticArc((0.,0.,0.),(1.,0.,0.),(0.,1.,0.),0.,sweep)
    chord=LinePath(tuple(arc.evaluate(1.)),tuple(arc.evaluate(0.)))
    # Direct rational expansion of (theta-sin(theta))/2, independent of the
    # implementation's decimal primitive and trigonometric argument reduction.
    theta=Fraction(sweep)
    reference=sum(((-1)**(index+1)*theta**(2*index+1)/math.factorial(2*index+1)
                   for index in range(1,40)),Fraction(0))/2
    assert area((arc,chord))==pytest.approx(float(reference),rel=2e-15,abs=0.)


@pytest.mark.parametrize('shift',((0.,0.,0.),(1e9,-1e9,10.)))
def test_full_ellipse_translation_shear_reflection_and_reversal(shift):
    arc=EllipticArc(shift,(2.,1.,0.),(-1.,3.,0.),0.,math.tau)
    support=Plane(shift,(2.,0.,0.),(1.,1.,0.))
    expected=7*math.pi/2
    forward=area((arc,),support)
    reverse=area((arc.subcurve(1.,0.),),support)
    assert forward==pytest.approx(expected,rel=1e-15)
    assert reverse==-forward
    reflected=EllipticArc(shift,(2.,1.,0.),(1.,-3.,0.),0.,math.tau)
    assert area((reflected,),support)==-forward


def test_polynomial_bezier_area_uses_exact_affine_projection(monkeypatch):
    support=Plane((2.,3.,0.),(2.,0.,0.),(1.,1.,0.))
    curve=BezierPath(((2.,3.,0.),(5.,4.,0.),(6.,3.,0.)))
    chord=LinePath(curve.controls[-1],curve.controls[0])
    def forbidden(*args,**kwargs):
        raise AssertionError('analytic planar area must not use quadrature or least squares')
    monkeypatch.setattr(np.polynomial.legendre,'leggauss',forbidden)
    monkeypatch.setattr(np.linalg,'lstsq',forbidden)
    # In the plane basis x=2t and y=2t(1-t); integral y dx = 2/3.
    assert area((curve,chord),support)==-float(Fraction(2,3))
    assert area((curve.subcurve(1.,0.),chord.subcurve(1.,0.)),support)==float(Fraction(2,3))
