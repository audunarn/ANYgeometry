from decimal import Decimal, localcontext
import math

import numpy as np
import pytest

from anygeometry import Cylinder, CylinderIntersectionCurve
from anygeometry.arrangement_geometry import curve_junctions
from anygeometry.cylinder_curve_events import cylinder_roots


def test_three_support_joint_isolated_without_a_sample_oracle():
    common = Cylinder((0,0,0),(0,0,1),(1,0,0),1,5)
    first = Cylinder((0,0,0),(0,1,0),(1,0,0),2,5)
    second = Cylinder((0,0,0),(1,0,0),(0,1,0),2,5)
    a = CylinderIntersectionCurve(common,first,.2,1.2,1)
    b = CylinderIntersectionCurve(common,second,.2,1.2,1)
    hits = curve_junctions(a,b)
    assert len(hits) == 1
    ta,tb=hits[0]
    point=a.evaluate(ta)
    np.testing.assert_allclose(point,b.evaluate(tb),atol=1e-13)
    with localcontext() as context:
        context.prec=60
        expected=np.array([float(Decimal('.5').sqrt()),float(Decimal('.5').sqrt()),float(Decimal('3.5').sqrt())])
    np.testing.assert_allclose(point,expected,atol=1e-13)
    assert ta == pytest.approx((math.pi/4-.2)/1.2,abs=1e-13)


def test_support_identity_is_independent_of_native_origin_and_height():
    common=Cylinder((0,0,0),(0,0,1),(1,0,0),1,5)
    other=Cylinder((0,0,0),(0,1,0),(1,0,0),2,5)
    curve=CylinderIntersectionCurve(common,other,.2,1.2,1)
    equivalent=Cylinder((0,3,0),(0,-1,0),(1,0,0),2,2)
    assert cylinder_roots(curve,equivalent) is None
    changed=Cylinder((0,0,0),(0,1,0),(1,0,0),3,5)
    assert cylinder_roots(curve,changed) == ()


def test_equal_radius_double_transition_retains_both_small_branches():
    first=Cylinder((0,0,0),(0,0,1),(1,0,0),1,4)
    second=Cylinder((0,0,0),(0,1,0),(1,0,0),1,4)
    parameters=np.array((0.,1e-12,1e-10,1e-8,1e-6,.1,1.))
    for branch in (-1,1):
        curve=CylinderIntersectionCurve(first,second,0.,math.pi/4,branch)
        angles=parameters*math.pi/4
        np.testing.assert_allclose(curve.evaluate(parameters)[:,2],branch*np.sin(angles),
                                   rtol=2e-15,atol=1e-16)
        np.testing.assert_allclose(curve.derivative(parameters)[:,2],branch*math.pi/4*np.cos(angles),
                                   rtol=2e-15,atol=1e-15)


def test_second_derivative_implicit_equations_and_regular_endpoint_jet():
    first=Cylinder((0,0,0),(0,0,1),(1,0,0),1,4)
    second=Cylinder((0,0,0),(0,1,0),(1,0,0),2,4)
    for branch in (-1,1):
        curve=CylinderIntersectionCurve(first,second,.2,1.1,branch)
        t=np.linspace(0,1,11)
        angle=.2+1.1*t
        radius=np.sqrt(4-np.cos(angle)**2)
        expected=branch*1.1**2*((np.cos(angle)**2-np.sin(angle)**2)/radius
                  -(np.cos(angle)*np.sin(angle))**2/radius**3)
        np.testing.assert_allclose(curve.second_derivative(t)[:,2],expected,atol=1e-14)
    large=Cylinder((0,0,0),(0,0,1),(1,0,0),2,4)
    small=Cylinder((0,0,0),(0,1,0),(1,0,0),1,4)
    curve=CylinderIntersectionCurve(large,small,math.pi/3,math.pi/6,1,'left_square')
    expected=(math.pi/3)*np.array((-math.sqrt(3),1.,0.))
    np.testing.assert_allclose(curve.second_derivative(0.),expected,atol=1e-14)
