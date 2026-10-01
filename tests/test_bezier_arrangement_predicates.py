import math
import numpy as np
import pytest
from anygeometry import Cylinder,CylinderIntersectionCurve,GeometryError
from anygeometry.arrangement_geometry import BezierPath,curve_junctions


def _parabola():
    return BezierPath(((0.,.25,0.),(.5,-.25,0.),(1.,.25,0.)))


def test_polynomial_elimination_keeps_crossings_tangencies_and_spatial_residuals():
    first=_parabola()
    second=BezierPath(((0.,-.17,0.),(.5,.33,0.),(1.,-.17,0.)))
    pairs=curve_junctions(first,second)
    assert sorted(t for t,s in pairs)==pytest.approx((.3,.7),abs=1e-13)
    for t,s in pairs:
        np.testing.assert_allclose(first.evaluate(t),second.evaluate(s),atol=1e-13)
    tangent=BezierPath(((0.,0.,0.),(1.,0.,0.)))
    pairs=curve_junctions(first,tangent)
    np.testing.assert_allclose(pairs,((.5,.5),),rtol=0,atol=1e-13)
    raised=BezierPath(((0.,-.17,1.),(.5,.33,1.),(1.,-.17,1.)))
    assert curve_junctions(first,raised)==()
    with pytest.raises(GeometryError,match='cancelled'):
        curve_junctions(first,second,cancellation_check=lambda:True)


def test_common_polynomial_interval_is_qualified_over_its_full_definition():
    first=_parabola()
    for second,expected in ((first.subcurve(.25,.75),((.25,0.),(.75,1.))),
                            (first.subcurve(.75,.25),((.25,1.),(.75,0.)))):
        np.testing.assert_allclose(curve_junctions(first,second),expected,rtol=0,atol=1e-13)


def test_bezier_cylinder_branch_uses_both_implicit_supports():
    first=Cylinder((0,0,0),(0,0,1),(1,0,0),1,4)
    second=Cylinder((0,0,0),(0,1,0),(1,0,0),1,4)
    branch=CylinderIntersectionCurve(first,second,.1,1.3,1)
    height=math.sqrt(.5)
    bezier=BezierPath(((-2.,height,height),(2.,height,height)))
    pairs=curve_junctions(bezier,branch)
    assert len(pairs)==1
    t,s=pairs[0]
    assert t==pytest.approx((2+height)/4,abs=1e-13)
    assert s==pytest.approx((math.pi/4-.1)/1.3,abs=1e-13)
    np.testing.assert_allclose(bezier.evaluate(t),branch.evaluate(s),atol=1e-13)
