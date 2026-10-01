import math
import numpy as np
import pytest
from anygeometry import Cylinder, CylinderIntersectionCurve, EllipticArc, GeometryError
from anygeometry.arrangement_geometry import plane_roots, curve_junctions
from anygeometry.batch_intersections import _coincident


def _curve(branch=1):
    first=Cylinder((0,0,-2),(0,0,1),(1,0,0),1,4)
    second=Cylinder((0,-2,0),(0,1,0),(1,0,0),1,4)
    return CylinderIntersectionCurve(first,second,0.,math.pi/2,branch)


def test_exact_planar_branch_factor_and_coincident_ellipse():
    curve=_curve()
    assert plane_roots(curve,(0,1,-1),0.,tolerance=1e-10) is None
    assert plane_roots(curve,(0,1,1),0.,tolerance=1e-10)==pytest.approx((0.,))
    ellipse=EllipticArc((0,0,0),(1,0,0),(0,1,1),0.,math.pi/2)
    assert _coincident(curve,ellipse,1e-10)
    hits=curve_junctions(curve,ellipse,tolerance=1e-10)
    assert any(a==pytest.approx(0) and b==pytest.approx(0) for a,b in hits)
    assert any(a==pytest.approx(1) and b==pytest.approx(1) for a,b in hits)
    np.testing.assert_allclose(curve.evaluate([0,.25,.5,.75,1]),ellipse.evaluate([0,.25,.5,.75,1]),atol=1e-14)


def test_public_branch_chart_rejects_internal_branch_transitions():
    curve=_curve()
    with pytest.raises(GeometryError,match='split.*discriminant transition'):
        CylinderIntersectionCurve(curve.first,curve.second,0.,math.tau,1)
