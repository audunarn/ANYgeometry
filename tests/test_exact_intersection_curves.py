"""Analytic curve oracles independent of witness-based reconstruction."""
from copy import deepcopy
from decimal import Decimal, localcontext
import json
import hashlib
import numpy as np
import pytest

from anygeometry import (EllipticArc, CylinderIntersectionCurve, Cylinder,
                         GeometryModel, GeometryError, EntityRef, copy_entities,
                         to_dict, from_dict)


@pytest.mark.parametrize('angle',(0.,np.pi))
@pytest.mark.parametrize('sweep',(-np.pi/4,np.pi/4))
def test_double_transition_seam_uses_the_certified_one_sided_jet(angle,sweep):
    first=Cylinder((0,0,-1.5),(0,0,1),(1,0,0),1,3)
    second=Cylinder((0,-1.5,0),(0,1,0),(1,0,0),1,3)
    for branch in (-1,1):
        curve=CylinderIntersectionCurve(first,second,angle,sweep,branch)
        expected=np.asarray((-np.sin(angle)*sweep,np.cos(angle)*sweep,branch*abs(sweep)))
        assert curve.derivative(0.)==pytest.approx(expected,abs=1e-13)
        assert curve.second_derivative(0.)==pytest.approx(
            (-np.cos(angle)*sweep*sweep,-np.sin(angle)*sweep*sweep,0.),abs=1e-13)
        # Independent analytic solution is x^2+y^2=x^2+z^2=1.
        for parameter in (1e-12,1e-6,.5,1.):
            theta=angle+sweep*parameter
            assert curve.evaluate(parameter)==pytest.approx(
                (np.cos(theta),np.sin(theta),branch*abs(np.sin(theta))),abs=1e-13)


def model_curve(curve):
    geometry = GeometryModel()
    start, end = geometry.add_points(curve.evaluate(np.asarray((0., 1.))))
    edge = geometry.add_curve(start, end, curve)
    return geometry, edge


def test_ellipse_evaluation_derivative_bounds_split_and_serialization():
    curve = EllipticArc((3., 2., -1.), (2., 0., 0.), (.5, 1., 0.), -.3, 2.1)
    g, edge = model_curve(curve)
    t = np.linspace(0., 1., 101)
    angle = -.3+2.1*t
    expected = np.column_stack((3+2*np.cos(angle)+.5*np.sin(angle), 2+np.sin(angle), -np.ones(len(t))))
    np.testing.assert_allclose(g.sample_edge(edge, t), expected, atol=2e-15)
    lo, hi = curve.bounds()
    assert np.all(expected >= lo) and np.all(expected <= hi)
    expected_d = 2.1*np.column_stack((-2*np.sin(angle)+.5*np.cos(angle), np.cos(angle), np.zeros(len(t))))
    np.testing.assert_allclose(curve.derivative(t), expected_d, atol=2e-15)
    original = to_dict(g)
    restored = from_dict(json.loads(json.dumps(original)))
    assert to_dict(restored) == original
    assert original['version'] == 5
    _, (left, right) = g.split_edge(edge, .37)
    np.testing.assert_allclose(g.sample_edge(left, t), curve.evaluate(.37*t))
    np.testing.assert_allclose(g.sample_edge(right, t), curve.evaluate(.37+.63*t))
    assert g.validate_topology() == ()


def test_global_ellipse_projection_and_affine_copy():
    curve = EllipticArc((0., 0., 0.), (2., 0., 0.), (0., 1., 0.), 0., 1.5)
    g, edge = model_curve(curve)
    point = curve.evaluate(.327)
    made, t, distance = g.closest_edge_point(edge, point)
    assert distance <= g.tolerance.length
    np.testing.assert_allclose(made, point, atol=g.tolerance.length)
    assert t == pytest.approx(.327, abs=1e-9)
    matrix = np.asarray(((1., .3, 0., 4.), (0., -2., 0., 1.), (0., 0., 1., 2.), (0., 0., 0., 1.)))
    inserted = copy_entities(g, (EntityRef('edge', edge),), matrix=matrix)
    copied = inserted.entity_map[EntityRef('edge', edge)].id
    parameters = np.linspace(0, 1, 17)
    np.testing.assert_allclose(g.sample_edge(copied, parameters), curve.evaluate(parameters) @ matrix[:3,:3].T+matrix[:3,3])
    assert g.validate_topology() == ()


@pytest.mark.parametrize('start,sweep', ((-.7, 2*np.pi), (2.3, -4.1), (.2, .8)))
def test_circle_axis_projection_has_constant_distance_without_box_refinement(monkeypatch,start,sweep):
    curve=EllipticArc((2.,-3.,4.),(2.,0.,0.),(0.,2.,0.),start,sweep)
    def no_refinement(*args,**kwargs):
        pytest.fail('constant-distance ellipse projection must use its analytic derivative')
    monkeypatch.setattr(EllipticArc,'bounds',no_refinement)
    # closest_edge_point first asks for a whole-edge scale, independently of
    # the projection. Exercise the projection helper directly here.
    from anygeometry.exact_curves import project_analytic_curve
    made,parameter,distance=project_analytic_curve(curve,(2.,-3.,7.),1e-9)
    np.testing.assert_allclose(made,curve.evaluate(0.),atol=1e-14)
    assert parameter==0.
    assert distance==pytest.approx(np.sqrt(13.),abs=1e-14)


@pytest.mark.parametrize('start,sweep', ((0.,2*np.pi),(2.7,-5.8),(1.,.2)))
def test_ellipse_projection_includes_all_stationary_points_and_endpoints(start,sweep):
    from anygeometry.exact_curves import project_analytic_curve
    curve=EllipticArc((0.,0.,0.),(2.,0.,0.),(0.,1.,0.),start,sweep)
    # From the ellipse centre the global minima are exactly its two minor-axis
    # points when included, otherwise an endpoint of the requested arc.
    candidates=[0.,1.]
    for angle in (-3*np.pi/2,-np.pi/2,np.pi/2,3*np.pi/2,5*np.pi/2):
        parameter=(angle-start)/sweep
        if 0<=parameter<=1:candidates.append(parameter)
    expected=min(np.hypot(2*np.cos(start+sweep*t),np.sin(start+sweep*t)) for t in candidates)
    made,parameter,distance=project_analytic_curve(curve,(0.,0.,3.),1e-9)
    assert distance==pytest.approx(np.hypot(expected,3.),abs=1e-12)
    assert made==pytest.approx(curve.evaluate(parameter),abs=1e-14)


@pytest.mark.parametrize('branch', (-1, 1))
def test_cylinder_branch_decimal_equations_derivative_and_codec(branch):
    first = Cylinder((0,0,0), (0,0,1), (1,0,0), 1., 4.)
    second = Cylinder((0,0,0), (0,1,0), (1,0,0), 2., 4.)
    curve = CylinderIntersectionCurve(first, second, .2, 1.1, branch)
    g, edge = model_curve(curve)
    t = np.linspace(0., 1., 31)
    points = g.sample_edge(edge, t)
    # First is x^2+y^2=1, second is x^2+z^2=4. Decimal
    # checks the stored world coordinates, not the implementation's residual.
    with localcontext() as ctx:
        ctx.prec = 60
        for x,y,z in points:
            x,y,z = map(lambda v: Decimal.from_float(float(v)), (x,y,z))
            assert abs(x*x+y*y-1) < Decimal('1e-14')
            assert abs(x*x+z*z-4) < Decimal('1e-14')
    expected_z = branch*np.sqrt(4-np.cos(.2+1.1*t)**2)
    np.testing.assert_allclose(points[:,2], expected_z, atol=1e-14)
    derivative = curve.derivative(t)
    expected_dz = branch*1.1*np.cos(.2+1.1*t)*np.sin(.2+1.1*t)/np.sqrt(4-np.cos(.2+1.1*t)**2)
    np.testing.assert_allclose(derivative[:,2], expected_dz, atol=1e-14)
    for a,b in ((0,1), (.17,.53), (.9,.99)):
        lo,hi=curve.bounds(a,b)
        samples=curve.evaluate(np.linspace(a,b,101))
        assert np.all(samples >= lo) and np.all(samples <= hi)
    doc=to_dict(g)
    assert to_dict(from_dict(json.loads(json.dumps(doc)))) == doc


def test_regular_discriminant_endpoint():
    first=Cylinder((0,0,0),(0,0,1),(1,0,0),2.,4.)
    second=Cylinder((0,0,0),(0,1,0),(1,0,0),1.,4.)
    curve=CylinderIntersectionCurve(first,second,np.pi/3,np.pi/6,1,'left_square')
    assert np.all(np.isfinite(curve.evaluate(np.asarray((0., .5, 1.)))))
    assert np.all(np.isfinite(curve.derivative(np.asarray((0., .5, 1.)))))
    assert np.linalg.norm(curve.derivative(0.)) > 0
    reversed_curve = curve.subcurve(1., 0.)
    np.testing.assert_allclose(reversed_curve.evaluate(1.), curve.evaluate(0.))
    assert np.linalg.norm(reversed_curve.derivative(1.)) > 0
    with pytest.raises(GeometryError, match='leaves the real'):
        CylinderIntersectionCurve(first,second,0.,np.pi,1)


@pytest.mark.parametrize('branch', (-1, 1))
def test_double_discriminant_endpoint_has_one_sided_analytic_tangent(branch):
    first=Cylinder((0,0,0),(0,0,1),(1,0,0),1.,4.)
    second=Cylinder((0,0,0),(0,1,0),(1,0,0),1.,4.)
    curve=CylinderIntersectionCurve(first,second,0.,np.pi/2,branch)
    np.testing.assert_allclose(curve.derivative(0.), (0.,np.pi/2,branch*np.pi/2), atol=1e-14)
    reversed_curve=curve.subcurve(1.,0.)
    np.testing.assert_allclose(reversed_curve.derivative(1.), -curve.derivative(0.), atol=1e-14)


def test_schema_four_still_reads_all_semantics_and_refuses_new_curves():
    g=GeometryModel(); g.add_plate(g.add_points(((0,0,0),(1,0,0),(1,1,0),(0,1,0))))
    doc=to_dict(g); doc['version']=4
    encoded=json.dumps({k:v for k,v in doc.items() if k != 'checksum'},sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
    doc['checksum']={'algorithm':'sha256','value':hashlib.sha256(encoded).hexdigest()}
    restored=from_dict(doc)
    assert restored.model_id == g.model_id
    assert restored.revision == g.revision
    assert restored.id_state() == g.id_state()
    g,_=model_curve(EllipticArc((0,0,0),(2,0,0),(0,1,0),0,1))
    doc=to_dict(g); doc['version']=4
    with pytest.raises(GeometryError,match='require schema 5'):
        from_dict(doc)


def test_invalid_inputs_and_endpoint_edit_are_atomic():
    with pytest.raises(GeometryError):EllipticArc((0,0,0),(1,0,0),(2,0,0))
    g,edge=model_curve(EllipticArc((0,0,0),(2,0,0),(0,1,0),0,1))
    before=deepcopy(to_dict(g))
    with pytest.raises(GeometryError):g.move_point(g.edges[edge].start,9,9,9)
    assert to_dict(g)==before
    with pytest.raises(GeometryError):g.evaluate_edge_many(edge,(np.nan,))
