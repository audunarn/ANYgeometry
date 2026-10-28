"""Exact trace restrictions are enclosures, never surface regularity authority."""
from dataclasses import replace
from fractions import Fraction as Q
import pytest
from anygeometry import GeometryError,GeometryModel,Plane
from anygeometry.nurbs import NurbsSurface
from anygeometry.nurbs_metric import derivative_intervals,_trace_derivative_intervals,certify_projected_chart


def parabola():
    return NurbsSurface(2,1,(0.,0.,0.,1.,1.,1.),(0.,0.,1.,1.),
        tuple(tuple((u,v,z) for v in (0.,1.)) for u,z in ((0.,0.),(.5,0.),(1.,1.))))


def test_line_restriction_differentiates_before_collapsing_coordinate():
    surface=parabola();box=((Q(1,4),Q(3,4)),(Q(1,2),Q(1,2)))
    with pytest.raises(GeometryError,match='positive area'):derivative_intervals(surface,box)
    assert _trace_derivative_intervals(surface,box)==(((1,1),(0,0),(Q(1,2),Q(3,2))),((0,0),(1,1),(0,0)))
    assert _trace_derivative_intervals(surface,((0,1),(0,1)))==derivative_intervals(surface,((0,1),(0,1)))


@pytest.mark.parametrize('domain,factor',(((0.,1.),Q(1)),((1.,0.),Q(-1)),((.75,.25),Q(-1,2))))
def test_exact_rational_point_and_reversed_cropped_chart(domain,factor):
    surface=NurbsSurface(1,1,(0.,0.,1.,1.),(0.,0.,1.,1.),
        (((0.,0.,0.),(0.,1.,0.)),((1.,0.,0.),(1.,1.,0.))),weights=((2.,2.),(1.,1.)),u_range=domain)
    du,dv=_trace_derivative_intervals(surface,((Q(1,2),Q(1,2)),(Q(1,2),Q(1,2))))
    assert du==((factor*Q(8,9),factor*Q(8,9)),(0,0),(0,0))
    assert dv==((0,0),(1,1),(0,0))


def test_internal_knot_includes_both_one_sided_derivatives():
    surface=NurbsSurface(1,1,(0.,0.,.5,1.,1.),(0.,0.,1.,1.),
        tuple(((x,0.,0.),(x,1.,0.)) for x in (0.,.5,2.)))
    du,dv=_trace_derivative_intervals(surface,((Q(1,2),Q(1,2)),(Q(1,3),Q(2,3))))
    assert du==((1,3),(0,0),(0,0)) and dv==((0,0),(1,1),(0,0))


def test_degenerate_enclosure_does_not_qualify_singular_surface():
    surface=NurbsSurface(1,1,(0.,0.,1.,1.),(0.,0.,1.,1.),(((0.,0.,0.),)*2,)*2)
    assert _trace_derivative_intervals(surface,((0,0),(1,1)))==(((0,0),)*3,)*2
    with pytest.raises(GeometryError,match='immersion'):certify_projected_chart(surface)


@pytest.mark.parametrize('box',((((-1,-1),(0,1))),(((1,0),(0,0))),(((float('nan'),0),(0,0)))))
def test_invalid_trace_domain_refuses(box):
    with pytest.raises(GeometryError):_trace_derivative_intervals(parabola(),box)


@pytest.fixture
def mixed_loop():
    from anygeometry.support_curves import PlaneNurbsCurve
    from anygeometry.entities import OrientedEdge
    from anygeometry.planar_exact import ExactPlane
    surface=parabola();plane=Plane((0.,.5,0.),(1.,0.,0.),(0.,0.,1.))
    curve=PlaneNurbsCurve.from_plane(plane,surface)
    model=GeometryModel()
    points=model.add_points(tuple(tuple(float(x) for x in row[2]) for row in curve._exact_points((0,1))))
    curved=model.add_curve(*points,curve);line=model.add_line(points[1],points[0])
    return model,(OrientedEdge(curved,True),OrientedEdge(line,True)),ExactPlane.from_plane(plane)


def test_authentic_plane_section_mixed_loop_area_with_zero_width_chart(mixed_loop):
    from anygeometry.plane_section_plate import loop_area
    from anygeometry.regular_source_reference import source_node,source_side
    model,loop,frame=mixed_loop;curve=model.edges[loop[0].edge].curve
    nodes=tuple(source_node(curve,t) for t in (Q(0),Q(1)))
    chart=source_side(curve,(Q(0),Q(1)),*nodes).box(Q(0),Q(1))
    assert chart[1][0]==chart[1][1]
    result=loop_area(model,loop,frame,lambda:None)
    assert result[0]>0 or result[1]<0
    expected=Q(1,6) if result[0]>0 else Q(-1,6)
    assert result[0]<=expected<=result[1]


def test_authentic_trace_area_preserves_callback_exception(mixed_loop):
    from anygeometry.plane_section_plate import loop_area
    token=GeometryError('original area cancellation')
    def stop():raise token
    with pytest.raises(GeometryError) as caught:loop_area(*mixed_loop,stop)
    assert caught.value is token
