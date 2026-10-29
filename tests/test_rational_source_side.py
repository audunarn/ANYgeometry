from dataclasses import replace
from fractions import Fraction as Q
import pytest
from anygeometry import GeometryError
from anygeometry.nurbs import NurbsCurve,CurveOnSurface
from anygeometry.regular_source_reference import (source_side,source_node,SourceSide,
    RationalSourceSide,parameter_transport_bound)
from test_local_curve_geometry import flat


def side(pcurve,a=Q(0),b=Q(1)):
    curve=CurveOnSurface(flat(),pcurve)
    result=source_side(curve,(a,b),source_node(curve,a),source_node(curve,b))
    assert type(result) is RationalSourceSide
    return result


def parabola(domain=(0.,1.)):
    return NurbsCurve(2,(0.,0.,0.,1.,1.,1.),((0.,0.),(.5,0.),(1.,1.)),domain=domain)


@pytest.mark.parametrize('domain',[(0.,1.),(.125,.875),(.875,.125)])
@pytest.mark.parametrize('a,b',[(Q(0),Q(1)),(Q(1,4),Q(3,4)),(Q(3,4),Q(1,4))])
def test_exact_point_and_derivative_crop_reverse_zero_width(domain,a,b):
    result=side(parabola(domain),a,b)
    c,d=map(Q,domain);speed=(d-c)*(b-a)
    for t in (Q(0),Q(1,3),Q(1,2),Q(1)):
        u=c+(d-c)*(a+(b-a)*t)
        assert result.box(t,t)==((u,u),(u*u,u*u))
        assert result.derivative(t,t)==((speed,speed),(2*u*speed,2*u*speed))
    assert parameter_transport_bound(result)==0
    for lo,hi in ((Q(0),Q(1)),(Q(1,8),Q(7,8))):
        box=result.box(lo,hi);derivative=result.derivative(lo,hi)
        for t in (lo,(lo+hi)/2,hi):
            point=result.box(t,t);jet=result.derivative(t,t)
            assert all(x<=v[0]<=y for (x,y),v in zip(box,point))
            assert all(x<=v[0]<=y for (x,y),v in zip(derivative,jet))


def test_positive_rational_weights_exact_quotient_endpoint_and_centre():
    result=side(NurbsCurve(2,(0.,0.,0.,1.,1.,1.),((0.,0.),(.5,1.),(1.,0.)),weights=(1.,2.,1.)))
    assert result.box(Q(1,2),Q(1,2))==((Q(1,2),Q(1,2)),(Q(2,3),Q(2,3)))
    assert result.derivative(Q(1,2),Q(1,2))==((Q(2,3),Q(2,3)),(Q(0),Q(0)))
    assert result.derivative(Q(0),Q(0))==((Q(2),Q(2)),(Q(4),Q(4)))
    assert result.derivative(Q(1),Q(1))==((Q(2),Q(2)),(Q(-4),Q(-4)))


def test_c0_knot_uses_both_one_sided_derivatives_and_complete_range():
    result=side(NurbsCurve(1,(0.,0.,.5,1.,1.),((0.,0.),(.5,.5),(1.,0.))))
    assert result.box(Q(1,2),Q(1,2))==((Q(1,2),Q(1,2)),(Q(1,2),Q(1,2)))
    assert result.derivative(Q(1,2),Q(1,2))==((Q(1),Q(1)),(Q(-1),Q(1)))
    assert result.box(Q(0),Q(1))==((Q(0),Q(1)),(Q(0),Q(1,2)))


def test_endpoint_identity_mutation_and_native_cache_refuse():
    result=side(parabola())
    with pytest.raises(GeometryError,match='endpoint'):
        source_side(result.curve,result.parameters,replace(result.start,identity=('foreign',)),result.end)
    object.__setattr__(result.curve.pcurve,'controls',((0.,0.),(.5,.25),(1.,1.)))
    with pytest.raises(GeometryError):result.box(Q(0),Q(1))
    result=side(parabola())
    result.curve.pcurve._point_values[b'\x00'*8]=(9.,9.)
    with pytest.raises(GeometryError):result.derivative(Q(0),Q(1))


def test_existing_straight_source_side_route_is_unchanged():
    curve=CurveOnSurface(flat(),NurbsCurve(1,(0.,0.,1.,1.),((0.,0.),(1.,1.)),weights=(1.,2.)))
    result=source_side(curve,(Q(0),Q(1)),source_node(curve,0),source_node(curve,1))
    assert type(result) is SourceSide and result.curve is None


def test_omitted_incident_face_restriction_is_hard_refusal(monkeypatch):
    from types import SimpleNamespace as NS
    import anygeometry.local_approximation_field as module
    from test_local_approximation_field import fixture
    _,prepared,_,_,_,allocation=fixture()
    query=module.prepare_local_boundary_limit_query(prepared,allocation)
    monkeypatch.setattr(type(prepared),'faces_using_edge',lambda self,edge:(987654321,))
    with pytest.raises(GeometryError,match='incident'):
        query(next(iter(prepared.edges)),Q(0),Q(1))


def test_rational_side_uses_real_point_witness_and_whole_fan_proof():
    from anygeometry._source_cell_point_error import fan_point_error_lower_squared
    from anygeometry.curvilinear_reference_maps import certify_fan_subtriangle
    result=side(parabola());support=result.curve.support
    xyz=((0.,0.,0.),(1.,1.,0.));centre=(Q(1,2),Q(3,4));centre_xyz=(*centre,Q(0))
    checks=[]
    lower=fan_point_error_lower_squared(support,result,centre,xyz,centre_xyz,check=lambda:checks.append(True))
    assert lower is not None and 0<lower<=Q(1,16) and len(checks)>=2
    bound,pieces=certify_fan_subtriangle(support,result,centre,xyz,centre_xyz,
        orientation=1,surface_error=Q(3,10))
    assert 0<bound<=Q(3,10) and pieces>0


def test_curved_cos_loop_source_cell_qualification_and_foreign_support_refusal():
    from anygeometry import (GeometryModel,CurvilinearMaterialCell,query_material_surface_regions,
        query_prepared_material_boundary_chords,validate_prepared_material_curvilinear_cells,
        validate_prepared_material_curvilinear_cells_binding)
    from anygeometry.source_cell_complex import _boundary_proof
    model=GeometryModel();support=flat()
    points=((.75,.5),(.5,.75),(.25,.5),(.5,.25))
    corners=((.75,.75),(.25,.75),(.25,.25),(.75,.25))
    curves=tuple(CurveOnSurface(support,NurbsCurve(2,(0.,0.,0.,1.,1.,1.),
        (points[i],corners[i],points[(i+1)%4]))) for i in range(4))
    vertices=tuple(model.add_point(*curve.exact_point(0)) for curve in curves)
    edges=tuple(model.add_curve(vertices[i],vertices[(i+1)%4],curve) for i,curve in enumerate(curves))
    face=model.add_face(edges,surface=support)
    regions=query_material_surface_regions(model,(face,))
    chords=query_prepared_material_boundary_chords(model,regions,
        tuple((model.handle('edge',edge),0.,1.) for edge in edges),chord_error=.2)
    handle=model.handle('face',face);centre=(.5,.5);centre_xyz=(.5,.5,0.)
    cells=tuple(CurvilinearMaterialCell(handle,(i,(i+1)%4,4),(points[i],points[(i+1)%4],centre),
        ((*points[i],0.),(*points[(i+1)%4],0.),centre_xyz),((0,i,True),)) for i in range(4))
    receipt=validate_prepared_material_curvilinear_cells(model,regions,chords,cells,surface_error=.2)
    validate_prepared_material_curvilinear_cells_binding(model,regions,receipt)
    assert len(receipt.cells)==4 and all(0<row[-1]<=.2 for row in receipt.source_map_bounds)
    other=replace(support,u_range=(0.,.5))
    with pytest.raises(GeometryError,match='polynomial boundary family is not qualified'):
        _boundary_proof(model,model.faces[face],other)
