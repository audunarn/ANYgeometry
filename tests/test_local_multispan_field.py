from dataclasses import replace
from fractions import Fraction as Q
import pytest
import anygeometry as owner
from anygeometry.nurbs import NurbsSurface,NurbsCurve,CurveOnSurface
from anygeometry._local_curvature import LocalApproximationNotCertified
from anygeometry.local_approximation_field import (query_local_approximation_field,
    validate_local_approximation_field,bind_local_approximation_field,prepare_local_cell_limit_query)
from anygeometry._local_piecewise_surface_star import SpanBudget,surface_span_atlas,span_hull
from anygeometry._local_spatial_clearance import SpatialBudget,_HullMemo,_hull_data
from test_local_piecewise_surface_star import inserted_surface


def model_surface(surface=None):
    model=owner.GeometryModel()
    surface=surface or inserted_surface(Q(0))
    # Authored trim intervals retain the actual support-knot feature anchors.
    a,b=map(Q,surface.u_range)
    cuts=sorted({Q(0),Q(1),*((Q(k)-a)/(b-a) for k in surface.knots_u if 0<(Q(k)-a)/(b-a)<1)})
    uv=tuple((float(u),0.) for u in cuts)+tuple((float(u),1.) for u in reversed(cuts))
    curves=tuple(CurveOnSurface(surface,NurbsCurve(1,(0.,0.,1.,1.),(uv[i],uv[(i+1)%len(uv)]))) for i in range(len(uv)))
    vertices=tuple(model.add_point(*curve.exact_point(0)) for curve in curves)
    edges=tuple(model.add_curve(vertices[i],vertices[(i+1)%len(vertices)],curve) for i,curve in enumerate(curves))
    face=model.add_face(edges,surface=surface);model.add_sheet((face,))
    return model,face


def field_query(model,face,**kwargs):
    return query_local_approximation_field(model,(face,),relative_accuracy=.001,scale_error=.001,
        policy_floor=1e-8,**kwargs)


@pytest.mark.parametrize('warp',(Q(0),Q(1,128)))
def test_real_multispan_field_issued_with_complete_bound_span_certificate(warp):
    model,face=model_surface(inserted_surface(warp))
    before=owner.to_dict(model)
    field=field_query(model,face)
    assert len(field.span_certificates)==1
    f,source,atlas,boxes,star=field.span_certificates[0]
    assert f==face and source and atlas
    assert boxes==(((Q(0),Q(1,2)),(Q(0),Q(1))),((Q(1,2),Q(1)),(Q(0),Q(1))))
    assert star[3]>0 and star[-1]==((0,0),(0,1),(1,0),(1,1))
    validate_local_approximation_field(model,field)
    assert owner.to_dict(model)==before
    object.__setattr__(field,'span_certificates',())
    with pytest.raises(owner.GeometryError):validate_local_approximation_field(model,field)


def test_complete_cross_knot_competitor_hull_and_all_piece_directions():
    surface=inserted_surface(Q(1,8))
    atlas=surface_span_atlas(surface,budget=SpanBudget())
    budget=SpatialBudget(lambda:None);memo=_HullMemo({id(surface):atlas})
    full=span_hull(atlas,((0,1),(0,1)),take=budget.take)
    box=((Q(1,4),Q(3,4)),(Q(1,8),Q(7,8)))
    hull,directions=_hull_data(surface,box,full,budget,memo)
    for u in (Q(1,4),Q(1,2),Q(3,4)):
        for v in (Q(1,8),Q(1,2),Q(7,8)):
            point=surface.exact_point(u,v)
            assert all(h.lo<=x<=h.hi for x,h in zip(point,hull))
            assert all(lo<=sum(q[k]*point[k] for k in range(3))<=hi for q,lo,hi in directions)


def test_changed_span_implementation_invalidates_issued_field(monkeypatch):
    model,face=model_surface();field=field_query(model,face)
    import anygeometry._local_piecewise_surface_star as spans
    original=spans._c1_seam
    monkeypatch.setattr(spans,'_c1_seam',lambda *args:original(*args))
    with pytest.raises(owner.GeometryError,match='context|runtime'):
        validate_local_approximation_field(model,field)


def test_callback_mutates_source_no_multispan_field_issued():
    model,face=model_surface();surface=model.faces[face].surface
    def mutate(phase):
        if phase=='local approximation field':object.__setattr__(surface,'u_range',(.125,.875))
    with pytest.raises(owner.GeometryError):field_query(model,face,cancellation_check=mutate)


def test_actual_public_multispan_flat_mesh_uses_owner_local_field():
    import anymesher.hybrid as mesher
    from anymesher.native_v2 import NativeMeshingOptions
    model,face=model_surface();before=owner.to_dict(model)
    result=mesher.generate_hybrid_mesh_result(model,target_size=.5,strategy='native',native_backend='auto',
        order='linear',certification_mode='none',structural_preparation=False,
        native_options=NativeMeshingOptions(point_placement='frontal_delaunay',metric_mode='isotropic_spatial'))
    assert result.mesh.elements_of_face[face]
    assert result.mesh.hybrid_diagnostics['geometry_approximation']['adaptation']=='local'
    assert owner.to_dict(model)==before


@pytest.mark.parametrize('reversed_chart',(False,True))
def test_cross_knot_whole_footprint_uses_infimum_after_owner_transport(reversed_chart):
    surface=inserted_surface(Q(1,2))
    if reversed_chart:surface=replace(surface,u_range=(1.,0.),v_range=(1.,0.))
    model,face=model_surface(surface)
    field=query_local_approximation_field(model,(face,),relative_accuracy=.001,
        scale_error=.1,policy_floor=1e-8)
    rows=field.face_tiles[0][2]
    assert len({limit for box,limit,curvature in rows})>1
    prepared=owner.clone_prepared_geometry(model,new_preparation_epoch=True)
    regions=owner.query_material_surface_regions(prepared,(face,))
    allocation=bind_local_approximation_field(model,prepared,field,regions)
    query=prepare_local_cell_limit_query(prepared,allocation)
    footprint=((Q(1,4),Q(3,4)),(Q(1,4),Q(3,4)))
    limit,(_,mapped)=query(face,(footprint,))
    selected=[value for box,value,curvature in rows if any(
        all(a<=d and c<=b for (a,b),(c,d) in zip(box,covered)) for covered in mapped)]
    selected.extend(value for f,box,value,identity in field.boundary_tiles if f==face and any(
        all(a<=d and c<=b for (a,b),(c,d) in zip(box,covered)) for covered in mapped))
    assert limit==min(selected)
