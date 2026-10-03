"""Exact region unions retain physical cuts while cancelling artificial seams."""
from dataclasses import replace

import numpy as np
import pytest

from anygeometry import (GeometryModel, GeometryError, EntityRef, to_dict,
    plan_intersections, apply_intersections, query_material_surface_regions,
    validate_material_surface_regions_binding, evaluate_material_surface_region)
from anygeometry.definition_binding import definition_checksum


def _interior_cut():
    model=GeometryModel()
    points=model.add_points(((0,0,0),(1,1,0),(2,-1,0),(3,0,0)))
    wall=model.extrude([model.add_spline(points[0],points[1:-1],points[-1])],(0,0,2))[0]
    cut=model.add_plate(model.add_points(((1,-2,.7),(2,-2,.7),(2,2,.7),(1,2,.7))))
    result=apply_intersections(model,plan_intersections(model,(wall,cut),policy='connect'),policy='connect')
    return model,wall,result


def test_exact_region_conserves_area_and_retains_physical_internal_cut():
    model,wall,applied=_interior_cut()
    before=to_dict(model)
    result=query_material_surface_regions(model)
    descendants={ref.id for ref in model.resolve_ref(EntityRef('face',wall))}
    region=next(region for region in result.regions if descendants & {face.id for face in region.faces})
    assert {face.id for face in region.faces}==descendants
    assert len(region.faces)>1 and region.cancelled_seams
    assert all(not path.decomposition for loop in region.boundaries for path in loop)
    assert {path.source_edge for path in region.interior_constraints} & {edge.id for edge in applied.joint_edges}
    # Independent Bernstein derivative and Gauss integration of |B'(t) x d|.
    nodes,weights=np.polynomial.legendre.leggauss(96)
    t=(nodes+1)/2
    dy=3*(1-6*t+6*t*t)
    expected=float(np.sum(weights*np.sqrt(9+dy*dy)))
    assert region.material_area==pytest.approx(expected,abs=1e-9)
    values=evaluate_material_surface_region(model,result,wall if wall in descendants else min(descendants),
                                           [[[.2,.3],[.8,.7]]],require_material=True)
    assert values.shape==(1,2,3)
    du,dv=evaluate_material_surface_region(model,result,min(descendants),[[.2,.3]],derivatives=True)
    np.testing.assert_allclose(du,[[3,3*(1-6*.2+6*.2**2),0]],atol=1e-13)
    np.testing.assert_allclose(dv,[[0,0,2]],atol=1e-13)
    for source in region.sources:
        uv=source.support.local_uv_many(values.reshape(-1,3))
        np.testing.assert_allclose(source.support.evaluate_many(uv),values.reshape(-1,3),atol=1e-12)
    assert definition_checksum(query_material_surface_regions(model,tuple(reversed(tuple(model.faces)))))==definition_checksum(result)
    assert to_dict(model)==before


def test_region_binding_rejects_tamper_staleness_and_cancellation_without_mutation():
    model,_,_=_interior_cut()
    result=query_material_surface_regions(model)
    region=next(region for region in result.regions if region.cancelled_seams)
    before=to_dict(model)
    for changed in (replace(region,cancelled_seams=()),
                    replace(region,interior_constraints=()),replace(region,material_area=region.material_area+1)):
        altered=replace(result,regions=tuple(changed if item is region else item for item in result.regions))
        with pytest.raises(GeometryError,match='definition binding changed'):
            validate_material_surface_regions_binding(model,altered)
    with pytest.raises(GeometryError,match='another model'):
        validate_material_surface_regions_binding(model.clone(),result)
    with pytest.raises(GeometryError,match='cancelled'):
        query_material_surface_regions(model,cancellation_check=lambda _:True)
    assert to_dict(model)==before
    with pytest.raises(GeometryError,match='outside material'):
        evaluate_material_surface_region(model,result,region.faces[0].id,[[2,2]],require_material=True)
    model.add_point(20,20,20)
    with pytest.raises(GeometryError,match='stale'):
        evaluate_material_surface_region(model,result,region.faces[0].id,[[.2,.3]])


def test_protected_seam_and_partial_selection_keep_source_constraints():
    model,_,_=_interior_cut()
    original=query_material_surface_regions(model)
    region=next(region for region in original.regions if region.cancelled_seams)
    seam=region.cancelled_seams[0]
    model.add_to_group('retained seam',[EntityRef('edge',seam.id)])
    result=query_material_surface_regions(model)
    assert seam.id not in {edge.id for item in result.regions for edge in item.cancelled_seams}
    assert seam.id in {path.source_edge for item in result.regions
                      for loop in (*item.boundaries,item.interior_constraints) for path in loop}
    face=region.faces[0].id
    partial=query_material_surface_regions(model,(face,))
    assert len(partial.regions)==1
    assert not partial.regions[0].cancelled_seams


@pytest.mark.parametrize('protection',('face_use','coedge','member_orientation'))
def test_occurrence_and_orientation_reference_semantics_protect_seams(protection):
    model,_,_=_interior_cut()
    region=next(region for region in query_material_surface_regions(model).regions if region.cancelled_seams)
    seam=region.cancelled_seams[0].id
    incident=model.faces_using_edge(seam)
    if protection=='member_orientation':
        axis=model.add_line(*model.add_points(((10,0,0),(11,0,0))))
        model.add_member((axis,),orientation_reference=('edge',seam))
        assert not model.members_using_edge(seam)
    else:
        with model.transaction():
            if protection=='face_use':
                use=next(use for use in model.face_uses.values() if use.face_id==incident[0])
                model._put_structural('face_use',replace(use,metadata={'section':'separate occurrence'}))
            else:
                use=next(use for use in model.coedges.values() if use.edge_id==seam)
                model._put_structural('coedge',replace(use,metadata={'joint':'retain boundary occurrence'}))
    before=to_dict(model)
    result=query_material_surface_regions(model)
    assert seam not in {edge.id for item in result.regions for edge in item.cancelled_seams}
    assert seam in {path.source_edge for item in result.regions
                    for loop in (*item.boundaries,item.interior_constraints) for path in loop}
    assert to_dict(model)==before


def test_singleton_region_retains_an_isolated_vertex_face_attachment():
    from anygeometry.structural import (AttachmentKind,AttachmentTargetKind,
        AttachmentEvidence,ParameterRange)
    model=GeometryModel()
    face=model.add_plate(model.add_points(((0,0,0),(2,0,0),(2,2,0),(0,2,0))))
    vertex=model.add_point(1,1,0)
    attachment=model.ensure_attachment(None,AttachmentKind.VERTEX_ON_FACE,
        AttachmentTargetKind.FACE,face,ParameterRange(0,0),
        (ParameterRange(.5,.5),ParameterRange(.5,.5)),source_kind='vertex',source_id=vertex,
        evidence=AttachmentEvidence.EXACT,tolerance_used=model.tolerance.length)
    before=to_dict(model)
    result=query_material_surface_regions(model,(face,))
    assert [handle.id for handle in result.regions[0].retained_vertices]==[vertex]
    assert [handle.id for handle in result.regions[0].source_attachments]==[attachment]
    assert to_dict(model)==before
