"""Original-domain coverage has no current-fragment or mesh acceptance meaning."""
from dataclasses import replace
import json

import numpy as np
import pytest

from anygeometry import (GeometryModel, GeometryError, Plane, BezierDirectrix, ExtrudedSurface,
                         plan_intersections, apply_intersections, to_dict)
from anygeometry.arrangement_geometry import BezierPath
from anygeometry.entities import OrientedEdge
from anygeometry.authored_boundary_correspondence import query_prepared_authored_boundary_correspondence
from anygeometry.authored_domain_coverage import validate_prepared_authored_face_triangles as validate, _original_domain


def prepared(model, face):
    plan=plan_intersections(model,tuple(model.faces),policy='connect')
    apply_intersections(model,plan,policy='connect')
    return model,query_prepared_authored_boundary_correspondence(model,face)


def planar(*,concave=False,holes=False,fragment=True):
    model=GeometryModel()
    points=(((0,0,0),(3,0,0),(3,1,0),(1,1,0),(1,3,0),(0,3,0)) if concave else
            ((0,0,0),(4,0,0),(4,4,0),(0,4,0)))
    face=model.add_plate(model.add_points(points))
    model.set_face_surface(face,Plane((0,0,0),(1,0,0),(0,1,0)))
    if holes:
        loops=[]
        for x in (.9,2.6):
            edges=model.add_polyline(model.add_points(((x,x,0),(x,x+.2,0),
                (x+.2,x+.2,0),(x+.2,x,0))),close=True)
            loops.append(tuple(OrientedEdge(edge,True) for edge in edges))
        with model.transaction():model._put_entity('face',replace(model.faces[face],holes=tuple(loops)))
    if fragment:
        model.add_plate(model.add_points(((3,-1,-1),(3,5,-1),(3,5,1),(3,-1,1))))
    return prepared(model,face)


def cubic(cropped=False,implicit=False):
    model=GeometryModel()
    controls=((0.,0.,0.),(1.,2.,0.),(2.,-1.,0.),(3.,1.,0.))
    vector=(.25,0.,1.5)
    if not cropped:
        points=model.add_points(controls)
        edge=model.add_spline(points[0],points[1:-1],points[-1])
        face=model.extrude((edge,),vector)[0]
        if not implicit:
            model.set_face_surface(face,ExtrudedSurface(BezierDirectrix(controls),vector))
        model.add_plate(model.add_points(((1.5,-3,-1),(1.5,3,-1),(1.5,3,3),(1.5,-3,3))))
    else:
        # Dyadic de Casteljau coefficients are exact for this small fixture.
        base=np.asarray(BezierPath(controls).subcurve(.25,.75).controls)
        lower=model.add_points(base+np.asarray(vector)*.125)
        upper=model.add_points(base+np.asarray(vector)*.875)
        bottom=model.add_spline(lower[0],lower[1:-1],lower[-1])
        right=model.add_line(lower[-1],upper[-1])
        top=model.add_spline(upper[0],upper[1:-1],upper[-1])
        left=model.add_line(lower[0],upper[0])
        face=model.add_face_from_loop(tuple(OrientedEdge(edge,forward) for edge,forward in
            ((bottom,True),(right,True),(top,False),(left,False))), (0,1,2,3),
            surface=ExtrudedSurface(BezierDirectrix(controls),vector,(.25,.75),(.125,.875)))
    return prepared(model,face)


def test_planar_original_coverage_across_prepared_fragments_without_publication():
    model,result=planar(); source=to_dict(model)
    assert len(result.descendants)>1
    triangles=np.array([[[0,0],[4,0],[2,3]],[[1,1],[1,2],[2,1]]],float)
    before=triangles.tobytes()
    assert validate(model,result,triangles) is None
    assert to_dict(model)==source and triangles.tobytes()==before
    assert not hasattr(result,'material_coverage_qualified')


@pytest.mark.parametrize('cropped',[False,True])
def test_original_cubic_domain_uses_original_support_ranges(cropped):
    model,result=cubic(cropped); source=to_dict(model)
    validate(model,result,[[[0,0],[1,0],[.5,.75]],[[.1,.2],[.8,.2],[.3,.8]]])
    with pytest.raises(GeometryError,match='outside'):
        validate(model,result,[[[-.01,.1],[.5,.1],[.25,.5]]])
    assert to_dict(model)==source


def test_original_concave_side_crossing_and_off_domain_vertex_refuse():
    model,result=planar(concave=True,fragment=False)
    with pytest.raises(GeometryError,match='side outside'):
        validate(model,result,[[[.5,2.5],[2.5,.5],[.25,.25]]])
    with pytest.raises(GeometryError,match='vertex outside'):
        validate(model,result,[[[5,5],[6,5],[5,6]]])


def test_all_original_holes_are_kept_and_enclosed_hole_refuses():
    model,result=planar(holes=True)
    assert len(json.loads(result.authored_definition.definition_json)['face']['holes'])==2
    with pytest.raises(GeometryError,match='encloses a hole'):
        validate(model,result,[[[.25,.25],[3,.25],[.25,3]]])
    with pytest.raises(GeometryError,match='vertex outside'):
        validate(model,result,[[[2.65,2.65],[3,2.65],[2.65,3]]])
    validate(model,result,[[[3.2,3.2],[3.8,3.2],[3.2,3.8]]])


def test_stale_and_tampered_correspondence_refuse_before_using_original_json():
    model,result=planar()
    forged=replace(result,authored_definition=replace(result.authored_definition,definition_json='{}'))
    with pytest.raises(GeometryError,match='definition binding'):
        validate(model,forged,[[[0,0],[1,0],[0,1]]])
    model.add_point(20,20,20)
    with pytest.raises(GeometryError,match='stale'):
        validate(model,result,[[[0,0],[1,0],[0,1]]])


def test_input_snapshot_and_cancellation_identity_preserve_source():
    model,result=planar(); source=to_dict(model)
    triangles=np.array([[[0.,0.],[1.,0.],[0.,1.]]])
    def mutate_input(_):triangles[:]=99
    validate(model,result,triangles,cancellation_check=mutate_input)
    error=RuntimeError('original coverage cancellation'); calls=0
    def cancel(phase):
        nonlocal calls
        if phase=='authored domain coverage':
            calls+=1
            if calls==3:raise error
    with pytest.raises(RuntimeError) as caught:
        validate(model,result,[[[0,0],[1,0],[0,1]]],cancellation_check=cancel)
    assert caught.value is error and to_dict(model)==source


def test_post_kernel_callback_mutation_cannot_return_success(monkeypatch):
    import anygeometry.authored_domain_coverage as module
    model,result=planar(); done=False
    original=module._validate_domain_triangles
    def completed(*args):
        nonlocal done
        original(*args);done=True
    monkeypatch.setattr(module,'_validate_domain_triangles',completed)
    def mutate(_):
        nonlocal done
        if done:
            done=False
            model.add_point(20,20,20)
    with pytest.raises(GeometryError):
        validate(model,result,[[[0,0],[1,0],[0,1]]],cancellation_check=mutate)


@pytest.mark.parametrize('triangles',[np.zeros((3,2)),np.full((1,3,2),np.nan),np.full((1,3,2),1+2j)])
def test_invalid_input_precedes_callbacks(triangles):
    model,result=planar(fragment=False)
    def callback(_):raise AssertionError('invalid input reached callback')
    with pytest.raises(GeometryError,match='finite'):
        validate(model,result,triangles,cancellation_check=callback)


def test_original_non_polynomial_trim_parser_refuses_without_fit():
    model,result=planar(fragment=False)
    payload=json.loads(result.authored_definition.definition_json)
    payload['edges'][0]['curve']={'type':'bezier_quadric_intersection'}
    definition=replace(result.authored_definition,definition_json=json.dumps(payload))
    # Parsing-only discriminator; the public binding would reject this forgery.
    with pytest.raises(GeometryError,match='original polynomial trims'):
        _original_domain(definition,lambda:None)


def test_generated_implicit_coons_has_exact_original_extrusion_identity():
    model,result=cubic(implicit=True)
    assert json.loads(result.authored_definition.definition_json)['face']['surface']['type']=='coons'
    source=to_dict(model)
    domain=_original_domain(result.authored_definition,lambda:None)
    assert isinstance(domain.support,ExtrudedSurface)
    assert domain.support.directrix.controls==((0.,0.,0.),(1.,2.,0.),(2.,-1.,0.),(3.,1.,0.))
    assert tuple(domain.support.vector)==(.25,0.,1.5)
    validate(model,result,[[[0,0],[1,0],[.5,.5]]])
    assert to_dict(model)==source


def test_non_none_original_parameterization_refuses_explicitly():
    model=GeometryModel()
    face=model.add_plate(model.add_points(((0,0,0),(1,0,0),(1,1,0),(0,1,0))))
    model.set_face_parameterization(face,Plane((0,0,0),(2,0,0),(0,2,0)))
    model,result=prepared(model,face)
    with pytest.raises(GeometryError,match='original parameterization'):
        validate(model,result,[[[0,0],[1,0],[.5,.5]]])


@pytest.mark.parametrize('mutation,match',[
    ('near_control','exact coefficient translations'),
    ('nonparallel','exact coefficient translations'),
    ('curved_connector','straight connectors'),
    ('explicit_coons','four-edge implicit Coons'),
    ('different_corners','four-edge implicit Coons'),
    ('unrepresentable_vector','exactly representable'),
])
def test_coons_recognition_refuses_unproved_original_coefficients(mutation,match):
    _,result=cubic(implicit=True)
    payload=json.loads(result.authored_definition.definition_json)
    face=payload['face']; edges={item['id']:item for item in payload['edges']}
    vertices={item['id']:item for item in payload['vertices']}
    bottom=edges[face['loop'][0][0]]; right=edges[face['loop'][1][0]]
    top=edges[face['loop'][2][0]]
    if mutation=='near_control':
        vertices[top['curve']['control_vertices'][0]]['position'][1]+=2.**-40
    elif mutation=='nonparallel':
        vertices[top['end']]['position'][1]+=.125
    elif mutation=='curved_connector':
        identifier=max(vertices)+1
        a=np.asarray(vertices[right['start']]['position']);b=np.asarray(vertices[right['end']]['position'])
        payload['vertices'].append({'id':identifier,'position':((a+b)/2+np.array([0,.125,0])).tolist()})
        right['curve']={'type':'spline','control_vertices':[identifier]}
    elif mutation=='explicit_coons':
        face['surface']['bottom']=[[0,0,0],[3,1,0]]
    elif mutation=='different_corners':
        face['corners']=[1,2,3,0]
    else:
        for identifier in (bottom['start'],*bottom['curve']['control_vertices'],bottom['end']):
            vertices[identifier]['position'][2]=-2.**-60
    changed=replace(result.authored_definition,definition_json=json.dumps(payload))
    # These are recognition-kernel counterexamples, not authenticated public
    # results. Public validation rejects such a changed snapshot beforehand.
    with pytest.raises(GeometryError,match=match):_original_domain(changed,lambda:None)
