"""Extrusion retains exact spline/NURBS profile geometry and differential shape."""
from dataclasses import replace
import numpy as np
import pytest
from anygeometry import (GeometryModel, SketchDefinition, SketchEntity, SketchWire,
    plane_sketch_plane, materialize_sketch_on_plane, to_dict, from_dict)
from anygeometry.nurbs import NurbsCurve, CurveOnSurface, NurbsSurface
from anygeometry.surfaces import CoonsSurface
from anygeometry.errors import GeometryError


@pytest.mark.parametrize('kind',['bezier','nurbs'])
@pytest.mark.parametrize('rotated',[False,True])
def test_exact_translated_profile_surface_derivatives_and_roundtrip(kind,rotated):
    geometry=GeometryModel()
    matrix=np.array(((1,0,0),(0,0,-1),(0,1,0))) if rotated else np.eye(3)
    controls=np.array(((0,0,0),(1,2,0),(3,-1,0),(4,0,0)))@matrix.T+np.array((2,3,4))
    vector=matrix@np.array((.3,.1,2.))
    if kind=='bezier':
        vertices=geometry.add_points(controls)
        edge=geometry.add_spline(vertices[0],vertices[1:-1],vertices[-1])
    else:
        curve=NurbsCurve(3,(2,2,2,2,5,5,5,5),tuple(map(tuple,controls)),(1.,.6,1.2,1.),(5.,2.))
        ends=geometry.add_points(curve.evaluate(np.array((0.,1.))))
        edge=geometry.add_curve(*ends,curve)
    face,=geometry.extrude((edge,),vector)
    assert isinstance(geometry.faces[face].surface,CoonsSurface)
    top=geometry.faces[face].loop[2].edge
    parameters=np.linspace(.05,.95,9)
    np.testing.assert_allclose(geometry.sample_edge(top,parameters),geometry.sample_edge(edge,parameters)+vector,atol=2e-12)
    for candidate in (geometry,from_dict(to_dict(geometry))):
        uv=np.array([(u,v) for u in parameters for v in (.0,.25,.75,1.)])
        expected=np.array([candidate.sample_edge(edge,np.array((u,)))[0]+v*vector for u,v in uv])
        np.testing.assert_allclose(candidate.evaluate_face_many(face,uv),expected,atol=3e-12)
        du,dv=candidate.face_derivatives_many(face,uv)
        # Independent finite differences expose the old chord derivative.
        h=1e-6
        expected_du=np.array([(candidate.sample_edge(edge,np.array((u+h,)))[0]-candidate.sample_edge(edge,np.array((u-h,)))[0])/(2*h) for u,v in uv])
        np.testing.assert_allclose(du,expected_du,atol=2e-8)
        np.testing.assert_allclose(dv,np.tile(vector,(len(uv),1)),atol=3e-12)
        expected_normals=np.cross(du,dv);expected_normals/=np.linalg.norm(expected_normals,axis=1)[:,None]
        np.testing.assert_allclose(candidate.face_normal_many(face,uv),expected_normals,atol=2e-12)
        assert not candidate.validate_topology()


@pytest.mark.parametrize('kind',['spline','nurbs'])
@pytest.mark.parametrize('normal',[(0,0,1),(0,1,0)])
def test_v2_curve_sketch_extrudes_regenerates_and_reopens(kind,normal):
    geometry=GeometryModel();geometry.features.capture_baseline(geometry)
    definition=SketchDefinition({'a':(0,0),'b':(1,1),'c':(3,1),'d':(4,0)},version=2,
        entities=(SketchEntity('profile',kind,('a','b','c','d')),),
        wires=(SketchWire('wire',('profile',),False),),extrusion=2.)
    parameters=definition.to_parameters();parameters['plane']={'origin':[1,2,3],'normal':list(normal),'x_axis':[1,0,0]}
    feature=geometry.features.append('geometry.sketch.plane',parameters=parameters)
    report=geometry.regenerate_features();assert report.success,report
    for candidate in (geometry,from_dict(to_dict(geometry))):
        outputs=candidate.features.get(feature.feature_id).outputs
        faces=[ref.id for key,ref in outputs.items() if key.startswith('extrusion/face/')]
        assert faces
        for face in faces:
            bottom=candidate.faces[face].loop[0].edge
            for u in (.13,.37,.81):
                np.testing.assert_allclose(candidate.face_point(face,u,.4),candidate.sample_edge(bottom,np.array((u,)))[0]+.8*np.array(normal),atol=3e-12)
        parameters['extrusion']=3.
        candidate.features.update(feature.feature_id,parameters=parameters)
        report=candidate.regenerate_features();assert report.success,report
        assert not candidate.validate_topology()


def test_nurbs_support_curve_refusal_is_atomic():
    geometry=GeometryModel()
    support=NurbsSurface(1,1,(0,0,1,1),(0,0,1,1),(((0,0,0),(0,1,0)),((1,0,0),(1,1,0))))
    curve=CurveOnSurface(support,NurbsCurve(1,(0,0,1,1),((0,0),(1,0))))
    ends=geometry.add_points(curve.evaluate(np.array((0.,1.))))
    edge=geometry.add_curve(*ends,curve)
    before=to_dict(geometry)
    with pytest.raises(GeometryError,match='A3b B-S1'):
        geometry.extrude((edge,),(0,0,1))
    assert to_dict(geometry)==before




@pytest.mark.parametrize('points,degree,expected_degree',[
    (((0,0),(1,1.5),(4,0)),3,2),
    (((0,0),(1,0),(4,0)),3,2),
    (((0,0),(4,0)),3,1),
    (((0,0),(1,1.5),(4,0)),1,1),
])
def test_three_fit_points_are_quadratic_and_keep_explicit_line_cases(points,degree,expected_degree):
    from anygeometry.sketch_entities import _nurbs
    curve=_nurbs(points,degree,(),True,False)
    assert curve.degree==expected_degree
    stations=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    stations/=stations[-1]
    np.testing.assert_allclose(curve.evaluate(stations),points,atol=1e-12)
    if np.all(np.asarray(points)[:,1]==0):
        np.testing.assert_allclose(curve.evaluate(np.linspace(0,1,21))[:,1],0.,atol=1e-12)
    if expected_degree==2 and points[1][1]!=0:
        # A smooth quadratic changes tangent continuously instead of having
        # the old two-span polyline's corner at the middle fit point.
        t=stations[1]
        np.testing.assert_allclose(curve.derivative(t-1e-7),curve.derivative(t+1e-7),atol=1e-5)
        geometry=GeometryModel();ends=geometry.add_points(np.c_[curve.evaluate(np.array((0.,1.))),[0.,0.]])
        world=replace(curve,controls=tuple((*p,0.) for p in curve.controls))
        edge=geometry.add_curve(*ends,world);face,=geometry.extrude((edge,),(0,0,2))
        for u in (.1,.5,.9):
            np.testing.assert_allclose(geometry.face_point(face,u,.4),np.r_[curve.evaluate(u),.8],atol=2e-12)
