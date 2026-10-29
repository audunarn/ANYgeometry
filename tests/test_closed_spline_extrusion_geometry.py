"""Closed curve extrusion stays exact; point queries skip unused derivatives."""
import numpy as np
import pytest
from anygeometry import (GeometryModel,SketchDefinition,SketchEntity,SketchWire,
    plane_sketch_plane,materialize_sketch_on_plane,to_dict,from_dict)
from anygeometry.sketch_entities import _nurbs


def fixture(kind):
    plane=plane_sketch_plane((0,0,0),(1,0,0),(0,1,0))
    definition=SketchDefinition({'a':(0.,0.),'b':(2.,0.),'c':(1.,2.)},version=2,
        entities=(SketchEntity('curve',kind,('a','b','c'),closed=True),),
        wires=(SketchWire('outline',('curve',),True),),extrusion=3.)
    geometry=GeometryModel();outputs=materialize_sketch_on_plane(geometry,plane,definition)
    return geometry,plane,definition,outputs


@pytest.mark.parametrize('kind',['spline','nurbs'])
def test_closed_subcurve_domains_and_sweeps_remain_exact(kind):
    geometry,plane,definition,outputs=fixture(kind)
    curve=_nurbs(list(definition.points.values()),3,(),kind=='spline',True)
    assert len(geometry.faces)==4
    edges=[outputs[f'entity/curve/edge/{i}'].id for i in range(4)]
    assert len(geometry.order_loop(edges))==4
    for piece,face in enumerate(geometry.faces):
        for u in (0.,.125,.5,.875,1.):
            for v in (0.,.4,1.):
                expected=plane.world(curve.evaluate((piece+u)/4))+3*v*plane.normal
                np.testing.assert_allclose(geometry.face_point(face,u,v),expected,atol=1e-12)
                # Differential evaluation remains the original scalar oracle;
                # the point-only route must preserve its arithmetic exactly.
                np.testing.assert_array_equal(geometry.face_point(face,u,v),geometry._topology_face_point_and_derivatives(face,u,v)[0])
        bend=geometry.face_point(face,.25,.5)-2*geometry.face_point(face,.5,.5)+geometry.face_point(face,.75,.5)
        assert np.linalg.norm(bend)>1e-3


def test_point_queries_do_not_request_discarded_analytic_derivatives(monkeypatch):
    geometry,plane,definition,outputs=fixture('nurbs')
    face=next(iter(geometry.faces));geometry.face_point(face,.3,.4)
    original=geometry._topology_face_points_many
    calls=[]
    original_edge=geometry._edge_parameter_derivative_object
    edge_calls=[]
    def edge_derivative(edge,parameter):
        edge_calls.append(edge.id);return original_edge(edge,parameter)
    monkeypatch.setattr(geometry,'_edge_parameter_derivative_object',edge_derivative)
    def derivative(face,parameters,**options):
        if options.get('_derivatives',False):calls.append(face)
        return original(face,parameters,**options)
    monkeypatch.setattr(geometry,'_topology_face_points_many',derivative)
    geometry.face_point(face,.31,.42)
    geometry.evaluate_face_many(face,np.array(((.2,.3),(.7,.8))))
    assert not calls
    assert not edge_calls
    geometry.face_derivatives_many(face,np.array(((.2,.3),(.7,.8))))
    assert calls


def test_closed_curve_extrusion_save_reopen_keeps_piece_output_geometry():
    geometry,plane,definition,outputs=fixture('nurbs')
    restored=from_dict(to_dict(geometry))
    assert set(restored.faces)==set(geometry.faces)
    uv=np.array(((.13,.2),(.37,.7),(.79,.4)))
    for face in geometry.faces:
        np.testing.assert_array_equal(restored.evaluate_face_many(face,uv),geometry.evaluate_face_many(face,uv))
    for piece in range(4):
        edge=outputs[f'entity/curve/edge/{piece}'].id
        assert edge in restored.edges
