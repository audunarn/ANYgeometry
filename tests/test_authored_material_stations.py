"""CURRENT trace oracle is independent of rounded ancestral split parameters."""
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F

import numpy as np
import pytest

from anygeometry import GeometryError, GeometryModel, Plane, to_dict, apply_intersections, plan_intersections
from anygeometry.authored_boundary_correspondence import query_prepared_authored_boundary_correspondence
from anygeometry.authored_boundary_stations import query_prepared_authored_boundary_stations as ancestral
from anygeometry.authored_material_stations import (
    query_prepared_authored_material_stations as query,
    validate_prepared_authored_material_station_coordinates as validate,
)
from examples.authored_material_stations_handoff import build, unpack


def coordinates(receipt):
    return np.array([[float(value) for value in row] for row in unpack(receipt.current_points)],float).reshape((-1,3))


def test_all_current_root2_endpoints_have_one_canonical_exact_uv():
    model,bindings,_,_=build()
    binding=bindings[1]
    before=to_dict(model)
    seen={}
    differences=[]
    exterior={edge for loop in binding.exterior_loops for _,_,edges in loop for edge in edges}
    edges=exterior|{edge for edge,_ in binding.interior_incidence}
    for edge in sorted(edges):
        receipt=query(model,binding,edge,(0,1))
        points=unpack(receipt.current_points)
        uv=unpack(receipt.authored_uv)
        assert uv==tuple(((point[1]+1)/6,(point[2]+1)/2) for point in points)
        assert unpack(receipt.authored_points)==points
        assert receipt.endpoint_ids==(model.edges[edge].start,model.edges[edge].end)
        for vertex,value in zip(receipt.endpoint_ids,uv):
            if vertex in seen:assert seen[vertex]==value
            seen[vertex]=value
        validate(model,receipt,coordinates(receipt))
        if edge in exterior:
            legacy=ancestral(model,binding,edge,(0,1))
            differences.extend((edge,a,b) for a,b in zip(unpack(legacy.authored_uv),uv) if a!=b)
    assert differences, 'ancestral source fractions must remain distinct, not silently replaced'
    assert {F(1,6),F(5,6)}<=set(value[0] for value in seen.values())
    assert to_dict(model)==before


@pytest.mark.parametrize('trace',('exterior','interior'))
def test_order_duplicates_and_rational_parameters_follow_current_stored_direction(trace):
    model,bindings,_,joint=build()
    binding=bindings[1]
    edge=joint if trace=='interior' else binding.exterior_loops[0][0][2][0]
    parameters=(F(1,3),np.float32(.25),np.int64(1),0,F(1,3))
    receipt=query(model,binding,edge,parameters)
    first,last=unpack(receipt.endpoint_points)
    expected=tuple(tuple((1-F(p))*a+F(p)*b for a,b in zip(first,last))
                   for p in (F(1,3),F(1,4),1,0,F(1,3)))
    assert unpack(receipt.current_points)==expected
    assert unpack(receipt.authored_uv)==tuple(((p[1]+1)/6,(p[2]+1)/2) for p in expected)
    assert receipt.trace_kind==('paired_interior' if trace=='interior' else 'exterior')
    validate(model,receipt,coordinates(receipt))
    with pytest.raises(FrozenInstanceError):receipt.edge_id=0


@pytest.mark.parametrize('plane',(
    Plane((0,0,0),(1,0,0),(0,1,0)),
    Plane((10,-2,3),(2,0,0),(1,3,0)),
))
def test_translated_nonorthogonal_plane_owner_lift(plane):
    model=GeometryModel()
    points=tuple(tuple(plane.origin+u*plane.u_vector+v*plane.v_vector) for u,v in ((0,0),(1,0),(1,1),(0,1)))
    face=model.add_plate(model.add_points(points))
    model.set_face_surface(face,plane)
    apply_intersections(model,plan_intersections(model,tuple(model.faces),policy='connect'),policy='connect')
    binding=query_prepared_authored_boundary_correspondence(model,face)
    edge=binding.exterior_loops[0][0][2][0]
    receipt=query(model,binding,edge,(F(1,3),0,1))
    assert unpack(receipt.authored_uv)==((F(1,3),F(0)),(F(0),F(0)),(F(1),F(0)))
    validate(model,receipt,coordinates(receipt))


@pytest.mark.parametrize('parameters',((True,),(-1,),('0.5',),(np.inf,),(np.nan,),(F(4,3),)))
def test_invalid_parameters_are_typed(parameters):
    model,bindings,_,edge=build()
    with pytest.raises(GeometryError):query(model,bindings[1],edge,parameters)


def test_binding_tamper_stale_wrong_model_and_coordinate_error_refuse():
    model,bindings,_,edge=build()
    receipt=query(model,bindings[1],edge,(0,1))
    with pytest.raises(GeometryError):validate(model,replace(receipt,authored_uv=()),coordinates(receipt))
    other,_,_,_=build()
    with pytest.raises(GeometryError):validate(other,receipt,coordinates(receipt))
    points=coordinates(receipt);points[0,0]+=.001
    with pytest.raises(GeometryError,match='coordinate error'):validate(model,receipt,points)
    key=bindings[1].descendants[0]
    model._faces[key]=replace(model.faces[key],metadata={'stale':True})
    with pytest.raises(GeometryError):validate(model,receipt,coordinates(receipt))


def test_cancellation_identity_and_final_callback_guard():
    model,bindings,_,edge=build();before=to_dict(model)
    error=RuntimeError('cancel material station')
    def cancel(_):raise error
    with pytest.raises(RuntimeError) as caught:query(model,bindings[1],edge,(0,),cancellation_check=cancel)
    assert caught.value is error and to_dict(model)==before
    calls=0
    def count(_):
        nonlocal calls
        calls+=1
    query(model,bindings[1],edge,(0,),cancellation_check=count)
    current=0
    def mutate_last(_):
        nonlocal current
        current+=1
        if current==calls:
            key=bindings[1].descendants[0]
            model._faces[key]=replace(model.faces[key],metadata={'late':True})
    with pytest.raises(GeometryError):query(model,bindings[1],edge,(0,),cancellation_check=mutate_last)


def test_transient_endpoint_callback_edit_uses_detached_controls():
    model,bindings,_,edge=build()
    baseline=query(model,bindings[1],edge,(F(1,3),))
    key=baseline.endpoint_ids[0];old=model.vertices[key];calls=0
    def temporary(_):
        nonlocal calls
        calls+=1
        if calls==1:model._vertices[key]=replace(old,position=old.position+np.array((.125,0,0)))
        elif calls==2:model._vertices[key]=old
    assert query(model,bindings[1],edge,(F(1,3),),cancellation_check=temporary)==baseline


def test_caller_array_conversion_cannot_change_definition_or_input_ownership():
    model,bindings,_,edge=build();receipt=query(model,bindings[1],edge,(0,1))
    calls=0
    class Input:
        def __array__(self,dtype=None,copy=None):
            nonlocal calls
            calls+=1
            key=receipt.endpoint_ids[0]
            model._vertices[key]=replace(model.vertices[key],position=model.vertices[key].position+np.array((.125,0,0)))
            return coordinates(receipt)
    with pytest.raises(GeometryError):validate(model,receipt,Input())
    assert calls==1


def test_empty_batch_and_unsupported_root_or_trace():
    model,bindings,_,edge=build()
    receipt=query(model,bindings[1],edge,())
    validate(model,receipt,np.empty((0,3)))
    with pytest.raises(GeometryError,match='authenticated root trace'):
        query(model,bindings[1],model.faces[bindings[0].descendants[0]].loop[-1].edge,(0,))
    from test_authored_domain_coverage import cubic
    curved,binding=cubic(cropped=True)
    first=binding.exterior_loops[0][0][2][0]
    with pytest.raises(GeometryError):query(curved,binding,first,(0,1))
    connector=binding.exterior_loops[0][1][2][0]
    with pytest.raises(GeometryError,match='original exact Plane'):
        query(curved,binding,connector,(0,1))


def test_tolerance_close_current_trace_is_not_exactly_on_original_plane():
    model=GeometryModel()
    root=model.add_plate(model.add_points(((0,0,0),(4,0,.4),(4,4,.4),(0,4,0))))
    model.set_face_surface(root,Plane((0,0,0),(1,0,.1),(0,1,0)))
    model.add_plate(model.add_points(((3,-1,-1),(3,5,-1),(3,5,1),(3,-1,1))))
    apply_intersections(model,plan_intersections(model,tuple(model.faces),policy='connect'),policy='connect')
    binding=query_prepared_authored_boundary_correspondence(model,root)
    edge=binding.interior_incidence[0][0]
    endpoint=model.vertex_position(model.edges[edge].start)
    assert F(float(endpoint[2]))!=F(float(endpoint[0]))*F(.1)
    before=to_dict(model)
    with pytest.raises(GeometryError,match='exactly on support'):
        query(model,binding,edge,(0,1))
    assert to_dict(model)==before


def test_id_and_parameter_coercion_cannot_change_current_binding():
    model,bindings,_,edge=build()
    class Identifier(int):
        calls=0
        def __int__(self):
            self.calls+=1
            key=bindings[1].descendants[0]
            model._faces[key]=replace(model.faces[key],metadata={'coercion':True})
            return super().__int__()
    identifier=Identifier(edge)
    with pytest.raises(GeometryError):query(model,bindings[1],identifier,(0,1))
    assert identifier.calls==1


@pytest.mark.parametrize('action', ('model', 'receipt', 'cancel', 'exception'))
def test_coordinate_assertion_callbacks_cannot_publish_changed_inputs(action):
    model, bindings, _, edge = build()
    receipt = query(model, bindings[1], edge, (0, 1))
    points = coordinates(receipt)
    before = to_dict(model)
    failure = RuntimeError('coordinate assertion cancellation')
    calls = 0

    def callback(stage):
        nonlocal calls
        if stage != 'authored material coordinate assertion':
            return False
        calls += 1
        if calls != 1:
            return False
        if action == 'exception':
            raise failure
        if action == 'cancel':
            return True
        if action == 'model':
            face = bindings[1].descendants[0]
            model._faces[face] = replace(model.faces[face], metadata={'late': 'coordinate callback'})
        else:
            object.__setattr__(receipt, 'trace_kind', 'forged')
        return False

    with pytest.raises(RuntimeError if action == 'exception' else GeometryError) as caught:
        validate(model, receipt, points, cancellation_check=callback)
    assert calls >= 1
    if action == 'exception':
        assert caught.value is failure
    if action != 'model':
        assert to_dict(model) == before
