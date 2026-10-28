import math
from dataclasses import replace
import pytest
from anygeometry import (GeometryModel, GeometryError, query_trim_enclosure_scale,
    validate_trim_enclosure_scale_binding)

def rectangle(model, size=1., shift=0.):
    return model.add_plate(model.add_points(((shift,0,0),(shift+2*size,0,0),
        (shift+2*size,size,0),(shift,size,0))))

@pytest.mark.parametrize('size,shift', [(1.,0.), (10.,0.), (.01,0.), (1.,1000.)])
def test_local_enclosure_scale_and_binding(size, shift):
    model=GeometryModel(); face=rectangle(model,size,shift)
    result=query_trim_enclosure_scale(model,(face,))
    assert result.reference_scale == pytest.approx(math.sqrt(5)*size)
    assert result.scale_definition == 'minimum_trim_enclosure_span'
    assert result.face_ids == (face,)
    validate_trim_enclosure_scale_binding(model,result)
    with pytest.raises(GeometryError,match='binding changed'):
        validate_trim_enclosure_scale_binding(model,replace(result,reference_scale=result.reference_scale*2))

def test_unrelated_geometry_does_not_loosen_reference():
    model=GeometryModel(); face=rectangle(model)
    first=query_trim_enclosure_scale(model,(face,))
    rectangle(model,1000.,10000.)
    second=query_trim_enclosure_scale(model,(face,))
    assert first.reference_scale == second.reference_scale
    with pytest.raises(GeometryError):
        validate_trim_enclosure_scale_binding(model,first)

def test_selected_small_loop_controls_reference_and_largest_controls_floor():
    model=GeometryModel(); a=rectangle(model,.01); b=rectangle(model,10.,100.)
    result=query_trim_enclosure_scale(model,(a,b))
    assert result.reference_scale == pytest.approx(math.sqrt(5)*.01)
    length=max(result.loop_spans)
    assert result.policy_floor == max(model.tolerance.effective_length(length),
        model.tolerance.effective_surface_residual(length),model.tolerance.effective_curve_fit_residual(length))

def test_empty_and_cancellation_refuse():
    model=GeometryModel(); face=rectangle(model)
    with pytest.raises(GeometryError): query_trim_enclosure_scale(model,())
    with pytest.raises(GeometryError,match='cancel'):
        query_trim_enclosure_scale(model,(face,),cancellation_check=lambda phase: True)

def test_callback_raw_mutation_cannot_return_stale_scale():
    model=GeometryModel(); face=rectangle(model); changed=False
    def callback(phase):
        nonlocal changed
        if phase == 'approximation trim enclosure' and not changed:
            changed=True; model.add_point(20,20,20)
        return False
    with pytest.raises(GeometryError):
        query_trim_enclosure_scale(model,(face,),cancellation_check=callback)

def test_hole_enclosure_is_included_in_minimum():
    from anygeometry import OrientedEdge, Plane
    model=GeometryModel(); face=rectangle(model,10.)
    model.set_face_surface(face,Plane((0,0,0),(1,0,0),(0,1,0)))
    edges=model.add_polyline(model.add_points(((1,1,0),(1,1.25,0),
        (1.5,1.25,0),(1.5,1,0))),close=True)
    hole=tuple(OrientedEdge(edge,True) for edge in edges)
    with model.transaction():
        model._put_entity('face',replace(model.faces[face],holes=(hole,)))
    result=query_trim_enclosure_scale(model,(face,))
    assert len(result.loop_spans)==2
    assert result.reference_scale == pytest.approx(math.hypot(.5,.25))
