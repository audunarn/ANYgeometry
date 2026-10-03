"""Original-domain success cannot grant a current child's reference scope."""
from dataclasses import replace

import numpy as np
import pytest

from anygeometry import (
    GeometryError, GeometryModel, Plane, apply_intersections, plan_intersections,
    query_prepared_authored_boundary_correspondence,
    validate_prepared_authored_face_triangles,
    validate_prepared_authored_face_child_triangles as validate, to_dict,
)
from anygeometry.authored_child_coverage import _support_correspondence
from anygeometry.surfaces import ExtrudedSurface
from test_authored_domain_coverage import planar, cubic


def children(model, correspondence):
    return sorted(correspondence.descendants, key=lambda face: min(
        model.vertex_position(model.oriented_start_vertex(use))[0]
        for use in model.faces[face].loop))


def test_split_left_pressure_whole_root_cell_refuses_but_each_selected_child_cell_passes():
    model, binding = planar()
    left, right = children(model, binding)
    crossing = [[[2,.5],[3.5,.5],[2,1.5]]]
    # Independent fixture cut at x=3: the original is the whole 4x4 square.
    validate_prepared_authored_face_triangles(model, binding, crossing)
    with pytest.raises(GeometryError, match='triangle vertex outside material'):
        validate(model, binding, left, crossing)
    with pytest.raises(GeometryError):
        validate(model, binding, right, crossing)
    before = to_dict(model)
    assert validate(model, binding, left, [[[1,.5],[2,.5],[1,1.5]]]) is None
    assert validate(model, binding, model.handle('face',right), [[[3,.5],[4,.5],[3,1.5]]]) is None
    assert to_dict(model) == before


def test_current_child_native_coordinates_are_not_original_uv_and_wrong_child_refuses():
    model, binding = planar()
    _, right = children(model, binding)
    # These would be inside a child's unit-square chart, but mean world x<3
    # in the authenticated original unit-basis plane.
    with pytest.raises(GeometryError):
        validate(model, binding, right, [[[.1,.1],[.4,.1],[.1,.4]]])


def test_independent_affine_original_frame_and_rebased_child_preserve_coordinates():
    model = GeometryModel()
    face = model.add_plate(model.add_points(((0,0,0),(4,0,0),(4,4,0),(0,4,0))))
    # Reversed v and shear; the same world square, non-unit source coordinates.
    model.set_face_surface(face, Plane((0,4,0),(2,0,0),(1,-2,0)))
    model.add_plate(model.add_points(((3,-1,-1),(3,5,-1),(3,5,1),(3,-1,1))))
    apply_intersections(model, plan_intersections(model,tuple(model.faces),policy='connect'),policy='connect')
    binding = query_prepared_authored_boundary_correspondence(model,face)
    left, right = children(model,binding)
    # World (x,y)=(1,1),(2,1),(1,2) => (u,v)=(-.25,1.5),(.25,1.5),(0,1).
    triangle = [[[-.25,1.5],[.25,1.5],[0,1]]]
    validate(model,binding,left,triangle)
    with pytest.raises(GeometryError):
        validate(model,binding,right,triangle)


def test_original_and_actual_child_holes_and_concavity_are_checked():
    model, binding = planar(holes=True)
    left = children(model,binding)[0]
    with pytest.raises(GeometryError):
        validate(model,binding,left,[[[.5,.5],[1.5,.5],[.5,1.5]]])
    validate(model,binding,left,[[[.1,.1],[.4,.1],[.1,.4]]])
    model,binding=planar(concave=True,fragment=False)
    child=binding.descendants[0]
    with pytest.raises(GeometryError):
        validate(model,binding,child,[[[.5,2.5],[2.5,.5],[.5,.5]]])


@pytest.mark.parametrize('implicit', [False,True])
def test_rounded_split_controls_do_not_gain_literal_child_coverage(implicit):
    model,binding=cubic(implicit=implicit)
    left=children(model,binding)[0]
    with pytest.raises(GeometryError,match='exact polynomial chart correspondence is unresolved'):
        validate(model,binding,left,[[[.1,.2],[.3,.2],[.1,.4]]])
    crossing=[[[.3,.2],[.7,.2],[.3,.4]]]
    validate_prepared_authored_face_triangles(model,binding,crossing)
    with pytest.raises(GeometryError):
        validate(model,binding,left,crossing)


def test_nominal_dyadic_cubic_cut_does_not_hide_rounded_literal_controls():
    from anygeometry import BezierDirectrix
    model=GeometryModel()
    controls=((0,0,0),(1,1,0),(2,1,0),(3,0,0))
    vertices=model.add_points(controls)
    spline=model.add_spline(vertices[0],vertices[1:-1],vertices[-1])
    face=model.extrude((spline,),(0,0,2))[0]
    model.set_face_surface(face,ExtrudedSurface(BezierDirectrix(controls),(0,0,2)))
    model.add_plate(model.add_points(((1.5,-2,-1),(1.5,2,-1),(1.5,2,3),(1.5,-2,3))))
    apply_intersections(model,plan_intersections(model,tuple(model.faces),policy='connect'),policy='connect')
    binding=query_prepared_authored_boundary_correspondence(model,face)
    left=children(model,binding)[0]
    with pytest.raises(GeometryError,match='exact polynomial chart correspondence is unresolved'):
        validate(model,binding,left,[[[.1,.2],[.3,.2],[.1,.4]]])
    with pytest.raises(GeometryError):
        validate(model,binding,left,[[[.3,.2],[.7,.2],[.3,.4]]])


@pytest.mark.parametrize('triangles', [np.empty((0,3,2)), [[[.1,.1],[.4,.1],[.1,.4]]]])
def test_identity_unsplit_cropped_original_extrusion(triangles):
    model,binding=cubic(cropped=True)
    validate(model,binding,binding.descendants[0],triangles)


def test_support_correspondence_does_not_infer_same_surface_from_equal_boundary_points():
    original=Plane((0,0,0),(1,0,0),(0,1,0))
    with pytest.raises(GeometryError,match='not exactly'):
        _support_correspondence(original,Plane((0,0,1e-15),(1,0,0),(0,1,0)),np.empty((0,3,2)))
    from anygeometry import BezierDirectrix
    source=ExtrudedSurface(BezierDirectrix(((0,0,0),(1,1,0),(2,0,0))),(0,0,1))
    changed=ExtrudedSurface(BezierDirectrix(((0,0,0),(1,2,0),(2,0,0))),(0,0,1))
    with pytest.raises(GeometryError,match='carrier differs'):
        _support_correspondence(source,changed,np.empty((0,3,2)))


@pytest.mark.parametrize('bad', [True,1.0,0,-1,999])
def test_bad_or_unrelated_child_identity_refuses(bad):
    model,binding=planar()
    with pytest.raises(GeometryError):
        validate(model,binding,bad,[[[1,.5],[2,.5],[1,1.5]]])
    other=next(face for face in model.faces if face not in binding.descendants)
    with pytest.raises(GeometryError):
        validate(model,binding,other,[[[1,.5],[2,.5],[1,1.5]]])


@pytest.mark.parametrize('bad', [[[0,0],[1,0],[0,1]], [[[0,0],[1,0],[2,0]]],
    [[[0,0],[1,0],[0,float('nan')]]], [[[0,0],[1,0],[0,1j]]]])
def test_invalid_or_degenerate_triangle_inputs_refuse(bad):
    model,binding=planar()
    with pytest.raises(GeometryError):
        validate(model,binding,children(model,binding)[0],bad)


def test_input_is_detached_and_query_cancellation_preserves_model():
    model,binding=planar()
    left=children(model,binding)[0]
    rows=np.asarray([[[1,.5],[2,.5],[1,1.5]]])
    before=to_dict(model)
    def mutate_input(_):
        rows[:]=99
        return False
    validate(model,binding,left,rows,cancellation_check=mutate_input)
    assert to_dict(model)==before
    with pytest.raises(GeometryError,match='cancelled'):
        validate(model,binding,left,[[[1,.5],[2,.5],[1,1.5]]],cancellation_check=lambda _:True)
    assert to_dict(model)==before


@pytest.mark.parametrize('stage', ['callback','array','raw','revision'])
def test_stale_or_changed_source_never_returns_child_coverage(stage):
    model,binding=planar()
    left=children(model,binding)[0]
    rows=[[[1,.5],[2,.5],[1,1.5]]]
    def mutate(_):
        model._faces[left]=replace(model.faces[left],metadata={'changed': True})
        return False
    class Input:
        def __array__(self,dtype=None,copy=None):
            mutate(None)
            return np.asarray(rows,dtype=dtype)
    if stage=='raw':mutate(None)
    if stage=='revision':model.add_point(50,50,50)
    with pytest.raises(GeometryError):
        validate(model,binding,left,Input() if stage=='array' else rows,
                 cancellation_check=mutate if stage=='callback' else None)


@pytest.mark.parametrize('mutation', ['loop', 'support'])
def test_transient_callback_changes_cannot_replace_the_authenticated_child(monkeypatch, mutation):
    import anygeometry.authored_child_coverage as owner
    model,binding=planar()
    left,right=children(model,binding)
    before=to_dict(model)
    original_face=model.faces[left]
    saved_origin=original_face.surface.origin.copy()
    stage={'original':False,'changed':False}
    reconstruct=owner._original_domain

    def restore():
        model._faces[left]=original_face
        original_face.surface.origin[:]=saved_origin

    def reconstruct_with_callback(definition,check):
        stage['original']=True
        result=reconstruct(definition,check)
        stage['original']=False
        return result

    def transient_change(_):
        if stage['original'] and not stage['changed']:
            stage['changed']=True
            if mutation=='loop':
                model._faces[left]=replace(original_face,loop=model.faces[right].loop,
                                           holes=model.faces[right].holes)
            else:
                original_face.surface.origin[2]=1
        elif not stage['original'] and stage['changed']:
            restore()
        return False

    monkeypatch.setattr(owner,'_original_domain',reconstruct_with_callback)
    try:
        if mutation=='loop':
            # This is wholly right of the independently constructed x=3 cut.
            with pytest.raises(GeometryError,match='triangle vertex outside material'):
                validate(model,binding,left,[[[3.1,.5],[3.8,.5],[3.1,1.5]]],
                         cancellation_check=transient_change)
        else:
            # Deep-copy the Plane arrays too, so a temporary alias mutation
            # cannot corrupt the captured proof support.
            validate(model,binding,left,[[[1,.5],[2,.5],[1,1.5]]],
                     cancellation_check=transient_change)
        assert stage['changed']
        assert to_dict(model)==before
    finally:
        restore()


def test_literal_extrusion_ranges_include_reversed_child_ranges():
    from anygeometry import BezierDirectrix
    directrix=BezierDirectrix(((0,0,0),(1,1,0),(2,0,0)))
    original=ExtrudedSurface(directrix,(0,0,1))
    child=ExtrudedSurface(directrix,(0,0,1),(.75,.25),(1,0))
    _support_correspondence(original,child,np.asarray([[[.3,.1],[.7,.1],[.3,.8]]]))
    with pytest.raises(GeometryError,match='leaves literal support range'):
        _support_correspondence(original,child,np.asarray([[[.2,.1],[.7,.1],[.3,.8]]]))


def test_explicit_sampled_coons_surface_does_not_gain_exact_child_coverage():
    from anygeometry import CoonsSurface
    model=GeometryModel()
    face=model.add_plate(model.add_points(((0,0,0),(4,0,0),(4,4,0),(0,4,0))))
    model.set_face_surface(face,CoonsSurface(((0,0,0),(4,0,0)),((4,0,0),(4,4,0)),
                                           ((0,4,0),(4,4,0)),((0,0,0),(0,4,0))))
    apply_intersections(model,plan_intersections(model,(face,),policy='connect'),policy='connect')
    binding=query_prepared_authored_boundary_correspondence(model,face)
    with pytest.raises(GeometryError):
        validate(model,binding,binding.descendants[0],[[[1,.5],[2,.5],[1,1.5]]])
