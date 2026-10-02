import numpy as np
import pytest

from anygeometry import (GeometryError, GeometryModel, apply_intersections,
    clone_prepared_geometry, has_current_intersection_preparation,
    plan_intersections, set_prepared_face_corners, to_dict)
from anygeometry.generators import cylinder, cone


def prepared(kind='plane'):
    if kind == 'plane':
        model = GeometryModel()
        model.add_plate(model.add_points(((0,0,0),(2,0,0),(2,2,0),(0,2,0))))
    elif kind in ('cylinder', 'cone'):
        model = cylinder(1,2,circumferential_segments=4) if kind == 'cylinder' else (
            cone(1,2,2,circumferential_segments=4))
    else:
        model = GeometryModel()
        points = model.add_points(((0,0,0),(1,1,0),(2,0,0)))
        edge = model.add_spline(points[0], points[1:-1], points[-1])
        face, = model.extrude((edge,), (0,0,2))
        from anygeometry import BezierDirectrix, ExtrudedSurface
        model.set_face_surface(face, ExtrudedSurface(
            BezierDirectrix(((0.,0.,0.),(1.,1.,0.),(2.,0.,0.))), (0.,0.,2.)))
    face = next(iter(model.faces))
    model.split_edge(model.faces[face].loop[0].edge, .5)
    plan = plan_intersections(model, tuple(model.faces), policy='connect')
    apply_intersections(model, plan, policy='connect')
    return model, face


@pytest.mark.parametrize('kind', ['plane', 'cylinder', 'cone', 'extruded'])
def test_owner_certifies_only_corner_changes_and_clone_keeps_binding(kind):
    model, face = prepared(kind)
    before = to_dict(model)
    expected = model.face_point(face, .3, .6)
    corners = next(c for c in ((0,1,2,3),(0,1,2,4)) if c != model.faces[face].corners)
    set_prepared_face_corners(model, {face: corners})
    assert model.faces[face].corners == corners
    np.testing.assert_array_equal(model.face_point(face,.3,.6), expected)
    assert has_current_intersection_preparation(model)
    assert has_current_intersection_preparation(clone_prepared_geometry(model))
    after = to_dict(model)
    before['revision'], before['checksum'] = after['revision'], after['checksum']
    next(f for f in before['faces'] if f['id'] == face)['corners'] = list(corners)
    assert after == before


def test_corner_batch_failure_is_atomic_and_nested_calls_are_rejected():
    model, face = prepared()
    before = to_dict(model)
    for updates in ({face:(0,1,2,99)}, {face:(0,1,2,3),999:(0,1,2,3)}):
        with pytest.raises(GeometryError):
            set_prepared_face_corners(model, updates)
        assert to_dict(model) == before
        assert has_current_intersection_preparation(model)
    with model.transaction():
        with pytest.raises(GeometryError, match='active transaction'):
            set_prepared_face_corners(model, {face:(0,1,2,3)})
    assert to_dict(model) == before


def test_stale_proof_and_parameterization_cannot_be_refreshed():
    model, face = prepared()
    model.set_face_parameterization(face, model.faces[face].surface)
    before = to_dict(model)
    with pytest.raises(GeometryError, match='current complete'):
        set_prepared_face_corners(model, {face:(0,1,2,3)})
    assert to_dict(model) == before
    apply_intersections(model,plan_intersections(model,tuple(model.faces),policy='connect'),policy='connect')
    before = to_dict(model)
    with pytest.raises(GeometryError, match='corner-independent'):
        set_prepared_face_corners(model, {face:(0,1,2,3)})
    assert to_dict(model) == before


def test_two_face_batch_rolls_back_and_old_application_stays_stale():
    model, face = prepared('cylinder')
    other = next(f for f in model.faces if f != face)
    before = to_dict(model)
    with pytest.raises(GeometryError):
        set_prepared_face_corners(model, {face:(0,1,2,3), other:(0,1,2,99)})
    assert to_dict(model) == before
    plan = plan_intersections(model, tuple(model.faces), policy='connect')
    apply_intersections(model, plan, policy='connect')
    corners = next(c for c in ((0,1,2,3),(0,1,2,4)) if c != model.faces[face].corners)
    observed = []
    model.add_change_hook(lambda *_: observed.append(has_current_intersection_preparation(model)))
    set_prepared_face_corners(model, {face:corners})
    assert observed and not any(observed)
    assert has_current_intersection_preparation(model)
    committed = to_dict(model)
    with pytest.raises(GeometryError):
        apply_intersections(model, plan, policy='connect')
    assert to_dict(model) == committed
