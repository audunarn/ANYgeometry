"""Actual XYZ containment: independent rational affine and material fixtures."""
from dataclasses import replace
from fractions import Fraction as F

import numpy as np
import pytest

from anygeometry import (
    GeometryError, GeometryModel, Plane, BezierDirectrix, ExtrudedSurface,
    query_material_surface_regions, to_dict,
    validate_material_surface_region_triangles_xyz as validate,
)
from anygeometry.entities import OrientedEdge


def fixture(points=((0, 0, 0), (6, 0, 0), (6, 6, 0), (0, 6, 0)),
            support=None, hole=None):
    model = GeometryModel()
    face = model.add_plate(model.add_points(points))
    model.set_face_surface(face, support or Plane((0, 0, 0), (6, 0, 0), (0, 6, 0)))
    if hole:
        edges = model.add_polyline(model.add_points(hole), close=True)
        with model.transaction():
            model._put_entity('face', replace(model.faces[face],
                holes=(tuple(OrientedEdge(edge, True) for edge in edges),)))
    return model, face, query_material_surface_regions(model, (face,))


@pytest.mark.parametrize('origin,u,v', [
    ((0, 0, 0), (6, 0, 0), (0, 6, 0)),
    ((-2, 3, 5), (6, 0, 0), (0, 6, 0)),
    ((0, 0, 0), (-6, 0, 0), (0, 6, 0)),
    ((0, 0, 0), (0, 6, 0), (-6, 0, 0)),
    ((0, 0, 0), (6, 0, 3), (2, 6, 1)),
])
def test_scaled_skewed_reflected_rotated_plane_uses_exact_actual_xyz(origin, u, v, monkeypatch):
    # Independently construct integer outer vertices and rational cell corners.
    # u=1/6 has no exact binary64 representation; no floating inverse is allowed.
    def lift(pair):
        return tuple(F(a)+F(b)*pair[0]+F(c)*pair[1] for a, b, c in zip(origin, u, v))
    outer = [lift(pair) for pair in ((0, 0), (1, 0), (1, 1), (0, 1))]
    model, face, result = fixture(outer, Plane(origin, u, v))
    expected_uv = ((F(1, 6), F(1, 6)), (F(5, 6), F(1, 6)), (F(1, 6), F(5, 6)))
    xyz = np.array([[lift(pair) for pair in expected_uv]], dtype=object)
    source = to_dict(model)
    original = xyz.copy()
    monkeypatch.setattr(model, 'face_support_local_uv', lambda *_: pytest.fail('float inverse used'))
    monkeypatch.setattr(model, 'project_to_face', lambda *_: pytest.fail('sample projection used'))
    validate(model, result, model.handle('face', face), xyz)
    validate(model, result, face, xyz[:, ::-1])
    assert to_dict(model) == source and np.array_equal(xyz, original)


def test_binary64_xyz_with_nonbinary_exact_uv_passes_without_rounding():
    model, face, result = fixture()
    xyz = np.array([[(1, 1, 0), (5, 1, 0), (1, 5, 0)]], dtype=float)
    assert F(float(1/6))*6 != 1  # independent discriminator against float UV.
    validate(model, result, face, xyz)
    # Arbitrarily small but nonzero off-carrier coordinates cannot become proof.
    xyz[0, 0, 2] = np.nextafter(0., 1.)
    with pytest.raises(GeometryError, match='exact Plane support correspondence'):
        validate(model, result, face, xyz)


def test_actual_cells_crossing_concavity_or_enclosing_unsampled_hole_refuse():
    model, face, result = fixture(((0, 0, 0), (3, 0, 0), (3, 1, 0),
                                  (1, 1, 0), (1, 3, 0), (0, 3, 0)))
    with pytest.raises(GeometryError, match='side outside'):
        validate(model, result, face, [[(.5, 2.5, 0), (2.5, .5, 0), (.25, .25, 0)]])
    model, face, result = fixture(hole=((.9, .9, 0), (.9, 1.1, 0),
                                      (1.1, 1.1, 0), (1.1, .9, 0)))
    with pytest.raises(GeometryError, match='encloses a hole'):
        validate(model, result, face, [[(.25, .25, 0), (3, .25, 0), (.25, 3, 0)]])
    validate(model, result, face, [[(2, 2, 0), (3, 2, 0), (2, 3, 0)]])


@pytest.mark.parametrize('bad', [np.zeros((3, 3)), np.full((1, 3, 3), np.nan),
    np.full((1, 3, 3), np.inf), np.full((1, 3, 3), 1+2j),
    [[[10**1000, 0, 0], [1, 0, 0], [0, 1, 0]]]])
def test_invalid_xyz_refuses_without_mutating_owner(bad):
    model, face, result = fixture()
    before = to_dict(model)
    with pytest.raises(GeometryError, match='finite'):
        validate(model, result, face, bad)
    assert to_dict(model) == before


def test_unsupported_carrier_refuses_even_if_uv_coverage_would_qualify():
    model = GeometryModel()
    points = model.add_points(((0, 0, 0), (.5, .5, 0), (1, 0, 0)))
    edge = model.add_spline(points[0], points[1:-1], points[-1])
    face = model.extrude((edge,), (0, 0, 1))[0]
    model.set_face_surface(face, ExtrudedSurface(
        BezierDirectrix(((0, 0, 0), (.5, .5, 0), (1, 0, 0))), (0, 0, 1)))
    assert isinstance(model.faces[face].surface, ExtrudedSurface)
    result = query_material_surface_regions(model, (face,))
    with pytest.raises(GeometryError, match='implicit Plane'):
        validate(model, result, face, [[(0, 0, 0), (.5, .25, 0), (.25, .1875, .5)]])


def test_stale_wrong_face_and_callback_owner_definition_refuse():
    model, face, result = fixture()
    xyz = [[(1, 1, 0), (5, 1, 0), (1, 5, 0)]]
    other, foreign, _ = fixture()
    with pytest.raises(GeometryError, match='bound face'):
        validate(model, result, other.handle('face', foreign), xyz)
    before = to_dict(model)
    error = RuntimeError('cancel actual XYZ')
    def cancel(_):
        raise error
    with pytest.raises(RuntimeError) as caught:
        validate(model, result, face, xyz, cancellation_check=cancel)
    assert caught.value is error and to_dict(model) == before
    with pytest.raises(GeometryError, match='cancelled'):
        validate(model, result, face, xyz, cancellation_check=lambda _: True)
    def mutate(_):
        object.__setattr__(model.faces[face].surface, 'origin', np.array((0, 0, .125)))
        return False
    with pytest.raises(GeometryError, match='changed|stale|binding'):
        validate(model, result, face, xyz, cancellation_check=mutate)


def test_input_coercion_does_not_change_detached_owner_truth():
    model, face, result = fixture()
    plane = model.faces[face].surface
    source = to_dict(model)
    class Input:
        def __array__(self, dtype=None, copy=None):
            before = plane.origin
            object.__setattr__(plane, 'origin', np.array((0, 0, 100.)))
            made = np.array([[(1, 1, 100), (5, 1, 100), (1, 5, 100)]], dtype=dtype)
            object.__setattr__(plane, 'origin', before)
            return made
    with pytest.raises(GeometryError, match='exact Plane support correspondence'):
        validate(model, result, face, Input())
    assert to_dict(model) == source


@pytest.mark.parametrize('during_coercion', [False, True])
def test_coordinated_owner_and_receipt_rebinding_cannot_replace_detached_domain(during_coercion):
    model, face, result = fixture()
    xyz = np.array([[(1, 1, 0), (5, 1, 0), (1, 5, 0)]], float)
    replaced = []
    def rebind():
        if replaced:
            return
        replaced.append(True)
        edges = model.add_polyline(model.add_points(((1.125, 1.125, 0),
            (1.125, 1.25, 0), (1.25, 1.25, 0), (1.25, 1.125, 0))), close=True)
        with model.transaction():
            model._put_entity('face', replace(model.faces[face],
                holes=(tuple(OrientedEdge(edge, True) for edge in edges),)))
        fresh = query_material_surface_regions(model, (face,))
        object.__setattr__(result, 'source', fresh.source)
        object.__setattr__(result, 'regions', fresh.regions)
    class Input:
        def __array__(self, dtype=None, copy=None):
            rebind()
            return np.array(xyz, dtype=dtype, copy=True)
    def callback(_):
        rebind()
        return False
    with pytest.raises(GeometryError, match='binding|changed|stale'):
        validate(model, result, face, Input() if during_coercion else xyz,
                 cancellation_check=None if during_coercion else callback)
    assert replaced
