"""Independent small material-coverage examples; no mesher or large model."""
from dataclasses import replace
from fractions import Fraction as F

import numpy as np
import pytest

from anygeometry import (GeometryModel, GeometryError, Plane, BezierDirectrix,
    ExtrudedSurface, query_material_surface_regions, to_dict,
    validate_material_surface_region_triangles as validate)
from anygeometry.entities import OrientedEdge
from anygeometry.arrangement_geometry import BezierPath, LinePath
from anygeometry.material_cell_coverage import (_frame, _lift, _value, _roots, _events,
    _chart_polynomial, _loop_location, _inside_exact)


def plane(points=((0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0)), hole=None):
    model = GeometryModel()
    face = model.add_plate(model.add_points(points))
    model.set_face_surface(face, Plane((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    if hole is not None:
        edges = model.add_polyline(model.add_points(hole), close=True)
        # Author a topological clockwise polygon hole, independently of the
        # coverage implementation. Source validation still qualifies the model.
        with model.transaction():
            model._put_entity('face', replace(model.faces[face],
                holes=(tuple(OrientedEdge(edge, True) for edge in edges),)))
    return model, face, query_material_surface_regions(model, (face,))


def extrusion(controls=((0, 0, 0), (.5, .5, 0), (1, 0, 0))):
    model = GeometryModel()
    points = model.add_points(controls)
    edge = model.add_spline(points[0], points[1:-1], points[-1])
    face = model.extrude((edge,), (0, 0, 1))[0]
    return model, face, query_material_surface_regions(model, (face,))


def test_plane_closed_child_and_reversed_orientation_preserve_definition():
    model, face, result = plane()
    triangles = np.array([((0, 0), (2, 0), (1, 1)), ((1, 1), (1, 3), (3, 1))], float)
    source = to_dict(model); before = triangles.tobytes()
    assert validate(model, result, face, triangles) is None
    assert validate(model, result, model.handle('face', face), triangles[:, ::-1]) is None
    assert to_dict(model) == source and triangles.tobytes() == before


@pytest.mark.parametrize('controls', [((0, 0, 0), (.5, .5, 0), (1, 0, 0)),
    ((0, 0, 0), (.25, .5, 0), (.75, -.25, 0), (1, 0, 0))])
def test_bezier_extrusion_whole_child_and_partial_polynomial_boundary_overlap(controls):
    model, face, result = extrusion(controls)
    before = to_dict(model)
    validate(model, result, face, [[[0, 0], [.5, 0], [.25, .5]],
                                  [[.25, .25], [.75, .25], [.5, .75]]])
    assert to_dict(model) == before


def test_quadratic_lift_matches_independent_exact_formula_without_rounded_controls():
    support = ExtrudedSurface(BezierDirectrix(((0, 0, 0), (.5, .5, 0), (1, 0, 0))), (1, 0, 2))
    powers = _lift(_frame(support), (F(1, 4), F(1, 8)), (F(3, 4), F(7, 8)))
    for t in (F(0), F(1, 3), F(1)):
        u, v = F(1, 4)+t/2, F(1, 8)+3*t/4
        assert tuple(_value(row, t) for row in powers) == (u+v, u-u*u, 2*v)


def test_cubic_reversed_subpatch_lift_exact_bernstein_oracle():
    controls = ((0, 0, 0), (.25, .5, 0), (.75, -.25, 0), (1, 0, 0))
    support = ExtrudedSurface(BezierDirectrix(controls), (0, 0, 2), (.75, .25), (.125, .875))
    powers = _lift(_frame(support), (F(0), F(1)), (F(1), F(0)))
    for s in (F(0), F(2, 7), F(1)):
        t, v = F(3, 4)-s/2, F(7, 8)-3*s/4
        weights = ((1-t)**3, 3*t*(1-t)**2, 3*t*t*(1-t), t**3)
        expected = tuple(sum(F(float(point[k]))*w for point, w in zip(controls, weights))
                         +(2*v if k == 2 else 0) for k in range(3))
        assert tuple(_value(row, s) for row in powers) == expected


def test_inside_vertices_do_not_certify_side_crossing_concave_trim():
    model, face, result = plane(((0, 0, 0), (3, 0, 0), (3, 1, 0),
                                (1, 1, 0), (1, 3, 0), (0, 3, 0)))
    with pytest.raises(GeometryError, match='side outside'):
        validate(model, result, face, [[[.5, 2.5], [2.5, .5], [.25, .25]]])


def test_wholly_enclosed_hole_is_not_certified_by_three_sides():
    hole = ((.9, .9, 0), (.9, 1.1, 0), (1.1, 1.1, 0), (1.1, .9, 0))
    model, face, result = plane(hole=hole)
    with pytest.raises(GeometryError, match='encloses a hole'):
        validate(model, result, face, [[[.25, .25], [3, .25], [.25, 3]]])
    validate(model, result, face, [[[2, 2], [3, 2], [2, 3]]])


def test_repeated_tangent_root_retained_once_without_proximity_merge():
    assert _roots(((F(1, 4), -F(1), F(1)),), lambda: None) == [(F(1, 2), F(1, 2))]
    # Nearby DISTINCT dyadic roots are retained, not merged by a world epsilon.
    delta = F(1, 2**30)
    roots = _roots(((F(1, 4)+delta/2, -F(1)-delta, F(1)),), lambda: None)
    assert roots == [(F(1, 2), F(1, 2)), (F(1, 2)+delta, F(1, 2)+delta)]


def test_unproved_common_image_reparameterization_refuses():
    # Same line traced quadratically versus linearly; no assumed affine map.
    with pytest.raises(GeometryError, match='unresolved polynomial overlap'):
        _events(((F(0), F(0), F(1)), (F(0),), (F(0),)),
                BezierPath(((0., 0., 0.), (1., 0., 0.))), lambda: None)


def test_distinct_algebraic_roots_in_identical_brackets_refuse():
    first = (-F(1, 2), F(0), F(1))
    second = (-F(1, 2)-F(1, 2**60), F(0), F(1))
    with pytest.raises(GeometryError, match='clustered'):
        _roots((first, second), lambda: None)
    assert _roots((first, tuple(2*x for x in first)), lambda: None) == _roots((first,), lambda: None)


@pytest.mark.parametrize('bad', [np.zeros((3, 2)), np.full((1, 3, 2), np.nan),
                                np.full((1, 3, 2), np.inf),
                                [[[10**1000, 0], [1, 0], [0, 1]]],
                                np.full((1, 3, 2), 1+2j)])
def test_invalid_input_refuses_without_mutation(bad):
    model, face, result = plane(); before = to_dict(model)
    with pytest.raises(GeometryError, match='finite'):
        validate(model, result, face, bad)
    assert to_dict(model) == before


def test_degenerate_outside_stale_and_tampered_cases_refuse():
    model, face, result = plane()
    with pytest.raises(GeometryError, match='degenerate'):
        validate(model, result, face, [[[0, 0], [1, 0], [2, 0]]])
    with pytest.raises(GeometryError, match='outside'):
        validate(model, result, face, [[[5, 5], [6, 5], [5, 6]]])
    altered = replace(result, regions=(replace(result.regions[0], material_area=20),))
    with pytest.raises(GeometryError, match='binding changed'):
        validate(model, altered, face, [[[1, 1], [2, 1], [1, 2]]])
    model.add_point(10, 10, 10)
    with pytest.raises(GeometryError, match='stale'):
        validate(model, result, face, [[[1, 1], [2, 1], [1, 2]]])


def test_cancellation_identity_and_input_snapshot():
    model, face, result = plane(); before = to_dict(model)
    error = RuntimeError('cancel exact coverage')
    def cancel(phase):
        if phase == 'material cell coverage':
            raise error
    with pytest.raises(RuntimeError) as caught:
        validate(model, result, face, [[[1, 1], [2, 1], [1, 2]]], cancellation_check=cancel)
    assert caught.value is error and to_dict(model) == before


def test_nonmonotone_and_extrapolated_profile_refusals():
    support = ExtrudedSurface(BezierDirectrix(((0, 0, 0), (1, 1, 0), (-1, 1, 0), (0, 0, 0))), (0, 0, 1))
    with pytest.raises(GeometryError, match='injectivity'):
        _frame(support)
    support = ExtrudedSurface(BezierDirectrix(((0, 0, 0), (.5, .5, 0), (1, 0, 0))),
                              (0, 0, 1), (-.25, .75))
    with pytest.raises(GeometryError, match='extrapolated'):
        _frame(support)


def test_exact_branch_support_crossing_tangency_and_transformed_refusal():
    from anygeometry.branch_algebra import BezierRuledSupport
    from anygeometry.branch_curves import BezierQuadricCurve
    from anygeometry.quadric_algebra import QuadricSupport
    first = BezierRuledSupport(((0., 0., 0.), (.5, 0., 0.), (1., 1., 0.)), (0., 0., 1.))
    second = QuadricSupport('general', matrix=(0., 0., 0., 0., 0., 0., 0., 0., 1.),
                            linear=(-.5, 0., 0.))
    curve = BezierQuadricCurve(first, second, 0., 1., parameterization='left_square')
    crossing = ((F(0), F(1)), (F(0), F(0), F(1)), (F(1, 2),))
    events, overlap = _events(crossing, curve, lambda: None)
    assert not overlap and _roots(events, lambda: None) == [(F(1, 4), F(1, 4))]
    tangent = (crossing[0], crossing[1], (F(1, 4), F(1)))
    events, overlap = _events(tangent, curve, lambda: None)
    assert not overlap and _roots(events, lambda: None) == [(F(1, 4), F(1, 4))]
    matrix = np.eye(4); matrix[0, 1] = .125; matrix[0, 3] = .1
    with pytest.raises(GeometryError, match='transformed branch'):
        _events(crossing, curve.transformed(matrix), lambda: None)


def test_input_is_detached_before_callback_mutation_and_end_binding_rechecked():
    model, face, result = plane()
    triangles = np.array([[[1., 1.], [2., 1.], [1., 2.]]])
    def mutate_input(phase):
        triangles[:] = 50.
    validate(model, result, face, triangles, cancellation_check=mutate_input)
    calls = 0
    def mutate_model(phase):
        nonlocal calls
        calls += 1
        if calls == 3:
            model.add_point(10, 10, 10)
    with pytest.raises(GeometryError, match='stale'):
        validate(model, result, face, [[[1, 1], [2, 1], [1, 2]]], cancellation_check=mutate_model)


def test_periodic_support_refuses_explicitly():
    from anygeometry import Cylinder
    with pytest.raises(GeometryError, match='unsupported support'):
        _frame(Cylinder((0, 0, 0), (0, 0, 1), (1, 0, 0), 1., 1.))


def test_collapsed_first_projection_retains_vertical_generator_contact():
    # The trim projects to a point in xy. A degree-zero Sylvester determinant
    # of one would erase the actual t=1/2 event on the parabolic wall.
    side = ((F(0), F(1)), (F(0), F(0), F(1)), (F(1, 2),))
    events, overlap = _events(side, LinePath((.5, .25, 0.), (.5, .25, 1.)), lambda: None)
    assert not overlap and _roots(events, lambda: None) == [(F(1, 2), F(1, 2))]


def test_tolerance_offset_tilted_plane_trim_is_not_exact_coverage():
    delta = 2.**-40
    model = GeometryModel()
    face = model.add_plate(model.add_points(((0, 0, delta), (1, 0, 1+delta),
                                            (1, 1, 1+delta), (0, 1, delta))))
    model.set_face_surface(face, Plane((0, 0, 0), (1, 0, 1), (0, 1, 0)))
    result = query_material_surface_regions(model, (face,))
    before = to_dict(model)
    with pytest.raises(GeometryError, match='not exactly on support'):
        validate(model, result, face, [[[delta/4, .5], [.75, .25], [.75, .75]]])
    assert to_dict(model) == before


def test_exact_large_offset_skew_plane_chart_kernel():
    # The historical region query cannot qualify this large-offset fixture;
    # exercise the new exact correspondence/winding proof directly, without
    # claiming public query acceptance or changing that prerequisite.
    origin = 2.**40
    support = Plane((origin, origin, origin), (1, 0, 1), (0, 2, 0))
    points = ((origin, origin, origin), (origin+2, origin, origin+2),
              (origin+2, origin+4, origin+2), (origin, origin+4, origin))
    loop = tuple(_chart_polynomial(_frame(support), LinePath(first, second), lambda: None)
                 for first, second in zip(points, (*points[1:], points[0])))
    assert _loop_location(loop, (F(1, 3), F(1, 3)), lambda: None) == 1
    assert _loop_location(loop, (-F(1, 3), F(1, 3)), lambda: None) == -1


def _rational_rectangle(left, right, bottom=F(0), top=F(1)):
    points = ((left, bottom), (right, bottom), (right, top), (left, top))
    return tuple(tuple((a, b-a) if b != a else (a,) for a, b in zip(first, second))
                 for first, second in zip(points, (*points[1:], points[0])))


def test_rational_winding_does_not_round_into_another_component():
    left, delta = F(1, 2), F(1, 2**60)
    loop = _rational_rectangle(left, left+delta)
    inside, outside = (left+delta/2, F(1, 2)), (left-delta/2, F(1, 2))
    assert float(inside[0]) == float(outside[0]) == .5
    assert _loop_location(loop, inside, lambda: None) == 1
    assert _loop_location(loop, outside, lambda: None) == -1
    assert _inside_exact((loop,), inside, lambda: None)
    assert not _inside_exact((loop,), outside, lambda: None)


def test_exact_winding_tangent_and_boundary_are_distinguished():
    # Upper arch v=4t(1-t), closed by the base: horizontal ray at its maximum
    # has a repeated root and must not count a crossing.
    arch = ((F(0), F(1)), (F(0), F(4), -F(4)))
    base = ((F(1), -F(1)), (F(0),))
    assert _loop_location((arch, base), (-F(1), F(1)), lambda: None) == -1
    assert _loop_location((arch, base), (F(1, 2), F(1, 2)), lambda: None) == 1
    assert _loop_location((arch, base), (F(1, 2), F(1)), lambda: None) == 0


def test_extrusion_tolerance_only_trim_correspondence_refuses():
    frame = _frame(ExtrudedSurface(BezierDirectrix(((0, 0, 0), (.5, .5, 0), (1, 0, 0))), (0, 0, 1)))
    with pytest.raises(GeometryError, match='chart correspondence'):
        _chart_polynomial(frame, BezierPath(((0., 0., 0.), (.5, .5+2.**-40, 0.),
                                             (1., 0., 0.))), lambda: None)


def test_exact_chart_refuses_branch_trim_even_with_identity_transform():
    from anygeometry.branch_algebra import BezierRuledSupport
    from anygeometry.branch_curves import BezierQuadricCurve
    from anygeometry.quadric_algebra import QuadricSupport
    support = ExtrudedSurface(BezierDirectrix(((0, 0, 0), (.5, 0, 0), (1, 1, 0))), (0, 0, 1))
    quadric = QuadricSupport('general', matrix=(0., 0., 0., 0., 0., 0., 0., 0., 1.),
                             linear=(-.5, 0., 0.))
    curve = BezierQuadricCurve(BezierRuledSupport.from_surface(support), quadric, 0., 1.,
                              parameterization='left_square')
    with pytest.raises(GeometryError, match='unsupported polynomial trim'):
        _chart_polynomial(_frame(support), curve, lambda: None)
