"""Analytic repeated-root oracles for parallel-wall joints on one pipe."""
import math

import numpy as np
import pytest

from anygeometry import GeometryError
from anygeometry.branch_algebra import BezierRuledSupport
from anygeometry.branch_curves import BezierQuadricCurve
from anygeometry.branch_events import bezier_quadric_junctions
from anygeometry.branch_wall_events import parallel_wall_parameters
from anygeometry.quadric_algebra import QuadricSupport


def pair(offset=0., reverse=False):
    # x=3t; first y=3t^3; second y=3t^3+3(2t-1)^2+offset.
    a = ((0., 0., 0.), (1., 0., 0.), (2., 0., 0.), (3., 3., 0.))
    b = tuple((float(i), y + offset, 0.) for i, y in enumerate((3., -1., -1., 6.)))
    pipe = QuadricSupport('general', matrix=(0., 0., 0., 0., 1., 0., 0., 0., 1.),
                          linear=(0., 0., 0.), constant=-4.)
    start, sweep = (.65, -.4) if reverse else (.25, .4)
    return tuple(BezierQuadricCurve(BezierRuledSupport(c, (0., 0., 1.)), pipe, start, sweep)
                 for c in (a, b))


@pytest.mark.parametrize('offset', [0., -.03125, .03125])
@pytest.mark.parametrize('reverse', [False, True])
def test_tangent_secant_and_separated_wall_branches(offset, reverse):
    a, b = pair(offset, reverse)
    expected = [] if offset > 0 else [.5] if offset == 0 else [
        .5 * (1 - math.sqrt(-offset / 3)), .5 * (1 + math.sqrt(-offset / 3))]
    for first, second in ((a, b), (b, a)):
        roots = bezier_quadric_junctions(first, second)
        assert len(roots) == len(expected)
        actual = sorted(first.start + first.sweep * u for u, _ in roots)
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)
        for u, v in roots:
            t = first.start + first.sweep * u
            y = 3*t**3
            witness = (3*t, y, math.sqrt(4-y*y))
            np.testing.assert_allclose(first.evaluate(u), witness, atol=1e-13, rtol=0)
            np.testing.assert_allclose(second.evaluate(v), witness, atol=1e-13, rtol=0)


def test_common_affine_image_and_opposite_branch():
    a, b = pair()
    transform = np.array(((1., .25, 0., 4.), (0., -2., .5, -1.), (.5, 0., 3., 2.), (0., 0., 0., 1.)))
    a, b = a.transformed(transform), b.transformed(transform)
    roots = bezier_quadric_junctions(a, b)
    assert len(roots) == 1
    np.testing.assert_allclose(a.evaluate(roots[0][0]), b.evaluate(roots[0][1]), atol=1e-13)
    a, b = pair()
    opposite = BezierQuadricCurve(b.first, b.second, b.start, b.sweep, -1)
    assert bezier_quadric_junctions(a, opposite) == ()


def test_generator_predicate_cancellation_and_common_component():
    a, b = pair()
    with pytest.raises(GeometryError, match='cancelled'):
        parallel_wall_parameters(a, b, cancellation_check=lambda: True)
    assert parallel_wall_parameters(a, a) is None


def test_all_visits_of_a_self_crossing_directrix_are_qualified():
    a, _ = pair()
    first = ((-.5, .25, 0.), (0., -.25, 0.), (.5, .25, 0.))
    second = ((3/16, -3/32, 0.), (-7/48, 13/96, 0.),
              (-7/48, -13/96, 0.), (3/16, 3/32, 0.))
    curves = [BezierQuadricCurve(BezierRuledSupport(c, (0., 0., 1.)), a.second, 0., 1.)
              for c in (first, second)]
    roots = bezier_quadric_junctions(*curves)
    for expected in ((.5, .25), (.5, .75)):
        assert any(np.linalg.norm(np.asarray(found) - expected) < 1e-12 for found in roots)
    assert len(roots) == 3


def test_scaled_junctions_obey_world_tolerance_or_refuse_explicitly():
    a, b = pair(-.03125)
    transform = np.diag((1e6, 1e6, 1e6, 1.))
    a, b = a.transformed(transform), b.transformed(transform)
    try:
        roots = bezier_quadric_junctions(a, b, tolerance=1e-10)
    except GeometryError as error:
        assert 'world tolerance' in str(error)
    else:
        assert len(roots) == 2
        for u, v in roots:
            assert np.linalg.norm(a.evaluate(u) - b.evaluate(v)) <= 1e-10


def test_tangent_walls_and_pipe_form_one_shared_junction_in_either_insertion_order():
    from anygeometry import apply_intersections, plan_intersections, to_dict, from_dict
    from test_branch_models import spline_wall, add_pipe, material_area

    controls = (((0., 0., 0.), (4., 0., 0.), (8., 0., 0.), (12., 3., 0.)),
                ((0., 3., 0.), (4., -1., 0.), (8., -1., 0.), (12., 6., 0.)))
    expected = np.array((6., .375, math.sqrt(4 - .375**2)))
    signatures = []
    for ordered in (controls, controls[::-1]):
        model, _ = spline_wall(ordered[0], vector=(0., 0., 4.))
        model.insert_model(spline_wall(ordered[1], vector=(0., 0., 4.))[0])
        faces = add_pipe(model, list(model.faces), radius=2., y=0., z=0., length=16.)
        before, area = to_dict(model), material_area(model)
        plan = plan_intersections(model, [model.handle('face', f) for f in faces], policy='connect')
        assert to_dict(model) == before
        applied = apply_intersections(model, plan, policy='connect')
        assert model.validate_topology() == ()
        assert material_area(model) == pytest.approx(area, rel=1e-12)
        vertices = [v.id for v in model.vertices.values() if np.linalg.norm(v.position - expected) < 1e-12]
        assert len(vertices) == 1
        incident = set(model.edges_using_vertex(vertices[0])) & {e.id for e in applied.joint_edges}
        assert len(incident) >= 4
        assert len({f for e in incident for f in model.faces_using_edge(e)}) >= 3
        after = to_dict(model)
        assert apply_intersections(model, plan, policy='connect').reused
        assert to_dict(model) == after
        assert to_dict(from_dict(after)) == after
        signatures.append(sorted(tuple(np.round(model.sample_edge(e.id, np.array([.5]))[0], 10))
                                 for e in applied.joint_edges))
    assert signatures[0] == signatures[1]
