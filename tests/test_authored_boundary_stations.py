"""Owner source/UV correspondence, never inferred global node identity."""
from dataclasses import replace
from fractions import Fraction as F

import numpy as np
import pytest

from anygeometry import GeometryError, to_dict
from anygeometry.authored_boundary_stations import (
    query_prepared_authored_boundary_stations as query,
    validate_prepared_authored_boundary_station_coordinates as validate)


def unpack(rows):
    return tuple(tuple(F(*value) for value in row) for row in rows)


@pytest.mark.parametrize('cropped', (False, True))
def test_original_cubic_station_matches_independent_power_curve_and_preserves_xyz(cropped):
    from test_authored_domain_coverage import cubic
    model, correspondence = cubic(cropped)
    source = to_dict(model)
    edge = correspondence.exterior_loops[0][0][2][0]
    parameters = (F(0), F(1, 3), F(1))
    result = query(model, correspondence, edge, parameters)
    uv = unpack(result.authored_uv)
    xyz = unpack(result.authored_points)
    for (u, v), (x, y, z) in zip(uv, xyz):
        t = F(1, 4)+F(1, 2)*u if cropped else u
        s = F(1, 8)+F(3, 4)*v if cropped else v
        assert (x, y, z) == (3*t+s/4, 6*t-15*t*t+10*t*t*t, 3*s/2)
    points = model.sample_edge(edge, np.array(tuple(map(float, parameters))))
    before = points.tobytes()
    assert validate(model, result, points) is None
    assert points.tobytes() == before and to_dict(model) == source
    assert not hasattr(result, 'global_node_ids')
    assert not hasattr(result, 'publication_qualified')


def test_planar_original_uv_keeps_actual_source_parameters_on_reversed_hole_edge():
    from test_authored_domain_coverage import planar
    model, correspondence = planar(holes=True)
    edge = correspondence.exterior_loops[1][0][2][0]
    result = query(model, correspondence, edge, (F(0), F(1, 7), F(1)))
    assert tuple(row[:2] for row in unpack(result.authored_points)) == unpack(result.authored_uv)
    validate(model, result, model.sample_edge(edge, np.array((0., 1/7, 1.))))


@pytest.mark.parametrize('parameters', ((float('nan'),), (float('inf'),), (-.1,), (1.1,), (complex(1, 0),)))
def test_invalid_station_parameters_refuse(parameters):
    from test_authored_domain_coverage import cubic
    model, correspondence = cubic()
    edge = correspondence.exterior_loops[0][0][2][0]
    with pytest.raises(GeometryError):
        query(model, correspondence, edge, parameters)


def test_wrong_edge_forged_station_and_outside_coordinate_assertion_refuse():
    from test_authored_domain_coverage import cubic
    model, correspondence = cubic()
    with pytest.raises(GeometryError, match='exterior'):
        query(model, correspondence, correspondence.interior_incidence[0][0], (F(1, 2),))
    edge = correspondence.exterior_loops[0][0][2][0]
    result = query(model, correspondence, edge, (F(1, 2),))
    points = model.sample_edge(edge, np.array((.5,)))
    with pytest.raises(GeometryError, match='binding changed'):
        validate(model, replace(result, tolerance=(1, 1)), points)
    with pytest.raises(GeometryError, match='coordinate error'):
        validate(model, result, points+1e-4)
    for bad in (points.astype(complex)+1j, np.array(((np.nan, 0, 0),)), np.zeros((1, 2))):
        with pytest.raises(GeometryError):
            validate(model, result, bad)


def test_coordinate_inputs_are_detached_before_callbacks_and_source_is_not_modified():
    from test_authored_domain_coverage import cubic
    model, correspondence = cubic()
    edge = correspondence.exterior_loops[0][0][2][0]
    result = query(model, correspondence, edge, (.5,))
    points = model.sample_edge(edge, np.array((.5,)))
    source = to_dict(model)
    def mutate_input(_):
        points[:] = 100
        return False
    assert validate(model, result, points, cancellation_check=mutate_input) is None
    assert (points == 100).all() and to_dict(model) == source
    with pytest.raises(GeometryError, match='cancelled'):
        validate(model, result, model.sample_edge(edge, np.array((.5,))), cancellation_check=lambda _: True)


def test_late_callback_cannot_change_the_station_assertion(monkeypatch):
    from test_authored_domain_coverage import cubic
    import anygeometry.authored_boundary_stations as module
    model, correspondence = cubic()
    edge = correspondence.exterior_loops[0][0][2][0]
    stations = query(model, correspondence, edge, (.5,))
    points = model.sample_edge(edge, np.array((.5,)))
    ready = False
    done = False
    original = module.query_prepared_authored_boundary_stations
    def mark(*args, **kwargs):
        nonlocal ready
        result = original(*args, **kwargs)
        ready = True
        return result
    monkeypatch.setattr(module, 'query_prepared_authored_boundary_stations', mark)
    def callback(_):
        nonlocal done
        if ready and not done:
            done = True
            object.__setattr__(stations, 'tolerance', (1, 1))
        return False
    with pytest.raises(GeometryError, match='binding changed'):
        validate(model, stations, points, cancellation_check=callback)
    assert done


@pytest.mark.parametrize('point', (-.6, 1.4))
def test_both_coordinate_balls_are_required_independently(point):
    from anygeometry.authored_boundary_stations import _assert_coordinate_errors
    # Analytic one-dimensional centers 0 and 4/5, common radius 1.
    # -3/5 is inside only the original ball; 7/5 only the current ball.
    original = (((0, 1), (0, 1), (0, 1)),)
    represented = (((4, 5), (0, 1), (0, 1)),)
    with pytest.raises(GeometryError, match='coordinate error'):
        _assert_coordinate_errors(np.array(((point, 0, 0),)), original, represented, F(1), lambda: None)
    _assert_coordinate_errors(np.array(((.4, 0, 0),)), original, represented, F(1), lambda: None)
