"""Independent affine examples and compatibility of the document frame API."""

from __future__ import annotations

import numpy as np
import pytest

import anygeometry as ag
from anygeometry.automation import AutomationError, Command, CommandBatch, apply_plan, plan_commands

CONVERTERS = (ag.model_to_world_points, ag.world_to_model_points,
              ag.model_to_world_vectors, ag.world_to_model_vectors)


@pytest.mark.parametrize("convert", CONVERTERS)
@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 1, 3), (0, 3)])
def test_identity_is_an_independent_array_with_original_shape(convert, shape):
    model = ag.GeometryModel()
    values = np.arange(np.prod(shape), dtype=float).reshape(shape)
    made = convert(model, values)
    assert made.shape == shape
    np.testing.assert_array_equal(made, values)
    assert not np.shares_memory(made, values)


@pytest.mark.parametrize("linear", [
    np.eye(3),
    np.array(((0., -1., 0.), (1., 0., 0.), (0., 0., 1.))),
    np.diag((-2., 3., 4.)),
    np.array(((2., 1., 0.), (0., 3., 1.), (0., 0., 4.))),
])
def test_affine_basis_translation_vectors_and_inverse(linear):
    model = ag.GeometryModel()
    matrix = np.eye(4)
    matrix[:3, :3] = linear
    matrix[:3, 3] = (10., -20., 30.)
    model.set_document_settings(coordinate_transform=matrix)
    # Basis vectors map to the columns of the linear map, by definition.
    expected_vectors = linear.T
    expected_points = expected_vectors + (10., -20., 30.)
    np.testing.assert_allclose(ag.model_to_world_points(model, np.eye(3)), expected_points, atol=1e-14)
    np.testing.assert_allclose(ag.model_to_world_vectors(model, np.eye(3)), expected_vectors, atol=1e-14)
    np.testing.assert_allclose(ag.world_to_model_points(model, expected_points), np.eye(3), atol=1e-14)
    np.testing.assert_allclose(ag.world_to_model_vectors(model, expected_vectors), np.eye(3), atol=1e-14)
    np.testing.assert_allclose(ag.model_to_world_vectors(model, [0., 0., 0.]), [0., 0., 0.])


def test_units_origin_crs_and_queries_never_change_document_or_input():
    model = ag.GeometryModel()
    model.add_point(1., 2., 3.)
    matrix = np.eye(4)
    matrix[:3, 3] = (10000., 20000., 30000.)
    model.set_document_settings(units="mm", coordinate_transform=matrix,
                                local_origin=(1e6, -2e6, 3e6), crs_metadata={"name": "opaque-label"})
    before = ag.to_dict(model)
    source = np.array([[[1000., 0., 0.], [0., 2000., 0.]]])
    snapshot = source.copy()
    world = ag.model_to_world_points(model, source)
    np.testing.assert_array_equal(world[0, 0], [11000., 20000., 30000.])
    np.testing.assert_allclose(ag.world_to_model_points(model, world), source, atol=1e-12)
    for convert in CONVERTERS:
        convert(model, source)
    np.testing.assert_array_equal(source, snapshot)
    assert ag.to_dict(model) == before
    restored = ag.from_dict(before)
    np.testing.assert_array_equal(ag.model_to_world_points(restored, source), world)


@pytest.mark.parametrize("convert", CONVERTERS)
@pytest.mark.parametrize("invalid", [1., [], [1., 2.], [[1., 2.], [3., 4.]],
                                     [np.nan, 0., 0.], [np.inf, 0., 0.],
                                     [1j, 0., 0.], ["bad", 0., 0.], [10**400, 0, 0]])
def test_invalid_input_is_a_geometry_error(convert, invalid):
    model = ag.GeometryModel()
    before = ag.to_dict(model)
    with pytest.raises(ag.GeometryError):
        convert(model, invalid)
    assert ag.to_dict(model) == before


def test_overflow_and_unusable_homogeneous_inverse_fail_closed():
    model = ag.GeometryModel()
    matrix = np.eye(4)
    matrix[0, 0] = 2.
    model.set_document_settings(coordinate_transform=matrix)
    for convert in (ag.model_to_world_points, ag.model_to_world_vectors):
        with pytest.raises(ag.GeometryError, match="nonfinite"):
            convert(model, [np.finfo(float).max, 0., 0.])

    # The existing setter admits near-affine bottom rows. Preserve their full
    # homogeneous meaning, including rejection of a singular 4x4 inverse.
    matrix = np.eye(4)
    matrix[0, 3], matrix[3, 0] = 2.**54, 2.**-54
    model.set_document_settings(coordinate_transform=matrix)
    before = ag.to_dict(model)
    with pytest.raises(ag.GeometryError, match="inverse"):
        ag.world_to_model_points(model, [0., 0., 0.])
    with pytest.raises(ag.GeometryError, match="homogeneous"):
        ag.model_to_world_points(model, [-2.**54, 0., 0.])
    assert ag.to_dict(model) == before


def test_near_affine_homogeneous_normalization_is_preserved():
    model = ag.GeometryModel()
    matrix = np.eye(4)
    matrix[3, 0] = 1e-9
    model.set_document_settings(coordinate_transform=matrix)
    expected = np.array([2., 3., 4.]) / (1. + 2e-9)
    np.testing.assert_array_equal(ag.model_to_world_points(model, [2., 3., 4.]), expected)
    np.testing.assert_allclose(ag.world_to_model_points(model, expected), [2., 3., 4.], atol=1e-14)


def test_automation_converts_metres_once_and_keeps_vector_translation_separate():
    model = ag.GeometryModel()
    matrix = np.array(((0., -1., 0., 10000.), (1., 0., 0., 20000.),
                       (0., 0., 1., 30000.), (0., 0., 0., 1.)))
    model.set_document_settings(units="mm", coordinate_transform=matrix, local_origin=(99., 88., 77.))
    q = lambda value, unit="m": {"value": value, "unit": unit, "frame": "world"}
    batch = CommandBatch(1, "world-example", model.model_id, model.revision, (
        Command("p", "create_point", {"position": q((10., 22., 30.))}),
        Command("shift", "translate", {"targets": ["p.vertex"], "vector": q((1., 0., 0.))}),
        Command("turn", "rotate", {"targets": ["p.vertex"], "axis_point": q((10., 20., 30.)),
                                    "axis_direction": q((0., 0., 1.), "1"),
                                    "angle": {"value": 90., "unit": "deg"}}),
    ))
    result = apply_plan(model, plan_commands(model, batch))
    handle = result.outputs["p.vertex"][0]
    np.testing.assert_allclose(model.vertex_position(handle.id), [1000., 2000., 0.], atol=1e-10)


def test_automation_retains_unknown_unit_and_frame_errors():
    model = ag.GeometryModel()
    for quantity, code in (({"value": (1., 2., 3.), "unit": "furlong", "frame": "world"}, "UNKNOWN_UNIT"),
                           ({"value": (1., 2., 3.), "unit": "m"}, "UNKNOWN_FRAME")):
        batch = CommandBatch(1, "invalid", model.model_id, model.revision,
                             (Command("p", "create_point", {"position": quantity}),))
        with pytest.raises(AutomationError) as caught:
            plan_commands(model, batch)
        assert caught.value.code == code
