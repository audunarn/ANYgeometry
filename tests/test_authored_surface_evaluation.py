"""Independent original-support formulas and bound batch evaluation contracts."""
from dataclasses import replace

import numpy as np
import pytest

from anygeometry import GeometryError, GeometryModel, Plane, CoonsSurface, to_dict
from anygeometry.authored_surface_evaluation import evaluate_prepared_authored_face as evaluate
from test_authored_domain_coverage import planar, cubic, prepared


@pytest.mark.parametrize('cropped,implicit', [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize('extension', [False, True])
def test_cubic_original_power_formula_and_derivatives(cropped, implicit, extension):
    model, result = cubic(cropped=cropped, implicit=implicit)
    source = to_dict(model)
    uv = np.array([[[0., 0.], [.125, .75]], [[.5, .25], [1., 1.]]])
    if extension:
        # Dyadic values go beyond both patch and directrix ranges even when
        # cropped; independent power formulas distinguish any clamping.
        uv = np.array([[[-2., -.5], [3., 1.5]], [[-.75, 2.], [1.75, -1.]]])
    t = .25 + .5*uv[..., 0] if cropped else uv[..., 0]
    s = .125 + .75*uv[..., 1] if cropped else uv[..., 1]
    expected = np.stack((3*t+.25*s, 6*t-15*t*t+10*t*t*t, 1.5*s), axis=-1)
    expected_du = np.stack((np.full_like(t, 3), 6-30*t+30*t*t, np.zeros_like(t)), axis=-1)
    expected_dv = np.broadcast_to([.25, 0., 1.5], expected.shape)
    if cropped:
        expected_du *= .5
        expected_dv = expected_dv*.75
    xyz = evaluate(model, result, uv)
    du, dv = evaluate(model, result, uv, derivatives=True)
    np.testing.assert_allclose(xyz, expected, rtol=0, atol=2e-15)
    np.testing.assert_allclose(du, expected_du, rtol=0, atol=2e-15)
    np.testing.assert_allclose(dv, expected_dv, rtol=0, atol=2e-15)
    assert xyz.shape == du.shape == dv.shape == (2, 2, 3)
    assert to_dict(model) == source


def test_tilted_plane_independent_formula_and_derivatives():
    model = GeometryModel()
    origin = np.array([2., -1., 3.])
    first = np.array([2., 0., 1.]); second = np.array([0., 3., 2.])
    face = model.add_plate(model.add_points([origin, origin+first, origin+first+second, origin+second]))
    model.set_face_surface(face, Plane(origin, first, second))
    model, result = prepared(model, face)
    uv = np.array([[.125, .25], [.75, .5]])
    np.testing.assert_array_equal(evaluate(model, result, uv), origin+uv[:, :1]*first+uv[:, 1:]*second)
    du, dv = evaluate(model, result, uv, derivatives=True)
    np.testing.assert_array_equal(du, np.tile(first, (2, 1)))
    np.testing.assert_array_equal(dv, np.tile(second, (2, 1)))


@pytest.mark.parametrize('shape', [(2,), (0, 2), (2, 0, 2), (2, 3, 2)])
@pytest.mark.parametrize('derivatives', [False, True])
def test_scalar_empty_and_multiaxis_shapes_are_detached(shape, derivatives):
    model, result = planar(fragment=False)
    uv = np.zeros(shape)
    output = evaluate(model, result, uv, derivatives=derivatives)
    arrays = output if derivatives else (output,)
    assert len(arrays) == (2 if derivatives else 1)
    for array in arrays:
        assert array.shape == (*shape[:-1], 3)
        assert not np.shares_memory(array, uv)
    if derivatives:
        assert not np.shares_memory(*arrays)


def test_holes_and_outside_points_evaluate_without_membership_claim():
    model, result = planar(holes=True, fragment=False)
    uv = np.array([[1., 1.], [2.7, 2.7], [-2., 7.]])
    np.testing.assert_array_equal(evaluate(model, result, uv), np.column_stack((uv, np.zeros(3))))


@pytest.mark.parametrize('uv', [1, [], [1, 2, 3], [[1], [2]], [[np.inf, 0]],
    [[np.nan, 0]], np.array([[1+0j, 0j]]), np.array([[1+2j, 0]], dtype=object),
    [[10**400, 0]], [['1', '2']]])
def test_invalid_input_refuses_before_callbacks(uv):
    model, result = planar(fragment=False)
    def callback(_):
        raise AssertionError('invalid input reached callback')
    with pytest.raises(GeometryError, match='finite real'):
        evaluate(model, result, uv, cancellation_check=callback)


def test_snapshot_precedes_callback_and_outputs_do_not_mutate_source():
    model, result = planar(fragment=False); source = to_dict(model)
    uv = np.array([[.25, .75]])
    def mutate(_):
        uv[:] = 99
    xyz = evaluate(model, result, uv, cancellation_check=mutate)
    np.testing.assert_array_equal(xyz, [[.25, .75, 0.]])
    xyz[:] = -22
    np.testing.assert_array_equal(evaluate(model, result, [[.25, .75]]), [[.25, .75, 0.]])
    assert to_dict(model) == source


def test_stale_tampered_and_wrong_owner_refuse():
    model, result = planar(fragment=False)
    forged = replace(result, authored_definition=replace(result.authored_definition, definition_json='{}'))
    with pytest.raises(GeometryError, match='definition binding'):
        evaluate(model, forged, [0, 0])
    with pytest.raises(GeometryError):
        evaluate(GeometryModel(), result, [0, 0])
    model.add_point(20, 20, 20)
    with pytest.raises(GeometryError, match='stale'):
        evaluate(model, result, [0, 0])


@pytest.mark.parametrize('late', [False, True])
def test_cancellation_exception_identity_at_entry_and_after_evaluation(monkeypatch, late):
    import anygeometry.authored_surface_evaluation as module
    model, result = planar(fragment=False); source = to_dict(model)
    completed = False
    original = module._evaluate_surface_many
    def measured(*args):
        nonlocal completed
        output = original(*args); completed = True
        return output
    monkeypatch.setattr(module, '_evaluate_surface_many', measured)
    error = RuntimeError('cancel authored evaluation')
    def cancel(_):
        if completed or not late:
            raise error
    with pytest.raises(RuntimeError) as caught:
        evaluate(model, result, [0, 0], cancellation_check=cancel)
    assert caught.value is error and to_dict(model) == source


def test_truthy_cancellation_is_typed():
    model, result = planar(fragment=False)
    with pytest.raises(GeometryError, match='cancelled'):
        evaluate(model, result, [0, 0], cancellation_check=lambda _: True)


@pytest.mark.parametrize('derivatives', [False, True])
def test_late_callback_mutation_cannot_publish(monkeypatch, derivatives):
    import anygeometry.authored_surface_evaluation as module
    model, result = planar(fragment=False)
    name = '_surface_derivatives_many' if derivatives else '_evaluate_surface_many'
    original = getattr(module, name); completed = False
    def measured(*args):
        nonlocal completed
        output = original(*args); completed = True
        return output
    monkeypatch.setattr(module, name, measured)
    def mutate(_):
        nonlocal completed
        if completed:
            completed = False
            model.add_point(20, 20, 20)
    with pytest.raises(GeometryError):
        evaluate(model, result, [0, 0], derivatives=derivatives, cancellation_check=mutate)


@pytest.mark.parametrize('kind', ['parameterization', 'coons'])
def test_unsupported_original_support_or_parameterization_refuses(kind):
    model = GeometryModel()
    face = model.add_plate(model.add_points(((0,0,0),(1,0,0),(1,1,0),(0,1,0))))
    if kind == 'parameterization':
        model.set_face_parameterization(face, Plane((0,0,0),(2,0,0),(0,2,0)))
    else:
        model.set_face_surface(face, CoonsSurface())
    model, result = prepared(model, face)
    with pytest.raises(GeometryError, match='parameterization|Bezier boundaries'):
        evaluate(model, result, [0, 0])


def test_last_binding_callback_mutation_is_rejected():
    model, result = planar(fragment=False)
    phases = []
    evaluate(model, result, [0, 0], cancellation_check=lambda phase: phases.append(phase))
    remaining = len(phases)
    def mutate_last(_):
        nonlocal remaining
        remaining -= 1
        if remaining == 0:
            model.add_point(20, 20, 20)
    with pytest.raises(GeometryError):
        evaluate(model, result, [0, 0], cancellation_check=mutate_last)
    assert remaining == 0


@pytest.mark.parametrize('option', ['False', 0, 1, None, np.array(True)])
def test_non_boolean_derivative_option_refuses_before_callback(option):
    model, result = planar(fragment=False)
    def callback(_):
        raise AssertionError('invalid option reached callback')
    with pytest.raises(GeometryError, match='Boolean scalar'):
        evaluate(model, result, [0, 0], derivatives=option, cancellation_check=callback)


def test_hostile_derivative_truth_method_is_never_called():
    model, result = planar(fragment=False); source = to_dict(model)
    class Hostile:
        def __bool__(self):
            model.add_point(20, 20, 20)
            raise AssertionError('invalid option was truth-tested')
    def callback(_):
        raise AssertionError('invalid option reached callback')
    with pytest.raises(GeometryError, match='Boolean scalar'):
        evaluate(model, result, [0, 0], derivatives=Hostile(), cancellation_check=callback)
    assert to_dict(model) == source


@pytest.mark.parametrize('option', [np.bool_(False), np.bool_(True)])
def test_numpy_boolean_scalar_selects_expected_return(option):
    model, result = planar(fragment=False)
    output = evaluate(model, result, [.25, .5], derivatives=option)
    if bool(option):
        assert isinstance(output, tuple) and len(output) == 2
        np.testing.assert_array_equal(output[0], [1., 0., 0.])
        np.testing.assert_array_equal(output[1], [0., 1., 0.])
    else:
        np.testing.assert_array_equal(output, [.25, .5, 0.])
