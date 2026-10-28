"""Batched NURBS evaluation in ``evaluate_material_surface_region`` (UNRUN definitions).

Real owner admitted NURBS region from the ``square_cells`` fixture; counters delegate to the actual
NurbsSurface kernels. The scalar ``support.evaluate`` is only the reference oracle for the batch result.
"""
import numpy as np
import pytest

from anygeometry import GeometryError, evaluate_material_surface_region, query_material_surface_regions
from anygeometry.nurbs import NurbsSurface
from test_prepared_material_curvilinear import square_cells


def region_fixture():
    model, regions, chords, cells = square_cells()
    face = next(iter(model.faces))
    support = regions.regions[0].support
    assert type(support) is NurbsSurface
    return model, regions, face, support


def reference(support, parameters):
    values = np.asarray(parameters, dtype=float)
    rows = values.reshape(-1, 2)
    made = np.asarray([support.evaluate(*uv) for uv in rows], dtype=float).reshape(-1, 3)
    return made.reshape((*values.shape[:-1], 3))


@pytest.mark.parametrize('shape', ((2,), (1, 2), (0, 2), (5, 2), (2, 3, 2)))
def test_batched_values_match_the_scalar_support_for_every_row_shape(shape):
    model, regions, face, support = region_fixture()
    rng = np.random.default_rng(7)
    parameters = rng.uniform(.05, .95, size=shape)
    result = evaluate_material_surface_region(model, regions, face, parameters)
    assert result.shape == (*shape[:-1], 3) and result.dtype == float
    assert np.allclose(result, reference(support, parameters), rtol=0., atol=1e-12)


def test_one_batched_kernel_call_replaces_per_point_scalar_evaluation(monkeypatch):
    model, regions, face, support = region_fixture()
    batch_calls, scalar_calls = [], []
    real_many, real_one = NurbsSurface.evaluate_many, NurbsSurface.evaluate

    def many(self, *args, **kwargs):
        batch_calls.append(len(args[0]))
        return real_many(self, *args, **kwargs)

    def one(self, *args, **kwargs):
        scalar_calls.append(1)
        return real_one(self, *args, **kwargs)

    parameters = np.random.default_rng(3).uniform(.05, .95, size=(40, 2))
    monkeypatch.setattr(NurbsSurface, 'evaluate_many', many)
    monkeypatch.setattr(NurbsSurface, 'evaluate', one)
    evaluate_material_surface_region(model, regions, face, parameters)
    assert batch_calls == [40] and scalar_calls == []


def test_result_is_a_copy_and_nonfinite_or_malformed_parameters_refuse():
    model, regions, face, support = region_fixture()
    first = evaluate_material_surface_region(model, regions, face, [[.25, .25]])
    first[:] = 99.
    second = evaluate_material_surface_region(model, regions, face, [[.25, .25]])
    assert np.all(second < 98.)
    for bad in ([[np.nan, .5]], [[.5, np.inf]], [.5, .5, .5], 'abc'):
        with pytest.raises(GeometryError):
            evaluate_material_surface_region(model, regions, face, bad)


def _outcome(call):
    try:
        return ('ok', np.asarray(call()))
    except GeometryError as error:
        return ('error', type(error), str(error))


@pytest.mark.parametrize('point', ([.5, .5], [2., 2.]))
def test_require_material_outcome_is_unchanged_by_batching(monkeypatch, point):
    """NURBS ``require_material`` membership has a PRE-EXISTING public limitation (the domain predicate
    needs a CurveOnSurface path; a LinePath point lookup is refused). Batching must not change the outcome:
    compare against the original scalar position path, without asserting success or a specific refusal."""
    import anygeometry.material_regions as module
    model, regions, face, support = region_fixture()

    def call():
        return evaluate_material_surface_region(model, regions, face, [point], require_material=True)

    batched = _outcome(call)

    def scalar(support, rows):
        return np.asarray([support.evaluate(*uv) for uv in rows], dtype=float).reshape(-1, 3)

    monkeypatch.setattr(module, '_support_positions', scalar)
    original = _outcome(call)
    assert batched[0] == original[0]
    if batched[0] == 'ok':
        assert np.allclose(batched[1], original[1], rtol=0., atol=1e-12)
    elif point == [2., 2.]:
        # outside the normalized chart: the exact exception class and bounds refusal are the contract; the
        # human label differs (batched native kernel vs scalar curve-parameter check)
        assert batched[1] is original[1]
        for message in (batched[2], original[2]):
            assert 'finite and in [0, 1]' in message
    else:
        assert batched[1:] == original[1:]                          # admitted UV: same type and message


def test_derivative_return_shape_is_preserved():
    model, regions, face, support = region_fixture()
    arrays = evaluate_material_surface_region(model, regions, face, [[.3, .4], [.6, .7]], derivatives=True)
    assert len(arrays) == 2 and all(array.shape == (2, 3) for array in arrays)
    single = evaluate_material_surface_region(model, regions, face, [.3, .4], derivatives=True)
    assert all(array.shape == (3,) for array in single)


def test_binding_and_cancellation_frontiers_are_preserved():
    model, regions, face, support = region_fixture()
    with pytest.raises(GeometryError, match='cancelled'):
        evaluate_material_surface_region(model, regions, face, [[.5, .5]], cancellation_check=lambda phase: True)
    edits = []

    def mutate(phase):
        if not edits:
            edits.append(model.add_point(9., 9., 9.))
        return False

    with pytest.raises(GeometryError):
        evaluate_material_surface_region(model, regions, face, [[.5, .5]], cancellation_check=mutate)
    assert edits

# Not covered (cheap fixture unavailable): weighted, transformed and reversed NURBS supports, which DO use
# the batched path and are untested here. Non-NURBS supports retain the unchanged scalar path.
