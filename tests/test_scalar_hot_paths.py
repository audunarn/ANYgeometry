"""Scalar hot-path parity tests: LinePath.evaluate and the private _cross3.

The scalar fast paths must be bit-identical to the vectorized originals for
every input they accept, preserve shapes, dtypes and exceptional behavior, and
fall back to the original route wherever identity is not guaranteed.
"""
import numpy as np
import pytest

from anygeometry import GeometryModel
from anygeometry._cross3 import _cross3
from anygeometry.arrangement_geometry import LinePath, freeze_edge


def vectorized_evaluate(path, parameters):
    t = np.asarray(parameters, dtype=float)
    return np.asarray(path.start)+t[..., None]*(np.asarray(path.end)-path.start)


SCALARS = [0., 1., .5, 1e-8, -3.75, 1e300, -1e-300, 5e-324, float('inf'),
           float('-inf'), float('nan')]
LINES = [((0., 0., 0.), (4., 2., 1.)),
         ((1e300, -1e-300, 5.), (-1e300, 1e300, -5.)),
         ((0., 0., 0.), (0., 0., 0.)),
         ((-0., -0., -0.), (0., 0., 0.)),
         ((np.float64(1.5), np.float64(-2.5), np.float64(0.)),
          (np.float64(3.25), np.float64(0.75), np.float64(-1.)))]


@pytest.mark.parametrize('start,end', LINES)
def test_scalar_evaluate_is_bit_identical_to_the_vectorized_oracle(start, end):
    path = LinePath(start, end)
    for t in SCALARS + [np.float64(value) for value in SCALARS]:
        result = path.evaluate(t)
        reference = vectorized_evaluate(path, t)
        assert result.shape == (3,)
        assert result.dtype == np.float64
        assert np.array_equal(result, reference, equal_nan=True)


def test_non_scalar_parameters_fall_back_to_the_vectorized_route():
    path = LinePath((0., 0., 0.), (4., 2., 1.))
    for parameters in [(0., 1.), np.linspace(0., 1., 7), np.asarray(0.5),
                       np.float32(0.5), 1, True, [0.25]]:
        result = path.evaluate(parameters)
        reference = vectorized_evaluate(path, parameters)
        assert np.array_equal(result, reference, equal_nan=True)
        assert result.dtype == reference.dtype
        assert result.shape == reference.shape


def test_non_float_coordinates_fall_back_to_the_vectorized_route():
    for start, end in [((0, 0, 0), (4, 2, 1)),
                       ((0., 0, 0.), (4., 2, 1)),
                       ((0., 0., 0.), (4., 2, 1))]:
        path = LinePath(start, end)
        for t in SCALARS[:6]:
            assert np.array_equal(path.evaluate(t), vectorized_evaluate(path, t),
                                  equal_nan=True)


def test_other_lengths_keep_their_original_shapes():
    path = LinePath((0., 0.), (4., 2.))
    result = path.evaluate(0.5)
    assert result.shape == (2,)
    assert np.array_equal(result, vectorized_evaluate(path, 0.5))


def test_subcurve_and_bounds_stay_consistent_with_evaluate():
    path = LinePath((0., 1., -2.), (4., 1., 2.))
    piece = path.subcurve(.25, .75)
    assert np.array_equal(piece.evaluate(0.), path.evaluate(.25))
    assert np.array_equal(piece.evaluate(1.), path.evaluate(.75))
    lower, upper = path.bounds(.25, .75)
    assert np.all(lower <= path.evaluate(np.linspace(.25, .75, 11)) + 0.)
    assert np.all(upper >= path.evaluate(np.linspace(.25, .75, 11)) - 0.)


def _adversarial_vectors():
    rng = np.random.default_rng(20261007)
    values = [rng.normal(size=3)*10.0**rng.integers(-300, 300) for _ in range(500)]
    values += [np.array((0., 0., 0.)), np.array((5e-324, -5e-324, 1e-308)),
               np.array((1.7976931348623157e308, -1.7976931348623157e308, 1.)),
               np.array((float('inf'), 0., -float('inf'))),
               np.array((float('nan'), 1., -2.))]
    return values


def test_cross3_is_bit_identical_to_numpy_cross_for_float64_vectors():
    vectors = _adversarial_vectors()
    for a in vectors:
        for b in vectors[::101]:
            result = _cross3(a, b)
            reference = np.cross(a, b)
            assert result.dtype == np.float64
            assert np.array_equal(result, reference, equal_nan=True)


def test_cross3_falls_back_to_numpy_cross_for_other_inputs():
    fallbacks = [(np.array((1, 0, 0)), np.array((0, 1, 0))),
                 (np.array((1., 0., 0.), dtype=np.float32),
                  np.array((0., 1., 0.), dtype=np.float32)),
                 ([1., 0., 0.], [0., 1., 0.]),
                 (np.array([[1., 0., 0.]]), np.array([[0., 1., 0.]]))]
    for a, b in fallbacks:
        assert np.array_equal(_cross3(a, b), np.cross(a, b), equal_nan=True)
        assert _cross3(a, b).dtype == np.cross(a, b).dtype
    # Exceptional behavior is preserved: an unsupported 4-vector raises
    # exactly like numpy.cross.
    with pytest.raises(ValueError) as reference:
        np.cross(np.array((1., 0., 0., 0.)), np.array((0., 1., 0., 0.)))
    with pytest.raises(ValueError) as actual:
        _cross3(np.array((1., 0., 0., 0.)), np.array((0., 1., 0., 0.)))
    assert str(actual.value) == str(reference.value)


class _PlainFloatSubclass(float):
    """An honest float subclass; still not fast-path eligible."""


class _LyingFloat(float):
    """A float subclass whose arithmetic lies; only the fallback is safe."""

    def __add__(self, other):
        return 42.0

    __radd__ = __add__

    def __mul__(self, other):
        return 42.0

    __rmul__ = __mul__


def test_overflowing_coordinates_preserve_numpy_exception_behavior():
    # Review counterexample: under np.errstate(over='raise') the vectorized
    # route raises on the subtraction; Python float arithmetic would
    # silently produce -inf instead.
    path = LinePath((1e308, 0., 0.), (-1e308, 1., 0.))
    with np.errstate(over='raise', invalid='raise'):
        with pytest.raises(FloatingPointError):
            vectorized_evaluate(path, 0.5)
        with pytest.raises(FloatingPointError):
            path.evaluate(0.5)
    assert np.array_equal(path.evaluate(0.5), vectorized_evaluate(path, 0.5),
                          equal_nan=True)


def test_invalidating_coordinates_preserve_numpy_exception_behavior():
    path = LinePath((float('inf'), 0., 0.), (float('inf'), 1., 0.))
    with np.errstate(invalid='raise'):
        with pytest.raises(FloatingPointError):
            vectorized_evaluate(path, 0.5)
        with pytest.raises(FloatingPointError):
            path.evaluate(0.5)
    assert np.array_equal(path.evaluate(0.5), vectorized_evaluate(path, 0.5),
                          equal_nan=True)


def test_nonfinite_scalars_fall_back_to_the_vectorized_route():
    path = LinePath((0., 0., 0.), (4., 2., 1.))
    for t in (float('inf'), float('-inf'), float('nan')):
        with np.errstate(over='raise', invalid='raise'):
            result = path.evaluate(t)
        assert np.array_equal(result, vectorized_evaluate(path, t), equal_nan=True)


def test_float_subclass_scalars_fall_back_to_the_vectorized_route():
    path = LinePath((0., 0., 0.), (4., 2., 1.))
    # Exact np.float64 scalars are fast-path eligible (covered below); only
    # genuine float subclasses fall back.
    for t in (_PlainFloatSubclass(0.5), _LyingFloat(0.5)):
        result = path.evaluate(t)
        reference = vectorized_evaluate(path, t)
        assert result.dtype == reference.dtype
        assert result.shape == reference.shape
        assert np.array_equal(result, reference, equal_nan=True)


def test_float_subclass_coordinates_fall_back_to_the_vectorized_route():
    for start, end in [((_PlainFloatSubclass(0.), 0., 0.), (4., 2., 1.)),
                       ((_LyingFloat(0.), 0., 0.), (4., 2., 1.))]:
        path = LinePath(start, end)
        for t in (0.5, 0., 1.):
            assert np.array_equal(path.evaluate(t), vectorized_evaluate(path, t),
                                  equal_nan=True)

def test_exact_np_float64_inputs_use_the_fast_path(monkeypatch):
    # Production freeze_edge stores exact np.float64 coordinates, so the
    # fast path must accept them and scalar np.float64 parameters, not only
    # exact Python floats. Blocking the vectorized route proves the fast
    # path handled the call.
    cases = [((0., 0., 0.), (4., 2., 1.)),
             ((np.float64(1.5), np.float64(-2.5), np.float64(0.)),
              (np.float64(3.25), np.float64(0.75), np.float64(-1.)))]
    references = {}
    for start, end in cases:
        path = LinePath(start, end)
        for t in (0.5, 1e-8, -3.75):
            references[(start, end, t)] = vectorized_evaluate(path, t)

    def blocked(*args, **kwargs):
        raise AssertionError('vectorized route used')

    monkeypatch.setattr(np, 'asarray', blocked)
    results = []
    for start, end in cases:
        path = LinePath(start, end)
        for t in (0.5, 1e-8, -3.75):
            for scalar in (t, np.float64(t)):
                results.append((start, end, t, path.evaluate(scalar)))
    monkeypatch.undo()
    for start, end, t, result in results:
        reference = references[(start, end, t)]
        assert result.shape == (3,)
        assert result.dtype == np.float64
        assert (result == reference).all()


def test_production_freeze_edge_route_uses_the_fast_path(monkeypatch):
    # freeze_edge builds LinePath coordinates as tuple(vertex_position(...)),
    # i.e. exact np.float64 values; the accelerated route must be the one
    # production traffic takes, not only synthetic Python-float tuples.
    model = GeometryModel()
    v0, v1 = model.add_points(((0., 0., 0.), (4., 2., 1.)))
    edge_id = model.add_line(v0, v1)
    path = freeze_edge(model, edge_id)
    assert isinstance(path, LinePath)
    assert all(type(value) is np.float64 for value in path.start+path.end)
    reference = vectorized_evaluate(path, 0.25)

    def blocked(*args, **kwargs):
        raise AssertionError('vectorized route used')

    monkeypatch.setattr(np, 'asarray', blocked)
    evaluated = [path.evaluate(0.25), path.evaluate(np.float64(0.25))]
    monkeypatch.undo()
    for result in evaluated:
        assert result.shape == (3,)
        assert result.dtype == np.float64
        assert (result == reference).all()


def test_underflowing_coordinates_preserve_numpy_exception_behavior():
    # Review counterexample: a finite result does not certify absence of
    # underflow. Under np.errstate(under='raise') the vectorized route
    # raises while Python-float arithmetic silently returns zero, so the
    # fast path must not run while NumPy underflow handling is enabled.
    path = LinePath((0., 0., 0.), (1e-200, 0., 0.))
    with np.errstate(under='raise'):
        with pytest.raises(FloatingPointError):
            vectorized_evaluate(path, 1e-200)
        with pytest.raises(FloatingPointError):
            path.evaluate(1e-200)
        # Without an actual underflow the evaluation still succeeds.
        assert np.array_equal(path.evaluate(0.5), vectorized_evaluate(path, 0.5))
    with np.errstate(under='warn'):
        with pytest.warns(RuntimeWarning):
            vectorized_evaluate(path, 1e-200)
        with pytest.warns(RuntimeWarning):
            path.evaluate(1e-200)
    assert np.array_equal(path.evaluate(1e-200), vectorized_evaluate(path, 1e-200))
