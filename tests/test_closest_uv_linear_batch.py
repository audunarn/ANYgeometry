"""The optional NumPy kernel keeps the public least-squares oracle."""
import numpy as np
import pytest
from anygeometry import surfaces


def systems():
    rng = np.random.default_rng(884)
    a = rng.normal(size=(256, 3, 2))
    a[::13, :, 1] = a[::13, :, 0]
    a[::37] = 0.
    a[1] = [[1., 1.], [0., 1.e-16], [0., 0.]]
    b = rng.normal(size=(len(a), 3))
    return a, b


def test_delta_batch_matches_public_zero_rank_and_near_singular():
    a, b = systems()
    got = surfaces._closest_uv_deltas(a[:, :, 0], a[:, :, 1], b)
    if np.__version__ != '2.4.6':
        assert got is None
        return
    expected = np.asarray([np.linalg.lstsq(x, y, rcond=None)[0] for x, y in zip(a, b)])
    np.testing.assert_array_equal(got, expected)


def test_version_public_protocol_and_private_capability_fallback(monkeypatch):
    a, b = systems()
    original = np.linalg.lstsq
    monkeypatch.setattr(np.linalg, 'lstsq', lambda *args, **kwargs: original(*args, **kwargs))
    assert surfaces._closest_uv_deltas(a[:, :, 0], a[:, :, 1], b) is None
    monkeypatch.setattr(np.linalg, 'lstsq', original)
    monkeypatch.setattr(np, '__version__', 'other')
    assert surfaces._closest_uv_deltas(a[:, :, 0], a[:, :, 1], b) is None


def test_nonfinite_custom_arrays_and_error_mode_fallback():
    a, b = systems()
    with np.errstate(invalid='raise'):
        assert surfaces._closest_uv_deltas(a[:, :, 0], a[:, :, 1], b) is None
    b[0, 0] = np.nan
    assert surfaces._closest_uv_deltas(a[:, :, 0], a[:, :, 1], b) is None
    class Custom(np.ndarray):
        pass
    assert surfaces._closest_uv_deltas(a[:, :, 0].view(Custom), a[:, :, 1], b) is None


@pytest.mark.skipif(surfaces._numpy_linalg is None or np.__version__ != '2.4.6', reason='optional inspected NumPy kernel')
def test_private_kernel_error_falls_back(monkeypatch):
    a, b = systems()
    class Failing:
        signature = '(m,n),(m,nrhs),()->(n,nrhs),(nrhs),(),(p)'
        def __call__(self, *args, **kwargs):
            raise np.linalg.LinAlgError('failed')
    failing = Failing()
    monkeypatch.setattr(surfaces, '_LSTSQ_GUFUNC', failing)
    monkeypatch.setattr(surfaces._numpy_linalg._umath_linalg, 'lstsq', failing)
    assert surfaces._closest_uv_deltas(a[:, :, 0], a[:, :, 1], b) is None


def test_custom_public_lstsq_is_called_for_each_row(monkeypatch):
    calls = []
    public = np.linalg.lstsq
    def custom(*args, **kwargs):
        calls.append(1)
        return public(*args, **kwargs)
    def evaluate(uv):
        return np.column_stack((uv, np.zeros(len(uv))))
    points = np.array([[.2,.3,0.],[-1.,2.,0.],[0.,0.,0.],[1.,1.,0.]])
    expected = surfaces._closest_uv_many(None, points, _evaluate=evaluate)
    monkeypatch.setattr(np.linalg, 'lstsq', custom)
    actual = surfaces._closest_uv_many(None, points, _evaluate=evaluate, _batch_samples=True)
    np.testing.assert_array_equal(actual, expected)
    assert calls


@pytest.mark.skipif(surfaces._numpy_linalg is None, reason='optional NumPy capability')
def test_removed_private_capability_falls_back(monkeypatch):
    a, b = systems()
    monkeypatch.setattr(surfaces._numpy_linalg, '_umath_linalg', None)
    assert surfaces._closest_uv_deltas(a[:, :, 0], a[:, :, 1], b) is None
