"""Expected-byte reuse never treats mutable actual runtime caches as truth."""
import copy
from dataclasses import replace
import gc
import weakref

import numpy as np
import pytest

from anygeometry import GeometryError, GeometryModel
from anygeometry import backend
from anygeometry.nurbs import NurbsCurve
from anygeometry.nurbs_runtime_binding import validate_curve_runtime, validate_model_curve_runtimes
from anygeometry import _canonical_curve_runtime as memo


@pytest.fixture(autouse=True)
def clean_expected_memos():
    with memo._LOCK:
        memo._MODELS.clear(); memo._SHARED.clear(); memo._STANDALONE.clear()
    yield
    with memo._LOCK:
        memo._MODELS.clear(); memo._SHARED.clear(); memo._STANDALONE.clear()


def curve(offset=0., domain=(0., 1.)):
    return NurbsCurve(2, (0., 0., 0., 1., 1., 1.),
        ((offset, 0., 0.), (offset+1., 2., 0.), (offset+2., 0., 0.)), (1., 2., 1.), domain)


def counts(monkeypatch):
    actual = {'prepared': 0, 'row': 0}
    for name, key in (('_compute_prepared', 'prepared'), ('_compute_row', 'row')):
        original = getattr(memo, name)
        def measured(*args, _original=original, _key=key):
            actual[_key] += 1
            return _original(*args)  # real independent owner backend; no mocked proof
        monkeypatch.setattr(memo, name, measured)
    return actual


def test_all128_actual_rows_checked_with_zero_warm_expected_evaluations(monkeypatch):
    assert memo.kernel_context() is not None
    c = curve()
    c.evaluate(0.); c.evaluate(1.)
    for i in range(128):
        c.evaluate((i+1)/129)
    assert len(c._point_values) == 128
    calls = counts(monkeypatch)
    validate_curve_runtime(c)
    assert calls == {'prepared': 1, 'row': 130}
    validate_curve_runtime(c)
    assert calls == {'prepared': 2, 'row': 130}
    c.evaluate(.123456789)
    validate_curve_runtime(c)
    assert calls == {'prepared': 3, 'row': 131}
    validate_curve_runtime(c)
    assert calls == {'prepared': 4, 'row': 131}
    expected, _context = memo.expected_prepared(c)
    assert len(expected.rows) <= memo._ROW_LIMIT
    for array in memo.buffers(expected.prepared)[1:]:
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_equivalent_curve_copies_reuse_canonical_truth_and_keep_actual_copy_semantics(monkeypatch):
    c = curve()
    c.evaluate(.25)
    calls = counts(monkeypatch)
    validate_curve_runtime(c)
    first = dict(calls)
    other = copy.copy(c)
    other.evaluate(.25)
    validate_curve_runtime(other)
    assert calls == {'prepared': first['prepared']+1, 'row': first['row']}
    # Existing validator allowed canonical-valued writable/noncontiguous
    # arrays: memoization adds no readonly/layout refusal for actual data.
    object.__setattr__(other, '_prepared', (other._prepared[0], *(a.copy() for a in other._prepared[1:])))
    other._prepared[1][0] = -0.  # fresh original prepared check keeps np.array_equal semantics
    key, value = next(iter(other._point_values.items()))
    noncontiguous = np.repeat(value, 2, axis=-1)[..., ::2]
    assert not noncontiguous.flags.c_contiguous
    object.__setattr__(other, '_point_values', {key: noncontiguous})
    validate_curve_runtime(other)
    assert calls == {'prepared': first['prepared']+2, 'row': first['row']}


@pytest.mark.parametrize('attack', ('prepared', 'endpoint', 'point', 'shape', 'dtype', 'nan',
                                   'key', 'nonfinite_station', 'outside_domain', 'size'))
def test_warm_canonical_truth_refuses_each_actual_cache_corruption(attack):
    c = curve(); c.evaluate(0.); c.evaluate(.25)
    validate_curve_runtime(c)
    key, row = next(iter(c._point_values.items()))
    if attack == 'prepared':
        arrays = [a.copy() for a in c._prepared[1:]]
        arrays[1][0, 0] += 1
        object.__setattr__(c, '_prepared', (c._prepared[0], *arrays))
    elif attack == 'endpoint':
        object.__setattr__(c, '_endpoint_values', (c._endpoint_values[0]+1, None))
    else:
        points = dict(c._point_values)
        if attack == 'point': points[key] = row+1
        elif attack == 'shape': points[key] = row.reshape(1, 3)
        elif attack == 'dtype': points[key] = row.astype(np.float32)
        elif attack == 'nan': points[key] = row*np.nan
        elif attack == 'key': points = {b'bad': row}
        elif attack == 'nonfinite_station': points = {np.float64(np.nan).tobytes(): row}
        elif attack == 'outside_domain': points = {np.float64(2.).tobytes(): row}
        else: points = {np.float64((i+1)/130).tobytes(): row for i in range(129)}
        object.__setattr__(c, '_point_values', points)
    with pytest.raises(GeometryError):
        validate_curve_runtime(c)


@pytest.mark.parametrize('domain', ((0., 1.), (1., 0.), (.125, .875)))
def test_declaration_bits_domains_and_signed_zero_are_independently_qualified(domain):
    plus = curve(domain=domain)
    minus = replace(plus, knots=(-0., -0., -0., 1., 1., 1.))
    assert memo.declaration(plus) != memo.declaration(minus)
    for c in (plus, minus):
        c.evaluate(0.); c.evaluate(.25); c.evaluate(1.)
        validate_curve_runtime(c)
    object.__setattr__(plus, 'domain', (.2, .8))
    with pytest.raises(GeometryError):
        validate_curve_runtime(plus)


def test_known_numeric_context_change_retires_expected_state_and_requalifies(monkeypatch):
    c = curve(); c.evaluate(.25)
    calls = counts(monkeypatch)
    validate_curve_runtime(c)
    old = dict(calls)
    monkeypatch.setattr(backend, 'MAX_WEIGHT_RATIO', 1.)
    with pytest.raises(GeometryError):
        validate_curve_runtime(c)
    assert calls['prepared'] == old['prepared']+1
    monkeypatch.setattr(backend, 'MAX_WEIGHT_RATIO', 1e6)
    validate_curve_runtime(c)
    assert calls['prepared'] == old['prepared']+2 and calls['row'] == old['row']+1


def test_stable_mutable_backend_wrapper_never_acquires_expected_memo(monkeypatch):
    c = curve(); c.evaluate(.25)
    original = backend._evaluate_prepared_bspline_curve
    state = {'offset': 0., 'calls': 0}
    def mutable_wrapper(*args, **kwargs):
        state['calls'] += 1
        return original(*args, **kwargs)+state['offset']
    monkeypatch.setattr(backend, '_evaluate_prepared_bspline_curve', mutable_wrapper)
    assert memo.kernel_context() is None
    calls = counts(monkeypatch)
    validate_curve_runtime(c)
    validate_curve_runtime(c)
    assert state['calls'] == 2 and calls == {'prepared': 2, 'row': 0}
    state['offset'] = 1.
    with pytest.raises(GeometryError):
        validate_curve_runtime(c)
    assert state['calls'] == 3


def test_numpy_module_alias_is_unknown_even_with_the_same_function_attributes(monkeypatch):
    from types import SimpleNamespace
    replacement = SimpleNamespace(**{name: getattr(np, name) for name in backend._RUNTIME_NUMPY_NAMES})
    monkeypatch.setattr(backend, 'np', replacement)
    assert memo.kernel_context() is None


def test_numpy_error_context_and_handlers_take_the_required_fresh_routes(monkeypatch):
    c = curve(); c.evaluate(.25)
    calls = counts(monkeypatch)
    validate_curve_runtime(c)
    first = dict(calls)
    errors, handler = np.geterr(), np.geterrcall()
    try:
        np.seterr(under='raise' if errors['under'] != 'raise' else 'ignore')
        validate_curve_runtime(c)
        assert calls == {'prepared': first['prepared']+1, 'row': first['row']+1}
        np.seterrcall(lambda *args: None)
        assert memo.kernel_context() is None
        validate_curve_runtime(c)
    finally:
        np.seterr(**errors); np.seterrcall(handler)


def test_weak_model_bank_retains_all_live_declarations_without_fixed_size_thrashing(monkeypatch):
    model = GeometryModel()
    for i in range(40):
        c = curve(float(i)*3); c.evaluate(.25)
        a, b = model.add_points((c.evaluate(0.), c.evaluate(1.)))
        model.add_curve(a, b, c)
    calls = counts(monkeypatch)
    validate_model_curve_runtimes(model)
    first = dict(calls)
    assert first['prepared'] == 40
    assert len(memo._MODELS[model].declarations) == 40
    validate_model_curve_runtimes(model)
    assert calls == {'prepared': first['prepared']*2, 'row': first['row']}
    reference = weakref.ref(model)
    del model
    gc.collect()
    assert reference() is None


def test_standalone_and_row_retention_eviction_only_recomputes(monkeypatch):
    monkeypatch.setattr(memo, '_STANDALONE_LIMIT', 2)
    curves = [curve(float(i)*3) for i in range(3)]
    for c in curves: c.evaluate(.25)
    calls = counts(monkeypatch)
    for c in curves: validate_curve_runtime(c)
    assert len(memo._STANDALONE) == 2 and calls == {'prepared': 3, 'row': 3}
    validate_curve_runtime(curves[0])
    assert calls == {'prepared': 4, 'row': 4}
    monkeypatch.setattr(memo, '_ROW_LIMIT', 2)
    for t in (.5, .75): curves[0].evaluate(t)
    validate_curve_runtime(curves[0])
    expected, _ = memo.expected_prepared(curves[0])
    assert len(expected.rows) <= 2


def test_actual_public_callback_poison_is_rejected_after_expected_memo_warms():
    from test_prepared_material_curvilinear import s_bend_boundary
    from anygeometry import validate_material_surface_regions_binding
    model, regions, edge = s_bend_boundary()
    pcurve = model.edges[edge.id].curve.pcurve
    pcurve.evaluate(.314159)
    validate_material_surface_regions_binding(model, regions)
    original = pcurve._point_values
    key, row = next(iter(original.items()))
    def poison(phase):
        object.__setattr__(pcurve, '_point_values', {**original, key: row+1})
    try:
        with pytest.raises(GeometryError, match='point cache'):
            validate_material_surface_regions_binding(model, regions, cancellation_check=poison)
    finally:
        object.__setattr__(pcurve, '_point_values', original)
    validate_material_surface_regions_binding(model, regions)


def _finite_offset_einsum(*operands, out=None, optimize=False, **kwargs):
    # This code is installed into NumPy's OWN implementation globals, where
    # c_einsum is the genuine native alias. No external closure is involved.
    if out is not None:
        kwargs['out'] = out
    return c_einsum(*operands, **kwargs)+1.


@pytest.mark.parametrize('attack', ('implementation_code', 'native_alias'))
def test_same_public_einsum_identity_cannot_hide_changed_effective_kernel(monkeypatch, attack):
    c = curve(); c.evaluate(.25)
    validate_curve_runtime(c)
    dispatcher = np.einsum
    implementation = getattr(dispatcher, '_implementation', getattr(dispatcher, '__wrapped__', dispatcher))
    assert 'c_einsum' in implementation.__globals__
    if attack == 'implementation_code':
        monkeypatch.setattr(implementation, '__code__', _finite_offset_einsum.__code__)
    else:
        original = implementation.__globals__['c_einsum']
        def offset(*args, **kwargs):
            return original(*args, **kwargs)+1.
        monkeypatch.setitem(implementation.__globals__, 'c_einsum', offset)
    assert np.einsum is dispatcher
    assert memo.kernel_context() is None
    if backend._CORE is None:
        with pytest.raises(GeometryError, match='point cache'):
            validate_curve_runtime(c)
    else:
        # Compiled point kernels do not consume Python einsum. The context
        # still conservatively takes the fresh route; numerical truth agrees.
        validate_curve_runtime(c)


def test_same_public_searchsorted_identity_cannot_hide_mutable_wrapfunc_alias(monkeypatch):
    c = NurbsCurve(2, (0., 0., 0., .5, 1., 1., 1.),
        ((0., 0., 0.), (1., 2., 0.), (2., -1., 0.), (3., 0., 0.)), (1., 2., 1., 1.))
    c.evaluate(.25)
    validate_curve_runtime(c)
    dispatcher = np.searchsorted
    implementation = getattr(dispatcher, '_implementation', getattr(dispatcher, '__wrapped__', dispatcher))
    original, code = implementation.__globals__['_wrapfunc'], implementation.__code__
    state = {'shift': 0}
    def mutable_wrapfunc(obj, method, *args, **kwargs):
        result = original(obj, method, *args, **kwargs)
        return result+state['shift'] if method == 'searchsorted' else result
    monkeypatch.setitem(implementation.__globals__, '_wrapfunc', mutable_wrapfunc)
    calls = counts(monkeypatch)
    assert np.searchsorted is dispatcher and implementation.__code__ is code
    assert memo.kernel_context() is None
    validate_curve_runtime(c)
    validate_curve_runtime(c)
    assert calls == {'prepared': 2, 'row': 0}  # actual fresh path on EVERY call
    state['shift'] = 1
    assert implementation.__globals__['_wrapfunc'] is mutable_wrapfunc
    if backend._CORE is None:
        with pytest.raises(GeometryError, match='point cache'):
            validate_curve_runtime(c)
    else:
        validate_curve_runtime(c)  # native point kernel does not consume Python searchsorted
