"""Independent expected runtime bytes; actual mutable kernels are NEVER truth."""
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import RLock
from weakref import WeakKeyDictionary, WeakValueDictionary
import numpy as np

from . import backend
from .errors import GeometryError

_LOCK = RLock()
_MODELS = WeakKeyDictionary()
_SHARED = WeakValueDictionary()
_STANDALONE = OrderedDict()
_STANDALONE_LIMIT = 32  # retention only: eviction always recomputes, never refuses
_ROW_LIMIT = 130  # all128 currently owned point rows plus two domain endpoints
_CONTEXT = object()


def kernel_context():
    """Effective kernel bindings/limits, not mutable environment intentions."""
    return backend._canonical_curve_runtime_context()


def _retire_context(context):
    global _CONTEXT
    if _CONTEXT != context:
        _MODELS.clear()
        _SHARED.clear()
        _STANDALONE.clear()
        _CONTEXT = context


def begin_validation():
    context = kernel_context()
    with _LOCK:
        _retire_context(context)
    return context


def declaration(curve):
    """Complete binary declaration, including signed zero and directed domain.

    Noncanonical raw public wrappers keep the original fresh validator route,
    rather than obtaining authority through a normalized memo key.
    """
    if (type(curve.degree) is not int or type(curve.knots) is not tuple
            or type(curve.controls) is not tuple or type(curve.domain) is not tuple
            or len(curve.domain) != 2 or any(type(row) is not tuple for row in curve.controls)
            or curve.weights is not None and type(curve.weights) is not tuple):
        return None
    values = (*curve.knots, *curve.domain, *(x for row in curve.controls for x in row),
              *(() if curve.weights is None else curve.weights))
    if any(type(value) is not float or not np.isfinite(value) for value in values):
        return None
    controls = np.asarray(curve.controls, dtype=np.float64)
    if controls.ndim != 2:
        return None
    return (curve.degree, np.asarray(curve.knots, dtype=np.float64).tobytes(), controls.shape,
            controls.tobytes(), None if curve.weights is None else np.asarray(curve.weights, dtype=np.float64).tobytes(),
            np.asarray(curve.domain, dtype=np.float64).tobytes())


def descriptor(array):
    return array.dtype.str, array.shape, array.tobytes(order='C')


def buffers(prepared):
    degree, arrays = prepared
    return (degree, *(np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape) for dtype, shape, data in arrays))


@dataclass
class _Expected:
    prepared: tuple
    rows: OrderedDict = field(default_factory=OrderedDict)


@dataclass
class _Bank:
    context: tuple
    declarations: dict = field(default_factory=dict)


def model_bank(model, curves):
    context = kernel_context()
    with _LOCK:
        _retire_context(context)
        if context is None:
            _MODELS.pop(model, None)
            return None
        bank = _MODELS.get(model)
        if bank is None or bank.context != context:
            bank = _Bank(context)
            _MODELS[model] = bank
        try:
            active = {key for curve in curves if (key := declaration(curve)) is not None}
        except (ValueError, TypeError, OverflowError) as error:
            raise GeometryError('NURBS evaluated definition is invalid') from error
        bank.declarations = {key: value for key, value in bank.declarations.items() if key in active}
        return bank


def expected_prepared(curve, bank=None, prepared=None):
    context, key = kernel_context(), declaration(curve)
    with _LOCK:
        _retire_context(context)
        if context is None or key is None:
            return None
        if bank is not None and bank.context != context:
            raise GeometryError('NURBS backend changed during runtime validation')
        # Fresh preparation is ALWAYS performed by the validator. Its exact
        # descriptor participates in the row key; no prepared proof shortcut.
        if prepared is None:
            prepared = _compute_prepared(curve)
        prepared_signature = prepared[0], tuple(descriptor(array) for array in prepared[1:])
        shared_key = context, key, prepared_signature
        expected = _SHARED.get(shared_key)
        if expected is None:
            if kernel_context() != context:
                raise GeometryError('NURBS backend changed during canonical preparation')
            expected = _Expected(prepared_signature)
            _SHARED[shared_key] = expected
        if bank is not None:
            bank.declarations[key] = expected
        else:
            _STANDALONE[shared_key] = expected
            _STANDALONE.move_to_end(shared_key)
            while len(_STANDALONE) > _STANDALONE_LIMIT:
                _STANDALONE.popitem(last=False)
        return expected, context


def expected_row(expected, context, station):
    with _LOCK:
        result = expected.rows.get(station)
        if result is None:
            if kernel_context() != context:
                raise GeometryError('NURBS backend changed during runtime validation')
            result = _compute_row(expected.prepared, station)
            if kernel_context() != context:
                raise GeometryError('NURBS backend changed during canonical point evaluation')
            expected.rows[station] = result
            while len(expected.rows) > _ROW_LIMIT:
                expected.rows.popitem(last=False)
        else:
            expected.rows.move_to_end(station)
        return result


def _compute_prepared(curve):
    return backend._prepare_bspline_curve(curve.degree, curve.knots, curve.controls, curve.weights)


def _compute_row(prepared, station):
    row = backend._evaluate_prepared_bspline_curve(buffers(prepared), np.frombuffer(station, dtype=np.float64), 0)
    return descriptor(row)
