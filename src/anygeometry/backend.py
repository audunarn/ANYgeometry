"""Selection of the compiled or pure-Python kernel backend.

ANYgeometry may carry an optional compiled module, ``anygeometry._core``
(C++17, CPython stable ABI). Every compiled kernel has a pure-Python reference
with the same signature in :mod:`anygeometry.reference`. The backend is chosen
once, when this module is imported, and never changes within a process:

* ``ANYGEOMETRY_DISABLE_NATIVE=1`` selects the Python references;
* otherwise the compiled module is used when it is installed. If it is
  installed but cannot be imported, the import of ANYgeometry fails: a broken
  build is an error, not a reason to fall back silently;
* ``ANYGEOMETRY_REQUIRE_NATIVE=1`` makes the import of ANYgeometry fail when the
  compiled module is not in use.

:func:`backend_info` reports the choice and its reason. Documents and results
never record the backend.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
from types import MappingProxyType

import numpy as np

from . import reference

__all__ = ["backend_info", "bspline_curve", "plane_evaluate"]

_TRUE = {"1", "true", "TRUE", "yes", "YES"}


def _flag(name: str) -> bool:
    return os.environ.get(name, "0").strip() in _TRUE


def _load():
    if _flag("ANYGEOMETRY_DISABLE_NATIVE"):
        return None, "disabled by ANYGEOMETRY_DISABLE_NATIVE"
    spec = importlib.util.find_spec(f"{__package__}._core")
    if spec is None or spec.origin is None:
        return None, "compiled module not installed"
    # Present but unloadable is an error, never a quiet fallback.
    return importlib.import_module(f"{__package__}._core"), "compiled module loaded"


_CORE, _REASON = _load()
if _CORE is None and _flag("ANYGEOMETRY_REQUIRE_NATIVE"):
    raise ImportError(f"ANYGEOMETRY_REQUIRE_NATIVE is set but the compiled core is not in use: {_REASON}")

_BUILD = MappingProxyType(dict(_CORE.build_info())) if _CORE is not None else None
_KERNELS = _CORE if _CORE is not None else reference

plane_evaluate = _KERNELS.plane_evaluate
bspline_evaluate = _KERNELS.bspline_evaluate

MAX_BSPLINE_DEGREE = 15
_MIN_SPAN = 2.0 ** -968
# Owner decision 2026-10-13: weight ratio max/min <= 1e6.
MAX_WEIGHT_RATIO = 1e6
# Numerical range of the homogeneous arithmetic: with weights scaled into
# (5e-7, 1) and nonzero coordinates within [2**-500, 2**500], every weighted
# control is a normal float and the denominator is at least 5e-7 / (degree + 1).
_MIN_MAGNITUDE = 2.0 ** -500
_MAX_MAGNITUDE = 2.0 ** 500


def _aligned(values) -> np.ndarray:
    return np.require(values, dtype=np.float64, requirements=["C", "A"])


def check_clamped_knots(degree: int, knots: np.ndarray, count: int) -> None:
    """Raise ``ValueError`` unless ``knots`` is a clamped knot vector the kernels support.

    End values repeat exactly ``degree + 1`` times, interior values at most
    ``degree`` times (so every joint keeps at least C0 continuity), knots are
    finite and non-decreasing, and every non-empty span is longer than
    ``64 * eps`` times the knot scale and than ``2**-968`` (so basis ratios
    never involve subnormal differences).
    """
    p = degree
    if knots.shape != (count + p + 1,) or not np.all(np.isfinite(knots)) or np.any(np.diff(knots) < 0):
        raise ValueError("knots must be finite and non-decreasing with n + degree + 1 values")
    lower, upper = knots[p], knots[count]
    if not (np.all(knots[: p + 1] == lower) and np.all(knots[count:] == upper)):
        raise ValueError("knots must be clamped: each end value repeated degree + 1 times")
    if not lower < upper or knots[p + 1] == lower or knots[count - 1] == upper:
        raise ValueError("knots must be clamped with a non-empty domain and end multiplicity degree + 1")
    interior = knots[p + 1: count]
    values, multiplicity = np.unique(interior, return_counts=True)
    if values.size and int(multiplicity.max()) > p:
        raise ValueError("interior knot multiplicity must not exceed the degree")
    spans = np.diff(np.concatenate(([lower], values, [upper])))
    floor = max(64 * np.finfo(float).eps * max(upper - lower, abs(lower), abs(upper)), _MIN_SPAN)
    if np.any(spans <= floor):
        raise ValueError("knot spans must be longer than 64 eps times the knot scale (and normal floats)")


def bspline_curve(degree, knots, controls, weights, t, order=0):
    """Points and derivatives of a B-spline/NURBS curve, shape ``(m, order+1, dim)``.

    Validates the definition once (finite clamped knots per
    :func:`check_clamped_knots`, positive finite weights) and runs the selected
    kernel. ``weights`` is ``None`` for a polynomial curve; their ratio is
    limited to 1e6 and they are scaled by an exact power of two (the curve is
    unchanged); nonzero control coordinates lie in [2**-500, 2**500]. A
    non-finite result is refused, never returned.

    Accuracy contract (checked against exact rational evaluation in the
    tests): the k-th derivative is within ``1e3 * eps * R**(k+1) * C * (degree / h)**k``
    of the exact value, where ``C`` is the largest control coordinate
    magnitude, ``R`` the weight ratio (1 for polynomial curves) and ``h`` the
    shortest non-empty knot span. The error is absolute relative to the curve
    scale: a position far below ``eps * C`` is not resolved. Raises ``ValueError`` for invalid input or a
    parameter outside ``[knots[degree], knots[n]]``.
    """
    degree, order = int(degree), int(order)
    if not 1 <= degree <= MAX_BSPLINE_DEGREE or not 0 <= order <= 2:
        raise ValueError("unsupported B-spline degree or derivative order")
    definition = _prepare_bspline_curve(degree, knots, controls, weights, _immutable=False)
    return _evaluate_prepared_bspline_curve(definition, t, order)


def _prepare_bspline_curve(degree, knots, controls, weights, *, _immutable=True):
    """Validate one private definition; persistent payloads own immutable bytes."""
    degree = int(degree)
    if not 1 <= degree <= MAX_BSPLINE_DEGREE:
        raise ValueError("unsupported B-spline degree or derivative order")
    knots = _aligned(knots)
    controls = _aligned(controls)
    if controls.ndim != 2 or not 1 <= controls.shape[1] <= 4 or len(controls) < degree + 1:
        raise ValueError("controls must be (n, dim) with dim <= 4 and n > degree")
    count = len(controls)
    check_clamped_knots(degree, knots, count)
    if not np.all(np.isfinite(controls)):
        raise ValueError("B-spline controls must be finite")
    magnitudes = np.abs(controls[controls != 0])
    if magnitudes.size and (magnitudes.min() < _MIN_MAGNITUDE or magnitudes.max() > _MAX_MAGNITUDE):
        raise ValueError("B-spline control coordinates must be 0 or within [2**-500, 2**500] in magnitude")
    if weights is None:
        weights = np.empty(0)
    else:
        weights = _aligned(weights)
        if weights.shape != (count,) or not np.all(np.isfinite(weights)) or not np.all(weights > 0):
            raise ValueError("weights must be positive and finite, one per control point")
        if weights.max() / MAX_WEIGHT_RATIO > weights.min():
            raise ValueError("B-spline weight ratio max/min must not exceed 1e6")
        weights = _aligned(np.ldexp(weights, -np.frexp(weights.max())[1]))
    if _immutable:
        def frozen(array):
            return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
        knots, controls, weights = map(frozen, (knots, controls, weights))
    return degree, knots, controls, weights


def _evaluate_prepared_bspline_curve(definition, t, order=0):
    """Evaluate validated private data through the unchanged selected kernel."""
    degree, knots, controls, weights = definition
    order = int(order)
    if not 0 <= order <= 2:
        raise ValueError("unsupported B-spline degree or derivative order")
    count = len(controls)
    t = _aligned(np.atleast_1d(np.asarray(t, dtype=np.float64)))
    if t.ndim != 1 or not np.all((t >= knots[degree]) & (t <= knots[count])):
        raise ValueError("parameter outside the knot domain")
    out = np.empty((len(t), order + 1, controls.shape[1]))
    bspline_evaluate(degree, knots, controls, weights, t, order, out)
    if not np.all(np.isfinite(out)):
        raise ValueError("B-spline evaluation is not representable in float64 (numerical failure)")
    return out


def backend_info() -> dict:
    """The kernel backend of this process: ``{"backend", "reason", "module", "build"}``."""
    return {
        "backend": "native" if _CORE is not None else "python",
        "reason": _REASON,
        "module": getattr(_CORE, "__file__", None),
        "build": dict(_BUILD) if _BUILD is not None else None,
    }


# Freeze trust at the owning backend's definition/load point, NOT at a lazy
# runtime validator import (which could otherwise bless a custom wrapper).
_RUNTIME_NUMPY_NAMES = ('abs', 'all', 'any', 'arange', 'array', 'asarray', 'atleast_1d',
    'clip', 'concatenate', 'diff', 'einsum', 'empty', 'finfo', 'float64', 'frexp',
    'frombuffer', 'geterr', 'geterrcall', 'isfinite', 'ldexp', 'require', 'searchsorted', 'unique', 'where', 'zeros')
_RUNTIME_OWNER_NAMES = ('_prepare_bspline_curve', '_evaluate_prepared_bspline_curve',
                        '_aligned', 'check_clamped_knots', 'bspline_evaluate')
_RUNTIME_PYTHON_GLOBALS = getattr(bspline_evaluate, '__globals__', None)
_RUNTIME_REFERENCE_NAMES = (('find_spans', 'basis_derivatives')
                          if _CORE is None else ())
_RUNTIME_TRUSTED_FUNCTIONS = tuple((name, globals()[name], getattr(globals()[name], '__code__', None))
                                 for name in _RUNTIME_OWNER_NAMES)
_RUNTIME_TRUSTED_NUMPY = tuple((name, getattr(np, name), getattr(getattr(np, name), '__code__', None))
                             for name in _RUNTIME_NUMPY_NAMES)
_RUNTIME_TRUSTED_REFERENCE = tuple((name, _RUNTIME_PYTHON_GLOBALS[name],
                                  getattr(_RUNTIME_PYTHON_GLOBALS[name], '__code__', None))
                                 for name in _RUNTIME_REFERENCE_NAMES)
_RUNTIME_TRUSTED_CORE, _RUNTIME_TRUSTED_KERNELS = _CORE, _KERNELS
_RUNTIME_TRUSTED_NUMPY_MODULE = np
_RUNTIME_TRUSTED_BUILD = repr(_BUILD)


def _capture_point_implementations():
    """Finite effective point-path edges, not general Python closure inspection."""
    functions, edges, seen = [], [], set()
    aliases = ('array', 'asanyarray', '_wrapreduction_any_all', '_wrapfunc', 'c_einsum', 'umr_all', 'umr_any')
    def visit(function):
        if id(function) in seen or not callable(function):
            return
        seen.add(id(function))
        defaults = getattr(function, '__defaults__', None)
        kwdefaults = getattr(function, '__kwdefaults__', None)
        functions.append((function, getattr(function, '__code__', None), defaults, kwdefaults,
                          None if kwdefaults is None else tuple(kwdefaults.items())))
        for attribute in ('_implementation', '__wrapped__'):
            child = getattr(function, attribute, None)
            edges.append((function, attribute, child))
            if child is not None:
                visit(child)
        namespace = getattr(function, '__globals__', None)
        if namespace is not None:
            for name in aliases:
                if name in namespace:
                    child = namespace[name]
                    edges.append((namespace, name, child))
                    visit(child)
    for name in ('all', 'any', 'arange', 'asarray', 'atleast_1d', 'clip', 'concatenate',
                 'einsum', 'empty', 'require', 'searchsorted', 'where', 'zeros'):
        visit(getattr(np, name))
    return tuple(functions), tuple(edges)


_RUNTIME_POINT_IMPLEMENTATIONS, _RUNTIME_POINT_EDGES = _capture_point_implementations()


def _canonical_curve_runtime_context():
    """Known owner configuration only; arbitrary mutable wrappers use fresh validation."""
    if (np is not _RUNTIME_TRUSTED_NUMPY_MODULE
            or _CORE is not _RUNTIME_TRUSTED_CORE or _KERNELS is not _RUNTIME_TRUSTED_KERNELS
            or repr(_BUILD) != _RUNTIME_TRUSTED_BUILD):
        return None
    bindings = ((globals(), _RUNTIME_TRUSTED_FUNCTIONS),
                (vars(np), _RUNTIME_TRUSTED_NUMPY),
                (_RUNTIME_PYTHON_GLOBALS, _RUNTIME_TRUSTED_REFERENCE))
    for namespace, expected in bindings:
        for name, function, code in expected:
            current = namespace.get(name)
            if current is not function or getattr(current, '__code__', None) is not code:
                return None
    if _RUNTIME_REFERENCE_NAMES and _RUNTIME_PYTHON_GLOBALS.get('np') is not np:
        return None
    for function, code, defaults, kwdefaults, values in _RUNTIME_POINT_IMPLEMENTATIONS:
        if (getattr(function, '__code__', None) is not code
                or getattr(function, '__defaults__', None) is not defaults
                or getattr(function, '__kwdefaults__', None) is not kwdefaults):
            return None
        if values is not None and (set(kwdefaults) != {key for key, _ in values}
                                   or any(kwdefaults[key] is not value for key, value in values)):
            return None
    for parent, name, child in _RUNTIME_POINT_EDGES:
        current = parent.get(name) if type(parent) is dict else getattr(parent, name, None)
        if current is not child:
            return None
    errors = tuple(sorted(np.geterr().items()))
    if np.geterrcall() is not None or any(mode in ('call', 'log') for _, mode in errors):
        return None  # callback/logging handlers may have mutable state behind a fixed identity
    limits = (MAX_BSPLINE_DEGREE, MAX_WEIGHT_RATIO, _MIN_SPAN, _MIN_MAGNITUDE, _MAX_MAGNITUDE)
    if any(type(value) not in (int, float) or not np.isfinite(value) for value in limits):
        return None
    # Callable/code ids cannot be reused: originals are pinned above for the
    # life of this backend. Expected memo values retain numerical metadata only.
    identities = tuple((id(function), id(code)) for group in
        (_RUNTIME_TRUSTED_FUNCTIONS, _RUNTIME_TRUSTED_NUMPY, _RUNTIME_TRUSTED_REFERENCE)
        for _, function, code in group)
    limit_bits = tuple((type(value).__name__, value if type(value) is int else np.float64(value).tobytes())
                       for value in limits)
    return identities, limit_bits, np.__version__, errors, id(_CORE)
