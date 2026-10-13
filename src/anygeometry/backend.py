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
    kernel. ``weights`` is ``None`` for a polynomial curve; they are scaled by
    an exact power of two (the curve is unchanged) so the homogeneous
    arithmetic cannot overflow through the weights. A non-finite result is
    refused, never returned. Raises ``ValueError`` for invalid input or a
    parameter outside ``[knots[degree], knots[n]]``.
    """
    degree, order = int(degree), int(order)
    if not 1 <= degree <= MAX_BSPLINE_DEGREE or not 0 <= order <= 2:
        raise ValueError("unsupported B-spline degree or derivative order")
    knots = _aligned(knots)
    controls = _aligned(controls)
    t = _aligned(np.atleast_1d(np.asarray(t, dtype=np.float64)))
    if controls.ndim != 2 or not 1 <= controls.shape[1] <= 4 or len(controls) < degree + 1:
        raise ValueError("controls must be (n, dim) with dim <= 4 and n > degree")
    count = len(controls)
    check_clamped_knots(degree, knots, count)
    if not np.all(np.isfinite(controls)):
        raise ValueError("B-spline controls must be finite")
    if weights is None:
        weights = np.empty(0)
    else:
        weights = _aligned(weights)
        if weights.shape != (count,) or not np.all(np.isfinite(weights)) or not np.all(weights > 0):
            raise ValueError("weights must be positive and finite, one per control point")
        weights = _aligned(np.ldexp(weights, -np.frexp(weights.max())[1]))
        if not np.all(weights > 0):
            raise ValueError("weight ratio is not representable in float64")
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
