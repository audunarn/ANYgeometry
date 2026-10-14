"""Surface capability registry (generality programme slice A1a, architecture section 1).

An adapter registry, not a base class: the existing surface dataclasses are
untouched (equality, hashing, repr, records and checksums unchanged).
``capabilities(surface)`` returns a ``SurfaceCapabilities`` record whose
callables delegate to the existing implementations, so every value is
bit-identical to what the engine computes today.

Proof objects (``enclosure``, ``jacobian_bounds``) are sound or ``None``;
``None`` means a route that needs them refuses (architecture 1.1 rule 4).
This slice provides them for ``Plane``; the other built-in types gain them
when a route needs them. ``exact_uv`` (rule 3) is the only inversion a
decision may use and exists for planes only; ``invert_local`` is a local
inversion for coordinates, never a closest-point guarantee and never a
decision input (R.1 A6).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
import math
import sys
import threading
from typing import Callable
import weakref

import numpy as np

from .errors import GeometryError
from .exact_predicates import exact, exact_cross, exact_dot, exact_vector
from .surfaces import (Cone, CoonsSurface, Cylinder, ExtrudedSurface, Plane, RuledSurface,
                       _evaluate_surface_many, _surface_derivatives_many)

UVBox = tuple  # ((u_low, u_high), (v_low, v_high))


class SurfaceKind(str, Enum):
    PLANE = "plane"
    CYLINDER = "cylinder"
    CONE = "cone"
    EXTRUDED = "extruded"
    RULED = "ruled"
    COONS = "coons"
    NURBS = "nurbs"


@dataclass(frozen=True, slots=True)
class Enclosure:
    """An outward-rounded box proven to contain the exact surface over a parameter box."""
    lower: tuple
    upper: tuple


@dataclass(frozen=True, slots=True)
class JacobianBounds:
    """Certified bounds of the singular values of ``[S_u S_v]`` over a parameter box."""
    sigma_max: float
    sigma_min: float            # 0.0: no lower bound


@dataclass(frozen=True, slots=True)
class ImplicitForm:
    """``F(x) = 0`` on the surface. Degree 1 (planes) in this programme.

    ``coefficients`` are exact rationals of the stored doubles: ``(a, b, c, d)``
    with ``F = a x + b y + c z + d`` (``(a, b, c) = u_vector x v_vector``).
    """
    degree: int
    coefficients: tuple
    gradient_inf: float


@dataclass(frozen=True, slots=True)
class InversionResult:
    """Local inversion of world points: chart parameters and the world residual of each."""
    uv: np.ndarray
    residual: np.ndarray


@dataclass(frozen=True, slots=True)
class SurfaceCapabilities:
    kind: SurfaceKind
    definition: object
    support_key: object
    domain: tuple
    periodic_u: float | None
    evaluate: Callable
    derivatives: Callable
    invert_local: Callable
    exact_uv: Callable | None
    enclosure: Callable | None
    jacobian_bounds: Callable | None
    implicit: ImplicitForm | None
    transformed: Callable
    is_exact: bool


def _uv_array(uv) -> np.ndarray:
    values = np.ascontiguousarray(uv, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("surface parameters must be an (n, 2) array")
    return values


def _points_array(points) -> np.ndarray:
    values = np.ascontiguousarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("world points must be an (n, 3) array")
    return values


def _evaluate(surface):
    return lambda uv: _evaluate_surface_many(surface, _uv_array(uv))


def _derivatives(surface):
    def derivatives(uv, order: int = 1):
        if order != 1:
            raise ValueError("only first derivatives are provided for the built-in surfaces")
        return _surface_derivatives_many(surface, _uv_array(uv))
    return derivatives


def _invert_local(surface):
    def invert(points, initial=None, tolerance=0.0):
        points = _points_array(points)
        uv = np.array([surface.local_uv(point) for point in points], dtype=np.float64).reshape(-1, 2)
        residual = np.linalg.norm(_evaluate_surface_many(surface, uv) - points, axis=1) if len(points) else np.zeros(0)
        return InversionResult(uv, residual)
    return invert


def _transformed(surface):
    def transformed(matrix):
        from .operations import _transform_surface
        return _transform_surface(surface, np.asarray(matrix, dtype=float))
    return transformed


_FLOAT_MAX = Fraction(sys.float_info.max)


def _down(value: Fraction) -> float:
    """The largest double <= ``value`` (``-inf`` below the float range)."""
    if value < -_FLOAT_MAX:
        return -math.inf
    if value > _FLOAT_MAX:
        return sys.float_info.max
    rounded = float(value)
    return math.nextafter(rounded, -math.inf) if Fraction(rounded) > value else rounded


def _up(value: Fraction) -> float:
    """The smallest double >= ``value`` (``inf`` above the float range)."""
    if value > _FLOAT_MAX:
        return math.inf
    if value < -_FLOAT_MAX:
        return -sys.float_info.max
    rounded = float(value)
    return math.nextafter(rounded, math.inf) if Fraction(rounded) < value else rounded


def _sqrt_bounds(value: Fraction) -> tuple[Fraction, Fraction]:
    """Exact rational bounds ``lower <= sqrt(value) <= upper`` with relative width below ``2**-60``.

    No float intermediate, so any magnitude is fine (review A1a-2).
    """
    if value < 0:
        raise GeometryError("square root of a negative quantity")
    if value == 0:
        return Fraction(0), Fraction(0)
    shift = max(0, 64 - (value.numerator.bit_length() - value.denominator.bit_length()) // 2)
    scaled = value * (1 << (2 * shift))                          # >= 2**126
    whole = scaled.numerator // scaled.denominator
    root = math.isqrt(whole)
    lower = Fraction(root, 1 << shift)
    exact_root = root * root == whole and scaled.denominator == 1
    return lower, lower if exact_root else Fraction(root + 1, 1 << shift)


def _endpoint(value):
    """A box end kept exact, or refused (review A1a rounds 2-3).

    Rationals (``Fraction`` and subclasses) and integers (Python, NumPy; not
    booleans) become exact ``Fraction``s; real floats (Python, NumPy of any
    width) become exact ``Fraction``s when finite and stay ``+-inf``/NaN
    otherwise. Everything else (``Decimal``, strings, complex, booleans) is refused.
    """
    if isinstance(value, Fraction):
        return Fraction(value.numerator, value.denominator)
    if isinstance(value, (bool, np.bool_)):
        raise GeometryError("chart box bounds must be real numbers, not booleans")
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    if isinstance(value, (float, np.floating)):
        if math.isfinite(value):
            return Fraction(*value.as_integer_ratio())
        return float(value)
    raise GeometryError(f"chart box bounds must be real numbers, not {type(value).__name__}")


def _box_interval(low, high) -> tuple:
    """A chart interval in either order; one infinite end allowed, both ends at the same infinity refused."""
    values = sorted(_endpoint(value) for value in (low, high))
    if any(type(value) is float and math.isnan(value) for value in values):
        raise GeometryError("chart box bounds must not be NaN")
    if values[0] == math.inf or values[1] == -math.inf:
        raise GeometryError("chart box interval must not lie at infinity")
    return values[0], values[1]


def _term(parameter, coefficient: Fraction):
    """``parameter * coefficient`` with ``0 * inf = 0`` (a zero direction contributes nothing)."""
    if coefficient == 0:
        return Fraction(0)
    if type(parameter) is float and math.isinf(parameter):
        return math.copysign(math.inf, parameter) * (1 if coefficient > 0 else -1)
    return exact(parameter) * coefficient


def _extended_sum(terms, direction: int) -> float:
    """Outward-rounded sum of exact terms and same-signed infinities (direction -1: lower, +1: upper)."""
    infinite = [term for term in terms if type(term) is float]
    if infinite:
        return infinite[0]
    total = sum(terms, Fraction(0))
    return _down(total) if direction < 0 else _up(total)


def _plane_enclosure(chart):
    origin, u_vector, v_vector = chart.origin, chart.u_vector, chart.v_vector

    def enclosure(box):
        (u_low, u_high), (v_low, v_high) = (_box_interval(*interval) for interval in box)
        lower, upper = [], []
        for o, a, b in zip(origin, u_vector, v_vector):
            low_terms = [o, _term(u_low if a > 0 else u_high, a), _term(v_low if b > 0 else v_high, b)]
            high_terms = [o, _term(u_high if a > 0 else u_low, a), _term(v_high if b > 0 else v_low, b)]
            lower.append(_extended_sum(low_terms, -1))
            upper.append(_extended_sum(high_terms, +1))
        return Enclosure(tuple(lower), tuple(upper))
    return enclosure


def _plane_proofs(plane: Plane):
    """The plane's exact derived data; it holds no reference to the definition (memo can drop it)."""
    from .planar_exact import ExactPlane
    chart = ExactPlane.from_plane(plane)
    u_vector, v_vector = chart.u_vector, chart.v_vector
    # [u v] is constant: its singular values are the square roots of the Gram matrix eigenvalues.
    uu, uv_, vv = exact_dot(u_vector, u_vector), exact_dot(u_vector, v_vector), exact_dot(v_vector, v_vector)
    trace, determinant = uu + vv, uu * vv - uv_ * uv_
    _, root_up = _sqrt_bounds(trace * trace - 4 * determinant)                         # >= sqrt(disc)
    lambda_max = (trace + root_up) / 2                                                  # upper bound
    lambda_min = max(Fraction(0), (trace - root_up) / 2)                                # lower bound
    bounds = JacobianBounds(_up(_sqrt_bounds(lambda_max)[1]), max(0.0, _down(_sqrt_bounds(lambda_min)[0])))
    normal = exact_cross(u_vector, v_vector)
    implicit = ImplicitForm(1, (*normal, -exact_dot(normal, chart.origin)), _up(max(abs(value) for value in normal)))
    return chart, _plane_enclosure(chart), bounds, implicit


def _plane_capabilities(plane: Plane) -> SurfaceCapabilities:
    chart, enclosure, bounds, implicit = _derived(plane, lambda: _plane_proofs(plane))
    return SurfaceCapabilities(
        kind=SurfaceKind.PLANE, definition=plane, support_key=("plane", *map(tuple, (plane.origin, plane.u_vector,
                                                                                     plane.v_vector))),
        domain=((-math.inf, math.inf), (-math.inf, math.inf)), periodic_u=None,
        evaluate=_evaluate(plane), derivatives=_derivatives(plane), invert_local=_invert_local(plane),
        exact_uv=chart.uv, enclosure=enclosure, jacobian_bounds=lambda box: bounds, implicit=implicit,
        transformed=_transformed(plane), is_exact=True)


def _generic_capabilities(surface, kind: SurfaceKind, *, periodic_u=None, is_exact=True) -> SurfaceCapabilities:
    return SurfaceCapabilities(
        kind=kind, definition=surface, support_key=(kind.value, id(surface)),
        domain=((0.0, 1.0), (0.0, 1.0)), periodic_u=periodic_u,
        evaluate=_evaluate(surface), derivatives=_derivatives(surface), invert_local=_invert_local(surface),
        exact_uv=None, enclosure=None, jacobian_bounds=None, implicit=None,
        transformed=_transformed(surface), is_exact=is_exact)


def _full_turn(surface) -> float | None:
    return 1.0 if abs(abs(float(surface.sweep_angle)) - math.tau) <= 1e-12 else None


_DERIVED: dict = {}
_LOCK = threading.RLock()                     # re-entrant: a finalizer may run inside the lock (gc)


def _forget(key: int, reference) -> None:
    with _LOCK:
        entry = _DERIVED.get(key)
        if entry is not None and entry[0] is reference:
            del _DERIVED[key]


def _derived(surface, build):
    """Derived data of a definition, memoised per definition object (architecture 1.1 rule 6).

    The definitions hold arrays and are not hashable, so the memo is keyed by
    identity with a weak reference; the value must not reference the
    definition, so the entry is dropped when the definition is collected
    (review A1a-1). Thread-safe: concurrent misses may build twice, but every
    caller gets the first stored value. Definitions that cannot be weakly
    referenced are not memoised.
    """
    key = id(surface)
    with _LOCK:
        entry = _DERIVED.get(key)
        if entry is not None and entry[0]() is surface:
            return entry[1]
    value = build()
    with _LOCK:
        entry = _DERIVED.get(key)
        if entry is not None and entry[0]() is surface:
            return entry[1]
        try:
            reference = weakref.ref(surface)
        except TypeError:
            return value
        _DERIVED[key] = (reference, value)
    weakref.finalize(surface, _forget, key, reference)
    return value


def capabilities(surface) -> SurfaceCapabilities:
    """The capability record of a built-in surface.

    Records are cheap views (callables bound to the definition) built per
    call; the derived proof data behind them is memoised per definition by
    ``_derived``, which never keeps a definition alive.
    """
    if isinstance(surface, Plane):
        return _plane_capabilities(surface)
    if isinstance(surface, Cone):
        return _generic_capabilities(surface, SurfaceKind.CONE, periodic_u=_full_turn(surface))
    if isinstance(surface, Cylinder):
        return _generic_capabilities(surface, SurfaceKind.CYLINDER, periodic_u=_full_turn(surface))
    if isinstance(surface, ExtrudedSurface):
        return _generic_capabilities(surface, SurfaceKind.EXTRUDED)
    if isinstance(surface, RuledSurface):
        return _generic_capabilities(surface, SurfaceKind.RULED, is_exact=False)
    if isinstance(surface, CoonsSurface):
        return _generic_capabilities(surface, SurfaceKind.COONS, is_exact=False)
    raise TypeError(f"no surface capabilities for {type(surface).__name__}")


__all__ = ["Enclosure", "ImplicitForm", "InversionResult", "JacobianBounds", "SurfaceCapabilities",
           "SurfaceKind", "capabilities"]
