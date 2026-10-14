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
from typing import Callable
import weakref

import numpy as np

from .exact_predicates import correctly_rounded_sqrt, exact, exact_cross, exact_dot, exact_vector
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


def _down(value: Fraction) -> float:
    rounded = float(value)
    return math.nextafter(rounded, -math.inf) if Fraction(rounded) > value else rounded


def _up(value: Fraction) -> float:
    rounded = float(value)
    return math.nextafter(rounded, math.inf) if Fraction(rounded) < value else rounded


def _plane_capabilities(plane: Plane) -> SurfaceCapabilities:
    from .planar_exact import ExactPlane
    chart = ExactPlane.from_plane(plane)
    origin, u_vector, v_vector = chart.origin, chart.u_vector, chart.v_vector

    def exact_uv(point):
        return chart.uv(point)

    def enclosure(box):
        (u0, u1), (v0, v1) = box
        corners = [tuple(o + exact(u) * a + exact(v) * b for o, a, b in zip(origin, u_vector, v_vector))
                   for u in (u0, u1) for v in (v0, v1)]
        return Enclosure(tuple(_down(min(c[k] for c in corners)) for k in range(3)),
                         tuple(_up(max(c[k] for c in corners)) for k in range(3)))

    # [u v] is constant: its singular values are the square roots of the Gram matrix eigenvalues.
    uu, uv_, vv = exact_dot(u_vector, u_vector), exact_dot(u_vector, v_vector), exact_dot(v_vector, v_vector)
    trace, determinant = uu + vv, uu * vv - uv_ * uv_
    discriminant = trace * trace - 4 * determinant
    root_up = Fraction(math.nextafter(correctly_rounded_sqrt(discriminant), math.inf))   # >= sqrt(disc)
    lambda_max = (trace + root_up) / 2                                                  # upper bound
    lambda_min = max(Fraction(0), (trace - root_up) / 2)                                # lower bound
    sigma_max = math.nextafter(correctly_rounded_sqrt(lambda_max), math.inf)
    sigma_min = max(0.0, math.nextafter(correctly_rounded_sqrt(lambda_min), -math.inf))
    bounds = JacobianBounds(sigma_max, sigma_min)
    normal = exact_cross(u_vector, v_vector)
    implicit = ImplicitForm(1, (*normal, -exact_dot(normal, origin)), _up(max(abs(value) for value in normal)))
    return SurfaceCapabilities(
        kind=SurfaceKind.PLANE, definition=plane, support_key=("plane", *map(tuple, (plane.origin, plane.u_vector,
                                                                                     plane.v_vector))),
        domain=((-math.inf, math.inf), (-math.inf, math.inf)), periodic_u=None,
        evaluate=_evaluate(plane), derivatives=_derivatives(plane), invert_local=_invert_local(plane),
        exact_uv=exact_uv, enclosure=enclosure, jacobian_bounds=lambda box: bounds, implicit=implicit,
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


_CACHE: dict = {}


def capabilities(surface) -> SurfaceCapabilities:
    """The capability record of a built-in surface, memoised per definition object.

    The surface dataclasses hold arrays (not hashable), so the memo is keyed by
    object identity and an entry is dropped when its definition is collected.
    """
    cached = _CACHE.get(id(surface))
    if cached is not None and cached[0]() is surface:
        return cached[1]
    if isinstance(surface, Plane):
        result = _plane_capabilities(surface)
    elif isinstance(surface, Cone):
        result = _generic_capabilities(surface, SurfaceKind.CONE, periodic_u=_full_turn(surface))
    elif isinstance(surface, Cylinder):
        result = _generic_capabilities(surface, SurfaceKind.CYLINDER, periodic_u=_full_turn(surface))
    elif isinstance(surface, ExtrudedSurface):
        result = _generic_capabilities(surface, SurfaceKind.EXTRUDED)
    elif isinstance(surface, RuledSurface):
        result = _generic_capabilities(surface, SurfaceKind.RULED, is_exact=False)
    elif isinstance(surface, CoonsSurface):
        result = _generic_capabilities(surface, SurfaceKind.COONS, is_exact=False)
    else:
        raise TypeError(f"no surface capabilities for {type(surface).__name__}")
    try:
        key = id(surface)
        _CACHE[key] = (weakref.ref(surface), result)
        weakref.finalize(surface, _CACHE.pop, key, None)
    except TypeError:                                  # not weak-referenceable: no memo
        pass
    return result


__all__ = ["Enclosure", "ImplicitForm", "InversionResult", "JacobianBounds", "SurfaceCapabilities",
           "SurfaceKind", "capabilities"]
