"""Lightweight parametric surfaces for structural shell geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Union, runtime_checkable

import numpy as np

from .errors import GeometryError

__all__ = [
    "CoonsSurface",
    "Cone",
    "Cylinder",
    "Plane",
    "RuledSurface",
    "Surface",
    "SurfaceProtocol",
    "closest_uv",
    "surface_normal",
]


def _vector3(value: object, name: str) -> np.ndarray:
    vector = np.array(value, dtype=float, copy=True)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise GeometryError(f"{name} must be a finite 3-vector")
    vector.flags.writeable = False
    return vector


def _unit(value: object, name: str) -> np.ndarray:
    vector = _vector3(value, name)
    length = float(np.linalg.norm(vector))
    if length <= 0.0:
        raise GeometryError(f"{name} must be non-zero")
    result = vector / length
    result.flags.writeable = False
    return result


@runtime_checkable
class SurfaceProtocol(Protocol):
    """Structural contract shared by all explicit surface types."""

    def evaluate(self, u: float, v: float) -> np.ndarray: ...

    def local_uv(self, point: object) -> tuple[float, float]: ...


@dataclass(frozen=True)
class Plane:
    """Planar patch ``origin + u*u_vector + v*v_vector``."""

    origin: np.ndarray
    u_vector: np.ndarray
    v_vector: np.ndarray

    def __post_init__(self) -> None:
        origin = _vector3(self.origin, "origin")
        u_vector = _vector3(self.u_vector, "u_vector")
        v_vector = _vector3(self.v_vector, "v_vector")
        if float(np.linalg.norm(np.cross(u_vector, v_vector))) <= 1.0e-14:
            raise GeometryError("plane parameter vectors must be independent")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "u_vector", u_vector)
        object.__setattr__(self, "v_vector", v_vector)

    def evaluate(self, u: float, v: float) -> np.ndarray:
        return self.origin + float(u) * self.u_vector + float(v) * self.v_vector

    def local_uv(self, point: object) -> tuple[float, float]:
        offset = _vector3(point, "point") - self.origin
        matrix = np.column_stack((self.u_vector, self.v_vector))
        uv, *_ = np.linalg.lstsq(matrix, offset, rcond=None)
        return float(uv[0]), float(uv[1])

    @property
    def normal(self) -> np.ndarray:
        return _unit(np.cross(self.u_vector, self.v_vector), "plane normal")


@dataclass(frozen=True)
class Cylinder:
    """Cylindrical patch parameterized by angle ``u`` and axial ``v``."""

    origin: np.ndarray
    axis: np.ndarray
    radial_direction: np.ndarray
    radius: float
    height: float
    start_angle: float = 0.0
    sweep_angle: float = 2.0 * np.pi

    def __post_init__(self) -> None:
        origin = _vector3(self.origin, "origin")
        axis = _unit(self.axis, "axis")
        radial = _vector3(self.radial_direction, "radial_direction")
        radial = radial - float(radial @ axis) * axis
        radial = _unit(radial, "radial_direction")
        radius = float(self.radius)
        height = float(self.height)
        sweep = float(self.sweep_angle)
        if not np.isfinite(radius) or radius <= 0.0:
            raise GeometryError("cylinder radius must be finite and positive")
        if not np.isfinite(height) or height == 0.0:
            raise GeometryError("cylinder height must be finite and non-zero")
        if not np.isfinite(sweep) or sweep == 0.0:
            raise GeometryError("cylinder sweep_angle must be finite and non-zero")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "radial_direction", radial)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "start_angle", float(self.start_angle))
        object.__setattr__(self, "sweep_angle", sweep)

    @property
    def circumferential_direction(self) -> np.ndarray:
        return np.cross(self.axis, self.radial_direction)

    def evaluate(self, u: float, v: float) -> np.ndarray:
        angle = self.start_angle + float(u) * self.sweep_angle
        radial = (
            np.cos(angle) * self.radial_direction
            + np.sin(angle) * self.circumferential_direction
        )
        return self.origin + self.radius * radial + float(v) * self.height * self.axis

    def local_uv(self, point: object) -> tuple[float, float]:
        offset = _vector3(point, "point") - self.origin
        axial = float(offset @ self.axis)
        radial = offset - axial * self.axis
        angle = float(
            np.arctan2(
                radial @ self.circumferential_direction,
                radial @ self.radial_direction,
            )
        )
        delta = _angle_on_sweep(angle, self.start_angle, self.sweep_angle)
        return delta / self.sweep_angle, axial / self.height


@dataclass(frozen=True)
class Cone:
    """Conical frustum with radii varying linearly along the axis."""

    origin: np.ndarray
    axis: np.ndarray
    radial_direction: np.ndarray
    radius_start: float
    radius_end: float
    height: float
    start_angle: float = 0.0
    sweep_angle: float = 2.0 * np.pi

    def __post_init__(self) -> None:
        cylinder = Cylinder(
            self.origin,
            self.axis,
            self.radial_direction,
            max(float(self.radius_start), float(self.radius_end), 1.0),
            self.height,
            self.start_angle,
            self.sweep_angle,
        )
        r0 = float(self.radius_start)
        r1 = float(self.radius_end)
        if not np.isfinite(r0) or not np.isfinite(r1) or min(r0, r1) < 0.0:
            raise GeometryError("cone radii must be finite and non-negative")
        if max(r0, r1) <= 0.0:
            raise GeometryError("at least one cone radius must be positive")
        object.__setattr__(self, "origin", cylinder.origin)
        object.__setattr__(self, "axis", cylinder.axis)
        object.__setattr__(self, "radial_direction", cylinder.radial_direction)
        object.__setattr__(self, "height", cylinder.height)
        object.__setattr__(self, "start_angle", cylinder.start_angle)
        object.__setattr__(self, "sweep_angle", cylinder.sweep_angle)
        object.__setattr__(self, "radius_start", r0)
        object.__setattr__(self, "radius_end", r1)

    @property
    def circumferential_direction(self) -> np.ndarray:
        return np.cross(self.axis, self.radial_direction)

    def evaluate(self, u: float, v: float) -> np.ndarray:
        v = float(v)
        radius = (1.0 - v) * self.radius_start + v * self.radius_end
        angle = self.start_angle + float(u) * self.sweep_angle
        radial = (
            np.cos(angle) * self.radial_direction
            + np.sin(angle) * self.circumferential_direction
        )
        return self.origin + radius * radial + v * self.height * self.axis

    def local_uv(self, point: object) -> tuple[float, float]:
        offset = _vector3(point, "point") - self.origin
        axial = float(offset @ self.axis)
        radial = offset - axial * self.axis
        angle = float(
            np.arctan2(
                radial @ self.circumferential_direction,
                radial @ self.radial_direction,
            )
        )
        delta = _angle_on_sweep(angle, self.start_angle, self.sweep_angle)
        return delta / self.sweep_angle, axial / self.height


@dataclass(frozen=True)
class RuledSurface:
    """Surface linearly joining two sampled boundary curves."""

    first_boundary: np.ndarray
    second_boundary: np.ndarray

    def __post_init__(self) -> None:
        first = np.array(self.first_boundary, dtype=float, copy=True)
        second = np.array(self.second_boundary, dtype=float, copy=True)
        if (
            first.ndim != 2
            or first.shape[1:] != (3,)
            or first.shape != second.shape
            or len(first) < 2
            or not np.all(np.isfinite(first))
            or not np.all(np.isfinite(second))
        ):
            raise GeometryError("ruled boundaries must be matching finite (n, 3) arrays")
        first.flags.writeable = False
        second.flags.writeable = False
        object.__setattr__(self, "first_boundary", first)
        object.__setattr__(self, "second_boundary", second)

    @staticmethod
    def _sample(points: np.ndarray, u: float) -> np.ndarray:
        parameter = np.clip(float(u), 0.0, 1.0) * (len(points) - 1)
        index = min(int(np.floor(parameter)), len(points) - 2)
        local = parameter - index
        return (1.0 - local) * points[index] + local * points[index + 1]

    def evaluate(self, u: float, v: float) -> np.ndarray:
        first = self._sample(self.first_boundary, u)
        second = self._sample(self.second_boundary, u)
        return (1.0 - float(v)) * first + float(v) * second

    def local_uv(self, point: object) -> tuple[float, float]:
        return closest_uv(self, point)


@dataclass(frozen=True)
class CoonsSurface:
    """Four-boundary Coons patch, or a topology-backed marker when empty.

    Explicit boundaries are matching sampled ``(n, 3)`` arrays in logical
    parameter directions: bottom/top increase with ``u`` and left/right with
    ``v``.  ``GeometryModel`` supplies those curves from topology when all
    four values are omitted.
    """

    bottom: np.ndarray | None = None
    right: np.ndarray | None = None
    top: np.ndarray | None = None
    left: np.ndarray | None = None

    def __post_init__(self) -> None:
        supplied = (self.bottom, self.right, self.top, self.left)
        if all(value is None for value in supplied):
            return
        if any(value is None for value in supplied):
            raise GeometryError("a Coons surface needs all four boundaries")
        arrays = tuple(np.array(value, dtype=float, copy=True) for value in supplied)
        if any(
            array.ndim != 2
            or array.shape[1:] != (3,)
            or len(array) < 2
            or not np.all(np.isfinite(array))
            for array in arrays
        ):
            raise GeometryError("Coons boundaries must be finite (n, 3) arrays")

        # Keep the public orientation contract explicit: bottom and top run
        # from left to right, while left and right run from bottom to top.
        # A relative tolerance based on the patch extent admits harmless
        # floating-point roundoff without making the result depend on an
        # arbitrary global coordinate offset.
        points = np.vstack(arrays)
        extent = float(np.linalg.norm(np.ptp(points, axis=0)))
        tolerance = max(1.0e-13, 1.0e-10 * extent)
        corner_pairs = (
            ("bottom[0]", arrays[0][0], "left[0]", arrays[3][0]),
            ("bottom[-1]", arrays[0][-1], "right[0]", arrays[1][0]),
            ("top[0]", arrays[2][0], "left[-1]", arrays[3][-1]),
            ("top[-1]", arrays[2][-1], "right[-1]", arrays[1][-1]),
        )
        for first_name, first, second_name, second in corner_pairs:
            gap = float(np.linalg.norm(first - second))
            if gap > tolerance:
                raise GeometryError(
                    "Coons boundaries have incompatible oriented corners: "
                    f"{first_name} and {second_name} differ by {gap:.6g} "
                    f"(tolerance {tolerance:.6g})"
                )
        for name, array in zip(("bottom", "right", "top", "left"), arrays):
            array.flags.writeable = False
            object.__setattr__(self, name, array)

    @property
    def has_boundaries(self) -> bool:
        return self.bottom is not None

    @staticmethod
    def _sample(points: np.ndarray, parameter: float) -> np.ndarray:
        scaled = np.clip(float(parameter), 0.0, 1.0) * (len(points)-1)
        index = min(int(np.floor(scaled)), len(points)-2)
        local = scaled-index
        return (1.0-local)*points[index] + local*points[index+1]

    def evaluate(self, u: float, v: float) -> np.ndarray:
        if not self.has_boundaries:
            raise GeometryError(
                "topology-backed Coons evaluation requires GeometryModel.face_point"
            )
        assert self.bottom is not None and self.right is not None and self.top is not None and self.left is not None
        bottom = self._sample(self.bottom, u)
        top = self._sample(self.top, u)
        left = self._sample(self.left, v)
        right = self._sample(self.right, v)
        corner_00, corner_10 = self.bottom[0], self.bottom[-1]
        corner_01, corner_11 = self.top[0], self.top[-1]
        return (
            (1.0-v)*bottom + v*top + (1.0-u)*left + u*right
            - ((1.0-u)*(1.0-v)*corner_00 + u*(1.0-v)*corner_10 + u*v*corner_11 + (1.0-u)*v*corner_01)
        )

    def local_uv(self, point: object) -> tuple[float, float]:
        if not self.has_boundaries:
            raise GeometryError(
                "topology-backed Coons inversion requires GeometryModel.face_local_uv"
            )
        return closest_uv(self, point)


Surface = Union[Plane, Cylinder, Cone, RuledSurface, CoonsSurface]


def _angle_on_sweep(angle: float, start: float, sweep: float) -> float:
    raw = float(angle - start)
    tolerance = 64.0 * np.finfo(float).eps * max(
        1.0, abs(float(angle)), abs(float(start)), abs(float(sweep))
    )
    if abs(raw) <= tolerance:
        return 0.0
    if sweep > 0.0:
        delta = float(raw % (2.0 * np.pi))
        return 0.0 if abs(delta - 2.0 * np.pi) <= tolerance else delta
    delta = -float((-raw) % (2.0 * np.pi))
    return 0.0 if abs(delta + 2.0 * np.pi) <= tolerance else delta


def closest_uv(
    surface: SurfaceProtocol,
    point: object,
    *,
    initial: tuple[float, float] = (0.5, 0.5),
    iterations: int = 30,
) -> tuple[float, float]:
    """Deterministic bounded Gauss-Newton closest-point parameters."""

    target = _vector3(point, "point")
    uv = np.asarray(initial, dtype=float).copy()
    step = 1.0e-6
    for _ in range(int(iterations)):
        current = surface.evaluate(float(uv[0]), float(uv[1]))
        du = (surface.evaluate(float(uv[0] + step), float(uv[1])) - current) / step
        dv = (surface.evaluate(float(uv[0]), float(uv[1] + step)) - current) / step
        jacobian = np.column_stack((du, dv))
        delta, *_ = np.linalg.lstsq(jacobian, target - current, rcond=None)
        uv += delta
        uv[:] = np.clip(uv, 0.0, 1.0)
        if float(np.linalg.norm(delta)) <= 1.0e-12:
            break
    return float(uv[0]), float(uv[1])


def surface_normal(surface: SurfaceProtocol, u: float, v: float) -> np.ndarray:
    """Numerical unit normal shared by all surface implementations."""

    step = 1.0e-6
    u_value, v_value = float(u), float(v)

    def derivative(parameter: float, axis: int) -> np.ndarray:
        if parameter <= step:
            lower, upper = parameter, parameter + step
        elif parameter >= 1.0 - step:
            lower, upper = parameter - step, parameter
        else:
            lower, upper = parameter - step, parameter + step
        if axis == 0:
            first = surface.evaluate(lower, v_value)
            second = surface.evaluate(upper, v_value)
        else:
            first = surface.evaluate(u_value, lower)
            second = surface.evaluate(u_value, upper)
        return (second - first) / (upper - lower)

    du = derivative(u_value, 0)
    dv = derivative(v_value, 1)
    return _unit(np.cross(du, dv), "surface normal")
