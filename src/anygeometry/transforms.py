"""Immutable affine transforms shared by editing and pattern operations."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Sequence, TypeAlias

import numpy as np

from .errors import GeometryError

__all__ = ["AffineLike", "AffineTransform", "coerce_affine_transform"]


def _vector3(value: object, name: str) -> np.ndarray:
    try:
        made = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise GeometryError(f"{name} must be a finite 3-vector") from error
    if made.shape != (3,) or not np.all(np.isfinite(made)):
        raise GeometryError(f"{name} must be a finite 3-vector")
    return made


def _validated_matrix(value: object) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise GeometryError("transform must be a finite 4x4 matrix") from error
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise GeometryError("transform must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-14):
        raise GeometryError("only affine homogeneous transforms are supported")
    singular_values = np.linalg.svd(matrix[:3, :3], compute_uv=False)
    scale = float(singular_values.max())
    if scale <= 0.0 or float(singular_values.min()) <= 1.0e-12 * scale:
        raise GeometryError("singular affine transforms are not supported")
    made = np.array(matrix, dtype=float, copy=True)
    made[3] = (0.0, 0.0, 0.0, 1.0)
    made.flags.writeable = False
    return made


@dataclass(frozen=True, slots=True, init=False, eq=False)
class AffineTransform:
    """A validated immutable 3D affine transform.

    ``first @ second`` applies ``second`` first and ``first`` second, matching
    homogeneous column-vector matrix multiplication. Angles are radians and
    rotations follow the right-hand rule.
    """

    _matrix: np.ndarray

    def __init__(self, matrix: object) -> None:
        object.__setattr__(self, "_matrix", _validated_matrix(matrix))

    @property
    def matrix(self) -> np.ndarray:
        """Return a read-only view of the homogeneous 4x4 matrix."""

        made = self._matrix.view()
        made.flags.writeable = False
        return made

    @classmethod
    def identity(cls) -> "AffineTransform":
        return cls(np.eye(4))

    @classmethod
    def translation(cls, vector: Sequence[float]) -> "AffineTransform":
        matrix = np.eye(4)
        matrix[:3, 3] = _vector3(vector, "translation vector")
        return cls(matrix)

    @classmethod
    def rotation(
        cls,
        axis_point: Sequence[float],
        axis_direction: Sequence[float],
        angle: float,
    ) -> "AffineTransform":
        point = _vector3(axis_point, "rotation axis point")
        direction = _vector3(axis_direction, "rotation axis direction")
        length = float(np.linalg.norm(direction))
        if length <= 0.0:
            raise GeometryError("rotation axis direction must be non-zero")
        try:
            made_angle = float(angle)
        except (TypeError, ValueError) as error:
            raise GeometryError("rotation angle must be finite") from error
        if isinstance(angle, bool) or not np.isfinite(made_angle):
            raise GeometryError("rotation angle must be finite")
        x, y, z = direction / length
        cosine, sine = float(np.cos(made_angle)), float(np.sin(made_angle))
        cross = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
        rotation = (
            cosine * np.eye(3)
            + sine * cross
            + (1.0 - cosine) * np.outer((x, y, z), (x, y, z))
        )
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = point - rotation @ point
        return cls(matrix)

    @classmethod
    def scale(
        cls,
        factors: float | Sequence[float],
        center: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> "AffineTransform":
        if isinstance(factors, bool):
            raise GeometryError("scale factors must be finite and non-zero")
        if isinstance(factors, Real):
            values = np.repeat(float(factors), 3)
        else:
            values = _vector3(factors, "scale factors")
        if not np.all(np.isfinite(values)) or np.any(values == 0.0):
            raise GeometryError("scale factors must be finite and non-zero")
        point = _vector3(center, "scale center")
        matrix = np.eye(4)
        matrix[:3, :3] = np.diag(values)
        matrix[:3, 3] = point - matrix[:3, :3] @ point
        return cls(matrix)

    @classmethod
    def reflection(
        cls,
        plane_point: Sequence[float],
        plane_normal: Sequence[float],
    ) -> "AffineTransform":
        point = _vector3(plane_point, "reflection plane point")
        normal = _vector3(plane_normal, "reflection plane normal")
        length = float(np.linalg.norm(normal))
        if length <= 0.0:
            raise GeometryError("reflection plane normal must be non-zero")
        normal /= length
        linear = np.eye(3) - 2.0 * np.outer(normal, normal)
        matrix = np.eye(4)
        matrix[:3, :3] = linear
        matrix[:3, 3] = point - linear @ point
        return cls(matrix)

    def inverse(self) -> "AffineTransform":
        return AffineTransform(np.linalg.inv(self._matrix))

    def then(self, following: "AffineTransform") -> "AffineTransform":
        """Return a transform that applies this transform, then ``following``."""

        if not isinstance(following, AffineTransform):
            raise TypeError("following transform must be an AffineTransform")
        return following @ self

    def apply_points(self, points: object) -> np.ndarray:
        """Transform one ``(3,)`` point or an arbitrary ``(..., 3)`` array."""

        try:
            values = np.asarray(points, dtype=float)
        except (TypeError, ValueError) as error:
            raise GeometryError("points must be a finite (..., 3) array") from error
        if values.ndim < 1 or values.shape[-1] != 3 or not np.all(np.isfinite(values)):
            raise GeometryError("points must be a finite (..., 3) array")
        return values @ self._matrix[:3, :3].T + self._matrix[:3, 3]

    def __matmul__(self, other: object) -> "AffineTransform":
        if not isinstance(other, AffineTransform):
            return NotImplemented
        return AffineTransform(self._matrix @ other._matrix)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AffineTransform) and bool(
            np.array_equal(self._matrix, other._matrix)
        )

    def __hash__(self) -> int:
        return hash(self._matrix.tobytes())


AffineLike: TypeAlias = AffineTransform | Sequence[Sequence[float]] | np.ndarray


def coerce_affine_transform(
    value: AffineLike | None,
    *,
    default_identity: bool = False,
) -> AffineTransform:
    """Normalize a public transform input without exposing a mutable matrix."""

    if value is None:
        if default_identity:
            return AffineTransform.identity()
        raise GeometryError("a transform is required")
    return value if isinstance(value, AffineTransform) else AffineTransform(value)
