"""Translation-invariant tolerance policy for geometry predicates.

The absolute values in :class:`TolerancePolicy` are expressed in the model's
current units.  Relative contributions are derived only from a participating
feature's local extent; global point magnitudes are deliberately never used.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Iterable

import numpy as np

from .errors import GeometryError

__all__ = ["DEFAULT_TOLERANCE_POLICY", "TolerancePolicy", "feature_extent"]


def _positive_finite(value: object, name: str) -> float:
    try:
        made = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise GeometryError(f"{name} tolerance must be a positive finite number") from error
    if not np.isfinite(made) or made <= 0.0:
        raise GeometryError(f"{name} tolerance must be a positive finite number")
    return made


def _non_negative_finite(value: object, name: str) -> float:
    try:
        made = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise GeometryError(f"{name} must be a non-negative finite number") from error
    if not np.isfinite(made) or made < 0.0:
        raise GeometryError(f"{name} must be a non-negative finite number")
    return made


def _stable_norm(vector: np.ndarray) -> float:
    """Return a finite-safe Euclidean norm without origin-based scaling."""

    largest = float(np.max(np.abs(vector))) if vector.size else 0.0
    if largest == 0.0:
        return 0.0
    made = largest * float(np.linalg.norm(vector / largest))
    return made if np.isfinite(made) else float("inf")


def feature_extent(points: Iterable[object] | np.ndarray) -> float:
    """Return the diagonal of a participating point set's local AABB.

    Subtracting coordinate-wise minima from maxima makes this invariant under
    translating every point.  A single point has zero extent.  Invalid or
    non-finite point sets are rejected rather than silently influencing a
    classification.
    """

    try:
        made = np.asarray(tuple(points) if not isinstance(points, np.ndarray) else points, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise GeometryError("feature points must be a finite (n, 3) array") from error
    if made.ndim == 1 and made.shape == (3,):
        made = made.reshape(1, 3)
    if made.ndim != 2 or made.shape[1:] != (3,) or not np.all(np.isfinite(made)):
        raise GeometryError("feature points must be a finite (n, 3) array")
    if len(made) < 2:
        return 0.0
    return _stable_norm(np.ptp(made, axis=0))


@dataclass(frozen=True, slots=True)
class TolerancePolicy:
    """Model-owned numerical and intentional geometry tolerances.

    ``length``, ``area``, and ``surface_residual`` are computational
    tolerances.  ``merge_length`` is intentionally separate because deciding
    whether to heal geometry is a modelling policy, not a predicate side
    effect.  ``relative_length`` and ``relative_area`` add local, feature-size
    dependent terms without inspecting distance from the global origin.
    """

    length: float = 1.0e-9
    merge_length: float = 1.0e-7
    angular: float = 1.0e-10
    parameter: float = 1.0e-10
    area: float = 1.0e-18
    surface_residual: float = 1.0e-8
    relative_length: float = 1.0e-12
    relative_area: float = 1.0e-14

    def __post_init__(self) -> None:
        for item in fields(self):
            object.__setattr__(
                self,
                item.name,
                _positive_finite(getattr(self, item.name), item.name),
            )

    def effective_length(self, extent: float = 0.0) -> float:
        """Computational length tolerance for a local participating extent."""

        local = _non_negative_finite(extent, "feature extent")
        return max(self.length, self.relative_length * local)

    def effective_merge_length(self, extent: float = 0.0) -> float:
        """Intentional merge/heal tolerance for a local participating extent."""

        local = _non_negative_finite(extent, "feature extent")
        return max(self.merge_length, self.relative_length * local)

    def effective_parameter(self, feature_length: float, extent: float | None = None) -> float:
        """Dimensionless parameter tolerance for a bounded curve.

        The length contribution is divided by the curve length, so uniformly
        scaling both geometry and policy leaves this value unchanged.
        """

        length = _non_negative_finite(feature_length, "feature length")
        if length == 0.0:
            return self.parameter
        local = length if extent is None else _non_negative_finite(extent, "feature extent")
        return max(self.parameter, self.effective_length(local) / length)

    def effective_area(self, extent: float = 0.0) -> float:
        """Computational area tolerance for a local participating extent."""

        local = _non_negative_finite(extent, "feature extent")
        length = self.effective_length(local)
        return max(self.area, self.relative_area * local * local, length * length)

    def effective_surface_residual(self, extent: float = 0.0) -> float:
        """Maximum verified residual for a local surface calculation."""

        local = _non_negative_finite(extent, "feature extent")
        return max(self.surface_residual, self.relative_length * local)

    def scaled(self, factor: float) -> "TolerancePolicy":
        """Return the same physical policy after a uniform unit/geometry scale."""

        scale = _positive_finite(factor, "scale factor")
        return TolerancePolicy(
            length=self.length * scale,
            merge_length=self.merge_length * scale,
            angular=self.angular,
            parameter=self.parameter,
            area=self.area * scale * scale,
            surface_residual=self.surface_residual * scale,
            relative_length=self.relative_length,
            relative_area=self.relative_area,
        )


DEFAULT_TOLERANCE_POLICY = TolerancePolicy()
