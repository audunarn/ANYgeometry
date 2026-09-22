"""Read-only document-unit conversions between model-local and world frames.

The document transform is the entire mapping. local_origin and CRS metadata
are descriptive, not extra transformations. Vectors here are not normals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .errors import GeometryError

if TYPE_CHECKING:
    from .model import GeometryModel

__all__ = [
    "model_to_world_points", "world_to_model_points",
    "model_to_world_vectors", "world_to_model_vectors",
]


def _convert(model: GeometryModel, values: object, *, inverse: bool, points: bool) -> np.ndarray:
    try:
        raw = np.asarray(values)
        if np.iscomplexobj(raw):
            raise ValueError("complex coordinates are not supported")
        array = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise GeometryError("coordinates must be a finite real (..., 3) array") from error
    if array.ndim < 1 or array.shape[-1] != 3 or not np.all(np.isfinite(array)):
        raise GeometryError("coordinates must be a finite real (..., 3) array")

    stored = model.coordinate_transform
    if stored is None:
        return array.copy()
    matrix = np.asarray(stored, dtype=float)
    if not points:
        matrix = matrix[:3, :3]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if inverse:
            try:
                matrix = np.linalg.inv(matrix)
            except np.linalg.LinAlgError as error:
                raise GeometryError("coordinate transform has no usable inverse") from error
        if not np.all(np.isfinite(matrix)):
            raise GeometryError("coordinate transform is nonfinite")
        if points:
            flat = array.reshape(-1, 3)
            homogeneous = np.concatenate((flat, np.ones((len(flat), 1))), axis=1) @ matrix.T
            weight = homogeneous[:, 3:4]
            if np.any(np.abs(weight) <= np.finfo(float).eps):
                raise GeometryError("coordinate transform produced an invalid homogeneous point")
            result = (homogeneous[:, :3] / weight).reshape(array.shape)
        else:
            result = array @ matrix.T
    if not np.all(np.isfinite(result)):
        raise GeometryError("coordinate transform produced nonfinite coordinates")
    return result


def model_to_world_points(model: GeometryModel, points: object) -> np.ndarray:
    """Map finite (..., 3) model-unit points through the document transform.

    Returns a new array. A missing transform is identity. No unit conversion,
    local-origin offset, CRS projection or model mutation is performed.
    """
    return _convert(model, points, inverse=False, points=True)


def world_to_model_points(model: GeometryModel, points: object) -> np.ndarray:
    """Map world points, expressed in model units, through the inverse mapping."""
    return _convert(model, points, inverse=True, points=True)


def model_to_world_vectors(model: GeometryModel, vectors: object) -> np.ndarray:
    """Apply the linear model-to-world map without translation or normalization.

    This maps vectors, not surface normals. Length-bearing vectors are in
    document units; dimensionless directions remain unnormalized directions.
    """
    return _convert(model, vectors, inverse=False, points=False)


def world_to_model_vectors(model: GeometryModel, vectors: object) -> np.ndarray:
    """Apply the inverse linear map without translation or normalization."""
    return _convert(model, vectors, inverse=True, points=False)
