"""Vectorized, deterministic geometry evaluation entry points."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .model import GeometryModel

__all__ = [
    "edge_tangent_many",
    "evaluate_edge_many",
    "evaluate_face_many",
    "face_derivatives_many",
    "face_normal_many",
    "project_to_face_many",
]


def evaluate_edge_many(
    geometry: GeometryModel,
    edge_id: int,
    parameters: Sequence[float] | np.ndarray,
) -> np.ndarray:
    return geometry.evaluate_edge_many(edge_id, parameters)


def edge_tangent_many(
    geometry: GeometryModel,
    edge_id: int,
    parameters: Sequence[float] | np.ndarray,
) -> np.ndarray:
    return geometry.edge_tangent_many(edge_id, parameters)


def evaluate_face_many(
    geometry: GeometryModel,
    face_id: int,
    uv: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    return geometry.evaluate_face_many(face_id, uv)


def face_derivatives_many(
    geometry: GeometryModel,
    face_id: int,
    uv: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return geometry.face_derivatives_many(face_id, uv)


def face_normal_many(
    geometry: GeometryModel,
    face_id: int,
    uv: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    return geometry.face_normal_many(face_id, uv)


def project_to_face_many(
    geometry: GeometryModel,
    face_id: int,
    xyz: Sequence[Sequence[float]] | np.ndarray,
    initial_uv: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return geometry.project_to_face_many(
        face_id, xyz, initial_uv=initial_uv
    )
