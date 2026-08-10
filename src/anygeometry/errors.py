"""Geometry-domain exceptions without meshing dependencies."""

from __future__ import annotations

__all__ = ["GeometryError", "GeometryTopologyError"]


class GeometryError(ValueError):
    """Raised when a geometric construction or query is invalid."""


class GeometryTopologyError(GeometryError):
    """Raised when entity relationships do not form valid topology."""
