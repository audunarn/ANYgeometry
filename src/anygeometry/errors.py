"""Geometry-domain exceptions without meshing dependencies."""

from __future__ import annotations

__all__ = ["GeometryError", "GeometryTopologyError", "SourceCellRefinementRequired"]


class GeometryError(ValueError):
    """Raised when a geometric construction or query is invalid."""


class GeometryTopologyError(GeometryError):
    """Raised when entity relationships do not form valid topology."""


class SourceCellRefinementRequired(GeometryError):
    """Certified fixed-FE approximation violation; request finer resolution.

    This control signal is neither geometric admission nor a proof receipt.
    Existing GeometryError handlers remain compatible.
    """

    def __init__(self, face_id, node_ids, error_limit, lower_error_squared, *, requested_error=None):
        from fractions import Fraction
        if (type(face_id) is not int or face_id < 0 or type(node_ids) is not tuple
                or len(node_ids) not in (3, 4) or any(type(n) is not int or n < 0 for n in node_ids)
                or len(set(node_ids)) != len(node_ids)):
            raise GeometryError('source-cell refinement connectivity is malformed')
        try:
            limit, lower = Fraction(error_limit), Fraction(lower_error_squared)
            requested = limit if requested_error is None else Fraction(requested_error)
        except (TypeError, ValueError, OverflowError, ZeroDivisionError) as cause:
            raise GeometryError('source-cell refinement bounds must be finite rational values') from cause
        if limit <= 0 or requested < limit or lower <= limit*limit:
            raise GeometryError('source-cell refinement requires a strict certified error violation')
        self.face_id, self.node_ids = face_id, node_ids
        self.error_limit, self.lower_error_squared = limit, lower
        self.requested_error = requested
        super().__init__(f'source-cell approximation requires finer mesh resolution (face {face_id})')
