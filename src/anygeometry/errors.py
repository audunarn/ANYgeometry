"""Geometry-domain exceptions without meshing dependencies."""

from __future__ import annotations

__all__ = ["GeometryError", "GeometryTopologyError", "SourceCellRefinementRequired",
           "SourceCellApproximationInconclusive"]


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


class _FanApproximationInconclusive(GeometryError):
    """Private terminal-box result; the source-cell owner must bind it freshly."""

    def __init__(self, error_limit, upper_error, jacobian_lower, proof_box, proof_depth):
        self.error_limit, self.upper_error = error_limit, upper_error
        self.jacobian_lower, self.proof_box, self.proof_depth = jacobian_lower, proof_box, proof_depth
        super().__init__('regular source fan approximation is inconclusive at the existing depth limit')


class SourceCellApproximationInconclusive(GeometryError):
    """Not certified at this cell resolution; neither violation nor admission.

    upper_error bounds only the unresolved terminal proof subbox, not the
    whole cell. It is not a lower witness. A consumer may make an already-
    bounded resolution attempt, but must obtain a fresh full proof.
    """

    def __init__(self, face_id, node_ids, error_limit, upper_error, *, requested_error,
                 jacobian_lower, proof_box, proof_depth, source_model_id, source_revision,
                 source_checksum, chord_checksum):
        from fractions import Fraction
        from uuid import UUID
        if (type(face_id) is not int or face_id < 0 or type(node_ids) is not tuple
                or len(node_ids) not in (3,4) or any(type(n) is not int or n < 0 for n in node_ids)
                or len(set(node_ids)) != len(node_ids)):
            raise GeometryError('inconclusive source-cell connectivity is malformed')
        try:
            limit, upper, requested, wedge = map(Fraction,(error_limit,upper_error,requested_error,jacobian_lower))
        except (TypeError,ValueError,OverflowError,ZeroDivisionError) as cause:
            raise GeometryError('inconclusive source-cell bounds must be finite rational values') from cause
        if limit <= 0 or requested < limit or upper <= limit or wedge <= 0:
            raise GeometryError('inconclusive approximation requires a positive wedge and an upper bound above allocation')
        if (type(proof_depth) is not int or proof_depth < 0 or type(proof_box) is not tuple
                or len(proof_box) != 2 or any(type(row) is not tuple or len(row) != 2
                or any(type(x) is not Fraction for x in row) or not 0 <= row[0] < row[1] <= 1
                for row in proof_box)):
            raise GeometryError('inconclusive source-cell terminal box is malformed')
        if (type(source_model_id) is not UUID or type(source_revision) is not int or source_revision < 0
                or any(type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value)
                       for value in (source_checksum,chord_checksum))):
            raise GeometryError('inconclusive source-cell binding is malformed')
        self.face_id,self.node_ids = face_id,node_ids
        self.error_limit,self.upper_error,self.requested_error = limit,upper,requested
        self.jacobian_lower,self.proof_box,self.proof_depth = wedge,proof_box,proof_depth
        self.source_model_id,self.source_revision = source_model_id,source_revision
        self.source_checksum,self.chord_checksum = source_checksum,chord_checksum
        super().__init__(f'source-cell approximation not certified at current resolution (face {face_id})')
