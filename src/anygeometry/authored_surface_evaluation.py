"""Bound batch evaluation of an authenticated original face support."""
import numbers

import numpy as np

from .authored_boundary_correspondence import validate_prepared_authored_boundary_correspondence_binding
from .authored_domain_coverage import _original_domain
from .errors import GeometryError
from .material_cell_coverage import _frame
from .surfaces import _evaluate_surface_many, _surface_derivatives_many


def evaluate_prepared_authored_face(model, correspondence, parameters, *,
                                    derivatives=False, cancellation_check=None):
    """Evaluate the ORIGINAL support at real finite ``(..., 2)`` parameters.

    Returns detached ``(..., 3)`` positions, or a pair of derivative arrays in
    the original u/v coordinates when ``derivatives=True``. Qualified supports
    are Plane and the exact polynomial extrusions recognized by original-domain
    coverage, including qualified implicit Coons charts. Explicit original
    parameterizations refuse. All finite parameters evaluate the support's
    polynomial extension; no membership in its outer loop, holes or patch is
    asserted. No current-fragment partition, cell or mesh claim is made.
    ``derivatives`` must be a Python or NumPy Boolean scalar.
    """
    if not isinstance(derivatives, (bool, np.bool_)):
        raise GeometryError('authored surface evaluation derivatives must be a Boolean scalar')
    derivatives = bool(derivatives)
    message = 'authored surface evaluation requires finite real (..., 2) parameters'
    try:
        raw = np.asarray(parameters)
        if (raw.dtype.kind not in 'biufO' or
                (raw.dtype.kind == 'O' and any(not isinstance(value, numbers.Real)
                                               for value in raw.flat))):
            raise GeometryError(message)
        values = np.array(raw, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise GeometryError(message) from error
    if values.ndim < 1 or values.shape[-1] != 2 or not np.all(np.isfinite(values)):
        raise GeometryError(message)
    rows = values.reshape((-1, 2))

    def check():
        if cancellation_check is not None and cancellation_check('authored surface evaluation'):
            raise GeometryError('authored surface evaluation cancelled')

    check()
    validate_prepared_authored_boundary_correspondence_binding(
        model, correspondence, cancellation_check=cancellation_check)
    domain = _original_domain(correspondence.authored_definition, check)
    _frame(domain.support)
    check()
    arrays = (_surface_derivatives_many(domain.support, rows) if derivatives else
              (_evaluate_surface_many(domain.support, rows),))
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise GeometryError('authored surface evaluation produced nonfinite results')
    result = tuple(np.array(array, dtype=float, copy=True).reshape((*values.shape[:-1], 3))
                   for array in arrays)
    check()
    validate_prepared_authored_boundary_correspondence_binding(
        model, correspondence, cancellation_check=cancellation_check)
    return result if derivatives else result[0]
