"""Private scalar cross-product kernel for demonstrated geometry hot paths.

``_cross3`` performs the same three float64 multiply/subtract operations as
:func:`numpy.cross` for two shape-``(3,)`` ``float64`` vectors, so results are
bit-identical without numpy's dispatch overhead. Every other input falls back
to :func:`numpy.cross` unchanged. NumPy itself is never monkeypatched.
"""
import math

import numpy as np


def _cross3(a, b):
    if (type(a) is np.ndarray and type(b) is np.ndarray
            and a.shape == (3,) and b.shape == (3,)
            and a.dtype == np.float64 and b.dtype == np.float64):
        a0, a1, a2 = a[0], a[1], a[2]
        b0, b1, b2 = b[0], b[1], b[2]
        return np.array((a1*b2-a2*b1, a2*b0-a0*b2, a0*b1-a1*b0))
    return np.cross(a, b)


_GAUSS_LEGENDRE = {}


def _gauss_legendre(count):
    """``numpy.polynomial.legendre.leggauss(count)``, computed once per count.

    The returned arrays are read-only; the values are identical to a fresh call.
    """
    rule = _GAUSS_LEGENDRE.get(count)
    if rule is None:
        nodes, weights = np.polynomial.legendre.leggauss(count)
        nodes.flags.writeable = False
        weights.flags.writeable = False
        rule = _GAUSS_LEGENDRE[count] = (nodes, weights)
    return rule

