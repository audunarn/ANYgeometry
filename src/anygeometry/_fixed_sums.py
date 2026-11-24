"""Weighted quadrature sums with one platform-independent value (0.4.7 gate, class A, 2026-11-24).

A BLAS dot product is not a defined value: on Intel macOS numpy links Apple Accelerate, whose 1-D dot
(and matrix-vector product with a long inner dimension) gives results that differ in the last bits
with the memory offset of the same values. Quadrature sums therefore never go through ``@``/``dot``.

Contract (owner, 2026-11-24): inputs are real float64 arrays only (anything else - complex, object,
integer, float32, a wider long double - is refused with TypeError; nothing is cast). The products
are formed elementwise in IEEE float64. Non-finite products are classified before any summation:
any NaN, or +inf together with -inf, gives NaN; infinities of one sign give that infinity. Only
all-finite products are summed, as the exact sum rounded once: ``math.fsum``, or, where it raises on
intermediate overflow, the exact rational sum rounded once (a signed infinity beyond the range).
The elementwise multiplication itself may emit numpy's floating-point warnings, as ``@`` did.
Shapes must match exactly: nothing is broadcast.
"""
import math
from fractions import Fraction

import numpy as np


def _sum(products):
    """The sum of one 1-D float64 array of products, under the module contract."""
    finite = np.isfinite(products)
    if not finite.all():
        if np.isnan(products).any():
            return math.nan
        positive, negative = bool((products == math.inf).any()), bool((products == -math.inf).any())
        return math.nan if positive and negative else (math.inf if positive else -math.inf)
    values = products.tolist()
    try:
        return math.fsum(values)
    except OverflowError:                   # finite products whose partial sums overflow
        exact = sum(Fraction(value) for value in values)
        try:
            return float(exact)
        except OverflowError:
            return math.inf if exact > 0 else -math.inf


def _real(array, name):
    array = np.asarray(array)
    if array.dtype != np.float64:
        raise TypeError(f'{name} must be a real float64 array, got dtype {array.dtype}')
    return array


def weighted_sum(weights, values):
    """``sum_i weights[i]*values[i]`` for two 1-D arrays of the same length."""
    weights, values = _real(weights, 'weights'), _real(values, 'values')
    if weights.ndim != 1 or values.shape != weights.shape:
        raise ValueError(f'weighted_sum needs two 1-D arrays of one length, got {weights.shape} and {values.shape}')
    return _sum(weights*values)


def weighted_rows(values, weights):
    """``values @ weights`` for ``values`` of shape ``(m, n)`` and ``weights`` of shape ``(n,)``: ``m`` sums."""
    weights, values = _real(weights, 'weights'), _real(values, 'values')
    if weights.ndim != 1 or values.ndim != 2 or values.shape[1] != weights.shape[0]:
        raise ValueError(f'weighted_rows needs (m, n) and (n,), got {values.shape} and {weights.shape}')
    return np.array([_sum(row) for row in values*weights], dtype=float).reshape(values.shape[0])


def weighted_columns(weights, values):
    """``weights @ values`` for ``weights`` of shape ``(n,)`` and ``values`` of shape ``(n, k)``: ``k`` sums."""
    weights, values = _real(weights, 'weights'), _real(values, 'values')
    if weights.ndim != 1 or values.ndim != 2 or values.shape[0] != weights.shape[0]:
        raise ValueError(f'weighted_columns needs (n,) and (n, k), got {weights.shape} and {values.shape}')
    return np.array([_sum(column) for column in (values*weights[:, None]).T], dtype=float).reshape(values.shape[1])
