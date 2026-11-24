"""Weighted quadrature sums with one platform-independent value (0.4.7 gate, class A, 2026-11-24).

A BLAS dot product is not a defined value: on Intel macOS numpy links Apple Accelerate, whose 1-D dot
(and matrix-vector product with a long inner dimension) gives results that differ in the last bits
with the memory offset of the same values. Quadrature sums therefore never go through ``@``/``dot``.

Each sum here is the IEEE elementwise products followed by one correctly rounded sum
(``math.fsum``). When the products are not all finite, or their exact sum overflows, ``math.fsum``
raises; the value is then the IEEE result of numpy's own (repeatable) summation, which is ``inf``,
``-inf`` or ``nan`` - the value a dot product gave - so callers keep their non-finite handling.
Shapes must match exactly: nothing is broadcast.
"""
import math

import numpy as np


def _sum(products):
    try:
        return math.fsum(products.tolist())
    except (OverflowError, ValueError):
        return float(np.sum(products))


def weighted_sum(weights, values):
    """``sum_i weights[i]*values[i]`` for two 1-D arrays of the same length."""
    weights, values = np.asarray(weights, dtype=float), np.asarray(values, dtype=float)
    if weights.ndim != 1 or values.shape != weights.shape:
        raise ValueError(f'weighted_sum needs two 1-D arrays of one length, got {weights.shape} and {values.shape}')
    return _sum(weights*values)


def weighted_rows(values, weights):
    """``values @ weights`` for ``values`` of shape ``(m, n)`` and ``weights`` of shape ``(n,)``: ``m`` sums."""
    weights, values = np.asarray(weights, dtype=float), np.asarray(values, dtype=float)
    if weights.ndim != 1 or values.ndim != 2 or values.shape[1] != weights.shape[0]:
        raise ValueError(f'weighted_rows needs (m, n) and (n,), got {values.shape} and {weights.shape}')
    return np.array([_sum(row) for row in values*weights], dtype=float).reshape(values.shape[0])


def weighted_columns(weights, values):
    """``weights @ values`` for ``weights`` of shape ``(n,)`` and ``values`` of shape ``(n, k)``: ``k`` sums."""
    weights, values = np.asarray(weights, dtype=float), np.asarray(values, dtype=float)
    if weights.ndim != 1 or values.ndim != 2 or values.shape[0] != weights.shape[0]:
        raise ValueError(f'weighted_columns needs (n,) and (n, k), got {weights.shape} and {values.shape}')
    return np.array([_sum(column) for column in (values*weights[:, None]).T], dtype=float).reshape(values.shape[1])
