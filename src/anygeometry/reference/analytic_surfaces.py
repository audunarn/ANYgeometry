"""References for analytic surface kernels (same signatures as ``anygeometry._core``)."""
from __future__ import annotations

import numpy as np


def plane_evaluate(frame: np.ndarray, uv: np.ndarray, out: np.ndarray) -> None:
    """Write ``origin + u*u_vector + v*v_vector`` for each ``(u, v)`` row into ``out``.

    ``frame`` holds origin, u_vector and v_vector (9 float64 values); ``uv`` is
    ``(n, 2)`` and ``out`` is ``(n, 3)``, both C-contiguous float64.
    """
    origin, u_vector, v_vector = frame[0:3], frame[3:6], frame[6:9]
    out[...] = (origin + uv[:, 0:1] * u_vector) + uv[:, 1:2] * v_vector
