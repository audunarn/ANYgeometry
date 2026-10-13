"""Smoke-test an installed ANYgeometry and its kernel backend outside the source tree.

    python tools/native_wheel_smoke.py --require-native   # the compiled core must be in use
    python tools/native_wheel_smoke.py --expect-python    # the pure-Python references must be in use

Exits non-zero when ANYgeometry is imported from a source checkout, when the
backend differs from the expectation, or when a kernel disagrees with its
reference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--require-native", action="store_true")
    group.add_argument("--expect-python", action="store_true")
    args = parser.parse_args()

    import numpy as np

    import anygeometry
    from anygeometry import backend, reference

    location = Path(anygeometry.__file__).resolve()
    if "site-packages" not in location.parts and "dist-packages" not in location.parts:
        sys.exit(f"anygeometry imported from {location}, not from an installed distribution")
    info = anygeometry.backend_info()
    print(json.dumps({"python": sys.version.split()[0], "anygeometry": anygeometry.__version__, **info}))
    expected = "native" if args.require_native else "python"
    if info["backend"] != expected:
        sys.exit(f"backend is {info['backend']} ({info['reason']}), expected {expected}")

    rng = np.random.default_rng(7)
    frame, uv = rng.standard_normal(9), rng.uniform(-1.0, 2.0, size=(257, 2))
    expected_points, points = np.empty((257, 3)), np.empty((257, 3))
    reference.plane_evaluate(frame, uv, expected_points)
    backend.plane_evaluate(frame, uv, points)
    if not np.array_equal(points, expected_points):
        sys.exit("plane_evaluate disagrees with its reference")

    # A rational quadratic full circle: radius exact to rounding, kernel = reference.
    half = 0.5 ** 0.5
    knots = np.array([0, 0, 0, .25, .25, .5, .5, .75, .75, 1, 1, 1.])
    controls = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0.]])
    weights = np.array([1, half, 1, half, 1, half, 1, half, 1.])
    t = np.linspace(0.0, 1.0, 1001)
    values = backend.bspline_curve(2, knots, controls, weights, t, 2)
    if np.max(np.abs(np.linalg.norm(values[:, 0], axis=1) - 1.0)) > 4 * np.finfo(float).eps:
        sys.exit("bspline_curve circle radius error exceeds 4 eps")
    expected_values = np.empty_like(values)
    scaled = np.ldexp(weights, -np.frexp(weights.max())[1])
    reference.bspline_evaluate(2, knots, controls, scaled, t, 2, expected_values)
    if np.max(np.abs(values - expected_values)) > 1e-13:
        sys.exit("bspline_evaluate disagrees with its reference")

    model = anygeometry.GeometryModel()
    model.add_plate(model.add_points(((0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0))))
    model.validate_topology()
    print("ok")


if __name__ == "__main__":
    main()
