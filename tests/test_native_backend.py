"""The optional compiled core: backend selection, reporting and differential agreement.

The backend is fixed at import (``anygeometry.backend``). Environment switches
are checked in fresh interpreters. Kernel agreement with the pure-Python
references is checked whenever the compiled module is in use; without it those
tests are skipped with the reported reason.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import anygeometry
from anygeometry import backend, reference

SOURCE = str(Path(anygeometry.__file__).resolve().parents[1])
NATIVE = backend.backend_info()["backend"] == "native"
needs_native = pytest.mark.skipif(not NATIVE, reason=f"compiled core not in use: {backend.backend_info()['reason']}")


def _probe(**environment):
    """backend_info() of a fresh interpreter with the given environment (None removes a variable)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = SOURCE + os.pathsep + env.get("PYTHONPATH", "")
    for key, value in environment.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return subprocess.run(
        [sys.executable, "-c", "import json, anygeometry; print(json.dumps(anygeometry.backend_info()))"],
        capture_output=True, text=True, env=env, timeout=120)


def test_backend_info_reports_one_consistent_choice():
    info = anygeometry.backend_info()
    assert set(info) == {"backend", "reason", "module", "build"}
    assert info["backend"] in {"native", "python"}
    if info["backend"] == "native":
        assert info["module"] and Path(info["module"]).name.startswith("_core")
        assert info["build"]["abi"] == "abi3" and info["build"]["core_api"] == 1
    else:
        assert info["module"] is None and info["build"] is None
    assert backend.plane_evaluate is (backend._CORE.plane_evaluate if NATIVE else reference.plane_evaluate)


def test_disable_switch_selects_the_python_references():
    result = _probe(ANYGEOMETRY_DISABLE_NATIVE="1", ANYGEOMETRY_REQUIRE_NATIVE=None)
    assert result.returncode == 0, result.stderr
    info = json.loads(result.stdout)
    assert info == {"backend": "python", "reason": "disabled by ANYGEOMETRY_DISABLE_NATIVE",
                    "module": None, "build": None}


def test_require_switch_refuses_to_import_without_the_compiled_core():
    result = _probe(ANYGEOMETRY_DISABLE_NATIVE="1", ANYGEOMETRY_REQUIRE_NATIVE="1")
    assert result.returncode != 0
    assert "ANYGEOMETRY_REQUIRE_NATIVE is set but the compiled core is not in use" in result.stderr


@needs_native
def test_require_switch_accepts_the_compiled_core():
    result = _probe(ANYGEOMETRY_DISABLE_NATIVE=None, ANYGEOMETRY_REQUIRE_NATIVE="1")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["backend"] == "native"


def _plane_cases():
    rng = np.random.default_rng(20261013)
    for scale in (1e-3, 1.0, 1e4):
        frame = rng.standard_normal(9) * scale
        uv = rng.uniform(-2.0, 3.0, size=(1000, 2))
        yield frame, uv
    # Adversarial: mixed magnitudes, signed zeros, exact cancellations.
    yield (np.array([1e16, -0.0, 3.0, 1.0, 1e-300, -1.0, -1.0, 2.0, 1e300]),
           np.array([[0.0, 0.0], [-0.0, 1.0], [1.0, -1.0], [1e-16, 1e-16], [2.5, 0.5]]))


@needs_native
@pytest.mark.parametrize("case", range(4))
def test_plane_evaluate_agrees_bit_for_bit_with_the_reference(case):
    frame, uv = list(_plane_cases())[case]
    expected, actual = np.empty((len(uv), 3)), np.empty((len(uv), 3))
    reference.plane_evaluate(frame, uv, expected)
    backend._CORE.plane_evaluate(frame, uv, actual)
    # Same operations in the same order with contraction disabled: bit identity.
    assert np.array_equal(expected.view(np.int64), actual.view(np.int64))


@needs_native
def test_plane_evaluate_rejects_inconsistent_or_mistyped_buffers():
    core = backend._CORE
    with pytest.raises(ValueError, match="inconsistent buffer sizes"):
        core.plane_evaluate(np.zeros(9), np.zeros((4, 2)), np.zeros((3, 3)))
    with pytest.raises(TypeError, match="uv must be a float64 buffer"):
        core.plane_evaluate(np.zeros(9), np.zeros((4, 2), dtype=np.float32), np.zeros((4, 3)))
    with pytest.raises((BufferError, ValueError)):
        core.plane_evaluate(np.zeros(9), np.zeros((4, 2)), np.zeros((4, 3))[:, :2].copy().T)
    readonly = np.zeros((4, 3))
    readonly.flags.writeable = False
    with pytest.raises((BufferError, TypeError, ValueError)):
        core.plane_evaluate(np.zeros(9), np.zeros((4, 2)), readonly)


def test_reference_plane_evaluate_matches_the_plane_surface():
    plane = anygeometry.Plane((1.0, 2.0, 3.0), (2.0, 0.0, 0.5), (0.0, 1.5, -1.0))
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.25, 0.75]])
    out = np.empty((3, 3))
    frame = np.concatenate((plane.origin, plane.u_vector, plane.v_vector))
    backend.plane_evaluate(frame, uv, out)
    for row, (u, v) in zip(out, uv):
        np.testing.assert_allclose(row, plane.evaluate(u, v), rtol=0, atol=1e-15)
