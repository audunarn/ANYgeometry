"""Selection of the compiled or pure-Python kernel backend.

ANYgeometry may carry an optional compiled module, ``anygeometry._core``
(C++17, CPython stable ABI). Every compiled kernel has a pure-Python reference
with the same signature in :mod:`anygeometry.reference`. The backend is chosen
once, when this module is imported, and never changes within a process:

* ``ANYGEOMETRY_DISABLE_NATIVE=1`` selects the Python references;
* otherwise the compiled module is used when it is installed. If it is
  installed but cannot be imported, the import of ANYgeometry fails: a broken
  build is an error, not a reason to fall back silently;
* ``ANYGEOMETRY_REQUIRE_NATIVE=1`` makes the import of ANYgeometry fail when the
  compiled module is not in use.

:func:`backend_info` reports the choice and its reason. Documents and results
never record the backend.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
from types import MappingProxyType

from . import reference

__all__ = ["backend_info", "plane_evaluate"]

_TRUE = {"1", "true", "TRUE", "yes", "YES"}


def _flag(name: str) -> bool:
    return os.environ.get(name, "0").strip() in _TRUE


def _load():
    if _flag("ANYGEOMETRY_DISABLE_NATIVE"):
        return None, "disabled by ANYGEOMETRY_DISABLE_NATIVE"
    spec = importlib.util.find_spec(f"{__package__}._core")
    if spec is None or spec.origin is None:
        return None, "compiled module not installed"
    # Present but unloadable is an error, never a quiet fallback.
    return importlib.import_module(f"{__package__}._core"), "compiled module loaded"


_CORE, _REASON = _load()
if _CORE is None and _flag("ANYGEOMETRY_REQUIRE_NATIVE"):
    raise ImportError(f"ANYGEOMETRY_REQUIRE_NATIVE is set but the compiled core is not in use: {_REASON}")

_BUILD = MappingProxyType(dict(_CORE.build_info())) if _CORE is not None else None
_KERNELS = _CORE if _CORE is not None else reference

plane_evaluate = _KERNELS.plane_evaluate


def backend_info() -> dict:
    """The kernel backend of this process: ``{"backend", "reason", "module", "build"}``."""
    return {
        "backend": "native" if _CORE is not None else "python",
        "reason": _REASON,
        "module": getattr(_CORE, "__file__", None),
        "build": dict(_BUILD) if _BUILD is not None else None,
    }
