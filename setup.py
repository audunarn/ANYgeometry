"""Build the optional compiled core ``anygeometry._core`` (C++17, CPython stable ABI).

The compiled core is optional: every kernel has a pure-Python reference and
``anygeometry.backend_info()`` reports which one is in use.

Environment switches at build time:

* ``ANYGEOMETRY_BUILD_NATIVE=1`` builds the extension. Until the core carries
  production kernels the default is a pure-Python build, so installs and the
  existing ``py3-none-any`` release artifacts are unchanged.
* ``ANYGEOMETRY_REQUIRE_NATIVE=1`` (with the above) turns a failed compile into
  a build error instead of a pure-Python package.

One ``cp311-abi3`` wheel per platform serves CPython 3.11 and newer.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from setuptools import Extension, setup

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:  # setuptools < 70.1
    from wheel.bdist_wheel import bdist_wheel

_TRUE = {"1", "true", "TRUE", "yes", "YES"}
LIMITED_API = "0x030B0000"


def _flag(name: str) -> bool:
    return os.environ.get(name, "0").strip() in _TRUE


def _ensure_msvc_tools_on_path() -> None:
    """Repair setuptools environments that discover MSVC headers but not tools (as ANYmesher)."""

    if sys.platform != "win32" or shutil.which("cl.exe") is not None:
        return
    target = "x64" if sys.maxsize > 2**32 else "x86"
    candidates: list[Path] = []
    for variable in ("ProgramFiles(x86)", "ProgramFiles"):
        root = os.environ.get(variable)
        if root:
            for host in ("HostX64", "HostX86"):
                candidates.extend((Path(root) / "Microsoft Visual Studio").glob(
                    f"*/*/VC/Tools/MSVC/*/bin/{host}/{target}/cl.exe"))
    if candidates:
        def version_key(compiler: Path) -> tuple[int, ...]:
            try:
                return tuple(int(part) for part in compiler.parents[3].name.split("."))
            except ValueError:
                return (0,)
        compiler = max(candidates, key=lambda item: (version_key(item), str(item)))
        os.environ["PATH"] = str(compiler.parent) + os.pathsep + os.environ.get("PATH", "")


class AbiThreeWheel(bdist_wheel):
    """Tag a wheel holding the compiled core as cp311-abi3."""

    def get_tag(self):
        python, abi, platform = super().get_tag()
        if self.distribution.has_ext_modules() and python.startswith("cp"):
            return "cp311", "abi3", platform
        return python, abi, platform


extensions = []
build_native = _flag("ANYGEOMETRY_BUILD_NATIVE") or _flag("ANYGEOMETRY_REQUIRE_NATIVE")
if build_native and not _flag("ANYGEOMETRY_DISABLE_NATIVE"):
    _ensure_msvc_tools_on_path()
    if sys.platform == "win32":
        # Precise IEEE semantics and no contraction into fused multiply-adds.
        compile_args = ["/std:c++17", "/O2", "/fp:precise", "/fp:contract-", "/EHsc"]
        link_args = ["/MANIFEST:NO"]
    else:
        compile_args = ["-std=c++17", "-O2", "-ffp-contract=off", "-fno-fast-math", "-fvisibility=hidden"]
        link_args = []
    extensions.append(Extension(
        "anygeometry._core",
        ["native/_core.cpp"],
        depends=[str(path) for path in sorted(Path("native/include").glob("*.hpp"))],
        include_dirs=["native/include"],
        language="c++",
        define_macros=[("Py_LIMITED_API", LIMITED_API)],
        py_limited_api=True,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        optional=not _flag("ANYGEOMETRY_REQUIRE_NATIVE"),
    ))

setup(ext_modules=extensions, cmdclass={"bdist_wheel": AbiThreeWheel})
