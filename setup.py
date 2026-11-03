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
import hashlib
import json
import shutil
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

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
        extensions = self.distribution.ext_modules or []
        if extensions and all(e.py_limited_api for e in extensions) and python.startswith("cp"):
            return "cp311", "abi3", platform
        return python, abi, platform


class ExcludeStaleExtensions(build_py):
    """OFF builds omit binaries left in a reused build directory as well."""
    def run(self):
        super().run()
        directory = Path(self.build_lib).resolve() / 'anygeometry'
        if directory == (Path('src') / 'anygeometry').resolve():
            raise RuntimeError('build output must not replace the source package')
        builder = self.get_finalized_command('build_ext')
        allowed = {Path(builder.get_ext_filename(ext.name)).name
                   for ext in self.distribution.ext_modules or []}
        for pattern in ('*.pyd', '*.so'):
            for path in directory.glob(pattern):
                if path.name not in allowed:
                    path.unlink()


def _validate_runtime_inspector_build():
    raw = Path('native/runtime_inspector_profile.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != '23754f3aa260f172b0f0344e86fc98b3ef63940a59129b8751764d3f13e9c579':
        raise RuntimeError('passive inspector profile differs from adopted identity')
    profile = json.loads(raw)
    if (sys.platform != 'win32' or sys.maxsize != 2**63-1 or sys.version != profile['runtime_version']
            or sysconfig.get_config_var('Py_GIL_DISABLED') or sysconfig.get_config_var('Py_DEBUG')):
        raise RuntimeError('passive inspector build runtime is outside the exact profile')
    if any(os.environ.get(name, '').strip() for name in ('CL', '_CL_', 'CFLAGS', 'CPPFLAGS', 'LDFLAGS')):
        raise RuntimeError('passive inspector does not accept compiler environment overrides')
    include = Path(sysconfig.get_path('include')).resolve()
    expected = profile['header_files']
    actual = {path.relative_to(include).as_posix(): path for path in include.rglob('*.h')}
    if len(expected) != 283 or set(actual) != set(expected):
        raise RuntimeError('passive inspector header set differs from exact profile')
    if any(hashlib.sha256(actual[name].read_bytes()).hexdigest() != digest for name, digest in expected.items()):
        raise RuntimeError('passive inspector header contents differ from exact profile')
    # Resolve the image containing the actual CPython entry, not a guessed DLL path.
    import ctypes
    from ctypes import wintypes as W
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    containing = kernel.GetModuleHandleExW
    containing.argtypes = [W.DWORD, ctypes.c_void_p, ctypes.POINTER(W.HMODULE)]
    containing.restype = W.BOOL
    filename = kernel.GetModuleFileNameW
    filename.argtypes = [W.HMODULE, W.LPWSTR, W.DWORD]
    filename.restype = W.DWORD
    handle = W.HMODULE()
    if not containing(0x6, ctypes.cast(ctypes.pythonapi.Py_GetVersion, ctypes.c_void_p), ctypes.byref(handle)):
        raise RuntimeError('cannot locate build runtime image')
    buffer = ctypes.create_unicode_buffer(32768)
    size = filename(handle, buffer, len(buffer))
    if not 0 < size < len(buffer):
        raise RuntimeError('cannot resolve build runtime image path')
    image = Path(buffer.value).read_bytes()
    if len(image) != 6760792 or hashlib.sha256(image).hexdigest() != '7ffe3ab11342db03cc18b026f27485c0f74f5bf10f239f1f6573394900596394':
        raise RuntimeError('passive inspector build runtime image differs')
    return profile


class ProfiledBuildExt(build_ext):
    def build_extension(self, ext):
        if ext.name == 'anygeometry._planning_runtime_inspector':
            _validate_runtime_inspector_build()
            include = Path(sysconfig.get_path('include')).resolve()
            # /I precedes INCLUDE. Put the validated complete tree first and
            # reject local quoted-header shadows before invoking the compiler.
            ext.include_dirs = [str(include), str(Path('native/include').resolve())]
            if (Path('native')/'Python.h').exists() or (Path('native')/'internal').exists():
                raise RuntimeError('local inspector Python header shadow')
            if self.define or self.undef or self.debug:
                raise RuntimeError('inspector does not accept build_ext macro overrides')
            if ('AG_INSPECTOR_HEADERS_VALIDATED', '1') not in ext.define_macros:
                ext.define_macros.append(('AG_INSPECTOR_HEADERS_VALIDATED', '1'))
        super().build_extension(ext)


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

build_planning = _flag("ANYGEOMETRY_BUILD_PLANNING_NATIVE") or _flag("ANYGEOMETRY_REQUIRE_PLANNING_NATIVE")
if build_planning and not _flag("ANYGEOMETRY_DISABLE_PLANNING_NATIVE"):
    if sys.platform != "win32":
        if _flag("ANYGEOMETRY_REQUIRE_PLANNING_NATIVE"):
            raise RuntimeError("the planning resource helper requires Windows")
    else:
        _ensure_msvc_tools_on_path()
        extensions.append(Extension(
            "anygeometry._planning_native_core", ["native/_planning_native_core.cpp"],
            language="c++", define_macros=[("Py_LIMITED_API", LIMITED_API)],
            py_limited_api=True, extra_compile_args=["/std:c++17", "/O2", "/EHsc"],
            extra_link_args=["/MANIFEST:NO"],
            optional=not _flag("ANYGEOMETRY_REQUIRE_PLANNING_NATIVE"),
        ))

build_inspector = _flag('ANYGEOMETRY_BUILD_RUNTIME_INSPECTOR') or _flag('ANYGEOMETRY_REQUIRE_RUNTIME_INSPECTOR')
if build_inspector and not _flag('ANYGEOMETRY_DISABLE_RUNTIME_INSPECTOR'):
    if sys.platform != 'win32':
        if _flag('ANYGEOMETRY_REQUIRE_RUNTIME_INSPECTOR'):
            raise RuntimeError('passive runtime inspection requires its exact Windows profile')
    else:
        _ensure_msvc_tools_on_path()
        extensions.append(Extension(
            'anygeometry._planning_runtime_inspector', ['native/_planning_runtime_inspector.cpp'],
            depends=['native/include/runtime_inspector_profile.hpp','native/runtime_inspector_profile.json'],
            include_dirs=['native/include'], language='c++',
            define_macros=[('Py_BUILD_CORE_MODULE','1')], py_limited_api=False,
            libraries=['bcrypt'], extra_compile_args=['/std:c++17','/O2','/EHsc','/Zp8','/showIncludes'],
            extra_link_args=['/MANIFEST:NO'], optional=not _flag('ANYGEOMETRY_REQUIRE_RUNTIME_INSPECTOR'),
        ))

setup(ext_modules=extensions, cmdclass={"bdist_wheel": AbiThreeWheel, 'build_ext': ProfiledBuildExt,
                                      'build_py': ExcludeStaleExtensions})
