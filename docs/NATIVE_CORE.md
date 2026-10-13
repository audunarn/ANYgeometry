# Optional compiled core

ANYgeometry can carry an optional compiled module, `anygeometry._core`, written
in C++17 against the CPython stable ABI. It holds batch kernels only; every
kernel has a pure-Python reference with the same signature in
`anygeometry.reference`. Status: scaffold (one kernel, `plane_evaluate`); the
production kernels follow in the generality programme
(`docs/agent_plans/GENERALITY_PROGRAMME_PLAN.md`, slices N1-N4).

## Backend selection and reporting

The backend is chosen once, when `anygeometry` is imported:

| Situation | Backend | `backend_info()["reason"]` |
|---|---|---|
| `ANYGEOMETRY_DISABLE_NATIVE=1` | python | `disabled by ANYGEOMETRY_DISABLE_NATIVE` |
| compiled module not installed | python | `compiled module not installed` |
| compiled module installed and importable | native | `compiled module loaded` |
| compiled module installed but not importable | – | the import of `anygeometry` fails |
| `ANYGEOMETRY_REQUIRE_NATIVE=1` and no native backend | – | the import of `anygeometry` fails |

`anygeometry.backend_info()` returns `{"backend", "reason", "module", "build"}`;
`build` describes the compiler and ABI of the loaded module. Documents and
results never record the backend. Kernels and references are compared by the
differential tests in `tests/test_native_backend.py` (bit identity where the
kernel performs the reference's operations in the same order; otherwise the
bound of owner decision D1).

## Building

Sources: `native/_core.cpp` and `native/include/*.hpp`, outside the package so
they never enter a wheel.

```powershell
$env:ANYGEOMETRY_BUILD_NATIVE = "1"          # opt in while the core is a scaffold
python -m pip wheel --no-deps -w dist .      # -> anygeometry-<v>-cp311-abi3-<platform>.whl
python setup.py build_ext --inplace          # development build next to the sources
```

Without `ANYGEOMETRY_BUILD_NATIVE=1` the build is pure Python
(`py3-none-any`), exactly as before. `ANYGEOMETRY_REQUIRE_NATIVE=1` builds the
extension and makes a failed compile an error; without it a failed compile
yields a pure-Python package.

Compiler settings: MSVC `/std:c++17 /O2 /fp:precise /fp:contract-`;
GCC/Clang `-std=c++17 -O2 -ffp-contract=off -fno-fast-math`. A fast-math build
is refused at compile time.

One `cp311-abi3` wheel per platform serves CPython 3.11 and newer
(`Py_LIMITED_API=0x030B0000`; the buffer protocol is in the limited API from
3.11). Free-threaded interpreters cannot load abi3 modules and use the Python
references.

## Verification

- `tests/test_native_backend.py`: backend reporting, the environment switches
  in fresh interpreters, differential agreement, buffer validation.
- `tools/native_wheel_smoke.py --require-native | --expect-python`: run against
  an installed distribution outside the source tree.
- `.github/workflows/native-wheels.yml` (manual): abi3 wheels on Windows,
  Linux (manylinux and musllinux x86_64) and macOS (arm64 and x86_64), each
  wheel installed on Python 3.11-3.14, and a pure-Python sdist install. It
  does not publish; the release workflow is unchanged.

## Kernel input contract

Kernels assume validated input and check their own buffers (float64,
C-contiguous, aligned, consistent sizes, parameters in the domain); offsets use
`Py_ssize_t`. The validating entry points in `anygeometry.backend` copy to
aligned storage and establish the mathematical contract first. For B-spline
curves (`bspline_curve`): degree 1-15, clamped knots (end multiplicity exactly
degree + 1, interior multiplicity at most the degree, non-empty spans longer
than 64 eps of the knot scale and than 2**-968), finite controls with nonzero
magnitudes in [2**-500, 2**500], positive finite weights with ratio at most 1e6
(owner decision) scaled by an exact power of two. A non-finite result is refused
with `ValueError`, never returned. Accuracy contract, tested against exact
rational evaluation: the k-th derivative is within
`1e3 * eps * R * C * (degree / h)**k` of the exact value (`C` largest control
magnitude, `R` weight ratio, `h` shortest span); errors are absolute relative
to the curve scale.

Open (architecture section 5.4): import-time probes of the effective
floating-point contraction and rounding mode are not implemented yet; the
compiler flags are set in `setup.py` only.
