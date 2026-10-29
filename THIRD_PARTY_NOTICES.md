# Third-party notices

ANYgeometry does not bundle its Python dependencies. The following direct
dependencies are installed separately and remain subject to their upstream
license terms. Exact declared requirements and normalized license identifiers
are maintained in [dependency-licenses.json](dependency-licenses.json).

| Dependency | Declared requirement | Purpose | Upstream | License | Bundled |
|---|---|---|---|---|---|
| NumPy | `numpy>=1.26` | Runtime numerical arrays | https://numpy.org/ | BSD-3-Clause and permissive component licenses | No |
| Shapely | `shapely>=2.0` | Optional planar operations | https://shapely.readthedocs.io/ | BSD-3-Clause | No |
| SciPy | `scipy>=1.13` | Optional sketch operations | https://scipy.org/ | BSD-3-Clause; installed wheels carry separate bundled-component notices | No |
| mpmath | `mpmath>=1.3` | Optional test extra (CH3 reference values; never a runtime dependency) | https://mpmath.org/ | BSD-3-Clause | No |
| setuptools | `setuptools>=77.0.3` | Build backend | https://github.com/pypa/setuptools | MIT | No |
| wheel | `wheel` | Wheel build support | https://github.com/pypa/wheel | MIT | No |
| build | `build>=1.2` | Development/release build frontend | https://github.com/pypa/build | MIT | No |
| pytest | `pytest>=8` | Development/testing | https://pytest.org/ | MIT | No |
| Twine | `twine>=5` | Development/release metadata and upload client | https://twine.readthedocs.io/ | Apache-2.0 | No |

This notice is informational and does not replace the license files supplied
by the upstream projects.

SciPy is installed separately, not bundled by ANYgeometry. Its upstream project
license is BSD-3-Clause. The inspected SciPy 1.17.1 Windows distribution also
includes notices for OpenBLAS (BSD-3-Clause), LAPACK (BSD-3-Clause-Open-MPI), and
the GCC runtime library (GPL-3.0-or-later WITH GCC-exception-3.1). These component
notices remain part of that distribution; they do not change ANYgeometry's own
license or authorize accepting arbitrary GPL metadata. Consult the exact
installed SciPy distribution's license files for its component terms.

The installed-metadata checker recognizes only the reviewed SciPy 1.17.1
composite legacy `License` field with SHA256
`804e09cf4c9fbb3d8083685570891ccc5b20e9038623dff3e04bb3651d4dd18f`
after stripping surrounding whitespace and normalizing CRLF to LF. This binds
the complete component notices and GCC exception, rather than permitting GPL
by classifier or substring. Unknown versions/text and conflicting explicit SPDX
metadata still require review. This recognition does not approve arbitrary
wheels, redistribution, relinking, or release. Primary references are the
[SciPy 1.17.1 upstream BSD license](https://github.com/scipy/scipy/blob/v1.17.1/LICENSE.txt)
and the exact installed distribution's `scipy-1.17.1.dist-info/METADATA` and
`LICENSE.txt` files; the legacy metadata fingerprint above identifies the
reviewed Windows distribution text, including all bundled-component notices.
The supporting installed `LICENSE.txt` has a distinct LF/stripped SHA256,
`13e9f93803066ebace3a790961c122281a4ce50d3bf67669bd91d4ca99b75098`;
it is not substituted for the metadata fingerprint or accepted as another form.

Static inventory checks cover every declared extra. `--check-installed` keeps
requiring all direct non-build dependencies by default. An explicit profile,
for example `--check-installed --installed-extra dev --installed-extra planar`,
requires runtime/development dependencies and selected extras, while still
checking every declared optional dependency found in that environment.
Only missing unselected optional dependencies may be absent under a profile.
