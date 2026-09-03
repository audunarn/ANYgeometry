# Third-party notices

ANYgeometry does not bundle its Python dependencies. The following direct
dependencies are installed separately and remain subject to their upstream
license terms. Exact declared requirements and normalized license identifiers
are maintained in [dependency-licenses.json](dependency-licenses.json).

| Dependency | Declared requirement | Purpose | Upstream | License | Bundled |
|---|---|---|---|---|---|
| NumPy | `numpy>=1.26` | Runtime numerical arrays | https://numpy.org/ | BSD-3-Clause and permissive component licenses | No |
| Shapely | `shapely>=2.0` | Optional planar operations | https://shapely.readthedocs.io/ | BSD-3-Clause | No |
| setuptools | `setuptools>=77.0.3` | Build backend | https://github.com/pypa/setuptools | MIT | No |
| wheel | `wheel` | Wheel build support | https://github.com/pypa/wheel | MIT | No |
| build | `build>=1.2` | Development/release build frontend | https://github.com/pypa/build | MIT | No |
| pytest | `pytest>=8` | Development/testing | https://pytest.org/ | MIT | No |
| Twine | `twine>=5` | Development/release metadata and upload client | https://twine.readthedocs.io/ | Apache-2.0 | No |

This notice is informational and does not replace the license files supplied
by the upstream projects.
