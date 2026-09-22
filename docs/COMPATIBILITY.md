# Compatibility and verification status

The released baseline is ANYgeometry **0.4.3**, published on **2026-09-16**.
The maintenance changes under Unreleased do not create a new release. Geometry
schema 4 (read schemas 1–4), automation protocol 1 and MPL-2.0 remain unchanged.

## Released evidence

- [Immutable release ledger](release/anygeometry-0.4.3-ledger.json).
- [Published release](https://github.com/audunarn/ANYgeometry/releases/tag/v0.4.3).
- [Release-head CI](https://github.com/audunarn/ANYgeometry/actions/runs/35087208977).
- [Publication workflow](https://github.com/audunarn/ANYgeometry/actions/runs/35087360492).

Earlier design plans, candidate hashes and local checkpoints remain historical.
Their wording must not override the published artifact identity or imply that
new source changes were included in that artifact.

## Verification matrix

| Configuration | Python | Status |
| --- | --- | --- |
| Windows and Linux kernel | 3.11–3.14 | Released baseline CI passed; changed source requires fresh CI |
| macOS 15 Apple silicon (`macos-15`) kernel | 3.11–3.14 | Maintenance CI configured; hosted result required |
| macOS 15 Intel (`macos-15-intel`) kernel | 3.11–3.14 | Maintenance CI configured; hosted result required |
| Installed candidate wheel, all four platforms | 3.13 | Maintenance gate; exact artifact report required |
| NumPy 1.26.0 + Shapely 2.0.0, Linux | 3.11 | Minimum-dependency gate; hosted result required |
| Complete consumer set, Windows/Linux | 3.13 | Isolated baseline/candidate comparison required |
| Mesher and MCP consumers, macOS Apple silicon | 3.13 | Isolated baseline/candidate comparison required |

Configured jobs are not passing evidence. Tests on one operating system or
architecture do not qualify another. macOS Intel downstream native packages,
other macOS versions and desktop application packaging are not covered here.

## Dependencies and consumers

| Package | Contract / frozen consumer baseline |
| --- | --- |
| NumPy | Mandatory, >=1.26 |
| Shapely | Optional `planar` extra, >=2.0; required for strict planar workflows |
| ANYmesher | 0.5.0; consumes ANYgeometry[planar]>=0.4.3,<0.5 |
| ANYfem | 0.4.1; headless tests with ANYsolver 0.4.6 and ANYmaterial 0.2.0 |
| ANYfileio | 0.3.2; real semantic owners, no OCCT activation |
| ANYgeometry-mcp | Source 4253c001b07263fa2e9015082cf4950db65cd2d3, built wheel; MCP SDK >=2,<3 |

The MCP source pin is not a claim of a published adapter release. Reports record
resolved distributions, downloaded artifact hashes, installed module origins,
the candidate wheel identity and the released-geometry control run. Source-only
tests with an ambient SDK 1.x fallback do not verify the declared SDK 2.x install.

## Capability boundaries

- Standalone Cylinder patches support bounded orthogonal/notched boundaries and
  **at most one hole**. Internal multi-hole proof tests do not extend this API.
- Complete cylinders use the separate certified sector-atlas contract; a single
  periodic face is not admitted by the standalone patch query.
- General certified curved-region area is unavailable; unresolved overlaps fail
  closed. Planar fragmentation still requires straight boundaries and Shapely.
- Coordinate helpers use document-unit affine coordinates. `local_origin` and
  CRS metadata are not additional transformations; no CRS reprojection occurs.
- Coordinate helper availability does not activate a CAD export adapter.

See [owner handoffs](OWNER_HANDOFFS.md) for pending cross-repository decisions.

## Reproducing the installed-package checks

Build once with `python -m build`, then use the same candidate wheel for every
platform. The runner requires Python 3.13 for the declared installed-package
matrix; additional local interpreter runs are recorded separately.

```text
python tools/check_compatibility.py wheel --candidate <wheel> --output <new-external-directory>
python tools/check_compatibility.py consumers --candidate <wheel> --mcp-source <adapter-checkout> --profile full --output <another-new-external-directory>
```

Use `--profile macos` for the Apple silicon mesher/MCP subset, and
`--expected-machine arm64` or `--expected-machine x86_64` to assert the selected
Mac architecture. The MCP checkout must contain the frozen commit above; its
committed archive is built, never its potentially edited working files.

Outputs must be new directories outside source checkouts. They retain
`report.json`, complete command logs, dependency-wheel identities, per-lane
`install.json` and `probe.json`, and isolated environments. CI uploads the JSON
and command logs rather than entire virtual environments. No global packages
are installed and no artifacts or environments are removed automatically.

For offline reruns, `--wheelhouse <directory>` supplies dependency wheels and
`--baseline-wheel <wheel>` supplies the control artifact. Its SHA-256 must match
the published 0.4.3 ledger. The frozen MCP build requires setuptools and wheel
in the runner interpreter. Network/setup failures are failed evidence, never
silently interpreted as dependency absence or compatible consumer behavior.
