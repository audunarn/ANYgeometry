# Compatibility and verification status

The current release is ANYgeometry **0.4.5**, dated **2026-10-01**, with its
[immutable release ledger](release/anygeometry-0.4.5-ledger.json). It writes
schema 5 and reads schemas 1–5. Version 0.4.4 retains its schema-4 scope and
[ledger](release/anygeometry-0.4.4-ledger.json).

The [general-intersection contract](general_intersections.md) passed a new
32-job source gate and 48 installed native meshes. The exact 0.4.5 release
wheel was reviewed and tested outside checkouts, and its published PyPI hashes
match the ledger. Automation protocol 1 and MPL-2.0 remain unchanged. Use the
[current compatibility table and consumer handoff](general_intersections_release_handoff.md)
for exact inputs and limits. Consumer versions remain development identities;
the broader FEM scientific gate and Qt acceptance remain unaccepted.

## Historical 0.4.3 release evidence

- [Immutable release ledger](release/anygeometry-0.4.3-ledger.json).
- [Published release](https://github.com/audunarn/ANYgeometry/releases/tag/v0.4.3).
- [Release-head CI](https://github.com/audunarn/ANYgeometry/actions/runs/35087208977).
- [Publication workflow](https://github.com/audunarn/ANYgeometry/actions/runs/35087360492).

Earlier design plans, candidate hashes and local checkpoints remain historical.
Their wording must not override the published artifact identity or imply that
new source changes were included in that artifact.

## Historical maintenance verification matrix

The maintenance implementation at `14e360f5dde8b4125a674df8593a3e4a6d3429bd`
passed [all 29 hosted jobs](https://github.com/audunarn/ANYgeometry/actions/runs/35708515183)
on 2026-09-22. These results identify the development candidate, not the
published 0.4.3 wheel. See the [completed maintenance record](MAINTENANCE_20260922.md)
and [per-platform dependency/artifact evidence](verification/maintenance-20260922.json).

| Configuration | Python | Status |
| --- | --- | --- |
| Windows and Linux kernel | 3.11–3.14 | Passed: 953 tests per configuration |
| macOS 15 Apple silicon (`macos-15`) kernel | 3.11–3.14 | Passed: 953 tests per configuration; arm64 asserted |
| macOS 15 Intel (`macos-15-intel`) kernel | 3.11–3.14 | Passed: 953 tests per configuration; x86_64 asserted |
| Installed candidate wheel, all four platforms | 3.13 | Passed with and without Shapely; same wheel verified |
| NumPy 1.26.0 + Shapely 2.0.0, Linux | 3.11 | Passed: 953 tests; 11 Shapely intersection warnings |
| Complete consumer set, Windows/Linux | 3.13 | Candidate and released control passed public and MCP gates |
| Mesher and MCP consumers, macOS Apple silicon | 3.13 | Candidate and released control passed both gates |

The results above apply to the recorded source and artifacts. Changed inputs
require new evidence. Tests on one operating system or architecture do not
qualify another. macOS Intel downstream native packages, other macOS versions
and desktop application packaging are not covered here. No numerical acceptance
tolerances were loosened for these results.

Public consumer and private MCP jobs are independent required gates. The
private `audunarn/ANYgeometry-mcp` checkout needs the repository secret
`ANYGEOMETRY_MCP_READ_TOKEN`, scoped to Contents: read on that repository only.
Without it, MCP CI reports a blocked prerequisite and fails explicitly; the
public consumer checks still run. A successful public subset is not acceptance
of the complete consumer set. Credentials are neither created nor copied by
these workflows. Configure the secret only for trusted CI access to that source.

The scoped secret was supplied for the recorded run. Attempts 1 and 2 failed
before MCP consumer execution (absent secret, then checkout authentication);
attempt 3 passed on all three platforms using MCP SDK 2.2.0. These setup failures
remain recorded and were not treated as successful consumer coverage.

## Maintenance dependencies and frozen consumer baselines

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

Use `--profile macos` for the Apple silicon mesher/MCP subset. Hosted CI splits
this into `--profile mesher` and `--profile mcp`; Windows/Linux split the full set
into `--profile public` and `--profile mcp`. The public/mesher profiles require no
MCP checkout. Use
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
