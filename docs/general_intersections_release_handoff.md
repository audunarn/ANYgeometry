# General intersections: release and consumer handoff

ANYgeometry 0.4.5 publishes the batch API required by ANYfem. The earlier
generated-plate suppression defect is fixed, as independently confirmed by
ANYfem. These exact inputs supersede geometry main `7e797791…`; matching
consumer version numbers alone do not identify the coordinated artifacts.

| Identity | Exact commit |
| --- | --- |
| Geometry implementation main / artifact source | `7ec4d9f5678e35366694d5d5bf9ff9ad95fcbaa8` |
| Geometry qualified review head | `dbd784d81f743c60cdfccae02b93ecd38e22767c` |
| Geometry CI synthetic merge used for candidate wheel | `d785dc4f6efa2a555942dfa39ff35415f3adeda9` |
| Sole-child release ledger / tag `v0.4.5` | `254cdc33ec1ae4b07c57b62791eb461103e7ff90` |
| Geometry release-ledger main merge | `51843d14998993e43ffcfc57a4d2672cda452989` |
| Qualified installed mesher source | `e21c0fc93662776762430e14450d54ac9192e2e8` |
| Mesher review integration | `edfaa28d86251c360ee87de8aa257cd576357cec` |
| Qualified installed FEM source | `83f3d6c81e405ca45f5b7edd7cb7c2afd32804c4` |
| FEM review integration, unmerged | `f9a5e53d89d754004b2d8ebd88b833fd9d5f2a7f` |
| Frozen solver in broader FEM CI | `d04199ac851c0d0f61430c2bc40136582aa8d659` |

Geometry review and implementation merge have identical trees. Candidate and
release archives have different container hashes but all 71 wheel members and
180 sdist members are byte-identical. Mesher and FEM integration commits change
dependency CI metadata only; their runtime source trees equal the qualified
installed inputs. The [delivery record](../reports/release_045/delivery.json)
records subsequent main merges and current owner CI dispositions.

| Geometry artifact | SHA256 |
| --- | --- |
| CI candidate wheel, 0.4.5 | `ad160bd91519ea6bd54a2308a4aa3af4a354b5c393951d9d387eea5c987a6c06` |
| Released wheel, 0.4.5 | `b68066aa44052aeeaf5d3feb8d6bc733ce81bf04752580aebc3c1b7915ce8fad` |
| Released sdist, 0.4.5 | `3995ff69850d239a8c266874ad6453fb1b6ed3d5c9ff5b75e84a0f0e05384a1c` |
| Previous qualified development wheel, 0.4.4 | `2f154358ec2b873db48994d14be1fd4659411a128751407772d60becb667c5f6` |

Every native mesher/FEM candidate filename, size and hash is in the
[installed qualification](../reports/general_intersections/installed-qualification-36834513207.json).
Do not substitute a published consumer solely because its version matches.
Neither consumer is published by this geometry release.

## Accepted scope

[Run 36834513207](https://github.com/audunarn/ANYgeometry/actions/runs/36834513207)
passes 32/32 jobs and accepts 48 installed native meshes. The source gate and
exact release-bundle checks are bound in the
[qualification](../reports/release_045/qualification.json),
[separate artifact and analytic review](../reports/release_045/independent_review.json)
and [release ledger](release/anygeometry-0.4.5-ledger.json).

| Configuration or check | Result |
| --- | --- |
| Windows, Linux, macOS 15 Apple silicon, macOS 15 Intel; Python 3.11–3.14 kernel | All 16 combinations pass |
| Installed universal geometry wheel, Python 3.13; all four platforms | Pass; origins, CLI, schema, typing, serialization and optional Shapely boundaries |
| Linux/Python 3.11 minimum dependencies | Pass; NumPy 1.26.0, Shapely 2.0.0 |
| Public consumer baselines and SDK-2 MCP candidate | Applicable Windows/Linux/Apple-silicon gates pass |
| Installed general native consumers; Windows/Linux/Apple silicon, Python 3.13 | 16 accepted meshes per platform |
| Supplied exact replay at target sizes 0.5 and 0.25 | Accepted; undo/replacement, material, references and lifecycle checks pass |
| New Windows skew-cylinder mesh | 632.484 seconds, unchanged 900-second budget; four independently checked branches |
| 260-sector cylinder on all three native platforms | Accepted; area 2π, all 260 shared panel interfaces, source preservation and save/load |
| Exact release wheel outside source checkouts | Pass with Shapely present/absent; installed batch, serialization and idempotence checks pass |

The historical 256-FaceUse direct API retains its original qualification.
Supported primary meshing selects general charts before that API for larger
components and trim-occurrence counts. The installed 260-sector probe qualifies
this route; it does not extend the old direct certificate.

The earlier [31/32 Windows timeout](../reports/general_intersections/installed-qualification-36825533614.json)
is preserved. The fix removes repeated full ownership/chart work and adds
rejection prechecks to repair trials. Improving candidates and final meshes
still pass complete topology, physical-quality and solver-admission checks.
Tolerances and budgets are unchanged. The user authorized the existing
15-degree default admission floor; stricter caller policies remain enforced
and direct automatic recovery remains opt-in.

## Remaining consumer work

[ANYfem PR10](https://github.com/audunarn/ANYfem/pull/10) is deliberately unmerged.
Its broader scientific [run 36834324382](https://github.com/audunarn/ANYfem/actions/runs/36834324382)
has a completed Linux/Python 3.13 cell with 1,041 passed, 130 explicit optional/GUI
skips and three capacity-workflow setup errors. They are
`PlaneStressConvergenceError` under frozen solver `d04199…`, at point 2837 after
30 iterations, reported maximum scaled yield residual 1e-10. Complete failure
logs and baseline failures are retained locally, with hashes in the delivery
record. Other platform results are reported individually.

The existing FEM main baseline failed earlier on missing batch APIs, so it
cannot establish whether these newly reached scientific errors were introduced
by this integration. No solver pin, tolerance or acceptance threshold changed.
ANYfem should pin the coordinated inputs, resolve/adjudicate that gate, then
rerun Qt parity and scientific acceptance. This does not authorize a Qt default
switch, Tk removal or an ANYfem release.

[ANYmesher PR9](https://github.com/audunarn/ANYmesh/pull/9) has a broader owner
matrix. Its final result and any main merge are recorded in `delivery.json`;
geometry publication accepts the installed 48-mesh scope without inferring a
pending owner matrix passed.

Other consumers must decode schema 5 and both new exact curve definitions
before reading newly saved documents. Schemas 1–4 remain readable by this
release. Intel macOS native consumers remain unqualified. Positive-area overlap
needs an explicit ownership policy. Ambiguous feature or attachment remaps
refuse atomically. CAD export remains outside this work.
