# Maintenance owner handoffs

Implementation is confined to ANYgeometry. These are reviewable follow-up
requirements for the other owners, not edits or messages sent to those owners.

## ANYopenSoft

The following entries in `governance/ECOSYSTEM_LEDGER.md` need current-status
supplements. Line numbers identify the file inspected on 2026-09-22; use the
section/text anchors if subsequent owner edits move them.

| Existing entry | Replacement evidence and boundary |
| --- | --- |
| "Current public contracts", lines 161–162: ANYgeometry 0.2.1 and live API >=0.2,<0.3 | Published 0.4.3, released 2026-09-16, retains schemas 1–4 reading and canonical schema 4 writing. The current mesher-facing range is ANYgeometry[planar]>=0.4.3,<0.5. Use the existing release ledger and release-head CI below. |
| "Material decisions", lines 214–221: no public directional composition helper | Maintenance commit 14e360f5dde8b4125a674df8593a3e4a6d3429bd supplies the four owner helpers and regressions. Record the development artifact separately from published 0.4.3. CAD contract adoption and capture/read-lease requirements still belong to the adapter owner. |
| "Material decisions", following CAD capability/package-resolution paragraph | The installed ANYmesher 0.5.0, ANYmaterial 0.2.0 and ANYfileio 0.3.2 semantic probe is current consumer evidence. It does not widen historical CAD qualification or authorize native export. Preserve that distinction when the file-I/O owner updates its ranges. |

Reference the [0.4.3 release ledger](release/anygeometry-0.4.3-ledger.json),
[release-head CI](https://github.com/audunarn/ANYgeometry/actions/runs/35087208977),
[completed maintenance record](MAINTENANCE_20260922.md) and
[compatibility table](COMPATIBILITY.md). Maintenance CI passed all 29 jobs,
including both Mac architectures and hosted MCP, with exact identities in
[the evidence summary](verification/maintenance-20260922.json). Preserve
historical decisions rather than rewriting them as new qualifications. Retain
the public one-hole patch limit; internal proofs do not expand it.

## ANYfileIO / ANYfileio-occt

`ANYfileIO/docs/ANYGEOMETRY_0_2_ADAPTER.md` section 9.2 blocks external-frame
export until the geometry owner defines a directional composition. The
[coordinate contract](COORDINATES.md) now defines it, with regressions in
`tests/test_coordinates.py` and automation contract tests.

For a model-unit point p, world coordinates are the homogeneous application of
the stored coordinate_transform T; the inverse uses T^-1. Missing T is identity.
Do not add or subtract local_origin, and do not infer a CRS transformation.
Vectors use only the linear block. These helpers are not surface-normal
transforms and do not authorize affine deformation of CAD primitives.

For an explicit regression example, use millimetres, T with identity linear
block and translation (10000, 20000, 30000), and nonzero local_origin
(500, 600, 700). The model point (1000, 0, 0) maps to world (11000, 20000, 30000),
the inverse restores it, and vector (1, 2, 3) is unchanged. Converting the world
point to metres afterward gives (11, 20, 30). The origin is never applied again.
See the executable example in [COORDINATES.md](COORDINATES.md).

The 61 coordinate regressions in `tests/test_coordinates.py`, together with
existing automation tests, passed in all 16 kernel configurations. They cover
rotation/reflection/scale/shear, batches, invalid inputs, non-mutation,
nonzero origin and metre/millimetre automation. Adopt the exact source and
candidate hash in [the completed record](MAINTENANCE_20260922.md), not a
version-only dependency on the already published 0.4.3 wheel.

The adapter owner must review and version its own contract, pin the accepted
geometry source/artifact, and test translation, rotation, nonzero origin,
non-default units, forward/inverse behavior and unsupported CRS requests.
Perform model-unit to CAD-unit conversion exactly once after the owner point
mapping. Leave external export blocked until that adapter work is accepted.
Existing owner read-lease and transaction-visible capture requirements are
unchanged by these coordinate queries.

## Consumer gate failures

Hosted MCP access requires an owner-configured `ANYGEOMETRY_MCP_READ_TOKEN`
repository secret in ANYgeometry with only Contents: read access to the private
ANYgeometry-mcp repository. The default GitHub job token cannot read that
repository. Do not copy the private adapter source into the public repository
or relax its visibility to work around this boundary. Access was supplied and
all three hosted MCP candidate/control comparisons passed with SDK 2.2.0.
The earlier missing-secret and authentication failures remain recorded setup
failures. There are no unresolved consumer-owned failures from this run.
Future missing access must still fail the separate MCP gates explicitly;
public-subset success is not complete consumer coverage.

Each isolated run produces candidate and released-0.4.3 reports. A failure in
both is an existing consumer/environment issue; a candidate-only failure is a
geometry regression candidate. Neither classification turns a failed gate green.
Record the consumer version, artifact hashes, command, error and both reports
in the handoff. Do not change consumer source or silently skip a required cell.
