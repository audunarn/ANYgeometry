# Maintenance owner handoffs

Implementation is confined to ANYgeometry. These are reviewable follow-up
requirements for the other owners, not edits or messages sent to those owners.

## ANYopenSoft

The geometry/version entries in `governance/ECOSYSTEM_LEDGER.md` still describe
ANYgeometry 0.2.1 and older mesher constraints. Add a new current-status entry
pointing to the [0.4.3 release ledger](release/anygeometry-0.4.3-ledger.json),
[release-head CI](https://github.com/audunarn/ANYgeometry/actions/runs/35087208977)
and [compatibility record](COMPATIBILITY.md). Preserve historical decisions and
their evidence rather than rewriting them as fresh qualifications.

The new entry should distinguish published 0.4.3 from maintenance source,
configured macOS gates from executed results, and the public one-hole patch
limit from internal material-proof coverage. Use the exact maintenance commit
and CI/artifact identities when they become available; a package version alone
does not identify the changed source.

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

The adapter owner must review and version its own contract, pin the accepted
geometry source/artifact, and test translation, rotation, nonzero origin,
non-default units, forward/inverse behavior and unsupported CRS requests.
Perform model-unit to CAD-unit conversion exactly once after the owner point
mapping. Leave external export blocked until that adapter work is accepted.

## Consumer gate failures

Each isolated run produces candidate and released-0.4.3 reports. A failure in
both is an existing consumer/environment issue; a candidate-only failure is a
geometry regression candidate. Neither classification turns a failed gate green.
Record the consumer version, artifact hashes, command, error and both reports
in the handoff. Do not change consumer source or silently skip a required cell.
