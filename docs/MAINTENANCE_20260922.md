# Maintenance implementation completion — 2026-09-22

The maintenance, consumer compatibility and macOS plan is implemented and its
requested automated gates passed. The tested implementation is commit
`14e360f5dde8b4125a674df8593a3e4a6d3429bd`; the closeout documentation records
that immutable result. It does not relabel a later build as the tested artifact.

## Delivery against the plan

| Work | Delivered behavior and evidence |
| --- | --- |
| Documentation and release status | 0.4.3 is released on 2026-09-16; new work is Unreleased. Mesher range is >=0.4.3,<0.5. Orthogonal/notched cylinder patches retain the public one-hole limit. Historical qualification and release ledgers remain unchanged. |
| Installed consumers | Candidate/control checks run outside checkouts, reject editable/source imports, verify installed wheel bytes and retain dependency/artifact identities. Plate/cylinder edits, meshing, feature references and ownership, FEM round-trip/rollback, real file-I/O semantic owners and MCP boundaries passed. |
| Platform and dependency coverage | All 16 Windows/Linux/macOS 15 arm64/macOS 15 Intel kernel cells passed 953 tests each on Python 3.11–3.14. The single wheel passed all four Python 3.13 smoke cells, with/without Shapely. Linux/Python 3.11 minimum dependencies passed. |
| Coordinate contract | Four root-exported point/vector helpers share automation conversion, preserve shape and model state, and keep document units, homogeneous point semantics, linear vectors, origin metadata and CRS behavior explicit. 61 coordinate regressions cover the planned cases. |

Version 0.4.3, schema 4 and automation protocol 1 are unchanged. The newly added
helpers are in the development candidate; installing published 0.4.3 does not
provide them. No numerical acceptance tolerances were loosened.

## Immutable evidence

- [Hosted run 35708515183](https://github.com/audunarn/ANYgeometry/actions/runs/35708515183):
  all 29 jobs green, final attempt 3. Only the three MCP jobs needed reruns.
- [Machine-readable evidence](verification/maintenance-20260922.json): source
  and Git object identities, all 29 job identities, exact test counts, artifact
  IDs/digests, resolved dependency versions and wheel hashes, import origins,
  installed-file checks, and per-case results for every candidate/control lane.
- Candidate wheel SHA-256:
  `4de647b2f7468b76609a914018db977850d1459de8afbe02108f7fab969f92ad`.
- Published 0.4.3 control wheel SHA-256:
  `6ab8f398de8ad1cc19b5c0d70a9787e1a15d3f5253c1125134eeb5050ec4b8b5`.

All ten successful installed-package reports used that same candidate wheel.
The pinned MCP source was `4253c001b07263fa2e9015082cf4950db65cd2d3`; SDK 2.2.0
resolved on Windows, Linux and Apple silicon. Both candidate and released
control passed on each platform. The successful MCP artifact IDs are Linux
`10687945454`, Windows `10687860610` and Apple silicon `10687531114`.
Their ZIP hashes were verified against GitHub's artifact digests.

Download evidence by exact artifact ID: repeated attempts can have artifacts
with identical names. The committed summary retains identities and results
after the CI artifact retention period; full command logs and install/probe
reports remain in the run artifacts while retained.

## Failure history and coverage limits

MCP attempt 1 was blocked by an absent repository secret. Attempt 2 failed
private checkout authentication. After the user corrected the secret value,
attempt 3 passed without changes to source, candidate wheel or acceptance
criteria. Neither failed setup attempt ran consumer tests or counted as passing
coverage. No consumer regression or unresolved requested configuration remained
after attempt 3. Secret values are not included in this record.

The minimum-dependency run emitted 11 Shapely 2.0.0 intersection warnings and
passed all 953 assertions. Other macOS versions, Intel downstream native
consumers, desktop packaging, CAD export, CRS reprojection and broader curved
overlap algorithms remain outside the plan's coverage.

## Owner delivery and release boundary

[Owner handoffs](OWNER_HANDOFFS.md) identify the exact stale ANYopenSoft entries,
replacement evidence, and the ANYfileIO/ANYfileio-occt formula, examples and
regression evidence. Their adoption is work for those repositories; this change
does not activate CAD export or waive its capture/transaction requirements.

Automated implementation verification is complete. Independent review and merge
are separate repository review actions, not assertions made by this record.
A compatible 0.4.4 release requires separate release authorization. The existing
0.4.3 ledgers, artifacts and publishing authority remain unchanged.
