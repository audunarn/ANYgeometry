# Cylinder sector occurrence atlas (candidate owner contract)

This additive contract is under focused qualification. Do not infer acceptance
from the presence of public symbols or package version 0.4.3. The accepted planar
trim contract is source commit `12683ae5d6dbb2620f5020b67c0a7673f2601766`;
the atlas successor must be pinned separately after independent review.

Geometry schema 4, automation protocol 1 and MPL-2.0 are unchanged. Atlas data is
derived, read-only and not persisted in geometry documents. No source allocator
IDs, mesh nodes, triangulations, virtual Faces or fake seam Edges are created.

## Supported family and ownership

The input is an ordered selection of distinct existing Cylinder FaceUses and one
selected reference FaceUse. Source Faces remain separate. A connected sector ring
must describe one complete period, with valid individually owned material loops.
Sector sweep is at most a quarter turn (within the model's angular tolerance).
The initial boundary family is axial Straight and coaxial Arc. Source interfaces
must already be partitioned into whole shared Edge identities; a missing split
returns `CAPABILITY_MISSING`, never inferred coordinate welding.

Each Face parameterization must be absent or the identical support object. This
is a supported-family restriction, not a geometric inequality assertion about
other parameterizations. A single stored periodic Face, arbitrary Spline/oblique
trims, other surface families and virtual interior chart cuts are deferred.

Seam-crossing holes are existing notched-sector or hole trim fragments. Only
remaining material intervals are paired. A void never gets an alias. Unselected
radial Coedges remain labelled, including interfaces belonging to a separate plate.
The atlas does not author missing trim geometry or repair invalid owner records.

## Calls and binding

Public imports are available from `anygeometry` or `anygeometry.cylinder_charts`:

```python
atlas = query_cylinder_atlas(
    model, selected_face_uses, reference_face_use=reference,
    expected_revision=model.revision, cancellation_check=check_cancel,
)
validate_cylinder_atlas_binding(
    model, atlas, selected_face_uses, reference_face_use=reference,
    expected_revision=model.revision, cancellation_check=check_cancel,
)
rows = evaluate_cylinder_occurrences(
    model, atlas, requests, face_uses=selected_face_uses,
    reference_face_use=reference, expected_revision=model.revision,
    cancellation_check=check_cancel,
)
```

Validation and evaluation check the CALLER's expected ordered selection AND
reference, not merely whatever handles arrive in an atlas. Both requalify against
current records; the canonical SHA-256 digest is integrity evidence, not a proof
signature or permission token. Wrong model, stale revisions/results, malformed
requests/evidence and open transactions are typed errors.

`cancellation_check` receives a diagnostic phase string and may raise. The exact
exception propagates, including during fresh binding checks and evaluation.
There is no partial successful result or source publication on cancellation.

## Coordinates and identity

`CylinderOccurrenceRequest(occurrence_id, parameter)` uses the source Edge's native
parameter t in [0,1]. Coedge reversal changes traversal, NOT the identity of t.
Occurrence `start_vertex`, `end_vertex` and endpoint coordinate pairs are in native
Edge t=0/t=1 order. `source_range` records the oriented traversal separately.
FaceUse orientation remains separate from both.

Rows retain local sector UV, a separate lifted-reference pair (angular turns,
physical axial coordinate), enclosures and residual bounds. One local U increment
is a sector sweep, not one revolution. Equivalent reference starts/reversed axes
are explicit chart mappings, not changes to physical source ownership.

Two interface mates name the SAME Edge and use the SAME t; `traversal_same`
describes traversal ordering only. Reference seam mates report an integer lift
offset of +/-1. Endpoint samples use canonical Vertex identity; interior samples
use Edge identity plus the exact floating t representation. No rounded coordinate
or root proximity merges samples. Keys are model/revision/atlas-bound and contain
no mesh ID. ANYmesher owns sample placement, registry allocation and elements.

## Evidence and refusal

All exported records are immutable with finite tuple data. `QUALIFIED` requires
complete evidence for every selected loop/interface. `UNRESOLVED` and
`CAPABILITY_MISSING` expose diagnostics but no partial consumable atlas. Evaluation
of an unqualified atlas, or an unqualified row, returns a typed failure for the
whole request. Operational owner evaluator failures propagate unchanged.

Proof uses outward rational intervals, a participating-geometry local frame,
analytic harmonic/affine curve bounds, source-incidence pairing and periodic
material-boundary checks. Sampled polygons and modulo-unwrapping heuristics are
not certificates. Returned floating rows are checked separately against the
same model tolerance; no assumed libm ULP guarantee replaces residual evidence.

Default hard limits: 256 FaceUses, 4096 occurrences, 8192 vertices, 200000 interval
operations, 65536 pair tests and 4096 rows per call. Policies may lower limits,
not relax numerical evidence or raise caps. Fixed 80-bit outward arithmetic,
8192-bit rational intermediate cap and bounded series stop unresolved on exhaustion.
Work counts are accounting, not measured speedups. No full model clone/index scan
or source cache warming is permitted by this read-only contract.

## Qualification and distribution state

Gate A public authoring fixture passed before implementation: eight real sector
Faces with two notches form a seam-crossing hole, with exact shared retained seam
Edges and no physical Edge across the void. This is source authoring evidence,
NOT atlas numerical qualification. Focused matrix, independent proof/source review,
consumer integration and platform/package evidence must be recorded separately.

### Earlier local checkpoints, 2026-09-07 (superseded candidates)

The initial 36-case focused subset passed on both available interpreters:

| Gate | Exact child command | Result | Supervisor wall time |
| --- | --- | --- | --- |
| B | `C:\Python\Python313\python.exe -m pytest tests/test_cylinder_atlas_contract.py -q` | 36 passed in 10.90s; exit 0 | 13.307260s |
| C | `C:\Python\Python314\python.exe -m pytest tests/test_cylinder_atlas_contract.py -q` | 36 passed in 9.10s; exit 0 | 10.923818s |

Both ran once in `C:\Github\ANYgeometry`, with a 60-second subprocess deadline,
`PYTHONPATH=C:\Github\ANYgeometry\src`, `PYTHONNOUSERSITE=1`,
`PYTHONDONTWRITEBYTECODE=1`, and OMP/OpenBLAS/MKL thread limits of 1. Both child
processes were reaped. These are functional run durations, not speedup evidence.

The exact tested source SHA-256 is
`CA0D5BC3D878EE81FCAE6A76FA1CB088A7546681178525CAD691FD836E623714`;
the exact tested focused-file SHA-256 is
`95197A045FB6D71438BFF54561D3C31D58EC296C7302D17FEFC92C1A9DBCB9C0`.
Base HEAD remains `12683ae5d6dbb2620f5020b67c0a7673f2601766`.
These hashes identify uncommitted candidate files, not a distributable commit.

This subset includes the seam-hole identity graph, native-parameter pairing,
translation/scale and sign variants, caller binding, stale/tampered results,
purity snapshots, bounded inputs, cancellation exception identity, malformed
owner frames, rational allocation preflight, independent high-precision Decimal
trigonometric bounds, and unrelated-geometry work-count equality.

The expanded test file
`2B6274B745B0CB92CFDAE4F947A08E31CD2D3F89C9731AFCCD0242C5FF54BCDE`
then passed 49 cases: Python 3.13 in 18.31s (supervisor 20.058103s), Python 3.14
in 19.11s (supervisor 20.377970s). Both exited 0 with reaped children. This adds
public cylinder/plate CONNECT with external radial Coedges, actual Cylinder
descendant replacement, ordinary-interface/local-hole variants, reversed FaceUse,
parameterization mismatch, distinct identities, ill-conditioning injection,
cancellation phases and cold/warm cache comparisons.

The next test file
`C5C13E0813642639A756DAFF485129872679B7222D45F6006EDB46D29CD84C38`
passed 51 cases on Python 3.13 in 19.30s (supervisor 20.727883s, exit 0, child
reaped). It adds equivalent starts and missing shared Edge identity despite
matching endpoint IDs. Its Python 3.14 run was held for independent review.
These runs used the same source hash and commands/environment above; no failed
test or retry is hidden by the expanded matrices.

### Accepted independent review corrections

Independent review of the earlier candidate required two corrections despite
green functional tests: aggregate copying/validation work was not consistently
bounded, and reference sample enclosures omitted sector-frame uncertainty.
The corrected implementation limits total incidence entries before public copy,
checks cancellation during incidence copying, charges material-seed event,
candidate and clearance work, and bounds recursive evidence validation plus
canonical encoding by aggregate node/byte budgets. Binding work uses a
context-local budget; callback exceptions retain their identity.

Frame-map error is now retained and composed with the returned row's residual.
Inverse-frame row norms bound radial/axial coordinate errors without assuming an
orthonormal stored frame. Angular displacement uses a certified asin bound for
an off-radius point, not a same-radius chord or small-angle approximation.
Offset/tilted public fixtures use an independent high-precision reference inverse.

Corrected source SHA-256:
`2C1AC90E1AF0142DFBCB856A5CB574D90CBE00B4C73E435AD9EF7A5AFB6B5E30`.
Corrected test SHA-256:
`2A03A3B57E95F2FF166FAA5B2580D93ED97FA0E273F041D32276B5E15C47A14D`.
Corrected focused execution (same exact Gate B/C commands and environment):

| Interpreter | Result | Supervisor wall time | Process state |
| --- | --- | --- | --- |
| Python 3.13 | 56 passed in 22.45s; exit 0 | 24.188563s | child reaped |
| Python 3.14 | 56 passed in 23.06s; exit 0 | 24.359796s | child reaped |

There was no failure, timeout or retry in this correction run. The independent
reviewer accepted these exact source/test bytes and authorized a four-path local
direct-child commit atop the trim baseline. Full/platform/package verification
remains a separate resource-gated step. No consumer activation, push or publication
is implied by source acceptance or the local commit.

The accepted trim commit is now remote on GitHub; its hosted gate is tracked
separately. No wheel from that commit has yet been handed off as qualified.
Historical wheels and the existing public 0.4.3 label are not substitutes for an
exact source/artifact pin. Atlas code is not published or activated by this document.
# Principal-cut lifting

Vertex coordinates whose certified Cartesian interval crosses the negative
radial ray are represented by both principal-angle images. The atlas lifts each
image independently to the same sector/reference target and joins them only
when the lifted intervals overlap with width below pi. Origin-containing boxes,
ambiguous period choices and disconnected images still fail closed. Scalar
arithmetic, surface tolerances, carrier qualification and topology identity are
unchanged. Reference-vertex caches retain Cartesian intervals so that a target
does not inherit a prematurely selected principal branch.
