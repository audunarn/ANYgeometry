# Bounded trimmed-domain owner query

Implementation under focused qualification; no release or actual-case integration
claim. Governing plan: `ANYGEOMETRY_TRIM_DOMAIN_OWNER_CONTRACT_PLAN.md`, SHA-256
`D1AF0BAC87758111CC344285C8F41472E184A70C733D77B0B33D8981A181FE1E`.

`query_trim_domain_relation(model, first, second, *, expected_revision,
cancellation_check=None)` returns immutable `TrimDomainResult` evidence. Parents
must be distinct active model-bound face handles; replacements are not resolved
implicitly. `validate_trim_domain_binding(model, result, first, second, *,
expected_revision, cancellation_check=None)` checks the exact ordered parents,
UUID, revision and digest, then requalifies all evidence against the live pair.
A self-rehashed forged certificate is not accepted. Both calls are read-only.

The complementary-face exemption requires **all** of: successful live validation,
`relation == TrimInteriorRelation.DISJOINT_INTERIORS`, `complementary is True`,
and `boundary == TrimBoundaryContact.CURVE`. It means disjoint material interiors
with a shared ring, not disjoint closed sets. UNRESOLVED, shared edges alone or
an intersection result of dimension CURVE are not such proof. In particular,
the older planar query uses an area threshold, so tiny positive polygons can
appear as lower-dimensional results. This query does not equate that with zero area.

## Supported positive proof

Two exact common-plane Plane faces without alternate parameterizations. One is a
convex Straight outer ring with exactly one hole; the other is its topology-identical
hole-free filling. The filling has 3–64 all-Straight or all-minor-Arc edges.
Exact rational signs prove strict continuous polar progression, integer winding one,
and therefore simplicity. Every supporting circle is strictly inside every outer
halfspace with owner tolerance clearance; this conservatively encloses each full Arc.
No sampled area, chord-based curve intersection or approximate angle sum is used.

Noncircular Arc chains qualify only if they satisfy that same sufficient proof.
Two semicircles, major arcs, mixed curve rings, extra holes, unsupported supports,
non-star shapes and general curved overlays remain UNRESOLVED. Extra holes are an
unsupported configuration, not a declaration that the user's model is invalid.
Small nonzero common-plane determinants are not snapped; rounded arbitrary rotations
can therefore become unresolved. Do not claim general transform covariance.

Straight planar pair queries reuse existing owner predicates on a detached two-face
copy. Complete DISJOINT or REGION classifications retain their meanings; other
results cannot qualify the complementary exemption without the exact proof above.
Missing optional capability remains explicitly unresolved. Source model caches,
indices, allocators, incidence and revision are never modified or restored afterward.

Fixed bounds: 64 edges per loop, 256 unique edges, 512 defining vertices, 200,000
rational primitive operations, 8192-bit numerators/denominators including bounded
intermediates. Work exhaustion is unresolved. `work_counts` reports copies,
halfspace/winding/ring work, rational operations and owner-query calls.

## Errors and integration boundary

`TrimDomainError.code` distinguishes INVALID_REQUEST, WRONG_MODEL,
INACTIVE_ENTITY, STALE_REVISION, STALE_RESULT, INVALID_RESULT and BUSY_MODEL.
Open transactions cannot be certified. Callback/API/backend operational exceptions
propagate unchanged; cancellation receives `trim domain qualification`. Such
exceptions are never converted into a no-overlap exemption.

The initial acceptance fixture is an actual GeometryModel with a square outer and
12 shared minor Arcs, both operand orders/reversal and supported transforms. The
recovered cylinder/plate construction is now owner-reconstructed as recorded below;
separate consumer qualification remains open. ANYmesher must consume owner evidence rather than retain
a downstream proof or filter away contradictory owner overlap results.

Schema 4, automation protocol 1, MPL-2.0 and package version remain unchanged.
No commit, push, full suite, hosted test or publication is authorized by this slice.

## Focused implementation evidence (2026-09-06)

Base main remains `a4f798ad8e52b12f91396bec71a84428b5e17201`. Only the
registered module, package exports, focused test file and this document are edited.
Exact LIGHT command: `python -B -m pytest tests/test_trim_domain_contract.py -q`.

- Initial run: 28 passed, 10 failed in 2.73s. Nine hostile-fixture tests tried to
  serialize deliberately invalid topology before querying; the codec correctly
  rejected it. The other snapshot incorrectly included changes made by opening
  and closing the test's own outer transaction. These were test-observation bugs,
  not waivers of query invariants. Snapshotting now performs no serialization or
  validation and the busy-state check compares inside the same transaction.
- Corrected run: 38 passed in 1.73s.
- Expanded focused run: **50 passed in 2.20s**. Added chart scaling/reversal,
  full-Arc rather than endpoint containment, work/bit/loop limits, late cancellation,
  typed capability absence, wrong-model retained evidence, copying of result maps,
  invalid protocol data and self-rehashed proof tampering.

Source changes also retain the complementary proof's own physical tolerance when
checking consistency with a legacy straight query, and explicitly classify
nonfinite proof arithmetic as unresolved. No broad suite or performance claim.
Cold/warm live snapshots and unrelated-geometry work-count equality are functional
checks only. At the 50-test checkpoint, independent exact-diff review and actual
reconstruction were pending; the corrections/reconstruction below supersede that
checkpoint. Consumer integration and full regression/build/hosted evidence remain
separate gates.

## Review corrections and recovered-case evidence

The review-authorized corrections stay within the same four-file allowlist:

- Reused owner outcomes require a complete aggregate certificate **and every
  component certificate** before positive fallback or complementary consistency
  acceptance. The existing `classified` property already checked aggregate
  completeness, but did not establish complete components when an explicit
  aggregate certificate overrode them. Focused negatives cover incomplete/missing
  DISJOINT evidence, incomplete REGION evidence and complementary consistency.
- The rational Arc conditioning floor is now the maximum of `2**-40` and the
  owner's `_COLLINEAR_RTOL` (currently `1e-12`), with equality rejected. A pure
  cache-free numerical owner `arc_frame` call on detached records must also produce
  finite usable frame data. Evaluator exceptions propagate unchanged; no evaluator
  rejection is converted into a successful proof. Tests straddle the owner floor,
  preserve injected exception identity, reject nonfinite frames and check live state.

Exact LIGHT command unchanged: `python -B -m pytest tests/test_trim_domain_contract.py -q`.
Correction-only run: **59 passed in 2.36s**. Actual public reconstruction added:
**60 passed in 2.73s**. Final added radial-ownership and nonfinite-frame guards:
**61 passed in 2.74s**. No full suite, benchmark, build or hosted run was performed.

### Authoring to post-imprint lineage

Provenance: ANYmesher recovered the first two user GUI commands from attachment
`757dba6c-091a-4ca7-905f-20b171437fb8/pasted-text.txt`. No other commands, loads,
sections or supports were executed. This supplies exact pre-imprint authoring
parameters, not a persisted post-imprint GUI document or its byte identity.

Focused node: `test_recovered_cylinder_plate_public_reconstruction` in
`tests/test_trim_domain_contract.py`. The test uses only public owner APIs to
generate, materialize and imprint this construction:

1. `anygeometry.generators.cylinder(radius=0.5, height=2.0,
   circumferential_segments=12, origin=(0,0,0), axis=(0,0,1),
   radial_direction=(1,0,0), longitudinal_spacing=0.5, ring_spacing=1.0)`.
2. Capture the generated baseline; append/materialize `generator.plate` with
   `length=2.0`, `width=2.0`, `origin=(-1,-1,1)`, `u_direction=(1,0,0)`,
   `v_direction=(0,1,0)`, `semantic_group='shell'`.
3. `query_intersection` with model-bound plate/cylinder handles, then
   `plan_imprint` / `apply_imprint`, both with explicit `ConnectionIntent.CONNECT`.

Assertions establish this deterministic fresh reconstruction's local lineage:

| Authoring identity | Observed result |
| --- | --- |
| Cylinder shell faces 1–24 | All remain active with unchanged replacement resolutions |
| Plate feature output `face/1` -> face 25 | Replaced by faces 26 and 27; feature output resolves to both |
| Existing ring Arc edges 13–24 | Reused exactly; same Edge records and endpoint/via coordinate bytes before/after imprint |
| Face 26 / face 27 | One-hole outer / hole-free filling, both Plane supports |
| Each shared ring edge | Four geometry-face, FaceUse and Coedge incidences across two Sheets |

The Plane chart is asserted as origin `(-1,-1,1)`, u vector `(2,0,0)`, v vector
`(0,2,0)`. Apply emits exactly one ChangeSet. Public topology validation is clean.
For both ordered pairs (26,27) and (27,26), the new owner query returns
DISJOINT_INTERIORS, complementary=true, boundary=CURVE and exactly those twelve
shared edge handles; fresh binding requalification succeeds without changing live
state. The proof consumes the **generated records**, not inferred trigonometric
coordinates or a copy of the earlier synthetic fixture.

This qualifies the supplied construction through ANYgeometry's current public
path. It does not claim equality to unavailable historical GUI post-imprint bytes,
an ANYmesher mesh result, general curved-region overlay or general covariance.
Independent correction review and consumer integration remain pending. Nothing
has been committed, pushed, tagged, built for publication or uploaded by this slice.
