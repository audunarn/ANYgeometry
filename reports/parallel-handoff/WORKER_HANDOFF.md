# Parallel geometry handoff — implementation owner

State: **verified against the assigned focused development criteria**. Root
acceptance, installed consumer integration and performance assessment remain
separate. Source candidate is frozen; no commit, push, branch/history change,
schema/version bump, sibling edit or mesher execution was performed.

Repository: `C:/Github/ANYgeometry/.worktrees/parallel-handoff`.
Branch provided by root: `codex/parallel-handoff`.
Base: `c6dc4261995a1e0c1ac9c19ba8ed96131d24b880`.
Configured OpenAI implementation fallback was used after cached Mistral
unavailability; no provider probes or configuration changes were made.

## Delivered behavior

- Root-exported conservative independent component partition with complete
  model/revision/content/selection binding, deterministic roots, structured
  merge explanations and explicit nonexecutable refusals.
- Complete shared dependency incidence, including standalone orientation
  references, shared attachment targets, vertex/control/edge/face dependencies,
  complete sheets, every attachment intent, junctions and transitive active
  lineage. Referenced geometry enlarges consumer bounds. Active Part lineage
  closes full children; ordinary context Parts may remain partial. Omitted
  dependent/interacting owners and partial sheets refuse rather than disappear.
- Planar exact trim boxes and full-revolution Cylinder/Cone enclosures with
  conservative boundary axial extents, legal extrapolation and outward rounding.
  ExtrudedSurface/RuledSurface/CoonsSurface/absent/unknown support certification
  explicitly refuses until exact trim containment is available. Supported edge
  curves use analytic/control-hull bounds. Eccentric/offset metadata without a
  geometry-owned extent contract refuses every nonempty selection, including
  declarations by an unselected owner with a disjoint root-axis box.
- Plain JSON/pickle-safe closure transport, preserving the current geometry
  schema, work UUID/local IDs, source UUID/revision/selection and complete
  bidirectional mappings of all eleven entity kinds. Incomplete, duplicate,
  wrong-kind or nonbijective maps reject; decoded arrays and read-only maps are
  independent of the original model. Existing geometry serialization retains
  sparse local IDs, UUID and revision.
- Existing all-or-error chart query retains its mathematical results and
  deterministic ordering. A new typed per-face batch preserves each original
  face error and returns successful rows as ordinary bound chart collections.
  Invalid source/selection, cancellation and source mutation abort the entire
  query. Complete persisted content and all eleven validation-relevant reverse
  incidence maps are freshly checked for reuse; revision-only caching is absent.
- Public `to_dict` still genuinely validates every call, then seeds content plus
  incidence qualification reuse. Post-qualified planning/application/provenance
  guards reuse qualification only for exactly matching fresh content and index
  snapshots. Changed content or derived indexes receives qualification/rejection.
  Direct invalid source edits retain the original chart binding error order.
- Definition checksums fast-path exact primitive values, preserving canonical
  JSON/checksum bytes, mutable-array detection and public output independence.
  No global entity/revision checksum cache was introduced.

Public API and limitations: `docs/PARALLEL_COMPONENT_HANDOFF.md`.
Request 4 (authored face regions/interface stations) remains deliberately later.

## Owned changed files and exact candidate identity

`reports/parallel-handoff/candidate-sha256.json` and the explicit frozen
`candidate03-sha256.json` record SHA256 of the eleven owned
source/test/doc/harness files. Final manifest bytes SHA256:

`9e302becbceb9774e2164ba7b97ffc4c831d5c79ffea89f4f49f231269877816`.

Candidate02 remains preserved in `candidate02-sha256.json`, with manifest bytes
SHA256 `d5276dfae950ec2ba71fe801f0fbccb15d9e45d96f1007a142b734d4aef6487e`.

The original measured candidate manifest remains byte-for-byte preserved in
`candidate01-sha256.json`, with manifest bytes SHA256:

`4a880b9a4bd427457db0ef57c6a5ac2bed472e2bf0451a25c460d01f7160b263`.

The manifest includes:

- `src/anygeometry/__init__.py`
- `src/anygeometry/component_partition.py` (new)
- `src/anygeometry/closure.py`
- `src/anygeometry/serialization.py`
- `src/anygeometry/trimmed_charts.py`
- `src/anygeometry/definition_binding.py`
- `src/anygeometry/batch_intersections.py`
- `src/anygeometry/edge_subcurve_preimages.py`
- `tests/test_parallel_handoff.py` (new)
- `docs/PARALLEL_COMPONENT_HANDOFF.md` (new)
- `tools/run_parallel_handoff_tests.py` (new)

Root-owned `reports/parallel-handoff-root` was observed in Git status and was
not changed. Other worktrees and the root-owned living study note were untouched.
`git diff --check` passed (only normal Windows line-ending warnings).

## Checks actually run

Exact commands, complete candidate source/test hashes, effective Python path,
single-thread environment, precise process-wall accounting, status/exit,
stdout/stderr, JUnit counts and remaining budget are retained per attempt.
Runtime: `C:/Python/Python314/python.exe`, Python 3.14.2. Test child `PYTHONPATH`
was the owned worktree `src`; `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`,
`MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS` were all explicitly `1`.
No test process launches subprocess workers. Each launch was bounded by
`min(30 s, remaining aggregate 90 s)`; no timeout occurred.

| Evidence directory under `reports/parallel-handoff` | Tests | Failed | Process wall (s) |
|---|---:|---:|---:|
| `check-20261006T200418-69eeda7c` | 42 | 3 | 2.9924353999085724 |
| `check-20261006T200624-1cbda70e` | 19 | 0 | 1.8563160999910906 |
| `check-20261006T200846-6f3b6099` | 53 | 0 | 4.410211400012486 |
| `check-20261006T201150-6da80ee8` | 130 | 0 | 3.686811099993065 |
| `check-20261006T201504-083bb39a` | 96 | 1 | 5.703083100030199 |
| `check-20261006T201603-7d2769b9` | 4 | 0 | 1.6459663000423461 |
| `check-20261006T201635-5d659ed6` | 187 | 0 | 6.308376800036058 |
| `check-20261006T202525-fe2cf9eb` | 5 | 3 | 1.6845215999055654 |
| `check-20261006T202603-afbfdc14` | 5 | 0 | 1.6285004999954253 |
| `check-20261006T203055-87470d81` | 2 | 0 | 1.5675377999432385 |

Aggregate measured process wall: **31.483760099858046 s**.
Unused authorised focused budget: **58.516239900141954 s**. This is accounting,
not renewed authority for older closed diagnostic families.

The first failures identified an incorrect implicit-support fixture expectation,
a decoder swallowing its specific duplicate-record GeometryError, and a fixture
whose warped Plane could not legally be committed. Those were repaired. The
later failure exposed chart binding error-order compatibility after qualification
reuse; the expected-content guard repaired it. All failed evidence is preserved.

The final 187-test run covered the new partition/transport/batch/mutation tests,
existing trimmed chart tests, preparation binding, prepared-face provenance,
edge-subcurve provenance and detached batch publication, definition-binding byte
parity, current/legacy serialization, and selected batch overlap/refusal/rollback/
cancellation tests. It includes standalone orientation/target/lineage sharing,
promoted target extents, extracted non-Part dependency disjointness, sparse IDs,
transport duplicate source targets, raw source/chart mutations, cached derived
incidence corruption and last-policy-callback source/candidate corruption.

## Remaining acceptance obligations

- Root must inspect final ownership/diff and reconcile its independent review.
  Reviewer source findings (mapping injectivity, standalone dependencies, promoted
  bounds and derived incidence corruption) were repaired with focused regressions.
  The reviewer is a separate author from the same OpenAI model family; this
  does not confer different-family independence.
- Root owns frozen baseline/candidate measurements for fresh geometry-only
  16/32/48 connected fixtures, output/checksum/definition identity and call counts.
  No author performance measurements were run or savings claimed.
- Root owns isolated installed-artifact/public consumer integration. No full
  suite, mesher run, external qualification campaign or release was performed.
- Unsupported trimmed surface families and unknown physical eccentric reach
  retain the documented serial/refusal path; no partial certificate permits
  dispatch. Face-region decomposition remains outside immediate requests 1–3.

## Final bounded continuation after measured candidate freeze

Root confirmed all 18 candidate measurement runs exited successfully with exact
prepared/chart parity. The author performed no benchmarks. Root then authorised
a planner-only late correction: an unselected member/member-use/attachment with
unbounded declared eccentric/offset reach can interact with selected material
despite disjoint root axes. The refusal now applies globally to every nonempty
selection; an empty selection remains a certified empty workload.

Only the following candidate files changed after the measured candidate freeze:

| File | Candidate02 SHA256 |
|---|---|
| `src/anygeometry/component_partition.py` | `10ad6b9e9a64639e06f4ea680c9eaaf8d6e968ee80c353d1a9198c9f5277a896` |
| `tests/test_parallel_handoff.py` | `56ece2d528e9d1acff2d10a499494a0408f42791248eb63788eed9468156b09d` |
| `docs/PARALLEL_COMPONENT_HANDOFF.md` | `0efae532e53eea60a6c3d84b791ad5cf1d70eebc52e9638547054e679d525c5c` |

All eight remaining manifest entries, including every exercised preparation
source, match the measured candidate exactly. The benchmark does not execute the
partition API, so root assessed the preparation measurements as applicable
without a replay. This handoff evidence was also updated; no measured source or
root evidence was edited.

The continuation's first focused run exposed a misplaced preexisting fixture
assertion while inserting the new parametrized test; the three failures are
preserved at `check-20261006T202525-fe2cf9eb`. Correcting that fixture placement
yielded **5 passed** in `check-20261006T202603-afbfdc14`, covering unselected
unknown physical reach in all three declaring entity kinds, selected eccentric
refusal, empty selection, existing unsupported support refusal, raw mutation,
certificate evidence tampering and cancellation. No broader unchanged test or
performance input was rerun. Final `git diff --check` passed. Installed-artifact
integration and final acceptance remain root-owned.

## Candidate03 — final guard proof without product edits

Root raised a final-callback derived-index concern based on an older source read.
Inspection confirmed candidate02 already calls `_qualified_model_state` after
the final partition callback. The certificate validator reconstructs through
that same guarded planner. No product repair was required, and root explicitly
adopted keeping the actual package source unchanged.

A focused regression now clears `_face_structural_uses` in that final callback,
while proving that the complete persisted document and revision remain equal.
Both partition planning and retained-certificate validation reject with the
original reverse-incidence GeometryError. **Two tests passed** at
`check-20261006T203055-87470d81`; its precise process-wall time is included in the
same cumulative ledger above. No benchmark, installed check or broader test was
run by the author for this evidence-only continuation.

Candidate03 differs from candidate02 in exactly one manifest entry:
`tests/test_parallel_handoff.py` SHA256
`b5e35fab7cd9f8d3251d9aa6cbec3e93dbbaf4dab405612ef29f72d47e75bcb5`.
All package source, documentation and test harness bytes remain identical.
The root-installed candidate02 wheel can therefore be assessed against these
same final package hashes without a rebuild; artifact acceptance remains root's
responsibility. This handoff and the candidate03/current manifests were updated,
and candidate01/02 manifests and all evidence remain preserved.
