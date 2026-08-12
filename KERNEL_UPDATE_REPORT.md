# ANYgeometry 0.2.1 kernel gap-closure report

Status: implementation, repository tests, performance qualification, offline
package build, Twine validation, isolated installed-wheel smoke, local commits,
branch publication, and draft PR creation are complete. The downstream
public-tree rerun and ecosystem closeout remain pending.

This report covers the strict mixed plate/member kernel and the additive
mesher-facing gap closure. All source changes are confined to ANYgeometry.
ANYmesh/ANYmesher, ANYfem, ANYstructure, ANYsolver, and other sibling
repositories were not edited in this task.

## Authority, branch, and baseline

- Original governing plan:
  `C:\Users\AudunArnesenNyhus\Downloads\ANYgeometry_kernel_update_codex_plan.md`
- Coordination ledger: `docs/KERNEL_UPDATE_OVERVIEW.md`
- Repository: `C:\Github\ANYgeometry`
- Authoritative base: local `kernel_update` at
  `19f16746ceee76838b7df9d08a95028699e3a738`
- `origin/kernel_update` did not exist; `main` was never substituted.
- Working branch: `native_hybrid_mesher`
- Registration head before final gap-closure edits:
  `f2d7793d7d32a6dcd772c7ed8701aca11b459288`
- Package target: ANYgeometry 0.2.1
- Geometry writer schema: 4
- Supported geometry readers: schemas 1, 2, 3, and 4

The coordinated branch is an approved improvement over the earlier
`codex/strict-kernel-update` name because it retains the required base while
providing the agreed ANYmesher read contract. It does not authorize writes to
the downstream repositories.

## Delivered kernel

### Persistent identity, ownership, and transactions

- Public `EntityHandle(model_id, kind, id)` values bind persistent identity to
  the model UUID; wrong-model, active, replaced, deleted, and unknown outcomes
  are explicit. `EntityRef` remains a local compatibility value.
- Geometry and structural allocators use monotonic high-water marks. Failed
  transactions, snapshot restore, feature regeneration, and alternate edits
  do not reuse retired IDs.
- Public geometry, structural, group, tag, coordinate, and feature-history
  views are owner-controlled. Mutations publish one deterministic `ChangeSet`
  per successful outer transaction.
- Delta journals capture changed records and dependency bounds, roll back
  topology/semantics/ownership/caches/spatial state atomically, and leave
  allocator gaps on failure.
- Reverse incidence covers topology, Sheet/FaceUse/Coedge ownership,
  Member/MemberEdgeUse axes, Attachments/Junctions, curve controls, and
  orientation references. Local structural validation expands only the
  affected semantic closure; complete Sheet validation remains intentional
  when manifold/orientation invariants require it.

### Mixed plate/member topology

- `Part`, `Sheet`, `FaceUse`, and persistent `Coedge` records own oriented
  shell topology. `Member` and ordered `MemberEdgeUse` chains represent
  physical beams independently of geometry edges and finite elements.
- Member identity survives geometry subdivision. Explicit Member split records
  structural lineage. Reverse/copy/insert operations retain or remap ownership,
  qualified relationships, parameterization, construction ownership, and
  model-bound identity atomically.
- `Attachment` and `Junction` persist explicit connection intent, parameter
  witnesses, evidence strength, residual/tolerance, context, provenance, and
  lineage. Unknown or dangling relationships fail closed.
- Qualified face/face `CONNECT` persists a shell/sheet T-junction as shared
  B-rep edge topology in both Sheets' FaceUse/Coedge records. No downstream
  coordinate-proximity connection inference is required or permitted.
- Straight sketch extrusion walls are canonical Plane supports; aligned arc
  extrusion walls are exact Cylinders. Genuine non-planar Coons cases remain
  typed unsupported.

### Query, plan, apply, and tolerance

- `query_intersection` returns immutable typed, model-bound components with
  dimension, parent/subparent identity, parameters, witnesses, residual,
  tolerance, quality, and diagnostics.
- `plan_imprint` produces a deterministic revision-bound plan. `apply_imprint`
  revalidates staleness and the explicit policy before one atomic edit.
- Typed outcomes include disjoint, touch/tangent/cross, containment, overlap,
  coincidence, unsupported, capability-missing, and unclassified results.
- Supported workflows include planar multi-component/hole-aware face
  clipping/imprint, coplanar fragmentation, qualified Plane/Cylinder cases,
  straight and same-circle edge/member cases, and member/sheet relations.
  Unsupported combinations fail before mutation.
- `TolerancePolicy` separates computational, coincidence, healing, fitting,
  surface, angular, parameter, area, and AABB tolerances. Effective tolerance
  depends on participating local extents, not world origin or unrelated model
  size. Uniform model/unit scaling uses `TolerancePolicy.scaled`.

### Audit, spatial index, and mesher-facing read contract

- A deterministic maintained AABB tree supports incremental mutation,
  changed-region lookup, nearest queries, overlap candidate selection, and
  strict-audit broad phase. Conservative face bounds cover Plane, Cylinder,
  Cone, Ruled, and Coons interiors.
- `strict_audit()` classifies every admitted full-model candidate and fails
  closed on unsupported/ill-conditioned geometry, incomplete accounting,
  unverified relationships, invalid ownership, or index inconsistency.
- `audit_changed_region(model, change_set, policy=None)` reports explicit
  `CHANGED_REGION` scope and can never claim full certification.
- `extract_model_closure` returns a working model, bidirectional model-bound
  handle maps, source UUID/revision, and canonical source handles. It preserves
  selected dependency/structural closure without silently dropping provenance.
- Batch edge/face evaluation, derivatives, normals, and closest projection are
  exposed as model methods and module functions for Plane, Cylinder, Cone,
  RuledSurface, and CoonsSurface. Built-in primitive projection is batched and
  deterministic; unsupported custom cases retain qualified fallback behavior.
- Public coplanar-overlap query accepts selected face IDs, changed AABBs, or
  precomputed candidate pairs and uses the maintained index by default.

## Serialization, certification, and compatibility

ANYgeometry 0.2.1 reads geometry schemas 1–4 and writes canonical checksummed
schema 4. Schemas 1–3 migrate deterministically and one-way. Schema 4 stores
model identity/revision, coordinates/CRS, full tolerance policy, allocator
high-water marks, support and optional parameterization, construction/control
ownership, geometry and structural topology, qualified relationships,
semantics, extensions, lineage, and feature history.

Legacy relationship evidence that lacks persisted qualification data becomes
`UNVERIFIED`; zero residual/tolerance placeholders never imply exactness and
block certification. A 0.2.0 reader intentionally rejects schema 4 even though
the live Python dependency range remains `ANYgeometry>=0.2,<0.3`.

`certified=True` is a write-time strict-audit gate, not a persisted certificate.
Ordinary and certified schema-4 payload shapes are identical. An `AuditReport`
is evidence only for its exact model UUID, revision, and audit policy; a
consumer must retain or recompute that report for qualified handoff.

## Downstream contract state

The ANYmesher task accepted the public Python range and schema-4 boundary. Its
independent contract evidence reported 31 passing integration tests covering
canonical API import, real planar query/plan/apply, wrong-model rejection,
legacy quarantine, persistence/migration, and future-version failure. A later
solver regression reported 109 passing tests. These results support but do not
replace ANYgeometry's own release qualification.

The authoritative downstream symbols are:

```text
extract_model_closure
evaluate_edge_many
edge_tangent_many
evaluate_face_many
face_derivatives_many
face_normal_many
project_to_face_many
audit_changed_region
find_coplanar_overlaps
query_intersection
plan_imprint
apply_imprint
```

## Supported intersection and mutation matrix

| Operands | Qualified query support | Mutation support |
|---|---|---|
| infinite line / line | analytic disjoint, cross, parallel, coincident | query only |
| line / Plane | analytic disjoint, touch, cross, contained | query only |
| Plane / Plane | analytic disjoint, curve, coincident | bounded face workflow where trims qualify |
| line / Cylinder | analytic zero, tangent, or two-point result | qualified bounded Plane/Cylinder face workflow |
| straight edge / straight edge | disjoint, touch, cross, collinear overlap | weld/connect/reuse according to explicit policy |
| coincident same-circle Arc / Arc | touch, contained, overlap, coincident | qualified subdivision/reuse where representable |
| straight edge / planar Face | all material intervals, including concavity and holes | imprint or persistent Attachment where the pair permits it |
| planar Face / Face | disjoint, point, curve, containment, overlap region, coincidence | curve `IMPRINT`/`CONNECT`; region `IMPRINT` only |
| bounded Plane / Cylinder faces | qualified analytic components | supported qualified face imprint cases |
| Member / Member | piecewise straight and coincident same-circle axes | `CONNECT`, `KEEP_DISCONNECTED`, `CONTACT_ONLY`, or strict reuse |
| Member / Face or Sheet | straight-chain point/interval classification | persistent typed Attachment/Junction; no inferred weld |
| Sheet / Sheet curve intersection | face/face query with model-bound Sheet context | shared edge plus persistent FaceUse/Coedge topology |

`DISJOINT`, same-parent, and face/face point-only results plan
`NO_TOPOLOGY`. `REUSE_EXISTING` is non-creative and revision-neutral when the
compatible edge or relationship exists; absence rejects before mutation.

## Fail-closed unsupported matrix

| Case | Typed result | Strict-audit result | Mutation | Required capability or fallback |
|---|---|---|---|---|
| general mixed Spline/Arc or non-coincident Arc pairs | `UNSUPPORTED` | `BLOCKER`, `UNSUPPORTED_CANDIDATE` | blocked | add a verified analytical/numerical qualifier |
| general bounded non-Plane/Cylinder surface pair | `UNSUPPORTED` | `BLOCKER`, `UNSUPPORTED_CANDIDATE` | blocked | add a qualified surface predicate |
| inconclusive or ill-conditioned predicate | `UNCLASSIFIED` | `BLOCKER`, `UNCLASSIFIED_CANDIDATE` | blocked | improve conditioning or provide verified evidence |
| general planar overlay without Shapely | `CAPABILITY_MISSING` | `BLOCKER`, `CAPABILITY_MISSING` | blocked | install the `planar` extra (`shapely>=2.0`) |
| region-dimensional shell `CONNECT` | `UNSUPPORTED` plan | blocking if unresolved | blocked | choose explicit `MutationPolicy.IMPRINT` |
| face/face point-only imprint | `UNSUPPORTED` plan | blocking if conformality is required | blocked | record an appropriate non-topological intent downstream |
| legacy relationship with no residual/tolerance evidence | loaded as `UNVERIFIED` | `BLOCKER`, `UNVERIFIED_CLASSIFICATION` | not promoted to exact topology | re-query/requalify under schema 4 |
| closure request with `include_features=True` | explicit `GeometryError` | not certifiable as a feature closure | blocked | materialize a supported feature baseline separately |

No unsupported case is converted into sampling-based topology, an empty
success, or coordinate-proximity connection inference.

## Verification evidence

### Environment and baseline

- Windows 11 `10.0.26200`, CPython 3.13.9, NumPy 2.4.3, Shapely 2.1.2.
- Base commit test collection: 132 tests; 131 passed and one failed because the
  declared `src/anygeometry/py.typed` marker was absent from the base commit.
- Base commit: `19f16746ceee76838b7df9d08a95028699e3a738`.

### Final tests

- Full repository suite: `python -m pytest -q` -> **389 passed in 11.94 s**.
- Schema-4 persistence/feature milestone matrix: **116 passed**.
- Final evaluator contract independently reproduced by coordination:
  **26 passed**.
- Final intersection/policy/radial workflow matrix independently reproduced:
  **67 passed in 7.66 s**.
- Final generalized Attachment/source/radial/curved-split matrix independently
  reproduced: **45 passed in 0.25 s**.
- The full suite ran with Shapely installed. It also contains an explicit
  import-denial regression proving the general planar path returns typed
  `CAPABILITY_MISSING` when Shapely is absent.
- No formatter, linter, or static type checker is configured in
  `pyproject.toml`; none was invented for this release. The installed wheel
  independently verified the PEP 561 `py.typed` marker.

### CLI and package qualification

- Direct source `python -m anygeometry --version` initially failed with
  `No module named anygeometry` because the checkout is not installed and the
  pytest-only `pythonpath` setting does not affect normal Python. With
  `PYTHONPATH=src`, version, `--write-example`, and readback `--json` all
  passed; readback reported 12 vertices, 17 edges, six faces, and valid
  topology. The generated example was removed.
- Build tools: build 1.5.0, wheel 0.45.1, Twine 6.2.0, pip 26.1.1.
- The first isolated build attempt truthfully failed when build isolation tried
  to fetch `wheel` through the restricted network. The registered offline
  command `python -m build --no-isolation --outdir dist_gap_closure` then built
  both artifacts without network access. Twine reported **PASSED** for both.
- A first in-repository wheel-origin harness was invalid because its target and
  origin assertion could not be independent of the checkout. A second external
  TEMP harness installed the wheel but failed before import because Windows
  stripped nested `python -c` quotes; its 8.3/long-path cleanup guard also
  refused, and the exact orphan was subsequently removed under an explicit
  canonical-path guard. No package claim was made from either run.
- The final registered external-TEMP smoke decoded its verifier over stdin,
  cleared Python path/home/venv state, installed with `--no-index --no-deps`,
  and passed. `sys.executable`, `sys.prefix`, and `anygeometry.__file__` were
  all below the fresh TEMP venv and the module was outside this repository;
  version was 0.2.1, `py.typed` existed, packaged CLI printed 0.2.1, and exact
  cleanup reported `target_exists=False`.

Artifacts (generated evidence; intentionally not committed):

- `anygeometry-0.2.1.tar.gz`: 327,454 bytes, SHA-256
  `2E923E14407480FA25489963948A19691E8121C0674DC78C1ACCDC96E3F265F1`.
- `anygeometry-0.2.1-py3-none-any.whl`: 274,758 bytes, SHA-256
  `99D3035806E109341E92475B555D21CA89EBB12E6D9410C13132920122CA5E95`.

## Performance qualification

`benchmarks/kernel_benchmarks.py` uses public APIs and records wall time,
traced peak Python allocation where practical, entity counts, changed keys,
spatial updates, local structural-validation visits, and broad/narrow-phase
candidate accounting.

The renewed runner adds changed-region audit, schema migration,
feature-closure checksum, typed intersection query/plan/apply, a 22,500-face
indexed-query grid, and separate local Member and plate edits inside one
persistent mixed Sheet/Member model. The 150 x 150 larger grid is the practical
release cap on the supported runner; this report does not imply that a
near-100,000-face wall-clock qualification was executed.

The first renewed run exposed a lifecycle measurement defect and a real cold
construction cost: the 22,500-face query built an index for all 90,601
vertex/edge/face records through sequential inserts under allocation tracing
before returning four candidates (214.412 s, 135,371,776 B). The initial
changed-region record similarly included a cold 40,401-record build and took
99.312 s. Those values remain failure evidence, not local-query timings.

The fixed constructor bulk-builds a deterministic balanced tree. The final
22,500-face cold query took 17.2843 s under tracing and returned the same four
candidates as the 0.000948 s steady query. Cold time fell 91.94% relative to
the defective run, while traced peak allocation **increased 5.86%** to
143,303,312 B; this report makes no memory-improvement claim. The 10,000-face
cold/steady pair was 7.27885 s / 0.000873 s with nine identical candidates.

### Comparable Wave-0 to final measurements

`seconds` below is allocation-traced when a peak is present. Final
`untraced_seconds` is normal-runtime evidence, but Wave 0 did not record an
untraced counterpart, so no cross-mode time delta is claimed.

| Workload | Wave 0 -> final traced seconds | Time change | Peak change | Final untraced |
|---|---:|---:|---:|---:|
| 10k-face construction | 1.5882 -> 12.6854 | +698.73% | +54.28% | 2.2330 s |
| 10k persistent-member lattice | 0.3530 -> 6.6457 | +1782.52% | +311.80% | 2.7641 s |
| mixed stiffened panel | 19.8431 -> 5.7519 | -71.01% | +396.52% | 2.1858 s |
| cylinder with members | 2.9802 -> 47.5142 | +1494.33% | +455.33% | 13.8698 s |
| duplicate/crossing construction | 0.1304 -> 1.8149 | +1291.45% | +254.18% | n/a |
| large-model local edit | 12.0664 -> 0.3692 | **-96.94%** | **-99.23%** | n/a |
| schema serialization | 0.9368 -> 6.1808 | +559.80% | +7.61% | 1.0447 s |
| lineage construction | 0.2802 -> 1.4579 | +420.34% | +44.86% | n/a |
| lineage resolution | 0.1840 -> 0.002894 | **-98.43%** | **-52.62%** | n/a |

The construction regressions exceed the plan's 10%/25% review thresholds and
are explicitly accepted by the lead for correctness scope, not dismissed:
Wave 0's member benchmarks stored bare edges/groups, whereas the final cases
construct and validate persistent Parts, Sheets, Members, ordered uses,
relationships, reverse incidence, monotonic identity, and rollback journals.
Schema-4 serialization writes and validates identity, tolerance, ownership,
evidence, lineage, feature history, and checksum that schema 2 did not carry.
`peak_python_bytes` is transient construction allocation, not retained topology
size, but the added persistent records also have a real documented storage
cost. The decisive locality measurements improve sharply and avoid complete
model copying/audit on ordinary edits.

### Final locality and audit evidence

- Changed-region audit after index materialization: 2.91818 s traced,
  813,600 B, 1,348 candidates = 1,348 narrow-phase classifications,
  16,719 node visits, 2,914 leaf tests, nine index updates, and correctly
  non-certifiable `CHANGED_REGION` scope.
- Mixed-model local Member edit: 0.01149 s, 51 structural changes, 153
  structural keys visited, and `full_structural_validation=false`.
- Mixed-model local plate edit: 0.00247 s, two geometry changes, three index
  updates, and no structural closure scan.
- 10,000-face full audit: 60.2833 s untraced, 299,000 candidates = narrow
  phases = classifications, zero unclassified, clean. A 400-face smoke audit
  admitted 11,800 candidates; 25x faces produced 25.34x candidates, showing
  near-linear sparse growth rather than quadratic enumeration.
- Duplicate/crossing audit: 2.40234 s untraced, 30,000 fully classified
  candidates, zero unclassified, 6,000 issues.
- Strict-audit transient allocation is reported by the traced 400-face smoke
  audit: 5,444,774 B. Qualification audits intentionally disable tracing so
  candidate/classification timing is not distorted.
- Query/plan/apply: 0.003168 / 0.000205 / 0.018382 s; apply changed 19 records.
- Feature checksum: 0.000352 s and a 64-character SHA-256.
- Schema migration: 19.8157 s traced, 127,488,153 B.

Deterministic unit tests additionally enforce AVL/index invariants, brute-force
candidate oracles, zero constructor refits/rotations, bounded local validation
visits, and subquadratic sparse work counts. Correctness and performance retain
equal release priority: every admitted audit candidate is classified, while
ordinary local edits avoid whole-model snapshots and full strict audit.

The renewed qualification ran on the fully integrated working tree before its
final commit, rather than from a previously clean commit. This is a documented
procedural deviation from the original clean-checkout preference. Coordination
accepted the full/spatial/package evidence after inspection; no source, test,
or benchmark code changed after that run, and the exact result JSON is included
with the same implementation state. Only this report and overview were
reconciled afterward.

## Exact final commands

```powershell
python -m pytest -q
python benchmarks/kernel_benchmarks.py --output benchmarks/wave_gap_closure_smoke.json
python benchmarks/kernel_benchmarks.py --qualification --output benchmarks/wave_gap_closure_qualification.json --baseline benchmarks/wave0_qualification.json
python -m build --no-isolation --outdir dist_gap_closure
python -m twine check dist_gap_closure\anygeometry-0.2.1.tar.gz dist_gap_closure\anygeometry-0.2.1-py3-none-any.whl
$env:PYTHONPATH=(Resolve-Path src).Path; python -m anygeometry --version
python -m anygeometry --write-example strict-kernel-example.anygeometry.json
python -m anygeometry strict-kernel-example.anygeometry.json --json
```

The installed-wheel command used a fresh GUID child of the canonical long TEMP
root, `python -m venv --copies --system-site-packages`, offline `pip install`,
and a Base64-decoded verifier streamed to `python -`; its complete registered
PowerShell guard is retained in the ecosystem coordination ledger.

## Known typed boundaries

- This is a structural-surface kernel, not a general solid/NURBS CAD kernel.
- General mixed spline/arc pairs, non-coincident arc pairs, and general bounded
  non-Plane/Cylinder surface intersections remain typed `UNSUPPORTED` or
  `CAPABILITY_MISSING`; no sampling result is promoted to exact topology.
- Region-dimensional shell `CONNECT` is unsupported; explicit coplanar
  `MutationPolicy.IMPRINT` is required.
- `extract_model_closure(..., include_features=True)` is explicitly unsupported;
  selecting structural handles while disabling structural closure rejects.
- A complete Sheet may be expanded for manifold/orientation validation. A
  selected ownership parent includes only the complete children required by
  the selected semantic closure unless that parent was explicitly selected.
- Full strict audit is intentionally more expensive than transaction-time
  changed-closure validation.
- Finite-element generation, seeding, numbering, smoothing, mapped
  decomposition, materials, sections, loads, supports, solver state, and
  results remain outside ANYgeometry.

## Commit, artifact, and closeout record

- Base: `19f16746ceee76838b7df9d08a95028699e3a738`
- Pre-gap-closure branch head: `f2d7793d7d32a6dcd772c7ed8701aca11b459288`
- Kernel implementation and tests:
  `5e1c1d9250737db6d5c1fb23868bdc3f6d6ca658`.
- Contracts, documentation, benchmark runner, and accepted result JSON:
  `6c2e8ae7ee2de9320e7d5b93e3132cb5de9c70bd`.
- Initial reconciled report metadata:
  `8828019e0f940b0d6f240b98f8be17d6f306155b`.
- Publication metadata: the branch-tip commit containing this record; its exact
  SHA is included in the completion packet and draft pull request.
- Release artifacts and hashes: recorded above; generated artifacts are
  intentionally excluded from Git.
- Remote branch: `origin/native_hybrid_mesher`.
- Draft pull request: `https://github.com/audunarn/ANYgeometry/pull/2`, targeting
  `main` because `origin/kernel_update` does not exist.
- Ecosystem closeout verdict: **PENDING `ECOSYSTEM CLOSEOUT: OK`**
