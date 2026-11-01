# PP5c Part 2 blocked handoff

PP5 release 3 is BLOCKED. Part 1 is separately committed and verified at
e26e4ab71dfd8a6cef6d8610cce959ab059ef3bd. Part 2 is a preserved partial implementation,
not accepted production code. Branch: claude/pp5-stage-partition-20261031.
No independent review, integration, release, push or remote operation was performed.

## Blocking evidence and stop condition

The real sector60 pair assignment still refuses with
`batch module closure/binding changed`. Both attempted parallel runs fall back to
serial and preserve the existing plan checksum; neither is a parallel success.
They publish no accepted stage result and never reach final applied-model CI.
The exact changed module/binding was not identified by the retained gate message.

The three-failure stop is conservatively accounted as:
1. pp5c-sector60-parallel1: real assignment refuses module closure.
2. pp5c-late-imports: diagnostic setup fails `ledger creation live coverage mismatch`.
3. pp5c-sector60-parallel2: real assignment still refuses after four explicit eager imports.
There were TWO failed numerical parallel runs and ONE failed diagnostic attempt,
not three numerical runs. No retries or numerical checks were launched after item 3.
The successful intermediate late-imports2 diagnostic executed only the first saved
assignment task per worker without a ledger/reset; it is investigative evidence,
not an admitted worker execution or acceptance check. It identified branch_supports,
plane_ruled_support, quadric_supports and quadric_curve_events as late imports in
that partial census. Eager loading those four did not solve full-assignment closure.
No wildcard admission, tolerance change, source-pin refresh or gate weakening was used.

## Implemented, with limits

- Closed typed transport for real pair paths and face arrangements: exact registered
  record classes, finite primitive/scalar values, IEEE float bytes (signed zeros
  distinguished), native float64 array bytes/shape/writeability, shared-object graph
  references. Cycles, unsupported types/dtypes, distinct shared-memory arrays and
  unknown/modified instance extras refuse. Decode invokes constructors, checks
  fields, then re-encodes for exact canonical equality. Plane.normal is recomputed
  and bound by a derived manifest; Cylinder uses the Part 1 adapter. Empty known
  caches are admitted with reference checks; nonempty caches refuse.
- One model decode per worker/stage assignment. A fresh constructor-created model
  view and selected face/support prevent the cold retained model acquiring caches;
  source packet is reverified at assignment boundaries. Generic copy was removed
  after its GeometryModel.__slotnames__ mutation was correctly refused by the gate.
- Real pair and arrangement tasks use the batch gate with an exact WorkerModel
  context; descriptor checks, ledger resets, staged-result discard and checks at
  assignment boundaries remain. K is the whole assignment by default.
- Public plan_intersections has a private control hook behind the existing opt-in;
  default OFF follows its original path. ON requires explicit expected/observed
  plan and applied-model CI. Unsupported budget/cancellation/nonexact policies
  decline to their serial routes. Member planning and final normalization/sealing
  stay on the host. PP3 numerical-budget/error adapters are not completed. PP4 has
  an admission hook, not a completed memory/loaded-host qualification policy.
- Cleanup precedes host serial sampling. Sampling compares exact encoded results;
  final CI compares projected plan and applied-model checksums. Canonical source
  bytes are checked before/after CI and before fallback, so CI refusal cannot silently
  rerun a mutated source. The small public integration test passed both real stages
  with actual CI; this is not proof for sector60/r7.
- Dispatch seeds classes across workers when possible, then uses deterministic
  largest-cost-first placement and original-order reduction. Pair cost is one plus
  boundary-edge product; face cost is traces squared plus points plus one. Tests
  cover 1/2/8/16 workers and mixed assignments. Actual production stages are
  homogeneous and dependent: each starts a FRESH planner and worker pool. There
  is no cross-stage retained worker pool or startup amortization claim.
- Configurable payload/type-walk limits propagate through host, planner and worker
  bootstrap, model and result transport, gate descriptors and model checks. A real
  4.1 MB extension packet passed with 6 MB/200000-node limits. The legacy diagnostic
  task retains its three-face cap; real stage tasks do not. Whole N2c planning was
  never run. This does not establish N2c runtime capacity or final production limits.
- Deadline and cancellation are terminal typed outcomes, with no serial replay.
  Cleanup failures remain terminal. Infrastructure failures have typed fallback
  after retirement. Finite JSON parsing, one-read verified request hash, independent
  worker reconstruction hash, poll/result race handling, environment-before-import
  and end-of-worker BLAS checks were added or strengthened.
- The evidence runner recognizes only the exact owned system console command after
  Job membership, checks owned Python commands, and records cleanup preflight.
  It uses a 110-second child bound under the 118-second task cap, unique pytest
  basetemp and no cache writes. A historical host-red cleanup failure is retained:
  unexpected owned executable was not recorded by the old guard. PID 59000 was
  later absent, but cleanup is NOT retroactively marked verified. Outer Job handle
  lifetime was kill-on-close. New passing receipts and both sector failures verify
  cleanup with zero active owned processes.

## Mathematical and identity claims

For a checked original record x and constructor R over its dataclass fields f(x),
acceptance requires identical key/schema constraints and binary equality of each
admitted derived array with R(f(x)): dtype, shape, contiguous layout, finite values
and bytes, including signed zero. Thus the checked binary discrepancy is zero;
no positive tolerance is introduced. Both computations may contain the same
floating-point rounding error. There is no new bound on their error relative to
exact real arithmetic and no universal geometry/alias/purity theorem.

RAW NATIVE VIEWS are an ACCEPTED RESIDUAL RISK. The alias domain is the traversed
model graph, owner reference counts, NumPy owner/base flags and GC census. The
from_address counterexample remains a documented non-detected limitation. Fresh
worker constructor arrays, serial sampling and CI are mitigations, not a proof
that raw host aliases cannot exist. Digest comparisons assume no hash collision;
exact byte checks are distinguished from checksum evidence.

Part 1's 471 passing tests, final admission census and three OFF identities apply
to e26e4ab, before Part 2 changes. See PP5C-PART1-EVIDENCE.md and PP5C-CONTRACT.md.
Source has 1083 faces / 540 Cylinder derived arrays / 985821 payload bytes; N2c
source has 561 / 18 / 900300; its closure 561 / 18 / 842545. These are Part 1 census
observations, not a new Part 2 type-walk count. Exact roundtrip and source identity
passed there. Closure permits only the existing absent-to-empty extension transition.

Part 1 OFF checksum identities:
- sector60: 1078d6bf9c5b9c33f84e0d68954fbf89f52e677368b23c5f7641b8c9323df153
- r7: 160166d64deb7b576e6c18bc5fa4dcea5e394b5fc1de896890629b183b1e0e70
- face200: 2ab517de9f4bc32af5f0aada85b234084282f982c313a73224f5bedda400c51f

Part 2 real serial sector60 and both refused parallel requests preserve the first
checksum and source bytes. Serial apply checksum:
d6b136d97cb57a20d1c80e6c15450d61432480d9b5129ff39231d2a0aacf1122.
The stage-codec2 and stage-decoded1 probes preserved sector60 plan/apply golden
checks with transported values returned to the serial host; these do not establish
parallel worker admission. Final source matches every input hash recorded by the
LAST FAILED sector60 receipt (zero mismatches); no final full passing suite exists.

The stage census observed 298 pair-kernel calls and 112 arrangement calls,
including nested calls, not scheduler job counts. Its 12 observed record classes include Plane,
Cylinder, LinePath, ArrangementPath, ArrangementCell,
MaterialArrangement, EllipticArc, CylinderIntersectionCurve, _CylinderSupport,
BezierQuadricCurve, QuadricSupport and PlaneRuledSupport. See the raw census for occurrence counts. Cylinder extras were observed 70 times and Plane.normal 42 times. The initial census label 'warm' for 68 cache dicts
was misleading: the constructor-rebuild probe found zero NONEMPTY caches. Preserve
that distinction; the codec does not admit arbitrary warm state.


| Observed record class | Occurrences |
|---|---:|
| anygeometry.arrangement_geometry.LinePath | 367 |
| anygeometry.branch_curves.BezierQuadricCurve | 68 |
| anygeometry.exact_curves.CylinderIntersectionCurve | 17 |
| anygeometry.exact_curves.EllipticArc | 142 |
| anygeometry.exact_curves._CylinderSupport | 34 |
| anygeometry.material_arrangement.ArrangementCell | 112 |
| anygeometry.material_arrangement.ArrangementPath | 450 |
| anygeometry.material_arrangement.MaterialArrangement | 112 |
| anygeometry.plane_ruled.PlaneRuledSupport | 68 |
| anygeometry.quadric_algebra.QuadricSupport | 68 |
| anygeometry.surfaces.Cylinder | 70 |
| anygeometry.surfaces.Plane | 42 |

ArrangementPoint is additionally supported by the codec; it was not observed in this census.

## Test evidence and corrections

The table below is extracted from saved pytest summaries. Counts are per invocation,
not distinct tests or final-candidate coverage. Expected red failures are retained.
A setup/import failure is not an algorithm regression proof. All exact child argv,
cwd, input hashes, exits, cleanup and stdout/stderr paths are in
.pp1-evidence/PP5C-PART2-COMMANDS.json. Full output remains on disk.

The face-setup-red invocation actually PASSED: it was not a discriminating red
case. The bounds-only attempted fix did not solve the public failure. Pair history
populates Plane.normal; carrying and validating that support manifest fixed the
small public test on its third attempt. The later face-support-red is the real
regression demonstration. Limits-green1's two failures were old injected mocks
missing new keyword arguments; limits-large's first failure was an invalid
extension key lacking a colon, corrected in the fixture. The stage-process failures
were a forbidden cache mutation, resolved by constructor-created views on the third
attempt. None of these earlier successes renew the final closure blocker budget.

| Receipt (pp5c- prefix) | Status | Passed | Failed | Deselected |
|---|---|---:|---:|---:|
| 345-green | passed | 52 | 0 | 0 |
| 345-green-final | passed | 53 | 0 | 0 |
| 345-real | passed | 4 | 0 | 0 |
| 345-red | failed | 1 | 14 | 0 |
| ccepted-focused | passed | 57 | 0 | 0 |
| ccepted-real | passed | 4 | 0 | 0 |
| ccepted-ring | passed | 7 | 0 | 0 |
| ccepted-seam | passed | 14 | 0 | 0 |
| ccepted-t2b | passed | 170 | 0 | 0 |
| ccepted-unit | passed | 219 | 0 | 0 |
| lias-extra-red | failed | 0 | 2 | 19 |
| i-mutation-green | passed | 1 | 0 | 1 |
| i-mutation-red | failed | 0 | 1 | 1 |
| onstructor-cleanup-red | failed | 0 | 1 | 15 |
| onstructor-green-01 | failed | 59 | 6 | 0 |
| onstructor-green-02 | passed | 65 | 0 | 0 |
| onstructor-red | failed | 13 | 6 | 0 |
| onstructor-red-02 | failed | 13 | 6 | 0 |
| ace-setup-green | passed | 8 | 0 | 0 |
| ace-setup-red | passed | 1 | 0 | 7 |
| ace-support-green | passed | 9 | 0 | 0 |
| ace-support-red | failed | 0 | 1 | 8 |
| ield-preflight-red | failed | 0 | 1 | 24 |
| inal-schema-green | passed | 41 | 0 | 0 |
| inal-schema-red | failed | 1 | 3 | 37 |
| inal-unit | failed | 212 | 3 | 0 |
| ost-green1 | failed | 0 | 1 | 0 |
| ost-green2 | failed | 0 | 1 | 0 |
| ost-green3 | passed | 1 | 0 | 0 |
| ost-red | cleanup_failed | 0 | 1 | 0 |
| edger-isolation | passed | 194 | 0 | 0 |
| imits-green1 | failed | 83 | 2 | 0 |
| imits-green2 | passed | 85 | 0 | 0 |
| imits-large | failed | 0 | 1 | 1 |
| imits-large2 | passed | 1 | 0 | 1 |
| imits-process | passed | 6 | 0 | 0 |
| imits-red | failed | 0 | 3 | 0 |
| etadata-red | failed | 0 | 2 | 21 |
| issing-extra-red | failed | 0 | 1 | 23 |
| art1-real | passed | 4 | 0 | 0 |
| art1-ring | passed | 7 | 0 | 0 |
| art1-seam | passed | 14 | 0 | 0 |
| art1-t2b | passed | 170 | 0 | 0 |
| art1-unit | passed | 218 | 0 | 0 |
| unner-green | passed | 1 | 0 | 0 |
| unner-red | failed | 0 | 1 | 0 |
| ealed-real | passed | 4 | 0 | 0 |
| ealed-ring | passed | 7 | 0 | 0 |
| ealed-seam | passed | 14 | 0 | 0 |
| ealed-t2b | passed | 170 | 0 | 0 |
| ealed-unit | passed | 217 | 0 | 0 |
| tage-context-green | passed | 24 | 0 | 0 |
| tage-context-red | failed | 0 | 1 | 6 |
| tage-process-green1 | failed | 0 | 1 | 0 |
| tage-process-green2 | failed | 0 | 1 | 0 |
| tage-process-green3 | passed | 1 | 0 | 0 |
| tage-process-red | failed | 0 | 1 | 0 |
| tages-green1 | passed | 6 | 0 | 0 |
| tages-red | failed | 0 | 6 | 0 |
| ask-float-red | failed | 0 | 1 | 16 |
| alues-alias-red | failed | 0 | 1 | 19 |
| alues-green1 | passed | 16 | 0 | 0 |
| alues-green2 | passed | 19 | 0 | 0 |
| alues-green3 | passed | 20 | 0 | 0 |
| alues-red | failed | 0 | 16 | 0 |
| alues-scalars-red | failed | 0 | 3 | 16 |

## Requested timing evidence

All rows are the 97-face sector60 subset, 32 logical CPUs. CPU is a sampled system
percentage, not a controlled-host utilization guarantee. Launch load and pre-plan
load are separately recorded; unrelated process identities were not inspected.

| Request | Actual outcome | Process wall s | Plan s | Apply s | Before launch Python / CPU % | Before plan Python / CPU % |
|---|---|---:|---:|---:|---|---|
| Serial | Passed serial golden | 32.63777349999873 | 20.13880370001425 | 4.674875700002303 | 1 / 31.5 | 2 / 3.2 |
| Parallel attempt 1, 8 workers | REFUSED; serial fallback | 43.856526700023096 | 38.39707609999459 | not reached | 1 / 0.0 | 2 / 0.6 |
| Parallel attempt 2, 8 workers | REFUSED; serial fallback | 52.43178959999932 | 46.29206800000975 | not reached | 1 / 5.6 | 2 / 6.4 |

Fallback plan duration includes failed parallel work and serial replay. No speedup,
savings, production balance or complete parallel critical-path measurement is
claimed. Stage-level startup/load/task/gate fields exist in saved worker receipts,
but neither sector run accepted a stage; no complete pair-plus-face timing/balance
result exists. Fresh planner/worker startup is paid twice by the implemented design.
Historical PP2 startup figures must not be represented as measurements of this code.

## NOT CHECKED / unfinished acceptance

- Final PP1/planning/T2b/ring/seam/57 groups and three OFF identities after the last
  Part 2 source edits: not run because the closure blocker reached the stop limit.
  Part 1 success does not qualify the modified executor/transport/public route.
- Sector60 accepted parallel plan+applied-model CI, then r7 comparison, complete
  8/16-worker performance, stratified sampling report and balance: blocked before
  sector60 stage admission. r7 timing/parallel tests were not started.
- Full closure across every sector task: the first-task diagnostic was insufficient.
- N2c planning/apply/mesh: not authorized. No whole-model runs performed. N2c final
  payload/type-walk capacity, memory policy and loaded-host qualification unproven.
- Complete PP3 budget/error integration and PP4 memory/host policy: hooks only;
  unsupported policies remain serial. Broader configuration edge cases unverified.
- Universal alias freedom, raw-native-view detection and new real-arithmetic error
  bounds: expressly not established by bounded admission or binary equality.
- Independent Opus/forseti review, final acceptance, integration, publication: pending.
  No push or other remote operation occurred.

## Next release options (not current authority)

An explicit newer inbox release is required. Preserve the current failed evidence.
First add diagnostic reporting of added/removed/rebound module names at the existing
closure refusal, without relaxing the check. Census complete authorized assignments
and establish a deterministic prewarm/import closure, then rerun the sector60 gate.
Only after admitted sector60 equality should r7 and final regression/identity groups
be run within a new explicit budget. Alternatively retain only Part 1 while
redesigning the stage execution boundary. No wildcard imports into the admission
allowlist, hash refresh, tolerance change or serial fallback relabelled as success.

## Artifacts and change locations

.pp1-evidence/PP5C-PART2-CHANGES.json records each tracked diff hunk and each new
source/test/tool/doc at line 1 with function/class anchors and SHA256. Generated
receipt/stdout/stderr paths and hashes are in PP5C-PART2-ARTIFACTS.json (line 1 for
JSON/plain generated files); nested worker files remain under their named receipt
folders. These generated artifacts are intentionally not added to Git.
PP5C-PART2-MANIFEST.json binds the current source to the final failed receipt.
PP5C-PART2-COMMANDS.json contains exact supervised commands, not fabricated reruns.
This handoff generator is .pp1-evidence/pp5c_part2_handoff.py; invoked as
`& 'C:\Python\Python314\python.exe' -B .pp1-evidence/pp5c_part2_handoff.py`.
Git inspection commands: `git status --short`, `git diff --check`, `git diff --stat`,
`git diff --numstat`, `git branch --show-current`, `git rev-parse HEAD`.
The preservation command and resulting SHA are recorded in the append-only outbox.
