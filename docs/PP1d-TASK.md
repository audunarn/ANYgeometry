# PP1d fourth release

Authority: inbox PP1d, parent 27a1ec30, existing pp1 worktree/branch only.
No delegation, pool activation, whole-model run, remote operation or DET-1 edit.

Plan and acceptance:
1. Add parent-red regressions for F1 live rebinding/new stores, F2 nested default
   memos and protected-content changes, F3 callable binding, F4 extra instance
   state/live inputs, F6 interpreter pins. Preserve the original failures.
2. Implement fresh census at each boundary, direct-object definition roles,
   content validation, module/global callable binding, and a narrow immutable
   payload boundary decoded after reset. Unknown stores/codecs decline.
3. Measure file verification and reset costs in an owned bounded process; retain
   full preflight failures instead of broadening the definition allowlist to pass.
4. Run focused regressions and unchanged serial identity gates. Commit and append
   exact commands, counts, identities, measurements and remaining gaps to outbox.

Dependencies: step 3 uses the final boundary from step 2; no acceptance claim
before verification. Risks: stricter classification can expose previously hidden
unqualified state; a declined full closure is a blocker, not a reason to weaken
the gate. The input codec is intentionally narrow; PP2 transport remains open.
F7/F8 and broad value-equal-key analysis remain explicitly open. Reset removes
inter-task cache history; it does not establish equality with warm serial history
or remove collisions within a task. Independent review remains external.

## Implemented boundary and remaining blocker

F1: each prepare and post-task check repeats the census of the declared module
objects. Module bindings, every non-reset-descendant store path, object identity
and type must remain unchanged. Rebound caches decline. Fresh stores outside a
reset root decline. New descendants of a known reset root may be removed by that
root's concrete clear operation. No snapshot object handle substitutes for the
live census. This deliberately may refuse harmless changes in alias traversal.
Aliases from outside reset roots into a reset descendant decline: clearing a
parent container would not clear the externally retained descendant itself.
The real gate requires this ledger; the injected observer remains an isolated
unit-test seam and cannot establish real admission evidence.

F2: the inventory records direct metadata object roles at actual module, class and
function attributes. Neither an arbitrary path component nor a fixed-root prefix
protects descendants. Exact fixed roots protect only themselves. Supported content
has a bounded type-tagged SHA256 digest, checked before reset and after tasks.
Scalar type and IEEE float bits are encoded distinctly. Unsupported leaves decline;
no repr, pickle or opaque-object wildcard blesses their contents. Function/type
symbols are process-local identity leaves; this is not F7 loaded-code attestation.

F3: only a top-level Python function bound at the pinned module attribute, using
that exact module globals dictionary and no closure, can execute through the gate.

F4: direct live arguments decline. `run_json` accepts immutable bytes and an
expected SHA256; after reset it validates the checksum, decodes fresh JSON data,
checks its schema and input types, then calls the bound task. Duplicate keys,
invalid JSON, nonfinite constants and unsupported payload shapes decline. No
pickle/custom decoder/factory is admitted. Model/array transport is NOT implemented.
The existing model/dataclass inspection adapters also reject top-level extra
instance state, cached-property values, declared warm caches and extra state in
model dataclass records before serialization can discard it.

F5: full file hashing remains enabled before and after tasks. No stat-only cache
or assumed performance saving is introduced. The dedicated measurement driver
records both captures/verification, aggregate machine load, clock resolution and
a separate successful one-store reset microfixture. Full-ledger reset is NOT
measured when the ledger declines; a microfixture does not stand in for that cost.

F6: built-in/frozen module pins name hashes of the actual interpreter/runtime DLL
paths resolved through this process's handles. The Windows adapter records
python314.dll, loaded python3.dll, vcruntime140.dll and vcruntime140_1.dll here.
The launcher executable remains a separate environment file. Missing/unresolved
runtime provenance and unsupported platforms decline.

The strict full census currently declines hundreds of previously protected
stores: notably dataclass Field contents, other unsupported metadata leaves,
annotation closure dictionaries, and mutable descendants of runtime binding roots.
This is a **BLOCKED worker delivery**, not qualification of the production closure.
No real startup canary is launched after that refusal. Completing admission needs
explicit, tested content adapters/decisions for those stores; restoring the old
path wildcard would reintroduce F2. Finite unit checks do not supply that missing
argument. PP2 remains unreleased.

## Determinism, equality and open items

Given a complete admitted closure and successful reset, prior task cache contents
cannot select the next task's first witness. This conditional history-independence
argument does not establish equality to the serial planner's warm history. Equal
keys can still collide *within* a task. Python dictionaries equate `1`, `1.0`, `True`
and compatible NumPy scalars; untyped LRU compound keys can do so as well. Typed-bit
digest tests do not repair those numerical cache keys. All 41 reset-root key/value
semantics across these families remain OPEN except the previously exercised three
signed-zero examples. DET-1 is a separately owned branch; it is not inspected,
changed, merged or claimed here. Full serial/parallel plan/apply parity remains PP6.

F7 (loaded Python bytecode/GUI-versus-worker identity) and F8 (independent reference
provenance) remain open as directed. Existing canary reference bytes are unchanged;
this release does not independently establish their provenance or regenerate them.
Independent Opus, Sol-high and forseti checks are not performed by this worker.

SHA256 comparisons are engineering integrity controls subject to collision and
trusted-runtime assumptions, not mathematical native-image proofs. No geometry
tolerance or error bound changes. Timing observations include scheduling and I/O
effects; clock resolution bounds only quantization, not measurement variability
or future task cost. Full-run performance and statistical confidence are unmeasured.

## Verification result

Parent 27a1ec30: the initial 12 regressions failed before implementation. Exact
read-only parent-module replays later show 21 failures, plus the separately added
external-alias regression failing (22 distinct new cases). Four cases require new
API/digest primitives absent on the parent; the remaining failures expose behavior.
Additional candidate-red evidence caught nested-model extras and the missing
mandatory-ledger requirement before those corrections. All failed evidence remains.

Final boundary group: **64 passed** (22 new + 42 prior). Existing focused groups:
**191 passed**, **57 passed / 3 overflow warnings**. Total reported groups: **312
passed**. No numerical source, tolerance or numerical regression test was changed.
Existing gate tests were adapted to bound top-level functions and explicit live
module roots; their reset, mutation-refusal and canary-mismatch assertions remain.
The 191/57 groups precede only final changes to the unwired gate/ledger checks;
their inputs do not import these modules. They are reused by impact inspection,
not represented as reruns after those final edits.

Pool-off identity: sector60 97 faces, checksum
`1078d6bf9c5b9c33f84e0d68954fbf89f52e677368b23c5f7641b8c9323df153`;
r7 240 faces, checksum
`160166d64deb7b576e6c18bc5fa4dcea5e394b5fc1de896890629b183b1e0e70`.
Both applied, respectively 110/360 vertex-on-face checks, zero over tolerance.
face200 canonical checksum
`2ab517de9f4bc32af5f0aada85b234084282f982c313a73224f5bedda400c51f`,
area exactly 4.218529952235817. These finite projections are the ordinary-output
identity evidence; no universal identity claim or PP6 acceptance follows.

Final preflight/cost artifact: `.pp1-evidence/pp1d-measure-alias-final/`.
1,395 stores: 651 protected, 413 declined, 269 require-empty, 41 reset roots,
21 covered descendants. 700 modules / 644 files. Both full hash verifications:
0.250475 s and 0.247934 s, total 0.498409 s. Final refusal took 0.152230 s;
no full-ledger reset or numerical startup task occurred. The separate 1,000-entry,
one-dictionary fixture reset took 0.0000129 s, census 0.0004602 s; these are NOT
production-task reset costs. Load: aggregate CPU 3.8% before / 8.5% after,
32 logical cores, both BLAS pools one thread. Clock resolution 1e-7 s. No repeat
variance estimate: three observations accompany changed source, not a statistical
sample. Interpreter/BLAS/geometry evidence and all exact commands remain on disk.

Status: **BLOCKED**, candidate ready for review, production admission unproved.
No whole-model run, push, independent review, or next-task claim. To continue,
a newer release should own explicit content adapters and mutable-store decisions
for the retained decline inventory, then requalify startup and measure actual
task reset cost. F7/F8, DET-1 value-equal keys and PP2 remain separate open work.
