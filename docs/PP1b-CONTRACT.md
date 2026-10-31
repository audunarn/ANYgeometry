# PP1b engineering admission contract (second release)

Authority: CODEX-INBOX-allfader PP1b, parent 8d29f99b, same worktree/branch.
This release replaces the attempted universal proof with pinned engineering
qualification and controls. It does not change numerical tolerances or repair
serial signed-zero cache keys. No pool or full-model run is enabled here.

Plan: (1) write parent-red negatives for unknown modules/types, altered artifact
hashes, non-single-thread BLAS, canary mismatch and cache reset; (2) implement
snapshot comparison, mandatory reset-or-decline and startup canary orchestration;
(3) exercise real pair/face canaries, existing focused tests and the three serial
identity gates, retain exact evidence, commit and report. Independent reviews
remain with Opus/forseti. No delegation is permitted for this file-managed task.

Proposed boundary: an owned, single-thread worker snapshots its loaded module
files, Python executable/runtime, numpy/scipy versions, native BLAS files/identity,
CPU identity and effective thread counts against an independently supplied pin.
Pins are baseline inputs, never updated automatically to accept a candidate.
Only explicitly pinned loaded types are accepted in task inputs. Modules or types
outside the pin, missing files, changed hashes or unverifiable thread counts refuse.
Current-file hashing is an engineering control, not loaded-image attestation.

Every supplied declined store is emptied before each canary and task; every
store must have a concrete supported reset operation and empty postcondition.
Unsupported stores refuse before task execution. Reset is allowed only in the
owned worker, never the GUI/serial process. Definitions/metadata are not silently
treated as disposable caches. Caller must supply its complete task cache registry;
an incomplete classification is not permission to clear arbitrary Python state.

Startup requires fixed expected digests for real pair/face work. Mismatch or
exception poisons the gate; no task result is admitted. Check code/environment
again after canaries and task execution. PP6 will compare serial and partitioned
plan AND applied-model digests with the same projection and expected inputs;
the equality hook is fail-closed and cannot update its reference.

Risks/open work: reset may cost more than a warm cache; no timing claim. Canaries
are finite coverage, not proof of every geometry or cache equivalence. PP2 owns
process lifecycle and the complete task registry binding; PP6 owns full plan/apply
wiring. Whole-model and independent acceptance remain unperformed in this slice.

Status: **BLOCKED / partial implementation**, not permission to release PP2.
Twelve new tests were parent-red, then green; the final focused new/PP1 group
passes 33 tests. Real baseline canaries ran two clipped plane-pair tasks and one
four-cell face arrangement with fixed expected digests. Startup admission has
not succeeded in a fresh worker. Three pin-assembly attempts were retained:
origin-less Cython namespace; descriptor-valued native type name; then refusal of
a native type without a static qualified name. No fourth attempt is authorized.
Known Cython namespaces are explicitly linked to the hashes of the complete
loaded native-extension set; other origin-less modules continue to decline.
The reset-registry names/types must exactly match the pin. The rehearsal names
the three reviewed LRUs; it does not classify every full-planner store, and no
full-planner task is admitted. Clearing arbitrary metadata/authority is not claimed.

Next-release option: keep unnameable native types out of the task-input allowlist
(still decline such inputs), while retaining explicit native-module/provider pins;
then qualify the resulting complete worker startup and task-specific reset ledger.
This option is not implemented here. A reset can expose a different first witness
than warm serial execution; canaries and the PP6 plan/apply control detect only
their exercised cases. DET-1 remains separate. No universal equivalence claim.

Final compatibility: 281 tests pass (33 new/PP1 + 191 + 57 existing), with three
retained overflow warnings. sector60, r7 and face200 retain their historical
digests; no tolerance/gate changed. Independent review and worker admission are
not complete. Evidence and failed attempts: `.pp1-evidence/pp1b-*` in this worktree.
