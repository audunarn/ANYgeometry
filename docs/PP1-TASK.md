# PP1: conservative purity prerequisites

Authority: PP1 first release, CODEX-INBOX-allfader; parent
21ede55f8124e382297b1f11935c07d3c26c89bd, branch claude/pp1-purity-20261031.
Only this worktree and append-only outbox may change. No delegation, remote,
whole-model run, tolerance change or production pool activation.

Plan and prospective acceptance:
1. Add regressions for warm weak stores, hidden/aliased state, signed-zero LRU
   first witnesses, late imports and H0 loaded-code skew. Preserve baseline
   failures in .pp1-evidence. No production implementation before the red run.
2. Implement internal, opt-in prerequisites: actual-type store discovery,
   explicit conservative decisions, Python loaded-code fingerprints and scoped
   import guards. Unknown state/native provenance and incomplete closure refuse.
   No cache clearing or changes to the original serial planning path.
3. Run focused new/existing tests and sector60/r7/face200 identity checks in
   individually bounded processes. Commit only source/tests/docs, report exact
   commands, counts, decisions and remaining gates.

Risks: a Python fingerprint is not native-image attestation; a sys.meta_path
guard alone cannot block cached imports, reload or arbitrary loader execution.
Do not label the complete numerical closure admitted while these proofs are
missing. Fixed/task-scoped cache decisions require evidence beyond inventory;
this slice may explicitly decline stores rather than change legacy semantics.
PP2-PP6 integration, native admission, review and final acceptance remain separate.

State: PARTIAL IMPLEMENTATION; PP1 admission is BLOCKED, not ready for PP2.

Implemented (unwired internal prerequisites): bounded actual-type inventory with
aliases and conservative store refusals; Python-only code identity; static local
import discovery; scoped sticky import/reload guards. Original serial modules,
cache behavior, tolerances and gates are unchanged.

Verification so far: 21 new tests failed on the parent because the prerequisite
API was absent. Before reaching that absence, both signed-zero tests established
actual first-witness collisions in existing caches. First candidate: 17 passed,
4 failed. Python 3.14 LOAD_SMALL_INT import operands were then handled. Import
guard fixtures were corrected to name only math, rather than accidentally
including unrelated custom pytest module types; custom modules remain explicitly
refused in the descriptor test. No assertion or tolerance was removed.
Second candidate: 21 passed. This is three regression-suite attempts including
the required red run; no further implementation iteration is made in this release.
Existing groups: 191 + 57 passed (3 existing overflow warnings in the latter).
Full stdout/stderr and owned-process cleanup receipts are in .pp1-evidence.

Observed registry: 130 loaded anygeometry modules, 1,389 stores, 6,388 unknowns,
19,936 aliases, no bound truncation. All stores declined. This is the observed
loaded set, NOT the complete possible planning closure. Instance __dict__ access
is deliberately declined where a native/custom getset cannot safely be invoked;
the walker does not manufacture completeness by executing descriptors.

Blocking limits found by inspection:
- ImportFreeze alone cannot intercept retained direct-loader aliases or arbitrary
  native/dynamic code execution. Its entry-point/meta-path/membership checks are
  useful diagnostics, not proof of the whole admitted closure.
- eager_closure is conservative static discovery, not a complete callable graph.
  Class dispatch/native module bodies decline; arbitrary object/reflection paths
  are not qualified. It is not a production admission controller.
- python_token binds tested Python code/defaults/closure/global values; it is NOT
  an attested native-image/runtime token. Full callable metadata and arbitrary
  decorators/reflection are not qualified. A Python-only H0 test is not GUI-to-
  planner native attestation. The observed numerical module set declines.
- No complete host/planner admission lifecycle is wired. None of these helpers
  may be used alone to enable numerical workers. No new parallel work is admitted.

Next release options: retain the declined numerical closure and specify a narrow,
fully qualified callable/import/native admission boundary; or complete the broader
loaded-image/runtime/loader proof before consuming these prerequisites. Neither
option is implemented or authorized by this note. Review remains independent
Opus/forseti work; this author has not performed it.

Final serial identity checks (pool workers 0, mode exact, seam override unset):
- sector60: 97 faces; plan checksum
  1078d6bf9c5b9c33f84e0d68954fbf89f52e677368b23c5f7641b8c9323df153;
  apply passed, 110 vertex-on-face records, zero over tolerance.
- r7: 240 faces; plan checksum
  160166d64deb7b576e6c18bc5fa4dcea5e394b5fc1de896890629b183b1e0e70;
  apply passed, 360 vertex-on-face records, zero over tolerance.
- face200: canonical summary
  2ab517de9f4bc32af5f0aada85b234084282f982c313a73224f5bedda400c51f;
  area exactly 4.218529952235817. Historical canonicalization excludes only
  origins and timing fields. All three equal the prior recorded references.
Each child asserted that the new module was not imported by the pool-off path.
Nine supervised receipts: seven successful, two retained failures, zero timeouts;
all within cap, each cleanup verified zero remaining owned processes.

Mathematical scope: binary64 +0 and -0 differ in the sign bit but compare and
hash equal. A value-keyed LRU therefore identifies two representations. The
tests exhibit a directrix whose retained control sign differs and bounds whose
synthetic evaluator returns +1 versus -1. A key equivalence that identifies
different observable outputs is insufficient for partition-independent reuse.
This is an exact discrete counterexample, with no rounding allowance or new
geometric certificate. Declining those stores avoids asserting purity; it does
not repair legacy history dependence. Historical digest equality is bounded
regression evidence, not a universal determinism/native-identity proof.

NOT CHECKED: complete admitted closure, retained direct-loader/reflection paths,
native image/dependency/CPU dispatch identity, full function-metadata identity,
formal S0/D3 runtime controller, GUI/planner integration, numerical worker
admission, complete suite, whole-model run, performance, independent reviews.
These are unproved/outside this bounded partial delivery, not waived gates.
Early red/candidate source hashes were not captured contemporaneously; preserved
stdout contains the failures. Final files/receipts are hashed in the manifest.
Ignored pyreadline3 shutdown errors remain in pytest stderr; focused-b also
retains three expected overflow warnings. Process exit and pytest counts are
reported separately; neither stderr nor failed runs were removed.
