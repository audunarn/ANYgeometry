# PP2 owned planner skeleton (release history and current contract)

Plan and acceptance, before implementation:

1. Add strict checksummed JSON transport and cold-model admission. Tests must fail
   on 41c20b0a because the PP2 API is absent. Unknown state, corruption, wrong
   threads, incomplete results and worker refusal must select original serial.
2. Add a bounded Windows owned planner and nested workers. No pickle, remote
   service, GUI process mutation or production plan publication. Default OFF;
   default requested pool size eight, sixteen optional, two for the small check.
3. Use the existing PP1 canaries, trusted-image qualification and per-task resets.
   Default chunk is the whole finite assignment; smaller positive K is explicit.
   Hold all results until every worker ends and parent samples pass.
4. Measure at least 600 transported small-model tasks for whole-assignment K and
   K=256, with requested aggregate load and startup overlap accounting. Run the
   PP1 plus 191/57 groups and all three unchanged pool-off identities afterwards.

Risks/open boundaries: this slice returns task content digests, not sealed plans.
PP3 owns production error attribution, cancellation and global numerical budgets;
PP4 owns memory admission; PP5/PP6 own stage wiring and host-plan publication.
Those options must decline in this skeleton. No whole-model runs are authorised.
Independent review and acceptance remain coordinator-owned. No certified guards
may be refreshed. Direct implementation stays with allfader under the task's
file-only worker protocol (no delegation).

## Handoff: BLOCKED, partial implementation

This section records release 1. Release 2 below explicitly authorises the exact
extension adapter; the old failed evidence remains unchanged in the outbox.

The 24 new tests failed on the base because the new API was absent. Candidate
check 1: 22 passed, 2 failed. Candidate check 2: 23 passed, 1 failed. The remaining
failure is the cold decoded model round-trip: serialization.from_dict adds an
empty `_serialization_extensions` field, whereas the existing PP1 check_types
gate requires exactly the constructor's instance fields. freeze_model correctly
propagates that refusal. No field was stripped, no gate or certified pin changed,
and the failing acceptance test remains in place.

The code is an uncommitted candidate, NOT an accepted executor. Real process
containment, per-worker qualification, the two-worker equality case, the 600-task
K matrix, overlap timings and final regression/identity groups have NOT RUN.
No performance, ordinary-output identity, transport completeness or production
admission claim follows from the 23 unit passes. Refusal/cancellation tests use
injected process lifecycle objects; they do not verify Windows job cleanup.

A follow-up release must settle the model-state adapter contract: explicitly
qualify decoder-owned extension state (including nonempty/behavior-bearing and
alias negatives) with preserved PP1 admission guarantees, or select a different
transport boundary that meets those guarantees. Simply deleting the field or
bypassing check_types would not establish the required claim.

Additional incomplete design work: bind host loaded-code identity (file hashes
alone do not attest it); distinguish source mutation/deadline from replayable
qualification refusal; ensure unsupported numerical budgets/callback policies
cannot enter this diagnostic API. PP3/PP4/PP6 integration remains open. The
three-face task explicitly uses the PP1 diagnostic scale, not general model
planning; it must never be advertised as a production stage adapter.

## Release 2: exact adapter and real owned processes

Base 6560336162b94c0fcd2e9caedfcced7fa5ae1945, same PP2 branch/worktree.
The coordinator's second release authorises the decoder-owned extension schema,
adoption of PP1i and real-process checks. No source pin guarding certified code
is updated. PP1's remaining checks, numerical tolerances and goldens stay intact.

The exact model key set is either the original constructor set or that set plus
`_serialization_extensions`. That field must be an exact dict. Its value graph
allows only exact dict/string keys, lists, None, bool, int, finite float and str.
No container cycles/repeated aliases or sharing with the other declarative model
state is admitted. Existing serialization namespace/schema constraints also apply.
Canonical JSON SHA256 records the extension structure and signed zero; the model
transport envelope includes and verifies this record. No field is stripped and
no caller object is mutated to obtain admission. Decode must reproduce the full
serialized document by canonical bytes (including signed zero), not Python value
equality. An absent source extension may decode as the decoder's
known empty dict only. This is not arbitrary-object or behavior transport.

OwnedPlanner starts a fresh bounded Windows planner at construction, optionally
while the caller loads its project; it must be closed even if never submitted.
The planner starts nested owned workers, sets thread environment before import,
and verifies actual one-thread pools. Each worker uses PP1i trusted-image/ledger
admission, actual canaries, resets and final staged-result checks. Every child is
assigned suspended to a kill-on-close Job before resume. Cleanup inspects only
that Job's members and their commands; the exact system-directory conhost.exe
0x4 helper is recognised as an owned Windows console child. No image-name kill
or unrelated process scan is used. Anonymous Python counts are measurement load
metadata only. Requests/results are size-bounded checksummed JSON, never pickle.

Default requested pool size is eight; sixteen is accepted as an option. Real
checks/measurements use two. Default K is the full finite per-worker assignment;
smaller K is explicit and coalescing is enabled. A nonempty whole assignment
has FOUR heavy checks including startup. K=256 with300 tasks has FIVE. Worker
results remain private until all workers finish and nested cleanup is verified.
Host checks coverage, source/model identity and mandatory PP1i stratified sampling
with actual (worker,position,class) rows and the canonical dispatch-request digest
(model, ordered tasks, worker count, K and source files), not only the model digest.

execute always uses require_ci=True for acceptance. A caller must supply both
expected and observed CI components. This skeleton's tests compare ordered task
digests and unchanged model content; these are diagnostic controls, NOT the real
PP6 plan/applied-model integration. Missing controls decline to original serial
after owned cleanup. Default OFF never inspects the model or starts a process.
Injected lifecycle unit tests are explicitly separate from real-process tests.

Real cancellation is a host-owned cancel.flag, checked during the transaction;
it terminates the owned Job tree and raises PlannerCancelled without replay.
This is bounded process cancellation, not cooperative kernel cancellation/error
attribution or global numerical-budget parity (PP3). The diagnostic task API
returns pair/face content digests for a three-face model at the PP1 canary scale;
it does not create a public IntersectionPlan, execute production stage seams,
publish/rebind host topology or apply a model. PP4 memory/host-loaded-code admission
and PP5/PP6 production integration remain open. File hashes do not attest arbitrary
in-memory host monkeypatches. No production enabling/default change is made.

Measurement matrix:600 alternating pair/face diagnostic tasks, two workers,
K=whole and K=256. The first pair of measurements is retained as preliminary;
repeat both after the canonical-roundtrip regression repair and again after
binding sampling to the complete dispatch request. Only the last pair is the
final source-bound comparison; all earlier observations remain preserved.
Every result is also compared against a full ordinary
serial sequence after cleanup. Report each worker's assignment cost minus the
same assigned serial-task sum as an observed difference, not a confidence bound.
Startup overlap is the actual tiny JSON model load overlapping planner startup
on Windows' shared performance counter; no artificial wait or inferred N2c gain.
Record anonymous Python count, CPU and native-thread counts. No whole-model run.

Acceptance: final PP1/PP2 unit and real-process tests, 191/57 groups, three unchanged
pool-off identities, both measurements, actual wrong-thread startup refusal,
source/receipt audit and commit. Independent review/production release remain
coordinator-owned. Cancellation failures and all partial evidence are retained.
