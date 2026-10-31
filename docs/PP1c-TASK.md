# PP1c, third release

Continue 9e4f3609 on claude/pp1-purity-20261031 in this worktree only.
Plan/criteria: non-numerical module-binding preflight first; parent-red tests for
removing exported-native-type discovery, explicit model input types, complete
ledger coverage/refusals and active contexts; implement; assemble a reference
environment.json and run one owned fresh worker with fixed real pair/face canaries;
measure reset/startup and warm/reset digest equality; focused and serial identity
gates; commit/report. Each process <=118 s, one heavy process, max three attempts
per problem. No whole-model run, remote, delegation or independent-review claim.

Engineering boundary: native binaries/BLAS are file-pinned; no exported native
type enumeration. Task input adapters have an explicit exact-type allowlist.
The full loaded geometry-store census gets a ledger row: reviewed numerical
caches reset, definition/runtime metadata file-pinned, idle context/recursion
state must be empty, descendants covered by a parent reset, anything else declines.
No invented reset for opaque objects, no clearing GUI state or active budget
contexts. This is engineering qualification with finite canaries, not a universal
purity proof. The ledger and any remaining declines are reported in full.

Outcome: verified against PP1c's bounded worker criteria; independent Opus/forseti
acceptance pending. No pool switch or executor integration is enabled by this work.
The non-numerical preflight reports each module/binding failure and scanned 672
module bindings without executing canaries. The assembled reference pins 702
loaded modules and 642 files, including numpy 2.4.6, scipy 1.17.1 and two OpenBLAS
libraries. Exported native types are not used to construct the input allowlist;
20 explicit primitive/NumPy/model/domain types have adapters. Unknown exact types,
object arrays and warm domain implementation caches decline. GeometryModel input
definitions pass through the existing owner serialization checks.

Ledger: 1,394 observed stores, all covered: 41 primary resets, 21 descendants
covered by those resets, 267 context/recursion/default stores required inactive,
1,065 protected definition/runtime metadata stores tied to pinned code/runtime.
Zero uncovered/declined rows for the exercised closure. This is the full loaded
geometry census, not a promise that every unexecuted lazy route is qualified.
Any newly loaded module fails the environment pin. Arbitrary runtime rebinding or
hostile in-process code injection is outside this controlled-worker contract.

Final owned worker: startup checks/canaries 1.0621025 s (excludes interpreter/import
startup); task resets 0.57199999 and 0.57270000 ms; startup resets 0.58500000 and
0.57500000 ms. Aggregate CPU 3.5% before / 1.0% after, 32 logical CPUs, both BLAS
libraries one thread. Clock resolution 1e-7 s. Small-case observations only.
Warm and reset pair/face digests exactly match the fixed references; no difference
on these tasks. Signed-zero counterexamples still demonstrate that reset is not
universally equivalent to warm serial history. No serial key repair (DET-1).

Validation: nine new regression cases fail the parent (five initially, four added
cases by read-only parent-module replay); final admission groups 42 passed,
existing focused groups 191 + 57 passed (3 retained overflow warnings).
The 191 group predates only the isolated exact-type identity comparison fix;
its numerical sources/import path are unchanged. All three final serial identity
checks preserve sector60/r7/face200 digests and face200 area exactly.

Retained failures: initial red tests, inactive-False context misclassification,
JSON tuple/list pin transport mismatch, and overloaded-metaclass equality bypass.
Each was tested before its fix. Three reference assemblies and three owned worker
rehearsals only. The last source pin was explicitly versioned for the single-line
type-identity fix, proving all other pinned files unchanged; numerical references
were not recalibrated. See .pp1-evidence/pp1c-type-fix-pin/delta.json.

NOT CHECKED: whole tripod, broader/unloaded geometry routes, full plan/apply or
mesher-entry parallel equivalence (PP6 hook only), complete suite, statistical
performance, loaded-image mathematical attestation, arbitrary mutated model or
callback state, independent reviews. PP2 still needs its explicit release.
