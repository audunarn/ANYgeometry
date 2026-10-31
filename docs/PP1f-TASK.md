# PP1f, sixth release

Authority: inbox PP1f; branch claude/pp1-purity-20261031; parent 2b545a68.
No remote, push, other worktree edits or delegation.

Plan and acceptance:
1. Parent-red tests for registered live namespaces, code pins, mutation refusal,
   exact constant state and mandatory worker reference pins.
2. Add bounded binding/code digest without erasing namespaces. Explicitly classify
   every one of the 22 observed nonempty stores. Decline unproved state.
3. One N2c 8-worker forward counts-only call-reachability trace in PP1F scratch,
   cap 300 s. No exclusion based merely on absence of Python call events.
4. Attempt actual worker admission; report reset/startup only if admitted.
5. After last source edit, PP1 + focused 191/57 tests and three pool-off identity
   checks. Inspect evidence, commit authorised files, append full outbox handoff.

Risks: binding identity is not deep purity of opaque objects; live code pins are
reference measurements, not proof of source-to-bytecode provenance; ForwardRef
state and NumPy runtime namespaces may require explicit declines. Trace uses the
old prototype base and cannot establish candidate worker acceptance.

Implemented boundary: one registered implementation namespace has local binding
and function/code identity digests, owner source hashes and portable code SHA256
reference pins required by WorkerGate before/after tasks. Attested objects are
retained to prevent ID reuse. Fifteen exact named constant stores use typed content
digests; six opaque runtime stores remain explicit declines. ForwardRef remains
one explicit decline; no evaluation or name-only approximation is used.

Evidence found during review: a new weakref regression demonstrated that a
binding-only ID table could lose the attested function/code objects. Both function
rebinding and code replacement cases failed before retention was implemented.

The single authorized N2c 8-worker forward prototype trace completed: 5139 pairs,
561 face tasks, 561 arrangements; 17 files, 157187413 Python call events and 1283
symbols. No apply/mesh ran, all eight workers exited zero, and owned-job cleanup
reported zero active processes. No stores are excluded. Python call tracing cannot
prove absence of native calls or global/attribute reads; the prototype base is not
the candidate worker. The observed absence of reference.bspline Python calls is
not an unreachability certificate.

Actual candidate worker admission refused before its first canary, with 1395
stores: 1039 pinned-definition, 271 require-empty, 41 reset, 16 require-live-binding,
7 decline, 21 covered-by-reset. All 65 watched stores remained unchanged; zero
resets executed. Successful canary startup and real per-task reset cost remain
unmeasured. The six runtime declines plus ForwardRef require an explicit follow-up
qualification/risk decision; this task does not relax the refusal policy.

Checks and measured costs are in .pp1-evidence/pp1f-* receipts and the append-only
outbox. Ordinary-output identity is only claimed for the three required gates.
No source-to-bytecode provenance proof, universal purity proof, cross-process code
pin portability, independent review, production acceptance or publication claimed.
