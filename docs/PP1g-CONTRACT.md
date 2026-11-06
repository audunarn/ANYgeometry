# PP1g: RISK-ACCEPTED QUALIFICATION, not a purity proof

**Authority and scope.** Seventh inbox release; parent 0e282129; branch
claude/pp1-purity-20261031. This is an internal admission/sampling hook. Production
executor integration remains PP2; whole-plan/apply wiring remains PP6. No remote,
whole-model run, tolerance change or claim of universal bit identity is authorised.

**Pinned trusted image.** The parent and every owned worker must have identical
Python/runtime, NumPy/SciPy versions, loaded binary-file SHA256 hashes, loaded module
closure, source hashes, settings and one-thread BLAS/OpenMP. Hashes identify loaded
image files, not in-memory machine-code pages. The trusted-image record names exactly:

- anygeometry.backend._RUNTIME_PYTHON_GLOBALS.__builtins__
- anygeometry.backend._RUNTIME_TRUSTED_NUMPY[12][1]._finfo_cache
- anygeometry.backend._RUNTIME_POINT_EDGES[3][0]
- anygeometry.backend._RUNTIME_POINT_EDGES[33][0]
- anygeometry.backend._RUNTIME_POINT_EDGES[52][0]
- anygeometry.backend._RUNTIME_POINT_EDGES[61][0]
- anygeometry.exact_curves._BranchConstants.__annotate_func__.cell[0]

PP1i clarifies these canonical paths: five stores are checked against their
actual builtins/NumPy module namespaces, finfo against its actual owning class,
and the owned annotation dictionary against its annotate-function closure.
The annotation dictionary additionally has a structural content digest at every
boundary and source-file provenance. The six runtime dictionary contents are
NOT reset or fully content-checked. File/binary hashes do not pin their mutable
bytes. All remain risk-accepted and not purity-proven. Strict mode still declines
them; nothing outside the exact list inherits trust.

**Worker controls.** Reset all 41 registered cache roots per task; their 21 nested
stores remain covered by those resets. The named opaque finfo store belongs to the
trusted-image exception and is not newly cleared. Preserve live implementation
bindings; check their code/reference pins. Every worker must pass the fixed actual
pair/face startup canaries before tasks. Recheck pins and ledger at task boundaries;
changed binaries, extra modules, unknown stores or a canary mismatch decline admission.

**Parent verification (updated PP1i).** New pool callers must provide actual
worker id, execution position and pair/face class per task plus the plan digest.
Verify each worker's last task and each worker/class tail, with at least
min(Nw,max(2,ceil(Nw/20))) samples per worker. Additional samples use a recorded
plan-digest-derived offset in the late half. See PP1i-CONTRACT.md for the exact
selection. The legacy metadata-free API retains its old stride only as a
diagnostic compatibility path and cannot establish worker coverage.

Hold all digests until checking exact equality with no tolerance. Mismatch
discards the whole candidate and reruns serial. Retire workers first; cleanup
failures, serial errors and cancellation propagate. Deterministic sampling does
not certify unsampled tasks. require_ci=True mandates both plan/apply controls
and dispatch metadata before startup; PP6 must use it with actual plans. Without
that flag and with both CI arguments None, CI is explicitly skipped.

**Opt-in.** ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING=1 enables the hook. Absent or
other values stay serial without starting workers. This slice launches no production
pool; it provides the default-off control for the later executor.

**Evidence and limits.** E=.pp1-evidence/pp1g-worker-01: a real child passed both
canaries and 40 repetitions of those same canaries (20 pair, 20 face). The actual parent checked pair/face samples
2/40. Startup 2.946483 s; median reset 0.000050400 s; parent sampling 0.020095 s.
CPU point samples: child 7.5% then 7.6%, parent after sampling 6.6%; 32 logical CPUs,
BLAS 1/1. These are finite small-task measurements, not an N2c speed estimate.
Full boundary overhead is separately reported in the outbox. Independent reviews,
production installation, PP2/PP6 integration and full-model acceptance remain open.

**Work plan and acceptance.** Parent-red tests -> trusted-image and fallback hooks ->
real-worker measurement -> final PP1 plus 191/57 regressions and sector60/r7/face200
pool-off identities -> bounded evidence, commit and outbox handoff. No gate is waived.
