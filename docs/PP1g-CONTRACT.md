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

Each is recorded as “trusted runtime image, pinned by versions and binary hashes,
not purity-proven”. These are the canonical inventory paths for the inbox shorthand.
Strict mode still declines them; nothing outside this exact list inherits trust.

**Worker controls.** Reset all 41 registered cache roots per task; their 21 nested
stores remain covered by those resets. The named opaque finfo store belongs to the
trusted-image exception and is not newly cleared. Preserve live implementation
bindings; check their code/reference pins. Every worker must pass the fixed actual
pair/face startup canaries before tasks. Recheck pins and ledger at task boundaries;
changed binaries, extra modules, unknown stores or a canary mismatch decline admission.

**Parent verification.** Hold canonical task digest results until checking
ceil(N/20) evenly spaced samples, minimum two (or all when N<2). Compare exact digest
strings, with no tolerance. A mismatch discards the entire candidate run and reruns
serial tasks. Retire workers before sampling/fallback. Cleanup failures, serial errors
and cancellation propagate. Deterministic sampling does not certify unsampled tasks.
The CI hook compares both plan and applied-model digests; PP6 must wire actual plans.

**Opt-in.** ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING=1 enables the hook. Absent or
other values stay serial without starting workers. This slice launches no production
pool; it provides the default-off control for the later executor.

**Evidence and limits.** E=.pp1-evidence/pp1g-worker-01: a real child passed both
canaries and 40 real small kernel tasks. The actual parent checked pair/face samples
2/40. Startup 2.946483 s; median reset 0.000050400 s; parent sampling 0.020095 s.
CPU point samples: child 7.5% then 7.6%, parent after sampling 6.6%; 32 logical CPUs,
BLAS 1/1. These are finite small-task measurements, not an N2c speed estimate.
Full boundary overhead is separately reported in the outbox. Independent reviews,
production installation, PP2/PP6 integration and full-model acceptance remain open.

**Work plan and acceptance.** Parent-red tests -> trusted-image and fallback hooks ->
real-worker measurement -> final PP1 plus 191/57 regressions and sector60/r7/face200
pool-off identities -> bounded evidence, commit and outbox handoff. No gate is waived.
