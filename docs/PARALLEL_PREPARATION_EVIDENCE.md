# Parallel handoff development evidence

2026-10-06. Requests 1–3 of ANYmesher's `ANYGEOMETRY_PARALLEL_HANDOFF`
are implemented on `codex/parallel-handoff`, based on geometry
`c6dc4261995a1e0c1ac9c19ba8ed96131d24b880`. The public contracts and
explicit refusal cases are in [PARALLEL_COMPONENT_HANDOFF.md](PARALLEL_COMPONENT_HANDOFF.md).
No version/schema change, package release, native mesh acceptance, new
authored-region permission or relaxation of scientific gates is included.

## Measured connected preparation

Frozen plain documents represent a base plate and N/2 X webs plus N/2 Y webs,
pitch 1, web height 0.5, base side N/2+1. Every face has its own sheet.
Exactly the same UUIDs, revisions and local IDs are used on both inputs.
Each child queries the original faces, plans all sheets and applies once.
No mesher is imported. Document and immutable chart-definition hashes are
computed outside the timed path. All 27 baseline/candidate runs retain identical
prepared documents, chart definitions, topology sizes and joint-edge counts
for their respective frozen inputs.

Times below are means of two unprofiled fresh-process runs in seconds;
a third separate cProfile run records call counts. Windows, Python 3.14.2,
NumPy 2.4.6, same machine/runtime. Six numerical-thread environment variables
were set to 1; native thread-pool enforcement is not separately established.
These are measured geometry preparation savings on these fixtures, not
end-to-end mesher speed claims or a portable runtime promise.

| Webs | Baseline individual queries | Candidate individual queries | Candidate per-face batch | Batch reduction |
|---:|---:|---:|---:|---:|
| 16 | 1.755 | 1.592 | 1.540 | 12.2% |
| 32 | 6.848 | 6.240 | 6.137 | 10.4% |
| 48 | 16.892 | 15.575 | 15.383 | 8.9% |

At 32 webs, public geometry `to_dict` calls fall from 81 to 8 and topology
validations from 82 to 11. The existing individual-query path still freshly
encodes complete source content (83 calls); the new per-face batch reduces that
to 19. The 66 arrangement calls and 10,255 definition checksum calls are retained.
Primitive dispatch reduces checksum overhead without an identity/revision cache.
Arrangement mathematics remains the largest preparation cost.

The baseline consumes 107.29878590011504 process seconds across nine children.
The candidate consumes 196.00333179999143 process seconds across eighteen
children (both query routes), within its aggregate 240-second and per-child
60-second bounds. No failed or timed-out benchmark run is omitted.

## Evidence and acceptance boundary

`reports/parallel-handoff-root/comparison.json` binds every retained raw
measurement artifact by SHA256 and records exact repeats, phase times, hashes,
call counts and topology sizes. `measure.py`, `compare.py` and the three frozen
documents retain the reproduction. Full raw evidence remains on disk in
`baseline-01` and `candidate-01`; the immutable baseline source archive is
separate from the current candidate source.

The implementation worker's main focused slice passes 187 tests. Final narrow
follow-ups pass 5 unknown-extent tests and 2 last-callback index tests. Aggregate
test process time is 31.483760099858046 of the same 90-second allowance. Earlier
failed fixtures/error-order checks and all precise process-wall accounting are
retained in `reports/parallel-handoff/WORKER_HANDOFF.md`. Independent source
review repaired mapping injectivity, transitive standalone references, promoted
reference bounds and raw derived-index qualification. It is independent
authorship within one model family, not different-family qualification.

After measurement, the partition refusal was tightened for unselected owners
with unknown physical offset reach; only that unexecuted planner, its tests and
API documentation changed. Every exercised preparation source remains byte
identical, so its recorded output/timing evidence is retained rather than rerun.

The final development wheel SHA256 is
`ef30c7d9fc5726e80e6abc4ade40aa2706d8b22a94a5efd67a7ac23e7f75378f`.
Every packaged Python source matches the final candidate. An offline wheel was
installed into an external temporary venv (reusing existing system NumPy 2.4.6);
Windows spawn workers verified site-packages origins, two independent stiffened
components, 20 complete identity-map records per component, current document
round-trips, chart bindings and unchanged authored geometry. This is a bounded
installed transport smoke, not a clean full downstream dependency campaign.
An initial build failed because the sandbox denied pip's user cache; that evidence
is retained in `installed-01`. The `--no-cache-dir` repair passed in `installed-02`.
Commands, exact process accounting, origin checks and final byte reconciliation
are in `reports/parallel-handoff-root/acceptance.json` and adjacent raw evidence.

Full platform CI, large/mixed native mesh acceptance and mesher-side adoption
are not claimed by this development evidence. Request 4 remains a later
material/reference/interface-station milestone. The old two ten-shot diagnostic
families remain closed; these new geometry-only checks do not renew them.
