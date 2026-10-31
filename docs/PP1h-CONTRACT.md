# PP1h — risk-accepted qualification, not a purity proof

Eighth release, parent ba746b2f, claude/pp1-purity-20261031. No remote or whole-model
run. No certified numerical source-pin guard may be refreshed or changed.

Plan: parent-red batch publication/counter tests; separate batch worker lifecycle;
real child measurements at K=16 and K=64, 200 tasks each (100 pair, 100 face), with
serial kernel comparison in the parent; final PP1 + 191/57 regressions and three
pool-off identity gates; audited commit and append-only handoff.

Contract: the existing WorkerGate remains unchanged. The new finite-assignment
worker captures full environment once at startup, passes both actual canaries,
checks file/code/module/store hashes before/after each chunk, and resets the known
cache handles before each task. K is explicit/configurable. K=1 preserves the
before/after-each-task heavy-check cadence; version/CPU discovery remains startup-only.
The final partial chunk's post-check is the mandatory worker-end boundary; an empty
assignment still checks an end boundary. No assignment results escape before that
end check. A failure clears the entire candidate assignment and reports the current
unverified batch count, then raises a qualification decline for PP1g serial fallback.
Cancellation also clears staged results and propagates. Only one assignment per
startup is allowed, preventing accidental omission of the final boundary.

Between boundaries, resets use the concrete previously verified registry handles.
A rebound root or altered module/store is caught at the next boundary and discards
results. A transient change restored before the next boundary is not proved absent;
this is the explicitly accepted batch risk, not a purity claim. No tolerance changes,
certified-source pin refresh, or relaxation of legacy per-task tests is permitted.


Measured finite workload: 200 small real kernel tasks per K, 100 pair and 100 face,
with one admitted child per run; every result matched the serial parent and the
unchanged canary goldens. K=16: average assignment cost 101.246 ms/task versus serial
11.343 ms/task; K=64: 38.202 versus 9.392 ms/task. These include amortized boundary
checks but exclude startup; raw differences are 89.903 and 28.811 ms/task. They are
not statistical estimates or N2c timings. Worker pre-run load: K16 3 Python processes,
CPU 1.9%; K64 3, CPU 0.0%; 32 logical CPUs, BLAS 1/1. Serial-parent pre-run load:
K16 2 Python processes, CPU 0.7%; K64 2, CPU 0.0%. Only anonymous aggregate process
counts were collected as requested; no unrelated command lines or PIDs were inspected.

One full environment capture occurred per worker. Including two startup boundaries,
K16 made 28 heavy checks and K64 made 10; each reset 202 times (two startup canaries
plus 200 tasks). The final partial chunk passed its mandatory end check. The per-task
path does no environment/source/store census/hash work; the existing task-payload
checksum check remains mandatory. Default K=64 is configurable; production selection,
N2c performance, transport/backpressure and independent reviews remain open.
