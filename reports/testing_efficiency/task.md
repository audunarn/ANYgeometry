# Testing efficiency

2026-10-01. Scope: ANYgeometry development feedback. No consumer repository
changes, version bump, release, scientific tolerance change or new qualification claim.

Question: is repeated cross-platform/native qualification dominating ordinary feedback?
Existing accepted run 36834513207 ran 32 jobs: Windows native consumers took
1083 seconds; Intel macOS/Python 3.12 kernel took 827 seconds. These measurements
describe historical CI, not a projected speedup.

Bounded implementation: retain all full qualification jobs, matrices, source pins,
budgets and evidence steps in the explicitly dispatched Tests workflow. Add an
automatic development workflow with full Linux/Windows Python 3.13 kernels,
one candidate build, Linux/Apple-silicon installed smoke and installed mesher checks.
Add local focused selection, conservative full-suite fallback and saved timings.
Documentation changes receive hygiene checks without numerical acceptance claims.

Prospective procedure transition: routine changes use development evidence;
platform, dependency, native topology and release claims still require applicable
full qualification. Existing immutable release records and consumed execution
authority remain unchanged. Full qualification is explicitly dispatched for the
exact review head; a development green result does not replace it.

Verification: selector failure modes, one local kernel run with durations, workflow
job-definition preservation, then hosted development evidence. Report actual timings
and outstanding gates; do not claim unmeasured runtime savings.
