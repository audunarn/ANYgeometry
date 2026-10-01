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

## Implementation and local evidence

- Full qualification job bodies verified byte-for-byte unchanged against main;
  only automatic event triggers removed. Automatic development expands to eight jobs.
- Five selector safeguards pass, including packaged-license changes, shared fixtures,
  deleted/new tests and preservation of failed subprocess exit codes.
- Full local kernel: 1,099 passed, 184.87 seconds; Windows/Python 3.13.9,
  pytest 9.0.1, NumPy 2.4.3, Shapely 2.1.2. No numerical code changed.
- Initial smoke: 133 passed in 45.63 seconds. Its release-authority mutation
  fixtures dominated feedback. Decision: keep those fixtures in full suites,
  retain owner/import/CLI contracts in the smoke scope. Revised smoke: 102 passed
  in 1.21 seconds, 2.78 seconds including process startup. Coverage is intentionally
  narrower and is not qualification equivalence.
- Local raw logs and JUnit are retained outside this worktree under
  `C:/Github/ANYgeometry/reports/testing-efficiency-*`; evidence.json binds their
  hashes. This task changes no consumer implementation or publication authority.
- Hosted first implementation: run 36867936439, head
  2a577a474cb3db9647f45c8422c89af3c53e0904; build, both installed wheel checks
  and installed mesher checks passed. All eight jobs subsequently passed.
- Final refinement also checks real Git committed/staged/deleted/untracked inputs
  in a disposable repository. Five selector tests passed in 0.48 seconds.
  The packaged license always requires runtime checks. The final hosted head is
  tracked in PR11; its runtime checks must pass before delivery acceptance.

## Final runtime acceptance

- Runtime/tooling head `433dfc1bc71a237c0578ffe032aedeaccc923829`: run
  36869241786 passed 8/8 jobs, including both complete 1,100-test kernels.
- Observed hosted elapsed time: 404 seconds. This is a development
  lane result, not full platform/native qualification.
- Raw evidence `testing-efficiency-hosted-final.json` SHA-256
  `45a036dd1885ed5e7890509e0ef531c2c2c3e62b653942a8c2d988eb7b5b1cfc` (retained in primary reports).
- Raw evidence `testing-efficiency-hosted-final.log` SHA-256
  `3e1e77f89e309e2dc70d51de2e8b881d36a865233538d1aab1c708a95af68b19` (retained in primary reports).
- Final documentation-only follow-up records these outcomes. Its runtime,
  tests, packaging and workflow inputs must match the accepted runtime head;
  no full-kernel repetition is required for this prose-only addition.
