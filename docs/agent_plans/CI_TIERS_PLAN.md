# CI tiers: a development path that stays fast and a final path that stays complete

Status: accepted (2026-10-16). Owner decisions: D-CI1 a, D-CI2 a, D-CI3 a, D-CI4 a; "Go C0-C7".

## Where we are (measured, run 38892370520 on main f864e6b)

| Path | Workflow | Trigger | What runs | Measured |
| --- | --- | --- | --- | --- |
| Development | `development.yml` | push to main, PR | classify changes; for any source change (almost always) the **full suite** on Ubuntu and Windows / Python 3.13; wheel build; installed-wheel and mesher checks; gate | Ubuntu kernel 22 min, Windows kernel **41 min** of a 60 min limit |
| Final | `ci.yml` (Tests) | `workflow_dispatch` only | 32 jobs: full suite on Windows / Ubuntu / macOS arm / macOS Intel x Python 3.11-3.14 (16 kernel jobs, **30 min limit each**), minimum dependencies, planar absence, consumers, wheels | last run 2026-10-01: failed; never run since the generality work. With today's suite it would time out at least on Windows and macOS Intel |

The suite (3,738 tests) is very uneven:

| Group | Ubuntu | Windows |
| --- | --- | --- |
| 3,544 / 3,423 tests under 1 s | 4.3 min | 5.5 min |
| 29 / 45 tests over 10 s | 10.3 min | 22.4 min |
| `test_hydrostatics.py` alone | 7.1 min | 13.4 min |
| half of all time | slowest 34 tests | slowest 35 tests |

The ten slowest tests (Windows) are hydrostatics, B1 and section-face certificates: free-trim fallback
cases (130 s, 59 s), an excursion refusal in section faces (117 s), the heeled full-hull reference (112 s),
an exact ill-conditioned root (76 s), the full-hull audit (51 s), plus fixture rebuilds of the B1 hulls
(about 17 s per module that builds them). Every new certified-geometry slice adds tests of this kind, so
the suite's time grows faster than its test count: splitting jobs alone is not enough.

## Goals and constraints

- Development path: fast feedback on every push/PR (target: under 15 min wall time on the slowest OS),
  without reducing what is checked before a merge to main counts as "development green".
- Final path: complete qualification exactly as `docs/testing.md` defines it (all platforms, Python
  3.11-3.14, consumers, minimum dependencies), runnable within the job limits, and actually run before a
  release and on a regular schedule.
- Unchanged: acceptance criteria, scientific tolerances, refusal behaviour, release authority and ledgers.
  A faster path never weakens a test; it changes when and where tests run.
- `docs/testing.md` says no extra test-runner dependency is needed. Parallelism via `pytest-xdist` would
  add one (decision D-CI2).

## Plan (slices with acceptance criteria)

| Slice | Output | Acceptance | Depends |
| --- | --- | --- | --- |
| C0 | **Timing baseline**: a script that turns the JUnit artifacts of a run into a per-test duration table (`reports/ci/durations.json`, tests above 1 s), refreshed from CI on demand | the table reproduces the totals above within 2 %; checked in; a CI step uploads the JUnit files it reads | – |
| C1 | **Make the slow tests cheaper without weakening them**: session-scope fixtures for the B1 half/full hulls (built once per run, not per module); the heeled reference and the bracketed free-trim cases computed once and reused across assertions; float-path references at the order that their stated tolerance needs (measured, not guessed) | every changed test keeps its assertions and tolerances; the suite's slowest-50 time drops by at least 40 % on Ubuntu (C0 table before/after); full suite green | C0 |
| C2 | **Tiers by marker**: `@pytest.mark.slow` on tests above a threshold from the C0 table (proposal: 10 s on Windows), registered in `pyproject.toml`; `pytest -m "not slow"` and `pytest -m slow` partition the suite exactly; a guard test fails when an unmarked test exceeds the threshold in the latest timing table (so new slow tests get marked) | `-m "not slow"` + `-m slow` collect exactly the full suite (no test lost or doubled); the fast tier runs in under 8 min on Windows | C0 |
| C3 | **Sharding** (option A, no new dependency): `tools/ci_shards.py` splits the collected test ids into N shards balanced by the C0 durations (unknown tests spread round-robin); the kernel job becomes a matrix over shards; a final aggregation step checks that the union of the shards' JUnit test ids equals the collected suite | union check exact; shard walls within 25 % of each other; with N = 3 the Windows kernel wall time under 15 min. (Option B instead: `pytest-xdist -n auto` inside each job; acceptance: suite xdist-safe, the runtime-evidence JSON files and basetemps per worker, same pass/fail set as serial.) | C0, D-CI2 |
| C4 | **Development path**: on PRs and pushes, the fast tier always runs on Ubuntu and Windows; the slow tier runs (sharded) when the change touches the geometry kernel or the tests' own files, as today's full scope, so a merge to main still has the complete suite green on two OSes; documentation-only changes unchanged | the development gate requires both tiers whenever the full scope applies; the slow tier is never silently skipped (gate fails on a skipped applicable tier); measured wall time of a typical source change under 15 min | C2, C3 |
| C5 | **Final path made runnable**: `ci.yml` kernel jobs sharded like C3 with the job limit kept at 30 min per shard; the 16 platform x Python lanes keep the full suite each (no lane reduced); consumers, minimum dependencies, wheels and planar absence unchanged | a dispatched run on main completes green, every lane within its limit, total wall time reported; the aggregation check proves every lane ran every test | C3 |
| C6 | **When the final path runs**: (a) required before any release tag (release ledger records the run id, as today); (b) a weekly scheduled run on main, plus (c) on demand; failures open an issue/notify rather than block development | schedule present; the release checklist in `docs/testing.md` names the run; first scheduled run green or its failures triaged | C5, D-CI3 |
| C7 | **Keep it from creeping back**: the C2 guard plus a job-summary table of the 20 slowest tests per run; a soft budget (proposal: fast tier under 8 min, any single test under 120 s on Windows) reported in the summary, not enforced as a failure | budgets visible in every run summary; the plan's living record notes when they are exceeded | C2 |

Order: C0 -> (C1, C2 in parallel) -> C3 -> C4 -> C5 -> C6; C7 alongside C4.

## Decisions for the owner

- **D-CI1 — development-path content.** (a) Fast tier always, slow tier only for kernel-touching changes
  (as today's full scope, recommended); (b) fast tier only on PRs, slow tier only after merge to main;
  (c) keep the full suite on every change, just sharded.
- **D-CI2 — parallelism.** (a) Own duration-balanced sharding across jobs (no new dependency, more jobs,
  recommended); (b) `pytest-xdist` within jobs (fewer jobs, a new test dependency, needs xdist-safety
  work for the runtime-evidence files); (c) both.
- **D-CI3 — final-path cadence.** Before every release (as now) plus (a) weekly scheduled (recommended),
  (b) nightly, or (c) only on demand.
- **D-CI4 — final-path breadth.** Keep all 16 kernel lanes with the full suite (recommended, as
  `docs/testing.md` requires), or run the slow tier only on the newest and oldest Python per OS. (b) is a
  change to the qualification definition and needs an explicit decision.

## Risks and open questions

1. Hosted runner speed varies (Windows ~1.8x Ubuntu here); shard balancing uses Windows timings.
2. Tests that write shared evidence files (`*_RUNTIME_EVIDENCE`) must keep working when split across jobs
   (they already run in one process per job; xdist would need per-worker paths).
3. The final path also contains consumer and native-wheel jobs whose duration is not measured here;
   C5 measures them on the first dispatched run.
4. Hosted minutes: sharding raises job count (development: about 2 x 3 kernel shards instead of 2 jobs;
   final: 16 x 2-3 shards). Wall time drops; billed minutes rise slightly (setup per job, ~1-2 min).
5. The ubuntu-latest image moves to Ubuntu 26 from 2026-10-19, and the Node.js 20 action deprecation is
   pending; C4/C5 should pin or update the actions in the same change.
