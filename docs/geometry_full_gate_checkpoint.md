# Workflow 2 full geometry gate checkpoint — 2026-10-28

State: scope audited; full applicable regression gate starting. Source commit `27efeba`; only existing dirty file is root-owned `docs/WORKFLOW_2_PROJECT.md` and it is protected.

New authority is routine Workflow 2 unit/regression verification, not reuse of `tools/run_main_integration_tests.py`, consumed common-main budgets, closed ten-shot families or scientific/provider campaigns. The unit-tree scope audit found geometry-only contracts in mesher-named tests; small ancestry/station fixtures explicitly forbid mesher execution; perturbation harness tests run only their small unit fixtures. No actual meshing provider run is included.

Plan and criteria: (1) run full tests through existing CI 3-shard selection and `development_environment()` runtime receipts, retain exact identities/stdout/stderr/JUnit; acceptance all selected tests accounted without changing skips/xfails. (2) classify failures against source and baseline; repair only Workflow 2-related compatibility within this repository. (3) verify changed slices and CI shard coverage, update this note and deliver evidence. Root owns independent acceptance and installed/platform gates.

Fresh budget: <=1200s per process, <=1800s cumulative process-wall including repair reserve. Historical CI timing data has302 recorded tests totalling1543.35sWindows; start first sequential CI shard with600s supervisor, adjust later bounds only within remaining budget while holding repair reserve. No baseline rerun is implicitly authorized beyond this budget. All failures/timeouts retained. Windows Git fixture tests require outside-sandbox execution; escalation is for the established disposable-fixture environment, not publication.

Evidence will be unique `../evidence/geometry-full-20261028-*` folders, with one-thread runtime and separate CI/runtime receipt directories. No source edits yet. Independent review pending root.

## First execution checkpoint

Shard1 source identities: `../evidence/geometry-full-20261028-shard1-5ff0b1274a`; CI selection/runtime receipts: `../evidence/geometry-full-20261028-shard1-runtime`. Exact full selection is4401 tests, shard1 assigns1466 including13 slow-marked cases. At approximately130s process progress reached39%, no failure emitted and three existing compiled-core conditional skips. Escalated bounded launch succeeded; no Git fixture permission workaround or repository configuration edits were needed. Source remains unchanged. Remaining work: finish shard1, use measured time to budget shards2/3 while preserving repair reserve, classify any failures, verify shard union and runtime receipts, final handoff.

## Retained completion / continuation

Shard1 completed **1461 passed,5 skipped,2935 deselected**; full selected4401 / assigned1466. Receipt `geometry-full-20261028-shard1-5ff0b1274a/result.json`: **436.296s** process-wall; cleanup verified zero owned processes. No source change. Interrupted model service sampling did not interrupt or restart the test process; terminal evidence was recovered by root. Original1800s budget now has **1363.704s remaining**, with approximately160s repair reserve held. Continuing the same authorized gate with shards2/3, <=600s each, no budget renewal. Prior unrelated exact-envelope Read request superseded by root; no further native or provider review/execution performed by this worker.

## Shard2 completion

Shard2 terminal receipt `../evidence/geometry-full-20261028-shard2-cade071727/result.json`: exit0, **402.188s**, cleanup verified zero owned processes. Runtime receipts `arc-runtime.json` and `epoch-runtime.json` confirm source origin, interpreter and thread settings1. No source repair needed. Cumulative shard1+2 process-wall **838.484s**, original budget remaining **961.516s**. Shard3 launched under600s bound, retaining at least361.516s for final verification/repair. No completed shard restarted.

## Final full-source gate — passed, root acceptance pending

All three unchanged-source CI shards completed. Full selection **4401**, **4386 passed /15 skipped /0 failures /0 errors**. Existing skipped cases all report `compiled core not in use: compiled module not installed`; no skip/xfail/selection or tolerance was added or changed. No source compatibility repair or focused retry was required.

| Existing CI shard | Assigned/results | Precise process-wall | Retained evidence folder |
|---|---|---:|---|
|1/3|1461 passed,5 skipped|436.2960000000021 s|`geometry-full-20261028-shard1-5ff0b1274a`|
|2/3|1462 passed,6 skipped|402.18799999999464 s|`geometry-full-20261028-shard2-cade071727`|
|3/3|1463 passed,4 skipped|479.6570000000065 s|`geometry-full-20261028-shard3-d490d0cc6c`|

Evidence folders are under `C:/ANYworkspaceWork/workflow2-20261027/evidence`. The exact child command in each is:

`C:/ANYworkspaceWork/workflow2-20261027/runtime/Scripts/python.exe -m pytest -q --durations=15 tests --junitxml=<that evidence folder>/pytest.xml`

Each launch used `tools.dev_checks.development_environment()` before the established `ANYmesh/tools/workflow2_check.py` Windows Job Object supervisor, timeout600s, explicit owned cwd, owned geometry src plus read-only sibling ANYmesh src on PYTHONPATH, and `ANYGEOMETRY_TEST_SHARD=1/3`, `2/3`, or `3/3`. No imported mesher/provider execution was added. Existing Git tests used their isolated local temporary remotes under escalated unit-test supervision. Native enable/disable/require flags were unset in the recorded launch identities; no forced skip flags.

The six receipt paths were set independently under the corresponding `geometry-full-20261028-shardN-runtime` directories. All six source-runtime receipt tests ran in their assigned shards; actual receipts confirm Python3.11.9, NumPy2.4.6, pytest9.1.1, the owned source checkout and OMP/OPENBLAS/MKL thread variables all1. The340 source/test/CI helper hashes, interpreter and PYTHONPATH match across every shard.

CI reports were copied byte-for-byte from runtime directories adjacent to their supervisor JUnit XML to use the existing verification interface. Completed checks:

- `python tools/ci_shards.py verify <shard1-folder>/shard-1.json <shard2-folder>/shard-2.json <shard3-folder>/shard-3.json --results pytest.xml`: **passed exit0**, **0.8280000000086147s**, `geometry-full-20261028-coverage-d57c6dcd07`. Verifies identical full selections, assigned-test XML presence, complete union and no duplicates.
- `python tools/ci_shards.py guard --junit <shard1-folder>/pytest.xml <shard2-folder>/pytest.xml <shard3-folder>/pytest.xml --reports <shard1-folder>/shard-1.json <shard2-folder>/shard-2.json <shard3-folder>/shard-3.json --limit 20`: **passed exit0**, **0.8280000000086147s**, `geometry-full-20261028-ci-guard-14c878341c`. Existing slow classification/timing threshold unchanged.
- `git diff --check`: **passed**. Only pre-existing root-owned `docs/WORKFLOW_2_PROJECT.md` and this new checkpoint are dirty. Root-owned file was not edited; its pre-existing LF/CRLF Git warning was not changed.

**Cumulative new-milestone process-wall:1319.7970000000205s /1800s**, remaining **480.2029999999795s**. No timeout, failed full-gate attempt or process restart. Every supervisor receipt verifies zero remaining owned processes. The initial collection was root's separate retained evidence; this accounting continues the original worker budget after shard1 and does not reset it.

Complete exact commands, sequential cumulative accounting, source identities, all test/skip counts, runtime receipts and SHA-256 hashes of every raw artifact are in:

`C:/ANYworkspaceWork/workflow2-20261027/evidence/geometry-full-20261028-summary.json`

Review status: **root independent acceptance pending**. No source changes require a new implementation review. This completes the assigned Windows/Python3.11 source unit/regression gate only. Optional compiled geometry-core execution (15 existing skips), installed wheel gates, other supported platforms/Python versions, provider/native meshing qualification, scientific/performance campaigns and publication were **not run or claimed by this worker**. Root retains those remaining Workflow2 gates. Previously retained failed studies/fixtures and closed common-main/ten-shot authority remain unchanged.
