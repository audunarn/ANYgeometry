# Development checks and qualification

Run commands from ANYgeometry, using the development environment with
`pip install -e '.[dev,planar]'`. No additional test-runner dependency is needed.

```console
python tools/dev_checks.py --scope smoke
python tools/dev_checks.py --scope coordinates
python tools/dev_checks.py --scope intersections
python tools/dev_checks.py --plan
python tools/dev_checks.py --report /path/outside/checkout/development.json
python tools/dev_checks.py --scope full
```

The default compares committed changes against `origin/main`, then includes staged,
unstaged and untracked files. Fetch the base first; use `--base <ref>` for another
review base. A missing base is an error, never an empty successful selection.
Artifact-heavy checkouts can conservatively select the full suite because of
untracked outputs. Use explicit scopes or a clean worktree for focused feedback;
the selector does not silently discard unfamiliar inputs.
Explicit scopes are convenient development probes. They do not claim complete
coverage of every dependency of a module. Automatic selection currently focuses
only coordinate/automation and serialization changes, plus changed test files.
Unknown source, tooling, packaging, shared fixtures and deleted tests require the
full kernel suite. New test files are included. Markdown-only changes run
`git diff --check`; this establishes no numerical or public-contract acceptance.
Reports record the selection, interpreter, command, duration and exit status;
pytest retains ordinary output and prints the slowest 20 tests. Save reports
outside the checkout so they do not become inputs to subsequent selection.

On the measured local Windows/Python 3.13 environment, smoke passed 102 tests
in 1.21 seconds (2.78 seconds including startup); the full kernel passed 1,099
in 184.87 seconds. These scopes establish different coverage. Smoke deliberately
excludes release-authority mutation fixtures, which remain in full kernel CI.

## Automatic development CI

The **Development** workflow runs on pull requests and updates to main. Kernel
changes (anything the selector does not narrow) run the complete kernel on Windows
and Linux/Python 3.13 in three duration-balanced shards per platform; other runtime
changes (changed test files, coordinate and serialization modules) run the fast tier
(every test not marked `slow`) in three shards plus the changed test files completely.
A **Development kernel shards complete** job proves from each shard's report that every
selected test ran in exactly one shard, and that no test outside the `slow` tier took
longer than 20 s (otherwise the duration table must be refreshed, below). It builds one
candidate wheel, check it in isolated Linux and Apple-silicon macOS environments
with and without Shapely, and exercise installed ANYmesher against candidate and
released baseline wheels. Import origins, artifact hashes, dependency resolution,
license checks and consumer failure classification retain their existing rules.
JUnit results include individual timings. Installed-artifact logs and reports are
retained. The aggregate **Development gate** fails when an applicable job fails,
is cancelled or is unexpectedly skipped. Documentation-only changes run change
classification and whitespace checks, then the aggregate gate.

For a prose-only follow-up inside a runtime PR, CI can reuse a complete successful
development run. It checks the same PR/base, all thirteen successful jobs and the Git
delta from the recorded run head to the actual checkout (including the base merge).
Any source, test, workflow, packaging or packaged-license change prevents reuse.
Skipped, failed, missing or inaccessible evidence never qualifies. The lookup is
bounded to five recent successful runs; no usable evidence means ordinary checks.
The retained plan identifies the reused run, source SHA and exact prose delta.
This applies only to development evidence; scientific/release qualification is
not reused or granted by this mechanism.

There are thirteen expanded automatic jobs for runtime changes (six kernel shards).
The shards cut wall time; acceptance criteria are unchanged.

## Test tiers, shards and the duration table

`reports/ci/durations.json` records per-test CI durations (tests of at least 1 s on
either platform, the rest at their mean) from one CI run. `tests/conftest.py` marks
every test of at least 10 s on the Windows runner `slow` (`pytest -m slow`, `pytest -m
"not slow"`; together exactly the full suite) and, with `ANYGEOMETRY_TEST_SHARD=k/N`,
keeps shard `k` of the selected tests: longest first onto the least loaded shard, by
the recorded Windows durations. `ANYGEOMETRY_SHARD_REPORT=<file>` records a shard's
tests for the completeness check (`tools/ci_shards.py verify`). Each kernel shard adds
its 20 slowest tests to the job summary (`tools/ci_durations.py summary`; a soft
120 s budget per test is reported, not enforced). Refresh the table when tests become
slower or new slow tests appear (the shards job reports them), from the JUnit results
of a green run on main:

```console
gh run download <run-id> --repo audunarn/ANYgeometry --pattern 'development-kernel-*' --dir runs
w=runs/development-kernel-windows-latest
u=runs/development-kernel-ubuntu-latest
python tools/ci_durations.py build --run <run-id> \
    "windows=$w-1/kernel-results.xml,$w-2/kernel-results.xml,$w-3/kernel-results.xml" \
    "ubuntu=$u-1/kernel-results.xml,$u-2/kernel-results.xml,$u-3/kernel-results.xml"
```

Each platform takes its shards' JUnit files, comma-separated (merged before use).
`check` confirms the table reproduces the run's totals.

## Full qualification

The **Tests** workflow (`ci.yml`) is the final path. It runs weekly on main (Monday
03:00 UTC) and is dispatched for the exact review branch/head before a release or
whenever full qualification is needed; each platform x Python lane runs the full suite
in three duration-balanced shards (five on macOS Intel, which runs about 2x slower), and **pytest shards complete** proves every lane ran
every test exactly once:

```console
gh workflow run ci.yml --repo audunarn/ANYgeometry --ref <review-branch>
```

Its 32 jobs, Windows/Linux/macOS architectures, Python 3.11–3.14 kernel matrix,
minimum dependencies, all installed consumers, native fixtures, timeouts, scientific
tolerances, source pins and retained evidence are unchanged. Intel macOS downstream
native support remains unqualified. Release and support claims require applicable
full qualification and normal release authority; a green development gate alone
is insufficient. Numerical/topology changes require their applicable scientific
and native-consumer gates before acceptance. Dependency or platform changes need
the affected matrix lanes. Reuse evidence only when the changed inputs cannot
invalidate it, documenting the impact decision in the living task record.

This split changes when expensive jobs run. It neither changes their acceptance
criteria nor renews a consumed diagnostic budget. Preserve failed evidence; do not
rerun a failed scientific case merely to obtain a green result. Qualification and
release ledgers remain immutable. Consumer defects stay with their owners.

## Runner images

Workflows use the `*-latest` labels plus `macos-15` and `macos-15-intel`, so Linux
follows GitHub's current image (Ubuntu 26 from 2026-10-19). Pinning a version (for
example `ubuntu-24.04`) changes the job names: update `DEVELOPMENT_JOBS` in
`tools/dev_checks.py` and any required status checks in the same change. Actions are
pinned to commit SHAs; all run on Node 24.

## Known local failures

A full local run on Windows (`python -m pytest tests`, Python 3.14) has 15 failures that
CI does not have. They are environmental, not product failures; any other failure is new.

- Nine `tests/test_import_boundaries_cli.py` release-authority tests
  (`test_release_authority_neutralizes_external_git_configuration[*]`, six cases;
  `test_release_authority_rejects_mutation[noncanonical-tag-ref|paired-replacement|unmerged-tag-child]`):
  a `git push` into the test's temporary repository fails during the full run. The file
  passes when run on its own.
- Six native runtime tests read the workflows' `env:` block and fail with a `KeyError`
  without it: the thread settings `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
  `MKL_NUM_THREADS` set to `1`, and the `*_RUNTIME_EVIDENCE` output files (copy the
  `env:` block of `.github/workflows/development.yml` to run them locally):
  `test_native_arc_parameter_maps.py::test_effective_source_runtime`,
  `test_native_arc_station_incidence.py::test_effective_source_runtime`,
  `test_native_attachment_reference_scope.py::test_source_runtime`,
  `test_native_curved_planar_material_scope.py::test_runtime`,
  `test_native_material_reference_scope.py::test_source_runtime`,
  `test_preparation_epoch.py::test_runtime`.
