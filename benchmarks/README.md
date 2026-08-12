# Kernel qualification benchmarks

`kernel_benchmarks.py` exercises only public ANYgeometry APIs and records both
wall time and deterministic work counts. It is a representative workload, not
a machine-independent pass/fail microbenchmark.

Run the quick profile:

```powershell
python benchmarks/kernel_benchmarks.py --output benchmarks/wave_gap_closure_smoke.json
```

Run the release profile (10,000 plate faces and 10,000 persistent members):

```powershell
python benchmarks/kernel_benchmarks.py --qualification `
  --output benchmarks/wave_gap_closure_qualification.json `
  --baseline benchmarks/wave0_qualification.json
```

The current runner records the original grid/member/audit workloads plus a
22,500-face indexed-query grid, changed-region audit, schema migration,
feature-closure checksum, typed intersection query/plan/apply, and separate
local Member and plate edits inside one persistent mixed Sheet/Member model.
Cold maintained-index materialization and the identical steady-state local
query are separate records. The 10,000-face maintained index is explicitly
materialized before its local edit, so changed-region audit measures
post-index affected-scope work rather than silently including a cold global
build. Cold and steady queries must return identical candidates. Local-edit
records include geometry and structural changes, spatial updates, structural
validation visits, and whether validation expanded to the complete model.

The 150 x 150 larger grid is the practical release profile on the supported
Windows/Python runner. It demonstrates deterministic locality and candidate
counts without implying that a near-100,000-face wall-clock run was executed.

`seconds` includes Python allocation tracing where `peak_python_bytes` is
present. Selected construction and serialization cases also record
`untraced_seconds`, which is the normal-runtime comparison. Full qualification
audits deliberately disable tracing and instead retain candidate,
classification, tree-visit, and narrow-phase counts.

Wave 0 files are measurements from base commit
`19f16746ceee76838b7df9d08a95028699e3a738`. Wave 5 files are the release
measurements for 0.2.0. `wave_gap_closure_qualification.json` is regenerated
from the final 0.2.1 tree and is the authoritative performance evidence
summarized in `KERNEL_UPDATE_REPORT.md`.
