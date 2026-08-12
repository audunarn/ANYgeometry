# Kernel qualification benchmarks

`kernel_benchmarks.py` exercises only public ANYgeometry APIs and records both
wall time and deterministic work counts. It is a representative workload, not
a machine-independent pass/fail microbenchmark.

Run the quick profile:

```powershell
python benchmarks/kernel_benchmarks.py --output benchmarks/wave5_smoke.json
```

Run the release profile (10,000 plate faces and 10,000 persistent members):

```powershell
python benchmarks/kernel_benchmarks.py --qualification `
  --output benchmarks/wave5_qualification.json
```

`seconds` includes Python allocation tracing where `peak_python_bytes` is
present. Selected construction and serialization cases also record
`untraced_seconds`, which is the normal-runtime comparison. Full qualification
audits deliberately disable tracing and instead retain candidate,
classification, tree-visit, and narrow-phase counts.

Wave 0 files are measurements from base commit
`19f16746ceee76838b7df9d08a95028699e3a738`. Wave 5 files are the release
measurements summarized in `KERNEL_UPDATE_REPORT.md`.
