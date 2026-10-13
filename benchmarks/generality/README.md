# Generality + speed corpus (G0)

Workloads for the ANYgeometry generality and speed programme
(`docs/agent_plans/GENERALITY_PROGRAMME_PLAN.md`). Public APIs only, so the same
corpus runs against any source tree on `PYTHONPATH`.

```powershell
$env:PYTHONPATH = "C:\path\to\tree\src"
python benchmarks/generality/run.py --output <new-dir> --repeat 3
python benchmarks/generality/compare.py <base-dir> <candidate-dir> --bound 1e-9
```

`run.py` records, per workload, the best of `--repeat` wall times, a topology
digest (canonical document with every float replaced by a marker), the float
vector in document order, and for the intersection workloads the total material
area of the trimmed charts and the largest distance of sampled edge points from
the faces that use them (joint residual). Faces whose chart cannot be qualified
are listed in `chart_failures`.

`compare.py` applies owner decision D1: identical topology, coordinates within
`--bound` relative to the model extent, no quality loss. Timings on a shared
machine vary by 10-20 %; compare repeated runs before quoting a speed-up.

Known on main 0a3a602: `tilted8` applies, but 3 of its 914 resulting faces fail
`query_trimmed_surface_charts` with "arrangement failed material conservation".
