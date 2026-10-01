# Quadric generalization study scripts

Throwaway study scripts behind `reports/general_intersections/quadric_generalization_study.md`.
They are not part of the package or its qualification. Each takes the directory that contains its
siblings (this directory) as the first argument, and needs `anygeometry` importable (`PYTHONPATH=src`).

| Script | Purpose |
| --- | --- |
| `cone_probe.py`, `cone_e1b.py`, `cone_e1c.py`, `cone_e2.py` | What the batch engine and the legacy certified engine do with cone x cylinder facets |
| `cyl_cyl_10.py` | Exact reference: cylinder x cylinder 10 degrees off perpendicular through the batch engine |
| `qb_verify.py` | Independent grid oracle for the prototype's charts |
| `qb_fuzz.py`, `qb_fuzz2.py` | Random pairings checked against the oracle |
| `qb_roots.py` | Exact branch/quadric roots against dense sign changes |
| `qb_vs_old.py` | Parity with the production `cylinder_cylinder_support` charts |
| `qb_sphere.py` | A sphere as the unbounded second support |
| `qb_time.py` | Timing of the exact support intersection over all facet pairs |
| `cone_tracer.py` | Cone facets arranged with Bezier-fitted traces (isolates the chart change) |
| `curve_repr_cost.py` | Analytic versus Bezier traces in the same arrangement |
| `mutual2.py` | Mutually crossing plates, through one point and in generic position |
