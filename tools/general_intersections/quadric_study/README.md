# Quadric generalization study scripts

Throwaway study scripts behind `reports/general_intersections/quadric_generalization_study.md`.
They are not part of the package or its qualification. Each takes the directory that contains its
siblings (this directory) as the first argument, and needs `anygeometry` importable (`PYTHONPATH=src`).

| Script | Purpose |
| --- | --- |
| `cone_probe.py`, `cone_e1b.py`, `cone_e1c.py`, `cone_e2.py` | What the batch engine and the legacy certified engine do with cone x cylinder facets |
| `cyl_cyl_10.py` | Exact reference: cylinder x cylinder 10 degrees off perpendicular through the batch engine |
| `curve_repr_cost.py` | Analytic versus Bezier traces in the same arrangement |
| `mutual2.py` | Mutually crossing plates, through one point and in generic position |
