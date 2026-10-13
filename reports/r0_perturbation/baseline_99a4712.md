# Perturbation harness report

- anygeometry: `C:\Github\ANYgeometry\.worktrees\r0-harness\src\anygeometry\__init__.py`
- python 3.14.2, numpy 2.4.6, seed 0
- variants: none, outputs/+1, outputs/-1, outputs/random, outputs+evaluations/+1, outputs+evaluations/-1, outputs+evaluations/random, route/numpy, route/scalar, route/fma, reverse
- cases: 35, operations: 74, flips: 75, wall: 425.7 s

## Per case

| Case | Operation | Flipping variants | Reversal |
|---|---|---|---|
| plane_family/plane | query[first,second] | - | same |
| plane_family/cylinder | query[first,second] | outputs/+1, outputs/random, outputs+evaluations/+1, outputs+evaluations/random | same |
| plane_family/cone | query[first,second] | outputs/+1, outputs/random, outputs+evaluations/+1, outputs+evaluations/random | same |
| plane_family/ruled | query[first,second] | - | same |
| plane_family/coons | query[first,second] | outputs/-1, outputs/random, outputs+evaluations/-1 | same |
| plane_u0_edge/seed1 | query[first,second] | route/scalar | same |
| plane_u0_edge/seed1 | points[face] | - | n/a |
| plane_u0_edge/seed1 | plan_apply | - | same |
| plane_u0_edge/seed2 | query[first,second] | - | same |
| plane_u0_edge/seed2 | points[face] | - | n/a |
| plane_u0_edge/seed2 | plan_apply | - | same |
| plane_u0_edge/seed3 | query[first,second] | - | same |
| plane_u0_edge/seed3 | points[face] | - | n/a |
| plane_u0_edge/seed3 | plan_apply | - | same |
| exact_tol_vertex/0.75tol | plan_apply | - | same |
| exact_tol_vertex/0.75tol | query[first,second] | - | same |
| exact_tol_vertex/0.75tol | points[face] | - | n/a |
| exact_tol_vertex/prev(tol) | plan_apply | outputs/-1, outputs/random, outputs+evaluations/-1 | same |
| exact_tol_vertex/prev(tol) | query[first,second] | - | same |
| exact_tol_vertex/prev(tol) | points[face] | outputs/-1, outputs/random, outputs+evaluations/-1, outputs+evaluations/random | n/a |
| exact_tol_vertex/tol | plan_apply | outputs/-1, outputs/random, outputs+evaluations/-1, outputs+evaluations/random | same |
| exact_tol_vertex/tol | query[first,second] | - | same |
| exact_tol_vertex/tol | points[face] | outputs/-1, outputs+evaluations/-1, outputs+evaluations/random | n/a |
| exact_tol_vertex/next(tol) | plan_apply | outputs/-1, outputs/random, outputs+evaluations/-1 | same |
| exact_tol_vertex/next(tol) | query[first,second] | - | same |
| exact_tol_vertex/next(tol) | points[face] | outputs/-1, outputs/random, outputs+evaluations/-1, outputs+evaluations/random | n/a |
| exact_tol_vertex/1.5tol | plan_apply | - | same |
| exact_tol_vertex/1.5tol | query[first,second] | - | same |
| exact_tol_vertex/1.5tol | points[face] | - | n/a |
| exact_tol_vertex/offset1e-170 | plan_apply | - | same |
| exact_tol_vertex/offset1e-170 | query[first,second] | outputs+evaluations/random | same |
| exact_tol_vertex/offset1e-200 | plan_apply | - | same |
| exact_tol_vertex/offset1e-200 | query[first,second] | outputs/random, outputs+evaluations/random | same |
| exact_tol_vertex/offset5e-324 | plan_apply | - | same |
| exact_tol_vertex/offset5e-324 | query[first,second] | outputs/random | same |
| exact_tol_vertex/tol@1e-150 | plan_apply | - | same |
| exact_tol_vertex/tol@1e-150 | query[first,second] | outputs/+1, outputs/-1, outputs/random, outputs+evaluations/+1, outputs+evaluations/-1, route/scalar, route/fma | same |
| exact_tol_vertex/tol@1e-100 | plan_apply | - | same |
| exact_tol_vertex/tol@1e-100 | query[first,second] | outputs/+1, outputs/-1, outputs+evaluations/+1, outputs+evaluations/-1, outputs+evaluations/random | same |
| exact_tol_vertex/tol@1e+20 | plan_apply | outputs/-1, outputs/random, outputs+evaluations/-1 | same |
| exact_tol_vertex/tol@1e+20 | query[first,second] | - | same |
| exact_tol_vertex/tol@1e+50 | plan_apply | outputs/-1, outputs+evaluations/-1, route/scalar, route/fma | same |
| exact_tol_vertex/tol@1e+50 | query[first,second] | - | same |
| scaled_crossing/1e-100 | plan_apply | outputs+evaluations/+1, outputs+evaluations/-1, outputs+evaluations/random | same |
| scaled_crossing/1e-100 | query[first,second] | outputs/+1, outputs/-1, outputs+evaluations/+1, outputs+evaluations/-1, outputs+evaluations/random | same |
| scaled_crossing/1000 | plan_apply | - | same |
| scaled_crossing/1000 | query[first,second] | - | same |
| scaled_crossing/1e+08 | plan_apply | - | same |
| scaled_crossing/1e+08 | query[first,second] | - | same |
| float32_tilted3 | plan_apply | - | same |
| float32_tilted3 | query[first,second] | - | same |
| seam_cylinder | points[face] | - | n/a |
| seam_cylinder | query[first,second] | - | same |
| seam_cylinder | plan_apply | - | same |
| seam_cylinder | charts | - | n/a |
| arc_opening/flat | points[face] | - | n/a |
| arc_opening/flat | query[first,second] | - | same |
| arc_opening/flat | query[probe,first] | - | same |
| arc_opening/flat | plan_apply | outputs+evaluations/random | same |
| arc_opening/flat | charts | - | n/a |
| arc_opening/tilted | points[face] | - | n/a |
| arc_opening/tilted | query[first,second] | outputs/-1, outputs+evaluations/random | same |
| arc_opening/tilted | query[probe,first] | - | same |
| arc_opening/tilted | plan_apply | outputs/-1 | same |
| arc_opening/tilted | charts | - | n/a |
| native_arc_station/nice | station_0.75tol | - | n/a |
| native_arc_station/nice | station_1.5tol | - | n/a |
| native_arc_station/split_cylinder | station_selection | outputs/+1, outputs/random, outputs+evaluations/+1, outputs+evaluations/-1, outputs+evaluations/random, route/fma | n/a |
| g0/pipes6 | plan_apply | - | same |
| g0/rings8 | plan_apply | - | same |
| g0/grid16 | plan_apply | - | same |
| g0/tilted8 | plan_apply | outputs+evaluations/random | same |
| g0/cubic_wall_pipe | plan_apply | - | same |
| g0/cone_pipe | plan_apply | - | same |

## Per perturbation site

| Site | Inventory | Flips |
|---|---|---|
| (not attributed) |  | 3: exact_tol_vertex/tol points[face] outputs+evaluations/random; exact_tol_vertex/next(tol) points[face] outputs+evaluations/random; arc_opening/tilted query[first,second] outputs+evaluations/random |
| Cone.local_uv | A2, A4, F2, F3 | 4: plane_family/cone query[first,second] outputs/+1; plane_family/cone query[first,second] outputs/random; plane_family/cone query[first,second] outputs+evaluations/+1; plane_family/cone query[first,second] outputs+evaluations/random |
| Cylinder.local_uv | A2, A3, A4, D1, D7 | 4: plane_family/cylinder query[first,second] outputs/+1; plane_family/cylinder query[first,second] outputs/random; plane_family/cylinder query[first,second] outputs+evaluations/+1; plane_family/cylinder query[first,second] outputs+evaluations/random |
| EllipticArc.evaluate | B2, B5, B11, C1, C3 | 3: native_arc_station/split_cylinder station_selection outputs+evaluations/+1; native_arc_station/split_cylinder station_selection outputs+evaluations/random; native_arc_station/split_cylinder station_selection route/fma |
| LinePath.evaluate | B1, B3, B11, C1, D1, D6 | 8: scaled_crossing/1e-100 plan_apply outputs+evaluations/+1; scaled_crossing/1e-100 plan_apply outputs+evaluations/-1; scaled_crossing/1e-100 plan_apply outputs+evaluations/random; arc_opening/flat plan_apply outputs+evaluations/random; native_arc_station/split_cylinder station_selection outputs+evaluations/+1; native_arc_station/split_cylinder station_selection outputs+evaluations/-1 ... |
| Plane.local_uv | A1, D1, D4, E4 | 53: plane_family/coons query[first,second] outputs/-1; plane_family/coons query[first,second] outputs/random; plane_family/coons query[first,second] outputs+evaluations/-1; plane_u0_edge/seed1 query[first,second] route/scalar; exact_tol_vertex/prev(tol) plan_apply outputs/-1; exact_tol_vertex/prev(tol) plan_apply outputs/random ... |
| freeze_edge.arc | B2, B11, C1, C4 | 6: arc_opening/tilted query[first,second] outputs/-1; arc_opening/tilted plan_apply outputs/-1; native_arc_station/split_cylinder station_selection outputs/+1; native_arc_station/split_cylinder station_selection outputs/random; native_arc_station/split_cylinder station_selection outputs+evaluations/+1; native_arc_station/split_cylinder station_selection outputs+evaluations/random |

## Flips

### plane_family/cylinder / query[first,second] / outputs/+1

- sites: Cylinder.local_uv (inventory A2, A3, A4, D1, D7)
- changed: `/components: 1 -> 0`; `/dimension: "curve" -> "none"`; `/kind: "cross" -> "disjoint"`
- reference: `{"classified": true, "components": 1, "dimension": "curve", "kind": "cross"}`
- perturbed: `{"classified": true, "components": 0, "dimension": "none", "kind": "disjoint"}`
- reproducer: `python tools/perturbation_harness.py --case "plane_family/cylinder" --op "query[first,second]" --variant "outputs/+1"`

### plane_family/cylinder / query[first,second] / outputs/random

- sites: Cylinder.local_uv (inventory A2, A3, A4, D1, D7)
- changed: `/components: 1 -> 7`
- reference: `{"classified": true, "components": 1, "dimension": "curve", "kind": "cross"}`
- perturbed: `{"classified": true, "components": 7, "dimension": "curve", "kind": "cross"}`
- reproducer: `python tools/perturbation_harness.py --case "plane_family/cylinder" --op "query[first,second]" --variant "outputs/random"`

### plane_family/cylinder / query[first,second] / outputs+evaluations/+1

- sites: Cylinder.local_uv (inventory A2, A3, A4, D1, D7)
- changed: `/components: 1 -> 0`; `/dimension: "curve" -> "none"`; `/kind: "cross" -> "disjoint"`
- reference: `{"classified": true, "components": 1, "dimension": "curve", "kind": "cross"}`
- perturbed: `{"classified": true, "components": 0, "dimension": "none", "kind": "disjoint"}`
- reproducer: `python tools/perturbation_harness.py --case "plane_family/cylinder" --op "query[first,second]" --variant "outputs+evaluations/+1"`

### plane_family/cylinder / query[first,second] / outputs+evaluations/random

- sites: Cylinder.local_uv (inventory A2, A3, A4, D1, D7)
- changed: `/components: 1 -> 7`
- reference: `{"classified": true, "components": 1, "dimension": "curve", "kind": "cross"}`
- perturbed: `{"classified": true, "components": 7, "dimension": "curve", "kind": "cross"}`
- reproducer: `python tools/perturbation_harness.py --case "plane_family/cylinder" --op "query[first,second]" --variant "outputs+evaluations/random"`

### plane_family/cone / query[first,second] / outputs/+1

- sites: Cone.local_uv (inventory A2, A4, F2, F3)
- changed: `/components: 1 -> 12`
- reference: `{"classified": true, "components": 1, "dimension": "curve", "kind": "cross"}`
- perturbed: `{"classified": true, "components": 12, "dimension": "curve", "kind": "cross"}`
- reproducer: `python tools/perturbation_harness.py --case "plane_family/cone" --op "query[first,second]" --variant "outputs/+1"`

### plane_family/cone / query[first,second] / outputs/random

- sites: Cone.local_uv (inventory A2, A4, F2, F3)
- changed: `/components: 1 -> 12`
- reference: `{"classified": true, "components": 1, "dimension": "curve", "kind": "cross"}`
- perturbed: `{"classified": true, "components": 12, "dimension": "curve", "kind": "cross"}`
- reproducer: `python tools/perturbation_harness.py --case "plane_family/cone" --op "query[first,second]" --variant "outputs/random"`

### plane_family/cone / query[first,second] / outputs+evaluations/+1

- sites: Cone.local_uv (inventory A2, A4, F2, F3)
- changed: `/components: 1 -> 11`
- reference: `{"classified": true, "components": 1, "dimension": "curve", "kind": "cross"}`
- perturbed: `{"classified": true, "components": 11, "dimension": "curve", "kind": "cross"}`
- reproducer: `python tools/perturbation_harness.py --case "plane_family/cone" --op "query[first,second]" --variant "outputs+evaluations/+1"`

### plane_family/cone / query[first,second] / outputs+evaluations/random

- sites: Cone.local_uv (inventory A2, A4, F2, F3)
- changed: `/components: 1 -> 14`
- reference: `{"classified": true, "components": 1, "dimension": "curve", "kind": "cross"}`
- perturbed: `{"classified": true, "components": 14, "dimension": "curve", "kind": "cross"}`
- reproducer: `python tools/perturbation_harness.py --case "plane_family/cone" --op "query[first,second]" --variant "outputs+evaluations/random"`

### plane_family/coons / query[first,second] / outputs/-1

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/components: 1 -> 0`; `/dimension: "point" -> "none"`; `/kind: "touch_point" -> "disjoint"`
- reference: `{"classified": true, "components": 1, "dimension": "point", "kind": "touch_point"}`
- perturbed: `{"classified": true, "components": 0, "dimension": "none", "kind": "disjoint"}`
- reproducer: `python tools/perturbation_harness.py --case "plane_family/coons" --op "query[first,second]" --variant "outputs/-1"`

### plane_family/coons / query[first,second] / outputs/random

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/components: 1 -> 0`; `/dimension: "point" -> "none"`; `/kind: "touch_point" -> "disjoint"`
- reference: `{"classified": true, "components": 1, "dimension": "point", "kind": "touch_point"}`
- perturbed: `{"classified": true, "components": 0, "dimension": "none", "kind": "disjoint"}`
- reproducer: `python tools/perturbation_harness.py --case "plane_family/coons" --op "query[first,second]" --variant "outputs/random"`

### plane_family/coons / query[first,second] / outputs+evaluations/-1

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/components: 1 -> 0`; `/dimension: "point" -> "none"`; `/kind: "touch_point" -> "disjoint"`
- reference: `{"classified": true, "components": 1, "dimension": "point", "kind": "touch_point"}`
- perturbed: `{"classified": true, "components": 0, "dimension": "none", "kind": "disjoint"}`
- reproducer: `python tools/perturbation_harness.py --case "plane_family/coons" --op "query[first,second]" --variant "outputs+evaluations/-1"`

### plane_u0_edge/seed1 / query[first,second] / route/scalar

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/components: 0 -> 1`; `/dimension: "none" -> "curve"`; `/kind: "disjoint" -> "cross"`
- reference: `{"classified": true, "components": 0, "dimension": "none", "kind": "disjoint"}`
- perturbed: `{"classified": true, "components": 1, "dimension": "curve", "kind": "cross"}`
- reproducer: `python tools/perturbation_harness.py --case "plane_u0_edge/seed1" --op "query[first,second]" --variant "route/scalar"`

### exact_tol_vertex/prev(tol) / plan_apply / outputs/-1

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/digest: "a425e70ceefad07d" -> "1d063e60b70dac6c"`; `/shape/vertices: 8 -> 7`
- reference: `{"digest": "a425e70ceefad07d", "shape": {"edges": 8, "faces": 2, "vertices": 8}}`
- perturbed: `{"digest": "1d063e60b70dac6c", "shape": {"edges": 8, "faces": 2, "vertices": 7}}`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/prev(tol)" --op "plan_apply" --variant "outputs/-1"`

### exact_tol_vertex/prev(tol) / plan_apply / outputs/random

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/digest: "a425e70ceefad07d" -> "1d063e60b70dac6c"`; `/shape/vertices: 8 -> 7`
- reference: `{"digest": "a425e70ceefad07d", "shape": {"edges": 8, "faces": 2, "vertices": 8}}`
- perturbed: `{"digest": "1d063e60b70dac6c", "shape": {"edges": 8, "faces": 2, "vertices": 7}}`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/prev(tol)" --op "plan_apply" --variant "outputs/random"`

### exact_tol_vertex/prev(tol) / plan_apply / outputs+evaluations/-1

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/digest: "a425e70ceefad07d" -> "1d063e60b70dac6c"`; `/shape/vertices: 8 -> 7`
- reference: `{"digest": "a425e70ceefad07d", "shape": {"edges": 8, "faces": 2, "vertices": 8}}`
- perturbed: `{"digest": "1d063e60b70dac6c", "shape": {"edges": 8, "faces": 2, "vertices": 7}}`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/prev(tol)" --op "plan_apply" --variant "outputs+evaluations/-1"`

### exact_tol_vertex/prev(tol) / points[face] / outputs/-1

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/corner_b/chart_contains: false -> true`; `/edge_b/chart_contains: false -> true`
- reference: `{"corner_b": {"chart_contains": false, "chart_interior": false, "face_contains_uv": true, "projection_withi...`
- perturbed: `{"corner_b": {"chart_contains": true, "chart_interior": false, "face_contains_uv": true, "projection_within...`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/prev(tol)" --op "points[face]" --variant "outputs/-1"`

### exact_tol_vertex/prev(tol) / points[face] / outputs/random

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/corner_b/chart_contains: false -> true`; `/edge_b/chart_contains: false -> true`
- reference: `{"corner_b": {"chart_contains": false, "chart_interior": false, "face_contains_uv": true, "projection_withi...`
- perturbed: `{"corner_b": {"chart_contains": true, "chart_interior": false, "face_contains_uv": true, "projection_within...`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/prev(tol)" --op "points[face]" --variant "outputs/random"`

### exact_tol_vertex/prev(tol) / points[face] / outputs+evaluations/-1

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/corner_b/chart_contains: false -> true`; `/edge_b/chart_contains: false -> true`
- reference: `{"corner_b": {"chart_contains": false, "chart_interior": false, "face_contains_uv": true, "projection_withi...`
- perturbed: `{"corner_b": {"chart_contains": true, "chart_interior": false, "face_contains_uv": true, "projection_within...`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/prev(tol)" --op "points[face]" --variant "outputs+evaluations/-1"`

### exact_tol_vertex/prev(tol) / points[face] / outputs+evaluations/random

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/edge_b/chart_contains: false -> true`
- reference: `{"corner_b": {"chart_contains": false, "chart_interior": false, "face_contains_uv": true, "projection_withi...`
- perturbed: `{"corner_b": {"chart_contains": false, "chart_interior": false, "face_contains_uv": true, "projection_withi...`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/prev(tol)" --op "points[face]" --variant "outputs+evaluations/random"`

### exact_tol_vertex/tol / plan_apply / outputs/-1

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/digest: "a425e70ceefad07d" -> "1d063e60b70dac6c"`; `/shape/vertices: 8 -> 7`
- reference: `{"digest": "a425e70ceefad07d", "shape": {"edges": 8, "faces": 2, "vertices": 8}}`
- perturbed: `{"digest": "1d063e60b70dac6c", "shape": {"edges": 8, "faces": 2, "vertices": 7}}`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/tol" --op "plan_apply" --variant "outputs/-1"`

### exact_tol_vertex/tol / plan_apply / outputs/random

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/digest: "a425e70ceefad07d" -> "1d063e60b70dac6c"`; `/shape/vertices: 8 -> 7`
- reference: `{"digest": "a425e70ceefad07d", "shape": {"edges": 8, "faces": 2, "vertices": 8}}`
- perturbed: `{"digest": "1d063e60b70dac6c", "shape": {"edges": 8, "faces": 2, "vertices": 7}}`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/tol" --op "plan_apply" --variant "outputs/random"`

### exact_tol_vertex/tol / plan_apply / outputs+evaluations/-1

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/digest: "a425e70ceefad07d" -> "1d063e60b70dac6c"`; `/shape/vertices: 8 -> 7`
- reference: `{"digest": "a425e70ceefad07d", "shape": {"edges": 8, "faces": 2, "vertices": 8}}`
- perturbed: `{"digest": "1d063e60b70dac6c", "shape": {"edges": 8, "faces": 2, "vertices": 7}}`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/tol" --op "plan_apply" --variant "outputs+evaluations/-1"`

### exact_tol_vertex/tol / plan_apply / outputs+evaluations/random

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/digest: "a425e70ceefad07d" -> "656ed79add0e58b5"`
- reference: `{"digest": "a425e70ceefad07d", "shape": {"edges": 8, "faces": 2, "vertices": 8}}`
- perturbed: `{"digest": "656ed79add0e58b5", "shape": {"edges": 8, "faces": 2, "vertices": 8}}`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/tol" --op "plan_apply" --variant "outputs+evaluations/random"`

### exact_tol_vertex/tol / points[face] / outputs/-1

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/corner_b/chart_contains: false -> true`; `/edge_b/chart_contains: false -> true`
- reference: `{"corner_b": {"chart_contains": false, "chart_interior": false, "face_contains_uv": true, "projection_withi...`
- perturbed: `{"corner_b": {"chart_contains": true, "chart_interior": false, "face_contains_uv": true, "projection_within...`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/tol" --op "points[face]" --variant "outputs/-1"`

### exact_tol_vertex/tol / points[face] / outputs+evaluations/-1

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/corner_b/chart_contains: false -> true`; `/edge_b/chart_contains: false -> true`
- reference: `{"corner_b": {"chart_contains": false, "chart_interior": false, "face_contains_uv": true, "projection_withi...`
- perturbed: `{"corner_b": {"chart_contains": true, "chart_interior": false, "face_contains_uv": true, "projection_within...`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/tol" --op "points[face]" --variant "outputs+evaluations/-1"`

### exact_tol_vertex/tol / points[face] / outputs+evaluations/random

- sites: - (inventory -)
- changed: `/corner_b/chart_contains: false -> true`; `/edge_b/chart_contains: false -> true`
- reference: `{"corner_b": {"chart_contains": false, "chart_interior": false, "face_contains_uv": true, "projection_withi...`
- perturbed: `{"corner_b": {"chart_contains": true, "chart_interior": false, "face_contains_uv": true, "projection_within...`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/tol" --op "points[face]" --variant "outputs+evaluations/random"`

### exact_tol_vertex/next(tol) / plan_apply / outputs/-1

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/digest: "a425e70ceefad07d" -> "656ed79add0e58b5"`
- reference: `{"digest": "a425e70ceefad07d", "shape": {"edges": 8, "faces": 2, "vertices": 8}}`
- perturbed: `{"digest": "656ed79add0e58b5", "shape": {"edges": 8, "faces": 2, "vertices": 8}}`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/next(tol)" --op "plan_apply" --variant "outputs/-1"`

### exact_tol_vertex/next(tol) / plan_apply / outputs/random

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/digest: "a425e70ceefad07d" -> "656ed79add0e58b5"`
- reference: `{"digest": "a425e70ceefad07d", "shape": {"edges": 8, "faces": 2, "vertices": 8}}`
- perturbed: `{"digest": "656ed79add0e58b5", "shape": {"edges": 8, "faces": 2, "vertices": 8}}`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/next(tol)" --op "plan_apply" --variant "outputs/random"`

### exact_tol_vertex/next(tol) / plan_apply / outputs+evaluations/-1

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/digest: "a425e70ceefad07d" -> "1d063e60b70dac6c"`; `/shape/vertices: 8 -> 7`
- reference: `{"digest": "a425e70ceefad07d", "shape": {"edges": 8, "faces": 2, "vertices": 8}}`
- perturbed: `{"digest": "1d063e60b70dac6c", "shape": {"edges": 8, "faces": 2, "vertices": 7}}`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/next(tol)" --op "plan_apply" --variant "outputs+evaluations/-1"`

### exact_tol_vertex/next(tol) / points[face] / outputs/-1

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/corner_b/chart_contains: false -> true`; `/edge_b/chart_contains: false -> true`
- reference: `{"corner_b": {"chart_contains": false, "chart_interior": false, "face_contains_uv": true, "projection_withi...`
- perturbed: `{"corner_b": {"chart_contains": true, "chart_interior": false, "face_contains_uv": true, "projection_within...`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/next(tol)" --op "points[face]" --variant "outputs/-1"`

### exact_tol_vertex/next(tol) / points[face] / outputs/random

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/corner_b/chart_contains: false -> true`
- reference: `{"corner_b": {"chart_contains": false, "chart_interior": false, "face_contains_uv": true, "projection_withi...`
- perturbed: `{"corner_b": {"chart_contains": true, "chart_interior": false, "face_contains_uv": true, "projection_within...`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/next(tol)" --op "points[face]" --variant "outputs/random"`

### exact_tol_vertex/next(tol) / points[face] / outputs+evaluations/-1

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/corner_b/chart_contains: false -> true`; `/edge_b/chart_contains: false -> true`
- reference: `{"corner_b": {"chart_contains": false, "chart_interior": false, "face_contains_uv": true, "projection_withi...`
- perturbed: `{"corner_b": {"chart_contains": true, "chart_interior": false, "face_contains_uv": true, "projection_within...`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/next(tol)" --op "points[face]" --variant "outputs+evaluations/-1"`

### exact_tol_vertex/next(tol) / points[face] / outputs+evaluations/random

- sites: - (inventory -)
- changed: `/corner_b/chart_contains: false -> true`
- reference: `{"corner_b": {"chart_contains": false, "chart_interior": false, "face_contains_uv": true, "projection_withi...`
- perturbed: `{"corner_b": {"chart_contains": true, "chart_interior": false, "face_contains_uv": true, "projection_within...`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/next(tol)" --op "points[face]" --variant "outputs+evaluations/random"`

### exact_tol_vertex/offset1e-170 / query[first,second] / outputs+evaluations/random

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/components: 1 -> 0`; `/dimension: "point" -> "none"`; `/kind: "touch_point" -> "disjoint"`
- reference: `{"classified": true, "components": 1, "dimension": "point", "kind": "touch_point"}`
- perturbed: `{"classified": true, "components": 0, "dimension": "none", "kind": "disjoint"}`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/offset1e-170" --op "query[first,second]" --variant "outputs+evaluations/random"`

### exact_tol_vertex/offset1e-200 / query[first,second] / outputs/random

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/components: 1 -> 0`; `/dimension: "point" -> "none"`; `/kind: "touch_point" -> "disjoint"`
- reference: `{"classified": true, "components": 1, "dimension": "point", "kind": "touch_point"}`
- perturbed: `{"classified": true, "components": 0, "dimension": "none", "kind": "disjoint"}`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/offset1e-200" --op "query[first,second]" --variant "outputs/random"`

### exact_tol_vertex/offset1e-200 / query[first,second] / outputs+evaluations/random

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/components: 1 -> 0`; `/dimension: "point" -> "none"`; `/kind: "touch_point" -> "disjoint"`
- reference: `{"classified": true, "components": 1, "dimension": "point", "kind": "touch_point"}`
- perturbed: `{"classified": true, "components": 0, "dimension": "none", "kind": "disjoint"}`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/offset1e-200" --op "query[first,second]" --variant "outputs+evaluations/random"`

### exact_tol_vertex/offset5e-324 / query[first,second] / outputs/random

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/components: 1 -> 0`; `/dimension: "point" -> "none"`; `/kind: "touch_point" -> "disjoint"`
- reference: `{"classified": true, "components": 1, "dimension": "point", "kind": "touch_point"}`
- perturbed: `{"classified": true, "components": 0, "dimension": "none", "kind": "disjoint"}`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/offset5e-324" --op "query[first,second]" --variant "outputs/random"`

### exact_tol_vertex/tol@1e-150 / query[first,second] / outputs/+1

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/components: 1 -> 0`; `/dimension: "point" -> "none"`; `/kind: "touch_point" -> "disjoint"`
- reference: `{"classified": true, "components": 1, "dimension": "point", "kind": "touch_point"}`
- perturbed: `{"classified": true, "components": 0, "dimension": "none", "kind": "disjoint"}`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/tol@1e-150" --op "query[first,second]" --variant "outputs/+1"`

### exact_tol_vertex/tol@1e-150 / query[first,second] / outputs/-1

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/components: 1 -> 0`; `/dimension: "point" -> "none"`; `/kind: "touch_point" -> "disjoint"`
- reference: `{"classified": true, "components": 1, "dimension": "point", "kind": "touch_point"}`
- perturbed: `{"classified": true, "components": 0, "dimension": "none", "kind": "disjoint"}`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/tol@1e-150" --op "query[first,second]" --variant "outputs/-1"`

### exact_tol_vertex/tol@1e-150 / query[first,second] / outputs/random

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/components: 1 -> 0`; `/dimension: "point" -> "none"`; `/kind: "touch_point" -> "disjoint"`
- reference: `{"classified": true, "components": 1, "dimension": "point", "kind": "touch_point"}`
- perturbed: `{"classified": true, "components": 0, "dimension": "none", "kind": "disjoint"}`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/tol@1e-150" --op "query[first,second]" --variant "outputs/random"`

### exact_tol_vertex/tol@1e-150 / query[first,second] / outputs+evaluations/+1

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/components: 1 -> 0`; `/dimension: "point" -> "none"`; `/kind: "touch_point" -> "disjoint"`
- reference: `{"classified": true, "components": 1, "dimension": "point", "kind": "touch_point"}`
- perturbed: `{"classified": true, "components": 0, "dimension": "none", "kind": "disjoint"}`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/tol@1e-150" --op "query[first,second]" --variant "outputs+evaluations/+1"`

### exact_tol_vertex/tol@1e-150 / query[first,second] / outputs+evaluations/-1

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/components: 1 -> 0`; `/dimension: "point" -> "none"`; `/kind: "touch_point" -> "disjoint"`
- reference: `{"classified": true, "components": 1, "dimension": "point", "kind": "touch_point"}`
- perturbed: `{"classified": true, "components": 0, "dimension": "none", "kind": "disjoint"}`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/tol@1e-150" --op "query[first,second]" --variant "outputs+evaluations/-1"`

### exact_tol_vertex/tol@1e-150 / query[first,second] / route/scalar

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/components: 1 -> 0`; `/dimension: "point" -> "none"`; `/kind: "touch_point" -> "disjoint"`
- reference: `{"classified": true, "components": 1, "dimension": "point", "kind": "touch_point"}`
- perturbed: `{"classified": true, "components": 0, "dimension": "none", "kind": "disjoint"}`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/tol@1e-150" --op "query[first,second]" --variant "route/scalar"`

### exact_tol_vertex/tol@1e-150 / query[first,second] / route/fma

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/components: 1 -> 0`; `/dimension: "point" -> "none"`; `/kind: "touch_point" -> "disjoint"`
- reference: `{"classified": true, "components": 1, "dimension": "point", "kind": "touch_point"}`
- perturbed: `{"classified": true, "components": 0, "dimension": "none", "kind": "disjoint"}`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/tol@1e-150" --op "query[first,second]" --variant "route/fma"`

### exact_tol_vertex/tol@1e-100 / query[first,second] / outputs/+1

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/components: 1 -> 0`; `/dimension: "point" -> "none"`; `/kind: "touch_point" -> "disjoint"`
- reference: `{"classified": true, "components": 1, "dimension": "point", "kind": "touch_point"}`
- perturbed: `{"classified": true, "components": 0, "dimension": "none", "kind": "disjoint"}`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/tol@1e-100" --op "query[first,second]" --variant "outputs/+1"`

### exact_tol_vertex/tol@1e-100 / query[first,second] / outputs/-1

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/components: 1 -> 0`; `/dimension: "point" -> "none"`; `/kind: "touch_point" -> "disjoint"`
- reference: `{"classified": true, "components": 1, "dimension": "point", "kind": "touch_point"}`
- perturbed: `{"classified": true, "components": 0, "dimension": "none", "kind": "disjoint"}`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/tol@1e-100" --op "query[first,second]" --variant "outputs/-1"`

### exact_tol_vertex/tol@1e-100 / query[first,second] / outputs+evaluations/+1

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/components: 1 -> 0`; `/dimension: "point" -> "none"`; `/kind: "touch_point" -> "disjoint"`
- reference: `{"classified": true, "components": 1, "dimension": "point", "kind": "touch_point"}`
- perturbed: `{"classified": true, "components": 0, "dimension": "none", "kind": "disjoint"}`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/tol@1e-100" --op "query[first,second]" --variant "outputs+evaluations/+1"`

### exact_tol_vertex/tol@1e-100 / query[first,second] / outputs+evaluations/-1

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/components: 1 -> 0`; `/dimension: "point" -> "none"`; `/kind: "touch_point" -> "disjoint"`
- reference: `{"classified": true, "components": 1, "dimension": "point", "kind": "touch_point"}`
- perturbed: `{"classified": true, "components": 0, "dimension": "none", "kind": "disjoint"}`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/tol@1e-100" --op "query[first,second]" --variant "outputs+evaluations/-1"`

### exact_tol_vertex/tol@1e-100 / query[first,second] / outputs+evaluations/random

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/components: 1 -> 0`; `/dimension: "point" -> "none"`; `/kind: "touch_point" -> "disjoint"`
- reference: `{"classified": true, "components": 1, "dimension": "point", "kind": "touch_point"}`
- perturbed: `{"classified": true, "components": 0, "dimension": "none", "kind": "disjoint"}`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/tol@1e-100" --op "query[first,second]" --variant "outputs+evaluations/random"`

### exact_tol_vertex/tol@1e+20 / plan_apply / outputs/-1

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/digest: "a425e70ceefad07d" -> null`; `/error: null -> "GeometryError"`; `/message: null -> "intersection material arrangement face:#: arrangement faile`; `/shape: {"faces": 2, "edges": 8, "vertices": 8} -> null`
- reference: `{"digest": "a425e70ceefad07d", "shape": {"edges": 8, "faces": 2, "vertices": 8}}`
- perturbed: `{"error": "GeometryError", "message": "intersection material arrangement face:#: arrangement failed materia...`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/tol@1e+20" --op "plan_apply" --variant "outputs/-1"`

### exact_tol_vertex/tol@1e+20 / plan_apply / outputs/random

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/digest: "a425e70ceefad07d" -> null`; `/error: null -> "GeometryError"`; `/message: null -> "intersection material arrangement face:#: arrangement faile`; `/shape: {"faces": 2, "edges": 8, "vertices": 8} -> null`
- reference: `{"digest": "a425e70ceefad07d", "shape": {"edges": 8, "faces": 2, "vertices": 8}}`
- perturbed: `{"error": "GeometryError", "message": "intersection material arrangement face:#: arrangement failed materia...`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/tol@1e+20" --op "plan_apply" --variant "outputs/random"`

### exact_tol_vertex/tol@1e+20 / plan_apply / outputs+evaluations/-1

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/digest: "a425e70ceefad07d" -> null`; `/error: null -> "GeometryError"`; `/message: null -> "intersection material arrangement face:#: arrangement faile`; `/shape: {"faces": 2, "edges": 8, "vertices": 8} -> null`
- reference: `{"digest": "a425e70ceefad07d", "shape": {"edges": 8, "faces": 2, "vertices": 8}}`
- perturbed: `{"error": "GeometryError", "message": "intersection material arrangement face:#: arrangement failed materia...`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/tol@1e+20" --op "plan_apply" --variant "outputs+evaluations/-1"`

### exact_tol_vertex/tol@1e+50 / plan_apply / outputs/-1

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/digest: "a425e70ceefad07d" -> null`; `/error: null -> "GeometryError"`; `/message: null -> "intersection material arrangement face:#: arrangement faile`; `/shape: {"faces": 2, "edges": 8, "vertices": 8} -> null`
- reference: `{"digest": "a425e70ceefad07d", "shape": {"edges": 8, "faces": 2, "vertices": 8}}`
- perturbed: `{"error": "GeometryError", "message": "intersection material arrangement face:#: arrangement failed materia...`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/tol@1e+50" --op "plan_apply" --variant "outputs/-1"`

### exact_tol_vertex/tol@1e+50 / plan_apply / outputs+evaluations/-1

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/digest: "a425e70ceefad07d" -> null`; `/error: null -> "GeometryError"`; `/message: null -> "intersection material arrangement face:#: arrangement faile`; `/shape: {"faces": 2, "edges": 8, "vertices": 8} -> null`
- reference: `{"digest": "a425e70ceefad07d", "shape": {"edges": 8, "faces": 2, "vertices": 8}}`
- perturbed: `{"error": "GeometryError", "message": "intersection material arrangement face:#: arrangement failed materia...`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/tol@1e+50" --op "plan_apply" --variant "outputs+evaluations/-1"`

### exact_tol_vertex/tol@1e+50 / plan_apply / route/scalar

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/digest: "a425e70ceefad07d" -> null`; `/error: null -> "GeometryError"`; `/message: null -> "intersection material arrangement face:#: arrangement faile`; `/shape: {"faces": 2, "edges": 8, "vertices": 8} -> null`
- reference: `{"digest": "a425e70ceefad07d", "shape": {"edges": 8, "faces": 2, "vertices": 8}}`
- perturbed: `{"error": "GeometryError", "message": "intersection material arrangement face:#: arrangement failed materia...`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/tol@1e+50" --op "plan_apply" --variant "route/scalar"`

### exact_tol_vertex/tol@1e+50 / plan_apply / route/fma

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/digest: "a425e70ceefad07d" -> null`; `/error: null -> "GeometryError"`; `/message: null -> "intersection material arrangement face:#: arrangement faile`; `/shape: {"faces": 2, "edges": 8, "vertices": 8} -> null`
- reference: `{"digest": "a425e70ceefad07d", "shape": {"edges": 8, "faces": 2, "vertices": 8}}`
- perturbed: `{"error": "GeometryError", "message": "intersection material arrangement face:#: arrangement failed materia...`
- reproducer: `python tools/perturbation_harness.py --case "exact_tol_vertex/tol@1e+50" --op "plan_apply" --variant "route/fma"`

### scaled_crossing/1e-100 / plan_apply / outputs+evaluations/+1

- sites: LinePath.evaluate (inventory B1, B11, B3, C1, D1, D6)
- changed: `/digest: null -> "bed19206b7c8fa93"`; `/error: "GeometryError" -> null`; `/message: "intersection material arrangement face:#: arrangement conta -> null`; `/shape: null -> {"faces": 2, "edges": 10, "vertices": 12}`
- reference: `{"error": "GeometryError", "message": "intersection material arrangement face:#: arrangement contains an un...`
- perturbed: `{"digest": "bed19206b7c8fa93", "shape": {"edges": 10, "faces": 2, "vertices": 12}}`
- reproducer: `python tools/perturbation_harness.py --case "scaled_crossing/1e-100" --op "plan_apply" --variant "outputs+evaluations/+1"`

### scaled_crossing/1e-100 / plan_apply / outputs+evaluations/-1

- sites: LinePath.evaluate (inventory B1, B11, B3, C1, D1, D6)
- changed: `/digest: null -> "bed19206b7c8fa93"`; `/error: "GeometryError" -> null`; `/message: "intersection material arrangement face:#: arrangement conta -> null`; `/shape: null -> {"faces": 2, "edges": 10, "vertices": 12}`
- reference: `{"error": "GeometryError", "message": "intersection material arrangement face:#: arrangement contains an un...`
- perturbed: `{"digest": "bed19206b7c8fa93", "shape": {"edges": 10, "faces": 2, "vertices": 12}}`
- reproducer: `python tools/perturbation_harness.py --case "scaled_crossing/1e-100" --op "plan_apply" --variant "outputs+evaluations/-1"`

### scaled_crossing/1e-100 / plan_apply / outputs+evaluations/random

- sites: LinePath.evaluate (inventory B1, B11, B3, C1, D1, D6)
- changed: `/digest: null -> "bed19206b7c8fa93"`; `/error: "GeometryError" -> null`; `/message: "intersection material arrangement face:#: arrangement conta -> null`; `/shape: null -> {"faces": 2, "edges": 10, "vertices": 12}`
- reference: `{"error": "GeometryError", "message": "intersection material arrangement face:#: arrangement contains an un...`
- perturbed: `{"digest": "bed19206b7c8fa93", "shape": {"edges": 10, "faces": 2, "vertices": 12}}`
- reproducer: `python tools/perturbation_harness.py --case "scaled_crossing/1e-100" --op "plan_apply" --variant "outputs+evaluations/random"`

### scaled_crossing/1e-100 / query[first,second] / outputs/+1

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/components: 1 -> 0`; `/dimension: "point" -> "none"`; `/kind: "touch_point" -> "disjoint"`
- reference: `{"classified": true, "components": 1, "dimension": "point", "kind": "touch_point"}`
- perturbed: `{"classified": true, "components": 0, "dimension": "none", "kind": "disjoint"}`
- reproducer: `python tools/perturbation_harness.py --case "scaled_crossing/1e-100" --op "query[first,second]" --variant "outputs/+1"`

### scaled_crossing/1e-100 / query[first,second] / outputs/-1

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/components: 1 -> 0`; `/dimension: "point" -> "none"`; `/kind: "touch_point" -> "disjoint"`
- reference: `{"classified": true, "components": 1, "dimension": "point", "kind": "touch_point"}`
- perturbed: `{"classified": true, "components": 0, "dimension": "none", "kind": "disjoint"}`
- reproducer: `python tools/perturbation_harness.py --case "scaled_crossing/1e-100" --op "query[first,second]" --variant "outputs/-1"`

### scaled_crossing/1e-100 / query[first,second] / outputs+evaluations/+1

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/components: 1 -> 0`; `/dimension: "point" -> "none"`; `/kind: "touch_point" -> "disjoint"`
- reference: `{"classified": true, "components": 1, "dimension": "point", "kind": "touch_point"}`
- perturbed: `{"classified": true, "components": 0, "dimension": "none", "kind": "disjoint"}`
- reproducer: `python tools/perturbation_harness.py --case "scaled_crossing/1e-100" --op "query[first,second]" --variant "outputs+evaluations/+1"`

### scaled_crossing/1e-100 / query[first,second] / outputs+evaluations/-1

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/components: 1 -> 0`; `/dimension: "point" -> "none"`; `/kind: "touch_point" -> "disjoint"`
- reference: `{"classified": true, "components": 1, "dimension": "point", "kind": "touch_point"}`
- perturbed: `{"classified": true, "components": 0, "dimension": "none", "kind": "disjoint"}`
- reproducer: `python tools/perturbation_harness.py --case "scaled_crossing/1e-100" --op "query[first,second]" --variant "outputs+evaluations/-1"`

### scaled_crossing/1e-100 / query[first,second] / outputs+evaluations/random

- sites: Plane.local_uv (inventory A1, D1, D4, E4)
- changed: `/components: 1 -> 0`; `/dimension: "point" -> "none"`; `/kind: "touch_point" -> "disjoint"`
- reference: `{"classified": true, "components": 1, "dimension": "point", "kind": "touch_point"}`
- perturbed: `{"classified": true, "components": 0, "dimension": "none", "kind": "disjoint"}`
- reproducer: `python tools/perturbation_harness.py --case "scaled_crossing/1e-100" --op "query[first,second]" --variant "outputs+evaluations/random"`

### arc_opening/flat / plan_apply / outputs+evaluations/random

- sites: LinePath.evaluate (inventory B1, B11, B3, C1, D1, D6)
- changed: `/digest: "408b4a37e4c5345b" -> "3170cb4ebd63f2c0"`
- reference: `{"digest": "408b4a37e4c5345b", "shape": {"edges": 40, "faces": 12, "vertices": 36}}`
- perturbed: `{"digest": "3170cb4ebd63f2c0", "shape": {"edges": 40, "faces": 12, "vertices": 36}}`
- reproducer: `python tools/perturbation_harness.py --case "arc_opening/flat" --op "plan_apply" --variant "outputs+evaluations/random"`

### arc_opening/tilted / query[first,second] / outputs/-1

- sites: Plane.local_uv, freeze_edge.arc (inventory A1, B11, B2, C1, C4, D1, D4, E4)
- changed: `/components: 2 -> 3`
- reference: `{"classified": true, "components": 2, "dimension": "curve", "kind": "cross"}`
- perturbed: `{"classified": true, "components": 3, "dimension": "curve", "kind": "cross"}`
- reproducer: `python tools/perturbation_harness.py --case "arc_opening/tilted" --op "query[first,second]" --variant "outputs/-1"`

### arc_opening/tilted / query[first,second] / outputs+evaluations/random

- sites: - (inventory -)
- changed: `/components: 2 -> 3`
- reference: `{"classified": true, "components": 2, "dimension": "curve", "kind": "cross"}`
- perturbed: `{"classified": true, "components": 3, "dimension": "curve", "kind": "cross"}`
- reproducer: `python tools/perturbation_harness.py --case "arc_opening/tilted" --op "query[first,second]" --variant "outputs+evaluations/random"`

### arc_opening/tilted / plan_apply / outputs/-1

- sites: Plane.local_uv, freeze_edge.arc (inventory A1, B11, B2, C1, C4, D1, D4, E4)
- changed: `/digest: "1361ec20b307aecf" -> "fedc1aea99deed8d"`; `/shape/edges: 40 -> 32`; `/shape/faces: 12 -> 8`; `/shape/vertices: 36 -> 32`
- reference: `{"digest": "1361ec20b307aecf", "shape": {"edges": 40, "faces": 12, "vertices": 36}}`
- perturbed: `{"digest": "fedc1aea99deed8d", "shape": {"edges": 32, "faces": 8, "vertices": 32}}`
- reproducer: `python tools/perturbation_harness.py --case "arc_opening/tilted" --op "plan_apply" --variant "outputs/-1"`

### native_arc_station/split_cylinder / station_selection / outputs/+1

- sites: freeze_edge.arc (inventory B11, B2, C1, C4)
- changed: `/records[3]/current_zero: false -> true`; `/records[7]/current_zero: true -> false`
- reference: `{"digest": "5fa1174281038f7b", "records": [{"ancestor_zero": false, "classification": "bounded", "current_z...`
- perturbed: `{"digest": "5fa1174281038f7b", "records": [{"ancestor_zero": false, "classification": "bounded", "current_z...`
- reproducer: `python tools/perturbation_harness.py --case "native_arc_station/split_cylinder" --op "station_selection" --variant "outputs/+1"`

### native_arc_station/split_cylinder / station_selection / outputs/random

- sites: freeze_edge.arc (inventory B11, B2, C1, C4)
- changed: `/records[7]/current_zero: true -> false`
- reference: `{"digest": "5fa1174281038f7b", "records": [{"ancestor_zero": false, "classification": "bounded", "current_z...`
- perturbed: `{"digest": "5fa1174281038f7b", "records": [{"ancestor_zero": false, "classification": "bounded", "current_z...`
- reproducer: `python tools/perturbation_harness.py --case "native_arc_station/split_cylinder" --op "station_selection" --variant "outputs/random"`

### native_arc_station/split_cylinder / station_selection / outputs+evaluations/+1

- sites: EllipticArc.evaluate, LinePath.evaluate, freeze_edge.arc (inventory B1, B11, B2, B3, B5, C1, C3, C4, D1, D6)
- changed: `/records[7]/current_zero: true -> false`
- reference: `{"digest": "5fa1174281038f7b", "records": [{"ancestor_zero": false, "classification": "bounded", "current_z...`
- perturbed: `{"digest": "5fa1174281038f7b", "records": [{"ancestor_zero": false, "classification": "bounded", "current_z...`
- reproducer: `python tools/perturbation_harness.py --case "native_arc_station/split_cylinder" --op "station_selection" --variant "outputs+evaluations/+1"`

### native_arc_station/split_cylinder / station_selection / outputs+evaluations/-1

- sites: LinePath.evaluate (inventory B1, B11, B3, C1, D1, D6)
- changed: `/records[7]/current_zero: true -> false`
- reference: `{"digest": "5fa1174281038f7b", "records": [{"ancestor_zero": false, "classification": "bounded", "current_z...`
- perturbed: `{"digest": "5fa1174281038f7b", "records": [{"ancestor_zero": false, "classification": "bounded", "current_z...`
- reproducer: `python tools/perturbation_harness.py --case "native_arc_station/split_cylinder" --op "station_selection" --variant "outputs+evaluations/-1"`

### native_arc_station/split_cylinder / station_selection / outputs+evaluations/random

- sites: EllipticArc.evaluate, LinePath.evaluate, freeze_edge.arc (inventory B1, B11, B2, B3, B5, C1, C3, C4, D1, D6)
- changed: `/records[6]/current_zero: false -> true`; `/records[7]/current_zero: true -> false`
- reference: `{"digest": "5fa1174281038f7b", "records": [{"ancestor_zero": false, "classification": "bounded", "current_z...`
- perturbed: `{"digest": "5fa1174281038f7b", "records": [{"ancestor_zero": false, "classification": "bounded", "current_z...`
- reproducer: `python tools/perturbation_harness.py --case "native_arc_station/split_cylinder" --op "station_selection" --variant "outputs+evaluations/random"`

### native_arc_station/split_cylinder / station_selection / route/fma

- sites: EllipticArc.evaluate (inventory B11, B2, B5, C1, C3)
- changed: `/records[7]/current_zero: true -> false`
- reference: `{"digest": "5fa1174281038f7b", "records": [{"ancestor_zero": false, "classification": "bounded", "current_z...`
- perturbed: `{"digest": "5fa1174281038f7b", "records": [{"ancestor_zero": false, "classification": "bounded", "current_z...`
- reproducer: `python tools/perturbation_harness.py --case "native_arc_station/split_cylinder" --op "station_selection" --variant "route/fma"`

### g0/tilted8 / plan_apply / outputs+evaluations/random

- sites: LinePath.evaluate (inventory B1, B11, B3, C1, D1, D6)
- changed: `/digest: "06f08c8db11be1c4" -> "7236326a70bdf6eb"`
- reference: `{"digest": "06f08c8db11be1c4", "shape": {"edges": 1928, "faces": 914, "vertices": 994}}`
- perturbed: `{"digest": "7236326a70bdf6eb", "shape": {"edges": 1928, "faces": 914, "vertices": 994}}`
- reproducer: `python tools/perturbation_harness.py --case "g0/tilted8" --op "plan_apply" --variant "outputs+evaluations/random"`

## Site coverage (operation runs in which the site was called)

| Site | Runs |
|---|---|
| Plane.local_uv | 535 |
| Cylinder.local_uv | 72 |
| Cone.local_uv | 16 |
| ExtrudedSurface.local_uv | 6 |
| ExtrudedSurface.local_uv_many | 6 |
| RuledSurface.local_uv | 6 |
| CoonsSurface.local_uv | 0 |
| closest_uv | 12 |
| freeze_edge.arc | 96 |
| Plane.evaluate | 250 |
| Cylinder.evaluate | 35 |
| Cone.evaluate | 6 |
| ExtrudedSurface.evaluate | 0 |
| _evaluate_surface_many | 66 |
| _surface_derivatives_many | 15 |
| LinePath.evaluate | 270 |
| LinePath.derivative | 108 |
| BezierPath.evaluate | 3 |
| BezierPath.derivative | 3 |
| EllipticArc.evaluate | 80 |
| EllipticArc.derivative | 39 |
| freeze_edge.line | 0 |
| freeze_edge.spline | 0 |
