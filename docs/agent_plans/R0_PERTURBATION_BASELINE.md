# R0 perturbation harness: baseline flip report

Slice R0, instrument 2 of `GENERALITY_ARCHITECTURE.md` R.4. Branch
`claude/geom-r0-harness`; engine source = local main `99a4712` (the branch only
adds `benchmarks/generality` from `claude/geom-g0-corpus`, the two tools, one
test file, this note and the report files). No product code is changed.

## Files

| File | Content |
|---|---|
| `tools/perturbation_harness.py` | perturbation / route-switch / reversal harness, JSON + Markdown reports, merge |
| `tools/adversarial_corpus.py` | adversarial cases plus the G0 workloads, each a builder and its operations |
| `tests/test_perturbation_harness.py` | tests that the harness works (18 tests, < 1 s) |
| `reports/r0_perturbation/baseline_99a4712.{json,md}` | the full baseline run below (all flips with reproducers, site coverage) |

## How to run

```powershell
$env:PYTHONPATH = "<worktree>\src"
python tools/perturbation_harness.py --list                       # cases, operations, sites
python tools/perturbation_harness.py --group adversarial --json adv.json --markdown adv.md   # ~2.8 min
python tools/perturbation_harness.py --group g0 --json g0.json --markdown g0.md               # ~4.4 min
python tools/perturbation_harness.py --merge adv.json g0.json --json all.json --markdown all.md
# one flip:
python tools/perturbation_harness.py --case "plane_family/coons" --op "query[first,second]" --variant "outputs/-1" [--sites Plane.local_uv]
# tests of the harness itself
python -m pytest tests/test_perturbation_harness.py -p no:cacheprovider -o addopts=""
```

Options: `--case NAME` (repeatable; `prefix*` allowed), `--op`, `--variant`
(repeatable), `--sites a,b` (restrict perturbed or route-switched sites; also
opt-in for the stored-data sites `freeze_edge.line` / `freeze_edge.spline`),
`--seed`, `--heavy` (adds `g0/grid32`), `--no-attribution`.

## What it does

For each case and operation it builds a fresh model, empties every
`functools` cache in `anygeometry`, runs the operation and records a float-free
outcome: the topology digest of the resulting document (floats replaced by a
marker, as `benchmarks/generality/run.py split_document`) and entity counts for
plan/apply; kind, `classified`, component count and dimension of an
`IntersectionResult`; the point-membership decisions (`MaterialDomain.contains`
of the trimmed chart with closed and open boundary, `face_contains_uv` of
`face_local_uv`, `project_to_face` distance `<= tol`); the chart structure of
`query_trimmed_surface_charts`; native-arc station classifications; or the
raised error type and message with numbers masked. Coordinates are not
compared (R.3 allows them to move within a bound; that check is not part of
this slice).

Variants, each compared with the unperturbed run of the same operation:

| Variant | Effect |
|---|---|
| `none` | no wrapping; a repeat run (determinism check) |
| `outputs/{+1,-1,random}` | every chart inversion output (`local_uv` of Plane, Cylinder, Cone, Extruded, Ruled, Coons; `local_uv_many`; `closest_uv`) and the derived `freeze_edge` product of an arc (the `EllipticArc` centre and axes computed from three vertices) moved by one ulp |
| `outputs+evaluations/{+1,-1,random}` | as above plus every evaluation and derivative at the capability boundary: surface `evaluate`, `_evaluate_surface_many`, `_surface_derivatives_many`, `LinePath`/`BezierPath`/`EllipticArc` `evaluate` and `derivative` |
| `route/numpy` | the product's vectorized NumPy formula also where it takes a scalar fast path (`LinePath.evaluate`) |
| `route/scalar` | Python-scalar formulas: closed-form Cramer on the normal equations for `Plane.local_uv`, `math.atan2` and scalar dot products for Cylinder/Cone `local_uv`, `math.cos/sin` for Cylinder and `EllipticArc` evaluation, scalar Plane/Line evaluation |
| `route/fma` | the scalar route with `math.fma` (Python 3.14 here) in dot products, 2x2 determinants and affine combinations |
| `reverse` | unperturbed, operands reversed: face handle order for plan/apply, argument order for `query_intersection` (reversible operations only) |

Patterns: `+1` / `-1` move every finite float of the value up / down by one
ulp (`0.0` becomes `±5e-324`), `random` draws the direction per float from a
generator seeded by `crc32(seed, case, operation, variant)`. A value is
perturbed only at the outermost wrapped call (a wrapped call inside another is
passed through), so a consumer sees exactly one ulp per value. Wrapping is by
attribute assignment in the harness process (module-level functions are
replaced in every `anygeometry` module that imported them) and is restored
after each run; the tests check the restore.

Attribution: when a combined variant flips, the harness re-runs it with each
site that was actually called enabled alone; the sites that flip alone are
listed with their R.1 inventory ids. "(combined only)" means no single site
reproduces the flip (an interaction of sites or of the random sign stream).
Inventory ids are the R.1 entries whose decisions consume that site's values:
they name the candidate decision sites, not a proven single culprit.

## What it proves and what it does not

It **falsifies**: every flip below is a deterministic reproducer of a decision
that depends on the last bit of an intermediate (or on the arithmetic route),
i.e. evidence of finding F1 on that path. A flip under `outputs` / `route/*`
models a legitimate alternative rounding; a flip under `outputs+evaluations`
also counts, since the engine treats evaluations as data (B11).

It **does not prove** robustness anywhere: a case without flips was tested
with three perturbation patterns, three routes and one seed on a finite
corpus. Perturbation happens only at the wrapped boundary (intermediates
inside a wrapped function, NumPy internals and unwrapped helpers are not
moved), so a path that never crosses a wrapped site is not exercised at all;
the site coverage table in the report shows which sites the corpus reaches
(Extruded, Coons `local_uv` and Bezier evaluation are not reached by these
cases). Absence of flips on the scanned modules is instrument 1's job
(structural audit), not this one. Outcomes are compared on decisions and
topology only, not coordinates against a bound `B`; no decision trace is
compared (the trace in `runtime_diagnostics` is a separate R0 deliverable).

## Baseline on local main 99a4712

Python 3.14.2, NumPy 2.4.6, seed 0, all 11 variants, default case set (35
cases, 74 operations; `g0/grid32` and `g0/doc60` excluded). Wall time: 165 s
(adversarial) + 261 s (G0) = **426 s**.

Summary:

- 75 flips in 19 of 35 cases. No flip under `none` (the runs are
  deterministic), **no reversal flip** (all 57 reversible operations give the
  same outcome in both operand orders), no flip under `route/numpy`.
- By variant: `outputs` 33, `outputs+evaluations` 36, `route/scalar` 3,
  `route/fma` 3.
- By attribution (sites that reproduce the flip alone): `Plane.local_uv`
  alone 51, `LinePath.evaluate` alone 6, `Cylinder.local_uv` 4,
  `Cone.local_uv` 4, `freeze_edge.arc` alone 2, `EllipticArc.evaluate`
  alone 1; several sites each alone 4 (`Plane.local_uv` + `freeze_edge.arc` 2;
  `EllipticArc.evaluate` + `LinePath.evaluate` + `freeze_edge.arc` 2);
  combined only 3.
- Inventory entries with flips (through their sites): A1, A2/A4 (cylinder,
  cone chart), B1/B3/B11/C1/D6 (line evaluation), B2/C1/C4 (arc definition),
  E4/D1 (planar membership).
- G0: only `tilted8` flips (`outputs+evaluations/random`, `LinePath.evaluate`),
  matching the earlier `perturb.py` baseline (only `tilted8` under line
  evaluation). The other five G0 workloads are stable under all 11 variants.
- The S1 seam case (`seam_cylinder`) shows **no decision flip** in this
  construction: the scalar/FMA routes reproduce the review's `u = 0.0` vs
  `1.42e-14` difference (asserted in the tests) but every membership decision
  on that face is the same, because the seam is the face boundary here. The
  review's `MaterialDomain.contains` flip needs its original domain, which is
  not reconstructed (open item).

Flip table (variant groups abbreviated: `o` = `outputs`, `oe` =
`outputs+evaluations`; all reproducers are in the report; the general form is
`python tools/perturbation_harness.py --case "<case>" --op "<operation>" --variant "<variant>"`,
add `--sites <site>` to reproduce the single-site attribution):

| Case | Operation | Variants that flip | Site(s) | What changes |
|---|---|---|---|---|
| plane_family/cylinder | query | o/+1, oe/+1 | Cylinder.local_uv | CROSS -> DISJOINT |
| plane_family/cylinder | query | o/random, oe/random | Cylinder.local_uv | CROSS, 1 -> 7 components |
| plane_family/cone | query | o/+1, o/random, oe/+1, oe/random | Cone.local_uv | CROSS, 1 -> 11..14 components |
| plane_family/coons | query | o/-1, o/random, oe/-1 | Plane.local_uv | TOUCH_POINT -> DISJOINT |
| plane_u0_edge/seed1 | query | route/scalar | Plane.local_uv | DISJOINT -> CROSS |
| exact_tol_vertex/prev(tol) | plan_apply | o/-1, o/random, oe/-1 | Plane.local_uv | digest; 8 -> 7 vertices |
| exact_tol_vertex/prev(tol) | points | o/-1, o/random, oe/-1, oe/random | Plane.local_uv | chart_contains False -> True |
| exact_tol_vertex/tol | plan_apply | o/-1, o/random, oe/-1, oe/random | Plane.local_uv | digest; 8 -> 7 vertices |
| exact_tol_vertex/tol | points | o/-1, oe/-1, oe/random (combined only) | Plane.local_uv | chart_contains False -> True |
| exact_tol_vertex/next(tol) | plan_apply | o/-1, o/random, oe/-1 | Plane.local_uv | digest |
| exact_tol_vertex/next(tol) | points | o/-1, o/random, oe/-1, oe/random (combined only) | Plane.local_uv | chart_contains False -> True |
| exact_tol_vertex/offset1e-170 | query | oe/random | Plane.local_uv | TOUCH_POINT -> DISJOINT |
| exact_tol_vertex/offset1e-200 | query | o/random, oe/random | Plane.local_uv | TOUCH_POINT -> DISJOINT |
| exact_tol_vertex/offset5e-324 | query | o/random | Plane.local_uv | TOUCH_POINT -> DISJOINT |
| exact_tol_vertex/tol@1e-150 | query | o/±1, o/random, oe/±1, route/scalar, route/fma | Plane.local_uv | TOUCH_POINT -> DISJOINT |
| exact_tol_vertex/tol@1e-100 | query | o/±1, oe/±1, oe/random | Plane.local_uv | TOUCH_POINT -> DISJOINT |
| exact_tol_vertex/tol@1e+20 | plan_apply | o/-1, o/random, oe/-1 | Plane.local_uv | success -> "arrangement failed material conservation" |
| exact_tol_vertex/tol@1e+50 | plan_apply | o/-1, oe/-1, route/scalar, route/fma | Plane.local_uv | success -> "arrangement failed material conservation" |
| scaled_crossing/1e-100 | plan_apply | oe/+1, oe/-1, oe/random | LinePath.evaluate | "unresolved bridge" refusal -> success |
| scaled_crossing/1e-100 | query | o/±1, oe/±1, oe/random | Plane.local_uv | TOUCH_POINT -> DISJOINT |
| arc_opening/flat | plan_apply | oe/random | LinePath.evaluate | digest |
| arc_opening/tilted | query | o/-1; oe/random (combined only) | Plane.local_uv, freeze_edge.arc | CROSS, 2 -> 3 components |
| arc_opening/tilted | plan_apply | o/-1 | Plane.local_uv, freeze_edge.arc | digest; 12 -> 8 faces |
| native_arc_station/split_cylinder | station_selection | o/+1, o/random | freeze_edge.arc | which record has `current_bound == 0` |
| native_arc_station/split_cylinder | station_selection | oe/+1, oe/-1, oe/random | EllipticArc.evaluate, LinePath.evaluate, freeze_edge.arc | which record has `current_bound == 0` |
| native_arc_station/split_cylinder | station_selection | route/fma | EllipticArc.evaluate | which record has `current_bound == 0` |
| g0/tilted8 | plan_apply | oe/random | LinePath.evaluate | digest |

Cases without any flip under all variants: `plane_family/plane`,
`plane_family/ruled`, `plane_u0_edge/seed2|3` (query, plan/apply, points),
`exact_tol_vertex/0.75tol`, `exact_tol_vertex/1.5tol`, the plan/apply of the
underflow-offset and `tol@1e-150|1e-100` cases, `scaled_crossing/1000|1e+08`,
`float32_tilted3`, `seam_cylinder`, `arc_opening/*` points and charts,
`native_arc_station/nice` (0.75 tol bounded, 1.5 tol refused: both stable,
these use exact `Fraction` proofs), and G0 `pipes6`, `rings8`, `grid16`,
`cubic_wall_pipe`, `cone_pipe`.

Observations for the R-slices (from the baseline, not acted on):

- `Plane.local_uv` dominates (53 of 75 flips name it). A 1-ulp move of `u = 0.0` to
  `-5e-324` changes vertex identity at exact tolerance (plan/apply digests),
  planar membership, and query kinds, including at unit scale
  (`plane_family/coons`). R1's exact planar chart is the first lever.
- `plane_u0_edge`: on points stored as the rounded evaluation of the `u = 0`
  edge, the point-membership decisions did not flip on any seed (the
  `<= tol` boundary test absorbs the sign of `u`); the face x plate query of
  seed 1 flips DISJOINT -> CROSS under the scalar (Cramer) chart route.
- Exact-tolerance construction is not monotone at baseline: `prev(tol)`,
  `tol` and `next(tol)` all give the 8-vertex result while `0.75 tol` merges to 7;
  perturbation of `Plane.local_uv` flips `prev(tol)` and `tol` to 7.
- A 1-ulp change of a cylinder or cone chart output turns a plane x cylinder
  CROSS into DISJOINT or into 7-14 components (A2/E4/E5 on the query route).
- `native_arc_station/split_cylinder`: the record selected by
  `current_bound == 0` in `test_direct_ancestor_versus_double_tolerance`
  depends on the arc frame's last bit and on FMA in ellipse evaluation (R.0
  item confirmed).
- At extreme scales refusals and successes swap (`tol@1e+20`, `tol@1e+50`,
  `scaled_crossing/1e-100`). Whole-model scales of `1e100` and above are
  refused at construction ("invalid surface geometry") and are not in the
  corpus; `scaled_crossing` at `1e-150`/`1e-160` and `1e10..1e30` also fails
  unperturbed and is not included.

## Open items

- Coordinate bound `B` and decision-trace comparison (R.4) are not
  implemented; outcomes are decisions/topology only.
- Corpus gaps from R.4: the rev-4 paraboloid-cap plane section and a Bezier
  opening (the arc opening is included); Extruded, Coons chart inversion and
  Bezier evaluation are never reached by the default case set.
- The seam case's chart-membership flip from review s1c is not reproduced
  (see above).
- Single-seed baseline; `--seed` variation and `g0/grid32` (`--heavy`) were
  not run.
- `tools/dev_checks.py` maps new `tools/*.py` files to the full suite; the
  full suite was not run (out of scope; no product code changed).
