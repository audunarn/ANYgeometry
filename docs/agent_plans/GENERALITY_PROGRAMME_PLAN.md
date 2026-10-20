# ANYgeometry generality + speed programme — plan rev 2 (2026-10-13)

Living plan. Study and measurements: `GENERALITY_AND_SPEED_DEEP_DIVE_20261013.md`.
Written on branch `claude/generality-programme`; on main since 2026-10-19. The sections
below the status section are the history as written at the time.

## Current status (2026-10-19) — read this first

Everything listed as merged below is on GitHub main (df0fe27 and later); statements of
"local main, not pushed" are history. Delivered 2026-10-13..16:

- Robust planar decisions (R0-R1c, T-b/D8 E1-E3 + E2a): exact or certified containment,
  junctions, ordering, cells and areas on straight/arc loops; perturbation flips 75 -> 14
  (none on planes).
- NURBS track to B1: NURBS curves, surfaces and faces (schema 7); exact plane x NURBS
  sections (`PlaneNurbsCurve`, records `plane_nurbs_exact_v2` recomputed on load); plane and
  NURBS faces trimmed by sections; strict audit with the tolerance-band face-pair contract
  (P4e); the B1 test hull (half and full), closed shell, volume and centroid. B1 here means
  the plan's P5 sense (frame faces); stiffener and frame *curves* on the shell
  (architecture 4.10) are not done.
- H1 hydrostatics for ANYstability: volume, centre of buoyancy, waterplane, BM, tables, GZ
  with free trim, closed waterlines. Quantities are estimates with a refusal threshold, not
  rigorous bounds (rigorous bounds are R5).
- Speed: bit-identical S1 subset (~1.05-1.16x); S2 below its acceptance (branch deleted);
  N0 + N1a opt-in native scaffold (wheel CI never run).
- CI tiers (`CI_TIERS_PLAN.md`): development path 42 -> 14 min; weekly sharded final path;
  final-path portability fixes (incl. the chart-bound snap in `_child_support`).

Remaining, in the recommended order (owner to confirm each slice):

1. Hygiene: DONE 2026-10-19 (final path green on 9fb68e2; download-artifact on Node 24;
   stale worktrees and branches removed, E5 restored; this document on main; known local
   failures in `docs/testing.md`).
2. Confirmation review of the final fixes merged without a confirming round: DONE
   2026-10-20. 18 commits, 13 confirmed; 5 major + 2 minor findings fixed on
   `claude/confirm-fixes` (arc axes far from scale 1; nested lobes of a compound hole walk;
   NURBS trims with an explicit parameterization and `curve_samples`; `_tight_x` charged to
   the qualification budget; hydrostatic sweeps with the caller's exact normal, exact
   bracket heights, any finite normal), four Sol rounds to clean.
3. Real-bow probe: decide Q2 (bow form) and build the 15-20 m bow joined to the midship
   planes; expected refusals: a patch boundary in the collision-bulkhead plane (P1), the
   stem (two patches need NURBS x NURBS adjacency, 4L), smooth runs tangent to flat
   bottom/side planes (plane x NURBS is transversal-only).
4. B2 design revision: A1d `(PLANE, NURBS)` dispatch, A4c' junction pairs, A4d'
   attachments, rewritten for the exact P-series design (architecture 4.5 still describes
   the superseded interval-Newton `BranchPath`), with the R2/R3 scope it needs.
5. M1 prerequisite: `trim_loops_uv_certified` for faces with section edges (today a typed
   refusal) and the A3c chart API; then M1 (ANYmesh) and F1 (ANYfem bow example).

Later: A5 NURBS x NURBS, quadric/extrusion x NURBS, A6 joint-edge remap, A7 STEP handoff,
revolved surfaces, P1 coverage gaps (closed sections, singular boxes, boundary in plane),
A2b NURBS edge predicates, R4-R6, native N1b-N4 (after the bow probe measures exact-predicate
cost at barge size). E5 (exact plane x plane trace) waits for an exact apply-side vertex
mapping; its work is on local branch `claude/geom-tb-e5` (restored 2026-10-19 at 29bf4c9).

Known defects and limits: a rebased Cylinder/Cone child with a sub-tolerance angular bound
offset fails "loop self-intersects" (fail-closed; intersection code); joint stretches near
a shared plane x NURBS corner reach JOINT_BALL (JOINT_BALL + 1) tolerances; near-tangent
sections and descents leaving the chart refuse; `iso_curve` refuses coordinates below
2**-500; projection on general pcurve edges is slow; 558 of 966 decision sites are
unassigned in the R0 audit (471 in `intersections`/`curved_intersections`); 14
perturbation flips remain (cylinder/cone, R4).

Superseded statements below: the "retry limit reached" and "local, unpushed" headers;
the slice table's "planned" status of G0/ARCH/S1/S2/N0; Q1 (closed 2026-10-13: hydrostatic
pressure, clamp at X = 0) and Q3 (closed: test CI runs on push and weekly; the wheel
workflow still needs the owner) listed as open; "volume with an error bound" in P5 (an
estimate with a refusal threshold was delivered). Order: B2 (A4e) comes before M1/F1, as
in the architecture; the slice table's "M1 depends on A3/A4" is incomplete. Bow form is
undecided (plan: one smooth shell; architecture section 6: two patches sharing the stem;
Q2: one patch per side).

## Owner decisions (2026-10-13)

| Id | Decision |
|---|---|
| D1 | Speed-ups may change coordinates within a stated error bound if topology, classification and certificates are identical and geometry quality is maintained. Every equivalence claim states the bound and a quality check. |
| D2 | Native core in C++ (C++17, CPython C API, stable ABI). |
| D3 | Native core optional; Claude's call. Chosen: optional with an explicit, reported pure-Python fallback while the reference implementation exists; revisit at the first release that ships it. |
| D4 | Target Windows, Linux and macOS; use the runners/toolchains available now. |
| D5 | Driving case: the ANYfem large barge with a curved bow. NURBS first, extend from there. |
| D6 | perf-hunt branch dropped. Its leads may be re-derived in S1 on current main. |
| D7 | Independent review by Codex (GPT 6.1 Sol high; GPT 6 Astra high/max for very complex reviews). Claude's own work: Opus 5.5 medium default, Fable for architecture/very complex work. |

## Driving case: curved-bow barge (proposal, details to be confirmed)

Base: ANYfem `floating_barge_large()` (15 m x 11 m x 2.5 m, bulkheads X = 5, 10,
girders, 700 mm longitudinals, bulkhead openings with R250 corners, flat bars).
Proposed variant `floating_barge_bow()`:

- the midship part 0 .. 15 m unchanged; a collision bulkhead at X = 15 m;
- a bow 15 .. 20 m: one smooth hull shell (port + starboard, joined at a rounded
  stem) as a NURBS surface: bottom rising in a curved rake to the deck, sides
  tapering in plan;
- the deck at 2.5 m continues forward as a planar face trimmed by the bow's deck-edge
  NURBS curve (a plane x NURBS intersection);
- longitudinals continue onto the bow shell as members on a curved face (curves on
  the surface, i.e. plane x NURBS intersections of the stiffener planes); one or two
  transverse web frames in the bow (plane x NURBS);
- loading: the end moment at X = 15 no longer fits; proposed hydrostatic pressure on
  the wet surface (ANYfem wet-surface load) plus the existing clamp at X = 0.
  **Open question Q1 for the owner.**

Milestones on this case:

- B1 geometry builds and passes `strict_audit` (ANYgeometry only);
- B2 `plan/apply_intersections` connect deck, bulkheads, frames and stiffeners to the
  bow shell (ANYgeometry);
- B3 meshes through ANYmesher with a quality report (consumer slice in ANYmesh);
- B4 solves in ANYfem with an independent check (e.g. pressure resultant equilibrium,
  convergence on refinement).

## Slices

Status key: planned / in progress / done (SHA) / reviewed.

### Foundation

| Id | Slice | Acceptance | Deps | Status |
|---|---|---|---|---|
| G0 | Benchmark corpus + harness in `benchmarks/generality/` (pipes6, rings8, grid16/32, tilted8, doc60, cone/Bezier wall, barge preparation through ANYmesher when available) with topology digests, coordinate-deviation report, work counters | Repeatable digests on two runs; JSON results | – | planned |
| ARCH | Architecture design (Fable): surface capability interface, general intersection-curve contract, NURBS data model + schema record, native-core layout and Python/native boundary, how ANYmesh consumes generic charts | Design doc reviewed by Codex (Astra) and accepted by owner | study | planned |

### Speed (pure Python)

| Id | Slice | Acceptance | Deps |
|---|---|---|---|
| S1 | Scalar 3-vectors in hot paths, validation at API boundaries | Topology digests identical on G0; coordinate deviation within D1 bound; >= 1.5x pipes6/rings8; full suite: same failures as main | G0 |
| S2 | Cached digests on immutable definitions (same checksum format), fewer whole-model passes, cheaper identity checks | Identical documents/checksums; doc60 >= 2x | G0 |

### Native core

| Id | Slice | Acceptance | Deps |
|---|---|---|---|
| N0 | `anygeometry._core` scaffold: abi3 C++17 extension, `ANYGEOMETRY_DISABLE_NATIVE` / `ANYGEOMETRY_REQUIRE_NATIVE`, backend in runtime diagnostics, CI wheel matrix (cibuildwheel) for Windows/Linux/macOS, installed-wheel smoke test, sdist w/o compiler -> reported Python fallback | Local build + import on Windows; abi3 wheel tag; tests pass with native on and off; CI config ready (running CI = owner) | ARCH (layout) |
| N1 | Batch evaluate / invert / closest point for current curves and surfaces | Differential tests vs Python reference within D1 bound; >= 20x on batches | N0 |
| N2 | Robust predicates: adaptive expansion arithmetic + exact fallback replacing `Fraction` hot paths | Exact agreement with Fraction oracle on degenerate corpus | N0 |
| N3 | Native UV arrangement | Identical topology/certificates on G0; >= 5x on arrangement share | N1, N2 |
| N4 | Root isolation + interval enclosures | Same root counts as exact oracle | N2 |

### Generality (NURBS first)

| Id | Slice | Acceptance | Deps |
|---|---|---|---|
| A1 | Capability interface + general intersection-curve contract; existing families mapped without behaviour change | No output change on G0 | ARCH |
| A2 | NURBS curves (multi-span, rational) as edge geometry: native eval/derivatives, knot insertion, Bezier extraction, transforms, schema record, Python reference | Independent references (textbook examples, de Boor vs Bezier extraction, exact circle as rational NURBS), round trip, transforms | A1, N0 |
| A3 | NURBS surfaces (+ trimmed faces with NURBS trim curves in UV); sphere/torus/revolved later | Same kind of references; analytic area/normals for exact rational cases | A2 |
| A4 | Plane/quadric x NURBS certified intersection: query, then plan/apply | Independent oracles (grid minima + Newton, role swap, area conservation); typed refusals | A3, N4 |
| A5 | NURBS x NURBS certified intersection | Same; tangential/overlap fail closed | A4 |
| A6 | Engine-wide joint-edge parameter remap | Removes the refusal for all curve families | A1 |
| A7 | STEP B-spline mapping handoff (ANYfileIO) | Contract agreed | A3 |
| M1 | ANYmesher consumer: mesh trimmed NURBS faces through the generic chart | B3 | A3, A4 |
| F1 | ANYfem `floating_barge_bow` example + test | B4 | M1 |

Order: G0 + ARCH now; then S1/S2, N0 and A1 in parallel; A2 -> A3 -> A4 -> M1 -> F1
is the critical path to the curved-bow barge.

## Status (2026-10-13, end of day 1) — retry limit reached, owner decision needed

All branches are local, unpushed and unmerged (worktrees under
`C:/Github/ANYgeometry/.worktrees/`). Base: main 0a3a602. Reviews: Codex
(GPT 6 Astra high for the architecture, GPT 6.1 Sol high for code); results in
the session scratchpad (`review_*_result.md`).

| Slice | Branch @ commit | State |
|---|---|---|
| G0 corpus | claude/geom-g0-corpus @ 5a7d485 | done (not reviewed) |
| ARCH | claude/generality-programme @ 685af6e | rev 3 after 3 Astra reviews: 13 -> 5 -> 5 findings, each narrower; rev 3 still has 3 blocking + 2 major in the certified-intersection algorithms (parameters_of existence, start matching across systems, norm conversion, preimage isolation, oracle keys). Retry limit reached. |
| S1 scalar hot paths | claude/geom-s1-scalar @ 3c4e1ca | 3 Sol rounds; round 3 still finds D1 cases: scalar Cylinder.local_uv changes a seam snap and thus material membership; the area shortcut returns NaN where main refuses; float32 input to the vertex grid; batch skip overflow with reversed order. Full suite: no new failures (22, subset of main's 30). Topology identical on G0; pipes6 ~1.5x, rings8 ~1.5x. |
| S2 bookkeeping | claude/geom-s2-bookkeeping @ 50a00f2 | identical output, ~1.07x on grid32/tilted8; acceptance (doc60 >= 2x) not met; not reviewed |
| N0 + N1a native | claude/geom-n1a-bspline @ afa7179 | 3 Sol rounds; all memory-safety, packaging and workflow findings resolved; round 3: two major numerical cases at extreme ranges (rational derivative cancellation at weight ratio ~2**200; basis underflow with knot spans ~2**1000). 48 tests pass from source and from an installed abi3 wheel. |

Key finding (F1, confirmed three times): the current engine's classification at
tangencies and seams depends on the last bits of chart inversion and evaluation.
Any change of rounding (pure Python or native) can flip such a decision. Under D1
as worded ("classification identical") even small speed-ups need bit-identical
arithmetic on those paths, and a native core is not feasible there until the
predicates are certified (robust to rounding).

Other findings: F2 tilted8 chart failures on main (3 of 914 faces, thin planar
faces); F3 S2 gain small (whole-model passes dominate).

## Second round (owner, 2026-10-13: "proceed as you recommend")

Decisions: robustness first (certify tangency/seam/membership/vertex decisions before
rounding-changing speed-ups; meanwhile only bit-identical speed-ups merge); design plane x
NURBS rigorously first; weight ratio <= 1e6; bow loading = hydrostatic pressure on the bow
with the clamp at X = 0; NURBS records in schema 7; wheel CI not run without the owner.

- MERGED to local main (not pushed): claude/geom-s1-identical, main 0a3a602 -> 8910cd3.
  Bit-identical speed-ups (vertex grid lookup, guarded batch canonical skip, Gauss-Legendre
  cache, flat-primitive content binding, exact float ratios). Evidence: G0 documents
  identical at bound 0 on all 8 workloads; full suite 22 failures, all in main's set; four
  Sol review rounds; the last remaining remark (a Python profiler hook can observe skipped
  recursion) is outside the equivalence contract. Gains: tilted8 ~1.16x, grid32 ~1.05x.
- Native (claude/geom-n1a-bspline @ 0bb41c4): weight ratio 1e6 and an accuracy contract
  tested against an exact rational oracle; not merged (opt-in scaffold, awaiting more kernels).
- Perturbation probe (G0 c341b56): 34/35 workload-site pairs robust to 1-ulp perturbation;
  tilted8 changes topology under line-evaluation perturbation (near-degenerate).
- Architecture on the narrowed scope, three attempts (retry limit reached): rev 4 69a30d2
  (2 blocking + 6 major) -> rev 5 693cf1a (2 blocking + 4 major) -> rev 6 7136104 (2 blocking +
  1 major). Sound per review: T3 canonical refusal policy, parameter-wide contraction, joins,
  in-box pairing (clipping contract), root-defined endpoints, parameters_of, persistence
  equivalence, reverse dispatch, authored-chart attachments, junction pair table. Open: (1)
  positional enclosure must include the independent-coordinate error (simple fix); (2) D5 cell
  membership for cells with several boundary cycles (needs a certified boundary-free
  connection from the oriented edge to the probe); (3) inventory sites (three-point arc
  frames, broad-phase pruning, derived tolerance extents, the overlap probe at
  batch_intersections.py:276). Owner decision requested on how to proceed.
- A0 NURBS definitions (claude/geom-a0-nurbs @ 2a40923, on the native branch + main): schema-7
  records, validation, evaluation; full suite no new failures; Sol rounds 1-2 fixed, round 3
  running.

## Third round (owner, 2026-10-13)

- Architecture rev 6 (7136104) ACCEPTED as is; its open items become R-slice tasks:
  (T-a) positional enclosure includes the independent-coordinate error (A1b),
  (T-b) certified cell membership for cells with several boundary cycles (R1, D5),
  (T-c) inventory sites: three-point arc frames, broad-phase pruning, derived tolerance
  extents, overlap probe at batch_intersections.py:276 (R0 audit, R1).
- NURBS defaults accepted: max degree 7, clamped knots, numeric ranges.
- MERGED to local main (not pushed): claude/geom-a0-nurbs, 8910cd3 -> 99a4712 = optional
  native core (opt-in build) + B-spline kernel + NURBS definitions/schema-7 records. Gates:
  full suite no new failures; merge confirmation review found one blocking issue (rational
  derivative cancellation), fixed by local-frame evaluation; accuracy contract corrected to
  R**(k+1) after a 600-curve exact-oracle sweep (worst 2.2 eps-units).
- Next: R0 (exact_predicates toolkit, decision trace, structural audit, perturbation harness,
  adversarial corpus). The independent non-rational oracle uses Python's decimal module
  (mpmath is not installed; installing it needs owner approval).

## R0 done (2026-10-13)

- MERGED to local main (not pushed): claude/geom-r0, 99a4712 -> 0a80f31. No engine route
  changed. Contents: exact_predicates toolkit (T0-T4, T7, T8), evidence tool, structural
  decision-site audit (reporting mode), perturbation harness + adversarial corpus + baseline
  report, G0 corpus.
- Evidence: 5e5 canonical enclosures contain an independent decimal oracle (0 failures);
  1e6 filter decisions vs canonical: 614,200 decided, 0 disagreements, 0 refusals.
- Baselines on main: audit 966 decision sites (408 mapped to the inventory, 558 unassigned;
  471 of those in intersections/curved_intersections); perturbation 75 flips in 19 of 35
  cases, none under operand reversal; Plane.local_uv involved in 53.
- Reviews: Sol three rounds on the combined branch; all findings fixed (atan2 branch cut,
  filter scope to registered kinds with required magnitude, refusal evidence, typed
  errors, exact norms, harness digests). Full suite: no new failures.
- Next: R1 (planar chart and membership, station model in arrange_material, generated
  curves, D5 multi-cycle cells (T-b)); first target: drive the Plane.local_uv flips to zero.

## R1a + R1b step 1 merged (2026-10-13)

- Local main 0a80f31 -> 7760c99 (R1a: exact planar clip_line_to_face and planar
  face-face query; decision vs report planes for inferred faces; 7 documented decision
  changes) -> 276b2bc (R1b step 1: exact planar material containment for straight
  loops; the exact route only for exact query points on all-straight Plane domains,
  float query points keep main's route; 2 documented reference changes: points at
  prev(tol)/tol from the boundary are boundary). Not pushed.
- Evidence: harness 75 -> 49 (R1a) -> 21 (R1b) flips; G0 identical at bound 0; planar
  workloads 3-6% slower (interleaved). Sol reviews: R1a 3 rounds, R1b 2 rounds, clean.
  Full suite: no new failures (main 15 env/release-authority failures).
- Remaining plane-only flips: `scaled_crossing/1e-100` and `arc_opening/flat`
  plan_apply, both decided at `LinePath.subcurve` (rounded sub-curve ends, B11);
  `arc_opening/tilted` query and plan_apply at `MaterialDomain.uv` on arcs (E4 arcs).
  Cylinder/cone cases belong to R4.

## R1b step 2 plan: station model for straight paths on planes

Question: the arrangement build stores every split piece as a `LinePath` between
rounded evaluated points; later junctions, vertex identity and duplicate tests decide
on those roundings. Change: on a Plane support, every straight piece is a station pair
on its source line `(source LinePath, exact a, exact b)`; positions are exact
rationals; decisions on straight pieces use them; the float `LinePath` of a piece is
emitted with correctly rounded endpoints (coordinates may move within rounding, D1).
Arcs and other curves keep the float route (reported as uncertified until E4/B2).

| Slice | Content | Acceptance | Depends |
|---|---|---|---|
| 2a | exact line-line junctions and point parameters (tolerance rules through squares), exact clip intervals (exact midpoint stations into R1b containment), exact `_parameters` merge, exact vertex identity (lowest index within tol) and duplicate edges, for straight pieces on Plane supports | `scaled_crossing/1e-100` plan_apply stable; generic float-route agreement tests (same topology); G0 topology identical, coordinate deviation reported within bound; suite no new failures; every changed unperturbed decision with its exact argument | R1b step 1 |
| 2b | seams D6/D7 through exact stations (planar) | interior-ended trace cases stable under harness | 2a |
| 2c | half-edge ordering D2 by exact chart directions (ties by exact curvature, else refuse as now) | harness flips on ordering sites zero for straight cases | 2a |
| 2d | exact cell-area threshold and conservation (D3, rational planar area), certified probe-in-cycle and hole ownership (D5, T-b) | probe/area sites stable; multi-cycle cell tests | 2a-2c |

### R1b step 2a merged (2026-10-13)

- Local main 276b2bc -> 9fc114b (17e6578 + review fixes f29620b, 7eb31f0, audit anchor
  9fc114b). Straight pieces on planes keep exact stations; clip/splits/junctions/merge
  and straight-straight vertex identity are exact replicas of the float rules; curved
  pieces keep main's float rules (incl. vertex identity and grid eligibility).
- Evidence: harness 21 -> 18 flips (scaled_crossing/1e-100 plan_apply stable;
  unperturbed answers unchanged); G0 topology identical, coordinates <= 4.8e-16
  relative; interleaved timing tilted8 0.95x, grid16 0.84x. Sol: 3 rounds, clean.
  Full suite: one new failure (decision-audit D8 anchor), fixed in 9fc114b; otherwise
  main's failures.
- Remaining plane-only flips: arc_opening/flat plan_apply (random pattern) decided at the
  interior-ended trace endpoint test (D6, step 2b) and curved subcurve ends;
  arc_opening/tilted at MaterialDomain.uv on arcs (E4). Found: scaled_crossing/1e-100
  plan_apply refuses ("unresolved bridge") on main and branch -- a scale bug to study.
- Open: speed of exact junctions (grid16 -16%): dyadic-integer kernel candidate.

### R1b step 2b merged (2026-10-13)

- Local main 9fc114b -> 05e5011 (be05ff2 + review fix 05e5011). Decomposition seams on
  planes go through exact stations (D6/D7) with main's seam parameterization (float
  extent * unit direction around the exact station; a longer exact seam changed the
  32-eps clip merge, Sol blocking finding) and main's stable order.
- Evidence: unperturbed outcomes of all corpus cases identical to main; harness 18 -> 17
  (arc_opening/flat stable); the only plane-only flips left are arc_opening/tilted
  (E4 arcs). G0 topology identical. Sol: 2 rounds, clean. Full suite (be05ff2): main's
  failures plus one release-authority flake (passes in isolation on both).
- Next: 2c ordering (D2), 2d areas/probes (D3/D5), then E4 arcs on planes.

### R1b step 2c merged (2026-10-13)

- Local main 05e5011 -> 610dd92 (f6330e5 + review round 1). Straight-only vertices on
  planes are ordered by exact chart directions (D2); exactly equal directions keep main's
  tie key; vertices with a curved edge keep main's atan2 order. The exact order can
  differ from main's wherever main's least-squares chart error exceeds the angular gaps
  (inside the 64-eps window and, on ill-conditioned charts, beyond it); it is the exact
  answer for the stored data.
- Evidence: unperturbed outcomes of all corpus cases identical to main; harness 17
  flips (unchanged; plane-only: arc_opening/tilted); G0 topology identical. Sol: 2
  rounds, clean. Full suite: main's failures plus the release-authority flake.
- Found: nearly coincident straight traces (20-1000 eps apart) fail on main and branch
  alike ("unresolved bridge" / "failed material conservation") -- step 2d (D3/D5).

### R1b step 2d merged (2026-10-13)

- Local main 610dd92 -> 9d1244c (08bb03d + review round 1). For all-straight planar
  arrangements, main's cell phase runs on exact stations: main's planar_loop_area
  formula on each piece's own stations, exact |area| <= tol^2 drop and conservation,
  main's probe (exact position, correctly rounded physical length), main's tolerance-based
  material test and hole ownership with exact containment.
- A face certificate (probe segment meets no other edge, zero-tolerance material) was
  tried and withdrawn after Sol review: tolerance-merged vertices keep their own stations,
  so the edge network is not an exactly closed planar embedding; the certificate accepted
  a probe outside its cell and the polygon areas changed main's answers. It needs T-b.
- Evidence: unperturbed outcomes of all corpus cases identical to main; harness 17 flips
  (plane-only: arc_opening/tilted); G0 topology identical; interleaved timing vs main
  610dd92: grid16 1.01x, tilted8 0.95-0.97x, rings8 1.0x. Sol: 2 rounds, clean. Suite:
  main's failures plus the release-authority flake.

### Findings for T-b / D8 (from steps 2c-2d)

- T-b needs an exactly closed planar embedding: canonical vertex positions (one station
  per vertex) with edges re-anchored to them and a planarity guarantee, before probes can
  be certified. With it, two known main defects become fixable: thin cells next to the
  boundary (a trace 1.2 tol above an edge: 2 cells or "failed material conservation"
  depending on the outer loop's start vertex) and nested trace outlines 2 tol apart
  (ownership [0,0,2] where the inner outline belongs to the ring).
- D8: _parameters merges split parameters closer than a fixed 32 eps regardless of the
  tolerance; nearly coincident straight traces (20-1000 eps apart) then lose a crossing
  and the arrangement fails (bridge / conservation) on main and branch alike.

### R1c E4a merged (2026-10-14)

- Local main 9d1244c -> 0c4ab4a (02cd235 + review rounds fedb651, 0c4ab4a).
  planar_arcs.py: containment for loops of lines and elliptic arcs on planes with an
  exact query station is an exact replica of main's rules with canonical enclosures
  for every transcendental quantity and certified decisions (refusal on undecidable
  ties): first-harmonic roots via x = tan(t/2), certified turn range from a rational pi
  enclosure, clamps keeping parameter and angle consistent, main's endpoint snapping
  and coincident-arc point rule (incl. endpoint acceptance). Per-engine memo (history-
  independent comparisons).
- Evidence: harness 17 -> 14 flips; no plane-only flips left (arc_opening/flat and
  /tilted stable; the tilted flip was a ray through an arc vertex). Unperturbed
  outcomes of all corpus cases identical to main; G0 identical; arc workloads
  1.00-1.04x main. Astra (high): 3 rounds, clean (turn completeness, endpoint
  acceptance, clamp consistency, cache/budget reproducibility fixed).
- Cost: an exact arc containment query is ~40 ms (float ~0.4 ms); rare in G0.
- Next (owner, 2026-10-13): T-b closed planar embedding + D8 parameter merge, design
  with Fable.

### T-b / D8 design: retry limit reached (2026-10-14), owner decision needed

- TB_D8_CLOSED_EMBEDDING_DESIGN.md rev 1-3 (Fable), Astra high reviews: rev 1 not
  acceptable (4 blocking, 5 major), rev 2 (1 blocking, 3 major), rev 3 (3 major, no
  blocking): S10 persisted-embedding certificate lacks component nesting (a
  disconnected cycle can move between faces with planarity, rotation system and
  area signs unchanged); S8 weakly-simple validator accepts an exterior "hole" whose
  vertices all lie on the outer boundary (needs open-edge and sector tests); S8's
  opposite-winding rule rejects main's accepted reversed-corner touching hole
  (normalize winding or approve the change). Astra: specification changes for the
  owner's decision, not another review round.
- Recommended direction unchanged: candidate C with conservative closure, persisted
  certification in apply, horizontal-ray ownership, multiplicity parity, D8 by
  tolerance-scaled rules after shared identity, E5 exact plane-plane line, E4 deferred.
  Owner questions Q-a..Q-i with recommendations in section 8.

### Owner decisions on T-b / D8 (2026-10-14: "Proceed as recommended")

- Implement E1 now (D8 tolerance-scaled merge after shared identity, 128-eps snap
  replacement, short-piece-safe canonical junction roles, deferred removal).
- Rev 4 of the design for the three round-3 gaps (component nesting in the persisted
  certificate, open-edge/sector validation of touching holes, winding normalisation);
  then E2, E2a, E3.
- Q-a stored-vertex representatives (2 tol); Q-b accept the <= 2 tol deformation
  contract under Q13; Q-c refuse coincident loop occurrences; Q-d keep the tol^2 sliver
  drop; Q-e refusal identities may change (listed); Q-f E5 in this programme; Q-g
  conservative closure first, repair opt-in; Q-h canonical data key for derived pieces
  (T7); Q-i atomic batch refusal when the persisted embedding is not preserved.

### T-b / D8 design accepted (2026-10-14)

- TB_D8_CLOSED_EMBEDDING_DESIGN.md rev 4 (Fable) + 4.1/4.2 (per-loop simplicity on
  stored stations and again after re-anchoring): Astra confirmation "(a) acceptable for
  implementation of E2/E2a/E3". Order: E1 (in review) -> E2 -> E2a + E3 (reviewed
  jointly; E2a before E2 merges) -> E2b opt-in, E5; E4 deferred.

### T-b/D8 E1 merged (2026-10-14, owner option a)

- Local main 0c4ab4a -> 17d3046 (477f818 + four review/scope rounds). All-straight planar
  arrangements: tolerance-scaled split rules applied once on shared vertex identity (S2/S4),
  length-rule endpoint snaps (S1a), canonical junction roles (S1b, T7), short-piece handling
  (S1c, incl. short line vs curve), exact directions for straight edges and seam rays, the
  derived-piece degeneracy rule, refusal of edges collapsing at stored coordinates.
  Arrangements with curved paths keep main's per-path merge until E4.
- Decision changes listed in the commits (Q13/Q-e): nearly coincident traces (4 cells, main
  refused), fans near a corner, corner cuts, short traces (point-like), subnormal slivers.
- Evidence: unperturbed corpus outcomes identical to main; harness 14 flips; G0 identical;
  full suite no new failures; Sol: 3 rounds + owner decision (option a) after the retry limit;
  timing vs main ~0.94-1.00x.
- E2 + E3 + E2a implemented on claude/geom-tb-e2 (b9e3eef), joint review pending.

### T-b/D8 E2 + E2a + E3 merged (2026-10-14, owner option a after the retry limit)

- Local main 17d3046 -> e2b8674. All-straight planar arrangements: one exact position per
  vertex (stored loop corner, else first station; chart-coincident stations one vertex),
  re-anchored edges, conservative closure (refuse crossings/T-junctions/overlaps), support
  precondition on inputs as given, deformation bound (world units); certified cells (validity,
  horizontal-ray ownership with vertex events, parity material, exact areas); apply re-certifies
  every certified face once after the whole batch's vertex mapping (planarity, oriented edge
  use, slits, winding and lobes vs the certified classification, hole regions never nesting,
  cells not overlapping) and refuses the detached candidate otherwise.
- Fixed main defects: thin cells by the boundary (2 cells for every start vertex), nested
  outlines ([0,1,1]), the merged-start case (2 cells, area 16), three tilted8 faces main cannot
  chart (areas match the exact stored polygons), the 1e-180 seam sliver.
- Listed changes: near-corner fan refuses (conservative closure); invalid domains refuse;
  coincident loop occurrences refuse (Q-c); scaled_crossing/1e-100 refuses as input off support.
- Evidence: unperturbed corpus outcomes = main except that refusal; harness 14 flips; G0 =
  main except tilted8's recovered faces (and a signed-zero coordinate); full suite no new
  failures; timing ~0.9-1.05x. Astra: 3 joint rounds (all findings fixed; the last one fixed
  without a fourth round, owner option a).
- E5 (exact plane x plane trace) parked on claude/geom-tb-e5: it connects plates at exactly tol
  and routes them into apply's float vertex mapping (rounding-dependent at tol). Owner: faces
  below ~1e-7 are not real-world; the 1e-100 stress test is not a requirement.

## NURBS track toward the curved-bow barge (owner, 2026-10-14: functionality before speed)

Owner chose the NURBS barge path next (speed work N1b/S1 re-admission deferred). Architecture
rev 6 was accepted as a whole (incl. section 4), so the "not yet" gate of section 8a on
A1b/A1c no longer applies. Order to B1-pre (a bow face built, arranged, audited) and B1 (deck
and bottom trimmed by plane x NURBS curves):

| Slice | Output | Acceptance | Depends |
|---|---|---|---|
| A1a-lite | `surface_capabilities.py`: registry + adapters for the six types; Plane complete (evaluate, derivatives, invert_local, exact_uv, enclosure, Jacobian bounds, implicit); others delegate evaluate/derivatives/invert/transform with enclosure/Jacobian `None` (rule 4: route refuses) until a route needs them | bit-identical delegation on 1e4 UVs per type; Plane enclosure contains samples and corners; exact_uv = ExactPlane; G0 byte-identical | A0 |
| A2a | NURBS edges in the engine (evaluation, add_curve, freeze_edge, bounds, transform, exact circular arcs) | transform commutes; arc radius/length vs analytic | A1a-lite, N1a |
| A3a | NurbsSurface capabilities (evaluate, derivatives, invert_local, enclosure, Jacobian bounds, iso_curve, cylinder_to_nurbs) | enclosures/Jacobian bounds contain 1e5 samples; bilinear patch exact | A2a |
| A1e | MaterialDomain over capabilities (NURBS chart route), area with bound | area of a rational cylinder patch vs r*sweep*h <= 1e-10 rel. | A3a |
| A3b | NURBS faces (trim rules, arrange_material on NURBS charts, strict_audit, transform): **B1-pre** | bow face audited; area vs Gauss-Legendre | A1e |
| A1b/A1c, A2b, A4a, A4b' | certified plane x NURBS curves, NURBS edge predicates, plane x NURBS route: **B1** | per architecture section 8 | A3b |

Risks: performance (Fractions in the hot arrangement loop; mitigate with a float
filter that only falls through to exact arithmetic near thresholds, measured on G0);
coordinate changes in G0 digests (expected within rounding, reported per workload);
replicating float tie rules exactly where they are not rational (64-eps angle grouping:
replaced by exact comparison, changes argued case by case).

### NURBS track status (2026-10-14)

A1a-lite (be69d9e), A2a (e77aefa) and A3a (bff3af7) merged to local main after Sol reviews
(owner option a on the last fix of each). Known limits: projection from a circle's centre
~38 s; NURBS curve x face queries are UNCLASSIFIED until A4; revolve of NURBS profiles
refused; `iso_curve` refuses coordinates below 2**-500.

### A1e-lite plan (what A3b needs; the rest of architecture A1e waits for A1d/R5)

`MaterialDomain` gains a NURBS chart branch; existing branches are untouched (byte
preservation for every built-in support). On a `NurbsSurface` support every boundary path is
a `CurveOnSurface` on that support; chart points, tangents and curvature come from the
pcurve (no world inversion, architecture 4.6); anything else is refused.

| Slice | Output | Acceptance | Depends |
|---|---|---|---|
| L1 | `CurveOnSurface.derivative`/`second_derivative` (chain rule through the surface), `chart`/`chart_derivative`; batch chart evaluation | derivatives vs finite differences; batch evaluation bit-identical to per-point | A3a |
| L2 | `MaterialDomain` NURBS branch: `uv`, `tangent`, `curvature`, `horizontal_roots` (Bernstein subdivision on the pcurve's rational numerator, float route), `contains_loop`, chart `area_loop` (Green on the pcurve, Gauss per span), `world_delta`/`left_probe`/`area_jacobian` from Jacobian bounds | chart areas of pcurve polygons/circles exact to 1e-13; containment of known points; all other domains unchanged (suite) | L1 |
| L3 | `world_area_loop` NURBS: `oint G dv`, `G(u, v) = int_0^u J(s, v) ds` per knot span (Gauss, doubling until settled), plus `world_area_estimate` returning `(value, estimate)` labelled *estimated* (rigorous bound = R5) | rational cylinder patch area vs `r |sweep| h` <= 1e-10 rel.; sub-rectangle vs `r |dtheta| dh`; parallelogram bilinear patch vs exact | L2 |

Deferred from architecture A1e: `BranchPath`, `_target_bounds` NURBS branch (A4), A1d
dispatch, rigorous area bound (R5). Risk: the float `horizontal_roots` route is not a
certified decision; NURBS-chart decisions get certified predicates in A3b/A4.

### A1e-lite merged (2026-10-14, owner option a after the retry limit)

Local main 216b917. L1-L3 delivered (cylinder area 4e-16 vs the 1e-10 target). Three float
attempts at NURBS-chart level roots failed review (tangencies, runs, close crossings); the
owner chose exact decisions: `exact_levels.py` (exact Bezier extraction, Sturm isolation of
the square-free numerator, exact gap signs, `event_side`, `has_root_in`) under
`nurbs_domain.contains_loop` (exact ray parity, exact tolerance band). Six Sol rounds; the
last fix (dyadic roots below the isolation width, exact band edges) merged without another
round. Full suite: no new failures. Limits: world area carries an estimate, not a bound (R5);
~0.1 s per degree-7 query with crossings. Next: A3b.

### A3b plan (owner option a, 2026-10-14): A3b-core, then a single-patch B1-pre

Finding: strict_audit classifies every broad-phase pair and blocks on unclassified ones,
so the architecture's B1-pre (several bow patches plus authored deck/bulkhead/frame planes)
needs plane x NURBS (A4a), an adjacent-patch NURBS x NURBS certificate (deferred 4L) and a
self-intersection proof. Owner chose: A3b-core, B1-pre = one curved bow face audited clean
in isolation; multi-patch bows and planes move to B1 (after A4a).

| Slice | Output | Acceptance | Depends |
|---|---|---|---|
| B-S1 | `CurveOnSurface` as a model edge: `add_curve` (exact end test via the exact surface point at the exact pcurve end), `bounds` (exact pcurve sub-hull -> surface enclosure -> transform), length/breakpoints, projection, split, transform (transforms the support; identity transform kept), schema-7 record `curve_on_surface`; extrude/revolve and curve x face queries refuse | edge API on an iso-curve of the rational cylinder (length vs r*dtheta, bounds contain samples, split keeps the support), round trip byte-identical, refusals | A1e-lite |
| B-S2 | NURBS faces: `add_face(..., surface=NurbsSurface)` rules (every loop/hole edge a `CurveOnSurface` on that support, pcurves chain exactly in the chart, closed, positive chart area), `MaterialDomain.from_model`, face area, face bounds, model transform, schema-7 faces | rational cylinder face area vs r*sweep*h <= 1e-10; a trimmed face vs Gauss-Legendre; refusals of 3.3/4.9 (world edge on a NURBS face, other support, open chart loop); transform commutes; round trip | B-S1 |
| B-S3 | `trim_loops_uv_certified(face, tol)`: chart polylines with a world chord bound (exact pcurve sub-hull distance to the chord x sqrt2 sigma_max) | the bound holds against 1e4 samples per edge; polyline counts shrink with tol | B-S2 |
| B-S4 | **B1-pre**: one curved bow face (rational NURBS, iso-curve boundary) audited clean: chart-decided edge-edge/vertex-edge pairs of its own boundary on an injectivity-certified patch (exact Bernstein sign of the projected Jacobian + simple projected boundary), `strict_audit` clean | `strict_audit` clean on the bow face; the injectivity certificate refuses a folded patch; area vs Gauss-Legendre | B-S3 |

Risks: projection on `CurveOnSurface` edges with box bounds only converges linearly (measure;
add a hull distance bound if needed); the injectivity certificate may need subdivision on
real bow shapes; audit pair kinds beyond the face's own boundary stay unclassified (by design).

### A3b progress (2026-10-14)

Branch claude/geom-a3b-s2 (worktree .worktrees/a3b2): B-S1 ac0e7ed + review-round-1 fixes
5439ca4, B-S2 575fca7, B-S3 4d020bd, B-S4 10be5d5. B1-pre reached: a rational half-bow face
bounded by its iso-curves audits clean (0.02 s), area vs Gauss-Legendre <= 1e-10. Change from
the plan: B-S4 certifies edge pairs in world space (exact iso-curve hulls; plane through the
shared vertex from the two tangents) instead of an injectivity certificate. Also found: the
strict audit had no bounds for any analytic curve family besides Straight/Arc/Spline (they
fail closed); A3b adds NURBS curves and curves on NURBS supports.

### A3b merged (2026-10-14, owner option a)

Local main d1a897c (B-S1..B-S4 + review rounds; last fix merged without another round).
Full suite: no new failures. B1-pre reached. Review-driven changes: exact NURBS trim
validation (certify_simple, contact-explicit piece certificates, exact hole containment,
relative chart area), no general-route fallback for curves on NURBS supports in the audit,
chord bounds for the surface image of chart chords incl. rounded ends. Limits: projection on
general (non-iso) pcurve edges is slow (box bounds); B1 needs A4a (plane x NURBS) and an
adjacent-patch certificate for multi-patch bows. Next: A1b/A1c, A2b, A4a toward B1.

### B1 plan: plane x NURBS (architecture 2.1, 4.1-4.5) - proposed 2026-10-14

Goal B1: a bow face (one NURBS patch) with a deck plane and a bottom plane trimmed by their
exact plane x NURBS section curves, sharing those edges with the bow face; frame planes cut
the shell; `strict_audit` clean. Exact rational arithmetic first (the oracle of 5.3); an
interval/native fast path only after it is correct (owner: functionality before speed).

| Slice | Output | Acceptance | Depends |
|---|---|---|---|
| P1 | `plane_nurbs.py` topology: exact substitution `G = n.S_h - d W` per Bezier sub-patch (exact extraction, A1e/A3b helpers); quad-tree boxes classified exactly (empty / regular in u or v / singular -> subdivide, refuse at depth); box-edge roots by exact univariate isolation (`exact_levels` Sturm machinery on edge restrictions, corner roots exact); clipping contract per gap; degree check, paths vs cycles, accounting; `CertifiedBranch` (roots, box chain) | rational cylinder cut by planes = the analytic ellipse/lines (counts, ends exact); wavy patches with constructed counts; refusals: closed section, boundary tangency, singular box, coincidence | A3b |
| P2 | `PlaneNurbsCurve` (pieces with parameter-wide interval Newton certificates in exact/Fraction intervals, `kappa`, `delta`), `evaluate` (uv, delta), `derivative`, `bounds` (enclosure over uv +- delta), `subcurve`/`transformed`; as a model edge (exact end test via the certified patch-boundary roots) | evaluation within `1e-12 * extent` of the analytic ellipse on the rational cylinder and `delta` bounds the deviation at 1e5 samples; edge API (length, split, projection) | P1 |
| P3 | schema-7 record `plane_nurbs_intersection` + validation on load (recompute boxes/roots; tampered records refused) | round trip byte-identical; tampered box/root refused with index; older reader refusal | P2 |
| P4 | faces sharing a section curve: plane faces trimmed by `PlaneNurbsCurve` + lines (two-support rule 3.3: the curve's `plane` and `surface` are the two faces' supports); NURBS face edges may be `PlaneNurbsCurve` on their support; audit: plane face x NURBS face classified by P1 (their full section equals the shared edges), shared-edge and line x section-curve pairs via `roots_on` (4.4) | deck/bottom faces built on the bow; audit pair classes all certified | P2 |
| P5 | **B1**: bow face + deck + bottom trimmed + frame planes (authored faces cut by the shell where they meet it) ; `strict_audit` clean; closed-shell certificate (every edge used by exactly two faces with opposite orientation, outward normals) and enclosed volume + centroid (divergence theorem over the faces, with an error bound) | B1 model audits clean; areas vs Gauss-Legendre; **closed shell certified and enclosed volume and centroid vs an independent Gauss-Legendre reference** (seed for H1, owner 2026-10-14); refusals of 4.9 reachable here exercised | P4 |

Progress 2026-10-14: P1 merged to local main 7152b01 (2 Astra rounds). P2 merged to local main
97f7dad (owner a): float evaluation failed 3 rounds -> owner chose exact evaluation by definition
(points = exact bisection midpoints refined into a 2**-50 plane slab; exact world points and
knot partials; knot_patch_hull enclosures; hull-and-slab distance bound). Exact design: Astra
rounds 1-2 blocking, round 3 no blocking, Sol confirmation no blocking/major; last minor fix
unreviewed. Suite 3526 passed, 13 known environment failures, none new.
P3 merged to local main 35f94a5 (owner a, 2026-10-15): record = plane, surface, path, domain
+ branch (path roots and arc boxes as canonical exact rational texts, algorithm
plane_nurbs_exact_v1); load recomputes the section and refuses differences with the index;
end vertices checked on load; extraction budgets before any work. 2 Sol rounds (round 2
no blocking/major; its minors fixed, unreviewed). Suite 3560 passed, none new. Next P4.

#### P4 sub-plan (2026-10-15)

Today a plane face with a `PlaneNurbsCurve` edge falls through to the generic *sampled*
trim check (`_validate_face_geometry`), and a NURBS face accepts only `CurveOnSurface`
edges with exact chart continuity; face pairs involving a NURBS face go through
`query_intersection`, which cannot classify plane x NURBS. P4 makes all three certified.

Joint rule (design decision for P4): a section curve's ends are algebraic roots, so
an adjacent edge can never end exactly there. Joints are topology vertices (as today);
geometrically, the adjacent edge's exact end must lie within the model's curve-fit
tolerance of every point of the root's world enclosure (`end_box`, the `add_curve` rule)
and, on a NURBS chart, its exact chart end within the chart tolerance of the root's
isolating rectangle. Exact equality stays required between two `CurveOnSurface` ends.

| Sub-slice | Output | Acceptance | Depends |
|---|---|---|---|
| P4a | `branch_predicates.py`: `roots_on(curve, plane2)` - roots of `h = n2 . S - d2` along a `PlaneNurbsCurve`, exact (substitution into the arcs' box polynomials: common zeros of `G1`, `G2` in each arc's knot rectangle, isolated by exact 1D elimination on the arc), with a typed refusal for tangency/undecided; certified "only shared ends" for a section curve and a straight line on its plane (line = plane through the line, normal to the face) | frame plane x deck-edge curve roots vs the analytic ellipse crossing (cylinder corpus); a line through the curve's end root certified to meet only there; a crossing line found; tangent line refused | P3 |
| P4b | plane faces whose loop mixes lines and `PlaneNurbsCurve` edges: two-support rule (curve's plane is the face's plane, exact), certified trim validation (pairwise: line x line exact as today, line x section curve by P4a, section x section on different surfaces refused unless their bounds are disjoint), joint rule, loop area with a bound (Green line integral along the curve) and its sign | a half-cylinder end-cap face (deck-like: one section curve + one line) validates; its area vs the analytic half ellipse; crossing/overlapping loops refused; a curve on another plane refused | P4a |
| P4c | NURBS faces whose loop mixes `CurveOnSurface` and `PlaneNurbsCurve` edges on the face's support: chart continuity by the joint rule, trim validation pairs (section x iso-curve by exact univariate isolation on the iso line; section x general pcurve refused; section x section of two planes by certified disjoint arc-box hulls or refused), chart area | the B1 bow face trimmed by a deck plane (deck-edge curve + three iso-curve pieces) validates; area vs Gauss-Legendre over the trimmed chart; refusals | P4a |
| P4d | `strict_audit`: plane face x NURBS face classified by P1 - their complete section (all paths) must equal their shared section-curve edges (branch identity: same plane, surface, path; domain covering), else reported; edge pairs (section x line, section x iso-curve, section x section) via P4a/P4c certificates; no fallback | the bow face + deck face sharing the deck-edge curve audit clean; a deck face that pierces the bow elsewhere is reported; all pair classes certified | P4b, P4c |

P4a merged to local main d1a35e1 (owner A, 2026-10-15) after 4 Astra rounds (round 4: no
blocking, its 2 majors fixed unreviewed): branch_predicates.py section_roots and
certify_line_section on the mathematical section curve C*; P2 points and joints certified within
delta = 2**-50 of the surface magnitude of C* (world), exact parameter map; records
plane_nurbs_exact_v2. The line certificate decides intersections/contacts within tolerance, not
proximity. Suite 3579 passed, none new. Next P4b.

P4b merged to local main 195a491 (owner a, 2026-10-15) after 3 Sol rounds (round 3: no
blocking, its 2 majors fixed unreviewed): section_faces.py validate_plane_section_face (exact
two-support rule, line x section by P4a, world simplicity of section edges by monotone-projection
joint certificates, Green area on certified monotone pieces with a rigorous bound, holes by ray
parity); P4a gained projection_monotone; P2 bounds resolution follows the parameter range. Suite
3597 passed, none new. Limits: refusals 12-90 s; only straight + section edges; no public
straight-hole API. Next P4c.

P4c merged to local main 89fd8ed (owner a, 2026-10-15) after 2 Sol rounds (no blocking; last
two fixes unreviewed): nurbs_section_faces.py (chart-level validation of NURBS faces with section
edges; straight chart segments only next to sections; P4a engine with chart-line patches);
_engine accepts any per-box polynomial; trim_loops_uv_certified refuses section faces (typed);
face_trim_loops_uv fixed for all NURBS faces (chart samples). Suite 3610 passed, none new.
Next P4d (audit).

P4d merged to local main f86e6f0 (owner a, 2026-10-15; suite 3625 passed, none new): owner chose (a) a strict
uniform tolerance contract for the audit (contacts within tolerance only in certified joint
departure stretches) and, after 3 Sol rounds, (b) for plane x NURBS faces sharing section edges:
their exact intersection must be the shared edges; the tolerance band along the shared edge is
NOT certified yet. Follow-up task P4e (open): a world-space face-band certificate (a lower bound
of the surface-plane angle from exact derivative bounds divided by sigma_max, at least the
angular tolerance, so the band lies within the departure radius).

Open risks: the joint rule's tolerance is the only non-exact decision (reviewers should
attack it); P4c's chart area over a trimmed NURBS region needs a certified bound
(A1e area is an estimate); roots_on cost in exact Python.

Open risks: (1) exact quad-tree cost in Python (degree (2,1)-(3,3) patches; measure on the
bow; an interval filter falling back to exact near zero if needed); (2) the clipping contract
and joins are the most error-prone parts (reviews will concentrate there); (3) frame planes
need plane faces whose boundary includes a section curve AND lines: P4 covers it, but a
frame crossing the deck edge needs `roots_on` junctions (4.4/4.5); (4) A2b (NURBS edge
predicates) is only needed where B1 arrangements need it - deferred unless P4 hits it.

### H1 track: hydrostatics for ANYstability (owner 2026-10-14, after B1)

Purpose: the owner plans a future ANYstability (ship stability) application that needs
volume calculations from ANYgeometry. The waterline is a plane x hull section, so B1
(P1-P2 `PlaneNurbsCurve`) is its foundation; the plane is already arbitrary (heel/trim).

Quantities: submerged volume and centre of buoyancy at given draft/heel/trim; waterplane
area, centroid and second moments (BM = I / V); later sweeps (hydrostatic tables, GZ curves).

Design notes (to be planned in slices when B1 is done):
- Face integrals as one moments API (orders 0-2 over a face, with an error bound),
  generalising the A1e area integral, instead of area-only.
- Submerged volume by the divergence theorem with `F = (0, 0, z - z_wl)` in the waterplane
  frame: `F` vanishes on the waterplane, so no waterplane face is built; only the hull faces
  below the waterline (clipped by the section curve) contribute.
- Waterplane area and moments as Green line integrals along the waterline `PlaneNurbsCurve`.
- Speed matters here: a stability sweep evaluates thousands of waterplanes, and exact Python
  evaluation costs about 0.5 ms per point; this is where the optional native/interval fast
  path (decision D3) becomes necessary.
- Cross-check reference: panel-mesh hydrostatics in ANYhydro/ANYmarine.

#### H1 sub-plan (2026-10-15; owner: H1 before P4e)

Method. Work in the waterplane frame (origin on the plane, unit normal n up out of the water,
axes e1 (longitudinal) and e2 (transverse) in the plane; frame coordinates xi, eta, zeta). Every
quantity is a hull-surface integral over the part below the plane, int_{zeta<0} phi (n . N) dA with
the outward area normal N, because the fields vanish on the waterplane or are divergence-free:

- volume: F = zeta n (div 1); first moments: xi zeta n, eta zeta n, zeta^2/2 n;
- waterplane area: F = n (div 0, F . n = 1 on the waterplane): A = -int_{hull, zeta<0} n . N dA;
  its first and second moments with xi, eta, xi^2, xi eta, eta^2 times n likewise.

So no waterline curve and no clipped faces are built. Each face is a chart (NURBS: its own chart;
plane: the plane's own (u, v) frame, any orientation) and contributes by Green's theorem in the
chart, oint G dv with G(u, v) = int_{u0}^{u} g(s, v) H(-zeta) ds: the inner Gauss rule is split at
knots and at the iso-line's waterline roots (plane: exact linear root; NURBS: roots of the iso-curve's
rational zeta per Bezier span), the boundary rule at knots and at the edges' waterline crossings.
Results are quadrature values with an order-doubling estimate and a refusal threshold (like P5c),
not rigorous bounds.

| Slice | Output | Acceptance | Depends |
| --- | --- | --- | --- |
| H1a | `face_moments`: the moment engine above (any plane orientation, NURBS faces; optional clipping plane); `enclosed_volume` rebuilt on it | P5 tests unchanged; B1 volume equal to P5c within 1e-13 relative; a rotated (tilted-face) box and a cube at 1e6 exact within 1e-12 relative; tilted faces no longer refused | P5 |
| H1b | `hydrostatics(model, faces, waterplane)`: displaced volume, centre of buoyancy, waterplane area, its centroid, second moments about the centroid (I_T, I_L), BM_T, BM_L, estimate | a heeled and trimmed box against closed forms within 1e-10 relative; B1 upright at drafts 1.0/1.75/2.5 against the independent reference (bottom to draft) and against `enclosed_volume` of `build_b1_half_hull(deck=T)` within 1e-9; B1 trimmed (z = T + a x) against an extended independent reference within 1e-8; dV/dT = A_wp (Richardson) within 1e-7 relative; waterplane fully above/below the hull handled (0 / full volume) | H1a |
| H1c | sweeps for ANYstability: hydrostatic table over drafts; draft for a displacement (Newton with A_wp); GZ at heel angles with free sinkage (and optional free trim) for a given displacement and KG | box GZ against the wall-sided formula within 1e-8; the B1 table monotone and consistent with H1b point values; time per waterplane measured and reported (decides the D3 native fast path) | H1b |

Open risks: (1) the waterline turning in v on a NURBS face (a square-root kink in G) slows Gauss
convergence: measured by the estimate, refused above the threshold; (2) the B1 test model is a half
hull with a centreline face: upright and trimmed waterplanes are meaningful, heeled ones need a full
hull (a mirrored B1 or a closed test hull; H1b tests heel on boxes); (3) a waterplane through a vertex
or along an edge exactly (root splitting at boundary points); (4) speed: exact geometry is not on the
float quadrature path, so H1 should be fast; measured in H1c.

H1 outcome (2026-10-15): H1a-c plus five Sol rounds (r1 e994c9f, r2 59140a5, r3 86a1573, r4
06db0f9, r5 fixes 21d0637 unreviewed, owner option b). MERGED to local main 21d0637 (suite 3682
passed, none new; not pushed). Final design: no sampled searches on the clipping path. Boundary
events are exact or certified per edge kind (straight edges linear; straight chart segments and
iso-lines by exact Bernstein restrictions of the P1 sub-patches with Sturm isolation and exact
bisection; section edges by section_roots); waterline turning levels by exact Bernstein subdivision
of the P1 boxes (tau^2 substitution at them); the plane offset exact from the caller's unnormalised
normal and shifts. Refusals: a closed waterline loop inside a NURBS face's chart box (not yet
supported), a NURBS face in the waterplane, unresolved tangencies, far-shifted knot domains, a
longitudinal direction within 1e-6 rad of the normal, unsettled quadrature. B1 waterplane ~0.8 s
(exact predicates dominate; the D3 native path is the next speed step). Follow-ups: closed waterline
loops (P1 cycles), free trim in gz_curve, a full (mirrored) B1 hull for heeled tests.

#### H1 follow-ups (owner 2026-10-16: "Finish the H1 follow-ups")

| Slice | Output | Acceptance | Depends |
| --- | --- | --- | --- |
| H1-F1 | closed waterline loops inside a NURBS face: the turning levels from the exact Bernstein sub-patches of the face's chart box directly (no P1 path topology, which refuses cycles); a surface lying in the waterplane still refuses | the basin wet disk (r = 0.01): V = pi r^4 / 2 and A = pi r^2 within 1e-8 relative (was refused); the existing H1 tests unchanged; B1 waterplane time not worse than 1.2x | H1 |
| H1-F2 | free trim in `gz_curve` (`free_trim=True`): at each heel the waterplane with volume V0 and (B - G) . xi = 0 (2-D Newton on sinkage and trim, finite-difference Jacobian, bracketed fallback) | a box with LCG off-centre at heel 0 against the wall-sided trim equation (root found independently) within 1e-9 rad; a symmetric box: free-trim GZ = fixed-trim GZ within 1e-10; residuals V and longitudinal lever below 1e-10 | H1 |
| H1-F3 | the full B1 hull (`build_b1_hull(full=True)`: both bow sides, deck/bottom/frames spanning port to starboard); `closed_shell` and the strict audit on it (the audit's plane faces with section edges of two surfaces) | closed shell; volume = 2 x half-hull within 1e-12; upright B on the centreline; heeled +-phi mirror-symmetric; heeled volume and B against an independent reference (z-slices, y-limits by Newton, kinks split) within 1e-8; dV/dzeta = A_wp heeled within 1e-7; strict audit clean or the remaining refusal documented | H1-F1 |

Open risks: (1) the sub-patch subdivision near a waterline touching a surface tangentially (cost; budget
refusal); (2) the 2-D trim Newton near flat lever curves; (3) the audit route for plane faces bounded by
sections of two different NURBS surfaces (P5 refused other section boundaries).

H1 follow-ups outcome (2026-10-16): F1 6b4a908, F2 1047e45, F3 85c9243 plus three Sol rounds (r1 da05007,
r2 09c0c34, r3 fix f864e6b unreviewed, owner option b). MERGED to local main f864e6b (suite 3708 passed,
none new; not pushed). Closed waterlines need no P1 topology (turns from exact sub-patch subdivision plus
knot-edge crossings; distinct levels all kept); free trim by Newton with a residual-checked bracketed
fallback (refuses ill-conditioned equilibria it cannot resolve); the full B1 hull audits clean (other
surfaces' sections on plane faces cleared by the distance engine). Costs: full-hull audit ~35 s; the
bracketed free-trim fallback ~40-110 s per call; local full suite ~28-33 min.

#### P5 sub-plan (2026-10-15)

A closed B1 region needs the bow face bounded entirely by section curves (deck z = 3, bottom
z = 1/2, two frames x = a, x = b): its corners are intersections of two section curves on the NURBS
face, which P4c still refuses. The bow surface's own boundaries (stem at y = 0, aft end at x = 0)
lie in the centreline/transom planes, which P1 cannot section (a plane containing a patch boundary);
the B1 bow therefore uses a chart range (``u_range``) that stops short of the stem, so the centreline
plane misses the support, and the shell is closed between the two frames.

| Sub-slice | Output | Acceptance | Depends |
|---|---|---|---|
| P5a | section x section joints on NURBS faces (P4c): corner vertices at the exact intersection of two section curves (section_roots), the pair certified by P4b's world piece certificate | a bow piece bounded by four section curves validates; a crossing pair is refused | P4d |
| P5b | the B1 closed half-hull between two frames: bow piece (four sections), deck and bottom faces (section + lines), two frame faces (section + lines), the centreline face (lines) | every edge shared by exactly two faces; ``strict_audit`` clean | P5a |
| P5c | closed-shell certificate (each edge used twice with opposite orientations, consistent outward normals) and the enclosed volume and centroid by the divergence theorem (``F = (0, 0, z)``: deck/bottom from the certified P4b areas and first moments, the bow from a chart quadrature with an error estimate) | volume and centroid vs an independent Gauss-Legendre reference (v-quadrature, frame limits by Newton in u) within 1e-9 relative; a shell with a missing face refused | P5b |

P5 outcome (2026-10-15): P5a-c plus four Astra rounds (r1 06d1a0b, r2 8a131ae, r3 85f25a5, r4 68d5ccb).
Contract decisions in the fixes: every section-audit joint (curve x curve, curve x plane, line x patch,
line x section) must keep its departure stretch within a fixed joint ball of JOINT_BALL = 8 tolerances
(departure angles below about 15 degrees refuse); plane x plane faces must overlap along their
intersection line only on shared edges; enclosed_volume integrates in a local frame and refuses when
the volume or the first moments have not settled (rtol 1e-8). Owner chose option c after round 4: fix
the volume finding, merge, and defer two sub-tolerance face-contract findings to P4e:
(1) plane_faces' rounding slack 2**-40 x coordinate scale can hide an overlap of that size (a model
translated by 1024 hides a 9e-10 overlap): uncertain overlap must refuse; (2) _outside_plane_face
decides an unshared section's side from its midpoint, so a section entering a plane face within a
joint (2.5e-10) passes face_face: certify the whole piece outside.

P4e scope (extended): the world-space face band (above) plus findings (1) and (2).

#### P4e sub-plan (2026-10-15; owner: contract a, the tolerance band, for face pairs)

Contract (owner 2026-10-15, after H1): face pairs follow the edge-level contract (a). Two faces that
share edges or vertices are clean when every pair of their points closer than the tolerance lies
within the joint band: within JOINT_BALL (8) tolerances of a shared vertex, or of a shared edge;
everywhere else they are certified farther apart than the tolerance. This replaces the P4d option-b
reading (exact intersection = shared edges). The two deferred P5 round-4 findings (a 9e-10 overlap past
a shared edge's end; a 2.5e-10 entry at a corner) lie inside the band and are acceptable; their tests
say so, with counterparts beyond the band refused.

| Slice | Output | Acceptance | Depends |
| --- | --- | --- | --- |
| P4e-1 | plane x plane faces (`plane_faces`): where the non-shared boundaries keep off the other plane, each face meets the other plane's tolerance slab exactly in its inside stretches of L times the strip, so contacts are where the two faces' stretches come within the tolerance; those must lie in shared edge spans or vertex balls widened by JOINT_BALL tol, and along a shared edge the dihedral angle must keep the band within the ball (sin theta >= 1 / JOINT_BALL, exact); the 2**-40 coordinate slack goes | B1 clean; the round-3/4 overlap past a shared end (also translated by 1024) clean as a band contact; the same overlap pushed 20 tol past the end refused; existing crossing tests refused | P5 |
| P4e-2 | an unshared section piece in a plane face's plane (`_outside_plane_face`): clearance from the face's straight edges by certify_line_section at the band radius (JOINT_BALL + 1) tol (contacts only in joint stretches), and one point outside both end balls outside the face | B1 clean; the round-4 corner entry (2.5e-10) clean; an entry 20 tol from the corner refused | P4e-1 |
| P4e-3 | plane x NURBS faces sharing section edges (`face_face`): a world band certificate over the NURBS face's chart box: every sub-rectangle is either farther than the tolerance from the plane, or crossed by the plane along a chart axis s with exact bounds m <= |d dist / ds| and M >= |dS/ds| (world) and the zero reached inside the rectangle on every s-line, so points within the tolerance lie within (M / m) tol of the exact section; M / m <= JOINT_BALL required | B1 clean (bow x deck, bottom, frames); a shallow-angle crossing (band wider than the ball) refused; a NURBS face touching the plane within the tolerance away from the shared edges refused; time per pair measured | P4e-2 |

Open risks: (1) the band certificate's rectangle subdivision near the face's chart box edge where the
section lies on the box boundary (zero reached at the rectangle edge: the rectangle is widened into the
support); (2) cost of exact derivative ranges; (3) uncovered section pieces near the plane face's
boundary must be cleared at the band radius, not the tolerance (P4e-2). P5 merged to local main 68d5ccb (2026-10-15; suite 3646 passed, none new; not pushed).

P4e outcome (2026-10-15): P4e-1..3 plus three Astra rounds (r1 b27465b, r2 55fb8a7, r3 fixes 825f4a5
unreviewed, owner option b). MERGED to local main 825f4a5 (suite 3697 passed, none new; not pushed).
Final design: plane x plane by inside stretches of L with joint intervals (shared spans, vertex
chords) widened by JOINT_BALL tol, no coordinate slack, refusal when coordinate rounding exceeds
tol/16; plane x NURBS by path coverage, uncovered section parts cleared at (JOINT_BALL+1) tol by the
distance-based pair engine (joints only at shared vertices), and _descent_band_clean (monotone plane
distance along a chart axis across Bezier pieces, M/m <= JOINT_BALL, crossed in the rectangle or a
two-sided extension inside the chart). Known: near a shared plane x NURBS corner the clearance's joint
stretches reach JOINT_BALL (JOINT_BALL + 1) tolerances; descents leaving the chart and near-tangent
sections that P1 cannot section refuse.

## Open questions

- Q1 bow loading (proposal above).
- Q2 bow form: proposal is a simple single NURBS patch per side (degree 3 x 3); a real
  hull lines plan can replace it later.
- Q3 CI wheel runs: who triggers them (no workflow runs without the owner).
- Architecture questions Q4-Q11 are listed in GENERALITY_ARCHITECTURE.md section 9.

## Log

- 2026-10-13: study written; owner decisions D1-D7 recorded; plan rev 2.
- 2026-10-13: G0, S1, S2 (partial), N0, N1a implemented on local branches; architecture
  rev 1-3 with two Codex (Astra) reviews; third review and Sol reviews of S1, N0+N1a running.
- 2026-10-19: current-status section added (from an API status review checked against the repo); plan and architecture moved to main.
