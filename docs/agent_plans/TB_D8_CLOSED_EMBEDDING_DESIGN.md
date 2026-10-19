# T-b / D8: an exactly closed planar embedding for `arrange_material` on planes

Design slice (Fable), revision 4, 2026-10-14. Design only; no source changes. Base:
local main 0c4ab4a. Companion documents: `GENERALITY_ARCHITECTURE.md` rev 6 (R.1 C1-C4,
D1-D8, R.2 T0-T8, R.3, R.5), `GENERALITY_PROGRAMME_PLAN.md` ("R1b step 2", "Findings
for T-b / D8", "R1c E4a merged"). Reviewed by GPT 6 Astra (high) three times; section
9 lists every finding and its disposition. The owner decided Q-a..Q-i on 2026-10-14
exactly as recommended (section 8); the contract below is written under those
decisions. E1 is being implemented by the main session; its contract (S1, S2, S4 and
the E1 row) is unchanged in this revision.

## 1. The problem, restated on the code

`arrange_material` (`src/anygeometry/material_arrangement.py`) builds, per face, a
half-edge graph whose vertices are tolerance clusters of stations and whose edges are
pieces that keep their *own* stations. Three consequences, verified on 0c4ab4a with
read-only probes (session scratchpad; nothing committed, no benchmarks):

1. **Not closed (T-b).** Two stations identified as one vertex can be `tol` apart, so
   incident edges do not meet at a point; cycle polygons overlap or leave gaps,
   `_exact_loop_area` is not the area of any polygon, and no point can be certified to
   lie in a cell. Main's probe (`32 tol` left of the first edge's midpoint, D5) answers
   for the wrong region whenever a cell is thinner than `32 tol`: the 1.2-tol thin cell
   depends on the outer loop's start vertex, the 2-tol nested outlines get ownership
   `[0, 0, 2]`. The step-2d certificate was unsound for this reason.
2. **Lost crossings (D8).** `_parameters` / `planar_stations.merge_parameters` merge
   split parameters closer than `32 eps` in parameter space, i.e. `32 eps * L` in
   length on a piece of length `L`, independent of `tol`. When `32 eps * L > tol` two
   geometrically distinct stations are merged on one path but kept distinct by vertex
   identity on the other: a T-junction with a gap, hence "unresolved bridge" or
   "failed material conservation". Probe (4x4 square, traces through `(2, 2)` at
   angles `0.3` and `0.3 + 20 eps`): main fails at `tol = 1e-20` and `1e-30`; with a
   tolerance-only merge the same input gives 4 cells / 12 paths. The 128-eps endpoint
   snap of `point_parameters` (both implementations) is the same kind of constant.
3. **`scaled_crossing/1e-100` is neither T-b nor D8.** Both faces receive **no traces**
   and one `ArrangementPoint((0, 0, 0))`, `0.5e-100` off face 2's plane (`tol =
   1e-109`). `batch_intersections._pair_paths` builds the trace as `witness ±
   (extent + 1.) * direction`; the absolute `+1.` makes the rounded ends exactly
   `±direction`, so the stored trace passes through the origin, misses both faces
   except at face 1's corner (an "isolated contact"), `MaterialDomain.contains`
   projects that point onto face 2 without an off-plane check, and the seams (extent
   floor `max(..., 1.)`) reach no boundary. A tolerance-only merge does not change
   this. The repair is A7 (slice E5).

## 2. Semantics to build

Plane support, straight pieces (boundary edges, clipped traces, seams); arcs: section
6. Decisions are taken in the exact chart of `planar_exact.ExactPlane` (affine: stored
doubles have exact rational chart coordinates); outputs are world positions of
vertices. Scaled chart forms (`U = offset . dual_u`, `V = offset . dual_v`) avoid
divisions where possible; stations at junction parameters are general rationals
(denominators up to 56 bits on grid16, probe), so this is a constant-factor saving.

**S1 Stations and tolerance junctions.** `planar_stations.junctions`,
`point_parameters`, the clip and the seams D6/D7 stay exact replicas of main's rules
with three changes. (a) Every parameter-space constant that *creates or moves a
station* becomes a length rule on the piece: the 32-eps merge (D8) and the 128-eps
endpoint snaps of `point_parameters` / `_exact_horizontal_roots` (snap iff the
sub-piece to the endpoint has exact length `<= tol`). The 8-eps windows of
`contains_loop` are parity bookkeeping; E1 audits them and leaves them unless the
audit shows a station-equivalent effect. (b) **Canonical operand roles (T7).** The
line/curve roles of `_line_line` are assigned by a data key, not by path index: the
segment with the larger exact squared length is the line; ties by the lexicographic
order of the exact station tuples `(start, end)`. Forward and reversed calls then
evaluate the same expression and return the same stations and the same refusals; the
key replaces `definition_checksum` for derived pieces (which have no stored record,
Q-h). (c) **Short pieces.** A segment with exact squared length `<= tol^2` never takes
the line role: if the other segment is longer it takes the role (b already ensures
this); if both are short, `_line_line` is skipped and the candidates are the four
endpoint `point_parameters` tests, which is what `_line_line` returns for coincident
planes anyway. `_check_line` (zero exact or zero float length) still refuses, as main
does. Hence the `_line_line` refusal "degenerate arrangement line" for a first segment
`<= tol` (`planar_stations.py:171`) is no longer reachable from the pair loop.

**S2 Deferred removal.** Clipping (`_clip_line_exact`, also for seams) keeps all
exactly distinct parameters and tests every gap's midpoint; it merges nothing. Pieces
of length `<= tol` therefore reach the pair loop (handled by S1c) and global
registration; their removal happens once, by S4. Registration order is unchanged
except where main's merges removed a station (listed cases only).

**S3 Vertex identity.** Main's rule, unchanged: a station joins the lowest-index vertex
whose first station is within `tol` (exact squared distance); registration order =
boundary paths, traces, seams, pieces in parameter order (`ExactVertexLookup`). This
is the only place where `tol` enters the topology. Added exact rules: two stations
with equal exact chart coordinates and different world positions (possible under S9)
are one vertex; a cluster containing two stored face vertices refuses (main refuses
through `start == end`).

**S4 Split rule (D8 replacement).** A sub-piece is dropped iff both stations identify
to the same vertex and its exact length is `<= tol`; a sub-piece longer than `tol`
whose stations identify to one vertex refuses (typed; main's `start == end` refusal,
named). The rule is evaluated on the shared identity, so it is identical on every
path and in both operand orders.

**S5 Representatives and re-anchoring.** Each vertex has one exact position: the
stored face vertex when the cluster contains one (R.3 3a), else the first station
(Q-a). Every piece becomes the segment between its vertices' representatives, in world
(output) and in the chart (decisions); a re-anchored piece whose direction reverses
relative to its stations is a legal edge (direction is not graph data; the parameter
map is re-derived). **Deformation contract (Q-b)**: every vertex is within `2 tol` of
each station it identifies and every re-anchored edge within `2 tol` of its piece
(convexity) — a tolerance-scale deformation, non-zero only where a cluster holds more
than one distinct station (on generic inputs only at exact crossings, where the two
stations differ by the skew of rounded coplanar lines, rounding level).

**S6 Overlay and closure policy.** The re-anchored segments are overlaid with zero
tolerance over all edge pairs passing the float box filter, including pairs sharing a
vertex: shared vertex — collinear overlap (`cross == 0`, `dot > 0`) or the longer edge
containing the shorter's far endpoint; disjoint vertices — proper crossing, T-junction
(vertex interior to an edge), collinear overlap; same vertex pair — duplicate, merged
with owners, flags and boundary multiplicity united (for straight pieces main's "same
canonical endpoints => same image" is now exactly true). **C-conservative
(recommended, Q-g)**: any crossing, T-junction or overlap refuses (typed: "arrangement
not closable within tolerance"); the embedding is the re-anchored segments, planar by
the check, every vertex a station or stored vertex. **C-repair (opt-in, E2b)**: exact
repair vertices, not re-identified, one pass, terminates by construction; needs S10.
`embedding_is_planar(E)` (a graph generalisation of `region_is_simple`) is the test
oracle and the diagnostics check.

**S7 Cycle walk.** `_compare_outgoing` (step 2c) on re-anchored directions; exactly
equal directions cannot occur on a planar embedding; bridges refuse as today.

**S8 Domain validity, faces, material.** *Validity* of the re-anchored loops
(`region_is_weakly_simple`, new), checked before any classification and refusing the
domain (typed: "invalid material domain") otherwise. Loops are treated as *unoriented*
point sets: main does not check hole winding (a reversed hole is accepted, probe:
corner-touching triangle with the same sign as the outer loop gives area 14.5), so no
sign rule is imposed; the stored `forward` flags, `source_edge`s and the
`orientation` flag (sign of the stored outer loop, as today) keep their provenance,
and every polygon test below is winding-agnostic. Rules, on the embedding after the
junction phase has split every edge at every contact point: (0) **every loop is
simple on its own**, checked on the loop's own stored stations before splitting
(`planar_exact.region_is_simple` applied to the single loop: no two non-adjacent edges
of the same loop meet, adjacent edges meet only at their shared vertex) -- touching is
allowed only between different loops; a self-crossing or self-touching loop refuses
(Astra round 4: the bow-tie outer loop `(0,0),(4,4),(0,4),(2,0)` split at its
crossing `(4/3,4/3)` has no proper crossings, no overlaps and non-zero area 4, passes
(1)-(3), and then breaks S9's partition identity: parity lobes 16/3 + 4/3). Rule (0) is checked twice: on the stored stations, and again **after S3-S5** (identification, removal, re-anchoring) on each boundary loop's occurrence sequence of vertices -- a loop that visits a vertex twice, or whose non-adjacent re-anchored edges meet, refuses (typed "invalid material domain") before classification (Astra round 4.1: stored outer loop `(0,0),(6,0),(6,6),(4,6),(3,e),(2,6),(0,6)` with `0 < e < tol` is simple, but a bottom-edge station at `(3,0)` identifies with the notch tip and re-anchoring makes the loop self-touching); (1) every
loop has non-zero exact area; (2) no two edges cross properly and no two edges overlap
collinearly (a shared edge between two loop occurrences is boundary multiplicity 2
and refuses, Q-c decided); edges may share vertices, including a loop vertex that lay
in the interior of another loop's edge (now a shared vertex); (3) **open-edge
containment**: because of (2) an open edge meets no other loop, so it lies entirely
inside or entirely outside each loop polygon and its midpoint decides exactly: the
midpoint of every hole edge is strictly inside the outer polygon and strictly outside
every other hole polygon; the midpoint of every outer edge is strictly outside every
hole polygon (`point_in_polygon`, never `EQUAL` by (2)). Rule (3) replaces the
vertex-only test of revision 3, which accepted an exterior "hole" with all vertices on
the outer boundary (outer `(0,0),(6,0),(6,6),(4,6),(4,2),(2,2),(2,6),(0,6)`, hole
`(2,3),(4,4),(3,2)` inside the notch: every hole-edge midpoint is outside and the
domain now refuses). Shared-vertex sectors need no separate rule: if a hole's wedge at
a shared vertex contained outer-boundary edges, those edges' open intervals would lie
inside the hole polygon and fail (3). Touching holes are valid, as on main (probe: a
hole touching the outer corner gives 1 cell, 0 holes, area 14.5; a hole vertex on an
outer edge gives 1 cell, area 14.0 — the walk absorbs the touching hole into the outer
cycle, and the design reproduces this because the walk and the parity rule below do
not depend on loop disjointness or winding). `region_is_simple` stays in the query
route (`intersections.py:136, 297`), which is not changed here. *Boundary incidence* is
a per-edge multiplicity from loop membership by path index (`paths[:len(boundaries)]`),
inherited by splits, summed on duplicate merges; traces and seams contribute zero.
*Hole ownership*: for a negative cycle `N` with leftmost vertex `v` (min `U`, then min
`V`), the open horizontal segment leaving `v` toward `-U` lies in the face `F` on `N`'s
left, which owns `N`. Among edges not incident to `v`, the nearest hit on that ray is
computed exactly with the half-open rule (an edge counts iff `V_lo <= V_v < V_hi`;
horizontal edges never). Edge-interior hit: the half-edge with `v` on its left names
the cycle `C` bounding `F` there. Vertex hit `w`: `C` is the cycle of the wedge at `w`
containing `+U` in `w`'s exact cyclic order (the wedge between the last and the first
outgoing half-edge; when a half-edge at `w` points exactly `+U`, the wedge above it —
its interior would otherwise have been hit first). An edge incident to `v` pointing
exactly `-U` (it cannot belong to `N`): `C` is the cycle of its half-edge whose left
side is `N`'s wedge at `v`. Positive `C`: owner; negative `C`: `owner(N) = owner(C)`
(terminates: `C` contains a point with `U < U_v`); no hit: unbounded face. No
direction changes, no refusal on ordinary or touching polygons. *Material by parity*
on the dual graph: the unbounded face is non-material; crossing an edge flips the state
iff its boundary multiplicity is odd; with a valid domain this is the exact region
membership. Main's probe is a diagnostic counter only.

**S9 Preconditions and areas.** Inputs on the support within `tol` (exact residual)
else refuse (typed; replaces the 1e-100 "unresolved bridge" until E5). Exact shoelace
per cycle. Conservation: (i) exact partition identity `sum(material faces) + dropped
slivers == area(outer) - sum(area(holes))` on the re-anchored loops, an assertion;
(ii) deformation evidence `|area(re-anchored loops) - area(stored loops)| <= 2 tol *
perimeter(stored loops)`, refused if violated (guards the implementation). Main's
`|area| <= tol^2` sliver drop kept (Q-d).

**S10 Certification of the persisted embedding.** `apply`
(`batch_intersections.py:923-967`) maps edge endpoints to model vertices by reusing any
model vertex within `tol` sharing a source, merging several candidates, and otherwise
creating a vertex at the rounded position; ownership accumulates across faces. The
persisted geometry (double coordinates, possibly moved by up to `tol` through reuse)
can therefore differ from the certified embedding without any two vertices
collapsing: a vertex can land on an edge, two edges can cross, the cyclic order at a
vertex can change. Specification: (1) the arrangement exports its vertex table
(representatives, `endpoints_by_edge`, the certified rotation system = the cyclic
order of half-edges at every vertex, and the boundary multiplicities); (2) apply
resolves the mapping **batch-wide first**: every embedding vertex of every face of the
batch to a model vertex (existing, merged, or new at the canonical rounding), without
writing; (3) for every face of the batch it then re-certifies the **persisted
embedding** on the final model coordinates (doubles as exact rationals, in the face's
exact chart) by **recomputing the whole certified classification and comparing it**:
`embedding_is_planar` holds, every vertex maps injectively, the exact cyclic order at
every vertex equals the certified rotation system (this fixes the boundary walks of
every connected component), and then — because a rotation system says nothing about
which component lies in which face of another component — the cycle walk, the
ownership ray of S8, the unbounded face, the parity material labels and the cycle
area signs are recomputed on the persisted coordinates and the resulting cells (outer
chain, hole chains, material label, boundary multiplicities) must equal the certified
cells exactly; (4) any failure refuses the whole batch before any write (typed:
"certified embedding not preserved on application"); apply is already the caller's
atomic transaction. This covers rounding, nearby-vertex reuse, multi-candidate
merges, cross-face effects and component nesting by checking the complete result
rather than the mechanism; its cost is one more cell phase per face on doubles. Separability
(two distinct vertices whose canonical roundings coincide) refuses at construction.
Under C-conservative every vertex is a station or stored vertex, so failures need a
non-transitive cluster or a pre-existing model vertex within `tol` of a certified
vertex — the cases main resolves silently by merging today.

Operand-order symmetry (T7) is structural: junction predicates use canonical roles
(S1b), identity uses the registration order (the plan's canonical entity order), and
every later step is a function of the embedding. The harness reversal runs are
evidence that the implementation follows this, not the argument.

## 3. Candidates compared

**A. Snap-merge then iterated closure.** Identify within `tol`, re-anchor, add repair
vertices, identify again, repeat to a round cap: cascades, cap-dependent refusals, no
bound stronger than `rounds * tol`.

**B. Exact arrangement first, tolerance as an output filter.** Overlay own stations at
zero tolerance, drop `|area| <= tol^2`, let apply merge within `tol`: loses main's
tolerance junctions (a trace ending `0.5 tol` inside the face dangles) and creates
slivers that apply collapses into degenerate faces — the contraction problem moves to
apply, where S10 shows it already lives.

**C. Tolerance identification once, exact overlay of the re-anchored segments**
(section 2) with the conservative or the repair closure. A' (own stations, cycle
polygons closed by `<= tol` connectors) rejected: connectors cross other edges, cycle
polygons overlap, "inside the cycle polygon" does not imply "inside the cell".

| | A | B | C-conservative | C-repair |
|---|---|---|---|---|
| Planarity | capped iteration | by construction | exact check, else refuse | by construction |
| Tolerance semantics | main's + cascades | changed | main's | main's |
| Invented vertices | yes | no | none | exact crossings, `<= 2 tol` |
| Persisted certification (S10) | needed | needed | needed | needed |
| New refusals | cap-dependent | — | tol-scale degenerate inputs | none |
| Reuse of 2a-2d, R1c | full | partial | full | full |

**Recommendation**: C-conservative first (E2), repair as an opt-in slice (E2b) once
refusal counts on real workloads justify it. Every vertex then is a station or a stored
vertex, each decision change is "the exact answer for the stored data under main's own
identity rule", and the only new behaviour is a typed refusal where main gave a
probe-dependent answer or silently deformed on apply.

## 4. Performance (grid16/grid32/tilted8 within 10-20 % of main)

Measured on grid16 (one plan+apply: 1.66 s, of which `arrange_material` 0.62 s; 17
faces; at most 180 edges, 110 vertices, 81 cells per face; median 28 edges):
non-adjacent edge pairs with overlapping float boxes total **102** for the whole
workload (adjacent pairs 1200, needing only the collinearity test). Junction
parameters have denominators up to 56 bits; an orientation test on such Fractions is
of the order of 10 us, so the overlay tests cost ~1-10 ms on grid16. Candidate search:
a full pairwise NumPy box test per face (`E^2` booleans; ~400^2 x 33 faces on grid32:
a few MB, milliseconds); no sweep or bucket structure. Ray hits: one NumPy prefilter
per negative cycle. Parity BFS O(E). The S10 re-certification repeats the planarity
candidate search and the cyclic-order comparison once per face on doubles: the same
cost again. Probe containment calls leave the decision path (diagnostic only on
request). Acceptance per slice: interleaved timing (step-2a protocol) `<= 1.05x` main
on grid16/grid32/tilted8, hard limit `1.10x`; grid32 (`HEAVY`) once per slice under
the resource rules.

## 5. Effect on public outputs and the Q13 contract

- Vertex coordinates: main's, except a cluster containing a stored face vertex not
  registered first (then the stored vertex; within `tol`).
- Piece coordinates: vertex positions; rounding-level on generic inputs, up to `2 tol`
  on degenerate inputs — this needs the owner's acceptance of the S5 contract (Q-b).
  Junction stations of skew rounded pairs may move at rounding level where the
  canonical role (S1b) differs from the index role.
- Topology digests identical on G0 (every slice). Expected changes, each argued:
  (a) D8 merges with `32 eps * L > tol` and 128-eps snaps longer than `tol` no longer
  happen — stations farther than `tol` apart are distinct vertices, as identity says;
  (b) thin boundary cells (1.2 tol) are material for every start vertex — a face of
  the subdivision with area `> tol^2` inside the region; (c) nested outlines 2 tol
  apart: `[0, 1, 1]`, the ownership ray hits the ring's outer square; (d) Sol's
  merged-start input: main refuses ("no hole owner"); the trace start `(3e-7,
  9.5e-7)` is `0.996 tol` from `(0, 0)` and `1.19 tol` from `(0, 2e-6)`, identifies
  with the corner, answer two cells of total area 16; (e) probe/parity
  disagreements listed individually with the exact check "the probe lies outside its
  own cycle polygon" where that is the explanation, else the slice is blocked;
  (f) typed refusals (off-support input, collapsed piece, not closable, invalid domain,
  separability, persisted embedding not preserved) replace "unresolved bridge" /
  "conservation" on the inputs of section 1 and (a)-(d), and the audit's refusal set
  changes accordingly (Q-e).
- Reuse: `planar_stations` (`Segment`, `junctions` with S1b/c, `point_parameters`
  with S1a, `boxes_separated`, `ExactVertexLookup`), `_clip_line_exact` (S2),
  `_add_exact_planar_seams`, `_compare_outgoing`, `_contains_loop_exact` (clip
  midpoints, queries), R1c `planar_arcs`. Replaced: `merge_parameters`/`_parameters`
  on the exact route, `_exact_planar_cells`, `_exact_loop_area`. Gating stays
  `exact_cells = Plane and all pieces straight`.

## 6. Arcs and curves: deferred (E4), open contracts named

(i) arc-arc overlay needs root isolation for ellipse-ellipse intersections (a quartic
in `tan(t/2)`), canonical enclosures, refusal when undecided; (ii) tangent incident arcs
need exact curvature ordering, else refuse (D2); (iii) an arc station cannot be
re-anchored: stored-vertex priority vs an arc endpoint station is resolved by a
connector edge bounded by the cluster diameter (`<= 2 tol`) or by refusal (owner
choice when E4 is designed); (iv) `contains_loop_exact_curved` and R1c line-arc roots
cover point-in-cycle and line-arc crossings. Until E4, faces with arcs keep main's
float cell phase, reported as uncertified.

## 7. Slices

Each slice: G0 digests identical; full suite (main's failures only); harness zero new
flips and zero reversal flips; interleaved timing; Codex review; completion checklist.

| Id | Output | Acceptance (hand-derived expectations) | Deps |
|---|---|---|---|
| E1 | S1a-c, S2, S4: length rules for the 32-eps merge and 128-eps snaps; canonical roles and short-piece handling in `junctions`; deferred removal in clip and seams; drop/refuse rule on shared identity; audit list of remaining eps windows | nearly coincident traces (20/100/1000 eps, `tol` 1e-20 and 1e-30): exactly 4 cells / 12 paths, both operand orders; non-transitive case (stations 0.9/0.9/1.1 tol): hand-derived vertex count; a short clipped piece (`<= tol`) preceding an overlapping trace and a seam, both operand roles, decided without refusal; `junctions(a, b)` and `junctions(b, a)` return identical station sets on 1e4 random pairs incl. skew pairs; G0 identical; `<= 1.05x` | — |
| E2 | S3, S5, S6 conservative, S9, `embedding_is_planar`; walk on the embedding; main's cell phase kept on the re-anchored polygons (so E2's decision changes are only re-anchoring's: none expected on G0) | `embedding_is_planar` holds on all G0 and corpus arrangements; zero "not closable" refusals on G0; overlay unit cases: crossing created by re-anchoring, `(0,0)-(2,0)` vs `(0,0)-(1,0)`, T-junction, reversed piece, duplicate, zero-length, chart-coincident vertices, off-support point; perturbation tests as in 2a; `<= 1.05x` | E1 |
| E2a | S10: vertex table and rotation system exported; batch-wide mapping; persisted-embedding re-certification; atomic typed refusal; separability refusal | plan-apply-plan on G0 and corpus: byte-identical documents per step **and** the persisted embedding certified on every face; constructed cases: (1) two exact vertices rounding to equal doubles refuse at construction; (2) a pre-existing model vertex within `tol` of a certified vertex whose reuse puts that vertex on another edge refuses at apply with no write; (3) a reuse that changes the cyclic order at a vertex without collapse refuses; (4) two faces sharing a trace vertex where reuse is valid on one face and invalid on the other: whole batch refused; (5) multi-candidate merge of two existing model vertices both incident to batch faces: re-certified or refused; (6) a disconnected island cycle (an isolated trace outline inside a cell) whose persisted vertices, through reuse within `tol`, move it across a thin edge into the neighbouring cell with planarity, rotation system and area signs unchanged: refused by the ownership/material comparison | E2 |
| E3 | S8 (weakly simple validator, multiplicity, ownership ray with vertex events, parity), S9 conservation identity and deformation bound; probe demoted | thin cell: 2 cells for all four start vertices; nested rings `[0, 1, 1]`, area 16; touching holes in both windings: corner 1 cell / 0 holes / 14.5, edge-vertex 1 cell / 14.0 (as main); the exterior notch "hole" of S8 refuses as an invalid domain; the self-crossing outer loop `(0,0),(4,4),(0,4),(2,0)` refuses as an invalid domain before classification (S8 rule 0); the notch loop `(0,0),(6,0),(6,6),(4,6),(3,e),(2,6),(0,6)`, `e = 0.5 tol`, with a station at `(3,0)` (a trace crossing the bottom edge there) refuses after re-anchoring (rule 0, second check); a hole enclosing another hole refuses; ray hitting a vertex exactly; two holes on one horizontal; islands/holes fixture unchanged; shared loop edge refuses; probe-disagreement counter zero on G0, every corpus disagreement listed | E2 |
| E2b | S6 repair (opt-in flag) | repair cases with hand-derived vertex positions; G0 unchanged | E2a, E3 |
| E5 | A7 plane x plane: `planar_exact.plane_plane_line` carried as exact stations through clipping, box-relative extent, float `LinePath` at the output boundary; seams from exact stations and directions with box-relative extent | `scaled_crossing/{1e-100, 1e3, 1e8}` succeed with the scale-1 topology digest; G0 identical | E2 |
| E4 | arcs (section 6) | deferred until (i)-(iv) are designed | E3, R1c |

E2 and E3 are reviewed together as one semantic change; E2a is required before any
merge of E2 into main because without it the certified embedding is not what gets
written.

## 8. Owner decisions (2026-10-14: Q-a..Q-i decided as recommended)

- Q-a representative: stored vertex when present (bound `2 tol`) — **decided**
  (persisted operands never move inside the arrangement); alternative was the first
  station (bound `tol`).
- Q-b deformation under Q13: the S5 contract (`<= 2 tol`, rounding-level on generic
  inputs) is an allowed output change — **decided**.
- Q-c boundary multiplicity: coincident loop occurrences refuse — **decided**.
- Q-d slivers: main's `|area| <= tol^2` drop kept — **decided**.
- Q-e refusal set: the audit's refusal identity may change under Q13 as decisions do,
  each change listed — **decided**.
- Q-f E5 scope: in this programme — **decided**.
- Q-g closure policy: conservative first, repair opt-in later — **decided**.
- Q-h T7 canonical order for derived pieces: the data key of S1b (exact squared
  length, then exact station tuples) in place of `definition_checksum` — **decided**.
- Q-i persisted certification: refuse the whole batch atomically on any face whose
  persisted embedding is not preserved — **decided**.
- Round 3 added no new owner choice: loop winding is not a decision operand (S8 treats
  loops as unoriented, which is main's behaviour), the open-edge containment rule is a
  strengthening of validity under Q-c's refusal policy, and the complete persisted
  certificate is the implementation of Q-i.

Risks: non-transitive identity kept (main's rule; S10 makes its consequences
visible); grid32 timing unverified until measured per slice; refusal texts change on
degenerate inputs; E5 is the first use of `ExactLine` in the batch route and needs
station provenance across the plan/clip boundary. Not verified: timings beyond the
grid16 probe; the harness under reversal on the new corpus cases.

## 9. Review findings and dispositions

Round 1 (Astra): 1 apply re-merges vertices — accepted, S10 (now re-certification of
the persisted embedding, see round 2); 2 shared-vertex overlaps — accepted, S6; 3
fallback rays — accepted, S8 single horizontal ray with vertex events; 4 boundary
provenance — accepted, S8 multiplicity by path index; 5 deformation — accepted, S5
contract, Q-b, S9 split; 6 D8 shared identity — accepted, S2 deferred removal, S1a
snaps; 7 E5 exact line — accepted; 8 E4 — deferred with contracts; 9 performance —
accepted, measured, dyadic claim withdrawn; 10 owner questions — accepted.

Round 2 (Astra):

1. S10 checked injectivity, not preservation (blocking) — **accepted**, verified at
   `batch_intersections.py:923-967` (nearby reuse within `tol`, multi-candidate merge,
   ownership accumulation). S10 now: batch-wide mapping first, then per-face
   re-certification of the persisted embedding (planarity + identical rotation system
   + area signs) on the final coordinates, atomic typed refusal before any write; E2a
   tests (2)-(5) cover topology change without collapse and cross-face effects.
2. Deferred clipping exposes short pieces to `_line_line`'s refusal (major) —
   **accepted**, verified at `planar_stations.py:171`. S1b/c: canonical roles (longer
   segment is the line) and short-short pairs decided by endpoint tests; the refusal
   is unreachable from the pair loop; E1 tests both operand roles.
3. Symmetry statement conflicts with T7 (major) — **accepted**. S1b makes junction
   predicates structurally symmetric through a canonical data key; the reversal runs
   are evidence only; Q-h asks the owner to accept the key for derived pieces.
4. Touching domains vs `region_is_simple` (major) — **accepted**, verified
   (`planar_exact.py:250-286` rejects shared vertices; main's arrangement nevertheless
   succeeds on touching holes, probe values in S8). S8 defines
   `region_is_weakly_simple` for the arrangement route (touching allowed, overlap
   refused), ownership and parity unchanged; E3 fixtures carry main's hand-checkable
   outcomes; `region_is_simple` stays in the query route.

Round 3 (Astra):

1. S10's certificate omits component nesting (major) — **accepted**: a rotation
   system fixes each component's boundary walks, not which face of another component
   contains it. S10 now recomputes the full classification (walk, ownership ray,
   unbounded face, parity labels, area signs) on the persisted coordinates and
   requires the cells to equal the certified cells; E2a case (6) moves a disconnected
   island across a thin edge with planarity, rotation system and signs unchanged.
2. Vertex-only containment accepts an exterior "hole" (major) — **accepted**, checked
   by inspection (the three hole vertices lie on the notch edges `x = 2`, `x = 4`,
   `y = 2`, whose interior side is outside the region). S8 rule (3): open-edge
   containment by exact midpoint tests after contact splitting, for hole edges
   against the outer and the other holes and for outer edges against the holes;
   sectors at shared vertices are covered by the same rule (argument in S8); invalid
   domains refuse before any classification; E3 fixtures added (notch "hole", nested
   holes).
3. Opposite-winding rule rejects main's accepted reversed corner fixture (major) —
   **accepted**, verified (`probe_touch.py` reverses the triangles; main gives 14.5
   regardless). S8 treats loops as unoriented for validity and uses winding-agnostic
   polygon tests, keeping `forward` flags, `source_edge`s and the `orientation` flag
   as provenance (main's behaviour, no new compatibility change); E3 runs the touching
   fixtures in both windings.

Round 4 (Astra): S10 nesting, S8 open-edge containment and winding resolved; new major -- S8 did not require each loop to be simple on its own (bow-tie outer loop passes rules 1-3 after splitting, breaks S9) -- **accepted**, S8 rule (0) per-loop simplicity on the stored stations, E3 regression added (main session edit).
Round 4.1 (Astra): rule (0) on stored stations only misses self-contacts created by re-anchoring -- **accepted**, rule (0) re-checked after S3-S5 on boundary occurrence sequences, E3 regression added (main session edit).
