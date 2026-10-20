# Certified hydrostatics: acceptance manifest (CH1, fixed 2026-10-20)

Prospective criteria for CH1b, CH1c and the CH1d/CH2 assessment runs, written before any of
those runs (policy `ANY_ECOSYSTEM_RISK_PROPORTIONATE_ENGINEERING_V1`, section 5). Design:
`CERTIFIED_HYDROSTATICS_DESIGN.md` revision 5. A criterion changed after a run is a new decision
with a new assessment; the earlier result stays as recorded.

## 1. Comparison rules

- Exact forms: exact equality (`Fraction` equality of rationals; for `Q / |n|` forms equality of
  `Q` and of `|n|^2`).
- Binary64 enclosures: section 6 rules of the design (`lo == hi` equal to the exact value, or
  adjacent doubles with `lo < exact < hi`).
- Decimal enclosures: contain the exact value; width at most the requested width.
- States, contacts, loop counts, nesting and identities: exact equality with the expected
  values below.
- No tolerance anywhere. Timings are recorded, not judged (except the resource limits, section 6).

## 2. Closed-form fixtures (Q01)

All fixtures use body coordinates in metres, `o = (0, 0, 0)` and `n = (0, 0, 1)` unless stated
(so `|n| = 1` and every quantity is rational). **Registered waterplane basis** for an axis-aligned
normal (a valid basis needs `(e1 x e2) . n != 0`): `n = +-e_x`: `e1 = (0, 1, 0)`, `e2 = (0, 0, 1)`;
`n = +-e_y`: `e1 = (1, 0, 0)`, `e2 = (0, 0, 1)`; `n = +-e_z`: `e1 = (1, 0, 0)`, `e2 = (0, 1, 0)`;
a fixture may state another basis explicitly. Expected values follow from elementary integration;
the fixture code states each one next to its definition. (Amended 2026-10-20 before any
assessment run: revision 1 gave one default basis, invalid for the `x`- and `y`-normal planes of
F5 and F9; Codex review of CH1b.)

| Id | Solid | Plane(s) | Expected |
|---|---|---|---|
| F1 | box `[0,a] x [0,b] x [0,c]` with `a, b, c = 2, 3, 5` | `z <= h` for `h = 0, 1, 5/2, 5, 6` and `z <= -1` | `V = a b min(max(h,0),c)`; `J` of the clipped box; `A = a b` and moments of the rectangle for `0 < h < c`; states per section 3 of the design (h = 0: occupancy EMPTY, below ABSENT, above PRESENT; h = c: FULL, below PRESENT, above ABSENT; h = -1: EMPTY, generic, no section; h = 6: FULL, generic, no section) |
| F2 | tetrahedron `(0,0,0), (1,0,0), (0,1,0), (0,0,1)` | `z <= h`, `h = 0, 1/4, 1/2, 1` | `V(h) = (1 - (1-h)^3) / 6`; `A(h) = (1-h)^2 / 2` for `0 < h < 1`; at `h = 1`: below PRESENT with one outer loop of limit area 0 (collapsing loop), above ABSENT, occupancy FULL |
| F3 | triangular prism: triangle `(0,0), (2,0), (0,1)` extruded `0 <= z <= 3`, cut by `x <= s` (normal `(1,0,0)`, basis `o=0, e1=(0,1,0), e2=(0,0,1)`) | `s = 0, 1, 2` | `V(s)`, `A(s)` of the trapezoid/triangle sections, closed form |
| F4 | extruded U: polygon `(0,0),(3,0),(3,3),(2,3),(2,1),(1,1),(1,3),(0,3)` in x-y extruded `0 <= z <= 1`, cut by `y <= t` (normal `(0,1,0)`, basis `o=0, e1=(1,0,0), e2=(0,0,1)`) | `t = 1/2, 2` | `t = 2`: two components (rectangles `[0,1] x [0,1]` and `[2,3] x [0,1]` in `(x, z)`), `A = 2`; `t = 1/2`: one component, `A = 3` |
| F5 | box `[0,3] x [0,3] x [0,1]` with a through bore `[1,2] x [1,2]` | `z <= 1/2`; `x <= 3/2` | `z <= 1/2`: one component with one hole, `A = 8`, `V = 4`; `x <= 3/2`: section is two rectangles (the bore splits it) |
| F6 | saddle solid `-1 <= x, y <= 1`, `-2 <= z <= |x| - |y|` (faces triangulated as planar facets) | `z <= 0` | degenerate: below one loop, above two loops; equal waterplane quantities on both sides; contacts list the saddle vertex |
| F7 | pyramidal recess: box `[-2,2] x [-2,2] x [0,2]` with an inverted square pyramid hollow, apex `(0,0,1)`, base `[-1,1]^2` at `z = 2` | `z <= 1` | above: one outer loop with one hole collapsing at the apex (hole of limit area 0); below: no hole |
| F8 | two holes in one face with a collinear vertex/edge (the round-3 example): plate `[-4,4] x [-2,2] x [0,1]` with bores `[-3,-2] x [-1,1]` and `[2,3] x [-1,1]` | `z <= 1/2` | one component with two holes; no `SYMBOLIC_DEGENERACY` |
| F9 | unit cube `[0,1]^3` | `n = +-e_k`, `d = +-1/2` (6 planes) | `A = 1`, first and second moments of the unit square in the given basis; orientation factor check |

## 3. Transformations (Q02)

For F1, F2 and F4: translations by `(7, -3, 11)`; rotations by exact rational orthogonal matrices
(from Pythagorean triples, e.g. `[[3/5, -4/5, 0], [4/5, 3/5, 0], [0, 0, 1]]` and a 3-axis product
with denominator 1105); reflection in `x = 0` with the face orientations reversed; the plane and
basis transformed with the solid. A model stores float coordinates, so before a rotation with
denominator `q` the solid is scaled by `q` (5 or 1105), making the rotated coordinates exact floats.
Expected: for a scale `s`, `V' = s^3 V`, `J' = s^4 R J + V' t` (exact), `A' = s^2 A`, first
waterplane moments `s^3` and second moments `s^4` times the originals in the transformed basis;
without scaling (`s = 1`) `V`, `A` and the waterplane moments are unchanged and `J' = R J + V t`.
(Amended 2026-10-20 before any assessment run to describe the scaling the fixtures need.) Planes built with ANYstability's convention
(`R = Ry(-trim) Rx(heel)`, half-space `R[2] . r <= -origin_z`, basis `R[0]`, `R[1]`) at heel
`0.1, 0.5` and trim `0.02` (floats as given; the exact results are for those floats).

## 4. Exact derivative check

For F1 (`z <= h`, `h` in `(0, 5)`), F2 (`h` in `(0, 1)`) and F3: choose four distinct `d` in one
open interval between consecutive vertex heights, recover the cubic `V(d)` exactly by
interpolation, differentiate exactly at a fifth interior `d`, and require equality with `A / |n|`
(rational) at that `d`.

## 5. Verifier attacks (CH1c)

Each must refuse with `VERIFICATION_FAILED` and the stated sub-reason: a result for another query
with equal quantities (F1 at `h = 6` presented for `h = 7`: `query`); altered `V`, a waterplane
quantity, a bound, a state, an accuracy flag, a loop count or an identity (`value`, `enclosure`,
`state`, `accuracy flag`, `topology`); a requested decimal form deleted and the proof regenerated
(`request`); an enclosure wider than requested (`[0, 2]` for `V = 1` at accuracy `1e-12`:
`enclosure`); non-adjacent binary64 endpoints (`enclosure`); altered contacts (`value`); a
substituted model or face set (`model`, `face set`); a stale snapshot (`snapshot`); a claimed
Python or dependency version differing from the running one, and a wrong expected artifact identity
(`artifact`); a malformed proof, an oversized proof, an integer token over the digit limit
(`MALFORMED_PROOF`).

## 6. Refusal coverage per phase and resource limits

`BUDGET_EXHAUSTED`, `DEADLINE_EXCEEDED`, `CANCELLED` and `BIT_LENGTH_EXCEEDED` each raised in
preparation, query, verification and `check_preparation`, with work counts on the refusal.
Resource limits for the CH1d/CH2 runs: per run an operation budget of 10^8 units, a bit-length
cap of 4096 bits, a wall-time ceiling of 600 s per solid and plane set; exceeding one is a
recorded refusal, not a pass.

## 7. recipe3 plane set (CH1d)

Source: `C:/ANYworkspaceWork/barge-source-reconciliation-20261019/reference-barge` (read, not
copied; file SHA-256 recorded in the run record). For every solid (`HULL`, `PUMPROOM`, `V1..V4`,
`WB1..WB8`):
- upright planes `z <= h` for `h` = the solid's lowest vertex height, `+0.5`, `+1.0`, its mid
  height, its highest vertex height, and every distinct vertex height of the solid (degenerate
  planes included);
- heel/trim planes with ANYstability's convention through the solid's centroid height: heel in
  `{-0.3, -0.1, 0.1, 0.3}` x trim in `{-0.02, 0, 0.02}`;
- the WB1/WB2/WB7/WB8 main-deck query exactly as ANYstability registers it (upright, offset 2 m),
  expected to be degenerate with both one-sided limits returned.
Criteria: every plane admits, queries and verifies (CH1c verifier, expected identity of the run's
artifact); recorded per plane: state, degeneracy, contacts, loop counts, work counts, wall time.
