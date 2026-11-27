# Display tessellation (`anygeometry.display_tessellation`)

Edge samples for viewers, shared by every face using an edge, so neighbouring faces cannot crack
apart. The module reads the model through public calls only and is not re-exported from
`anygeometry`.

Contract (owner decision 2026-11-25, N1): only lines, circles and ellipses on planes, cylinders
and cones, each divided in **closed form**. Everything else is a typed refusal; there is no
measured sampling (no probes, sub-samples, margins or breakpoints).

## Supported set

| | Supported | Refused |
|---|---|---|
| Incident supports (`model.faces_using_edge`) | `Plane`, `Cylinder`, `Cone` | any other support (`ExtrudedSurface`, NURBS, Coons marker, none, ...) or a face with an explicit `parameterization`: `SUPPORT_UNSUPPORTED` (the caller keeps its own route for that edge) |
| Edge curves | `Straight`, `LineRange` whose carrier IS the edge's segment (exact domain `[0, 1]`, both carrier endpoints the evaluated ends and the edge's topology vertices within `model.tolerance.length`, round 5) (lines); `Arc` (circle, via `model.arc_frame`); `EllipticArc` (a circle when `|u| = |v|` and `u.v = 0` to 8 eps relative, else an ellipse; `u`, `v` need not be orthogonal) | every other curve (`Spline`, `NurbsCurve`, `CurveOnSurface`, intersection curves, ...), and a `LineRange` drawn over part of its carrier (any other exact domain, reversed included): `EDGE_UNSUPPORTED` |

An edge used by no face (a lone curve) has no supports and is divided by its tangent alone.
Supports are checked before the curve type.

**`LineRange` (owner decision, N1 round 4, the recommended option).** A `LineRange` is supported
only when its carrier endpoints ARE the edge's segment: its exact domain is `[0, 1]` and both
carrier endpoints equal the edge's evaluated endpoints within `model.tolerance.length`. Any other
`LineRange` is `EDGE_UNSUPPORTED` ("a line range drawn over part of its carrier is not
supported"), after the coordinate checks (a carrier beyond 1e9 stays `OUT_OF_DOMAIN`). Round 4:
a 1-unit range retained over `[0, 2^-26]` of a carrier at x = 9e8 was drawn as one interval
whose exact carrier turned 0.4636476358228977 rad against the limit 0.4636476328426804, its
exact end 1.49e-8 off the drawn segment at sag 1e-10: the float ends of a retained range need
not lie on the exact carrier to the bounds, and an oracle built from those ends cannot see it.
Round 5 (owner decision, the recommended option): the carrier endpoints must also equal the
edge's TOPOLOGY vertices (`model.vertex_position`) within `model.tolerance.length`, not only its
own evaluated ends; otherwise `EDGE_UNSUPPORTED`. Round 5: a `[0, 1]` carrier from `(0, 0, 0)` to
`(3, 0, 0)` with both vertices 1e-6 off in Y (valid at `curve_fit_residual = 1e-5`) was drawn
although the model length is 1e-9.

## Exact divisions

`edge_display_samples(model, edge_id, DisplayTolerance(angle, sag=None))` returns `EdgeSamples`:
parameters strictly increasing from exactly 0 to exactly 1, points read-only; `points[0]` and
`points[-1]` are the vertex positions exactly, interior points are
`model.evaluate_edge_many(edge, parameters)`. The parameters are the sorted union of the following
divisions; every interval of the no-merge union lies inside one interval of each, so all bounds
hold on it, and an interval of the merged union is kept only when verified against every bound
(below: merge, then verify with exact forward maps; restore on failure).

Circles and ellipses are divided in their **local angle** `s = theta - start = sweep t`: the frame is
rotated once, `u' = cos(start) u + sin(start) v`, `v' = -sin(start) u + cos(start) v`, so
`c + cos(s) u' + sin(s) v'` is exactly the curve at local angle `s` and no absolute angle (up to
1e6 rad) is formed and rounded before the start is subtracted. Lines are taken from their
evaluated ends (`model.evaluate_edge_many(edge, [0, 1])`, the exact evaluator), never from a
`LineRange` carrier's float fields (a retained range can be far narrower than its carrier).

Counts are plain ceilings, `n = ceil(span / step)`; one piece fewer only when its emitted step
`span / (n - 1)` is verified to be at most `step` to the span's own rounding (4 eps relative: an
exact multiple can come out a few ulps above an integer). There is no fixed slack.

1. **Tangent** (at most `angle` per interval). Lines: none. A curve `c + cos(theta) u + sin(theta) v`
   has the tangent `M w(theta + pi/2)` with `M = [u v]` in an in-plane basis (det > 0) and
   `w(x) = (cos x, sin x)`; its direction angle `psi` is monotone. Circles: uniform in `theta`.
   Ellipses: `n = ceil(|delta psi| / angle)` equal steps of `psi`, each mapped back to
   `theta` exactly: `w ∝ M^-1 (cos psi, sin psi)`, lifted with the polar decomposition
   `M = R(alpha) S` (the angle between `e` and `S e` stays within pi/2).
2. **Support azimuth** (at most `angle` per interval, for each incident cylinder or cone: the
   turning of a cylinder's normal, an upper bound for a cone's). The curve is projected onto the
   plane normal to the axis, in the module's OWN orthonormal frame about the support's unit axis
   (owner decision, N1 round 10, the recommended option): `e1` the normalised cross product of the
   axis with the basis vector least aligned with it, `e2 = axis x e1`; never the support's stored
   radial / circumferential directions, so the turning and every branch decision are independent
   of the frame's phase (round 10: a cylinder whose radial seed lay nearly along its axis stored a
   radial 2.2e-5 off orthogonal; projecting onto it moved `g` by 1.1e-5 against a bound of
   1.2e-13, drawing an arc with an interior turnback and refusing a strictly monotone one). A
   cylinder or cone whose stored frame is self-inconsistent (orthonormality error above 1e-9) is
   refused (domain table).
   - Lines: every decision uses the quantity that is accurate for it, never a reconstruction
     (round 3: `B' = A' + P(b - a)` cancelled an end's own 1.5e-9 radius to 0 next to a 2e7 start,
     and the line was drawn as one chord). Each end's radius from its OWN projection,
     `A' = P(a - o)`, `B' = P(b - o)` (rounded by `(4 eps + F) |p - o|`, `F = 4 eps` the module frame's deviation, round 10); the direction from the
     projection `D' = P(b - a)` of the exact difference (a ruling's is exactly 0 where its
     separately rounded ends are not); the segment's interior closest approach (from both ends'
     estimates, `t* = -(A'.D') / |D'|^2` and `1 - (B'.D') / |D'|^2`, distance `|P' x D'| / |D'|`,
     counted when either lies inside the segment beyond its ends' rounding) BEFORE any end
     shortcut. An end on the axis to the frame's rounding (projected radius
     `<= 8 eps max(1, |end|, |axis point|)`) is only possible where the support reaches its axis,
     a cone's apex (a zero radius): the projection is a ray from the axis point, constant azimuth,
     no division; on a cylinder or a frustum such an end is `OUT_OF_DOMAIN` (round 3: a line from
     1.5e-15 behind a 2e-9 cylinder's axis hid an interior axis crossing); both ends on the axis:
     `OUT_OF_DOMAIN`. Collinear (`|A' x D'| <= 8 eps |A'| |D'|` AND `|B' x D'| <= 8 eps |B'| |D'|`,
     `D' = 0` included; when the two estimates disagree the line is divided, the conservative
     branch) with `A'`, `B'` on the same side: constant azimuth (cylinder rulings, cone
     generators), no division; on opposite sides, or with an interior closest approach: the line
     crosses the axis, `EDGE_UNSUPPORTED`. Otherwise `A' + t D'` (`cross(A' + t D', D')` constant)
     has a monotone azimuth spanning less than pi (the wrapped difference of the ends' own
     azimuths): equal azimuth steps, each `t = -cross(A', e) / cross(D', e)`. There is no
     apex-proximity shortcut beyond the frame's rounding: a line from near an apex is divided like
     any other.
   - Circles and ellipses: `c' + M' w(theta) = N (z + w(x))` (`N = M' J` with det > 0, `z = N^-1 c'`).
     `arg(z + w)` is monotone while `g = 1 + z.w` keeps its sign (its extremes over the interval
     are found in closed form). The branch is decided from `g` only when its sign is decidable
     with margin (owner decision, N1 round 9, the recommended option): `|g|` at both ends, at
     every interior extreme and at the interval's midpoint must exceed its rigorous rounding
     bound `g_bound` (stability table); otherwise the azimuth is near-stationary and the edge is
     `OUT_OF_DOMAIN` ("azimuth near-stationary: branch not decidable"). No value is snapped to 0
     and no branch is inferred from a value within its bound (round 9: at the minimum sweep a
     strictly monotone edge whose `g` evaluated to 0 was refused as a turnback, and a nearby edge
     with a real 1.8e-10 rad turnback was drawn after its end values were snapped). Consequence: a
     valid arc whose azimuth is exactly stationary at an end (`g = 0` there: the curve tangent to
     a ray from the axis, round 1 #6) changed from drawn to `OUT_OF_DOMAIN`. A sign change with
     margin, or the projection reaching the axis, is `EDGE_UNSUPPORTED`. Equal azimuth steps, each solved exactly (`A cos(theta) + B sin(theta) = C`
     on the monotone branch: on the ray `e`, with `z = (z.e) e + h e_perp`, the point is
     `w = sign(g) sqrt((1 - h)(1 + h)) e - h e_perp`, formed directly: the former
     `(z.e)^2 - |z|^2 + 1` and `lambda e - z` cancelled terms of size `|z|^2` and `|z|` on a thin
     projection).
     This divides circles off a cylinder's axis (admitted within the model's tolerance) and plane
     sections of cones (the axis at a focus of the projected ellipse) exactly.
3. **Sag** (chord within the allowance). Allowance: `tol.sag`, else the smallest over incident
   cylinders and cones of `r_max (1 - cos(angle / 2))` (cylinder radius, cone's larger radius);
   none without either.
   - Vertex offsets: the first and last drawn chords end at the vertex positions, which may lie
     up to `|vertex - curve(0 or 1)|` from the curve's ends. Such a chord is the analytic chord
     with an end moved, each of its points by at most that offset, so its distance from the curve
     is at most the exact sag plus the offset. End intervals are bisected until that bound holds
     (an offset at the coordinates' resolution, 4 eps x max(1, |vertex coordinate|), is rounding
     and counts as 0). An offset above the allowance is `EDGE_UNSUPPORTED` ("a vertex lies farther
     from its curve than the sag allowance"). Lines: the sag is the vertex offsets only.
   - Circle: a chord over a step `delta` leaves the circle by exactly `rho (1 - cos(delta / 2))`,
     so the uniform count is raised to `ceil(|sweep| / (4 asin(sqrt(sag / (2 rho)))))`; only an
     end interval with a vertex offset is checked (and bisected) further.
   - Ellipse: the sag of every interval is computed exactly, and an interval above the
     allowance is bisected in `theta` (deterministically) until none is; sub-intervals keep the
     tangent and azimuth bounds. *Proof:* `P(theta) = c + M w(theta)` is an affine image of the
     unit circle. The signed distance of `P` from the chord's line is an affine function vanishing
     on that line, hence a fixed multiple of the circle point's signed distance from the preimage
     chord; both are largest at the circle arc's parametric midpoint `m` (where its tangent is
     parallel to the chord), for `|delta theta| <= pi`. On an emitted interval the tangent turns by
     at most `angle <= pi/2` (so `|delta theta| < pi`, affine maps keeping parallel tangents) and
     the projection onto the chord is monotone, so the distance from the segment is the distance
     from the line. In closed form (no cancellation of large coordinates):
     `sag = 2 sin^2(delta / 4) |u x v| / |T(m)|`, `T(m) = -sin(m) u' + cos(m) v'` (local `m`).
   - The computed ellipse sag is raised by its own rounding, `(1 + 8 eps x aspect)`, before it
     is compared (fail closed: `|u x v|` and `|T(m)|` of a skew frame round by a relative
     `eps x aspect` at most).
4. A curved edge whose single DRAWN chord could leave it by more than
   `model.tolerance.surface_residual` gets at least two intervals (conservative). The decision is
   on the chord between the drawn endpoints (the vertex positions): the exact sag of the analytic
   chord raised by its rounding (`1 + 64 eps + 8 eps x aspect`), plus the larger vertex offset as
   measured (never zeroed as rounding), plus the evaluator's rounding `3 E` (allowance, below),
   bounds the distance of the evaluated curve from the drawn chord. Round 4: vertices one float
   (1.19e-7) off a lone arc at x = 9e8 at `surface_residual = 1e-8` were drawn as one chord (the
   offsets had been zeroed as rounding), and a lone unit circle at 999994 rad, sweep .4, at
   `surface_residual = 2 sin^2(.1)` (its analytic sag; the evaluated sag exceeds it by 2.31e-12)
   was one chord; both now get two.

The union is **merge, then verify with exact forward maps; restore on failure** (round 7: the
round-6 merge rule checked each dropped cut alone, and Codex found three in-domain violations:
two moved ends of one interval spending the same headroom twice, 0.8000606 rad against the
published 0.8000418; a circle interval lengthened past its sag, 0.07894671 against 0.07894327,
because the circle path skips the sag check; and the minimum-two split merged into the edge's
start, returning one chord).

*Merge* (a proposal; one sort and one pass). Two adjacent cuts of different divisions are merged
when the cut dropped lies, in its OWN division's angle (tangent or azimuth, radians), within
`max(headroom_D, 64 eps max(1, |angle|))` of the cut kept (`headroom_D = max(T, Z...) - term_D`).
The stricter division's cut is kept (the smaller headroom), the earlier on ties; a cut that has
absorbed others is replaced only by a cut within every absorbed cut's threshold; two cuts of one
division are never merged; the ends are never dropped. Nothing in the proposal is relied on for a
bound.

*Verify.* Every final interval with a dropped cut at either end is checked with exact quantities:
each division's angle span over it from its own forward map (the tangent angle; each support's
azimuth), accepted only when `span + term_D <= angle + allowance.angle` (the computed span plus
that division's own documented rounding term, so the true span meets the published bound); and,
for a circle with a sag allowance, its exact sag `2 rho sin^2(delta / 4)` (raised by its
rounding) against `sag (1 + 64 eps)` (the published `sag + allowance.sag` less the `3 E` that the
drawn points' rounding uses). An ellipse's sag, and a circle's end interval with a vertex offset,
are verified exactly afterwards by the sag refinement on every final interval, as before.
*Mandatory cuts:* the ends are never dropped, and the minimum-two-interval split is protected:
when the tangent plan forced two intervals, a union that would be one chord is a failure (its
dropped cuts, the split among them, are restored); the split may merge into another interior cut
(the edge still has two intervals). A forced split whose inverse rounds its cut onto an end
(not strictly inside (0, 1)) uses the parametric midpoint `t = .5` instead (exactly representable
and distinct; N1 round 8); it is not an equal-angle cut, so the intervals on both sides of it are
verified like a merged interval after the union, and a failure is `EDGE_UNSUPPORTED` (the cut is
never moved). Inside the domain the tangent of a forced split turns less than `tol.angle` over the
whole edge, so each half meets the angle bound by monotonicity.

*Restore.* An interval that fails first lets another ORIGINAL cut of each end's merge group stand
for that group, then restores every cut of the group (a merge group is a contiguous run of the
sorted cuts, so this restores every cut that lay in the interval); the pieces are verified again.
No cut is ever invented. An interval without a dropped cut is an interval of the no-merge union,
which meets every bound by construction, so the loop ends, at worst at "no merge at all". The
bounds therefore hold by the verification, not by the merge rule's argument. Over the corpus the
verification failed two merged intervals (the `tripod_skins` circle edges 19 and 35 at the fine
step without sag: the merged tangent interval sagged 1.42e-14 and 1.47e-14 relative above the
sag allowance, beyond its `64 eps` share) and
both were met by the group's other original cut (a move of 2.2e-16); no sliver had to be kept and
no group was fully restored. Every division's count
(tangent and each support's) is computed and checked against the cap, and then their sum
`1 + sum(n - 1)` (the most intervals the union can have), before any division is built, so
building costs at most 4096 cuts; more than 4096 intervals is `EDGE_UNSUPPORTED` (round 3: a
circle coaxial with its cylinder needing 2100 tangent and 2100 azimuth intervals is now refused
from the sum, where both divisions were built and merged before). No raw arithmetic error escapes a division: every division is guarded, and
any `ArithmeticError` or `ValueError` left is typed `EDGE_UNSUPPORTED`.

Measured at the Fine step (7.5 deg): a tripod mitre ellipse takes 48 intervals, the tripod
circles 24, `winding_fixture`'s two ellipses 45 and 43; the lone 20:1 half ellipse 24 (no sag
allowance), 78 at sag 1e-3 and 230 at sag 1e-4 (unchanged by the round-1 fixes: every count over
the corpus at both steps, with and without sag 1e-4, is the same; parameters moved by at most
5.6e-14). Unchanged again by the round-2 fixes: all 5744 corpus results (every edge, both steps,
with and without sag 1e-4) keep their counts and refusal codes; 40 parameter arrays (tripod
mitre ellipses and the three `cone_section` cases, through the rewritten ray inverse) moved by at
most 2.2e-16. Unchanged by the round-3 fixes: all 5744 corpus results keep their counts, codes
and parameters bit for bit. Unchanged by the round-4 fixes: all 5744 corpus results (parameters
and points) identical bit for bit to c47336c. Unchanged by the round-5 fixes: all 5744
corpus results (parameters, points and refusal codes) identical bit for bit to a49e037; 168
rounding allowances (circles seen along their axis, now of projected aspect exactly 1) moved by
at most 3.5e-16 relative. Round 6 (the aspect bound lowered to 1e4, the in-plane frame's small
diagonal from the residual's norm): no corpus result is newly refused and no refusal code or
rounding allowance changed (all 5744 compared with 2427aa3); 42 results changed through the
frame, the more accurate small diagonal moving the tangent inverse by ulps: 26 parameter arrays
(tripod mitre ellipses, `winding_fixture`, `box_corner_cut`, `shared_circle`, `skew_ellipse_edge`)
keep their counts and moved by at most 8.9e-16 (points by at most 1.1e-14), and the 16 results of
the `steep_cuts` and `steep_cuts_with_plane` ellipses (two edges each, both steps, with and without sag 1e-4)
gained one interval at 514732a: their tangent cut at the symmetric midpoint moved from
0.5 - 7.1e-15 to 0.5 + 1.04e-14, just beyond the former union's 64-ulp gap (9.0e-15) from the
azimuth cut at 0.5 (a sliver interval). With the round-6 follow-up's merge rule (below), against
2427aa3: all 5744 results keep their counts and refusal codes (the `steep_cuts` counts are back),
no rounding allowance changed, 205 parameter arrays moved by at most 7.0e-15 (the frame, and which
of two merged cuts is kept), and the shortest interval over the corpus is 4.6e-5.

## Declared numeric domain (owner decision 2026-11-25)

Checked before any division. Inside the domain the bounds above hold to rounding; outside it the
call raises `DisplayDomainRefused` (a `DisplayTrimRefused` with code `OUT_OF_DOMAIN`).

| Bound | Value, applied to | Why |
|---|---|---|
| `MAX_DISPLAY_COORDINATE` | 1e9 model units: vertex positions, arc centre and (`EllipticArc`) centre + u, centre + v, an `Arc`'s via vertex, `LineRange` carrier endpoints and a line's evaluated ends, every incident support's origin (and a plane's u, v vectors), every emitted point | float spacing 1.2e-7 there; ship and offshore models in m or mm are far inside |
| `MAX_DISPLAY_ANGLE` | 1e6 rad: `EllipticArc` start and start + sweep; cylinder and cone start and start + sweep | float spacing 1.2e-10 rad, four orders below the smallest display angle (1e-6), so `start + sweep t` keeps its steps |
| `MIN_DISPLAY_RADIUS_RATIO` | 1e-9 x max(1, max \|coordinate\|): every incident cylinder radius (scale: the edge's definition points and the support origin); every point used for a line's azimuth about a cylinder or cone: its ends by their own projections (except an end on the axis to the frame's rounding, `8 eps max(1, \|coordinate\|)`, at a cone's exact apex: an exactly zero radius and the end within `model.tolerance.length` of the apex point, axial position included, round 4; elsewhere, and both ends on it, refused) and its interior closest approach `\|P' x D'\| / \|D'\|` (scale: the edge's definition points, i.e. the vertices and the line's ends, and the support origin: an evaluated point `a + t (b - a)` is rounded at the scale of the ends, round 3); a cylinder or cone arc's closest approach (one rule for both) (a lower bound, `sigma_min(N) min \|z + w\|`, each candidate evaluated directly: `\| \|z\| - 1 \|` at the analytic critical angle, `\|z + w\|` at the ends) must exceed BOTH terms: (1) 1e-9 x the conic's own scale, its definition points (vertices, `c`, `c + u`, `c + v`) and `max(1, \|stored angle\|) sigma_max(M')`, the evaluator's rounding of `start + sweep t` moving the projected point by eps times that, NOT the support origin (round 5: the implementation had added it, refusing a closest approach .001 to a frustum whose origin is at -1e8); and (2) `64 eps \|c - o\|`, the frame's own rounding: the centre's offset from the support's axis point `o` is a float subtraction, rounded by eps times its result (round 5) | an azimuth computed from coordinates of size \|p\| at distance r from the axis is rounded by about eps \|p\| / r: at the bound 2.2e-7 rad, below the smallest display angle. Measured without the stored-angle term: a .5 circle 2e-9 from a frustum's axis at 999994 rad turns 0.116 rad per interval at `angle = .1` |
| support frame (round 10) | every incident `Cylinder` and `Cone`: a sanity check for self-inconsistent supports only: its STORED frame's orthonormality error, the three lengths `\|\|axis\| - 1\|`, `\|\|radial_direction\| - 1\|`, `\|\|circumferential_direction\| - 1\|` and the three pairwise dot products `\|axis . radial_direction\|`, `\|axis . circumferential_direction\|`, `\|radial_direction . circumferential_direction\|` (absolute), each at most `1e-9` (round 11 added the axis and circumferential lengths and the radial . circumferential product: a finite axis seed whose squared norm is subnormal stored an axis and circumferential direction 1.125 long, admitted before) (NaN refused), else `OUT_OF_DOMAIN` ("support frame not orthonormal"), for every edge on it (lines included), before any division (owner decision, N1 round 10, corrected: the first version's `8 eps` refused 65 of 20,000 random ordinary cylinders, worst 117 eps) | the divisions use only the stored axis (`_axis_frame`), so a stored frame's rounding enters no bound directly. It enters only through the support's own `evaluate()`: a frame off by `delta` makes the evaluated surface a slightly skewed cylinder or cone whose normal leaves the direction about the axis that the azimuth bound measures by at most about `2 delta` (radial and circumferential each tilted by up to `delta` out of the plane normal to the axis, or scaled by `1 + delta`): at `delta <= 1e-9` that is `2e-9` rad, a tenth of the 2 % rounding share (`2e-8` rad) at the smallest display angle and five orders below the share at `angle = 1e-3`, so it is not added to the allowance. Ordinary frames are far inside: the support orthogonalises and normalises its seed, off by about `eps / sin(seed, axis)`; measured up to `117 eps = 2.6e-14` over 20,000 random cylinders with ordinary seeds, and up to `1.8e-11` over the test's 20,000 cylinders and cones whose axis and seed components span six orders of magnitude (all admitted; the 8 eps version refused 1756 of them), so only a seed within about `2e-7` rad of the axis is refused. Round 10's seed `(1 + 1e-11, 1 - 1e-11, 0)` on the axis `(1, 1, 0)` stored a radial `2.2e-5` off orthogonal: refused |
| `MIN_DISPLAY_SWEEP` | 1e-8 rad (owner decision, N1 round 8, the recommended option), inclusive: `\|sweep\|` of every `EllipticArc` (its stored sweep) and every `Arc` (the model's arc frame), checked before any division; below it (zero and NaN included) `OUT_OF_DOMAIN` | round 8: both MEDIUM findings needed sweeps of 1e-16 and 5e-11 rad, where the stored angles' own float spacing is a sizeable share of the sweep: the tangent inverse rounded a forced cut onto an end (a 1e-16 rad ellipse drawn as one chord 1e-4 off the curve) and the azimuth branch test saw both end values of `g` inside its snapping band (a strictly monotone 5e-11 rad edge refused as a turnback). With `\|stored angle\| <= 1e4` (every non-circular conic, aspect >= 1, and every conic with a cylinder or cone support, by `MAX_DISPLAY_ELLIPSE_ASPECT`) the float spacing of the stored angles is at most 1.8e-12 rad, four orders below the minimum sweep, so the tangent inverse and the azimuth branch see a resolved interval; a lone circle's stored angles (up to `MAX_DISPLAY_ANGLE`, spacing 1.2e-10) lie two orders below it. A three-point `Arc` cannot reach the bound: the model's collinearity rule refuses arcs below a sweep of about 4e-6. Test geometries over 8e-9 and 1e-14 rad moved to 1e-8 (their assertions unchanged) |
| rounding share | the edge's rounding allowance (`display_rounding_allowance`, below) at most 2 % of `tol.angle` (else `OUT_OF_DOMAIN`) and of the sag allowance (else `EDGE_UNSUPPORTED`, the unresolvable-sag code) | round 3: the allowance must be small against what is asked. It binds only near the other bounds: at the radius bound the azimuth term is about `64 eps / 1e-9 = 1.4e-5` rad, so angles below about 7e-4 rad are refused there; at coordinates 1e9 the sag term is about 5e-6, so sags below about 2.7e-4 are refused there (a test at sag 3e-5 there moved to 3e-4). Over the corpus the largest share is 8.7e-8 (angle) and 2.0e-6 (sag) |
| `MAX_DISPLAY_ELLIPSE_ASPECT` | 1e4 (owner decision, N1 round 6, the recommended option: was 1e6), inclusive: the computed product is compared with `1e4 x (1 + 4 eps)` (the rounding of an aspect computed from exactly orthogonal axes at the bound; round 6: the SVD quotient of `u = (1e6, 1e6, 0)`, `v = (-1, 1, 0)` came out `1e6 (1 + 2.2e-16)` and was refused at the then-bound; for a skew frame the computed aspect is accurate to a relative `eps x aspect`, at most 2.2e-12 here). Applied to a non-circular ellipse's aspect `a / b` (its principal semi-axes, the singular values of `[u v]`) times `max(1, \|start\|, \|start + sweep\|)`; and, for a circle or an ellipse, the same product for its PROJECTION across every incident cylinder's or cone's axis (the 2x2 frame `M'`, whose aspect can be far above the curve's: an ellipse of aspect 100 tilted almost into the axis direction projects at 1e8). A projected frame that is a similarity to 8 eps relative (equal column norms, orthogonal columns: a circle seen along its axis) has aspect exactly 1 (round 5: the computed `1 + 2 eps` at stored angle `nextafter(1e6, 0)` was refused), so on a cylinder or cone a circle's stored angles are bounded by 1e4 too (lone circles and circles on planes keep `MAX_DISPLAY_ANGLE`). The curve's own bound is checked before the parallel-axes test: a nondegenerate ellipse (`sigma_min > 64 eps sigma_max`) beyond it is `OUT_OF_DOMAIN`; only axes parallel to that resolution are `EDGE_UNSUPPORTED` (round 5) | the tangent direction moves up to `a / b` times faster than the arc angle, and the evaluator rounds the absolute angle `start + sweep t` by eps \|theta\|: the tangent error is at most `1e4 x eps = 2.2e-12` rad, six orders below the smallest display angle. Measured with `a / b <= 1e6` alone: aspect 1e6 at 999994 rad turns 10 % over `angle = 1e-3` per interval, local divisions notwithstanding. The azimuth is resolved through `M'^-1` and back (`chi`, then the polar part of `M'`), which stretches an angle error by up to `aspect(M')`: within the bound it is at most `1e4 x eps = 2.2e-12` rad. Measured outside it: the 1e8 projection above turned 0.266 rad on one interval at `angle = .1`. Why 1e4 (round 6): at the old bound conic rounding amplified by the aspect was measured beyond the published bound at products 9.9e5 (round 3, a projected aspect 990099) and 6.8e5 (round 6, a skew frame whose interval turned 1.04e-5 rad beyond the angle), and rounding refused exact boundary cases at 1e6 (rounds 5 and 6): every measured overshoot lay within a factor ~1e2 of the old bound. The new bound is 68 times below the smallest product with a measured overshoot; every term the aspect scales (`32 eps x aspect x theta*`, tangent and projected) is at most 7.1e-11 rad, 3.6e-3 of the 2 % share (2e-8 rad) at the smallest display angle (it was 0.36 of it); 312 random skew frames up to the bound, each at an angle putting every interval at the limit, turned no interval beyond the angle |

No apex is recognised by proximity: a line end counts as on the axis only when its projection is
0 to the frame's own rounding (`8 eps max(1, |coordinates|)`: the evaluator's exact apex of a
tilted cone projects about 3e-16 off it), and a generator is recognised by the collinearity of
`A'` and `D'`,
so a frustum's narrow end (both radii positive, however different) is never an apex and a line
from within `model.tolerance.length` of an apex is divided by its azimuth like any other.
An end on the axis counts as the apex only for a cone with an EXACTLY zero radius
(`radius_start == 0`: the apex is `origin`; `radius_end == 0`: `origin + height axis`) and only
when the end lies within `model.tolerance.length` (plus the apex point's own float rounding,
`8 eps max(1, |apex coordinate|)`) of that apex point, its axial position included; otherwise
`OUT_OF_DOMAIN`. Round 4: a frustum with inner radius 1e-16 and a tip at the origin, and an exact
cone with its tip on the axis 1e-6 above the apex, were drawn as one chord; both are refused.
An edge id not in the model is a plain `GeometryError` (not a typed refusal).
**Argument validation** (owner decision, N2 round 4 (4); Codex N2 r4 Low b): every public entry
refuses an invalid argument with `GeometryError` (a `ValueError` subclass), never a raw exception.
`DisplayTolerance(angle, sag)` takes real numbers only (`numbers.Real`, not `bool` or `np.bool_`;
strings, complex numbers, arrays and `None` as the angle are refused; `DisplayTolerance(10**400)`
raised `OverflowError`, now a `GeometryError`). `edge_display_samples` and
`display_rounding_allowance` take a `GeometryModel`, an integer edge id of any integer type
(`numbers.Integral`, not a bool: a NaN edge id raised `ValueError`, and a float such as `3.0` was
silently truncated; both are refused now) and a `DisplayTolerance`. `DisplayTrimRefused(code, message)`
with a code that is not one of `REFUSAL_CODES` (or not a string) raises `GeometryError` (it raised a
plain `ValueError` before). The result records (`EdgeSamples`, `DisplayRoundingAllowance`) are built
by the module and take no validation. Valid arguments give the same bits as before: the N1 corpus
(5744 results) is bit-identical (0 changes).
Round 5 (owner decision, N2 round 5 (1); Codex N2 r5 M3) replaces the ABC checks by EXACT CONCRETE
TYPES (see **Trust model** below): NumPy registers `np.timedelta64` as both `numbers.Integral` and
`numbers.Real`, so a day duration raised a raw `TypeError` and a nanosecond duration was accepted as
an edge id or an angle; a Python int above the decimal conversion limit raised a raw `ValueError`
when a diagnostic formatted it. Now an edge id is `type(x) is int` (so not `bool`) or exactly one of
`np.int8`, `np.int16`, `np.int32`, `np.int64`, `np.uint8`, `np.uint16`, `np.uint32`, `np.uint64`,
converted to a Python int and checked arithmetically to `|id| < 2**63`; an angle or sag is exactly a
`float`, an `int` (not `bool`; compared with `2**1023` before `float()`, so no overflow), an
`np.float32` or an `np.float64`. `timedelta64`, `datetime64`, `Fraction`, `Decimal`, NumPy
integers as a tolerance, `bool`, `np.bool_` and any other object are a `GeometryError`; the model
and tolerance are checked with `issubclass(type(x), ...)` (never `isinstance`, which looks up a
foreign `__class__`). `DisplayTrimRefused(code, message)` takes `type(code) is str` and `type(message)
is str` and formats neither otherwise. Changed results (stated): `Fraction`, `Decimal` and NumPy
integer tolerances (accepted in round 4) and edge ids beyond `2**63` are now refused; ordinary calls
(int ids, float or int tolerances) are bit-identical (N1 corpus rerun below).

Inside the domain, a sag allowance at or below 4 eps x max(1, |vertex coordinates|) on a curved
edge is `EDGE_UNSUPPORTED` (unresolvable), as are distinct parameters evaluating to one point and
evaluator errors.

## Rounding allowance (`display_rounding_allowance`)

"Holds to rounding" means one thing, given by ONE public function,
`display_rounding_allowance(model, edge_id, tol) -> DisplayRoundingAllowance(angle, sag,
axis_floor, axis_floors)`: inside the declared domain, every emitted interval's tangent turns, and every
incident cylinder's or cone's azimuth changes, by at most `tol.angle + angle`, and every drawn
chord lies within the sag allowance `+ sag` of its curve. The dense oracle in
`tests/test_display_tessellation_edges.py` checks exactly this, with no other slack, and checks
that the allowance is at most 2 % (which `edge_display_samples` enforces). `axis_floors`
(round 5) gives each incident cylinder's or cone's OWN floor, in the order of
`model.faces_using_edge` (first use): a line's `1e-9 x` the scale of its definition points and
that support's origin; a conic's the larger of the two closest-approach terms of the domain
table. The oracle measures the azimuth about each support at points at least its own floor from
its axis (round 5: a floor shared across supports, .5 from a second cylinder's origin at
5e8, skipped every point of a .01 cylinder, and a one-chord replacement turning .5 rad passed).
`axis_floor` is a summary only: `MIN_DISPLAY_RADIUS_RATIO x scale`, the scale taken over the
edge's definition points and EVERY incident support's origin. It is not a measurement threshold:
it can exceed a small support's every distance from its axis (round 6: in the two-cylinder case
above it is .5 while one cylinder's whole chord lies within .01 of its axis). Callers measuring
support turning use `axis_floors`, one per support.

With `eps` the float epsilon and `theta* = max(1, |start|, |start + sweep|)` (1 for a line):

- `E`, the rounding of an evaluated point: a conic, `8 eps (|c| + |u| + |v| + sigma_max theta*)`
  (the evaluator forms `start + sweep t`, rounded by `2 eps theta*`, which moves the point by up to
  `sigma_max` times that; round 3: at 999994 rad a .2 rad interval was drawn 7e-11 rad wide, its
  chord 3.5e-12 beyond the sag); a line, `8 eps (|a| + |b|)`. Per incident cylinder or cone,
  `E_o = E + 8 eps |o|` (the projection relative to its axis point `o`), and `r` the smallest
  distance from its axis the division uses (a conic: its closest-approach lower bound; a line: its
  ends' and interior closest approach).
- `angle = 64 eps x tol.angle + max(T, Z for each support)`:
  - `T = 32 eps W`, `W = aspect x theta*` (aspect 1 for a circle; a line: `8 eps`): each end's
    tangent direction is off by `aspect (2 eps theta* + 4 eps)` (the evaluated angle and the
    vector), and the tangent division's inverse by a few `eps x aspect`;
  - a conic's `Z = 32 eps aspect(M') theta* + 8 E_o / r`: the projected forward map and inverse
    (`N^-1`, the polar part `S`) stretch a rounding by `aspect(M')` (round 3: at a projected aspect
    of 990099 the forward map under-estimated a span by 1.45e-10 rad and an interval turned
    2.1e-10 rad beyond the angle); each evaluated point's azimuth is off by `E_o / r`, at both chord
    ends and at every measured point;
  - a line's `Z = 2 G / r`, `G = err(A') + err(B') + err(D') + 8 eps max(|A'|, |B'|) + 2 X` (how
    far its true position across the axis can be from the computed one: each cut, or a constant
    azimuth's collinearity, is off by at most `G / r`; `X = eps max |coordinate|` of the evaluated
    ends, each end's absolute coordinate rounding, round 4), plus `4 E_o / r` when interior samples
    are drawn; a ray from an apex: `2 (rho + err + X) / axis_floor` (`rho` the apex end's projected
    radius).
- `sag = 64 eps x sag + 3 E` for a conic (both chord ends and the measured point move by `E`);
  `64 eps x sag + 8 eps max(1, |b - a|) + X` for a line (plus `E` when interior samples are drawn;
  `X` as above, round 4: the drawn ends are the true ends rounded, so the published bound holds
  for every supported line); plus any vertex offset counted as 0 at the coordinates' resolution.

Worst value inside the domain: 2 % of the angle and of the sag allowance, by the rounding-share
bound (above); over the corpus at most 8.7e-8 of the angle and 2.0e-6 of the sag. Recomputed with
the round-6 aspect bound 1e4: the aspect terms (`T = 32 eps W` and a conic's `32 eps aspect(M')
theta*`) are at most `32 eps x 1e4 = 7.1e-11` rad (7.1e-9 at the old bound), so they alone can no
longer approach the 2 % share at any admitted angle (2e-8 rad at the smallest, 1e-6); the share
still binds only through the radius terms (`8 E_o / r`, `2 G / r`, about 1.4e-5 rad at the radius
bound, as in the domain table) and the sag terms at large coordinates.

## Stability within the domain (round-2 class audit)

The class audited: a float rounding or cancellation, at an admitted scale, that breaks a
closed-form bound or raises. Every formula of the module, and why it is stable inside the
declared domain (or which bound makes it so):

| Formula | Form used | Why stable within the domain |
|---|---|---|
| counts `ceil(span / step)` | quotient checked finite and `<= 4096` before any division is built; one fewer only when `span / (n - 1) <= step (1 + 4 eps)` | `step >= 1e-6`; the span is a difference of computed angles (4 eps relative) |
| `_AngleMap` determinant, inverse | `det` of the 2x2 frame, refused when `|det| <= 1e-12 |M|^2`; inverse = adjugate / det | relative error `eps x aspect`, at most 2.2e-12 by the aspect bound (curve and projection) |
| singular values `sigma_max`, `sigma_min`, aspect | `sigma_max = (hypot(m00 + m11, m10 - m01) + hypot(m00 - m11, m01 + m10)) / 2`, `sigma_min = det / sigma_max` | sums of squares and a quotient: no cancellation (the former `sqrt(F^2 - 4 det^2)` cancelled for near-circles) |
| polar angle `alpha` | `atan2(m10 - m01, m00 + m11)` | the pair's norm is `>= sqrt(2 det) > 0`; atan2 is relative-accurate |
| `z = N^-1 c'` | adjugate product | `|z|` large on a thin projection: relative error `eps x aspect(M')` (projected aspect bound) |
| sign of `g = 1 + z.w` | `1 + |z| cos(x - arg z)` at the ends, the interior extremes and the midpoint; each `|g|` must exceed `g_bound = 2 (I + Z + V)` (N1 round 9, owner decision), else `OUT_OF_DOMAIN` (near-stationary), never snapped | a decision with margin. `g_bound` bounds `|computed g - exact g|` rigorously (first order, doubled for the second-order terms), with `s = sigma_min(N) = det / sigma_max`: **I (inputs)** `g = cross(p, p') / det N` (`p = c' + N w`, `p' = N w'`), so absolute errors `E_N` of the projected frame and `E_c` of the projected offset move it by at most `(E_c + E_N) / s + (|c'| + sigma_max) E_N / (sigma_max s)`; on a cylinder or cone `E_N = (8 eps + F) (|u'| + |v'|)` (one rotation, 2 eps, and a dot product with a unit frame vector, 3 eps plus its unit length) and `E_c = (8 eps + F) |c - o|` (the subtraction, eps, then the projection), `F = 4 eps` the module frame's deviation from an exact orthonormal frame about the axis (round 10; it was absent, and the stored frame's deviation unbounded); the tangent map's offset is exactly 0 (`g = 1`, no input term); **Z (`z = N^-1 c'`)** the computed det is off by a relative `eps |N|_F^2 / det <= 2 eps aspect`, the adjugate product by `2 eps` per term: `8 eps (aspect + 2) |c'| / s`; **V (evaluation)** hypot, atan2, cos, the product and the sum to an ulp each and `x - arg z` to `eps (|x| + pi)`: `8 eps (1 + |z| (4 + |x|))`. Each term is a sum or quotient of positive quantities. Margin at every end and interior extreme proves one strict sign over the interval (`g`'s extremes are among them; a critical point moved by rounding across an end changes `g` by second order only, within `V`). Over the corpus no edge is near this bound (all 5744 results unchanged) |
| closest approach (conic) | `det / sigma_max x min(hypot(z0 + cos x, z1 + sin x), ||z| - 1|)`, each evaluated directly | absolute error `eps (1 + |z|)` of a distance, far below the 1e-9 radius ratio; the scale includes `max(1, |stored angle|) sigma_max(M')` for cylinders and cones alike |
| forward azimuth `angle(theta)` | `chi = x + atan2(cross(w, q), w.q)`, `Phi = chi + alpha + atan2(cross(e, S e), e.S e)` | `e.S e >= sigma_min > 0`; `cross(w, q)` subtracts terms of size `|z|` and `S` has aspect `aspect(M')`: the error is up to a few `eps x aspect(M') x theta*` (round 3: 1.45e-10 rad at aspect 990099), covered by the allowance's `32 eps aspect(M') theta*` and at most 7.1e-11 rad by the projected aspect bound (7.1e-9 at the old bound 1e6) |
| ray inverse `solve(phi)` | `h = cross(e, z)`, `w = sign(g) sqrt((1 - h)(1 + h)) e - h e_perp` | nothing of size `|z|^2` cancels (was `(z.e)^2 - |z|^2 + 1` and `lambda e - z`); its remaining error is the forward map's, in the same allowance term |
| azimuth frame (round 10) | `axis = stored axis / \|stored axis\|` (round 11: normalised here, the stored axis need not be unit: 0.9999999995 from a subnormal seed entered `e2` directly and turned an interval 2e-11 rad beyond the bound); `e1 = (axis x b) / \|axis x b\|`, `b` the basis vector of the smallest `\|axis component\|`; `e2 = axis x e1`; the stored axis direction only | `axis x b` is exact (its components are axis components or 0) and has norm `>= sqrt(2/3)`: no cancellation; `e1` is one normalisation off (about eps), `e2` adds the cross product's rounding (about 2.5 eps): within `F = 4 eps` of an exact orthonormal frame about the axis per vector (measured over 2000 random axes: orthonormality defects at most 4 eps; test), carried in every projection's error. The stored radial / circumferential directions are not used, so a stored frame's own error (unbounded before: a radial 2.2e-5 off orthogonal, round 10) cannot enter the azimuth; its phase is irrelevant (cuts bit-identical across stored phases; test). Confirmed for the corrected 1e-9 sanity check: no bound (`g_bound`, `G`, the projection terms, the floors, the sag allowance) contains a stored-frame term, so admitting stored frames off by up to 1e-9 changes none of them; the only effect, through the support's own normal, is accounted in the domain table |
| other uses of the stored support (round-10 audit; axis length, round-11 audit) | stored-angle domain checks (`\|start\|`, `\|start + sweep\|`), the sag allowance (radius, a cone's larger radius), a cone's apex point `origin + height axis`, the cylinder radius bound; the module forms no axial coordinate | none uses the stored radial or circumferential direction: the angles bound the evaluator's own rounding, the radii and the apex depend only on origin, axis, height and radii (a cone's evaluate at a zero radius is `origin + v height axis` whatever its radial frame). The axis length enters only the apex, which keeps the STORED axis on purpose: the apex is the support's own evaluated point (`evaluate` uses the stored axis), so normalising would move the apex test away from where the surface ends (by up to `1e-9 \|height\|`); the angle checks, the radius bound and the sag allowance do not use the axis at all; every projection uses the normalised axis (`_axis_frame`) |
| line projection | `A' = P(a - o)`, `B' = P(b - o)` (each end's own), `D' = P(b - a)` | no reconstruction `A' + D'` (round 3: it cancelled a 1.5e-9 end radius next to a 2e7 start); each rounds by `(4 eps + F) \|p - o\|`, `D'` by `(4 eps + F) \|b - a\|` (`F` the module frame's deviation, round 10): the allowance's `G` |
| line on-axis, collinearity | radius `<= 8 eps max(1, |coordinates|)`, only at a cone's apex; both `|A' x D'| <= 8 eps |A'| |D'|` and `|B' x D'| <= 8 eps |B'| |D'|` (`D' = 0` included) | decisions at the frame's rounding; when the two estimates disagree the line is divided (conservative); a near-collinear line that misses the test is divided generically (span below one step: no cut) or refused by its closest approach (fail closed); the constant azimuth's residual turning `2 G / r` is in the allowance |
| line closest approach | both ends' estimates `t* = -(A'.D') / |D'|^2`, `1 - (B'.D') / |D'|^2`, distance `|P' x D'| / |D'|`, before any end shortcut | `|D'| > 0` on this path (guarded); interior when either estimate is inside beyond its ends' rounding and its own error; the smaller distance is checked (conservative) |
| line cut `t = -cross(A', e) / cross(D', e)` | direct | `|cross(D', e)| = |D'| h / |p|`; the cut's azimuth is off by at most `G / r` (in the allowance); a zero denominator is refused |
| circle classification | `|u| = |v|` and `u.v = 0` to 8 eps relative | the circle path's uniform steps and radius `max(|u|, |v|)` are then exact to 16 eps relative (was 1e-12: a 9e-13 skew overshot by 1.8e-13 rad and 2.3e-8 sag) |
| circle sag count | `4 asin(sqrt(sag / (2 rho)))`, allowance `< 2 rho` | no cancellation (`2 rho sin^2(delta / 4)`, not `1 - cos`); exact for the analytic chord, the evaluated chord ends moving by `E` (the allowance's `3 E`, round 3) |
| local frame `u'`, `v'` | one rotation by the stored start | no absolute angle formed; rounding `eps (|u| + |v|)` |
| in-plane matrix (Gram-Schmidt) | `|u'|`, `v'.e1`, and the small diagonal as the residual's NORM `|v' - (v'.e1) e1|` (round 6: it was `v'.e2`, `e2` the normalised residual) | absolute error `eps |M|`, a relative `eps x aspect` of the small diagonal: tangent-angle error `eps x aspect <= 2.2e-12` rad by the aspect bound. `v'.e2` added `(v'.e1)(e1.rest) / |rest|`, `e1.rest` being the residual's own nonorthogonality `eps |v'|`: a relative `eps x aspect^2` (round 6: 1e-5 rad over the angle at aspect 4e5, stored angle 1.7) |
| projections onto a nearly dependent frame (round-6 audit) | the parallel-axes test (the residual's norm against `_DEGENERATE |v|`); the in-plane matrix (above); every other projection is onto the module's orthonormal frame about the support axis (`_axis_frame`, round 10; before, the support's stored radial and circumferential directions), never onto a frame built from the curve | the residual's norm is the only quantity recovered from a nearly dependent pair; no dot product with a normalised residual remains |
| aspect bound comparison | computed `aspect x theta* <= 1e4 (1 + 4 eps)` (`_aspect_admitted`), curve and projection | the bound is inclusive: an aspect computed from exactly orthogonal axes at the bound rounds by a few eps (round 6); a skew frame's computed aspect is off by a relative `eps x aspect <= 2.2e-12`, immaterial to every term it scales |
| ellipse sag | `2 sin^2(delta / 4) |u x v| / |T(m)|`, raised by `(1 + 8 eps x aspect)` before comparing | `|u x v|` and `|T(m)|` of a skew frame round by a relative `eps x aspect` each; the raise makes the comparison fail closed; `|T(m)| = 0` is refused; the evaluated chord ends move by `E` (the allowance's `3 E`, round 3) |
| parameters `s / sweep` | direct | `\|sweep\| >= MIN_DISPLAY_SWEEP` (round 8; `sweep = 0` refused before) |
| union merge proposal | `|angle_D(k) - angle_D(v)| <= max(headroom_D, 64 eps max(1, |angle|))`, `headroom_D = max(T, Z...) - term_D`, in the dropped cut's own division | a proposal only: no bound relies on it (round 7) |
| union verification | per merged interval: `|angle_D(b) - angle_D(a)| + term_D <= angle + allowance.angle` for every division; a circle's `2 rho sin^2(delta / 4) (1 + 8 eps) <= sag (1 + 64 eps)` | the division angles are local to the edge (forward maps of the stability rows above, their error within `term_D`); a failure restores original cuts, ending at the no-merge union |
| sag allowance default | `2 r sin^2(angle / 4)` | no `1 - cos` cancellation |
| bisection | `a + (b - a) / 2`, refused when not strictly inside | float resolution typed |
| combined cap | `1 + sum(n - 1)` over every division's count, before any is built | integers |
| rounding allowance | `display_rounding_allowance` (above), at most 2 % of the angle and sag | every term is a sum or quotient of positive quantities (no cancellation); `r > 0` by the radius bound |
| raw arithmetic | any `ArithmeticError` or `ValueError` from a division | typed `EDGE_UNSUPPORTED` |

## Trust model (owner decision, N2 round 5 (1))

Callers are in-process library code passing ordinary values; the modules (`display_tessellation`
and `_slab_triangulation`) are not a sandbox against hostile objects. What they guarantee is a
TYPED refusal (`GeometryError`, or a `DisplayTrimRefused` with its code) for every input outside
the accepted types, decided by checks against EXACT CONCRETE TYPES only: `type(x) is T` or
`issubclass(type(x), T)` for the module's own classes; never a `numbers` ABC, never `isinstance` on
a caller value, never an attribute of a foreign object (only of an exact `np.ndarray`), never
`str()`, `repr()`, `format` or `__name__` of a foreign object or dtype. Every refusal message is
FIXED text plus, at most, values the module itself produced: counts, indices, lengths and shapes it
computed from exact ndarrays as ints, its own constants, floats it computed (or converted with
`float()` from the model's own fields), and an edge id after it was converted to a Python int and
range checked. No caller object, dtype or unbounded int is ever formatted (owner decision N2 round 6
(1): EVERY formatted integer, model-produced ids included, goes through `_id_text`, which prints it
only if |v| < 10**18 and otherwise the fixed text "<id out of display range>"; 34 sites audited, a
face id of 10**5000 is a typed refusal). Note that form refusals use memory O(number of rings): one capped tuple snapshot of the ring list, at most 8 bytes x `MAX_SLAB_RINGS` (16,666 = 50_000 // 3), about 135 kB (135,355 bytes traced at the cap); domain/finiteness refusals
of an accepted, capped array may make one bounded copy (at most 8 bytes x the cap, about 1.8 MB
measured for the largest capped `float64` `extra_x`). A refusal runs no caller code (Codex N2 r5 M1: a metaclass whose `__name__` raised escaped as a
raw `RuntimeError`, and a dtype with a 2e6-character field name allocated about 6 MB while it was
refused; both are now `TRIM_INVALID` under 100 kB traced). Audit of every raise site (round 5): all
77 `raise` statements at 1ab837b (53 in `display_tessellation.py`, 22 in `_slab_triangulation.py`) and
the three message builders (`_unsupported`, `_too_complex`, `_form_error`); 16 messages formatted a
caller value and are now fixed text: a foreign type name in 11 (8 + 3), a dtype and shape in 2, a model
or arithmetic exception's text in 3 (one of them also a type name, counted once), the refusal code's
`repr` in 1, and two model fields (a stored angle, a cylinder radius) are converted with `float()`
before they are formatted; 2 raise sites are new (a non-str refusal message, an edge id beyond
`2**63`), 79 in all. Every other message already held only module-produced values. A source test
keeps `__name__`, `numbers`, a formatted dtype and exception text out of both modules. A lower-level exception stays attached as the refusal's `__cause__`.

## Refusal codes (`REFUSAL_CODES`)

| Code | Meaning |
|---|---|
| `SUPPORT_UNSUPPORTED` | an incident face's support is not a plane, cylinder or cone, or the face has an explicit parameterization |
| `EDGE_UNSUPPORTED` | the curve is not a line, circle or ellipse; degenerate frame or projection; support azimuth not monotone along the edge or reaching the axis (a line crossing it); unresolvable sag (including a sag rounding allowance above 2 % of the sag allowance); a vertex farther from its curve than the sag allowance; more than 4096 intervals (each division, or their sum); lost precision |
| `OUT_OF_DOMAIN` | outside the declared numeric domain (`DisplayDomainRefused`), including an angle rounding allowance above 2 % of `tol.angle` |
| `TRIM_INVALID`, `TRIM_WINDS_SUPPORT` | face trims: raised by `face_display_mesh` (N3, "Face lift (N3)" below), not by edge sampling |

## Trim ring triangulation (`anygeometry._slab_triangulation`, private, N2)

`triangulate_rings(rings, extra_x=None)` triangulates the region of `rings[0]` minus `rings[1:]`
(each ring closed implicitly; ring orientation is free) in pure NumPy, without Shapely. It is not
wired into a public display mesh by `face_display_mesh` (N3, below).

**Input contract (owner decision, N2 round 1, the recommended option): simple rings.** The outer
ring and every hole are simple polygons: no proper crossing, no self-touch, no zero-width retrace
(repeated consecutive points are allowed and collapse to one vertex; collinear runs are fine). The
holes lie strictly inside the outer ring and are pairwise disjoint: no two boundaries share a point
or an edge, and no hole lies inside another. Anything else is `TRIM_INVALID`. (The first version
described an even-odd rule; it is dropped: validity then depended on the cut list, and
self-touches were refused or accepted by an area comparison, not by the rule.)

**Validity is decided exactly and does not depend on `extra_x`** (owner decision, N2 round 1). The
decision uses only ordinate comparisons on slab lines (vertical lines through every vertex and
every `extra_x`; extra lines only refine the structure, and the predicates are exact, so the
verdict is the same with or without them).

**Cut points are correctly rounded** (owner decision, N2 round 2, the recommended option). A cut
point on the segment from `(xa, ya)` to `(xb, yb)` (canonical: `a` the endpoint of lower x, so
every traversal gives the same bits) at a slab line `x` has the exact fraction
`f* = (x - xa) / (xb - xa)` and ordinate `y* = ya + f* (yb - ya)`; the result stores `RN(f*)` and
`RN(y*)`, the round-to-nearest binary64 values. Method: a float candidate `c` (the plain formula,
then one correction step) is certified when the residual `R = (v* - c)(xb - xa)`, evaluated from
error-free transformations (two-sum, Dekker's two-product; exact for domain values, the products
guarded against underflow: an operand is zero or at least `2^-450`), satisfies
`|R| + E < h (xb - xa)`, `h` half the smaller gap of `c` to its float neighbours and
`E = 16 u (sum |terms| + |R|) + 2^-1060` a rigorous bound on the residual's evaluation error (nine
terms: at most `8.01 u` from the summation, `u` per inexact cross product, `u |R|` from the last
sum, with margin for `E`'s own rounding; the absolute term covers gradual underflow); or when the
residual is exactly zero. Rows it cannot certify (a residual within `E` of a rounding boundary, a
cancelling ordinate near 0, tiny operands) are rounded from exact integers (CPython's integer
true division is correctly rounded, subnormals included). Measured: no row of the timing trims
needs the exact route; 1022 of the 4005 directed hard rows in the tests do.

Correct rounding is monotone, so the exact strict order of the boundary points on a slab line
becomes a NON-STRICT order of the stored ordinates, never a reversal. Therefore:

- a strictly smaller stored ordinate proves a strictly smaller exact one, and only equal stored
  ordinates (ties) are compared exactly (`fractions.Fraction`);
- ties of distinct boundaries are ACCEPTED: the pieces they bound have a vertical side of zero
  height and are exactly degenerate; they are dropped (below). The area they lose is already in the
  stated bound (each cut ordinate's error times half its neighbours' x span), with the error now at
  most half an ulp of the stored ordinate;
- a reversal is unreachable; the check stays as a fail-closed guard (`OUT_OF_DOMAIN`, "rounded
  ordinates reverse their exact order", never seen in the fuzz);
- a fraction that rounds to 0.0 or 1.0 at a strictly interior x is ACCEPTED: the exact fraction is
  then within half an ulp of the end (`|f* - f| <= 2^-54` at 1, at most `2^-1075` at 0); the 2D
  mesh is unaffected (its cut point is the correctly rounded `(x, RN(y*))`), and the fraction is
  provenance metadata only (see "Certified product" below). Both round-1 resolution refusals ("slab too narrow", "feature below display
  resolution") are gone; the two Codex round-2 reproductions (a real N1 face whose right corners
  are one ulp apart, a triangular hole cut one ulp right of its tip) are meshed and pass the exact
  oracle.

The validity checks, on the exact order:

1. on each line, all boundary points (vertices, cut points) in exact order: two distinct points at
   the same exact location are a contact;
2. a vertical segment's two endpoints are adjacent in that order (nothing else touches it, no
   vertical retrace);
3. in each slab the spanning segments keep their order from the left line to the right line (a
   reversal is a proper crossing; equal at both lines is an overlap);
4. walking each slab bottom-up, a hole is entered only inside the outer ring and while no other
   hole is open.

Cost and budgets (owner decisions, N2 rounds 1 and 2): incidence processing is `O(I log I)` in the number `I` of
segment-slab incidences (sorts and linear scans, no pairs), and input preparation is `O(input size)`
(a copy and scan of the capped arrays, including out-of-range `extra_x` values); the exact comparisons run only on
lines with a tie. The preflight counts what is allocated and emitted, before anything is built
(a cheap upper bound, segments x lines, skips the exact count for small trims): `N`, ALL input
points (duplicates and the points of vertical segments included, each also a returned point and
a segment), at most `MAX_SLAB_RING_POINTS = 50000`; `I` (and the number of `extra_x` values) at
most `MAX_SLAB_INCIDENCES = 200000`; and the triangle bound `I + 2 (N + cuts)` (`cuts = I - sloped
segments`; a trapezoid gives `L + R - 2` triangles for `L`, `R` points on its sides, and each
point is inside at most one left and one right side) at most `MAX_SLAB_OUTPUT = 600000`. Above any
of them the trim is `OUT_OF_DOMAIN` ("trim too complex for display"); the caller keeps its old
route. Why these values: the measured ordinary trims are a 400-vertex ring with 20 holes
(`N = 800`, `I = 4.2e3`, 4280 triangles), a plain 400-vertex ring (`N = 400`, `I = 0.8e3`) and a
400-vertex star (`N = 400`, `I = 2.0e4`, 19640 triangles, the densest ordinary trim measured); the
budgets are 60x (`N`), 10x (`I`) and 10x (triangle bound, 5.9e4 for the star) above them. A
subdivided vertical side (Codex round 2: 200002 triangles from 2 incidences) is refused by `N`.
Staggered holes need about `2 m^2` incidences: `m = 314` is accepted (199706 triangles, 0.16 s),
`m >= 315` is refused by the triangle bound (loose by about 3x there). `I` is not `O(n)` in
general (a non-monotone ring can need `O(n^2)`); the budgets bound it.

Method: in each slab the spanning segments, in their exact order, are paired bottom/top, giving
trapezoids whose union is the region. x-monotone single rings take a certified fast path (pure
Python up to 48 vertices, NumPy above) giving the same trapezoids: it returns only when the
correctly rounded chains are strictly apart at every interior line (then the exact chains are
too and the ring is simple); otherwise the general path decides, ties included. Tests force the
general path with the private keyword `_fast=False` and compare.

**Caller error state** (owner decision, N2 round 2): every numeric section runs under a local
`np.errstate(all="ignore")` with explicit finiteness checks after, so results and refusals do not
depend on the caller's `np.seterr`; the contract tests run every call under
`np.errstate(all="raise")`. Audit of the merged N1 module (`display_tessellation.py`) the same
way: all public calls of its edge tests (`edge_display_samples`, `display_rounding_allowance`)
run under `np.errstate(all="raise")` through a test fixture. Round 2 found no escape there; Codex
round 3 found one the fixture did not cover: an ellipse with semiaxes 2 and 1 and a SUBNORMAL
out-of-plane component (`u = (2, 0, 2^-1074)`) samples under the default state, but under
`raise` both public calls raised `FloatingPointError` (underflow in `u / |u|`) in their setup,
before the arithmetic guard. Fix (owner decision, N2 round 3 (4)): both public calls run their
own setup and arithmetic under a local `np.errstate(all="ignore")`; the existing explicit
finiteness and domain guards decide; MODEL calls made from inside (vertex positions, incident
faces, arc frames, `evaluate_edge_many`) run under NumPy's DEFAULT error state, whatever the
caller's, so the model's own state-checked fast paths see the state they were measured under.
Nothing changes under the default state: the N1 corpus (5744 results: samples, points and
allowances, every corpus edge at both steps and both sag settings) is bit-identical before and
after (0 changes). The fixture now also carries the reproduction and six subnormal or
huge-but-in-domain variants (subnormal components in both axes, a circle with a subnormal tilt, a
subnormal centre, a centre near 1e9 with a stored angle of 1e5, a millimetre ellipse with a
subnormal component, a line with subnormal coordinates), each compared bit for bit under
`raise`, `ignore`, `warn` and mixed states against the default. The corpus tests make no public
call. (Building the corpus models under `raise` does fail, in `exact_curves` bounding boxes and
`nurbs`, outside the display module; not changed.)

**Input contract and size caps first** (owner decision, N2 round 4 (1), the recommended option;
it replaces the permissive input handling of rounds 1-3, whose input-form findings recurred in
every review round). The module is private and its only caller will be N3, so the contract is
strict and checked FIRST, in O(number of rings), from types and shapes alone, without touching
coordinate data:

- `rings` is a `list` or `tuple` (exact types, no subclass) of 1 to `MAX_SLAB_RINGS =
  MAX_SLAB_RING_POINTS / 3 = 16666` items; each item is exactly `numpy.ndarray` (`type(a) is
  np.ndarray`: no subclass, so no masked array, `np.matrix`, `np.memmap` or user subclass),
  `dtype == np.float64`, `ndim == 2`, `shape[1] == 2`, `shape[0] >= 3`; the total number of points
  (the sum of the shapes) is at most `MAX_SLAB_RING_POINTS`.
- `extra_x` is `None` or exactly a `numpy.ndarray` with `dtype == np.float64` and `ndim == 1`, at
  most `MAX_SLAB_INCIDENCES` values.
- Any other form is `TRIM_INVALID` ("input must be a list/tuple of float64 (n, 2) ndarrays",
  with the offending item named); above a cap it is `OUT_OF_DOMAIN` ("too complex"). Nothing is
  converted, no caller container other than the outer list or tuple is iterated (it is
  snapshotted, then only `type`, `dtype` and `shape` of its items are read), and `np.asarray` is
  never applied to a foreign object. Lists of points, other dtypes (integers, `float32`, a
  `np.longdouble` wider than binary64, complex, object, strings, booleans, big-endian float64),
  stacked `(k, n, 2)` arrays, ranges, generators, iterators, nested lists, scalars and subclasses
  are all refused by this one rule. Views of float64 arrays (strided, broadcast, Fortran order) are
  admitted and capped by their shapes without being materialised.
- Then the coordinates (float64 already, so nothing can be narrowed): finite and
  `|c| <= MAX_DISPLAY_COORDINATE`, else `OUT_OF_DOMAIN`.

The conversion code of rounds 1-3 (the binary64 round trip, masked-array handling, the
sequence and scalar walks) is deleted; there is one input path. Codex round 4 reproductions, all
refused now with a traced peak allocation under 100 kB (`tracemalloc`): an empty stacked array
`np.empty((100000, 0, 2))` (14 MB traced before), a list of 128 wide-string point arrays (17 MB),
`range(10**40)` as `extra_x` or as the rings (a raw `OverflowError` before), a stacked
`ndarray` subclass whose indexing doubled what is stored (a unit square meshed to area 4), and a
nested `extra_x` list (`[[.25], [.5]]`). The parametrised refusal test also carries every
input-form reproduction of rounds 1-3 (oversized integers, complex with a NaN imaginary part,
integers above `2^53`, generators and iterators, masked arrays and `np.ma.masked`) and other
foreign forms (45 cases). The size caps keep their round-3 test (a broadcast view of 1e6 points,
20000 rings, a broadcast and a strided `extra_x`: under 1 MB traced). Changed results (stated):
inputs the earlier rounds converted are now `TRIM_INVALID`: Python sequences of points, integer,
`float32`, boolean, complex and object arrays (complex, object and boolean were `OUT_OF_DOMAIN`,
integers above `2^53` "not exactly representable" `OUT_OF_DOMAIN`), stacked arrays, scalars and
sequences as `extra_x`, and ndarray subclasses other than masked arrays (read as stored before).

Contract of the result (`SlabTriangulation`):

- `points` are the input vertices in order (rings concatenated, unchanged bits), then *cut points*;
  cut point `k` is `(x, RN(y*))` on the slab line `x` of ring segment `cut_ends[k] = (a, b)`, `a`
  its endpoint of lower x. `cut_fraction[k] = RN(f*)`, in `[0, 1]` (0 or 1 only within half an
  ulp, see above), is the fraction of that point from `a`. `cut_ends` and `cut_fraction` are
  provenance METADATA (which segment, roughly where), not a certified lift (below). No other
  points are created.
- `triangles` are counter-clockwise with strictly positive area, decided from the slab geometry,
  not from a determinant (owner decision, N2 round 1): each triangle has one vertical side between
  two points of one slab line and its third corner on the other line, so its exact area is
  (side height) x (slab width) / 2, and the side height is the difference of two floats, exactly
  signed. A piece is dropped exactly when that height is zero (one shared point, or a tie).
- The triangles tile, exactly in rational arithmetic, the *rounded* polygon (the rings with each cut
  point inserted on its segment): no overlap, nothing outside, every point on a trapezoid's
  vertical side used (no T-junctions). The rounded polygon's area differs from the exact one by at
  most `sum_k |RN(y*_k) - y*_k| (x_next,k - x_prev,k) / 2` over the cut points (moving one by `d`
  changes the area by `d` times half its neighbours' x distance), each error at most half an ulp.
- Coverage (owner decision, N2 round 5 (2), the recommended option): the 2D mesh covers exactly the
  CLOSURE OF THE POSITIVE-AREA INTERIOR of the rounded trim. Zero-width parts of
  the rounded trim that are not in the closure of the positive-area interior (a hole edge and an
  outer edge rounded onto the same segment) are omitted and invisible in display; points IN that closure (for example the round-4
  hole-tip tie, where a slab line meets a boundary at a point) are covered and may appear as vertices.
  N3 must not assume every trim boundary segment is a triangle edge.
  Codex round 5 reproduction (kept as a test): outer ring `(0, nextafter(1, 0)), (4, 1), (4, 3),
  (0, 3)`, hole `(2.5, 1), (3.5, 1), (3.5, 2), (2.5, 2)`: the outer edge's cuts at `x = 2.5` and `3.5`
  round onto the hole's bottom corners, the exact oracle passes, and no triangle covers the interior
  of the coincident segment (for example `(3, 1)`). Area accounting: the zero-width part has zero
  area, so the triangles' exact area is the rounded polygon's area (outer minus hole, with the cut
  points inserted), as for every accepted mesh.
- `area` is the EXACT area of the emitted triangles, correctly rounded once (owner decision, N2
  round 4 (3), the recommended option): `RN(A)` with `2 A = sum (xr - xl) ((ytl - ybl) + (ytr -
  ybr))` over the trapezoids (the triangles tile them exactly; a dropped piece has zero area), every
  value a stored float. Up to 24 trapezoids it is computed from exact integers (a common
  power-of-two denominator, one correctly rounded integer division); above, by a certified float
  evaluation (error-free two-sum and two-product of the main terms, an exact split-and-sum of the
  products, a rigorous bound on the small rest), which returns only when the bound proves the
  rounding (else the integers decide). The value is the same bits on the fast and the general path
  (round 3 left 23 of 6000 trims differing in the last bits). `area` is the correctly rounded exact
  area, which may round to 0 for subnormal-scale trims (owner decision, N2 round 5 (4); Codex round 5:
  a square of side `2^-538` is accepted on both paths with two strictly positive triangles, and its
  exact area `2^-1076` rounds to `0.0`; kept as a test); it is never zeroed by the computation
  itself: Codex round 4 showed a rectangle with `s = 2^-537`, width `nextafter(s, inf)` and height `s / 2`, exact area
  rounding to `5e-324`, reported 0.0 with cuts at `[0.09 s, 0.34 s]`; now `5e-324` on both paths, with
  and without cuts (round 3's `2^-536` square stays `2^-1072`). The exact oracle now checks
  `area == float(Fraction(exact triangle sum))` for every accepted mesh. Changed results: `area`
  moves by at most a few ulps where the plain float sum was not correctly rounded (measured below);
  points, triangles and cut data are unchanged.
- The result is deterministic (bit-identical on repeat).

**Certified product** (owner decisions, N2 rounds 3 and 8). The certified product of N2 is the 2D
mesh: the vertices and triangles in the trim plane, with the guarantees above. `cut_ends` and
`cut_fraction` are provenance metadata only. Lifting this mesh onto a support is N3's contract; N3
must not rely on cut_fraction reconstruction for orientation or coverage (see the two documented
counterexamples). Counterexample 1 (kept as a test): outer ring `(-1, -1), (1, -1), (1, 3), (-1, 3)`,
hole `(-0.21, 1), (0.75, 0.75), (0.75, 1.25)`, `extra_x = [nextafter(-0.21, -inf)]`: the 2D mesh passes
the exact oracle, but rebuilding its cut points at `cut_fraction` inverts triangle `[7, 4, 10]`.
Counterexample 2 (kept as a test): outer ring `(0, 0), (3, yb), (3, 4), (0, 4)` with `yb = 0.1 * 3`,
hole `(0.5, 0.5), (1, tip), (2, 1), (1, 2)` with `tip = float(Fraction(yb) / 3)`: the outer cut at
`x = 1` rounds onto the hole tip; the fraction rebuild separates the two points, every triangle stays
counter-clockwise, yet a point the 2D mesh covers lies in no lifted triangle.

Refusals, in this order (no raw `IndexError`/`ValueError`/`OverflowError`/`FloatingPointError`):

| code | when |
|---|---|
| `TRIM_INVALID` / `OUT_OF_DOMAIN` | the input contract and size caps, from types and shapes only, before any data is read (owner decision, N2 round 4): anything but a list/tuple of 1 to `MAX_SLAB_RINGS` exact float64 `(n, 2)` ndarrays with `n >= 3` and an `extra_x` of None or an exact float64 1-D ndarray (`TRIM_INVALID`, "input must be a list/tuple of float64 (n, 2) ndarrays"); more than `MAX_SLAB_RINGS` rings, `MAX_SLAB_RING_POINTS` points or `MAX_SLAB_INCIDENCES` extra lines (`OUT_OF_DOMAIN`, "too complex") |
| `TRIM_INVALID` (`SlabInvalid`) | fewer than three distinct consecutive vertices |
| `OUT_OF_DOMAIN` (`DisplayDomainRefused`, as N1) | the declared numeric domain, checked before any arithmetic: finite, `\|c\| <= MAX_DISPLAY_COORDINATE = 1e9` for rings and `extra_x` (float64 by the contract, so nothing is converted or narrowed); then the incidence and triangle budgets. Inside the domain no overflow is possible: differences are at most 2e9, shoelace products at most 4e18 |
| `TRIM_INVALID` | the input contract above (crossing, self-touch, retrace, touching rings, hole outside or nested), decided exactly |
| `OUT_OF_DOMAIN` | a ring whose EXACT area is at most `16 n eps extent^2` of ITS OWN extent (the float value of that expression; one rule on both paths, owner decision N2 round 2: a float shoelace filter with a rigorous bound, exact integers where it cannot decide; before, the fast path tested its slab sum and the general path a cancelling float shoelace, and they disagreed on a ring at the floor); the reversal guard (unreachable) |

The tests carry an independent exact oracle (`tests/test_display_tessellation_slab_contract.py`):
exact orientation (Shewchuk's filter with an absolute underflow allowance, then rationals: a
subnormal collinear triple used to pass the filter as non-zero), exact pairwise
interior-disjointness (bounding-box prefilter, no cap), exact containment of corners, edge
midpoints (closed) and centroids (strict) in the rounded polygon, exact equality of the triangle
area sum with the rounded polygon's area, the stated bound against the exact polygon (each cut
charged once, found by its segment `cut_ends[k]` and index `k` in its own ring; round 2 matched
cuts by rounded coordinates across rings, so an outer cut rounded onto a hole vertex was charged
to both rings: +25 % on the round-3 test case), every cut
fraction and ordinate equal to `float(Fraction)` of the exact value, and a brute-force exact
validity verdict.

Fuzz (N2 round 1, 6000 cases: `display-tess-evidence-20261125/n2r1_fuzz.py`; integer-grid, star,
snapped and one-ulp-nudged star, and x-monotone generators, half with random extra lines, some one
ulp from an end; each case also run with the general path forced): 0 raw exceptions; 0 exact-oracle
failures among the 2810 accepted; the brute-force exact verdict matched on all 6000 (2957 invalid,
all `TRIM_INVALID`); fast and general paths gave the same verdict and the same triangles; the
verdict never changed with `extra_x`. 233 valid cases were `OUT_OF_DOMAIN` (190 feature below
display resolution, 39 slab too narrow, 4 area floor). Round 2, the same 6000 cases (`n2r2_fuzz.py`,
same seed, the stricter oracle): 0 raw exceptions; 0 exact-oracle failures among the 3039 accepted;
the brute-force verdict matched on all 6000 (2957 invalid, all `TRIM_INVALID`); fast and general
paths agreed (verdict and triangles); the verdict never changed with `extra_x`; the reversal guard
never fired. Of the 233 valid cases refused in round 1, 229 are now accepted; the remaining 4
`OUT_OF_DOMAIN` are the area floor (as before). Round 3, the same 6000 cases and oracle (per-cut area
allowance): identical counts (3039 accepted, 0 oracle failures, 2957 `TRIM_INVALID`, 4
`OUT_OF_DOMAIN` area floor, 0 raw exceptions). Ordinary results unchanged: 3000 seeded star trims
with holes and extra lines (6000 calls, both paths; 5284 accepted) give bit-identical points,
triangles, cut data and areas before and after round 3. Round 4 (`n2r4_fuzz.py`: the same 6000 cases passed as float64 ndarrays, the oracle now also
checking `area == RN(exact area)`): identical counts (3039 accepted, 0 oracle failures, 2957
`TRIM_INVALID`, 4 `OUT_OF_DOMAIN` area floor, 0 raw exceptions). Ordinary results (`n2r4_bits.py`,
3000 seeded star trims with holes and extra lines, both paths, 6000 calls, 5614 accepted): 0 verdict
changes, points, triangles and cut data bit-identical; `area` changed in 2543 calls by at most 3
ulps (the plain float sum was not correctly rounded there); fast and general paths now agree on
every area (15 differed at c59accd).

Measured (best of 30 or 300 runs, loaded PC, round 1 / round 2): a 400-vertex ring with 20 holes
2.13 / 2.41 ms, a plain 400-vertex ring 0.21 / 0.41 ms, a 400-vertex star 5.1 / 7.8 ms; a unit
square 43 / 48 us, the square with 9 lines 61 / 85 us, a 24-vertex monotone ring 64 / 71 us, a
square with one hole 0.24 / 0.22 ms (same session, the round-1 code from `git archive dd5061c`). Round 3
(76c07d2 / round 3, same session, better of two passes): 2.54 / 2.51 ms, 0.42 / 0.43 ms, 7.6 /
7.1 ms; 48 / 51 us, 87 / 90 us, 73 / 77 us, 0.22 / 0.23 ms (the shape preflight costs about 3 us
per call). The
certified rounding costs about 0.15 ms of NumPy overhead per call above 64 cut points (below that,
exact integers are cheaper) and about 0.13 us per cut point. Round 4 (c59accd / round 4, same session, best of
three alternating passes, `n2r4_bench.py`): 2.52 / 2.57 ms, 0.43 / 0.47 ms, 8.3 / 8.5 ms; 51 / 51 us,
89 / 90 us, 76 / 86 us, 232 / 224 us. The exact area costs about 40 us of NumPy overhead above 24
trapezoids (certified float route) and about 1.7 us per trapezoid below (integers); the strict
input check is cheaper than the conversion it replaces.

Round 5 (`n2r4_fuzz.py` rerun, the same 6000 cases and oracle): identical counts (3039 accepted, 0
oracle failures, 2957 `TRIM_INVALID`, 4 `OUT_OF_DOMAIN` area floor, 0 raw exceptions). N1 corpus
(`n2r4_corpus.py`, 5744 results, 1ab837b against round 5): 0 sample changes and 0 allowance changes,
refusal codes and messages included. Timing (`n2r4_bench.py`, best of three alternating passes,
1ab837b / round 5): 2.53 / 2.57 ms, 0.475 / 0.473 ms, 8.9 / 8.2 ms; 49 / 50 us, 89 / 90 us, 84 / 83
us, 223 / 223 us (unchanged within the loaded PC's noise; only refusal text changed).

## Face lift (N3)

`face_display_mesh(model, face_id, tol) -> FaceDisplayMesh` (public in
`anygeometry.display_tessellation`, not re-exported from `anygeometry`; implementation in the private
`anygeometry._display_lift`) returns a certified triangle mesh of one face whose support is a
`Plane`, `Cylinder` or `Cone` and whose loops are edges N1 draws (lines, circles, ellipses), or a
typed `DisplayTrimRefused`, never a partial mesh (owner decisions 2026-11-26 on SPEC-N3 Q1-Q9;
slices N3a plane, N3b cylinder, N3c cone). Same trust model as N1 and N2: exact argument types,
the module's arithmetic under `np.errstate(all="ignore")` with explicit guards, model calls through
N1, no raw arithmetic error escapes, and the result does not depend on the caller's `np.seterr`.
Nothing in it runs in a GUI thread by itself.

**Steps.** (1) Support check. (2) Rings: the N1 samples of each coedge (reversed for a reversed
coedge, last point dropped); consecutive coedges must meet bit for bit. (3) Chart of every ring
point: on a plane its coordinates in an orthonormal frame of the plane; on a cylinder or cone
`(theta, h)` about the module's own axis frame (`_axis_frame`), theta unwrapped along each ring
(N1 bounds every azimuth step by the angle plus 2 %), holes shifted by multiples of 2 pi onto the
outer ring's sheet. (4) Seam, winding and apex refusals. (5) Boundary-chord certificate against
T_face. (6)-(7) N2 (`triangulate_rings`), refined by slab bisection where needed (cylinder,
cone). (8) Lift by index: ring vertices are the N1 points; a cut point (a point N2 makes on a ring
chord) is ON the 3D N1 chord, `A + f (B - A)` (`cut_fraction` is provenance only: no claim is
derived from it). (9) Combinatorial, orientation and deviation certificates.

**`FaceDisplayMesh`.** `points` (n, 3): first the ring vertices, the N1 `EdgeSamples` points bit
for bit (outer loop, then each hole), then the cut points; `triangles` (m, 3) int64,
counter-clockwise about the face normal `sense x` the support normal; `chart` (n, 2) the chart
points; `ring_vertex_count`; `source` (n, 2) `(edge id, interval index)` of the N1 chord each point
lies on; `sense` the sign of the outer ring's exact chart area; `face_tolerance` = T_face;
`deviation_bound` (certified, `<= face_tolerance`); `trim_sag_bound`; `degenerate_count` (lifted
triangles with an exactly zero 3D normal, dropped from `triangles`: R3 C3).

### Contract

- **Domain and map (Q3).** The domain is the closure of the positive-area interior of the chart
  polygon of the N1 samples (as N2 defines it); the map is the index-wise lift, linear per
  triangle. Ring vertices are N1's points bit for bit; cut points lie on N1 chords to rounding.
- **Coverage (Q2), certified combinatorially and exactly:** (i) the index-to-chart map is
  injective; (ii) every directed triangle edge occurs once, and an edge without its reverse lies on
  one ring chord; (iii) those boundary edges are exactly the chain of every ring (its vertices with
  each chord's cut points in order, directed so the region is on the left); (iv) every chart
  triangle is strictly counter-clockwise (Shewchuk's filter, then rationals). With all triangles
  positive, the 2-chain's boundary is the rings' chain, so every point off the edges is covered
  exactly by its winding number: a conforming, non-overlapping covering of the chart region.
  The certificate is about the chart; it carries over to the support only where the chart map is
  injective (R2 H2, "Injectivity" below): on a cylinder or cone the unwrapped azimuth extent of
  the rings must be certifiably below 2 pi (`TRIM_WINDS_SUPPORT` otherwise); a plane's affine
  chart is injective by its non-degenerate frame (sine at least 1e-12, else `OUT_OF_DOMAIN`).
- **Paired deviation (Q4).** A mesh point at barycentric weights `w` of a triangle corresponds to
  the support point at the same weights of the chart vertices (parametric correspondence, not the
  radial distance: on the r6 frustum the radial distance 0.2929 passes a 0.30 limit, the paired
  0.3800 does not). `deviation_bound` bounds it over every triangle in closed form, never sampled.
- **T_face (Q5, owner decision).** `tol.sag` if given; else on a plane
  `model.tolerance.surface_residual`; else `r_max (tol.angle + a)^2 / 8 + surface_residual` on a
  cylinder (`r_max` its radius) or a cone (`r_max` the larger of its two radii), `a` the largest N1
  `allowance.angle` of the face's edges. Vertex offsets count toward it. The default admits every
  N1 circle chord on its own support (Taylor bound `r dT^2 / 8`, `dT <= angle + a`) as long as the
  rounding allowance `A_face` stays within the slack (see the large-radius note).
- **Analytic trim (Q6): approximated, not covered.** The mesh covers the polygon of the N1 chords,
  not the analytic trim region. `trim_sag_bound` certifies the largest distance of the drawn
  boundary chords from the analytic curves (N1's exact chord sag, its rounding and the vertex
  offsets), plus every boundary point N3 creates: the cut points' own rounding (R1 M2); the D1
  identifications move no lifted point (R3 C1: bitwise equal), so they add nothing; always reported, enforced (`EDGE_UNSUPPORTED`)
  only when `tol.sag` is given, as `sag (1 + 64 eps)` plus the offsets N1 treats as rounding
  (decision D3) on N1's chords (N3's own rounding terms are reported on top). A plane with a circular
  hole is accepted at the default tolerance with `trim_sag_bound >= r (1 - cos(angle / 2))`.
- **T-vertices (Q8): crack-free to rounding, not vertex-shared.** A cut point on a boundary chord is
  not inserted into the neighbour's samples; it lies on the shared 3D N1 chord (to the rounding of
  `A + f (B - A)`, at most `2 eps (|A_i| + |B_i|)` per coordinate; `4 eps |(|A| + |B|)|` is counted),
  so neighbouring faces meet without cracks above rounding, with T-vertices. N1 samples are never
  moved and no other face is edited. **Identifications are exact (R3 C1, owner decision
  2026-11-27, replacing R1/R2 H1):** a D1 merge identifies two cut points only when their lifted
  3D points are bitwise equal, and a chain edge cancels only when both of its lifted ends are
  bitwise equal (`L = 0` exactly); otherwise `TRIM_INVALID`. No allowance enters, so the mesh
  boundary is the lifted chain of the ring chords point for point: no identification opens a gap.
- **Orientation (Q7), no margin (R3 C3, owner decision 2026-11-27).** Every published triangle is
  CERTIFIABLY positively oriented against `sense x` the support normal over its WHOLE chart
  footprint (not at its centroid: codex-n2r8's 2e-4-sweep fixture passes the centroid test and
  folds elsewhere). A float filter certifies most triangles (its bound is the filter's own error
  budget, never a margin that admits anything: plane `(16 eps + 2 alpha) |d1| |d2|`, cylinder
  `40 eps |d1| |d2|`, cone `40 eps (1 + |k|) |d1| |d2|`, about three times the derived error, which
  assumes libm's cosine and sine within a few ulps). Every triangle it cannot certify is decided
  from its EXACT normal (integers over the stored points' common power-of-two scale; R1 M1): on
  a plane a second, tight filter and then exactly against `u x v` (exact); on a cylinder or cone
  by an ENCLOSURE (below, "Exact fallback on curved supports"): kept only if certifiably positive.
  A triangle whose exact normal is zero (zero area) is DROPPED from `triangles` and counted in
  `degenerate_count`; anything else (negative or undecided) is a fold (`TRIM_INVALID`), whatever
  its width or world position. **Dropping keeps the certificates:** the three lifted points of a
  zero-normal triangle are collinear (or coincide), so its point set is its longest edge, which
  any two of its edges cover; it is dropped only if that set provably lies in the kept
  triangles: an edge is covered when its reverse is in a kept triangle or in a dropped triangle
  already shown covered, a dropped triangle when two of its edges, or one longest edge (exact
  squared lengths), are covered, iterated to a fixpoint (an ulp slab gives a strip of zero-area
  triangles along one chord); else `TRIM_INVALID` ("a zero-area lifted triangle is not covered by
  the triangles kept beside it"). Then the kept triangles contain every dropped one as point
  sets, so the lifted surface and the lifted boundary chain are unchanged as sets (a dropped
  chain edge lies in the kept mesh: no gap at a shared trim); the
  chart covering (Q2) is certified on the full chart mesh, and `deviation_bound` is taken over all
  of it, so the support over a dropped triangle's footprint is within `deviation_bound` of the kept
  mesh. The published mesh then has unpaired edges along each dropped triangle (overlapping in
  3D, zero width).

### Bounds per support

`eps = 2^-52`, `FE = _FRAME_ERROR` (the module frame's distance from an exactly orthonormal one).
Common lemma: for the exact chart map `S*`, a mesh point `x = sum W_j X_j` with chart
`c = sum W_j c_j + kappa` (`W` convex weights over the ring vertices `j` that the triangle's
vertices stand for: a cut at fraction `f` of chord `AB` stands for `A` and `B` with weights
`1 - f`, `f`; `kappa` the computed distance of a cut's chart point from that interpolation),

    |x - S*(c)| <= max_j |X_j - S*(c_j)| + |sum W_j S*(c_j) - S*(cbar)| + Lip |kappa| + 16 eps max|X|,

and `|X_j - S*(c_j)| <= delta_j (1 + 8 eps) + A_face` with `delta_j = |X_j - S(c_j)|` computed.

**Plane (decision D2).** The chart is measured through an exactly orthonormal chart: with the
module's frame `e1 = u / |u|`, `n = (u x v) / |u x v|`, `e2 = n x e1` and `S*` built on the exact
frame, an isometric affine chart, `|sum w_i (X_i - S*(c_i))| <= max_i |X_i - S*(c_i)|` (no
curvature term). `A_face = 16 eps (|o| + max|X| + H) + 2 alpha max|c|`, `alpha = 8 eps / s + 16 eps`
(the frame's tilt; `s` the sine between `u` and `v`, `s < 1e-12` is `OUT_OF_DOMAIN`), `H` the
largest coordinate extent. `deviation_bound = max over triangles of max_i delta_i (1 + 8 eps) +
A_face`. Orientation: `nT . (sense n) > (16 eps + 2 alpha) |d1| |d2|`, once per triangle.
Where that bound exceeds `T_face` at a point (R1 M3: `|o|` and `max|X|` are world magnitudes, so
an exactly planar triangle on `z = 2e6` failed `T_face = 1e-8` by rounding alone), the point's
bound is the smaller of it and the origin-relative bound `delta_loc (1 + 8 eps) + A_loc`,
`delta_loc = |w - c_x e1 - c_y e2|`, `w = X - o` (rounded by half an ulp of its own result),
`A_loc = 16 eps (max|w| + max|c|) + 2 alpha max|c|`; a point still over is decided exactly:
`D^2 = |w|^2 + |c|^2 - 2 c_x (w.u) / sqrt(u.u) - 2 c_y (w.v') / sqrt(v'.v')` (`v' = v - (v.u /
u.u) u`, the exactly orthonormal frame) in rationals with both square roots enclosed to `2^-128`,
`2^-512`, `2^-2048` relative; proven `D <= T_face` accepts (bound `<= T_face`), proven over or
undecided refuses (`EDGE_UNSUPPORTED` at a ring vertex, `TRIM_INVALID` at a cut). A face the
first bound accepts is unchanged bit for bit.

**Cylinder.** `S*(theta, h) = o + r c(theta) + h a*`, `c = cos e1* + sin e2*`;
`A_face = 32 eps (|o| + r + H + max|X|) + 2 FE (r + H)`, `H = max |h|`. `S*_tt = -r c`, every
other second derivative 0, so the middle term is at most `(r / 2) Var_W(theta)`. By the law of
total variance `Var_W(theta) = sum_i w_i V_i + Var_w(m_i)`, `V_i = f_i (1 - f_i) (theta_B -
theta_A)^2` (0 for a ring vertex), `m_i` within `kappa_t` of `theta_i`. The maximum over the
triangle of `G(w) = sum w_i V_i + Var_w(theta_i)` lies on an edge (`G` depends on `w` through the
image point `(sum w_i (V_i + theta_i^2), sum w_i theta_i)`; for each second coordinate the largest
first one lies on the image triangle's boundary), and on the edge `p, q` (`tau = theta_q -
theta_p`) `G = (1 - l) V_p + l V_q + l (1 - l) tau^2` has the exact maximum `U = max(V_p, V_q) +
max(tau^2 - |V_q - V_p|, 0)^2 / (4 tau^2)`. Per triangle

    E(T) = [max_j delta_j (1 + 8 eps) + (r / 2) (U (1 + 32 eps) + 8 eps (dT^2 + max V))
            + A_face + 16 eps max|X| + r k_t + k_h + (r / 2) (2 k_t dT_max + k_t^2)] (1 + 8 eps).

Boundary chords (two ring vertices): `max delta (1 + 8 eps) + r dT^2 / 8 (1 + 8 eps) + A_face <=
T_face`, else `EDGE_UNSUPPORTED` (a fixed N1 chord cannot be refined inside). Where a chord or a
triangle exceeds T_face with this bound, the origin-relative variant decides as on a plane (R1, M3
class): offsets `|(X - o) - S_loc(c)|`, `S_loc = rho c(theta) + h a`, and `A_loc = 32 eps (r + H +
max|X - o|) + 2 FE (r + H)`, the cuts' `16 eps max|X|` replaced by their chords' `cut_rounding`
(the same on a cone with `rho_max` and `rho`'s rounding); a face the first bound accepts without
refinement is unchanged bit for bit. Orientation:
`n . c(theta) = R cos(theta - phi)`, whose set above a margin `m >= 0` is an arc shorter than pi,
so on a footprint narrower than pi the minimum exceeds `m` iff both end values do: filtered as
`sense n . c(theta_end) > 40 eps |d1| |d2|` at both ends (the filter's error bound, about 13 eps
derived: rounded differences and cross product 2.5, dots 2, frame `sqrt(2) FE` 5.7, cosine, sine
and sums 2.5; it certifies, never exempts); a footprint of pi or more is refused; the rest by the
exact enclosure (R3 C3, "Codex round 3").

**Decision E1.** The bound above is SPEC-N3's Taylor/Popoviciu bound taken over the ring vertices
each cut stands for, with the exact maximum on each triangle edge. It replaces the spec's
`E(T) = max_i delta_i + r dT^2 / 8`, which counted a cut's chord sag twice (the cut's own offset
already holds it): 288 of the 2663 tripod triangles exceeded T_face at 7.5 deg with that form, none
with this one.

**Cone (N3c).** `S*(theta, h) = o + rho(h) c(theta) + h a*`, `rho(h) = r0 + k h`,
`k = (r1 - r0) / (height |axis|)` (the support's `evaluate` puts `v height axis` at radius
`(1 - v) r0 + v r1`; `height` may be negative). `A_face = 32 eps (|o| + rho_max + H + max|X|) +
2 FE (rho_max + H) + 8 eps (|r0| + |k| H)` (the last term the rounding of `rho`, `k` included).
Since `rho` is linear in `h`, `sum W_j rho_j = rho(hbar)` and exactly

    sum W_j S*(c_j) - S*(cbar) = sum W_j rho_j (c(theta_j) - c(thetabar))
                               = k Cov_W(theta, h) c'(thetabar) + sum W_j rho_j R_j,   |R_j| <= dt_j^2 / 2,

so the middle term is at most `(1 / 2) sum W_j rho_j dt_j^2 + |k| |Cov_W(theta, h)|` (`rho_j > 0`
by the apex rule). This is SPEC-N3's `B_cone = rho_max dT^2 / 8 + |k| dT dH / 4` (Cauchy-Schwarz
and Popoviciu give `|Cov_w| <= dT dH / 4` for a triangle's own vertices), taken in a form that does
not add a circle chord's variance to the mixed term at a different weight: by the laws of total
variance and covariance, with each cut's own chord (`V_i`, `C_i = f (1 - f) dtheta dh`, 0 on a
circle) and the cross term `|k| C_i |m_i - thetabar|` (rho varies along a skew chord),

    middle term <= H(w) = sum_i w_i P_i + sum_{i<j} w_i w_j D_ij,
    P_i = (rho_i / 2) V_i + |k| C_i (1 + dT + 2 k_t),    D_ij = (rho_T / 2) tau_ij^2 + |k| |tau_ij dh_ij|

(`rho_i` the larger radius of a cut's chord ends, `rho_T` the largest over the triangle; the
triangle vertices' own variance and covariance are `sum_{i<j} w_i w_j tau_ij^2` and
`sum_{i<j} w_i w_j tau_ij dh_ij`). The maximum of `H` over the triangle is taken exactly: on each
edge `max(P_p, P_q) + max(D - |P_q - P_p|, 0)^2 / (4 D)`; where `H` is strictly concave on the
plane (`D01 > 0`, `det = 4 D01 D02 - (D01 + D02 - D12)^2` above its rounding and above
`1e-6 (D01 + D02)^2`, so the condition number is at most 4e6) and its critical point lies in the
triangle (to 1e-3), also the critical value (a concave quadratic's global maximum, an
upper bound wherever it lies); where it is concave only to rounding, the boundary maximum plus
`(det + its rounding) / (D01 + D02)` (along the flattest direction the boundary is within that).
Per triangle

    E(T) = [max_j delta_j (1 + 8 eps) + H_max (1 + 64 eps) + 32 eps (sum D + max P)
            + ((rho_T / 2)(2 k_t dT_max + k_t^2) + |k| ((k_h dT + k_t dH) / 2 + k_t k_h)) (1 + 8 eps)
            + A_face + 16 eps max|X| + rho_max k_t + (1 + |k|) k_h] (1 + 8 eps)

(`|dS/dtheta| = rho`, `|dS/dh| = |k c + a| <= 1 + |k|` for the `kappa` shift). Boundary chords:
`max delta (1 + 8 eps) + (rho dT^2 / 8 + |k| dT dH / 4)(1 + 8 eps) + A_face <= T_face`, else
`EDGE_UNSUPPORTED`. On the r6 frustum's naive two-triangle mesh `B_cone = 0.3084 + 0.3888 = 0.697`
bounds the oracle's 0.3800 (the curvature term alone, 0.3084, would not). Orientation:
`S_theta x S_h = rho (c(theta) - k a)`, so the sign of `f(theta) = sense n . c(theta) - sense k
n . a = R cos(theta - phi) + level` decides; its minimum over a footprint narrower than pi is at
an end unless the sinusoid's minimum lies inside, which needs a falling slope at the low end and
a rising one at the high end; "possibly inside" within the filter bound is taken as inside, and
then `level - R` must exceed the bound too. Filter bound `40 eps (1 + |k|) |d1| |d2|` (the
cylinder's budget plus about 17 eps |k|: the level's dot and product, `|k| FE` and the slope's own
rounding); the rest by the exact enclosure (R3 C3).

**Refinement (decision E3).** Every slab holding a triangle with `E(T) > T_face` gets an N2 extra
slab line at its middle; N2 runs again and every certificate is re-run on the new mesh, at most
`_REFINE_ROUNDS = 8` rounds (2^8 sub-slabs per slab). Still failing: `TRIM_INVALID` ("a lifted
triangle's deviation bound exceeds the face tolerance"); a slab too narrow to bisect likewise; more
than `MAX_SLAB_INCIDENCES` lines: `OUT_OF_DOMAIN` ("too complex"). Bisection rather than one
uniform step: as `tau` shrinks the edge maximum falls to `max(P_p, P_q)` as soon as
`D <= |P_q - P_p|`, so a staggered cut near a chord's middle usually needs one or two halvings. A
face needing none is N2's mesh itself. On a cone the mixed term needs slabs about
`4 (T_face - chord sag) / (|k| dH)` wide: the r6 frustum (`k = 0.99`, `dH = 1`) takes 5-8 rounds at
0.05-0.3 rad and is refused after 8 at 0.02 rad. Test: with `tol.sag` exactly the worst
boundary-chord bound, 8 rounds then `TRIM_INVALID`.

**Decision E2 (azimuth join).** Chart azimuths closer than their own rounding,
`16 eps (max(1, |theta|) + reach / r)` (on a cone `r` is the smallest `rho` at the ring points;
`reach` = `max |P - o|` plus the stored coordinates' representation projected onto the radial
plane, `sum_i (|P_i| + |o_i|) (|e1_i| + |e2_i|)`, R2 M3; it was the largest world coordinate plus
the origin's, so an axial translation to 2e6 joined two ring vertices 1e-8 rad apart), are joined
to the smallest:
the two ends of a ruling, or a hole's ruling against another ring's vertex, otherwise make
ulp-wide slabs whose cuts on a long chord stand tangentially off the chart order (a true sliver
fold, 3D width ~1e-3). The join changes chart values only: every ring vertex still lifts to its N1
sample bit for bit (tested), the offsets `delta` measure the change, and a join that would make
two distinct N1 samples one chart point is `TRIM_INVALID` ("two ring points coincide in the
chart", tested).

**Decision D1 (aligned holes).** Two or more CUT points whose chart points are bitwise equal become
one mesh vertex before the certificates (the boundary chain is then a 1-chain whose zero-width
spikes cancel); a chart point shared with a ring vertex stays `TRIM_INVALID`. **R1 H1:** the
identification must also hold in 3D. Let `d` be the largest 3D distance between a merged cut and
its kept representative and `L` the total 3D length of the chord pieces whose chain edges cancel
(a cancelled piece is not on the mesh boundary; every point of it lies within `L` of a node that
is, a cancelled run along a ring ending at nodes of that ring's remaining chain; a ring whose
whole chain cancels is `TRIM_INVALID`). R1/R2 bounded the gap by `d + L`, added to
`trim_sag_bound` and limited to T_face (R1) then `A_face` (R2); R3 found `A_face` itself inflated
to 15646 by an ill-conditioned plane basis (a 0.125 gap accepted). **R3 C1 (owner decision,
replacing both):** `d = 0` and `L = 0` exactly, i.e. the merged cuts' lifted points bitwise equal
and both lifted ends of every cancelled chain edge bitwise equal, else `TRIM_INVALID` ("merged cut
points, or the ends of a boundary interval their merge cancels, are not bitwise equal in 3D (lift
not certified: a gap in the trim)"); `trim_sag_bound` carries no identification term. Codex's fixture (outer and hole chords ulps apart in the chart, 8e-9 apart in 3D, the
interval x = 3..3.25 cancelled) is refused at every sag. E2 joins move chart azimuths only, never a 3D point (the
offsets measure them), so they identify no vertices. **D2:** the plane
deviation is measured directly through an exactly orthonormal chart (above). **D3:** the trim-sag
check `sag (1 + 64 eps)` plus N1's rounding offsets (above). **D4:** an arithmetic error in the
lift is `TRIM_INVALID` with fixed text; a seam (one edge used twice) is `TRIM_WINDS_SUPPORT`; the
tests build trims the model would refuse with `_put_entity`.

**Domain (no new constant).** A cone trim reaching the apex (a ring point within N1's apex rule of
an exactly zero radius, or `rho(h)` at most its rounding at a ring point) is `OUT_OF_DOMAIN`; otherwise `rho > 0`
on the whole chart region (`rho` is linear in `h`, every mesh point's `h` lies between ring
points'). Small radii are bounded by N1's `MIN_DISPLAY_RADIUS_RATIO`, large ones by the slab
budget. Large-radius note (N3c finding): at radius 5e8 `A_face` (about 9e-6) exceeds the Q5
default's slack (`surface_residual = 1e-8` plus `r (2 angle a + a^2) / 8`), so an N1 chord at
exactly the full angle step on the largest radius (sweep / angle an integer) is `EDGE_UNSUPPORTED`
by rounding alone (the 3e-6 / 5e8 frustum at sweep .5, angle .05); the T_face formula (Q5) was not
changed.

### Refusals (fixed texts; `REFUSAL_CODES` unchanged, owner decision)

| code | text (after `face <id>: `) |
|---|---|
| `SUPPORT_UNSUPPORTED` | "the face has an explicit parameterization"; "its support is not a plane, cylinder or cone" |
| `TRIM_WINDS_SUPPORT` | "a loop winds around the support's axis" (its steps, the closing one included, sum to more than pi in magnitude); "the trim's azimuth extent is not certifiably below 2 pi (the support would be covered twice)" (R2 H2); "the face uses one edge twice (a seam or slit)" |
| `OUT_OF_DOMAIN` | N1's and N2's domain refusals (N2's ring cap and its ring-point cap against the coedge count, "trim too complex for display: at least N ring points (MAX_SLAB_RING_POINTS = ...)", before the seam scan, R2 L3; the ring-point cap while the rings are sampled, with N2's text "... N ring points (...)", R1 M4); "a nonzero coordinate or support parameter is below 2**-200 in magnitude (the lift's domain)" (R2 M1/M2); "a coordinate or support parameter is above 2**200 in magnitude, or not finite (the lift's domain)" and "a derived quantity of the lift is not finite, or a nonzero one rounded to 0 (the lift's domain)" (R5 H1); "its plane's vectors are parallel to within a sine of 1e-12"; "its refinement needs more than 200000 slab lines (too complex)"; "its trim reaches the cone's apex (...)" |
| `EDGE_UNSUPPORTED` | N1's edge refusals; "a boundary chord deviates from the plane / cylinder / cone by up to X, above the face tolerance T"; "edge E's drawn chords leave its curve by up to X, above the sag S" (only with `tol.sag`) |
| `TRIM_INVALID` (trims) | "a loop has no coedges"; "consecutive coedges do not meet bit for bit"; "two ring points coincide in the chart"; N2's trim refusals (crossing, touching rings, ...) |
| `TRIM_INVALID` (uncertified lifts) | "two mesh vertices coincide in the chart (lift not certified: ...)"; "the mesh boundary is not the chain of the ring chords (lift not certified)"; "a mesh edge is not shared by exactly two oppositely oriented triangles (lift not certified)"; "a chart triangle is not strictly counter-clockwise (lift not certified)"; "a lifted triangle folds against the face normal (lift not certified)"; "a lifted triangle's deviation bound exceeds the face tolerance (lift not certified)"; "merged cut points, or the ends of a boundary interval their merge cancels, are not bitwise equal in 3D (lift not certified: a gap in the trim)" (R3 C1); "a zero-area lifted triangle is not covered by the triangles kept beside it (lift not certified: it cannot be dropped)" (R3 C3); "the lifted boundary is not simple: its projection back to the chart crosses or touches itself (lift not certified: lifted triangles may overlap)", "the lifted boundary's loops are not nested as the trim's: not exactly one positively oriented loop in the projection back to the chart (lift not certified)", "a lifted triangle's projection back to the chart is not positively oriented (lift not certified: lifted triangles may overlap)", "a lifted point is too close to the support's axis, or too far from its chart point, for the projection back to the chart (lift not certified)" (R4); "an arithmetic error in the lift" |

### Evidence

Tests: `tests/test_display_tessellation_lift.py` (N3a, plane), `..._lift_cylinder.py` (N3b; E2, E3;
the lift's frame and cross product pinned bit for bit to N1's), `..._lift_cone.py` (N3c: T12, T13,
every corpus cone face with its expected code, window patches on tilted, narrowing, equal-radius
and negative-height cones). Every accepted mesh is checked by independent oracles: exact Fraction
combinatorics, a 50-digit mpmath paired deviation on barycentric grids and the footprint
orientation against the exact frame. Evidence folder `display-tess-evidence-20261125`: fuzz
`n3a_fuzz.py` (planar), `n3b_fuzz.py` (cylinder), `n3c_fuzz.py` (cone), 0 oracle failures and 0 raw
exceptions each (`n3c-fuzz.txt`); the plane and cylinder lifts are bit-identical to 57dc038 across
the N3c changes (`n3c_bits.py`, `n3c-bits-compare.txt`); timing `n3c-bench.txt`.

### Codex round 1 (R1): class audit

`tests/test_display_tessellation_lift_n3r1.py` holds every reproduction (failing on ee0bc2c:
`n3r1-old-run.txt`). Each finding's class was searched through the module:

| class | sites | outcome |
|---|---|---|
| H1 vertex identifications certified in the chart only | D1 cut merge; boundary-edge cancellation; E2 azimuth join; N2's equal ring points; bit-equal coedge ends | merge + cancellation: `d + L` measured, counted, `<= T_face` (above); E2 moves no 3D point; equal ring chart points and broken coedge ends stay refused |
| M1/M3 decisions scaled by world placement | Q7 thin gate (`A_face`); plane `A_face`; cylinder/cone `A_face`; cuts' `16 eps max|X|`; E2 join width; N1 apex rule; trim-sag resolution | thin: exact decision; plane, cylinder, cone: origin-relative stage then (plane) exact; cuts: `cut_rounding` from their chords in the local stage; E2, apex and trim resolution are the stored points' own representation (`ulp(|X|)`), kept |
| M2 created points' rounding omitted | ring vertices (N1, counted); cut points incl. refinement cuts (now in `trim_sag_bound`; already in the deviation, measured on a plane, `rho` on curved supports); merged points (`d + L`) | fixed |
| M4 budgets after work | ring points and rings (now while sampling); refinement lines (before N2); exact fallbacks (per thin triangle or per point over the float bound) | fixed |
| L1 sampled oracles | cylinder and cone orientation (now the exact footprint minimum); curved paired deviation (kept, named "sampled"); plane on-chord test (now exact rationals with the rounding of `A + f (B - A)`) | fixed |

Changed results: `trim_sag_bound` of every face with cut points (plus `cut_rounding`, about
`8 eps |X|`); `degenerate_count` now counts only exempt triangles (zero or undecidable normals);
planar faces through N2 slab lines an ulp from a vertex (the private `_extra_x` seam of the fuzz)
whose rounding-thin triangles truly fold are refused (`n3r1-bits-compare.txt`); far-from-origin
planar, cylinder and cone faces that rounding alone refused are accepted.

### Codex round 2 (R2): decisions and class audit

`tests/test_display_tessellation_lift_n3r2.py` holds every reproduction (failing on 8cb91bc:
`n3r2-old-run.txt`). Owner decisions (standing approval, the recommended options):

- **H1 (Q8 wins).** An identification is allowed only within the rounding allowance `A_face`
  (above, Decision D1): no accepted mesh has a boundary gap beyond rounding.
- **H2 (injectivity).** On a cylinder or cone the unwrapped azimuth extent of all rings, raised by
  twice E2's azimuth rounding, must be below 2 pi, else `TRIM_WINDS_SUPPORT`. **Injectivity:** the
  exact map `S*(theta, h) = o + rho(h) c(theta) + h a*` is injective on any strip of azimuth width
  below 2 pi (`rho > 0` on the region, apex refused). Every mesh chart point lies in the rings'
  azimuth range (cut points on ring chords, slab lines strictly inside it). The chart covering is
  certified non-overlapping (Q2), so its image under `S*` covers the face's part of the support
  without a second sheet: two chart triangles that are disjoint in the chart are disjoint on the
  support. The lifted mesh is within `deviation_bound` of that image pointwise (Q4), so 3D overlap
  beyond the certified deviation is excluded; Codex's zero-winding loop (9/8 of a turn and back,
  18 triangles, one 3D point in two triangle interiors) is refused. Refinement adds no point
  outside the range. A plane's chart is affine with a frame of sine at least 1e-12 (else
  `OUT_OF_DOMAIN`, N3a): injective, nothing to add.
- **M1 + M2 (magnitude-dependent arithmetic).** Chosen: a domain guard for the float arithmetic
  plus scale-invariant exact decisions. Domain: every nonzero ring-point coordinate and support
  parameter (origin, axis or plane vectors, radii, height) is at least `2^-200` in magnitude, else
  `OUT_OF_DOMAIN` (N1 bounds the coordinates, the origin and the plane vectors above by
  `MAX_DISPLAY_COORDINATE`, not a cone's height or radii: the upper side is round 5's, below). Why: then a nonzero
  difference of two such numbers is at least `2^-253` and a product of up to four is normal, so the
  relative error analyses of the filters and bounds hold for every input-derived quantity; the
  guard costs nothing for real models (rounding residues such as `6e-17` are far inside) and is
  simpler than a scale-invariant rewrite of every bound. What can still underflow (derived
  quantities: a cut a tiny fraction along a chord) is decided exactly or bounded upward: every
  bound's norm (`_norms`: bit for bit where the squares are normal, else scaled by an exact power
  of two and raised); every float orientation filter carries an underflow slack and sends what it
  cannot certify to the exact decision (the slack only routes); the exact thin decision on curved
  supports scales the exact normal by a power of two (no absolute floor; Codex's 1e-304 normal is
  a fold); the exact plane bound is converted with upward rounding (`_sqrt_up`; a positive
  distance never becomes 0 or smaller; Codex's 1e-200 plane is now `OUT_OF_DOMAIN`, and the exact
  bound of a 1e-170 offset is within two ulps above it).
- **M3 (world placement).** The E2 threshold uses `max |P - o|` plus the stored coordinates'
  representation projected onto the radial plane (above); an axial translation of model and
  support (Codex's strip at z = 0 and 2e6) changes no decision. Purely support-frame magnitudes
  were tried and refused translated faces whose ruling ends differ by their stored coordinates'
  rounding (a window face and a frustum with a hole at a lateral 1e3 to 1e5: their azimuths stayed
  ulp-wide slabs whose wall triangles fold), so the stored representation stays in, projected. The
  rounding-thin width limit and H1's allowance stay `A_face` (the stored points' own
  representation, world-scaled by nature). Limit found: the tripod's x-axis skins translated by
  1e4 or more get consecutive N1 samples 3e-13 apart (N1's output on the translated edges; at 1e6
  N1 refuses the edge), which N3 refuses `TRIM_INVALID`; the invariance test uses 1e3 and 3e4.
- **L1.** The exact plane decision compares squared quantities (`D^2 = q + b1 sqrt(U) + b2
  sqrt(V)` against `limit^2`, signs of sums of square roots decided by squaring): a vertex exactly
  T_face off the plane at z = 2e6 is accepted (it was refused `EDGE_UNSUPPORTED`).
- **L2.** The orientation oracles admit a non-positive triangle only under the module's exact rule:
  on a plane an exactly zero normal; on a cylinder or cone a zero normal, or a footprint minimum
  within `64 eps (1 + |k|) |n|` (twice the module's margin) of a rounding-thin triangle. The ee0bc2c
  module's exempt fold on Codex's 1e6 face fails it.
- **L3.** N2's ring cap and its ring-point cap (against the coedge count, each coedge giving at
  least one ring point) are checked before the seam scan.
- **P1.** The planar orientation of what the first filter cannot certify: a vectorised tight
  filter (Shewchuk's bound per normal component and the float `u x v` error) decides almost all
  (polygon400_holes20: all 625), integers only for the rest.

| class | sites | outcome |
|---|---|---|
| H1 identifications beyond rounding | D1 merge `d`; chain cancellation `L`; E2 join (chart only); equal ring chart points | merge + cancellation `<= A_face`; E2 moves no 3D point, a join of two N1 samples refused (unchanged) |
| H2 injectivity of the chart -> 3D map | cylinder / cone azimuth unwrap (net winding only); holes' 2 pi sheet shift; refinement lines; cone apex; plane frame | extent check over all rings after the shift; lines inside the range; apex `OUT_OF_DOMAIN` (unchanged); plane sine `>= 1e-12` (unchanged) |
| M1/M2 magnitude-dependent arithmetic | curved thin decision (absolute `2^-1000` floor, underflowing `|n|`); plane exact bound (`float(D^2)` to 0); ring-sign and chart-orientation filters (absolute floor, exact fallback); 3D orientation filters (margin underflow); bound norms (offsets, sizes, cut rounding, identification); trim sag (N1's functions) | normalised exact normal, no floor; `_sqrt_up`; filters kept (the floor only routes to the exact decision); 3D filters: underflow slack, undecided go exact; `_norms`; N1 unchanged; inputs below `2^-200`: `OUT_OF_DOMAIN` |
| M3 world-placement thresholds | E2 join; extent rounding; thin width limit; H1 allowance; N1 apex rule; trim-sag resolution; local/exact fallbacks | E2 and extent: support frame plus projected representation; thin, H1: `A_face` (representation, kept); apex and trim resolution: N1's own rules (kept) |
| L1 inverse-square-root enclosures | plane exact decision | exact squared comparison |
| L2 oracle exemptions | plane, cylinder, cone orientation oracles | the module's exact rule |
| L3 budgets after scans | seam scan; N1 sampling (R1); refinement lines (before N2) | ring and coedge counts first |

Changed results: none on the bit-identity sets vs 8cb91bc (`n3r2_bits.py`, `n3r2-bits-compare.txt`:
the planar 1200-face fuzz and corpus, 1000 cylinder and 600 cone fuzz faces, every corpus plane,
cylinder and cone face at both steps: points, triangles, chart, source, scalars and refusals
identical). By construction they change only for: identifications beyond `A_face` (now refused),
cylinder or cone trims of azimuth extent near or above 2 pi (refused), inputs with nonzero
magnitudes below `2^-200` (`OUT_OF_DOMAIN`), faces far from the origin along a coordinate-aligned
axis whose azimuths the world reach had joined (no longer joined; Codex's strip accepted), wide
triangles the float filter could not certify but whose exact orientation is positive (accepted;
before refused unseen), and points exactly at T_face off a plane (accepted). Fuzz (0 oracle failures,
0 raw exceptions each): planar 1141 / 1200 accepted, cylinder 1148 / 1177, cone 947 / 1046
(`n3r2-*-fuzz.txt`).

### Codex round 3 (R3): owner contract decisions C1-C3 and class audit

`tests/test_display_tessellation_lift_n3r3.py` holds every reproduction (failing on 692e819:
`n3r3-old-run.txt`, 18 of 21; the two M3 placement tests and one decision sanity test pass there,
as C2 makes a placement-dependent refusal no defect). Three classes came back a third time, each
through a new allowance (the retry limit), so the owner simplified the contract (2026-11-27,
replacing R1/R2 H1 and the R1/R2 M1/M3 rules where they conflict):

- **C1 Identifications are exact.** Vertices are identified (D1 merge, any dedup) only when their
  lifted 3D points are bitwise equal; a chain edge cancels only when both lifted ends are bitwise
  equal (`L = 0`); coedge ends must meet bitwise (N1's shared vertices). Otherwise `TRIM_INVALID`.
  The `A_face`-based `d + L` gate is removed and `trim_sag_bound` carries no identification term.
  **E2 stays** (chart-only azimuth join, moving no 3D point): it can never identify unequal 3D
  points because every identification goes through the bitwise check: a join making two N1 samples
  one chart point is refused ("two ring points coincide"), a cut landing on a ring vertex's chart
  point is refused ("two mesh vertices coincide"), and cuts made equal in the chart are merged
  only if bitwise equal in 3D. Codex's r3 fixture (ill-conditioned plane, `A_face` 15646, a 0.125
  gap) and the r1/r2 fixtures are refused. Cost: faces whose holes align with other rings to
  within ulps in the chart while their 3D cuts differ by rounding are refused (the grid-aligned
  polygon400_holes20, the tip fixture; see the changed results).
- **C2 Translation invariance is not a requirement; only fail-open matters.** Thresholds are kept;
  the translation tests assert only "accepted -> certified". **Known limit:** the tripod skins
  moved by 1e3 are all accepted; moved by 1e4 along a coordinate axis 6 to 12 of the 18 are
  refused (N1 returns consecutive samples 3e-13 apart, "two ring points coincide", or folds), at
  1e5 N1 itself refuses 11 edges. A placement-dependent refusal takes the consumer's old route.
  Codex's r3 oblique strip (refused after an axial translation along `(1, 1, 0)`) may be refused
  or accepted; accepted, it is certified.
- **C3 No orientation margin** (Q7 above). Exact fallback on curved supports
  (`_thin_curved`): with the exact normal `n` (integers) and `g(t) = P cos t + Q sin t + L`
  (`P = sense n . e1`, `Q = sense n . e2`, `L = -sense k n . a`, exact rationals of the stored
  floats), `|g*(t) - g(t)| <= E = |n| (sqrt(2) FE + (|k| + kerr) FE + kerr (1 + FE))` for the exact
  support normal `c*(t) - k* a*` (frame within `FE` per unit vector, `kerr = 8 eps |k|` for the
  computed slope, `|n|` rounded up). The cosine and sine at the footprint ends are enclosed
  rationally (fixed-point Taylor series in integers, 128 bits, counted truncation; no libm
  assumption). Certified positive iff: footprint below pi (exactly, against the float pi); both
  ends' lower bounds of `g - E` positive; on a cone also `L - E > sqrt(P^2 + Q^2)` (by squares) or
  `g'(lo) > 0` or `g'(hi) < 0` (enclosed: the minimum not inside). Zero normal: dropped;
  otherwise a fold. Codex's r3 M1 predicate (minimum `-32.0156 eps |n|`) is a fold.
- **L1.** The cylinder and cone orientation oracles have no exemption (no 64 eps, no width limit):
  a published triangle with a non-positive exact footprint minimum fails them (Codex's M1 case
  does); the planar oracle requires an exactly positive normal; `_check_combinatorics` restores the
  dropped triangles (an exact cover of zero-normal, chart-CCW triangles, checked against
  `degenerate_count` and the drop rule) and checks the covering with them, and every cancelled
  chain edge with bitwise-equal ends.
- **L2.** The ring cap is checked from `len(face.holes)` before the ring list is built (Codex's
  100000-hole face: peak memory below 100 kB, was 1.6 MB).

| class | every path of the kind | outcome |
|---|---|---|
| C1 identifications with an allowance | D1 merge (`_merge`, chart-equal cuts); chain cancellation (`_combinatorial`); coedge joins (`_ring`, was `np.array_equal`: -0.0 == 0.0); E2 azimuth join (`_join`, `_SNAP`); equal ring chart points (plane and curved); cut on a ring vertex; N2's own slab-line dedup (`np.unique` of x, chart only, pinned); refinement slabs (`np.unique`, chart intervals) | merge + cancellation: bitwise 3D (`_check_identified`); coedge joins: bitwise; E2: chart only, guarded as above; equal ring points and cut-on-vertex: refused (unchanged); N2 and slab dedups identify no 3D points |
| C1 allowances in reported bounds | `trim_sag_bound` (`d + L` removed; `cut_rounding` kept: a cut's own rounding on its chord, not an identification); `_trim_sag`'s rounding-level vertex offsets (Q6 enforcement, D3) | identification term removed; the rest unchanged (not identifications) |
| C3 orientation margins | curved fallback `32 eps (1 + |k|) |n|` exemption and its `A_face` width limit; the fallback's `inside` test with `+-margin`; planar zero-normal exemption (kept in the mesh); the float filters (plane `(16 eps + 2 alpha)`, cylinder `40 eps`, cone `40 eps (1 + |k|)` times `|d1| |d2|`); chart orientation (`_orient_positive`), ring signs (`_ring_sign`) | exemption, width limit and margin removed (enclosure); zero normals dropped and counted; float filters kept as certify-only error bounds (derived budget about 13 eps + 17 eps |k|, threefold slack, libm cosine/sine assumed within a few ulps; nothing they cannot certify is accepted without the exact decision); chart and ring signs exact (unchanged) |
| C2 placement-dependent refusals | E2 threshold, extent rounding, `A_face`, local/exact fallbacks | kept (C2); tests assert "accepted -> certified" |
| L2 budgets after copies | ring list expansion; coedge count; ring-point cap while sampling | ring count from `len(holes)` first; the rest unchanged |

Changed results vs 692e819 (`n3r2_bits.py`, `n3r3-bits-compare.txt`): refused now, the grid-aligned
polygon400_holes20 at both steps (C1: merged cuts not bitwise equal in 3D; the jittered bench face
is unchanged) and 7 of the 1200 planar fuzz faces, all in the private ulp-slab "extra" mode (1 by
C1, 6 because a zero-area triangle holding an ulp-scale spike of the ring, e.g. a ring vertex
2.7e-16 from two cuts, is not covered by kept triangles); `triangles` only (the zero-area ones
dropped, every other field identical, `degenerate_count` the same number) on 59 planar, 37 of the
763 cylinder and 20 of the 447 cone fuzz faces; identical: the other accepted fuzz faces, every
refusal, the tripod skins, the 400-face cylinder and every other corpus plane, cylinder and cone
face at both steps. No face is newly accepted.

Fuzz on the r3 tree (0 oracle failures, 0 raw exceptions each; the oracles verify the module's
dropped triangles as a witness): planar 1134 / 1200 accepted (`n3r3-planar-fuzz.txt`), cylinder
1148 / 1177 (`n3r3-cylinder-fuzz.txt`), cone 947 / 1046 (`n3r3-cone-fuzz.txt`); the scripts' line
"rounding-thin (exempt) triangles" now counts dropped zero-area triangles. Timing vs 692e819
within noise (`n3r3-bench.txt`: tripod 18 skins 75.6 -> 76.9 ms, 400-face cylinder 775 -> 774 ms,
polygon400_holes20 62.7 -> 62.1 ms at 7.5 deg). All display tests: `n3r3-display-run.txt`.

### Codex round 4 (R4): global injectivity of the lift

`tests/test_display_tessellation_lift_n3r4.py` holds the reproductions (all 12 failing on 5c4d38c:
`n3r4-old-run.txt`). Codex's r4 High: a topology-valid cylinder face whose chart is certified and
whose lifted triangles are each oriented and within the deviation bound, yet two lifted triangles
cover the same open patch: the two sides of a notch are 1e-5 apart in the chart, less than the
lifted chords' azimuth deviation (about 1e-4), so the lifted sides cross. The same CLASS as r2 H2
(chart non-overlap plus per-triangle orientation does not imply that the lifted triangles do not
overlap in 3D), now through a chord deviation instead of a winding. The aborted run's lead (a loop
that self-crosses in 3D on the plane x = 1 but not in the chart, forced in topology-invalid) was
accepted with overlapping triangles too.

**Decision (owner standing approval, recommended option): a certificate implying global 3D
injectivity**, not another local check (`_check_projection`, run last on every accepted mesh, so no
earlier refusal or accepted output changes):

- **The projection.** `P` maps 3D back to the chart and is a function on a neighbourhood of the
  support: on a plane `P(X) = ((X - o).e1, (X - o).e2)`; on a cylinder and a cone `P(X) =
  (azimuth of (x, y), z)` with `(x, y, z) = M (X - o)`, `M` the rows `e1, e2, a`. The module's FLOAT
  frame vectors are taken as exact rationals: `P` is then an exactly defined map (it need not be
  orthonormal) whose predicates are rational. Two distinct points of the lifted mesh at the same 3D
  position have the same `P`, so it suffices that `P` maps the published mesh injectively.
  Preconditions (refused `TRIM_INVALID` otherwise): every mesh point's distance from the axis in
  M-space `rho > 16 err` (`err = 4 eps |fl(X - o)|_1` bounds `x, y, z`), its computed azimuth within
  pi / 4 of its chart azimuth (the unwrapping), every published triangle's P-azimuth range below pi
  (so it misses the axis), every boundary edge's below pi / 2 (exact `cross2` comparisons of pairs),
  and the whole mesh's P-azimuth range below 2 pi (`TRIM_WINDS_SUPPORT`: `P` single-valued, the
  branch cut outside the mesh). On a cone the trim already avoids the apex (`rho > 0`, T13).
- **(i) Local.** Every published triangle's P-image is positively oriented: the Jacobian of `P` on
  the flat triangle. Plane: `n . (e1 x e2)` (float filter `24 eps |d1| |d2|`, else the exact normal
  against the exact `e1 x e2`). Cylinder and cone: `n . (cos t f1 + sin t f2)`, `f1 = e2 x a`, `f2 =
  a x e1` (Binet-Cauchy: `(M d1 x M d2) . (cos t, sin t, 0)`), over the triangle's P-azimuth range
  (the hull of its vertices' azimuths, widened by their error bounds; a convex set in an open
  half-plane through the axis). Filter `40 eps |d1| |d2|` at both ends (an arc argument as for
  Q7), else exactly: the vertices' exact azimuths checked inside the range with the rational
  cosine/sine enclosures, then `_thin_curved` with the exact frame `(f1, f2)` and `E = 0`. **How Q7
  relates:** on a plane Q7's sign `n . (u x v)` is the Jacobian of the orthogonal projection, the
  float frame's `e1 x e2` within `alpha` of it, so (i) is a second decision of the same quantity
  (no accepted face changed); on a cylinder Q7's `n . c*(t)` is the Jacobian of the orthonormal
  `P*` over the chart footprint, (i) the float frame's over the P-azimuth range (they differ by the
  frame error and by the cut points' chart-vs-lift azimuths); **on a cone they differ in kind**:
  Q7 uses the support normal `n . (c - k a)`, `P`'s Jacobian is `n . c` (radial, perpendicular to
  the axis): a triangle whose normal is far from the radial direction (a steep `k`) can pass Q7 and
  fold under `P`; (i) refuses it (test).
- **(ii) Boundary.** The P-image of the lifted boundary chain (the published mesh's boundary edges,
  after the chart chain's D1 cancellations; chain vertices with bitwise-equal lifted points, which
  C3 makes when two cuts an ulp apart in the chart lift onto one point, are one point and the
  zero-length edge between them is dropped) must be a set of pairwise disjoint simple closed
  curves: every vertex once a tail and once a head; adjacent edges meet only at their shared
  vertex; non-adjacent ones not at all. On a curved support the P-image of a lifted chord is a
  curve (`z` linear, the azimuth monotone), so the predicates are exact on the curve, not on its
  chord: two edges meet iff a common height and a common ray (`cross2 = 0`; the pair's azimuths lie
  within pi) is reached at parameters in `[0, 1]`; with the heights eliminated `cross2` is a
  quadratic in one parameter, and a root in the feasible interval is decided exactly (ends and the
  rational vertex); adjacent edges from `B` (`U = A - B`, `V = C - B`) meet again iff `s Uz = t Vz`
  and `s (cross2(U, B) + (Uz / Vz) cross2(B, V)) + s^2 (Uz / Vz) cross2(U, V) = 0` at `0 < s, t <=
  1` (linear after the root `s = 0`), horizontal edges by their azimuth intervals. Candidates: the
  adjacent pairs behind a float filter (apart in azimuth, apart in height, or `|alpha'| > min(|Uz|,
  |Vz|) |beta|`; plane: not collinear or pointing apart), the non-adjacent pairs from a sweep over
  the edges' bounding boxes (sorted by azimuth, `O(n log n + pairs)`; more than 4e6 box pairs is
  `OUT_OF_DOMAIN`). Otherwise `TRIM_INVALID` ("the lifted boundary is not simple").
- **(iii) Nesting.** Each loop's orientation at its lexicographically smallest P-vertex (exact:
  the azimuth is monotone along every edge, so the extreme is a vertex; the sign of the tangents'
  cross product there, `cross2(v, W) Uz - Wz cross2(v, U)`, on a plane the orientation of `v`,
  next, previous). Exactly one loop may be positive (the outer one, holding vertex 0), the others
  negative; else `TRIM_INVALID` ("not nested as the trim's").

**Argument** (preconditions above). Let `F = P o L` on the chart region `R` (`L` the lift by
vertices, affine on each chart triangle). By (i) `F` is an orientation-preserving local
homeomorphism on every published triangle (and injective on it: two of its points on one fibre ray
would put the ray in the triangle's plane, where the Jacobian vanishes). For a point `y` off the
image of the edges and of the dropped zero-area triangles (a null set), the degree of `F` at `y`
is the number of its preimages (each counts +1) and equals the winding number of `F(dR)` around
`y` (the boundary chain's image, a zero-length chain edge a point). By (ii) and (iii) that is the
sum over disjoint simple closed curves of +1 inside the one positive loop and -1 inside each
negative one, hence at most 1. So no two published triangles' P-images share an interior point
(their interiors are open sets, and a.e. point of an intersection would have two preimages); as
`P` is a function on 3D, no two published lifted triangles share an interior point in 3D, and the
boundary curves are injective. As the degree is also `>= 0` everywhere, every negative loop lies
inside the positive one and outside the others: the trim's nesting.

Codex's notch face and the lead are refused ("the lifted boundary is not simple"). The oracles:
`_check_cylinder` and `_check_cone` now call `_projection_overlaps`, an independent EXACT test of
P-image overlap (P about the stored axis direction, in Fractions; every pair of mesh edges meets
at most at a shared vertex, no vertex strictly inside another triangle's P-image (its fibre ray
meets the triangle's interior), every triangle P-positive, 50 digits): the Codex mesh with the
certificate switched off fails it.

| class: chart-only reasoning for a 3D property | where | outcome |
|---|---|---|
| chart covering implies a 3D covering | `_combinatorial` (Q2) | kept (a chart statement); 3D non-overlap now by the P certificate |
| chart extent below 2 pi implies a single sheet | `_check_extent` (H2) | kept; the P certificate also checks the lifted points' own P-azimuth range |
| triangle orientation over the CHART footprint | Q7 (`_thin_decision`, plane filter) | kept (Q7 is a statement about the paired support normal); the P-Jacobian (i) over the lifted P-azimuth range added |
| cut points ordered along their chord in the chart | `_combinatorial` chain order | the lifted cuts' order (rounded `A + f (B - A)`) is now covered by (ii) (adjacent chain edges meet only at their shared vertex) |
| distinct chart points are distinct 3D points | `_merge`, `_RING_COINCIDE`, E2 join | identifications bitwise in 3D (C1, unchanged); two chain points equal in 3D are one point in (ii) and a self-touch is refused; equal in 3D interior points are covered by (i)-(iii) |
| chart notch width vs chord deviation | the r4 High | (ii) |
| winding / sheet of the chart | `_unwrap`, `_join` | unchanged (r2 H2); P single-valued by its own range check |
| zero-area triangles dropped | `_drop_zero` | exact 3D coverage (unchanged); null set in the degree argument |
| deviation bounds | `_CylinderBound`, `_ConeBound`, plane bounds | paired (3D point vs `S*` of its chart point): a 3D property by construction, unchanged |

Changed results vs 5c4d38c (`n3r2_bits.py`, `n3r4-bits-compare-raw.txt`): none. Every face of the
bit-identity sets (1200 planar fuzz faces and the planar corpus/bench faces, 763 cylinder and 447
cone fuzz faces, every corpus plane, cylinder and cone face at both steps) has the same outcome and,
when accepted, bit-identical output; the tripod skins and the 400-face cylinder stay accepted.
Fuzz on the r4 tree (0 oracle failures with the new overlap oracle, 0 raw exceptions each, same
accept counts as r3): planar 1134 / 1200 (`n3r4-planar-fuzz.txt`), cylinder 1148 / 1177
(`n3r4-cylinder-fuzz.txt`), cone 947 / 1046 (`n3r4-cone-fuzz.txt`). Timing vs 5c4d38c
(`n3r4-bench-raw.txt`, 7.5 / 22.5 deg, loaded PC): tripod 18 skins 76.0 -> 82.1 / 51.8 -> 56.6 ms
(+8 / +9 %), 400-face cylinder 780 -> 888 / 787 -> 896 ms (+14 %), polygon400_holes20 62.8 -> 76.2
/ 59.6 -> 67.7 ms (+21 / +13 %). Reasons: the certificate is new work on every accepted face, about
0.25 ms fixed per face in NumPy call overhead (the 400-face cylinder's faces have 4 triangles
each); on the jittered polygon about 520 non-adjacent boundary edge pairs pass the float filters
(ulp-short chain edges from slab lines an ulp apart) and are decided exactly, and 625 sliver
triangles' P orientation goes exact.

Remaining assumption (documented, as for the float filters since N3b): the libm `atan2`, `cos`,
`sin` within a few ulps in the float filters and in the boxes that prune candidate pairs; every
decision the filters cannot make is exact (rationals; the azimuth enclosures of (i)'s fallback are
the fixed-point ones of C3).

### Codex round 5 (R5): the magnitude domain, upper side; speed of the R4 certificate

`tests/test_display_tessellation_lift_n3r5.py` holds the reproduction (9 of its 13 H1 tests fail on
75c52fc, `n3r5-old-run.txt`; the other 4 are acceptance checks; 5 more test the P1 scalar path
against the array path). Codex's r5 High: a topology-valid
cone face (`radius_end = 1e307`, `height = float max`) for which `height |axis|` overflowed, so
`_ConeChart` computed the slope `k = 0` (exactly about 0.0556); the face was accepted with
`deviation_bound` 3.23e-4 against a paired deviation of 3.76e-4 at a returned vertex, and its first
triangle folds against the exact cone normal. The class: R2 M1/M2 bounded the magnitudes from below
only; N1 bounds the ring coordinates, the origin and the plane vectors by 1e9 but not a cone's
height or radii or a cylinder's radius.

**Decision (owner standing approval, recommended option): the symmetric upper guard.** Every
ring-point coordinate and every stored support number (origin, axis, plane vectors, the stored
radial and circumferential directions, radii, height, sweep and start angle) must be finite and at
most `2^200` in magnitude, else `OUT_OF_DOMAIN` ("a coordinate or support parameter is above
2**200 in magnitude, or not finite (the lift's domain)"), checked with the lower side, before any
lift arithmetic on them (`_check_magnitudes`). In defence in depth (the audit shows they cannot
fire inside the domain): the cone slope must be finite and nonzero when `r1 != r0`, and a cylinder's
or cone's `T_face` finite, else `OUT_OF_DOMAIN` ("a derived quantity of the lift is not finite, or
a nonzero one rounded to 0 (the lift's domain)", `_check_derived`). The Codex cone is refused by the
guard; with the guard bypassed the slope check alone refuses it; with both bypassed the old mesh
comes back and fails the exact-slope oracle (tests). The same face shape with in-domain parameters
(the Codex slope at heights 9e59 and `2^200`, and a nearly cylindrical `k = 2^-180`) is accepted
and passes the independent cone oracle (exact slope, deviation, footprint orientation, no overlap).

**Audit of every derived float quantity inside the domain.** Domain facts: ring points, mesh points
(cuts on ring chords) and the origin within `2^30` (N1), so `|X - o|`, `h`, the frame coordinates
`x, y` within `2^32`; the axis and stored frame unit to 1e-9 (N1); `r, r0, r1, |height|` in
`[2^-200, 2^200]` (or a zero radius: the apex rule); the display angle at most pi / 2 (N1); chart
azimuths within `2^4` (each ring's steps sum below pi, holes shifted onto the outer sheet); every
nonzero input at least `2^-200`, every nonzero difference of inputs at least `2^-252`. Underflow:
every positive bound is a sum that contains `A_face >= 32 eps max |X| >= 2^-250` and is raised by
`1 + 8 eps` at the end, so the absolute loss of a positive term underflowing (at most a few
`2^-1074` per operation, a few hundred operations) is absorbed; every float filter carries an
underflow slack and only routes to an exact decision (R2).

| quantity | where | range inside the domain | outcome |
|---|---|---|---|
| cone slope `k = (r1 - r0) / (height |axis|)` | `_ConeChart` | denominator in `[2^-201, 2^201]`; `|k|` in `[2^-453, 2^402]`, exactly 0 iff `r1 = r0`; its relative error the stated `8 eps` (no over/underflow) | proven; also checked (`_check_derived`): the r5 High |
| `rho = r0 + k h`, `rho_rounding` | cone chart, bounds | `<= 2^435` | proven finite |
| `T_face = r_max (angle + a)^2 / 8 + residual` | cylinder, cone | `<= 2^202 + residual` | proven; also checked |
| plane sine `|u x v| / (|u| |v|)`, `alpha` | `_PlaneChart` | products of two inputs in `[2^-400, 2^62]`, normal; `alpha <= 2^-9` (sine >= 1e-12) | proven (R2) |
| `A_face`, `A_loc` (cylinder, cone, plane) | `rounding`, `local_rounding` | `<= 2^436` | proven finite |
| boundary-chord bounds `rho dT^2 / 8 + |k| dT dH / 4` | step (5) | `<= 2^441` | proven finite |
| cylinder Taylor bound `E(T)` (variances, `r kt`, edge maxima `gap^2 / (4 tau^2)`) | `_CylinderBound` | `<= 2^212`; the quotient only where `tau^2 > 0`, at most `tau^2 / 4` | proven finite |
| cone bound `H(w)`: `D_ij`, `P_i`, `det = 4 D01 D02 - m^2`, `det_error`, `flat^2`, the critical point | `_ConeBound` | `D, P <= 2^446`, every product `<= 2^894` (the module's largest float value; `2^130` headroom), the critical point only where `det > 1e-6 flat^2` (condition at most 4e6) | proven finite |
| offsets `delta = |X - S(c)|` (squared norms) | `offsets`, `local_offsets` | `S(c) <= 2^436`, squares `<= 2^874`; underflow by `_norms` (R2) | proven |
| Q7 filters (`40 eps (1 + |k|) |d1| |d2|`, `level = -k n . a`, `hypot(p, q)`) | `_lift_cylinder`, `_lift_cone` | `|d| <= 2^33`, `|n| <= 2^67`, values `<= 2^470` | proven finite; undecided go exact |
| exact Q7 / projection (`kerr = 8 eps |k|`, `E`) | `_thin_curved` | rationals of the stored floats; `kerr` valid because `k` is computed without over/underflow | exact |
| E2 join threshold `SNAP (max(1, |theta|) + reach / radius)` and `_check_extent`'s rounding | `_join`, `_check_extent` | `reach <= 2^35`; `radius = r >= 2^-200` (cylinder) or `> 8 eps |r0| >= 2^-252` (cone), so `<= 2^288`; only for a cone whose apex is the support origin (`r0 = 0`, `radius > 8 eps |k| H`) can it overflow (an axial reach `H` near the cancellation level) | finite, except that apex case: there the join or `_check_extent` (infinite rounding) refuses (fail closed; a large threshold only joins more or refuses) |
| refinement slab middle `(left + right) / 2` | `_bisection` | chart azimuths `<= 2^4` | proven; an unsplittable slab refuses |
| projection P (R4): `err`, `rho`, `aerr = 2.2 err / rho + ...`, boxes, product filters | `_CurvedProjection`, `_PlaneProjection`, `_check_boundary` | `err <= 2^35`, `rho > 16 err`, `aerr <= 2^5`, products `<= 2^140` | proven finite |
| plane exact bound, `_sqrt_up` | `exact_bound` | rationals; rounded up (R2) | exact |
| cut rounding `4 eps |(|A| + |B|)|` | `_cut_rounding` | `<= 2^-17` | proven |
| trim sag | N1's functions (N1's domain) | N1 | unchanged |

**Q7 and the projection on a cone use consistent slopes.** `P = (azimuth, axial)` does not involve
`k`: its Jacobian sign on a lifted triangle is `n . c(theta)` (radial). Q7 and the deviation bounds
use the one slope `_ConeChart.k`, within `8 eps |k|` of the exact one inside the domain (the exact
fallback carries that `kerr`); the oracle uses the exact slope. The two certificates are separate
and both must pass, so an accepted cone triangle satisfies both: Q7's `n . (c - k a) > 0` over its
footprint and `n . c > 0` over its P-azimuth range (a test checks both exactly at every vertex of an
accepted frustum mesh). The Codex triangle passed the projection check correctly (its radial
Jacobian is positive) and Q7 only because `k` had become 0.

**Speed (P1; the R4 certificate cost ~0.25 ms per face).** No published value changes; every float
filter still only routes to an exact predicate:

- a scalar path (`_SMALL_POINTS = 40` mesh points or fewer: the 400-face cylinder's faces): the same
  float quantities on Python floats (the same formulas and error bounds; `math`'s `cos`, `sin`,
  `atan2`, `hypot` within the few ulps already assumed), the same filters, the exact predicates
  inherited from the array classes, the same checks in the same order with the same texts; a chain
  with two tails of equal computed P (the bitwise identification) takes the array path;
- the exact normals Q7 already computed are shared with the projection certificate (`_cached_normals`;
  every use is a sign homogeneous in the normal);
- on a plane, (i)'s undecided triangles go through Q7's tight filter (`_tight_dot`, the reference
  `e1 x e2` within `_cross_error`) before the exact decision: the polygon400 slivers (625) decide in
  floats;
- a separating-axis filter along each segment for non-adjacent plane boundary pairs (`_beyond`: both
  ends of one segment strictly beyond an end of the other in its direction, from differences relative
  to that end): the ~520 pairs of pieces of one chord separated by ulp-short chain edges decide in
  floats;
- `_fixed` by shifts, the box sweep's height test on sorted copies.

Changed results vs 75c52fc (`n3r2_bits.py`, `n3r5-bits-compare.txt`): none. Every face of the
bit-identity sets (1200 planar fuzz faces and the planar corpus/bench faces, 763 cylinder and 447
cone fuzz faces, every corpus plane, cylinder and cone face at both steps) has the same outcome
(refusal code and text) and, when accepted, bit-identical output; the tripod skins and the
400-face cylinder stay accepted. No face of these sets has a parameter above `2^200`. Scalar and
array paths: random small chains and triangles on a plane and a cylinder give the same outcome on
both (tests). Fuzz on the r5 tree (0 oracle failures, 0 raw exceptions each, the same accept counts
as r4): planar 1134 / 1200 (`n3r5-planar-fuzz.txt`), cylinder 1148 / 1177
(`n3r5-cylinder-fuzz.txt`), cone 947 / 1046 (`n3r5-cone-fuzz.txt`). The lift test files also pass on
Python 3.13.9 (`n3r5-py313-run.txt`).

Timing (`n3r5-bench.txt`: 4 runs, two bench invocations of 2 interleaved rounds each, one session;
the PC was heavily loaded, absolute numbers about 1.5x the r4 session's; totals, medians, 7.5 /
22.5 deg; 5c4d38c -> 75c52fc -> r5): tripod 18 skins 75.9 -> 82.2 -> 83.5 (+10 %) / 75.4 -> 84.9 ->
83.1 (+10 %); 400-face cylinder 1164 -> 1317 -> 1217 (+5 %) / 1145 -> 1331 -> 1212 (+6 %);
polygon400_holes20 96.0 -> 114.0 -> 107.4 (+12 %) / 95.0 -> 102.6 -> 96.6 (+2 %). The certificate
itself (warm per-function medians): 400-face cylinder 139 -> 37 ms, polygon400 13.1 -> 4.6 ms (7.5
deg), tripod 5.8 -> 5.6 ms. What remains is the certificate's own work on the array path: the
tripod skins (more than 40 points per face, where the arrays are faster than the scalar path,
measured 14.7 ms scalar) and the polygon's 5594-edge chain (the box sweep 1.4 ms, about 80 000
u-overlapping candidates for 6841 box pairs; a uniform grid measured slower, 5.0 ms).


### Codex round 6 (R6): scalar and array paths decide alike; the upper guard covers every stored field

`tests/test_display_tessellation_lift_n3r6.py` holds the reproductions (5 of its 9 tests fail on 17c0de7,
`n3r6-old-run.txt`; the others are acceptance checks).

**M1 (the scalar certificate accepted a face the array certificate refused).** At the strict
`rho > 16 err` cutoff the stdlib `hypot` returned one ulp more than `np.hypot` for the same lifted vertex:
the round-5 scalar path published the triangle, the array path refused `TRIM_INVALID`. The class: the
scalar path duplicated array-path arithmetic with other primitives. Contract (owner standing approval,
recommended option): the scalar path gives exactly the array path's decision AND values. Audit of every
computation the scalar path duplicates (`_display_lift.py`, `_SmallPlaneProjection`,
`_SmallCurvedProjection`, `_projection_triangles_small`, `_check_boundary_small`):

| Computation | Array path | Scalar path (r6) |
| --- | --- | --- |
| `fl(X - o)`, the dots with `e1, e2, a`, `err = 4 eps \|d\|_1 ...` | NumPy elementwise (sum of 3 left to right) | the same operations on floats, same order (IEEE `+ - *`: identical) |
| `rho = hypot(x, y)`, the `rho > 16 err` cutoff | `np.hypot` | `np.hypot` on one short array (was `math.hypot`: an ulp apart) |
| `atan2`, `round((theta - raw) / 2 pi)`, `u`, `aerr` | `np.arctan2`, `np.round`, array expressions | the same NumPy expressions on one short array (was `math.atan2`, Python `round`: `round(nan)` raised) |
| triangle cross products, `\|d1\|\|d2\|` (`sqrt`), `24 eps` / `40 eps` bounds | elementwise `-`, `*`, `np.sqrt` | the same terms, `math.sqrt` (correctly rounded: identical) |
| the dots `n . normal_f`, `n . f1`, `n . f2` | `normal @ v` (BLAS: order and fused multiply-add not fixed) | explicit `nx v0 + ny v1 + nz v2` in BOTH paths (the array path changed: a filter only, no published value) |
| `cos`, `sin` of the azimuth range in the orientation filter | `np.cos`, `np.sin` | `np.cos`, `np.sin` on one short array (was `math`) |
| `nextafter`, `min`/`max`, boxes, chain checks, adjacent and separating-axis filters | NumPy | the same expressions (checked term by term; order of raises equal) |
| exact predicates, `_fixed`, exact azimuth enclosures | shared classes | shared |

A differential test (`tests/test_display_tessellation_lift_n3r6.py`) runs corpus faces with the fast path on and
forced off (`_SMALL_POINTS = 0`) and demands an identical outcome (refusal code and text, or bitwise
points, triangles, chart, source and scalars); `n3r6_diff.py` runs every planar (1200), cylinder (1600) and cone
(1400) fuzz face and every corpus face at both steps (`n3r6-diff.txt`: 0 mismatches). Changed result: the
Codex reproduction is now refused on BOTH paths (`TRIM_INVALID`, near the axis; was accepted on the fast path).
No other face of the bit-identity sets changes.

**L1 (the upper guard omitted `Cylinder.height`).** `_upper_parameters` is now derived from the support
instance's stored numeric attributes (every array component and float of `vars(support)`), not a hand list;
a test sets each stored numeric attribute of `Plane`, `Cylinder` and `Cone` to `2**201` and requires the
guard's list to exceed the bound (a missed or new field fails it). A cylinder of stored height `2**201` is
`OUT_OF_DOMAIN`.
