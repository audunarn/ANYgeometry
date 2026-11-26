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

## Refusal codes (`REFUSAL_CODES`)

| Code | Meaning |
|---|---|
| `SUPPORT_UNSUPPORTED` | an incident face's support is not a plane, cylinder or cone, or the face has an explicit parameterization |
| `EDGE_UNSUPPORTED` | the curve is not a line, circle or ellipse; degenerate frame or projection; support azimuth not monotone along the edge or reaching the axis (a line crossing it); unresolvable sag (including a sag rounding allowance above 2 % of the sag allowance); a vertex farther from its curve than the sag allowance; more than 4096 intervals (each division, or their sum); lost precision |
| `OUT_OF_DOMAIN` | outside the declared numeric domain (`DisplayDomainRefused`), including an angle rounding allowance above 2 % of `tol.angle` |
| `TRIM_INVALID`, `TRIM_WINDS_SUPPORT` | reserved for face trims (not raised by edge sampling) |

## Trim ring triangulation (`anygeometry._slab_triangulation`, private, N2)

`triangulate_rings(rings, extra_x=None)` triangulates the region of `rings[0]` minus `rings[1:]`
(each ring closed implicitly; ring orientation is free) in pure NumPy, without Shapely. It is not
wired into any public display mesh yet (N3/N4).

**Input contract (owner decision, N2 round 1, the recommended option): simple rings.** The outer
ring and every hole are simple polygons: no proper crossing, no self-touch, no zero-width retrace
(repeated consecutive points are allowed and collapse to one vertex; collinear runs are fine). The
holes lie strictly inside the outer ring and are pairwise disjoint: no two boundaries share a point
or an edge, and no hole lies inside another. Anything else is `TRIM_INVALID`. (The first version
described an even-odd rule; it is dropped: validity then depended on the cut list, and
self-touches were refused or accepted by an area comparison, not by the rule.)

**Validity is decided exactly and does not depend on `extra_x`** (owner decision, N2 round 1). The
decision uses only ordinate comparisons on slab lines. A cut point's ordinate is computed in one
canonical form from the segment's lower-x endpoint, `y = ya + f (yb - ya)`, `f = (x - xa) / (xb - xa)`,
so every traversal of a segment gives bit-identical values; vertices are exact. Float filter: the
computed ordinate is within `B = 4 eps (|ya| + |yb|) + 2^-1000` of the exact one (five roundings on
the product and one on the sum give at most `6.01 u (|ya| + |yb|)`, `u = eps / 2`; `B` is that with
a 1.33 margin, which also absorbs the filter's own difference; the absolute term covers
underflow). Two ordinates on a line are ordered by the filter when they differ by more than the sum
of their bounds, otherwise by exact rational arithmetic (`fractions.Fraction`). The checks, on the
slab structure (vertical lines through every vertex and every `extra_x`; extra lines only refine
it, and the predicates are exact, so the verdict is the same with or without them):

1. on each line, all boundary points (vertices, cut points) in exact order: two distinct points at
   the same exact location are a contact;
2. a vertical segment's two endpoints are adjacent in that order (nothing else touches it, no
   vertical retrace);
3. in each slab the spanning segments keep their order from the left line to the right line (a
   reversal is a proper crossing; equal at both lines is an overlap);
4. walking each slab bottom-up, a hole is entered only inside the outer ring and while no other
   hole is open.

Cost (owner decision, N2 round 1): `O(I log I)` in the number `I` of segment-slab incidences
(sorts and linear scans, no pairs); the exact fallback runs only on lines whose adjacent pairs the
filter cannot separate. `I` is counted before anything is allocated (preflight; a cheap upper
bound, segments x lines, skips the count for small trims) and above `MAX_SLAB_INCIDENCES = 200000`
(also the limit on the number of `extra_x` values) the trim is `OUT_OF_DOMAIN` ("trim too complex
for display"); the caller keeps its old route. Why 2e5: measured `I` is 4.2e3 for a 400-vertex ring
with 20 holes, 0.8e3 for a plain 400-vertex ring and 2.0e4 for a 400-vertex star (the densest
ordinary trim measured), a factor 10 below; results are linear in `I` (cut points and triangles at
most about one per incidence) and a trim just under the budget (staggered holes, `m = 315`,
`I = 199712`, 200972 triangles) took about 0.1 s. Staggered holes need about `2 m^2` incidences, so
`m >= 316` is refused. `I` is not `O(n)` in general (a non-monotone ring can need `O(n^2)`); the
budget bounds it.

Method: in each slab the spanning segments, in their exact order, are paired bottom/top, giving
trapezoids whose union is the region. x-monotone single rings take a certified fast path (pure
Python up to 48 vertices, NumPy above) giving the same trapezoids: it returns only when the float
filter proves the two chains strictly apart at every interior line (then the ring is simple and
the rounded ordinates are ordered); otherwise the general path decides. Tests force the general
path with the private keyword `_fast=False` and compare.

Contract of the result (`SlabTriangulation`):

- `points` are the input vertices in order (rings concatenated, unchanged bits), then *cut points*;
  cut point `k` lies on ring segment `cut_ends[k] = (a, b)`, `a` its endpoint of lower x, at
  fraction `cut_fraction[k]` from `a`, strictly inside `(0, 1)`, so a caller can place it on the
  segment's own image. No other points are created.
- `triangles` are counter-clockwise with strictly positive area, decided from the slab geometry,
  not from a determinant (owner decision, N2 round 1): each triangle has one vertical side between
  two distinct points of one slab line and its third corner on the other line, so its exact area
  is (side height) x (slab width) / 2 > 0. A piece is dropped only when that side has zero height
  (one shared point).
- The triangles tile, exactly in rational arithmetic, the *rounded* polygon (the rings with each cut
  point inserted on its segment): no overlap, nothing outside, every point on a trapezoid's
  vertical side used (no T-junctions). The rounded polygon's area differs from the exact one by at
  most `sum_k B_k (x_next,k - x_prev,k) / 2` over the cut points (moving one by `d` changes the area
  by `d` times half its neighbours' x distance).
- The result is deterministic (bit-identical on repeat).

Refusals, in this order (no raw `IndexError`/`ValueError`/`OverflowError`/`FloatingPointError`):

| code | when |
|---|---|
| `TRIM_INVALID` (`SlabInvalid`) | not a sequence of `(n, 2)` arrays, non-numeric (string) data, fewer than three (distinct consecutive) vertices |
| `OUT_OF_DOMAIN` (`DisplayDomainRefused`, as N1) | the declared numeric domain (owner decision, N2 round 1), checked before any arithmetic: real finite coordinates with `\|c\| <= MAX_DISPLAY_COORDINATE = 1e9` for rings and `extra_x` (complex, boolean and object arrays, e.g. integers too large for a float, are refused, never converted; non-finite values moved from `TRIM_INVALID` to `OUT_OF_DOMAIN`); then the preflight budget. Inside the domain no overflow is possible: differences are at most 2e9, shoelace products at most 4e18 |
| `TRIM_INVALID` | the input contract above (crossing, self-touch, retrace, touching rings, hole outside or nested), decided exactly |
| `OUT_OF_DOMAIN` | the valid input cannot be resolved in floats: a cut fraction rounds to 0 or 1 at a strictly interior x ("slab too narrow"); rounded ordinates on a slab line not strictly in their exact order ("feature below display resolution"); a ring's float area at most `16 n eps extent^2` of ITS OWN extent (a backstop; the floor used to be relative to all rings and refused a 1/64 hole in a 1e6-long face) |

The tests carry an independent exact oracle (`tests/test_display_tessellation_slab_contract.py`):
exact orientation, exact pairwise interior-disjointness (bounding-box prefilter, no cap), exact
containment of corners, edge midpoints (closed) and centroids (strict) in the rounded polygon,
exact equality of the triangle area sum with the rounded polygon's area, the stated bound against
the exact polygon, cut provenance recomputed exactly, and a brute-force exact validity verdict.

Fuzz (N2 round 1, 6000 cases: `display-tess-evidence-20261125/n2r1_fuzz.py`; integer-grid, star,
snapped and one-ulp-nudged star, and x-monotone generators, half with random extra lines, some one
ulp from an end; each case also run with the general path forced): 0 raw exceptions; 0 exact-oracle
failures among the 2810 accepted; the brute-force exact verdict matched on all 6000 (2957 invalid,
all `TRIM_INVALID`); fast and general paths gave the same verdict and the same triangles; the
verdict never changed with `extra_x`. 233 valid cases were `OUT_OF_DOMAIN` (190 feature below
display resolution, 39 slab too narrow, 4 area floor): near-contacts at the ulp level (e.g. a
snapped vertex 1e-10 off its own edge after an offset) or an extra line one ulp from a vertex.

Measured (best of 30 or 300 runs, loaded PC, before/after round 1): a 400-vertex ring with 20 holes
2.2 / 2.3 ms, a plain 400-vertex ring 0.25 / 0.23 ms, a 400-vertex star 7.1 / 5.6 ms; a unit square
58 / 45 us, a square with one hole 0.19 / 0.26 ms (the general path's exact checks).
