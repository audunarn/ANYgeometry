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
| Edge curves | `Straight`, `LineRange` whose carrier IS the edge's segment (exact domain `[0, 1]`, both carrier endpoints the evaluated ends within `model.tolerance.length`) (lines); `Arc` (circle, via `model.arc_frame`); `EllipticArc` (a circle when `|u| = |v|` and `u.v = 0` to 8 eps relative, else an ellipse; `u`, `v` need not be orthogonal) | every other curve (`Spline`, `NurbsCurve`, `CurveOnSurface`, intersection curves, ...), and a `LineRange` drawn over part of its carrier (any other exact domain, reversed included): `EDGE_UNSUPPORTED` |

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

## Exact divisions

`edge_display_samples(model, edge_id, DisplayTolerance(angle, sag=None))` returns `EdgeSamples`:
parameters strictly increasing from exactly 0 to exactly 1, points read-only; `points[0]` and
`points[-1]` are the vertex positions exactly, interior points are
`model.evaluate_edge_many(edge, parameters)`. The parameters are the sorted union of the following
divisions; every interval of the union lies inside one interval of each, so all bounds hold on it.

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
   plane normal to the axis, in the support's (radial, circumferential) frame.
   - Lines: every decision uses the quantity that is accurate for it, never a reconstruction
     (round 3: `B' = A' + P(b - a)` cancelled an end's own 1.5e-9 radius to 0 next to a 2e7 start,
     and the line was drawn as one chord). Each end's radius from its OWN projection,
     `A' = P(a - o)`, `B' = P(b - o)` (rounded by `4 eps |p - o|`); the direction from the
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
     are found in closed form): strictly inside the interval, non-strictly at its ends (`g = 0` at
     an end is the curve tangent there to a ray from the axis: a stationary end of a monotone
     azimuth). An end value within rounding of 0 (`|g| <= 16 eps (1 + |z|)`) of the wrong sign
     counts as 0 only when the turnback it could hide, about `g^2 / (2 |z + w|^3)`, is below eps
     rad. A sign change inside, or the projection reaching the axis, is `EDGE_UNSUPPORTED`. Equal azimuth steps, each solved exactly (`A cos(theta) + B sin(theta) = C`
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

The union is one sort and one pass. Every tangent-division value is kept. A value `v` of a support
division within 64 ulps of the stored angle (in edge-parameter units: closer parameters evaluate
to the same theta) of a kept value `k` is dropped only when that division's own angle at `k`
equals its angle at `v` to rounding (`64 eps max(1, |angle|)`, the angles being local to the
edge), so every interval still lies inside one interval of that division up to that rounding;
otherwise it is kept (and a cut too close to evaluate apart is refused). Every division's count
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
and points) identical bit for bit to c47336c.

## Declared numeric domain (owner decision 2026-11-25)

Checked before any division. Inside the domain the bounds above hold to rounding; outside it the
call raises `DisplayDomainRefused` (a `DisplayTrimRefused` with code `OUT_OF_DOMAIN`).

| Bound | Value, applied to | Why |
|---|---|---|
| `MAX_DISPLAY_COORDINATE` | 1e9 model units: vertex positions, arc centre and (`EllipticArc`) centre + u, centre + v, an `Arc`'s via vertex, `LineRange` carrier endpoints and a line's evaluated ends, every incident support's origin (and a plane's u, v vectors), every emitted point | float spacing 1.2e-7 there; ship and offshore models in m or mm are far inside |
| `MAX_DISPLAY_ANGLE` | 1e6 rad: `EllipticArc` start and start + sweep; cylinder and cone start and start + sweep | float spacing 1.2e-10 rad, four orders below the smallest display angle (1e-6), so `start + sweep t` keeps its steps |
| `MIN_DISPLAY_RADIUS_RATIO` | 1e-9 x max(1, max \|coordinate\|): every incident cylinder radius (scale: the edge's definition points and the support origin); every point used for a line's azimuth about a cylinder or cone: its ends by their own projections (except an end on the axis to the frame's rounding, `8 eps max(1, \|coordinate\|)`, at a cone's exact apex: an exactly zero radius and the end within `model.tolerance.length` of the apex point, axial position included, round 4; elsewhere, and both ends on it, refused) and its interior closest approach `\|P' x D'\| / \|D'\|` (scale: the edge's definition points, i.e. the vertices and the line's ends, and the support origin: an evaluated point `a + t (b - a)` is rounded at the scale of the ends, round 3); a cylinder or cone arc's closest approach (one rule for both) (a lower bound, `sigma_min(N) min \|z + w\|`, each candidate evaluated directly: `\| \|z\| - 1 \|` at the analytic critical angle, `\|z + w\|` at the ends; scale: the edge's definition points and `max(1, \|stored angle\|) sigma_max(M')`, the evaluator's rounding of `start + sweep t` moving the projected point by eps times that) | an azimuth computed from coordinates of size \|p\| at distance r from the axis is rounded by about eps \|p\| / r: at the bound 2.2e-7 rad, below the smallest display angle. Measured without the stored-angle term: a .5 circle 2e-9 from a frustum's axis at 999994 rad turns 0.116 rad per interval at `angle = .1` |
| rounding share | the edge's rounding allowance (`display_rounding_allowance`, below) at most 2 % of `tol.angle` (else `OUT_OF_DOMAIN`) and of the sag allowance (else `EDGE_UNSUPPORTED`, the unresolvable-sag code) | round 3: the allowance must be small against what is asked. It binds only near the other bounds: at the radius bound the azimuth term is about `64 eps / 1e-9 = 1.4e-5` rad, so angles below about 7e-4 rad are refused there; at coordinates 1e9 the sag term is about 5e-6, so sags below about 2.7e-4 are refused there (a test at sag 3e-5 there moved to 3e-4). Over the corpus the largest share is 8.7e-8 (angle) and 2.0e-6 (sag) |
| `MAX_DISPLAY_ELLIPSE_ASPECT` | 1e6: a non-circular ellipse's aspect `a / b` (its principal semi-axes, the singular values of `[u v]`) times `max(1, \|start\|, \|start + sweep\|)`; and, for a circle or an ellipse, the same product for its PROJECTION across every incident cylinder's or cone's axis (the 2x2 frame `M'`, whose aspect can be far above the curve's: an ellipse of aspect 100 tilted almost into the axis direction projects at 1e8) | the tangent direction moves up to `a / b` times faster than the arc angle, and the evaluator rounds the absolute angle `start + sweep t` by eps \|theta\|: the tangent error is at most `1e6 x eps = 2.2e-10` rad, four orders below the smallest display angle. Measured with `a / b <= 1e6` alone: aspect 1e6 at 999994 rad turns 10 % over `angle = 1e-3` per interval, local divisions notwithstanding. The azimuth is resolved through `M'^-1` and back (`chi`, then the polar part of `M'`), which stretches an angle error by up to `aspect(M')`: within the bound it is at most `1e6 x eps = 2.2e-10` rad. Measured outside it: the 1e8 projection above turned 0.266 rad on one interval at `angle = .1` |

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
axis_floor)`: inside the declared domain, every emitted interval's tangent turns, and every
incident cylinder's or cone's azimuth changes, by at most `tol.angle + angle`, and every drawn
chord lies within the sag allowance `+ sag` of its curve. The dense oracle in
`tests/test_display_tessellation_edges.py` checks exactly this, with no other slack, and checks
that the allowance is at most 2 % (which `edge_display_samples` enforces). `axis_floor` is
`MIN_DISPLAY_RADIUS_RATIO x scale` (the edge's coordinate scale): the azimuth is defined and
measured at points at least that far from every axis (only a cone's apex is nearer).

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
bound (above); over the corpus at most 8.7e-8 of the angle and 2.0e-6 of the sag.

## Stability within the domain (round-2 class audit)

The class audited: a float rounding or cancellation, at an admitted scale, that breaks a
closed-form bound or raises. Every formula of the module, and why it is stable inside the
declared domain (or which bound makes it so):

| Formula | Form used | Why stable within the domain |
|---|---|---|
| counts `ceil(span / step)` | quotient checked finite and `<= 4096` before any division is built; one fewer only when `span / (n - 1) <= step (1 + 4 eps)` | `step >= 1e-6`; the span is a difference of computed angles (4 eps relative) |
| `_AngleMap` determinant, inverse | `det` of the 2x2 frame, refused when `|det| <= 1e-12 |M|^2`; inverse = adjugate / det | relative error `eps x aspect`, at most 2.2e-10 by the aspect bound (curve and projection) |
| singular values `sigma_max`, `sigma_min`, aspect | `sigma_max = (hypot(m00 + m11, m10 - m01) + hypot(m00 - m11, m01 + m10)) / 2`, `sigma_min = det / sigma_max` | sums of squares and a quotient: no cancellation (the former `sqrt(F^2 - 4 det^2)` cancelled for near-circles) |
| polar angle `alpha` | `atan2(m10 - m01, m00 + m11)` | the pair's norm is `>= sqrt(2 det) > 0`; atan2 is relative-accurate |
| `z = N^-1 c'` | adjugate product | `|z|` large on a thin projection: relative error `eps x aspect(M')` (projected aspect bound) |
| sign of `g = 1 + z.w` | `1 + |z| cos(x - arg z)` at the ends and interior extremes | a decision: an end within `16 eps (1 + |z|)` counts as 0 only when the hidden turnback is below eps rad; an interior `g` near 0 means `|z|` near 1, so the closest approach `sigma_min ||z| - 1|` is below the radius bound and the edge is refused (fail closed) |
| closest approach (conic) | `det / sigma_max x min(hypot(z0 + cos x, z1 + sin x), ||z| - 1|)`, each evaluated directly | absolute error `eps (1 + |z|)` of a distance, far below the 1e-9 radius ratio; the scale includes `max(1, |stored angle|) sigma_max(M')` for cylinders and cones alike |
| forward azimuth `angle(theta)` | `chi = x + atan2(cross(w, q), w.q)`, `Phi = chi + alpha + atan2(cross(e, S e), e.S e)` | `e.S e >= sigma_min > 0`; `cross(w, q)` subtracts terms of size `|z|` and `S` has aspect `aspect(M')`: the error is up to a few `eps x aspect(M') x theta*` (round 3: 1.45e-10 rad at aspect 990099), covered by the allowance's `32 eps aspect(M') theta*` and at most 7.1e-9 rad by the projected aspect bound |
| ray inverse `solve(phi)` | `h = cross(e, z)`, `w = sign(g) sqrt((1 - h)(1 + h)) e - h e_perp` | nothing of size `|z|^2` cancels (was `(z.e)^2 - |z|^2 + 1` and `lambda e - z`); its remaining error is the forward map's, in the same allowance term |
| line projection | `A' = P(a - o)`, `B' = P(b - o)` (each end's own), `D' = P(b - a)` | no reconstruction `A' + D'` (round 3: it cancelled a 1.5e-9 end radius next to a 2e7 start); each rounds by `4 eps |p - o|`, `D'` by `4 eps |b - a|`: the allowance's `G` |
| line on-axis, collinearity | radius `<= 8 eps max(1, |coordinates|)`, only at a cone's apex; both `|A' x D'| <= 8 eps |A'| |D'|` and `|B' x D'| <= 8 eps |B'| |D'|` (`D' = 0` included) | decisions at the frame's rounding; when the two estimates disagree the line is divided (conservative); a near-collinear line that misses the test is divided generically (span below one step: no cut) or refused by its closest approach (fail closed); the constant azimuth's residual turning `2 G / r` is in the allowance |
| line closest approach | both ends' estimates `t* = -(A'.D') / |D'|^2`, `1 - (B'.D') / |D'|^2`, distance `|P' x D'| / |D'|`, before any end shortcut | `|D'| > 0` on this path (guarded); interior when either estimate is inside beyond its ends' rounding and its own error; the smaller distance is checked (conservative) |
| line cut `t = -cross(A', e) / cross(D', e)` | direct | `|cross(D', e)| = |D'| h / |p|`; the cut's azimuth is off by at most `G / r` (in the allowance); a zero denominator is refused |
| circle classification | `|u| = |v|` and `u.v = 0` to 8 eps relative | the circle path's uniform steps and radius `max(|u|, |v|)` are then exact to 16 eps relative (was 1e-12: a 9e-13 skew overshot by 1.8e-13 rad and 2.3e-8 sag) |
| circle sag count | `4 asin(sqrt(sag / (2 rho)))`, allowance `< 2 rho` | no cancellation (`2 rho sin^2(delta / 4)`, not `1 - cos`); exact for the analytic chord, the evaluated chord ends moving by `E` (the allowance's `3 E`, round 3) |
| local frame `u'`, `v'` | one rotation by the stored start | no absolute angle formed; rounding `eps (|u| + |v|)` |
| in-plane matrix (Gram-Schmidt) | `|u'|`, `v'.e1`, `|v' - (v'.e1) e1|` | absolute error `eps |M|`: tangent-angle error `eps x aspect <= 2.2e-10` rad by the aspect bound |
| ellipse sag | `2 sin^2(delta / 4) |u x v| / |T(m)|`, raised by `(1 + 8 eps x aspect)` before comparing | `|u x v|` and `|T(m)|` of a skew frame round by a relative `eps x aspect` each; the raise makes the comparison fail closed; `|T(m)| = 0` is refused; the evaluated chord ends move by `E` (the allowance's `3 E`, round 3) |
| parameters `s / sweep`, union gap `/ |sweep|` | direct | `sweep = 0` refused |
| union drop rule | `|angle(k) - angle(v)| <= 64 eps max(1, |angle|)` | the division angles are local to the edge |
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
