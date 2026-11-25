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
| Edge curves | `Straight`, `LineRange` (lines); `Arc` (circle, via `model.arc_frame`); `EllipticArc` (a circle when `|u| = |v|` and `u.v = 0` to 1e-12 relative, else an ellipse; `u`, `v` need not be orthogonal) | every other curve (`Spline`, `NurbsCurve`, `CurveOnSurface`, intersection curves, ...): `EDGE_UNSUPPORTED` |

An edge used by no face (a lone curve) has no supports and is divided by its tangent alone.
Supports are checked before the curve type.

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
   - Lines: the ends' projections `A'`, `B'` relative to the axis point. An end exactly on the
     axis (`A' = 0`, a cone's exact apex): the projection is a ray from the axis point, constant
     azimuth, no division. `A'`, `B'` collinear with the axis point (`|A' x B'| <= 8 eps |A'| |B'|`)
     on the same side: constant azimuth (cylinder rulings, cone generators), no division; on
     opposite sides: the line crosses the axis, `EDGE_UNSUPPORTED`. Otherwise `A' + t D'`
     (`cross(A' + t D', D')` constant) has a monotone azimuth spanning less than pi: equal azimuth
     steps, each `t = -cross(A', e) / cross(D', e)`. There is no apex-proximity shortcut: a line
     from near an apex is divided like any other.
   - Circles and ellipses: `c' + M' w(theta) = N (z + w(x))` (`N = M' J` with det > 0, `z = N^-1 c'`).
     `arg(z + w)` is monotone while `g = 1 + z.w` keeps its sign (its extremes over the interval
     are found in closed form): strictly inside the interval, non-strictly at its ends (`g = 0` at
     an end is the curve tangent there to a ray from the axis: a stationary end of a monotone
     azimuth). An end value within rounding of 0 (`|g| <= 16 eps (1 + |z|)`) of the wrong sign
     counts as 0 only when the turnback it could hide, about `g^2 / (2 |z + w|^3)`, is below eps
     rad. A sign change inside, or the projection reaching the axis, is `EDGE_UNSUPPORTED`. Equal azimuth steps, each solved exactly (`A cos(theta) + B sin(theta) = C`
     on the monotone branch: the ray root `lambda = z.e + sign(g) sqrt((z.e)^2 - |z|^2 + 1)`).
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
4. A curved edge whose single chord could leave it by more than `model.tolerance.surface_residual`
   gets at least two intervals.

The union is one sort and one pass. Every tangent-division value is kept. A value `v` of a support
division within 64 ulps of the stored angle (in edge-parameter units: closer parameters evaluate
to the same theta) of a kept value `k` is dropped only when that division's own angle at `k`
equals its angle at `v` to rounding (`64 eps max(1, |angle|)`, the angles being local to the
edge), so every interval still lies inside one interval of that division up to that rounding;
otherwise it is kept (and a cut too close to evaluate apart is refused). Each division's count is
checked against the cap before it is built; more than 4096 intervals is `EDGE_UNSUPPORTED`.

Measured at the Fine step (7.5 deg): a tripod mitre ellipse takes 48 intervals, the tripod
circles 24, `winding_fixture`'s two ellipses 45 and 43; the lone 20:1 half ellipse 24 (no sag
allowance), 78 at sag 1e-3 and 230 at sag 1e-4 (unchanged by the round-1 fixes: every count over
the corpus at both steps, with and without sag 1e-4, is the same; parameters moved by at most
5.6e-14).

## Declared numeric domain (owner decision 2026-11-25)

Checked before any division. Inside the domain the bounds above hold to rounding; outside it the
call raises `DisplayDomainRefused` (a `DisplayTrimRefused` with code `OUT_OF_DOMAIN`).

| Bound | Value, applied to | Why |
|---|---|---|
| `MAX_DISPLAY_COORDINATE` | 1e9 model units: vertex positions, arc centre and (`EllipticArc`) centre + u, centre + v, an `Arc`'s via vertex, `LineRange` carrier endpoints and a line's evaluated ends, every incident support's origin (and a plane's u, v vectors), every emitted point | float spacing 1.2e-7 there; ship and offshore models in m or mm are far inside |
| `MAX_DISPLAY_ANGLE` | 1e6 rad: `EllipticArc` start and start + sweep; cylinder and cone start and start + sweep | float spacing 1.2e-10 rad, four orders below the smallest display angle (1e-6), so `start + sweep t` keeps its steps |
| `MIN_DISPLAY_RADIUS_RATIO` | 1e-9 x max(1, max \|coordinate\|): every incident cylinder radius (scale: the edge's definition points and the support origin); every point used for a line's azimuth about a cylinder or cone: its ends (except an end exactly on the axis) and its closest approach `\|A' x B'\| / \|D'\|` between them (scale: that point and the support origin); a cone arc's closest approach (a lower bound, `sigma_min(N) min \|z + w\|`, each candidate evaluated directly: `\| \|z\| - 1 \|` at the analytic critical angle, `\|z + w\|` at the ends; scale: the edge's definition points and `max(1, \|stored angle\|) sigma_max(M')`, the evaluator's rounding of `start + sweep t` moving the projected point by eps times that) | an azimuth computed from coordinates of size \|p\| at distance r from the axis is rounded by about eps \|p\| / r: at the bound 2.2e-7 rad, below the smallest display angle. Measured without the stored-angle term: a .5 circle 2e-9 from a frustum's axis at 999994 rad turns 0.116 rad per interval at `angle = .1` |
| `MAX_DISPLAY_ELLIPSE_ASPECT` | 1e6: a non-circular ellipse's aspect `a / b` (its principal semi-axes, the singular values of `[u v]`) times `max(1, \|start\|, \|start + sweep\|)` | the tangent direction moves up to `a / b` times faster than the arc angle, and the evaluator rounds the absolute angle `start + sweep t` by eps \|theta\|: the tangent error is at most `1e6 x eps = 2.2e-10` rad, four orders below the smallest display angle. Measured with `a / b <= 1e6` alone: aspect 1e6 at 999994 rad turns 10 % over `angle = 1e-3` per interval, local divisions notwithstanding |

No apex is recognised by proximity: a line end counts as on the axis only when its projection is
exactly 0, and a generator is recognised by the collinearity of its ends with the axis point,
so a frustum's narrow end (both radii positive, however different) is never an apex and a line
from within `model.tolerance.length` of an apex is divided by its azimuth like any other.
An edge id not in the model is a plain `GeometryError` (not a typed refusal).

Inside the domain, a sag allowance at or below 4 eps x max(1, |vertex coordinates|) on a curved
edge is `EDGE_UNSUPPORTED` (unresolvable), as are distinct parameters evaluating to one point and
evaluator errors.

## Refusal codes (`REFUSAL_CODES`)

| Code | Meaning |
|---|---|
| `SUPPORT_UNSUPPORTED` | an incident face's support is not a plane, cylinder or cone, or the face has an explicit parameterization |
| `EDGE_UNSUPPORTED` | the curve is not a line, circle or ellipse; degenerate frame or projection; support azimuth not monotone along the edge or reaching the axis (a line crossing it); unresolvable sag; a vertex farther from its curve than the sag allowance; more than 4096 intervals; lost precision |
| `OUT_OF_DOMAIN` | outside the declared numeric domain (`DisplayDomainRefused`) |
| `TRIM_INVALID`, `TRIM_WINDS_SUPPORT` | reserved for face trims (not raised by edge sampling) |
