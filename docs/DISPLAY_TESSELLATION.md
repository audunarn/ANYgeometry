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

1. **Tangent** (at most `angle` per interval). Lines: none. A curve `c + cos(theta) u + sin(theta) v`
   has the tangent `M w(theta + pi/2)` with `M = [u v]` in an in-plane basis (det > 0) and
   `w(x) = (cos x, sin x)`; its direction angle `psi` is monotone. Circles: uniform in `theta`.
   Ellipses: `n = ceil(|delta psi| / angle - 1e-9)` equal steps of `psi`, each mapped back to
   `theta` exactly: `w ∝ M^-1 (cos psi, sin psi)`, lifted with the polar decomposition
   `M = R(alpha) S` (the angle between `e` and `S e` stays within pi/2).
2. **Support azimuth** (at most `angle` per interval, for each incident cylinder or cone: the
   turning of a cylinder's normal, an upper bound for a cone's). The curve is projected onto the
   plane normal to the axis, in the support's (radial, circumferential) frame.
   - Lines: `A' + t D'`; `cross(A' + t D', D')` is constant, so the azimuth is monotone and spans
     less than pi unless the projection passes through the axis point. Equal azimuth steps, each
     `t = -cross(A', e) / cross(D', e)`. Cylinder rulings (projection a point) and generators
     through a cone's exact apex have constant azimuth: no division.
   - Circles and ellipses: `c' + M' w(theta) = N (z + w(x))` (`N = M' J` with det > 0, `z = N^-1 c'`).
     `arg(z + w)` is monotone while `g = 1 + z.w` keeps its sign (its extremes over the interval
     are found in closed form); a sign change, or the projection reaching the axis, is
     `EDGE_UNSUPPORTED`. Equal azimuth steps, each solved exactly (`A cos(theta) + B sin(theta) = C`
     on the monotone branch: the ray root `lambda = z.e + sign(g) sqrt((z.e)^2 - |z|^2 + 1)`).
     This divides circles off a cylinder's axis (admitted within the model's tolerance) and plane
     sections of cones (the axis at a focus of the projected ellipse) exactly.
3. **Sag** (chord within the allowance). Allowance: `tol.sag`, else the smallest over incident
   cylinders and cones of `r_max (1 - cos(angle / 2))` (cylinder radius, cone's larger radius);
   none without either. Lines: 0.
   - Circle: a chord over a step `delta` leaves the circle by exactly `rho (1 - cos(delta / 2))`,
     so the uniform count is raised to `ceil(|sweep| / (4 asin(sqrt(sag / (2 rho)))))`.
   - Ellipse: sag `<= (L / 2) sin(h)` and `L <= rho_max h` over an interval of tangent turning
     `h <= pi/2`, with `rho_max = a^2 / b` (principal semi-axes `a >= b` of `M`, closed-form SVD).
     *Proof sketch:* the chord's direction is one of the interval's tangent directions (mean value
     theorem), so every tangent is within `h` of it; a point at arc length `s` is at most
     `min(s, L - s) sin h <= (L / 2) sin h` from the chord's line (integrate `sin(psi - psi_c)` from
     the nearer end), and its foot lies on the segment because the projection onto the chord grows
     at `cos(psi - psi_c) >= cos h >= 0`; `L = int rho dpsi <= rho_max h`. The bound grows with `h`,
     so the psi-uniform `n` is doubled until `(rho_max h / 2) sin(h) <= sag` holds.
4. A curved edge whose single chord could leave it by more than `model.tolerance.surface_residual`
   gets at least two intervals.

A parameter of a later division within 64 ulps of the stored angle (in edge-parameter units) of
one already kept is not repeated (the same division point reached twice, up to rounding). More
than 4096 intervals is `EDGE_UNSUPPORTED`.

The ellipse sag bound is conservative (measured on the corpus): the tripod's 6 m mitre ellipses
take 118 intervals at the Fine step (24 for its circles); a lone 20:1 half ellipse needs 2048 at
sag 1e-3 and more than 4096 (refused) at sag 1e-4.

## Declared numeric domain (owner decision 2026-11-25)

Checked before any division. Inside the domain the bounds above hold to rounding; outside it the
call raises `DisplayDomainRefused` (a `DisplayTrimRefused` with code `OUT_OF_DOMAIN`).

| Bound | Value, applied to | Why |
|---|---|---|
| `MAX_DISPLAY_COORDINATE` | 1e9 model units: vertex positions, arc centre and (`EllipticArc`) centre + u, centre + v, an `Arc`'s via vertex, `LineRange` carrier endpoints, every incident support's origin (and a plane's u, v vectors), every emitted point | float spacing 1.2e-7 there; ship and offshore models in m or mm are far inside |
| `MAX_DISPLAY_ANGLE` | 1e6 rad: `EllipticArc` start and start + sweep; cylinder and cone start and start + sweep | float spacing 1.2e-10 rad, four orders below the smallest display angle (1e-6), so `start + sweep t` keeps its steps |
| `MIN_DISPLAY_RADIUS_RATIO` | 1e-9 x max(1, max \|coordinate\|): every incident cylinder radius (scale: the edge's definition points and the support origin); every cone point used for an azimuth: a line's ends and closest approach (scale: that point and the cone origin), an arc's closest approach (a lower bound, `sigma_min(N) min |z + w|`; scale: the edge's definition points) - except a cone's exact apex | an azimuth computed from coordinates of size \|p\| at distance r from the axis is rounded by about eps \|p\| / r: at the bound 2.2e-7 rad, below the smallest display angle |

A cone has an apex only if one of its radii is exactly 0 (the apex is `origin + v height axis`,
`v` = 0 or 1); a vertex is the apex iff it is within `model.tolerance.length` of it. A frustum's
narrow end (both radii positive, however different) is never an apex.

Inside the domain, a sag allowance at or below 4 eps x max(1, |vertex coordinates|) on a curved
edge is `EDGE_UNSUPPORTED` (unresolvable), as are distinct parameters evaluating to one point and
evaluator errors.

## Refusal codes (`REFUSAL_CODES`)

| Code | Meaning |
|---|---|
| `SUPPORT_UNSUPPORTED` | an incident face's support is not a plane, cylinder or cone, or the face has an explicit parameterization |
| `EDGE_UNSUPPORTED` | the curve is not a line, circle or ellipse; degenerate frame or projection; support azimuth not monotone along the edge or reaching the axis; unresolvable sag; more than 4096 intervals; lost precision |
| `OUT_OF_DOMAIN` | outside the declared numeric domain (`DisplayDomainRefused`) |
| `TRIM_INVALID`, `TRIM_WINDS_SUPPORT` | reserved for face trims (not raised by edge sampling) |
