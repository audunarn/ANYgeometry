The local exact curve-geometry service partitions an already admitted affine CurveOnSurface pcurve at exact preimages of the support knot lines. It does not add a curve family, sampled fit, inverse root search or floating parameter snap.

Admission remains a literal 2D NurbsCurve pcurve on literal NurbsSurface support with existing native/definition checks. The new partition is restricted further to degree1, exactly two controls, four clamped knots and equal positive weights (or implicit unit weights). Other pcurves retain their existing partition/refusal behavior. A LinePath pcurve is not admitted.

For clamped knots `(L,L,H,H)`, controls A,B and valid domain `(alpha,beta)`, the exact pcurve is `q(t)=a+t*d`, where `a=A+(B-A)*(alpha-L)/(H-L)` and `d=(B-A)*(beta-alpha)/(H-L)`. Thus trimmed, nonunit and reversed domains are handled explicitly. Every stored number is lifted to Fraction before this algebra.

A support knot k in axis range `(U0,U1)` has normalized coordinate `(k-U0)/(U1-U0)`. If that pcurve coordinate has nonzero d, its exact preimage is `(normalized-a_i)/d_i`; only roots strictly inside the requested interval join the existing descriptor/pcurve knot anchors. Constant coordinates introduce no isolated root. Sets deduplicate simultaneous crossings.

Each resulting interval still enters the original charged box queue. All old depth, box, active-work, native-state, definition, policy, cancellation, speed and curvature checks remain. No continuity is inferred at a knot: pieces retain the existing one-sided endpoint semantics. Exact structural anchors and subsequent adaptive subdivisions keep their existing distinct accounting conventions.
