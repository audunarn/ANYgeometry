# Double-fold station fixture

`examples/authored_fold_stations_handoff.py` constructs a cubic Bezier wall
with original support

```
B(t) = (3t, 25/16 - 3t + 3t^2, -1)
D    = (0, 0, 2)
X(t,s) = B(t) + s D
```

The unit pipe has its axis along global x. Substitution into the independent
pipe equation gives `(25/16-3t+3t^2)^2 + (-1+2s)^2 = 1`.
The discriminant vanishes at `t=1/4,3/4`. Between them, the profile y coordinate
stays between `13/16` and `1`; each z branch stays inside one 45-degree pipe
panel. Axial ends, wall ends and internal panel seams do not clip these
branches. Arrangement normalization may split one branch to regularize its
lens; the public prepared model retains a `both_sine` branch on the other.

The focused tests verify the actual prepared branch, original polynomial
coordinates, independently constructed cylinder residual, fold endpoints,
original derivatives and source/coordinate non-mutation. They use source
edge identity and owner parameters, with no inverse-XYZ inference. This adds
observed public fixture coverage of `both_sine` to the contract in
`AUTHORED_SURFACE_EVALUATION.md`; it does not change the production API.

The station contract still lifts evaluated binary64 carrier coordinates.
It does not certify algebraic intersection coordinates, material membership,
curve approximation, global node identities or mesh acceptance. QIC internal
station mapping remains unqualified. Existing resource and consumer quality
gates remain unchanged.
