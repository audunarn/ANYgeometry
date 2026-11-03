# R10 correctness and regression repairs

These changes preserve the existing numerical tolerances, depth limits and
piece/predicate budgets. Focused evidence does not establish a full CI pass or
N2c parallel/performance qualification.

## Exact quadratic exclusion at the existing depth boundary

`branch_predicates._quadratic_affine_disjoint` accepts exact `Fraction` tensor
Bernstein coefficients only, converts patches of total degree at most two to
power coefficients, and requires the second equation to be affine. Eliminating
one coordinate gives `q(t) = A t² + B t + C` using rational arithmetic.

For `A != 0` and `D = B² - 4AC < 0`,

```
4 A q(t) = (2 A t + B)² - D > 0.
```

Thus there is no common real zero, including outside the parameter patch.
A nonzero constant also excludes a zero. Unsupported coefficients/degrees,
linear roots, contacts, crossings and identically zero reductions receive no
certificate from this helper. Rational-algebra error is zero. This concerns the
existing mathematical represented curve; its existing numerical world-error
contract remains unchanged.

The helper runs only after all old successful classification branches, at the
unchanged default depth 48, where the old engine would otherwise refuse. It does
not relax an explicitly reduced depth budget. A previously successful call never
needed this new branch, so its successful classification/accumulation path is
unchanged. Five ordinary root/error controls were compared with the original
source in both execution orders, including raw IEEE world bounds and exact
rational intervals. This is scoped identity evidence, not a blanket kernel claim.

## Finite straight segments in mixed-loop containment

`plane_section_plate._contains` computes ray parameter `s` and segment parameter
`r` by exact rational determinants. An intersection with `r < 0` or `r > 1`
belongs to the supporting-line extension, not the finite segment. It is excluded
before testing contact at `s == 0`. Actual segment/vertex contacts retain the
original refusal. Curved-edge checks, ray slopes and all limits are unchanged.
The added algebra has zero rounding error; it introduces no geometric movement
or tolerance. Exact endpoint/contact negatives and the original public hole
case remain required. The owner separately authorized only the two matching
host/constructor source-pin updates for this reviewed file.

## Independent quadratic reference oracle

The CH3 test oracle uses the closed-form quadratic solution, selecting the
larger-magnitude `b +/- sqrt(D)` for the first root and Viete's product for the
second. Linear, constant, zero-polynomial and zero-constant cases are explicit.
It retains the existing 60 guard bits, caller precision and real/domain filters;
results round after the temporary precision scope. Near repeated roots remain
ill-conditioned: residual and Viete tests are not an interval or forward-error
certificate. No production root solver is substituted into the independent
oracle.

## Portable regression preconditions

The explicitly approved fixture migration preserves complete binary64 arithmetic
inputs rather than requiring every CPU/BLAS reduction to reproduce historical
bits. Historical circle definitions are checked exactly; their stored length
approximations have a checked rational error of at most one ULP for those inputs.
That is an error term for the fixture, not a production tolerance. Six genuine
synthetic counterexamples retain old overshoots of 99, 22 and 30 eps in each edge
orientation. Historical 12/10/9-eps cases, the 8-eps failure threshold, derived
bounds, exact endpoint/derivative tests and separate live-producer checks remain.

Fallback-specific unit tests explicitly inject only a preliminary retained-sum
refusal. The helper defaults to no injection. Actual cell geometry, final
conservation/crossing/closure validation, cancellation and refusal assertions
remain real. The unforced control validates the returned geometry independently.
Original test files are retained byte-for-byte under
`historical-planning-tests/r10-cpu-fixtures/`; failed attempts remain evidence.

## Runtime-independent negative-test preconditions

The remote-endpoint joint regression now checks both public-edit atomic refusal
and the original exact `2**-40` corrupted-definition refusal. A public editor's
stronger surface-validity check must not prevent testing the lower-level guard.
The depth regression retains the depth-32 positive and depth-1 refusal, using
`x + y = 999/1000` on `x² + y² = 1/2` for the latter. Its two distinct roots
satisfy `(x-y)² = 1999/10⁶ > 0`; an easy already-isolated root does not establish
a depth exhaustion precondition. Production algorithms and limits are unchanged.
