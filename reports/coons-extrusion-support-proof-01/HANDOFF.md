# Coons / Bezier extrusion support proof — implemented / unverified

State: implemented but UNVERIFIED; execution prohibited. OpenAI fallback owns the
bounded slice after cached worker reliability failures. No Python/test/geometry
evaluation, native, Git or worker execution is authorized or performed.

Proposed representation: prospectively capture an implicit Coons face's four
oriented polynomial boundary coefficient arrays and corner layout directly from
stored topology; capture an ExtrudedSurface's raw Bezier controls, vector and
parameter ranges. Explicit sampled Coons and multi-edge/nonpolynomial sides
remain unsupported (their unsupported snapshot must not masquerade as a proof).

Exact equation: in logical directions, bottom B(u), top T(u), right R(v), left
L(v). Prove T=B+D coefficientwise, R=B(1)+vD, L=B(0)+vD with exact endpoints.
Then the tensor Coons blend reduces identically to B(u)+vD. No evaluator samples,
coordinate snapping, idealized carrier or binary64 normalization is used.

Child mapping: prove child directrix equals B(t) or B(1-t), plus a constant
multiple of D; prove child extrusion vector equals a nonzero exact multiple of
D. Compose stored ranges into the exact affine root UV map. Require mapped
rectangle inside [0,1]^2 and nonzero determinant; retain its orientation sign.
Power coefficients permit exact degree elevation but rounded mismatches refuse.
Strict same-sign projected Bernstein derivative coefficients perpendicular to D
provide a sufficient regularity/injectivity certificate. This is deliberately
not a complete recognizer of all equivalent parametric surfaces.

Integration: add captured support kinds, a dedicated pure proof helper, and
support-only evidence inside the existing native material/reference row. Keep
document-material/reference/meshing predicates false for this unsupported trim
family. Equality of supports and parameter orientation is NOT proof of trimmed
partition conservation, semantic reference remapping or floating evaluator
parity. No schema or root-export change is proposed.

Planned tests are source-only: exact cubic, reversed boundary storage/ranges,
degree elevation, exact affine world coefficients, support/closure mismatch,
singularity/noninjectivity conservative refusal, missing legacy capture,
query/callback binding and unchanged material/reference refusal. Final handoff
will state implemented/unverified status and exact owned source hashes.

## Final delivered slice

Changed existing files:

- `src/anygeometry/native_support_snapshots.py`: prospective detached implicit
  Coons boundary/corner capture and Bezier extrusion declared/raw controls,
  vectors and ranges. Explicit sampled Coons is captured under a different,
  unsupported kind; it is never interpreted as Bernstein data.
- `src/anygeometry/native_material_reference_scope.py`: the existing public
  query now emits `support_only` rows when the exact proof succeeds, retaining
  signed parameter maps in evidence. Physical support is true only for that
  proof; document material stays false with an explicit missing-partition
  refusal. Selection of these families forces native-reference qualification
  false. The historical orientation-remap predicates remain unchanged.

New files:

- `src/anygeometry/polynomial_extrusion_support.py`: rational coefficient
  compilation, exact Coons cancellation identity, sufficient rank/injectivity
  proof, same/reversed carrier and ruling-scale correspondence, exact contained
  chart rectangle and signed determinant. Work loops use the caller's existing
  aggregate charge callback; no pool or threshold is added.
- `tests/test_polynomial_extrusion_support.py`: independent coefficient/range/
  orientation/refusal definitions and real small public preparation/query
  regression definitions, all unexecuted.
- `docs/POLYNOMIAL_EXTRUSION_SUPPORT.md`: supported/refused scope, equations,
  snapshot/parameter semantics and explicit remaining material/reference gaps.

The helper permits degree elevation through exact polynomial identity, but
does not attempt general affine directrix reparameterization or a complete
regular-surface recognizer. Exact transformed world coefficients can qualify;
rounded near-equality cannot. Only closed unit-chart parameter correspondence
is claimed, not extrapolated Coons evaluation or finite evaluator parity.

## Checks, failures and evidence

No Python imports, pytest, compilation, numerical geometry evaluation, models,
native code, Git commands or workers ran. Consequently there is NO pass count,
installed validation, performance claim or executed actual-wall acceptance.
Verification consists only of source/formula/caller review and file hash reads.
Several bounded PowerShell source reads accidentally used word-valued `-First`
arguments (`fifty`/`sixty`) and failed; later integer-range reads supplied the
needed source. These were inspection-command mistakes, not product/test runs.

Authored regression definitions cover exact generated cubic and independent
power formulas; stored-edge reversal; cropped/reversed ranges; negative ruling;
degree elevation; affine world coefficients; tiny rounded/top/connector changes;
wrong corners/multiple sides; raw/declarative mismatch; invalid/legacy ranges;
nonfinite data; zero ruling and unproved injectivity; explicit sampled Coons;
charge exhaustion/callback identity; actual prospective capture; public caller
support-only flags; raw same-revision mutations, tamper and late callbacks.
Existing small cubic-fragmentation fixture is reused by the future query tests;
it has NOT been constructed/evaluated in this assignment.

Root's source-review/selected checks remain necessary. No edits were made to
whole-cell/branch proof gates, reference algorithms, root exports, schema,
tolerances, external consumer files or other dirty evidence.

## Scope and next unresolved proof

This is foundational authored/current support qualification, not the sole
current mesh blocker. Root noted that proper authored-input grouping can use
the existing CURRENT region API without this new original-partition proof.
This slice does not diagnose or repair the consumer's physical quality failure.

The actual wall's original/current positive trimmed-material partition still
needs a proof for actual BQC boundaries and rounded fragment provenance. Signed
support correspondence cannot replace that proof or qualify reference remaps,
FaceUse sense transfer, exterior loads, station movement or mesh publication.
Legacy snapshots lacking source coefficients cannot recover them from current
geometry. Source-only scope is frozen for independent review; root retains
scientific adoption, execution authority, integration and acceptance.

## Frozen SHA-256

- `native_support_snapshots.py`: `73131774d496f8102fda6f1e990f34807a23417ca9db43d3140a40bbc32bfec7`
- `native_material_reference_scope.py`: `e9e03c77e53498396b9411bf7c2b61302f85e427d3842a0bc8c41b14b969a7bc`
- `polynomial_extrusion_support.py`: `80998a2820b9a0ebd49493af26cb841eac5d23be90845266281b523d20a1b978`
- `test_polynomial_extrusion_support.py`: `2aa002714aaa808060b842b7b242fce15d430f06e2d78b93d3f2a178c2618703`
- `POLYNOMIAL_EXTRUSION_SUPPORT.md`: `841556e7113c8945a3fd32d2536e0afc3b4d921897d2f66fab5bce21f622c0a1`
