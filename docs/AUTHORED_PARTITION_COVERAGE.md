# Complete straight planar authored partitions

`validate_prepared_authored_face_partition(model, correspondence,
child_triangles_uv, *, cancellation_check=None)` is root-exported. The mapping
must contain **every** authenticated current descendant ID, exactly once. Values
are finite real `(n, 3, 2)` triangle arrays in the **original** UV coordinates.
Success returns `None`; a missing/unsupported/incomplete proof raises
`GeometryError`. An empty child batch cannot cover a positive-area child.

Per-cell containment alone admits an incomplete mesh. The portable split-left
pressure case has a 4-by-4 original face, left area 12 and right area 4. Two
left-side triangles fit their child but leave the right side unmeshed. This API
requires cells for both children and proves complete coverage separately from
mere containment.

The initial scope is deliberately narrow: the original and every literal child
must use exactly coplanar Plane supports and one simple straight outer loop.
Different affine frames and orientations are allowed. Holes, curved supports or
trims, explicit parameterizations and unresolved support correspondence refuse.
Existing sampled topology checks do not establish exact regularity for arbitrary
curved loops. This addition makes no acceptance claim for those families.

The proof checks exact rational closure, distinct/nonzero polygon edges, no
nonadjacent segment contact/crossing and no adjacent backtracking. Every whole
closed cell must lie inside both the original material and its assigned literal
child. The absolute rational cell-area sum must equal each literal child's
shoelace area; the sum of child areas must equal the original area. All cell
interiors must be disjoint, using exact triangle separating axes with a
deterministic rational bounding-box sweep. Boundary contacts are allowed.

Simple polygon domains and finite triangle unions are regular closed sets.
Containment, equal area and disjoint interiors therefore prove complete closed
material equality of the cells, every literal child and the original domain.
The common original affine plane chart has a nonzero constant area Jacobian,
so exact UV measure equality also preserves physical material area. Equal
areas **without** regularity, containment and disjointness do not suffice.

Input arrays and all literal domains are detached and authenticated before any
cancellation callback. Input conversion may execute caller code and is followed
by reauthentication. Source freshness is checked again before success. Queries
do not mutate model state. Cancellation or a caller's exhausted-work exception
returns no successful proof. No count cap, renewed diagnostic budget or new
resource allowance is introduced. Worst-case work can remain quadratic;
performance at 100/1,000 operands is unmeasured.

The proof does not certify edge/node conformity, required joint stations,
attachments or FEM references, quality, high-order elements, complete project
semantics or publication permission. ANYmesher must retain its independent
owner/project/closure bindings, source XYZ and station chains, work ledger,
quality/solver gates and atomic publication. It must validate these exact cells
and current references immediately before publication. No schema, version or
release change is made.

`examples/authored_partition_coverage_handoff.py` demonstrates complete coverage
and the incomplete left-only refusal. Source regressions exercise permutation,
repeat queries, stale/callback mutation, explicit family refusals and cancellation.
An independent rational polygon-clipping oracle checks separating axes and
candidate completeness; display samples are not the oracle. These are contract
checks, not accepted large/mixed meshing.
