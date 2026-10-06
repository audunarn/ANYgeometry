# Stored polynomial extrusion support correspondence

This is additive **support-only** evidence in
`query_prepared_native_material_reference_scope`. It does not qualify an
original/current trimmed material partition, reference remapping, binary64
evaluation preservation, or meshing. Existing Plane/Cylinder material proofs
and all whole-cell branch-trim refusals remain separate and unchanged.

The prospective native snapshot now retains two additional support kinds:

- `coons_polynomial_v1`: implicit topology-backed Coons corner indices and each
  original loop edge's ID, stored direction, Straight/Spline kind and complete
  stored point coefficients. This captures the pre-operation definition rather
  than trying to reconstruct a retired source from descendants.
- `bezier_extrusion_v1`: declared and actual evaluated Bezier arrays, declared
  and actual ruling vectors, and stored parameter ranges. Raw arrays/vector
  changes therefore cannot hide behind unchanged serialized public fields.

Explicit Coons arrays remain piecewise-linear samples, not Bernstein controls.
They are captured as `coons_sampled` but not proved by this slice. Nonpolynomial
or multi-edge topology sides, explicit parameterizations and absent legacy
source coefficients do not acquire a polynomial proof. No document schema or
root export is added. Normal qualified capture/composition/clone/corner-binding
paths carry these tuples through the existing prepared-owner snapshots.

For four single-edge sides with corner indices `(0,1,2,3)`, let bottom/top run
in increasing logical u and left/right in increasing logical v. Stored edge
directions are applied before reversing the loop's top/left sides. Exact
rational conversion of every binary64 coefficient proves

`top(u)=bottom(u)+D`, `right(v)=bottom(1)+vD`,
`left(v)=bottom(0)+vD`.

These identities and exact corner closure cancel the Coons bilinear correction,
leaving `S(u,v)=B(u)+vD` on the closed unit square. Power-coefficient comparison
admits exact degree elevation. A rounded coefficient mismatch, even if within
the historical world tolerance, refuses this exact support proof. It does not
change that tolerance or historical geometry classification.

A child Bezier extrusion must have polynomial carrier `B(t)` or `B(1-t)`, plus
a constant multiple of D, and ruling vector equal to a nonzero multiple of D.
Stored u/v ranges compose into an exact diagonal affine child-unit-UV to
original-unit-UV map. Its image must remain inside the original unit square;
the determinant and orientation sign are retained explicitly. Reversed ranges
and negative ruling directions are not mistaken for forward orientation.
Arbitrary shifted/scaled directrix parameter substitutions are not recognized
in this bounded implementation. A world affine transform is supported only
when its *resulting stored coefficients* satisfy the same exact identities;
transform history never substitutes for a coefficient proof.

Regularity and injectivity use a sufficient exact certificate. For one Cartesian
axis e, set `q=e*(D·D)-D*D_e`. Then `q·D=0`. If all degree-scaled consecutive
Bernstein-control differences dotted with q have the same strict sign,
`q·B(t)` is strictly monotone. Thus B'(t) is never parallel to D, and equality
`B(t)+sD=B(r)+zD` forces t=r and s=z. Unproved cases refuse; this is not a
complete test for all regular extrusions. Ranges extrapolating the captured
directrix beyond `[0,1]` are unsupported.

The exact relation `S_child = S_original ∘ map` implies corresponding tangent
cross products differ by the map determinant. This is support/chart orientation
evidence, not a comparison of FaceUse senses or an authorization to remap
orientation-dependent references. Finite floating implementations, topology
chain length normalization, inversion and sampling have separate obligations.

Successful rows have `classification='support_only'`,
`physical_support_qualified=True`, `document_material_qualified=False`, an
explicit missing-partition refusal, and packed rational evidence in
`domain_evidence_json['polynomial_support_correspondence']`. A mismatch returns
the existing refused-row form. The complete receipt's
`geometry_native_reference_maps_qualified` is forced false when this support
family is selected, and `meshing_permitted` remains false. No orientation or
reference permission predicate is opened merely by the new support evidence.

Polynomial coefficient and derivative loops charge the existing aggregate
owner proof work pool; they create no new budget. Callback failures retain
their identity through the existing query wrapper. Entry snapshots are detached
before proof callbacks, and the existing final scope/native-support/edge-binding
checks still guard receipt issuance and later validation.

Remaining work for the cubic-wall material question includes an exact positive
trimmed-domain partition census, actual BQC boundary/chart correspondence,
rounded fragment provenance where needed, and independently qualified native
reference maps. Equal supports or an area sum cannot supply those missing
proofs. This source-only slice has no executed test or model acceptance claim.

This is foundational authored/current qualification work, not a prerequisite
for every use of the existing current material-region API. A consumer retaining
its proper authored grouping may already coalesce qualified CURRENT extrusion
seams under that API's constraints. This new support proof neither disables
that route nor claims to resolve its meshing quality or input-grouping issues.
