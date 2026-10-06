# Prepared native material and geometry references

`query_prepared_native_material_reference_scope(model, authored_face_ids)` issues
one immutable, read-only batch receipt after actual `plan_intersections` /
`apply_intersections` preparation. Its selected root IDs are explicit. The binding
validator authenticates the issued selection; callers must separately check that
the selection satisfies their own scope. Legacy preparations without prospective
actual support snapshots refuse this query. No persisted schema/version changes.

The query detaches complete original/current documents, owner lineage, native
support coefficients and edge ancestry before callbacks. Cylinder snapshots
include the actual stored circumferential coefficient omitted by serialization.
One request shares its arithmetic cache, support frames and aggregate configurable
`NativeArcParameterMapPolicy` budget. There are no face/edge/occurrence count caps.
Document fingerprints are constant per batch. Final guards are callback-free;
cancellation, exhausted work or stale binding issues no accepted partial receipt.
Issued outputs use weak ownership seals, so copying or repairing forged output
does not produce an authentic receipt. This is owner authentication, not an
external signature or an external-reference inventory.

For supported Plane trims, exact rational source UV coordinates describe straight
closed loops, simple outer domains and strictly inside, disjoint holes. A bounding
box sweep supplies segment predicate candidates. The census includes every
intersection breakpoint, then tests every open slab cell. It proves
`sum(1_Dchild) = 1_Dsource` away from their boundaries. Material indicators are
positive outer-minus-holes indicators independent of stored traversal; duplicate
interior domains with opposite loops add multiplicity and refuse. Curved Plane
trims remain unqualified.

The initial Cylinder family uses an exactly unit axial vector orthogonal to both
stored radial coefficients, a nonsingular radial frame, positive radius, finite
angular span less than one period, common source/current physical support, axial
straight trims and horizontal three-point circular trims. Each circle strictly
encloses the projection axis; its center need not equal that axis. For a circular
radial trace `k + rho(cos(theta)e1 + sin(theta)e2)`, the angular derivative numerator
is bounded below by `rho^2 - rho*||k|| > 0`. Thus the stored-dot angular projection
has a monotone horizontal chart image, even when refitting changes the circle.

The chart is the existing native projection:
`z=(P-O).a`, `phi=atan2((P-O-z*a).stored_c,(P-O-z*a).stored_r)`.
Endpoint rays are ordered by exact half-plane/cross-product predicates relative
to a source boundary ray, avoiding a global principal-angle seam. All source and
child loops must certify the same branch, close, remain simple and pass the
positive indicator census. Symbolic angle ranks preserve these horizontal/vertical
domain indicators. They are neither angles nor area coordinates. Common injective
support plus pointwise indicator equality implies equal physical material; no
numerical density integration or display samples establish acceptance.

Retained density bounds are explicitly for harmonic coordinates
`x=R*alpha,y=z`, with `alpha` the native harmonic angle. They are not density bounds
for stored-dot `phi` or symbolic ranks. For example `r=X,c=2Y,a=Z` gives
`phi=atan2(4*sin(alpha),cos(alpha))`; the physical density with respect to phi is
one half at alpha zero, unlike the harmonic density there.

Document Arc trim semantics use the exact three stored points, directed via span
and stored endpoints. Captured harmonic Arc functions and runtime float/libm
evaluation are separate. Bounded native Arc parameter-map evidence never implies
exact parameter preservation, document circle identity or floating-evaluation
preservation. A material chart proof can qualify even while an existing coarse
geometric boundary correspondence refuses; that original refusal is retained and
does not gain admission through this receipt.

The complete geometry inventory includes boundaries, interior constraints,
start/end/via/control vertex dependencies, isolated vertices, Members,
MemberEdgeUses, Attachments, Junctions, Parts, Sheets, FaceUses and Coedges. A
dependency graph includes selected and outside roots through shared carrier
vertices and structural owners. Native member/coedge maps must authenticate the
intended original model/revision/checksum, be exact or bounded, retain payload and
traversal, and cover each original use completely without overlap or gaps.
The bounded native-function residual and a MemberEdgeUse parent-range parameter
residual are separate quantities. Their individual tolerances do not establish a
composed physical referenced-coordinate error within the length tolerance; that
claim additionally needs the appropriate Jacobian/composition and evaluator
roundoff bounds. The receipt retains these obligations without granting admission.

FaceUse preservation additionally requires the physical native normal orientation
relation; equal enums cannot preserve a mirrored Plane basis or reversed Cylinder
sweep/height. Literal face references require supported, identical actual native
snapshots as well as complete edge dependency identity. Unsupported snapshots
are absence of proof. Unchanged member-source attachments and authored Sheet
joint attachments require the complete corresponding owner maps.

Preparation-created `SHEET_ON_JOINT` / `SHEET_JOINT` records can qualify as
`derived_exact_joint_incidence`: their exact public creation contract, complete
actual Sheet incidence, all full `[0,1]` edge attachments, supported material
trace, connect intent, exact evidence and owner coverage must hold. Unknown,
extra, missing or wrong-Sheet records refuse. This proves current derived joint
incidence; it does not remap original parameters. Original split/remapped
attachments still require a separate semantic map and presently refuse.

`document_material_qualified` and `geometry_native_reference_maps_qualified` are
separate outcomes. Opaque metadata/groups/tags retain unknown consumer semantics;
consumer-owned external references remain an explicit obligation. Floating
evaluation preservation and meshing permission stay false. General tilted/radially
drifting Cylinder trims, full-period/concave angular branches, curved Plane loops,
and nonparallel cubic families remain explicit supported-family gaps.
