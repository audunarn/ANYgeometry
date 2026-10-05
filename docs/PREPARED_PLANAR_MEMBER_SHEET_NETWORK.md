# Prepared planar Member/Sheet network

`query_prepared_planar_member_sheet_network(model, current_joint_edge_id)` returns
a distinct immutable `PreparedPlanarMemberSheetNetwork` owner receipt for a whole
connected Plane/literal-Straight document. The existing boundary-only receipts
keep their existing admission rules.

The receipt accounts for all Sheet owners, Parts, Member chains, edge-local
source point Attachments, generated Member overlap and point Junctions, interior
carrier edges, and their endpoint vertices. Every authored and current face must have a
selected Sheet owner and appear in the component records; unrelated unowned
faces refuse rather than disappearing from the whole-document receipt.
Original Member carriers must lie in
the union of selected authored Sheet materials. Exact rational polygon boundary
events prove every carrier interval, including concave outer loops and holes;
display sampling and orphan faces cannot establish material membership.

Public original-edge ancestry remains required. Straight current carriers are
certified restrictions of original carriers under their unchanged recorded
tolerances. Exact endpoint residual bounds extend to full affine intervals by
convexity. Current split Sheet trim vertices are projected into their actual
bound Plane charts only after their support residuals satisfy the same owner
tolerances; original authored boundary supports remain exact.

Point contacts are independently reconstructed from original straight carriers
and owned authored Plane materials. Each contact binds a canonical current
Member endpoint vertex and persisted global station occurrence. At each vertex,
the producer's union of transverse Sheet contacts and Member station occurrences
determines the complete Cartesian point-contact inventory. Every supplied point
Attachment must prove its original material, declared current UV, named child
trim, identical incident vertex, source/current station residual, and owner
sets. Exact source-to-current, UV-to-current, and source-to-UV squared residual
bounds each satisfy the same unchanged owner tolerance, preventing accumulated
pairwise error from moving the declared contact farther from its source anchor.
Malformed contact ranges and degenerate projected segments refuse with typed
geometry errors. Omissions, duplicate occurrences, grouped spatially separate contacts, and
nearby but topologically separate vertices refuse. Overlap inventory is derived
from all live MemberEdgeUses with face incidence, and each overlap Junction must
name exactly every live Sheet owner of its edge. Each generated overlap covers
the full persisted Member use range and full target edge interval `[0, 1]`;
coherently narrowed Attachment and Junction ranges cannot omit carrier portions.

`validate_prepared_planar_member_sheet_network_binding(model, receipt)` rederives
the complete receipt and pins the immutable supplied definition before and after
callbacks and model guards. Queries and validation do not mutate the model.
Cancellation, stale revisions and changed receipt definitions refuse.

This receipt does not qualify beam discretization, load transfer, solver use,
external reference transfer, or publication. Curved Members, nonplanar supports,
original Junction remapping, orientation references, point-range boundary
Attachments, and other Member attachment semantics remain reserved and refuse.
There is no schema/version or operand-count acceptance cap. Focused tests cover
strip10, hub4 and hub10; these tests establish development behavior only and do
not authorize larger qualification campaigns or meshing.
