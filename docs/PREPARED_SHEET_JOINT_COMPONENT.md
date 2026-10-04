# Prepared Sheet joint component

`query_prepared_sheet_joint_component(model, current_joint_edge_id)` returns an
immutable current dependency receipt. The receipt and its query/binding/selection
APIs are exported from `anygeometry`.
It requires a current all-face `PreparedModelScope` and a declared multi-Sheet
CONNECT Sheet joint. It grants no mesh, parameter remapping, load equivalence,
geometric partition or publication permission.

The fixed point follows every declared joint touching the selected Sheets,
their geometry edges or vertices, including joints reached only through a Sheet.
Every selected Sheet includes all of its FaceUses, coedges, literal faces,
boundary edges and endpoints. All connected owner Sheets are included, without
an owner-count cap. Literal membership is cross-checked against derived owner
indexes on every included edge. Every multi-owner included edge must have a
complete public joint declaration; an undeclared shared edge refuses rather
than silently omitting its outside Sheet. Unrelated components stay outside the
selected records; the complete incoming persisted model remains visible in the
embedded scope.

The qualified family has stable original/current Sheet and Part IDs, original
and current Plane faces with literal Straight boundaries, and full-range EXACT
`sheet_on_joint` attachments. Sheet policy, name and metadata must match the
original, apart from derived membership. Part name and metadata must match.
Touching Members, member Junctions, other Attachments, source-less structural
owners and unproved original Attachment/Junction remapping refuse explicitly.
Previously authored joint links and declarations are qualified only when their
full payloads remain identical on the same unchanged edge definition and endpoint
records. `preserved_joint_attachment_ids` and `preserved_joint_junction_ids`
identify those literal preserved records in the fresh working-model source
namespace. They do not map Project IDs or qualify splitting/rebinding, arbitrary
attachment semantics, property transfer or parameter remapping. Complete Sheet
closure extraction retains its Sheet junction declarations; a reopened design
can therefore undergo a new witnessed extraction and preparation without
reconstructing prior preparation authority or pruning existing joints.
Nonempty qualified Attachment lineage refuses; otherwise unrelated attachments
whose lineage alone touches the component also refuse. Original incoming
attachment/junction references are followed to a fixed point before preserved
relation qualification; their disappearance from the current snapshot is not
evidence that the source design had no such dependency.

`occurrence_mapping_qualified=True` has a precise separate meaning: for every
`(Sheet ID, authored root face ID)` there is exactly one original FaceUse, and
exactly one current FaceUse for every authenticated descendant of that root.
Each retains the original orientation and metadata. `occurrence_correspondence`
exposes the Sheet ID, authored root ID, original FaceUse ID and all current
FaceUse IDs. This is a root/Sheet occurrence proof, **not FaceUse handle ancestry**.
All raw current coedge IDs, orientations, definitions and attachment parameter
ranges are retained in the detached records.

`semantic_mapping_qualified=False` remains independent of that occurrence proof.
Groups, tags and arbitrary metadata, feature or extension meanings are bound
and visible, not certified as equivalent property or external-reference
assignments. Scoped source/current group and tag records can differ. The full
scope retains all persisted incoming records; external project references remain
the consuming application's responsibility.
Selected Part records retain their literal complete `sheet_ids`, even when some
of those Sheets are outside this component. The scoped record subset therefore
is not a standalone model; those contextual records remain available in the
full scope. No membership IDs are filtered or rewritten.

`validate_prepared_sheet_joint_component_binding` rederives the receipt and
rejects stale, wrong-owner or tampered data. Selection validation requires every
connected authored root; selecting only one Sheet's root refuses. Extra known
roots are allowed in the selection but are not qualified by this receipt.
Inputs are normalized before callbacks, working definitions are detached, and
final owner/index checks follow the final cancellation callback. Cancellation
exceptions propagate unchanged, with no partial receipt or model edits.

The portable example prepares two explicit Sheets before a 4×4 plate is cut at
x=3. Joint edge 25 links both Sheets; all eight current faces and their sibling
FaceUses are captured. A third joint reached through the plate expands the
component to three Sheets. No native mesher is invoked.
`examples/reopened_sheet_joint_component_handoff.py` demonstrates a plain saved
child design, fresh complete eight-face/two-Sheet extraction and new preparation.
All eight child roots and occurrences remain distinct, with the existing two
joint attachments and one declaration retained. General semantic mapping and
publication flags remain false.
