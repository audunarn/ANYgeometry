# Prepared mixed Sheet network receipt

`query_prepared_mixed_sheet_joint_network` (in
`src/anygeometry/prepared_mixed_sheet_network.py`) captures one complete
connected mixed Sheet-only document after a complete current intersection
preparation, and `validate_prepared_mixed_sheet_joint_network_binding`
rederives the same receipt from the live model so no omitted or forged
record can pass. The receipt proves CURRENT structural truth only.

## Supported cases

- Sheet-only documents (zero Members) whose current faces mix Plane,
  Cylinder/Cone panels and supported extruded/Coons walls, provided every
  current face descends from an authored source face and the document is one
  connected whole-document Sheet component.
- Literal incidence/index correspondence per canonical edge, and a complete
  qualified generated Sheet joint on every shared canonical edge.
- Root/source Sheet FaceUse occurrence correspondence: unique complete
  root/Sheet occurrences with unchanged orientation and metadata.
- Complete unchanged original joint inventory, recorded as
  `preserved_joint_attachment_ids` and `preserved_joint_junction_ids` and
  qualified payload-for-payload against the current joints.
- Complete public trimmed charts, one per current face, including holes and
  seam sides, bound to the live model.
- Exact recorded original and current surface definitions per current face
  with an explicitly limited correspondence verdict:
  - `exact_fields` — the serialized carrier field dictionaries are literally
    equal. Restricted parametric fields `start_angle`, `sweep_angle` (and
    `height` for Cylinder carriers only) may narrow on a qualified split; a
    Cone carrier must retain its serialized `height` exactly. Zero floating
    plane residuals alone do NOT establish exact serialized fields.
  - `recorded_only` — differing Plane definitions and Coons/extruded wall
    transitions are recorded verbatim with no carrier-equivalence claim.
    The three Plane origin/basis residuals are individually checked against
    the current chart tolerance. Those diagnostics establish no whole-material
    distance bound or parameter correspondence: their contributions can add
    and the chart's UV extent can amplify them.
- Deterministic, nonmutating queries; typed refusal helpers
  `require_prepared_mixed_sheet_authored_material_coverage` and
  `require_prepared_mixed_sheet_reference_parameter_mapping` always refuse.

## Refused cases

- Documents containing Members or member edge uses; disconnected or
  partially owned documents; batch-created source-less Sheets/Parts; changed
  Sheet/Part semantic fields; empty or unselected Parts (every current and
  source Part must be selected by the connected Sheet component, and the
  document must carry at least one Part).
- Parameterized faces (current or original), unsupported surface-family
  transitions, unsupported occurrence orientation values, unsupported
  Junction/Attachment semantics, unowned or source-less owners, original
  attachment/junction remaps, and singular or nonfinite Plane carrier proofs.
- Forged receipts: any changed inventory tuple, record JSON, chart row,
  occurrence row, carrier verdict or qualification flag is refused by
  rederivation; qualification flags other than `occurrence_mapping_qualified`
  must be false.
- Stale receipts: any model mutation (including from a `cancellation_check`
  callback) invalidates the binding; callbacks that mutate the model during a
  query or binding validation are refused.

## Remaining limits

- No authored material coverage: exact curved source arrangement/domain
  inclusion and nonoverlap are future work; area totals, retained root IDs
  and samples never promote this receipt.
- No reference parameter mapping: no original-to-current parameter
  correspondence is certified; surface records and chart bindings are
  visibility, not a remap.
- No raw metadata/group/tag/feature/extension semantics, beam/load/solver
  equivalence or publication permission; no authored-root permission.
- Coons/extruded wall surface-definition correspondence is recorded only.
