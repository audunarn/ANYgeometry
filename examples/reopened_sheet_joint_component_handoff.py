"""Fresh extraction/preparation of a reopened child design; geometry only."""
from anygeometry import (
    apply_intersections, extract_model_closure, from_dict, plan_intersections,
    query_prepared_sheet_joint_component, to_dict,
    validate_prepared_sheet_joint_component_binding,
    validate_prepared_sheet_joint_component_selection,
)
from examples.prepared_sheet_joint_component_handoff import build as build_original


def build(*, unrelated=False):
    original, _, _, joint = build_original(unrelated=unrelated)
    source = from_dict(to_dict(original))
    handles = tuple(source.handle('face', key) for key in source.faces) + tuple(
        source.handle('sheet', key) for key in source.sheets)
    closure = extract_model_closure(source, handles, include_structural_closure=True,
                                    include_features=False)
    work = closure.working_model
    roots = tuple(closure.source_to_work[source.handle('face', key)].id for key in source.faces)
    work_joint = closure.source_to_work[source.handle('edge', joint)].id
    before = to_dict(work)
    apply_intersections(work, plan_intersections(work, roots, policy='connect'), policy='connect')
    return source, closure, roots, work_joint, before


def verify():
    source, closure, roots, joint, before = build()
    work = closure.working_model
    source_before, work_before = to_dict(source), to_dict(work)
    receipt = query_prepared_sheet_joint_component(work, joint)
    validate_prepared_sheet_joint_component_binding(work, receipt)
    validate_prepared_sheet_joint_component_selection(work, receipt, roots)
    assert to_dict(source) == source_before and to_dict(work) == work_before
    assert receipt.authored_face_ids == roots == receipt.current_face_ids
    assert receipt.source_records['attachments'] == receipt.current_records['attachments']
    assert receipt.source_records['junctions'] == receipt.current_records['junctions']
    assert len(before['structural']['junctions']) == 1
    assert len(receipt.attachment_ids) == 2 and len(receipt.junction_ids) == 1
    return dict(authored_roots=roots, current_joint_edge=joint,
                preserved_joint_attachment_ids=receipt.preserved_joint_attachment_ids,
                preserved_joint_junction_ids=receipt.preserved_joint_junction_ids,
                occurrence_count=len(receipt.occurrence_correspondence),
                semantic_mapping_qualified=receipt.semantic_mapping_qualified,
                publication_qualified=receipt.publication_qualified)


if __name__ == '__main__':
    print(verify())
