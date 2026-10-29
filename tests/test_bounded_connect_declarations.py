"""Real bounded CONNECT declaration; geometry-only IMPRINT remains separate."""
import pytest
from anygeometry import (GeometryError,IntersectionBatchPolicy,ConnectionIntent,
    plan_intersections,apply_intersections)
from anygeometry.serialization import to_dict
from anygeometry.joint_edges import query_joint_edge
from anygeometry.structural import AttachmentKind,AttachmentEvidence,ParameterRange
from anygeometry.entities import EntityRef
from test_local_sweep_competition import closed_source


def test_bounded_connect_declares_authentic_joints_and_atomic_apply():
    model,faces,plane=closed_source();before=to_dict(model)
    original_revision=model.revision
    original_sheets=set(model.sheets);original_faces=set(model.faces)
    sheet_parents={sid:{model.face_uses[u].face_id for u in record.face_use_ids} for sid,record in model.sheets.items()}
    plan=plan_intersections(model,(*faces,plane),policy='connect')
    assert to_dict(model)==before  # declarations belong only to detached candidate
    cancel=IntersectionBatchPolicy(ConnectionIntent.CONNECT,cancellation_check=lambda:True)
    with pytest.raises(GeometryError,match='cancelled'):apply_intersections(model,plan,policy=cancel)
    assert to_dict(model)==before
    token=GeometryError('actual callback token')
    def stop():raise token
    with pytest.raises(GeometryError) as caught:
        apply_intersections(model,plan,policy=IntersectionBatchPolicy(ConnectionIntent.CONNECT,cancellation_check=stop))
    assert caught.value is token and to_dict(model)==before
    face=model.faces[plane];metadata=face.metadata
    def mutate():
        object.__setattr__(face,'metadata',dict(metadata,callback_mutation=True))
        return False
    try:
        with pytest.raises(GeometryError,match='changed'):
            apply_intersections(model,plan,policy=IntersectionBatchPolicy(ConnectionIntent.CONNECT,cancellation_check=mutate))
        assert set(model.faces)==original_faces and not model.attachments and not model.junctions
    finally:object.__setattr__(face,'metadata',metadata)
    assert to_dict(model)==before
    result=apply_intersections(model,plan,policy='connect')
    assert model.revision==original_revision+1
    assert len(result.joint_edges)==4 and len(model.faces)==10
    assert set(model.sheets)==original_sheets
    descendants={parent:{ref.id for ref in model.resolve_ref(EntityRef('face',parent))} for parent in original_faces}
    assert all(len(children)==2 for children in descendants.values())
    for sheet,parents in sheet_parents.items():
        assert {model.face_uses[u].face_id for u in model.sheets[sheet].face_use_ids}==set().union(*(descendants[p] for p in parents))
    for edge in result.joint_edges:
        incident=model.faces_using_edge(edge.id)
        assert len(incident)==4
        assert all(sum(use.edge==edge.id for loop in (model.faces[f].loop,*model.faces[f].holes) for use in loop)==1 for f in incident)
        joint=query_joint_edge(model,edge.id)
        assert joint.declared and joint.junctions and len(joint.sheets)==2
        attachments=[model.attachments[a] for handle in joint.junctions
            for a in model.junctions[handle.id].attachment_ids]
        assert {item.source_id for item in attachments}=={sheet.id for sheet in joint.sheets}
        assert all(item.kind is AttachmentKind.SHEET_ON_JOINT
            and item.evidence is AttachmentEvidence.EXACT
            and item.connection_intent is ConnectionIntent.CONNECT
            and item.target_kind=='edge' and item.target_id==edge.id
            and item.target_parameters==(ParameterRange(0.,1.),)
            and item.max_residual==0 for item in attachments)
    assert not model.validate_topology()


def test_bounded_imprint_does_not_promote_structural_connections():
    model,faces,plane=closed_source();before=to_dict(model);sheets=set(model.sheets)
    plan=plan_intersections(model,(*faces,plane),policy='imprint')
    assert to_dict(model)==before
    result=apply_intersections(model,plan,policy='imprint')
    assert len(result.joint_edges)==4 and len(model.faces)==10
    assert set(model.sheets)==sheets and not model.attachments and not model.junctions
    assert all(not query_joint_edge(model,edge.id).declared for edge in result.joint_edges)
