"""Explicit, topology-bound Sheet joint records shared with mesh consumers."""
from dataclasses import dataclass, replace
from .identity import EntityHandle
from .errors import GeometryError
from .structural import (AttachmentKind,AttachmentTargetKind,AttachmentEvidence,
                         ConnectionIntent,JunctionKind,ParameterRange,NonManifoldPolicy)


@dataclass(frozen=True,slots=True)
class JointEdge:
    edge: EntityHandle
    sheets: tuple[EntityHandle,...]
    junctions: tuple[EntityHandle,...]
    owner_policy_declared: bool = False

    @property
    def declared(self):
        return bool(self.junctions) or self.owner_policy_declared


def _radial_count(model,sheet,edge):
    return sum(use.edge == edge
        for face_use in sheet.face_use_ids
        for loop in (model.faces[model.face_uses[face_use].face_id].loop,
                     *model.faces[model.face_uses[face_use].face_id].holes)
        for use in loop)


def query_joint_edge(model,edge):
    """Qualify explicit Sheet joint coverage of one current canonical edge.

    Metadata or samples cannot authorize a joint. Every participating Sheet
    must use the edge and have a full-range exact attachment in the Junction.
    """
    if isinstance(edge,EntityHandle):
        if edge.model_id!=model.model_id or edge.kind!="edge":
            raise GeometryError("joint query needs an edge from this model")
        identifier=edge.id
    else:
        identifier=edge
    handle=model.handle("edge",identifier)
    owners=tuple(model.sheets_using_edge(identifier))
    owner_policy_declared = False
    if len(owners) == 1:
        sheet = model.sheets[owners[0]]
        uses = _radial_count(model,sheet,identifier)
        owner_policy_declared = (
            uses > 2 and sheet.policy.non_manifold is NonManifoldPolicy.ALLOW_DECLARED
            and identifier in sheet.declared_non_manifold_edges)
    junctions=[]
    candidates={joint for owner in owners for joint in model._sheet_junctions.get(owner,())}
    for joint_id in sorted(candidates):
        joint=model.junctions[joint_id]
        if joint.kind is not JunctionKind.SHEET_JOINT or set(joint.sheet_ids)!=set(owners):
            continue
        if joint.connection_intent is not ConnectionIntent.CONNECT:
            continue
        sources=set()
        for attachment_id in joint.attachment_ids:
            attachment=model.attachments[attachment_id]
            if (attachment.kind is AttachmentKind.SHEET_ON_JOINT
                and attachment.source_kind=="sheet" and attachment.source_id in owners
                and attachment.target_kind is AttachmentTargetKind.EDGE and attachment.target_id==identifier
                and attachment.connection_intent is ConnectionIntent.CONNECT
                and attachment.evidence is AttachmentEvidence.EXACT
                and attachment.target_parameters==(ParameterRange(0.,1.),)
                and attachment.max_residual<=attachment.tolerance_used):
                sources.add(attachment.source_id)
        if sources==set(owners) and len(owners)>=2:
            junctions.append(model.handle("junction",joint_id))
    return JointEdge(handle,tuple(model.handle("sheet",owner) for owner in owners),
                     tuple(junctions),owner_policy_declared)


def declare_joint_edge(model,edge_id,tolerance):
    owners=tuple(model.sheets_using_edge(edge_id))
    if len(owners) == 1 and _radial_count(model,model.sheets[owners[0]],edge_id) > 2:
        # CONNECT explicitly authorizes this physical joint. Preserve its Sheet
        # identity and admit only the current canonical non-manifold edge.
        sheet = model.sheets[owners[0]]
        model._put_structural("sheet", replace(sheet,
            policy=replace(sheet.policy,non_manifold=NonManifoldPolicy.ALLOW_DECLARED),
            declared_non_manifold_edges=tuple(sorted(set(
                sheet.declared_non_manifold_edges) | {edge_id}))))
        return
    if len(owners)<2 or query_joint_edge(model,edge_id).declared:
        return
    attachments=tuple(model.ensure_attachment(None,AttachmentKind.SHEET_ON_JOINT,
        AttachmentTargetKind.EDGE,edge_id,ParameterRange(0.,1.),(ParameterRange(0.,1.),),
        source_kind="sheet",source_id=sheet,connection_intent=ConnectionIntent.CONNECT,
        evidence=AttachmentEvidence.EXACT,max_residual=0.,tolerance_used=tolerance,
        provenance={"contract":"ANYGEOMETRY_ANALYTIC_JOINT_EDGE_V1"}) for sheet in owners)
    model.ensure_junction(JunctionKind.SHEET_JOINT,(),sheet_ids=owners,
        attachment_ids=attachments,connection_intent=ConnectionIntent.CONNECT,
        provenance={"contract":"ANYGEOMETRY_ANALYTIC_JOINT_EDGE_V1"})
