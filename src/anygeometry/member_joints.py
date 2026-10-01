"""Explicit structural relations for atomically prepared member topology."""
import numpy as np
from .entities import EntityRef
from .structural import (AttachmentKind, AttachmentTargetKind, AttachmentEvidence,
    ConnectionIntent, JunctionKind, JunctionMemberUse, ParameterRange, Orientation)
from .errors import GeometryError


def _member_point(model,member_id,parameter):
    for identifier in model.members[member_id].edge_use_ids:
        use=model.member_edge_uses[identifier]
        if use.parent_range.contains(parameter,tolerance=model.tolerance.parameter):
            local=(parameter-use.parent_range.start)/use.parent_range.length
            if use.orientation is Orientation.REVERSED:
                local=1-local
            return model.sample_edge(use.edge_id,np.asarray((min(1.,max(0.,local)),)))[0]
    raise GeometryError('joint parameter is outside the member axis')


def declare_member_contacts(model,contacts,check,*,sheet_target_ids=(), intent=ConnectionIntent.CONNECT):
    """Contacts carry exact parent parameters; topology supplies the station."""
    grouped={}
    for contact,vertex in contacts:
        members,faces,tolerance=grouped.setdefault(vertex,(set(),set(),contact.world_tolerance))
        members.update(contact.member_parameters)
        if contact.face_id is not None:
            faces.update(ref.id for ref in model.resolve_ref(EntityRef('face',contact.face_id)))
    for vertex,(parameters,faces,tolerance) in sorted(grouped.items()):
        check()
        position=model.vertex_position(vertex)
        incident={face for edge in model.edges_using_vertex(vertex) for face in model.faces_using_edge(edge)}
        faces.intersection_update(incident)
        sheets={sheet for face in faces for use in model._face_structural_uses.get(face,())
                for sheet in (model.face_uses[use].sheet_id,)}
        selected={}
        for face in sorted(faces):
            for use in model._face_structural_uses.get(face,()):
                selected.setdefault(model.face_uses[use].sheet_id,face)
        attachments=[]
        # A bend in one continuous axis is topology, not a structural joint.
        if len({member for member,_parameter in parameters}) == 1 and not faces:
            values=sorted(parameter for _member,parameter in parameters)
            if (len(values)==1 or max(values)-min(values)<=model.tolerance.parameter
                    or all(min(abs(value),abs(value-1.))<=model.tolerance.parameter for value in values)):
                continue
        endpoint=any(parameter in (0.,1.) for _member,parameter in parameters)
        for member,parameter in sorted(parameters):
            residual=float(np.linalg.norm(_member_point(model,member,parameter)-position))
            if residual>tolerance:
                raise GeometryError('canonical member contact exceeds its qualified tolerance')
            targets=[(sheet,face) for sheet,face in sorted(selected.items())]
            targets.extend((None,face) for face in sorted(faces)
                           if not model._face_structural_uses.get(face))
            for sheet,face in targets:
                uv=model.face_local_uv(face,position)
                sheet_target=sheet in sheet_target_ids
                at_endpoint=parameter in (0.,1.)
                kind=(AttachmentKind.MEMBER_ENDPOINT_ON_SHEET if at_endpoint
                      else AttachmentKind.MEMBER_CROSS_SHEET) if sheet_target else AttachmentKind.MEMBER_THROUGH_FACE
                attachments.append(model.ensure_attachment(member,kind,
                    AttachmentTargetKind.SHEET if sheet_target else AttachmentTargetKind.FACE,
                    sheet if sheet_target else face,ParameterRange(parameter,parameter),
                    tuple(ParameterRange(float(value),float(value)) for value in uv),
                    sheet_id=sheet,connection_intent=intent,
                    evidence=AttachmentEvidence.EXACT,max_residual=residual,tolerance_used=tolerance,
                    metadata={'face_sequence':[face]},
                    provenance={'contract':'ANYGEOMETRY_ANALYTIC_MEMBER_JOINT_V1'}))
        members=tuple(sorted({member for member,_ in parameters}))
        if len(members)==2 and endpoint:
            source=next((member,parameter) for member,parameter in sorted(parameters)
                        if parameter in (0.,1.))
            target=next((member,parameter) for member,parameter in sorted(parameters)
                        if member!=source[0])
            attachments.append(model.ensure_attachment(source[0],AttachmentKind.MEMBER_ENDPOINT_ON_MEMBER,
                AttachmentTargetKind.MEMBER,target[0],ParameterRange(source[1],source[1]),
                (ParameterRange(target[1],target[1]),),connection_intent=intent,
                evidence=AttachmentEvidence.EXACT,max_residual=float(np.linalg.norm(
                    _member_point(model,*source)-_member_point(model,*target))),tolerance_used=tolerance,
                provenance={'contract':'ANYGEOMETRY_ANALYTIC_MEMBER_JOINT_V1'}))
        uses=tuple(JunctionMemberUse(member,ParameterRange(parameter,parameter)) for member,parameter in sorted(parameters))
        kind=(JunctionKind.MULTI_WAY if len(parameters)+len(sheets)>2 else
              JunctionKind.ENDPOINT if all(parameter in (0.,1.) for _member,parameter in parameters)
              else JunctionKind.CROSSING)
        if len(uses)+len(sheets)>=2:
            model.ensure_junction(kind,
                uses,sheet_ids=tuple(sorted(sheets)),attachment_ids=tuple(attachments),
                connection_intent=intent,
                provenance={'contract':'ANYGEOMETRY_ANALYTIC_MEMBER_JOINT_V1'})


def declare_member_boundaries(model,selected_members,check,*,intent=ConnectionIntent.CONNECT):
    """Retain each exact parent interval on its canonical shell edge."""
    selected=set(selected_members)
    for use in sorted(model.member_edge_uses.values(),key=lambda item:item.id):
        if use.member_id not in selected:
            continue
        faces=tuple(model.faces_using_edge(use.edge_id))
        sheets=tuple(model.sheets_using_edge(use.edge_id))
        if not faces:
            continue
        check()
        tolerance=model.tolerance.effective_length(model.edge_length(use.edge_id))
        attachment=model.ensure_attachment(use.member_id,AttachmentKind.MEMBER_ON_FACE_BOUNDARY,
            AttachmentTargetKind.EDGE,use.edge_id,use.parent_range,(ParameterRange(0.,1.),),
            connection_intent=intent,evidence=AttachmentEvidence.EXACT,
            max_residual=0.,tolerance_used=tolerance,
            provenance={'contract':'ANYGEOMETRY_ANALYTIC_MEMBER_JOINT_V1'})
        if sheets:
            model.ensure_junction(JunctionKind.OVERLAP,(JunctionMemberUse(use.member_id,use.parent_range),),
                sheet_ids=sheets,attachment_ids=(attachment,),connection_intent=intent,
                provenance={'contract':'ANYGEOMETRY_ANALYTIC_MEMBER_JOINT_V1'})
