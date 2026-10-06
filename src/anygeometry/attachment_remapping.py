"""Analytic remapping of point and axis attachments to material descendants.

Witness samples are never used to decide which descendant retains a relation.
Target parameter rectangles enclose the retained axis interval; they do not
replace the source interval or the exact material clipping certificate.
"""
from dataclasses import replace
import numpy as np

from .arrangement_geometry import LinePath, BezierPath, freeze_edge
from .errors import GeometryError
from .exact_curves import EllipticArc
from .material_arrangement import MaterialDomain, _clip_intervals
from .member_arrangements import _support_roots
from .structural import ParameterRange, Orientation, AttachmentTargetKind
from .surfaces import Plane, Cylinder


def capture_face_attachments(model,face_id,check):
    snapshots=[]
    for identifier in sorted(model._target_attachments.get(('face',face_id),())):
        check()
        attachment=model.attachments[identifier]
        if len(attachment.target_parameters)!=2:
            raise GeometryError('face attachment needs two target parameter ranges')
        u,v=attachment.target_parameters
        if u.start==u.end and v.start==v.end:
            point=tuple(model.face_point(face_id,u.start,v.start))
            snapshots.append(('target',attachment,((None,point,attachment.member_range),)))
            continue
        if attachment.source_kind!='member' or attachment.member_id is None:
            raise GeometryError('non-point face attachment needs an exact source axis definition')
        axes=[]
        for use_id in model.members[attachment.member_id].edge_use_ids:
            use=model.member_edge_uses[use_id]
            lower=max(use.parent_range.start,attachment.member_range.start)
            upper=min(use.parent_range.end,attachment.member_range.end)
            if upper<=lower:
                continue
            a=(lower-use.parent_range.start)/use.parent_range.length
            b=(upper-use.parent_range.start)/use.parent_range.length
            if use.orientation is Orientation.REVERSED:
                a,b=1-a,1-b
            axes.append((freeze_edge(model,use.edge_id).subcurve(a,b),None,ParameterRange(lower,upper)))
        if not axes:
            raise GeometryError('face attachment has no retained source axis interval')
        snapshots.append(('target',attachment,tuple(axes)))
    for identifier in sorted(model._source_attachments.get(('face',face_id),())):
        attachment=model.attachments[identifier]
        if any(item[1].id==identifier for item in snapshots):
            raise GeometryError('self face attachments need an explicit ownership remap')
        if attachment.target_kind is AttachmentTargetKind.VERTEX:
            paths=((None,tuple(model.vertex_position(attachment.target_id)),None),)
        elif attachment.target_kind is AttachmentTargetKind.EDGE:
            curve=freeze_edge(model,attachment.target_id)
            interval=attachment.target_parameters[0]
            paths=((curve.subcurve(interval.start,interval.end),None,interval),)
        else:
            raise GeometryError('source face attachment needs an exact point or edge target')
        snapshots.append(('source',attachment,paths))
    return tuple(snapshots)


def _target_bounds(domain,curve):
    support=domain.support
    if isinstance(support,Plane):
        normal=support.normal
        inverse=np.linalg.inv(np.column_stack((support.u_vector,support.v_vector,normal)))
        transform=np.eye(4); transform[:3,:3]=inverse
        transform[:3,3]=-inverse @ support.origin
        if isinstance(curve,LinePath):
            points=np.asarray((curve.start,curve.end)) @ inverse.T+transform[:3,3]
            lower,upper=points.min(axis=0),points.max(axis=0)
        else:
            lower,upper=curve.transformed(transform).bounds()
        return tuple(ParameterRange(float(np.clip(lower[i],0.,1.)),
                                    float(np.clip(upper[i],0.,1.))) for i in (0,1))
    # Angular extrema of a curve on the secondary cylinder need not be at
    # its endpoints. Its native chart is a certified enclosing rectangle.
    # Retain that conservative angular range and bound the linear axial part.
    lower,upper=curve.bounds()
    center=.5*(lower+upper)-support.origin; half=.5*(upper-lower)
    axial=float(center @ support.axis)/support.height
    radius=float(np.abs(support.axis) @ half)/abs(support.height)
    return (ParameterRange(0.,1.),ParameterRange(max(0.,axial-radius),min(1.,axial+radius)))


def remap_face_attachments(model,old_face,descendants,snapshots,check):
    for role,attachment,paths in snapshots:
        candidates=[]
        for face_id in sorted(descendants):
            domain=MaterialDomain.from_model(model,face_id)
            tolerance=model.tolerance.effective_length(max(
                (np.linalg.norm(hi-lo) for loop in domain.boundaries for path in loop
                 for lo,hi in (path.curve.bounds(),)),default=1.))
            for curve,point,source_interval in paths:
                check()
                if curve is None:
                    probe=LinePath(point,point)
                    if domain.contains(probe,0.,tolerance):
                        uv=domain.support.local_uv(point)
                        parameters=tuple(ParameterRange(min(1.,max(0.,float(value))),
                            min(1.,max(0.,float(value)))) for value in uv)
                        candidates.append((face_id,source_interval,parameters))
                    continue
                if _support_roots(curve,domain.support,tolerance,check) is not None:
                    raise GeometryError('attachment axis lacks a whole-curve support certificate')
                for a,b in _clip_intervals(domain,curve,tolerance,check):
                    interval=ParameterRange(source_interval.start+a*source_interval.length,
                                            source_interval.start+b*source_interval.length)
                    candidates.append((face_id,interval,_target_bounds(domain,curve.subcurve(a,b))))
        if not candidates:
            raise GeometryError(f'attachment {attachment.id} lost all material descendants')
        lineage=tuple(dict.fromkeys((*attachment.lineage,('attachment',attachment.id),('face',old_face.id))))
        identifiers=[]
        for index,(face_id,interval,parameters) in enumerate(candidates):
            changes={'lineage':lineage}
            if role=='target':
                changes.update(target_id=face_id,target_parameters=parameters,member_range=interval)
            else:
                changes.update(source_id=face_id)
                if attachment.target_kind is AttachmentTargetKind.EDGE:
                    changes['target_parameters']=(interval,)
            identifier=attachment.id if index==0 else model._allocate_structural('attachment')
            model._put_structural('attachment',replace(attachment,id=identifier,**changes))
            identifiers.append(identifier)
        for junction_id in sorted(model._attachment_junctions.get(attachment.id,())):
            junction=model.junctions[junction_id]
            expanded=tuple(dict.fromkeys(identifier for old in junction.attachment_ids
                for identifier in (identifiers if old==attachment.id else (old,))))
            model._put_structural('junction',replace(junction,attachment_ids=expanded))
