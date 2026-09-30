import pytest
import numpy as np
from dataclasses import replace
from anygeometry import GeometryModel, plan_intersections, apply_intersections, to_dict
from anygeometry.structural import (AttachmentKind, AttachmentEvidence, ParameterRange,
                                   JunctionKind, JunctionMemberUse)


def test_axis_attachment_retains_thin_fragment_and_junction_references():
    model=GeometryModel()
    face=model.add_plate(model.add_points(((0,0,0),(4,0,0),(4,4,0),(0,4,0))))
    sheet=model.add_sheet((face,))
    member=model.add_member((model.add_line(*model.add_points(((0,2,0),(4,2,0)))),))
    attachment=model.add_attachment(member,AttachmentKind.MEMBER_ON_FACE,'face',face,
        ParameterRange(0,1),(ParameterRange(0,1),ParameterRange(.5,.5)),
        evidence=AttachmentEvidence.EXACT,tolerance_used=1e-8,sheet_id=sheet)
    junction=model.ensure_junction(JunctionKind.OVERLAP,(JunctionMemberUse(member,ParameterRange(0,1)),),
                                  attachment_ids=(attachment,),sheet_ids=(sheet,))
    cuts=[model.add_plate(model.add_points(((x,0,-1),(x,4,-1),(x,4,1),(x,0,1))))
          for x in (.75,.8)]
    application=apply_intersections(model,plan_intersections(model,(face,*cuts),policy='connect'),policy='connect')
    assert not application.reused
    related=[model.attachments[identifier] for identifier in model.junctions[junction].attachment_ids]
    ranges=sorted((item.member_range.start,item.member_range.end) for item in related)
    assert np.asarray(ranges)==pytest.approx(np.asarray(((0.,.1875),(.1875,.2),(.2,1.))))
    assert all(item.target_id in model.faces and ('face',face) in item.lineage for item in related)
    assert model.validate_topology()==()
    from anygeometry import from_dict
    assert to_dict(from_dict(to_dict(model)))==to_dict(model)
    from anygeometry import GeometryError
    before=to_dict(model)
    first=min(related,key=lambda item:item.member_range.start)
    with pytest.raises(GeometryError,match='does not cover'):
        with model.transaction():
            model._put_structural('attachment',replace(first,
                member_range=ParameterRange(first.member_range.start,first.member_range.end-.0001)))
    assert to_dict(model)==before


def test_point_attachment_on_cut_retains_every_incident_material_face():
    model=GeometryModel()
    face=model.add_plate(model.add_points(((0,0,0),(4,0,0),(4,4,0),(0,4,0))))
    member=model.add_member((model.add_line(*model.add_points(((2,2,-1),(2,2,1)))),))
    model.add_attachment(member,AttachmentKind.MEMBER_THROUGH_FACE,'face',face,
        ParameterRange(.5,.5),(ParameterRange(.5,.5),ParameterRange(.5,.5)),
        evidence='exact',tolerance_used=1e-8)
    cut=model.add_plate(model.add_points(((2,0,-1),(2,4,-1),(2,4,1),(2,0,1))))
    apply_intersections(model,plan_intersections(model,(face,cut),policy='connect'),policy='connect')
    related=[item for item in model.attachments.values() if item.member_id==member]
    assert len(related)==2
    for item in related:
        assert model.face_point(item.target_id,*(r.start for r in item.target_parameters))==pytest.approx((2,2,0))
    assert model.validate_topology()==()
