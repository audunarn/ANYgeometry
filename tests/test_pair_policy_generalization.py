import numpy as np
import pytest
from anygeometry import (GeometryModel, EntityRef, GeometryError, ConnectionIntent, plan_imprint, apply_imprint,
                         query_intersection, query_trimmed_surface_charts, to_dict)
from anygeometry.entities import OrientedEdge
from anygeometry.operations import trim_face


@pytest.mark.parametrize('intent', (ConnectionIntent.CONNECT, ConnectionIntent.IMPRINT,
                                   ConnectionIntent.CONTACT_ONLY, ConnectionIntent.KEEP_DISCONNECTED))
def test_multiple_member_crossings_keep_intent_and_all_occurrences(intent):
    model = GeometryModel()
    first = model.add_member((model.add_line(*model.add_points(((-2,0,0),(2,0,0)))),))
    points = model.add_points(((-1,-1,0),(-.5,1,0),(.5,1,0),(1,-1,0)))
    second = model.add_member(tuple(model.add_line(a,b) for a,b in zip(points,points[1:])))
    result = query_intersection(model, model.handle('member',first), model.handle('member',second))
    assert len(result.components) == 2
    before = to_dict(model)
    plan = plan_imprint(model, result, policy=intent)
    assert (plan.batch_plan is not None) == (intent in (ConnectionIntent.CONNECT,ConnectionIntent.IMPRINT))
    assert to_dict(model) == before
    applied = apply_imprint(model, plan, policy=intent)
    joints = [model.junctions[handle.id] for handle in applied.relations if handle.kind=='junction']
    assert len(joints) == 2 and all(joint.connection_intent is intent for joint in joints)
    parameters = sorted(use.member_range.start for joint in joints for use in joint.member_uses
                        if use.member_id == first)
    np.testing.assert_allclose(parameters,(.3125,.6875),rtol=0.,atol=1e-14)
    assert model.validate_topology() == ()
    after = to_dict(model)
    if plan.batch_plan is None:
        with pytest.raises(GeometryError,match='stale'):
            apply_imprint(model,plan,policy=intent)
        plan=plan_imprint(model,model.handle('member',first),model.handle('member',second),policy=intent)
    assert apply_imprint(model,plan,policy=intent).reused
    assert to_dict(model) == after
    if intent in (ConnectionIntent.CONTACT_ONLY,ConnectionIntent.KEEP_DISCONNECTED):
        assert all(to_dict(model)[key]==before[key] for key in ('vertices','edges','faces'))
    reuse=plan_imprint(model,model.handle('member',first),model.handle('member',second),policy='reuse_existing')
    assert apply_imprint(model,reuse,policy='reuse_existing').reused
    assert to_dict(model) == after


@pytest.mark.parametrize('owned', (False, True))
def test_imprint_member_material_intervals_across_multiple_holes(owned):
    model=GeometryModel()
    face=model.add_plate(model.add_points(((0,0,0),(4,0,0),(4,2,0),(0,2,0))))
    holes=[]
    for x in (.5,1.5,2.5):
        vertices=model.add_points(((x,.5,0),(x+.3,.5,0),(x+.3,1.5,0),(x,1.5,0)))
        holes.append(tuple(OrientedEdge(edge,True) for edge in model.add_polyline(vertices,close=True)))
    trim_face(model,face,tuple(holes))
    target=model.handle('sheet',model.add_sheet((face,))) if owned else model.handle('face',face)
    member=model.add_member((model.add_line(*model.add_points(((-1,1,0),(5,1,0)))),))
    before=to_dict(model)
    plan=plan_imprint(model,model.handle('member',member),target,policy=ConnectionIntent.IMPRINT)
    assert plan.batch_plan is not None and len(plan.result.components)==4
    assert to_dict(model)==before
    applied=apply_imprint(model,plan,policy=ConnectionIntent.IMPRINT)
    intervals=sorted((item.member_range.start,item.member_range.end)
        for item in model.attachments.values() if item.source_id==member and not item.member_range.is_point)
    np.testing.assert_allclose(intervals,np.array(((1,1.5),(1.8,2.5),(2.8,3.5),(3.8,5)))/6,
                               rtol=0.,atol=1e-14)
    assert applied.relations and all(model.attachments[handle.id].connection_intent is ConnectionIntent.IMPRINT
                                   for handle in applied.relations if handle.kind=='attachment')
    assert sum(chart.material_area for chart in query_trimmed_surface_charts(model).charts)==pytest.approx(7.1)
    assert len(model.sheets)==int(owned)
    assert model.validate_topology()==()
    after=to_dict(model)
    assert apply_imprint(model,plan,policy=ConnectionIntent.IMPRINT).reused
    assert to_dict(model)==after


def test_imprint_unowned_face_point_retains_relation_without_inventing_owner():
    model=GeometryModel()
    face=model.add_plate(model.add_points(((-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0))))
    member=model.add_member((model.add_line(*model.add_points(((0,0,-1),(0,0,1)))),))
    plan=plan_imprint(model,model.handle('member',member),model.handle('face',face),policy='imprint')
    applied=apply_imprint(model,plan,policy='imprint')
    attachments=[model.attachments[item.id] for item in applied.relations if item.kind=='attachment']
    descendants={item.id for item in model.resolve_ref(EntityRef('face',face))}
    assert attachments and {item.target_id for item in attachments}==descendants
    assert all(item.connection_intent is ConnectionIntent.IMPRINT for item in attachments)
    assert not model.sheets and model.validate_topology()==()


def test_reuse_requires_every_member_occurrence_and_contact_recreates_only_missing_relation():
    model=GeometryModel()
    first=model.add_member((model.add_line(*model.add_points(((-2,0,0),(2,0,0)))),))
    vertices=model.add_points(((-1,-1,0),(-.5,1,0),(.5,1,0),(1,-1,0)))
    second=model.add_member(tuple(model.add_line(a,b) for a,b in zip(vertices,vertices[1:])))
    operands=(model.handle('member',first),model.handle('member',second))
    apply_imprint(model,plan_imprint(model,*operands,policy='contact_only'),policy='contact_only')
    joints=tuple(sorted(model.junctions))
    retained=model.junctions[joints[0]]
    model.remove_junction(joints[1])
    before=to_dict(model)
    plan=plan_imprint(model,*operands,policy='reuse_existing')
    with pytest.raises(GeometryError,match='requires compatible existing'):
        apply_imprint(model,plan,policy='reuse_existing')
    assert to_dict(model)==before
    applied=apply_imprint(model,plan_imprint(model,*operands,policy='contact_only'),policy='contact_only')
    assert not applied.reused and len(model.junctions)==2
    assert model.junctions[retained.id]==retained


def test_imprint_relation_failure_rolls_back_exactly(monkeypatch):
    model=GeometryModel()
    face=model.add_plate(model.add_points(((-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0))))
    sheet=model.add_sheet((face,))
    member=model.add_member((model.add_line(*model.add_points(((-2,0,0),(2,0,0)))),))
    plan=plan_imprint(model,model.handle('member',member),model.handle('sheet',sheet),policy='imprint')
    before=to_dict(model)
    original=GeometryModel.ensure_attachment
    calls=0
    def fail_after_create(self,*args,**kwargs):
        nonlocal calls
        calls+=1
        original(self,*args,**kwargs)
        raise GeometryError('injected member imprint relation failure')
    monkeypatch.setattr(GeometryModel,'ensure_attachment',fail_after_create)
    with pytest.raises(GeometryError,match='injected member imprint relation failure'):
        apply_imprint(model,plan,policy='imprint')
    assert calls==1 and to_dict(model)==before
