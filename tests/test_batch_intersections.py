from copy import deepcopy
from dataclasses import replace

import pytest

from anygeometry import GeometryModel, EntityRef, GeometryError, to_dict
from anygeometry.batch_intersections import plan_intersections, apply_intersections, IntersectionBatchPolicy
from anygeometry.structural import ConnectionIntent


def test_finite_cylinder_point_contact_is_shared_and_bound():
    import numpy as np
    from anygeometry import from_dict, query_trimmed_surface_charts
    from anygeometry.generators import cylinder
    model=cylinder(1.,1.,circumferential_segments=4)
    first=set(model.faces)
    model.insert_model(cylinder(1.,1.,origin=(2.,0.,1.),circumferential_segments=4))
    second=set(model.faces)-first
    before=to_dict(model)
    plan=plan_intersections(model,tuple(model.faces),policy=ConnectionIntent.CONNECT)
    assert plan.face_contacts
    assert all(np.linalg.norm(np.asarray(contact.position)-(1.,0.,1.))<1e-9
               for contact in plan.face_contacts)
    assert to_dict(model)==before
    with pytest.raises(GeometryError,match='content binding'):
        apply_intersections(model,replace(plan,face_contacts=()),policy=ConnectionIntent.CONNECT)
    apply_intersections(model,plan,policy=ConnectionIntent.CONNECT)
    vertices=[vertex.id for vertex in model.vertices.values()
              if np.linalg.norm(np.asarray(vertex.position)-(1.,0.,1.))<1e-9]
    assert len(vertices)==1
    incident={face for edge in model.edges_using_vertex(vertices[0])
              for face in model.faces_using_edge(edge)}
    assert incident & first and incident & second
    assert {attachment.target_id for attachment in model.attachments.values()
            if attachment.source_kind=='vertex' and attachment.source_id==vertices[0]}==incident
    assert model.validate_topology()==()
    assert sum(chart.material_area for chart in query_trimmed_surface_charts(model).charts)==pytest.approx(4*np.pi)
    assert from_dict(to_dict(model)).validate_topology()==()
    after=to_dict(model)
    assert apply_intersections(model,plan,policy=ConnectionIntent.CONNECT).reused
    assert to_dict(model)==after


def test_member_endpoint_on_interior_axis_retains_qualified_attachment():
    from anygeometry import plan_imprint,apply_imprint
    from anygeometry.structural import AttachmentKind,JunctionKind
    model=GeometryModel()
    members=[model.add_member((model.add_line(*model.add_points(points)),))
             for points in (((-1.,0.,0.),(1.,0.,0.)),((0.,0.,0.),(0.,1.,0.)))]
    plan=plan_imprint(model,*(model.handle('member',member) for member in members),policy=ConnectionIntent.CONNECT)
    apply_imprint(model,plan,policy=ConnectionIntent.CONNECT)
    assert len(model.junctions)==1
    assert next(iter(model.junctions.values())).kind is JunctionKind.CROSSING
    attachments=[item for item in model.attachments.values() if item.kind is AttachmentKind.MEMBER_ENDPOINT_ON_MEMBER]
    assert len(attachments)==1
    assert attachments[0].source_id==members[1] and attachments[0].target_id==members[0]
    assert attachments[0].target_parameters[0].start==pytest.approx(.5)
    assert model.validate_topology()==()


@pytest.mark.parametrize('third_axis',(False,True))
def test_repeated_member_visits_retain_distinct_parent_occurrences(third_axis):
    import numpy as np
    from anygeometry import from_dict
    from anygeometry.structural import JunctionKind,Junction,JunctionMemberUse,ParameterRange
    model=GeometryModel()
    points=model.add_points(((-1.,-1.,0.),(1.,1.,0.),(-1.,1.,0.),(1.,-1.,0.)))
    edges=[model.add_line(a,b) for a,b in zip(points,points[1:])]
    member=model.add_member(edges)
    selected=[model.handle('member',member)]
    if third_axis:
        axis=model.add_member((model.add_line(*model.add_points(((0.,0.,-1.),(0.,0.,1.)))),))
        selected.append(model.handle('member',axis))
    before=to_dict(model)
    plan=plan_intersections(model,tuple(reversed(selected)),policy='connect')
    assert to_dict(model)==before
    assert plan==plan_intersections(model,selected,policy='connect')
    apply_intersections(model,plan,policy='connect')
    crossings=[joint for joint in model.junctions.values() if any(
        use.member_id==member and 0.<use.member_range.start<1. for use in joint.member_uses)]
    assert len(crossings)==1
    crossing=crossings[0]
    assert crossing.kind is (JunctionKind.MULTI_WAY if third_axis else JunctionKind.CROSSING)
    visits=[use.member_range.start for use in crossing.member_uses if use.member_id==member]
    expected=np.sqrt(2.)/(2*np.sqrt(8.)+2.)
    np.testing.assert_allclose(visits,(expected,1-expected),rtol=0.,atol=1e-14)
    assert crossing.member_ids==tuple(item.id for item in selected)
    center=[vertex.id for vertex in model.vertices.values() if np.linalg.norm(vertex.position)<1e-12]
    assert len(center)==1 and len(model.edges_using_vertex(center[0]))==(6 if third_axis else 4)
    assert model.validate_topology()==()
    assert from_dict(to_dict(model)).junctions[crossing.id]==crossing
    after=to_dict(model)
    repeated=plan_intersections(model,selected,policy='connect')
    assert apply_intersections(model,repeated,policy='connect').reused
    assert to_dict(model)==after
    duplicate=JunctionMemberUse(member,ParameterRange.point(visits[0]))
    with pytest.raises(GeometryError,match='interval can participate only once'):
        Junction(999,JunctionKind.CROSSING,(duplicate,duplicate))


def _plates():
    model = GeometryModel()
    faces = []
    for points in (((0,0,0),(5,0,0),(5,5,0),(0,5,0)),
                   ((0,2.5,-2.5),(5,2.5,-2.5),(5,2.5,2.5),(0,2.5,2.5)),
                   ((2.5,0,-2.5),(2.5,5,-2.5),(2.5,5,2.5),(2.5,0,2.5))):
        faces.append(model.add_plate(model.add_points(points)))
    return model, faces


def test_original_operand_batch_is_read_only_atomic_and_high_valence():
    model, faces = _plates()
    before = deepcopy(to_dict(model))
    plan = plan_intersections(model, faces, policy=ConnectionIntent.CONNECT)
    with pytest.raises(GeometryError,match='content binding'):
        apply_intersections(model,replace(plan,arrangements=()),policy=ConnectionIntent.CONNECT)
    with pytest.raises(GeometryError,match='policy does not match'):
        apply_intersections(model,plan,policy=IntersectionBatchPolicy(face_connections=False))
    assert to_dict(model) == before
    applied = apply_intersections(model, plan, policy=ConnectionIntent.CONNECT)
    assert applied.change_set.revision_after == before['revision']+1
    assert len(model.faces) == 12
    assert model.validate_topology() == ()
    center = [vertex.id for vertex in model.vertices.values() if tuple(vertex.position) == (2.5,2.5,0.)]
    assert len(center) == 1
    assert len(model.edges_using_vertex(center[0])) == 6
    assert all(len(model.faces_using_edge(edge.id)) == 4 for edge in applied.joint_edges)
    committed=deepcopy(to_dict(model))
    second_plan=plan_intersections(model,tuple(model.faces),policy=ConnectionIntent.CONNECT)
    repeated=apply_intersections(model,second_plan,policy=ConnectionIntent.CONNECT)
    assert repeated.reused
    assert to_dict(model)==committed
    assert apply_intersections(model,plan,policy=ConnectionIntent.CONNECT).reused
    assert to_dict(model)==committed
    model.add_point(9,9,9)
    with pytest.raises(GeometryError, match='stale'):
        apply_intersections(model, plan, policy=ConnectionIntent.CONNECT)


def test_bilinear_boundary_contact_is_preserved_but_interior_crossing_refused():
    model=GeometryModel()
    plate=model.add_plate(model.add_points(((0,0,0),(4,0,0),(4,2,0),(0,2,0))))
    corners=model.add_points(((.5,.35,0),(3.5,1.65,0),(3.5,1.8,1),(.5,.35,1)))
    edges=[model.add_line(corners[i],corners[(i+1)%4]) for i in range(4)]
    wall=model.add_face(edges)
    plan=plan_intersections(model,(plate,wall),policy=ConnectionIntent.CONNECT)
    assert {item.face_id for item in plan.arrangements}=={plate}
    apply_intersections(model,plan,policy=ConnectionIntent.CONNECT)
    assert model.validate_topology()==()
    assert len(model.faces_using_edge(edges[0]))>=2
    assert wall in model.faces
    # A plane through the interior has corners of both signs. It must not be
    # accepted using this boundary-only compatibility certificate.
    cutting=model.add_plate(model.add_points(((0,0,.5),(4,0,.5),(4,2,.5),(0,2,.5))))
    before=to_dict(model)
    with pytest.raises(GeometryError,match='interior Coons'):
        plan_intersections(model,(wall,cutting),policy=ConnectionIntent.CONNECT)
    assert to_dict(model)==before


def test_high_valence_joint_with_one_sheet_retains_owner_and_explicit_policy():
    from anygeometry import query_joint_edge, query_trimmed_surface_charts, from_dict
    from anygeometry.structural import SheetTopologyPolicy, ConnectivityPolicy, NonManifoldPolicy
    model, faces = _plates()
    sheet = model.add_sheet(faces,policy=SheetTopologyPolicy(
        connectivity=ConnectivityPolicy.ALLOW_DISCONNECTED))
    source_part = model.sheets[sheet].part_id
    applied = apply_intersections(model,plan_intersections(
        model,(model.handle('sheet',sheet),),policy='connect'),policy='connect')
    assert set(model.sheets)=={sheet}
    assert model.sheets[sheet].part_id==source_part
    assert model.sheets[sheet].policy.non_manifold is NonManifoldPolicy.ALLOW_DECLARED
    assert set(model.sheets[sheet].declared_non_manifold_edges)=={
        edge.id for edge in applied.joint_edges}
    assert all(query_joint_edge(model,edge).declared for edge in applied.joint_edges)
    assert model.validate_topology()==()
    assert sum(chart.material_area for chart in query_trimmed_surface_charts(
        model,tuple(model.faces)).charts)==pytest.approx(75.)
    restored=from_dict(to_dict(model))
    assert all(query_joint_edge(restored,edge.id).declared for edge in applied.joint_edges)
    assert restored.validate_topology()==()


def test_failure_and_cancellation_restore_all_state_and_allocators():
    model, faces = _plates()
    before = deepcopy(to_dict(model))
    with pytest.raises(GeometryError, match='budget'):
        plan_intersections(model, faces, policy=IntersectionBatchPolicy(max_predicates=1))
    assert to_dict(model) == before
    plan = plan_intersections(model, faces, policy=ConnectionIntent.CONNECT)
    calls = 0
    def cancelled():
        nonlocal calls
        calls += 1
        return calls > 5
    with pytest.raises(GeometryError, match='cancelled'):
        apply_intersections(model, plan, policy=IntersectionBatchPolicy(cancellation_check=cancelled))
    assert calls > 5
    assert to_dict(model) == before


def test_positive_area_overlap_requires_explicit_ownership():
    model=GeometryModel()
    first=model.add_plate(model.add_points(((0,0,0),(2,0,0),(2,2,0),(0,2,0))))
    second=model.add_plate(model.add_points(((1,0,0),(3,0,0),(3,2,0),(1,2,0))))
    before=deepcopy(to_dict(model))
    with pytest.raises(GeometryError, match='positive-area'):
        plan_intersections(model, (first,second), policy=ConnectionIntent.CONNECT)
    assert to_dict(model)==before


@pytest.mark.parametrize('count',(1,8,17))
def test_growing_member_set_has_no_model_count_cap(count):
    from anygeometry import query_trimmed_surface_charts
    model=GeometryModel()
    face=model.add_plate(model.add_points(((0,0,0),(10,0,0),(10,10,0),(0,10,0))))
    sheet=model.add_sheet((face,))
    members=[]
    for index in range(count):
        y=10*(index+1)/(count+1)
        edge=model.add_line(*model.add_points(((0,y,0),(10,y,0))))
        members.append(model.add_member((edge,)))
    operands=(model.handle('sheet',sheet),*[model.handle('member',member) for member in members])
    plan=plan_intersections(model,operands,policy='connect')
    apply_intersections(model,plan,policy='connect')
    assert model.validate_topology()==()
    charts=query_trimmed_surface_charts(model,tuple(model.faces))
    assert sum(chart.material_area for chart in charts.charts)==pytest.approx(100.,abs=1e-8)
    for member in members:
        uses=tuple(model.member_edge_uses[use_id] for use_id in model.members[member].edge_use_ids)
        assert uses
        assert all(len(model.faces_using_edge(use.edge_id))==2 for use in uses)
    before=to_dict(model)
    repeated=apply_intersections(model,plan,policy='connect')
    assert repeated.reused
    assert to_dict(model)==before


@pytest.mark.parametrize('count',(1,8,25))
def test_growing_interior_ended_plate_cuts_preserve_each_original_material_domain(count):
    from anygeometry import query_trimmed_surface_charts
    model=GeometryModel()
    host=model.add_plate(model.add_points(((0,0,0),(100,0,0),(100,100,0),(0,100,0))))
    cutters=[]
    for index in range(count):
        x=100*(index+1)/(count+1)
        cutters.append(model.add_plate(model.add_points(((x,3,-1),(x,97,-1),(x,97,1),(x,3,1)))))
    original=to_dict(model)
    plan=plan_intersections(model,(host,*cutters),policy='connect')
    assert to_dict(model)==original
    result=apply_intersections(model,plan,policy='connect')
    assert len(result.joint_edges)==count
    assert model.validate_topology()==()
    charts=query_trimmed_surface_charts(model).charts
    area_by_face={chart.face.id:chart.material_area for chart in charts}
    for parent,area in ((host,10000.),*((face,188.) for face in cutters)):
        descendants=model.resolve_ref(EntityRef('face',parent))
        assert sum(area_by_face[face.id] for face in descendants)==pytest.approx(area,abs=1e-8)
    committed=to_dict(model)
    assert apply_intersections(model,plan,policy='connect').reused
    assert to_dict(model)==committed


def test_explicit_pair_budget_counts_operands_and_reuse_never_creates_topology():
    model,faces=_plates()
    before=to_dict(model)
    with pytest.raises(GeometryError,match='maximum_candidate_pairs=1'):
        plan_intersections(model,faces,policy=IntersectionBatchPolicy(max_candidate_pairs=1))
    assert to_dict(model)==before
    plan=plan_intersections(model,faces,policy='reuse_existing')
    with pytest.raises(GeometryError,match='compatible existing topology'):
        apply_intersections(model,plan,policy='reuse_existing')
    assert to_dict(model)==before


def test_member_crossings_and_interior_face_contact_share_vertices():
    model=GeometryModel()
    face=model.add_plate(model.add_points(((0,0,0),(4,0,0),(4,4,0),(0,4,0))))
    members=[]
    for points in (((.5,2,0),(3.5,2,0)), ((2,.5,0),(2,3.5,0)), ((2,2,-1),(2,2,1))):
        vertices=model.add_points(points)
        members.append(model.add_member((model.add_line(*vertices),)))
    operands=(model.handle('face',face), *(model.handle('member',member) for member in members))
    source=deepcopy(to_dict(model))
    plan=plan_intersections(model,operands,policy=ConnectionIntent.CONNECT)
    assert to_dict(model)==source
    apply_intersections(model,plan,policy=ConnectionIntent.CONNECT)
    assert model.validate_topology()==()
    center=[vertex.id for vertex in model.vertices.values() if tuple(vertex.position)==(2.,2.,0.)]
    assert len(center)==1
    incident=model.edges_using_vertex(center[0])
    assert {member for edge in incident for member in model.members_using_edge(edge)}==set(members)
    assert any(set(junction.member_ids)==set(members) for junction in model.junctions.values())
    assert any(attachment.member_id==members[2] and attachment.member_range.is_point
               for attachment in model.attachments.values())
    for member in members[:2]:
        uses=[use for use in model.member_edge_uses.values() if use.member_id==member]
        assert len(uses)==2
        assert all(model.faces_using_edge(use.edge_id) for use in uses)
        assert sorted((use.parent_range.start,use.parent_range.end) for use in uses)==[(0.,.5),(.5,1.)]
    committed=deepcopy(to_dict(model))
    again=plan_intersections(model,(*model.faces,*(model.handle('member',member) for member in members)),
                             policy=ConnectionIntent.CONNECT)
    assert apply_intersections(model,again,policy=ConnectionIntent.CONNECT).reused
    assert to_dict(model)==committed
