import numpy as np
import pytest
from dataclasses import replace
from anygeometry import (GeometryModel, GeometryError, query_trimmed_surface_charts,
    evaluate_trimmed_surface_chart, validate_trimmed_surface_charts_binding, to_dict,
    plan_intersections, apply_intersections, ConnectionIntent)


def test_bound_material_charts_preserve_source_and_check_batches():
    model=GeometryModel()
    face=model.add_plate(model.add_points(((0,0,0),(4,0,0),(4,4,0),(0,4,0))))
    before=to_dict(model)
    result=query_trimmed_surface_charts(model)
    assert to_dict(model)==before
    assert result.charts[0].material_area==pytest.approx(16)
    values=evaluate_trimmed_surface_chart(model,result,face,np.array([[[.25,.25],[.5,.5]]]),require_material=True)
    assert values.shape==(1,2,3)
    np.testing.assert_allclose(values,[[[1,1,0],[2,2,0]]])
    with pytest.raises(GeometryError,match='outside material'):
        evaluate_trimmed_surface_chart(model,result,face,[[2,2]],require_material=True)
    assert evaluate_trimmed_surface_chart(model,result,face,np.empty((0,2))).shape==(0,3)
    for changed in (replace(result.charts[0],material_area=17.),
                    replace(result.charts[0],world_tolerance=1.),
                    replace(result.charts[0],domain=replace(result.charts[0].domain,face_id=999))):
        with pytest.raises(GeometryError,match='binding changed'):
            validate_trimmed_surface_charts_binding(model,replace(result,charts=(changed,)))
    other=model.clone()
    with pytest.raises(GeometryError,match='another model'):
        validate_trimmed_surface_charts_binding(other,result)
    model.add_point(9,9,9)
    with pytest.raises(GeometryError,match='stale'):
        evaluate_trimmed_surface_chart(model,result,face,[[.5,.5]])


def test_fragmented_material_charts_conserve_analytic_area():
    model=GeometryModel()
    a=model.add_plate(model.add_points(((0,0,0),(4,0,0),(4,4,0),(0,4,0))))
    b=model.add_plate(model.add_points(((2,1,-1),(2,3,-1),(2,3,1),(2,1,1))))
    plan=plan_intersections(model,(a,b),policy=ConnectionIntent.CONNECT)
    apply_intersections(model,plan,policy=ConnectionIntent.CONNECT)
    result=query_trimmed_surface_charts(model)
    assert sum(chart.material_area for chart in result.charts)==pytest.approx(20)


def test_public_charts_preserve_construction_seams_and_physical_joint_identity():
    from anygeometry import EntityRef
    model=GeometryModel()
    a=model.add_plate(model.add_points(((0,0,0),(4,0,0),(4,4,0),(0,4,0))))
    b=model.add_plate(model.add_points(((2,1,-1),(2,3,-1),(2,3,1),(2,1,1))))
    apply_intersections(model,plan_intersections(model,(a,b),policy=ConnectionIntent.CONNECT),
                        policy=ConnectionIntent.CONNECT)
    before=to_dict(model)
    result=query_trimmed_surface_charts(model)
    paths=[path for chart in result.charts for loop in chart.boundaries for path in loop]
    expected={edge for edge in model.edges if 'intersection_decomposition_seam'
              in model.tags_for(EntityRef('edge',edge))}
    assert expected
    assert {path.source_edge for path in paths if path.decomposition} == expected
    assert any(not path.decomposition for path in paths)
    validate_trimmed_surface_charts_binding(model,result)
    assert to_dict(model)==before
    chart=next(chart for chart in result.charts if any(path.decomposition
               for loop in chart.boundaries for path in loop))
    loops=tuple(tuple(replace(path,decomposition=False) for path in loop) for loop in chart.boundaries)
    changed=replace(result,charts=tuple(replace(item,domain=replace(item.domain,boundaries=loops))
                       if item.face==chart.face else item for item in result.charts))
    with pytest.raises(GeometryError,match='definition binding changed'):
        validate_trimmed_surface_charts_binding(model,changed)


@pytest.mark.parametrize('intent', (ConnectionIntent.CONNECT, ConnectionIntent.IMPRINT))
def test_later_physical_cut_promotes_a_construction_seam(intent):
    from anygeometry import EntityRef, from_dict
    model=GeometryModel()
    a=model.add_plate(model.add_points(((0,0,0),(4,0,0),(4,4,0),(0,4,0))))
    b=model.add_plate(model.add_points(((2,1,-1),(2,3,-1),(2,3,1),(2,1,1))))
    apply_intersections(model,plan_intersections(model,(a,b),policy=intent),policy=intent)
    seams={edge for edge in model.edges if 'intersection_decomposition_seam'
           in model.tags_for(EntityRef('edge',edge))}
    seam=min(seams)
    record=model.edges[seam]
    start=np.asarray(model.vertex_position(record.start))
    end=np.asarray(model.vertex_position(record.end))
    offset=np.array((0.,0.,1.))
    model.add_plate(model.add_points((start-offset,end-offset,end+offset,start+offset)))
    before=to_dict(model)
    plan=plan_intersections(model,tuple(model.faces),policy=intent)
    assert to_dict(model)==before
    result=apply_intersections(model,plan,policy=intent)
    assert seam in {handle.id for handle in result.joint_edges}
    assert 'intersection_decomposition_seam' not in model.tags_for(EntityRef('edge',seam))
    paths=[path for chart in query_trimmed_surface_charts(model).charts
           for loop in chart.boundaries for path in loop if path.source_edge==seam]
    assert len(paths)>=3
    assert all(not path.decomposition for path in paths)
    assert model.validate_topology()==()
    assert sum(chart.material_area for chart in query_trimmed_surface_charts(model).charts)==pytest.approx(
        20+2*np.linalg.norm(end-start))
    after=to_dict(model)
    assert apply_intersections(model,plan,policy=intent).reused
    assert to_dict(model)==after
    restored=from_dict(after)
    assert 'intersection_decomposition_seam' not in restored.tags_for(EntityRef('edge',seam))


def test_same_sheet_authored_coplanar_boundaries_still_connect():
    from anygeometry.structural import SheetTopologyPolicy, ConnectivityPolicy
    model=GeometryModel()
    faces=[model.add_plate(model.add_points(points)) for points in (
        ((0,0,0),(1,0,0),(1,1,0),(0,1,0)),
        ((1,0,0),(2,0,0),(2,1,0),(1,1,0)))]
    model.add_sheet(faces,policy=SheetTopologyPolicy(connectivity=ConnectivityPolicy.ALLOW_DISCONNECTED))
    assert not (set(use.edge for use in model.faces[faces[0]].loop) &
                set(use.edge for use in model.faces[faces[1]].loop))
    result=apply_intersections(model,plan_intersections(model,faces,policy='connect'),policy='connect')
    assert len(result.joint_edges)==1
    assert set(model.faces_using_edge(result.joint_edges[0].id))==set(faces)
    assert model.validate_topology()==()


def test_validated_chart_reuse_still_rejects_changed_evidence_and_direct_edits(monkeypatch):
    from anygeometry.material_arrangement import MaterialDomain
    model=GeometryModel()
    face=model.add_plate(model.add_points(((0,0,0),(4,0,0),(4,4,0),(0,4,0))))
    result=query_trimmed_surface_charts(model)
    before=to_dict(model)
    integration=MaterialDomain.area_loop
    calls=[]
    def measured(*args,**options):
        calls.append(1)
        return integration(*args,**options)
    monkeypatch.setattr(MaterialDomain,'area_loop',measured)
    topology_validation=GeometryModel.validate_topology
    topology_calls=[]
    def measured_topology(*args,**options):
        topology_calls.append(1)
        return topology_validation(*args,**options)
    monkeypatch.setattr(GeometryModel,'validate_topology',measured_topology)
    validate_trimmed_surface_charts_binding(model,result)
    count=len(calls)
    assert count
    qualified_count=len(topology_calls)
    assert qualified_count
    validate_trimmed_surface_charts_binding(model,result)
    assert len(calls)==count
    assert len(topology_calls)==qualified_count
    assert to_dict(model)==before
    changed=replace(result,charts=(replace(result.charts[0],material_area=17.),))
    with pytest.raises(GeometryError,match='material area binding changed'):
        validate_trimmed_surface_charts_binding(model,changed)
    assert len(calls)>count
    # A direct dictionary edit does not advance revision, but its complete
    # document checksum must invalidate previously qualified evidence.
    model._faces[face]=replace(model.faces[face],metadata={'changed':True})
    with pytest.raises(GeometryError,match='source binding changed'):
        validate_trimmed_surface_charts_binding(model,result)


def test_content_fingerprint_cannot_replace_public_topology_qualification():
    from anygeometry.serialization import _serialized_model_state
    model=GeometryModel()
    face=model.add_plate(model.add_points(((0,0,0),(4,0,0),(4,4,0),(0,4,0))))
    qualified=to_dict(model)
    assert _serialized_model_state(model)==qualified
    result=query_trimmed_surface_charts(model)
    validate_trimmed_surface_charts_binding(model,result)
    # Simulate an invalid direct edit without the normal mutation/revision API.
    edge=model.faces[face].loop[0].edge
    model._edges[edge]=replace(model.edges[edge],end=model.edges[edge].start)
    assert _serialized_model_state(model)['checksum']!=qualified['checksum']
    with pytest.raises(GeometryError,match='source binding changed'):
        validate_trimmed_surface_charts_binding(model,result)
    with pytest.raises(GeometryError,match='invalid topology'):
        to_dict(model)


def test_alternating_chart_collections_reuse_only_completed_qualification(monkeypatch):
    from anygeometry.material_arrangement import MaterialDomain
    model=GeometryModel()
    for x in (0.,2.):
        model.add_plate(model.add_points(((x,0,0),(x+1,0,0),(x+1,1,0),(x,1,0))))
    both=query_trimmed_surface_charts(model)
    one=replace(both,charts=both.charts[:1])
    validate_trimmed_surface_charts_binding(model,both)
    validate_trimmed_surface_charts_binding(model,one)
    original=MaterialDomain.original_world_area
    def unexpected(*args,**kwargs):
        raise AssertionError('unchanged collection was requalified')
    monkeypatch.setattr(MaterialDomain,'original_world_area',unexpected)
    validate_trimmed_surface_charts_binding(model,both)
    validate_trimmed_surface_charts_binding(model,one)
    with pytest.raises(GeometryError,match='cancelled'):
        validate_trimmed_surface_charts_binding(model,both,cancellation_check=lambda _: True)
    monkeypatch.setattr(MaterialDomain,'original_world_area',original)
    forged=replace(one,charts=(replace(one.charts[0],material_area=2.),))
    for _ in range(2):
        with pytest.raises(GeometryError,match='material area binding changed'):
            validate_trimmed_surface_charts_binding(model,forged)
    face=one.charts[0].face.id
    model._faces[face]=replace(model.faces[face],metadata={'changed':True})
    for evidence in (one,both):
        with pytest.raises(GeometryError,match='source binding changed'):
            validate_trimmed_surface_charts_binding(model,evidence)


def test_chart_cache_eviction_requalifies_without_limiting_collections(monkeypatch):
    from anygeometry.material_arrangement import MaterialDomain
    model=GeometryModel()
    for x in range(9):
        model.add_plate(model.add_points(((2*x,0,0),(2*x+1,0,0),(2*x+1,1,0),(2*x,1,0))))
    all_charts=query_trimmed_surface_charts(model)
    collections=[replace(all_charts,charts=(chart,)) for chart in all_charts.charts]
    calls=[]
    original=MaterialDomain.original_world_area
    def measured(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(MaterialDomain,'original_world_area',measured)
    for evidence in collections:
        validate_trimmed_surface_charts_binding(model,evidence)
    assert len(calls)==9
    validate_trimmed_surface_charts_binding(model,collections[-1])
    assert len(calls)==9
    validate_trimmed_surface_charts_binding(model,collections[0])
    assert len(calls)==10


def test_cancelled_chart_qualification_is_not_cached(monkeypatch):
    from anygeometry.material_arrangement import MaterialDomain
    model=GeometryModel()
    model.add_plate(model.add_points(((0,0,0),(1,0,0),(1,1,0),(0,1,0))))
    evidence=query_trimmed_surface_charts(model)
    calls=[]
    original=MaterialDomain.original_world_area
    def measured(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(MaterialDomain,'original_world_area',measured)
    with pytest.raises(GeometryError,match='cancelled'):
        validate_trimmed_surface_charts_binding(model,evidence,
            cancellation_check=lambda phase: phase=='trimmed surface chart validation complete')
    assert len(calls)==1
    validate_trimmed_surface_charts_binding(model,evidence)
    assert len(calls)==2
