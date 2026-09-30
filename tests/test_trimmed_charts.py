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
