"""Operation-local chart arithmetic reuses computation, never Work authority."""
from fractions import Fraction as Q
from dataclasses import replace
import json

import anygeometry as owner
import anygeometry.prepared_material_curvilinear as module
import pytest
from _planning_numpy_profile_contract import numpy_profile_test
from anygeometry.plane_nurbs import CURRENT_WORK,PlaneNurbsNotCertified
from anygeometry.predicates import DEFAULT_INTERSECTION_QUALIFICATION_POLICY as POLICY
from anygeometry.section_audit import Work


def _plane_square(offset=0.):
    model=owner.GeometryModel()
    vertices=model.add_points(((0.,0.,offset),(1.,0.,offset),(1.,1.,offset),(0.,1.,offset)))
    edges=tuple(model.add_polyline(vertices,close=True))
    face=model.add_face(edges,surface=owner.Plane((0.,0.,offset),(1.,0.,0.),(0.,1.,0.)))
    regions=owner.query_material_surface_regions(model,(face,))
    seeds=tuple((model.handle('edge',edge),(Q(0),Q(1,2),Q(1))) for edge in edges)
    return model,regions,seeds,face


def _geometry(result):
    return (result.edge_parameters,result.chords.intervals,result.chords.reference_uv_by_face,
            result.chord_error,result.error_measure,result.proposal_fraction)


def test_public_sampling_builds_each_real_chart_endpoint_once(monkeypatch,tmp_path):
    actual=module._uv_box;calls=[]
    def record(support,curve,t):
        calls.append((id(support),id(curve),t))
        return actual(support,curve,t)
    monkeypatch.setattr(module,'_uv_box',record)
    model,regions,seeds,_=_plane_square()
    original=owner.to_dict(model)
    with module.prepared_material_boundary_sampling_operation():
        first=module.query_prepared_material_boundary_sampling(model,regions,seeds,chord_error=.1)
        used=len(calls);spent=CURRENT_WORK.get().boxes
        necessary=sum(len(parameters) for _,parameters in first.edge_parameters)
        (tmp_path/'first-builder-counts.json').write_text(json.dumps(dict(
            actual_endpoint_builders=used,unique_face_edge_stations=necessary,
            actual_parent_boxes=spent,intervals=len(first.chords.intervals)),indent=2))
        assert used==necessary
        second=module.query_prepared_material_boundary_sampling(
            model,regions,first.edge_parameters,chord_error=.1)
        assert len(calls)==used
        assert CURRENT_WORK.get().boxes>=spent
        assert second is not first and second.chords is not first.chords
        assert _geometry(second)==_geometry(first)
        (tmp_path/'cold-warm-counts.json').write_text(json.dumps(dict(
            cold_endpoint_builders=used,warm_endpoint_builders=len(calls)-used,
            cold_boxes=spent,warm_boxes=CURRENT_WORK.get().boxes-spent),indent=2))
        module.validate_prepared_material_boundary_sampling_binding(model,regions,first)
        module.validate_prepared_material_boundary_sampling_binding(model,regions,second)
    assert owner.to_dict(model)==original


def _record_boxes(monkeypatch):
    actual=module._uv_box;calls=[]
    def record(support,curve,t):
        calls.append((support,curve,t))
        return actual(support,curve,t)
    monkeypatch.setattr(module,'_uv_box',record)
    return calls


def test_keys_separate_models_faces_edges_and_stations(monkeypatch):
    calls=_record_boxes(monkeypatch)
    model,_,seeds,face=_plane_square()
    shared=model.edges[seeds[0][0].id];a,b=shared.start,shared.end
    e,f=model.add_points(((0.,0.,1.),(1.,0.,1.)))
    side_edges=model.add_polyline((a,e,f,b))
    other_face=model.add_face((shared.id,*side_edges),surface=owner.Plane((0.,0.,0.),(0.,0.,1.),(1.,0.,0.)))
    regions=owner.query_material_surface_regions(model,(face,other_face))
    other,other_regions,other_seeds,other_id=_plane_square(2.)
    with module.prepared_material_boundary_sampling_operation():
        get=module._sampling_chart_boxes(model,regions,lambda:None)
        first=get(face,shared.id,Q(1,2))
        assert get(face,shared.id,Q(1,2))==first and len(calls)==1
        assert get(other_face,shared.id,Q(1,2))!=first and len(calls)==2
        get(face,shared.id,Q(0));assert len(calls)==3
        get(face,seeds[1][0].id,Q(1,2));assert len(calls)==4
        foreign=module._sampling_chart_boxes(other,other_regions,lambda:None)
        foreign(other_id,other_seeds[0][0].id,Q(1,2));assert len(calls)==5


def test_refusals_are_not_cached(monkeypatch):
    actual=module._uv_box;calls=[];failure=RuntimeError('actual enclosure failure')
    def fail_once(support,curve,t):
        calls.append(t)
        if len(calls)==1:raise failure
        return actual(support,curve,t)
    monkeypatch.setattr(module,'_uv_box',fail_once)
    model,regions,seeds,face=_plane_square()
    with module.prepared_material_boundary_sampling_operation():
        get=module._sampling_chart_boxes(model,regions,lambda:None)
        with pytest.raises(RuntimeError) as caught:get(face,seeds[0][0].id,Q(0))
        assert caught.value is failure
        value=get(face,seeds[0][0].id,Q(0))
        assert get(face,seeds[0][0].id,Q(0))==value and len(calls)==2


def test_real_plane_nurbs_endpoint_box_is_reused_without_narrowing(monkeypatch):
    from test_plane_source_cells import direct_plane_disk
    from anygeometry.support_curves import PlaneNurbsCurve
    model,face=direct_plane_disk()
    regions=owner.query_material_surface_regions(model,(face.id,))
    edge=face.loop[0].edge;curve=model.edges[edge].curve
    assert type(curve) is PlaneNurbsCurve
    expected=module._uv_box(face.surface,curve,Q(1,2))
    calls=_record_boxes(monkeypatch)
    with module.prepared_material_boundary_sampling_operation():
        get=module._sampling_chart_boxes(model,regions,lambda:None)
        first=get(face.id,edge,Q(1,2));spent=CURRENT_WORK.get().boxes
        assert first==expected and spent>0 and len(calls)==1
        assert get(face.id,edge,Q(1,2))==expected
        assert CURRENT_WORK.get().boxes==spent and len(calls)==1


@pytest.mark.parametrize('fault',['callback','cancel','calculator','work-context','source','builder-cell',
                                  'work-take','work-guard','work-counter-descriptor'])
def test_hits_recheck_authority_and_preserve_failure_identity(monkeypatch,fault):
    model,regions,seeds,face=_plane_square();edge=seeds[0][0].id
    armed=False;failure=RuntimeError('actual callback failure');signal=owner.CancellationSignal()
    changed_calls=[]
    def changed(*args):
        changed_calls.append(True)
        pytest.fail('changed work calculator executed')
    def check():
        if armed:
            if fault=='callback':raise failure
            if fault=='cancel':module._check(signal,'chart endpoint')
    saved_cell=None;token=None
    try:
        with pytest.raises((owner.GeometryError,RuntimeError)) as caught:
            with module.prepared_material_boundary_sampling_operation():
                get=module._sampling_chart_boxes(model,regions,check)
                get(face,edge,Q(0));armed=True
                if fault=='cancel':signal.cancel()
                elif fault=='calculator':monkeypatch.setattr(module,'_uv_box',lambda *args:pytest.fail('changed calculator executed'))
                elif fault=='work-context':token=CURRENT_WORK.set(Work())
                elif fault=='source':model.add_point(9.,9.,9.)
                elif fault=='work-take':monkeypatch.setattr(Work,'take',changed)
                elif fault=='work-guard':monkeypatch.setattr(module._SamplingWork,'guard',changed)
                elif fault=='work-counter-descriptor':monkeypatch.setattr(Work,'boxes',property(changed),raising=False)
                elif fault=='builder-cell':
                    cell=dict(zip(module._sampling_chart_boxes.__code__.co_freevars,module._sampling_chart_boxes.__closure__))['builder']
                    saved_cell=cell,cell.cell_contents
                    cell.cell_contents=lambda *args:pytest.fail('changed producer builder executed')
                try:get(face,edge,Q(0))
                finally:
                    if token is not None:
                        CURRENT_WORK.reset(token);token=None
        if fault=='callback':assert caught.value is failure
        else:assert type(caught.value) is not RuntimeError
        assert changed_calls==[]
    finally:
        if token is not None:CURRENT_WORK.reset(token)
        if saved_cell is not None:saved_cell[0].cell_contents=saved_cell[1]


@pytest.mark.parametrize('outer',[False,True])
def test_hits_at_capacity_are_free_but_exhausted_work_remains_terminal(outer):
    from anygeometry.arrangement_geometry import freeze_edge
    model,regions,seeds,face=_plane_square();edge=seeds[0][0].id
    parent=Work(128,32) if outer else None
    token=CURRENT_WORK.set(parent) if outer else None
    try:
        with module.prepared_material_boundary_sampling_operation(qualification=replace(POLICY,max_boxes_per_pair=128)):
            work=parent if outer else CURRENT_WORK.get()
            get=module._sampling_chart_boxes(model,regions,lambda:None)
            value=get(face,edge,Q(0))
            support=regions.regions[0].sources[0].support;curve=freeze_edge(model,edge)
            while work.boxes<work.max_boxes:module._uv_box(support,curve,Q(1,4))
            at_max=work.boxes
            assert get(face,edge,Q(0))==value and work.boxes==at_max==128
            with pytest.raises(PlaneNurbsNotCertified,match='qualification budget exhausted'):
                module._uv_box(support,curve,Q(3,4))
            spent=work.boxes;assert spent==129
            with pytest.raises(PlaneNurbsNotCertified,match='qualification budget exhausted'):
                get(face,edge,Q(0))
            assert work.boxes==spent
    finally:
        if token is not None:CURRENT_WORK.reset(token)


def test_nested_and_exited_scopes_never_reuse_another_bank(monkeypatch):
    calls=_record_boxes(monkeypatch);model,regions,seeds,face=_plane_square();edge=seeds[0][0].id
    with module.prepared_material_boundary_sampling_operation():
        first=module._sampling_chart_boxes(model,regions,lambda:None);first(face,edge,Q(0))
        with module.prepared_material_boundary_sampling_operation():
            nested=module._sampling_chart_boxes(model,regions,lambda:None);nested(face,edge,Q(0))
            assert len(calls)==2
        first(face,edge,Q(0));assert len(calls)==2
    with pytest.raises(owner.GeometryError,match='context changed'):first(face,edge,Q(0))
    with module.prepared_material_boundary_sampling_operation():
        module._sampling_chart_boxes(model,regions,lambda:None)(face,edge,Q(0))
        assert len(calls)==3


def test_exact_boxes_share_the_same_bounded_bank_with_world_bounds(request):
    # Component storage test only: these private namespaces issue no geometry receipt.
    state,cleanup=module._new_sampling_arithmetic();world,chart=object(),object();model=owner.GeometryModel();calls=[]
    def bound():calls.append(1);return Q(1,2),((Q(0),Q(1)),)
    def box():return ((Q(0),Q(0)),(Q(1),Q(1)))
    try:
        first=state[0](world,model,(),0,Q(1),bound)
        for key in range(4095):state[0](chart,model,(),key,None,box)
        assert state[0](world,model,(),0,Q(1),bound)==first and len(calls)==1
        state[0](chart,model,(),4095,None,box)
        assert state[0](world,model,(),0,Q(1),bound)==first and len(calls)==2
    finally:cleanup()
    request.node.user_properties.append(('execution_scope','bounded-arithmetic-bank-component'))


@pytest.mark.parametrize('kind',['plane','cylinder'])
@numpy_profile_test
def test_sampling_changes_preserve_cold_warm_loaded_host_admission(kind):
    from anygeometry import _planning_manifest as manifest
    if kind=='plane':model,_,_,_=_plane_square()
    else:
        model=owner.GeometryModel()
        points=model.add_points(((1.,0.,0.),(1.,1.,0.),(0.,1.,0.)))
        arc=model.add_arc(*points);model.extrude((arc,),(0.,0.,1.))
    cold=manifest.capture_loaded_manifest(model=model)
    owner.query_trimmed_surface_charts(model,tuple(model.faces))
    cold.verify()
    warm=manifest.capture_loaded_manifest(model=model)
    warm.verify()
    assert cold.wire['callables_sha256']==warm.wire['callables_sha256']
