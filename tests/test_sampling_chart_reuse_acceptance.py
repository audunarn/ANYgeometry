"""Targeted acceptance controls for the frozen chart-arithmetic candidate."""
from fractions import Fraction as Q

import pytest
import anygeometry as owner
import anygeometry.prepared_material_curvilinear as module
from anygeometry.arrangement_geometry import freeze_edge
from anygeometry.definition_binding import definition_checksum
from anygeometry.plane_nurbs import CURRENT_WORK
from anygeometry.section_audit import Work
from test_sampling_chart_arithmetic_reuse import _plane_square, _record_boxes


@pytest.mark.parametrize('reset',['local','parent'])
def test_cached_hit_refuses_counter_reset_and_retains_parent_consumption(reset):
    model,regions,seeds,face=_plane_square();original=owner.to_dict(model)
    parent=Work(128,32);token=CURRENT_WORK.set(parent)
    spent=None
    reason=('^boundary sampling work counters changed$' if reset=='local' else
            '^boundary sampling parent work counters decreased; consumed work restored$')
    try:
        with pytest.raises(owner.GeometryError,match=reason):
            with module.prepared_material_boundary_sampling_operation():
                get=module._sampling_chart_boxes(model,regions,lambda:None)
                get(face,seeds[0][0].id,Q(0))
                local=CURRENT_WORK.get().local;spent=parent.boxes
                assert spent>0 and local.boxes>0
                (local if reset=='local' else parent).boxes=0
                get(face,seeds[0][0].id,Q(0))
        assert parent.boxes==spent
        assert CURRENT_WORK.get() is parent
        assert owner.to_dict(model)==original
    finally:CURRENT_WORK.reset(token)


def test_producer_without_bank_still_refuses_revision_drift():
    assert module._SAMPLING_ARITHMETIC.get() is None
    assert module._SAMPLING_ARITHMETIC_ISSUER.get() is None
    model,regions,seeds,face=_plane_square()
    get=module._sampling_chart_boxes(model,regions,lambda:None)
    get(face,seeds[0][0].id,Q(0));revision=model.revision
    model.add_point(7.,7.,7.)
    assert model.revision>revision
    with pytest.raises(owner.GeometryError,match='^boundary sampling arithmetic producer source changed$'):
        get(face,seeds[0][0].id,Q(0))


def test_ineligible_cos_boxes_and_operands_match_original_occurrence_selection(monkeypatch):
    from anygeometry.nurbs import CurveOnSurface,NurbsSurface
    from test_adaptive_rational_sampling import rounded
    model,regions,seeds=rounded();original=owner.to_dict(model)
    chart=regions.regions[0].sources[0];face=chart.face.id;edge=seeds[0][0].id
    references=module._sampling_references(model,regions)
    reference=references[face][1]
    # Exact OLD _query_boundary_chords occurrence selection, not new interval math.
    old_owner=reference.owner if hasattr(reference,'owner') else reference.model
    old_curve=(old_owner.edges[edge].curve if old_owner is not model else freeze_edge(model,edge))
    assert type(chart.support) is NurbsSurface and type(old_curve) is CurveOnSurface
    parameters=(Q(0),Q(1,2),Q(1))
    actual=module._uv_box
    expected=tuple(actual(chart.support,old_curve,t) for t in parameters)
    operands=[]
    def record(support,curve,t):
        operands.append((support,curve,t))
        return actual(support,curve,t)
    monkeypatch.setattr(module,'_uv_box',record)
    with module.prepared_material_boundary_sampling_operation():
        get=module._sampling_chart_boxes(model,regions,lambda:None)
        for attempt in range(2):
            assert tuple(get(face,edge,t) for t in parameters)==expected
            assert len(operands)==(attempt+1)*len(parameters)  # ineligible: genuine builders again
    for support,curve,t in operands:
        assert support is chart.support
        assert type(curve) is type(old_curve)
        assert definition_checksum(curve)==definition_checksum(old_curve)
        assert t in parameters
    assert owner.to_dict(model)==original


def test_region_only_tamper_refuses_retained_box():
    model,regions,seeds,face=_plane_square();original=owner.to_dict(model)
    selected=regions.regions;signature=definition_checksum(regions)
    try:
        with pytest.raises(owner.GeometryError,match='^boundary sampling arithmetic producer source changed$'):
            with module.prepared_material_boundary_sampling_operation():
                get=module._sampling_chart_boxes(model,regions,lambda:None)
                get(face,seeds[0][0].id,Q(0))
                object.__setattr__(regions,'regions',())
                assert definition_checksum(regions)!=signature
                get(face,seeds[0][0].id,Q(0))
    finally:object.__setattr__(regions,'regions',selected)
    assert owner.to_dict(model)==original


def test_exact_box_is_unchanged_when_evicted_and_recomputed(request):
    # Component storage only: private namespace data does not issue a geometry receipt.
    state,cleanup=module._new_sampling_arithmetic();chart,world=object(),object()
    model=owner.GeometryModel();calls=[]
    expected=((Q(-2,3),Q(4,5)),(Q(7,11),Q(9,7)))
    def box():calls.append(True);return expected
    def bound():return Q(1,2),((Q(0),Q(1)),)
    try:
        first=state[0](chart,model,(),'target',None,box)
        assert first==expected
        for key in range(4095):state[0](world,model,(),key,Q(1),bound)
        assert state[0](chart,model,(),'target',None,box)==expected and len(calls)==1
        state[0](world,model,(),4095,Q(1),bound)
        third=state[0](chart,model,(),'target',None,box)
        assert third==first==expected and len(calls)==2
        assert all(type(x) is Q for pair in third for x in pair)
    finally:cleanup()
    request.node.user_properties.append(('execution_scope','bounded-arithmetic-bank-component'))


def test_unknown_parent_protocol_recomputes_actual_boxes(monkeypatch):
    class InheritedBudget(Work):
        pass
    calls=_record_boxes(monkeypatch)
    model,regions,seeds,face=_plane_square();parent=InheritedBudget(128,32)
    token=CURRENT_WORK.set(parent)
    try:
        with module.prepared_material_boundary_sampling_operation():
            get=module._sampling_chart_boxes(model,regions,lambda:None)
            first=get(face,seeds[0][0].id,Q(0));spent=parent.boxes
            assert get(face,seeds[0][0].id,Q(0))==first
            assert len(calls)==2 and parent.boxes>spent>0
    finally:CURRENT_WORK.reset(token)
