from dataclasses import replace
from fractions import Fraction as Q
import pytest
import anygeometry as owner
from anygeometry.plane_nurbs import CURRENT_WORK,charge_box,PlaneNurbsNotCertified
from anygeometry.predicates import DEFAULT_INTERSECTION_QUALIFICATION_POLICY as POLICY
from anygeometry.section_audit import Work
from anygeometry.prepared_material_curvilinear import (
    prepared_material_boundary_sampling_operation as operation,
    query_prepared_material_boundary_sampling as query,
    validate_prepared_material_boundary_sampling_binding as validate)
from test_adaptive_rational_sampling import rounded


def test_nested_scope_charges_parent_and_local_without_renewal():
    initial=CURRENT_WORK.get()
    with operation(qualification=replace(POLICY,max_boxes_per_pair=2)):
        parent=CURRENT_WORK.get();charge_box()
        with operation():
            local=CURRENT_WORK.get();charge_box()
            assert local.boxes==1 and parent.boxes==2
            with pytest.raises(PlaneNurbsNotCertified):charge_box()
        assert CURRENT_WORK.get() is parent and parent.boxes==3
    assert CURRENT_WORK.get() is initial


def test_local_cap_still_applies_under_larger_parent():
    with operation():
        parent=CURRENT_WORK.get()
        with operation(qualification=replace(POLICY,max_boxes_per_pair=1)):
            charge_box()
            with pytest.raises(PlaneNurbsNotCertified):charge_box()
        assert parent.boxes==2


def test_scope_restores_on_callback_exception_and_limit_mutation():
    initial=CURRENT_WORK.get();error=RuntimeError('callback')
    with pytest.raises(RuntimeError) as caught:
        with operation():raise error
    assert caught.value is error and CURRENT_WORK.get() is initial
    with pytest.raises(owner.GeometryError,match='limits changed'):
        with operation():CURRENT_WORK.get().max_depth+=1
    assert CURRENT_WORK.get() is initial


def test_actual_sampling_local_diagnostics_and_aggregate_parent():
    model,regions,seeds=rounded()
    with operation():
        parent=CURRENT_WORK.get()
        first=query(model,regions,seeds,chord_error=.1,error_measure='cell_reference',proposal_fraction=.25)
        used=parent.boxes
        second=query(model,regions,seeds,chord_error=.1,error_measure='cell_reference',proposal_fraction=.25)
        assert parent.boxes>used>0
        assert first.edge_parameters==second.edge_parameters
        validate(model,regions,first);validate(model,regions,second)


def test_actual_query_refuses_exhausted_parent_and_restores_context():
    model,regions,seeds=rounded();parent=Work(1,32);parent.boxes=1
    token=CURRENT_WORK.set(parent)
    try:
        with pytest.raises(owner.GeometryError):query(model,regions,seeds,chord_error=.1)
        assert CURRENT_WORK.get() is parent and parent.boxes>1
    finally:CURRENT_WORK.reset(token)


def square_with_weighted_line():
    from anygeometry.nurbs import CurveOnSurface,NurbsCurve
    from test_local_curve_geometry import flat
    model=owner.GeometryModel();support=flat();uv=((0.,0.),(1.,0.),(1.,1.),(0.,1.))
    vertices=model.add_points(tuple((*p,0.) for p in uv));edges=[]
    for i,p in enumerate(uv):
        curve=CurveOnSurface(support,NurbsCurve(1,(0.,0.,1.,1.),(p,uv[(i+1)%4]),weights=(1.,4.) if i==0 else None))
        edges.append(model.add_curve(vertices[i],vertices[(i+1)%4],curve))
    face=model.add_face(edges,surface=support)
    return model,owner.query_material_surface_regions(model,(face,)),tuple((model.handle('edge',e),(Q(0),Q(1))) for e in edges)


def test_linear_image_reference_does_not_inherit_descriptor_speed():
    model,regions,seeds=square_with_weighted_line()
    cell=query(model,regions,seeds,chord_error=.02,error_measure='cell_reference',proposal_fraction=.25)
    descriptor=query(model,regions,seeds,chord_error=.02,error_measure='parameterized',proposal_fraction=.25)
    assert len(cell.edge_parameters[0][1])==2
    assert len(descriptor.edge_parameters[0][1])>2
    validate(model,regions,cell)
    object.__setattr__(cell,'error_measure','geometric')
    with pytest.raises(owner.GeometryError):validate(model,regions,cell)


def test_rational_reference_certifies_full_vector_interpolation():
    model,regions,seeds=rounded()
    result=query(model,regions,seeds,chord_error=.05,error_measure='cell_reference',proposal_fraction=.25)
    # Proposal sets need not match a coarser descriptor enclosure. Check the
    # actual full-vector interpolation at independent stations on each segment.
    import numpy as np
    for handle,parameters in result.edge_parameters:
        curve=model.edges[handle.id].curve
        for a,b in zip(parameters,parameters[1:]):
            first,last=(np.asarray(curve.exact_point(t),dtype=float) for t in (a,b))
            for k in range(17):
                t=Q(k,16)
                actual=np.asarray(curve.exact_point(a+t*(b-a)),dtype=float)
                linear=first+float(t)*(last-first)
                assert np.linalg.norm(actual-linear)<=.05*.25
    validate(model,regions,result)


def test_true_plane_section_reference_uses_plane_occurrence():
    from test_plane_source_cells import direct_plane_disk
    model,face=direct_plane_disk();regions=owner.query_material_surface_regions(model,(face.id,))
    seeds=tuple((model.handle('edge',u.edge),(Q(0),Q(1))) for u in face.loop)
    result=query(model,regions,seeds,chord_error=.1,error_measure='cell_reference',proposal_fraction=.25)
    validate(model,regions,result)
    assert result.error_measure=='cell_reference'
    assert all(len(parameters)>=2 for _,parameters in result.edge_parameters)


def test_actual_callback_cancel_and_raw_source_mutation_refuse():
    model,regions,seeds=rounded();error=RuntimeError('sampling cancel')
    def cancel(phase):raise error
    previous=CURRENT_WORK.get()
    with pytest.raises(RuntimeError) as caught:
        with operation():query(model,regions,seeds,chord_error=.1,error_measure='cell_reference',cancellation_check=cancel)
    assert caught.value is error and CURRENT_WORK.get() is previous
    edge=seeds[0][0].id;curve=model.edges[edge].curve
    def mutate(phase):
        object.__setattr__(curve.pcurve,'weights',(1.,2.,1.))
    with pytest.raises(owner.GeometryError):
        with operation():query(model,regions,seeds,chord_error=.1,error_measure='cell_reference',cancellation_check=mutate)
    assert CURRENT_WORK.get() is previous


def test_every_actual_shared_trace_occurrence_is_checked(monkeypatch):
    from test_plane_split import bowl
    from anygeometry.regular_source_reference import NurbsReference
    model,face,plane=bowl()
    result=owner.apply_plane_split(model,owner.plan_plane_split(model,(face,),plane),policy='imprint')
    regions=owner.query_material_surface_regions(model,tuple(model.faces))
    trace=result.trace_edges[0];expected=set(model.faces_using_edge(trace));seen=set()
    original=NurbsReference.side
    def observed(reference,edge,*args):
        if edge==trace:seen.add(reference.face.id)
        return original(reference,edge,*args)
    monkeypatch.setattr(NurbsReference,'side',observed)
    query(model,regions,((model.handle('edge',trace),(Q(0),Q(1,4),Q(1,2),Q(3,4),Q(1))),),
        chord_error=.1,error_measure='cell_reference',proposal_fraction=.25)
    assert len(expected)==2 and seen==expected


def test_optional_rational_jet_overenclosure_is_unavailable_not_false_chart_exit():
    from anygeometry.prepared_material_curvilinear import _cos_c2_world_chord_bound,_world_chord
    model,regions,seeds=square_with_weighted_line();curve=model.edges[seeds[0][0].id].curve
    with operation():
        assert _cos_c2_world_chord_bound(curve,Q(0),Q(1),Q(0),lambda:None) is None
        bound,_=_world_chord(curve,Q(0),Q(1),((0.,0.,0.),(1.,0.,0.)),.01,lambda:None)
        assert bound<=.01


@pytest.mark.parametrize('when',[1,2])
def test_optional_rational_unavailable_preserves_callback_exception_identity(when):
    from anygeometry.prepared_material_curvilinear import _cos_c2_world_chord_bound
    model,regions,seeds=square_with_weighted_line();curve=model.edges[seeds[0][0].id].curve
    error=RuntimeError('original callback');calls=[]
    def check():
        calls.append(1)
        if len(calls)==when:raise error
    with pytest.raises(RuntimeError) as caught:
        with operation():_cos_c2_world_chord_bound(curve,Q(0),Q(1),Q(0),check)
    assert caught.value is error


def test_optional_rational_unavailable_preserves_work_and_mutation_refusals():
    from anygeometry.prepared_material_curvilinear import _cos_c2_world_chord_bound
    model,regions,seeds=square_with_weighted_line();curve=model.edges[seeds[0][0].id].curve
    work=Work(0,32);token=CURRENT_WORK.set(work)
    try:
        with pytest.raises(PlaneNurbsNotCertified):
            _cos_c2_world_chord_bound(curve,Q(0),Q(1),Q(0),lambda:None)
    finally:CURRENT_WORK.reset(token)
    calls=[]
    def mutate():
        calls.append(1)
        if len(calls)==2:object.__setattr__(curve.pcurve,'weights',(1.,3.))
    with pytest.raises(owner.GeometryError,match='changed|definition binding is invalid'):
        with operation():_cos_c2_world_chord_bound(curve,Q(0),Q(1),Q(0),mutate)


def test_optional_rational_real_chart_exit_still_refuses():
    from anygeometry.prepared_material_curvilinear import _cos_c2_world_chord_bound
    model,regions,seeds=square_with_weighted_line();curve=model.edges[seeds[0][0].id].curve
    object.__setattr__(curve.pcurve,'controls',((0.,0.),(1.25,0.)))
    with pytest.raises(owner.GeometryError):
        with operation():_cos_c2_world_chord_bound(curve,Q(0),Q(1),Q(0),lambda:None)


@pytest.mark.parametrize('counter',['boxes','subdivisions'])
def test_public_sampling_callback_cannot_reset_consumed_raw_parent(counter):
    from anygeometry.plane_nurbs import charge_split
    model,regions,seeds=rounded();before=owner.to_dict(model)
    parent=Work(POLICY.max_boxes_per_pair,POLICY.max_subdivision_depth)
    parent.boxes=10;parent.subdivisions=4
    previous=CURRENT_WORK.get();token=CURRENT_WORK.set(parent);calls=[]
    def reset(phase):
        calls.append(phase)
        charge_box();charge_split()  # actual additional charge in this query
        setattr(parent,counter,0)
    try:
        with pytest.raises(owner.GeometryError,match='parent work counters decreased'):
            query(model,regions,seeds,chord_error=.1,error_measure='cell_reference',cancellation_check=reset)
        assert calls and CURRENT_WORK.get() is parent
        assert parent.boxes>=11 and parent.subdivisions>=5
        assert owner.to_dict(model)==before
    finally:CURRENT_WORK.reset(token)
    assert CURRENT_WORK.get() is previous


@pytest.mark.parametrize('replacement',['parent','local','receiver'])
def test_public_sampling_callback_cannot_replace_work_identity(replacement):
    model,regions,seeds=rounded();before=owner.to_dict(model)
    parent=Work(POLICY.max_boxes_per_pair,POLICY.max_subdivision_depth)
    token=CURRENT_WORK.set(parent)
    def replace_work(phase):
        current=CURRENT_WORK.get()
        other=Work(parent.max_boxes,parent.max_depth)
        if replacement=='receiver':parent.take=other.take
        else:setattr(current,replacement,other)
    try:
        with pytest.raises(owner.GeometryError,match='identity changed|receiver changed'):
            query(model,regions,seeds,chord_error=.1,cancellation_check=replace_work)
        assert CURRENT_WORK.get() is parent and owner.to_dict(model)==before
    finally:CURRENT_WORK.reset(token)


def test_parent_watermark_allows_nested_charges_and_preserves_original_exception():
    from anygeometry.plane_nurbs import charge_split
    initial=CURRENT_WORK.get();error=RuntimeError('original cancellation')
    with pytest.raises(RuntimeError) as caught:
        with operation():
            parent=CURRENT_WORK.get();charge_box();charge_split()
            with operation():charge_box();charge_split()
            parent.guard()
            assert parent.boxes==2 and parent.subdivisions==2
            raise error
    assert caught.value is error and CURRENT_WORK.get() is initial


def test_public_callback_reset_then_exception_retains_consumption_and_exception():
    model,regions,seeds=rounded();before=owner.to_dict(model)
    parent=Work(POLICY.max_boxes_per_pair,POLICY.max_subdivision_depth);parent.boxes=10
    token=CURRENT_WORK.set(parent);error=RuntimeError('original reset cancellation')
    def reset_then_raise(phase):
        charge_box();parent.boxes=0;raise error
    try:
        with pytest.raises(RuntimeError) as caught:
            query(model,regions,seeds,chord_error=.1,cancellation_check=reset_then_raise)
        assert caught.value is error and parent.boxes>=11
        assert CURRENT_WORK.get() is parent and owner.to_dict(model)==before
    finally:CURRENT_WORK.reset(token)


@pytest.mark.parametrize('reverse', (False, True))
def test_cell_reference_reuses_only_terminal_arithmetic_and_bounded_reference_sets(monkeypatch,reverse):
    import anygeometry.prepared_material_curvilinear as module
    model,regions,seeds=rounded()
    if reverse:
        seeds=tuple((handle,parameters[::-1]) for handle,parameters in seeds)
    original=module._sampling_references
    calls=[]
    def references(*args):
        calls.append(1)
        return original(*args)
    def redundant(*args,**kwargs):
        raise AssertionError('terminal cell-reference arithmetic must not be reproved')
    monkeypatch.setattr(module,'_sampling_references',references)
    monkeypatch.setattr(module,'_world_chord',redundant)
    monkeypatch.setattr(module,'_parameterized_world_chord',redundant)
    result=query(model,regions,seeds,chord_error=.05,error_measure='cell_reference',proposal_fraction=.25)
    # The guarded arithmetic producer owns its canonical reference set; final
    # metadata uses a separate operation-local set. Neither rebuilds per interval.
    assert len(calls)==2
    assert all(chord.world_error<=.05*.25 for chord in result.chords.intervals)
    assert all((parameters[0]>parameters[-1])==reverse for _,parameters in result.edge_parameters)
    validate(model,regions,result)


def test_affine_fraction_eligibility_uses_actual_world_derivatives():
    import anygeometry.prepared_material_curvilinear as module
    from test_prepared_material_curvilinear import s_bend_boundary
    model,regions,seeds=square_with_weighted_line()
    references=module._sampling_references(model,regions)
    assert module._cell_reference_is_world_affine(model,references,seeds[0][0].id,Q(0),Q(1),lambda:None)
    # Its chart pcurve is a line, but the source surface gives a curved 3D image.
    model,regions,edge=s_bend_boundary()
    references=module._sampling_references(model,regions)
    assert not module._cell_reference_is_world_affine(model,references,edge.id,Q(0),Q(1),lambda:None)


def test_terminal_reuse_still_refuses_final_callback_mutation(monkeypatch):
    model,regions,seeds=rounded()
    def mutate(phase):
        if phase=='curvilinear boundary qualification complete':
            object.__setattr__(model.edges[seeds[0][0].id].curve.pcurve,'degree',1)
    with pytest.raises(owner.GeometryError):
        query(model,regions,seeds,chord_error=.05,error_measure='cell_reference',cancellation_check=mutate)


def test_sampling_bypasses_fraction_only_for_certified_world_affine_side(monkeypatch):
    import anygeometry.prepared_material_curvilinear as module
    model,regions,seeds=square_with_weighted_line()
    original=module._cell_reference_world_chord
    limits=[]
    def bound(model,references,edge,a,b,endpoints,error,check):
        limits.append(error)
        return original(model,references,edge,a,b,endpoints,error,check)
    monkeypatch.setattr(module,'_cell_reference_world_chord',bound)
    result=query(model,regions,seeds,chord_error=.02,error_measure='cell_reference',proposal_fraction=.25)
    assert limits and all(limit==.02 for limit in limits)
    validate(model,regions,result)


@pytest.mark.parametrize('target', ('source_ranges','source_validate','memo_init','memo_getattr',
    'point_jet','curve_jet','metric_code','metric_mul'))
@pytest.mark.parametrize('phase', ('curvilinear boundary station refinement',
    'curvilinear boundary qualification complete'))
def test_reused_terminal_context_refuses_private_calculator_mutation(monkeypatch,target,phase):
    import anygeometry.prepared_material_curvilinear as module
    from anygeometry import regular_source_reference as source,curvilinear_reference_maps as maps
    from anygeometry import _local_curve_geometry as jets,nurbs_metric as metric
    model,regions,seeds=rounded()
    original=metric.derivative_intervals.__code__
    changed=[]
    def replacement(*args,**kwargs):
        return None
    targets={'source_ranges':(source.RationalSourceSide,'_ranges'),
        'source_validate':(source.RationalSourceSide,'_validate'),
        'memo_init':(maps._SideMemo,'__init__'),'memo_getattr':(maps._SideMemo,'__getattr__'),
        'point_jet':(source,'_pcurve_point_jet'),'curve_jet':(jets,'_curve_jet'),
        'metric_mul':(metric,'mul')}
    def mutate(current):
        if current==phase and not changed:
            changed.append(True)
            if target=='metric_code':
                metric.derivative_intervals.__code__=replacement.__code__
            else:
                owner,name=targets[target]
                monkeypatch.setattr(owner,name,replacement)
    try:
        with pytest.raises(owner.GeometryError,match='context changed'):
            query(model,regions,seeds,chord_error=.1,error_measure='cell_reference',
                proposal_fraction=.25,cancellation_check=mutate)
        assert changed
    finally:
        metric.derivative_intervals.__code__=original


def test_context_guard_preserves_original_callback_exception_after_private_mutation(monkeypatch):
    from anygeometry import regular_source_reference as source
    model,regions,seeds=rounded()
    error=RuntimeError('original callback exception')
    def mutate_and_raise(phase):
        monkeypatch.setattr(source.RationalSourceSide,'_ranges',lambda *args:None)
        raise error
    with pytest.raises(RuntimeError) as caught:
        query(model,regions,seeds,chord_error=.1,error_measure='cell_reference',cancellation_check=mutate_and_raise)
    assert caught.value is error


@pytest.mark.parametrize('target', ('source_enclosure','_context','_canonical_literals','_keywords',
    '_FUNCTIONS','_BINDINGS','_METHODS','caps'))
@pytest.mark.parametrize('phase', ('curvilinear boundary station refinement',
    'curvilinear boundary qualification complete'))
def test_reused_terminal_context_refuses_source_enclosure_dispatch_mutation(monkeypatch,target,phase):
    from anygeometry import _source_box_hull as enclosure
    model,regions,seeds=rounded()
    changed=[]
    def replacement(*args,**kwargs):
        return None
    def mutate(current):
        if current==phase and not changed:
            changed.append(True)
            monkeypatch.setattr(enclosure,target,() if target.startswith('_') and target.isupper() else replacement)
    with pytest.raises(owner.GeometryError,match='context changed'):
        query(model,regions,seeds,chord_error=.1,error_measure='cell_reference',
            proposal_fraction=.25,cancellation_check=mutate)
    assert changed



def _record_real_cell_bounds(monkeypatch):
    import anygeometry.prepared_material_curvilinear as module
    actual=module._cell_reference_world_chord
    calls=[]
    def record(model,references,edge,a,b,endpoints,error,check):
        calls.append((edge,a,b,error))
        return actual(model,references,edge,a,b,endpoints,error,check)
    monkeypatch.setattr(module,'_cell_reference_world_chord',record)
    return calls


def test_scope_reuses_real_terminal_arithmetic_with_fresh_receipts(monkeypatch):
    calls=_record_real_cell_bounds(monkeypatch)
    model,regions,seeds=rounded()
    with operation():
        first=query(model,regions,seeds,chord_error=.1,error_measure='cell_reference',proposal_fraction=.25)
        used=len(calls);spent=CURRENT_WORK.get().boxes
        second=query(model,regions,first.edge_parameters,chord_error=.1,error_measure='cell_reference',proposal_fraction=.25)
        assert used>0 and len(calls)==used
        assert CURRENT_WORK.get().boxes>=spent
        assert second is not first and second.chords is not first.chords
        assert second.edge_parameters==first.edge_parameters
        validate(model,regions,first);validate(model,regions,second)


def test_scope_balance_split_proves_only_new_intervals(monkeypatch):
    calls=_record_real_cell_bounds(monkeypatch)
    model,regions,seeds=rounded()
    with operation():
        first=query(model,regions,seeds,chord_error=.1,error_measure='cell_reference',proposal_fraction=.25)
        used=len(calls)
        handle,parameters=first.edge_parameters[0];a,b=parameters[:2];mid=(a+b)/2
        balanced=((handle,tuple(sorted((*parameters,mid)))),*first.edge_parameters[1:])
        second=query(model,regions,balanced,chord_error=.1,error_measure='cell_reference',proposal_fraction=.25)
        assert {(edge,x,y) for edge,x,y,_ in calls[used:]}=={(handle.id,a,mid),(handle.id,mid,b)}
        assert len(calls)==used+2
        validate(model,regions,second)


def test_scope_tighter_limit_recomputes_and_preserves_consumption(monkeypatch):
    calls=_record_real_cell_bounds(monkeypatch)
    model,regions,seeds=rounded()
    with operation():
        first=query(model,regions,seeds,chord_error=.1,error_measure='cell_reference',proposal_fraction=.25)
        used=len(calls);spent=CURRENT_WORK.get().boxes
        second=query(model,regions,first.edge_parameters,chord_error=.005,error_measure='cell_reference',proposal_fraction=.25)
        assert len(calls)>used and CURRENT_WORK.get().boxes>spent
        assert all(row.world_error<=.005*.25 for row in second.chords.intervals)
        validate(model,regions,second)


def test_scope_bank_isolates_nested_and_exited_lifetimes(monkeypatch):
    calls=_record_real_cell_bounds(monkeypatch)
    model,regions,seeds=rounded()
    with operation():
        first=query(model,regions,seeds,chord_error=.1,error_measure='cell_reference')
        used=len(calls)
        with operation():
            query(model,regions,first.edge_parameters,chord_error=.1,error_measure='cell_reference')
            assert len(calls)>used
        nested=len(calls)
        query(model,regions,first.edge_parameters,chord_error=.1,error_measure='cell_reference')
        assert len(calls)==nested
    with operation():
        query(model,regions,first.edge_parameters,chord_error=.1,error_measure='cell_reference')
        assert len(calls)>nested


@pytest.mark.parametrize('mutation', ('calculator','source','bank_identity','bank_value'))
def test_scope_hit_refuses_changed_prior_source_or_calculator(monkeypatch,mutation):
    from anygeometry import regular_source_reference as source
    import anygeometry.prepared_material_curvilinear as module
    model,regions,seeds=rounded()
    with pytest.raises(owner.GeometryError):
        with operation():
            first=query(model,regions,seeds,chord_error=.1,error_measure='cell_reference')
            if mutation=='calculator':
                monkeypatch.setattr(source.RationalSourceSide,'_ranges',lambda *args:None)
            elif mutation=='source':
                curve=model.edges[seeds[0][0].id].curve.pcurve
                object.__setattr__(curve,'degree',1)
            elif mutation=='bank_identity':
                from contextvars import ContextVar
                monkeypatch.setattr(module,'_SAMPLING_ARITHMETIC',ContextVar('poison',default=None))
            else:
                module._SAMPLING_ARITHMETIC.set(None)
            query(model,regions,first.edge_parameters,chord_error=.1,error_measure='cell_reference')


def test_scope_hit_keeps_actual_callback_cancellation_and_exception():
    model,regions,seeds=rounded();error=RuntimeError('cached query callback failure')
    def failure(phase):raise error
    with operation():
        first=query(model,regions,seeds,chord_error=.1,error_measure='cell_reference')
        spent=CURRENT_WORK.get().boxes
        with pytest.raises(owner.GeometryError,match='cancelled'):
            query(model,regions,first.edge_parameters,chord_error=.1,error_measure='cell_reference',
                cancellation_check=lambda phase:phase=='curvilinear boundary station refinement')
        with pytest.raises(RuntimeError) as caught:
            query(model,regions,first.edge_parameters,chord_error=.1,error_measure='cell_reference',cancellation_check=failure)
        assert caught.value is error and CURRENT_WORK.get().boxes>=spent
        validate(model,regions,first)


def test_scope_hit_cannot_erase_exhausted_parent_work():
    model,regions,seeds=rounded()
    with operation():
        first=query(model,regions,seeds,chord_error=.1,error_measure='cell_reference')
        parent=CURRENT_WORK.get()
        # Cached bounds cannot bypass a fresh tiny per-query allowance.
        spent=parent.boxes
        with pytest.raises(owner.GeometryError):
            query(model,regions,first.edge_parameters,chord_error=.1,error_measure='cell_reference',
                qualification=replace(POLICY,max_boxes_per_pair=1))
        assert parent.boxes>=spent



def test_scope_shared_edge_preserves_selected_incident_occurrences(monkeypatch):
    from anygeometry.nurbs import CurveOnSurface,NurbsCurve
    from test_local_curve_geometry import flat
    model=owner.GeometryModel();support=flat()
    uv=((0.,0.),(.5,0.),(1.,0.),(0.,1.),(.5,1.),(1.,1.))
    vertices=model.add_points(tuple((*p,0.) for p in uv))
    def edge(a,b):
        return model.add_curve(vertices[a],vertices[b],CurveOnSurface(support,
            NurbsCurve(1,(0.,0.,1.,1.),(uv[a],uv[b]))))
    shared=edge(1,4)
    left=model.add_face((edge(0,1),shared,edge(4,3),edge(3,0)),surface=support)
    right=model.add_face((edge(1,2),edge(2,5),edge(5,4),shared),surface=support)
    both=owner.query_material_surface_regions(model,(left,right))
    one=owner.query_material_surface_regions(model,(left,))
    seeds=((model.handle('edge',shared),(Q(0),Q(1))),)
    calls=_record_real_cell_bounds(monkeypatch)
    with operation():
        first=query(model,both,seeds,chord_error=.1,error_measure='cell_reference')
        used=len(calls)
        second=query(model,both,seeds,chord_error=.1,error_measure='cell_reference')
        assert len(calls)==used
        assert len(first.chords.intervals[0].occurrences)==len(second.chords.intervals[0].occurrences)==2
        single=query(model,one,seeds,chord_error=.1,error_measure='cell_reference')
        assert len(calls)>used and len(single.chords.intervals[0].occurrences)==1
        validate(model,both,second);validate(model,one,single)



def test_callback_cannot_seed_curved_terminal_key_with_detached_straight_operands(monkeypatch):
    import anygeometry.prepared_material_curvilinear as module
    from anygeometry.arrangement_geometry import LinePath
    from anygeometry.native_support_snapshots import capture_native_supports
    from anygeometry.definition_binding import definition_checksum
    calls=_record_real_cell_bounds(monkeypatch)
    model,regions,seeds=rounded();handle=seeds[0][0]
    curve=model.edges[handle.id].curve
    endpoints=tuple(tuple(float(x) for x in curve.exact_point(t)) for t in (Q(0),Q(1)))
    straight=LinePath(*endpoints);attempted=[]
    def poison(phase):
        if phase=='curvilinear boundary station refinement' and not attempted:
            attempted.append(True)
            state=module._SAMPLING_ARITHMETIC.get()
            epoch=model.revision,module._checksum(model),capture_native_supports(model)
            key=definition_checksum(regions),'cell_reference',handle.id,Q(0),Q(1),endpoints
            # This was a real namespace producer write: the target identity was
            # curved while its detached builder operands proved a straight line.
            with pytest.raises(TypeError):
                module._sampling_terminal_arithmetic(state,model,epoch,key,.01,
                    model,{},handle.id,Q(0),Q(1),endpoints,.01,lambda:None,straight,'geometric')
    with operation():
        first=query(model,regions,seeds,chord_error=.01,error_measure='cell_reference',cancellation_check=poison)
        assert attempted and len(first.edge_parameters[0][1])>2
        assert calls and any(row.world_error>0 for row in first.chords.intervals)
        used=len(calls)
        second=query(model,regions,first.edge_parameters,chord_error=.01,error_measure='cell_reference')
        assert len(calls)==used
        validate(model,regions,first);validate(model,regions,second)


@pytest.mark.parametrize('raise_original', (False,True))
def test_scope_close_code_mutation_restores_all_contexts_and_original_error(raise_original):
    import anygeometry.prepared_material_curvilinear as module
    initial_work=CURRENT_WORK.get()
    initial_bank=module._SAMPLING_ARITHMETIC.get()
    initial_issuer=module._SAMPLING_ARITHMETIC_ISSUER.get()
    model,regions,seeds=rounded();original_error=RuntimeError('original caller failure')
    changed=[];saved=[]
    # close has one closure cell (trusted cleanup); match that shape while
    # ensuring invocation would mask the original caller exception.
    def malicious_close_factory():
        retained=None
        def malicious_close():
            if retained is None:
                raise AssertionError('untrusted close executed')
        return malicious_close
    def attack(phase):
        if not changed:
            state=module._SAMPLING_ARITHMETIC.get();saved.append((state[2],state[2].__code__))
            state[2].__code__=malicious_close_factory().__code__;changed.append(True)
            if raise_original:
                raise original_error
    try:
        expected=RuntimeError if raise_original else owner.GeometryError
        with pytest.raises(expected) as caught:
            with operation():
                query(model,regions,seeds,chord_error=.1,error_measure='cell_reference',cancellation_check=attack)
        if raise_original:
            assert caught.value is original_error
        assert changed and CURRENT_WORK.get() is initial_work
        assert module._SAMPLING_ARITHMETIC.get() is initial_bank
        assert module._SAMPLING_ARITHMETIC_ISSUER.get() is initial_issuer
    finally:
        for function,code in saved:
            function.__code__=code



def test_interval_arithmetic_hits_do_not_repeat_full_native_snapshot_walk(monkeypatch):
    import anygeometry.prepared_material_curvilinear as module
    from anygeometry import native_support_snapshots as native
    actual=native.capture_native_supports;calls=[]
    def snapshot(*args,**kwargs):
        calls.append(1)
        return actual(*args,**kwargs)
    monkeypatch.setattr(native,'capture_native_supports',snapshot)
    model,regions,seeds=rounded();handle=seeds[0][0];curve=model.edges[handle.id].curve
    a,b=Q(0),Q(1,4)
    endpoints=tuple(tuple(float(x) for x in curve.exact_point(t)) for t in (a,b))
    with operation():
        arithmetic=module._sampling_terminal_arithmetic(model,regions,lambda:None)
        bound,pieces=arithmetic(handle.id,a,b,endpoints,.1,'cell_reference')
        before=len(calls)
        assert before>0
        for _ in range(8):
            assert arithmetic(handle.id,a,b,endpoints,.1,'cell_reference')==(bound,pieces)
        assert len(calls)==before
    # Arithmetic is never a receipt and cannot satisfy public admission.
    with pytest.raises(owner.GeometryError,match='not issued'):
        validate(model,regions,(bound,pieces))


@pytest.mark.parametrize('phase', ('entry','curvilinear boundary station refinement',
    'curvilinear boundary qualification complete','curvilinear boundary sampling complete'))
@pytest.mark.parametrize('runtime', ('kernel','point_cache'))
def test_cached_query_runtime_mutation_refuses_active_final_and_publication(phase,runtime):
    import numpy as np
    import anygeometry.prepared_material_curvilinear as module
    model,regions,seeds=rounded();curve=model.edges[seeds[0][0].id].curve.pcurve
    changed=[]
    with operation():
        first=query(model,regions,seeds,chord_error=.1,error_measure='cell_reference')
        prepared=curve._prepared;points=curve._point_values
        other=replace(curve,weights=(1.,2.,1.))
        issued=set(module._SAMPLINGS)
        def attack(current):
            if current==phase and not changed:
                changed.append(True)
                if runtime=='kernel':
                    object.__setattr__(curve,'_prepared',other._prepared)
                else:
                    object.__setattr__(curve,'_point_values',
                        {np.float64(.5).tobytes():np.ones((1,1,2))})
        try:
            if phase=='entry':
                attack('entry')
            with pytest.raises(owner.GeometryError,match='evaluated kernel|point cache'):
                query(model,regions,first.edge_parameters,chord_error=.1,error_measure='cell_reference',
                    cancellation_check=None if phase=='entry' else attack)
            assert changed and set(module._SAMPLINGS).issubset(issued)
            # Successful cached arithmetic cannot keep an issued receipt valid
            # after its live evaluated source has been poisoned.
            with pytest.raises(owner.GeometryError,match='evaluated kernel|point cache'):
                validate(model,regions,first)
        finally:
            object.__setattr__(curve,'_prepared',prepared)
            object.__setattr__(curve,'_point_values',points)
        validate(model,regions,first)



def test_cached_query_passive_signal_preserves_cancellation_without_external_code():
    from anygeometry.cancellation import CancellationSignal
    model,regions,seeds=rounded();signal=CancellationSignal()
    with operation():
        first=query(model,regions,seeds,chord_error=.1,error_measure='cell_reference')
        second=query(model,regions,first.edge_parameters,chord_error=.1,error_measure='cell_reference',
            cancellation_check=signal)
        assert first.edge_parameters==second.edge_parameters
        validate(model,regions,second)
        spent=CURRENT_WORK.get().boxes
        signal.cancel()
        with pytest.raises(owner.GeometryError,match='cancelled'):
            query(model,regions,first.edge_parameters,chord_error=.1,error_measure='cell_reference',
                cancellation_check=signal)
        assert CURRENT_WORK.get().boxes>=spent
