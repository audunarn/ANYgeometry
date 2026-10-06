"""Small wrapper-local ancestry fixtures; no mesher or mixed-model execution."""
from dataclasses import replace, FrozenInstanceError
from fractions import Fraction as F
import math

import pytest

from anygeometry import GeometryModel, GeometryError, plan_intersections, to_dict, from_dict
from anygeometry.definition_binding import definition_checksum
from anygeometry.edge_subcurve_preimages import (
    _capture_edge_subcurve_preimages as capture,
    _record_edge_subcurve_split as record,
    _finalize_edge_subcurve_preimages as finalize,
    _publish_edge_subcurve_preimages as publish,
    _copy_current_edge_subcurve_preimages as copy_receipt,
    _edge_subcurve_definition as definition,
    _restrict_controls,
    _rebind_edge_subcurve_incidence as rebind,
    _SubcurveEnclosureUnavailable,
    query_prepared_edge_subcurve_preimages as query,
    validate_prepared_edge_subcurve_preimages_binding as validate,
)


def fixture(reverse=False):
    model = GeometryModel()
    face = model.add_plate(model.add_points(((0, 0, -2), (4, 0, -2), (4, 4, -2), (0, 4, -2))))
    points = ((0., 0., 0.), (1., 2., 0.), (2., -1., 0.), (3., 1., 0.))
    vertices = model.add_points(points[::-1] if reverse else points)
    edge = model.add_spline(vertices[0], vertices[1:-1], vertices[-1])
    # A real, complete one-face plan supplies the preparation receipt's content
    # binding; these tests simulate only the authorized wrapper's publication.
    plan = plan_intersections(model, (face,), policy='connect')
    return model, edge, plan


def split(candidate, draft, edge, parameter=.1, tolerance=1e-9):
    before = definition(candidate, edge)
    _, children = candidate.split_edge(edge, parameter)
    record(draft, edge, parameter, children, tolerance, model=candidate, parent_definition=before)
    return children


def commit(source, candidate, draft, plan):
    prepared = finalize(candidate, draft)
    source.restore_topology(candidate.topology_snapshot())
    checksum = to_dict(source)['checksum']['value']
    source._intersection_preparation_receipt = (plan, source.revision, checksum, tuple(sorted(source.faces)))
    publish(source, prepared, checksum)
    return query(source)


def independent_restriction(controls, a, b):
    """Power expansion/composition/back conversion, independent of blossom."""
    n = len(controls)-1
    output = [[F(0)]*3 for _ in range(n+1)]
    for axis in range(3):
        power = [sum(F(controls[i][axis])*(-1)**(k-i)*math.comb(n,i)*math.comb(n-i,k-i)
                     for i in range(k+1)) for k in range(n+1)]
        composed = [sum(power[j]*math.comb(j,k)*a**(j-k)*(b-a)**k for j in range(k,n+1))
                    for k in range(n+1)]
        for i in range(n+1):
            output[i][axis] = sum(composed[k]*F(math.comb(i,k),math.comb(n,k)) for k in range(i+1))
    return tuple(map(tuple,output))


@pytest.mark.parametrize('reverse', [False, True])
def test_repeated_actual_float_splits_keep_ancestor_and_exact_enclosure(reverse):
    model, edge, plan = fixture(reverse)
    original = to_dict(model); draft = capture(model, allow_seed=True)
    candidate = model.clone(preserve_identity=True)
    with candidate.transaction():
        first, second = split(candidate, draft, edge, .1)
        middle, last = split(candidate, draft, second, .3)
    assert to_dict(model) == original
    binding = commit(model, candidate, draft, plan)
    rows = {r.edge_id:r for r in binding.records}
    a, mid = F(.1), F(.1)+(1-F(.1))*F(.3)
    expected_intervals = {first:(F(0),a),middle:(a,mid),last:(mid,F(1))}
    for child, interval in expected_intervals.items():
        row = rows[child]
        assert tuple(F(*x) for x in row.interval) == interval
        assert row.ancestor.definition.edge_id == edge
        assert row.ancestor.source_checksum == original['checksum']['value']
        controls = tuple(tuple(F(*x) for x in p) for p in row.ancestor.definition.controls)
        exact = independent_restriction(controls,*interval)
        actual = tuple(tuple(F(*x) for x in p) for p in row.current_definition.controls)
        error = tuple(tuple(x-y for x,y in zip(p,q)) for p,q in zip(actual,exact))
        assert error == tuple(tuple(F(*x) for x in p) for p in row.error_controls)
        squared = max(sum(x*x for x in p) for p in error)
        assert F(*row.squared_distance_bound) == squared <= F(1e-9)**2
    validate(model,binding)
    definition_checksum(binding)  # every exact scalar is JSON-compatible
    with pytest.raises(FrozenInstanceError):
        binding.revision = 0


@pytest.mark.parametrize('interval', [(F(4,5),F(1,7)),(F(0),F(1)),(F(1),F(0))])
def test_reversed_exact_interval_math(interval):
    controls=((F(0),F(0),F(0)),(F(1),F(2),F(0)),(F(2),-F(1),F(0)),(F(3),F(1),F(0)))
    assert _restrict_controls(controls,*interval) == independent_restriction(controls,*interval)


def test_changed_child_and_changed_parent_never_rebind():
    model, edge, _ = fixture(); draft = capture(model,allow_seed=True)
    candidate=model.clone(preserve_identity=True)
    first,_=split(candidate,draft,edge)
    child_control=candidate.edges[first].curve.control_vertices[0]
    candidate.move_point(child_control,9,9,9)
    with pytest.raises(GeometryError,match='tracked child changed'):
        finalize(candidate,draft)
    candidate=model.clone(preserve_identity=True); draft=capture(model,allow_seed=True)
    control=candidate.edges[edge].curve.control_vertices[0]
    candidate.move_point(control,1,2.0000000001,0)
    before=definition(candidate,edge)
    _,children=candidate.split_edge(edge,.1)
    with pytest.raises(GeometryError,match='parent definition changed'):
        record(draft,edge,.1,children,1e-9,model=candidate,parent_definition=before)


def test_unknown_new_edges_are_explicitly_unavailable_and_requested_scope_refuses():
    model,edge,plan=fixture(); draft=capture(model,allow_seed=True)
    candidate=model.clone(preserve_identity=True)
    with candidate.transaction():
        split(candidate,draft,edge)
        points=candidate.add_points(((8,0,0),(9,0,0)))
        unknown=candidate.add_line(*points)
    binding=commit(model,candidate,draft,plan)
    assert unknown in binding.unavailable_edge_ids
    with pytest.raises(GeometryError,match='unavailable'):
        query(model,edge_ids=(unknown,))
    with pytest.raises(GeometryError,match='not active'):
        query(model,edge_ids=(99999,))
    assert unknown not in capture(model,allow_seed=True).records


def test_stale_tampered_wrong_owner_load_and_unqualified_clone_refuse():
    model,edge,plan=fixture(); draft=capture(model,allow_seed=True)
    candidate=model.clone(preserve_identity=True); split(candidate,draft,edge)
    binding=commit(model,candidate,draft,plan)
    with pytest.raises(GeometryError,match='definition binding changed'):
        validate(model,replace(binding,records=binding.records[:-1]))
    other,_,_=fixture()
    with pytest.raises(GeometryError,match='another model'):
        validate(other,binding)
    for other in (from_dict(to_dict(model)),model.clone(preserve_identity=True)):
        with pytest.raises(GeometryError): query(other)
        assert capture(other,allow_seed=True) is None
    model.add_point(12,12,12)
    with pytest.raises(GeometryError): query(model)
    assert capture(model,allow_seed=True) is None


def test_seed_authority_is_current_source_not_recovered_loaded_history():
    model,edge,_=fixture()
    loaded=from_dict(to_dict(model))
    assert capture(loaded) is None
    seeded=capture(loaded,allow_seed=True)
    assert seeded.records[edge].ancestor.source_checksum==to_dict(loaded)['checksum']['value']
    loaded.split_edge(edge,.1)
    assert capture(loaded,allow_seed=True) is None


def test_cancellation_exact_identity_and_draft_record_atomicity():
    model,edge,_=fixture(); original=to_dict(model)
    error=RuntimeError('stop owner ancestry')
    def cancel(_): raise error
    with pytest.raises(RuntimeError) as caught: capture(model,allow_seed=True,cancellation_check=cancel)
    assert caught.value is error and to_dict(model)==original
    draft=capture(model,allow_seed=True); candidate=model.clone(preserve_identity=True)
    before=definition(candidate,edge); _,children=candidate.split_edge(edge,.1)
    original_records=dict(draft.records); calls=0
    def second(_):
        nonlocal calls
        calls+=1
        if calls==2: raise error
    draft.check=second
    with pytest.raises(RuntimeError) as caught:
        record(draft,edge,.1,children,1e-9,model=candidate,parent_definition=before)
    assert caught.value is error and draft.records==original_records
    assert to_dict(model)==original


def test_query_callback_mutation_and_partial_preparation_refuse():
    model,edge,plan=fixture(); draft=capture(model,allow_seed=True)
    candidate=model.clone(preserve_identity=True); split(candidate,draft,edge)
    commit(model,candidate,draft,plan)
    calls=0
    def mutate(_):
        nonlocal calls
        calls+=1
        if calls==2:model.add_point(20,20,20)
    with pytest.raises(GeometryError):query(model,cancellation_check=mutate)


def test_capture_and_publish_refuse_outer_transactions():
    model,_,_=fixture(); draft=capture(model,allow_seed=True)
    prepared=finalize(model.clone(preserve_identity=True),draft)
    with model.transaction():
        with pytest.raises(GeometryError,match='committed'): capture(model,allow_seed=True)
        with pytest.raises(GeometryError,match='committed'):
            publish(model,prepared,to_dict(model)['checksum']['value'])


def test_qualified_clone_requires_identical_current_document():
    model,edge,plan=fixture(); draft=capture(model,allow_seed=True)
    candidate=model.clone(preserve_identity=True); split(candidate,draft,edge)
    binding=commit(model,candidate,draft,plan)
    clone=model.clone(preserve_identity=True)
    clone._intersection_preparation_receipt=model._intersection_preparation_receipt
    copy_receipt(model,clone)
    assert query(clone)==binding
    clone.add_point(20,20,20)
    with pytest.raises(GeometryError,match='qualified cloning'):copy_receipt(model,clone)


def test_arc_seed_exposes_circle_ancestry_without_parameter_qualification():
    model,edge,_=fixture()
    vertices=model.add_points(((8,0,0),(9,1,0),(10,0,0)))
    arc=model.add_arc(*vertices)
    plan=plan_intersections(model,tuple(model.faces),policy='connect')
    draft=capture(model,allow_seed=True); candidate=model.clone(preserve_identity=True)
    split(candidate,draft,edge)
    binding=commit(model,candidate,draft,plan)
    assert arc not in binding.unavailable_edge_ids
    selected = query(model, edge_ids=(arc,))
    assert len(selected.arc_records) == 1
    assert selected.arc_records[0].classification == 'exact'
    assert selected.arc_records[0].parameter_mapping_qualified is False


def test_rounding_tolerance_refusal_leaves_draft_unchanged():
    model,edge,_=fixture(); draft=capture(model,allow_seed=True)
    candidate=model.clone(preserve_identity=True); before=definition(candidate,edge)
    _,children=candidate.split_edge(edge,.1); snapshot=dict(draft.records)
    with pytest.raises(_SubcurveEnclosureUnavailable,match='exceeds existing tolerance'):
        record(draft,edge,.1,children,1e-30,model=candidate,parent_definition=before)
    assert draft.records==snapshot
    with pytest.raises(GeometryError,match='orientation/incidence'):
        record(draft,edge,.1,children[::-1],1e-9,model=candidate,parent_definition=before)
    assert draft.records==snapshot


def test_changed_live_source_blocks_detached_finalization():
    model,edge,_=fixture(); draft=capture(model,allow_seed=True)
    candidate=model.clone(preserve_identity=True); split(candidate,draft,edge)
    model.add_point(15,15,15)
    with pytest.raises(GeometryError,match='source changed'):finalize(candidate,draft)
    assert not hasattr(model,'_edge_subcurve_preimages_receipt')


def test_incomplete_preparation_and_transplanted_receipt_cannot_query():
    model,edge,plan=fixture(); draft=capture(model,allow_seed=True)
    candidate=model.clone(preserve_identity=True); split(candidate,draft,edge)
    commit(model,candidate,draft,plan)
    clone=model.clone(preserve_identity=True)
    clone._intersection_preparation_receipt=model._intersection_preparation_receipt
    clone._edge_subcurve_preimages_receipt=model._edge_subcurve_preimages_receipt
    with pytest.raises(GeometryError,match='another owner'):query(clone)
    receipt=model._intersection_preparation_receipt
    model._intersection_preparation_receipt=(*receipt[:3],())
    with pytest.raises(GeometryError,match='complete current preparation'):query(model)


@pytest.mark.parametrize('control', [False, True])
def test_explicit_equal_coordinate_incidence_merge_preserves_exact_receipt(control):
    from anygeometry.intersections import _merge_vertex
    model,edge,plan=fixture(); draft=capture(model,allow_seed=True)
    candidate=model.clone(preserve_identity=True)
    original=draft.records[edge]
    with candidate.transaction():
        old=(candidate.edges[edge].curve.control_vertices[0] if control else candidate.edges[edge].start)
        new=candidate.add_point(*candidate.vertex_position(old))
        prior={i:definition(candidate,i) for i in candidate.edges_using_vertex(old)}
        _merge_vertex(candidate,old,new)
        rebind(draft,prior,model=candidate)
    current=draft.records[edge]
    assert current.current_definition != original.current_definition
    assert replace(current,current_definition=original.current_definition)==original
    binding=commit(model,candidate,draft,plan)
    assert edge not in binding.unavailable_edge_ids


def test_near_coordinate_incidence_merge_becomes_unavailable():
    from anygeometry.intersections import _merge_vertex
    model,edge,plan=fixture(); draft=capture(model,allow_seed=True)
    candidate=model.clone(preserve_identity=True)
    with candidate.transaction():
        old=candidate.edges[edge].start
        position=list(candidate.vertex_position(old)); position[1]+=1e-12
        new=candidate.add_point(*position)
        prior={i:definition(candidate,i) for i in candidate.edges_using_vertex(old)}
        _merge_vertex(candidate,old,new)
        rebind(draft,prior,model=candidate)
    assert edge not in draft.records
    binding=commit(model,candidate,draft,plan)
    assert edge in binding.unavailable_edge_ids
    with pytest.raises(GeometryError,match='unavailable'):query(model,edge_ids=(edge,))


def test_incidence_rebind_cannot_launder_stale_prior_seal():
    from anygeometry.intersections import _merge_vertex
    model,edge,_=fixture(); draft=capture(model,allow_seed=True)
    candidate=model.clone(preserve_identity=True); records=dict(draft.records)
    with candidate.transaction():
        old=candidate.edges[edge].curve.control_vertices[0]
        candidate.move_point(old,1,2.000000000001,0)
        new=candidate.add_point(*candidate.vertex_position(old))
        prior={i:definition(candidate,i) for i in candidate.edges_using_vertex(old)}
        _merge_vertex(candidate,old,new)
        with pytest.raises(GeometryError,match='prior definition changed'):
            rebind(draft,prior,model=candidate)
    assert draft.records==records


def straight_fixture():
    model=GeometryModel()
    face=model.add_plate(model.add_points(((0,0,-2),(4,0,-2),(4,4,-2),(0,4,-2))))
    vertices=model.add_points(((0.,0.,0.),(4.,0.,0.)))
    edge=model.add_line(*vertices)
    plan=plan_intersections(model,(face,),policy='connect')
    return model,edge,plan


@pytest.mark.parametrize('displacement,retained', [(1e-12,True),(1e-9,True),(1e-6,False)])
def test_straight_incidence_merge_residual_against_unchanged_tolerance(displacement,retained):
    from anygeometry.intersections import _merge_vertex
    model,edge,plan=straight_fixture(); draft=capture(model,allow_seed=True)
    candidate=model.clone(preserve_identity=True)
    with candidate.transaction():
        first,_=split(candidate,draft,edge)
    record=draft.records[first]
    with candidate.transaction():
        old=candidate.edges[first].start
        position=list(candidate.vertex_position(old)); position[1]+=displacement
        new=candidate.add_point(*position)
        prior={i:definition(candidate,i) for i in candidate.edges_using_vertex(old)}
        _merge_vertex(candidate,old,new)
        rebind(draft,prior,model=candidate)
    binding=commit(model,candidate,draft,plan)
    if not retained:
        assert first not in draft.records
        assert first in binding.unavailable_edge_ids
        with pytest.raises(GeometryError,match='unavailable'):
            query(model,edge_ids=(first,))
        return
    sealed=draft.records[first]
    assert sealed.ancestor==record.ancestor and sealed.interval==record.interval
    assert sealed.tolerance==record.tolerance  # never renewed or increased
    assert F(*sealed.squared_distance_bound)==F(displacement)**2<=F(*record.tolerance)**2
    assert sealed.current_definition==definition(candidate,first)
    assert first not in binding.unavailable_edge_ids
    validate(model,binding)


def test_straight_incidence_without_recorded_tolerance_still_requires_exact_controls():
    from anygeometry.intersections import _merge_vertex
    model,edge,plan=straight_fixture(); draft=capture(model,allow_seed=True)
    candidate=model.clone(preserve_identity=True)
    with candidate.transaction():
        old=candidate.edges[edge].start
        position=list(candidate.vertex_position(old)); position[1]+=1e-12
        new=candidate.add_point(*position)
        prior={i:definition(candidate,i) for i in candidate.edges_using_vertex(old)}
        _merge_vertex(candidate,old,new)
        rebind(draft,prior,model=candidate)
    assert edge not in draft.records
    binding=commit(model,candidate,draft,plan)
    assert edge in binding.unavailable_edge_ids
    with pytest.raises(GeometryError,match='unavailable'):
        query(model,edge_ids=(edge,))


def test_spline_incidence_change_never_reseals_even_within_tolerance():
    from anygeometry.intersections import _merge_vertex
    model,edge,plan=fixture(); draft=capture(model,allow_seed=True)
    candidate=model.clone(preserve_identity=True)
    with candidate.transaction():
        first,_=split(candidate,draft,edge)
    with candidate.transaction():
        old=candidate.edges[first].curve.control_vertices[0]
        position=list(candidate.vertex_position(old)); position[1]+=1e-12
        new=candidate.add_point(*position)
        prior={i:definition(candidate,i) for i in candidate.edges_using_vertex(old)}
        _merge_vertex(candidate,old,new)
        rebind(draft,prior,model=candidate)
    assert first not in draft.records
    binding=commit(model,candidate,draft,plan)
    assert first in binding.unavailable_edge_ids


def test_incidence_rebind_never_synthesizes_missing_lineage():
    from anygeometry.intersections import _merge_vertex
    model,edge,plan=straight_fixture(); draft=capture(model,allow_seed=True)
    candidate=model.clone(preserve_identity=True)
    with candidate.transaction():
        first,second=split(candidate,draft,edge)
    draft.records.pop(second)
    with candidate.transaction():
        old=candidate.edges[first].start
        new=candidate.add_point(*candidate.vertex_position(old))
        prior={i:definition(candidate,i) for i in candidate.edges_using_vertex(old)}
        prior[second]=definition(candidate,second)
        _merge_vertex(candidate,old,new)
        rebind(draft,prior,model=candidate)
    assert second not in draft.records
    binding=commit(model,candidate,draft,plan)
    assert second in binding.unavailable_edge_ids


def test_optional_enclosure_refusal_can_only_remove_not_publish_children():
    model,edge,plan=fixture(); draft=capture(model,allow_seed=True)
    candidate=model.clone(preserve_identity=True)
    with candidate.transaction():
        prior=definition(candidate,edge); _,children=candidate.split_edge(edge,.1)
        try:
            record(draft,edge,.1,children,1e-30,model=candidate,parent_definition=prior)
        except _SubcurveEnclosureUnavailable:
            for identifier in (edge,*children):draft.records.pop(identifier,None)
    binding=commit(model,candidate,draft,plan)
    assert set(children)<=set(binding.unavailable_edge_ids)
    with pytest.raises(GeometryError,match='unavailable'):query(model,edge_ids=children)


@pytest.mark.parametrize('callback_stage', [1, 2])
@pytest.mark.parametrize('reuse_seal_error', [False, True])
def test_enclosure_marker_identifies_only_own_seal_failure(callback_stage, reuse_seal_error):
    model,edge,_=fixture(); draft=capture(model,allow_seed=True)
    candidate=model.clone(preserve_identity=True)
    prior=definition(candidate,edge); _,children=candidate.split_edge(edge,.1)
    records=dict(draft.records)
    with pytest.raises(_SubcurveEnclosureUnavailable) as actual:
        record(draft,edge,.1,children,1e-30,model=candidate,parent_definition=prior)
    assert draft.enclosure_failure is actual.value
    error=actual.value if reuse_seal_error else _SubcurveEnclosureUnavailable('callback refusal')
    calls=0
    def callback(_):
        nonlocal calls
        calls+=1
        if calls==callback_stage:raise error
    draft.check=callback
    with pytest.raises(_SubcurveEnclosureUnavailable) as caught:
        record(draft,edge,.1,children,1e-9,model=candidate,parent_definition=prior)
    assert caught.value is error
    assert draft.enclosure_failure is None
    assert draft.records==records
