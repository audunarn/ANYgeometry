"""Actual CONNECT preparation and owner material transport, without cell proof."""
from dataclasses import replace
from fractions import Fraction as Q
from contextlib import contextmanager
import json
import time
import pytest
from anygeometry import (GeometryError,clone_prepared_geometry,plan_intersections,apply_intersections,
                         query_material_surface_regions)
from anygeometry.local_field_transport import query_local_field_transport
from anygeometry.local_contact_transport import (query_local_contact_transport,
    validate_local_contact_transport,query_local_contact_exclusion)
from anygeometry.nurbs_intersection_curve import intersection_curve_operation
from test_local_contact_stars import source_stars


@contextmanager
def phase(name):
    started=time.perf_counter()
    print('CONTACT_STAGE '+json.dumps({'phase':name,'event':'enter','monotonic':started}),flush=True)
    try:
        yield
    except BaseException as error:
        print('CONTACT_STAGE '+json.dumps({'phase':name,'event':'error','type':type(error).__name__,
            'elapsed':time.perf_counter()-started}),flush=True)
        raise
    else:
        print('CONTACT_STAGE '+json.dumps({'phase':name,'event':'return',
            'elapsed':time.perf_counter()-started}),flush=True)


@pytest.fixture(scope='module')
def realized(source_stars):
    source,parents,stars=source_stars
    with phase('clone'):
        prepared=clone_prepared_geometry(source)
    with intersection_curve_operation():
        with phase('plan_CONNECT'):
            plan=plan_intersections(prepared,parents,policy='connect')
        with phase('apply_CONNECT'):
            apply_intersections(prepared,plan,policy='connect')
    with phase('material_regions'):
        regions=query_material_surface_regions(prepared,tuple(prepared.faces))
    with phase('material_transport'):
        transport=query_local_field_transport(source,prepared,regions)
    with phase('contact_realization'):
        result=query_local_contact_transport(source,prepared,stars,transport)
    return source,prepared,stars,transport,result


def test_actual_pre_cell_realization_complete_uses_and_finite_lookup(realized):
    source,prepared,stars,transport,result=realized
    with phase('contact_binding'):
        validate_local_contact_transport(source,prepared,result)
    assert result.trace_fragments and all(len(r[3])==4 for r in result.trace_fragments)
    row=transport.footprints[0]
    assert row.source_from_child==((1,0,0),(0,1,0))
    star=stars.stars[0]
    with phase('contact_lookup_inner'):
        answer=query_local_contact_exclusion(source,prepared,result,row.prepared_face.id,star.inner_box)
    assert answer is not None and answer.excluded_partner_native_box==star.outer_box
    assert answer.partner_face!=answer.source_face and answer.retained_partner_native_rectangles
    assert answer.exterior_squared_lower==star.exterior_squared_lower
    with phase('contact_lookup_full'):
        assert query_local_contact_exclusion(source,prepared,result,row.prepared_face.id,((0,1),(0,1))) is None


def test_unprepared_clone_cannot_realize_source_intent(source_stars):
    source,_,stars=source_stars
    with phase('clone'):
        prepared=clone_prepared_geometry(source)
    with phase('material_regions'):
        regions=query_material_surface_regions(prepared,tuple(prepared.faces))
    with phase('material_transport'):
        transport=query_local_field_transport(source,prepared,regions)
    with pytest.raises(GeometryError): query_local_contact_transport(source,prepared,stars,transport)


def test_transport_copies_and_fake_material_receipts_refused(realized):
    source,prepared,stars,transport,result=realized
    with pytest.raises(GeometryError): validate_local_contact_transport(source,prepared,replace(result))
    with pytest.raises(GeometryError): query_local_contact_transport(source,prepared,stars,replace(transport))


@pytest.mark.parametrize('operand',('receipt','regions','trace','junction'))
def test_realization_raw_mutation_in_callback_is_refused(realized,operand):
    source,prepared,stars,transport,result=realized
    edge=prepared.edges[result.trace_fragments[0][2]]
    if operand=='receipt': target,name,bad=result,'trace_fragments',()
    elif operand=='regions': target,name,bad=transport.regions,'regions',()
    elif operand=='trace': target,name,bad=edge.curve,'domain',(Q(0),Q(1,2))
    else:
        target=next(iter(prepared.junctions.values())); name='attachment_ids'; bad=()
    old=getattr(target,name)
    def mutate(phase): object.__setattr__(target,name,bad)
    try:
        with pytest.raises(GeometryError): validate_local_contact_transport(source,prepared,result,cancellation_check=mutate)
    finally: object.__setattr__(target,name,old)
    with phase('contact_binding'):
        validate_local_contact_transport(source,prepared,result)


def test_realization_late_cancel_does_not_return_exclusion(realized):
    source,prepared,_,transport,result=realized
    with pytest.raises(GeometryError,match='cancelled'):
        query_local_contact_exclusion(source,prepared,result,transport.footprints[0].prepared_face.id,
                                      ((0,1),(0,1)),cancellation_check=lambda phase: True)


def test_realization_budget_and_missing_fragment_refuse(realized):
    from anygeometry.predicates import IntersectionQualificationPolicy
    source,prepared,stars,transport,result=realized
    with pytest.raises(GeometryError,match='budget'):
        query_local_contact_transport(source,prepared,stars,transport,
            policy=IntersectionQualificationPolicy(max_boxes_per_pair=1))
    edge_id=result.trace_fragments[0][2]
    # Deliberately corrupt real backing storage; public edges is read-only.
    before_edges=dict(prepared._edges)
    prepared._edges.pop(edge_id)
    try:
        with pytest.raises(GeometryError): query_local_contact_transport(source,prepared,stars,transport)
    finally:
        prepared._edges.clear()
        prepared._edges.update(before_edges)
    with phase('contact_binding'):
        validate_local_contact_transport(source,prepared,result)



def test_internal_frontier_uses_bindings_but_entry_and_exit_are_fresh(realized,monkeypatch):
    """Control test with genuine issued inputs; no fabricated positive receipt."""
    import anygeometry.local_contact_transport as service
    from anygeometry.predicates import DEFAULT_INTERSECTION_QUALIFICATION_POLICY as policy
    source,prepared,stars,transport,_=realized
    original=service._inputs; calls=[]
    def fresh(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(service,'_inputs',fresh)
    initial,check,finish=service._binding_frontier(source,prepared,stars,transport,policy,None)
    assert len(calls)==1
    for _ in range(5): assert check() is False
    assert len(calls)==1
    finish()
    assert len(calls)==2 and initial==service._snapshot(source,prepared,stars,transport)


def test_frontier_external_callback_has_both_fresh_gates(realized,monkeypatch):
    import anygeometry.local_contact_transport as service
    from anygeometry.predicates import DEFAULT_INTERSECTION_QUALIFICATION_POLICY as policy
    source,prepared,stars,transport,_=realized
    original=service._inputs; events=[]
    def fresh(*args):
        events.append('fresh')
        return original(*args)
    def callback(phase): events.append('callback')
    monkeypatch.setattr(service,'_inputs',fresh)
    _,check,finish=service._binding_frontier(source,prepared,stars,transport,policy,callback)
    check(); finish()
    assert events==['fresh','fresh','callback','fresh','fresh']


@pytest.mark.parametrize('operand',('source','stars','regions'))
def test_new_frontier_callback_mutations_cannot_finish(realized,operand):
    import anygeometry.local_contact_transport as service
    from anygeometry.predicates import DEFAULT_INTERSECTION_QUALIFICATION_POLICY as policy
    source,prepared,stars,transport,_=realized
    if operand=='source':
        target=source.faces[stars.face_ids[0]].surface; name='controls'
        bad=tuple(tuple((p[0],p[1],p[2]+1) for p in row) for row in target.controls)
    elif operand=='stars': target,name,bad=stars,'intent','IMPRINT'
    else: target,name,bad=transport.regions,'regions',()
    old=getattr(target,name)
    def mutate(phase): object.__setattr__(target,name,bad)
    _,check,_=service._binding_frontier(source,prepared,stars,transport,policy,mutate)
    try:
        with pytest.raises(GeometryError): check()
    finally: object.__setattr__(target,name,old)


def test_final_fresh_gate_detects_withdrawn_issuer_authority(realized):
    import anygeometry.local_contact_transport as service
    import anygeometry.local_field_transport as material
    from anygeometry.predicates import DEFAULT_INTERSECTION_QUALIFICATION_POLICY as policy
    source,prepared,stars,transport,_=realized
    _,check,finish=service._binding_frontier(source,prepared,stars,transport,policy,None)
    issued=material._ISSUED.pop(transport)
    try:
        assert check() is False  # literal identity alone never issues authority
        with pytest.raises(GeometryError): finish()
    finally: material._ISSUED[transport]=issued


def test_new_frontier_cancel_exception_and_cap_propagate(realized):
    import anygeometry.local_contact_transport as service
    from anygeometry.predicates import IntersectionQualificationPolicy
    source,prepared,stars,transport,_=realized
    policy=IntersectionQualificationPolicy(max_boxes_per_pair=1)
    _,check,_=service._binding_frontier(source,prepared,stars,transport,policy,None)
    check()
    with pytest.raises(GeometryError,match='budget'): check()
    _,check,_=service._binding_frontier(source,prepared,stars,transport,policy,lambda phase: True)
    with pytest.raises(GeometryError,match='cancelled'): check()
    class Stop(Exception): pass
    def stop(phase): raise Stop('caller exception unchanged')
    _,check,_=service._binding_frontier(source,prepared,stars,transport,policy,stop)
    with pytest.raises(Stop): check()

