"""Exact ordinary-distance shortcut and real saved circle-pair field integration."""
from dataclasses import replace
from fractions import Fraction as Q
import pytest
from anygeometry import GeometryError
from anygeometry.nurbs import NurbsSurface
from anygeometry._local_graph_clearance import GraphClearance
from anygeometry._local_spatial_clearance import SpatialBudget
from anygeometry._detached_geometry_read import NumericalBoundary

FULL=((Q(0),Q(1)),(Q(0),Q(1)))


def planes(slope=0,height=2,scale=1,shift=(0,0,0)):
    knots=(0.,0.,1.,1.)
    def support(upper):
        return NurbsSurface(1,1,knots,knots,tuple(tuple(tuple(float(scale*x+d) for x,d in zip(
            (u,v,slope*u+height if upper else 0),shift)) for v in (0,1)) for u in (0,1)))
    return support(False),support(True)


def operation(supports,slacks=()):
    holder=[]
    def guard():
        if holder:holder[0].guard()
    boundary=NumericalBoundary(guard,None)
    budget=SpatialBudget(boundary.tick)
    op=GraphClearance(dict(enumerate(supports)),budget,slacks);holder.append(op)
    return op,boundary


@pytest.mark.parametrize('slope,scale,shift',((0,1,(0,0,0)),(2,1,(0,0,0)),(2,8,(32,-16,4))))
def test_exact_plane_distance_oracle_and_complete_target_cover(slope,scale,shift):
    op,boundary=operation(planes(slope=slope,scale=scale,shift=shift))
    try:
        gap,leaves=op.query(0,1,FULL)
        assert gap==Q(4*scale*scale,1+slope*slope)
        assert leaves==((FULL,gap),)
        op.guard()
    finally:boundary.close()


@pytest.mark.parametrize('ranges',(((1.,0.),(1.,0.)),((.25,.75),(.25,.75))))
def test_common_crop_and_reversal_are_bound_in_native_coordinates(ranges):
    supports=tuple(replace(s,u_range=ranges[0],v_range=ranges[1]) for s in planes())
    op,boundary=operation(supports)
    try:assert op.query(0,1,FULL)[0]==4
    finally:boundary.close()


def test_nonuniform_weights_unequal_domain_and_intersection_keep_fallback():
    a,b=planes()
    alternatives=(replace(b,weights=((1.,2.),(1.,2.))),replace(b,u_range=(.25,.75)),planes(slope=2,height=-1)[1])
    for target in alternatives:
        op,boundary=operation((a,target))
        try:assert op.query(0,1,FULL) is None
        finally:boundary.close()


def test_callback_pair_mutation_exception_and_work_limit():
    op,boundary=operation(planes())
    try:
        op.query(0,1,FULL);pair=op.pairs[(0,1)];old=pair.gradient_squared_upper
        def mutate(phase):
            object.__setattr__(pair,'gradient_squared_upper',old/2)
            return False
        boundary._callback=mutate
        with pytest.raises(GeometryError,match='pair changed'):op.query(0,1,FULL)
        object.__setattr__(pair,'gradient_squared_upper',old)
        token=GeometryError('external token')
        def stop(phase):raise token
        boundary._callback=stop
        with pytest.raises(GeometryError) as caught:op.query(0,1,FULL)
        assert caught.value is token
        boundary._callback=lambda phase:True
        with pytest.raises(GeometryError,match='cancelled'):op.query(0,1,FULL)
        boundary._callback=None;op.budget.limit=op.budget.used
        with pytest.raises(GeometryError,match='budget'):op.query(0,1,FULL)
    finally:boundary.close()


def test_unknown_initial_runtime_falls_back_and_changed_runtime_refuses(monkeypatch):
    from anygeometry import _local_graph_clearance as graph
    op,boundary=operation(planes())
    old=graph._derive
    try:
        monkeypatch.setattr(graph,'_derive',lambda *args:None)
        with pytest.raises(GeometryError,match='context'):op.guard()
        unknown,other=operation(planes())
        try:assert unknown.query(0,1,FULL) is None
        finally:other.close()
    finally:
        monkeypatch.setattr(graph,'_derive',old);boundary.close()


def test_saved_circle_pair_actual_local_field_uses_complete_graph_clearance():
    from test_nurbs_pair_public import model_pair
    from anygeometry.serialization import to_dict,from_dict
    from anygeometry.local_approximation_field import (query_local_approximation_field,
        validate_local_approximation_field,query_local_approximation_limit)
    source,faces=model_pair()
    for face in faces:source.add_sheet((face,))
    source=from_dict(to_dict(source));before=to_dict(source)
    field=query_local_approximation_field(source,faces,relative_accuracy=.001,scale_error=.01,
        policy_floor=1e-10,connect_intent=True)
    validate_local_approximation_field(source,field)
    kinds=[kind for face,box,limit,rows in field.spatial_ledger for other,kind,record in rows]
    assert 'implicit_graph_clearance' in kinds and 'conditional_contact' in kinds
    for face,matrix,tiles in field.face_tiles:
        assert sum((box[0][1]-box[0][0])*(box[1][1]-box[1][0]) for box,limit,_ in tiles)==1
    with pytest.raises(GeometryError,match='realization'):query_local_approximation_limit(source,field,faces[0])
    assert to_dict(source)==before
    target=source.faces[faces[0]];old=target.metadata
    def mutate(phase):object.__setattr__(target,'metadata',dict(old,changed=True))
    try:
        with pytest.raises(GeometryError):validate_local_approximation_field(source,field,cancellation_check=mutate)
    finally:object.__setattr__(target,'metadata',old)



def test_oblique_translated_projection_retains_absolute_height_and_point_oracle():
    a,b=planes(slope=2)
    def oblique(s):
        return replace(s,controls=tuple(tuple((x+z+7,y-5,z+11) for x,y,z in row) for row in s.controls))
    a,b=oblique(a),oblique(b);op,boundary=operation((a,b))
    try:
        gap,_=op.query(0,1,FULL)
        # In these physical coordinates target plane is 3*(z-11)-2*(x-7)-2=0.
        assert gap==Q(4,13)
        for u in (Q(0),Q(1,2),Q(1)):
            for v in (Q(0),Q(1)):
                point=a.exact_point(u,v)
                for s in (Q(0),Q(1,3),Q(1)):
                    for t in (Q(0),Q(2,3),Q(1)):
                        target=b.exact_point(s,t)
                        assert gap<=sum((x-y)**2 for x,y in zip(point,target))
        op.guard()
    finally:boundary.close()


def test_multiple_native_spans_retain_complete_target_gradient_cover():
    knots=(0.,0.,.5,1.,1.);kv=(0.,0.,1.,1.)
    a=NurbsSurface(1,1,knots,kv,tuple(((x,0.,0.),(x,1.,0.)) for x in (0.,.5,1.)))
    b=replace(a,controls=tuple(tuple((x,y,2.) for x,y,z in row) for row in a.controls))
    op,boundary=operation((a,b))
    try:
        assert op.query(0,1,FULL)[0]==4
        op.guard()
    finally:boundary.close()


@pytest.mark.parametrize('side',(0,1))
def test_nonzero_authenticated_transport_never_uses_nominal_gap(side):
    supports=planes(height=2)
    # A displacement allowance of 2 can consume the entire nominal separation.
    # Either endpoint's allowance must disable the nominal exact-graph shortcut.
    op,boundary=operation(supports,{id(supports[side]):Q(2)})
    try:assert op.query(0,1,FULL) is None
    finally:boundary.close()
