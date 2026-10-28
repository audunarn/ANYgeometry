"""Exact point-preflight values and unchanged fallback/frontier semantics."""
from fractions import Fraction as Q
import sys
import pytest

from anygeometry import GeometryError
from anygeometry.surfaces import Plane
from anygeometry.regular_source_reference import SourceNode,SourceSide
from anygeometry.plane_source_reference import PlaneSide
from anygeometry import curvilinear_reference_maps as maps
from anygeometry import _point_fan_values as fast
from anygeometry._source_cell_point_error import fan_point_error_lower_squared
from anygeometry.plane_nurbs import CURRENT_WORK,charge_box
from anygeometry.section_audit import Work


def operands(width=Q(0)):
    uv=((Q(1,4),Q(1,4)),(Q(3,4),Q(1,4)),(Q(3,4),Q(3,4)),(Q(1,4),Q(3,4)))
    boxes=tuple(tuple((x-width,x+width) for x in point) for point in uv)
    nodes=tuple(SourceNode(('test',i),box) for i,box in enumerate(boxes))
    xyz=tuple((float(u),float(v),1.) for u,v in uv)
    return boxes,nodes,xyz


@pytest.mark.parametrize('width',(Q(0),Q(1,1000)))
@pytest.mark.parametrize('side',range(4))
@pytest.mark.parametrize('s',(Q(0),Q(1)))
def test_values_exactly_match_both_original_fan_outputs(width,side,s):
    boxes,nodes,xyz=operands(width)
    boundary=SourceSide(nodes[side],nodes[(side+1)%4])
    expected=(maps._source_quad_fan(boxes,boundary,side,(s,s),(Q(1,2),)*2)[0],
              maps._quad_fan(xyz,side,(s,s),(Q(1,2),)*2)[0])
    assert fast.canonical_point_values(boxes,boundary,side,s,Q(1,2),xyz)==expected


def run_probe(boundary,boxes,xyz,side,check):
    return fan_point_error_lower_squared(Plane((0.,0.,0.),(1.,0.,0.),(0.,1.,0.)),boundary,(Q(1,2),)*2,
        (xyz[side],xyz[(side+1)%4]),(Q(1,2),Q(1,2),Q(1)),check=check,
        quad_uv=boxes,quad_xyz=xyz,quad_side=side)


@pytest.mark.parametrize('side',range(4))
def test_exact_lower_witness_and_work_callbacks_match_original_route(monkeypatch,side):
    boxes,nodes,xyz=operands(Q(1,1000));boundary=SourceSide(nodes[side],nodes[(side+1)%4])
    outcomes=[]
    for original in (False,True):
        callbacks=[];work=Work(20,12);token=CURRENT_WORK.set(work)
        try:
            with monkeypatch.context() as patch:
                if original:patch.setattr(fast,'_context',lambda boundary:False)
                lower=run_probe(boundary,boxes,xyz,side,lambda:callbacks.append(1))
            outcomes.append((lower,work.boxes,len(callbacks)))
        finally:CURRENT_WORK.reset(token)
    assert outcomes[0]==outcomes[1]==(Q(1),2,4)


@pytest.mark.parametrize('plane_side',(False,True))
def test_curved_sides_keep_box_derivative_work_and_callbacks(monkeypatch,plane_side):
    boxes,nodes,xyz=operands();native=SourceSide(nodes[0],nodes[1],curve=object())
    boundary=PlaneSide(None,native,nodes[0],nodes[1]) if plane_side else native
    cls=PlaneSide if plane_side else SourceSide
    calls=[]
    def box(self,lo,hi):
        charge_box();calls.append('box')
        return ((Q(1,2),)*2,(Q(1,4),)*2)
    def derivative(self,lo,hi):
        charge_box();calls.append('derivative')
        return ((Q(1,2),)*2,(Q(0),)*2)
    monkeypatch.setattr(cls,'box',box);monkeypatch.setattr(cls,'derivative',derivative)
    assert fast.canonical_point_values(boxes,boundary,0,Q(0),Q(1,2),xyz) is None
    outcomes=[]
    for original in (False,True):
        calls.clear();callbacks=[];work=Work(100,12);token=CURRENT_WORK.set(work)
        try:
            with monkeypatch.context() as patch:
                if original:patch.setattr(fast,'_context',lambda boundary:False)
                lower=run_probe(boundary,boxes,xyz,0,lambda:callbacks.append(1))
            outcomes.append((lower,work.boxes,len(callbacks),tuple(calls)))
        finally:CURRENT_WORK.reset(token)
    assert outcomes[0]==outcomes[1]
    assert outcomes[0][1]>2 and 'derivative' in outcomes[0][3]


@pytest.mark.parametrize('name',('_source_quad_fan','_quad_fan','mul','interval'))
def test_changed_effective_mapping_uses_original_route(monkeypatch,name):
    boxes,nodes,xyz=operands();boundary=SourceSide(nodes[0],nodes[1])
    original=getattr(maps,name);calls=[]
    def wrapper(*args,**kwargs):
        calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(maps,name,wrapper)
    assert fast.canonical_point_values(boxes,boundary,0,Q(0),Q(1,2),xyz) is None
    assert run_probe(boundary,boxes,xyz,0,lambda:None)==1
    assert calls


def test_keyword_default_and_in_place_code_mutation_disable_specialization(monkeypatch):
    boxes,nodes,xyz=operands();boundary=SourceSide(nodes[0],nodes[1])
    with monkeypatch.context() as patch:
        patch.setattr(maps._source_quad_fan,'__kwdefaults__',{'corner_hull':True})
        assert fast.canonical_point_values(boxes,boundary,0,Q(0),Q(1,2),xyz) is None
    def replaced(xyz,side,s,t):
        return (((0,0),)*3,(),())
    monkeypatch.setattr(maps._quad_fan,'__code__',replaced.__code__)
    assert fast.canonical_point_values(boxes,boundary,0,Q(0),Q(1,2),xyz) is None


def test_unknown_family_or_nonprobe_inputs_fall_back():
    boxes,nodes,xyz=operands()
    class Unknown(SourceSide):pass
    assert fast.canonical_point_values(boxes,Unknown(nodes[0],nodes[1]),0,Q(0),Q(1,2),xyz) is None
    boundary=SourceSide(nodes[0],nodes[1])
    assert fast.canonical_point_values(boxes,boundary,0,Q(1,3),Q(1,2),xyz) is None
    assert fast.canonical_point_values(boxes,boundary,0,Q(0),Q(1,3),xyz) is None


def test_work_exhaustion_and_callback_refusals_are_not_skipped():
    boxes,nodes,xyz=operands();boundary=SourceSide(nodes[0],nodes[1])
    work=Work(0,12);token=CURRENT_WORK.set(work)
    try:
        with pytest.raises(GeometryError):run_probe(boundary,boxes,xyz,0,lambda:None)
    finally:CURRENT_WORK.reset(token)
    error=GeometryError('cancelled point preflight')
    def fail():raise error
    with pytest.raises(GeometryError) as caught:run_probe(boundary,boxes,xyz,0,fail)
    assert caught.value is error


def test_callback_mutation_rechecks_context_at_each_probe(monkeypatch):
    boxes,nodes,xyz=operands();boundary=SourceSide(nodes[0],nodes[1]);calls=[]
    original=maps._quad_fan
    def changed(xyz,side,s,t):
        return (((0,0),)*3,(),())
    def check():
        calls.append(1)
        if len(calls)==3:monkeypatch.setattr(original,'__code__',changed.__code__)
    assert run_probe(boundary,boxes,xyz,0,check)==1
    assert len(calls)==4


def test_public_preflight_does_not_compute_full_straight_fan_triples():
    boxes,nodes,xyz=operands();boundary=SourceSide(nodes[0],nodes[1]);calls=[]
    codes={maps._source_quad_fan.__code__,maps._quad_fan.__code__}
    def profile(frame,event,arg):
        if event=='call' and frame.f_code in codes:calls.append(frame.f_code)
    previous=sys.getprofile()
    try:
        sys.setprofile(profile)
        result=run_probe(boundary,boxes,xyz,0,lambda:None)
    finally:sys.setprofile(previous)
    assert result==1 and calls==[]


@pytest.mark.parametrize('side',range(4))
def test_asymmetric_interval_corners_preserve_dependency_and_hull_clamp(side):
    boxes,nodes,xyz=operands()
    boxes=tuple(tuple((a-Q(1+2*i+j,1000),b+Q(2+i+3*j,1000))
                      for j,(a,b) in enumerate(point)) for i,point in enumerate(boxes))
    boundary=SourceSide(SourceNode(('first',),boxes[side]),SourceNode(('last',),boxes[(side+1)%4]))
    xyz=((Q(1,3),Q(-2,7),Q(5,9)),(Q(4,5),Q(2,3),Q(-1,7)),
         (Q(-8,3),Q(4,9),Q(2)),(Q(5,11),Q(1,17),Q(-3,5)))
    for s in (Q(0),Q(1)):
        assert fast.canonical_point_values(boxes,boundary,side,s,Q(1,2),xyz)==(
            maps._source_quad_fan(boxes,boundary,side,(s,s),(Q(1,2),)*2)[0],
            maps._quad_fan(xyz,side,(s,s),(Q(1,2),)*2)[0])
