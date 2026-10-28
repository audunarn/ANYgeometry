"""Exact brute-force reference and refusal tests; no geometry authority stubs."""
from fractions import Fraction as Q
import pytest
from anygeometry._local_clearance_bvh import (
    CompetitorPiece,ClearanceBudget,ClearanceBVHRefused,
    build_clearance_bvh,query_clearance_bvh)


def box(x,y=0,z=0,width=0):
    return tuple((Q(v),Q(v)+Q(width)) for v in (x,y,z))


def distance(a,b):
    return sum((max(Q(0),lo-hi2,lo2-hi)**2 for (lo,hi),(lo2,hi2) in zip(a,b)),Q(0))


@pytest.mark.parametrize('query',(box(0),box(Q(1,3),2),box(20,4,-2),box(-3,1,1,Q(1,4))))
def test_exact_brute_force_parity_and_all_ties(query):
    records=tuple(CompetitorPiece(('piece',i),box(Q(i,3),i%3,i%2,Q(1,6))) for i in range(-8,9))
    budget=ClearanceBudget(10000)
    tree=build_clearance_bvh(records,budget=budget)
    result=query_clearance_bvh(tree,query,budget=budget)
    expected=min(distance(query,p.box) for p in records)
    assert result.minimum_squared==expected
    assert result.limiting_keys==tuple(p.key for p in records if distance(query,p.box)==expected)
    assert result.status==('zero' if expected==0 else 'positive')
    assert tree.record_count==len(records)
    assert result.cumulative==budget.counters
    assert result.counters.operations>0 and result.counters.operations==sum(result.counters[1:])


def test_remote_same_curve_piece_is_never_filtered():
    near=CompetitorPiece(('curve',7,(Q(0),Q(1,4))),box(9))
    remote=CompetitorPiece(('curve',7,(Q(3,4),Q(1))),box(Q(1,10)))
    budget=ClearanceBudget(100)
    tree=build_clearance_bvh((near,remote),budget=budget)
    result=query_clearance_bvh(tree,box(0),budget=budget)
    assert result.minimum_squared==Q(1,100) and result.limiting_keys==(remote.key,)


def test_overlap_touch_and_heterogeneous_key_ties_remain_inconclusive():
    records=(CompetitorPiece('first',box(0,width=1)),CompetitorPiece(2,box(1)),
        CompetitorPiece(('adjacent',Q(1,2)),box(0)))
    budget=ClearanceBudget(1000)
    result=query_clearance_bvh(build_clearance_bvh(records,budget=budget),box(0,width=1),budget=budget)
    assert result.minimum_squared==0 and result.status=='zero'
    assert result.limiting_keys==tuple(p.key for p in records)


def test_empty_is_explicit_absence_not_infinite_clearance():
    budget=ClearanceBudget(10)
    tree=build_clearance_bvh((),budget=budget)
    result=query_clearance_bvh(tree,box(0),budget=budget)
    assert tree.record_count==0
    assert result.minimum_squared is None and result.status=='empty' and result.limiting_keys==()
    assert budget.counters.operations==2


@pytest.mark.parametrize('records',(
    [CompetitorPiece('x',box(0))],
    (CompetitorPiece('x',((0.,1.),(0,1),(0,1))),),
    (CompetitorPiece('x',((1,0),(0,1),(0,1))),),
    (CompetitorPiece('x',((0,1),(0,1))),),
    (CompetitorPiece(['mutable'],box(0)),),
    (CompetitorPiece('duplicate',box(0)),CompetitorPiece('duplicate',box(1))),
))
def test_malformed_or_duplicate_records_refuse(records):
    with pytest.raises(ClearanceBVHRefused):build_clearance_bvh(records,budget=ClearanceBudget(100))


@pytest.mark.parametrize('cap',(True,False,0,-1,1.5,float('inf')))
def test_budget_requires_positive_finite_integer(cap):
    with pytest.raises(ClearanceBVHRefused):ClearanceBudget(cap)


def test_shared_budget_is_not_reset_between_builds_and_query():
    budget=ClearanceBudget(4)
    tree=build_clearance_bvh((CompetitorPiece('x',box(0)),),budget=budget)
    build_clearance_bvh((CompetitorPiece('y',box(2)),),budget=budget)
    assert budget.counters.operations==4
    with pytest.raises(ClearanceBVHRefused,match='exhausted'):query_clearance_bvh(tree,box(1),budget=budget)
    assert budget.counters.operations==4


def test_cancellation_exception_and_cap_mutation_are_not_swallowed():
    class Stop(BaseException):pass
    def stop(phase):raise Stop()
    budget=ClearanceBudget(10,cancellation_check=stop)
    with pytest.raises(Stop):build_clearance_bvh((),budget=budget)
    assert budget.counters.operations==1
    with pytest.raises(ClearanceBVHRefused,match='cancelled'):
        build_clearance_bvh((),budget=ClearanceBudget(10,cancellation_check=lambda phase:True))
    def mutate(phase):object.__setattr__(changed,'_configuration',(1000,64,None))
    changed=ClearanceBudget(2,cancellation_check=mutate)
    with pytest.raises(ClearanceBVHRefused,match='changed'):build_clearance_bvh((),budget=changed)


def test_active_owner_work_is_charged_without_reset_or_replacement():
    from anygeometry.plane_nurbs import CURRENT_WORK
    class Work:
        max_depth=64
        def __init__(self):self.calls=0
        def take(self):self.calls+=1;return self.calls<=2
    work=Work();token=CURRENT_WORK.set(work)
    budget=ClearanceBudget(100)
    try:
        tree=build_clearance_bvh((CompetitorPiece('x',box(0)),),budget=budget)
        with pytest.raises(Exception,match='budget exhausted'):query_clearance_bvh(tree,box(1),budget=budget)
        assert CURRENT_WORK.get() is work and work.calls==3
        assert budget.counters.operations==3
    finally:CURRENT_WORK.reset(token)


def test_deterministic_tree_and_query_and_immutable_boxes():
    records=tuple(CompetitorPiece(key,box(0)) for key in ('a',2,('c',Q(1,2))))
    first=build_clearance_bvh(records,budget=ClearanceBudget(100))
    second=build_clearance_bvh(records,budget=ClearanceBudget(100))
    assert first==second
    with pytest.raises(TypeError):first[0]=None
    with pytest.raises(AttributeError):records[0].box=box(1)
    result=query_clearance_bvh(first,box(2),budget=ClearanceBudget(100))
    assert result.limiting_keys==tuple(p.key for p in records)


def test_pruning_retains_exact_minimum_and_charges_prunes():
    budget=ClearanceBudget(1000)
    tree=build_clearance_bvh(tuple(CompetitorPiece(i,box(i*100)) for i in range(8)),budget=budget)
    result=query_clearance_bvh(tree,box(0),budget=budget)
    assert result.minimum_squared==0 and result.limiting_keys==(0,)
    assert result.counters.pruned_nodes>0 and result.counters.leaf_tests<8


def test_depth_limit_and_malformed_query_refuse():
    with pytest.raises(ClearanceBVHRefused,match='depth'):
        build_clearance_bvh((CompetitorPiece('a',box(0)),CompetitorPiece('b',box(1))),
            budget=ClearanceBudget(100,max_depth=0))
    budget=ClearanceBudget(100)
    tree=build_clearance_bvh((),budget=budget)
    with pytest.raises(ClearanceBVHRefused,match='exact'):
        query_clearance_bvh(tree,((0.,1.),(0,1),(0,1)),budget=budget)
