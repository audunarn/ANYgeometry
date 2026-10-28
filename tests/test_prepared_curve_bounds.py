"""Exact prepared/reference bounds, analytic jets and proof-frontier guards."""
from dataclasses import replace
from fractions import Fraction as Q
from types import SimpleNamespace

import numpy as np
import pytest

from anygeometry.nurbs import NurbsSurface
from anygeometry import GeometryError
from anygeometry.tolerance import TolerancePolicy
from anygeometry.nurbs_surface_intersection import (
    query_nurbs_surface_intersection,NurbsIntersectionCurve,IntersectionStatus,
    NurbsSurfaceIntersectionIntegrityError,NurbsSurfaceIntersectionCancelled)
from anygeometry.nurbs_surface_intersection._curve_bounds import (
    prepare_bound_evaluator,bound_segment,_BoundPreparation,ranges)
from anygeometry.nurbs_surface_intersection._intervals import Interval
from anygeometry.nurbs_intersection_curve import intersection_curve_operation


def sources(kind):
    knots = (0.,0.,0.,1.,1.,1.)
    if kind == 'generic':
        kv = (0.,0.,1.,1.)
        first = NurbsSurface(1,1,(0.,0.,.5,1.,1.),kv,
            tuple(tuple((x,y,0.) for y in (-1.,1.)) for x in (0.,.5,1.)))
        second = NurbsSurface(1,1,kv,kv,
            tuple(tuple((x,0.,z) for z in (-1.,1.)) for x in (-1.,2.)))
        return first,second
    weights = ((1.,)*3,(2.,)*3,(1.,)*3) if kind == 'rational' else None
    return tuple(NurbsSurface(2,2,knots,knots,
        tuple(tuple((float(i-1),float(j-1),
            float(offset*(j-1)) if weights else (1.,-1.,1.)[i]+.5*offset*(j-1))
            for j in range(3)) for i in range(3)),weights) for offset in (0,1))


@pytest.fixture(scope='module',params=('scalar','rational','generic'))
def problem(request):
    first,second = sources(request.param)
    policy = SimpleNamespace(max_boxes_per_pair=256,max_subdivision_depth=24) if request.param == 'generic' else None
    result = query_nurbs_surface_intersection(first,second,tolerances=TolerancePolicy(),qualification=policy)
    assert result.status is IntersectionStatus.CERTIFIED,result.reason
    return first,second,result


def test_prepared_full_bounds_equal_uncached_reference_including_point_and_repeated_intervals(problem):
    _,_,result = problem
    for ci,component in enumerate(result.components):
        prepared = prepare_bound_evaluator(result,ci)
        for ai in range(min(3,len(component.arcs))):
            for lo,hi in ((Q(1,4),Q(1,3)),(Q(1,2),Q(1,2)),(Q(1,4),Q(1,3))):
                interval = (Q(ai)+lo,Q(ai)+hi)
                expected = bound_segment(result,ci,ai,interval,(lo,hi))
                actual = prepared(ai,interval,(lo,hi),lambda:None)
                assert actual == expected
                assert prepared(ai,interval,(lo,hi),lambda:None,world_only=True) == (expected.world_lo,expected.world_hi)


def test_exact_partial_restriction_matches_reference_for_vector_grids_and_degenerate_axes(problem):
    prepared = _BoundPreparation(problem[2])
    grid = tuple(tuple(tuple(Q((7*i+3*j+k)%11-5,13) for k in range(4))
                       for j in range(4)) for i in range(3))
    for u in ((Q(0),Q(1)),(Q(1,7),Q(6,7)),(Q(1,3),Q(1,3))):
        for v in ((Q(0),Q(1)),(Q(1,5),Q(4,5)),(Q(0),Q(0)),(Q(1),Q(1))):
            expected = ranges(grid,(u,v))
            actual = prepared.ranges(grid,(u,v))
            assert actual == expected
            object.__setattr__(actual[0],'lo',Q(-999))
            assert prepared.ranges(grid,(u,v)) == expected


def test_public_parabolic_world_boxes_and_nonzero_analytic_jets():
    from test_nurbs_boundary_certification import graph_curve
    curve = graph_curve()
    with intersection_curve_operation():
        lo,hi = curve.bounds(Q(1,4),Q(1,3))
        # x=2t-1,z=(2t-1)^2. These extrema and derivatives are exact
        # continuous polynomial oracles, independent of sampled residuals.
        assert Q(float(lo[0])) <= Q(-1,2) and Q(float(hi[0])) >= Q(-1,3)
        assert Q(float(lo[2])) <= Q(1,9) and Q(float(hi[2])) >= Q(1,4)
        bound = curve.enclose(Q(1,4),Q(1,3))[0]
        assert bound.world_derivative[0].lo <= 2 <= bound.world_derivative[0].hi
        assert bound.world_derivative[2].lo <= -2 and bound.world_derivative[2].hi >= Q(-4,3)
        assert bound.world_second_derivative[2].lo <= 8 <= bound.world_second_derivative[2].hi
        point_lo,point_hi = curve.bounds(Q(1,4),Q(1,4))
        for a,b,x in zip(point_lo,point_hi,(Q(-1,2),Q(0),Q(1,4))):
            assert Q(float(a)) <= x <= Q(float(b))


def test_world_only_bridge_skips_jets_and_retains_hull_precision_without_point_samples(monkeypatch):
    from test_nurbs_boundary_certification import graph_curve
    import anygeometry.nurbs_surface_intersection._curve_bounds as module
    curve = graph_curve()
    def forbidden(*args,**kwargs):
        raise AssertionError('world-only box requested second-order jets')
    monkeypatch.setattr(module,'world_jet',forbidden)
    with intersection_curve_operation():
        shape = curve._shape()
        context_type = type(shape._context)
        def point(context,parameter):
            raise AssertionError('world box precision used a point evaluation target')
        monkeypatch.setattr(context_type,'evaluate',point)
        lo,hi = curve.bounds(Q(1,4),Q(1,3))
        assert np.all(lo <= hi)
        curve.bounds(Q(1,4),Q(1,4))


@pytest.mark.parametrize('reverse',(False,True))
def test_world_box_original_precision_across_arc_joint_and_reversed_subview(reverse):
    from test_nurbs_pair_public import model_pair
    from anygeometry.intersections import query_intersection
    model,faces = model_pair()
    curve = query_intersection(model,*faces).components[0].analytic_curve
    count = len(curve.owner.segments)
    assert count >= 2
    joint = max(1,count//2)
    view = curve.subcurve((Q(joint)-Q(1,2))/count,(Q(joint)+Q(1,2))/count)
    if reverse:
        view = view.subcurve(Q(1),Q(0))
    with intersection_curve_operation():
        for a,b in ((Q(1,2),Q(1,2)),(Q(1,4),Q(3,4))):
            complete = view.enclose(a,b)
            assert len(complete) == 2  # BOTH incident arcs, also at a singleton
            expected = (np.asarray([min(box.world_lo[k] for box in complete) for k in range(3)]),
                        np.asarray([max(box.world_hi[k] for box in complete) for k in range(3)]))
            actual = view.bounds(a,b)
            assert all(np.array_equal(x,y) for x,y in zip(actual,expected))


def test_world_results_detach_and_warmed_issuer_poison_refuses():
    from test_nurbs_boundary_certification import graph_curve
    import anygeometry.nurbs_intersection_curve as module
    curve = graph_curve()
    with intersection_curve_operation():
        first = curve.bounds(Q(1,4),Q(1,3))
        expected = tuple(x.copy() for x in first)
        first[0][:] = -999.; first[1][:] = 999.
        fresh = curve.bounds(Q(1,4),Q(1,3))
        assert all(np.array_equal(a,b) for a,b in zip(fresh,expected))
        module._OPERATIONS.get()[('shape',curve)] = object()
        with pytest.raises(GeometryError,match='cached prepared shape'):
            curve.bounds(Q(1,4),Q(1,3))


def test_prepared_bounds_mutation_conversion_and_later_cancellation_refuse():
    first,second = sources('scalar')
    result = query_nurbs_surface_intersection(first,second,tolerances=TolerancePolicy())
    owner = NurbsIntersectionCurve.from_result(result)
    state = {'cancel':False}
    context = owner.prepare(first=first,second=second,tolerances=TolerancePolicy(),
                            cancellation_check=lambda:state['cancel'])
    context.enclose(Q(1,4),Q(1,3))
    state['cancel'] = True
    with pytest.raises(NurbsSurfaceIntersectionCancelled):
        context._world_bounds(Q(1,4),Q(1,3))
    state['cancel'] = False
    shape = context.shape()
    class Change(int):
        @property
        def numerator(self):
            object.__setattr__(owner,'branch_anchor',('changed',))
            return 0
    with pytest.raises(NurbsSurfaceIntersectionIntegrityError):
        shape._world_bounds(Change(0),Q(1,3))


def test_warmed_world_cache_still_rejects_live_source_mutation_after_callback():
    first,second = sources('scalar')
    result = query_nurbs_surface_intersection(first,second,tolerances=TolerancePolicy())
    owner = NurbsIntersectionCurve.from_result(result)
    state = {'change':False}
    original = first.controls
    def callback():
        if state['change']:
            point = original[0][0]
            object.__setattr__(first,'controls',
                (((point[0],point[1],point[2]+.125),*original[0][1:]),*original[1:]))
        return False
    context = owner.prepare(first=first,second=second,tolerances=TolerancePolicy(),cancellation_check=callback)
    context._world_bounds(Q(1,4),Q(1,3))
    state['change'] = True
    try:
        with pytest.raises(NurbsSurfaceIntersectionIntegrityError):
            context._world_bounds(Q(1,4),Q(1,3))
    finally:
        object.__setattr__(first,'controls',original)
