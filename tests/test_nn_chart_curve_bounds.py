"""Position-only proofs retain full enclosure positions and owner frontiers."""
from fractions import Fraction as Q
from types import SimpleNamespace
import pytest
from anygeometry.nurbs_surface_intersection import (
    query_nurbs_surface_intersection, NurbsIntersectionCurve, IntersectionStatus,
    NurbsSurfaceIntersectionCancelled, NurbsSurfaceIntersectionIntegrityError)
from anygeometry.tolerance import TolerancePolicy
from anygeometry.errors import GeometryError
from test_prepared_curve_bounds import sources


def prepare(kind='scalar', callback=None):
    first,second = sources(kind)
    policy = SimpleNamespace(max_boxes_per_pair=256,max_subdivision_depth=24) if kind == 'generic' else None
    result = query_nurbs_surface_intersection(first,second,tolerances=TolerancePolicy(),qualification=policy)
    assert result.status is IntersectionStatus.CERTIFIED
    owner = NurbsIntersectionCurve.from_result(result)
    return first,second,owner.prepare(first=first,second=second,tolerances=TolerancePolicy(),
        qualification=policy,cancellation_check=callback)


def reference(bounds):
    return tuple(tuple((min(getattr(box,name)[k][0] for box in bounds),
                        max(getattr(box,name)[k][1] for box in bounds)) for k in range(2))
                 for name in ('first_chart','second_chart'))


@pytest.mark.parametrize('kind',('scalar','rational','generic'))
def test_chart_only_exact_full_reference_cold_warm_and_reversed(kind):
    _,_,context = prepare(kind)
    n = int(context.domain[1])
    requests = [(Q(0),Q(0)),(Q(n),Q(n)),(Q(n,4),Q(3*n,4)),(Q(0),Q(n))]
    if n>1:
        requests.extend(((Q(1),Q(1)),(Q(1,2),Q(3,2))))
    for a,b in requests:
        cold = context._chart_bounds(a,b)
        assert cold == reference(context.enclose(a,b))
        assert context._chart_bounds(a,b) == cold
    for start,end in ((Q(0),Q(1)),(Q(3,4),Q(1,4))):
        shape = context.shape().subcurve(start,end)
        for a,b in ((Q(0),Q(0)),(Q(1),Q(1)),(Q(0),Q(1)),(Q(1,3),Q(2,3))):
            cold = shape._chart_bounds(a,b)
            assert cold == reference(shape.enclose(a,b))
            assert shape._chart_bounds(a,b) == cold


def test_chart_only_omits_unused_jets_but_keeps_original_support_hulls(monkeypatch):
    _,_,context = prepare()
    import anygeometry.nurbs_surface_intersection._curve_bounds as bounds
    from anygeometry.nurbs_surface_intersection._atlas import _Surface
    original = _Surface.hull
    calls = []
    def hull(self,*args,**kwargs):
        calls.append(1)
        return original(self,*args,**kwargs)
    def no_jet(*args,**kwargs):
        raise AssertionError('chart position requested a derivative jet')
    monkeypatch.setattr(_Surface,'hull',hull)
    monkeypatch.setattr(bounds._BoundPreparation,'jet',no_jet)
    monkeypatch.setattr(bounds,'world_jet',no_jet)
    context._chart_bounds(Q(1,4),Q(1,3))
    assert len(calls)>=2


def test_chart_cache_issues_immutable_native_boxes_and_still_checks_later_callback():
    state={'cancel':False}
    _,_,context = prepare(callback=lambda:state['cancel'])
    original = context._chart_bounds(Q(1,4),Q(1,3))
    changed = context._chart_bounds(Q(1,4),Q(1,3))
    # Normalized chart_box from the existing bound engine contains native
    # outward binary64 endpoints, not mutable Fraction implementation objects.
    assert all(type(value) is float for support in changed for axis in support for value in axis)
    assert changed is not original
    with pytest.raises(TypeError):
        changed[0][0][0] = -999
    with pytest.raises((AttributeError,TypeError)):
        object.__setattr__(changed[0][0][0],'_numerator',-999)
    assert context._chart_bounds(Q(1,4),Q(1,3)) == original
    state['cancel']=True
    with pytest.raises(NurbsSurfaceIntersectionCancelled):
        context._chart_bounds(Q(1,4),Q(1,3))


def test_warmed_shape_chart_cache_rejects_revoked_issuer(monkeypatch):
    import importlib
    module=importlib.import_module('anygeometry.nurbs_surface_intersection.curve')
    _,_,context=prepare()
    shape=context.shape()
    shape._chart_bounds(Q(1,4),Q(1,3))
    monkeypatch.delitem(module._ISSUED_CONTEXTS,id(context))
    with pytest.raises(NurbsSurfaceIntersectionIntegrityError):
        shape._chart_bounds(Q(1,4),Q(1,3))


def test_chart_cache_raw_source_and_conversion_mutation_refuse():
    first,_,context = prepare()
    context._chart_bounds(Q(1,4),Q(1,3))
    controls = first.controls
    point=controls[0][0]
    object.__setattr__(first,'controls',(((point[0]+.125,*point[1:]),*controls[0][1:]),*controls[1:]))
    try:
        with pytest.raises(NurbsSurfaceIntersectionIntegrityError):
            context._chart_bounds(Q(1,4),Q(1,3))
    finally:
        object.__setattr__(first,'controls',controls)
    shape=context.shape()
    class Change(int):
        @property
        def numerator(self):
            shape._end += 1
            return 0
    with pytest.raises(NurbsSurfaceIntersectionIntegrityError):
        shape._chart_bounds(Change(0),Q(1,3))


def test_chart_contraction_refusal_and_callback_exception_propagate(monkeypatch):
    first,second,context = prepare()
    import anygeometry.nurbs_surface_intersection._curve_bounds as bounds
    sentinel=GeometryError('exact chart contraction refusal')
    def refuse(*args,**kwargs):
        raise sentinel
    monkeypatch.setattr(bounds._BoundPreparation,'ranges',refuse)
    with pytest.raises(GeometryError) as caught:
        context._chart_bounds(Q(1,4),Q(1,3))
    assert caught.value is sentinel
    state={'raise':False}
    def callback():
        if state['raise']:
            raise sentinel
        return False
    _,_,other=prepare(callback=callback)
    state['raise']=True
    with pytest.raises(GeometryError) as caught:
        other._chart_bounds(Q(1,4),Q(1,3))
    assert caught.value is sentinel


def test_public_adapter_uses_issued_chart_path_and_keeps_selected_support(monkeypatch):
    from test_nurbs_boundary_certification import graph_curve
    from anygeometry.nurbs_intersection_curve import intersection_curve_operation
    curve=graph_curve()
    with intersection_curve_operation():
        shape=curve._shape()
        expected=reference(shape.enclose(Q(1,4),Q(1,3)))[curve.chart_support]
        def no_full(*args,**kwargs):
            raise AssertionError('adapter requested full derivative bounds')
        monkeypatch.setattr(type(shape),'enclose',no_full)
        assert curve.chart_box(Q(1,4),Q(1,3)) == expected
        assert all(type(value) is Q for axis in curve.chart_box(Q(1,4),Q(1,3)) for value in axis)
        public=curve.chart_box(Q(1,4),Q(1,3))
        object.__setattr__(public[0][0],'_numerator',-999)
        assert curve.chart_box(Q(1,4),Q(1,3)) == expected
