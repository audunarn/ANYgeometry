"""Persistent component and continuous implicit bounds; author-unrun."""
from fractions import Fraction as F
from importlib import import_module
import dataclasses
import json

import pytest

from anygeometry.nurbs import NurbsSurface
from anygeometry.tolerance import TolerancePolicy
from anygeometry.nurbs_surface_intersection import (
    NurbsIntersectionCurve, IntersectionStatus, encode_curve,decode_curve,
    query_nurbs_surface_intersection,NurbsSurfaceIntersectionIntegrityError,
    NurbsSurfaceIntersectionCancelled)


def surfaces(crosscut=False, reverse=False):
    squares = (1.,-1.,1.)
    knots = (0.,0.,0.,1.,1.,1.)
    def make(second):
        controls = tuple(tuple((float(i-1),float(j-1),
            squares[i]+squares[j]+(0.5*(i-1) if crosscut else squares[i]+squares[j]-0.25)*second)
            for j in range(3)) for i in range(3))
        return NurbsSurface(2,2,knots,knots,controls,None,(1.,0.) if reverse else None,None)
    return make(False),make(True)


def owner(first,second):
    result = query_nurbs_surface_intersection(first,second,tolerances=TolerancePolicy())
    assert result.status is IntersectionStatus.CERTIFIED
    return NurbsIntersectionCurve.from_result(result)


def test_curve_literal_roundtrip_and_one_operation_revalidation(monkeypatch):
    first,second = surfaces()
    curve = owner(first,second)
    restored = decode_curve(json.loads(json.dumps(encode_curve(curve))))
    assert restored == curve
    module = import_module("anygeometry.nurbs_surface_intersection.query")
    original = module.query_nurbs_surface_intersection
    calls = []
    def count(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(module,"query_nurbs_surface_intersection",count)
    context = restored.prepare(first=first,second=second,tolerances=TolerancePolicy())
    shape = context.shape()
    points = shape.sample(5)
    assert len(points) == 5 and len(calls) == 1
    assert shape.breakpoints[0] == 0 and shape.breakpoints[-1] == 1
    assert curve.closed and curve.seam_identity == curve.branch_anchor
    assert points[0] == pytest.approx(points[-1],abs=1e-9)
    assert shape.chart(first,F(1,5)) == pytest.approx(shape.chart(second,F(1,5)),abs=1e-9)
    assert len(calls) == 1


@pytest.mark.parametrize("reverse",(False,True))
def test_continuous_circle_bounds_contract_and_derivatives_cover_both_slicing_axes(reverse):
    first,second = surfaces(reverse=reverse)
    curve = owner(first,second)
    context = curve.prepare(first=first,second=second,tolerances=TolerancePolicy())
    shape = context.shape()
    stations = shape.breakpoints
    assert len(stations) > 2
    for a,b in zip(stations,stations[1:]):
        t = (a+b)/2
        epsilon = (b-a)/10000
        bounds = shape.enclose(t-epsilon,t+epsilon)
        assert len(bounds) == 1
        bound = bounds[0]
        assert max(h-l for l,h in zip(bound.world_lo,bound.world_hi)) < 0.01
        left,right = shape.evaluate(t-epsilon),shape.evaluate(t+epsilon)
        for k,derivative in enumerate(bound.world_derivative):
            numerical = (right[k]-left[k])/float(2*epsilon)
            assert float(derivative.lo)-1e-4 <= numerical <= float(derivative.hi)+1e-4
        assert bound.world_derivative[2].lo <= 0 <= bound.world_derivative[2].hi
        assert bound.world_second_derivative[2].lo <= 0 <= bound.world_second_derivative[2].hi
        assert bound.first_chart == bound.second_chart


def test_crosscut_has_simultaneous_support_endpoints_and_source_changes_refuse():
    first,second = surfaces(crosscut=True)
    curve = owner(first,second)
    assert not curve.closed
    context = curve.prepare(first=first,second=second,tolerances=TolerancePolicy())
    endpoints = context.evaluate_many(curve.parameter_range)
    assert all(e.first_chart == e.second_chart for e in endpoints)
    object.__setattr__(first,"controls",surfaces()[1].controls)
    with pytest.raises(NurbsSurfaceIntersectionIntegrityError):
        context.evaluate(0)


def test_forged_anchor_and_cancellation_refuse():
    first,second = surfaces()
    curve = owner(first,second)
    forged = dataclasses.replace(curve,branch_anchor=("forged",))
    with pytest.raises(NurbsSurfaceIntersectionIntegrityError):
        forged.prepare(first=first,second=second,tolerances=TolerancePolicy())
    with pytest.raises(NurbsSurfaceIntersectionCancelled):
        curve.prepare(first=first,second=second,tolerances=TolerancePolicy(),cancellation_check=lambda:True)


def test_request_iterator_owner_mutation_refuses():
    first,second = surfaces()
    curve = owner(first,second)
    context = curve.prepare(first=first,second=second,tolerances=TolerancePolicy())
    def parameters():
        object.__setattr__(curve,"branch_anchor",("changed",))
        yield F(1,2)
    with pytest.raises(NurbsSurfaceIntersectionIntegrityError):
        context.evaluate_many(parameters())


def test_prepared_shape_cannot_relabel_proof_support_or_change_parameter_binding():
    from anygeometry.nurbs_surface_intersection import SourceDefinition
    first,second = surfaces()
    curve = owner(first,second)
    context = curve.prepare(first=first,second=second,tolerances=TolerancePolicy())
    unrelated = surfaces(crosscut=True)[1]
    shape = context.shape()
    shape._sources = (SourceDefinition.capture(unrelated),shape._sources[1])
    with pytest.raises(NurbsSurfaceIntersectionIntegrityError):
        shape.chart_enclosure(unrelated,F(1,4),F(1,3))
    shape = context.shape()
    shape._start = F(1)
    with pytest.raises(NurbsSurfaceIntersectionIntegrityError):
        shape.enclose(F(1,4),F(1,3))
    shape = context.shape()
    class MutatingInteger(int):
        @property
        def numerator(self):
            shape._start = F(1)
            return 0
    with pytest.raises(NurbsSurfaceIntersectionIntegrityError):
        shape.enclose(MutatingInteger(0),F(1,3))


def test_prepared_shape_rejects_unissued_duck_context_and_wrong_constructor_support():
    from types import SimpleNamespace
    from anygeometry.nurbs_surface_intersection import PreparedCurveShape
    first,second = surfaces()
    fake = SimpleNamespace(domain=(F(0),F(1)),breakpoints=(F(0),F(1)),
        support_index=lambda surface:0,enclose=lambda *args:())
    with pytest.raises(NurbsSurfaceIntersectionIntegrityError):
        PreparedCurveShape(fake,first,second)
    curve = owner(first,second)
    context = curve.prepare(first=first,second=second,tolerances=TolerancePolicy())
    with pytest.raises(NurbsSurfaceIntersectionIntegrityError):
        PreparedCurveShape(context,surfaces(crosscut=True)[1],second)


@pytest.mark.parametrize('selected',((.1,.8),(.8,.1)))
def test_exact_nondyadic_chart_scaling_and_rational_world_analytic_jets(selected):
    from test_rational_projection import _cheap_rational
    first,second = (dataclasses.replace(s,u_range=selected) for s in _cheap_rational())
    span = F(selected[1])-F(selected[0])
    assert span != F(selected[1]-selected[0]),'fixture must expose float-before-rational subtraction'
    curve = owner(first,second)
    context = curve.prepare(first=first,second=second,tolerances=TolerancePolicy())
    full = context.shape()
    for a,b in ((F(0),F(1)),(F(4,5),F(1,5))):
        shape = full.subcurve(a,b)
        bounds = shape.enclose(F(1,3),F(1,3))
        assert len(bounds) == 1
        bound = bounds[0]
        derivative = bound.first_chart_derivative[0]
        assert derivative.lo == derivative.hi
        assert abs(derivative.lo) == abs(b-a)
        assert bound.first_native[0][0] == bound.first_native[0][1]
        u = bound.first_native[0][0]
        denominator = 1+2*u-2*u*u
        # Literal first support: x(u)=(2u-1)/(1+2u-2u^2), y=0,z=0.
        x_prime = 4*(u*u-u+1)/denominator**2
        x_second = 4*(2*u-1)*(2*u*u-2*u+5)/denominator**3
        assert x_second != 0
        native_rate = derivative.lo*span
        expected_first,expected_second = x_prime*native_rate,x_second*native_rate*native_rate
        assert bound.world_derivative[0].lo <= expected_first <= bound.world_derivative[0].hi
        assert bound.world_second_derivative[0].lo <= expected_second <= bound.world_second_derivative[0].hi


@pytest.mark.parametrize('selected',((.1,.8),(.8,.1)))
def test_advisory_first_order_chart_witness_matches_rational_implicit_derivative(selected):
    import numpy as np
    from test_rational_projection import _cheap_rational
    from anygeometry.nurbs_intersection_curve import NurbsSurfaceIntersectionCurve,intersection_curve_operation
    from anygeometry.intersection_chart import IntersectionChart
    first,second = _cheap_rational()
    # Same separable rational projection; F=x(u)+y(v), so v=(1-x(u))/2.
    controls = tuple(tuple((p[0],p[1],p[0]+p[1]) for p in row) for row in second.controls)
    first = dataclasses.replace(first,u_range=selected)
    second = dataclasses.replace(second,u_range=selected,controls=controls)
    curve = NurbsSurfaceIntersectionCurve(owner(first,second))
    with intersection_curve_operation():
        for a,b in ((F(0),F(1)),(F(4,5),F(1,5))):
            edge = curve.subcurve(a,b)
            chart = IntersectionChart(edge)
            stations = np.asarray([.25,.6,.8])
            q = chart.evaluate(stations)
            derivative = chart.derivative_witness(stations)
            ends = chart.evaluate(np.asarray([0.,1.]))
            width = float(F(selected[1])-F(selected[0]))
            native_rate = width*(ends[1,0]-ends[0,0])
            u = selected[0]+width*q[:,0]
            denominator = 1+2*u-2*u*u
            x_prime = 4*(u*u-u+1)/denominator**2
            expected = np.column_stack((np.full(len(stations),native_rate/width),-x_prime*native_rate/2))
            np.testing.assert_allclose(derivative,expected,rtol=1e-10,atol=1e-10)


def test_first_order_witness_is_not_used_by_certified_enclosures_or_chart_proofs(monkeypatch):
    from anygeometry.nurbs_intersection_curve import NurbsSurfaceIntersectionCurve,intersection_curve_operation
    from anygeometry.intersection_chart import IntersectionChart
    from anygeometry.nurbs_surface_intersection._curve_witness import FirstOrderChartWitness
    first,second = surfaces()
    curve = NurbsSurfaceIntersectionCurve(owner(first,second))
    def forbidden(*args,**kwargs):
        raise AssertionError('advisory derivative witness entered a proof-bearing path')
    monkeypatch.setattr(FirstOrderChartWitness,'evaluate',forbidden)
    with intersection_curve_operation():
        assert curve.enclose(F(1,5),F(21,100))
        assert curve.chart_derivative_bounds(F(1,5),F(21,100))
        assert len(curve.bounds(F(1,5),F(21,100))) == 2
        chart = IntersectionChart(curve)
        assert len(chart.derivative(.2)) == 2
        points,bound = chart.certified_polyline(.1,max_pieces=256)
        assert len(points) >= 3 and bound <= .1


def test_advisory_witness_fallback_never_swallows_cancellation_integrity_or_budget(monkeypatch):
    import numpy as np
    from anygeometry.nurbs_intersection_curve import NurbsSurfaceIntersectionCurve,intersection_curve_operation
    from anygeometry.intersection_chart import IntersectionChart
    from anygeometry.nurbs_surface_intersection._curve_witness import FirstOrderWitnessNotQualified
    from anygeometry.nurbs_surface_intersection import NurbsSurfaceIntersectionRefused
    first,second = surfaces()
    curve = NurbsSurfaceIntersectionCurve(owner(first,second))
    chart = IntersectionChart(curve)
    with intersection_curve_operation():
        expected = chart.derivative(np.asarray([.2,.3]))
        def mathematical_refusal(*args,**kwargs):
            raise FirstOrderWitnessNotQualified('minor not separated')
        monkeypatch.setattr(NurbsSurfaceIntersectionCurve,'chart_derivative_witness',mathematical_refusal)
        np.testing.assert_array_equal(chart.derivative_witness(np.asarray([.2,.3])),expected)
        for error in (NurbsSurfaceIntersectionCancelled('cancelled'),
                      NurbsSurfaceIntersectionIntegrityError('source changed'),
                      NurbsSurfaceIntersectionRefused('work budget exhausted')):
            def refuse(*args,**kwargs):
                raise error
            monkeypatch.setattr(NurbsSurfaceIntersectionCurve,'chart_derivative_witness',refuse)
            with pytest.raises(type(error),match=str(error)):
                chart.derivative_witness(np.asarray([.2,.3]))


def test_array_pcurve_uses_one_certified_batch_and_derivative_reuses_its_points(monkeypatch):
    import numpy as np
    from anygeometry.nurbs_intersection_curve import NurbsSurfaceIntersectionCurve,intersection_curve_operation
    module = import_module('anygeometry.nurbs_surface_intersection._structural_evaluate')
    original = module._evaluate_validated_structural
    calls = []
    def count(result,requests,**kwargs):
        requests = tuple(requests)
        calls.append(len(requests))
        return original(result,requests,**kwargs)
    first,second = surfaces(reverse=True)
    curve = NurbsSurfaceIntersectionCurve(owner(first,second))
    monkeypatch.setattr(module,'_evaluate_validated_structural',count)
    stations = np.asarray([.1,.2,.4,.7])
    with intersection_curve_operation():
        q = curve.chart(stations)
        assert calls == [len(stations)]
        np.testing.assert_allclose((2*q[:,0]-1)**2+(2*q[:,1]-1)**2,.25,atol=2e-9,rtol=0)
        curve.chart_derivative_witness(stations)
        assert calls == [len(stations)]
        other = curve.for_support(second).chart(stations)
        np.testing.assert_array_equal(q,other)
        assert calls == [len(stations)]


@pytest.mark.parametrize('change',('shape','external_support','cancel'))
def test_pcurve_batch_checks_iterator_mutation_and_last_callback_before_issuance(change):
    first,second = surfaces()
    cancelled = [False]
    callback = lambda:cancelled[0]
    curve = owner(first,second)
    context = curve.prepare(first=first,second=second,tolerances=TolerancePolicy(),cancellation_check=callback)
    shape = context.shape()
    external = curve.first_source.rebuild()
    def parameters():
        yield F(1,5)
        if change == 'shape':
            shape._end = shape._end-F(1,10)
        elif change == 'external_support':
            # A caller support is separate from the context's actual inputs.
            # Even replacement with the OTHER canonical support must refuse.
            for field,value in zip(('degree_u','degree_v','knots_u','knots_v','controls','weights','u_range','v_range'),
                                   curve.second_source.values):
                object.__setattr__(external,field,value)
        else:
            cancelled[0] = True
    error = NurbsSurfaceIntersectionCancelled if change == 'cancel' else NurbsSurfaceIntersectionIntegrityError
    with pytest.raises(error):
        shape.chart_many(external,parameters())


def test_pcurve_batch_rechecks_external_support_after_final_evaluator_callback(monkeypatch):
    from anygeometry.nurbs_surface_intersection import SourceDefinition
    module = import_module('anygeometry.nurbs_surface_intersection._structural_evaluate')
    original = module._evaluate_validated_structural
    armed = [False]
    def arm_after_points(*args,**kwargs):
        points = original(*args,**kwargs)
        armed[0] = True
        return points
    monkeypatch.setattr(module,'_evaluate_validated_structural',arm_after_points)
    first,second = surfaces()
    external = SourceDefinition.capture(first).rebuild()
    replacement = SourceDefinition.capture(second)
    def callback():
        if armed[0]:
            armed[0] = False
            for field,value in zip(('degree_u','degree_v','knots_u','knots_v','controls','weights','u_range','v_range'),
                                   replacement.values):
                object.__setattr__(external,field,value)
        return False
    curve = owner(first,second)
    shape = curve.prepare(first=first,second=second,tolerances=TolerancePolicy(),cancellation_check=callback).shape()
    with pytest.raises(NurbsSurfaceIntersectionIntegrityError):
        shape.chart_many(external,(F(1,5),F(3,10)))


def test_generic_seam_curve_owner_and_continuous_derivatives():
    from test_generic import seam_planes
    from types import SimpleNamespace
    first,second = seam_planes()
    limits = SimpleNamespace(max_boxes_per_pair=256,max_subdivision_depth=24)
    result = query_nurbs_surface_intersection(first,second,tolerances=TolerancePolicy(),qualification=limits)
    assert result.status is IntersectionStatus.CERTIFIED
    assert result.algorithm == "generic-atlas"
    curve = decode_curve(json.loads(json.dumps(encode_curve(NurbsIntersectionCurve.from_result(result)))))
    context = curve.prepare(first=first,second=second,tolerances=TolerancePolicy(),qualification=limits)
    shape = context.shape()
    for t in (F(1,5),F(4,5)):
        point = shape.evaluate(t)
        assert abs(point[1]) <= 1e-9 and abs(point[2]) <= 1e-9
        bounds = shape.enclose(t-F(1,10000),t+F(1,10000))
        assert bounds
        assert max(max(h-l for l,h in zip(b.world_lo,b.world_hi)) for b in bounds) < 0.01
        assert all(b.world_derivative[1].lo <= 0 <= b.world_derivative[1].hi for b in bounds)


def test_rational_owner_uses_exact_native_correspondence():
    from test_rational_projection import _cheap_rational
    first,second = _cheap_rational()
    result = query_nurbs_surface_intersection(first,second,tolerances=TolerancePolicy())
    assert result.status is IntersectionStatus.CERTIFIED
    assert result.algorithm == "shared-separable-rational-projection-scalar"
    curve = NurbsIntersectionCurve.from_result(result)
    context = curve.prepare(first=first,second=second,tolerances=TolerancePolicy())
    shape = context.shape()
    assert shape.chart(first,F(1,2)) == pytest.approx(shape.chart(second,F(1,2)),abs=1e-9)
    bounds = shape.enclose(F(499,1000),F(501,1000))
    assert bounds and all(b.first_chart == b.second_chart for b in bounds)


@pytest.mark.parametrize("mode",("inward","opposed","tangent","nonroot"))
def test_simultaneous_physical_endpoint_requires_exact_root_and_common_inward_tangent(mode):
    from types import SimpleNamespace
    from anygeometry.nurbs_surface_intersection._continuation import Graphs
    from anygeometry.nurbs_surface_intersection._records import SectionRoot
    from anygeometry.nurbs_surface_intersection._intervals import Interval as I
    graph = object.__new__(Graphs)
    graph.first = graph.second = SimpleNamespace(ranges=((F(-1),F(1)),)*2)
    graph.knots = (((F(-1),F(1)),)*4,)
    graph.charge = lambda:None
    region = ((F(1,2),)*2,(F(1),)*2,(F(1,2),)*2,(F(1),)*2)
    root = SectionRoot(0,1,F(1),region)
    # F=(u-s,v-t,u): exact one-dimensional root, tangent along v=t.
    jac = ((I(1),I(0),I(-1),I(0)),
           (I(0),I(1),I(0),I(1 if mode == "opposed" else -1)),
           (I(1),I(0),I(0),I(0)))
    if mode == "tangent":
        jac = (jac[0],(I(0),I(0),I(0),I(1)),jac[2])
    pair = SimpleNamespace(enclose=lambda box:box,
        value_and_jacobian=lambda box:((I(1 if mode == "nonroot" else 0),I(0),I(0)),jac))
    graph.pair = lambda wi:pair
    assert graph.physical(root) is (mode == "inward")
