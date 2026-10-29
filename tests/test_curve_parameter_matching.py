"""Public conservative parameter covers, not finite-sample inverse claims."""
from dataclasses import dataclass, replace
from fractions import Fraction as Q
import math
import pytest

from anygeometry.curve_parameter_matching import (
    match_curve_parameter_regions, CurveParameterMatchError,
    CurveParameterMatchInconclusive,
)


@dataclass(frozen=True)
class Linear:
    domain: tuple = (0.,1.)
    def subcurve(self,a,b):return replace(self,domain=(a,b))
    def chart_bounds(self):return ((self.domain[0],0.),(self.domain[1],0.))
    def chart_enclosure(self,t):return ((Q(t),Q(t)),(Q(0),Q(0)))


@dataclass(frozen=True)
class Fold(Linear):
    def chart_bounds(self):
        a,b=map(Q,self.domain); values=[abs(2*a-1),abs(2*b-1)]
        if a<=Q(1,2)<=b:values.append(Q(0))
        return ((min(values),Q(0)),(max(values),Q(0)))
    def chart_enclosure(self,t):
        x=abs(2*Q(t)-1);return ((x,x),(Q(0),Q(0)))


@dataclass(frozen=True)
class Tangent(Linear):
    def chart_bounds(self):
        a,b=map(Q,self.domain); values=[(a-Q(1,2))**2,(b-Q(1,2))**2]
        if a<=Q(1,2)<=b:values.append(Q(0))
        return ((a,min(values)),(b,max(values)))
    def chart_enclosure(self,t):
        x=Q(t);y=(x-Q(1,2))**2;return ((x,x),(y,y))


def covered(result,t):return any(r.lower<=t<=r.upper for r in result.regions)


def test_linear_complete_cover_exact_bounds_and_immutable_result():
    result=match_curve_parameter_regions(Linear(),((Q(1,2),0),(Q(1,2),0)),Q(1,8))
    assert len(result.regions)==len(result.candidates)==1
    assert covered(result,Q(3,8)) and covered(result,Q(5,8))
    assert result.regions[0].lower<=Q(3,8)<=Q(5,8)<=result.regions[0].upper
    assert Q(result.regions[0].lower)>=Q(3,8)-Q(1e-12)
    assert Q(result.regions[0].upper)<=Q(5,8)+Q(1e-12)
    assert all(c.residual_squared<=Q(1,64) for c in result.candidates)
    assert isinstance(result.candidates[0].residual_squared,Q)
    assert result.work_used<=10000
    with pytest.raises(AttributeError):result.regions[0].lower=0


def test_fold_keeps_two_regions_and_zero_one_seam_separate():
    result=match_curve_parameter_regions(Fold(),((Q(1,2),0),(Q(1,2),0)),Q(1,64))
    assert len(result.regions)==2 and covered(result,.25) and covered(result,.75)
    assert result.regions[0].upper<result.regions[1].lower
    seam=match_curve_parameter_regions(Fold(),((1,0),(1,0)),Q(1,64))
    assert len(seam.regions)==2 and covered(seam,0) and covered(seam,1)


def test_tangent_possible_point_never_discarded():
    result=match_curve_parameter_regions(Tangent(),((Q(1,2),Q(-1,4)),(Q(1,2),Q(-1,4))),Q(1,4),param_resolution=1e-8)
    assert covered(result,.5)
    assert result.candidates[0].parameter==.5
    assert result.candidates[0].residual_squared==Q(1,16)


def test_possible_without_robust_representative_is_inconclusive_not_empty():
    with pytest.raises(CurveParameterMatchInconclusive) as error:
        match_curve_parameter_regions(Linear(),((.4,-.1),(.6,.1)),.01,param_resolution=.01)
    assert error.value.reason=='inconclusive'


def test_empty_requires_proved_outside_and_one_query():
    result=match_curve_parameter_regions(Linear(),((3,0),(3,0)),.1)
    assert result.regions==result.candidates==() and result.work_used==3


@pytest.mark.parametrize('settings,reason',[
    ({'max_boxes':1},'budget_exhausted'),
    ({'max_depth':0},'depth_exhausted'),
    ({'max_depth':100,'param_resolution':1e-20,'uv_tolerance':2**-54},'precision_exhausted'),
])
def test_exhaustion_never_partial(settings,reason):
    settings=dict(settings);tol=settings.pop('uv_tolerance',.01)
    with pytest.raises(CurveParameterMatchError) as error:
        match_curve_parameter_regions(Linear(),((1,0),(1,0)),tol,**settings)
    assert error.value.reason==reason


def test_checkpoint_exception_identity_and_query_budget():
    class Stop(BaseException):pass
    stop=Stop();units=[]
    def checkpoint():
        units.append(1)
        if len(units)==7:raise stop
    with pytest.raises(Stop) as error:
        match_curve_parameter_regions(Linear(),((.5,0),(.5,0)),.01,checkpoint=checkpoint)
    assert error.value is stop and len(units)==7


@pytest.mark.parametrize('settings',[
    {'uv_tolerance':0},{'uv_tolerance':-1},{'uv_tolerance':math.inf},
    {'uv_tolerance':True},{'max_boxes':0},{'max_boxes':1.5},{'max_boxes':True},
    {'max_depth':-1},{'max_depth':True},{'param_resolution':0},
    {'param_resolution':math.nan},{'checkpoint':1},
    {'target_box':((1,0),(0,0))},{'target_box':((0,math.nan),(0,1))},
    {'target_box':((0,),(1,))},
])
def test_bad_arguments(settings):
    arguments=dict(target_box=((.5,0),(.5,0)),uv_tolerance=.01);arguments.update(settings)
    with pytest.raises(CurveParameterMatchError) as error:
        match_curve_parameter_regions(Linear(),**arguments)
    assert error.value.reason=='invalid_input'


def test_whole_master_guard():
    with pytest.raises(CurveParameterMatchError,match='whole curve domain'):
        match_curve_parameter_regions(Linear((.2,.8)),((.5,0),(.5,0)),.01)


def test_query_count_bounded_and_failure_charged_before_query():
    calls=[]
    @dataclass(frozen=True)
    class Counted(Linear):
        def chart_bounds(self):calls.append('interval');return super().chart_bounds()
        def chart_enclosure(self,t):calls.append('point');return super().chart_enclosure(t)
    units=[]
    result=match_curve_parameter_regions(Counted(),((.5,0),(.5,0)),.01,checkpoint=lambda:units.append(1))
    assert len(calls)<=result.work_used==len(units)<=10000
    calls.clear()
    with pytest.raises(CurveParameterMatchError) as error:
        match_curve_parameter_regions(Counted(),((.5,0),(.5,0)),.01,max_boxes=1)
    assert error.value.reason=='budget_exhausted' and not calls
    with pytest.raises(CurveParameterMatchError) as error:
        match_curve_parameter_regions(Counted(),((.5,0),(.5,0)),.01,max_boxes=result.work_used-1)
    assert error.value.reason=='budget_exhausted'


def test_inconclusive_second_component_cannot_return_first_partial_success():
    @dataclass(frozen=True)
    class LooseRight(Fold):
        def chart_enclosure(self,t):
            return ((0,1),(0,1)) if t>.5 else super().chart_enclosure(t)
    with pytest.raises(CurveParameterMatchInconclusive):
        match_curve_parameter_regions(LooseRight(),((.5,0),(.5,0)),.01)


def test_malformed_public_enclosure_is_typed():
    @dataclass(frozen=True)
    class Invalid(Linear):
        def chart_bounds(self):return ((0,math.nan),(1,0))
    with pytest.raises(CurveParameterMatchError) as error:
        match_curve_parameter_regions(Invalid(),((.5,0),(.5,0)),.01)
    assert error.value.reason=='invalid_enclosure'
    @dataclass(frozen=True)
    class InvalidPoint(Linear):
        def chart_enclosure(self,t):return ((1,0),(0,0))
    with pytest.raises(CurveParameterMatchError) as error:
        match_curve_parameter_regions(InvalidPoint(),((.5,0),(.5,0)),.01)
    assert error.value.reason=='invalid_enclosure'


def test_numpy_real_input_and_exact_all_target_box_residual():
    import numpy as np
    result=match_curve_parameter_regions(Linear(),
        (np.array([.49,-.001],dtype=np.float32),np.array([.51,.001],dtype=np.float32)),np.float64(.02),param_resolution=.01)
    assert len(result.candidates)==1
    assert result.candidates[0].residual_squared<=Q(.02)**2


@pytest.mark.parametrize('which',['interval','point'])
def test_owner_query_refusal_is_typed_without_partial_result(which):
    from anygeometry.plane_nurbs import PlaneNurbsNotCertified
    refusal=PlaneNurbsNotCertified('deliberate owner query refusal')
    @dataclass(frozen=True)
    class Refusing(Linear):
        def chart_bounds(self):
            if which=='interval':raise refusal
            return super().chart_bounds()
        def chart_enclosure(self,t):
            if which=='point':raise refusal
            return super().chart_enclosure(t)
    with pytest.raises(CurveParameterMatchError) as error:
        match_curve_parameter_regions(Refusing(),((.5,0),(.5,0)),.01)
    assert error.value.reason=='enclosure_refused' and error.value.__cause__ is refusal


def cooperative_curve(level=.375):
    from anygeometry.surfaces import Plane
    from anygeometry.nurbs import NurbsSurface
    from anygeometry.support_curves import PlaneNurbsCurve
    surface=NurbsSurface(1,1,(0,0,1,1),(0,0,1,1),(((0,0,0),(0,1,0)),((1,0,0),(1,1,1))))
    return PlaneNurbsCurve.from_plane(Plane((0,0,level),(1,0,0),(0,1,0)),surface)


def test_public_context_factory_queries_and_matcher_share_one_ledger():
    from anygeometry.curve_parameter_matching import bounded_section_work
    from anygeometry.plane_nurbs import CURRENT_WORK
    units=[]
    with bounded_section_work(max_work=20000,max_depth=48,checkpoint=lambda:units.append(1)) as ledger:
        curve=cooperative_curve()
        assert ledger.work_used>0
        curve.chart_bounds();curve.chart_enclosure(.4)
        uv=curve.chart(.4);before=ledger.work_used
        result=match_curve_parameter_regions(curve,(uv,uv),1e-5)
        assert result.work_used==ledger.work_used-before
        assert len(units)==ledger.work_used<=20000
    assert CURRENT_WORK.get() is None


def test_inner_query_cancellation_identity_and_tokens_reset():
    from anygeometry.curve_parameter_matching import bounded_section_work
    from anygeometry.plane_nurbs import CURRENT_WORK
    from anygeometry.errors import GeometryError
    curve=cooperative_curve();stop=GeometryError('caller stop');calls=[]
    def checkpoint():
        calls.append(1)
        if len(calls)==5:raise stop
    with pytest.raises(GeometryError) as error:
        with bounded_section_work(max_work=20000,checkpoint=checkpoint):
            match_curve_parameter_regions(curve,((.7,.6),(.7,.6)),.01)
    assert error.value is stop and len(calls)==5 and CURRENT_WORK.get() is None
    with bounded_section_work() as ledger:assert ledger.work_used==0


def test_foreign_context_refuses_and_unscoped_hooks_do_not_charge_it():
    from anygeometry.curve_parameter_matching import bounded_section_work,_cooperative_tick
    from anygeometry.plane_nurbs import CURRENT_WORK
    from anygeometry.section_audit import Work
    foreign=Work();token=CURRENT_WORK.set(foreign)
    try:
        _cooperative_tick();_cooperative_tick()
        assert foreign.boxes==foreign.subdivisions==0
        with pytest.raises(CurveParameterMatchError) as error:
            with bounded_section_work():pass
        assert error.value.reason=='nested_context' and CURRENT_WORK.get() is foreign
    finally:CURRENT_WORK.reset(token)


def test_local_subbudget_exhaustion_then_recovery_in_same_outer_ledger():
    from anygeometry.curve_parameter_matching import bounded_section_work
    from anygeometry.plane_nurbs import CURRENT_WORK
    units=[]
    with bounded_section_work(max_work=20000,checkpoint=lambda:units.append(1)) as ledger:
        outer=CURRENT_WORK.get()
        with pytest.raises(CurveParameterMatchError) as error:
            match_curve_parameter_regions(Linear(),((.5,0),(.5,0)),.01,max_boxes=1)
        assert error.value.reason=='budget_exhausted' and CURRENT_WORK.get() is outer
        consumed=ledger.work_used
        result=match_curve_parameter_regions(Linear(),((.5,0),(.5,0)),.01)
        assert result.work_used==ledger.work_used-consumed and result.candidates
        assert ledger.work_used==len(units)
    assert CURRENT_WORK.get() is None


@pytest.mark.parametrize('ranges',[((0.,1.),(0.,1.)),((2.,4.),(-3.,1.)),((4.,2.),(1.,-3.))])
def test_real_plane_nurbs_public_whole_hyperbola(ranges):
    from anygeometry.surfaces import Plane
    from anygeometry.nurbs import NurbsSurface
    from anygeometry.support_curves import PlaneNurbsCurve
    from copy import deepcopy
    ur,vr=ranges
    surface=NurbsSurface(1,1,(min(ur),min(ur),max(ur),max(ur)),(min(vr),min(vr),max(vr),max(vr)),
        (((0,0,0),(0,1,0)),((1,0,0),(1,1,1))),u_range=ur,v_range=vr)
    curve=PlaneNurbsCurve.from_plane(Plane((0,0,.5),(1,0,0),(0,1,0)),surface)
    before=deepcopy(curve);parameter=.25
    uv=tuple(curve.chart(parameter))
    result=match_curve_parameter_regions(curve,(uv,uv),1e-5)
    assert covered(result,parameter) and len(result.candidates)==1
    assert result.candidates[0].residual_squared<=Q(1e-5)**2
    low,high=result.candidates[0].chart_box
    assert all(0<=a<=b<=1 for a,b in zip(low,high))
    assert curve==before
    with pytest.raises(CurveParameterMatchError,match='whole curve domain'):
        match_curve_parameter_regions(curve.subcurve(.1,.9),(uv,uv),1e-5)
