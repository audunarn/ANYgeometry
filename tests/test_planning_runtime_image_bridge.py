"""Dependency-isolated runtime guard units; never native qualification."""
import builtins
import sys
import types
from types import SimpleNamespace

import pytest
import _planning_contract_helpers as helpers
from anygeometry import _planning_qualification as qualification


def _facet(rows=None,*,required=False):
    """Actual runtime_files code, private import/sys/ctypes, no global patch."""
    provider=types.ModuleType(qualification.__name__)
    vars(provider).update(vars(qualification))
    platform=SimpleNamespace(platform='linux',modules=dict(sys.modules))
    platform.modules[qualification.__name__]=provider
    native_calls=[]
    def unexpected_native(*args,**kwargs):
        native_calls.append(True)
        raise AssertionError('Windows loader entered in POSIX guard facet')
    ctypes=SimpleNamespace(WinDLL=unexpected_native)
    def importing(name,globals=None,locals=None,fromlist=(),level=0):
        if name=='sys':return platform
        if name=='ctypes':return ctypes
        return builtins.__import__(name,globals,locals,fromlist,level)
    vars(provider)['__builtins__']=dict(vars(builtins),__import__=importing)
    actual=qualification.runtime_files
    provider.runtime_files=types.FunctionType(actual.__code__,vars(provider),actual.__name__,actual.__defaults__,actual.__closure__)
    assert provider.runtime_files.__code__ is actual.__code__
    try:provider.runtime_files()
    except qualification.QualificationDecline as error:
        assert type(error) is qualification.QualificationDecline
        reason=error.args[0]
    else:raise AssertionError('actual guard facet did not refuse')
    assert reason=='interpreter runtime adapter is Windows-only'
    observed=[]
    def capture(roots,**kwargs):
        observed.append((roots,kwargs))
        if rows is None:raise qualification.QualificationDecline(reason)
        error=qualification.QualificationDecline('callable closure refused: '+str(rows[:5]))
        error.refusals=list(rows)
        raise error
    namespace=dict(vars(helpers))
    namespace.update(sys=platform,qualification=provider,
        os=SimpleNamespace(environ={'ANYGEOMETRY_REQUIRE_QUALIFIED_PLANNING_TESTS':'1' if required else '0'}),
        manifest=SimpleNamespace(__file__=helpers.manifest.__file__,code_digest=helpers.manifest.code_digest,
                                 capture_callable_manifest=capture))
    for name,value in vars(helpers).items():
        if type(value) is types.FunctionType and value.__module__==helpers.__name__:
            cloned=types.FunctionType(value.__code__,namespace,value.__name__,value.__defaults__,value.__closure__)
            cloned.__kwdefaults__=value.__kwdefaults__
            namespace[name]=cloned
    request=SimpleNamespace(node=SimpleNamespace(user_properties=[]))
    return namespace,request,observed,native_calls


def _capture(facet,request):
    # Isolate this guard from the independently tested NumPy dispatcher boundary.
    return facet['_capture_with_runtime_image_boundary']((None,),request,'original-runtime-assertions',{})


def test_actual_guard_aggregate_is_initial_scope_not_legacy_predicate(request):
    rows=['root:None:0: interpreter runtime adapter is Windows-only',
          'anygeometry.tolerance.max: interpreter runtime adapter is Windows-only']
    facet,selected,calls,native=_facet(rows)
    assert _capture(facet,selected) is None
    assert len(calls)==1 and native==[]
    properties=dict(selected.node.user_properties)
    assert properties['postcapture_qualification'] is False
    assert properties['original_test_body_executed'] is False
    request.node.user_properties.append(('execution_scope','dependency-isolated-runtime-guard-unit'))


def _error(rows):
    error=qualification.QualificationDecline('callable closure refused: '+str(rows[:5]))
    error.refusals=list(rows)
    return error


@pytest.mark.parametrize('size',[1,5,6])
def test_all_aggregate_rows_are_checked_without_truncation(size):
    reason='interpreter runtime adapter is Windows-only'
    rows=[f'root:None:{index}: '+reason for index in range(size)]
    assert helpers._runtime_only_refusals(_error(rows))==tuple(rows)


@pytest.mark.parametrize('fault',['unknown-direct','mixed-first','hidden-sixth','nested-reason','empty','tuple',
                                  'empty-path','nonstring','message','cause','context','subclass'])
def test_unknown_mixed_or_malformed_refusals_never_classify(fault):
    reason='interpreter runtime adapter is Windows-only'
    rows=['root:'+str(i)+': '+reason for i in range(6)]
    error=_error(rows)
    if fault=='unknown-direct':error=qualification.QualificationDecline('other '+reason)
    elif fault=='mixed-first':rows[0]='root: source code changed';error=_error(rows)
    elif fault=='hidden-sixth':error.refusals[5]='root: source code changed'
    elif fault=='nested-reason':error.refusals[5]='root: source code changed: '+reason
    elif fault=='empty':error.refusals=[]
    elif fault=='tuple':error.refusals=tuple(rows)
    elif fault=='empty-path':error=_error([': '+reason])
    elif fault=='nonstring':
        class Hostile(str):
            def __repr__(self):pytest.fail('unknown row repr executed')
        error.refusals[5]=Hostile(rows[5])
    elif fault=='message':error.args=('unknown aggregate message',)
    elif fault=='cause':error.__cause__=RuntimeError('unknown source failure')
    elif fault=='context':error.__context__=RuntimeError('unknown source failure')
    elif fault=='subclass':
        class Foreign(qualification.QualificationDecline):pass
        error=Foreign(reason)
    assert helpers._runtime_only_refusals(error) is None


def test_individual_guard_and_required_positive_lane_remain_distinct():
    facet,request,calls,native=_facet()
    assert _capture(facet,request) is None
    assert dict(request.node.user_properties)['runtime_refusal_count']==1
    assert len(calls)==1 and native==[]
    strict,selected,_,native=_facet(required=True)
    with pytest.raises(pytest.fail.Exception,match='qualified planning test lane requires original assertions:'):
        _capture(strict,selected)
    assert selected.node.user_properties==[] and native==[]


def test_successful_pure_capture_is_not_preempted_by_posix_facet():
    facet,request,_,native=_facet();result=object();entered=[]
    def capture(roots,**kwargs):entered.append((roots,kwargs));return result
    facet['manifest'].capture_callable_manifest=capture
    assert _capture(facet,request) is result
    assert entered==[((None,),{})] and native==[]
    assert dict(request.node.user_properties)=={'execution_scope':'original-runtime-assertions'}


@pytest.mark.parametrize('skew',['code','namespace','filename','decline-provider'])
def test_runtime_guard_source_or_provider_skew_is_not_accepted(skew,tmp_path):
    facet,_,_,native=_facet();provider=facet['qualification'];function=provider.runtime_files
    if skew=='code':function.__code__=function.__code__.replace(co_consts=(*function.__code__.co_consts,'source skew'))
    elif skew=='namespace':provider.runtime_files=types.FunctionType(function.__code__,dict(vars(provider)))
    elif skew=='filename':provider.__file__=str(tmp_path/'_planning_qualification.py')
    else:provider.QualificationDecline=RuntimeError
    with pytest.raises(AssertionError):facet['_runtime_guard_evidence']()
    assert native==[]


@pytest.mark.parametrize('required',[False,True])
def test_unknown_actual_capture_refusal_propagates_without_scope_label(required):
    facet,request,_,native=_facet(['root: source code changed'],required=required)
    with pytest.raises(qualification.QualificationDecline,match='source code changed'):
        _capture(facet,request)
    assert request.node.user_properties==[] and native==[]


def test_real_pure_root_capture_keeps_its_actual_boundary(request):
    captured=helpers.capture_with_runtime_boundary((None,),request=request)
    if captured is None:
        assert dict(request.node.user_properties)['postcapture_qualification'] is False
        return
    captured.verify()
