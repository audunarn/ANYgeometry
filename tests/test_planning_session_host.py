"""Session transaction units with dummy v1 tasks and isolated admission.

These units do not prove package admission or real cleanup. Actual counterparts
are test_planning_session_cancellation_process and the public eight-worker canary
in test_planning_r8_integration, with unmodified kernels and real owned sessions.
"""
import importlib
import json

import pytest

from anygeometry import _planning_host as host
from anygeometry import _planning_executor as executor
from anygeometry import batch_intersections as bi
from anygeometry._planning_canaries import _fixture
from anygeometry._planning_values import encode_value
from anygeometry.serialization import from_dict, to_dict


def setup(tmp_path, monkeypatch, *, corrupt=False):
    module=importlib.import_module('anygeometry._planning_session')
    # Dummy task/reply fixtures exercise lifecycle ordering, not the package
    # input grammar. Leave the default package_stages=False at this unit seam.
    monkeypatch.setattr(host,'_admit_stage_package',lambda *a,**k:lambda:None)
    events=[]
    class Fake:
        def __init__(self,*args,**kw):events.append('start')
        def submit_stage(self,kind,packet,tasks,*,poll):
            events.append(kind);poll()
            value='corrupt' if corrupt and kind=='pair' else kind
            return dict(results=(json.dumps(encode_value(value),sort_keys=True,separators=(',',':')),),
                        task_layout=((0,0,kind),),request_sha256='b'*64,sampling_sha256='c'*64)
        def finish(self,*,poll):events.append('finish');poll()
        def close(self):events.append('retired')
    monkeypatch.setattr(module,'OwnedSession',Fake)
    model,_=_fixture();model=from_dict(to_dict(model))
    def observed(_):events.append('ci');return ('plan','apply')
    control=host.StageControl(tmp_path/'session',expected_ci=('plan','apply'),observed_ci=observed,
                              workers=2,reuse_workers=True)
    def serial_task(kind):
        def run(task):events.append('sample-'+kind);return kind
        return run
    def parallel(bridge):
        pair=bridge.run('pair',[dict(kind='pair',cost=1.)],serial_task('pair'))
        assert pair==(('corrupt',) if corrupt else ('pair',))
        assert not any(e.startswith('sample-') for e in events)
        face=bridge.run('face',[dict(kind='face',cost=1.)],serial_task('face'))
        assert face==('face',)
        return 'candidate'
    return model,control,events,parallel


def test_both_stages_retire_before_all_samples_and_ci(tmp_path,monkeypatch):
    model,control,events,parallel=setup(tmp_path,monkeypatch)
    result=host.qualified_plan(model,control,lambda:pytest.fail('replayed'),parallel,
                               policy=bi.IntersectionBatchPolicy(),mode='exact')
    assert result=='candidate' and control.ci_verified
    assert events==['start','pair','face','finish','retired','sample-pair','sample-face','ci']
    assert [r['stage'] for r in control.reports]==['pair','face']
    assert all(r['sampling']['plan_digest']=='c'*64 for r in control.reports)


def test_bad_pair_stops_deferred_face_sampling_and_replays_after_cleanup(tmp_path,monkeypatch):
    model,control,events,parallel=setup(tmp_path,monkeypatch,corrupt=True)
    result=host.qualified_plan(model,control,lambda:events.append('serial') or 'original',parallel,
                               policy=bi.IntersectionBatchPolicy(),mode='exact')
    assert result=='original'
    assert events==['start','pair','face','finish','retired','sample-pair','serial']
    assert control.mode=='serial' and not control.ci_verified


def test_provisional_construction_error_replays_original_after_retirement(tmp_path,monkeypatch):
    model,control,events,parallel=setup(tmp_path,monkeypatch)
    provisional=ValueError('invalid provisional value')
    original=bi.GeometryError('original serial attribution')
    def fail(bridge):parallel(bridge);raise provisional
    def serial():events.append('serial');raise original
    with pytest.raises(bi.GeometryError) as caught:
        host.qualified_plan(model,control,serial,fail,policy=bi.IntersectionBatchPolicy(),mode='exact')
    assert caught.value is original
    assert events==['start','pair','face','retired','serial']
    assert original.__context__.__cause__ is provisional


@pytest.mark.parametrize('error', [executor.PlannerCancelled('cancelled'),executor.PlannerDeadline('deadline'),
    executor.PlannerCleanupError('cleanup'),executor.PlannerSourceChanged('source'),KeyboardInterrupt()])
def test_terminal_error_never_becomes_provisional_replay(tmp_path,monkeypatch,error):
    model,control,events,parallel=setup(tmp_path,monkeypatch)
    def fail(bridge):parallel(bridge);raise error
    with pytest.raises(type(error)) as caught:
        host.qualified_plan(model,control,lambda:pytest.fail('terminal replay'),fail,
                            policy=bi.IntersectionBatchPolicy(),mode='exact')
    assert caught.value is error and events==['start','pair','face','retired']
    assert control.mode=='terminal'


@pytest.mark.parametrize('field',['max_predicates','max_candidate_pairs'])
def test_finite_budget_never_constructs_session_or_samples(tmp_path,monkeypatch,field):
    model,control,events,parallel=setup(tmp_path,monkeypatch)
    result=host.qualified_plan(model,control,lambda:'serial',parallel,
                              policy=bi.IntersectionBatchPolicy(**{field:1}),mode='exact')
    assert result=='serial' and events==[]


@pytest.mark.parametrize('error_type',[executor.PlannerDeadline,executor.PlannerCancelled,
    executor.PlannerSourceChanged,executor.PlannerCleanupError,executor.PlannerMemoryExceeded])
@pytest.mark.parametrize('prior',[False,True])
def test_serial_sample_terminal_crosses_kernel_retry_handler(tmp_path,error_type,prior):
    control=host.StageControl(tmp_path,expected_ci=('p','a'),observed_ci=lambda _:('p','a'),reuse_workers=True)
    error=error_type('original sample terminal');armed=[False];events=[]
    def poll():
        if armed[0]:raise error
    bridge=host.Bridge({},control,poll);bridge._inside_numerical_call=prior
    candidate=dict(sampling_sha256='c'*64,task_layout=((0,0,'pair'),),results=[json.dumps(encode_value(None))])
    def kernel(task):
        armed[0]=True
        try:
            try:bridge.check_deadline()
            except bi.GeometryError:events.append('swallowed');return None
        finally:armed[0]=False
    with pytest.raises(host._CallbackFailed) as caught:
        bridge.verify_candidate('pair',[{'kind':'pair'}],kernel,candidate)
    assert caught.value.error is error and events==[]
    assert bridge._inside_numerical_call is prior and control.reports==[]


@pytest.mark.parametrize('prior',['absent',False,True])
def test_serial_sample_restores_flag_after_original_exception(tmp_path,prior):
    control=host.StageControl(tmp_path,expected_ci=('p','a'),observed_ci=lambda _:('p','a'),reuse_workers=True)
    bridge=host.Bridge({},control)
    if prior!='absent':bridge._inside_numerical_call=prior
    error=RuntimeError('original sample exception')
    candidate=dict(sampling_sha256='c'*64,task_layout=((0,0,'pair'),),results=[])
    def kernel(task):
        assert bridge._inside_numerical_call is True
        raise error
    with pytest.raises(RuntimeError) as caught:bridge.verify_candidate('pair',[{}],kernel,candidate)
    assert caught.value is error
    if prior=='absent':assert not hasattr(bridge,'_inside_numerical_call')
    else:assert bridge._inside_numerical_call is prior


def test_serial_sample_releases_temporary_before_comparison(tmp_path,monkeypatch):
    import weakref
    control=host.StageControl(tmp_path,expected_ci=('p','a'),observed_ci=lambda _:('p','a'),reuse_workers=True)
    bridge=host.Bridge({},control);references=[]
    class Temporary:pass
    def kernel(task):
        assert bridge._inside_numerical_call is True
        value=Temporary();references.append(weakref.ref(value));return value
    def encode(value,**kwargs):
        assert not hasattr(bridge,'_inside_numerical_call')
        return encode_value(None)
    comparison=host.comparison_packet
    def compare(*args,**kwargs):
        assert references and references[-1]() is None
        return comparison(*args,**kwargs)
    monkeypatch.setattr(host,'encode_value',encode)
    monkeypatch.setattr(host,'comparison_packet',compare)
    candidate=dict(sampling_sha256='c'*64,task_layout=((0,0,'pair'),),results=[json.dumps(encode_value(None))])
    assert bridge.verify_candidate('pair',[{}],kernel,candidate)==(None,)
    assert not hasattr(bridge,'_inside_numerical_call')
