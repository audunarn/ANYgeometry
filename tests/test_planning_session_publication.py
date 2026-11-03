"""Atomic initialization evidence and terminal-result precedence, without processes."""
import importlib
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
from anygeometry._planning_executor import (
    PlannerCancelled,PlannerDeadline,PlannerCleanupError,PlannerSourceChanged,
)
from anygeometry._planning_qualification import QualificationDecline


def publication(tmp_path,monkeypatch,errors,*,platform='nt',oversleep=False):
    module=importlib.import_module('anygeometry._planning_session')
    original=Path.replace;calls=[];sleeps=[];clock=[0.]
    def replace(path,target):
        if path==tmp_path/'initialization-phases.tmp':
            calls.append(1)
            if errors:raise errors.pop(0)
        return original(path,target)
    def sleep(delay):
        sleeps.append(delay);clock[0]+=1. if oversleep else delay
    monkeypatch.setattr(Path,'replace',replace)
    # A private os namespace avoids changing global pathlib/platform behavior.
    monkeypatch.setattr(module,'os',SimpleNamespace(name=platform,getpid=os.getpid))
    import anygeometry._planning_executor as executor
    monkeypatch.setattr(executor,'os',SimpleNamespace(name=platform,getpid=os.getpid))
    monkeypatch.setattr(module.time,'perf_counter',lambda:2.)
    monkeypatch.setattr(module.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(module.time,'sleep',sleep)
    return module,calls,sleeps,clock


def denied(code):
    error=PermissionError(13,'original atomic replace denial')
    if code is not None:error.winerror=code
    return error


@pytest.mark.parametrize('code',[None,5,32,33])
def test_initialization_replace_retries_only_observed_windows_denials(tmp_path,monkeypatch,code):
    module,calls,sleeps,clock=publication(tmp_path,monkeypatch,[denied(code),denied(code)])
    module._record_initialization(tmp_path,'child_prepare',1.)
    row=module._read_initialization(tmp_path,os.getpid())
    assert row['phases']=={'child_prepare':1.} and len(calls)==3
    assert sleeps==[.002,.004] and clock[0]<=.05
    assert not (tmp_path/'initialization-phases.tmp').exists()


@pytest.mark.parametrize('oversleep',[False,True])
def test_initialization_replace_permanent_denial_is_bounded_and_preserves_last_failure(tmp_path,monkeypatch,oversleep):
    failures=[denied(5) for _ in range(20)];originals=list(failures)
    module,calls,sleeps,clock=publication(tmp_path,monkeypatch,failures,oversleep=oversleep)
    with pytest.raises(PermissionError) as caught:module._record_initialization(tmp_path,'child_prepare',1.)
    assert caught.value is originals[len(calls)-1]
    assert 1<=len(calls)<=8 and sum(sleeps)<=.05
    assert (tmp_path/'initialization-phases.tmp').is_file()
    assert not (tmp_path/'initialization-phases.json').exists()
    if oversleep:assert len(calls)==1


@pytest.mark.parametrize('platform,error',[
    ('posix',denied(5)),('nt',denied(87)),('nt',OSError('other failure')),
])
def test_initialization_replace_unrelated_errors_propagate_immediately(tmp_path,monkeypatch,platform,error):
    module,calls,sleeps,_=publication(tmp_path,monkeypatch,[error],platform=platform)
    with pytest.raises(type(error)) as caught:module._record_initialization(tmp_path,'child_prepare',1.)
    assert caught.value is error and len(calls)==1 and sleeps==[]
    assert (tmp_path/'initialization-phases.tmp').is_file()


@pytest.mark.parametrize('manifest',[None,{}])
@pytest.mark.parametrize('status,expected',[
    ('deadline',PlannerDeadline),('cancelled',PlannerCancelled),
    ('cleanup_error',PlannerCleanupError),('source_changed',PlannerSourceChanged),
    ('declined',QualificationDecline),
])
def test_failed_result_status_precedes_success_only_infrastructure(manifest,status,expected):
    module=importlib.import_module('anygeometry._planning_session')
    session=object.__new__(module.OwnedSession)
    session.numerical_manifest=manifest;session._infrastructure=None
    with pytest.raises(expected,match='original terminal reason'):
        session._verify_infrastructure({'status':status,'reason':'original terminal reason'})
    assert session._infrastructure is None


def report_fixture():
    module=importlib.import_module('anygeometry._planning_session')
    session=object.__new__(module.OwnedSession)
    session.numerical_manifest={'infrastructure_host':'host','resource':'resource','value_codec':'codec'}
    session._infrastructure=None
    value=dict(host='host',planner='a'*64,planner_nodes=1,resource='resource',
        planner_reference='b'*64,worker_reference='c'*64,value_codec='codec')
    return session,{'status':'passed','infrastructure':value}


@pytest.mark.parametrize('field',['host','planner','planner_nodes','resource','planner_reference','worker_reference','value_codec','extra'])
def test_passed_result_retains_every_infrastructure_binding(field):
    session,report=report_fixture()
    report['infrastructure'][field]=None
    with pytest.raises(QualificationDecline,match='infrastructure request binding'):
        session._verify_infrastructure(report)
    assert session._infrastructure is None


def test_passed_result_pins_first_stage_and_refuses_later_drift():
    session,report=report_fixture();session._verify_infrastructure(report)
    original=dict(session._infrastructure)
    session._verify_infrastructure(report)
    report['infrastructure']['planner']='d'*64
    with pytest.raises(QualificationDecline,match='changed between stages'):
        session._verify_infrastructure(report)
    assert session._infrastructure==original
