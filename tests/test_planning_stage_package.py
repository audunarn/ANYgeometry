"""Explicit verification controls use package admission without production policy."""
from _planning_numpy_profile_contract import numpy_profile_test
from _planning_platform_contract import windows_owned_test
import json
import types
import pytest
from anygeometry import _planning_host as host, _planning_session as session
from anygeometry import batch_intersections as batch
from anygeometry._planning_qualification import QualificationDecline
from anygeometry._planning_executor import PlannerCleanupError
from anygeometry._planning_values import encode_value, value_codec_contract
from anygeometry._planning_stages import dispatch


def control(tmp_path):
    result=host.StageControl(tmp_path,workers=2,expected_ci=('plan','apply'),observed_ci=lambda _:('plan','apply'))
    result.package_stages=True
    result.numerical_manifest=dict(value_codec=value_codec_contract(),common='a'*64,constructor='b'*64,
        host='c'*64,infrastructure_host='d'*64,resource={})
    return result


@pytest.mark.parametrize('failure',[None,'reply','finish','cleanup'])
def test_package_stage_sessions_retire_before_samples(tmp_path,monkeypatch,failure):
    ctl=control(tmp_path);events=[]
    tasks=(dict(kind='pair',first=1,second=2,tolerance=1e-6,face_tolerances=[1e-6,1e-6],cost=1.),)
    class Fake:
        def __init__(self,path,**kwargs):
            events.append(('open',path.name));assert kwargs['workers']==2
            assert kwargs['numerical_manifest'] is ctl.numerical_manifest
        def submit_stage(self,kind,packet,transported,**kwargs):
            assert transported is not tasks and transported==tasks
            events.append(('reply',kind))
            if failure=='reply':raise QualificationDecline('reply refused')
            return dict(results=[json.dumps(encode_value((),codec='v2'),separators=(',',':'))],task_layout=dispatch(tasks,2)[1],sampling_sha256='e'*64)
        def finish(self,**kwargs):
            events.append(('finish',None))
            if failure=='finish':raise QualificationDecline('finish refused')
        def close(self):
            events.append(('close',None))
            if failure=='cleanup':raise OSError('cleanup failed')
    monkeypatch.setattr(session,'OwnedSession',Fake)
    bridge=host._PackageStageBridge({},ctl)
    def sample(task):
        assert events[-1]==('close',None);events.append(('sample',None));return ()
    if failure:
        with pytest.raises(PlannerCleanupError if failure=='cleanup' else QualificationDecline):bridge.run('pair',tasks,sample)
        assert not ctl.reports and ('sample',None) not in events
    else:
        assert bridge.run('pair',tasks,sample)==((),)
        assert not getattr(ctl,'production',False) and ctl.reuse_workers is False
        assert bridge.session is None and bridge.retired
        assert ctl.reports[0]['stage']=='pair'
        assert bridge.run('face',tasks,sample)==((),)
        assert [item for item in events if item[0]=='open']==[('open','pair'),('open','face')]
    assert events.count(('close',None))==(2 if failure is None else 1)


@pytest.mark.parametrize('marker',[None,{},False])
def test_package_stage_missing_marker_never_downgrades(tmp_path,monkeypatch,marker):
    ctl=control(tmp_path);ctl.numerical_manifest=marker
    monkeypatch.setattr(session,'OwnedSession',lambda *a,**kw:pytest.fail('unadmitted session'))
    with pytest.raises(QualificationDecline):host._PackageStageBridge({},ctl).run('pair',({'kind':'pair'},),lambda _:())


def descriptor(monkeypatch):
    from anygeometry._planning_canaries import _fixture
    model,_=_fixture();saved={}
    def capture(*args,**kwargs):saved.update(kwargs);return None
    monkeypatch.setattr(host,'qualified_plan',capture)
    monkeypatch.setenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING','1')
    batch.plan_intersections(model,tuple(model.faces),policy='connect',_parallel_control=object())
    return model,saved['policy'],saved['preparation']


def test_public_package_descriptor_is_source_bound_and_not_executed(monkeypatch):
    model,policy,prepared=descriptor(monkeypatch)
    result=host._stage_descriptor(prepared,model,policy)
    assert result['operands']==[['face',identifier] for identifier in sorted(model.faces)]
    copied=types.FunctionType(prepared.__code__.replace(),prepared.__globals__,closure=prepared.__closure__)
    with pytest.raises(QualificationDecline):host._stage_descriptor(copied,model,policy)


def test_unknown_package_descriptor_never_executes(monkeypatch):
    model,policy,_=descriptor(monkeypatch)
    def foreign():pytest.fail('unknown descriptor executed')
    for value in (foreign,{},True):
        with pytest.raises(QualificationDecline):host._stage_descriptor(value,model,policy)


def test_replaced_public_descriptor_owner_never_executes_hooks(monkeypatch):
    model,policy,prepared=descriptor(monkeypatch)
    class Foreign:
        @property
        def __code__(self):pytest.fail('foreign callable metadata hook executed')
    monkeypatch.setattr(batch,'plan_intersections',Foreign())
    with pytest.raises(QualificationDecline):host._stage_descriptor(prepared,model,policy)


@windows_owned_test
@numpy_profile_test
def test_admitted_package_capability_and_manifest_skew_are_terminal(tmp_path):
    from anygeometry._planning_canaries import _fixture
    from anygeometry._planning_executor import freeze_model,PlannerSourceChanged
    from anygeometry.serialization import from_dict,to_dict
    model,_=_fixture();model=from_dict(to_dict(model));ctl=control(tmp_path)
    verify=host._admit_stage_package(model,ctl,batch.IntersectionBatchPolicy(),freeze_model(model))
    verify()
    original=ctl.numerical_manifest['common']
    ctl.numerical_manifest['common']='0'*64
    with pytest.raises(PlannerSourceChanged,match='capability'):verify()
    ctl.numerical_manifest['common']=original
    ctl.package_stages=False
    with pytest.raises(PlannerSourceChanged,match='capability'):verify()
    ctl.package_stages=True
    verify()


def test_package_ci_decline_rechecks_loaded_authority_before_replay(tmp_path,monkeypatch):
    from anygeometry._planning_canaries import _fixture
    from anygeometry._planning_executor import PlannerSourceChanged
    from anygeometry.serialization import from_dict,to_dict
    model,_=_fixture();model=from_dict(to_dict(model));ctl=control(tmp_path)
    ctl.reuse_workers=True;changed=[]
    def admit(*args):
        def verify():
            if changed:raise PlannerSourceChanged('loaded authority changed by CI')
        return verify
    monkeypatch.setattr(host,'_admit_stage_package',admit)
    def observed(candidate):changed.append(True);return ('wrong','apply')
    ctl.observed_ci=observed
    with pytest.raises(PlannerSourceChanged,match='loaded authority'):
        host.qualified_plan(model,ctl,lambda:pytest.fail('drift replayed'),lambda bridge:object(),
            policy=batch.IntersectionBatchPolicy(),mode='exact')
    assert ctl.mode=='terminal'
