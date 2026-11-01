"""Transcript replay must reconstruct legacy task and sample semantics exactly."""
import copy
import json
import pytest
from anygeometry import _planning_host as host
from anygeometry._planning_executor import freeze_model, source_files
from anygeometry._planning_canaries import _fixture
from anygeometry._planning_session import sampling_digest
from anygeometry._planning_admission import stratified_samples
from anygeometry._planning_stages import dispatch
from anygeometry._planning_values import encode_value
from anygeometry._planning_qualification import QualificationDecline


def prepared(tmp_path):
    model,_=_fixture();packet=freeze_model(model);sources=source_files()
    control=host.StageControl(tmp_path,expected_ci=('p','a'),observed_ci=lambda _:('p','a'),workers=2,reuse_workers=True)
    entries=[]
    for kind in ('pair','face'):
        tasks=[dict(kind=kind,cost=1.,support='warm' if kind=='face' else 'cold',zero=0.)]
        _,layout=dispatch(tasks,2)
        result=json.dumps(encode_value(kind),sort_keys=True,separators=(',',':'))
        seed=sampling_digest(packet,tasks,workers=2,batch_size=None,sources=sources)
        entries.append(dict(kind=kind,tasks=tasks,candidate=dict(results=[result],task_layout=layout,
            sampling_sha256=seed,preparation_samples=stratified_samples(layout,seed),request_sha256='a'*64)))
    return packet,control,sources,entries


def replay(tmp_path):
    packet,control,sources,entries=prepared(tmp_path)
    return host._TranscriptBridge(packet,control,entries,sources=sources),control,entries


def test_replay_pair_reference_runs_before_host_face_task_construction(tmp_path):
    bridge,control,entries=replay(tmp_path);state=[]
    def pair(task):state.append('warm');return 'pair'
    assert bridge.run('pair',entries[0]['tasks'],pair)==('pair',)
    assert state==['warm']
    # The caller builds its face task only after verified pair values return.
    actual=[dict(kind='face',cost=1.,support=state[0],zero=0.)]
    assert bridge.run('face',actual,lambda task:'face')==('face',)
    bridge.finish()
    assert [r['stage'] for r in control.reports]==['pair','face']


@pytest.mark.parametrize('change',['task','signed_zero','layout','layout_type','seed','missing_result','extra_result','missing_samples','wrong_samples'])
def test_transcript_tamper_refuses_before_any_host_reference(tmp_path,change):
    packet,control,sources,entries=prepared(tmp_path);actual=copy.deepcopy(entries[0]['tasks'])
    first=entries[0]
    if change=='task':first['tasks'][0]['support']='forged'
    elif change=='signed_zero':first['tasks'][0]['zero']=-0.
    elif change=='layout':first['candidate']['task_layout']=[[1,0,'pair']]
    elif change=='layout_type':first['candidate']['task_layout']=[[False,0,'pair']]
    elif change=='seed':first['candidate']['sampling_sha256']='0'*64
    elif change=='missing_result':first['candidate']['results']=[]
    elif change=='extra_result':first['candidate']['results']*=2
    elif change=='missing_samples':del first['candidate']['preparation_samples']
    else:first['candidate']['preparation_samples']['indices']=[]
    bridge=host._TranscriptBridge(packet,control,entries,sources=sources)
    with pytest.raises(QualificationDecline):
        bridge.run('pair',actual,lambda task:pytest.fail('unbound reference executed'))
    assert control.reports==[]


def test_transcript_reordered_stage_refuses(tmp_path):
    bridge,_,entries=replay(tmp_path)
    with pytest.raises(QualificationDecline):bridge.run('face',entries[1]['tasks'],lambda _:pytest.fail('sampled'))


def test_transcript_extra_stage_refuses_finish(tmp_path):
    bridge,_,entries=replay(tmp_path)
    bridge.run('pair',entries[0]['tasks'],lambda _:'pair')
    with pytest.raises(QualificationDecline):bridge.finish()


def test_transcript_missing_stage_refuses(tmp_path):
    packet,control,sources,entries=prepared(tmp_path)
    bridge=host._TranscriptBridge(packet,control,entries[:1],sources=sources)
    bridge.run('pair',entries[0]['tasks'],lambda _:'pair')
    with pytest.raises(QualificationDecline):bridge.run('face',entries[1]['tasks'],lambda _:pytest.fail('sampled'))


@pytest.mark.parametrize('collector',['collect_runtime_diagnostics','collect_runtime_records'])
def test_active_collectors_decline_before_probe_or_preparation(tmp_path,collector):
    from anygeometry import runtime_diagnostics as diagnostics
    from anygeometry.batch_intersections import IntersectionBatchPolicy
    model,_=_fixture();_,control,_,_=prepared(tmp_path)
    with getattr(diagnostics,collector)():
        result=host.qualified_plan(model,control,lambda:'serial',lambda _:pytest.fail('probe executed'),
            policy=IntersectionBatchPolicy(),mode='exact',preparation=lambda:pytest.fail('descriptor called'))
    assert result=='serial' and control.mode=='serial'


def test_preparation_rejects_unsupported_handles_before_owned_start(tmp_path,monkeypatch):
    from anygeometry import _planning_session as session
    monkeypatch.setattr(session,'OwnedSession',lambda *a,**k:pytest.fail('process started'))
    packet,control,_,_=prepared(tmp_path)
    spec=dict(operands=[['vertex',1]],seam='baseline',policy=dict(intent='connect',face_connections=True,
        member_connections=True,member_face_connections=True,planning_mode=None,seam_policy=None))
    with pytest.raises(QualificationDecline):
        bridge=host._PreparingBridge(packet,control,spec,poll=lambda:None)
        bridge.run('pair',[dict(kind='pair',cost=1.)],lambda _:pytest.fail('host sample'))


@pytest.mark.parametrize('sampled',[True,False])
def test_malformed_transcript_json_declines_before_host_samples(tmp_path,sampled):
    packet,control,sources,entries=prepared(tmp_path)
    tasks=[dict(kind='pair',cost=1.,ordinal=i) for i in range(6)]
    _,layout=dispatch(tasks,control.workers)
    seed=sampling_digest(packet,tasks,workers=control.workers,batch_size=None,sources=sources)
    sample=stratified_samples(layout,seed)
    index=sample['indices'][0] if sampled else next(i for i in range(6) if i not in sample['indices'])
    candidate=dict(results=[json.dumps(encode_value('pair'),separators=(',',':'))]*6,task_layout=layout,
        sampling_sha256=seed,preparation_samples=sample,request_sha256='b'*64)
    candidate['results'][index]='not valid JSON'
    bridge=host._TranscriptBridge(packet,control,[dict(kind='pair',tasks=tasks,candidate=candidate)],sources=sources)
    with pytest.raises(QualificationDecline,match='JSON'):
        bridge.run('pair',tasks,lambda _:pytest.fail('malformed input reached host reference'))


def test_mismatch_artifact_io_failure_remains_a_decline(tmp_path,monkeypatch):
    from pathlib import Path
    bridge,control,entries=replay(tmp_path)
    entries[0]['candidate']['results'][0]=json.dumps(encode_value('corrupt'),separators=(',',':'))
    original=Path.write_text;failure=OSError('diagnostic storage failed')
    def fail_artifact(path,*args,**kwargs):
        if path.name=='sample-mismatch-pair.json':raise failure
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'write_text',fail_artifact)
    with pytest.raises(QualificationDecline) as caught:
        bridge.run('pair',entries[0]['tasks'],lambda _:'pair')
    assert caught.value.__cause__ is failure
    assert control.reports==[]
