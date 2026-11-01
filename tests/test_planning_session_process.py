"""Real finite-session tests: tiny source only, with original serial/apply controls."""
import json

from anygeometry import batch_intersections as bi
from anygeometry._planning_host import StageControl
from anygeometry._planning_canaries import _fixture
from anygeometry.serialization import from_dict, to_dict


def test_public_session_reuses_workers_with_same_samples_and_plan_apply(tmp_path,monkeypatch):
    model,_=_fixture();document=to_dict(model)
    reference=from_dict(document)
    monkeypatch.delenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING',raising=False)
    serial=bi.plan_intersections(reference,tuple(reference.faces),policy='connect')
    bi.apply_intersections(reference,serial,policy='connect')
    expected=(serial.content_checksum,to_dict(reference)['checksum']['value'])
    def observed(plan):
        candidate=from_dict(document)
        bi.apply_intersections(candidate,plan,policy='connect')
        return plan.content_checksum,to_dict(candidate)['checksum']['value']
    monkeypatch.setenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING','1')
    controls=[]
    for reuse in (False,True):
        source=from_dict(document)
        control=StageControl(tmp_path/('session' if reuse else 'legacy'),expected_ci=expected,
                             observed_ci=observed,workers=2,reuse_workers=reuse)
        plan=bi.plan_intersections(source,tuple(source.faces),policy='connect',_parallel_control=control)
        assert control.mode=='parallel',control.decline_reason
        assert control.ci_verified and plan.content_checksum==serial.content_checksum
        assert to_dict(source)==document
        controls.append(control)
    old,new=controls
    assert [r['stage'] for r in new.reports]==['pair','face']
    for first,second in zip(old.reports,new.reports):
        assert first['sampling']['indices']==second['sampling']['indices']
        assert first['request_sha256']==second['sampling_sha256']
        assert first['task_layout']==second['task_layout']
        assert all(r['full_environment_checks']==1 and len(r['boundary_seconds'])==4
                   and r['gate_closed'] and not r['gate_poisoned'] for r in second['worker_reports'])
    pair,face=new.reports
    assert [r['pid'] for r in pair['worker_reports']]==[r['pid'] for r in face['worker_reports']]
    assert all(r['model_decode_count']==1 for report in new.reports for r in report['worker_reports'])
    folder=new.directory/'session'
    for name in ('cleanup.json','workers-cleanup.json'):
        cleanup=json.loads((folder/name).read_text())
        assert cleanup['verified'] and cleanup['active_processes']==0
    from anygeometry._planning_executor import read_packet
    finish=read_packet(folder/'result.json','result')
    assert finish['status']=='passed' and finish['stages']==['pair','face']
    assert all(r['gate_closures']==2 for r in finish['worker_reports'])
