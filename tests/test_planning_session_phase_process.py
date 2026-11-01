"""Finite one-stage sessions, empty assignments, and public no-work planning."""
import json
import pytest
from anygeometry import batch_intersections as bi
from anygeometry._planning_canaries import _fixture
from anygeometry._planning_executor import freeze_model,read_packet
from anygeometry._planning_session import OwnedSession
from anygeometry._planning_host import StageControl
from anygeometry._planning_values import encode_value
from anygeometry.serialization import from_dict,to_dict


@pytest.mark.parametrize('kind',['pair','face'])
def test_one_stage_finishes_with_empty_second_worker(tmp_path,kind):
    original,_=_fixture();model=from_dict(to_dict(original));packet=freeze_model(model);captured={}
    class Capture:
        def check_deadline(self):pass
        def run(self,stage,tasks,serial_task):
            values=tuple(serial_task(task) for task in tasks)
            captured[stage]=(tasks,[json.dumps(encode_value(value),sort_keys=True,separators=(',',':'),allow_nan=False) for value in values])
            return values
    bi._plan_intersections(model,tuple(model.faces),policy='connect',_stages=Capture())
    task=captured[kind][0][0];owned=OwnedSession(tmp_path/kind,workers=2,timeout=70)
    try:
        report=owned.submit_stage(kind,packet,[task]);owned.finish()
    finally:owned.close()
    assert report['results']==(captured[kind][1][0],)
    assert [r['task_count'] for r in report['worker_reports']]==[1,0]
    assert [len(r['boundary_seconds']) for r in report['worker_reports']]==[4,3]
    assert all(r['gate_closed'] and not r['gate_poisoned'] for r in report['worker_reports'])
    final=read_packet(owned.directory/'result.json','result')
    assert final['stages']==[kind] and all(r['gate_closures']==1 for r in final['worker_reports'])
    for name in ('cleanup.json','workers-cleanup.json'):
        cleanup=json.loads((owned.directory/name).read_text())
        assert cleanup['verified'] and cleanup['active_processes']==0


def test_public_empty_plan_creates_no_owned_session(tmp_path,monkeypatch):
    original,_=_fixture();document=to_dict(original);source=from_dict(document)
    monkeypatch.delenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING',raising=False)
    serial=bi.plan_intersections(source,(next(iter(source.faces)),),policy='connect')
    bi.apply_intersections(source,serial,policy='connect')
    expected=(serial.content_checksum,to_dict(source)['checksum']['value'])
    def observed(plan):
        target=from_dict(document);bi.apply_intersections(target,plan,policy='connect')
        return plan.content_checksum,to_dict(target)['checksum']['value']
    source=from_dict(document)
    control=StageControl(tmp_path/'empty',workers=2,reuse_workers=True,
        expected_ci=expected,observed_ci=observed)
    monkeypatch.setenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING','1')
    plan=bi.plan_intersections(source,(next(iter(source.faces)),),policy='connect',_parallel_control=control)
    assert plan.content_checksum==serial.content_checksum and control.ci_verified
    assert control.reports==[] and not control.directory.exists()
    assert to_dict(source)==document
