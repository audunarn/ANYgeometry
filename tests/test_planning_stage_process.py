"""Real owned stage subprocesses, distinct from host/control injection tests."""
import json
from anygeometry import _planning_executor as a
from anygeometry._planning_stages import WorkerModel,stage_task
from anygeometry._planning_canaries import _fixture
from anygeometry.serialization import from_dict,to_dict


def test_owned_stage_workers_decode_once_and_match_serial(tmp_path):
    model,_=_fixture();packet=a.freeze_model(from_dict(to_dict(model)));context=WorkerModel(packet)
    faces=list(context.model.faces);tol=context.model.tolerance.effective_length(5.)
    tasks=[dict(kind='pair',first=faces[0],second=f,tolerance=tol,face_tolerances=[tol,tol],cost=1.) for f in faces[1:]]*2
    expected=tuple(stage_task(context,t) for t in tasks)
    with a.OwnedPlanner(tmp_path/'stages',workers=2) as planner:
        report=planner.submit_stage(packet,tasks)
    assert report['results']==expected and report['actual_workers']==2
    assert all(row['model_decode_count']==1 for row in report['worker_reports'])
    assert all(row['model_sha256']==packet['sha256'] for row in report['worker_reports'])
    assert json.loads((tmp_path/'stages/cleanup.json').read_text())['active_processes']==0

def test_real_requested_payload_limit_exceeds_default(tmp_path):
    model,_=_fixture();model=from_dict(to_dict(model));model._serialization_extensions={'pp5:padding':{'value':'x'*4100000}}
    limits=dict(payload_limit=6000000,walk_limit=200000)
    packet=a.freeze_model(model,**limits);assert len(packet['payload'])>4000000
    context=WorkerModel(packet,**limits);faces=list(model.faces);tol=model.tolerance.effective_length(5.)
    tasks=[dict(kind='pair',first=faces[0],second=faces[1],tolerance=tol,face_tolerances=[tol,tol],cost=1.)]
    expected=stage_task(context,tasks[0])
    with a.OwnedPlanner(tmp_path/'large',workers=1,**limits) as planner:
        result=planner.submit_stage(packet,tasks)
    assert result['results']==(expected,)
    assert a.read_packet(tmp_path/'large/worker-0/limits.json','limits')==limits
