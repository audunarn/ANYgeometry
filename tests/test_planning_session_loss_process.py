"""Owned-worker loss and consumed deadlines cannot reuse a surviving session."""
import json
import time
from pathlib import Path
import psutil
import pytest
from anygeometry import batch_intersections as bi
from anygeometry._planning_canaries import _fixture
from anygeometry._planning_executor import freeze_model,PlannerDeadline
from anygeometry._planning_session import OwnedSession
from anygeometry._planning_qualification import QualificationDecline
from anygeometry.serialization import from_dict,to_dict


def first_pair():
    original,_=_fixture();model=from_dict(to_dict(original));packet=freeze_model(model)
    class Captured(BaseException):pass
    class Probe:
        def check_deadline(self):pass
        def run(self,kind,tasks,serial):
            if tasks:raise Captured(tasks[0])
            return ()
    with pytest.raises(Captured) as caught:
        bi._plan_intersections(model,tuple(model.faces),policy='connect',_stages=Probe())
    return packet,caught.value.args[0]


@pytest.mark.parametrize('fault',['worker-exit','deadline'])
def test_session_terminal_loss_retires_without_retry(tmp_path,fault):
    packet,task=first_pair();owned=OwnedSession(tmp_path/fault,workers=2,timeout=70)
    try:
        report=owned.submit_stage('pair',packet,[task])
        if fault=='worker-exit':
            process=psutil.Process(report['worker_reports'][0]['pid'])
            command=process.cmdline()
            assert command[-3]=='session-worker'
            assert Path(command[-2]).resolve()==(owned.directory/'worker-0').resolve()
            process.kill()  # Exact owned PID, role and directory checked immediately above.
        else:owned.deadline=time.monotonic()
        with pytest.raises(QualificationDecline if fault=='worker-exit' else PlannerDeadline):owned.finish()
        assert owned.failed
        with pytest.raises(QualificationDecline,match='terminal'):owned.submit_stage('pair',packet,[task])
    finally:owned.close()
    cleanup=json.loads((owned.directory/'cleanup.json').read_text())
    assert cleanup['verified'] and cleanup['active_processes']==0
