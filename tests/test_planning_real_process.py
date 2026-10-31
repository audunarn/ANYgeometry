"""PP2 real owned subprocess checks, separate from injected unit lifecycles."""
import json
import time
from pathlib import Path
import threading
import pytest
from anygeometry import _planning_executor as a
from anygeometry._planning_tasks import model_task
from anygeometry.definition_binding import definition_checksum
from test_planning_transport_schema import cold_model


def test_real_two_workers_transport_sampling_ci_cleanup(tmp_path):
    model = cold_model(); packet = a.freeze_model(model)
    tasks = ('pair', 'face') * 4
    reference = tuple(model_task(packet, name) for name in tasks)
    folder = tmp_path/'two-workers'
    result = a.execute(model, tasks, lambda name:model_task(packet, name), directory=folder,
        workers=2, opt_in=True, expected_ci=(definition_checksum(reference), packet['sha256']),
        observed_ci=lambda result:(definition_checksum(result['results']), result['model_sha256']))
    assert result['mode'] == 'parallel', result
    assert result['results'] == reference and result['ci_verified']
    assert set(result['sampling']['workers']) == {0, 1}
    assert all(r['full_environment_checks'] == 1 for r in result['worker_reports'])
    assert all(len(r['boundary_seconds']) == 4 for r in result['worker_reports'])
    assert json.loads((folder/'cleanup.json').read_text())['active_processes'] == 0
    assert json.loads((folder/'workers-cleanup.json').read_text())['active_processes'] == 0


def test_real_nested_cancellation_never_replays_serial(tmp_path):
    model = cold_model(); folder=tmp_path/'cancel'
    signal = folder/'cancel.flag'
    failures=[]
    def cancel_after_admission():
        deadline=time.monotonic()+65
        while time.monotonic()<deadline:
            if (folder/'worker-0/admitted.json').exists():
                signal.write_text('cancel', encoding='utf-8'); return
            if (folder/'result.json').exists():
                failures.append('planner ended before worker admission'); return
            time.sleep(.01)
        failures.append('worker admission deadline')
    watcher=threading.Thread(target=cancel_after_admission)
    watcher.start()
    try:
        with pytest.raises(a.PlannerCancelled):
            a.execute(model, ('pair','face')*300, lambda _:pytest.fail('cancel replayed'),
                directory=folder,workers=2,opt_in=True)
    finally:
        watcher.join(timeout=70)
    assert not watcher.is_alive() and not failures
    cleanup=json.loads((folder/'cleanup.json').read_text())
    assert cleanup['verified'] and cleanup['active_processes']==0
    assert any('worker' in row['command'] for row in cleanup['checked'])


def test_real_planner_corrupt_transport_declines_and_cleans(tmp_path):
    folder=tmp_path/'corrupt'
    with a.OwnedPlanner(folder,workers=2) as planner:
        packet=a.pack({'bad':'request'},'request')
        packet['payload']+=' '
        a.write_packet(folder/'request.json',packet)
        planner.tree.wait(planner.child,planner.deadline)
        result=a.read_packet(folder/'result.json','result')
        assert result['status']=='declined' and 'checksum' in result['reason']
    assert json.loads((folder/'cleanup.json').read_text())['active_processes']==0


def test_real_started_planner_fallback_uses_original_source(tmp_path):
    folder=tmp_path/'fallback'; model=object(); seen=[]
    result=a.execute(model,('pair','face'),lambda name:seen.append((model,name)) or name,
                     directory=folder,workers=2,opt_in=True)
    assert result['mode']=='serial' and 'model' in result['decline_reason']
    assert seen==[(model,'pair'),(model,'face')]
    cleanup=json.loads((folder/'cleanup.json').read_text())
    assert cleanup['verified'] and cleanup['active_processes']==0 and cleanup['commands']
