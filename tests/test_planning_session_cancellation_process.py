"""Real public planner callback/Job integration; tiny three-face fixture only."""
import json
import threading
import time
import pytest

from anygeometry import batch_intersections as bi
from anygeometry._planning_host import StageControl
from anygeometry._planning_executor import PlannerCancelled
from anygeometry._planning_canaries import _fixture
from anygeometry.serialization import from_dict, to_dict
from anygeometry.cancellation import CancellationSignal, _event


@pytest.mark.parametrize('failure', ['cancel', 'callback-error'])
@pytest.mark.parametrize('stage', ['pair', 'face'])
def test_session_callback_after_worker_admission_retires_tree(tmp_path, monkeypatch, failure, stage):
    model, _ = _fixture()
    document = to_dict(model)
    model = from_dict(document)
    folder = tmp_path/'public-cancel'
    seen = []
    error = OSError('caller cancellation provider failed')
    signal = CancellationSignal()
    stop = threading.Event()
    read = _event(signal).is_set
    if failure == 'callback-error':
        # An injected host event-read failure, not an admitted arbitrary callback.
        def failed_read():
            if read(): raise error
            return False
        _event(signal).is_set = failed_read
    def cancel_after_admission():
        deadline = time.monotonic()+90
        while not stop.wait(.01) and time.monotonic()<deadline:
            if (folder/'session'/'worker-0'/('admitted-'+('0' if stage=='pair' else '1')+'.json')).exists():
                seen.append('admitted')
                signal.cancel()
                return
    watcher = threading.Thread(target=cancel_after_admission)
    control = StageControl(folder, expected_ci=('plan', 'apply'),
                           observed_ci=lambda _: pytest.fail('cancelled CI'), workers=2,reuse_workers=True)
    policy = bi.IntersectionBatchPolicy(cancellation_check=signal)
    monkeypatch.setenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING', '1')
    watcher.start()
    try:
        with pytest.raises(PlannerCancelled if failure == 'cancel' else OSError) as caught:
            bi.plan_intersections(model, tuple(model.faces), policy=policy, _parallel_control=control)
    finally:
        stop.set()
        watcher.join(timeout=1)
    assert not watcher.is_alive()
    if failure == 'callback-error': assert caught.value is error
    assert seen == ['admitted'] and not control.ci_verified
    assert control.mode == 'terminal'
    assert control.reports == []
    if stage == 'pair': assert not (folder/'session/command-1.json').exists()
    assert to_dict(model) == document
    cleanup = json.loads((folder/'session/cleanup.json').read_text())
    assert cleanup['verified'] and cleanup['active_processes'] == 0
    assert any('session-worker' in row['command'] for row in cleanup['checked'])
