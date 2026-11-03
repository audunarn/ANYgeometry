"""A captured source is immutable evidence, not a new fallback opportunity."""
import pytest

from anygeometry import _planning_executor as executor
from anygeometry import _planning_host as host
from anygeometry._planning_canaries import _fixture
from anygeometry.batch_intersections import IntersectionBatchPolicy


def test_post_capture_source_drift_is_terminal(monkeypatch):
    monkeypatch.setattr(executor, 'source_files', lambda: {'owned.py': 'new'})
    with pytest.raises(executor.PlannerSourceChanged):
        executor.verify_sources({'owned.py': 'captured'})


def test_post_capture_unreadable_source_is_terminal(monkeypatch):
    error = OSError('captured source disappeared')
    def missing():
        raise error
    monkeypatch.setattr(executor, 'source_files', missing)
    with pytest.raises(executor.PlannerSourceChanged) as caught:
        executor.verify_sources({'owned.py': 'captured'})
    assert caught.value.__cause__ is error


def test_source_drift_cannot_reenter_serial_after_stage_dispatch(tmp_path, monkeypatch):
    model, _ = _fixture()
    control = host.StageControl(tmp_path, expected_ci=('p', 'a'), observed_ci=lambda _: ('p', 'a'))
    monkeypatch.setattr(executor, 'source_files', lambda: {'owned.py': 'new'})
    def parallel(bridge):
        executor.verify_sources({'owned.py': 'captured'})
    with pytest.raises(executor.PlannerSourceChanged):
        host.qualified_plan(model, control, lambda: pytest.fail('source drift replayed serial'),
                            parallel, policy=IntersectionBatchPolicy(), mode='exact')
    assert control.mode == 'terminal'
    assert not control.ci_verified


@pytest.mark.parametrize('role', ['planner', 'worker', 'session-planner', 'session-worker'])
def test_source_drift_is_terminal_across_child_wire(tmp_path, monkeypatch, role):
    import contextlib
    import sys
    from anygeometry import _planning_child as child
    from anygeometry import _planning_session as session
    packets = []
    def prepare(*,production=False):
        assert production is False  # This diagnostic fixture has no package marker.
        return {},{}
    monkeypatch.setattr(child, 'prepare', prepare)
    monkeypatch.setattr(child, 'threadpool_limits', lambda **k: contextlib.nullcontext())
    def drift(*args, **kwargs):
        raise executor.PlannerSourceChanged('source changed after capture')
    monkeypatch.setattr(child, 'planner', drift)
    monkeypatch.setattr(child, 'worker', drift)
    monkeypatch.setattr(session, 'session_planner', drift)
    monkeypatch.setattr(session, 'session_worker', drift)
    monkeypatch.setattr(child, 'write_packet', lambda path, packet: packets.append(packet))
    monkeypatch.setattr(sys, 'argv', [child.__file__, role, str(tmp_path), '10'])
    child.main()
    result = executor.unpack(packets[-1], 'result' if role in ('planner', 'session-planner') else 'worker-result')
    assert result['status'] == 'source_changed'
    with pytest.raises(executor.PlannerSourceChanged):
        executor.check_result_status(result)


def test_machine_load_uses_only_aggregate_observations(monkeypatch):
    from types import SimpleNamespace
    from anygeometry import _planning_child as child
    monkeypatch.setattr(child.psutil, 'process_iter', lambda *a, **k: pytest.fail('unrelated process enumeration'))
    monkeypatch.setattr(child.psutil, 'cpu_percent', lambda *a, **k: 12.5)
    monkeypatch.setattr(child.psutil, 'virtual_memory', lambda: SimpleNamespace(available=123456, total=234567))
    monkeypatch.setattr(child.os, 'cpu_count', lambda: 8)
    observed = child.load()
    assert observed['python_process_count'] is None
    assert observed['cpu_percent'] == 12.5
    assert observed['available_memory_bytes'] == 123456
    assert observed['logical_cpus'] == 8
