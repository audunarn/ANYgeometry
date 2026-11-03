"""Real public planner callback/Job integration; tiny three-face fixture only."""
from _planning_numpy_profile_contract import numpy_profile_test
from _planning_platform_contract import windows_owned_test
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


@windows_owned_test
@numpy_profile_test
def test_session_false_signal_retains_policy_plan_apply_and_source(tmp_path, monkeypatch):
    model, _ = _fixture()
    document = to_dict(model)
    model = from_dict(document)
    reference = from_dict(document)
    monkeypatch.delenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING', raising=False)
    serial = bi.plan_intersections(reference, tuple(reference.faces), policy='connect')
    bi.apply_intersections(reference, serial, policy='connect')
    expected = (serial.content_checksum, to_dict(reference)['checksum']['value'])
    def observed(plan):
        candidate = from_dict(document)
        bi.apply_intersections(candidate, plan, policy='connect')
        return plan.content_checksum, to_dict(candidate)['checksum']['value']
    signal = CancellationSignal()
    policy = bi.IntersectionBatchPolicy(cancellation_check=signal)
    control = StageControl(tmp_path/'success', expected_ci=expected, observed_ci=observed, workers=2,reuse_workers=True)
    monkeypatch.setenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING', '1')
    plan = bi.plan_intersections(model, tuple(model.faces), policy=policy, _parallel_control=control)
    assert control.mode == 'parallel' and control.ci_verified and not signal.cancelled
    assert plan.policy.cancellation_check is signal
    assert plan.content_checksum == serial.content_checksum and to_dict(model) == document
    assert [r['stage'] for r in control.reports] == ['pair', 'face']
    for kind in ('session',):
        cleanup = json.loads((control.directory/kind/'cleanup.json').read_text())
        assert cleanup['verified'] and cleanup['active_processes'] == 0
