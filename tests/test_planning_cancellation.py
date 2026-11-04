"""PP3a passive-signal lifecycle, with injected event-read faults for error paths.

Fault injection is not evidence qualifying altered signals or arbitrary callbacks.
"""
import pytest
from _planning_numpy_profile_contract import numpy_profile_test

from anygeometry import _planning_host as host
from anygeometry import _planning_executor as executor
from anygeometry import batch_intersections as bi
from anygeometry._planning_canaries import _fixture
from anygeometry._planning_qualification import QualificationDecline
from anygeometry.serialization import from_dict, to_dict
from anygeometry.cancellation import CancellationSignal, _event


def setup_call(tmp_path, callback):
    model, _ = _fixture()
    model = from_dict(to_dict(model))
    control = host.StageControl(tmp_path/'owned', expected_ci=('plan', 'apply'),
                                observed_ci=lambda _: ('plan', 'apply'))
    signal = None
    if callback is not None:
        signal = CancellationSignal()
        # Isolated owned Event read fault injection; no class/global patch or
        # production claim that arbitrary policy callbacks can be delegated.
        _event(signal).is_set = callback
    policy = bi.IntersectionBatchPolicy(cancellation_check=signal)
    return model, control, policy


def call(model, control, policy, parallel, serial):
    return host.qualified_plan(model, control, serial, parallel, policy=policy, mode='exact')


def test_false_callback_can_use_parallel_and_ci(tmp_path):
    polls = []
    model, control, policy = setup_call(tmp_path, lambda: polls.append(1) and False)
    result = call(model, control, policy, lambda bridge: 'candidate', lambda: 'serial')
    assert result == 'candidate' and control.ci_verified and polls


def test_initial_cancel_does_not_start_or_replay(tmp_path):
    model, control, policy = setup_call(tmp_path, lambda: True)
    with pytest.raises(executor.PlannerCancelled, match='intersection planning cancelled'):
        call(model, control, policy, lambda _: pytest.fail('started'), lambda: pytest.fail('replayed'))
    assert not control.directory.exists()


@pytest.mark.parametrize('error', [QualificationDecline('callback'), OSError('callback'),
                                  RuntimeError('callback'), KeyboardInterrupt('callback')])
def test_callback_exception_identity_is_not_reclassified(tmp_path, monkeypatch, error):
    events = []
    active = False
    def callback():
        if active:
            raise error
        return False
    model, control, policy = setup_call(tmp_path, callback)
    class Fake:
        def __init__(self, *args, **kw):
            nonlocal active
            active = True
        def submit_stage(self, packet, tasks, *, poll):
            poll()
            pytest.fail('callback failure returned')
        def close(self):
            events.append('retired')
    monkeypatch.setattr(host, 'OwnedPlanner', Fake)
    with pytest.raises(type(error)) as caught:
        call(model, control, policy, lambda b: b.run('pair', [{}], lambda _: None),
             lambda: pytest.fail('callback error replayed'))
    assert caught.value is error and events == ['retired']


def test_cancellation_after_refusal_retires_without_replay(tmp_path, monkeypatch):
    events = []
    cancelled = False
    model, control, policy = setup_call(tmp_path, lambda: cancelled)
    class Fake:
        def __init__(self, *args, **kw): pass
        def submit_stage(self, *args, **kw):
            nonlocal cancelled
            cancelled = True
            raise QualificationDecline('refused')
        def close(self): events.append('retired')
    monkeypatch.setattr(host, 'OwnedPlanner', Fake)
    with pytest.raises(executor.PlannerCancelled):
        call(model, control, policy, lambda b: b.run('pair', [{}], lambda _: None),
             lambda: pytest.fail('cancelled refusal replayed'))
    assert events == ['retired']


def test_cleanup_decline_is_terminal(tmp_path, monkeypatch):
    model, control, policy = setup_call(tmp_path, None)
    class Fake:
        def __init__(self, *args, **kw): pass
        def submit_stage(self, *args, **kw): raise QualificationDecline('refused')
        def close(self): raise QualificationDecline('cleanup failed')
    monkeypatch.setattr(host, 'OwnedPlanner', Fake)
    with pytest.raises(executor.PlannerCleanupError):
        call(model, control, policy, lambda b: b.run('pair', [{}], lambda _: None),
             lambda: pytest.fail('cleanup failure replayed'))


@pytest.mark.parametrize('point', ['result', 'ci'])
def test_cancel_before_sampling_or_publication(tmp_path, monkeypatch, point):
    events = []
    cancelled = False
    model, control, policy = setup_call(tmp_path, lambda: cancelled)
    class Fake:
        def __init__(self, *args, **kw): pass
        def submit_stage(self, *args, **kw):
            nonlocal cancelled
            cancelled = True
            return {}
        def close(self): events.append('retired')
    monkeypatch.setattr(host, 'OwnedPlanner', Fake)
    def observed(_):
        nonlocal cancelled
        cancelled = True
        return ('plan', 'apply')
    control.observed_ci = observed
    parallel = (lambda b: b.run('pair', [{}], lambda _: pytest.fail('sampled'))
                ) if point == 'result' else lambda b: 'candidate'
    with pytest.raises(executor.PlannerCancelled):
        call(model, control, policy, parallel, lambda: pytest.fail('replayed'))
    assert not control.ci_verified
    assert events == (['retired'] if point == 'result' else [])


def test_callback_mutation_cannot_publish(tmp_path):
    active = False
    def callback():
        if active: model._revision += 1
        return False
    model, control, policy = setup_call(tmp_path, callback)
    def parallel(bridge):
        nonlocal active
        active = True
        return 'candidate'
    with pytest.raises(executor.PlannerSourceChanged):
        call(model, control, policy, parallel, lambda: pytest.fail('mutation replay'))


@pytest.mark.parametrize('field', ['max_predicates', 'max_candidate_pairs'])
def test_budget_policies_keep_original_serial_path(tmp_path, field):
    model, control, _ = setup_call(tmp_path, None)
    policy = bi.IntersectionBatchPolicy(**{field: 1})
    assert call(model, control, policy, lambda _: pytest.fail('budget parallel'), lambda: 'serial') == 'serial'


def test_owned_exit_wait_polls_cancel():
    from anygeometry._planning_process import OwnedTree
    import subprocess
    events = []
    class Child:
        def wait(self, *, timeout):
            events.append('wait')
            raise subprocess.TimeoutExpired('owned', timeout)
    def poll():
        if events: raise executor.PlannerCancelled('late cancel')
    import time
    with pytest.raises(executor.PlannerCancelled):
        OwnedTree.wait(None, Child(), time.monotonic()+10, poll=poll)
    assert events == ['wait']


def _host_pair_callback_error(error,calls):
    import sys
    plan_code=bi._plan_intersections.__code__
    def callback():
        frame=sys._getframe(1)
        try:
            while frame is not None:
                if frame.f_code is plan_code:
                    calls.append(frame.f_locals.get('_stages') is not None)
                    raise error
                frame=frame.f_back
        finally:
            del frame
        return False
    return callback,plan_code


@numpy_profile_test
def test_public_host_callback_decline_is_not_serial_replay(tmp_path,monkeypatch):
    error=QualificationDecline('caller error during host pair enumeration')
    calls=[];callback,plan_code=_host_pair_callback_error(error,calls)
    model,control,policy=setup_call(tmp_path,callback)
    before=to_dict(model)
    monkeypatch.setenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING','1')
    with pytest.raises(QualificationDecline) as caught:
        bi.plan_intersections(model,tuple(model.faces),policy=policy,_parallel_control=control)
    assert caught.value is error and calls==[True]
    assert control.mode=='terminal' and control.decline_reason is None
    assert bi._plan_intersections.__code__ is plan_code and to_dict(model)==before


def test_unsupported_policy_initial_route_keeps_callback_exception_in_serial(tmp_path,monkeypatch,request):
    from dataclasses import replace
    error=QualificationDecline('caller error after explicit initial route refusal')
    calls=[];callback,plan_code=_host_pair_callback_error(error,calls)
    model,control,policy=setup_call(tmp_path,callback)
    policy=replace(policy,max_candidate_pairs=1)
    before=to_dict(model)
    monkeypatch.setenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING','1')
    with pytest.raises(QualificationDecline) as caught:
        bi.plan_intersections(model,tuple(model.faces),policy=policy,_parallel_control=control)
    assert caught.value is error and calls==[False]
    assert control.mode=='serial' and not control.ci_verified
    assert control.decline_reason=='QualificationDecline: PP3/special-route adapter unavailable'
    assert bi._plan_intersections.__code__ is plan_code and to_dict(model)==before
    request.node.user_properties.extend((('execution_scope','explicit-unsupported-policy-serial-callback'),
                                         ('parallel_callback_qualification',False)))


def test_callback_consuming_deadline_is_terminal(tmp_path, monkeypatch):
    clock = [0.0]
    control = host.StageControl(tmp_path/'never', expected_ci=('p', 'a'), observed_ci=lambda _: ('p', 'a'))
    monkeypatch.setattr(host.time, 'monotonic', lambda: clock[0])
    bridge = host.Bridge({}, control, poll=lambda: clock.__setitem__(0, 101.0))
    with pytest.raises(executor.PlannerDeadline):
        bridge.check_deadline()


def test_initial_callback_mutation_is_not_silently_snapshotted(tmp_path):
    def callback():
        model._revision += 1
        return False
    model, control, policy = setup_call(tmp_path, callback)
    with pytest.raises(executor.PlannerSourceChanged):
        call(model, control, policy, lambda _: pytest.fail('mutated input admitted'),
             lambda: pytest.fail('mutation replay'))


def test_cleanup_failure_has_precedence_but_preserves_callback_in_chain(tmp_path, monkeypatch):
    original = OSError('callback error')
    cleanup = QualificationDecline('cleanup error')
    active = False
    def callback():
        if active: raise original
        return False
    model, control, policy = setup_call(tmp_path, callback)
    class Fake:
        def __init__(self, *args, **kw):
            nonlocal active
            active = True
        def submit_stage(self, packet, tasks, *, poll): poll()
        def close(self): raise cleanup
    monkeypatch.setattr(host, 'OwnedPlanner', Fake)
    with pytest.raises(executor.PlannerCleanupError) as caught:
        call(model, control, policy, lambda b: b.run('pair', [{}], lambda _: None),
             lambda: pytest.fail('unsafe replay'))
    assert caught.value.__cause__ is cleanup
    assert cleanup.__context__.error is original


def test_arbitrary_callback_retains_original_serial_semantics(tmp_path):
    model, control, _ = setup_call(tmp_path, None)
    callback = lambda: pytest.fail('unsupported callback polled before serial')
    policy = bi.IntersectionBatchPolicy(cancellation_check=callback)
    assert call(model, control, policy, lambda _: pytest.fail('arbitrary callback delegated'),
                lambda: 'original serial') == 'original serial'


def test_forged_signal_is_not_delegated(tmp_path):
    from anygeometry.cancellation import CancellationSignal
    model, control, _ = setup_call(tmp_path, None)
    forged = object.__new__(CancellationSignal)
    policy = bi.IntersectionBatchPolicy(cancellation_check=forged)
    assert call(model, control, policy, lambda _: pytest.fail('forged signal delegated'),
                lambda: 'original serial') == 'original serial'


def test_terminal_control_cannot_start_again_or_replay(tmp_path):
    model, control, _ = setup_call(tmp_path, None)
    signal = CancellationSignal()
    signal.cancel()
    policy = bi.IntersectionBatchPolicy(cancellation_check=signal)
    with pytest.raises(executor.PlannerCancelled):
        call(model, control, policy, lambda _: pytest.fail('started'), lambda: pytest.fail('replayed'))
    assert control.mode == 'terminal'
    with pytest.raises(bi.GeometryError, match='stage control already used'):
        call(model, control, bi.IntersectionBatchPolicy(), lambda _: pytest.fail('restarted'),
             lambda: pytest.fail('terminal replay'))
