"""Production admission is separate from the diagnostic CI oracle."""
import pytest
from anygeometry import _planning_host as host


def test_public_opt_in_without_private_control_selects_production(monkeypatch):
    from anygeometry import _planning_controller as production
    seen=[]
    monkeypatch.setattr(production,'run',lambda *a,**k:seen.append(k) or 'production')
    result=host.qualified_plan(object(),None,lambda:'serial',lambda bridge:'candidate',policy=object(),mode='exact')
    assert result=='production' and seen


def test_production_memory_profile_is_explicit_and_fixed(monkeypatch):
    from anygeometry import _planning_controller as production
    monkeypatch.setattr(production,'observe_memory',lambda:dict(host_peak_bytes=123456,host_private_bytes=100,available_bytes=2**36))
    budget=production.memory_profile()
    assert budget.host_peak_estimate_bytes==123456+512*2**20
    assert budget.tree_limit_bytes==4*2**30
    assert budget.total_limit_bytes==8*2**30
    assert budget.free_fraction==(1,2)


@pytest.mark.parametrize('unsupported', ['native', 'mode', 'predicates', 'pairs'])
@pytest.mark.parametrize('raises', [False, True])
def test_unsupported_policy_keeps_original_serial_callback_and_error_contract(monkeypatch, unsupported, raises):
    from types import SimpleNamespace
    from anygeometry import GeometryError
    from anygeometry import _planning_controller as production
    events = []
    original = OSError('caller exception')
    wrapped = GeometryError('original serial wrapper')
    def serial():
        events.append('serial')
        if raises:
            raise wrapped from original
        return 'original result'
    policy = SimpleNamespace(cancellation_check=lambda: pytest.fail('extra host callback poll'),
                             max_predicates=1 if unsupported == 'predicates' else None,
                             max_candidate_pairs=1 if unsupported == 'pairs' else None)
    monkeypatch.setattr(production, '_storage', lambda: pytest.fail('unsupported policy allocated parallel scratch'))
    def invoke():
        return host.qualified_plan(object(), None, serial, lambda _: pytest.fail('unsupported parallel work'),
                                   policy=policy, mode='topology_certified' if unsupported == 'mode' else 'exact',
                                   eligible=unsupported != 'native',
                                   serial_replay=lambda _: pytest.fail('unsupported serial route was adapted'))
    if raises:
        with pytest.raises(GeometryError) as caught:
            invoke()
        assert caught.value is wrapped and caught.value.__cause__ is original
    else:
        assert invoke() == 'original result'
    assert events == ['serial']
