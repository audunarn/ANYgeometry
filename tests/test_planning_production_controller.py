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
