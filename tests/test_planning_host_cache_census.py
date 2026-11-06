"""Host cache census binds registered stores without visiting warm payloads."""
import sys
import types

import pytest

from anygeometry import _planning_ledger as ledger
from anygeometry._planning_qualification import QualificationDecline


def registered(monkeypatch):
    module = types.ModuleType('anygeometry._host_cache_census_fixture')
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def host_census(module):
    return ledger.loaded_geometry_ledger(modules=(module,), omit_cache_payloads=True)


def test_registered_warm_payload_does_not_consume_80000_node_bound(monkeypatch):
    module = registered(monkeypatch)
    module._MEMO = {i: object() for i in range(80001)}
    original = module._MEMO
    result = host_census(module)
    result.validate()
    assert not result.report.truncated
    assert result.rows[module.__name__ + '._MEMO']['action'] == 'reset'
    assert module._MEMO is original and len(original) == 80001


def test_host_census_still_checks_container_binding(monkeypatch):
    module = registered(monkeypatch)
    module._MEMO = {'warm': object()}
    result = host_census(module)
    module._MEMO = {}
    with pytest.raises(QualificationDecline, match='binding'):
        result.validate()


def test_cache_registration_does_not_hide_protected_definition_alias(monkeypatch):
    module = registered(monkeypatch)
    module._MEMO = {}
    module.__annotations__ = module._MEMO
    with pytest.raises(QualificationDecline, match='protected'):
        host_census(module)


def test_post_capture_protected_definition_alias_is_detected(monkeypatch):
    module = registered(monkeypatch)
    module._MEMO = {}
    result = host_census(module)
    module.__annotations__ = module._MEMO
    with pytest.raises(QualificationDecline):
        result.validate()


def test_registered_cache_cannot_alias_a_pinned_definition_root(monkeypatch):
    module = registered(monkeypatch)
    module._MEMO = {'protected': 1}
    module.DEFINITION = module._MEMO
    monkeypatch.setattr(ledger, '_FIXED_ROOTS', ledger._FIXED_ROOTS + (module.__name__ + '.DEFINITION',))
    with pytest.raises(QualificationDecline, match='protected'):
        host_census(module)


def test_registered_cache_cannot_alias_keyword_defaults(monkeypatch):
    module = registered(monkeypatch)
    module._MEMO = {'value': 1}
    def function(*, value=1):
        return value
    function.__kwdefaults__ = module._MEMO
    module.function = function
    with pytest.raises(QualificationDecline, match='protected'):
        host_census(module)


@pytest.mark.parametrize('protected_kind', ['fixed', 'annotation'])
def test_cache_exemption_cannot_hide_nested_definition_alias(monkeypatch, protected_kind):
    module = registered(monkeypatch)
    module._MEMO = {'warm': 1}
    if protected_kind == 'fixed':
        module.DEFINITION = {'nested': module._MEMO}
        monkeypatch.setattr(ledger, '_FIXED_ROOTS', ledger._FIXED_ROOTS + (module.__name__ + '.DEFINITION',))
    else:
        module.__annotations__ = {'nested': module._MEMO}
    with pytest.raises(QualificationDecline, match='protected'):
        host_census(module)


def test_unregistered_warm_store_is_not_exempt(monkeypatch):
    module = registered(monkeypatch)
    module.unknown = {i: object() for i in range(80001)}
    with pytest.raises(QualificationDecline, match='coverage|truncated'):
        host_census(module)


def test_unrelated_loaded_modules_are_not_added_as_host_roots(monkeypatch):
    module = registered(monkeypatch)
    module._MEMO = {}
    extra = types.ModuleType('anygeometry._unrelated_gui_fixture')
    extra.cache = {i: object() for i in range(80001)}
    monkeypatch.setitem(sys.modules, extra.__name__, extra)
    result = host_census(module)
    result.validate()
    assert result.modules == (module,)


def test_opaque_host_census_requires_explicit_roots():
    with pytest.raises(QualificationDecline, match='explicit'):
        ledger.loaded_geometry_ledger(omit_cache_payloads=True)


def test_host_admission_never_resets_registered_cache(monkeypatch):
    module = registered(monkeypatch)
    module._MEMO = {'warm': 12}
    result = host_census(module)
    monkeypatch.setattr(ledger, 'reset_declined', lambda *a, **k: pytest.fail('host cache reset'))
    result.validate_host()
    assert module._MEMO == {'warm': 12}


def test_host_admission_refuses_unregistered_small_store(monkeypatch):
    module = registered(monkeypatch)
    module.unknown = {'arbitrary': 1}
    result = host_census(module)
    with pytest.raises(QualificationDecline, match='declined ledger store'):
        result.validate_host()


def test_host_admission_refuses_active_context(monkeypatch):
    from contextvars import ContextVar
    module = registered(monkeypatch)
    module.context = ContextVar('active_test_context', default=None)
    token = module.context.set(1)
    try:
        result = host_census(module)
        with pytest.raises(QualificationDecline, match='active context'):
            result.validate_host()
    finally:
        module.context.reset(token)
