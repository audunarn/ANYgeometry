"""Real interpreter interfaces: compatibility never manufactures qualification."""
import ctypes
import sys

import pytest
from _planning_numpy_profile_contract import numpy_profile_test
from _planning_contract_helpers import capture_with_runtime_boundary, classify_initial_dispatcher_refusal

from anygeometry import _planning_code as code
from anygeometry import _planning_ledger as ledger
from anygeometry._planning_manifest import capture_callable_manifest, prepare_interfaces
from anygeometry._planning_qualification import QualificationDecline


def test_native_field_type_without_optional_public_export(monkeypatch):
    class Record(ctypes.Structure):
        _fields_ = [('value', ctypes.c_int)]
    expected = type(Record.value)
    monkeypatch.delattr(ctypes, 'CField', raising=False)
    assert code._ctypes_field_type() is expected


def test_rebound_field_export_is_not_trusted(monkeypatch):
    monkeypatch.setattr(ctypes, 'CField', object(), raising=False)
    with pytest.raises(QualificationDecline, match='ctypes field export binding'):
        code._ctypes_field_type()


def test_source_owned_slot_metadata_on_this_interpreter():
    from anygeometry import nurbs
    prepare_interfaces()
    assert type(nurbs._PreparedCurveState.__annotations__) is dict
    assert nurbs._PreparedCurveState.__annotations__ == {}


def test_real_owned_callable_without_external_annotation_module_assumption(request):
    from anygeometry.tolerance import TolerancePolicy
    captured = capture_with_runtime_boundary((TolerancePolicy,), request=request)
    if captured is None:
        return # Exact initial refusal is not postcapture coverage.
    captured.verify()


def test_owned_annotation_store_is_retained_and_structurally_checked():
    from anygeometry.exact_curves import _BranchConstants
    original = ledger._trusted_store_values()[-1]
    if sys.version_info < (3, 14):
        assert original is _BranchConstants.__annotations__
    before = ledger.trusted_annotation_digest(original)
    changed = dict(original)
    changed['unexpected'] = int
    with pytest.raises(QualificationDecline):
        ledger.trusted_annotation_digest(changed)
    assert ledger.trusted_annotation_digest(original) == before


def test_registered_rlock_requires_a_nonmutating_state_interface(request):
    if classify_initial_dispatcher_refusal(request):return # Exact initial refusal is not RLock coverage.
    from anygeometry import surface_capabilities
    from anygeometry._planning_executor import PlannerSourceChanged
    lock = surface_capabilities._LOCK
    assert not lock._is_owned()
    if 'locked' not in vars(type(lock)):
        with pytest.raises(QualificationDecline, match='runtime RLock state interface unavailable'):
            capture_callable_manifest((lock,))
        assert not lock._is_owned()
        return
    captured = capture_callable_manifest((lock,))
    captured.verify()
    with lock:
        with pytest.raises(PlannerSourceChanged):
            captured.verify()
    captured.verify()


@pytest.mark.parametrize('provider', ['ctypes', '_ctypes'])
@numpy_profile_test
def test_native_field_export_rebinding_after_capture_is_terminal(monkeypatch, provider):
    import importlib
    from anygeometry._planning_executor import PlannerSourceChanged
    captured = code.InfrastructurePin('host')
    module = importlib.import_module(provider)
    monkeypatch.setattr(module, 'CField', object(), raising=False)
    with pytest.raises(PlannerSourceChanged):
        captured.verify()
