"""R5 owns package definitions and value schemas; Plane profiles are diagnostic."""
import hashlib
from pathlib import Path
import numpy as np
import pytest
from _planning_contract_helpers import capture_with_runtime_boundary
from anygeometry._planning_executor import PlannerSourceChanged, freeze_model
from anygeometry._planning_qualification import QualificationDecline
from anygeometry._planning_values import encode_value, decode_value, same


def test_obsolete_diagnostic_source_pin_is_still_rejected(monkeypatch):
    from anygeometry import _planning_profiles as diagnostic, _planning_ledger as ledger
    from anygeometry import batch_intersections as batch
    pins = dict(diagnostic._SOURCE_PINS)
    assert hashlib.sha256(Path(batch.__file__).read_bytes()).hexdigest() != pins[batch.__name__]
    # Isolate this diagnostic unit's prerequisite, never a production admission.
    monkeypatch.setattr(ledger, 'runtime_infrastructure', lambda: (None, None))
    with pytest.raises(QualificationDecline, match='branch profile source not registered'):
        diagnostic.BranchProfiles()
    assert diagnostic._SOURCE_PINS == pins


def test_actual_domain_support_and_freeze_preserve_model():
    from anygeometry._planning_canaries import _fixture
    from anygeometry.material_arrangement import MaterialDomain
    from anygeometry.surfaces import Plane
    model, _ = _fixture()
    before = dict(vars(model))
    domains = [MaterialDomain.from_model(model, face) for face in model.faces]
    assert len(domains) == 3 and all(type(domain.support) is Plane for domain in domains)
    assert freeze_model(model)
    assert all(vars(model)[name] is value for name, value in before.items())


@pytest.mark.parametrize('shape', [(3,), (2, 3), (2, 2, 2), ()])
def test_stage_codec_retains_exact_float64_array_shape_and_bits(shape):
    value = np.ones(shape, dtype=np.float64)
    assert same(value, decode_value(encode_value(value)))


@pytest.mark.parametrize('value', [np.array([1], dtype=np.int64), np.array([float('nan')]),
    np.array([float('inf')]), np.ones((2, 3))[:, ::2]])
def test_stage_codec_refuses_arrays_outside_its_schema(value):
    with pytest.raises(QualificationDecline):
        encode_value(value)


def test_array_subclass_refuses_without_dispatching_hooks():
    class Foreign(np.ndarray):
        def __array_function__(self, *args, **kwargs):
            pytest.fail('foreign array hook executed')
    with pytest.raises(QualificationDecline, match='unsupported type'):
        encode_value(np.ones(3).view(Foreign))


def test_non_native_endian_nan_is_not_admitted_as_finite_native_bytes():
    value = np.array([float('nan')], dtype='>f8' if np.little_endian else '<f8')
    assert not value.dtype.isnative
    with pytest.raises(QualificationDecline):
        encode_value(value)


@pytest.mark.parametrize('method', ['effective_length', 'effective_area'])
def test_all_owned_methods_remain_bound_even_when_unselected(monkeypatch, method):
    from anygeometry.tolerance import TolerancePolicy
    captured = capture_with_runtime_boundary((TolerancePolicy.effective_length, TolerancePolicy))
    if captured is None:
        return
    captured.verify()
    monkeypatch.setattr(TolerancePolicy, method, lambda self, scale: 2.)
    with pytest.raises(PlannerSourceChanged):
        captured.verify()


def test_owned_code_identity_change_is_terminal_even_for_equal_behavior(monkeypatch):
    from anygeometry.tolerance import TolerancePolicy
    captured = capture_with_runtime_boundary((TolerancePolicy.effective_length,))
    if captured is None:
        return
    function = TolerancePolicy.effective_length
    monkeypatch.setattr(function, '__code__', function.__code__.replace())
    with pytest.raises(PlannerSourceChanged):
        captured.verify()
