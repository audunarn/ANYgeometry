"""Owner-authorized PP5d regressions; no relaxation of admission predicates."""
import sys
import hashlib
import json
from types import SimpleNamespace

import pytest

from anygeometry import _planning_batch as batch
from anygeometry import _planning_executor as executor
from anygeometry import _planning_qualification as qualification
from anygeometry.model import GeometryModel
from anygeometry.errors import GeometryError


def huge_integer_model():
    limit = sys.get_int_max_str_digits()
    assert limit > 0, 'this regression requires the configured finite interpreter limit'
    model = GeometryModel()
    model._serialization_extensions = {'pp2:data': 10 ** (limit + 1)}
    return model


def _refuse_extension_encoding(monkeypatch, extension):
    """Fault only the owner's selected extension encoder, never global JSON."""
    original = qualification.json
    error = ValueError('selected extension encoder failure')
    calls = []
    def dumps(value, *args, **kwargs):
        if value is extension:
            calls.append(value)
            raise error
        return original.dumps(value, *args, **kwargs)
    namespace = SimpleNamespace(**vars(original))
    namespace.dumps = dumps
    monkeypatch.setattr(qualification, 'json', namespace)
    return error, calls


@pytest.mark.parametrize('operation', [qualification.model_extension_record, executor.freeze_model])
def test_oversized_integer_encoding_is_typed_refusal(operation, monkeypatch):
    model = huge_integer_model()
    value = model._serialization_extensions['pp2:data']
    limit = sys.get_int_max_str_digits()
    # The real scalar encoder may accept this integer on some runtimes.
    # Deliberately exercise the caught-error contract, not an extra digit bound.
    error, calls = _refuse_extension_encoding(monkeypatch, model._serialization_extensions)
    with pytest.raises(qualification.QualificationDecline, match='extension encoding refused') as caught:
        operation(model)
    assert caught.value.__cause__ is error
    assert calls == [model._serialization_extensions]
    assert sys.get_int_max_str_digits() == limit
    assert model._serialization_extensions['pp2:data'] is value


def test_host_codec_refusal_retires_then_uses_original_serial(tmp_path, monkeypatch):
    model = huge_integer_model()
    events = []
    extension = model._serialization_extensions
    value = extension['pp2:data']
    limit = sys.get_int_max_str_digits()
    _, calls = _refuse_extension_encoding(monkeypatch, extension)

    class CodecPlanner:
        def submit(self, original, tasks):
            assert original is model
            return executor.freeze_model(original)

        def close(self):
            events.append('retired')

    result = executor.execute(model, ('pair',), lambda task: events.append(task) or 'serial-value',
                              directory=tmp_path / 'unused', planner=CodecPlanner(), opt_in=True)
    assert calls == [extension]
    assert sys.get_int_max_str_digits() == limit
    assert model._serialization_extensions is extension and extension['pp2:data'] is value
    assert events == ['retired', 'pair']
    assert result['mode'] == 'serial'
    assert result['results'] == ('serial-value',)
    assert 'QualificationDecline: extension encoding refused' in result['decline_reason']


@pytest.mark.parametrize('kind', ['added', 'removed', 'rebound'])
def test_module_boundary_refusal_identifies_change(kind, monkeypatch):
    original = object()
    registered = {'known': original}
    observed = dict(registered)
    changed = 'late.module' if kind == 'added' else 'known'
    if kind == 'added':
        observed[changed] = object()
    elif kind == 'removed':
        del observed[changed]
    else:
        observed[changed] = object()
    monkeypatch.setattr(batch, 'sys', SimpleNamespace(modules=observed))
    with pytest.raises(qualification.QualificationDecline) as error:
        batch.verify_boundary({}, registered)
    assert 'batch module closure/binding changed' in str(error.value)
    assert kind + '=' in str(error.value)
    assert changed in str(error.value)


@pytest.mark.parametrize('name', ['shapely', 'shapely._different_native_layout'])
def test_unqualified_shapely_layout_cannot_silently_omit_geos(name, monkeypatch):
    monkeypatch.setattr(qualification, 'sys', SimpleNamespace(modules={name: object()}))
    with pytest.raises(qualification.QualificationDecline, match='Shapely native module layout'):
        qualification.runtime_files()


def test_numerical_error_is_retired_and_propagated_without_serial_replay(tmp_path):
    events = []
    original = GeometryError('numerical failure: bridge attribution')

    class FailingPlanner:
        def submit(self, model, tasks):
            raise original

        def close(self):
            events.append('retired')

    with pytest.raises(GeometryError) as error:
        executor.execute(None, ('pair',), lambda task: events.append('serial'),
                         directory=tmp_path / 'unused', planner=FailingPlanner(), opt_in=True)
    assert error.value is original and events == ['retired']


def test_extension_encoder_fault_is_local_exact_and_restored(monkeypatch):
    extension = {'pp2:data': 17}
    original_module, original_dumps = qualification.json, json.dumps
    with monkeypatch.context() as scoped:
        error, calls = _refuse_extension_encoding(scoped, extension)
        assert qualification.json is not original_module
        assert json.dumps is original_dumps
        assert qualification.json.dumps({'ordinary': 2}, sort_keys=True) == original_dumps({'ordinary': 2}, sort_keys=True)
        with pytest.raises(ValueError) as caught:
            qualification.json.dumps(extension, sort_keys=True)
        assert caught.value is error and calls == [extension]
    assert qualification.json is original_module and json.dumps is original_dumps


def test_actual_huge_integer_encoder_outcome_matches_owner_raw_identity(request):
    """A real encoder may accept this scalar; no extra digit ceiling is claimed."""
    model = huge_integer_model()
    extension = model._serialization_extensions
    value = extension['pp2:data']
    limit = sys.get_int_max_str_digits()
    before = dict(vars(model))
    options = dict(sort_keys=True, separators=(',', ':'), allow_nan=False)
    try:
        raw = json.dumps(extension, **options).encode('utf-8')
    except ValueError:
        request.node.user_properties.append(('actual_extension_encoder', 'ValueError-refusal'))
        with pytest.raises(qualification.QualificationDecline, match='^extension encoding refused: ValueError$') as caught:
            qualification.model_extension_record(model)
        assert type(caught.value.__cause__) is ValueError
        with pytest.raises(qualification.QualificationDecline, match='^extension encoding refused: ValueError$'):
            executor.freeze_model(model)
    else:
        request.node.user_properties.append(('actual_extension_encoder', 'accepted-exact-bytes'))
        expected = dict(present=True, sha256=hashlib.sha256(raw).hexdigest())
        assert qualification.model_extension_record(model) == expected
        assert raw == ('{"pp2:data":' + '1' + '0' * (limit + 1) + '}').encode('utf-8')
        from anygeometry.serialization import to_dict
        expected_body = dict(document=to_dict(model), extension_record=expected)
        expected_raw = json.dumps(expected_body, **options).encode('utf-8')
        packet = executor.freeze_model(model)
        assert packet == dict(schema='PP2-v1', kind='model', payload=expected_raw.decode('utf-8'),
                              sha256=hashlib.sha256(expected_raw).hexdigest())
    assert sys.get_int_max_str_digits() == limit
    assert model._serialization_extensions is extension and extension['pp2:data'] is value
    assert set(vars(model)) == set(before)
    assert all(vars(model)[key] is item for key, item in before.items())
