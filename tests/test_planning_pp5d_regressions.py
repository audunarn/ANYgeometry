"""Owner-authorized PP5d regressions; no relaxation of admission predicates."""
import sys
from types import SimpleNamespace

import pytest

from anygeometry import _planning_batch as batch
from anygeometry import _planning_executor as executor
from anygeometry import _planning_qualification as qualification
from anygeometry.model import GeometryModel


def huge_integer_model():
    limit = sys.get_int_max_str_digits()
    assert limit > 0, 'this regression requires the configured finite interpreter limit'
    model = GeometryModel()
    model._serialization_extensions = {'pp2:data': 10 ** (limit + 1)}
    return model


@pytest.mark.parametrize('operation', [qualification.model_extension_record, executor.freeze_model])
def test_oversized_integer_encoding_is_typed_refusal(operation):
    model = huge_integer_model()
    value = model._serialization_extensions['pp2:data']
    limit = sys.get_int_max_str_digits()
    with pytest.raises(qualification.QualificationDecline, match='extension encoding refused'):
        operation(model)
    assert sys.get_int_max_str_digits() == limit
    assert model._serialization_extensions['pp2:data'] is value


def test_host_codec_refusal_retires_then_uses_original_serial(tmp_path):
    model = huge_integer_model()
    events = []

    class CodecPlanner:
        def submit(self, original, tasks):
            assert original is model
            return executor.freeze_model(original)

        def close(self):
            events.append('retired')

    result = executor.execute(model, ('pair',), lambda task: events.append(task) or 'serial-value',
                              directory=tmp_path / 'unused', planner=CodecPlanner(), opt_in=True)
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
