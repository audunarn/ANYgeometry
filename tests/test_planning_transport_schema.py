"""PP2 release 2 exact decoder-owned state contract; no omitted state."""
import pytest
from anygeometry import _planning_qualification as q
from anygeometry._planning_executor import freeze_model, thaw_model


def cold_model():
    from anygeometry._planning_canaries import _fixture
    from anygeometry.serialization import from_dict, to_dict
    model, _ = _fixture()
    return from_dict(to_dict(model))


@pytest.mark.parametrize('extension', [{}, {'pp2:data': [None, True, 2, -0.0, 'value', {'z': 3.0}]}])
def test_exact_extension_schema_records_digest_without_mutation(extension):
    model = cold_model()
    model._serialization_extensions = extension
    record = q.model_extension_record(model)
    assert record['present'] and len(record['sha256']) == 64
    assert model._serialization_extensions is extension
    packet = freeze_model(model)
    restored = thaw_model(packet)
    assert q.model_extension_record(restored) == record


@pytest.mark.parametrize('extension', [{'x': object()}, [], {'x': float('nan')}, {1: 'bad-key'}])
def test_unknown_or_wrong_extension_type_declines(extension):
    model = cold_model(); model._serialization_extensions = extension
    with pytest.raises(q.QualificationDecline): freeze_model(model)


@pytest.mark.parametrize('kind', ['within', 'model-state', 'cycle'])
def test_extension_alias_declines(kind):
    model = cold_model(); shared = []
    if kind == 'within': model._serialization_extensions = {'a': shared, 'b': shared}
    elif kind == 'model-state': model._serialization_extensions = model._arc_cache
    else:
        shared.append(shared); model._serialization_extensions = {'a': shared}
    with pytest.raises(q.QualificationDecline, match='alias|cycle'): freeze_model(model)


def test_extra_other_key_declines_even_with_valid_extension():
    model = cold_model(); model.other_state = {}
    with pytest.raises(q.QualificationDecline, match='non-field'): freeze_model(model)


def test_extension_digest_distinguishes_signed_zero():
    model = cold_model(); model._serialization_extensions = {'v': 0.0}
    first = q.model_extension_record(model)
    model._serialization_extensions['v'] = -0.0
    assert q.model_extension_record(model)['sha256'] != first['sha256']


def test_model_roundtrip_checks_canonical_float_bits(monkeypatch):
    # A decoder regression must not hide behind Python's -0.0 == +0.0.
    from copy import deepcopy
    from anygeometry import serialization
    from anygeometry._planning_executor import pack
    model = cold_model()
    document = deepcopy(serialization.to_dict(model))
    document['vertices'][0]['position'][0] = -0.0
    packet = pack(dict(document=document, extension_record=q.model_extension_record(model)), 'model')
    monkeypatch.setattr(serialization, 'from_dict', lambda ignored:model)
    with pytest.raises(q.QualificationDecline, match='values'):
        thaw_model(packet)
