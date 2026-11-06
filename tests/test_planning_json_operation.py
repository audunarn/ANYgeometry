"""Current stage wire integration; JSON internals have no purity/profile waiver."""
import json
import pytest
from anygeometry import _planning_values as values
from anygeometry._planning_qualification import QualificationDecline


@pytest.mark.parametrize('ascii_only', [True, False])
def test_closed_stage_value_survives_actual_json_transport(ascii_only):
    shared = [1, 2.5, True, None, '\u03a9']
    value = {'first': shared, 'second': shared}
    wire = json.dumps(values.encode_value(value), sort_keys=True, separators=(',', ':'),
                      allow_nan=False, ensure_ascii=ascii_only)
    decoded = values.decode_value(json.loads(wire))
    assert decoded == value
    assert decoded['first'] is decoded['second']


def test_unknown_object_added_before_encoding_refuses_without_hooks():
    calls = []
    class Foreign:
        def __repr__(self):
            calls.append('repr')
            return 'foreign'
    value = {'x': [1]}
    value['x'].append(Foreign())
    with pytest.raises(QualificationDecline, match='unsupported type'):
        values.encode_value(value)
    assert calls == []


def test_unknown_codec_version_cannot_bypass_the_current_transport():
    with pytest.raises(QualificationDecline, match='codec version'):
        values.encode_value({'x': 1}, codec='foreign')


def test_large_integer_uses_current_wire_contract_not_obsolete_json_profile():
    value = {'x': 1 << 2048}
    wire = json.dumps(values.encode_value(value), allow_nan=False)
    assert values.decode_value(json.loads(wire)) == value


@pytest.mark.parametrize('value', [float('nan'), float('inf'), object()])
def test_current_wire_refuses_nonfinite_or_unknown_values(value):
    with pytest.raises(QualificationDecline):
        values.encode_value({'x': value})


def test_actual_stage_decode_retains_four_megabyte_payload_limit():
    wire = json.dumps(values.encode_value({'x': 'a' * 4_000_001}), separators=(',', ':'))
    with pytest.raises(QualificationDecline, match='payload limit'):
        values.decode_stage_values([wire])


@pytest.mark.parametrize('where', ['value', 'key'])
def test_ascii_surrogate_expansion_is_bounded_at_actual_stage_decode(where):
    text = '\U0001f600' * 333334
    value = {'x': text} if where == 'value' else {text: 0}
    wire = json.dumps(values.encode_value(value), ensure_ascii=True, separators=(',', ':'))
    assert len(wire.encode('utf-8')) > 4_000_000
    with pytest.raises(QualificationDecline, match='payload limit'):
        values.decode_stage_values([wire])


def test_non_ascii_stage_transport_preserves_supplementary_characters():
    value = {'x': '\U0001f600' * 1000}
    wire = json.dumps(values.encode_value(value), ensure_ascii=False, separators=(',', ':'))
    assert values.decode_stage_values([wire])[0] == value
