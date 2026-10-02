"""Per-call reuse preserves the original definition and checksum contract."""

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import hashlib
import json
from uuid import UUID

import numpy as np
import pytest

from anygeometry.definition_binding import definition_checksum, definition_value


def reference_value(value):
    """Independent copy of the converter before memo reuse."""
    if isinstance(value, np.ndarray):
        return {"array": value.tolist(), "shape": list(value.shape)}
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {"type": type(value).__qualname__, "fields": {
            item.name: reference_value(getattr(value, item.name)) for item in fields(value)
            if item.metadata.get("definition") is not False}}
    if isinstance(value, Mapping):
        return {str(key): reference_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [reference_value(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, UUID):
        return str(value)
    return value


def payload(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def assert_original_binding(value):
    original = reference_value(value)
    expanded = definition_value(value)
    assert expanded == original
    assert payload(expanded) == payload(original)
    assert definition_checksum(value) == hashlib.sha256(payload(original)).hexdigest()


class Kind(Enum):
    CURVE = "curve"


@dataclass(frozen=True)
class Definition:
    points: object
    kind: Kind = Kind.CURVE
    identifier: UUID = UUID("00000000-0000-0000-0000-000000000001")
    gain: object = np.float64(1.25)
    hidden_value: int = field(default=3, compare=False, repr=False)
    cache: dict = field(default_factory=dict, metadata={"definition": False})


def test_nested_aliased_definitions_match_original_bytes():
    array = np.asarray(((1., 2., 3.), (4., 5., 6.)))
    definition = Definition(array, cache={"nonfinite": float("nan")})
    sequence = [definition, array]
    paths = (sequence, definition)
    mapping = {7: paths, "array": array, "definition": definition}
    assert_original_binding([mapping, mapping, paths, sequence, definition, array])
    converted = definition_value(definition)["fields"]
    assert "cache" not in converted
    assert converted["hidden_value"] == 3


@pytest.mark.parametrize("shared", [
    np.asarray(((1., 2.), (3., 4.))),
    Definition((1., 2.)),
    {"points": [1., 2.]},
    [1., 2.],
    (1., 2.),
])
def test_public_expansions_remain_independent_for_each_expanded_kind(shared):
    assert_original_binding([shared, shared])
    converted = definition_value([shared, shared])
    assert converted[0] is not converted[1]
    if isinstance(converted[0], dict):
        converted[0]["tampered"] = True
    else:
        converted[0].append("tampered")
    assert converted[1] == reference_value(shared)


class CountingMapping(dict):
    expansions = 0

    def items(self):
        self.expansions += 1
        return super().items()


def test_only_checksums_reuse_mapping_traversal_with_a_fresh_memo():
    shared = CountingMapping(points=[1., 2.])
    definition = Definition(shared)
    evidence = [definition, definition, shared, {"alias": shared}]
    definition_value(evidence)
    assert shared.expansions == 4
    definition_checksum(evidence)
    assert shared.expansions == 5
    definition_value(evidence)
    assert shared.expansions == 9
    definition_checksum(evidence)
    assert shared.expansions == 10


@pytest.mark.parametrize("array", [
    np.asarray(3, dtype=np.int64),
    np.asarray([], dtype=np.float64).reshape(0, 2),
    np.asarray((0, 255), dtype=np.uint8),
    np.asarray((1.25, -0.), dtype=np.float32),
    np.asarray((True, False)),
])
def test_array_shape_dtype_and_scalar_contract_match_original(array):
    assert_original_binding({"array": array, "alias": array, "scalar": array.dtype.type(1)})


def test_mutations_between_calls_cannot_reuse_previous_content():
    array = np.asarray((1., 2.))
    sequence = [array]
    mapping = {"path": sequence}
    definition = Definition(mapping)
    evidence = [definition, definition]
    before = definition_checksum(evidence)
    old_output = definition_value(evidence)
    array[0] = 9.
    sequence.append(np.int64(4))
    mapping["new"] = (7., 8.)
    assert_original_binding(evidence)
    assert definition_checksum(evidence) != before
    assert old_output[0]["fields"]["points"]["path"][0]["array"] == [1., 2.]
    old_output[0]["fields"]["points"].clear()
    assert_original_binding(evidence)
    before_cache = definition_checksum(evidence)
    definition.cache["inverse"] = float("inf")
    assert definition_checksum(evidence) == before_cache


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("container", ["scalar", "array", "numpy_scalar"])
def test_nonfinite_after_successful_call_is_still_refused(nonfinite, container):
    shared = {"value": 1.}
    evidence = [shared, shared]
    assert_original_binding(evidence)
    shared["value"] = {
        "scalar": nonfinite,
        "array": np.asarray((nonfinite,)),
        "numpy_scalar": np.float64(nonfinite),
    }[container]
    with pytest.raises(ValueError):
        payload(reference_value(evidence))
    with pytest.raises(ValueError):
        definition_checksum(evidence)


def test_recursive_input_remains_rejected():
    recursive = []
    recursive.append(recursive)
    with pytest.raises(RecursionError):
        reference_value(recursive)
    with pytest.raises(RecursionError):
        definition_value(recursive)
