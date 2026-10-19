"""Internal copy reduction must not expose mutable feature state."""
from copy import deepcopy

import numpy as np
import pytest
from anygeometry import GeometryModel, GeometryError
from anygeometry.features import _validate_json_value


def test_clone_snapshots_restore_and_failed_history_edit_remain_independent():
    model = GeometryModel()
    model.add_point(1, 2, 3)
    model.features.capture_baseline(model)
    record = model.features.append("vendor.optional", parameters={"nested": [1, 2]}, suppressed=True)
    saved = model.design_snapshot()
    expected = deepcopy(saved["features"])
    clone = model.clone()
    clone.features.set_suppressed(record.feature_id, False)
    assert model.features.snapshot() == expected
    model.restore_design(saved)
    saved["features"]["baseline"]["vertices"].clear()
    saved["features"]["records"][0].parameters["nested"].append(3)
    assert model.features.snapshot() == expected
    assert clone.features.snapshot()["records"][0].suppressed is False

    invalid = model.features.snapshot()
    invalid["baseline"]["invalid"] = float("nan")
    with pytest.raises(GeometryError, match="finite"):
        model.features.restore(invalid)
    assert model.features.snapshot() == expected


@pytest.mark.parametrize("value", [None, True, 1, 1.5, "text", np.int64(4), np.float64(2.5),
                                   {"nested": [1, (False, 2.0)]}, np.array([1.0, 2.0])])
def test_feature_json_validation_retains_builtin_and_numpy_support(value):
    _validate_json_value(value, "feature")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), np.float64("nan"),
                                   {"nested": [float("inf")]}, {"": 1}, {1: 1}, object()])
def test_feature_json_validation_refuses_invalid_payloads(value):
    with pytest.raises(GeometryError):
        _validate_json_value(value, "feature")
