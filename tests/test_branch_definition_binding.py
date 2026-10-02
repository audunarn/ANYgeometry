"""Bind branch geometry without binding its explicitly marked inversion cache."""

from dataclasses import dataclass, field, replace

import numpy as np
import pytest

from anygeometry.branch_algebra import BezierRuledSupport
from anygeometry.branch_curves import BezierQuadricCurve
from anygeometry.definition_binding import definition_checksum, definition_value
from anygeometry.quadric_algebra import QuadricSupport


def branch_curve():
    first = BezierRuledSupport(((0., 0., 0.), (.5, 0., 0.), (1., 1., 0.)), (0., 0., 1.))
    second = QuadricSupport("general", matrix=(0., 0., 0., 0., 1., 0., 0., 0., 1.),
                            linear=(0., 0., 0.), constant=-4.)
    return BezierQuadricCurve(first, second, .1, .8)


def test_point_inversion_cache_miss_preserves_definition_binding():
    curve = branch_curve()
    before = definition_checksum(curve)
    assert not curve._inversions
    point = curve.evaluate(.37)
    assert curve.parameters_of(point) == pytest.approx((.37,), abs=1e-14)
    assert len(curve._inversions) == 1
    assert definition_checksum(curve) == before
    assert "_inversions" not in definition_value(curve)["fields"]


@pytest.mark.parametrize("change", [
    lambda curve: replace(curve, start=.2, sweep=.7),
    lambda curve: replace(curve, sweep=.5),
    lambda curve: replace(curve, branch=-1),
    lambda curve: replace(curve, second=replace(curve.second, constant=-9.)),
    lambda curve: replace(curve, first=replace(curve.first, direction=(0., 0., 2.))),
    lambda curve: curve.transformed(np.asarray(((1., 0., 0., 1.), (0., 1., 0., 0.),
                                               (0., 0., 1., 0.), (0., 0., 0., 1.)))),
])
def test_true_geometry_changes_remain_bound(change):
    curve = branch_curve()
    assert definition_checksum(change(curve)) != definition_checksum(curve)


def test_noncomparing_fields_remain_bound_without_explicit_exclusion():
    @dataclass(frozen=True)
    class Definition:
        value: int = field(compare=False)

    assert Definition(1) == Definition(2)
    assert definition_checksum(Definition(1)) != definition_checksum(Definition(2))
