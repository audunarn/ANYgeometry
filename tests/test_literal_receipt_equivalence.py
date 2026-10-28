"""_literal receipt encoding must stay identical (including refusals)."""
from enum import Enum
from fractions import Fraction
from uuid import UUID

import pytest

from anygeometry import prepared_material_curvilinear as pmc
from anygeometry.errors import GeometryError


def _reference(value):
    kind = type(value)
    if value is None or kind in (bool, int, float, str):
        return kind.__name__, value
    if kind is pmc.Q:
        return 'Fraction', value.numerator, value.denominator
    if kind is UUID:
        return 'UUID', value.hex
    if isinstance(value, Enum):
        return type(value).__qualname__, _reference(value.value)
    if kind is tuple:
        return tuple(_reference(v) for v in value)
    raise GeometryError('x')


class _E(Enum):
    A = 1
    B = (Fraction(1, 3), 'q')


class _S(str):
    pass


VALUES = [None, True, 0, -3, 2.5, 'a', Fraction(-3, 7), UUID(int=5), _E.A, _E.B, (),
          (1, (2, (Fraction(1, 2), None, 'x', True)), ()), tuple(range(50)),
          (Fraction(1, 3),) * 9, (UUID(int=1), _E.A, (2.0, False))]


@pytest.mark.parametrize('value', VALUES)
def test_literal_matches_reference(value):
    assert pmc._literal(value) == _reference(value)
    assert repr(pmc._literal(value)) == repr(_reference(value))


@pytest.mark.parametrize('bad', [[1], {1: 2}, object(), _S('s'), (1, [2]), (1, (2, {3})), 1j, b'x'])
def test_literal_refuses_unsupported_like_reference(bad):
    with pytest.raises(GeometryError):
        pmc._literal(bad)
