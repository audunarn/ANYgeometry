"""Receipt encoding and pin reads retain their original observable dispatch."""
from dataclasses import fields
from enum import Enum
from fractions import Fraction as Q
from uuid import UUID

import pytest

from anygeometry import prepared_material_curvilinear as module
from anygeometry.errors import GeometryError


def reference_literal(value):
    kind = type(value)
    if value is None or kind in (bool, int, float, str):
        return kind.__name__, value
    if kind is Q:
        return 'Fraction', value.numerator, value.denominator
    if kind is UUID:
        return 'UUID', value.hex
    if isinstance(value, Enum):
        return type(value).__qualname__, reference_literal(value.value)
    if kind is tuple:
        return tuple(reference_literal(v) for v in value)
    if kind in (module.EntityHandle, module.BoundaryChordOccurrence, module.MaterialBoundaryChord,
                module.PreparedMaterialBoundaryChords, module.PreparedMaterialBoundarySampling,
                module.CurvilinearMaterialCell, module.PreparedMaterialCurvilinearCells):
        return kind.__name__, tuple((f.name, reference_literal(getattr(value, f.name))) for f in fields(value))
    raise GeometryError('curvilinear receipt contains a nonliteral or unsupported field')


def test_literal_value_repr_and_supported_receipts():
    class Choice(Enum):
        VALUE = 3
    value = (None, True, 42, -0., 'literal', Q(7, 9), UUID(int=73), Choice.VALUE,
             (('nested',), ()))
    assert module._literal(value) == reference_literal(value)
    assert repr(module._literal(value)) == repr(reference_literal(value))
    for kind in (module.BoundaryChordOccurrence, module.MaterialBoundaryChord,
                 module.PreparedMaterialBoundaryChords, module.PreparedMaterialBoundarySampling,
                 module.CurvilinearMaterialCell, module.PreparedMaterialCurvilinearCells):
        receipt = kind(**{field.name: value for field in fields(kind)})
        assert module._literal(receipt) == reference_literal(receipt)
        assert repr(module._literal(receipt)) == repr(reference_literal(receipt))


def test_literal_custom_metaclass_dispatch_occurs_once():
    calls = []
    class Observed(type):
        def __eq__(cls, other):
            calls.append(other.__name__)
            return False
    class Unsupported(metaclass=Observed):
        pass
    for value in (Unsupported(), (Unsupported(),), ((Unsupported(),),)):
        calls.clear()
        with pytest.raises(GeometryError) as reference:
            reference_literal(value)
        expected = calls[:]
        calls.clear()
        with pytest.raises(GeometryError) as actual:
            module._literal(value)
        assert str(actual.value) == str(reference.value)
        assert calls == expected
    assert not hasattr(module, '_LITERAL_SCALARS')


@pytest.mark.parametrize('keywords', [None, {}, {'z': object(), 'a': object()}])
def test_function_pin_reads_in_original_order(keywords):
    calls = []
    code, defaults = object(), object()
    class Function:
        def __getattribute__(self, name):
            calls.append(name)
            return {'__code__': code, '__defaults__': defaults, '__kwdefaults__': keywords}[name]
    function = Function()
    result = module._chord_function_pin(function)
    assert calls == ['__code__', '__defaults__', '__kwdefaults__']
    assert result == (function, code, defaults,
                      tuple((key, id(value)) for key, value in sorted((keywords or {}).items())))


def test_pin_stops_at_first_attribute_failure():
    calls = []
    class Function:
        def __getattribute__(self, name):
            calls.append(name)
            raise RuntimeError('attribute read failed')
    with pytest.raises(RuntimeError, match='attribute read failed'):
        module._chord_function_pin(Function())
    assert calls == ['__code__']
