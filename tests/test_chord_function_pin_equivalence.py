"""The pin used by every fail-closed calculator guard must keep its exact value."""
import functools
import math

from anygeometry import prepared_material_curvilinear as pmc


def _reference_pin(function):
    return (function, getattr(function, '__code__', None),
            getattr(function, '__defaults__', None),
            tuple((key, id(value)) for key, value in
                  sorted((getattr(function, '__kwdefaults__', None) or {}).items())))


def _plain(a, b=1, *, z=2, c=3): return a
def _bare(a): return a
class _K:
    def m(self, x=(1,), *, k=None): return x
    @classmethod
    def c(cls, *, q=5, a=1): return q
    @staticmethod
    def s(x=2): return x


def _samples():
    yield from (_plain, _bare, _K.m, _K().m, _K.c.__func__, _K.c, _K.s, math.isqrt,
                len, functools.lru_cache(None)(_bare), functools.partial(_bare),
                lambda: 0, int, object(), None)


def test_pin_identical_on_varied_callables():
    for f in _samples():
        assert pmc._chord_function_pin(f) == _reference_pin(f)
        assert type(pmc._chord_function_pin(f)[3]) is tuple


def test_pin_detects_code_defaults_and_kwdefault_mutation():
    def f(a=1, *, z=2): return a
    before = pmc._chord_function_pin(f)
    f.__defaults__ = (2,)
    assert pmc._chord_function_pin(f) != before
    f.__defaults__ = (1,)
    assert pmc._chord_function_pin(f) == before
    f.__kwdefaults__ = {'z': 3}
    assert pmc._chord_function_pin(f) != before
    f.__kwdefaults__ = {'z': 2}
    assert pmc._chord_function_pin(f) == _reference_pin(f)
    f.__code__ = _bare.__code__.replace()
    assert pmc._chord_function_pin(f)[1] is f.__code__ and pmc._chord_function_pin(f) != before


def test_calculator_context_equals_context_built_with_reference_pin(monkeypatch):
    new = pmc._cell_reference_calculator_context()
    monkeypatch.setattr(pmc, '_chord_function_pin', _reference_pin)
    assert pmc._cell_reference_calculator_context() == new
