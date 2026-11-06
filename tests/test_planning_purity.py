"""PP1 prerequisites. The baseline has no admission API; each gate is required."""
import builtins
import contextvars
import importlib
import math
import sys
import types
import weakref
from collections import OrderedDict, deque
from functools import lru_cache

import numpy as np
import pytest


def api():
    return importlib.import_module('anygeometry._planning_purity')


def module(name='_pp1_fixture', source='def f(x):\n    return x + 1\n'):
    value = types.ModuleType(name)
    exec(compile(source, name + '.py', 'exec'), value.__dict__)
    return value


def test_warm_unlisted_weak_store_refuses_without_mutation():
    m = module()
    class Key:
        pass
    key = Key()
    m.memo = weakref.WeakKeyDictionary({key: 19})
    before = tuple(m.memo.items())
    p = api()
    report = p.inventory((m,))
    assert any(r.path == m.__name__ + '.memo' and r.decision == 'declined'
               for r in report.stores)
    with pytest.raises(p.PurityDecline, match='store'):
        p.require_pure(report)
    assert m.memo[key] == 19 and tuple(m.memo.items()) == before


@pytest.mark.parametrize('factory', [dict, list, set, OrderedDict, deque,
                                    weakref.WeakValueDictionary,
                                    lambda: contextvars.ContextVar('pp1')])
def test_actual_store_type_is_found_under_innocent_name(factory):
    m = module()
    m.harmless = factory()
    p = api()
    report = p.inventory((m,))
    assert any(r.path.endswith('.harmless') and r.decision == 'declined'
               for r in report.stores)
    with pytest.raises(p.PurityDecline):
        p.require_pure(report)


def test_function_dictionary_alias_and_wrapped_state():
    m = module()
    m.f.hidden = {}
    m.alias = m.f.hidden
    m.f.__wrapped__ = lambda: None
    m.f.__wrapped__.other = []
    p = api()
    report = p.inventory((m,))
    assert any('other' in r.path for r in report.stores)
    assert report.aliases
    with pytest.raises(p.PurityDecline):
        p.require_pure(report)


def test_unknown_descriptor_never_executes_and_slots_are_found():
    calls = []
    class Hidden:
        __slots__ = ('bank',)
        @property
        def __dict__(self):
            calls.append('bad')
            raise AssertionError('descriptor executed')
    m = module()
    m.object = Hidden()
    m.object.bank = {}
    p = api()
    report = p.inventory((m,))
    assert calls == []
    assert any('.bank' in r.path for r in report.stores)
    with pytest.raises(p.PurityDecline):
        p.require_pure(report)
    class CustomModule(types.ModuleType):
        pass
    with pytest.raises(p.PurityDecline, match='unsupported module'):
        p.ImportFreeze((CustomModule('unqualified'),))


def test_truncated_walk_cannot_be_admitted():
    p = api()
    report = p.inventory((module(),), max_nodes=1)
    assert report.truncated
    with pytest.raises(p.PurityDecline, match='truncated'):
        p.require_pure(report)


def test_signed_zero_directrix_first_witness_is_declined():
    from anygeometry import branch_algebra as a
    plus = a.BezierRuledSupport(((0., 0., 0.), (1., 1., 0.), (2., 0., 0.)), (0., 0., 1.))
    minus = a.BezierRuledSupport(((-0., 0., 0.), (1., 1., 0.), (2., 0., 0.)), (0., 0., 1.))
    assert plus == minus and hash(plus) == hash(minus)
    a.directrix_of.cache_clear()
    try:
        first = a.directrix_of(plus)
        hit = a.directrix_of(minus)
        own = a.directrix_of.__wrapped__(minus)
        assert hit is first
        assert math.copysign(1., hit.controls[0][0]) == 1.
        assert math.copysign(1., own.controls[0][0]) == -1.
        p = api()
        report = p.inventory((a,))
        assert any(r.path == a.__name__ + '.directrix_of' and r.decision == 'declined'
                   for r in report.stores)
        with pytest.raises(p.PurityDecline):
            p.require_pure(report)
    finally:
        a.directrix_of.cache_clear()


def test_signed_zero_bounds_first_witness_is_declined():
    from anygeometry import branch_curves as c
    class Curve:
        def evaluate(self, t):
            return np.array([math.copysign(1., t), 0., 0.])
    curve = Curve()
    c._cached_bounds.cache_clear()
    try:
        first = c._cached_bounds(curve, 0., 0.)
        hit = c._cached_bounds(curve, -0., -0.)
        own = c._cached_bounds.__wrapped__(curve, -0., -0.)
        assert hit is first and not np.array_equal(hit[0], own[0])
        p = api()
        report = p.inventory((c,))
        assert any(r.path == c.__name__ + '._cached_bounds' and r.decision == 'declined'
                   for r in report.stores)
        with pytest.raises(p.PurityDecline):
            p.require_pure(report)
    finally:
        c._cached_bounds.cache_clear()


def test_loaded_code_h0_uses_executing_objects_not_disk(tmp_path):
    source = tmp_path / 'h0.py'
    source.write_text('def f(x):\n    return x + 1\n')
    host = module(source=source.read_text())
    host.__file__ = str(source)
    p = api()
    before = p.python_token((host,))
    source.write_text('def f(x):\n    return x + 2\n')
    planner = module(source=source.read_text())
    planner.__file__ = str(source)
    assert p.python_token((host,)) == before
    with pytest.raises(p.PurityDecline, match='code-skew'):
        p.require_matching_python(before, (planner,))


def test_token_binds_defaults_closure_and_signed_zero():
    p = api()
    a = module(source='def f(x=0.0):\n    return x\n')
    b = module(source='def f(x=-0.0):\n    return x\n')
    assert p.python_token((a,)) != p.python_token((b,))
    a.f.__defaults__ = (-0.,)
    assert p.python_token((a,)) != p.python_token((module(),))
    def enclosing(x):
        def f():
            return x
        return f
    a.f = enclosing(1)
    b.f = enclosing(2)
    assert p.python_token((a,)) != p.python_token((b,))


def test_native_or_mutable_identity_cannot_be_attested():
    p = api()
    m = module()
    m.native = math.sin
    with pytest.raises(p.PurityDecline, match='native|unsupported'):
        p.python_token((m,))
    del m.native
    m.bank = {}
    with pytest.raises(p.PurityDecline):
        p.python_token((m,))


def test_freeze_blocks_new_import_even_if_failure_is_caught():
    p = api()
    original = builtins.__import__
    with pytest.raises(p.PurityDecline, match='import'):
        with p.ImportFreeze((math,)):
            try:
                importlib.import_module('_pp1_never_loaded')
            except p.PurityDecline:
                pass
    assert builtins.__import__ is original


def test_freeze_rejects_reload_and_preinserted_modules():
    p = api()
    with pytest.raises(p.PurityDecline, match='reload'):
        with p.ImportFreeze((math,)):
            try:
                importlib.reload(math)
            except p.PurityDecline:
                pass
    try:
        with pytest.raises(p.PurityDecline, match='module'):
            with p.ImportFreeze((math,)):
                sys.modules['_pp1_inserted'] = module('_pp1_inserted')
    finally:
        sys.modules.pop('_pp1_inserted', None)


def test_freeze_allows_attested_cached_import_and_restores_hooks():
    p = api()
    original = builtins.__import__
    with p.ImportFreeze((math,)):
        import math as cached
        assert cached is math
    assert builtins.__import__ is original


def test_eager_closure_finds_function_local_import_before_freeze(tmp_path, monkeypatch):
    p = api()
    (tmp_path / '_pp1_root.py').write_text('def f():\n    import _pp1_leaf\n    return _pp1_leaf.VALUE\n')
    (tmp_path / '_pp1_leaf.py').write_text('VALUE = 7\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        closure = p.eager_closure(('_pp1_root',), allowed_names=('_pp1_root', '_pp1_leaf'))
        assert {m.__name__ for m in closure} == {'_pp1_root', '_pp1_leaf'}
        with p.ImportFreeze(closure):
            assert sys.modules['_pp1_root'].f() == 7
    finally:
        sys.modules.pop('_pp1_root', None)
        sys.modules.pop('_pp1_leaf', None)


def test_eager_closure_declines_dynamic_import_capability():
    p = api()
    m = module(source='def f(name):\n    return __import__(name)\n')
    sys.modules[m.__name__] = m
    try:
        with pytest.raises(p.PurityDecline, match='dynamic'):
            p.eager_closure((m.__name__,), allowed_names=(m.__name__,))
    finally:
        sys.modules.pop(m.__name__, None)
