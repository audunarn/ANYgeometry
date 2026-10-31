import importlib
import math
from copy import deepcopy
from functools import lru_cache
import types
import hashlib
from pathlib import Path

import numpy as np
import pytest


def api():
    return importlib.import_module('anygeometry._planning_qualification')


def pin():
    own = {'kind': 'python', 'path': str(Path(__file__).resolve()),
           'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return {'schema': 'PP1b-v1', 'modules': {'m': {'kind': 'python', 'sha256': 'a'*64}, __name__: own},
            'files': {'m.py': 'a'*64}, 'versions': {'numpy': 'test', 'scipy': 'test'},
            'python': {'version': 'test'}, 'cpu': {'machine': 'test'},
            'blas': [{'internal_api': 'openblas', 'num_threads': 1, 'sha256': 'b'*64}],
            'types': ['builtins.int', 'builtins.tuple', 'builtins.dict', 'builtins.str'],
            'settings': {'mode': 'exact'}, 'reset_registry': {}}


@pytest.mark.parametrize('change,reason', [
    (lambda p: p['modules'].update(extra={'kind': 'native', 'sha256': 'c'*64}), 'module'),
    (lambda p: p['files'].update({'m.py': 'c'*64}), 'hash'),
    (lambda p: p['blas'][0].update(num_threads=2), 'thread'),
    (lambda p: p['blas'].clear(), 'thread'),
    (lambda p: p['cpu'].update(machine='other'), 'cpu'),
])
def test_environment_negatives(change, reason):
    p = api()
    expected = pin()
    observed = deepcopy(expected)
    change(observed)
    with pytest.raises(p.QualificationDecline, match=reason):
        p.verify_environment(expected, observed)


def test_unknown_type_declines_even_when_its_module_name_is_pinned():
    p = api()
    with pytest.raises(p.QualificationDecline, match='type'):
        p.check_types(object(), pin())
    p.check_types((1, 2), {**pin(), 'types': ['builtins.int', 'builtins.tuple']})


def test_reset_covers_all_stores_and_refuses_unknown_before_mutating():
    p = api()
    with pytest.raises(p.QualificationDecline, match='reset registry'):
        p.WorkerGate(pin(), stores={'omitted': {}}, owned_worker=True, observe=pin)
    first = {'warm': 1}
    with pytest.raises(p.QualificationDecline, match='reset'):
        p.reset_declined({'first': first, 'unknown': object()}, owned_worker=True)
    assert first == {'warm': 1}
    with pytest.raises(p.QualificationDecline, match='owned'):
        p.reset_declined({'first': first}, owned_worker=False)
    assert first == {'warm': 1}
    second = [1]
    counts = p.reset_declined({'first': first, 'second': second}, owned_worker=True)
    assert first == {} and second == [] and counts == {'first': 0, 'second': 0}


def test_signed_zero_lrus_reset_between_tasks():
    from anygeometry import branch_algebra as a, branch_curves as c
    plus = a.BezierRuledSupport(((0.,0.,0.),(1.,1.,0.),(2.,0.,0.)),(0.,0.,1.))
    minus = a.BezierRuledSupport(((-0.,0.,0.),(1.,1.,0.),(2.,0.,0.)),(0.,0.,1.))
    class Curve:
        def evaluate(self, t):
            return np.array([math.copysign(1., t), 0., 0.])
    curve = Curve()
    caches = {'directrix_of': a.directrix_of, '_cached_bounds': c._cached_bounds,
              '_cached_poly_plan': a._cached_poly_plan}
    for cache in caches.values():
        cache.cache_clear()
    try:
        first = a.directrix_of(plus)
        assert a.directrix_of(minus) is first
        bounds = c._cached_bounds(curve, 0., 0.)
        assert c._cached_bounds(curve, -0., -0.) is bounds
        # PolyPlan retains its first support; no fabricated replacement kernel.
        from anygeometry.surfaces import Plane
        from anygeometry.quadric_algebra import QuadricSupport
        plane = QuadricSupport.from_surface(Plane((0., 0., 0.), (1., 0., 0.), (0., 0., 1.)))
        poly = a._cached_poly_plan(plus, plane)
        assert a._cached_poly_plan(minus, plane) is poly
        p = api()
        assert p.reset_declined(caches, owned_worker=True) == dict.fromkeys(caches, 0)
        negative = a.directrix_of(minus)
        assert math.copysign(1., negative.controls[0][0]) == -1.
        assert not np.array_equal(c._cached_bounds(curve, -0., -0.)[0], bounds[0])
        assert a._cached_poly_plan(minus, plane) is not poly
    finally:
        for cache in caches.values():
            cache.cache_clear()


def test_canary_mismatch_poisoned_gate_never_executes_task():
    p = api()
    calls = []
    gate = p.WorkerGate(pin(), observe=pin, stores={}, owned_worker=True)
    with pytest.raises(p.QualificationDecline, match='canary'):
        gate.start({'pair': lambda: 'wrong', 'face': lambda: 'face'},
                   {'pair': 'expected', 'face': 'face'})
    with pytest.raises(p.QualificationDecline):
        gate.run(lambda: calls.append(1))
    assert calls == []


def test_every_task_is_reset_and_postcheck_failure_discards_result():
    p = api()
    cache = {'old': 1}
    observation = pin()
    observation['reset_registry'] = {'cache': 'builtins.dict'}
    gate = p.WorkerGate(deepcopy(observation), observe=lambda: deepcopy(observation),
                        stores={'cache': cache}, owned_worker=True)
    gate.start({'pair': lambda: 'ok', 'face': lambda: 'ok'}, {'pair': 'ok', 'face': 'ok'})
    def task():
        assert cache == {}
        cache['new'] = 2
        return 7
    assert gate.run(task) == 7
    assert gate.run(task) == 7
    def altered():
        observation['files']['m.py'] = 'f'*64
        return 'must not escape'
    with pytest.raises(p.QualificationDecline, match='hash'):
        gate.run(altered)


def test_ci_control_requires_both_plan_and_applied_digests():
    p = api()
    p.check_ci_equality(('plan', 'applied'), ('plan', 'applied'))
    with pytest.raises(p.QualificationDecline, match='applied'):
        p.check_ci_equality(('plan', 'applied'), ('plan', 'wrong'))
    with pytest.raises(p.QualificationDecline):
        p.check_ci_equality(('plan', None), ('plan', None))


def test_capture_detects_changed_real_file(tmp_path):
    p = api()
    path = tmp_path/'pinned.py'
    path.write_text('VALUE = 1\n')
    m = types.ModuleType('_pp1b_pin')
    m.__file__ = str(path)
    before = p.pin_module(m)
    path.write_text('VALUE = 2\n')
    assert before['sha256'] != p.pin_module(m)['sha256']
