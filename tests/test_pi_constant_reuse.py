"""Exact precision constants retain interval, accounting and refusal semantics."""
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import pytest
import anygeometry.exact_predicates as ep


def legacy_pi(engine):
    # Frozen original Machin procedure, independent of the reuse implementation.
    if engine._pi is None:
        engine._pi = engine.sub(
            engine.mul(16, engine._atan_series(engine.box(Fraction(1, 5)))),
            engine.mul(4, engine._atan_series(engine.box(Fraction(1, 239)))))
    return engine._pi


@pytest.fixture(autouse=True)
def clear_constants():
    with ep._PI_CACHE_LOCK:
        ep._PI_CACHE.clear()
    yield
    with ep._PI_CACHE_LOCK:
        ep._PI_CACHE.clear()


def _cold_warm_body(level):
    oracle = ep.Canonical(level)
    expected = legacy_pi(oracle)
    cold, warm = ep.Canonical(level), ep.Canonical(level)
    assert cold.pi() == warm.pi() == expected
    assert cold._pi is warm._pi  # prove the native reuse path activated
    assert cold.operations == warm.operations == oracle.operations
    assert cold.pi() is cold._pi and warm.pi() is warm._pi
    assert cold.operations == oracle.operations  # repeated local access is free
    with pytest.raises(AttributeError):
        warm._pi.lo = Fraction(0)


_PI_CHILD=r'''
import copyreg,json,sys
from pathlib import Path
sys.path[:0]=[sys.argv[3],sys.argv[4]]
import test_pi_constant_reuse as contract
ep=contract.ep
assert Path(ep.__file__).resolve().is_relative_to(Path(sys.argv[3]).resolve())
assert Path(contract.__file__).resolve()==Path(sys.argv[4]).resolve()/'test_pi_constant_reuse.py'
level=int(sys.argv[1]);scenario=sys.argv[2]
assert ep._pi_cache_key(ep.Canonical(level)) is not None, 'pristine PI reuse guard refused'
contract._cold_warm_body(level)
report=dict(scope='clean-child-exact-PI-reuse-unit',level=level,scenario=scenario,
    executable=sys.executable,version=sys.version,origin=ep.__file__,
    original_positive_body_executed=True,pristine_reuse_guard_admitted=True)
if scenario=='derived_metadata':
    original={cls:dict(vars(cls)) for cls in (ep.Box,ep.Fraction,ep.Canonical)}
    slots=copyreg._slotnames(ep.Box)
    assert slots==['lo','hi'] and type(slots) is list
    assert set(vars(ep.Box))-set(original[ep.Box])=={'__slotnames__'}
    for cls,namespace in original.items():
        assert all(vars(cls).get(name) is value for name,value in namespace.items())
        if cls is not ep.Box:assert set(vars(cls))==set(namespace)
    assert ep._pi_cache_key(ep.Canonical(level)) is None
    oracle=ep.Canonical(level);expected=contract.legacy_pi(oracle)
    cold,warm=ep.Canonical(level),ep.Canonical(level)
    assert cold.pi()==warm.pi()==expected
    assert cold._pi is not warm._pi
    assert cold.operations==warm.operations==oracle.operations
    assert cold.pi() is cold._pi and warm.pi() is warm._pi
    assert cold.operations==warm.operations==oracle.operations
    with contract.pytest.raises(AttributeError):warm._pi.lo=contract.Fraction(0)
    report.update(guard_after_metadata=ep._pi_cache_key(cold),
        identity_reused_after_metadata=cold._pi is warm._pi,
        changed_namespace_keys=['__slotnames__'],operations=oracle.operations,
        exact_interval=[[expected.lo.numerator,expected.lo.denominator],
                        [expected.hi.numerator,expected.hi.denominator]])
elif scenario!='pristine':raise AssertionError('unknown PI unit scenario')
print(json.dumps(report))
'''


def _run_pi_child(tmp_path,request,level,scenario):
    """Pristine reuse is a child unit contract; host metadata stays untouched."""
    root=Path(__file__).resolve().parents[1]
    namespaces={cls:dict(vars(cls)) for cls in (ep.Box,ep.Fraction,ep.Canonical)}
    command=[sys.executable,'-B','-c',_PI_CHILD,str(level),scenario,str(root/'src'),str(root/'tests')]
    try:
        result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=15)
    except subprocess.TimeoutExpired as error:
        for name,value in (('stdout',error.stdout),('stderr',error.stderr)):
            text=value.decode('utf-8','replace') if isinstance(value,bytes) else value or ''
            (tmp_path/(name+'.txt')).write_text(text,encoding='utf-8')
        raise
    finally:
        for cls,namespace in namespaces.items():
            assert set(vars(cls))==set(namespace)
            assert all(vars(cls)[name] is value for name,value in namespace.items())
    (tmp_path/'stdout.txt').write_text(result.stdout,encoding='utf-8')
    (tmp_path/'stderr.txt').write_text(result.stderr,encoding='utf-8')
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads(result.stdout)
    assert report['level']==level and report['scenario']==scenario
    assert report['original_positive_body_executed'] is True
    assert report['pristine_reuse_guard_admitted'] is True
    (tmp_path/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    request.node.user_properties.extend((('execution_scope',report['scope']),
        ('original_test_body_executed',True),('host_namespaces_preserved',True),
        ('pristine_reuse_guard_admitted',True)))
    return report


@pytest.mark.parametrize('level', range(ep.L_MAX+1))
def test_cold_warm_intervals_and_charges_are_exact(level,tmp_path,request):
    _run_pi_child(tmp_path,request,level,'pristine')


@pytest.mark.parametrize('level',[0,2,4])
def test_real_derived_metadata_declines_reuse_with_exact_legacy_accounting(level,tmp_path,request):
    report=_run_pi_child(tmp_path,request,level,'derived_metadata')
    assert report['guard_after_metadata'] is None
    assert report['identity_reused_after_metadata'] is False
    assert report['changed_namespace_keys']==['__slotnames__']


@pytest.mark.parametrize('remaining', [0, 1, 5, 10])
def test_warm_reuse_exhausts_at_same_operation(remaining):
    ep.Canonical(0).pi()
    actual, oracle = ep.Canonical(0), ep.Canonical(0)
    actual.operations = oracle.operations = ep._MAX_OPERATIONS-remaining
    with pytest.raises(ep._BudgetExhausted):
        actual.pi()
    with pytest.raises(ep._BudgetExhausted):
        legacy_pi(oracle)
    assert actual.operations == oracle.operations == ep._MAX_OPERATIONS+1
    assert actual._pi is oracle._pi is None


def test_changed_budget_does_not_bypass_warm_charge(monkeypatch):
    ep.Canonical(0).pi()
    monkeypatch.setattr(ep, '_MAX_OPERATIONS', 2)
    actual, oracle = ep.Canonical(0), ep.Canonical(0)
    for call, engine in ((actual.pi, actual), (lambda: legacy_pi(oracle), oracle)):
        with pytest.raises(ep._BudgetExhausted):
            call()
        assert engine.operations == 3 and engine._pi is None


def test_failed_cold_computation_is_not_shared(monkeypatch):
    with monkeypatch.context() as local:
        local.setattr(ep, '_MAX_OPERATIONS', 1)
        with pytest.raises(ep._BudgetExhausted):
            ep.Canonical(0).pi()
    assert not ep._PI_CACHE
    actual, oracle = ep.Canonical(0), ep.Canonical(0)
    assert actual.pi() == legacy_pi(oracle)
    assert actual.operations == oracle.operations


def test_class_arithmetic_replacement_declines_reuse(monkeypatch):
    ep.Canonical(0).pi()
    monkeypatch.setattr(ep.Canonical, '_atan_series', lambda self, x: self.box(0))
    actual, oracle = ep.Canonical(0), ep.Canonical(0)
    assert actual.pi() == legacy_pi(oracle) == ep.Box(Fraction(0), Fraction(0))
    assert actual.operations == oracle.operations


def test_warm_box_constructor_replacement_is_not_bypassed(monkeypatch):
    ep.Canonical(0).pi()
    marker = RuntimeError('custom Box construction')
    def refuse(*args, **kwargs):
        raise marker
    monkeypatch.setattr(ep.Box, '__init__', refuse)
    actual, oracle = ep.Canonical(0), ep.Canonical(0)
    for call in (actual.pi, lambda: legacy_pi(oracle)):
        with pytest.raises(RuntimeError) as caught:
            call()
        assert caught.value is marker
    assert actual.operations == oracle.operations == 0


def test_custom_budget_comparison_uses_legacy_arithmetic(monkeypatch):
    ep.Canonical(0).pi()
    active = [None]
    class Budget(int):
        def __lt__(self, operation):
            if operation == 1:
                active[0].grid *= 2
            return int(self) < operation
    monkeypatch.setattr(ep, '_MAX_OPERATIONS', Budget(ep._MAX_OPERATIONS))
    actual, oracle = ep.Canonical(0), ep.Canonical(0)
    active[0] = actual
    result = actual.pi()
    active[0] = oracle
    assert result == legacy_pi(oracle)
    assert actual.operations == oracle.operations
    assert actual.grid == oracle.grid


def test_instance_hook_and_subclass_retain_original_route():
    ep.Canonical(0).pi()
    actual, oracle = ep.Canonical(0), ep.Canonical(0)
    for engine in (actual, oracle):
        engine._atan_series = lambda x, e=engine: e.box(0)
    assert actual.pi() == legacy_pi(oracle)
    assert actual.operations == oracle.operations

    class Custom(ep.Canonical):
        def _atan_series(self, x):
            return self.box(1)
    actual, oracle = Custom(0), Custom(0)
    assert actual.pi() == legacy_pi(oracle)
    assert actual.operations == oracle.operations


def test_changed_instance_precision_is_not_misidentified():
    ep.Canonical(0).pi()
    actual, oracle = ep.Canonical(0), ep.Canonical(0)
    for engine in (actual, oracle):
        engine.bits += 1
        engine.grid *= 2
    assert actual.pi() == legacy_pi(oracle)
    assert actual.operations == oracle.operations


def test_parallel_reuse_keeps_exact_results_and_accounting():
    expected = []
    for level in range(ep.L_MAX+1):
        engine = ep.Canonical(level)
        expected.append((legacy_pi(engine), engine.operations))
    def compute(level):
        engine = ep.Canonical(level)
        return engine.pi(), engine.operations
    levels = list(range(ep.L_MAX+1))*4
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(compute, levels))
    assert results == [expected[level] for level in levels]
    assert len(ep._PI_CACHE) <= 8
