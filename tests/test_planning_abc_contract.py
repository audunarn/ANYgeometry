"""Finite ABC unit contracts in clean children; the loaded host is untouched.

Other tests may legitimately import mpmath and register more numeric classes.
That initial refusal is not selected-predicate poison coverage. Each child
proves a real positive baseline before applying its one mutation.
"""
import json
from pathlib import Path
import subprocess
import sys

import pytest


_CHILD=r'''
import abc,json,numbers,sys
from pathlib import Path
sys.path.insert(0,sys.argv[2])
import anygeometry,numpy
from anygeometry import _planning_runtime as runtime
from anygeometry._planning_qualification import QualificationDecline
report=dict(scenario=sys.argv[1],executable=sys.executable,version=sys.version,
    numpy=numpy.__version__,numpy_origin=numpy.__file__,
    anygeometry_origin=anygeometry.__file__,runtime_origin=runtime.__file__,
    scope='clean-child-finite-ABC-unit')
assert Path(anygeometry.__file__).resolve().is_relative_to(Path(sys.argv[2]).resolve())
try:runtime.verify_numeric_abcs()
except QualificationDecline as error:
    report['baseline']=dict(status='refused',type=type(error).__name__,reason=str(error))
    print(json.dumps(report));raise
report['baseline']=dict(status='passed')
scenario=sys.argv[1]
if scenario=='warm':
    class Unrelated:pass
    assert not isinstance(Unrelated(),numbers.Rational)
    assert any(reference() is Unrelated for reference in abc._get_dump(numbers.Rational)[2])
    cleared=[]
    def refuse_clear(*args):
        cleared.append(1);raise AssertionError('host cache cleared')
    abc.ABCMeta._abc_caches_clear=refuse_clear
    runtime.verify_numeric_abcs()
    assert cleared==[]
    report.update(status='passed',negative_cache_warmed=True,cache_clear_calls=len(cleared))
elif scenario in ('__subclasscheck__','__instancecheck__'):
    original=getattr(abc.ABCMeta,scenario);calls=[]
    def poisoned(cls,value):
        selected=(value is int if scenario=='__subclasscheck__' else type(value) is int)
        if cls is numbers.Rational and selected:
            calls.append(1);return False
        return original(cls,value)
    setattr(abc.ABCMeta,scenario,poisoned)
    try:runtime.verify_numeric_abcs()
    except QualificationDecline as error:
        assert str(error)=='numeric ABC selected predicate differs from audited table'
        assert calls
        report.update(status='refused',type=type(error).__name__,reason=str(error),
                      selected_poison_calls=len(calls))
    else:raise AssertionError('selected predicate poison was accepted')
elif scenario=='registration':
    class Foreign:pass
    numbers.Real.register(Foreign)
    assert any(reference() is Foreign for reference in abc._get_dump(numbers.Real)[0])
    try:runtime.verify_numeric_abcs()
    except QualificationDecline as error:
        assert str(error)=='numeric ABC registrations/hook outside contract'
        report.update(status='refused',type=type(error).__name__,reason=str(error),
                      foreign_registration_present=True)
    else:raise AssertionError('unknown virtual registration was accepted')
else:raise AssertionError('unknown ABC unit scenario')
print(json.dumps(report))
'''


def _run_child(tmp_path,request,scenario):
    source=Path(__file__).resolve().parents[1]/'src'
    command=[sys.executable,'-B','-c',_CHILD,scenario,str(source)]
    try:
        result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',
                              errors='replace',timeout=15)
    except subprocess.TimeoutExpired as error:
        for name,value in (('stdout',error.stdout),('stderr',error.stderr)):
            text=value.decode('utf-8','replace') if isinstance(value,bytes) else value or ''
            (tmp_path/(name+'.txt')).write_text(text,encoding='utf-8')
        raise
    (tmp_path/'stdout.txt').write_text(result.stdout,encoding='utf-8')
    (tmp_path/'stderr.txt').write_text(result.stderr,encoding='utf-8')
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads(result.stdout)
    assert report['baseline']=={'status':'passed'}
    (tmp_path/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    request.node.user_properties.extend((('execution_scope',report['scope']),
        ('child_positive_baseline_passed',True),('child_executable',report['executable']),
        ('child_numpy',report['numpy']),('child_anygeometry',report['anygeometry_origin'])))
    return report


def test_warm_irrelevant_cache_history_preserves_selected_decisions(tmp_path,request):
    report=_run_child(tmp_path,request,'warm')
    assert report['status']=='passed' and report['negative_cache_warmed']
    assert report['cache_clear_calls']==0


@pytest.mark.parametrize('predicate',['__subclasscheck__','__instancecheck__'])
def test_poisoned_selected_predicate_refuses(tmp_path,request,predicate):
    report=_run_child(tmp_path,request,predicate)
    assert report['type']=='QualificationDecline' and report['selected_poison_calls']>0
    assert report['reason']=='numeric ABC selected predicate differs from audited table'
    request.node.user_properties.append(('selected_poison_calls',report['selected_poison_calls']))


def test_unknown_virtual_registration_refuses(tmp_path,request):
    report=_run_child(tmp_path,request,'registration')
    assert report['type']=='QualificationDecline' and report['foreign_registration_present']
    assert report['reason']=='numeric ABC registrations/hook outside contract'
