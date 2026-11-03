"""Effective NumPy thread/context settings belong to loaded-host admission."""
import types
import numpy as np
import pytest
from _planning_numpy_profile_contract import numpy_profile_test
from anygeometry._planning_executor import PlannerSourceChanged
from anygeometry._planning_qualification import QualificationDecline


@pytest.mark.parametrize('boundary',['runtime','dispatcher','profile','negative-dispatcher'])
def test_required_qualified_lane_rejects_initial_boundary_refusal(monkeypatch,request,boundary):
    import _planning_contract_helpers as helpers
    import _planning_numpy_profile_contract as profile
    reason='NumPy array dispatcher interface unavailable'
    monkeypatch.setenv('ANYGEOMETRY_REQUIRE_QUALIFIED_PLANNING_TESTS','1')
    monkeypatch.setattr(helpers,'runtime_boundary_refusal',lambda:reason)
    monkeypatch.setattr(helpers,'dispatcher_boundary_refusal',lambda:reason)
    monkeypatch.setattr(profile,'_initial_profile_refusal',lambda:reason)
    entered=[]
    def original():entered.append(True)
    actions={
        'runtime':lambda:helpers.capture_with_runtime_boundary((None,),request=request),
        'dispatcher':lambda:helpers.capture_with_dispatcher_boundary((None,),request=request),
        'profile':lambda:profile.numpy_profile_test(original)(request=request),
        'negative-dispatcher':lambda:helpers.classify_initial_dispatcher_refusal(request)}
    with pytest.raises(pytest.fail.Exception,match='qualified planning test lane requires original assertions: '+reason):
        actions[boundary]()
    assert entered==[]


def capture():
    from anygeometry._planning_numpy_context import capture_numpy_context
    return capture_numpy_context()


@numpy_profile_test
def test_capture_preserves_exact_effective_numpy_context():
    flags=np.geterr();callback=np.geterrcall();buffer=np.getbufsize()
    context=capture()
    assert context.wire==dict(errors=flags,callback=None,buffer_size=buffer)
    assert np.geterr()==flags and np.geterrcall() is callback and np.getbufsize()==buffer
    context.verify()


@pytest.mark.parametrize('kind',['divide','over','under','invalid','buffer'])
@numpy_profile_test
def test_effective_numpy_setting_change_is_terminal(kind):
    context=capture();flags=np.geterr();buffer=np.getbufsize()
    try:
        if kind=='buffer':np.setbufsize(buffer+16)
        else:np.seterr(**{kind:'raise' if flags[kind]!='raise' else 'ignore'})
        with pytest.raises(PlannerSourceChanged):context.verify()
    finally:
        np.seterr(**flags);np.setbufsize(buffer)


def test_error_callback_is_not_invoked_or_serialized():
    original=np.geterrcall()
    def foreign(*args):pytest.fail('NumPy error callback executed during capture')
    try:
        np.seterrcall(foreign)
        with pytest.raises(QualificationDecline):capture()
    finally:np.seterrcall(original)


def test_settings_getter_replacement_refuses_before_call(monkeypatch):
    monkeypatch.setattr(np,'geterr',lambda:pytest.fail('foreign getter invoked'))
    with pytest.raises(QualificationDecline):capture()


@numpy_profile_test
def test_loaded_manifest_binds_numpy_context_with_small_owned_root(monkeypatch):
    from anygeometry import _planning_manifest as manifest
    module=types.ModuleType('ppq_context_fixture');exec('def root():return 1',vars(module))
    monkeypatch.setattr(manifest,'numerical_roots',lambda:(module.root,))
    before=np.geterr()
    loaded=manifest.capture_loaded_manifest()
    try:
        np.seterr(over='raise' if before['over']!='raise' else 'ignore')
        with pytest.raises(PlannerSourceChanged):loaded.verify()
    finally:np.seterr(**before)


@numpy_profile_test
def test_loaded_manifest_callback_drift_is_terminal(monkeypatch):
    from anygeometry import _planning_manifest as manifest
    module=types.ModuleType('ppq_context_fixture');exec('def root():return 1',vars(module))
    monkeypatch.setattr(manifest,'numerical_roots',lambda:(module.root,))
    loaded=manifest.capture_loaded_manifest();before=np.geterrcall()
    try:
        np.seterrcall(lambda *args:None)
        with pytest.raises(PlannerSourceChanged):loaded.verify()
    finally:np.seterrcall(before)


@pytest.mark.parametrize('which',['python_code','native_alias'])
def test_pre_owner_modified_getter_is_not_executed(tmp_path,which):
    import os,subprocess,sys
    from pathlib import Path
    script='''
import importlib,json,numpy as np
prefix='numpy._core' if int(np.__version__.split('.')[0])>=2 else 'numpy.core'
config=importlib.import_module(prefix+'._ufunc_config')
if CASE=='python_code':
    def foreign():raise AssertionError('pre-owner getter executed')
    np.geterr.__code__=foreign.__code__
else:
    native=importlib.import_module(prefix+'._multiarray_umath')
    umath=importlib.import_module(prefix+'.umath')
    def foreign():raise AssertionError('pre-owner native alias executed')
    getter='_get_extobj_dict' if '_get_extobj_dict' in vars(native) else 'geterrobj'
    setattr(native,getter,foreign)
    setattr(umath,getter,foreign)
    if getter in vars(config):setattr(config,getter,foreign)
import anygeometry
from anygeometry._planning_numpy_context import capture_numpy_context
from anygeometry._planning_qualification import QualificationDecline
try:capture_numpy_context()
except QualificationDecline:pass
else:raise AssertionError('modified pre-owner getter admitted')
'''.replace('CASE',repr(which))
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'),PYTHONDONTWRITEBYTECODE='1',
             ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING='0')
    result=subprocess.run([sys.executable,'-B','-c',script],cwd=tmp_path,env=env,capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stdout+result.stderr
