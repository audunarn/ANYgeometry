"""Resource-helper live-export admission; no numerical planning execution."""
import importlib
import sys
import types

import pytest
from _planning_native_test_contract import native_resource_test
from _planning_numpy_profile_contract import numpy_profile_test

from anygeometry import _planning_native as adapter
from anygeometry._planning_qualification import QualificationDecline


@pytest.mark.parametrize('name',['identity','create','new_owner','close','wait'])
@native_resource_test
def test_initial_altered_export_refuses_before_any_replacement_call(monkeypatch,name):
    native=adapter.load();called=[]
    def altered(*args,**kwargs):
        called.append(name)
        if name=='identity':return {'abi':'cp311-abi3','resource_abi':1,'build':'forged'}
        raise AssertionError('altered resource export ran')
    monkeypatch.setattr(native,name,altered)
    with pytest.raises(QualificationDecline):adapter.identity()
    assert called==[]


@native_resource_test
def test_native_builtin_swapped_export_refuses_before_call(monkeypatch):
    native=adapter.load()
    monkeypatch.setattr(native,'identity',native.new_owner)
    with pytest.raises(QualificationDecline):adapter.identity()


@native_resource_test
def test_replacement_module_with_copied_metadata_and_exports_refuses(monkeypatch):
    native=adapter.load()
    fake=types.ModuleType(native.__name__);fake.__dict__.update(vars(native))
    monkeypatch.setitem(sys.modules,native.__name__,fake)
    import anygeometry
    monkeypatch.setattr(anygeometry,'_planning_native_core',fake)
    with pytest.raises(QualificationDecline):adapter.identity()


@native_resource_test
@numpy_profile_test
def test_postcapture_export_drift_is_terminal_through_resource_infrastructure(monkeypatch):
    from anygeometry._planning_code import InfrastructurePin
    from anygeometry._planning_executor import PlannerSourceChanged
    captured=InfrastructurePin('host')
    native=adapter.load();called=[]
    monkeypatch.setattr(native,'create',lambda *a:called.append('create'))
    with pytest.raises(PlannerSourceChanged):captured.verify()
    assert called==[]


@native_resource_test
def test_worker_resource_file_pin_has_no_native_import_dependency(monkeypatch):
    from anygeometry import _planning_qualification as q
    identity=adapter.identity();pin=q.resource_file_pin(identity)
    original=importlib.import_module
    def no_helper(name,*args,**kwargs):
        if name in ('anygeometry._planning_native','anygeometry._planning_native_core'):
            raise AssertionError('resource file check imported ownership helper')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(importlib,'import_module',no_helper)
    q.verify_resource_file_pin(pin)
    assert pin['identity']==identity and len(pin['files'])==2


@native_resource_test
def test_worker_resource_file_skew_refuses(monkeypatch):
    from anygeometry import _planning_qualification as q
    pin=q.resource_file_pin(adapter.identity());original=q._hash
    binary=next(path for path in pin['files'] if path.endswith('.pyd'))
    monkeypatch.setattr(q,'_hash',lambda path:'0'*64 if str(path)==binary else original(path))
    with pytest.raises(QualificationDecline,match='resource file'):q.verify_resource_file_pin(pin)


@native_resource_test
def test_worker_reference_projects_only_two_resource_modules():
    from anygeometry import _planning_qualification as q
    pin=q.resource_file_pin(adapter.identity())
    providers={'anygeometry._planning_native_core':q.pin_module(adapter.load()),'numerical.native':{'kind':'native'}}
    original={'modules':{'anygeometry._planning_native':q.pin_module(adapter),
        'anygeometry._planning_native_core':providers['anygeometry._planning_native_core'],
        'generated':{'kind':'explicit-generated-native','providers':providers},
        'unexpected.numerical':{'kind':'python'}},'files':dict(pin['files']),'resource_pin':pin}
    projected=q.worker_environment_reference(original)
    assert set(projected['modules'])=={'generated','unexpected.numerical'}
    assert projected['modules']['generated']['providers']=={'numerical.native':{'kind':'native'}}
    assert projected['files']==original['files'] and projected['resource_pin']==pin
    assert len(original['modules'])==4 and len(original['modules']['generated']['providers'])==2


@pytest.mark.parametrize('name',['anygeometry._planning_controller','anygeometry._planning_native','anygeometry._planning_native_core'])
@pytest.mark.parametrize('binding',['module','package'])
def test_worker_resource_guard_refuses_forbidden_import(monkeypatch,name,binding):
    from anygeometry import _planning_qualification as q
    import anygeometry
    for other in ('anygeometry._planning_controller','anygeometry._planning_native','anygeometry._planning_native_core'):
        monkeypatch.delitem(sys.modules,other,raising=False)
        monkeypatch.delattr(anygeometry,other.rsplit('.',1)[-1],raising=False)
    if binding=='module':monkeypatch.setitem(sys.modules,name,types.ModuleType(name))
    else:monkeypatch.setattr(anygeometry,name.rsplit('.',1)[-1],types.ModuleType(name),raising=False)
    with pytest.raises(QualificationDecline,match='worker ownership'):q.verify_worker_resource_absence()

@pytest.mark.parametrize('name',['reserve_child','job_create','state','query','configure','terminate','assign','observe','member','resume','exit_code','inject_fault'])
@native_resource_test
def test_remaining_resource_exports_are_bound(monkeypatch,name):
    native=adapter.load();called=[]
    monkeypatch.setattr(native,name,lambda *a:called.append(name))
    with pytest.raises(QualificationDecline):adapter.identity()
    assert not called


@pytest.mark.parametrize('target',['module_class','loader','spec','package_edge','extra_export','builtin_module'])
@native_resource_test
def test_mutable_native_metadata_and_type_bindings_refuse(monkeypatch,target):
    native=adapter.load();called=[]
    if target=='module_class':
        class Changed(types.ModuleType):
            def __getattribute__(self,name):
                if name=='identity':called.append(name)
                return super().__getattribute__(name)
        monkeypatch.setattr(native,'__class__',Changed)
    elif target=='loader':monkeypatch.setattr(native.__loader__,'path',__file__)
    elif target=='spec':
        from importlib.machinery import ModuleSpec
        replacement=ModuleSpec(native.__name__,native.__loader__,origin=native.__file__)
        monkeypatch.setattr(native,'__spec__',replacement)
    elif target=='package_edge':
        import anygeometry
        monkeypatch.setattr(anygeometry,'_planning_native_core',object())
    elif target=='extra_export':monkeypatch.setattr(native,'unknown_resource_call',lambda:called.append('extra'),raising=False)
    else:monkeypatch.setattr(native.create,'__module__','foreign.image')
    with pytest.raises(QualificationDecline):adapter.identity()
    assert not called


@native_resource_test
def test_existing_owner_uses_retained_cleanup_after_live_exports_drift(tmp_path,monkeypatch):
    from anygeometry._planning_process import OwnedTree
    tree=OwnedTree(tmp_path);native=tree.api.native;called=[]
    for name in ('state','query','terminate','close'):
        monkeypatch.setattr(native,name,lambda *args:called.append('altered cleanup'))
    with pytest.raises(QualificationDecline):adapter.identity()
    tree.close()
    assert tree.handle is None and called==[]


@native_resource_test
def test_export_wire_is_json_portable_and_offsets_are_image_relative():
    import json
    value=adapter.identity()
    assert json.loads(json.dumps(value))==value
    assert value['exports_schema']==1
    assert {record['name'] for record in value['exports']}==set(adapter._EXPORT_NAMES)
    assert all(type(record['image_offset']) is int and 0<record['image_offset']<2**32 for record in value['exports'])


@native_resource_test
def test_image_lookup_failure_is_refused_without_fallback(monkeypatch):
    original=adapter.ctypes.WinDLL
    class Failed:
        def __init__(self):
            self.GetModuleHandleExW=lambda *args:0
            self.GetModuleFileNameW=lambda *args:0
    monkeypatch.setattr(adapter.ctypes,'WinDLL',lambda *args,**kwargs:Failed())
    with pytest.raises(QualificationDecline,match='image lookup failed'):adapter.identity()


def test_owner_reload_cannot_rebaseline_changed_exports():
    with pytest.raises(RuntimeError,match='fresh process'):importlib.reload(adapter)


@native_resource_test
def test_pre_owner_import_mutation_is_refused_and_fresh_process_wire_matches(tmp_path):
    import json
    from pathlib import Path
    import subprocess
    source=str(Path(adapter.__file__).resolve().parents[1])
    code='''import importlib,json,sys
sys.path.insert(0,sys.argv[1])
helper=importlib.import_module('anygeometry._planning_native_core')
assert 'anygeometry._planning_native' not in sys.modules
called=[]
if sys.argv[2]=='altered':
 def changed():called.append('identity');return {'abi':'cp311-abi3','resource_abi':1,'build':'forged'}
 helper.identity=changed
from anygeometry import _planning_native as adapter
from anygeometry._planning_qualification import QualificationDecline
if sys.argv[2]=='altered':
 try:adapter.identity()
 except QualificationDecline:pass
 else:raise AssertionError('pre-owner mutation was accepted')
 assert called==[]
 print(json.dumps({'refused_before_call':True}))
else:print(json.dumps(adapter.identity()))
'''
    for mode in ('altered','clean'):
        command=[sys.executable,'-B','-c',code,source,mode]
        result=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True,timeout=15)
        (tmp_path/(mode+'-stdout.txt')).write_text(result.stdout)
        (tmp_path/(mode+'-stderr.txt')).write_text(result.stderr)
        assert result.returncode==0,result.stderr
        observed=json.loads(result.stdout)
        if mode=='clean':assert observed==adapter.identity()
        else:assert observed=={'refused_before_call':True}


def test_native_contract_required_lane_refuses_before_loading_helper(monkeypatch,request):
    import _planning_native_test_contract as contract
    from types import SimpleNamespace
    called=[]
    monkeypatch.setattr(contract,'os',SimpleNamespace(name='posix'))
    monkeypatch.setenv('ANYGEOMETRY_REQUIRE_QUALIFIED_PLANNING_TESTS','1')
    monkeypatch.setattr(contract,'_adapter',SimpleNamespace(load=lambda:called.append('load')))
    @contract.native_resource_test
    def original():called.append('body')
    with pytest.raises(pytest.fail.Exception,match='qualified planning test lane requires original assertions'):
        original(request=request)
    assert called==[]


def test_native_contract_rejects_an_unexpected_initial_reason(monkeypatch,request):
    import _planning_native_test_contract as contract
    from types import SimpleNamespace
    called=[]
    def decline():
        called.append('load')
        raise QualificationDecline('unexpected native refusal')
    monkeypatch.setattr(contract,'os',SimpleNamespace(name='posix'))
    monkeypatch.setenv('ANYGEOMETRY_REQUIRE_QUALIFIED_PLANNING_TESTS','0')
    monkeypatch.setattr(contract,'_adapter',SimpleNamespace(load=decline))
    @contract.native_resource_test
    def original():called.append('body')
    with pytest.raises(AssertionError):original(request=request)
    assert called==['load']
