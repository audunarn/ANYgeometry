"""Curated external generated-module linkage; controlled metadata is not native qualification."""
import ast
import json
from pathlib import Path
import sys
import types
import pytest
from anygeometry import _planning_qualification as q


@pytest.fixture
def generated_image(monkeypatch,tmp_path):
    import scipy,threadpoolctl
    path=tmp_path/'provider.pyd';path.write_bytes(b'controlled metadata; not loaded as executable')
    provider=types.ModuleType('provider');provider.__file__=str(path)
    modules={'provider':provider,'cython_runtime':types.ModuleType('cython_runtime')}
    for name in ('_cython_3_2_4','_cython_3_1_4','_cython_3_3_0'):modules[name]=types.ModuleType(name)
    monkeypatch.setattr(q,'sys',types.SimpleNamespace(modules=modules,executable=sys.executable,version=sys.version,platform=sys.platform,implementation=sys.implementation))
    monkeypatch.setattr(q,'runtime_files',lambda:{str(path):q._hash(path)})
    monkeypatch.setattr(threadpoolctl,'threadpool_info',lambda:[dict(filepath=str(path),num_threads=1,internal_api='controlled-metadata')])
    return modules,path


def _selected():
    if hasattr(q,'generated_runtime_modules'):return q.generated_runtime_modules()
    # Baseline uses the actual old child declaration without executing its startup.
    tree=ast.parse(Path(q.__file__).with_name('_planning_child.py').read_text())
    return next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(n,ast.Name) and n.id=='_GENERATED_MODULES' for n in node.targets))


def test_observed_cython_330_uses_curated_native_linkage(generated_image,request):
    observed=q.capture_environment(generated_modules=_selected())
    pin=observed['modules']['_cython_3_3_0']
    assert pin['kind']=='explicit-generated-native'
    assert pin['providers']=={'provider':observed['modules']['provider']}
    assert pin['members']==[['__doc__','NoneType'],['__loader__','NoneType'],['__name__','str'],['__package__','NoneType'],['__spec__','NoneType']]
    q.verify_environment(json.loads(json.dumps(observed)),observed)
    request.node.user_properties.extend((('execution_scope','bounded-generated-metadata-component'),('loaded_native_qualification',False)))


@pytest.mark.parametrize('present',[(),('_cython_3_3_0',),('_cython_3_2_4','_cython_3_1_4')])
def test_selection_is_only_present_curated_names(generated_image,present):
    modules,_=generated_image
    for name in tuple(modules):
        if name.startswith('_cython_') and name not in present:del modules[name]
    selected=q.generated_runtime_modules()
    assert selected==('cython_runtime',)+tuple(name for name in q._GENERATED_VERSION_MODULES if name in present)
    record=q.capture_environment(generated_modules=selected)
    assert q.generated_runtime_modules(reference=record)==selected


@pytest.mark.parametrize('change',['missing','extra','schema'])
def test_worker_reference_requires_exact_generated_set(generated_image,change):
    modules,_=generated_image
    expected=q.capture_environment(generated_modules=q.generated_runtime_modules())
    if change=='missing':del modules['_cython_3_3_0']
    elif change=='extra':expected['modules']['_cython_3_9_9']={'kind':'explicit-generated-native'}
    else:expected['modules']=[]
    with pytest.raises(q.QualificationDecline,match='generated module (selection differs from reference|reference schema)'):
        q.generated_runtime_modules(reference=expected)


def test_cython_runtime_remains_mandatory(generated_image):
    modules,_=generated_image;del modules['cython_runtime']
    with pytest.raises(q.QualificationDecline,match='^mandatory generated module missing: cython_runtime$'):
        q.generated_runtime_modules()


def test_unknown_synthetic_name_never_gains_generated_treatment(generated_image):
    modules,_=generated_image;modules['_cython_99_0_0']=types.ModuleType('_cython_99_0_0')
    selected=q.generated_runtime_modules()
    assert '_cython_99_0_0' not in selected
    with pytest.raises(q.QualificationDecline,match='^module has no qualified origin: _cython_99_0_0$'):
        q.capture_environment(generated_modules=selected)


@pytest.mark.parametrize('change',['member','member-type','provider-bytes','provider-missing','origin','type'])
def test_generated_and_native_pins_reject_mutation(generated_image,change):
    modules,path=generated_image
    selected=q.generated_runtime_modules();expected=q.capture_environment(generated_modules=selected)
    if change=='member':modules['_cython_3_3_0'].new_member=True
    elif change=='member-type':modules['_cython_3_3_0'].__doc__='changed type'
    elif change=='provider-bytes':path.write_bytes(b'changed controlled metadata')
    elif change=='provider-missing':del modules['provider']
    elif change=='origin':modules['_cython_3_3_0'].__file__=str(path)
    else:modules['_cython_3_3_0']=object()
    with pytest.raises(q.QualificationDecline):
        observed=q.capture_environment(generated_modules=q.generated_runtime_modules(reference=expected))
        q.verify_environment(expected,observed)


@pytest.mark.parametrize('change',['rebound','late-curated','late-unknown'])
def test_existing_batch_module_identity_boundary_is_retained(generated_image,monkeypatch,change):
    from anygeometry import _planning_batch as batch
    modules,_=generated_image
    if change=='late-curated':del modules['_cython_3_3_0']
    expected=q.capture_environment(generated_modules=q.generated_runtime_modules());registered=dict(modules)
    monkeypatch.setattr(batch,'sys',types.SimpleNamespace(modules=modules))
    if change=='rebound':modules['_cython_3_2_4']=types.ModuleType('_cython_3_2_4')
    else:
        name='_cython_3_3_0' if change=='late-curated' else '_cython_99_0_0'
        modules[name]=types.ModuleType(name)
    with pytest.raises(q.QualificationDecline,match='^batch module closure/binding changed:'):
        batch.verify_boundary(expected,registered)


def test_diagnostic_and_session_roles_use_the_same_selector():
    root=Path(q.__file__).parent
    child=ast.parse((root/'_planning_child.py').read_text());session=ast.parse((root/'_planning_session.py').read_text())
    for tree,names in [(child,('worker','planner','main')),(session,('session_planner','session_worker'))]:
        for name in names:
            fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='generated_runtime_modules']
            assert calls,name
            if name in ('worker','session_worker'):
                assert any(any(k.arg=='reference' for k in call.keywords) for call in calls)
