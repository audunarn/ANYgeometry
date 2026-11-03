"""Loaded callables and settings bind admission; source files alone do not."""
import sys
import types
import pytest
from _planning_numpy_profile_contract import numpy_profile_test
from _planning_contract_helpers import capture_with_runtime_boundary, capture_with_dispatcher_boundary, classify_initial_dispatcher_refusal
from anygeometry._planning_executor import PlannerSourceChanged
from anygeometry._planning_qualification import QualificationDecline


def api():
    from anygeometry import _planning_manifest
    return _planning_manifest


def fixture():
    module=types.ModuleType('ppq_manifest_fixture')
    exec('VALUE=2\ndef dependency(x=3,*,scale=4): return (x+VALUE)*scale\ndef root(): return dependency()\n',vars(module))
    return module


@pytest.mark.parametrize('change',['code','defaults','kwargs','global','alias'])
def test_loaded_callable_mutation_is_terminal_without_resnapshot(change,request):
    module=fixture();manifest=capture_with_dispatcher_boundary((module.root,),request=request)
    if manifest is None:return # Exact initial refusal is not postcapture coverage.
    if change=='code':module.dependency.__code__=(lambda:7).__code__
    elif change=='defaults':module.dependency.__defaults__=(8,)
    elif change=='kwargs':module.dependency.__kwdefaults__['scale']=8
    elif change=='global':module.VALUE=8
    else:module.dependency=lambda:8
    with pytest.raises(PlannerSourceChanged):manifest.verify()


def test_unrelated_module_does_not_enter_callable_closure(monkeypatch,request):
    module=fixture();manifest=capture_with_dispatcher_boundary((module.root,),request=request)
    if manifest is None:return # Exact initial refusal is not postcapture coverage.
    unrelated=types.ModuleType('ppq_unrelated');unrelated.opaque=object()
    monkeypatch.setitem(sys.modules,unrelated.__name__,unrelated)
    manifest.verify()
    assert manifest.strong_references


def test_root_module_symbol_replacement_is_terminal(request):
    module=fixture();manifest=capture_with_dispatcher_boundary((module.root,),request=request)
    if manifest is None:return # Exact initial refusal is not postcapture coverage.
    module.root=lambda:4
    with pytest.raises(PlannerSourceChanged):manifest.verify()


def test_closure_cells_are_bound(request):
    value=[1]
    def root():return value[0]
    manifest=capture_with_dispatcher_boundary((root,),request=request)
    if manifest is None:return # Exact initial refusal is not postcapture coverage.
    value[0]=2
    with pytest.raises(PlannerSourceChanged):manifest.verify()


def test_unknown_state_refuses():
    value=object()
    def root():return value
    with pytest.raises(QualificationDecline):api().capture_callable_manifest((root,))


@pytest.mark.parametrize('as_root',[True,False])
def test_unknown_class_property_is_not_executed(as_root):
    class Foreign:
        @property
        def __class__(self):pytest.fail('unknown __class__ property executed')
    value=Foreign()
    def root():return value
    with pytest.raises(QualificationDecline):api().capture_callable_manifest((value if as_root else root,))


@pytest.mark.parametrize('container',[dict,set,frozenset])
def test_nested_key_repr_is_not_executed(container):
    class Foreign:
        def __repr__(self):pytest.fail('unknown key repr executed')
    key=(Foreign(),)
    value={key:0} if container is dict else container((key,))
    def root():return value
    with pytest.raises(QualificationDecline):api().capture_callable_manifest((root,))


def test_node_limit_is_unchanged_and_truncation_refuses(request):
    if classify_initial_dispatcher_refusal(request):return # Exact initial refusal is not node-limit coverage.
    value=list(range(80001))
    def root():return value
    with pytest.raises(QualificationDecline,match='bound|truncat'):api().capture_callable_manifest((root,))


def test_foreign_cache_name_never_gains_content_exemption():
    module=fixture()
    exec('_MEMO={"opaque":object()}\ndef root():return _MEMO\n',vars(module))
    with pytest.raises(QualificationDecline):api().capture_callable_manifest((module.root,))


def test_owned_function_annotation_state_is_bound(request):
    module=fixture();module.root.__annotations__={'return':int}
    manifest=capture_with_runtime_boundary((module.root,),request=request)
    if manifest is None:return # Exact initial refusal is not postcapture coverage.
    module.root.__annotations__['return']=str
    with pytest.raises(PlannerSourceChanged):manifest.verify()


def test_static_relative_import_under_python314_is_resolved(request):
    module=types.ModuleType('anygeometry.ppq_manifest_fixture')
    module.__package__='anygeometry'
    exec('def root():\n from .errors import GeometryError\n return GeometryError\n',vars(module))
    captured=capture_with_runtime_boundary((module.root,),request=request)
    if captured is None:return # Exact initial refusal is not postcapture coverage.
    captured.verify()


@numpy_profile_test
def test_real_cold_and_warm_callable_capture_preserves_host_caches():
    from anygeometry._planning_canaries import _fixture
    from anygeometry import query_trimmed_surface_charts
    model,_=_fixture()
    manifest=api().capture_loaded_manifest()
    query_trimmed_surface_charts(model,tuple(model.faces))
    manifest.verify()
    warm=api().capture_loaded_manifest()
    assert warm.wire['callables_sha256']==manifest.wire['callables_sha256']


@numpy_profile_test
def test_effective_setting_and_native_binary_are_bound(monkeypatch):
    manifest=api().capture_loaded_manifest()
    monkeypatch.setenv('ANYGEOMETRY_SEAM_POLICY','foreign')
    with pytest.raises(PlannerSourceChanged):manifest.verify()


def test_declared_polynomial_interface_preparation_preserves_quadrature_cache():
    import numpy as np
    from anygeometry import _cross3 as kernel
    cache=kernel._GAUSS_LEGENDRE
    entries=tuple(cache.items())
    api().prepare_interfaces()
    polynomial=vars(np).get('polynomial')
    assert type(polynomial) is types.ModuleType
    assert polynomial is sys.modules['numpy.polynomial']
    legendre=vars(polynomial).get('legendre')
    assert type(legendre) is types.ModuleType
    assert legendre is sys.modules['numpy.polynomial.legendre']
    assert type(vars(legendre)['leggauss']) is types.FunctionType
    assert kernel._GAUSS_LEGENDRE is cache
    assert tuple(cache)==tuple(key for key,_ in entries)
    assert all(cache[key] is value for key,value in entries)


@pytest.mark.parametrize('edge',['polynomial','legendre','leggauss'])
def test_declared_polynomial_interface_rebinding_is_terminal(monkeypatch,edge,request):
    import numpy as np
    from anygeometry import _cross3 as kernel
    api().prepare_interfaces()
    captured=capture_with_runtime_boundary((kernel._gauss_legendre,),request=request)
    if captured is None:return # Exact initial refusal is not postcapture coverage.
    owner={'polynomial':np,'legendre':vars(np)['polynomial'],
           'leggauss':vars(vars(np)['polynomial'])['legendre']}[edge]
    monkeypatch.setattr(owner,edge,None)
    with pytest.raises(PlannerSourceChanged):captured.verify()


def test_unknown_lazy_module_attribute_is_not_resolved(request):
    if classify_initial_dispatcher_refusal(request):return # Exact initial refusal is not lazy-module coverage.
    module=types.ModuleType('ppq_foreign_lazy_interface')
    calls=[]
    def getattr_hook(name):
        calls.append(name)
        raise AssertionError('dynamic dependency hook executed')
    module.__getattr__=getattr_hook
    namespace={'__name__':'ppq_lazy_interface_fixture','foreign':module}
    exec('def root(): return foreign.unregistered',namespace)
    with pytest.raises(QualificationDecline,match='dynamic module attribute'):
        api().capture_callable_manifest((namespace['root'],))
    assert calls==[]


@pytest.mark.parametrize('metadata',[None,False,17,[]])
def test_non_string_function_module_refuses_passively(metadata,request):
    if classify_initial_dispatcher_refusal(request):return # Exact initial refusal is not function-metadata coverage.
    module=fixture();module.root.__module__=metadata
    with pytest.raises(QualificationDecline,match='function metadata'):
        api().capture_callable_manifest((module.root,))


@pytest.mark.parametrize('field',['__module__','__qualname__'])
@pytest.mark.parametrize('phase',['initial','retained','full'])
def test_function_string_subclass_metadata_never_executes_hooks(field,phase,request):
    calls=[]
    class Hostile(str):
        def startswith(self,*args):calls.append('startswith');raise AssertionError('metadata hook')
        def __hash__(self):calls.append('hash');raise AssertionError('metadata hook')
        def __eq__(self,other):calls.append('eq');raise AssertionError('metadata hook')
        def __str__(self):calls.append('str');raise AssertionError('metadata hook')
    module=fixture()
    captured=None if phase=='initial' else capture_with_dispatcher_boundary((module.root,),request=request)
    if phase!='initial' and captured is None:return # Exact initial refusal is not postcapture coverage.
    setattr(module.root,field,Hostile(getattr(module.root,field)))
    with pytest.raises(QualificationDecline if phase=='initial' else PlannerSourceChanged):
        if captured is None:api().capture_callable_manifest((module.root,))
        else:captured.verify(full=phase=='full')
    assert calls==[]


@pytest.mark.parametrize('full',[False,True])
def test_function_module_none_after_capture_is_terminal(full,request):
    module=fixture();captured=capture_with_dispatcher_boundary((module.root,),request=request)
    if captured is None:return # Exact initial refusal is not postcapture coverage.
    module.root.__module__=None
    with pytest.raises(PlannerSourceChanged):captured.verify(full=full)
