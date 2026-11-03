"""Loaded callables and settings bind admission; source files alone do not."""
import sys
import types
import pytest
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
def test_loaded_callable_mutation_is_terminal_without_resnapshot(change):
    module=fixture();manifest=api().capture_callable_manifest((module.root,))
    if change=='code':module.dependency.__code__=(lambda:7).__code__
    elif change=='defaults':module.dependency.__defaults__=(8,)
    elif change=='kwargs':module.dependency.__kwdefaults__['scale']=8
    elif change=='global':module.VALUE=8
    else:module.dependency=lambda:8
    with pytest.raises(PlannerSourceChanged):manifest.verify()


def test_unrelated_module_does_not_enter_callable_closure(monkeypatch):
    module=fixture();manifest=api().capture_callable_manifest((module.root,))
    unrelated=types.ModuleType('ppq_unrelated');unrelated.opaque=object()
    monkeypatch.setitem(sys.modules,unrelated.__name__,unrelated)
    manifest.verify()
    assert manifest.strong_references


def test_root_module_symbol_replacement_is_terminal():
    module=fixture();manifest=api().capture_callable_manifest((module.root,))
    module.root=lambda:4
    with pytest.raises(PlannerSourceChanged):manifest.verify()


def test_closure_cells_are_bound():
    value=[1]
    def root():return value[0]
    manifest=api().capture_callable_manifest((root,));value[0]=2
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


def test_node_limit_is_unchanged_and_truncation_refuses():
    value=list(range(80001))
    def root():return value
    with pytest.raises(QualificationDecline,match='bound|truncat'):api().capture_callable_manifest((root,))


def test_foreign_cache_name_never_gains_content_exemption():
    module=fixture()
    exec('_MEMO={"opaque":object()}\ndef root():return _MEMO\n',vars(module))
    with pytest.raises(QualificationDecline):api().capture_callable_manifest((module.root,))


def test_owned_function_annotation_state_is_bound():
    module=fixture();module.root.__annotations__={'return':int}
    manifest=api().capture_callable_manifest((module.root,))
    module.root.__annotations__['return']=str
    with pytest.raises(PlannerSourceChanged):manifest.verify()


def test_static_relative_import_under_python314_is_resolved():
    module=types.ModuleType('anygeometry.ppq_manifest_fixture')
    module.__package__='anygeometry'
    exec('def root():\n from .errors import GeometryError\n return GeometryError\n',vars(module))
    api().capture_callable_manifest((module.root,)).verify()


def test_real_cold_and_warm_callable_capture_preserves_host_caches():
    from anygeometry._planning_canaries import _fixture
    from anygeometry import query_trimmed_surface_charts
    model,_=_fixture()
    manifest=api().capture_loaded_manifest()
    query_trimmed_surface_charts(model,tuple(model.faces))
    manifest.verify()
    warm=api().capture_loaded_manifest()
    assert warm.wire['callables_sha256']==manifest.wire['callables_sha256']


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
def test_declared_polynomial_interface_rebinding_is_terminal(monkeypatch,edge):
    import numpy as np
    from anygeometry import _cross3 as kernel
    api().prepare_interfaces()
    captured=api().capture_callable_manifest((kernel._gauss_legendre,))
    owner={'polynomial':np,'legendre':vars(np)['polynomial'],
           'leggauss':vars(vars(np)['polynomial'])['legendre']}[edge]
    monkeypatch.setattr(owner,edge,None)
    with pytest.raises(PlannerSourceChanged):captured.verify()


def test_unknown_lazy_module_attribute_is_not_resolved():
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
def test_non_string_function_module_refuses_passively(metadata):
    module=fixture();module.root.__module__=metadata
    with pytest.raises(QualificationDecline,match='function metadata'):
        api().capture_callable_manifest((module.root,))


@pytest.mark.parametrize('field',['__module__','__qualname__'])
@pytest.mark.parametrize('phase',['initial','retained','full'])
def test_function_string_subclass_metadata_never_executes_hooks(field,phase):
    calls=[]
    class Hostile(str):
        def startswith(self,*args):calls.append('startswith');raise AssertionError('metadata hook')
        def __hash__(self):calls.append('hash');raise AssertionError('metadata hook')
        def __eq__(self,other):calls.append('eq');raise AssertionError('metadata hook')
        def __str__(self):calls.append('str');raise AssertionError('metadata hook')
    module=fixture()
    captured=None if phase=='initial' else api().capture_callable_manifest((module.root,))
    setattr(module.root,field,Hostile(getattr(module.root,field)))
    with pytest.raises(QualificationDecline if phase=='initial' else PlannerSourceChanged):
        if captured is None:api().capture_callable_manifest((module.root,))
        else:captured.verify(full=phase=='full')
    assert calls==[]


@pytest.mark.parametrize('full',[False,True])
def test_function_module_none_after_capture_is_terminal(full):
    module=fixture();captured=api().capture_callable_manifest((module.root,))
    module.root.__module__=None
    with pytest.raises(PlannerSourceChanged):captured.verify(full=full)
