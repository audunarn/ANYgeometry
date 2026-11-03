"""Real source-bound classes retain every owned definition binding.

R5 supersedes selected-body-only mutation acceptance. Synthetic package names
cannot establish source provenance; the old spoof fixture remains a refusal.
"""
import sys
import types
import pytest
from _planning_contract_helpers import capture_with_runtime_boundary, capture_with_dispatcher_boundary, classify_initial_dispatcher_refusal
from anygeometry._planning_manifest import capture_callable_manifest
from anygeometry._planning_executor import PlannerSourceChanged
from anygeometry._planning_qualification import QualificationDecline


def owned(monkeypatch):
    module=types.ModuleType('anygeometry.ppq_finite_type_fixture')
    module.__file__=__file__
    exec('class Record:\n def selected(self):return 1\n def unused(self,opaque=object()):return opaque\ndef root(value):return value.selected()\n',vars(module))
    monkeypatch.setitem(sys.modules,module.__name__,module)
    return module


def test_type_root_symbol_replacement_is_terminal(monkeypatch,request):
    from anygeometry import tolerance
    manifest=capture_with_runtime_boundary((tolerance.TolerancePolicy,),request=request)
    if manifest is None:return # Exact initial refusal is not postcapture coverage.
    monkeypatch.setattr(tolerance,'TolerancePolicy',type('Other',(),{}))
    with pytest.raises(PlannerSourceChanged):manifest.verify()


def test_selected_method_rebinding_is_terminal(monkeypatch,request):
    from anygeometry.tolerance import TolerancePolicy
    manifest=capture_with_runtime_boundary((TolerancePolicy.effective_length,TolerancePolicy),request=request)
    if manifest is None:return # Exact initial refusal is not postcapture coverage.
    monkeypatch.setattr(TolerancePolicy,'effective_length',lambda self,scale:2)
    with pytest.raises(PlannerSourceChanged):manifest.verify()


def test_unselected_owned_method_rebinding_is_terminal(monkeypatch,request):
    from anygeometry.tolerance import TolerancePolicy
    manifest=capture_with_runtime_boundary((TolerancePolicy.effective_length,TolerancePolicy),request=request)
    if manifest is None:return # Exact initial refusal is not postcapture coverage.
    monkeypatch.setattr(TolerancePolicy,'effective_area',lambda self,scale:2)
    with pytest.raises(PlannerSourceChanged):manifest.verify()


def test_outside_package_source_spoof_refuses_initial_capture(monkeypatch,request):
    if classify_initial_dispatcher_refusal(request):return # Exact initial refusal is not source-spoof coverage.
    module=owned(monkeypatch)
    with pytest.raises(QualificationDecline,match='source outside package'):
        capture_callable_manifest((module.root,module.Record))


def test_exact_builtin_type_constructor_binds_interpreter_image(request):
    manifest=capture_with_runtime_boundary((BaseException.__new__,),request=request)
    if manifest is None:return # Exact initial refusal is not postcapture coverage.
    manifest.verify()


def test_unregistered_native_like_callable_refuses():
    class Pretend:
        __module__=None
        __qualname__='BaseException.__new__'
        def __call__(self):return None
    with pytest.raises(QualificationDecline):capture_callable_manifest((Pretend(),))


def test_bound_builtin_instance_method_is_not_type_constructor():
    with pytest.raises(QualificationDecline):capture_callable_manifest(([].append,))


@pytest.mark.parametrize('module',[None,False,17,object()])
def test_non_string_type_metadata_refuses_without_coercion(module):
    class Unknown:
        def __call__(self):return None
    Unknown.__module__=module
    with pytest.raises(QualificationDecline):capture_callable_manifest((Unknown(),))


@pytest.mark.parametrize('field',['__module__','__qualname__'])
def test_string_subclass_type_metadata_never_dispatches_hooks(field):
    events=[]
    class Text(str):
        def __add__(self,other):events.append('add');raise AssertionError('metadata add hook')
        def __radd__(self,other):events.append('radd');raise AssertionError('metadata radd hook')
        def __str__(self):events.append('str');raise AssertionError('metadata str hook')
    class Unknown:pass
    setattr(Unknown,field,Text('pretend'))
    with pytest.raises(QualificationDecline):capture_callable_manifest((Unknown(),))
    assert events==[]


def test_type_metadata_native_reads_bypass_metaclass_hooks():
    events=[]
    class Meta(type):
        @property
        def __module__(cls):events.append('module');raise AssertionError('metadata property hook')
    class Unknown(metaclass=Meta):pass
    with pytest.raises(QualificationDecline):capture_callable_manifest((Unknown(),))
    assert events==[]


@pytest.mark.parametrize('full',[False,True])
def test_type_metadata_drift_is_terminal(full,request):
    value={'safe':1};captured=capture_with_dispatcher_boundary((value,),request=request)
    if captured is None:return # Exact initial refusal is not postcapture coverage.
    class Unknown:__module__=None
    value['new']=Unknown()
    with pytest.raises(PlannerSourceChanged):captured.verify(full=full)
