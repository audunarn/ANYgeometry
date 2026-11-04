"""Real source-bound classes retain every owned definition binding.

R5 supersedes selected-body-only mutation acceptance. Synthetic package names
cannot establish source provenance; the old spoof fixture remains a refusal.
"""
import sys
import types
import pytest
from _planning_numpy_profile_contract import numpy_profile_test
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


@numpy_profile_test
def test_type_root_symbol_replacement_is_terminal(monkeypatch,request):
    from anygeometry import tolerance
    manifest=capture_with_runtime_boundary((tolerance.TolerancePolicy,),request=request)
    if manifest is None:return # Exact initial refusal is not postcapture coverage.
    monkeypatch.setattr(tolerance,'TolerancePolicy',type('Other',(),{}))
    with pytest.raises(PlannerSourceChanged):manifest.verify()


@numpy_profile_test
def test_selected_method_rebinding_is_terminal(monkeypatch,request):
    from anygeometry.tolerance import TolerancePolicy
    manifest=capture_with_runtime_boundary((TolerancePolicy.effective_length,TolerancePolicy),request=request)
    if manifest is None:return # Exact initial refusal is not postcapture coverage.
    monkeypatch.setattr(TolerancePolicy,'effective_length',lambda self,scale:2)
    with pytest.raises(PlannerSourceChanged):manifest.verify()


@numpy_profile_test
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


@numpy_profile_test
def test_bounded_real_owned_class_binding_replacement_is_terminal(monkeypatch,request):
    from anygeometry.errors import GeometryError
    captured=capture_callable_manifest((GeometryError,))
    captured.verify()
    monkeypatch.setattr(GeometryError,'__str__',lambda self:'changed')
    with pytest.raises(PlannerSourceChanged):captured.verify()


def test_real_tolerance_dataclass_wrapper_boundary_is_explicit(request):
    import dataclasses
    from pathlib import Path
    from _planning_contract_helpers import (dispatcher_boundary_refusal,
        require_qualified_planning_lane)
    from anygeometry import tolerance
    roots=(tolerance.TolerancePolicy.effective_length,tolerance.TolerancePolicy)
    if dispatcher_boundary_refusal() is not None:
        captured=capture_with_runtime_boundary(roots,request=request)
        assert captured is None
        return
    wrapper=tolerance.TolerancePolicy.__dict__['__repr__']
    factory=vars(dataclasses).get('_recursive_repr')
    legacy=(type(factory) is types.FunctionType and type(wrapper) is types.FunctionType
        and wrapper.__globals__ is vars(dataclasses)
        and wrapper.__code__ in factory.__code__.co_consts
        and Path(wrapper.__code__.co_filename).resolve()==Path(dataclasses.__file__).resolve()
        and wrapper.__module__==tolerance.__name__
        and wrapper.__qualname__=='TolerancePolicy.__repr__')
    if legacy:
        reason="callable closure refused: ['anygeometry.tolerance.TolerancePolicy@pre-slots.__repr__: owned function namespace/origin mismatch']"
        require_qualified_planning_lane(reason)
        original=tolerance.TolerancePolicy
        method=original.effective_area
        with pytest.raises(QualificationDecline) as caught:capture_callable_manifest(roots)
        assert type(caught.value) is QualificationDecline and str(caught.value)==reason
        assert tolerance.TolerancePolicy is original and original.effective_area is method
        request.node.user_properties.extend((('execution_scope','initial-legacy-dataclass-wrapper-refusal'),
            ('runtime_refusal',reason),('original_test_body_executed',False),('postcapture_qualification',False)))
        return
    captured=capture_with_runtime_boundary(roots,request=request)
    if captured is not None:captured.verify()
