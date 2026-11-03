"""Real source-bound classes retain every owned definition binding.

R5 supersedes selected-body-only mutation acceptance. Synthetic package names
cannot establish source provenance; the old spoof fixture remains a refusal.
"""
import sys
import types
import pytest
from anygeometry._planning_manifest import capture_callable_manifest
from anygeometry._planning_executor import PlannerSourceChanged
from anygeometry._planning_qualification import QualificationDecline


def owned(monkeypatch):
    module=types.ModuleType('anygeometry.ppq_finite_type_fixture')
    module.__file__=__file__
    exec('class Record:\n def selected(self):return 1\n def unused(self,opaque=object()):return opaque\ndef root(value):return value.selected()\n',vars(module))
    monkeypatch.setitem(sys.modules,module.__name__,module)
    return module


def test_type_root_symbol_replacement_is_terminal(monkeypatch):
    from anygeometry import tolerance
    manifest=capture_callable_manifest((tolerance.TolerancePolicy,))
    monkeypatch.setattr(tolerance,'TolerancePolicy',type('Other',(),{}))
    with pytest.raises(PlannerSourceChanged):manifest.verify()


def test_selected_method_rebinding_is_terminal(monkeypatch):
    from anygeometry.tolerance import TolerancePolicy
    manifest=capture_callable_manifest((TolerancePolicy.effective_length,TolerancePolicy))
    monkeypatch.setattr(TolerancePolicy,'effective_length',lambda self,scale:2)
    with pytest.raises(PlannerSourceChanged):manifest.verify()


def test_unselected_owned_method_rebinding_is_terminal(monkeypatch):
    from anygeometry.tolerance import TolerancePolicy
    manifest=capture_callable_manifest((TolerancePolicy.effective_length,TolerancePolicy))
    monkeypatch.setattr(TolerancePolicy,'effective_area',lambda self,scale:2)
    with pytest.raises(PlannerSourceChanged):manifest.verify()


def test_outside_package_source_spoof_refuses_initial_capture(monkeypatch):
    module=owned(monkeypatch)
    with pytest.raises(QualificationDecline,match='source outside package'):
        capture_callable_manifest((module.root,module.Record))


def test_exact_builtin_type_constructor_binds_interpreter_image():
    manifest=capture_callable_manifest((BaseException.__new__,))
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
def test_type_metadata_drift_is_terminal(full):
    value={'safe':1};captured=capture_callable_manifest((value,))
    class Unknown:__module__=None
    value['new']=Unknown()
    with pytest.raises(PlannerSourceChanged):captured.verify(full=full)
