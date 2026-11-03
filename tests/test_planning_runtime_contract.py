"""Named provider contracts bind loaded adapters, not only source images."""
import pytest
from anygeometry._planning_executor import PlannerSourceChanged
from anygeometry._planning_qualification import QualificationDecline


def capture(provider,name):
    from anygeometry._planning_runtime import capture_operation
    return capture_operation(provider,name)


def test_numpy_dispatcher_has_live_python_adapter_binding(monkeypatch):
    import numpy as np
    binding=capture('numpy','einsum')
    monkeypatch.setattr(np,'einsum',lambda *a,**k:0)
    with pytest.raises(PlannerSourceChanged):binding.verify()


def test_numpy_wrapped_implementation_code_is_bound(monkeypatch):
    import numpy as np
    binding=capture('numpy','iscomplexobj')
    implementation=np.iscomplexobj.__wrapped__
    monkeypatch.setattr(implementation,'__code__',(lambda *a,**k:False).__code__)
    with pytest.raises(PlannerSourceChanged):binding.verify()


def test_unknown_runtime_operation_is_not_admitted():
    with pytest.raises(QualificationDecline):capture('numpy','arbitrary_unregistered_operation')


@pytest.mark.parametrize('keyword',[False,True])
def test_runtime_defaults_do_not_execute_unknown_repr(keyword):
    from anygeometry._planning_runtime import _fingerprint
    class Foreign:
        def __repr__(self):pytest.fail('unknown default repr executed')
    def function(value=None,*,option=None):return value
    if keyword:function.__kwdefaults__={'option':Foreign()}
    else:function.__defaults__=(Foreign(),)
    with pytest.raises(QualificationDecline):_fingerprint(function,[],[])


def test_runtime_binding_survives_unrelated_module(monkeypatch):
    import sys,types
    binding=capture('numpy','einsum')
    unrelated=types.ModuleType('ppq_extra_gui_module');unrelated.something=object()
    monkeypatch.setitem(sys.modules,unrelated.__name__,unrelated)
    binding.verify()
