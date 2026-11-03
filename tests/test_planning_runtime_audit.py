"""External interfaces are pinned; no whole-interpreter audit-absence claim."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from _planning_contract_helpers import capture_with_runtime_boundary
from anygeometry._planning_executor import PlannerSourceChanged
from anygeometry._planning_qualification import pin_module


def test_imported_external_module_is_bound_to_its_actual_source_image():
    pin = pin_module(json)
    assert pin['kind'] == 'python'
    assert pin['path'] == str(Path(json.__file__).resolve())
    assert pin['sha256'] == hashlib.sha256(Path(json.__file__).read_bytes()).hexdigest()


@pytest.mark.parametrize('change', ['symbol', 'module'])
def test_owned_caller_imported_numpy_binding_drift_is_terminal(monkeypatch, change):
    from anygeometry import tolerance
    captured = capture_with_runtime_boundary((tolerance._positive_finite,))
    if captured is None:
        return
    captured.verify()
    if change == 'symbol':
        monkeypatch.setattr(tolerance.np, 'isfinite', lambda *args: True)
    else:
        monkeypatch.setattr(tolerance, 'np', object())
    with pytest.raises(PlannerSourceChanged):
        captured.verify()


def test_unrelated_interpreter_module_is_not_a_package_owned_definition(monkeypatch):
    import types
    from anygeometry.tolerance import TolerancePolicy
    captured = capture_with_runtime_boundary((TolerancePolicy,))
    if captured is None:
        return
    foreign = types.ModuleType('r10_unrelated_interpreter_state')
    foreign.value = object()
    monkeypatch.setitem(sys.modules, foreign.__name__, foreign)
    captured.verify()
    foreign.value = object()
    captured.verify()


@pytest.mark.parametrize('when', ['before', 'after'])
def test_benign_python_audit_observer_does_not_imply_a_purity_claim(tmp_path, when):
    # Audit hooks cannot be removed, so exercise the trust boundary in a child.
    script = '''
import json,sys
import numpy as numpy_interface
from anygeometry._planning_manifest import capture_callable_manifest
calls=[]
def observer(event,args):calls.append(event)
if sys.argv[1]=='before':sys.addaudithook(observer)
from _planning_contract_helpers import capture_with_runtime_boundary
captured=capture_with_runtime_boundary((numpy_interface.geterr,))
if captured is None:
    print(json.dumps(dict(initial_refused=True,verified=False)))
    sys.exit(0)
if sys.argv[1]=='after':sys.addaudithook(observer)
captured.verify()
# We deliberately make no callback-count/absence or native-audit-list claim.
print(json.dumps(dict(verified=True,observer_installed=True)))
'''
    import anygeometry
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(Path(anygeometry.__file__).resolve().parent.parent), str(Path(__file__).resolve().parent))))
    result = subprocess.run([sys.executable, '-B', '-c', script, when], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=20)
    (tmp_path / 'stdout.txt').write_text(result.stdout)
    (tmp_path / 'stderr.txt').write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    from _planning_contract_helpers import runtime_boundary_refusal
    expected = (dict(initial_refused=True, verified=False) if runtime_boundary_refusal()
                else dict(verified=True, observer_installed=True))
    assert json.loads(result.stdout) == expected
