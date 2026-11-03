"""Base geometry import supports the declared NumPy1.x namespace."""
import os, subprocess, sys
from pathlib import Path

def test_geometry_import_when_numpy2_private_namespace_is_absent():
    code = """import builtins
original = builtins.__import__
def legacy(name, *args, **kwargs):
    if name == 'numpy._core._multiarray_umath':
        raise ModuleNotFoundError('NumPy2 namespace absent', name='numpy._core')
    return original(name, *args, **kwargs)
import numpy
builtins.__import__ = legacy
import anygeometry
assert anygeometry.GeometryModel().validate_topology() == ()
"""
    result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,timeout=20,
        env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src')))
    assert result.returncode == 0, result.stderr
