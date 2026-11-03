"""Base geometry import supports the declared NumPy1.x namespace."""
import os, subprocess, sys
import pytest
from pathlib import Path

@pytest.mark.parametrize('missing', ['numpy._core', 'numpy._core._multiarray_umath'])
def test_geometry_import_when_numpy2_private_namespace_is_absent(missing):
    code = """import builtins, sys
original = builtins.__import__
def legacy(name, *args, **kwargs):
    if name == 'numpy._core._multiarray_umath':
        raise ModuleNotFoundError('NumPy2 namespace absent', name=sys.argv[1])
    return original(name, *args, **kwargs)
import numpy
builtins.__import__ = legacy
import anygeometry
assert anygeometry.GeometryModel().validate_topology() == ()
"""
    result=subprocess.run([sys.executable,'-B','-c',code,missing],capture_output=True,text=True,timeout=20,
        env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src')))
    assert result.returncode == 0, result.stderr


def test_unrelated_native_import_error_is_not_hidden():
    code = """import builtins
import numpy
original=builtins.__import__
def broken(name,*args,**kwargs):
    if name == 'numpy._core._multiarray_umath':
        raise ModuleNotFoundError('unrelated missing dependency',name='unexpected_dependency')
    return original(name,*args,**kwargs)
builtins.__import__=broken
try:
    import anygeometry
except ModuleNotFoundError as error:
    assert error.name=='unexpected_dependency'
else:
    raise AssertionError('unrelated import error hidden')
"""
    result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,timeout=20,
        env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src')))
    assert result.returncode==0,result.stderr
