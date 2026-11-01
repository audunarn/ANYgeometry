"""Inject a late module into the probe's planar exercise; it must detect it."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import inspect
import importlib.machinery
import json
import pathlib
import runpy
import sys
import types
import shapely.ops

original=shapely.ops.unary_union
name='_pp5d_test_lazy_import'
calls=[]

def injected(*args,**kwargs):
    if pathlib.Path(inspect.currentframe().f_back.f_code.co_filename).name=='pp5d_prewarm_check.py':
        calls.append(True)
        module=types.ModuleType(name)
        module.__spec__=importlib.machinery.ModuleSpec(name,None,origin='built-in')
        sys.modules[name]=module
    return original(*args,**kwargs)

shapely.ops.unary_union=injected
try:
    try:runpy.run_path(str(pathlib.Path(__file__).with_name('pp5d_prewarm_check.py')),run_name='__main__')
    except AssertionError as error:
        assert name in str(error),str(error)
        assert calls==[True]
    else:raise AssertionError('probe silently accepted a late import during the planar exercise')
finally:
    shapely.ops.unary_union=original
    sys.modules.pop(name,None)
print(json.dumps(dict(status='passed',injected_late_module_refused=True)))
