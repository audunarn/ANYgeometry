"""Unknown pre-import NumPy state must not add OFF import-time execution."""
import os
from pathlib import Path
import subprocess
import sys
import pytest


@pytest.mark.parametrize('case',['properties','empty_cell','json_fake'])
def test_unknown_adapter_keeps_off_import_safe(tmp_path,case):
    script='''
import numpy as np
original=np.linalg.norm
if CASE=='properties':
    class Foreign:
        def __call__(self,*a,**k):return original(*a,**k)
        @property
        def _implementation(self):raise AssertionError('custom property executed during OFF import')
        @property
        def __wrapped__(self):raise AssertionError('custom property executed during OFF import')
    np.linalg.norm=Foreign()
elif CASE=='empty_cell':
    def make():
        hidden=1
        def foreign(*a,**k):return original(*a,**k) if a else hidden
        del hidden
        return foreign
    np.linalg.norm=make()
else:
    import json,_json
    class Encoder:
        __module__='_json'
        def __call__(self,*args):raise AssertionError('fake native encoder executed')
    _json.make_encoder=Encoder
    json.encoder.c_make_encoder=Encoder
import anygeometry
from anygeometry import model,_planning_runtime_anchor
from anygeometry._planning_qualification import QualificationDecline
try:_planning_runtime_anchor.verify_owner(model)
except QualificationDecline:pass
else:raise AssertionError('unsupported runtime was admitted')
assert not anygeometry.GeometryModel().faces
'''.replace('CASE',repr(case))
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'),
             ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING='0',PYTHONDONTWRITEBYTECODE='1')
    result=subprocess.run([sys.executable,'-B','-c',script],cwd=tmp_path,env=env,
        capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stdout+result.stderr
