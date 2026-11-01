"""Fresh-process startup closure and native dependency pin regression."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import json
import pathlib
import runpy
import sys

root=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
child=runpy.run_path(str(root/'src/anygeometry/_planning_child.py'),run_name='pp5d_startup_check')
child['prepare']()
before=set(sys.modules)
from shapely.geometry import Polygon
from shapely.ops import unary_union
from anygeometry import branch_events, quadric_events, branch_wall_events
assert set(sys.modules)==before, sorted(set(sys.modules)-before)
assert unary_union([Polygon([(0.,0.),(1.,0.),(0.,1.)])]).area==0.5
assert set(sys.modules)==before, sorted(set(sys.modules)-before)
q=child['q']
pins=q.runtime_files()
assert len([p for p in pins if pathlib.Path(p).name.startswith('geos')])==2
assert any(pathlib.Path(p).name.startswith('msvcp140-') for p in pins)
expected=q.capture_environment(generated_modules=child['_GENERATED_MODULES'],trusted_image=True)
assert expected['modules']['_cython_3_1_4']['kind']=='explicit-generated-native'
assert all(expected['files'][p]==h for p,h in pins.items())
observed=q.capture_environment(generated_modules=child['_GENERATED_MODULES'],trusted_image=True)
q.verify_environment(expected,observed)
geos=next(p for p in pins if pathlib.Path(p).name.startswith('geos'))
observed['files'][geos]='0'*64
try:q.verify_environment(expected,observed)
except q.QualificationDecline:pass
else:raise AssertionError('changed GEOS dependency pin accepted')
out=pathlib.Path(os.environ['TASK_EVIDENCE_DIRECTORY'])
(out/'closure.json').write_text(json.dumps(dict(status='passed',late_modules=0,runtime_files=pins,
    generated=child['_GENERATED_MODULES'],changed_dependency_refused=True),indent=2))
print(json.dumps(dict(status='passed',late_modules=0,runtime_files=len(pins),changed_dependency_refused=True)))
