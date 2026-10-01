"""Record the original replay using installed released owner capabilities."""
import json
import sys
from pathlib import Path
from importlib.metadata import version
from time import perf_counter
import inspect

from replay import project
from anygeometry import to_dict
import anymesher
from verify_replay import verify

made,commands=project()
before=to_dict(made.geometry)
size=float(sys.argv[1])
start=perf_counter()
options={'strategy':'auto','order':'linear'}
available=hasattr(anymesher,'MeshAutomationOptions')
if available and 'automation' in inspect.signature(made.generate_mesh).parameters:
    options['automation']=anymesher.MeshAutomationOptions()
report={'versions':{name:version(name) for name in ('ANYgeometry','ANYmesher','ANYfem')},
        'target_size':size,'automation_api_available':available,
        'baseline_comparison':True,'accepted_mesh':False}
try:
    mesh=made.generate_mesh(size,**options)
    report['checks']=verify(mesh,made.geometry)
    report.update(accepted_mesh=True,nodes=len(mesh.nodes),elements=len(mesh.shells))
except Exception as error:
    chain=[]
    while error is not None:
        chain.append({'type':type(error).__name__,'message':str(error)})
        error=error.__cause__
    report['failure_chain']=chain
finally:
    assert to_dict(made.geometry)==before
    report.update(source_unchanged=True,seconds=perf_counter()-start)
    Path(__file__).with_name(f'baseline-replay-{round(size*100):03d}.json').write_text(
        json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2,allow_nan=False),flush=True)
