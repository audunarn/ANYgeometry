"""Non-numerical preflight; reports module and binding, never scans native exports."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import scipy.linalg
from anygeometry import _planning_qualification as q
from anygeometry._planning_canaries import startup_canaries  # import only; no execution

declared = ('cython_runtime', '_cython_3_2_4')
rows = []
for name, module in tuple(sorted(sys.modules.items())):
    try:
        pin = q.pin_module(module)
        rows.append(dict(module=name, binding='__file__/__spec__', status='pinned', pin=pin))
    except Exception as exc:
        rows.append(dict(module=name, binding='__file__/__spec__',
                         status='explicit-generated' if name in declared else 'declined',
                         error=type(exc).__name__ + ': ' + str(exc)))
result = dict(numerical_tasks=0, exported_native_types_enumerated=0, bindings=rows,
              failures=[r for r in rows if r['status']=='declined'])
out = Path(os.environ['TASK_EVIDENCE_DIRECTORY'])
(out/'preflight.json').write_text(json.dumps(result, indent=2))
print(json.dumps(dict(modules=len(rows), numerical_tasks=0,
                     failures=result['failures'], explicit_generated=list(declared))))
assert not result['failures']
