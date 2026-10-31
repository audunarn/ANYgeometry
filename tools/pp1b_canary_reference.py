import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from anygeometry._planning_canaries import startup_canaries
from threadpoolctl import threadpool_limits, threadpool_info

with threadpool_limits(limits=1):
    pools = threadpool_info()
    assert pools and all(p['num_threads'] == 1 for p in pools)
    result = {name: task() for name, task in startup_canaries().items()}
    result['threads_verified'] = all(p['num_threads'] == 1 for p in threadpool_info())
    assert result['threads_verified']
result['kernel_sha256'] = {name: hashlib.sha256((ROOT/'src/anygeometry'/name).read_bytes()).hexdigest()
                          for name in ('batch_intersections.py', 'material_arrangement.py')}
folder = Path(os.environ['TASK_EVIDENCE_DIRECTORY'])
(folder/'canary-reference.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
