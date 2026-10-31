"""Fresh-process engineering gate rehearsal; pair/face subsets only."""
import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from anygeometry._planning_qualification import capture_environment, WorkerGate
from anygeometry._planning_canaries import startup_canaries, pair_canary, face_canary
from anygeometry import branch_algebra, branch_curves
import numpy
import scipy.linalg
from threadpoolctl import threadpool_limits, threadpool_info

folder = Path(os.environ['TASK_EVIDENCE_DIRECTORY'])
mode, reference = sys.argv[1:]
reference = Path(reference)
expected_canaries = json.loads((ROOT/'tests/data/pp1b-canaries.json').read_text())
generated = ('cython_runtime', '_cython_3_2_4')
stores = {'anygeometry.branch_algebra.directrix_of': branch_algebra.directrix_of,
          'anygeometry.branch_algebra._cached_poly_plan': branch_algebra._cached_poly_plan,
          'anygeometry.branch_curves._cached_bounds': branch_curves._cached_bounds}
with threadpool_limits(limits=1):
    if mode == 'reference':
        # Reference assembly is a rehearsal, never automatic admission. Numerical
        # oracle digests above were recorded before qualification implementation.
        actual = {name: task() for name, task in startup_canaries().items()}
        assert actual == expected_canaries
        pin = capture_environment(generated_modules=generated, declined_stores=stores)
        (folder/'environment.json').write_text(json.dumps(pin, indent=2))
        print(json.dumps(dict(kind='reference rehearsal', modules=len(pin['modules']),
            files=len(pin['files']), blas_pools=len(pin['blas']), canaries=actual)))
    elif mode == 'worker':
        pin = json.loads(reference.read_text())
        # The complete declared module list is imported before the candidate
        # snapshot. An extra/missing module still rejects at the gate.
        for name in pin['modules']:
            if name not in sys.modules:
                importlib.import_module(name)
        gate = WorkerGate(pin, stores=stores, owned_worker=True, generated_modules=generated)
        gate.start(startup_canaries(), expected_canaries)
        actual = dict(pair=gate.run(pair_canary), face=gate.run(face_canary))
        assert actual == expected_canaries
        result = dict(kind='worker rehearsal', startup_passed=True, task_digests=actual,
                      reset_counts=gate.reset_counts,
                      threads=[p['num_threads'] for p in threadpool_info()])
        (folder/'worker.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
    else:
        raise ValueError('unknown probe mode')
