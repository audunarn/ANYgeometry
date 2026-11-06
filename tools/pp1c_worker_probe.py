"""PP1c preflight, reference, and fresh-worker rehearsal (no full-model runs)."""
from collections import Counter
import importlib
import json
import os
from pathlib import Path
import sys
import time

import psutil
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy
import scipy.linalg
from threadpoolctl import threadpool_limits, threadpool_info
from anygeometry import _planning_qualification as q
from anygeometry._planning_ledger import loaded_geometry_ledger
from anygeometry._planning_canaries import startup_canaries, pair_canary, face_canary

GENERATED = ('cython_runtime', '_cython_3_2_4')
folder = Path(os.environ['TASK_EVIDENCE_DIRECTORY'])
mode = sys.argv[1]
expected_canaries = json.loads((ROOT/'tests/data/pp1b-canaries.json').read_text())
load = dict(cpu_percent_start=psutil.cpu_percent(interval=.1), logical_cpus=os.cpu_count())

def save(name, value):
    (folder/name).write_text(json.dumps(value, indent=2))

with threadpool_limits(limits=1):
    if mode == 'worker':
        expected = json.loads(Path(sys.argv[2]).read_text())
        for name in expected['modules']:
            if name not in sys.modules:
                importlib.import_module(name)
    # Pin-module preflight records each failing binding before any numerical task.
    preflight = []
    for name, module in tuple(sorted(sys.modules.items())):
        if name in GENERATED:
            continue
        try:
            q.pin_module(module)
        except Exception as exc:
            preflight.append(dict(module=name, binding='__file__/__spec__', error=str(exc)))
    save('preflight.json', dict(failures=preflight, numerical_tasks_before_preflight=0,
                               exported_native_types_enumerated=0))
    assert not preflight
    if mode == 'reference':
        warm = {name: task() for name, task in startup_canaries().items()}
        assert warm == expected_canaries
    ledger = loaded_geometry_ledger()
    save('ledger.json', ledger.record())
    decisions = dict(Counter(row['action'] for row in ledger.rows.values()))
    declined = [path for path, row in ledger.rows.items() if row['action']=='decline']
    if mode == 'preflight':
        print(json.dumps(dict(stores=len(ledger.rows), decisions=decisions, declined=declined)))
    elif mode == 'reference':
        ledger.prepare(owned_worker=True)
        pin = q.capture_environment(generated_modules=GENERATED, declined_stores=ledger.resets)
        save('environment.json', pin)
        save('reference.json', dict(warm=warm, decisions=decisions, load=load))
        print(json.dumps(dict(modules=len(pin['modules']), files=len(pin['files']),
                             stores=len(ledger.rows), decisions=decisions)))
    elif mode == 'worker':
        gate = q.WorkerGate(expected, stores=ledger.resets, owned_worker=True,
                           generated_modules=GENERATED, ledger=ledger)
        started = time.perf_counter()
        try:
            gate.start(startup_canaries(), expected_canaries)
        except Exception:
            observed = q.capture_environment(generated_modules=GENERATED, declined_stores=ledger.resets)
            save('decline-observed-environment.json', observed)
            save('decline-fields.json', {key: dict(expected=expected.get(key), observed=observed.get(key))
                 for key in set(expected)|set(observed) if expected.get(key)!=observed.get(key)})
            raise
        startup_seconds = time.perf_counter()-started
        warm = {name: task() for name, task in startup_canaries().items()}
        reset = dict(pair=gate.run(pair_canary), face=gate.run(face_canary))
        assert reset == warm == expected_canaries
        load['cpu_percent_end'] = psutil.cpu_percent(interval=.1)
        result = dict(startup_passed=gate.ready, warm=warm, reset=reset,
                      warm_reset_equal=warm==reset, startup_seconds=startup_seconds,
                      reset_seconds=ledger.costs, clock_resolution=time.get_clock_info('perf_counter').resolution,
                      load=load, stores=len(ledger.rows), decisions=decisions,
                      blas_threads=[p['num_threads'] for p in threadpool_info()])
        save('worker.json', result)
        save('ledger.json', ledger.record())
        print(json.dumps(result))
    else:
        raise ValueError('unknown mode')
