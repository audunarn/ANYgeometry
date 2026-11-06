"""F5 bounded costs and strict admission refusal; no numerical task is launched."""
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
import types

import psutil
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import scipy.linalg
from threadpoolctl import threadpool_limits, threadpool_info
from anygeometry import _planning_qualification as q
from anygeometry import _planning_ledger as l
from anygeometry import _planning_purity as p
from anygeometry._planning_canaries import startup_canaries

folder = Path(os.environ['TASK_EVIDENCE_DIRECTORY'])
def save(name, value):
    (folder / name).write_text(json.dumps(value, indent=2))

with threadpool_limits(limits=1):
    load = dict(cpu_start=psutil.cpu_percent(interval=.1), logical_cpus=os.cpu_count())
    ledger = l.loaded_geometry_ledger()
    save('ledger.json', ledger.record())
    declines = [row for row in ledger.rows.values() if row['action'] == 'decline']
    started = time.perf_counter()
    expected = q.capture_environment(generated_modules=('cython_runtime', '_cython_3_2_4'),
                                     declined_stores=ledger.resets)
    baseline_capture = time.perf_counter() - started
    save('environment.json', expected)
    verification = []
    for _ in range(2):
        started = time.perf_counter()
        observed = q.capture_environment(generated_modules=('cython_runtime', '_cython_3_2_4'),
                                         declined_stores=ledger.resets)
        q.verify_environment(expected, observed)
        verification.append(time.perf_counter() - started)
    started = time.perf_counter()
    try:
        ledger.prepare(owned_worker=True)
    except q.QualificationDecline as exc:
        refusal = str(exc)
    else:
        raise AssertionError('strict full closure unexpectedly admitted; inspect before running tasks')
    refusal_seconds = time.perf_counter() - started
    # Separate supported microfixture; no bypass of the full ledger above.
    module = types.ModuleType('anygeometry.pp1d_reset_fixture')
    module._MEMO = {i: i for i in range(1000)}
    objects = {}
    report = p.inventory((module,), store_objects=objects)
    fixture = l.build_ledger(report, objects, modules=(module,))
    counts = fixture.prepare(owned_worker=True)
    assert module._MEMO == {} and list(counts.values()) == [0]
    load['cpu_end'] = psutil.cpu_percent(interval=.1)
    result = dict(qualified=False, numerical_tasks=0, declined_stores=len(declines),
        stores=len(ledger.rows), decisions=dict(Counter(row['action'] for row in ledger.rows.values())),
        baseline_capture_seconds=baseline_capture, full_verification_seconds=verification,
        double_verification_seconds=sum(verification), files=len(expected['files']),
        modules=len(expected['modules']), runtime_files=q.runtime_files(),
        full_ledger_refusal=refusal, full_ledger_refusal_seconds=refusal_seconds,
        full_ledger_reset_measured=False, reset_fixture=dict(stores=1, entries_before=1000,
            entries_after=0, reset_seconds=fixture.costs, census_seconds=fixture.census_costs),
        load=load, blas_threads=[v['num_threads'] for v in threadpool_info()],
        clock_resolution=time.get_clock_info('perf_counter').resolution)
    save('measurement.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'runtime_files'}))
