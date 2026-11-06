"""Full-census decisions and boundary costs; no reachability exclusions or tasks."""
from collections import Counter
import contextvars
import functools
import hashlib
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

OUT = Path(os.environ['TASK_EVIDENCE_DIRECTORY'])
def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2))

def witness(value):
    if type(value) is dict:
        data = [(id(k), id(v)) for k, v in dict.items(value)]
    elif type(value) in (list, set):
        data = [id(v) for v in value]
    elif type(value) is functools._lru_cache_wrapper:
        data = list(value.cache_info())
    else:
        data = [id(value)]
    return hashlib.sha256(json.dumps(data).encode()).hexdigest()

with threadpool_limits(limits=1):
    load = dict(cpu_start=psutil.cpu_percent(interval=.1), logical_cpus=os.cpu_count())
    started = time.perf_counter()
    ledger = l.loaded_geometry_ledger()
    ledger_build = time.perf_counter() - started
    save('ledger.json', ledger.record())
    prior = json.loads((ROOT/'.pp1-evidence/pp1d-measure-alias-final/ledger.json').read_text())
    previous = [row for row in prior['stores'] if row['action'] == 'decline']
    delta = [dict(path=row['path'], previous_reason=row['reason'],
                  current=ledger.rows.get(row['path'], {'action': 'path-changed-review-required'}))
             for row in previous]
    save('original-413-decisions.json', delta)
    nonempty = []
    for path, row in ledger.rows.items():
        if row['action'] != 'require-empty':
            continue
        value = ledger.objects[path]
        if type(value) is contextvars.ContextVar:
            value = value.get(None)
            if value is None or (path == 'anygeometry.arrangement_geometry._JUNCTION_BOUNDS' and value is False):
                continue
            active = not (type(value) in (tuple, list, dict, set) and len(value) == 0)
            size = None
        else:
            size = len(value)
            active = size != 0
        if active:
            nonempty.append(dict(path=path, kind=row['kind'], size=size))
    save('nonempty-require-empty.json', nonempty)
    expected = q.capture_environment(generated_modules=('cython_runtime', '_cython_3_2_4'),
                                     declined_stores=ledger.resets)
    save('environment.json', expected)
    costs = []
    for _ in range(2):
        started = time.perf_counter()
        observed = q.capture_environment(generated_modules=('cython_runtime', '_cython_3_2_4'),
                                         declined_stores=ledger.resets)
        q.verify_environment(expected, observed)
        costs.append(time.perf_counter() - started)
    watched = {path: value for path, value in ledger.objects.items()
               if path in ledger.resets or path in l._RUNTIME_EMPTY_PATHS}
    before = {path: witness(value) for path, value in watched.items()}
    started = time.perf_counter()
    try:
        ledger.prepare(owned_worker=True)
    except q.QualificationDecline as exc:
        refusal = str(exc)
    else:
        raise AssertionError('admission changed; inspect before authorizing any numerical task')
    refusal_seconds = time.perf_counter() - started
    assert before == {path: witness(value) for path, value in watched.items()}
    assert ledger.costs == []
    fixture = types.ModuleType('anygeometry.pp1e_reset_fixture')
    fixture._MEMO = {i: i for i in range(1000)}
    objects = {}
    report = p.inventory((fixture,), store_objects=objects)
    small = l.build_ledger(report, objects, modules=(fixture,))
    small.prepare(owned_worker=True)
    assert fixture._MEMO == {}
    load['cpu_end'] = psutil.cpu_percent(interval=.1)
    result = dict(qualified=False, numerical_tasks=0, unreachable_exclusions=0,
        stores=len(ledger.rows), decisions=dict(Counter(row['action'] for row in ledger.rows.values())),
        original_413_decisions=dict(Counter(row['current']['action'] for row in delta)),
        nonempty_require_empty=len(nonempty), watched_unchanged=len(watched),
        first_refusal=refusal, ledger_build_seconds=ledger_build,
        full_verification_seconds=costs, double_verification_seconds=sum(costs),
        refusal_seconds=refusal_seconds, full_reset_measured=False,
        reset_fixture=dict(stores=1, entries_before=1000, entries_after=0,
            reset_seconds=small.costs, census_seconds=small.census_costs),
        files=len(expected['files']), modules=len(expected['modules']), load=load,
        blas_threads=[row['num_threads'] for row in threadpool_info()],
        clock_resolution=time.get_clock_info('perf_counter').resolution)
    save('measurement.json', result)
    print(json.dumps(result))
