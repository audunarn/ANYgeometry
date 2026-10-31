"""PP1f actual gate admission attempt, named decisions and honest cost evidence."""
from collections import Counter
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import psutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import scipy.linalg
from threadpoolctl import threadpool_limits, threadpool_info
from anygeometry import _planning_ledger as l
from anygeometry import _planning_qualification as q
from anygeometry._planning_canaries import startup_canaries
OUT=Path(os.environ['TASK_EVIDENCE_DIRECTORY'])
def save(name,value): (OUT/name).write_text(json.dumps(value,indent=2))
def witness(value):
    if type(value) is dict: data=[(id(k),id(v)) for k,v in value.items()]
    elif type(value) in (list,set): data=[id(v) for v in value]
    elif type(value) is functools._lru_cache_wrapper: data=list(value.cache_info())
    else: data=[id(value)]
    return hashlib.sha256(json.dumps(data).encode()).hexdigest()
with threadpool_limits(limits=1):
    load=dict(cpu_start=psutil.cpu_percent(interval=.1),logical_cpus=os.cpu_count())
    ledger=l.loaded_geometry_ledger()
    previous=json.loads((ROOT/'.pp1-evidence/pp1e-measure-01/nonempty-require-empty.json').read_text())
    decisions=[]
    for old in previous:
        path=old['path'];row=ledger.rows[path];value=ledger.objects[path]
        if path==l._LIVE_NAMESPACE_PATH:
            reason='Exact registered implementation globals; names/object/code identities plus code SHA256 and owner source pin; never clear. Builtins separately decline.'
        elif path.endswith('._finfo_cache'):
            reason='DECLINE: values are opaque numpy.finfo instances; safe disposal and alias independence not proved. No reset asserted.'
        elif path.endswith('.__builtins__'):
            reason='DECLINE: interactive builtins helper instances and their mutable state lack an admitted structural schema.'
        elif path in l._DECLINED_RUNTIME_PATHS:
            reason='DECLINE: NumPy namespace contains native dispatchers/partials/opaque runtime state; a shallow binding pin alone does not qualify transitive state.'
        else:
            reason='Exact named content guarded with typed digest before/after every task; plain static metadata/constant containers; never cleared.'
        decisions.append(dict(path=path,action=row['action'],reason=reason,size=len(value)))
    assert len(decisions)==22
    save('decisions.json',decisions)
    save('ledger.json',ledger.record())
    expected=q.capture_environment(generated_modules=('cython_runtime','_cython_3_2_4'),declined_stores=ledger.resets)
    expected['live_code_pins']=ledger.live_code_pins()
    save('environment.json',expected)
    gate=q.WorkerGate(expected,stores=ledger.resets,owned_worker=True,
        generated_modules=('cython_runtime','_cython_3_2_4'),ledger=ledger)
    watched={p:v for p,v in ledger.objects.items() if p in ledger.resets or p in l._RUNTIME_EMPTY_PATHS}
    before={p:witness(v) for p,v in watched.items()}
    goldens=json.loads((ROOT/'tests/data/pp1b-canaries.json').read_text())
    start=time.perf_counter()
    refusal=None
    try: gate.start(startup_canaries(),goldens)
    except q.QualificationDecline as exc: refusal=str(exc)
    duration=time.perf_counter()-start
    unchanged=before=={p:witness(v) for p,v in watched.items()}
    if refusal is not None:
        assert gate.poisoned and not gate.ready and ledger.costs==[] and unchanged
    load['cpu_end']=psutil.cpu_percent(interval=.1)
    result=dict(admitted=gate.ready,refusal=refusal,admission_attempt_seconds=duration,
        canary_startup_seconds=duration if gate.ready else None,
        reset_seconds=ledger.costs,per_task_reset_seconds=None,
        measured_admitted_task_count=0,stores=len(ledger.rows),
        decisions=dict(Counter(row['action'] for row in ledger.rows.values())),
        nonempty_decisions=dict(Counter(row['action'] for row in decisions)),
        declined=[dict(path=p,reason=row['reason']) for p,row in ledger.rows.items() if row['action']=='decline'],
        watched_store_count=len(watched),watched_unchanged=unchanged,
        verification_seconds=gate.verification_seconds,census_seconds=ledger.census_costs,
        files=len(expected['files']),modules=len(expected['modules']),
        live_code_function_count=sum(len(v) for v in expected['live_code_pins'].values()),
        load=load,blas_threads=[x['num_threads'] for x in threadpool_info()],
        clock_resolution=time.get_clock_info('perf_counter').resolution,
        unreachable_exclusions=0)
    save('measurement.json',result)
    print(json.dumps(result))
