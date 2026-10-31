"""PP1h risk-accepted qualification, not a purity proof: real child + parent samples."""
import ctypes
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from collections import Counter
import psutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import scipy.linalg
from threadpoolctl import threadpool_limits,threadpool_info
from anygeometry import _planning_ledger as l
from anygeometry import _planning_qualification as q
from anygeometry._planning_admission import run_qualified
from anygeometry._planning_batch import BatchWorkerGate
from anygeometry._planning_canaries import startup_canaries
spec=importlib.util.spec_from_file_location('pp1h_owned_job',Path.home()/'bin/claude-review.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
OUT=Path(os.environ['TASK_EVIDENCE_DIRECTORY'])
GENERATED=('cython_runtime','_cython_3_2_4')

def save(name,value): (OUT/name).write_text(json.dumps(value,indent=2))
def load():
    # Requested aggregate load only; do not expose or inspect unrelated PIDs/commands.
    python_count=sum(1 for process in psutil.process_iter(['name'],ad_value='')
                     if (process.info['name'] or '').lower() in ('python.exe','pythonw.exe','python','python3'))
    return dict(python_process_count=python_count,cpu_percent=psutil.cpu_percent(interval=.1),logical_cpus=os.cpu_count())

def child_main():
    expected=json.loads((OUT/'reference.json').read_text())
    for name in expected['modules']:
        if name not in sys.modules: importlib.import_module(name)
    ledger=l.loaded_geometry_ledger(trusted_image=True)
    save('child-ledger-before.json',ledger.record())
    batch_size=json.loads((OUT/'configuration.json').read_text())['batch_size']
    gate=BatchWorkerGate(expected,stores=ledger.resets,owned_worker=True,
        ledger=ledger,generated_modules=GENERATED,batch_size=batch_size)
    tasks=json.loads((OUT/'tasks.json').read_text())
    canaries=startup_canaries()
    goldens=json.loads((ROOT/'tests/data/pp1b-canaries.json').read_text())
    report=dict(pid=os.getpid(),load_start=load(),status='starting')
    try:
        started=time.perf_counter();gate.start(canaries,goldens)
        report['canary_startup_seconds']=time.perf_counter()-started
        report['startup_resets']=list(ledger.costs)
        report['startup_passed']=gate.ready
        payload=b'{"args":[],"kwargs":{}}';digest=hashlib.sha256(payload).hexdigest()
        started=time.perf_counter()
        digests=gate.run_tasks(tuple((canaries[name],payload,digest) for name in tasks))
        assignment_seconds=time.perf_counter()-started
        report.update(status='admitted',task_count=len(tasks),digests=digests,
            batch_size=batch_size,assignment_seconds=assignment_seconds,
            per_task_reset_seconds=ledger.costs[2:],task_execution_seconds=gate.task_seconds,
            boundary_seconds=gate.boundary_seconds,full_environment_checks=len(gate.verification_seconds),
            verification_seconds=gate.verification_seconds,load_end=load(),
            blas_threads=[p['num_threads'] for p in threadpool_info()],
            decisions=dict(Counter(row['action'] for row in ledger.rows.values())),
            clock_resolution=time.get_clock_info('perf_counter').resolution)
        assert len(report['per_task_reset_seconds'])==len(tasks)
        save('child-ledger-after.json',ledger.record())
    except BaseException as exc:
        report.update(status='failed',error=type(exc).__name__+': '+str(exc),
                      startup_passed=gate.ready,reset_count=len(ledger.costs))
        try:
            observed=q.capture_environment(generated_modules=GENERATED,declined_stores=ledger.resets,trusted_image=True)
            save('child-observed.json',observed)
            save('child-pin-difference.json',{k:dict(expected=expected.get(k),observed=observed.get(k))
                for k in set(expected)|set(observed) if k!='live_code_pins' and expected.get(k)!=observed.get(k)})
        except BaseException as probe: report['diagnostic_error']=str(probe)
        raise
    finally: save('child.json',report)

def parent_main():
    batch_size=int(sys.argv[1])
    if batch_size not in (16,64): raise ValueError('outside PP1h stated measurement matrix')
    save('configuration.json',dict(batch_size=batch_size,tasks=200))
    old=json.loads((ROOT/'.pp1-evidence/pp1g-worker-01/reference.json').read_text())
    changed_gate=ROOT/'src/anygeometry/_planning_ledger.py'
    protected=[]
    for path,digest in old['files'].items():
        if Path(path).resolve()==changed_gate.resolve(): continue
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('prior pinned source/binary changed; no pin refresh: '+path)
        protected.append(path)
    save('unchanged-prior-pins.json',dict(files=protected,matched=len(protected),
        permitted_noncertified_gate_edit=str(changed_gate),certified_guard_refreshes=0))
    canaries=startup_canaries()
    goldens=json.loads((ROOT/'tests/data/pp1b-canaries.json').read_text())
    assert {name:task() for name,task in canaries.items()}==goldens
    ledger=l.loaded_geometry_ledger(trusted_image=True)
    save('parent-ledger.json',ledger.record())
    ledger.prepare(owned_worker=True)
    expected=q.capture_environment(generated_modules=GENERATED,declined_stores=ledger.resets,trusted_image=True)
    expected['live_code_pins']=ledger.live_code_pins()
    save('reference.json',expected)
    tasks=('pair',)*100+('face',)*100
    save('tasks.json',tasks)
    command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'worker']
    state=dict(command=command,cwd=str(ROOT),cap_seconds=90,load_start=load())
    job=None;child=None;started=None
    def start():
        nonlocal job,child,started
        job=helper.WindowsJob();started=time.perf_counter()
        with (OUT/'child-stdout.txt').open('wb') as out,(OUT/'child-stderr.txt').open('wb') as err:
            child=subprocess.Popen(command,cwd=ROOT,env=os.environ.copy(),stdout=out,stderr=err,
                creationflags=subprocess.CREATE_NO_WINDOW|4)
            state['owned_pid']=child.pid;job.assign_and_resume(child)
            try: state['exit_code']=child.wait(timeout=90)
            except subprocess.TimeoutExpired:
                state['status']='timeout';raise q.QualificationDecline('owned worker deadline')
        state['status']='passed' if state['exit_code']==0 else 'failed'
        if state['exit_code']!=0: raise q.QualificationDecline('owned worker admission/task failed; see child.json')
    def stop():
        if job is None: return
        if child is not None and child.poll() is None:
            actual=psutil.Process(child.pid).cmdline()
            if actual!=command: raise RuntimeError('owned command changed before cleanup')
            state['checked_before_cleanup']=actual
        state['cleanup']=job.cleanup()
        state['process_wall_seconds']=time.perf_counter()-started
        save('child-process.json',state)
        if not state['cleanup']['verified'] or state['cleanup']['active_processes']!=0:
            raise RuntimeError('owned worker cleanup failed')
    def parallel(submitted):
        assert tuple(submitted)==tasks
        return json.loads((OUT/'child.json').read_text())['digests']
    serial_calls=[]
    def serial(name):
        serial_calls.append(name)
        return canaries[name]()
    result=run_qualified(tasks,start_workers=start,stop_workers=stop,
        parallel_tasks=parallel,serial_task=serial,opt_in=True)
    sample_calls=list(serial_calls)
    serial_times=[];serial_results=[];serial_load_start=load()
    for name in tasks:
        started=time.perf_counter();serial_results.append(canaries[name]())
        serial_times.append(time.perf_counter()-started)
    assert tuple(serial_results)==result['results']
    result.update(serial_sample_tasks=sample_calls,load_end=load(),
        serial_load_start=serial_load_start,serial_task_seconds=serial_times,
        serial_task_names=tasks,batch_size=batch_size,
        blas_threads=[p['num_threads'] for p in threadpool_info()],
        qualification='RISK-ACCEPTED QUALIFICATION, not a purity proof',
        full_model=False,task_scope='200 small real kernel tasks, 100 pair and 100 face canaries',
        plan_apply_ci='unit hook tested; full PP6 control not exercised here')
    save('parent.json',result)
    assert result['mode']=='parallel' and result['samples']==10
    assert sample_calls==['pair']*5+['face']*5
    assert result['results']==tuple(goldens[name] for name in tasks)
    print(json.dumps({k:v for k,v in result.items() if k not in ('results','serial_task_seconds','serial_task_names')}))

if __name__=='__main__':
    with threadpool_limits(limits=1):
        if len(sys.argv)>1 and sys.argv[1]=='worker': child_main()
        else: parent_main()
