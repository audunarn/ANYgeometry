"""PP1g risk-accepted qualification, not a purity proof: real child + parent samples."""
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
from anygeometry._planning_canaries import startup_canaries
spec=importlib.util.spec_from_file_location('pp1g_owned_job',Path.home()/'bin/claude-review.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
OUT=Path(os.environ['TASK_EVIDENCE_DIRECTORY'])
GENERATED=('cython_runtime','_cython_3_2_4')

def save(name,value): (OUT/name).write_text(json.dumps(value,indent=2))
def load(): return dict(cpu_percent=psutil.cpu_percent(interval=.1),logical_cpus=os.cpu_count())

def child_main():
    expected=json.loads((OUT/'reference.json').read_text())
    for name in expected['modules']:
        if name not in sys.modules: importlib.import_module(name)
    ledger=l.loaded_geometry_ledger(trusted_image=True)
    save('child-ledger-before.json',ledger.record())
    gate=q.WorkerGate(expected,stores=ledger.resets,owned_worker=True,
        ledger=ledger,generated_modules=GENERATED)
    tasks=json.loads((OUT/'tasks.json').read_text())
    canaries=startup_canaries()
    goldens=json.loads((ROOT/'tests/data/pp1b-canaries.json').read_text())
    report=dict(pid=os.getpid(),load_start=load(),status='starting')
    try:
        started=time.perf_counter();gate.start(canaries,goldens)
        report['canary_startup_seconds']=time.perf_counter()-started
        report['startup_resets']=list(ledger.costs)
        report['startup_passed']=gate.ready
        durations=[];digests=[]
        for name in tasks:
            started=time.perf_counter();digests.append(gate.run(canaries[name]))
            durations.append(time.perf_counter()-started)
        report.update(status='admitted',task_count=len(tasks),digests=digests,
            per_task_reset_seconds=ledger.costs[2:],task_boundary_seconds=durations,
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
    canaries=startup_canaries()
    goldens=json.loads((ROOT/'tests/data/pp1b-canaries.json').read_text())
    assert {name:task() for name,task in canaries.items()}==goldens
    ledger=l.loaded_geometry_ledger(trusted_image=True)
    save('parent-ledger.json',ledger.record())
    ledger.prepare(owned_worker=True)
    expected=q.capture_environment(generated_modules=GENERATED,declined_stores=ledger.resets,trusted_image=True)
    expected['live_code_pins']=ledger.live_code_pins()
    save('reference.json',expected)
    tasks=('pair',)*20+('face',)*20
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
    result.update(serial_sample_tasks=serial_calls,load_end=load(),
        blas_threads=[p['num_threads'] for p in threadpool_info()],
        qualification='RISK-ACCEPTED QUALIFICATION, not a purity proof',
        full_model=False,task_scope='40 small real kernel tasks, 20 pair and 20 face canaries',
        plan_apply_ci='unit hook tested; full PP6 control not exercised here')
    save('parent.json',result)
    assert result['mode']=='parallel' and result['samples']==2
    assert serial_calls==['pair','face']
    assert result['results']==tuple(goldens[name] for name in tasks)
    print(json.dumps({k:v for k,v in result.items() if k!='results'}))

if __name__=='__main__':
    with threadpool_limits(limits=1):
        if len(sys.argv)>1 and sys.argv[1]=='worker': child_main()
        else: parent_main()
