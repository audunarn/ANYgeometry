"""PP2 finite small-model matrix, never a production/full-model planning run."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from threadpoolctl import threadpool_limits, threadpool_info
from anygeometry import _planning_executor as executor
from anygeometry._planning_canaries import _fixture
from anygeometry._planning_tasks import model_task
from anygeometry.serialization import from_dict, to_dict
from anygeometry.definition_binding import definition_checksum


def load():
    count=sum(1 for p in psutil.process_iter(['name'],ad_value='')
        if (p.info['name'] or '').lower() in ('python.exe','pythonw.exe','python','python3'))
    return dict(python_process_count=count,cpu_percent=psutil.cpu_percent(interval=.1),logical_cpus=os.cpu_count())


def main():
    choice=sys.argv[1]
    if choice not in ('whole','256'):
        raise ValueError('outside PP2 declared two-configuration matrix')
    batch=None if choice=='whole' else 256
    out=Path(os.environ['TASK_EVIDENCE_DIRECTORY'])
    source,_=_fixture()
    document=to_dict(source)
    project=out/'small-model.json'
    project.write_text(json.dumps(document),encoding='utf-8')
    tasks=('pair','face')*300
    goldens=json.loads((ROOT/'tests/data/pp1b-canaries.json').read_text())
    expected=tuple(goldens[name] for name in tasks)
    before=load()
    # Start P at project-open, then do the actual small project decode in H.
    planner=executor.OwnedPlanner(out/'owned',workers=2,batch_size=batch)
    load_started=time.perf_counter()
    model=from_dict(json.loads(project.read_text(encoding='utf-8')))
    load_ended=time.perf_counter()
    packet=executor.freeze_model(model)
    result=executor.execute(model,tasks,lambda name:model_task(packet,name),directory=out/'unused',
        planner=planner,opt_in=True,expected_ci=(definition_checksum(expected),packet['sha256']),
        observed_ci=lambda candidate:(definition_checksum(candidate['results']),candidate['model_sha256']))
    assert result['mode']=='parallel',result
    assert result['results']==expected and result['ci_verified']
    # Original serial diagnostic work, after every owned process is retired.
    serial_load=load();serial_times=[];serial_results=[]
    for name in tasks:
        started=time.perf_counter();serial_results.append(model_task(packet,name))
        serial_times.append(time.perf_counter()-started)
    assert tuple(serial_results)==result['results']
    table=[]
    for index,worker in enumerate(result['worker_reports']):
        serial_work=sum(serial_times[index::2])
        table.append(dict(worker=index,tasks=worker['task_count'],batch_size=worker['batch_size'],
            assignment_seconds=worker['assignment_seconds'],serial_comparison_seconds=serial_work,
            raw_overhead_seconds=worker['assignment_seconds']-serial_work,
            boundary_count=len(worker['boundary_seconds']),boundary_seconds=sum(worker['boundary_seconds']),
            startup_seconds=worker['startup_seconds'],load_before=worker['load_before']))
    overlap=max(0,min(load_ended,result['ready_at'])-max(load_started,planner.started))
    report=dict(configuration=choice,tasks=600,workers=2,model_faces=3,full_model=False,
        source_sha256=hashlib.sha256(project.read_bytes()).hexdigest(),host_load_before=before,
        serial_load_before=serial_load,worker_table=table,
        project_load_seconds=load_ended-load_started,observed_boot_load_overlap_seconds=overlap,
        planner_ready_from_host_start_seconds=result['ready_at']-planner.started,
        remaining_startup_after_project_load_seconds=max(0,result['ready_at']-load_ended),
        remaining_startup_at_submit_seconds=max(0,result['ready_at']-planner.started-result['host_timing']['start_to_submit_seconds']),
        startup_overlap_basis='Windows shared performance counter; actual tiny JSON project load, no sleep/artificial delay',
        host_timing=result['host_timing'],samples=result['samples'],sampling=result['sampling'],
        ci_verified=result['ci_verified'],ci_scope='diagnostic ordered task digests and unchanged transported model; not PP6 plan/apply',
        serial_task_seconds=serial_times,blas_threads=[p['num_threads'] for p in threadpool_info()],
        clock_resolution=time.get_clock_info('perf_counter').resolution,
        caveat='one observation/configuration; raw worker-minus-serial differences, not statistical estimates or N2c forecasts')
    (out/'measurement.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'execution.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('sampling','serial_task_seconds')}))


if __name__=='__main__':
    with threadpool_limits(limits=1): main()
