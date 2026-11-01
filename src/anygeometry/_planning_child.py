"""Private executable entry point. Values cross the boundary only as JSON.

Running by file gives P and W the same __main__ source identity. All imports and
startup work precede the reference capture; no arbitrary code names are accepted
from transport. Parent launch already sets native thread variables before import.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Capture effective launcher settings BEFORE importing native numerical libraries.
_BOOT_THREAD_ENV = tuple((name, os.environ.get(name)) for name in
    ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
import scipy.linalg
from threadpoolctl import threadpool_info, threadpool_limits
from anygeometry import _planning_qualification as q
from anygeometry import _planning_ledger as ledger_api
from anygeometry._planning_batch import BatchWorkerGate
from anygeometry._planning_canaries import startup_canaries
from anygeometry._planning_executor import (pack, unpack, read_packet, write_packet, thaw_model,
    assignment_size, verify_threads, verify_sources, reduce_results, freeze_model,
    PlannerDeadline, PlannerCancelled, check_result_status)
from anygeometry._planning_process import OwnedTree
from anygeometry._planning_tasks import model_task
from anygeometry._planning_stages import WorkerModel,stage_task,dispatch
# Real sector60 import census: close these exact lazy kernel edges before admission.
from anygeometry import branch_supports,plane_ruled_support,quadric_supports,quadric_curve_events
from anygeometry import branch_events,quadric_events,branch_wall_events
# Planar overlap is an existing real-pair route. Import its exact dependency
# closure before capture; later additions/rebindings remain forbidden.
from shapely.geometry import Polygon
from shapely.ops import unary_union
_GENERATED_MODULES = ('cython_runtime', '_cython_3_2_4', '_cython_3_1_4')


def load():
    # Requested anonymous machine-load totals; never unrelated command/PID data.
    count = sum(1 for p in psutil.process_iter(['name'], ad_value='')
                if (p.info['name'] or '').lower() in ('python.exe', 'pythonw.exe', 'python', 'python3'))
    return dict(python_process_count=count, cpu_percent=psutil.cpu_percent(interval=.1),
                logical_cpus=os.cpu_count())


def prepare():
    if any(value != '1' for name, value in _BOOT_THREAD_ENV):
        raise q.QualificationDecline('native thread environment was not one before import')
    verify_threads(threadpool_info())
    canaries = startup_canaries()
    goldens = json.loads((Path(__file__).resolve().parents[2]/'tests/data/pp1b-canaries.json').read_text())
    if {name: task() for name, task in canaries.items()} != goldens:
        raise q.QualificationDecline('planner prewarm canary mismatch')
    # Warm the transport/task path too, before ledger closure freezes imports.
    from anygeometry._planning_canaries import _fixture
    from anygeometry.serialization import to_dict, from_dict
    model, _ = _fixture()
    from anygeometry._planning_executor import freeze_model
    packet = freeze_model(from_dict(to_dict(model)))
    for name in canaries:
        if model_task(packet, name) != goldens[name]:
            raise q.QualificationDecline('transport task differs from canary golden')
    # Warm the exact constructor-view task path before freezing the ledger.
    context=WorkerModel(packet)
    faces=list(context.model.faces);tolerance=context.model.tolerance.effective_length(5.)
    for face in faces[1:]:
        stage_task(context,dict(kind='pair',first=faces[0],second=face,
            tolerance=tolerance,face_tolerances=[tolerance,tolerance]))
    context.verify()
    return canaries, goldens


def worker(folder, deadline, canaries, goldens, limits):
    request = read_packet(folder/'request.json', 'assignment',limit=limits['payload_limit'])
    context = WorkerModel(request['model'],**limits) if request.get('stage') is True else None
    decoded = context.model if context is not None else thaw_model(request['model'],**limits)
    reconstructed = unpack(freeze_model(decoded,**limits), 'model',limit=limits['payload_limit'])
    recorded = unpack(request['model'], 'model',limit=limits['payload_limit'])
    if recorded['extension_record'] == dict(present=False, sha256=None):
        if decoded._serialization_extensions != {}:
            raise q.QualificationDecline('worker extension reconstruction mismatch')
        reconstructed['extension_record'] = recorded['extension_record']
    model_sha = pack(reconstructed, 'model',limit=limits['payload_limit'])['sha256']
    if model_sha != request['model']['sha256']:
        raise q.QualificationDecline('worker reconstructed model hash mismatch')
    expected = request['expected']
    ledger = ledger_api.loaded_geometry_ledger(trusted_image=True)
    gate = BatchWorkerGate(expected, stores=ledger.resets, owned_worker=True, ledger=ledger,
        generated_modules=_GENERATED_MODULES,
        batch_size=assignment_size(request['batch_size'], len(request['tasks'])), coalesce_boundaries=True)
    before = load()
    started = time.perf_counter()
    gate.start(canaries, goldens)
    startup = time.perf_counter()-started
    write_packet(folder/'admitted.json', pack(dict(admitted=True), 'admitted'))
    tasks = []
    for ordinal, name in request['tasks']:
        payload = json.dumps(dict(args=[name] if context is not None else [request['model'], name], kwargs={}),
                             sort_keys=True, separators=(',', ':')).encode()
        tasks.append((stage_task if context is not None else model_task, payload, hashlib.sha256(payload).hexdigest()))
    started = time.perf_counter()
    results = gate.run_tasks(tasks,context=context)
    elapsed = time.perf_counter()-started
    if time.monotonic() >= deadline:
        raise PlannerDeadline('worker deadline consumed')
    verify_threads(threadpool_info())
    return dict(rows=[[item[0], value] for item, value in zip(request['tasks'], results)],
        model_sha256=model_sha, model_decode_count=1 if context is not None else 1+len(tasks), load_before=before,
        task_count=len(tasks), batch_size=gate.batch_size, assignment_seconds=elapsed,
        startup_seconds=startup, task_seconds=gate.task_seconds,
        boundary_seconds=gate.boundary_seconds, reset_count=len(ledger.costs),
        full_environment_checks=len(gate.verification_seconds), blas_threads=threadpool_info())


def planner(folder, deadline, canaries, goldens, ready_at, binding, limits):
    write_packet(folder/'ready.json', pack(dict(ready_at=ready_at, load_before=load()), 'ready'))
    path = folder/'request.json'
    while not path.exists():
        if time.monotonic() >= deadline:
            raise PlannerDeadline('unused planner deadline consumed')
        time.sleep(.01)
    request, request_sha = read_packet(path, 'request', with_digest=True,limit=limits['payload_limit'])
    binding['request_sha256'] = request_sha
    verify_sources(request['sources'])
    thaw_model(request['model'],**limits)
    tasks = request['tasks']
    stage = request.get('stage',False)
    if type(stage) is not bool or type(tasks) is not list:
        raise q.QualificationDecline('invalid task schema')
    if not stage and any(type(t) is not str or t not in ('pair','face') for t in tasks):
        raise q.QualificationDecline('invalid diagnostic task')
    assignments, layout = dispatch(tasks if stage else [dict(kind=t,cost=1.) for t in tasks],request['workers'])
    count = len(assignments)
    ledger = ledger_api.loaded_geometry_ledger(trusted_image=True)
    ledger.prepare(owned_worker=True)
    expected = q.capture_environment(generated_modules=_GENERATED_MODULES,
                                    declined_stores=ledger.resets, trusted_image=True)
    expected['live_code_pins'] = ledger.live_code_pins()
    write_packet(folder/'reference.json', pack(expected, 'reference'))
    tree = OwnedTree(folder)
    workers = []
    reports = []
    try:
        for index in range(count):
            location = folder/('worker-'+str(index))
            location.mkdir()
            assigned = [[ordinal,tasks[ordinal]] for ordinal in assignments[index]]
            assignment = dict(expected=expected, model=request['model'], tasks=assigned,
                              batch_size=request['batch_size'],stage=stage)
            write_packet(location/'limits.json',pack(limits,'limits'))
            write_packet(location/'request.json', pack(assignment, 'assignment',limit=limits['payload_limit']))
            child = tree.spawn('worker', location, max(.1, deadline-time.monotonic()-5))
            workers.append((child, location))
        for child, location in workers:
            tree.wait(child, deadline-3)
            report = read_packet(location/'result.json', 'worker-result',limit=limits['payload_limit'])
            check_result_status(report)
            verify_threads(report.get('blas_threads', []))
            if report['model_sha256'] != request['model']['sha256']:
                raise q.QualificationDecline('worker model binding mismatch')
            reports.append(report)
    finally:
        tree.close()
        (folder/'cleanup.json').replace(folder/'workers-cleanup.json')
    rows = [row for report in reports for row in report['rows']]
    reduce_results(rows, len(tasks))
    verify_sources(request['sources'])
    return dict(status='passed', rows=rows, request_sha256=request_sha,
                worker_reports=reports, requested_workers=request['workers'], actual_workers=count,
                task_layout=layout,
                ready_at=ready_at, model_sha256=request['model']['sha256'])


def main():
    role, location, seconds = sys.argv[1:]
    folder = Path(location)
    deadline = time.monotonic()+min(float(seconds), 110)
    started = time.perf_counter()
    result = {}
    binding = {}
    limits = dict(payload_limit=4_000_000,walk_limit=100000)
    try:
        if (folder/'limits.json').exists():
            limits=read_packet(folder/'limits.json','limits',limit=4096)
        if type(limits) is not dict or set(limits)!={'payload_limit','walk_limit'} or any(type(v) is not int or v<1 for v in limits.values()):
            raise q.QualificationDecline('invalid child limits')
        with threadpool_limits(limits=1):
            canaries, goldens = prepare()
            if role == 'planner':
                result = planner(folder, deadline, canaries, goldens, time.perf_counter(), binding, limits)
            elif role == 'worker':
                result = worker(folder, deadline, canaries, goldens, limits)
            else:
                raise q.QualificationDecline('unknown child role')
            result['status'] = 'passed'
    except Exception as exc:
        traceback.print_exc()
        status = ('deadline' if isinstance(exc, PlannerDeadline) else
                  'cancelled' if isinstance(exc, PlannerCancelled) else 'declined')
        result = dict(status=status, reason=type(exc).__name__+': '+str(exc), **binding)
    finally:
        result['process_seconds'] = time.perf_counter()-started
        write_packet(folder/'result.json', pack(result, 'result' if role == 'planner' else 'worker-result',limit=limits['payload_limit']))


if __name__ == '__main__':
    main()
