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
import scipy.linalg
from threadpoolctl import threadpool_info, threadpool_limits
from anygeometry import _planning_qualification as q
from anygeometry import _planning_ledger as ledger_api
from anygeometry._planning_batch import BatchWorkerGate
from anygeometry._planning_canaries import startup_canaries
from anygeometry._planning_executor import (pack, unpack, read_packet, write_packet, thaw_model,
    assignment_size, verify_threads, verify_sources, reduce_results)
from anygeometry._planning_process import OwnedTree
from anygeometry._planning_tasks import model_task


def load():
    # Requested anonymous machine-load totals; never unrelated command/PID data.
    count = sum(1 for p in psutil.process_iter(['name'], ad_value='')
                if (p.info['name'] or '').lower() in ('python.exe', 'pythonw.exe', 'python', 'python3'))
    return dict(python_process_count=count, cpu_percent=psutil.cpu_percent(interval=.1),
                logical_cpus=os.cpu_count())


def prepare():
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
    return canaries, goldens


def worker(folder, deadline, canaries, goldens):
    request = read_packet(folder/'request.json', 'assignment')
    expected = request['expected']
    ledger = ledger_api.loaded_geometry_ledger(trusted_image=True)
    gate = BatchWorkerGate(expected, stores=ledger.resets, owned_worker=True, ledger=ledger,
        generated_modules=('cython_runtime', '_cython_3_2_4'),
        batch_size=assignment_size(request['batch_size'], len(request['tasks'])), coalesce_boundaries=True)
    before = load()
    started = time.perf_counter()
    gate.start(canaries, goldens)
    startup = time.perf_counter()-started
    write_packet(folder/'admitted.json', pack(dict(admitted=True), 'admitted'))
    tasks = []
    for ordinal, name in request['tasks']:
        payload = json.dumps(dict(args=[request['model'], name], kwargs={}),
                             sort_keys=True, separators=(',', ':')).encode()
        tasks.append((model_task, payload, hashlib.sha256(payload).hexdigest()))
    started = time.perf_counter()
    results = gate.run_tasks(tasks)
    elapsed = time.perf_counter()-started
    if time.monotonic() >= deadline:
        raise q.QualificationDecline('worker deadline')
    return dict(rows=[[item[0], value] for item, value in zip(request['tasks'], results)],
        model_sha256=request['model']['sha256'], load_before=before,
        task_count=len(tasks), batch_size=gate.batch_size, assignment_seconds=elapsed,
        startup_seconds=startup, task_seconds=gate.task_seconds,
        boundary_seconds=gate.boundary_seconds, reset_count=len(ledger.costs),
        full_environment_checks=len(gate.verification_seconds), blas_threads=threadpool_info())


def planner(folder, deadline, canaries, goldens, ready_at):
    write_packet(folder/'ready.json', pack(dict(ready_at=ready_at, load_before=load()), 'ready'))
    path = folder/'request.json'
    while not path.exists():
        if time.monotonic() >= deadline:
            raise q.QualificationDecline('unused planner deadline')
        time.sleep(.01)
    request = read_packet(path, 'request')
    request_sha = json.loads(path.read_text())['sha256']
    verify_sources(request['sources'])
    thaw_model(request['model'])
    tasks = request['tasks']
    if (type(tasks) is not list or any(type(t) is not str or t not in ('pair', 'face') for t in tasks)
            or type(request['workers']) is not int or request['workers'] not in (1, 2, 8, 16)):
        raise q.QualificationDecline('invalid task/worker schema')
    count = min(request['workers'], len(tasks))
    ledger = ledger_api.loaded_geometry_ledger(trusted_image=True)
    ledger.prepare(owned_worker=True)
    expected = q.capture_environment(generated_modules=('cython_runtime', '_cython_3_2_4'),
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
            assigned = [[ordinal, name] for ordinal, name in enumerate(tasks) if ordinal % count == index]
            assignment = dict(expected=expected, model=request['model'], tasks=assigned,
                              batch_size=request['batch_size'])
            write_packet(location/'request.json', pack(assignment, 'assignment'))
            child = tree.spawn('worker', location, min(85, max(.1, deadline-time.monotonic()-5)))
            workers.append((child, location))
        for child, location in workers:
            tree.wait(child, deadline-3)
            report = read_packet(location/'result.json', 'worker-result')
            if report.get('status') != 'passed':
                raise q.QualificationDecline('worker refused: '+report.get('reason', 'unknown'))
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
                task_layout=[[i % count, i // count, name] for i,name in enumerate(tasks)],
                ready_at=ready_at, model_sha256=request['model']['sha256'])


def main():
    role, location, seconds = sys.argv[1:]
    folder = Path(location)
    deadline = time.monotonic()+min(float(seconds), 100)
    started = time.perf_counter()
    result = {}
    try:
        with threadpool_limits(limits=1):
            canaries, goldens = prepare()
            if role == 'planner':
                result = planner(folder, deadline, canaries, goldens, time.perf_counter())
            elif role == 'worker':
                result = worker(folder, deadline, canaries, goldens)
            else:
                raise q.QualificationDecline('unknown child role')
            result['status'] = 'passed'
    except Exception as exc:
        traceback.print_exc()
        result = dict(status='declined', reason=type(exc).__name__+': '+str(exc))
        if role == 'planner' and (folder/'request.json').exists():
            result['request_sha256'] = json.loads((folder/'request.json').read_text()).get('sha256')
    finally:
        result['process_seconds'] = time.perf_counter()-started
        write_packet(folder/'result.json', pack(result, 'result' if role == 'planner' else 'worker-result'))


if __name__ == '__main__':
    main()
