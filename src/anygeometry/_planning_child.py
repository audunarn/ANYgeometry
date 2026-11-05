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
    PlannerDeadline, PlannerCancelled, PlannerCleanupError, PlannerSourceChanged, check_result_status)
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


def load():
    # Aggregate counters only: counting Python processes would enumerate other
    # owners' processes. Explicit unavailable is not a measured zero.
    return dict(python_process_count=None, cpu_percent=psutil.cpu_percent(interval=.1),
                available_memory_bytes=psutil.virtual_memory().available,
                logical_cpus=os.cpu_count())


def prepare(*,production=False):
    if any(value != '1' for name, value in _BOOT_THREAD_ENV):
        raise q.QualificationDecline('native thread environment was not one before import')
    verify_threads(threadpool_info())
    canaries = startup_canaries()
    # Fixed PP1b baseline, also checked against tests/data/pp1b-canaries.json.
    # Installed wheels carry this source; they do not carry checkout fixtures.
    goldens = {
        'pair': 'eddf551cfc6a929265e6147e81f711539fef97a42e67c3c6b810b7dbabc80785',
        'face': 'db97f6a31b9c34dfcafdb2c66f1f17fa07034c24fa9e90b51b43786e8bde409b',
    }
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
    # Resolve the same finite numerical lazy imports in both fresh roles.
    if production:
        from anygeometry._planning_manifest import prepare_interfaces
        prepare_interfaces()
    return canaries, goldens


def worker(folder, deadline, canaries, goldens, limits, generated):
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
    if generated != q.generated_runtime_modules(reference=expected):
        raise q.QualificationDecline('generated module selection changed after preparation')
    ledger = ledger_api.loaded_geometry_ledger(trusted_image=True)
    gate = BatchWorkerGate(expected, stores=ledger.resets, owned_worker=True, ledger=ledger,
        generated_modules=generated,
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


def planner(folder, deadline, canaries, goldens, ready_at, binding, limits, generated):
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
    if generated != q.generated_runtime_modules():
        raise q.QualificationDecline('generated module selection changed after preparation')
    expected = q.capture_environment(generated_modules=generated,
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
        try:
            tree.close()
            (folder/'cleanup.json').replace(folder/'workers-cleanup.json')
        except Exception as exc:
            raise PlannerCleanupError('planner worker cleanup failed') from exc
    rows = [row for report in reports for row in report['rows']]
    reduce_results(rows, len(tasks))
    verify_sources(request['sources'])
    return dict(status='passed', rows=rows, request_sha256=request_sha,
                worker_reports=reports, requested_workers=request['workers'], actual_workers=count,
                task_layout=layout,
                ready_at=ready_at, model_sha256=request['model']['sha256'])


def _publish_memory_failure(error,publication,binding):
    """Activate only an already-written, current-command primary OOM result."""
    if type(error) is not MemoryError or error.args!=():return False
    if type(publication) is not list or len(publication)!=1:return False
    capsule=publication[0]
    if type(capsule) is not tuple or len(capsule)!=7 or type(binding) is not dict:return False
    if type(capsule[6]) is not int or not 0<=capsule[6]<=2:return False
    if (binding.get('session_sha256')!=capsule[2] or binding.get('worker')!=capsule[3] or
            binding.get('worker_setup_sha256')!=capsule[4] or binding.get('request_sha256')!=capsule[5]):
        return False
    publication[0]=None
    os.replace(capsule[0],capsule[1])
    return True


def main():
    role, location, seconds = sys.argv[1:]
    # Import the complete optional session closure before any reference capture.
    if role in ('session-planner','session-worker'):
        from anygeometry._planning_session import session_planner, session_worker
    folder = Path(location)
    deadline = time.monotonic()+min(float(seconds), 110)
    started = time.perf_counter()
    result = {};production=False
    binding = {}
    emergency=[None];emergency_path=False
    limits = dict(payload_limit=4_000_000,walk_limit=100000)
    try:
        if (folder/'limits.json').exists():
            limits=read_packet(folder/'limits.json','limits',limit=4096)
        if type(limits) is not dict or set(limits)!={'payload_limit','walk_limit'} or any(type(v) is not int or v<1 for v in limits.values()):
            raise q.QualificationDecline('invalid child limits')
        with threadpool_limits(limits=1):
            production=(folder/'admission.json').exists()
            if production:
                admission=read_packet(folder/'admission.json','package-admission',limit=4096)
                if (role not in ('session-planner','session-worker') or type(admission) is not dict or
                        set(admission)!={'common','constructor','host','infrastructure_host','resource','value_codec'} or
                        any(type(admission[name]) is not str or len(admission[name])!=64 or any(c not in '0123456789abcdef' for c in admission[name])
                            for name in ('common','constructor','host','infrastructure_host'))):
                    raise q.QualificationDecline('invalid production admission marker')
                from anygeometry._planning_values import verify_value_codec_contract
                verify_value_codec_contract(admission['value_codec'])
                q.resource_file_pin(admission['resource'])
                if role=='session-worker':q.verify_worker_resource_absence()
            prepare_started=time.perf_counter()
            canaries, goldens = prepare(production=production)
            generated = q.generated_runtime_modules()
            if production:
                from anygeometry._planning_session import _record_initialization
                _record_initialization(folder,'child_prepare',prepare_started)
            if role == 'planner':
                result = planner(folder, deadline, canaries, goldens, time.perf_counter(), binding, limits, generated)
            elif role == 'worker':
                result = worker(folder, deadline, canaries, goldens, limits, generated)
            elif role == 'session-planner':
                result = session_planner(folder,deadline,canaries,goldens,binding,limits,generated)
            elif role == 'session-worker':
                result = session_worker(folder,deadline,canaries,goldens,binding,limits,generated,emergency=emergency)
            else:
                raise q.QualificationDecline('unknown child role')
            result['status'] = 'passed'
    except Exception as exc:
        if role=='session-worker' and type(exc) is MemoryError and exc.args==():
            # No traceback, JSON, text stream, or initialization publication on this path.
            emergency_path=True
            if not _publish_memory_failure(exc,emergency,binding):raise
        else:
            traceback.print_exc()
            status = ('deadline' if isinstance(exc, PlannerDeadline) else
                      'cancelled' if isinstance(exc, PlannerCancelled) else
                      'source_changed' if isinstance(exc, PlannerSourceChanged) else
                      'cleanup_error' if isinstance(exc, PlannerCleanupError) else 'declined')
            result = dict(status=status, reason=type(exc).__name__+': '+str(exc), **binding)
    finally:
        if not emergency_path:
            result['process_seconds'] = time.perf_counter()-started
            if production:
                try:
                    from anygeometry._planning_session import _record_initialization
                    _record_initialization(folder,'child_process_total',started)
                except Exception as evidence_error:
                    result['initialization_evidence_error']=type(evidence_error).__name__+': '+str(evidence_error)
            write_packet(folder/'result.json', pack(result, 'result' if role in ('planner','session-planner') else 'worker-result',limit=limits['payload_limit']))



if __name__ == '__main__':
    main()
