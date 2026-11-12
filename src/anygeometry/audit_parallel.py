"""Bounded process acceleration of the unchanged strict spatial classifiers.

Topology and broad phase execute once in the parent. Children receive public
checksummed trusted runtime snapshots and disjoint candidate batches. No geometric
predicate, qualification policy or classification is replaced here.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import threading
import time

from .audit import AuditCollector, AuditContext, AuditScope, run_audit
from .predicates import DEFAULT_INTERSECTION_QUALIFICATION_POLICY
from .model_transport import capture_model_transfer, _issue_classification_receipt, _restore_classification_transfer
from .strict_audit import _StrictAuditState

_spawn_environment_lock = threading.Lock()


class _BatchCollector(AuditCollector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emitted = []

    def add(self, issue):
        self.emitted.append(issue)
        return super().add(issue)


def _classify_batch(connection, transfer, receipt, context, policy, qualification, pairs):
    try:
        model = _restore_classification_transfer(transfer, receipt)
        if (str(model.model_id) != context.model_id or model.revision != context.revision
                or (len(model.vertices), len(model.edges), len(model.faces), len(model.members)) != transfer.entity_counts):
            raise RuntimeError('Parallel audit snapshot identity/checksum mismatch.')
        collector = _BatchCollector(context.model_id, context.revision, scope=context.scope, policy=policy)
        collector.record_candidates(len(pairs))
        state = _StrictAuditState(model, context, collector, qualification=qualification)
        state.classify_spatial_pairs(pairs)
        # finish() may synthesize aggregate incomplete/unverified issues. Only
        # merge original pair issues; the parent synthesizes aggregates once.
        try:
            from threadpoolctl import threadpool_info
            pools = {'status': 'available', 'libraries': [
                {'library': os.path.basename(item.get('filepath') or ''),
                 'version': item.get('version'), 'user_api': item.get('user_api'),
                 'num_threads': item.get('num_threads')} for item in threadpool_info()]}
        except Exception:
            pools = {'status': 'unknown', 'libraries': []}
        connection.send(('ok', collector.finish(), tuple(collector.emitted), pools))
    except BaseException as error:
        connection.send(('error', type(error).__name__, str(error)))
    finally:
        connection.close()


def _retire(processes):
    for process in processes:
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + 5
    for process in processes:
        process.join(timeout=max(0., deadline - time.monotonic()))
    survivors = [process for process in processes if process.is_alive()]
    for process in survivors:
        process.kill()
    deadline = time.monotonic() + 5
    for process in survivors:
        process.join(timeout=max(0., deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            raise RuntimeError('Parallel audit child retirement could not be verified.')


def _merge(collector, reports):
    """Merge only independent narrow-phase contributions, never broadphase."""
    for report, emitted in reports:
        for issue in emitted:
            collector.add(issue)
        metrics = report.metrics
        collector.record_narrow_phase(metrics.narrow_phase_tests)
        collector.record_classification(classified=True, count=metrics.classified_count)
        collector.record_classification(classified=False, count=metrics.unclassified_count)
        collector.record_intersection_work(boxes_examined=metrics.boxes_examined,
            subdivisions=metrics.subdivisions, trace_segments=metrics.trace_segments)
        collector.record_affected_structural_keys(metrics.affected_structural_keys)
        if not report.completed:
            collector.mark_incomplete()
        if not report.verified:
            collector.mark_unverified()


class _ParallelState(_StrictAuditState):
    def __init__(self, *args, workers, min_candidates, cancelled, execution, classification_started=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.workers = workers
        self.min_candidates = min_candidates
        self.cancelled = cancelled
        self.execution = execution
        self.classification_started = classification_started

    def audit_topology(self):
        binding = (str(self.model.model_id), self.model.revision)
        super().audit_topology()
        if (str(self.model.model_id), self.model.revision) != binding:
            raise RuntimeError('Source changed during root topology audit.')
        self._topology_audit_binding = binding

    def audit_spatial_pairs(self):
        timings = self.execution.setdefault('_timings', {})
        started = time.perf_counter()
        result = self.spatial_pair_candidates()
        timings['broadphase'] = time.perf_counter() - started
        pairs = result.pairs
        count = len(pairs)
        available_cpus = getattr(os, 'process_cpu_count', os.cpu_count)() or 1
        workers = min(self.workers, available_cpus, max(1, count))
        self.execution.update(route='serial', effective_workers=1, candidate_count=count,
                              requested_workers=self.workers, status='running')
        if workers < 2 or count < self.min_candidates:
            self.classify_spatial_pairs(pairs)
            self.execution['status'] = 'completed'
            return
        self.execution.update(route='process_pool', effective_workers=0)
        if mp.current_process().daemon:
            self.execution['status'] = 'refused_daemonic_parent'
            raise RuntimeError('Parallel strict audit requires a non-daemonic contained parent process.')
        started = time.perf_counter()
        transfer = capture_model_transfer(self.model)
        receipt = _issue_classification_receipt(transfer, self)
        timings['snapshot'] = time.perf_counter() - started
        self.execution['source_checksum'] = transfer.source_checksum
        self.execution['source_checksum_kind'] = 'runtime_transfer_sha256'
        self.execution['model_transport'] = 'trusted_runtime_v1'
        self.execution['validation_reused'] = 'root_topology_and_structural_audit'
        batches = [[] for _ in range(workers)]
        # Blocks preserve useful local caches while distributing pair kinds
        # and expensive face classifications evenly across all children.
        for start in range(0, count, 256):
            batches[(start // 256) % workers].extend(pairs[start:start + 256])
        processes, readers = [], []
        context = mp.get_context('spawn')
        reports, threadpools = {}, {}
        try:
            started = time.perf_counter()
            # Children inherit single-thread BLAS before Python/Numpy import.
            # Restore the parent environment even when process launch fails.
            with _spawn_environment_lock:
                names = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')
                previous = {name: os.environ.get(name) for name in names}
                try:
                    os.environ.update({name: '1' for name in names})
                    for index, batch in enumerate(batches):
                        if self.cancelled is not None and self.cancelled():
                            self.execution['status'] = 'cancelled'
                            raise RuntimeError('Parallel strict audit cancelled.')
                        reader, writer = context.Pipe(duplex=False)
                        process = context.Process(target=_classify_batch,
                            args=(writer, transfer, receipt, self.context, self.collector.policy,
                                  self.qualification, tuple(batch)), name=f'anygeometry-audit-{index}', daemon=False)
                        try:
                            process.start()
                        except BaseException:
                            reader.close()
                            writer.close()
                            raise
                        writer.close()
                        processes.append(process)
                        readers.append(reader)
                finally:
                    for name, value in previous.items():
                        if value is None:
                            os.environ.pop(name, None)
                        else:
                            os.environ[name] = value
            self.execution['effective_workers'] = len(processes)
            self.execution['blas_threads_requested'] = 1
            self.execution['blas_threads'] = 'unknown'
            timings['spawn'] = time.perf_counter() - started
            started = time.perf_counter()
            if self.classification_started is not None:
                self.classification_started()
            while len(reports) < len(processes):
                if self.cancelled is not None and self.cancelled():
                    self.execution['status'] = 'cancelled'
                    raise RuntimeError('Parallel strict audit cancelled.')
                received = False
                for index, reader in enumerate(readers):
                    if index in reports:
                        continue
                    if reader.poll():
                        payload = reader.recv()
                        if payload[0] != 'ok':
                            raise RuntimeError(f'Parallel classifier failed: {payload[1]}: {payload[2]}')
                        reports[index] = payload[1:3]
                        threadpools[index] = payload[3]
                        received = True
                    elif processes[index].exitcode is not None:
                        raise RuntimeError(f'Parallel classifier exited without a result: {processes[index].exitcode}')
                if not received:
                    time.sleep(.01)
            timings['classify_and_transport'] = time.perf_counter() - started
            started = time.perf_counter()
            _merge(self.collector, [reports[index] for index in range(workers)])
            self.execution['child_threadpools'] = [threadpools[index] for index in range(workers)]
            observed = [[library['num_threads'] for library in item['libraries'] if library['user_api'] == 'blas']
                        for item in threadpools.values()]
            if (all(item['status'] == 'available' for item in threadpools.values())
                    and all(values and all(type(count) is int and count > 0 for count in values) for values in observed)):
                self.execution['blas_threads'] = max(count for values in observed for count in values)
            timings['merge'] = time.perf_counter() - started
            self.execution['status'] = 'completed'
        except BaseException:
            if self.execution['status'] != 'cancelled':
                self.execution['status'] = 'failed'
            raise
        finally:
            started = time.perf_counter()
            for reader in readers:
                reader.close()
            _retire(processes)
            timings['retire'] = time.perf_counter() - started
            self.execution['children_retired'] = True


def strict_audit_parallel(model, *, workers=6, min_candidates=20000, policy=None,
                          qualification=None, cancelled=None, execution=None, classification_started=None):
    """Return the exact strict report, with optional bounded spatial processes.

    Existing strict_audit remains serial. Small workloads stay serial here too.
    Failures/refusals become the usual fail-closed owner audit blockers; effective
    route information is returned through the optional execution dictionary.
    """
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError('workers must be an integer in [1, 8]')
    if type(min_candidates) is not int or min_candidates < 0:
        raise ValueError('min_candidates must be a nonnegative integer')
    execution = {} if execution is None else execution
    execution.update(route='not_started', effective_workers=0, requested_workers=workers, status='running')
    qualification = DEFAULT_INTERSECTION_QUALIFICATION_POLICY if qualification is None else qualification
    def check(source, context, collector):
        try:
            _ParallelState(source, context, collector, qualification=qualification,
                workers=workers, min_candidates=min_candidates, cancelled=cancelled, execution=execution,
                classification_started=classification_started).run()
        except BaseException as error:
            execution['failure'] = f'{type(error).__name__}: {error}'
            if execution['status'] not in ('cancelled', 'refused_daemonic_parent'):
                execution['status'] = 'failed'
            raise
    check.audit_name = 'strict_audit_parallel'
    return run_audit(model, (check,), model_id_getter=lambda source: source.model_id,
                     revision_getter=lambda source: source.revision, scope=AuditScope.FULL_MODEL, policy=policy)
