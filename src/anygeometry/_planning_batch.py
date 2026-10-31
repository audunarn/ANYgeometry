"""PP1h batch cadence: risk-accepted qualification, not a purity proof.

No result leaves this finite-assignment API before the final verified boundary.
Full environment discovery is startup-only; image/code/store checks remain at
chunk boundaries. Transient mutations restored between checks are not disproved.
"""
import os
import sys
import time
import types
from threadpoolctl import threadpool_info
from . import _planning_qualification as q

_EMPTY = b'{"args":[],"kwargs":{}}'


def verify_boundary(expected, registered):
    """Risk-accepted boundary hash/census support, not full environment discovery."""
    if set(sys.modules) != set(registered) or any(sys.modules.get(n) is not m for n, m in registered.items()):
        raise q.QualificationDecline('batch module closure/binding changed')
    runtime = q.runtime_files()
    observed = {}
    generated = []
    for name, module in registered.items():
        if expected['modules'][name]['kind'] == 'explicit-generated-native':
            namespace = vars(module)
            if namespace.get('__file__') or namespace.get('__spec__') is not None:
                raise q.QualificationDecline('batch generated module origin changed')
            observed[name] = dict(kind='explicit-generated-native',
                members=sorted([key, type.__getattribute__(type(value), '__name__')]
                               for key, value in namespace.items()))
            generated.append(name)
        else:
            observed[name] = q.pin_module(module, runtime=runtime)
    providers = {name: pin for name, pin in observed.items() if pin['kind'] == 'native'}
    for name in generated:
        observed[name]['providers'] = providers
    if observed != expected['modules']:
        raise q.QualificationDecline('batch module/source/binary hash changed')
    if any(q._hash(path) != digest for path, digest in expected['files'].items()):
        raise q.QualificationDecline('batch pinned file hash changed')
    if {name: os.environ.get(name) for name in expected['settings']} != expected['settings']:
        raise q.QualificationDecline('batch environment settings changed')
    pools = [{**pool, 'sha256': q._hash(pool['filepath'])} for pool in threadpool_info()]
    if sorted(pools, key=lambda pool: pool['filepath']) != expected['blas']:
        raise q.QualificationDecline('batch BLAS/thread/image changed')


class BatchWorkerGate(q.WorkerGate):
    """Risk-accepted finite assignment, not a purity proof; K=1 heavy cadence compatible."""
    def __init__(self, *args, batch_size=64, coalesce_boundaries=False, **kwargs):
        if type(batch_size) is not int or batch_size < 1:
            raise q.QualificationDecline('batch size must be a positive integer')
        if type(coalesce_boundaries) is not bool:
            raise q.QualificationDecline('coalescing selection must be bool')
        super().__init__(*args, **kwargs)
        if self.ledger is None:
            raise q.QualificationDecline('batch gate requires the live ledger')
        self._batch_size = batch_size
        self.coalesce_boundaries = coalesce_boundaries
        self._registered = {}
        self._pending = []
        self.closed = False
        self.boundary_seconds = []
        self.task_seconds = []
        self.discarded_batch_tasks = 0
        self.discarded_run_tasks = 0

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def pending_count(self):
        return len(self._pending)

    def _boundary(self):
        started = time.perf_counter()
        try:
            verify_boundary(self.expected, self._registered)
            if self.expected.get('live_code_pins', {}) != self.ledger.live_code_pins():
                raise q.QualificationDecline('batch live code pins changed')
            self.ledger.validate()
        finally:
            self.boundary_seconds.append(time.perf_counter() - started)

    def _execute(self, task, payload, digest):
        if type(task) is not types.FunctionType:
            raise q.QualificationDecline('batch task callable type')
        module = self._registered.get(task.__module__)
        if (module is None or sys.modules.get(task.__module__) is not module or
                '.' in task.__qualname__ or vars(module).get(task.__qualname__) is not task or
                task.__globals__ is not vars(module) or task.__closure__ is not None):
            raise q.QualificationDecline('batch task binding/globals changed')
        self.reset_counts = self.ledger.reset_for_batch(owned_worker=self.owned_worker)
        args, kwargs = q.decode_task_payload(payload, digest)
        q.check_types((args, kwargs), self.expected)
        return task(*args, **kwargs)

    def start(self, canaries, expected_digests):
        if self.ready or self.poisoned or self.closed:
            raise q.QualificationDecline('batch startup may run only once')
        try:
            if set(canaries) != {'pair', 'face'} or set(expected_digests) != set(canaries):
                raise q.QualificationDecline('both batch startup canaries required')
            self._verify()
            self._registered = {name: sys.modules.get(name) for name in self.expected['modules']}
            self._boundary()
            digest = q.hashlib.sha256(_EMPTY).hexdigest()
            for name in sorted(canaries):
                actual = self._execute(canaries[name], _EMPTY, digest)
                if type(actual) is not str or not actual or actual != expected_digests[name]:
                    raise q.QualificationDecline('batch canary mismatch: ' + name)
            self._boundary()
            self.ready = True
        except BaseException:
            self.ready = False
            self.poisoned = True
            raise

    def run_tasks(self, tasks):
        """Risk-accepted assignment; final chunk post-check is worker-end check."""
        if not self.ready or self.poisoned or self.closed:
            raise q.QualificationDecline('batch worker is not admitted or already ended')
        completed = []
        try:
            tasks = tuple(tasks)
            if not tasks:
                self._boundary()
            for offset in range(0, len(tasks), self.batch_size):
                # No external code runs between the prior post-check and this
                # pre-check. Preserve legacy cadence unless explicitly selected.
                if not self.coalesce_boundaries or offset == 0:
                    self._boundary()
                for task, payload, digest in tasks[offset:offset + self.batch_size]:
                    started = time.perf_counter()
                    self._pending.append(self._execute(task, payload, digest))
                    self.task_seconds.append(time.perf_counter() - started)
                self._boundary()
                completed.extend(self._pending)
                self._pending.clear()
            self.closed = True
            self.ready = False
            return tuple(completed)
        except BaseException:
            self.discarded_batch_tasks = len(self._pending)
            self.discarded_run_tasks = len(completed) + len(self._pending)
            self._pending.clear()
            completed.clear()
            self.ready = False
            self.poisoned = True
            self.closed = True
            raise

    def run(self, *args, **kwargs):
        raise q.QualificationDecline('batch worker requires a finite run_tasks assignment')

    def run_json(self, *args, **kwargs):
        raise q.QualificationDecline('batch worker requires a finite run_tasks assignment')
