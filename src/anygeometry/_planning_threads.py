"""Opt-in in-process thread dispatch of the pair and face stages of intersection planning.

PROTOTYPE (thread-design slice, 2026-11-10). Serial planning is the default and is not changed.
The route is selected only by ``ANYGEOMETRY_PLANNING_THREADS=<1..8>`` and is useful only on a
free-threaded interpreter (GIL off); on a GIL build it gives the same plan, without speed-up.

Contract (see thread-design-note.md in the programme work folder):

* The planner hands the stage its ordinary task list and its own serial task function through the
  existing ``_stages`` hook; this module only decides WHERE each task runs. Results are returned in
  task order, so everything assembled from them is assembled in the serial order.
* Every task runs in its own copy of the dispatching context with private runtime-diagnostics
  collectors and a root-isolation memo that is private to its worker thread. Counters and records
  are published to the caller's collectors in task order after the stage.
* Refusals: every task with a lower index than a failed task runs to completion, so the failure
  reported is the first one in serial order, with the serial message. Counters of later tasks are
  discarded.
* The caller's ``cancellation_check`` is called on the dispatching thread only; workers poll an
  event.
* Not eligible (the request falls back to the unchanged serial route): predicate or candidate-pair
  budgets (their refusal point is order dependent), a planning mode other than exact, an epoch/native
  plan, a face support other than a plane or a cylinder, or an active tracing/work scope held in a
  context variable.
"""
import contextvars
import os
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from time import perf_counter_ns

from .errors import GeometryError
from . import runtime_diagnostics as _diagnostics
from .analytic_roots import _ROOT_MEMO

THREADS_ENVIRONMENT = 'ANYGEOMETRY_PLANNING_THREADS'
MAX_THREADS = 8
# Context variables that hold a live mutable work/trace object while their scope is active. A task
# that inherited one would share it with other threads, so an active scope keeps planning serial.
_SCOPE_VARIABLES = (
    ('anygeometry.exact_predicates', '_TRACE'),
    ('anygeometry.curve_parameter_matching', '_ACTIVE_WORK'),
    ('anygeometry.cylinder_charts', '_binding_budget'),
    ('anygeometry.support_curves', '_RUNTIME_CACHE_BANK'),
    ('anygeometry._quad_bound_reuse', '_ACTIVE'),
    ('anygeometry.plane_nurbs', 'CURRENT_WORK'),
    ('anygeometry.nurbs_intersection_curve', '_OPERATIONS'),
    ('anygeometry.nurbs_intersection_curve', '_FACT_ISSUER'),
    ('anygeometry.curvilinear_reference_maps', '_CLOSED_FAN_SCOPE'),
    ('anygeometry.curvilinear_reference_maps', '_CLOSED_FAN_ACTIVE'),
    ('anygeometry.prepared_material_curvilinear', '_SAMPLING_ARITHMETIC'),
    ('anygeometry.prepared_material_curvilinear', '_SAMPLING_ARITHMETIC_ISSUER'),
    ('anygeometry.certified_hydrostatics._admission', '_ACTIVE'),
    ('anygeometry.certified_hydrostatics._ch3_verify_math', '_active'),
)
_PHASES = {'pair': 'intersection_support_and_clipping', 'face': 'intersection_arrangements'}


def requested_threads():
    """The thread count asked for by the environment; 0 means the serial route."""
    raw = os.environ.get(THREADS_ENVIRONMENT)
    if raw is None or raw in ('', '0'):
        return 0
    if not (raw.isascii() and raw.isdecimal()) or not 1 <= int(raw) <= MAX_THREADS:
        raise GeometryError(
            f"{THREADS_ENVIRONMENT}={raw!r} is not supported (0 or unset: serial; 1..{MAX_THREADS}: threads)")
    return int(raw)


def _scope_active():
    for module_name, name in _SCOPE_VARIABLES:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        if getattr(module, name).get() not in (None, ()):
            return True
    return False


def _audited_supports(model):
    # The shared-store audit and the identity runs cover plane and cylinder supports. Other supports
    # (NURBS section curves use a validated shared runtime store) keep the serial route.
    from .surfaces import Cylinder, Plane
    return all(type(face.surface) in (Plane, Cylinder) for face in model.faces.values())


def thread_stages(model, policy, mode, native):
    """A stage dispatcher for this planning call, or None for the unchanged serial route."""
    threads = requested_threads()
    if not threads:
        return None
    if os.environ.get('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING') == '1':
        raise GeometryError(
            f"{THREADS_ENVIRONMENT} and ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING are mutually exclusive")
    if (mode != 'exact' or native is not None or policy.max_predicates is not None
            or policy.max_candidate_pairs is not None or _scope_active() or not _audited_supports(model)):
        return None
    return ThreadStages(threads, policy.cancellation_check)


class _Abandoned(BaseException):
    """A task stopped because an earlier task failed or planning was cancelled."""


class ThreadStages:
    in_process = True

    def __init__(self, threads, cancellation_check):
        self.threads = threads
        self._cancel = cancellation_check
        self._local = threading.local()
        self._owner = threading.get_ident()

    def check_deadline(self):
        # Same poll and message as the serial route; dispatching thread only.
        if self._cancel is not None and self._cancel():
            raise GeometryError("intersection planning cancelled")

    def run(self, kind, tasks, serial):
        if threading.get_ident() != self._owner:
            raise GeometryError("thread planning stage entered from a foreign thread")
        tasks = list(tasks)
        if not tasks:
            return []
        phase = _PHASES[kind]
        count = len(tasks)
        outcomes = [None] * count
        cancelled = threading.Event()
        lock = threading.Lock()
        first_failure = [count]
        parent = contextvars.copy_context()

        def execute(index):
            if cancelled.is_set() or index > first_failure[0]:
                return
            def check():
                if cancelled.is_set() or index > first_failure[0]:
                    raise _Abandoned()
            def body():
                counters, records = {}, {}
                _diagnostics._COLLECTORS.set((counters,))
                _diagnostics._RECORD_COLLECTORS.set((records,))
                memo = getattr(self._local, 'memo', None)
                if memo is None:
                    memo = self._local.memo = {}
                _ROOT_MEMO.set(memo)
                start = perf_counter_ns()
                try:
                    value, error = serial(tasks[index], check=check), None
                except _Abandoned:
                    return
                except BaseException as exc:  # reported in serial order by the dispatcher
                    value, error = None, exc
                    with lock:
                        if index < first_failure[0]:
                            first_failure[0] = index
                counters[phase + '_time_ns'] = counters.get(phase + '_time_ns', 0) + perf_counter_ns() - start
                outcomes[index] = (value, error, dict(counters=counters, records=records))
            parent.copy().run(body)

        order = sorted(range(count), key=lambda i: (-float(tasks[i].get('cost', 0.)), i))
        start = perf_counter_ns()
        pool = ThreadPoolExecutor(max_workers=min(self.threads, count), thread_name_prefix='anygeometry-plan')
        cancel_error = None
        try:
            pending = {pool.submit(execute, index) for index in order}
            futures = tuple(pending)
            while pending:
                if self._cancel is None or cancel_error is not None:
                    pending = wait(pending).not_done
                    continue
                pending = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED).not_done
                try:
                    self.check_deadline()
                except BaseException as exc:
                    cancel_error = exc
                    cancelled.set()
            for future in futures:
                future.result()  # an error of the dispatch machinery itself is never swallowed
        except BaseException:
            cancelled.set()
            raise
        finally:
            pool.shutdown(wait=True)
        _diagnostics._increment(f'intersection_thread_{kind}_stage_time_ns', perf_counter_ns() - start)
        if cancel_error is not None:
            raise cancel_error
        failed = first_failure[0]
        for index in range(min(failed + 1, count)):
            _diagnostics._publish_planning(outcomes[index][2])
        if failed < count:
            error = outcomes[failed][1]
            if isinstance(error, GeometryError):
                task = tasks[failed]
                if kind == 'pair':
                    raise GeometryError(
                        f"intersection batch face:{task['first']}/face:{task['second']}: {error}") from error
                raise GeometryError(f"intersection material arrangement face:{task['face']}: {error}") from error
            raise error
        return [outcome[0] for outcome in outcomes]

