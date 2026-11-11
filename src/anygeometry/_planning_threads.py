"""Opt-in in-process thread dispatch of the pair and face stages of intersection planning.

PROTOTYPE (thread-design slice, 2026-11-10; Codex round-1 fixes 2026-11-11). Serial planning is the
default and is not changed. The route is selected only by ``ANYGEOMETRY_PLANNING_THREADS=<1..8>`` and
is useful only on a free-threaded interpreter (GIL off); on a GIL build it gives the same plan,
without speed-up.

Contract (see thread-design-note.md in the programme work folder):

* The threaded route returns exactly what the serial route returns (plan, deterministic counters,
  records), or planning runs on the serial route. There are two ways to the serial route:
  the call is not eligible (``thread_stages`` returns ``None``), or the threaded attempt raised
  (``plan_with_threads``): its counters and records are discarded whole and planning is repeated
  serially, so a refusal carries serial's message, cause, counters and records by construction.
  The discarded attempt is visible only as the clock ``intersection_thread_discarded_attempt_time_ns``.
* The planner hands the stage its ordinary task list and its own serial task function through the
  existing ``_stages`` hook; this module only decides WHERE each task runs. Results are returned in
  task order, so everything assembled from them is assembled in the serial order.
* Every task runs in its own copy of the dispatching context with private runtime-diagnostics
  collectors and a root-isolation memo that is private to its worker thread. Counters and records
  are published to the caller's collectors in task order after the stage.
* A failed task stops the stage: tasks with a higher index are abandoned, the failure with the
  lowest index is raised with the serial wrapping (and then leads to the serial repeat above).
* Not eligible: a ``cancellation_check`` on the policy (a callback's poll sequence cannot be
  reproduced from several threads), predicate or candidate-pair budgets (their refusal point is order
  dependent), a planning mode other than exact, an epoch/native plan, a face support other than a
  plane or a cylinder, an edge curve other than a straight line, a circular arc or an elliptic arc
  (exact types; these are what the shared-store audit and the identity runs cover), or an active
  tracing/work scope held in a context variable.
* Declined during the attempt (``_NotAudited``, then the serial repeat above): the EFFECTIVE stage
  inputs are checked by exact type, not only the authored model. Before the pair stage every face must
  have a ``MaterialDomain`` whose support is a plane or a cylinder and whose boundary curves are audited
  path curves; before the face stage every trace (pair traces and member inputs) must be an
  ``ArrangementPath`` with an audited path curve and every point an ``ArrangementPoint``.
* Cell normalisation runs inside the face task, directly after that face's arrangement, as in serial.
* Identity excludes clock values only: the ``*_time_ns`` counters and the record fields named in
  ``RECORD_CLOCK_FIELDS``. Every other counter and record value is part of the contract.
* The switch is validated at the public entry before any route is chosen (``validate_switches``): the value
  is unset, empty, ``0`` or exactly one of ``1`` .. ``8``, and it excludes the process-route switch.
* Interrupts (``KeyboardInterrupt`` / ``SystemExit``, any non-``Exception``) are never turned into a serial
  repeat. ONE rule owns this, ``_Attempt.verdict``, applied by ``plan_with_threads`` (and by a ``ThreadStages.run``
  called on its own) when the attempt ends with an ordinary exception: (1) an interrupt seen anywhere in a stage
  of the attempt - recorded by a worker (task function, setup, bookkeeping, outcome storage; lowest task index
  first), met by the dispatching thread in ``wait()`` or in its cleanup - is raised instead of that exception;
  (2) otherwise, if the workers are not known to have finished (``pool.shutdown`` did not return), a typed
  ``GeometryError`` is raised and nothing is planned beside them; (3) only otherwise planning is repeated
  serially. An interrupt that propagates by itself is not caught at all.
  Limit of the rule: it sees what the route's own code observed. An ordinary failure of an interpreter
  primitive at the very moment of an interrupt can mask it only where the interrupt was never observed (for
  example inside ``shutdown`` before any Python statement of this module runs); that is outside the contract,
  as it is on the serial route.
* Inside a stage that ends normally the dispatcher still raises in this order: lowest-index worker interrupt,
  interrupt of the dispatching thread, ordinary machinery failure (lowest task index), lowest-index task
  failure with the serial wrapping.
"""
import contextvars
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from time import perf_counter_ns

from .errors import GeometryError
from . import runtime_diagnostics as _diagnostics
from .analytic_roots import _ROOT_MEMO
from .arrangement_geometry import LinePath
from .branch_curves import BezierQuadricCurve
from .curves import Arc, Straight
from .exact_curves import CylinderIntersectionCurve, EllipticArc
from .material_arrangement import ArrangementPath, ArrangementPoint, MaterialDomain
from .surfaces import Cylinder, Plane

THREADS_ENVIRONMENT = 'ANYGEOMETRY_PLANNING_THREADS'
PROCESS_ENVIRONMENT = 'ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING'
MAX_THREADS = 8
# Record fields that hold a clock value, by record key: the only record contents outside the identity contract.
RECORD_CLOCK_FIELDS = {'arrangement_snap_fallback': ('seconds',)}
# What the shared-store audit and the identity runs cover (exact types, no subclasses). NURBS section
# curves use a validated shared runtime store; splines and the other exact curves were never run threaded.
_AUDITED_SUPPORTS = (Plane, Cylinder)
_AUDITED_EDGE_CURVES = (Straight, Arc, EllipticArc)
# Frozen curves that may reach a stage inside a domain boundary or a trace (exact types): what the audited
# edges freeze to and what the pair routes generated in the identity runs (census of the 570 captured N2c
# face inputs, 2026-11-11: LinePath, EllipticArc, CylinderIntersectionCurve, BezierQuadricCurve; their
# per-object stores are rows 1-5 of the object-store audit). QuadricIntersectionCurve was never run threaded.
_AUDITED_PATH_CURVES = (LinePath, EllipticArc, CylinderIntersectionCurve, BezierQuadricCurve)
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
    # Canonical spelling only ('1' .. '8'): no sign, padding, leading zero or other digit alphabet.
    if raw not in _THREAD_COUNTS:
        shown = repr(raw) if len(raw) <= 16 else repr(raw[:16]) + f'... ({len(raw)} characters)'
        raise GeometryError(
            f"{THREADS_ENVIRONMENT}={shown} is not supported (0 or unset: serial; 1..{MAX_THREADS}: threads)")
    return _THREAD_COUNTS[raw]


_THREAD_COUNTS = {str(count): count for count in range(1, MAX_THREADS + 1)}


def validate_switches():
    """Typed refusal of an unsupported thread switch or of both parallel switches; called once per public
    planning call before any route (existing-material, bounded, process, thread, serial) is chosen."""
    if requested_threads() and os.environ.get(PROCESS_ENVIRONMENT) == '1':
        raise GeometryError(f"{THREADS_ENVIRONMENT} and {PROCESS_ENVIRONMENT} are mutually exclusive")


def _scope_active():
    for module_name, name in _SCOPE_VARIABLES:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        if getattr(module, name).get() not in (None, ()):
            return True
    return False


def _audited_model(model):
    return (all(type(face.surface) in _AUDITED_SUPPORTS for face in model.faces.values())
            and all(type(edge.curve) in _AUDITED_EDGE_CURVES for edge in model.edges.values()))


def thread_stages(model, policy, mode, native):
    """A stage dispatcher for this planning call, or None for the unchanged serial route."""
    validate_switches()
    threads = requested_threads()
    if not threads:
        return None
    if (mode != 'exact' or native is not None or policy.cancellation_check is not None
            or policy.max_predicates is not None or policy.max_candidate_pairs is not None
            or _scope_active() or not _audited_model(model)):
        return None
    return ThreadStages(threads)


class _Attempt:
    """What the stages of one threaded attempt observed; the single owner of the interrupt rule (``verdict``)."""
    __slots__ = ('worker_slots', 'dispatcher', 'running')

    def __init__(self):
        self.worker_slots = []     # per stage: the list of worker interrupts by task index (None = none)
        self.dispatcher = None     # the first interrupt met by the dispatching thread itself
        self.running = False       # True from the first submit until pool.shutdown(wait=True) has returned

    def note(self, exc):
        """Remember an interrupt of the dispatching thread (no allocation; ordinary exceptions are not kept)."""
        if self.dispatcher is None and not isinstance(exc, Exception):
            self.dispatcher = exc

    def interrupt(self):
        for slots in self.worker_slots:
            for exc in slots:
                if exc is not None:
                    return exc
        return self.dispatcher

    def verdict(self, failure):
        """What must be raised instead of the ordinary ``failure`` of the attempt, or None for the serial repeat."""
        interrupt = self.interrupt()
        if interrupt is not None:
            return interrupt
        if self.running:
            error = GeometryError("thread planning stage did not finish: its workers are not known to have "
                                  "stopped, so planning is not repeated serially")
            error.__cause__ = failure
            return error
        return None


_ATTEMPT = contextvars.ContextVar('anygeometry_planning_thread_attempt', default=None)


def plan_with_threads(threaded, serial):
    """``threaded()`` with its diagnostics published on success; on any failure the result of ``serial()``.

    The attempt runs under private collectors. A refusal (or any other exception) of the attempt may have
    counted work the serial route never reaches and may come from another task than serial's first
    failure, so nothing of it is kept: the serial route is run from the start and decides alone.
    ``KeyboardInterrupt`` and ``SystemExit`` pass through, and ``_Attempt.verdict`` decides whether an ordinary
    failure may lead to the serial repeat at all (not after an interrupt seen anywhere in the attempt, and not
    while workers are not known to have finished).
    """
    start = perf_counter_ns()
    attempt, failure, plan = _Attempt(), None, None
    token = _ATTEMPT.set(attempt)
    try:
        with _diagnostics._planning_transaction() as collected:
            plan = threaded()
    except Exception as exc:
        failure = exc
    finally:
        _ATTEMPT.reset(token)
    if failure is None:
        _diagnostics._publish_planning(collected)
        return plan
    # Outside the handler, so what follows has no threaded context. The one place that decides.
    instead = attempt.verdict(failure)
    if instead is not None:
        raise instead
    _diagnostics._increment('intersection_thread_discarded_attempt_time_ns', perf_counter_ns() - start)
    return serial()


class _NotAudited(Exception):
    """A stage input is outside the audited exact types: the attempt is dropped for the serial route.

    Deliberately not a GeometryError: it is never an answer about the geometry and never reaches a caller
    (``plan_with_threads`` repeats planning serially).
    """


def _audited_paths(paths):
    return all(type(path) is ArrangementPath and type(path.curve) in _AUDITED_PATH_CURVES for path in paths)


class _Abandoned(BaseException):
    """A task stopped because an earlier task failed or the dispatcher is unwinding."""


class ThreadStages:
    in_process = True

    def __init__(self, threads):
        self.threads = threads
        self._local = threading.local()
        self._owner = threading.get_ident()

    def check_deadline(self):
        # No caller callback exists on this route (a policy with one is not eligible).
        return None

    def admit_domains(self, domains, boundary_only):
        """Before the pair stage: every face has an audited effective material domain."""
        if boundary_only:
            raise _NotAudited('a face without a material domain')
        for domain in domains.values():
            if (type(domain) is not MaterialDomain or type(domain.support) not in _AUDITED_SUPPORTS
                    or not all(_audited_paths(loop) for loop in domain.boundaries)):
                raise _NotAudited('material domain outside the audited supports and boundary curves')

    def admit_face_inputs(self, traces, points):
        """Before the face stage: pair traces, member inputs and points are of audited exact types."""
        if not all(_audited_paths(paths) for paths in traces.values()):
            raise _NotAudited('trace outside the audited path curves')
        if not all(type(point) is ArrangementPoint for found in points.values() for point in found):
            raise _NotAudited('point input of an unaudited type')

    def run(self, kind, tasks, serial):
        if threading.get_ident() != self._owner:
            raise GeometryError("thread planning stage entered from a foreign thread")
        tasks = list(tasks)
        if not tasks:
            return []
        attempt = _ATTEMPT.get()
        if attempt is not None:                      # inside plan_with_threads: the boundary applies the rule
            return self._run(kind, tasks, serial, attempt)
        attempt = _Attempt()                         # called on its own: the same rule, applied here
        try:
            return self._run(kind, tasks, serial, attempt)
        except Exception as failure:
            caught, instead = failure, attempt.verdict(failure)
        raise instead if instead is not None else caught

    def _run(self, kind, tasks, serial, attempt):
        phase = _PHASES[kind]
        count = len(tasks)
        outcomes = [None] * count
        stopped = threading.Event()
        lock = threading.Lock()
        first_failure = [count]
        # Per task index, preallocated so that recording needs no allocation: a KeyboardInterrupt / SystemExit
        # (any non-Exception) seen anywhere in that task's worker, and an ordinary exception of the worker
        # machinery outside the task function (setup, bookkeeping, outcome storage).
        interrupts = [None] * count
        machinery = [None] * count
        attempt.worker_slots.append(interrupts)
        parent = contextvars.copy_context()

        def execute(index):
            # Nothing leaves a worker: whatever happens is recorded and judged after all workers have finished.
            try:
                if stopped.is_set() or index > first_failure[0]:
                    return
                parent.copy().run(body, index)
            except Exception as exc:
                machinery[index] = exc
                stopped.set()
            except BaseException as exc:
                interrupts[index] = exc
                stopped.set()

        def body(index):
            def check():
                if stopped.is_set() or index > first_failure[0]:
                    raise _Abandoned()
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
            except Exception as exc:          # a task failure: reported in serial order by the dispatcher
                value, error = None, exc
                with lock:
                    if index < first_failure[0]:
                        first_failure[0] = index
            # A non-Exception out of the task function (an interrupt) goes to execute, like one raised by the
            # setup above or the bookkeeping below.
            counters[phase + '_time_ns'] = counters.get(phase + '_time_ns', 0) + perf_counter_ns() - start
            outcomes[index] = (value, error, dict(counters=counters, records=records))

        order = sorted(range(count), key=lambda i: (-float(tasks[i].get('cost', 0.)), i))
        start = perf_counter_ns()
        pool = ThreadPoolExecutor(max_workers=min(self.threads, count), thread_name_prefix='anygeometry-plan')
        held = None              # what the dispatching thread itself met; judged after the workers have drained
        futures = []
        attempt.running = True
        try:
            try:
                for index in order:
                    futures.append(pool.submit(execute, index))
                wait(futures)
            except BaseException as exc:
                attempt.note(exc)        # first of all: an interrupt is kept whatever the cleanup below does
                held = exc
                stopped.set()
            finally:
                pool.shutdown(wait=True)
                attempt.running = False  # only now are all workers known to have finished
        except BaseException as exc:     # the cleanup itself failed or was interrupted: verdict decides
            attempt.note(exc)
            raise
        for future in futures:   # execute records everything; a future carries an error only if recording failed
            try:
                future.result()
            except BaseException as exc:
                if held is None or (isinstance(held, Exception) and not isinstance(exc, Exception)):
                    held = exc
        for exc in interrupts:   # 1. the lowest-index interrupt of a worker
            if exc is not None:
                raise exc
        if held is not None and not isinstance(held, Exception):   # 2. an interrupt of the dispatching thread
            raise held
        for exc in machinery:    # 3. an ordinary machinery failure (leads to the serial repeat)
            if exc is not None:
                raise exc
        if held is not None:
            raise held
        _diagnostics._increment(f'intersection_thread_{kind}_stage_time_ns', perf_counter_ns() - start)
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
