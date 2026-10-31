"""PP1g risk-accepted qualification, not a purity proof; PP2/PP6 wiring pending.

The parent holds canonical task digests until verification completes. Worker
startup must call WorkerGate.start on every actual worker. No pool is created by
this module; start/stop/dispatch are explicit executor hooks for PP2.
"""
import os
import time
from ._planning_qualification import QualificationDecline, check_ci_equality


def sample_indices(count):
    """Risk-accepted sample: 5 percent rounded up, minimum two, at most all."""
    if type(count) is not int or count < 0:
        raise QualificationDecline('invalid task count')
    size = min(count, max(2, (count + 19) // 20))
    return tuple(index * count // size for index in range(size))


def run_qualified(tasks, *, start_workers, stop_workers, parallel_tasks,
                  serial_task, opt_in=None, expected_ci=None, observed_ci=None):
    """Risk-accepted qualification, not a purity proof; stage then verify/fallback.

    Hooks receive immutable task descriptors/fresh task inputs under PP2's contract.
    They return exact canonical digest strings, not approximate numeric equality.
    stop_workers must retire owned activity before serial verification/fallback.
    Unexpected errors and cancellation propagate; only qualification declines fall
    back. This hook does not mutate or apply a model; PP6 owns publication.
    """
    tasks = tuple(tasks)
    enabled = (os.environ.get('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING') == '1'
               if opt_in is None else opt_in)
    if type(enabled) is not bool:
        raise QualificationDecline('opt-in must be bool')
    samples, sampling_seconds, reason = 0, 0.0, None
    if enabled and tasks:
        retired = False
        try:
            try:
                start_workers()
                candidate = tuple(parallel_tasks(tasks))
            finally:
                stop_workers()
                retired = True
            if len(candidate) != len(tasks) or any(type(value) is not str or not value for value in candidate):
                raise QualificationDecline('parallel task digest coverage/type mismatch')
        except QualificationDecline as exc:
            if not retired:
                raise
            reason = str(exc)
        if reason is None:
            started = time.perf_counter()
            try:
                for index in sample_indices(len(tasks)):
                    reference = serial_task(tasks[index])
                    samples += 1
                    if type(reference) is not str or reference != candidate[index]:
                        reason = 'sample digest mismatch at task ' + str(index)
                        break
            finally:
                sampling_seconds = time.perf_counter() - started
            if reason is None and (expected_ci is not None or observed_ci is not None):
                actual_ci = observed_ci() if observed_ci is not None else None
                try:
                    check_ci_equality(expected_ci, actual_ci)
                except QualificationDecline as exc:
                    reason = str(exc)
            if reason is None:
                return dict(mode='parallel', results=candidate, samples=samples,
                            sampling_seconds=sampling_seconds, decline_reason=None)
    return dict(mode='serial', results=tuple(serial_task(task) for task in tasks),
                samples=samples, sampling_seconds=sampling_seconds, decline_reason=reason)
