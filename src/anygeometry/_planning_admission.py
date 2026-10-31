"""PP1i risk-accepted qualification, not a purity proof; PP2/PP6 wiring pending.

The parent holds canonical task digests until verification completes. Worker
startup must call WorkerGate.start on every actual worker. No pool is created by
this module; start/stop/dispatch are explicit executor hooks for PP2.
"""
import os
import time
import hashlib
from ._planning_qualification import QualificationDecline, check_ci_equality


def sample_indices(count):
    """Legacy single-stream diagnostic sampler; not dispatch-aware or production CI."""
    if type(count) is not int or count < 0:
        raise QualificationDecline('invalid task count')
    size = min(count, max(2, (count + 19) // 20))
    return tuple(index * count // size for index in range(size))


def stratified_samples(layout, plan_digest):
    """Select immutable ordinal indices from (worker, position, class) rows.

    Every worker/class tail is mandatory. Fill each worker to at least max(2,
    ceil(Nw/20)) distinct samples, capped by Nw. Additional samples start in the
    late half, rotated by SHA256(plan digest + worker id), then the earlier half.
    This deterministic coverage is not a probabilistic accuracy certificate.
    """
    if (type(plan_digest) is not str or len(plan_digest) != 64 or
            any(c not in '0123456789abcdef' for c in plan_digest)):
        raise QualificationDecline('sampling requires a canonical plan digest')
    groups = {}
    for ordinal, row in enumerate(layout):
        if (type(row) not in (tuple, list) or len(row) != 3 or
                type(row[0]) is not int or row[0] < 0 or
                type(row[1]) is not int or row[1] < 0 or
                type(row[2]) is not str or row[2] not in ('pair', 'face')):
            raise QualificationDecline('invalid worker/position/class sampling metadata')
        groups.setdefault(row[0], []).append((row[1], ordinal, row[2]))
    selected, offsets = set(), []
    for worker, rows in sorted(groups.items()):
        rows.sort()
        if [r[0] for r in rows] != list(range(len(rows))):
            raise QualificationDecline('worker sampling positions must be unique and complete')
        selected.add(rows[-1][1])
        for kind in sorted({r[2] for r in rows}):
            selected.add(next(r[1] for r in reversed(rows) if r[2] == kind))
        seed = hashlib.sha256((plan_digest + ':' + str(worker)).encode('ascii')).hexdigest()
        late = rows[len(rows)//2:]
        offset = int(seed, 16) % len(late)
        offsets.append(dict(worker=worker, seed_sha256=seed, late_offset=offset))
        order = late[offset:] + late[:offset] + list(reversed(rows[:len(rows)//2]))
        quota = min(len(rows), max(2, (len(rows)+19)//20))
        present = sum(row[1] in selected for row in rows)
        for row in order:
            if present >= quota:
                break
            if row[1] not in selected:
                selected.add(row[1]); present += 1
    return dict(indices=tuple(sorted(selected)), workers=tuple(sorted(groups)),
                offsets=offsets, plan_digest=plan_digest,
                minimum='min(Nw,max(2,ceil(Nw/20))) per worker; >=1 per worker/class; every worker last')


def run_qualified(tasks, *, start_workers, stop_workers, parallel_tasks,
                  serial_task, opt_in=None, expected_ci=None, observed_ci=None,
                  task_layout=None, plan_digest=None, require_ci=False):
    """Risk-accepted qualification, not a purity proof; stage then verify/fallback.

    Hooks receive immutable task descriptors/fresh task inputs under PP2's contract.
    They return exact canonical digest strings, not approximate numeric equality.
    stop_workers must retire owned activity before serial verification/fallback.
    Unexpected errors and cancellation propagate; only qualification declines fall
    back. This hook does not mutate or apply a model; PP6 owns publication.
    task_layout is the actual dispatch (worker id, within-worker position, class).
    Omitting it preserves the legacy diagnostic API without worker guarantees.
    require_ci mandates both CI callbacks and dispatch metadata before startup.
    """
    tasks = tuple(tasks)
    enabled = (os.environ.get('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING') == '1'
               if opt_in is None else opt_in)
    if type(enabled) is not bool:
        raise QualificationDecline('opt-in must be bool')
    if type(require_ci) is not bool:
        raise QualificationDecline('mandatory CI mode must be bool')
    samples, sampling_seconds, reason = 0, 0.0, None
    sampling = None
    ci_verified = False
    if enabled and tasks:
        retired = False
        try:
            try:
                if require_ci and (expected_ci is None or observed_ci is None):
                    raise QualificationDecline('mandatory CI requires both plan/apply controls')
                if require_ci and task_layout is None:
                    raise QualificationDecline('mandatory CI requires dispatch sampling metadata')
                if task_layout is not None:
                    task_layout = tuple(task_layout)
                    if len(task_layout) != len(tasks):
                        raise QualificationDecline('sampling metadata coverage mismatch')
                    sampling = stratified_samples(task_layout, plan_digest)
                elif plan_digest is not None:
                    raise QualificationDecline('plan digest without dispatch sampling metadata')
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
                for index in (sampling['indices'] if sampling is not None else sample_indices(len(tasks))):
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
                    ci_verified = True
                except QualificationDecline as exc:
                    reason = str(exc)
            if reason is None:
                return dict(mode='parallel', results=candidate, samples=samples,
                            sampling_seconds=sampling_seconds, decline_reason=None,
                            sampling=sampling, ci_verified=ci_verified)
    return dict(mode='serial', results=tuple(serial_task(task) for task in tasks),
                samples=samples, sampling_seconds=sampling_seconds, decline_reason=reason,
                sampling=sampling, ci_verified=False)
