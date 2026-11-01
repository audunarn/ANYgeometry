"""Opt-in PP2 digest executor skeleton. No public plan/application integration.

Only closed, cold model values and the explicit pair/face diagnostic task schema
are transported. Unsupported caller policies stay in the original serial caller.
The checksum detects corruption; it is not an authenticity or purity proof.
"""
import hashlib
import json
import math
import psutil
import os
from pathlib import Path
import sys
import time

from ._planning_qualification import QualificationDecline, check_types, input_type_names, model_extension_record
from ._planning_admission import run_qualified
from .errors import GeometryError


class PlannerCancelled(GeometryError):
    """Owned cancellation is terminal and never grants serial replay."""


class PlannerDeadline(GeometryError):
    """Consumed owned deadline is terminal, not a fresh serial budget."""


class PlannerCleanupError(GeometryError):
    """Failed containment cleanup cannot grant a serial replay."""


class PlannerSourceChanged(GeometryError):
    """A changed caller source is not silently resnapshotted for fallback."""


class PlannerMemoryExceeded(GeometryError):
    """An admitted host memory overrun cannot grant serial replay."""


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise QualificationDecline('duplicate transport key')
        result[key] = value
    return result


def _constant(value):
    raise QualificationDecline('nonfinite transport value')


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise QualificationDecline('nonfinite transport value')
    return result


def check_result_status(result):
    if type(result) is not dict:
        raise QualificationDecline('non-dict owned result')
    if result.get('status') == 'deadline':
        raise PlannerDeadline(str(result.get('reason', 'owned deadline consumed')))
    if result.get('status') == 'cancelled':
        raise PlannerCancelled(str(result.get('reason', 'owned cancellation')))
    if result.get('status') == 'cleanup_error':
        raise PlannerCleanupError(str(result.get('reason', 'owned cleanup failed')))
    if result.get('status') != 'passed':
        raise QualificationDecline(str(result.get('reason', 'planner refused')))


def pack(value, kind, *, limit=4_000_000):
    if type(limit) is not int or limit < 1:raise QualificationDecline("invalid payload limit="+str(limit))
    try:
        payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise QualificationDecline('unsupported transport value') from exc
    if len(payload.encode('utf-8')) > limit:
        raise QualificationDecline('transport size limit='+str(limit))
    return dict(schema='PP2-v1', kind=kind, payload=payload,
                sha256=hashlib.sha256(payload.encode('utf-8')).hexdigest())


def unpack(packet, kind, *, limit=4_000_000):
    if (type(packet) is not dict or set(packet) != {'schema', 'kind', 'payload', 'sha256'}
            or packet['schema'] != 'PP2-v1' or packet['kind'] != kind):
        raise QualificationDecline('transport schema/kind mismatch')
    payload = packet['payload']
    if (type(payload) is not str or len(payload.encode('utf-8')) > limit
            or type(packet['sha256']) is not str
            or hashlib.sha256(payload.encode('utf-8')).hexdigest() != packet['sha256']):
        raise QualificationDecline('transport checksum/size mismatch limit='+str(limit))
    try:
        return json.loads(payload, object_pairs_hook=_pairs, parse_constant=_constant, parse_float=_finite_float)
    except QualificationDecline:
        raise
    except (ValueError, RecursionError, UnicodeError) as exc:
        raise QualificationDecline('invalid transport JSON') from exc


def write_packet(path, packet):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(packet, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def read_packet(path, kind, *, with_digest=False, limit=4_000_000):
    path = Path(path)
    if path.stat().st_size > 2*limit+100000:
        raise QualificationDecline('transport envelope size limit='+str(2*limit+100000))
    try:
        text = path.read_text(encoding='utf-8')
        if len(text.encode('utf-8')) > 2*limit+100000:
            raise QualificationDecline('transport envelope size limit='+str(2*limit+100000))
        packet = json.loads(text, object_pairs_hook=_pairs,
                            parse_constant=_constant, parse_float=_finite_float)
    except QualificationDecline:
        raise
    except (ValueError, RecursionError, UnicodeError) as exc:
        raise QualificationDecline('invalid transport envelope') from exc
    value = unpack(packet, kind,limit=limit)
    return (value, packet['sha256']) if with_digest else value


def freeze_model(model, *, payload_limit=4_000_000,walk_limit=100000):
    from .model import GeometryModel
    from .serialization import to_dict
    if type(model) is not GeometryModel:
        raise QualificationDecline('unsupported model type')
    check_types(model, {'types': input_type_names()},limit=walk_limit)
    from ._planning_constructor import audit
    body = dict(document=to_dict(model), extension_record=model_extension_record(model,limit=walk_limit))
    rows = audit(vars(model),limit=walk_limit)
    if rows:
        body['constructor_state'] = rows
    return pack(body, 'model',limit=payload_limit)


def thaw_model(packet, *, payload_limit=4_000_000,walk_limit=100000):
    from .serialization import from_dict, to_dict
    decoded = unpack(packet, 'model',limit=payload_limit)
    if type(decoded) is not dict or set(decoded) not in ({'document', 'extension_record'}, {'document', 'extension_record', 'constructor_state'}):
        raise QualificationDecline('model wrapper schema mismatch')
    document = decoded['document']
    try:
        model = from_dict(document)
        from ._planning_constructor import audit
        audit(vars(model), restore=decoded.get('constructor_state', []),limit=walk_limit)
        actual = model_extension_record(model,limit=walk_limit)
        recorded = decoded['extension_record']
        if recorded == dict(present=False, sha256=None):
            if model._serialization_extensions != {}:
                raise QualificationDecline('absent source extension decoded nonempty')
        elif recorded != actual:
            raise QualificationDecline('model extension structural digest mismatch')
        if pack(to_dict(model), 'model-document',limit=payload_limit) != pack(document, 'model-document',limit=payload_limit):
            raise QualificationDecline('model transport changed values')
        return model
    except QualificationDecline:
        raise
    except Exception as exc:
        raise QualificationDecline('model decoding refused: ' + type(exc).__name__) from exc


def verify_threads(pools):
    if not pools or any(type(p.get('num_threads')) is not int or p['num_threads'] != 1 for p in pools):
        raise QualificationDecline('native thread count is absent or not one')


def assignment_size(batch_size, count):
    if batch_size is None:
        return max(1, count)
    if type(batch_size) is not int or batch_size < 1:
        raise QualificationDecline('invalid batch size')
    return batch_size


def reduce_results(rows, count):
    if type(rows) not in (tuple,list):raise QualificationDecline('result rows must be a sequence')
    result = {}
    for row in rows:
        if (type(row) not in (tuple, list) or len(row) != 2 or type(row[0]) is not int
                or row[0] not in range(count) or row[0] in result
                or type(row[1]) is not str or not row[1]):
            raise QualificationDecline('result coverage/type mismatch')
        result[row[0]] = row[1]
    if len(result) != count:
        raise QualificationDecline('result coverage mismatch')
    return tuple(result[index] for index in range(count))


def source_files():
    """File identity only; does not attest arbitrary host in-memory code changes."""
    root = Path(__file__).resolve().parent
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*.py'))}


def verify_sources(expected):
    if source_files() != expected:
        raise QualificationDecline('source file identity changed')


class OwnedPlanner:
    """One bounded fresh P; constructor may overlap caller project loading.

    Directory must be a new caller-owned evidence/scratch directory. close() is
    mandatory, including if an early-start planner is never submitted.
    """
    def __init__(self, directory, *, workers=8, batch_size=None, timeout=100,payload_limit=4_000_000,walk_limit=100000,_role='planner',tree_memory_limit=None):
        if _role not in ('planner','session-planner'):
            raise QualificationDecline('unsupported owned planner role')
        if type(workers) is not int or workers not in (1, 2, 8, 16):
            raise QualificationDecline('unsupported worker count')
        assignment_size(batch_size, 0)
        if type(timeout) not in (int, float) or not 0 < timeout <= 110:
            raise QualificationDecline('invalid owned process deadline='+str(timeout)+' cap=110')
        from ._planning_process import OwnedTree,validate_memory_limit
        validate_memory_limit(tree_memory_limit)
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        if any(type(v) is not int or v<1 for v in (payload_limit,walk_limit)):
            raise QualificationDecline('invalid payload/walk limit')
        self.limits=dict(payload_limit=payload_limit,walk_limit=walk_limit)
        write_packet(self.directory/'limits.json',pack(self.limits,'limits'))
        self.workers, self.batch_size = workers, batch_size
        self.started = time.perf_counter()
        self.deadline = time.monotonic() + timeout
        self.tree = OwnedTree(self.directory,**({} if tree_memory_limit is None else dict(memory_limit_bytes=tree_memory_limit)))
        self.child = None
        self.closed = False
        self.used = False
        try:
            self.child = self.tree.spawn(_role, self.directory, timeout)
        except BaseException:
            try:
                self.close()
            except Exception as cleanup:
                raise PlannerCleanupError('owned planner constructor cleanup failed') from cleanup
            raise

    def submit(self, model, tasks):
        return self._submit(model,tasks)

    def submit_stage(self, packet, tasks, *, poll=None):
        return self._submit(None,tasks,stage_packet=packet,poll=poll)

    def _submit(self, model, tasks, *, stage_packet=None, poll=None):
        if self.closed or self.used:
            raise QualificationDecline('planner already used or closed')
        self.used = True
        tasks = tuple(tasks)
        limits=getattr(self,"limits",dict(payload_limit=4_000_000,walk_limit=100000))
        stage = stage_packet is not None
        if not stage and any(type(t) is not str or t not in ('pair', 'face') for t in tasks):
            raise QualificationDecline('unsupported task descriptor')
        packet = stage_packet if stage else freeze_model(model,**limits)
        sources = source_files()
        request = dict(model=packet, tasks=tasks, workers=self.workers,
                       batch_size=self.batch_size, sources=sources, stage=stage)
        request_packet = pack(request, 'request',limit=limits['payload_limit'])
        if poll is not None:poll()
        write_packet(self.directory/'request.json', request_packet)
        submitted = time.perf_counter()
        path = self.directory/'result.json'
        while not path.exists():
            if poll is not None:poll()
            if (self.directory/'cancel.flag').exists():
                raise PlannerCancelled('owned planner cancelled')
            if self.child.poll() is not None:
                if path.exists():
                    break
                raise QualificationDecline('planner exit before result')
            if time.monotonic() >= self.deadline:
                raise PlannerDeadline('owned planner deadline consumed')
            time.sleep(.01)
        if poll is not None:poll()
        result = read_packet(path, 'result',limit=limits['payload_limit'])
        if (self.directory/'cancel.flag').exists():
            raise PlannerCancelled('owned planner cancelled')
        # The producer must finish before either results or serial replay escape.
        if poll is None:self.tree.wait(self.child, self.deadline)
        else:self.tree.wait(self.child, self.deadline,poll=poll)
        if poll is not None:poll()
        if (self.directory/'cancel.flag').exists():
            raise PlannerCancelled('owned planner cancelled')
        check_result_status(result)
        if result.get('request_sha256') != request_packet['sha256']:
            raise QualificationDecline('result request binding mismatch')
        verify_sources(sources)
        if not stage and freeze_model(model,**limits) != packet:
            raise PlannerSourceChanged('source model changed while planning')
        result['results'] = reduce_results(result.pop('rows'), len(tasks))
        result['host_timing'] = dict(start_to_submit_seconds=submitted-self.started,
                                    start_to_return_seconds=time.perf_counter()-self.started)
        return result

    def close(self):
        if not self.closed:
            self.closed = True
            self.tree.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def execute(model, tasks, serial_task, *, directory, opt_in=None, workers=8,
            batch_size=None, planner=None, expected_ci=None, observed_ci=None):
    """Original caller owns serial semantics. No numerical budget/callback adapter.

    Cleanup failure and BaseException cancellation propagate; qualification
    refusal records its typed reason then invokes the original serial callback.
    """
    tasks = tuple(tasks)
    enabled = os.environ.get('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING') == '1' if opt_in is None else opt_in
    if type(enabled) is not bool:
        raise QualificationDecline('opt-in must be bool')
    reason = None
    if enabled and tasks:
        owned = planner
        try:
            try:
                if owned is None:
                    owned = OwnedPlanner(directory, workers=workers, batch_size=batch_size)
                candidate = owned.submit(model, tasks)
            except (QualificationDecline, OSError, psutil.AccessDenied) as exc:
                reason = type(exc).__name__ + ': ' + str(exc)
        finally:
            # Cleanup failure is terminal even when it is an infrastructure error.
            if owned is not None:
                owned.close()
        if reason is None:
            # Processes are already retired. PP1i owns sampling and mandatory
            # two-component CI acceptance. The caller supplies the meaning of
            # the CI digests; diagnostic checks are not PP6 plan publication.
            checked = run_qualified(tasks, start_workers=lambda:None, stop_workers=lambda:None,
                parallel_tasks=lambda _:candidate['results'], serial_task=serial_task, opt_in=True,
                task_layout=candidate['task_layout'], plan_digest=candidate['request_sha256'],
                expected_ci=expected_ci,
                observed_ci=(lambda:observed_ci(candidate)) if observed_ci is not None else None,
                require_ci=True)
            return dict(candidate, **checked)
    elif planner is not None:
        planner.close()
    return dict(mode='serial', results=tuple(serial_task(t) for t in tasks),
                samples=0, decline_reason=reason)
