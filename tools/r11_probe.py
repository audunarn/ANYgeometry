"""R11 single-process wire-feasibility probe. NEVER spawns children, workers or Jobs.

One serial stage-capture pass through the real ``_plan_intersections`` ``_stages`` interface.
Tasks run in-process through ``WorkerModel`` + ``stage_task`` under ``_planning_transaction``
(worker semantics, one compute per task); the decoded values go back to the planner exactly as
the production bridge returns them. Every packet is then formed with the SAME pure builders as
the normal route (``build_session_setup/build_session_command/build_worker_command/
build_worker_report/build_session_reply/build_infrastructure_record``, ``transport_tasks_v2``,
``dispatch``, ``compact_worker_report``, ``pack``) and each full session reply is checked by the
real ``verify_prepared_stage`` so the proof cannot drift from production schema checks.

Component classes in the output:
  exact          real bytes (setup, commands, result rows, diagnostics under probe-process semantics)
  upper-bound    worst-case placeholders for values that need real workers: finite float repr
                 (23 chars), pid 2**32-1, counters 2**63-1, BLAS info at the DECLARED production bound
                 (``reply_bounds()``, enforced on the host for every compact worker report)
Diagnostics are measured with the REAL per-task ``reset_for_batch(owned_worker=True)`` of a PackageWorkerLedger
built for the probe's own WorkerModel (dedicated process, registered declined caches only), the same task
payload decode/check_types and the same diagnostic transaction as BatchWorkerGate._execute. Not a native-worker
qualification: the ledger's worker-role absence invariant cannot hold here and is recorded.
No flag upgrades missing evidence. Open named gap: ``worker_setup.expected_environment_packet`` (the child's
expected-environment snapshot is not reproduced) => feasibility stays ``conditional``; ``complete`` only for
``passed``.

Usage:
  python tools/r11_probe.py --model DOC.json --sha256 HEX --output DIR
  python tools/r11_probe.py --fixture-small --output DIR
  python tools/r11_probe.py --fixture-small --output DIR --compare-dir CAPTURED_SESSION_DIR
  python tools/r11_probe.py --fixture-small --output DIR --compare-reply ACTUAL_SESSION_REPLY.json --compare-stage pair
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

MEASURE_LIMIT = 256 * 2**20
PAYLOAD_LIMIT = 4_000_000
ENVELOPE_LIMIT = 2 * PAYLOAD_LIMIT + 100_000
CEILINGS = dict(callable_nodes=80000, authority_nodes=80000, model_value_nodes=100000,
                payload_bytes=PAYLOAD_LIMIT, envelope_bytes=ENVELOPE_LIMIT)
FLOAT_WORST = 2.2250738585072014e-308      # smallest normal double: 17 digits + 'e-308' = 23 chars, the maximum repr length
MISSING_REPLY_FIELDS = ('diagnostics.ledger_reset_semantics',)


def walk_nodes(value):
    """Nodes of a decoded JSON value (every container, key and scalar)."""
    count = 0
    stack = [value]
    while stack:
        item = stack.pop()
        count += 1
        if type(item) is dict:
            count += len(item)
            stack.extend(item.values())
        elif type(item) in (list, tuple):
            stack.extend(item)
    return count


def measure_packet(name, body, kind, *, payload_limit=PAYLOAD_LIMIT, exact=True):
    """Pack with the real machinery (measurement limit) and compare to ceilings."""
    from anygeometry._planning_executor import pack
    packet = pack(body, kind, limit=MEASURE_LIMIT)
    payload_bytes = len(packet['payload'].encode('utf-8'))
    envelope_bytes = len(json.dumps(packet, allow_nan=False).encode('utf-8'))
    nodes = walk_nodes(json.loads(packet['payload']))
    return dict(name=name, kind=kind, exact=exact, payload_utf8_bytes=payload_bytes, envelope_bytes=envelope_bytes,
                walked_nodes=nodes, sha256=packet['sha256'],
                within=dict(payload=payload_bytes <= payload_limit,
                            envelope=envelope_bytes <= 2 * payload_limit + 100000,
                            walk=nodes <= 100000))


def stable_diagnostics(data):
    """Comparison content: wall-time counters (*_time_ns) and the one known record timing field
    (arrangement_snap_fallback['seconds']) are removed; everything else is compared exactly."""
    from anygeometry._planning_session import stable_diagnostics as shared
    return shared(data)


def bound_diagnostics(data):
    """Upper-bound form of a diagnostic transaction: every wall-time counter (variable-length integer) is set to
    the declared production int63 cap; all other content is the real measured content (stable)."""
    from anygeometry._planning_session import reply_bounds, timing_record_fields, check_diagnostic_timing
    check_diagnostic_timing(data)           # a malformed known timing field is a typed decline, never bounded away
    records, timing, counter_max = {}, timing_record_fields(), reply_bounds()['counter_max']
    for name, rows in data.get('records', {}).items():
        field = timing.get(name)
        records[name] = [dict(row, **{field: FLOAT_WORST}) if field and type(row) is dict else row for row in rows]
    return dict(counters={k: (counter_max if k.endswith('_time_ns') else v)
                          for k, v in data.get('counters', {}).items()},
                records=records)


class Recorder:
    """``_stages`` object: runs each task ONCE with WorkerModel+stage_task in this process and records it.

    With a ledger, each task first calls the real ``reset_for_batch(owned_worker=True)`` (registered declined
    caches of THIS dedicated process only), then decodes the task payload and runs check_types exactly like
    BatchWorkerGate._execute, inside the same diagnostic transaction. Logical worker partitions share this one
    context: every task starts from freshly reset registered caches and a fresh WorkerModel.view(), and a
    determinism self-check (small cases) re-executes the last task to demonstrate that."""
    def __init__(self, context, limits, workers=8, ledger=None):
        self.context, self.limits, self.workers, self.ledger = context, limits, workers, ledger
        self.stages = []
        self.reset_calls = 0
        self.reset_total = 0
        self.determinism = {}

    def check_deadline(self):
        return None

    def _execute(self, task):
        from anygeometry import _planning_qualification as q
        from anygeometry._planning_stages import stage_task
        from anygeometry.runtime_diagnostics import _planning_transaction
        payload = json.dumps(dict(args=[task], kwargs={}), sort_keys=True, separators=(',', ':')).encode()
        digest = hashlib.sha256(payload).hexdigest()
        if self.ledger is not None:
            counts = self.ledger.reset_for_batch(owned_worker=True)
            self.reset_calls += 1
            if isinstance(counts, dict):
                self.reset_total += sum(v for v in counts.values() if type(v) is int)
        args, kwargs = q.decode_task_payload(payload, digest, limit=self.limits['payload_limit'])
        q.check_types((args, kwargs), {'types': q.input_type_names()}, limit=self.limits['walk_limit'])
        with _planning_transaction() as data:
            encoded = stage_task(self.context, *args, **kwargs)
        return encoded, data

    def run(self, kind, tasks, serial_task):
        from anygeometry._planning_values import transport_tasks_v2, verify_transport_tasks_v2, decode_stage_values
        started = time.perf_counter()
        transported = transport_tasks_v2(tasks, limit=self.limits['walk_limit'], payload_limit=MEASURE_LIMIT)
        verify_transport_tasks_v2(tasks, transported, limit=self.limits['walk_limit'], payload_limit=MEASURE_LIMIT)
        encoded, diagnostics = [], []
        for task in transported:
            text, data = self._execute(task)
            encoded.append(text)
            diagnostics.append(data)
        if self.ledger is not None and 0 < len(transported) <= 16:
            again, data = self._execute(transported[-1])
            self.determinism[kind] = bool(again == encoded[-1] and stable_diagnostics(data) == stable_diagnostics(diagnostics[-1]))
        values = decode_stage_values(encoded, limit=self.limits['walk_limit'], payload_limit=MEASURE_LIMIT, codec='v2')
        self.stages.append(dict(kind=kind, tasks=list(tasks), transported=list(transported), encoded=encoded,
                                diagnostics=diagnostics, seconds=time.perf_counter() - started))
        return values


def worst_string(chars):
    """Worst JSON(ensure_ascii) form: U+10FFFF is two surrogate escapes = 12 ASCII characters per codepoint."""
    return '\U0010FFFF' * chars


def worst_blas():
    from anygeometry._planning_session import reply_bounds, _BLAS_STRING_FIELDS
    REPLY_BOUNDS = reply_bounds()
    entry = {name: worst_string(REPLY_BOUNDS['blas_filepath_chars'] if name == 'filepath' else REPLY_BOUNDS['blas_string_chars'])
             for name in _BLAS_STRING_FIELDS}
    entry['num_threads'] = 1            # verify_threads requires exactly one thread
    return [dict(entry) for _ in range(REPLY_BOUNDS['blas_entries'])]


def timing_worst(count, encoded_bytes):
    return dict(count=count, total_wall=FLOAT_WORST, total_cpu=FLOAT_WORST, setup_wall=FLOAT_WORST, setup_cpu=FLOAT_WORST,
                kernel_wall=FLOAT_WORST, kernel_cpu=FLOAT_WORST, encode_wall=FLOAT_WORST, encode_cpu=FLOAT_WORST,
                bookkeeping_wall=FLOAT_WORST, bookkeeping_cpu=FLOAT_WORST, encoded_utf8_bytes=encoded_bytes,
                max_total_wall=FLOAT_WORST, max_kernel_wall=FLOAT_WORST)


def build_stage_reply(stage, *, session, sequence, previous, seen, workers, packet, numerical, infrastructure_native,
                      worst=True):
    """Full session reply for one captured stage via the production builders.

    worst=True uses upper-bound placeholders; worst=False uses small realistic metadata (to emulate an
    accepted run in tests / bound-vs-actual comparison). Returns (parent body, packets dict, session body)."""
    from anygeometry._planning_session import (build_session_command, build_worker_command, build_worker_report,
        build_session_reply, build_infrastructure_record, compact_worker_report, reply_bounds)
    from anygeometry._planning_stages import dispatch
    from anygeometry._planning_executor import pack
    REPLY_BOUNDS = reply_bounds()
    kind = stage['kind']
    body = build_session_command(session, sequence, previous, kind, stage['transported'], seen=seen)
    request = pack(body, 'session-command', limit=MEASURE_LIMIT)['sha256']
    parts, layout = dispatch(body['tasks'], workers)
    seconds = FLOAT_WORST if worst else 0.0123456
    pid = REPLY_BOUNDS['pid_max'] if worst else 12345
    counter = REPLY_BOUNDS['counter_max'] if worst else 3
    blas = worst_blas() if worst else [dict(user_api='blas', internal_api='openblas', prefix='libopenblas',
        filepath='C:\\x\\openblas.dll', version='0.3.31', num_threads=1, threading_layer='pthreads', architecture='SkylakeX')]
    guard = dict(controller_loaded=False, native_adapter_loaded=False, native_core_loaded=False,
                 binary_sha256=infrastructure_native['binary_sha256'])
    commands, replies, compacts, rows, components, bodies = [], [], [], [], [], []
    for index in range(workers):
        part = parts[index] if index < len(parts) else []
        assigned = [[i, body['tasks'][i]] for i in part]
        command = build_worker_command(session, sequence, previous, kind, assigned, index, request)
        commands.append(measure_packet('%s worker-%d command' % (kind, index), command, 'session-worker-command'))
        command_sha = pack(command, 'session-worker-command', limit=MEASURE_LIMIT)['sha256']
        n = len(part)
        report = build_worker_report(
            dict(session_sha256=session, worker=index, worker_setup_sha256='0' * 64, request_sha256=command_sha),
            rows=[[i, stage['encoded'][i]] for i in part], model_sha256=packet['sha256'], pid=pid, task_count=n,
            batch_size=max(1, n), assignment_seconds=seconds, startup_seconds=seconds, task_seconds=[seconds] * n,
            task_timing=(timing_worst(n, sum(len(stage['encoded'][i].encode('utf-8')) for i in part)) if worst else
                         dict(timing_worst(n, sum(len(stage['encoded'][i].encode('utf-8')) for i in part)),
                              **{k: seconds for k in ('total_wall', 'total_cpu', 'setup_wall', 'setup_cpu', 'kernel_wall',
                                 'kernel_cpu', 'encode_wall', 'encode_cpu', 'bookkeeping_wall', 'bookkeeping_cpu',
                                 'max_total_wall', 'max_kernel_wall')})),
            boundary_seconds=[seconds] * (4 if n else 3), reset_count=counter, full_environment_checks=1,
            blas_threads=blas, gate_closed=True, gate_poisoned=False,
            diagnostics=[[i, bound_diagnostics(stage['diagnostics'][i]) if worst else stage['diagnostics'][i]] for i in part],
            resource_guard=guard,
            value_codec=numerical['value_codec'])
        replies.append(measure_packet('%s worker-%d reply' % (kind, index), report, 'session-worker-reply', exact=False))
        rows.extend(report['rows'])
        bodies.append((command, report))
        rows_text = json.dumps(report['rows'], sort_keys=True, separators=(',', ':'))
        stable_text = json.dumps([[i, stable_diagnostics(stage['diagnostics'][i])] for i in part], sort_keys=True,
                                 separators=(',', ':'))
        components.append(dict(worker=index, task_count=n, rows_sha256=hashlib.sha256(rows_text.encode()).hexdigest(),
            rows_bytes=len(rows_text.encode()), diagnostics_stable_sha256=hashlib.sha256(stable_text.encode()).hexdigest(),
            diagnostics_bytes=len(json.dumps(report['diagnostics'], sort_keys=True, separators=(',', ':')).encode())))
        compacts.append(compact_worker_report(report, part, payload_limit=MEASURE_LIMIT))
    infra = build_infrastructure_record(numerical, 'a' * 64, counter, infrastructure_native)
    infra.update(planner_reference='b' * 64, worker_reference='c' * 64)
    session_body = build_session_reply(session, request, kind, rows=rows, infrastructure=infra, worker_reports=compacts,
        requested_workers=workers, actual_workers=len(parts), task_layout=layout, model_sha256=packet['sha256'],
        startup=dict(configured_concurrency=workers, max_inflight_startups=workers, ready_workers=workers,
                     active_workers=len(parts)))
    return body, dict(parent_command=measure_packet(kind + ' session-command', body, 'session-command'),
                      worker_commands=commands, worker_replies=replies, compact_reports=compacts,
                      worker_components=components, worker_bodies=bodies), session_body


def verify_with_production_schema(stage, body, session_body, *, workers, packet, numerical, infrastructure_native):
    """The real host-side verifier must accept the constructed reply (drift guard)."""
    from anygeometry._planning_session import verify_prepared_stage
    from anygeometry._planning_executor import reduce_results
    report = dict(session_body)
    results = reduce_results(report.pop('rows'), len(body['tasks']))
    setup = dict(workers=workers, model=packet, startup_concurrency=workers,
                 resource_identity=infrastructure_native, value_codec=numerical['value_codec'])
    final = [{'pid': r['pid']} for r in report['worker_reports']]
    verify_prepared_stage(dict(report, results=list(results)), body['tasks'], setup, final, payload_limit=MEASURE_LIMIT)


def stage_proof(stage, *, session, sequence, previous, seen, workers, packet, numerical, infrastructure_native,
                payload_limit=PAYLOAD_LIMIT):
    out = dict(kind=stage['kind'], task_count=len(stage['tasks']), errors=[], components=[])
    try:
        body, packets, session_body = build_stage_reply(stage, session=session, sequence=sequence, previous=previous,
            seen=seen, workers=workers, packet=packet, numerical=numerical, infrastructure_native=infrastructure_native)
        verify_with_production_schema(stage, body, session_body, workers=workers, packet=packet, numerical=numerical,
                                      infrastructure_native=infrastructure_native)
        out['session_reply'] = measure_packet(stage['kind'] + ' session-reply', session_body, 'session-reply',
                                              payload_limit=payload_limit, exact=False)
        out['parent_command'] = packets['parent_command']
        out['worker_commands'] = packets['worker_commands']
        out['worker_replies'] = packets['worker_replies']
        out['worker_components'] = packets['worker_components']
        out['tasks_sha256'] = _h(list(stage['transported']))
        out['session_rows_sha256'] = hashlib.sha256(json.dumps(session_body['rows'], sort_keys=True,
                                                               separators=(',', ':')).encode()).hexdigest()
        out['compact_reports_utf8_bytes'] = [len(json.dumps(r, sort_keys=True, separators=(',', ':')).encode('utf-8'))
                                             for r in packets['compact_reports']]
        out['result_value_utf8_total'] = sum(len(v.encode('utf-8')) for v in stage['encoded'])
        out['diagnostics_canonical_bytes'] = sum(len(json.dumps(d, sort_keys=True, separators=(',', ':')).encode('utf-8'))
                                                 for d in stage['diagnostics'])
        for row in [out['parent_command'], out['session_reply'], *out['worker_commands'], *out['worker_replies']]:
            ok = row['payload_utf8_bytes'] <= payload_limit and row['envelope_bytes'] <= 2 * payload_limit + 100000
            out['components'].append(dict(name=row['name'], payload=row['payload_utf8_bytes'],
                                          envelope=row['envelope_bytes'], exact=row['exact'], within=ok))
            if not ok:
                out['errors'].append('limit: %s payload=%d envelope=%d' % (row['name'], row['payload_utf8_bytes'],
                                                                           row['envelope_bytes']))
    except BaseException as error:
        out['errors'].append('refused: %s: %s' % (type(error).__name__, error))
        if not isinstance(error, Exception):
            raise
    out['within'] = not out['errors']
    return out


def compare_actual(stage_result, reply_path):
    """Compare the proof bound for a stage against an ACTUAL accepted session reply packet file."""
    from anygeometry._planning_executor import unpack
    raw = Path(reply_path).read_bytes()
    packet = json.loads(raw)
    body = unpack(packet, 'session-reply', limit=MEASURE_LIMIT)
    payload = len(packet['payload'].encode('utf-8'))
    envelope = len(raw)
    bound = stage_result['session_reply']
    compact = [len(json.dumps(r, sort_keys=True, separators=(',', ':')).encode('utf-8')) for r in body['worker_reports']]
    return dict(actual_payload=payload, actual_envelope=envelope, bound_payload=bound['payload_utf8_bytes'],
                bound_envelope=bound['envelope_bytes'], actual_compact_reports=compact,
                bound_compact_reports=stage_result['compact_reports_utf8_bytes'],
                payload_ok=payload <= bound['payload_utf8_bytes'], envelope_ok=envelope <= bound['envelope_bytes'],
                compact_ok=len(compact) == len(stage_result['compact_reports_utf8_bytes']) and all(
                    a <= b for a, b in zip(compact, stage_result['compact_reports_utf8_bytes'])))


WORKER_SETUP_GAP = 'worker_setup.expected_environment_packet'


def verdict(result):
    """No flag can upgrade missing evidence. 'passed' needs: every stage within limits, a constructed ledger
    with real reset_for_batch simulation (and determinism when checked), and a PROVEN worker-setup bound."""
    stages = result.get('stages', [])
    missing = missing_conditions(result)
    if result.get('errors'):
        return 'incomplete', missing or list(MISSING_REPLY_FIELDS)       # named conditions survive generic errors
    if not stages or any(not stage['within'] for stage in stages):
        return ('refused' if stages else 'incomplete'), ([] if stages else missing)
    # A supplied actual-vs-proof comparison that fails refuses the verdict (never silently ignored).
    for key in ('compare_capture', 'compare_actual'):
        comparison = result.get(key)
        if comparison is not None and not (comparison.get('all_within') if key == 'compare_capture' else
                                           all(comparison.get(name) is True for name in ('payload_ok', 'envelope_ok', 'compact_ok'))):
            return 'refused', ['comparison.' + key]
    return ('passed' if not missing else 'conditional'), missing


def missing_conditions(result):
    """Every required condition, named. Absent or false evidence is missing (fail closed)."""
    missing = []
    setup = result.get('session_setup') or {}
    within = setup.get('within')
    if type(within) is not dict or set(within) != {'payload', 'envelope', 'walk'} or not all(v is True for v in within.values()):
        missing.append('session_setup.within')
    reset = result.get('reset_simulation') or {}
    if reset.get('pins_match_manifest') is not True:
        missing.append('reset_simulation.pins_match_manifest')
    if not reset.get('ledger_constructed') or reset.get('reset_calls', 0) < 1 or \
            any(value is False for value in (reset.get('determinism') or {}).values()):
        missing.append('diagnostics.ledger_reset_semantics')
    thaw = result.get('thaw_equivalence')
    if type(thaw) is not dict or not all(thaw.get(name) is True for name in ('constructor_equal', 'host_equal', 'common_equal')):
        missing.append('thaw_equivalence')           # --no-thaw-equivalence can therefore never qualify
    coverage = result.get('stage_coverage')
    if not coverage or not coverage.get('captured') or coverage.get('captured') != coverage.get('proven'):
        missing.append('stage_coverage')
    terminal = result.get('terminal')
    if type(terminal) is not dict or terminal.get('within') is not True or terminal.get('errors'):
        missing.append('terminal_wire_proof')
    if (result.get('worker_setup') or {}).get('bound_proven') is not True:
        missing.append(WORKER_SETUP_GAP)       # set only by worker_setup_bound() when the formula fits the limits
    return missing


def worker_setup_bound(base, *, payload_limit=PAYLOAD_LIMIT):
    """Source-derived worker-setup bound from the EXACT setup with expected={} (``base``, a measure_packet row) plus
    the adopted cap on the canonical expected reference (WORKER_EXPECTED_CAP, enforced at producer and child):
      payload  <= base + (cap_bytes - 2)            (the cap counts the full canonical text; '{}' is already in base)
      envelope <= base + 2 * (cap_bytes - 2)        (canonical text is ASCII; JSON string escaping at most doubles it)
      walk     <= base + (cap_nodes - 1)            (the expected root node is already in base)"""
    from anygeometry._planning_session import worker_expected_cap
    WORKER_EXPECTED_CAP = worker_expected_cap()
    extra = WORKER_EXPECTED_CAP['payload_bytes'] - 2
    bound = dict(payload_utf8_bytes=base['payload_utf8_bytes'] + extra,
                 envelope_bytes=base['envelope_bytes'] + 2 * extra,
                 walked_nodes=base['walked_nodes'] + WORKER_EXPECTED_CAP['walk_nodes'] - 1)
    proven = (bound['payload_utf8_bytes'] <= payload_limit and bound['envelope_bytes'] <= 2 * payload_limit + 100000
              and bound['walked_nodes'] <= 100000)
    return dict(bound_proven=proven, cap=dict(WORKER_EXPECTED_CAP), base=base, bound=bound,
                derivation='exact setup with expected={} + cap formulas; cap enforced by check_worker_expected_bound '
                           'at the producer (before workers start) and in the child on reading setup')


def _actual_packet(path, kind):
    from anygeometry._planning_executor import unpack
    raw = Path(path).read_bytes()
    packet = json.loads(raw)
    body = unpack(packet, kind, limit=MEASURE_LIMIT)
    if type(body) is not dict:
        # Every captured protocol packet kind carries an object; a valid-checksum null/scalar/list is a refusal,
        # never a silently skipped packet.
        raise ValueError('%s packet body must be a JSON object, got %s' % (kind, type(body).__name__))
    return body, dict(payload=len(packet['payload'].encode('utf-8')), envelope=len(raw), walk=walk_nodes(body),
                      sha256=packet['sha256'])


def _h(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def terminal_proof(seen, stage_requests, *, session, previous, workers, packet, numerical, infrastructure_native,
                   payload_limit=PAYLOAD_LIMIT, preparation=None):
    """Normal-terminal wire proof: parent finish command, every worker finish command, every worker result.json
    and the planner result.json (incl. _planning_child's process_seconds wrapper), all via the shared builders and
    validated by the real finish verifiers. Scope: the default R11 route (prepared_entries/preparation_sha256 are
    None); any alternative preparation transcript is refused, not claimed. Declined/error/initialization-evidence-
    error replies are explicit failures under the existing runtime packet limits and are not bounded here."""
    from anygeometry._planning_session import (build_session_command, build_worker_command, build_worker_final,
        build_planner_result, build_infrastructure_record, _verify_finish_workers, _verify_worker_resources, reply_bounds)
    from anygeometry._planning_child import with_process_seconds
    from anygeometry._planning_executor import pack
    REPLY_BOUNDS = reply_bounds()
    out = dict(scope='default production route: prepared_entries=None, preparation_sha256=None; alternative preparation '
                     'transcripts are unsupported and refused; declined/error replies are not bounded here',
               errors=[], components=[])
    try:
        if preparation is not None:
            raise ValueError('unsupported alternative preparation transcript')
        body = build_session_command(session, len(seen), previous, 'finish', [], seen=tuple(seen))
        request = pack(body, 'session-command', limit=MEASURE_LIMIT)['sha256']
        out['parent_command'] = measure_packet('finish session-command', body, 'session-command')
        guard = dict(controller_loaded=False, native_adapter_loaded=False, native_core_loaded=False,
                     binary_sha256=infrastructure_native['binary_sha256'])
        commands, results, finals = [], [], []
        for index in range(workers):
            command = build_worker_command(session, len(seen), previous, 'finish', [], index, request)
            commands.append(measure_packet('finish worker-%d command' % index, command, 'session-worker-command'))
            final = with_process_seconds(build_worker_final(
                dict(session_sha256=session, worker=index, worker_setup_sha256='0' * 64,
                     request_sha256=pack(command, 'session-worker-command', limit=MEASURE_LIMIT)['sha256']),
                model_sha256=packet['sha256'], stages=seen, gate_closures=len(seen), blas_threads=worst_blas(),
                pid=REPLY_BOUNDS['pid_max'], resource_guard=guard, value_codec=numerical['value_codec']), FLOAT_WORST)
            finals.append(final)
            results.append(measure_packet('worker-%d result' % index, final, 'worker-result', exact=False))
        _verify_finish_workers(finals, tuple(seen), workers, packet['sha256'])
        for final in finals:
            _verify_worker_resources(final, infrastructure_native, numerical['value_codec'])
        infra = build_infrastructure_record(numerical, 'a' * 64, REPLY_BOUNDS['counter_max'], infrastructure_native)
        infra.update(planner_reference='b' * 64, worker_reference='c' * 64)
        planner = with_process_seconds(build_planner_result(session, request, seen, infra, list(stage_requests), finals,
                                                            None, None), FLOAT_WORST)
        out['worker_commands'], out['worker_results'] = commands, results
        out['planner_result'] = measure_packet('planner result', planner, 'result', exact=False)
        for row in [out['parent_command'], *commands, *results, out['planner_result']]:
            ok = row['payload_utf8_bytes'] <= payload_limit and row['envelope_bytes'] <= 2 * payload_limit + 100000
            out['components'].append(dict(name=row['name'], payload=row['payload_utf8_bytes'],
                                          envelope=row['envelope_bytes'], within=ok))
            if not ok:
                out['errors'].append('limit: %s' % row['name'])
    except BaseException as error:
        out['errors'].append('refused: %s: %s' % (type(error).__name__, error))
        if not isinstance(error, Exception):
            raise
    out['within'] = not out['errors']
    return out


def binding_errors(result, session_dir):
    """Bind the captured packets to THIS proof: exact setup/model/sources/numerical/limits/resource identity and the
    complete session -> request -> worker -> model -> terminal chain (content checks, not checksum validity), plus
    the required terminal packets' sizes. Empty list = bound."""
    from anygeometry._planning_session import bounded_duration, validate_command
    root = Path(session_dir)
    binding = result.get('binding')
    if not binding:
        return ['binding evidence missing from the proof']
    errors = []
    workers = binding['workers']
    kinds = [stage['kind'] for stage in result.get('stages', [])]
    terminal = result.get('terminal') or {}
    def load(path, kind):
        try:
            body, info = _actual_packet(path, kind)
            return body, info
        except Exception as error:
            errors.append('%s/%s unreadable: %s: %s' % (Path(path).parent.name, Path(path).name, type(error).__name__, error))
            return None, None
    def size(name, info, bound):
        if not bound or info['payload'] > bound['payload_utf8_bytes'] or info['envelope'] > bound['envelope_bytes'] \
                or info['walk'] > bound['walked_nodes']:
            errors.append('%s exceeds the proof bound' % name)
    def expect(name, condition):
        if not condition:
            errors.append(name)
    setup, info = load(root / 'session.json', 'session')
    if setup is None:
        return errors
    session = info['sha256']
    expect('session setup packet differs from the proof setup', session == binding['session_sha256'])
    for key, field in (('model_packet_sha256', 'model'), ('sources_sha256', 'sources'), ('numerical_sha256', 'numerical_manifest'),
                       ('limits_sha256', 'limits')):
        expect('setup %s differs from the proof' % field, _h(setup.get(field)) == binding[key])
    expect('setup workers/startup/batch', (setup.get('workers'), setup.get('startup_concurrency'), setup.get('batch_size')) ==
           (workers, binding['startup_concurrency'], None))
    worker_setup = {}
    for k in range(workers):
        body, winfo = load(root / ('worker-%d' % k) / 'session.json', 'session-worker')
        if body is None:
            continue
        worker_setup[k] = winfo['sha256']
        expect('worker-%d setup binding' % k, body.get('session_sha256') == session and body.get('worker') == k and
               _h(body.get('model')) == binding['model_packet_sha256'] and _h(body.get('sources')) == binding['sources_sha256'] and
               body.get('workers') == workers and body.get('batch_size') is None)
        expected = body.get('expected')
        expect('worker-%d expected resource pin' % k, type(expected) is dict and _h(expected.get('resource_pin')) == binding['resource_pin_sha256'])
    previous, worker_previous, stage_requests, finish_request = session, dict(worker_setup), [], None
    for sequence, kind in enumerate(kinds + ['finish']):
        command, cinfo = load(root / ('command-%d.json' % sequence), 'session-command')
        if command is None:
            continue
        csha = cinfo['sha256']
        expect('%s command chain' % kind, command.get('session') == session and command.get('sequence') == sequence and
               command.get('previous') == previous and command.get('stage') == kind)
        try:        # the production grammar on EVERY captured parent command (incl. finish): fields, types, order
            validate_command(command, session=session, sequence=sequence, previous=previous, seen=tuple(kinds[:sequence]))
        except Exception as error:
            errors.append('%s command grammar refused: %s: %s' % (kind, type(error).__name__, error))
        previous = csha
        if kind == 'finish':
            finish_request = csha
            size('finish session-command', cinfo, terminal.get('parent_command'))
        else:
            stage_requests.append(csha)
            reply, _ = load(root / ('reply-%d.json' % sequence), 'session-reply')
            if reply is not None:
                expect('%s reply binding' % kind, reply.get('session_sha256') == session and reply.get('request_sha256') == csha and
                       reply.get('stage') == kind and reply.get('model_sha256') == binding['model_sha256'])
        for k in range(workers):
            folder = root / ('worker-%d' % k)
            wc, winfo = load(folder / ('command-%d.json' % sequence), 'session-worker-command')
            if wc is None:
                continue
            expect('%s worker-%d command chain' % (kind, k), wc.get('session') == session and wc.get('sequence') == sequence and
                   wc.get('stage') == kind and wc.get('worker') == k and wc.get('parent_request') == csha and
                   wc.get('previous') == worker_previous.get(k))
            try:        # production grammar on EVERY captured worker command, including finish
                validate_command(wc, session=session, sequence=sequence, previous=worker_previous.get(k),
                                 seen=tuple(kinds[:sequence]), worker=k)
            except Exception as error:
                errors.append('%s worker-%d command grammar refused: %s: %s' % (kind, k, type(error).__name__, error))
            worker_previous[k] = winfo['sha256']
            if kind == 'finish':
                size('finish worker-%d command' % k, winfo, (terminal.get('worker_commands') or [None] * workers)[k])
                final, finfo = load(folder / 'result.json', 'worker-result')
                if final is not None:
                    size('worker-%d result' % k, finfo, (terminal.get('worker_results') or [None] * workers)[k])
                    expect('worker-%d terminal binding' % k, final.get('status') == 'passed' and final.get('session_sha256') == session and
                           final.get('worker') == k and final.get('worker_setup_sha256') == worker_setup.get(k) and
                           final.get('request_sha256') == winfo['sha256'] and final.get('model_sha256') == binding['model_sha256'] and
                           final.get('stages') == kinds and final.get('gate_closures') == len(kinds) and
                           bounded_duration(final.get('process_seconds')))
            else:
                wr, _ = load(folder / ('reply-%d.json' % sequence), 'session-worker-reply')
                if wr is not None:
                    expect('%s worker-%d reply binding' % (kind, k), wr.get('session_sha256') == session and wr.get('worker') == k and
                           wr.get('worker_setup_sha256') == worker_setup.get(k) and wr.get('request_sha256') == winfo['sha256'] and
                           wr.get('model_sha256') == binding['model_sha256'])
    planner, pinfo = load(root / 'result.json', 'result')
    if planner is not None:
        size('planner result', pinfo, terminal.get('planner_result'))
        expect('planner terminal binding', planner.get('status') == 'passed' and planner.get('session_sha256') == session and
               planner.get('request_sha256') == finish_request and planner.get('stages') == kinds and
               planner.get('stage_requests') == stage_requests and len(planner.get('worker_reports') or []) == workers and
               planner.get('prepared_entries') is None and planner.get('preparation_sha256') is None and
               bounded_duration(planner.get('process_seconds')))
    errors.extend(protocol_errors(result, root))
    return errors


def protocol_errors(result, session_dir):
    """Reuse the REAL production verifiers on the captured packets (no second copy of their rules):
    _verify_worker_resources on every captured worker stage reply and final, _verify_finish_workers on the finals,
    _verify_infrastructure (via an un-initialized OwnedSession carrying the proof's numerical manifest) on every
    stage reply and the planner result (P-only fields must agree across all of them, never equated with H),
    verify_prepared_stage on each captured stage aggregate (rows->results) with the real setup and the captured
    finals; plus content binding of aggregate rows/worker_reports to the individual worker packets, worker commands
    to the parent command's tasks, parent tasks to the proof, planner-echoed finals to the independent finals, and
    refusal of terminal packets outside the normal scope (initialization_evidence_error)."""
    from anygeometry import _planning_session as s
    from anygeometry._planning_executor import reduce_results
    from anygeometry._planning_stages import dispatch
    root = Path(session_dir)
    binding = result.get('binding') or {}
    if any(name not in binding for name in ('resource', 'value_codec', 'numerical_manifest')):
        return ['protocol binding identities (resource/value_codec/numerical_manifest) missing from the proof']
    errors = []
    workers, native, codec = binding['workers'], binding['resource'], binding['value_codec']
    kinds = [stage['kind'] for stage in result.get('stages', [])]
    def load(path, kind):
        try:
            return _actual_packet(path, kind)[0]
        except Exception as error:
            errors.append('protocol: %s/%s unreadable: %s: %s' % (Path(path).parent.name, Path(path).name, type(error).__name__, error))
            return None
    def guarded(name, function, *args):
        try:
            function(*args)
        except Exception as error:
            errors.append('protocol: %s refused: %s: %s' % (name, type(error).__name__, error))
    finals, aggregates = [], {}
    for k in range(workers):
        final = load(root / ('worker-%d' % k) / 'result.json', 'worker-result')
        finals.append(final)
        if final is not None:
            if 'initialization_evidence_error' in final:
                errors.append('protocol: worker-%d terminal carries initialization_evidence_error (outside the normal proof scope)' % k)
            guarded('worker-%d final resource/codec' % k, s._verify_worker_resources, final, native, codec)
    if all(f is not None for f in finals):
        guarded('finish workers', s._verify_finish_workers, finals, tuple(kinds), workers, binding['model_sha256'])
    owned = s.OwnedSession.__new__(s.OwnedSession)             # only _verify_infrastructure's state; no process, no I/O
    owned.numerical_manifest, owned._infrastructure = binding['numerical_manifest'], None
    for sequence, kind in enumerate(kinds):
        command = load(root / ('command-%d.json' % sequence), 'session-command')
        reply = load(root / ('reply-%d.json' % sequence), 'session-reply')
        stage = result['stages'][sequence]
        if command is None or reply is None:
            continue
        tasks = command.get('tasks')
        if _h(tasks) != stage.get('tasks_sha256'):
            errors.append('protocol: %s parent command tasks differ from the proof tasks' % kind)
        guarded('%s infrastructure' % kind, owned._verify_infrastructure, reply)
        parts, _ = dispatch(tasks, workers)
        worker_reports = []
        for k in range(workers):
            wr = load(root / ('worker-%d' % k) / ('reply-%d.json' % sequence), 'session-worker-reply')
            wc = load(root / ('worker-%d' % k) / ('command-%d.json' % sequence), 'session-worker-command')
            if wr is None or wc is None:
                worker_reports.append(None)
                continue
            part = parts[k] if k < len(parts) else []
            if _h(wc.get('tasks')) != _h([[i, tasks[i]] for i in part]):
                errors.append('protocol: %s worker-%d command tasks differ from the parent command dispatch' % (kind, k))
            guarded('%s worker-%d reply resource/codec' % (kind, k), s._verify_worker_resources, wr, native, codec)
            try:
                worker_reports.append((wr, s.compact_worker_report(wr, part, payload_limit=MEASURE_LIMIT)))
            except Exception as error:
                worker_reports.append(None)
                errors.append('protocol: %s worker-%d reply refused: %s' % (kind, k, error))
        echoed = reply.get('worker_reports') or []
        for k, pair in enumerate(worker_reports):
            if pair is not None and (k >= len(echoed) or _h(echoed[k]) != _h(pair[1])):
                errors.append('protocol: %s aggregate worker report %d differs from the individual worker reply' % (kind, k))
        if all(pair is not None for pair in worker_reports):
            if _h(reply.get('rows')) != _h([row for wr, _ in worker_reports for row in wr['rows']]):
                errors.append('protocol: %s aggregate rows differ from the individual worker rows' % kind)
        if all(f is not None for f in finals):
            try:
                body = dict(reply)
                results = reduce_results(body.pop('rows'), len(tasks))
                setup = dict(workers=workers, model={'sha256': binding['model_sha256']},
                             startup_concurrency=binding['startup_concurrency'], resource_identity=native, value_codec=codec)
                s.verify_prepared_stage(dict(body, results=list(results)), tasks, setup, finals, payload_limit=MEASURE_LIMIT)
            except Exception as error:
                errors.append('protocol: %s stage aggregate refused: %s: %s' % (kind, type(error).__name__, error))
    planner = load(root / 'result.json', 'result')
    if planner is not None:
        if 'initialization_evidence_error' in planner:
            errors.append('protocol: planner terminal carries initialization_evidence_error (outside the normal proof scope)')
        guarded('planner infrastructure', owned._verify_infrastructure, planner)
        echoed = planner.get('worker_reports') or []
        for k, final in enumerate(finals):
            if final is not None and (k >= len(echoed) or _h(echoed[k]) != _h(final)):
                errors.append('protocol: planner worker_reports[%d] differs from the independent worker final' % k)
    return errors


def compare_capture(result, session_dir):
    """Size/rows/diagnostics comparison AND complete identity/chain/terminal binding of a captured real run."""
    comparison = _compare_capture_sizes(result, session_dir)
    errors = binding_errors(result, session_dir)
    comparison['binding'] = dict(errors=errors, ok=not errors)
    comparison['all_within'] = bool(comparison['all_within'] and not errors)
    return comparison


def _compare_capture_sizes(result, session_dir):
    """Compare ACTUAL raw packets of a real small run (session dir copy) with the proof bounds/simulation.

    Rows: setup, per-stage session command/reply, per-worker command/reply, plus an informational
    worker-setup row (no bound is claimed for it). Equality is required for result rows; diagnostics are reported
    against the simulated payload."""
    root = Path(session_dir)
    rows, ok = [], True
    def check(name, kind, path, bound, **extra):
        """One REQUIRED packet: missing/unreadable/over-bound => within False (and the whole comparison fails)."""
        nonlocal ok
        path = Path(path)
        if not path.exists() or not bound:
            ok = False
            rows.append(dict(name=name, missing=not path.exists(), actual=None, bound=None, within=False))
            return None
        try:
            body, actual = _actual_packet(path, kind)
        except Exception as error:
            ok = False
            rows.append(dict(name=name, missing=False, actual=None, bound=None, within=False,
                             error=type(error).__name__ + ': ' + str(error)))
            return None
        good = actual['payload'] <= bound['payload_utf8_bytes'] and actual['envelope'] <= bound['envelope_bytes'] \
            and actual['walk'] <= bound['walked_nodes']
        ok &= good
        rows.append(dict(name=name, missing=False, actual=actual, bound=dict(payload=bound['payload_utf8_bytes'],
                         envelope=bound['envelope_bytes'], walk=bound['walked_nodes']), within=good, **extra))
        return body
    workers = result.get('workers', 8)
    setup_body = check('session setup', 'session', root / 'session.json', result.get('session_setup'))
    if setup_body is not None and result.get('session_setup_field_bytes'):
        actual_fields = {k: len(json.dumps(v, sort_keys=True, separators=(',', ':')).encode()) for k, v in setup_body.items()}
        rows[-1].update(field_bytes_actual=actual_fields, field_bytes_simulated=result['session_setup_field_bytes'],
                        differing_fields=sorted(k for k in set(actual_fields) | set(result['session_setup_field_bytes'])
                                                if actual_fields.get(k) != result['session_setup_field_bytes'].get(k)))
    worker_setup = (result.get('worker_setup') or {}).get('bound')
    for index in range(workers):
        check('worker-%d setup' % index, 'session-worker', root / ('worker-%d' % index) / 'session.json', worker_setup)
    for sequence, stage in enumerate(result.get('stages', [])):
        kind = stage['kind']
        check(kind + ' session-command', 'session-command', root / ('command-%d.json' % sequence), stage['parent_command'])
        reply = check(kind + ' session-reply', 'session-reply', root / ('reply-%d.json' % sequence), stage['session_reply'])
        if reply is not None:
            rows_equal = hashlib.sha256(json.dumps(reply['rows'], sort_keys=True, separators=(',', ':')).encode()).hexdigest() \
                == stage.get('session_rows_sha256')
            rows[-1].update(rows_equal_simulated=rows_equal)
            ok &= rows_equal
        for index in range(workers):
            component = stage['worker_components'][index]
            folder = root / ('worker-%d' % index)
            check('%s worker-%d command' % (kind, index), 'session-worker-command',
                  folder / ('command-%d.json' % sequence), stage['worker_commands'][index])
            body = check('%s worker-%d reply' % (kind, index), 'session-worker-reply',
                         folder / ('reply-%d.json' % sequence), stage['worker_replies'][index])
            if body is None:
                continue
            rows_text = json.dumps(body['rows'], sort_keys=True, separators=(',', ':'))
            diag = len(json.dumps(body['diagnostics'], sort_keys=True, separators=(',', ':')).encode())
            stable = json.dumps([[row[0], stable_diagnostics(row[1])] for row in body['diagnostics']], sort_keys=True,
                                separators=(',', ':'))
            equal = hashlib.sha256(rows_text.encode()).hexdigest() == component['rows_sha256']
            stable_equal = hashlib.sha256(stable.encode()).hexdigest() == component['diagnostics_stable_sha256']
            within = diag <= component['diagnostics_bytes']
            rows[-1].update(rows_equal_simulated=equal, diagnostics_stable_equal=stable_equal,
                            diagnostics_actual_bytes=diag, diagnostics_simulated_bytes=component['diagnostics_bytes'],
                            diagnostics_within_simulated=within)
            ok &= equal and stable_equal and within
    return dict(rows=rows, all_within=bool(ok) and bool(rows) and all(r['within'] for r in rows))


def build_probe_ledger(context, resource_native):
    """Ledger for the probe's OWN WorkerModel, constructed like session_worker (role='worker', resource pin).

    The worker role requires the host controller/native modules to be absent, which cannot hold in this host
    process; the exact invariant is recorded and the SAME PackageWorkerLedger API is built without a role
    (identical registered stores/resets, no worker-absence claim). Never a native-worker qualification."""
    from anygeometry import _planning_ledger as ledger_api, _planning_qualification as q
    from anygeometry._planning_qualification import QualificationDecline
    try:
        return ledger_api.package_worker_ledger(context.model, resource_pin=q.resource_file_pin(resource_native),
                                                role='worker'), 'worker', None
    except QualificationDecline as error:
        return ledger_api.package_worker_ledger(context.model), 'probe-process-no-role', str(error)


def run_probe(model, operands, output, *, document_sha256=None, thaw_equivalence=True, workers=8,
              payload_limit=PAYLOAD_LIMIT, compare_reply=None, compare_stage=None, compare_dir=None):
    if bool(compare_reply) != bool(compare_stage):
        raise ValueError('compare_reply and compare_stage must be given together (no silently skipped comparison)')
    from anygeometry import batch_intersections as bi
    from anygeometry._planning_executor import freeze_model, thaw_model, source_files
    from anygeometry._planning_manifest import capture_loaded_manifest, prepare_interfaces
    from anygeometry._planning_code import InfrastructurePin
    from anygeometry._planning_values import value_codec_contract
    from anygeometry._planning_session import build_session_setup
    from anygeometry._planning_stages import WorkerModel
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    limits = dict(payload_limit=4_000_000, walk_limit=100000)
    result = dict(schema='R11-probe-v2', single_process=True, children_spawned=0, ceilings=CEILINGS,
                  document_sha256=document_sha256, stages=[], errors=[], scale_aware=True,
                  component_classes=dict(exact=['session setup', 'session/worker commands', 'result rows', 'diagnostics (probe-process semantics)'],
                                         upper_bound=['floats', 'pids', 'counters', 'BLAS info (declared bound)', 'infrastructure record hashes/nodes']))
    path = output / 'r11-probe.json'

    def save():
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding='utf-8')
        temporary.replace(path)

    def step(name, function):
        started = time.perf_counter()
        try:
            return function()
        except BaseException as error:
            result['errors'].append(dict(step=name, error=type(error).__name__ + ': ' + str(error)))
            save()
            raise
        finally:
            result.setdefault('seconds', {})[name] = time.perf_counter() - started
            save()
    try:
        packet = step('freeze', lambda: freeze_model(model))
        result['model_packet'] = dict(sha256=packet['sha256'], payload_utf8_bytes=len(packet['payload'].encode('utf-8')))
        prepare_interfaces()
        manifest = step('capture_manifest', lambda: capture_loaded_manifest(model=model))
        infrastructure = InfrastructurePin('host')
        authority = [t for t in manifest.roles['constructor']._last_tokens if len(t) > 2 and t[1] == 'request-document-authority']
        host_authority = [t for t in manifest.roles['host']._last_tokens if len(t) > 2 and t[1] == 'request-document-authority']
        result['manifest'] = dict(wire_sha256=manifest.wire['sha256'], common=manifest.callables.digest,
            constructor=manifest.roles['constructor'].digest, host=manifest.roles['host'].digest,
            infrastructure_host=infrastructure.digest, callable_nodes=manifest.nodes,
            common_graph=manifest.callables.graph, constructor_graph=manifest.roles['constructor'].graph,
            host_graph=manifest.roles['host'].graph,
            authority_constructor=[list(t[2]) if type(t[2]) is tuple else t[2] for t in authority],
            authority_host=[list(t[2]) if type(t[2]) is tuple else t[2] for t in host_authority],
            authority_nodes=[t[2][2] for t in authority if type(t[2]) is tuple])
        if thaw_equivalence:
            def equivalence():
                thawed = thaw_model(packet, **limits)
                second = capture_loaded_manifest(model=thawed)
                return dict(constructor_equal=second.roles['constructor'].digest == manifest.roles['constructor'].digest,
                            host_equal=second.roles['host'].digest == manifest.roles['host'].digest,
                            common_equal=second.callables.digest == manifest.callables.digest,
                            thawed_constructor=second.roles['constructor'].digest, thawed_host=second.roles['host'].digest)
            result['thaw_equivalence'] = step('thaw_equivalence', equivalence)
        numerical = dict(value_codec=value_codec_contract(), common=manifest.callables.digest,
            constructor=manifest.roles['constructor'].digest, host=manifest.roles['host'].digest,
            infrastructure_host=infrastructure.digest, resource=infrastructure.native)
        sources = source_files()
        setup = build_session_setup(packet, sources, workers=workers, batch_size=None, limits=limits,
                                    startup_concurrency=workers, numerical_manifest=numerical)
        result['session_setup'] = measure_packet('session setup', setup, 'session', payload_limit=payload_limit)
        result['source_file_count'] = len(sources)
        session = result['session_setup']['sha256']
        result['workers'] = workers
        result['session_setup_field_bytes'] = {k: len(json.dumps(v, sort_keys=True, separators=(',', ':')).encode())
                                               for k, v in setup.items()}
        from anygeometry._planning_session import build_worker_setup
        worker_base = measure_packet('worker setup (expected={})',
                                     build_worker_setup(session, 0, packet, sources, {}, workers, None), 'session-worker',
                                     payload_limit=payload_limit)
        result['worker_setup'] = worker_setup_bound(worker_base, payload_limit=payload_limit)
        context = step('worker_model', lambda: WorkerModel(packet, value_codec='v2', **limits))
        ledger, ledger_role, ledger_error = step('probe_ledger', lambda: build_probe_ledger(context, infrastructure.native))
        recorder = Recorder(context, limits, workers, ledger=ledger)
        result['reset_simulation'] = dict(ledger_constructed=True, role=ledger_role, role_invariant=ledger_error,
            live_code_pins=ledger.live_code_pins(),
            pins_match_manifest=ledger.live_code_pins() == dict(common=numerical['common'], constructor=numerical['constructor']),
            ledger_nodes=ledger.manifest.nodes, registered_stores=len(ledger.resets),
            source_sha256=hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
            scope='dedicated probe process only; registered declined caches; NOT native-worker qualification',
            partition_equivalence='one shared context; each task preceded by real reset_for_batch and a fresh WorkerModel.view()')
        policy = bi.IntersectionBatchPolicy()
        effective = bi._policy(policy)
        mode = bi.resolve_planning_mode(effective)
        seam = bi.resolve_seam_policy(effective)
        step('stage_capture', lambda: bi._plan_intersections(
            model, operands, policy=policy, _planning_mode=mode, _seam_policy=seam, _stages=recorder))
        seen, previous = (), session
        from anygeometry._planning_executor import pack
        for row in recorder.stages:
            if not row['tasks']:
                continue
            stage = stage_proof(row, session=session, sequence=len(seen), previous=previous, seen=seen, workers=workers,
                packet=packet, numerical=numerical, infrastructure_native=infrastructure.native, payload_limit=payload_limit)
            stage['capture_seconds'] = row['seconds']
            result['stages'].append(stage)
            if stage['errors']:
                result['errors'].extend(dict(step=row['kind'], error=text) for text in stage['errors']
                                        if not text.startswith('limit:'))
            if 'parent_command' in stage:
                previous = stage['parent_command']['sha256']
            seen += (row['kind'],)
            save()
        result['reset_simulation'].update(reset_calls=recorder.reset_calls, reset_total=recorder.reset_total,
                                          determinism=dict(recorder.determinism))
        result['stage_coverage'] = dict(captured=[r['kind'] for r in recorder.stages if r['tasks']],
                                        proven=[s['kind'] for s in result['stages'] if s.get('within')])
        result['terminal'] = terminal_proof(tuple(s['kind'] for s in result['stages']),
            [s['parent_command']['sha256'] for s in result['stages'] if 'parent_command' in s], session=session,
            previous=previous, workers=workers, packet=packet, numerical=numerical,
            infrastructure_native=infrastructure.native, payload_limit=payload_limit)
        from anygeometry import _planning_qualification as qualification
        result['binding'] = dict(session_sha256=session, workers=workers, startup_concurrency=workers,
            model_sha256=packet['sha256'], model_packet_sha256=_h(packet), sources_sha256=_h(sources),
            numerical_sha256=_h(numerical), limits_sha256=_h(limits), resource=infrastructure.native,
            value_codec=numerical['value_codec'], numerical_manifest=numerical,
            resource_pin_sha256=_h(qualification.resource_file_pin(infrastructure.native)))
        if result['terminal']['errors']:
            result['errors'].extend(dict(step='terminal', error=text) for text in result['terminal']['errors']
                                    if not text.startswith('limit:'))
        if compare_reply and compare_stage:
            match = [s for s in result['stages'] if s['kind'] == compare_stage]
            result['compare_actual'] = compare_actual(match[0], compare_reply) if match else dict(error='stage absent')
        if compare_dir:
            result['compare_capture'] = compare_capture(result, compare_dir)
        result['feasibility'], result['missing'] = verdict(result)
        result['complete'] = result['feasibility'] == 'passed'
    except BaseException as error:
        result['complete'] = False
        result['feasibility'] = 'incomplete'
        result['missing'] = list(MISSING_REPLY_FIELDS)
        if not result['errors']:
            result['errors'].append(dict(step='probe', error=type(error).__name__ + ': ' + str(error)))
        save()
        if not isinstance(error, Exception):
            raise
    save()
    return result


def extract_geometry(data):
    """Geometry serialization document from a plain document OR an .anyfem project JSON.

    A plain document has a top-level ``checksum``; a project carries it under ``geometry``. Both present,
    neither present, or a malformed ``geometry`` is refused (never guessed)."""
    if type(data) is not dict:
        raise SystemExit('model input must be a JSON object')
    direct, project = 'checksum' in data, 'geometry' in data
    if direct and project:
        raise SystemExit('ambiguous model input: both a document checksum and a project geometry are present')
    if project:
        geometry = data['geometry']
        if type(geometry) is not dict or 'checksum' not in geometry:
            raise SystemExit('malformed project geometry: expected a serialization document with a checksum')
        return geometry
    if direct:
        return data
    raise SystemExit('model input has neither a serialization checksum nor a project geometry')


def load_model(model_path, sha256):
    """The hash binds the WHOLE input file; only the explicitly identified geometry is deserialized."""
    data = Path(model_path).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != sha256.lower():
        raise SystemExit('model sha256 mismatch: expected %s actual %s' % (sha256, actual))
    from anygeometry.serialization import from_dict
    try:
        parsed = json.loads(data)
    except ValueError as error:
        raise SystemExit('model input is not valid JSON: %s' % error)
    return from_dict(extract_geometry(parsed)), actual


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--model')
    parser.add_argument('--sha256')
    parser.add_argument('--output', required=True)
    parser.add_argument('--fixture-small', action='store_true')
    parser.add_argument('--no-thaw-equivalence', action='store_true')
    parser.add_argument('--compare-dir', help='copy of a REAL small run session directory (see r11_profile --capture-packets)')
    parser.add_argument('--compare-reply')
    parser.add_argument('--compare-stage', choices=('pair', 'face'))
    args = parser.parse_args(argv)
    if bool(args.compare_reply) != bool(args.compare_stage):
        parser.error('--compare-reply and --compare-stage must be given together')       # before any expensive work
    if args.fixture_small:
        # Same construction as r11_profile --fixture-small (serialize, then deserialize) so the model packet matches.
        from anygeometry._planning_canaries import _fixture
        from anygeometry.serialization import from_dict, to_dict
        model, _ = _fixture()
        model = from_dict(to_dict(model))
        digest = None
    else:
        if not args.model or not args.sha256:
            parser.error('--model and --sha256 are required without --fixture-small')
        model, digest = load_model(args.model, args.sha256)
    result = run_probe(model, tuple(model.faces), args.output, document_sha256=digest,
                       thaw_equivalence=not args.no_thaw_equivalence,
                       compare_reply=args.compare_reply, compare_stage=args.compare_stage, compare_dir=args.compare_dir)
    print(json.dumps(dict(complete=result.get('complete'), feasibility=result.get('feasibility'),
                          missing=result.get('missing'), errors=result['errors'],
                          output=str(Path(args.output) / 'r11-probe.json'))))
    # 0 only when every field proof passed; conditional/refused/incomplete are non-zero.
    return 0 if result.get('complete') else 1


if __name__ == '__main__':
    sys.exit(main())
