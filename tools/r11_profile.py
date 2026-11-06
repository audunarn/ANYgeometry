"""R11 bounded subset profile: selected operands only, never whole-model planning.

Usage:
  python tools/r11_profile.py --fixture-json F.json --case elbow_r7 --output DIR --mode serial
  python tools/r11_profile.py --fixture-json F.json --case elbow_sector60 --output DIR --mode forced-parallel --thread-limit 1

Fixture JSON: {"document": <serialization document>, "subsets": {"<case>": [face ids...]}}
(``subsets`` may also be a list of {"case"|"name": str, "faces"|"operands": [...]}).

--mode serial           sets ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING=0 in THIS process only.
--mode forced-parallel  runs the real production controller with the PRIVATE
                        ``_planning_controller._forced_route`` context: only the cost
                        gate is skipped; H/P/W integrity, storage, memory lease, the
                        100 s allowance and retirement are unchanged. The receipt
                        records ``forced_selector``; such a run NEVER satisfies the
                        default-selector acceptance.
--thread-limit N        threadpoolctl limit for THIS process (host/serial kernels).
                        Child workers keep their own admitted thread profile.
The caller (not this tool) owns the wall-clock budget; nothing is renewed here.
"""
import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ENV = 'ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING'


def select_case(data, case):
    subsets = data.get('subsets')
    if type(subsets) is dict:
        faces = subsets.get(case)
    elif type(subsets) is list:
        faces = next((row.get('faces', row.get('operands')) for row in subsets
                      if type(row) is dict and case in (row.get('case'), row.get('name'))), None)
    else:
        faces = None
    if type(faces) is not list or not faces or any(type(f) is not int or f < 1 for f in faces):
        raise SystemExit('fixture has no valid subset for case %r' % case)
    return tuple(sorted(set(faces)))


def timing_summary(receipt):
    """Sum per-worker fixed-size task_timing aggregates (maxima are maxed, not summed)."""
    totals = dict(tasks=0)
    for stage in receipt.get('costs', {}).get('infrastructure', []):
        for worker in stage.get('workers', []):
            row = worker.get('task_timing')
            if not row:
                continue
            totals['tasks'] += row.get('count', 0)
            for key, value in row.items():
                if key == 'count':
                    continue
                if key.startswith('max_'):
                    totals[key] = max(totals.get(key, 0.), value)
                else:
                    totals[key] = totals.get(key, 0.) + value
    return totals


MATCHED_PHASES = ('intersection_support_and_clipping_time_ns', 'intersection_arrangements_time_ns')


def deterministic_diagnostics(diagnostics):
    """Counters comparable across modes: drop wall-time and cache-work counters."""
    kept, excluded = {}, []
    for name, value in sorted(dict(diagnostics).items()):
        if name.endswith('_time_ns') or 'cache' in name:
            excluded.append(name)
        else:
            kept[name] = value
    return kept, excluded


def _plain(value):
    if isinstance(value, (bytes, bytearray)):
        return value.hex()
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    return repr(value)


def export_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, default=_plain, indent=None), encoding='utf-8')
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classify(mode, receipt):
    """Actual disposition, never presenting a declined forced run as completed parallel."""
    if mode != 'forced-parallel':
        return dict(disposition='serial-mode', complete=True)
    if not receipt:
        return dict(disposition='no-receipt', complete=False, reason='no parallel_planning receipt')
    disposition = receipt.get('disposition')
    row = dict(disposition=disposition, retired=receipt.get('retired'), reason=receipt.get('reason'),
               accepted=[dict(stage=s.get('stage'), accepted=s.get('accepted'), total=s.get('total'),
                              accepted_weight=s.get('accepted_weight'), total_weight=s.get('total_weight'))
                         for s in receipt.get('stages', [])],
               forced_selector=receipt.get('forced_selector'))
    row['complete'] = bool(disposition == 'parallel' and receipt.get('retired') is True and receipt.get('stages'))
    return row


@contextlib.contextmanager
def capture_session_packets(destination):
    """Copy the raw session packets (*.json, incl. worker-K/) of a REAL run into a dedicated read-only diagnostic
    directory just before the controller removes its scratch ``work`` folder. Tool-level, restored on exit."""
    import shutil
    import stat
    real = shutil.rmtree
    destination = Path(destination)

    def rmtree(path, *args, **kwargs):
        source = Path(path) / 'session'
        if Path(path).name == 'work' and source.is_dir():
            manifest = {}
            for item in sorted(source.rglob('*.json')):
                relative = item.relative_to(source)
                target = destination / 'session' / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                raw = item.read_bytes()
                target.write_bytes(raw)
                target.chmod(stat.S_IREAD)
                manifest[str(relative)] = hashlib.sha256(raw).hexdigest()
            (destination / 'capture-manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
        return real(path, *args, **kwargs)
    shutil.rmtree = rmtree
    try:
        yield
    finally:
        shutil.rmtree = real


def small_fixture():
    """The tiny canary model as fixture data (document + one subset with all faces)."""
    from anygeometry._planning_canaries import _fixture
    from anygeometry.serialization import to_dict
    model, _ = _fixture()
    return dict(document=to_dict(model), subsets=dict(small=sorted(model.faces)))


def profile(fixture, case, output, mode, thread_limit=None, expect_faces=None, capture_packets=None):
    from anygeometry import batch_intersections as bi
    from anygeometry import _planning_controller as controller
    from anygeometry.runtime_diagnostics import collect_runtime_diagnostics, collect_runtime_records
    from anygeometry.serialization import from_dict, to_dict
    from threadpoolctl import threadpool_info, threadpool_limits
    if fixture is None:
        data = small_fixture()
        fixture_sha256 = hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()
    else:
        data = json.loads(Path(fixture).read_text(encoding='utf-8'))
        fixture_sha256 = hashlib.sha256(Path(fixture).read_bytes()).hexdigest()
    operands = select_case(data, case)
    if expect_faces is not None and len(operands) != expect_faces:
        raise SystemExit('selected face count %d != expected %d' % (len(operands), expect_faces))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = dict(schema='R11-profile-v1', case=case, mode=mode, selected_faces=len(operands),
                  fixture_sha256=fixture_sha256, capture_packets=str(capture_packets) if capture_packets else None,
                  thread_limit=thread_limit, forced_selector=None, whole_model_planning=False)
    path = output / ('r11-profile-%s-%s.json' % (case, mode))

    def save():
        path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding='utf-8')
    try:
        started = time.perf_counter()
        model = from_dict(data['document'])
        result['load_seconds'] = time.perf_counter() - started
        result['container_faces'] = len(model.faces)
        policy = bi.IntersectionBatchPolicy()
        previous = os.environ.get(ENV)
        limiter = threadpool_limits(limits=thread_limit) if thread_limit else contextlib.nullcontext()
        selector = contextlib.nullcontext()
        if mode == 'serial':
            os.environ[ENV] = '0'
        elif mode == 'forced-parallel':
            os.environ[ENV] = '1'
            result['forced_selector'] = 'r11-profile:' + case
            selector = controller._forced_route(result['forced_selector'])
        else:
            raise SystemExit('unknown mode')
        try:
            capture = capture_session_packets(capture_packets) if capture_packets and mode == 'forced-parallel' else contextlib.nullcontext()
            with limiter, selector, capture, collect_runtime_diagnostics() as diagnostics, collect_runtime_records() as records:
                result['threadpool_info'] = threadpool_info()
                started = time.perf_counter()
                plan = bi.plan_intersections(model, operands, policy=policy)
                result['plan_seconds'] = time.perf_counter() - started
                started = time.perf_counter()
                applied = bi.apply_intersections(model, plan, policy=policy)
                result['apply_seconds'] = time.perf_counter() - started
        finally:
            if previous is None:
                os.environ.pop(ENV, None)
            else:
                os.environ[ENV] = previous
        from anygeometry._planning_code import exact_numerical_record
        from anygeometry.model import GeometryModel
        result['plan_checksum'] = plan.content_checksum
        result['source_checksum'] = plan.source_checksum
        # Full canonical exports for exact control comparison (checksums alone are not enough).
        result['plan_json_sha256'] = export_json(output / 'plan.json', exact_numerical_record(plan))
        final = applied if isinstance(applied, GeometryModel) else model
        document = to_dict(final)
        result['application_json_sha256'] = export_json(output / 'application.json', document)
        result['document_checksum_after'] = document['checksum']['value']
        result['applied_type'] = type(applied).__name__
        kept, excluded = deterministic_diagnostics(diagnostics)
        result['diagnostics'] = kept
        result['diagnostics_excluded'] = excluded
        result['diagnostics_sha256'] = export_json(output / 'diagnostics.json', kept)
        result['matched_host_phases_seconds'] = {name: diagnostics.get(name, 0) / 1e9 for name in MATCHED_PHASES}
        receipts = records.get('parallel_planning', [])
        receipt = receipts[-1] if receipts else None
        result['receipt'] = receipt
        result.update(classify(mode, receipt))
        if mode == 'forced-parallel' and receipt:
            result['task_timing_totals'] = timing_summary(receipt)
        if mode == 'forced-parallel' and not result['complete']:
            result['error'] = 'forced parallel did not complete: %s (%s)' % (result.get('disposition'), result.get('reason'))
        else:
            result['complete'] = 'error' not in result
    except BaseException as error:
        result['complete'] = False
        result['error'] = type(error).__name__ + ': ' + str(error)
        save()
        if not isinstance(error, Exception):
            raise
    save()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--fixture-json')
    parser.add_argument('--fixture-small', action='store_true', help='the tiny canary model (case "small")')
    parser.add_argument('--case', required=True, choices=('elbow_r7', 'elbow_sector60', 'small'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', required=True, choices=('serial', 'forced-parallel'))
    parser.add_argument('--thread-limit', type=int)
    parser.add_argument('--expect-faces', type=int)
    parser.add_argument('--capture-packets', help='dedicated directory for raw session packets of a forced-parallel run')
    args = parser.parse_args(argv)
    if bool(args.fixture_json) == bool(args.fixture_small):
        parser.error('exactly one of --fixture-json / --fixture-small is required')
    result = profile(None if args.fixture_small else args.fixture_json, args.case, args.output, args.mode,
                     args.thread_limit, args.expect_faces, args.capture_packets)
    print(json.dumps({k: result.get(k) for k in ('complete', 'error', 'plan_seconds', 'apply_seconds', 'selected_faces')}))
    return 0 if result.get('complete') else 1


if __name__ == '__main__':
    sys.exit(main())
