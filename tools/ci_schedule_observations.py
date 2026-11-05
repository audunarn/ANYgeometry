"""Retain censored CI observations without inventing completed durations.

JUnit and progress-protocol clocks remain separate. Both can supply scheduling
cost policy; neither establishes process wall, complete suite cost, or fit.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def merge(table, records, provenance, *, semantics):
    from ci_isolated_tests import validate_nodes
    if semantics not in ('junit-testcase', 'progress-protocol'):
        raise ValueError('unknown observation timing semantics')
    if not provenance.get('run') or not provenance.get('head_sha') or not provenance.get('source_sha256'):
        raise ValueError('observation provenance incomplete')
    result = copy.deepcopy(table)
    scheduling = result.setdefault('scheduling', {})
    key = 'tests' if semantics == 'junit-testcase' else 'protocol_tests'
    tests = scheduling.setdefault(key, {})
    count = 0
    seen = set()
    for record in records:
        if record['outcome'] != 'passed':
            continue
        node = record['node']
        validate_nodes([node])
        identity = (node, record['runtime'], record['artifact'])
        if identity in seen:
            raise ValueError('duplicate observation from same runtime/artifact')
        seen.add(identity)
        seconds = record['seconds']
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError('observation seconds must be finite and nonnegative')
        bucket = tests.setdefault(node, {}).setdefault(record['runtime'],
            dict(minimum_seconds=seconds, maximum_seconds=seconds, observations=0, artifacts=[]))
        if record['artifact'] in bucket['artifacts']:
            raise ValueError('observation artifact already retained; do not double-count refresh')
        bucket['minimum_seconds'] = min(bucket['minimum_seconds'], seconds)
        bucket['maximum_seconds'] = max(bucket['maximum_seconds'], seconds)
        bucket['observations'] += 1
        bucket['artifacts'].append(record['artifact'])
        bucket['artifacts'].sort()
        bucket.setdefault('retained_seconds', []).append(seconds)
        count += 1
    scheduling.setdefault('additional_provenance', []).append(dict(provenance, semantics=semantics, passed_observations=count))
    scheduling['protocol_timing_semantics'] = 'phase progress start-to-complete-end elapsed; NOT JUnit or process-wall; unfinished cases omitted and censored'
    scheduling['censoring'] = 'historical first-failure prefixes and cancelled prefixes; missing outcomes remain unmeasured'
    return result


def retained_records(summary, junit, runtime='windows-latest/python3.13'):
    from ci_durations import node_id
    data = json.loads(Path(summary).read_text(encoding='utf-8'))
    protocol = []
    for shard, item in data.items():
        for cohort, group in item['groups'].items():
            # A cohort that never started has assignment/cost metadata only.
            # Absence of completion evidence supplies no measured duration.
            if 'completed' not in group:
                if set(group) == {'assigned', 'estimated_seconds'}:
                    continue
                raise ValueError('completion status absent from observed cohort')
            rows = group.get('completed_rows', [])
            if len(rows) != group['completed']:
                raise ValueError('summary completed count differs from retained rows')
            for row in rows:
                protocol.append(dict(node=row['nodeid'], seconds=row['protocol_elapsed_seconds'],
                    outcome=row['outcome'], runtime=runtime,
                    artifact='run42048877201-shard'+shard+'-'+cohort+'-progress'))
    junit_rows = []
    for case in ET.parse(junit).iter('testcase'):
        outcome = 'passed' if not any(case.find(tag) is not None for tag in ('failure', 'error', 'skipped')) else 'not-passed'
        junit_rows.append(dict(node=node_id(case.get('classname', ''), case.get('name', '')),
            seconds=float(case.get('time', 0)), outcome=outcome, runtime=runtime,
            artifact='run42048877201-shard10-core-junit'))
    return protocol, junit_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', type=Path, required=True)
    parser.add_argument('--progress-summary', type=Path, required=True)
    parser.add_argument('--junit', type=Path, required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--head-sha', required=True)
    args = parser.parse_args(argv)
    # The retained source schema currently binds these specific artifacts.
    if args.run != '42048877201' or args.head_sha != '2aae7ce6fd8321264bcffd69dcbc23d23c0825cc':
        raise ValueError('retained timeout evidence source identity differs')
    table = json.loads(args.table.read_text(encoding='utf-8'))
    progress, junit = retained_records(args.progress_summary, args.junit)
    for records, path, semantics in ((progress, args.progress_summary, 'progress-protocol'), (junit, args.junit, 'junit-testcase')):
        provenance = dict(run=args.run, head_sha=args.head_sha, source_path=path.as_posix(),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            applicability='Historical scheduling only; cancelled run, incomplete suite, no current-source qualification or fit bound')
        table = merge(table, records, provenance, semantics=semantics)
    args.table.write_text(json.dumps(table, indent=1, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps(dict(retained_protocol=len(progress), retained_junit=len(junit), release_acceptance=False)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
