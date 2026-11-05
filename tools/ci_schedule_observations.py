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


def artifact_records(folder, *, runtime, artifact_id, runtime_verification=None):
    """Read genuine phase journals/JUnit; keep censored lower bounds separate."""
    import re
    from ci_durations import node_id
    from ci_isolated_tests import validate_nodes
    folder = Path(folder)
    match = re.fullmatch(r'(windows-latest|ubuntu-latest|macos-15|macos-15-intel)/python(3\.(?:11|12|13|14))/(AMD64|x86_64|arm64)', runtime)
    architectures = {'windows-latest': 'AMD64', 'ubuntu-latest': 'x86_64', 'macos-15': 'arm64', 'macos-15-intel': 'x86_64'}
    if not match or match[3] != architectures[match[1]] or type(artifact_id) is not int or artifact_id <= 0:
        raise ValueError('explicit artifact runtime/architecture/identity invalid')
    runtime_path = folder/'arc-runtime.json'
    if runtime_path.exists():
        evidence = json.loads(runtime_path.read_text(encoding='utf-8'))
        verification = dict(kind='observed-python-runtime-json', python_version=evidence['python_version'], runtime=runtime)
    else:
        verification = runtime_verification or {}
        if (verification.get('kind') != 'declared-and-ci-verified' or verification.get('runtime') != runtime
                or not verification.get('job_id') or len(verification.get('log_sha256','')) != 64
                or len(verification.get('base_workflow_sha256','')) != 64):
            raise ValueError('runtime JSON absent; CI-verified declared runtime proof required')
    if not verification['python_version'].startswith(match[2]+'.'):
        raise ValueError('runtime Python evidence differs from explicit bucket')
    result = dict(protocol=[], junit=[], censored=[], outcomes={}, source_sha256={}, runtime_verification=verification)
    def retain(path):
        result['source_sha256'][path.relative_to(folder).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if runtime_path.exists():retain(runtime_path)
    for mode in ('core', 'oracle'):
        phase = folder/'ci-isolation'
        progress, config = phase/(mode+'-progress.jsonl'), phase/(mode+'-config.json')
        if not progress.exists():
            continue  # Never-started phase has no measured duration.
        retain(progress); retain(config)
        binding = hashlib.sha256(config.read_bytes()).hexdigest()
        row_config = json.loads(config.read_text(encoding='utf-8'))
        nodes = validate_nodes(row_config['nodes'])
        if row_config['mode'] != mode:
            raise ValueError('phase config mode differs')
        indices = {node: index for index, node in enumerate(nodes)}
        active = None; started = None; reports = {}; finished = set(); clock = -1.; completed = 0
        terminal_outcomes = {}
        pid = None
        artifact = str(artifact_id)+'/'+mode
        def censored():
            return dict(node=active, runtime=runtime, artifact=artifact,
                observed_lower_bound_seconds=clock-started,
                bound='start to last recorded event; termination/call completion unobserved')
        for index, line in enumerate(progress.read_text(encoding='utf-8').splitlines()):
            row = json.loads(line)
            now = row['monotonic_seconds']; node = row['nodeid']; event = row['event']
            if (row.get('schema') != 'ci-phase-progress-v1' or row['cohort'] != mode
                    or row['config_sha256'] != binding or row['assigned_total'] != len(nodes)
                    or type(now) not in (int, float) or not math.isfinite(now) or now < clock):
                raise ValueError('invalid/binding-drifted phase observation')
            clock = now
            if index == 0:
                pid = row['pid']
                if event != 'phase_start' or node is not None or row['completed_count'] != 0:
                    raise ValueError('phase journal start invalid')
                continue
            if row['pid'] != pid or node not in indices or row['assigned_index'] != indices[node]:
                raise ValueError('phase observation identity differs from assignment')
            if event == 'start':
                if active is not None or node in finished or row['completed_count'] != completed:
                    raise ValueError('phase start outside exact protocol')
                active = node; started = clock; reports = {}
            elif event == 'report':
                name, outcome = row['phase'], row['outcome']
                required = () if name == 'setup' else ('setup',) if name == 'call' or reports.get('setup') != 'passed' else ('setup','call')
                if (node != active or name not in ('setup','call','teardown') or name in reports
                        or outcome not in ('passed','failed','skipped') or tuple(reports) != required
                        or (name == 'call' and reports.get('setup') != 'passed')):
                    raise ValueError('phase report outside legal ordering')
                reports[name] = outcome
            elif event == 'end':
                if node != active or type(row['protocol_complete']) is not bool:
                    raise ValueError('phase end outside exact protocol')
                if row['protocol_complete']:
                    required = ('setup','call','teardown') if reports.get('setup') == 'passed' else ('setup','teardown')
                    if tuple(reports) != required:
                        raise ValueError('completed phase missing actual reports')
                    outcome = ('error' if any(reports.get(p)=='failed' for p in ('setup','teardown')) else
                               'failed' if reports.get('call')=='failed' else 'skipped' if 'skipped' in reports.values() else 'passed')
                    if row['outcome'] != outcome:
                        raise ValueError('phase terminal outcome differs from actual reports')
                    completed += 1
                    terminal_outcomes[node] = outcome
                    result['outcomes'][outcome] = result['outcomes'].get(outcome, 0)+1
                    if outcome == 'passed':
                        result['protocol'].append(dict(node=node, seconds=clock-started, outcome=outcome,
                            runtime=runtime, artifact=artifact+'/progress'))
                else:
                    result['censored'].append(censored())
                if row['completed_count'] != completed:
                    raise ValueError('phase completed count differs')
                finished.add(node); active = None; reports = {}
            else:
                raise ValueError('unexpected phase observation event')
        if active is not None:
            result['censored'].append(censored())
        junit = phase/(mode+'.xml')
        if junit.exists():
            retain(junit)
            seen = set()
            for case in ET.parse(junit).iter('testcase'):
                node = node_id(case.get('classname', ''), case.get('name', ''))
                if node not in indices or node in seen:
                    raise ValueError('JUnit identity outside exact phase assignment')
                seen.add(node)
                outcome = ('error' if case.find('error') is not None else 'failed' if case.find('failure') is not None
                           else 'skipped' if case.find('skipped') is not None else 'passed')
                if terminal_outcomes.get(node) != outcome:
                    raise ValueError('JUnit outcome differs from completed progress evidence')
                if outcome != 'passed':
                    continue
                seconds = float(case.get('time', 0))
                if not math.isfinite(seconds) or seconds < 0:
                    raise ValueError('invalid JUnit duration')
                result['junit'].append(dict(node=node, seconds=seconds, outcome='passed', runtime=runtime, artifact=artifact+'/junit'))
    if not result['protocol']:
        raise ValueError('artifact has no completed passed protocol evidence')
    return result


def ci_verified_runtime(base, entry, job, manifest):
    """Declared matrix runtime verified by successful steps and retained logs."""
    import re
    if (entry.get('workflow_commit') != manifest['head_sha']
            or not any(step['name']=='Set up Python' and step['conclusion']=='success' for step in job['steps'])
            or not any(step['name']=='Verify the requested Mac architecture' and step['conclusion']=='success' for step in job['steps'])):
        raise ValueError('CI runtime verification steps/source absent')
    log_path, workflow_path = Path(base)/entry['log'], Path(base)/entry['workflow']
    versions = set(re.findall(r'Successfully set up CPython \((3\.\d+\.\d+)\)',log_path.read_text(encoding='utf-8')))
    source = workflow_path.read_text(encoding='utf-8')
    if (len(versions)!=1 or "MAC_LABEL: ${{ matrix.os }}" not in source
            or "expected = 'x86_64' if os.environ['MAC_LABEL'] == 'macos-15-intel' else 'arm64'" not in source
            or 'assert platform.machine() == expected' not in source):
        raise ValueError('retained setup/architecture runtime proof incomplete')
    return dict(kind='declared-and-ci-verified', runtime=manifest['runtime'], python_version=versions.pop(),
                job_id=job['id'], log_sha256=hashlib.sha256(log_path.read_bytes()).hexdigest(),
                base_workflow_commit=entry['workflow_commit'],
                base_workflow_sha256=hashlib.sha256(workflow_path.read_bytes()).hexdigest())


def import_artifacts(table, manifest_path):
    """Manifest binds explicit runtime, job, run/head and immutable artifact IDs."""
    import re
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['schema'] != 'ci-artifact-import-v1':
        raise ValueError('unknown artifact import schema')
    result = copy.deepcopy(table)
    for entry in manifest['artifacts']:
        base = manifest_path.parent
        metadata_path, job_path = base/entry['metadata'], base/entry['job_metadata']
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        candidates = [row for row in metadata['artifacts'] if row['id'] == entry['id']]
        if len(candidates) != 1:
            raise ValueError('artifact identity absent or duplicated')
        artifact = candidates[0]; job = json.loads(job_path.read_text(encoding='utf-8'))
        parsed = re.fullmatch(r'pytest \(([^,]+), Python (3\.\d+), shard (\d+)/(\d+)\)', job['name'])
        if (not parsed or artifact['workflow_run']['id'] != int(manifest['run'])
                or artifact['workflow_run']['head_sha'] != manifest['head_sha']
                or job['head_sha'] != manifest['head_sha'] or job['run_id'] != int(manifest['run'])
                or artifact['name'] != 'kernel-runtime-'+parsed[1]+'-'+parsed[2]+'-'+parsed[3]
                or not manifest['runtime'].startswith(parsed[1]+'/python'+parsed[2]+'/')):
            raise ValueError('artifact/job/runtime source identity differs')
        if parsed[1].startswith('macos') and not any(step['name']=='Verify the requested Mac architecture' and step['conclusion']=='success' for step in job['steps']):
            raise ValueError('Mac architecture verification not demonstrated')
        runtime_proof = ci_verified_runtime(base,entry,job,manifest)
        observations = artifact_records(base/entry['directory'], runtime=manifest['runtime'], artifact_id=entry['id'],
            runtime_verification=runtime_proof)
        provenance = dict(run=manifest['run'], head_sha=manifest['head_sha'], artifact_id=entry['id'],
            artifact_name=artifact['name'], job_id=job['id'], runtime=manifest['runtime'],
            source_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            source_files_sha256=observations['source_sha256'],
            metadata_sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            job_sha256=hashlib.sha256(job_path.read_bytes()).hexdigest(), outcomes=observations['outcomes'],
            runtime_verification=observations['runtime_verification'],
            ci_runtime_verification=runtime_proof,
            applicability='Historical cancelled prefixes only; no current-source numerical qualification or fit bound')
        for key, semantics in (('protocol','progress-protocol'),('junit','junit-testcase')):
            if observations[key]:
                result = merge(result, observations[key], provenance, semantics=semantics)
        result['scheduling'].setdefault('censored_observations', []).extend(observations['censored'])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', type=Path, required=True)
    parser.add_argument('--progress-summary', type=Path)
    parser.add_argument('--junit', type=Path)
    parser.add_argument('--run')
    parser.add_argument('--head-sha')
    parser.add_argument('--artifact-manifest', type=Path,
                        help='explicit job/artifact/runtime manifest for retained raw phase journals')
    args = parser.parse_args(argv)
    if args.artifact_manifest:
        if any((args.progress_summary,args.junit,args.run,args.head_sha)):
            raise ValueError('artifact manifest cannot mix legacy source selectors')
        table = json.loads(args.table.read_text(encoding='utf-8'))
        result = import_artifacts(table,args.artifact_manifest)
        args.table.write_text(json.dumps(result,indent=1,sort_keys=True)+'\n',encoding='utf-8')
        print(json.dumps(dict(imported_artifact_manifest=args.artifact_manifest.as_posix(),release_acceptance=False)))
        return 0
    if not all((args.progress_summary,args.junit,args.run,args.head_sha)):
        raise ValueError('legacy refresh requires progress summary, JUnit, run and head SHA')
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
