"""Pure tooling contracts; executable without importing geometry or its conftest."""
from __future__ import annotations
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))


class ProfileTests(unittest.TestCase):
    # Use ordinary iterations: pytest reports successful unittest subtests as
    # additional call reports, whereas the isolated driver records one exact
    # setup/call/teardown protocol per collected test case.
    def profiles(self):
        return importlib.import_module('ci_test_profiles')

    def test_disjoint_exhaustive_and_unknown_defaults_integration(self):
        p = self.profiles()
        nodes = ['tests/test_ci_test_profiles.py::test_one',
                 'tests/test_certified_hydrostatics_verify.py::test_two[x]',
                 'tests/test_future.py::test_three']
        groups = p.partition(nodes)
        self.assertEqual(groups, {'unit': nodes[:1], 'integration': nodes[2:], 'qualification': nodes[1:2]})
        self.assertEqual(sum(map(len, groups.values())), len(nodes))
        self.assertEqual(set().union(*map(set, groups.values())), set(nodes))
        self.assertEqual(p.select(nodes, 'full'), nodes)
        self.assertEqual(p.select(nodes, 'fast'), nodes[:1])
        self.assertEqual(p.select(nodes, 'qualification'), nodes[1:2])

    def test_duplicate_or_malformed_inventory_refused(self):
        p = self.profiles()
        for nodes in (['tests/test_a.py::a'] * 2, ['../test_a.py::a'], ['tests/test_a.py'], [1], []):
            with self.assertRaises(ValueError, msg=repr(nodes)):
                p.partition(nodes)

    def test_full_selector_refuses_inherited_narrowing(self):
        p = self.profiles()
        for env in ({'PYTEST_ADDOPTS': '-k easy'}, {'ANYGEOMETRY_TEST_PROFILE': 'unit'},
                    {'ANYGEOMETRY_TEST_TIER': 'unit'}):
            with self.assertRaises(ValueError, msg=repr(env)):
                p.validate_selection('full', [], '', '', env)
        for flag in ('select', 'k', 'm'):
            with self.assertRaises(ValueError, msg=flag):
                p.validate_selection('full', ['tests/test_a.py'] if flag == 'select' else [],
                                     'easy' if flag == 'k' else '', 'slow' if flag == 'm' else '', {})
        p.validate_selection('full', [], '', '', {'ANYGEOMETRY_TEST_SHARD': '1/16'})
        p.validate_selection('unit', [], '', '', {})

    def test_module_selection_precedes_collection_and_records_partial_scope(self):
        p = self.profiles()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('test_ci_test_profiles.py', 'test_certified_hydrostatics_verify.py', 'test_future.py'):
                path = root / 'tests' / name
                path.parent.mkdir(exist_ok=True)
                path.write_text('def test_one(): pass\n', encoding='utf-8')
            receipt = p.module_manifest(root, 'unit')
            self.assertEqual(receipt['selected_modules'], ['tests/test_ci_test_profiles.py'])
            self.assertEqual(receipt['tier_counts'], {'unit': 1, 'integration': 1, 'qualification': 1})
            self.assertEqual(receipt['scope'], 'partial-development-only')
            self.assertFalse(receipt['release_acceptance'])
            self.assertEqual(len(receipt['inventory_sha256']), 64)
            self.assertEqual(p.module_manifest(root, 'full')['scope'], 'full-source-module-inventory')

    def test_receipt_binds_exact_node_manifest_without_release_claim(self):
        p = self.profiles()
        nodes = ['tests/test_ci_test_profiles.py::test_one', 'tests/test_unknown.py::test_two']
        receipt = p.node_receipt(nodes, 'full')
        self.assertEqual(receipt['tiers']['unit'], nodes[:1])
        self.assertEqual(receipt['tiers']['integration'], nodes[1:])
        self.assertEqual(receipt['selected'], nodes)
        self.assertEqual(receipt['inventory_count'], 2)
        self.assertFalse(receipt['release_acceptance'])
        self.assertNotEqual(receipt['inventory_sha256'], p.node_receipt(list(reversed(nodes)), 'full')['inventory_sha256'])

    def test_actual_driver_profile_allowlist_and_legacy_selector_compatibility(self):
        import ci_isolated_tests as ci
        for flags, expected_profile, allowed in ((['--profile', 'unit'], 'unit', True),
                                                  (['--select', 'tests/test_ci_test_profiles.py'], 'legacy-focused', False),
                                                  (['--profile', 'full'], 'full', False)):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                path = root / 'tests/test_ci_test_profiles.py'
                path.parent.mkdir()
                path.write_text('def test_one(): pass\n', encoding='utf-8')
                node = 'tests/test_ci_test_profiles.py::test_one'
                calls = []
                def fake_run(root, folder, name, config, env, **kwargs):
                    calls.append(config)
                    Path(config['manifest']).write_text(json.dumps({'nodes': [node]}))
                    if name == 'collection':
                        Path(env['ANYGEOMETRY_SHARD_REPORT']).write_text(json.dumps(
                            {'index': 1, 'count': 1, 'selected': [node], 'ran': [node], 'slow': []}))
                    return 0
                with mock.patch.dict(ci.os.environ, {}, clear=True), mock.patch.object(ci, 'run_phase', fake_run):
                    self.assertEqual(ci.main(['--root', str(root), '--collect-only', *flags]), 0)
                receipt = json.loads((root / 'ci-isolation/profile-receipt.json').read_text())
                self.assertEqual(receipt['profile'], expected_profile)
                self.assertEqual('allowed_modules' in calls[0], allowed)
                if allowed:
                    self.assertEqual(calls[0]['allowed_modules'], ['tests/test_ci_test_profiles.py'])
                self.assertTrue(any(row['mode'] == 'core' for row in calls))
        with mock.patch.dict(ci.os.environ, {}, clear=True), mock.patch.object(ci, 'run_phase') as run:
            with self.assertRaises(ValueError):
                ci.main(['--profile', 'full', '--select', 'tests/test_ci_test_profiles.py'])
            run.assert_not_called()

    def test_runner_and_workflows_are_integrated_without_replacing_release(self):
        source = (ROOT / 'tools/ci_isolated_tests.py').read_text(encoding='utf-8')
        self.assertIn("parser.add_argument('--profile'", source)
        self.assertIn('module_manifest(root,args.profile)', source)
        self.assertIn('profile-receipt.json', source)
        self.assertIn("collection['allowed_modules']", source)
        self.assertIn('validate_selection(', source)
        dev = (ROOT / '.github/workflows/development.yml').read_text(encoding='utf-8')
        self.assertIn('pull_request:', dev)
        self.assertIn('push:', dev)
        self.assertIn('--profile unit', dev)
        self.assertNotIn('matrix:', dev)
        self.assertIn('timeout-minutes: 10', dev)
        manual = (ROOT / '.github/workflows/test-profile.yml').read_text(encoding='utf-8')
        self.assertIn('options: [unit, integration, qualification]', manual)
        self.assertIn('ci_isolated_tests.py --profile', manual)
        full = (ROOT / '.github/workflows/ci.yml').read_text(encoding='utf-8')
        self.assertIn('workflow_dispatch:', full)
        self.assertIn('schedule:', full)
        self.assertIn('--profile full', full)
        self.assertIn('timeout-minutes: 30', full)
        self.assertIn('pytest-qualified:', full)
        self.assertIn('wheel-smoke:', full)

    def test_existing_integration_gates_remain_required_by_full_workflow(self):
        import re
        integration = (ROOT / '.github/workflows/integration.yml').read_text(encoding='utf-8')
        jobs = set(re.findall(r'^  ([a-z][a-z-]*):$', integration.split('jobs:', 1)[1], re.M))
        self.assertEqual(jobs, {'plan', 'kernel', 'shards', 'wheel', 'wheel-smoke', 'mesher', 'gate'})
        self.assertIn('workflow_call:', integration)
        self.assertIn('workflow_dispatch:', integration)
        self.assertNotIn('pull_request:\n', integration)
        self.assertIn("BASE_SHA: ''", integration)
        self.assertIn('needs: [plan, kernel, shards, wheel, wheel-smoke, mesher]', integration)
        self.assertIn("for name in ('kernel', 'shards', 'wheel', 'wheel-smoke', 'mesher'):", integration)
        self.assertIn("assert needs[name]['result'] == ('success' if runtime else 'skipped')", integration)
        full = (ROOT / '.github/workflows/ci.yml').read_text(encoding='utf-8')
        self.assertIn('uses: ./.github/workflows/integration.yml', full)
        dev = (ROOT / '.github/workflows/development.yml').read_text(encoding='utf-8')
        self.assertIn('name: Development gate', dev)
        self.assertIn('needs: unit', dev)


class SchedulingTests(unittest.TestCase):
    def artifact_fixture(self, folder):
        import hashlib
        phase = folder/'ci-isolation'
        phase.mkdir()
        nodes = ['tests/test_plain.py::test_pass', 'tests/test_plain.py::test_skip', 'tests/test_plain.py::test_active']
        config = phase/'core-config.json'
        config.write_text(json.dumps(dict(mode='core', nodes=nodes)))
        binding = hashlib.sha256(config.read_bytes()).hexdigest()
        rows = []
        def event(kind, index, clock, count, **details):
            rows.append(dict(schema='ci-phase-progress-v1', cohort='core', event=kind,
                nodeid=None if index is None else nodes[index], config_sha256=binding, pid=2556,
                monotonic_seconds=clock, assigned_total=3, assigned_index=index,
                completed_count=count, **details))
        event('phase_start', None, 10., 0)
        event('start', 0, 11., 0)
        for offset, name in enumerate(('setup', 'call', 'teardown'), 1):
            event('report', 0, 11.+offset, 0, phase=name, outcome='passed')
        event('end', 0, 15., 1, outcome='passed', protocol_complete=True, error_type=None)
        event('start', 1, 16., 1)
        event('report', 1, 17., 1, phase='setup', outcome='skipped')
        event('report', 1, 18., 1, phase='teardown', outcome='passed')
        event('end', 1, 19., 2, outcome='skipped', protocol_complete=True, error_type=None)
        event('start', 2, 20., 2)
        event('report', 2, 21., 2, phase='setup', outcome='passed')
        (phase/'core-progress.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        (phase/'core.xml').write_text('<testsuite><testcase classname="tests.test_plain" name="test_pass" time="3.5"/>'
                                    '<testcase classname="tests.test_plain" name="test_skip" time="1"><skipped/></testcase></testsuite>')
        (folder/'arc-runtime.json').write_text(json.dumps(dict(python_version='3.11.9 (Clang)', pytest_version='9.2.0')))
        return phase

    def test_mac_artifact_shape_keeps_passed_clocks_and_censored_active_case_separate(self):
        from ci_schedule_observations import artifact_records
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self.artifact_fixture(folder)
            result = artifact_records(folder, runtime='macos-15-intel/python3.11/x86_64', artifact_id=13443466657)
            self.assertEqual(len(result['protocol']), 1)
            self.assertEqual(result['protocol'][0]['seconds'], 4.)
            self.assertEqual(len(result['junit']), 1)
            self.assertEqual(result['junit'][0]['seconds'], 3.5)
            self.assertEqual(result['censored'][0]['node'], 'tests/test_plain.py::test_active')
            self.assertEqual(result['censored'][0]['observed_lower_bound_seconds'], 1.)
            self.assertNotIn('seconds', result['censored'][0])
            self.assertEqual(result['outcomes'], {'passed': 1, 'skipped': 1})
            self.assertEqual(result['protocol'][0]['runtime'], 'macos-15-intel/python3.11/x86_64')

    def test_mac_artifact_import_refuses_runtime_or_bound_config_drift(self):
        from ci_schedule_observations import artifact_records
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            phase = self.artifact_fixture(folder)
            with self.assertRaises(ValueError):
                artifact_records(folder, runtime='macos-15-intel/python3.11/arm64', artifact_id=13443466657)
            with self.assertRaises(ValueError):
                artifact_records(folder, runtime='macos-15-intel/python3.12/x86_64', artifact_id=13443466657)
            runtime_path = folder/'arc-runtime.json'
            runtime_text = runtime_path.read_text()
            runtime_path.unlink()
            with self.assertRaises(ValueError):
                artifact_records(folder, runtime='macos-15-intel/python3.11/x86_64', artifact_id=13443466657)
            runtime_path.write_text(runtime_text)
            config = phase/'core-config.json'
            config.write_text(config.read_text()+' ')
            with self.assertRaises(ValueError):
                artifact_records(folder, runtime='macos-15-intel/python3.11/x86_64', artifact_id=13443466657)

    def test_retained_unstarted_cohort_schema_is_censored_without_fabricated_completion(self):
        from ci_schedule_observations import retained_records
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            summary = folder/'summary.json'
            junit = folder/'core.xml'
            # Actual retained shard7 oracle shape: assignment and estimate only,
            # with no phase start, completed count, outcomes or completed rows.
            data = {'7': {'groups': {'oracle': {'assigned': 27, 'estimated_seconds': 12.341944},
                'core': {'assigned': 2, 'completed': 1, 'completed_rows': [dict(
                    nodeid='tests/test_plain.py::test_one', protocol_elapsed_seconds=174., outcome='passed')]}}}}
            summary.write_text(json.dumps(data))
            junit.write_text('<testsuite><testcase classname="tests.test_plain" name="test_one" time="1.5"/></testsuite>')
            protocol, cases = retained_records(summary, junit)
            self.assertEqual(len(protocol), 1)
            self.assertEqual(len(cases), 1)
            self.assertEqual(protocol[0]['seconds'], 174.)
            self.assertNotIn('oracle', protocol[0]['artifact'])
            data['7']['groups']['oracle']['completed_rows'] = []
            summary.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                retained_records(summary, junit)

    def test_retained_clocks_preserve_history_and_correct_fallback_cost(self):
        from ci_schedule_observations import merge
        from ci_shards import duration, slow_ids, assign, scheduling_receipt
        node = 'tests/test_heavy.py::test_case'
        table = {'tests': {}, 'rest_mean': {'windows': .1}, 'scheduling': {'tests': {}, 'provenance': {'old': True}}}
        record = dict(node=node, seconds=174.0, outcome='passed', runtime='windows313', artifact='progress-1')
        provenance = dict(run='run', head_sha='sha', source_sha256='0'*64)
        refreshed = merge(table, [record], provenance, semantics='progress-protocol')
        self.assertEqual(table['scheduling']['tests'], {})
        self.assertEqual(refreshed['scheduling']['provenance'], {'old': True})
        self.assertEqual(refreshed['tests'], table['tests'])
        self.assertEqual(refreshed['scheduling']['tests'], {})
        self.assertEqual(duration(refreshed, node), 174.0)
        self.assertIn(node, slow_ids(refreshed))
        bucket = refreshed['scheduling']['protocol_tests'][node]['windows313']
        self.assertEqual(bucket['retained_seconds'], [174.0])
        with self.assertRaises(ValueError):
            merge(refreshed, [record], provenance, semantics='progress-protocol')
        ids = [node, 'tests/test_unknown.py::test_one', 'tests/test_other.py::test_two']
        self.assertNotEqual(assign(ids, 2, table), assign(ids, 2, refreshed))
        receipt = scheduling_receipt(ids, 2, refreshed)
        self.assertEqual(receipt['setup_reserve_seconds'], 180)
        self.assertEqual(receipt['unobserved_count'], 2)
        self.assertFalse(receipt['measured_fit'])
        self.assertFalse(receipt['release_acceptance'])

    def test_unfinished_failed_or_invalid_observations_do_not_become_completed_costs(self):
        from ci_schedule_observations import merge
        provenance = dict(run='run', head_sha='sha', source_sha256='0'*64)
        record = dict(node='tests/test_heavy.py::test_case', seconds=999., outcome='interrupted', runtime='windows313', artifact='a')
        refreshed = merge({}, [record], provenance, semantics='progress-protocol')
        self.assertEqual(refreshed['scheduling']['protocol_tests'], {})
        for seconds in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError, msg=repr(seconds)):
                merge({}, [dict(record, outcome='passed', seconds=seconds)], provenance, semantics='progress-protocol')


if __name__ == '__main__':
    unittest.main()
