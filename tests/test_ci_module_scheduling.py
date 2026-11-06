"""Pure caller/coverage contracts for fixture-preserving CI assignment."""
from pathlib import Path
import ast
import importlib
import json
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import ci_shards as shards
import ci_isolated_tests as isolated


class ModuleSchedulingTests(unittest.TestCase):
    def inventory(self):
        ids=['tests/test_a.py::test_a[0]','tests/test_b.py::test_b','tests/test_a.py::test_a[1]','tests/test_c.py::test_c']
        table={'tests':{ids[0]:{'windows':8.},ids[1]:{'windows':7.},ids[2]:{'windows':6.},ids[3]:{'windows':2.}}}
        return ids,table

    def test_module_groups_are_deterministic_exhaustive_and_default_stays_node(self):
        ids,table=self.inventory()
        self.assertEqual(shards.assign(ids,2,table),{ids[0]:1,ids[1]:2,ids[2]:2,ids[3]:1})
        owner=shards.assign(ids,2,table,grouping='module')
        self.assertEqual(owner[ids[0]],owner[ids[2]])
        self.assertEqual(owner,shards.assign(list(reversed(ids)),2,table,grouping='module'))
        self.assertEqual(set(owner),set(ids))
        self.assertEqual(sum(sum(owner[node]==i for node in ids) for i in (1,2)),len(ids))
        self.assertEqual(owner[ids[0]],1)
        self.assertEqual(shards.scheduling_receipt(['legacy-label'],1,{} )['grouping'],'node')

    def test_unknown_grouping_refuses_and_receipt_reports_largest_module(self):
        ids,table=self.inventory()
        with self.assertRaises(ValueError):shards.assign(ids,2,table,grouping='guess')
        with self.assertRaises(ValueError):shards.scheduling_receipt(ids,2,table,grouping='guess')
        receipt=shards.scheduling_receipt(ids,2,table,grouping='module')
        self.assertEqual(receipt['grouping'],'module')
        self.assertEqual(receipt['largest_group']['key'],'tests/test_a.py')
        self.assertEqual(receipt['largest_group']['estimated_seconds'],14.)
        self.assertEqual(receipt['group_count'],3)
        self.assertFalse(receipt['measured_fit'])

    def test_only_actual_scoped_fixture_modules_are_admitted_as_groups(self):
        ids=['tests/test_fixture.py::test_a','tests/test_fixture.py::test_b',
             'tests/test_free.py::test_a','tests/test_free.py::test_b']
        table={'tests':{node:{'windows':cost} for node,cost in zip(ids,(7.,7.,857.343,857.343))}}
        owner=shards.assign(ids,2,table,grouping='module',group_modules=['tests/test_fixture.py'])
        self.assertEqual(owner[ids[0]],owner[ids[1]])
        self.assertNotEqual(owner[ids[2]],owner[ids[3]])
        self.assertEqual(set(owner),set(ids))
        with self.assertRaises(ValueError):
            shards.assign(ids,2,table,grouping='module',group_modules=['tests/missing.py'])

    def test_real_collection_hook_records_policy_and_keeps_module_together(self):
        tree=ast.parse((ROOT/'tests/conftest.py').read_text())
        function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='pytest_collection_finish')
        function.decorator_list=[]
        space={'Path':Path,'_ci_shards':lambda:shards}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'real-conftest-hook','exec'),space)
        ids,_=self.inventory()
        items=[SimpleNamespace(nodeid=node,get_closest_marker=lambda name:False,
            _fixtureinfo=SimpleNamespace(name2fixturedefs={'used':[SimpleNamespace(scope='module' if '/test_a.py::' in node else 'function')]})) for node in ids]
        session=SimpleNamespace(items=items,config=SimpleNamespace(hook=SimpleNamespace(pytest_deselected=lambda **kwargs:None)))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'shard.json'
            with mock.patch.dict(os.environ,{'ANYGEOMETRY_TEST_SHARD':'1/2','ANYGEOMETRY_SHARD_REPORT':str(path),'ANYGEOMETRY_SHARD_GROUPING':'module'},clear=True):
                space['pytest_collection_finish'](session)
            report=json.loads(path.read_text())
            self.assertEqual(report['grouping'],'module')
            self.assertEqual(report['group_modules'],['tests/test_a.py'])
            self.assertEqual([node for node in ids if node in report['ran']],report['ran'])
            self.assertEqual(len(set(node in report['ran'] for node in (ids[0],ids[2]))),1)

    def reports(self,folder,policies,split=False):
        ids,_=self.inventory()
        partitions=([ids[0],ids[2]],[ids[1],ids[3]]) if not split else ([ids[0],ids[1]],[ids[2],ids[3]])
        paths=[]
        for index,(policy,nodes) in enumerate(zip(policies,partitions),1):
            target=folder/str(index);target.mkdir()
            report=target/'shard.json'
            report.write_text(json.dumps(dict(index=index,count=2,selected=ids,ran=nodes,slow=[],grouping=policy,
                group_modules=['tests/test_a.py'] if policy=='module' else [])))
            rows=''.join('<testcase classname="tests.'+node.split('::')[0][6:-3]+'" name="'+node.split('::')[1]+'" time="1"/>' for node in nodes)
            (target/'kernel-results.xml').write_text('<testsuite>'+rows+'</testsuite>')
            paths.append(report)
        return paths

    def test_verifier_refuses_unknown_mixed_or_split_module_policy(self):
        for policies,split in ((('module','node'),False),(('guess','guess'),False),(('module','module'),True)):
            with tempfile.TemporaryDirectory() as temp:
                self.assertTrue(shards.verify(self.reports(Path(temp),policies,split)))
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(shards.verify(self.reports(Path(temp),('module','module'))),[])
        with tempfile.TemporaryDirectory() as temp:
            paths=self.reports(Path(temp),('module','module'))
            changed=json.loads(paths[1].read_text());changed['group_modules']=['tests/test_c.py']
            paths[1].write_text(json.dumps(changed))
            self.assertTrue(shards.verify(paths))

    def test_driver_binds_policy_to_receipts_and_does_not_leak_it_to_nested_tests(self):
        for report_policy,phase_policy in (('module','module'),('module','node'),('node','node'),('guess','guess')):
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp);(root/'tests').mkdir();(root/'tests/test_a.py').write_text('def test_a(): pass\n')
                node='tests/test_a.py::test_a';seen=[]
                def fake_run(root,folder,name,config,env,**kwargs):
                    seen.append((name,config,dict(env)))
                    Path(config['manifest']).write_text(json.dumps(dict(nodes=[node],grouping=report_policy if name=='collection' else phase_policy,group_modules=[])))
                    if name=='collection':
                        Path(env['ANYGEOMETRY_SHARD_REPORT']).write_text(json.dumps(dict(index=1,count=1,selected=[node],ran=[node],slow=[],grouping=report_policy,group_modules=[])))
                    return 0
                with mock.patch.dict(os.environ,{'ANYGEOMETRY_SHARD_GROUPING':'module'},clear=True),mock.patch.object(isolated,'run_phase',fake_run):
                    if report_policy!='module':
                        with self.assertRaises(ValueError):isolated.main(['--root',str(root),'--collect-only'])
                        self.assertEqual(len(seen),1)
                        continue
                    if phase_policy!='module':
                        with self.assertRaises(ValueError):isolated.main(['--root',str(root),'--collect-only'])
                        continue
                    self.assertEqual(isolated.main(['--root',str(root),'--collect-only']),0)
                for filename in ('schedule-receipt.json','profile-receipt.json','plan.json','complete.json'):
                    self.assertEqual(json.loads((root/'ci-isolation'/filename).read_text())['grouping'],'module')
                phases=[row for row in seen if row[0]=='core']
                self.assertEqual(phases[0][1]['grouping'],'module')
                self.assertNotIn('ANYGEOMETRY_SHARD_GROUPING',phases[0][2])

    def test_workflow_profiles_preserve_full_union_and_qualified_path(self):
        development=(ROOT/'.github/workflows/development.yml').read_text()
        manual=development.split('\n  manual-integration:\n',1)[1]
        self.assertIn("if: github.event_name == 'workflow_dispatch'",manual)
        self.assertIn('needs: unit',manual)
        self.assertIn('uses: ./.github/workflows/integration.yml',manual)
        self.assertIn('actions: read',manual)
        ci=(ROOT/'.github/workflows/ci.yml').read_text()
        regular=ci.split('\n  pytest:\n',1)[1].split('\n  pytest-qualified:\n',1)[0]
        qualified=ci.split('\n  pytest-qualified:\n',1)[1].split('\n  pytest-shards:\n',1)[0]
        self.assertIn("ANYGEOMETRY_SHARD_GROUPING: 'module'",regular)
        self.assertNotIn('ANYGEOMETRY_SHARD_GROUPING',qualified)
        self.assertIn('--profile full',regular)
        legacy=(ROOT/'.github/workflows/integration.yml').read_text().split('\n  kernel:\n',1)[1].split('\n  shards:\n',1)[0]
        self.assertIn('--profile integration',legacy)
        self.assertIn("ANYGEOMETRY_SHARD_GROUPING: 'module'",legacy)
        self.assertNotIn('python -m pytest',legacy)
        self.assertIn('repository: audunarn/ANYmesh',legacy)
        self.assertIn('ref: 2ef01d47e29fb461da1cf1ece73913c1708d4e49',legacy)
        self.assertIn('path: .worktrees/ci-mesher-consumer',legacy)
        self.assertIn('PYTHONPATH: ${{ github.workspace }}/.worktrees/ci-mesher-consumer/src',legacy)
        self.assertIn('CI_MESHER_SOURCE_ROOT:',legacy)
        self.assertIn('consumer-runtime.json',legacy)
        profiles=importlib.import_module('ci_test_profiles')
        inventory=profiles.module_manifest(ROOT,'full')['inventory_modules']
        nodes=[module+'::inventory_probe' for module in inventory]
        groups=profiles.partition(nodes)
        self.assertEqual(set().union(*map(set,groups.values())),set(nodes))
        self.assertEqual(sum(map(len,groups.values())),len(nodes))
        self.assertEqual(profiles.select(nodes,'full'),nodes)

    def test_module_ci_report_cannot_use_all_modules_none_sentinel(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'tests').mkdir();(root/'tests/test_a.py').write_text('def test_a(): pass\n')
            node='tests/test_a.py::test_a';calls=[]
            def fake_run(root,folder,name,config,env,**kwargs):
                calls.append(name)
                Path(config['manifest']).write_text(json.dumps(dict(nodes=[node],grouping='module',group_modules=None)))
                Path(env['ANYGEOMETRY_SHARD_REPORT']).write_text(json.dumps(dict(index=1,count=1,selected=[node],ran=[node],slow=[],grouping='module',group_modules=None)))
                return 0
            with mock.patch.dict(os.environ,{'ANYGEOMETRY_SHARD_GROUPING':'module'},clear=True),mock.patch.object(isolated,'run_phase',fake_run):
                with self.assertRaises(ValueError):isolated.main(['--root',str(root),'--collect-only'])
            self.assertEqual(calls,['collection'])

    def test_module_grouping_refuses_qualified_positive_host_before_collection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'tests').mkdir();(root/'tests/test_a.py').write_text('def test_a(): pass\n')
            with mock.patch.dict(os.environ,{'ANYGEOMETRY_SHARD_GROUPING':'module','ANYGEOMETRY_REQUIRE_QUALIFIED_PLANNING_TESTS':'1'},clear=True),mock.patch.object(isolated,'run_phase') as run:
                with self.assertRaises(ValueError):isolated.main(['--root',str(root),'--collect-only'])
                run.assert_not_called()


if __name__=='__main__':unittest.main()
