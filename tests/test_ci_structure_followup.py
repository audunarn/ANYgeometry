"""Small regressions for CI dependency installation and bounded driver fixtures."""
from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))


class FollowupTests(unittest.TestCase):
    def test_integration_partition_replaces_the_timed_out_five_way_layout(self):
        import re
        text=(ROOT/'.github/workflows/integration.yml').read_text()
        kernel=text.split('\n  kernel:\n',1)[1].split('\n  shards:\n',1)[0]
        axis=re.search(r'^        shard: \[([0-9, ]+)\]$',kernel,re.M)
        values=[int(value.strip()) for value in axis.group(1).split(',')]
        self.assertGreater(len(values),5)
        self.assertEqual(values,list(range(1,len(values)+1)))
        self.assertEqual(kernel.count('${{ matrix.shard }}/'+str(len(values))),2)
        self.assertIn('timeout-minutes: 40',kernel)
        self.assertIn('--profile integration',kernel)
        self.assertIn("ANYGEOMETRY_SHARD_GROUPING: 'module'",kernel)

    def test_windows_integration_installs_and_preflights_required_planner_helper(self):
        text = (ROOT/'.github/workflows/integration.yml').read_text()
        kernel = text.split('\n  kernel:\n', 1)[1].split('\n  shards:\n', 1)[0]
        install = kernel.split('name: Install integration test dependencies', 1)[1].split('      - ', 1)[0]
        self.assertIn("ANYGEOMETRY_REQUIRE_PLANNING_NATIVE: ${{ runner.os == 'Windows' && '1' || '0' }}", install)
        self.assertIn("python -m pip install -e '.[dev,planar,sketch,test]'", install)
        self.assertIn('name: Verify required integration planner helper', kernel)
        self.assertIn('import anygeometry._planning_native_core', kernel)
        self.assertLess(kernel.index('name: Verify required integration planner helper'), kernel.index('name: Run the kernel shard'))
        self.assertIn('timeout-minutes: 40', kernel)
        self.assertIn('shard: [1, 2, 3, 4, 5, 6, 7, 8]', kernel)
        self.assertNotIn('ANYGEOMETRY_BUILD_NATIVE', kernel)

    def tooling(self):
        spec = importlib.util.spec_from_file_location('bounded_fixture_source', ROOT/'tests/test_ci_isolated_tests.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_core_oracle_driver_fixture_has_only_two_tiny_modules(self):
        tooling = self.tooling()
        with tempfile.TemporaryDirectory() as temp:
            root = tooling.tiny_driver_checkout(Path(temp))
            modules = sorted(path.name for path in (root/'tests').glob('test_*.py'))
            self.assertEqual(modules, ['test_core.py', 'test_oracle.py'])
            self.assertIn('import mpmath', (root/'tests/test_oracle.py').read_text())
            self.assertIn('examples.namespace_probe', (root/'tests/test_core.py').read_text())
            self.assertFalse((root/'src').exists())
            self.assertLess(sum(path.stat().st_size for path in root.rglob('*.py')), 5000)

    def test_both_nested_regressions_use_tiny_checkout_without_timeout_increase(self):
        import ast
        tree = ast.parse((ROOT/'tests/test_ci_isolated_tests.py').read_text())
        names = {'test_real_selected_cases_keep_shard_assignment_and_complete_actual_union',
                 'test_bootstrap_preserves_checkout_namespace_imports_at_real_collection'}
        for function in [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]:
            calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
            self.assertTrue(any(isinstance(node.func, ast.Name) and node.func.id=='tiny_driver_checkout' for node in calls), function.name)
            self.assertTrue(any(isinstance(node.func, ast.Attribute) and node.func.attr=='run'
                                and any(key.arg=='timeout' and isinstance(key.value, ast.Constant) and key.value.value==40 for key in node.keywords)
                                for node in calls), function.name)


if __name__ == '__main__':
    unittest.main()
