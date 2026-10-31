"""Read-only exact-parent replay; never checkout or replace working source."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ['TASK_EVIDENCE_DIRECTORY'])
BASE = 'd58b8ff1ec1667e70c7cce6913d14dd8918dcd84'
sys.path.insert(0, str(ROOT/'src'))
import anygeometry
for short in ('_planning_purity', '_planning_ledger'):
    data = subprocess.check_output(['git', 'show', BASE+':src/anygeometry/'+short+'.py'], cwd=ROOT)
    path = OUT/(short+'.py')
    path.write_bytes(data)
    name = 'anygeometry.'+short
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    setattr(anygeometry, short, module)
import pytest
class Counts:
    failed = passed = 0
    def pytest_runtest_logreport(self, report):
        if report.when == 'call':
            self.failed += int(report.failed)
            self.passed += int(report.passed)
counts = Counts()
code = pytest.main(['-q', str(ROOT/'tests/test_planning_metadata.py'), '-p', 'no:cacheprovider',
                   '--basetemp', str(OUT/'pytest-temp')], plugins=[counts])
result = dict(parent=BASE, failed=counts.failed, passed=counts.passed, pytest_exit=code)
(OUT/'parent-replay.json').write_text(json.dumps(result, indent=2))
assert code == 1 and counts.failed == 19 and counts.passed == 0, result
print(json.dumps(result))
