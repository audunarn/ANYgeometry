"""Read-only parent-module replay for the four regressions added during PP1c."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ['TASK_EVIDENCE_DIRECTORY'])
BASE = '9e4f36099721af78772da4f0e5991e4b22557b71'
sys.path.insert(0, str(ROOT/'src'))
import anygeometry
# These are the only changed existing Python kernel files; every other existing
# dependency stays at BASE. The new ledger cannot be reached by these parent
# inventory calls because the store_objects argument does not exist on BASE.
for short in ('_planning_qualification', '_planning_purity'):
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
    def __init__(self):
        self.failed = self.passed = 0
    def pytest_runtest_logreport(self, report):
        if report.when == 'call':
            self.failed += int(report.failed)
            self.passed += int(report.passed)
counts = Counts()
code = pytest.main(['-q', str(ROOT/'tests/test_planning_ledger.py'),
    '-k', 'junction_switch or survives_json or metaclass_equality', '-p', 'no:cacheprovider',
    '--basetemp', str(OUT/'pytest-temp')], plugins=[counts])
result = dict(parent=BASE, parent_modules=['_planning_qualification','_planning_purity'],
              failed=counts.failed, passed=counts.passed, pytest_exit=code)
(OUT/'parent-replay.json').write_text(json.dumps(result, indent=2))
assert code == 1 and counts.failed == 4 and counts.passed == 0, result
print(json.dumps(result))
