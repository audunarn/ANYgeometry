"""Read-only replay of PP1d regression cases on exact parent boundary modules."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ['TASK_EVIDENCE_DIRECTORY'])
BASE = '27a1ec30b732b9c49cef4b07defa83bb1c9b2908'
sys.path.insert(0, str(ROOT/'src'))
import anygeometry
names = ('_planning_qualification', '_planning_purity', '_planning_ledger')
for short in names:
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
selection = sys.argv[1:]
code = pytest.main(['-q', str(ROOT/'tests/test_planning_review.py'), '-p', 'no:cacheprovider',
                   '--basetemp', str(OUT/'pytest-temp'), *selection], plugins=[counts])
result = dict(parent=BASE, parent_modules=names, failed=counts.failed,
              passed=counts.passed, pytest_exit=code)
(OUT/'parent-replay.json').write_text(json.dumps(result, indent=2))
expected = 1 if selection == ['-k', 'reset_descendant_with_external_alias'] else 22
assert code == 1 and counts.failed == expected and counts.passed == 0, result
print(json.dumps(result))
