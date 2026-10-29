"""CI tiers tools: duration table, shard assignment and completeness checks (CI tiers C0, C2, C3)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import ci_durations  # noqa: E402
import ci_shards  # noqa: E402


def _junit(path: Path, cases: dict[str, float]) -> Path:
    rows = "".join(f'<testcase classname="tests.{name.split("::")[0]}" name="{name.split("::")[1]}" time="{t}"/>'
                   for name, t in cases.items())
    path.write_text(f'<testsuites><testsuite name="pytest">{rows}</testsuite></testsuites>', encoding="utf-8")
    return path


def test_node_ids_follow_pytest():
    assert ci_durations.node_id("tests.test_hydrostatics", "test_x[1.0]") == "tests/test_hydrostatics.py::test_x[1.0]"


def test_activity_artifact_is_live_before_a_case_finishes(tmp_path):
    import os
    import shutil
    import subprocess
    project = tmp_path / "probe"
    project.mkdir()
    (project / "tools").mkdir()
    (project / "tests").mkdir()
    (project / "pytest.ini").write_text("[pytest]\ntestpaths = tests\n", encoding="utf-8")
    source = Path(__file__).resolve().parent
    shutil.copy2(source / "conftest.py", project / "tests/conftest.py")
    shutil.copy2(TOOLS / "ci_shards.py", project / "tools/ci_shards.py")
    (project / "tests/test_probe.py").write_text(
        "import json,os\nfrom pathlib import Path\n"
        "def test_reads_live_start():\n"
        "    row=json.loads(Path(os.environ['ANYGEOMETRY_ACTIVE_TEST_REPORT']).read_text())\n"
        "    assert row['phase']=='started'\n"
        "    assert row['nodeid']=='tests/test_probe.py::test_reads_live_start'\n"
        "def test_last():\n    pass\n", encoding="utf-8")
    report = project / "active.json"
    env = dict(os.environ, ANYGEOMETRY_ACTIVE_TEST_REPORT=str(report), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    for name in ("ANYGEOMETRY_TEST_SHARD", "ANYGEOMETRY_SHARD_REPORT"):
        env.pop(name, None)
    completed = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
        "--rootdir="+str(project), "--confcutdir="+str(project), "-c", str(project / "pytest.ini")],
        cwd=project, env=env, capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    row = json.loads(report.read_text(encoding="utf-8"))
    assert row["phase"] == "finished" and row["nodeid"] == "tests/test_probe.py::test_last"
    assert row["utc"] and row["shard"] is None
    assert not report.with_name(report.name + ".tmp").exists()


def test_the_table_reproduces_its_totals(tmp_path):
    a = _junit(tmp_path / "a.xml", {"test_m::t1": 12., "test_m::t2": .2, "test_m::t3": .4})
    b = _junit(tmp_path / "b.xml", {"test_m::t1": 6., "test_m::t2": .1, "test_m::t3": .3})
    table = ci_durations.build([f"windows={a}", f"ubuntu={b}"], 1.0, "run")
    assert set(table["tests"]) == {"tests/test_m.py::t1"}
    assert table["totals"]["windows"] == {"tests": 3, "seconds": 12.6}
    assert ci_durations.check(table) == []
    table["totals"]["windows"]["seconds"] = 20.
    assert ci_durations.check(table)


def test_partial_observations_preserve_baseline_and_missing_platforms(tmp_path):
    baseline = {"tests": {"old": {"windows": 30.}, "changed": {"windows": 4.}},
                "rest_mean": {"windows": .1}}
    path = tmp_path / "durations.json"
    path.write_text(json.dumps(baseline), encoding="utf-8")
    observed = {"changed": {"ubuntu": 40.}, "new": {"ubuntu": 15.}}
    path.with_name("duration-observations.json").write_text(
        json.dumps({"tests": observed, "partial_tests": {"failed": {"windows": 80.}}}), encoding="utf-8")
    table = ci_shards.load_table(path)
    assert table["tests"] == baseline["tests"]
    assert table["observations"] == observed
    assert ci_shards.duration(table, "old") == 30.
    assert ci_shards.duration(table, "changed") == 40.
    assert ci_shards.duration(table, "new") == 15.
    assert ci_shards.duration(table, "missing") == .1
    assert ci_shards.duration(table, "failed") == 80.
    assert ci_shards.slow_ids(table) == {"old", "changed", "new", "failed"}
    assert ci_shards.slow_ids(table, 35.) == {"changed", "failed"}


def test_fixture_setup_classification_follows_definitions_and_preserves_execution(tmp_path):
    import os
    import shutil
    import subprocess
    project = tmp_path / 'fixture-cost'
    (project/'tests').mkdir(parents=True)
    (project/'tools').mkdir()
    (project/'reports/ci').mkdir(parents=True)
    source = Path(__file__).resolve().parent
    shutil.copy2(source/'conftest.py', project/'tests/conftest.py')
    shutil.copy2(TOOLS/'ci_shards.py', project/'tools/ci_shards.py')
    (project/'pytest.ini').write_text('[pytest]\ntestpaths=tests\nmarkers=slow\n')
    (project/'reports/ci/durations.json').write_text(json.dumps({'tests': {}, 'rest_mean': {}}))
    (project/'reports/ci/duration-observations.json').write_text(json.dumps({
        'tests': {}, 'setup_tests': {'tests/test_cost.py::test_known': {'windows': 80.},
                                  'tests/test_cost.py::test_function_only': {'windows': 12.}}}))
    (project/'tests/test_cost.py').write_text('''import pytest
from pathlib import Path
@pytest.fixture(scope='module')
def data():
    p=Path('executions.txt');p.write_text(p.read_text()+'x' if p.exists() else 'x')
    return 1
@pytest.fixture(scope='module')
def indirect(data): return data
def test_known(data): assert data==1
@pytest.mark.parametrize('unused',(0,1))
def test_indirect(indirect,unused): assert indirect==1
@pytest.mark.parametrize('data',(9,))
def test_direct_parameter(data): assert data==9
class TestOverride:
    @pytest.fixture(scope='class')
    def data(self): return 2
    def test_fast(self,data): assert data==2
class TestParentOverride:
    @pytest.fixture(scope='class')
    def data(self,data): return data+1
    def test_slow(self,data): assert data==2
def test_unrelated(): pass
def test_builtin(tmp_path): assert tmp_path.exists()
def test_function_only(): pass
''')
    report = project/'selection.json'
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', ANYGEOMETRY_SHARD_REPORT=str(report))
    for name in ('ANYGEOMETRY_TEST_SHARD', 'ANYGEOMETRY_ACTIVE_TEST_REPORT'):
        env.pop(name, None)
    command = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
        '--rootdir='+str(project), '--confcutdir='+str(project), '-c', str(project/'pytest.ini')]
    completed = subprocess.run(command, cwd=project, env=env, capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stdout+completed.stderr
    row = json.loads(report.read_text())
    assert len(row['selected']) == len(row['ran']) == 9
    assert set(row['slow']) == {'tests/test_cost.py::test_known', 'tests/test_cost.py::test_indirect[0]',
                               'tests/test_cost.py::test_indirect[1]',
                               'tests/test_cost.py::TestParentOverride::test_slow',
                               'tests/test_cost.py::test_function_only'}
    assert (project/'executions.txt').read_text() == 'x'
    expected_slow = set(row['slow'])
    completed = subprocess.run(command+['-m', 'not slow'], cwd=project, env=env,
                               capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stdout+completed.stderr
    row = json.loads(report.read_text())
    expected_fast = {'tests/test_cost.py::TestOverride::test_fast',
                     'tests/test_cost.py::test_direct_parameter[9]',
                     'tests/test_cost.py::test_unrelated', 'tests/test_cost.py::test_builtin'}
    assert len(row['selected']) == len(row['ran']) == 4
    assert set(row['selected']) == set(row['ran']) == expected_fast
    assert row['slow'] == []
    assert (project/'executions.txt').read_text() == 'x'
    completed = subprocess.run(command+['-m', 'slow'], cwd=project, env=env,
                               capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stdout+completed.stderr
    row = json.loads(report.read_text())
    assert len(row['selected']) == len(row['ran']) == 5
    assert set(row['selected']) == set(row['ran']) == set(row['slow']) == expected_slow
    assert (project/'executions.txt').read_text() == 'xx'


def test_shards_partition_the_tests_and_balance_them():
    table = {"tests": {f"t{i}": {"windows": float(i)} for i in range(1, 31)}, "rest_mean": {"windows": .1}}
    ids = [f"t{i}" for i in range(1, 31)] + [f"u{i}" for i in range(40)]
    owner = ci_shards.assign(ids, 3, table)
    assert set(owner) == set(ids) and set(owner.values()) == {1, 2, 3}
    loads = [sum(ci_shards.duration(table, n) for n in ids if owner[n] == k) for k in (1, 2, 3)]
    assert max(loads) - min(loads) <= 30.                       # within the largest single test
    assert ci_shards.assign(ids, 3, table) == owner             # deterministic


def test_verify_finds_missing_doubled_and_unfinished_shards(tmp_path):
    calls = iter(range(100))

    def report(index, ran, selected=("a", "b", "c"), results=True):
        folder = tmp_path / f"lane-{index}-{next(calls)}"                 # a fresh folder per shard
        folder.mkdir()
        path = folder / f"shard-{index}.json"
        path.write_text(json.dumps({"index": index, "count": 2, "selected": list(selected), "ran": ran, "slow": []}))
        if results:
            _junit(folder / "kernel-results.xml", {f"test_m::{name.split('::')[-1]}": 1. for name in ran})
        return path
    ids = {n: f"tests/test_m.py::{n}" for n in "abc"}
    selected = tuple(ids.values())
    assert ci_shards.verify([report(1, [ids["a"], ids["b"]], selected), report(2, [ids["c"]], selected)]) == []
    assert any("no shard" in p for p in ci_shards.verify([report(1, [ids["a"]], selected), report(2, [ids["c"]], selected)]))
    assert any("two shards" in p for p in ci_shards.verify([report(1, [ids["a"], ids["b"]], selected),
                                                            report(2, [ids["b"], ids["c"]], selected)]))
    assert ci_shards.verify([report(1, list(selected), selected)])        # shard 2 missing
    # a shard cancelled after collection: its report exists, its results do not
    problems = ci_shards.verify([report(1, [ids["a"], ids["b"]], selected), report(2, [ids["c"]], selected, results=False)])
    assert any("cancelled or crashed" in p for p in problems)


def test_the_guard_reports_unmarked_slow_tests(tmp_path):
    junit = _junit(tmp_path / "j.xml", {"test_m::fast": 1., "test_m::marked": 50., "test_m::grown": 30.})
    report = tmp_path / "r.json"
    report.write_text(json.dumps({"slow": ["tests/test_m.py::marked"]}))
    problems = ci_shards.guard([junit], [report], 20.)
    assert len(problems) == 1 and "grown" in problems[0]


def test_the_slow_marks_follow_the_table():
    table = ci_shards.load_table()
    slow = ci_shards.slow_ids(table)
    assert slow and all(
        table["tests"].get(n, {}).get("windows", 0.) >= ci_shards.SLOW_SECONDS
        or max(table.get("observations", {}).get(n, {}).values(), default=0.) >= ci_shards.SLOW_SECONDS
        or max(table.get("failed_observations", {}).get(n, {}).values(), default=0.) >= ci_shards.SLOW_SECONDS
        or max(table.get("setup_observations", {}).get(n, {}).values(), default=0.) >= ci_shards.SLOW_SECONDS
        for n in slow)


@pytest.mark.parametrize("text", ["0/3", "4/3", "x/3"])
def test_bad_shard_specifications_are_refused(text):
    with pytest.raises(ValueError):
        ci_shards.parse_shard(text)
