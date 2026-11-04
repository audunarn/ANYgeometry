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


def test_the_table_reproduces_its_totals(tmp_path):
    a = _junit(tmp_path / "a.xml", {"test_m::t1": 12., "test_m::t2": .2, "test_m::t3": .4})
    b = _junit(tmp_path / "b.xml", {"test_m::t1": 6., "test_m::t2": .1, "test_m::t3": .3})
    table = ci_durations.build([f"windows={a}", f"ubuntu={b}"], 1.0, "run")
    assert set(table["tests"]) == {"tests/test_m.py::t1"}
    assert table["totals"]["windows"] == {"tests": 3, "seconds": 12.6}
    assert ci_durations.check(table) == []
    table["totals"]["windows"]["seconds"] = 20.
    assert ci_durations.check(table)


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
    assert slow and all(ci_shards.duration(table, n) >= ci_shards.SLOW_SECONDS for n in slow)


@pytest.mark.parametrize("text", ["0/3", "4/3", "x/3"])
def test_bad_shard_specifications_are_refused(text):
    with pytest.raises(ValueError):
        ci_shards.parse_shard(text)


def _observation(node, seconds, outcome="passed", runtime="generic"):
    return {"node": node, "seconds": seconds, "outcome": outcome, "runtime": runtime,
            "artifact": "artifact-1", "source_run": "run", "head_sha": "head"}


def test_positive_refresh_preserves_history_and_separate_runtime_evidence():
    table = {"tests": {"old": {"windows": 12.}}, "totals": {"windows": {"tests": 1, "seconds": 12.}},
             "rest_mean": {"windows": .1}}
    records = [_observation("new", 20.), _observation("new", 2.),
               _observation("new", 40., runtime="qualified"),
               _observation("old", 1.), _observation("failed", 999., "failed"),
               _observation("skipped", 999., "skipped"), _observation("setup_error", 999., "error")]
    refreshed = ci_durations.refresh(table, records, {"run": "run", "head_sha": "head"})
    assert refreshed["tests"] == table["tests"]
    assert refreshed["totals"] == table["totals"]
    assert "scheduling" not in table
    evidence = refreshed["scheduling"]["tests"]
    assert set(evidence) == {"new", "old"}
    assert evidence["new"]["generic"] == {"minimum_seconds": 2., "maximum_seconds": 20.,
                                          "observations": 2, "artifacts": ["artifact-1"]}
    assert evidence["new"]["qualified"]["maximum_seconds"] == 40.
    assert ci_shards.duration(refreshed, "new") == 40.
    assert ci_shards.duration(refreshed, "old") == 12.
    assert ci_shards.duration(refreshed, "unknown") == .1
    assert "new" in ci_shards.slow_ids(refreshed)
    assert ci_durations.check(refreshed) == []


@pytest.mark.parametrize("change", [{"seconds": -1.}, {"seconds": float("nan")},
                                    {"source_run": "other"}, {"head_sha": "other"}])
def test_positive_refresh_rejects_invalid_or_mismatched_evidence(change):
    record = dict(_observation("new", 1.), **change)
    with pytest.raises(ValueError):
        ci_durations.refresh({"tests": {}}, [record], {"run": "run", "head_sha": "head"})


def test_runtime_parameter_aliases_remain_distinct_in_sixteen_shards():
    ids = [f"tests/test_m.py::test_x[{value}]" for value in ("True", "np.True_", "int|str", "Union[int,str]")]
    ids += [f"unknown-{index}" for index in range(30)]
    table = {"tests": {ids[0]: {"windows": 20.}}, "rest_mean": {"windows": .1}}
    assignment = ci_shards.assign(ids, 16, table)
    partitions = [{name for name in ids if assignment[name] == index} for index in range(1, 17)]
    assert set.union(*partitions) == set(ids)
    assert sum(map(len, partitions)) == len(ids)
    assert all(not left & right for i, left in enumerate(partitions) for right in partitions[i+1:])
    assert ci_shards.assign(list(reversed(ids)), 16, table) == assignment


def test_full_ci_keeps_every_runtime_and_sixteen_shards_under_original_caps():
    text = (TOOLS.parent / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    regular = text.split("\n  pytest:\n", 1)[1].split("\n  pytest-shards:\n", 1)[0]
    minimum = text.split("\n  minimum-dependencies:\n", 1)[1].split("\n  wheel-smoke:\n", 1)[0]
    assert "os: [windows-latest, ubuntu-latest, macos-15, macos-15-intel]" in regular
    assert 'python-version: ["3.11", "3.12", "3.13", "3.14"]' in regular
    assert "python-version: '3.11'" in minimum and "runs-on: ubuntu-latest" in minimum
    for job in (regular, minimum):
        assert "shard: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]" in job
        assert "exclude:" not in job
        assert "timeout-minutes: 30" in job
        assert "ANYGEOMETRY_TEST_SHARD: ${{ matrix.shard }}/16" in job
        assert "--junitxml=kernel-results.xml" in job
    gate = text.split("\n  pytest-shards:\n", 1)[1].split("\n  planar-absence:\n", 1)[0]
    assert "needs: [pytest, minimum-dependencies]" in gate
    assert 'if [ "$PYTEST_RESULT" != success ]' in gate
    assert 'if [ "$MINIMUM_RESULT" != success ]' in gate
    assert "python tools/ci_shards.py verify lanes/kernel-runtime-$os-$py-*/shard-*.json" in gate
    assert "python tools/ci_shards.py verify lanes/minimum-dependencies-runtime-*/shard-*.json" in gate


def test_sixteen_partition_reports_still_require_every_assigned_junit_result(tmp_path):
    ids = [f"tests/test_m.py::t{i}" for i in range(16)]
    reports = []
    for index, node in enumerate(ids, 1):
        folder = tmp_path / str(index); folder.mkdir()
        path = folder / f"shard-{index}.json"
        path.write_text(json.dumps(dict(index=index, count=16, selected=ids, ran=[node], slow=[])))
        _junit(folder / "kernel-results.xml", {f"test_m::t{index-1}": .1})
        reports.append(path)
    assert ci_shards.verify(reports) == []
    _junit(reports[-1].parent / "kernel-results.xml", {})
    assert any("assigned tests have no result" in problem for problem in ci_shards.verify(reports))
