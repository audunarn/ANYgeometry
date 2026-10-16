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
    assert slow and all(table["tests"][n]["windows"] >= ci_shards.SLOW_SECONDS for n in slow)


@pytest.mark.parametrize("text", ["0/3", "4/3", "x/3"])
def test_bad_shard_specifications_are_refused(text):
    with pytest.raises(ValueError):
        ci_shards.parse_shard(text)
