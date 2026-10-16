"""Duration-balanced test shards for CI (CI tiers plan, slices C2 and C3).

The pytest side lives in ``tests/conftest.py``: it marks the tests that the duration table
(``reports/ci/durations.json``) records as slow, and with ``ANYGEOMETRY_TEST_SHARD=k/N`` keeps only
shard ``k`` of the tests selected for the run (after ``-m``/``-k`` deselection). Every collected test
goes to exactly one shard: longest first (Windows durations, unknown tests at the table's mean) onto
the shard with the least total, ties by node id; each shard runs its tests in collection order.
With ``ANYGEOMETRY_SHARD_REPORT=<path>`` a shard writes the ids it ran, all selected ids and the slow
ids to that JSON file.

``verify`` checks the reports of one run: identical selections, the union of the shards' ids equals
the selection and no id runs twice. ``guard`` checks JUnit results against the slow marks: a test that
is not marked slow but took longer than ``--limit`` seconds fails (refresh the duration table).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "reports" / "ci" / "durations.json"
SLOW_SECONDS = 10.0                 # Windows seconds at or above which a test is in the slow tier
BALANCE_PLATFORM = "windows"


def load_table(path: Path = TABLE) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"tests": {}, "rest_mean": {}}


def slow_ids(table: dict, threshold: float = SLOW_SECONDS) -> set[str]:
    return {name for name, times in table.get("tests", {}).items()
            if times.get(BALANCE_PLATFORM, max(times.values(), default=0.0)) >= threshold}


def duration(table: dict, name: str) -> float:
    times = table.get("tests", {}).get(name)
    if times:
        return times.get(BALANCE_PLATFORM, max(times.values()))
    return table.get("rest_mean", {}).get(BALANCE_PLATFORM, .1) or .1


def assign(ids: list[str], count: int, table: dict) -> dict[str, int]:
    """``{id: shard index 1..count}``: longest first onto the lightest shard (ties: lowest index)."""
    loads = [0.0] * count
    result = {}
    for name in sorted(ids, key=lambda n: (-duration(table, n), n)):
        index = min(range(count), key=lambda i: (loads[i], i))
        loads[index] += duration(table, name)
        result[name] = index + 1
    return result


def parse_shard(text: str) -> tuple[int, int]:
    index, _, count = text.partition("/")
    index, count = int(index), int(count)
    if not 1 <= index <= count:
        raise ValueError(f"shard {text!r}: expected k/N with 1 <= k <= N")
    return index, count


def verify(reports: list[Path], results: str = "kernel-results.xml") -> list[str]:
    """Also: next to each report, the shard's JUnit file (``results``) exists and records every test the
    shard was assigned (a cancelled or crashed shard has no complete results; its report, written at
    collection, alone proves nothing)."""
    from ci_durations import read_junit
    data = [json.loads(path.read_text(encoding="utf-8")) for path in reports]
    problems = []
    for path, item in zip(reports, data):
        junit = path.parent / results
        if not junit.is_file():
            problems.append(f"shard {item['index']}/{item['count']} ({path.parent.name}): no {results} "
                            "(cancelled or crashed)")
            continue
        recorded = set(read_junit(junit))
        absent = [n for n in item["ran"] if n not in recorded]
        if absent:
            problems.append(f"shard {item['index']}/{item['count']} ({path.parent.name}): {len(absent)} assigned "
                            f"tests have no result, e.g. {absent[0]}")
    selections = {tuple(sorted(item["selected"])) for item in data}
    if len(selections) != 1:
        problems.append("the shards saw different selections (collection differs between jobs)")
    counts = {item["count"] for item in data}
    indices = sorted(item["index"] for item in data)
    if len(counts) != 1 or indices != list(range(1, counts.pop() + 1)):
        problems.append(f"shard reports {indices} do not cover 1..N exactly once")
    selected = set(data[0]["selected"]) if data else set()
    seen: dict[str, int] = {}
    for item in data:
        for name in item["ran"]:
            seen[name] = seen.get(name, 0) + 1
    twice = sorted(n for n, k in seen.items() if k > 1)
    missing = sorted(selected - set(seen))
    extra = sorted(set(seen) - selected)
    problems += [f"ran in two shards: {n}" for n in twice[:20]]
    problems += [f"selected but run by no shard: {n}" for n in missing[:20]]
    problems += [f"run but not selected: {n}" for n in extra[:20]]
    return problems


def guard(junit: list[Path], reports: list[Path], limit: float) -> list[str]:
    from ci_durations import read_junit
    slow = set()
    for path in reports:
        slow.update(json.loads(path.read_text(encoding="utf-8")).get("slow", []))
    problems = []
    for path in junit:
        for name, seconds in read_junit(path).items():
            if seconds > limit and name not in slow:
                problems.append(f"{name}: {seconds:.1f} s, not marked slow (limit {limit:.0f} s): "
                                "refresh reports/ci/durations.json")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    v = sub.add_parser("verify")
    v.add_argument("reports", nargs="+", type=Path)
    v.add_argument("--results", default="kernel-results.xml", help="the JUnit file name next to each report")
    g = sub.add_parser("guard")
    g.add_argument("--junit", nargs="+", type=Path, required=True)
    g.add_argument("--reports", nargs="+", type=Path, required=True)
    g.add_argument("--limit", type=float, default=2 * SLOW_SECONDS)
    args = parser.parse_args(argv)
    if args.command == "verify":
        problems = verify(args.reports, args.results)
    else:
        problems = guard(args.junit, args.reports, args.limit)
    for problem in problems:
        print(problem, file=sys.stderr)
    if not problems:
        print(f"{args.command}: ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
