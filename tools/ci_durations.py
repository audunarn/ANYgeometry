"""Per-test CI durations from JUnit results (CI tiers plan, slices C0, C2 and C7).

``build``  : JUnit files of one CI run (``label=path``, one per platform) -> ``reports/ci/durations.json``
             with the run's totals per platform, every test of at least ``--min-seconds`` on any platform,
             and the mean duration of the remaining tests per platform (for balancing unknown tests).
``check``  : the totals recorded in the table are reproduced by its own entries (C0 acceptance).
``guard``  : a JUnit file from a run of the fast tier: tests slower than ``--limit`` seconds fail the
             step (C2: a test that became slow must be marked by refreshing the table).
``summary``: a Markdown table of the slowest tests of a JUnit file and the soft budgets (C7), for the
             GitHub job summary.

Node ids follow pytest: ``tests/<module>.py::<name>`` (JUnit ``classname`` ``tests.<module>``, or
``tests.<module>.<Class>`` -> ``tests/<module>.py::<Class>::<name>``).
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "reports" / "ci" / "durations.json"


def node_id(classname: str, name: str) -> str:
    parts = classname.split(".")
    for split in range(len(parts), 0, -1):
        module = Path(*parts[:split]).with_suffix(".py")
        if (ROOT / module).is_file() or split == 2:
            rest = parts[split:]
            return "::".join([module.as_posix(), *rest, name])
    return classname.replace(".", "/") + ".py::" + name


def read_junit(path: Path) -> dict[str, float]:
    """``{node id: seconds}`` of the test cases of a JUnit file (skipped ones included at their time)."""
    root = ET.parse(path).getroot()
    result: dict[str, float] = {}
    for case in root.iter("testcase"):
        result[node_id(case.get("classname", ""), case.get("name", ""))] = float(case.get("time", 0) or 0)
    return result


def build(sources: list[str], min_seconds: float, run: str | None) -> dict:
    platforms: dict[str, dict[str, float]] = {}
    for item in sources:
        label, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"expected label=path, got {item!r}")
        platforms[label] = read_junit(Path(path))
    names = set().union(*platforms.values())
    slow = sorted(n for n in names if any(p.get(n, 0.0) >= min_seconds for p in platforms.values()))
    table = {"run": run, "min_seconds": min_seconds, "platforms": sorted(platforms), "totals": {}, "rest_mean": {},
             "tests": {n: {label: round(platforms[label].get(n, 0.0), 3) for label in sorted(platforms)} for n in slow}}
    for label, times in platforms.items():
        rest = [t for n, t in times.items() if n not in table["tests"]]
        table["totals"][label] = {"tests": len(times), "seconds": round(sum(times.values()), 3)}
        table["rest_mean"][label] = round(sum(rest) / len(rest), 6) if rest else 0.0
    return table


def check(table: dict) -> list[str]:
    """Each platform's total against its slow entries plus the rest at their mean (within 2 %)."""
    problems = []
    for label, total in table["totals"].items():
        listed = sum(entry[label] for entry in table["tests"].values())
        rest = (total["tests"] - len(table["tests"])) * table["rest_mean"][label]
        if abs(listed + rest - total["seconds"]) > .02 * total["seconds"]:
            problems.append(f"{label}: {listed + rest:.1f} s reproduced, {total['seconds']:.1f} s recorded")
    return problems


def guard(path: Path, limit: float) -> list[str]:
    return [f"{n}: {t:.1f} s (unmarked, limit {limit:.0f} s): refresh reports/ci/durations.json"
            for n, t in sorted(read_junit(path).items(), key=lambda item: -item[1]) if t > limit]


def summary(path: Path, count: int, budgets: dict[str, float]) -> str:
    times = read_junit(path)
    total = sum(times.values())
    lines = [f"### Slowest {count} tests ({len(times)} tests, {total / 60:.1f} min)", "",
             "| seconds | test |", "| ---: | --- |"]
    lines += [f"| {t:.1f} | `{n}` |" for n, t in sorted(times.items(), key=lambda item: -item[1])[:count]]
    over = [f"- {name}: {value:.1f} s exceeds the soft budget of {budgets[name]:.0f} s"
            for name, value in (("slowest test", max(times.values(), default=0.0)),) if value > budgets[name]]
    lines += ["", "Soft budgets (reported, not enforced): " + ", ".join(f"{k} {v:.0f} s" for k, v in budgets.items())]
    lines += over or ["- within the budgets"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("sources", nargs="+", help="label=junit.xml")
    b.add_argument("--min-seconds", type=float, default=1.0)
    b.add_argument("--run", help="the CI run the files come from")
    b.add_argument("--output", type=Path, default=TABLE)
    c = sub.add_parser("check")
    c.add_argument("--table", type=Path, default=TABLE)
    g = sub.add_parser("guard")
    g.add_argument("junit", type=Path)
    g.add_argument("--limit", type=float, default=20.0)
    s = sub.add_parser("summary")
    s.add_argument("junit", type=Path)
    s.add_argument("--count", type=int, default=20)
    s.add_argument("--slowest-budget", type=float, default=120.0)
    args = parser.parse_args(argv)
    if args.command == "build":
        table = build(args.sources, args.min_seconds, args.run)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        problems = check(table)
    elif args.command == "check":
        problems = check(json.loads(args.table.read_text(encoding="utf-8")))
    elif args.command == "guard":
        problems = guard(args.junit, args.limit)
    else:
        sys.stdout.write(summary(args.junit, args.count, {"slowest test": args.slowest_budget}))
        return 0
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
