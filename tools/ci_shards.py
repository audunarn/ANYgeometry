"""Duration-balanced test shards for CI (CI tiers plan, slices C2 and C3).

The pytest side lives in ``tests/conftest.py``: it marks the tests that the duration table
(``reports/ci/durations.json``) records as slow, and with ``ANYGEOMETRY_TEST_SHARD=k/N`` keeps only
shard ``k`` of the tests selected for the run (after ``-m``/``-k`` deselection). Every collected test
goes to exactly one shard: longest first (the maximum of historical Windows cost and passed
observations across separately retained runtimes, unknown tests at the historical mean) onto
the shard with the least total, ties by node id; each shard runs its tests in collection order.
With ``ANYGEOMETRY_SHARD_REPORT=<path>`` a shard writes the ids it ran, all selected ids and the slow
ids to that JSON file.

Costs are scheduling policy estimates from incomplete historical observations. They are neither
measured full-suite job durations nor a guarantee that a shard fits its job cap.

The node policy remains the default. CI can opt into the module policy with an explicit list
of modules whose collected items use module/class-scoped fixtures. Those modules are indivisible;
fixture-free modules retain node assignment. Collection order within each shard is unchanged.

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
    names = (set(table.get("tests", {})) | set(table.get("scheduling", {}).get("tests", {}))
             | set(table.get("scheduling", {}).get("protocol_tests", {})))
    return {name for name in names if duration(table, name) >= threshold}


def duration(table: dict, name: str) -> float:
    times = table.get("tests", {}).get(name)
    if times:
        historical = times.get(BALANCE_PLATFORM, max(times.values()))
    else:
        historical = table.get("rest_mean", {}).get(BALANCE_PLATFORM, .1) or .1
    observed = table.get("scheduling", {}).get("tests", {}).get(name, {})
    protocols = table.get("scheduling", {}).get("protocol_tests", {}).get(name, {})
    return max(historical, max((bucket["maximum_seconds"] for bucket in observed.values()), default=0.0),
               max((bucket["maximum_seconds"] for bucket in protocols.values()), default=0.0))


def grouping_policy(value='node'):
    if value not in ('node','module'):
        raise ValueError('unknown CI shard grouping: '+str(value))
    return value


def group_modules_policy(grouping,modules=None):
    grouping=grouping_policy(grouping)
    if modules is None:return None if grouping=='module' else []
    if type(modules) is not list or any(type(name) is not str for name in modules) or len(modules)!=len(set(modules)):
        raise ValueError('invalid scoped fixture module list')
    if any(type(name) is not str or not name.startswith('tests/') or not name.endswith('.py')
           or '\\' in name or '..' in Path(name).parts for name in modules):
        raise ValueError('invalid scoped fixture module path')
    if grouping=='node' and modules:raise ValueError('node grouping cannot admit fixture modules')
    return sorted(modules)


def fixture_group_modules(items):
    """Read pytest's resolved used fixtures, including transitive/autouse fixtures."""
    modules=set()
    for item in items:
        info=getattr(item,'_fixtureinfo',None)
        definitions=getattr(info,'name2fixturedefs',{})
        if any(fixture.scope in ('module','class') for choices in definitions.values() for fixture in (choices or ())):
            modules.add(item.nodeid.split('::',1)[0])
    return sorted(modules)


def assignment_groups(ids,table,grouping,group_modules=None):
    grouping=grouping_policy(grouping)
    admitted=group_modules_policy(grouping,group_modules)
    if grouping=='node':return [(node,[node],duration(table,node)) for node in ids]
    if admitted is not None and not set(admitted)<={node.split('::',1)[0] for node in ids}:
        raise ValueError('admitted fixture module outside selected inventory')
    groups={}
    for node in ids:
        if '::' not in node:raise ValueError('module grouping requires test node IDs')
        module=node.split('::',1)[0]
        groups.setdefault(module if admitted is None or module in admitted else node,[]).append(node)
    return [(name,sorted(nodes),sum(duration(table,node) for node in sorted(nodes))) for name,nodes in groups.items()]


def assign(ids: list[str], count: int, table: dict, *, grouping='node',group_modules=None) -> dict[str, int]:
    """``{id: shard index 1..count}``: longest first onto the lightest shard (ties: lowest index)."""
    loads = [0.0] * count
    result = {}
    for name,nodes,cost in sorted(assignment_groups(ids,table,grouping,group_modules),key=lambda group:(-group[2],group[0])):
        index = min(range(count), key=lambda i: (loads[i], i))
        loads[index] += cost
        for node in nodes:result[node] = index + 1
    return result


def scheduling_receipt(ids: list[str], count: int, table: dict, *, setup_reserve_seconds=180, cap_seconds=1800, grouping='node',group_modules=None):
    """Policy estimates plus explicit setup/collection/phase/artifact reserve.

    Reserve is a planning allowance, not a measured bound. A passing estimate
    cannot establish actual fit, especially while individual cases are unknown.
    """
    if type(count) is not int or count < 1 or not 0 <= setup_reserve_seconds < cap_seconds <= 1800:
        raise ValueError('invalid scheduling envelope')
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('invalid scheduling inventory')
    grouping=grouping_policy(grouping)
    admitted=group_modules_policy(grouping,group_modules)
    owner = assign(ids, count, table,grouping=grouping,group_modules=admitted)
    groups=assignment_groups(ids,table,grouping,admitted)
    largest=min(groups,key=lambda group:(-group[2],group[0]))
    largest_module=(min(assignment_groups(ids,table,'module'),key=lambda group:(-group[2],group[0]))
                    if all('::' in node for node in ids) else largest)
    loads = [sum(duration(table, node) for node in ids if owner[node] == index) for index in range(1, count+1)]
    known = (set(table.get('tests', {})) | set(table.get('scheduling', {}).get('tests', {}))
             | set(table.get('scheduling', {}).get('protocol_tests', {})))
    return dict(schema='ci-scheduling-estimate-v1', scope='policy-estimate-only-not-measured-fit',
        cap_seconds=cap_seconds, setup_reserve_seconds=setup_reserve_seconds,grouping=grouping,group_modules=admitted,
        group_count=len(groups),largest_group=dict(key=largest[0],nodes=len(largest[1]),estimated_seconds=largest[2]),
        largest_module=dict(key=largest_module[0],nodes=len(largest_module[1]),estimated_seconds=largest_module[2]),
        reserve_scope='setup, collection, serial interpreter startup and artifact allowance',
        estimated_test_seconds=loads,
        estimated_with_reserve_seconds=[cost+setup_reserve_seconds for cost in loads],
        over_estimate_shards=[index for index, cost in enumerate(loads, 1) if cost+setup_reserve_seconds > cap_seconds],
        unobserved_count=sum(node not in known for node in ids), measured_fit=False, release_acceptance=False)


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
    policies=[];admissions=[]
    for item in data:
        try:policies.append(grouping_policy(item.get('grouping','node')))
        except ValueError as error:problems.append(str(error));policies.append(None)
        try:
            if policies[-1]=='module' and not isinstance(item.get('group_modules'),list):raise ValueError('module policy lacks admitted fixture list')
            admitted=group_modules_policy(policies[-1],item.get('group_modules',[]))
            if not set(admitted)<={node.split('::',1)[0] for node in item['selected']}:raise ValueError('admitted module outside selection')
            admissions.append(admitted)
        except (ValueError,TypeError) as error:problems.append(str(error));admissions.append(None)
    if len(set(policies))!=1:problems.append('the shards used different grouping policies')
    if admissions and any(value!=admissions[0] for value in admissions):problems.append('the shards admitted different fixture module lists')
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
    if policies and set(policies)=={'module'}:
        owners={}
        for item in data:
            for node in item['ran']:
                module=node.split('::',1)[0]
                if module in (admissions[0] or []):owners.setdefault(module,set()).add(item['index'])
        problems += ['module split across shards: '+name for name,indices in sorted(owners.items()) if len(indices)>1]
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
