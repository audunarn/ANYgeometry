"""Evidence runs for the robust-predicate toolkit (slice R0 acceptance).

    python tools/exact_predicate_evidence.py --oracle 100000 --filters 1000000 --output <file.json>

1. Canonical enclosures (levels 0 and L_MAX) of sqrt, sin, cos, atan and atan2
   at random arguments must contain the high-precision ``decimal`` oracle of
   tests/test_exact_predicates.py (independent arithmetic and series code).
2. Filter vs canonical policy: on random predicates (many constructed within
   1e-17 .. 1e-3 of their thresholds) a filter decision must equal the
   canonical outcome; deferrals and canonical refusals are counted.
Prints and writes counts; exits non-zero on any disagreement.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from anygeometry.exact_predicates import (  # noqa: E402
    Canonical, DegenerateWithinBound, L_MAX, certified_compare, filter_compare)
from test_exact_predicates import D, _cases, contains, d_atan, d_atan2, d_sin_cos  # noqa: E402


def oracle_run(count: int, seed: int) -> dict:
    rng = random.Random(seed)
    failures = []
    engines = (Canonical(0), Canonical(L_MAX))
    for index in range(count):
        x, y = rng.uniform(-50, 50), rng.uniform(-50, 50) or 1.0
        engine = engines[index % 2]
        s, c = engine.sincos(x)
        ds, dc = d_sin_cos(D(x))
        checks = (("sin", contains(s, ds)), ("cos", contains(c, dc)),
                  ("atan", contains(engine.atan(x), d_atan(D(x)))),
                  ("atan2", contains(engine.atan2(y, x), d_atan2(D(y), D(x)))))
        failures.extend((name, x, y, engine.level) for name, ok in checks if not ok)
    return {"inputs": count, "failures": failures[:20], "failure_count": len(failures)}


def filter_run(count: int, seed: int) -> dict:
    rng = random.Random(seed)
    decided = deferred = refused = disagreements = 0
    examples = []
    for threshold, estimate, error, quantity, magnitude in _cases(rng, count):
        fast = filter_compare("evidence", estimate, error, threshold, kind="trigonometric", magnitude=magnitude)
        if fast is None:
            deferred += 1
            continue
        decided += 1
        try:
            canonical = certified_compare("evidence", quantity, threshold, kind="trigonometric", magnitude=magnitude)
        except DegenerateWithinBound:
            canonical = None
            refused += 1
        if fast != canonical:
            disagreements += 1
            if len(examples) < 20:
                examples.append((threshold, estimate, error))
    return {"predicates": count, "filter_decided": decided, "deferred": deferred,
            "canonical_refused_where_filter_decided": refused, "disagreements": disagreements, "examples": examples}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=int, default=10000)
    parser.add_argument("--filters", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    start = time.perf_counter()
    report = {"oracle": oracle_run(args.oracle, args.seed), "filters": filter_run(args.filters, args.seed + 1)}
    report["seconds"] = round(time.perf_counter() - start, 1)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    bad = report["oracle"]["failure_count"] or report["filters"]["disagreements"] \
        or report["filters"]["canonical_refused_where_filter_decided"]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
