"""Merge repeated result directories of one tree into one, keeping the fastest run.

    python benchmarks/generality/merge_best.py <output-dir> <run-dir> [<run-dir> ...]

Every run must report the same topology digest per workload (otherwise the
merge fails); the float vector and quality of the first run are kept.
Interleave base and candidate runs on a shared machine, then merge each side.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def main() -> None:
    output, runs = Path(sys.argv[1]), [Path(item) for item in sys.argv[2:]]
    output.mkdir(parents=True, exist_ok=True)
    for first in sorted(runs[0].glob("*.json")):
        records = [json.loads((run / first.name).read_text(encoding="utf-8")) for run in runs]
        if len({record["topology_digest"] for record in records}) != 1:
            sys.exit(f"{first.stem}: topology differs between runs")
        merged = dict(records[0])
        merged["seconds"] = [value for record in records for value in record["seconds"]]
        merged["best_seconds"] = min(merged["seconds"])
        (output / first.name).write_text(json.dumps(merged, indent=2), encoding="utf-8")
        shutil.copy(runs[0] / f"{first.stem}.floats.npy", output / f"{first.stem}.floats.npy")


if __name__ == "__main__":
    main()
