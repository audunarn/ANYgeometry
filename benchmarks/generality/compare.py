"""Compare two G0 result directories under owner decision D1.

    python benchmarks/generality/compare.py <base-dir> <candidate-dir> [--bound 1e-9]

Per workload: the topology digests must be identical, the float vectors must
have the same length, the largest coordinate deviation relative to the model
extent must not exceed ``--bound``, and quality must not degrade (material area
within the same relative bound, joint residual not above the base value plus
the bound times the extent). Exit status 1 if any check fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--bound", type=float, default=1e-9)
    args = parser.parse_args()
    failed = False
    for base_file in sorted(args.base.glob("*.json")):
        name = base_file.stem
        candidate_file = args.candidate / base_file.name
        if not candidate_file.exists():
            print(f"{name:18s} MISSING in candidate")
            failed = True
            continue
        base, candidate = (json.loads(path.read_text(encoding="utf-8")) for path in (base_file, candidate_file))
        a = np.load(args.base / f"{name}.floats.npy")
        b = np.load(args.candidate / f"{name}.floats.npy")
        problems = []
        if base["topology_digest"] != candidate["topology_digest"]:
            problems.append("topology differs")
        deviation = float("nan")
        if a.shape != b.shape:
            problems.append(f"float count {a.size} vs {b.size}")
        else:
            finite = np.isfinite(a) & np.isfinite(b)
            if not np.array_equal(np.isfinite(a), np.isfinite(b)) or not np.array_equal(a[~finite], b[~finite]):
                problems.append("non-finite values differ")
            extent = max(1.0, float(np.max(np.abs(a[finite]))) if finite.any() else 1.0)
            deviation = float(np.max(np.abs(a[finite] - b[finite]))) / extent if finite.any() else 0.0
            if deviation > args.bound:
                problems.append(f"deviation {deviation:.2e} > {args.bound:.0e}")
            if "material_area" in base:
                area = abs(candidate["material_area"] - base["material_area"]) / max(1.0, abs(base["material_area"]))
                if area > args.bound:
                    problems.append(f"material area changed {area:.2e}")
                if candidate["joint_residual"] > base["joint_residual"] + args.bound * extent:
                    problems.append(f"joint residual {base['joint_residual']:.2e} -> {candidate['joint_residual']:.2e}")
        speedup = base["best_seconds"] / candidate["best_seconds"] if candidate["best_seconds"] else float("inf")
        status = "OK  " if not problems else "FAIL"
        failed |= bool(problems)
        print(f"{status} {name:18s} {base['best_seconds']:8.3f} -> {candidate['best_seconds']:8.3f} s "
              f"({speedup:5.2f}x)  max rel deviation {deviation:.1e}  {'; '.join(problems)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
