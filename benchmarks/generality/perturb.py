"""Last-bit perturbation probe for decision robustness (finding F1).

    python benchmarks/generality/perturb.py [--trials 6] [workload ...]

For each workload and each perturbation site, the site's floating-point
outputs are moved by one ulp up or down (pseudo-randomly per call, seeded per
trial) and the workload's topology digest is compared with the unperturbed
run. A changed digest or a raised error means a decision depends on the last
bit of that site: the engine's answer is not robust there.

This is a measurement harness: it wraps functions in this process only and
never changes the product. Sites cover chart inversion and curve evaluation,
the quantities finding F1 showed to drive tangency/seam decisions.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import anygeometry  # noqa: E402
from anygeometry import exact_curves, surfaces  # noqa: E402
from anygeometry import arrangement_geometry  # noqa: E402

from corpus import WORKLOADS  # noqa: E402
from run import split_document  # noqa: E402

_RNG = random.Random(0)


def _nudge(value):
    """Move every float in ``value`` by one ulp in a pseudo-random direction."""
    if isinstance(value, float):
        return float(np.nextafter(value, np.inf if _RNG.random() < 0.5 else -np.inf))
    if isinstance(value, np.ndarray) and value.dtype == np.float64:
        directions = np.where(np.array([_RNG.random() < 0.5 for _ in range(value.size)]).reshape(value.shape),
                              np.inf, -np.inf)
        return np.nextafter(value, directions)
    if isinstance(value, tuple):
        return tuple(_nudge(item) for item in value)
    return value


def _wrap(owner, name):
    original = getattr(owner, name)

    def wrapped(*args, **kwargs):
        return _nudge(original(*args, **kwargs))
    return original, wrapped


SITES = {
    "plane.local_uv": (surfaces.Plane, "local_uv"),
    "cylinder.local_uv": (surfaces.Cylinder, "local_uv"),
    "cone.local_uv": (surfaces.Cone, "local_uv"),
    "ellipse.evaluate": (exact_curves.EllipticArc, "evaluate"),
    "line.evaluate": (arrangement_geometry.LinePath, "evaluate"),
}


def run_once(name):
    try:
        model = WORKLOADS[name]()()
        return split_document(model)[0]
    except Exception as error:  # noqa: BLE001 - a raised error is an outcome to report
        return f"error: {type(error).__name__}: {error}"[:160]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workloads", nargs="*", default=[name for name in WORKLOADS if name != "doc60"])
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"anygeometry": anygeometry.__file__, "trials": args.trials, "results": {}}
    for name in args.workloads:
        reference = run_once(name)
        rows = {}
        for site, (owner, attribute) in SITES.items():
            original, wrapped = _wrap(owner, attribute)
            outcomes = []
            for trial in range(args.trials):
                _RNG.seed(1000 * trial + 7)
                setattr(owner, attribute, wrapped)
                try:
                    outcomes.append(run_once(name))
                finally:
                    setattr(owner, attribute, original)
            changed = [outcome for outcome in outcomes if outcome != reference]
            rows[site] = {"changed": len(changed), "outcomes": sorted(set(changed))}
            print(f"{name:16s} {site:20s} {len(changed)}/{args.trials} changed"
                  + (f"  e.g. {changed[0]}" if changed else ""), flush=True)
        report["results"][name] = {"reference": reference, "sites": rows}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
