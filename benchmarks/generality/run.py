"""Run G0 workloads and record time, topology digest, coordinates and quality.

    python benchmarks/generality/run.py --output <dir> [--repeat 3] [workload ...]

Writes ``<dir>/<workload>.json`` and ``<dir>/<workload>.floats.npy``. The
topology digest hashes the canonical document with every float replaced by a
marker, so it is unchanged by coordinate-only differences; the float vector
(document order) is kept separately so ``compare.py`` can bound the deviation
(owner decision D1: identical topology, coordinates within a stated bound).
"""
from __future__ import annotations

import argparse
import collections
import math
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import anygeometry  # noqa: E402
from anygeometry import query_trimmed_surface_charts, to_dict  # noqa: E402

from corpus import QUALITY, WORKLOADS  # noqa: E402


def split_document(model):
    """Return (topology digest, float vector) of the canonical document."""
    document = dict(to_dict(model))
    document.pop("model_id", None)
    document.pop("checksum", None)
    floats: list[float] = []

    def strip(value):
        if isinstance(value, float):
            floats.append(value)
            return "<f>"
        if isinstance(value, dict):
            return {key: strip(value[key]) for key in sorted(value)}
        if isinstance(value, (list, tuple)):
            return [strip(item) for item in value]
        return value

    skeleton = strip(document)
    digest = hashlib.sha256(json.dumps(skeleton, sort_keys=True, default=str).encode()).hexdigest()
    return digest[:16], np.asarray(floats, dtype=float)


def quality(model):
    """Material area and the largest distance of sampled edge points from their faces.

    Faces whose trimmed chart cannot be qualified are counted, not hidden.
    """
    areas, failures = [], []
    for face_id in sorted(model.faces):
        try:
            charts = query_trimmed_surface_charts(model, [model.handle("face", face_id)]).charts
        except anygeometry.GeometryError as error:
            failures.append([face_id, str(error)])
            continue
        areas.extend(chart.material_area for chart in charts)
    owners = collections.defaultdict(set)
    for face_id, face in model.faces.items():
        for use in (*face.loop, *(item for hole in face.holes for item in hole)):
            owners[use.edge].add(face_id)
    worst = 0.0
    samples = np.linspace(0.0, 1.0, 9)
    for edge_id, faces in owners.items():
        points = model.sample_edge(edge_id, samples)
        for face_id in faces:
            _, _, distance = model.project_to_face_many(face_id, points)
            worst = max(worst, float(np.max(distance)))
    return {"material_area": math.fsum(areas), "joint_residual": worst, "chart_failures": failures}


def measure(name: str, repeat: int, output: Path, with_quality: bool) -> dict:
    times = []
    result = None
    for _ in range(repeat):
        run = WORKLOADS[name]()
        start = time.perf_counter()
        result = run()
        times.append(time.perf_counter() - start)
    digest, floats = split_document(result)
    record = {
        "workload": name,
        "best_seconds": round(min(times), 4),
        "seconds": [round(value, 4) for value in times],
        "topology_digest": digest,
        "float_count": int(floats.size),
        "faces": len(result.faces),
        "edges": len(result.edges),
        "vertices": len(result.vertices),
        "anygeometry": anygeometry.__file__,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    if with_quality and name in QUALITY:
        record.update(quality(result))
    np.save(output / f"{name}.floats.npy", floats)
    (output / f"{name}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workloads", nargs="*", default=list(WORKLOADS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--no-quality", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in args.workloads:
        record = measure(name, args.repeat, args.output, not args.no_quality)
        print(json.dumps({key: record[key] for key in ("workload", "best_seconds", "topology_digest", "faces")}
                         | {key: record[key] for key in ("material_area", "joint_residual") if key in record}),
              flush=True)


if __name__ == "__main__":
    main()
