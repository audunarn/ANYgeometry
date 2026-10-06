"""Bounded root-owned preparation comparison; never invokes a mesher."""
import argparse
import cProfile
import hashlib
import json
import os
from pathlib import Path
import platform
import pstats
import subprocess
import sys
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_identity(source):
    return digest({p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted((source / "anygeometry").rglob("*.py"))})


def execute(args):
    source = Path(args.source).resolve()
    sys.path.insert(0, str(source))
    import anygeometry as g
    import numpy as np
    assert Path(g.__file__).resolve().is_relative_to(source)
    before = source_identity(source)
    fixture = Path(args.fixture)
    if args.mode == "fixture":
        model = g.GeometryModel()
        m, length = args.size // 2, args.size // 2 + 1.
        base = model.add_points(((0, 0, 0), (length, 0, 0), (length, length, 0), (0, length, 0)))
        model.add_sheet((model.add_plate(tuple(base)),))
        for i in range(m):
            c = i + 1.
            for points in (((0, c, 0), (length, c, 0), (length, c, .5), (0, c, .5)),
                           ((c, 0, 0), (c, length, 0), (c, length, .5), (c, 0, .5))):
                ids = model.add_points(points)
                model.add_sheet((model.add_plate(tuple(ids)),))
        fixture.write_text(json.dumps(g.to_dict(model), sort_keys=True), encoding="utf-8")
        return
    model = g.from_dict(json.loads(fixture.read_text(encoding="utf-8")))
    policy = g.IntersectionBatchPolicy(g.ConnectionIntent.CONNECT)
    from anygeometry.definition_binding import definition_checksum
    phases = {}
    profile = cProfile.Profile() if args.profile else None
    if profile:
        profile.enable()
    started = time.perf_counter()
    t = started
    if args.chart_mode == "by_face":
        rows = g.query_trimmed_surface_charts_by_face(model).results
        assert all(row.error is None and row.charts is not None for row in rows)
        charts = [row.charts for row in rows]
    else:
        charts = [g.query_trimmed_surface_charts(model, (face,)) for face in sorted(model.faces)]
    phases["queries"] = time.perf_counter() - t
    t = time.perf_counter()
    plan = g.plan_intersections(model, tuple(model.handle("sheet", i) for i in sorted(model.sheets)), policy=policy)
    phases["planning"] = time.perf_counter() - t
    t = time.perf_counter()
    applied = g.apply_intersections(model, plan, policy=policy)
    phases["application"] = time.perf_counter() - t
    elapsed = time.perf_counter() - started
    if profile:
        profile.disable()
        profile.dump_stats(str(Path(args.output).with_suffix(".pstats")))
    # Immutable chart definitions only: do not validate/evaluate stale charts.
    # This observational hash is outside the timed/profiled preparation path.
    chart_digest = digest([definition_checksum(chart) for chart in charts])
    counts = {}
    if profile:
        for (file, line, name), (cc, nc, tt, ct, callers) in pstats.Stats(profile).stats.items():
            if "anygeometry" in file and name in ("to_dict", "validate_topology", "definition_checksum", "arrange_material", "_serialized_model_state"):
                key = Path(file).name + ":" + name
                old = counts.setdefault(key, {"calls": 0, "cumulative_seconds": 0.})
                old["calls"] += nc
                old["cumulative_seconds"] += ct
    document = g.to_dict(model)
    result = {"schema": "anygeometry.parallel-handoff-measure-v1", "runtime": sys.version,
              "numpy": np.__version__, "origin": g.__file__, "machine": platform.node(),
              "source_identity": before, "source_unchanged": before == source_identity(source),
              "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
              "web_count": args.size, "profiled": args.profile, "chart_query_mode": args.chart_mode, "pipeline_seconds": elapsed,
              "phases": phases, "counts": counts, "document_digest": digest(document),
              "chart_digest": chart_digest, "joint_edges": len(applied.joint_edges),
              "topology_counts": {kind: len(getattr(model, kind)) for kind in ("vertices", "edges", "faces", "sheets", "members", "attachments", "junctions")},
              "threads": {name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


def parent(args):
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    attempts = []
    env = dict(os.environ)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        env[name] = "1"
    env["PYTHONOPTIMIZE"] = "0"
    env["PYTHONPATH"] = str(Path(args.source).resolve())
    routes = ("individual", "by_face") if args.compare_chart_modes else (args.chart_mode,)
    for route in routes:
      for size in (16, 32, 48):
        fixture = Path(args.fixture) / f"grid-{size}.json"
        for ordinal in range(3):
            remaining = 240. - sum(a["process_wall_seconds"] for a in attempts)
            if remaining <= 0:
                print(json.dumps({"status": "aggregate budget exhausted", "attempts": len(attempts)}))
                return 1
            output = root / f"{route}-grid-{size}-{ordinal}.json"
            cmd = [sys.executable, str(Path(__file__).resolve()), "--mode", "measure", "--source", args.source,
                   "--fixture", str(fixture), "--size", str(size), "--output", str(output), "--chart-mode", route]
            if ordinal == 2:
                cmd.append("--profile")
            launched = time.perf_counter()
            disposition = "completed"
            with output.with_suffix(".stdout.log").open("wb") as out, output.with_suffix(".stderr.log").open("wb") as err:
                process = subprocess.Popen(cmd, stdout=out, stderr=err, env=env)
                try:
                    code = process.wait(timeout=min(60., remaining))
                except subprocess.TimeoutExpired:
                    disposition = "timeout"
                    process.kill()
                    code = process.wait()
            attempt = {"command": cmd, "process_wall_seconds": time.perf_counter()-launched,
                       "status": disposition, "exit_code": code, "output": str(output)}
            attempts.append(attempt)
            (root / "run.json").write_text(json.dumps({"attempts": attempts, "process_seconds": sum(a["process_wall_seconds"] for a in attempts),
                        "aggregate_seconds": time.perf_counter()-started, "limit_seconds": 240., "per_process_limit_seconds": 60.}, indent=2), encoding="utf-8")
            if code:
                print(json.dumps(attempt))
                return 1
    print(json.dumps({"attempts": len(attempts), "process_seconds": sum(a["process_wall_seconds"] for a in attempts), "evidence": str(root)}))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fixture", "measure", "parent"), required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--size", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--chart-mode", choices=("individual", "by_face"), default="individual")
    parser.add_argument("--compare-chart-modes", action="store_true")
    arguments = parser.parse_args()
    if arguments.mode == "parent":
        sys.exit(parent(arguments))
    execute(arguments)
