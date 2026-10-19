"""Profile ANYgeometry workloads and classify self-time by layer.

Usage: python prof.py <workload> [--noprof]
Layers: anygeometry Python, numpy Python wrappers, numpy/C builtins, other.
"""
import cProfile, pstats, sys, time, math, json, os
from collections import defaultdict

import numpy as np
from anygeometry import ConnectionIntent, GeometryModel, apply_intersections, plan_intersections, to_dict, from_dict
from anygeometry.generators import cylinder, stiffened_panel


def _plan_apply(m):
    handles = [m.handle("face", f) for f in sorted(m.faces)]
    plan = plan_intersections(m, handles, policy=ConnectionIntent.CONNECT)
    apply_intersections(m, plan, policy=ConnectionIntent.CONNECT)
    return m


def pipes(k=6):
    m = GeometryModel()
    side = int(math.ceil(math.sqrt(k))); L = 2.0 * side + 2.0
    p = m.add_points(((0, 0, 0), (L, 0, 0), (L, L, 0), (0, L, 0))); m.add_plate(tuple(p))
    for i in range(k):
        x, y = 2.0 + 2.0 * (i % side), 2.0 + 2.0 * (i // side)
        m.insert_model(cylinder(0.4, 2.0, origin=(x + 0.013 * i, y + 0.007 * i, -1.0), axis=(0, 0, 1),
                                radial_direction=(1, 0, 0), circumferential_segments=8))
    return lambda: _plan_apply(m)


def rings(r=8):
    m = GeometryModel()
    m.insert_model(cylinder(1.0, 10.0, origin=(0, 0, 0), axis=(1, 0, 0), radial_direction=(0, 1, 0),
                            circumferential_segments=8))
    for i in range(r):
        x = 0.6 + i * 9.0 / r
        p = m.add_points(((x, -1.5, -1.5), (x, 1.5, -1.5), (x, 1.5, 1.5), (x, -1.5, 1.5))); m.add_plate(tuple(p))
    return lambda: _plan_apply(m)


def crossing_grid(n=16):
    """n/2 x n/2 crossing vertical web plates on a base plate (egg-crate)."""
    m = GeometryModel(); L = float(n // 2 + 1)
    p = m.add_points(((0, 0, 0), (L, 0, 0), (L, L, 0), (0, L, 0))); m.add_plate(tuple(p))
    for i in range(1, n // 2 + 1):
        x = float(i) + 0.01 * i
        p = m.add_points(((x, 0, 0), (x, L, 0), (x, L, 1), (x, 0, 1))); m.add_plate(tuple(p))
        y = float(i) + 0.007 * i
        p = m.add_points(((0, y, 0), (L, y, 0), (L, y, 1), (0, y, 1))); m.add_plate(tuple(p))
    return lambda: _plan_apply(m)


def tilted(n=8):
    """n mutually crossing tilted plates (generic position)."""
    m = GeometryModel(); rng = np.random.default_rng(3)
    for i in range(n):
        a = math.pi * i / n; c, s = math.cos(a), math.sin(a)
        tilt = 0.3 * rng.standard_normal()
        pts = [(-3 * c, -3 * s, -2 + tilt), (3 * c, 3 * s, -2 - tilt), (3 * c, 3 * s, 2 - tilt), (-3 * c, -3 * s, 2 + tilt)]
        pts = [(x + 0.05 * i, y + 0.03 * i, z) for x, y, z in pts]
        p = m.add_points(pts); m.add_plate(tuple(p))
    return lambda: _plan_apply(m)


def doc(n=60):
    m = GeometryModel(); pts = {}
    for i in range(n + 1):
        for j in range(n + 1):
            pts[i, j] = m.add_point(float(i), float(j), 0.0)
    for i in range(n):
        for j in range(n):
            m.add_plate((pts[i, j], pts[i + 1, j], pts[i + 1, j + 1], pts[i, j + 1]))
    def run():
        b = from_dict(to_dict(m)); b.validate_topology(); return b
    return run


W = {"pipes6": pipes, "rings8": rings, "grid16": crossing_grid, "grid32": lambda: crossing_grid(32),
     "tilted8": tilted, "doc60": doc}


def layer(key):
    fn = key[0].replace("\\", "/")
    if "anygeometry" in fn:
        return "anygeometry (Python)"
    if "/numpy/" in fn:
        return "numpy (Python wrappers)"
    if fn == "~":
        name = key[2]
        if "numpy" in name or "ndarray" in name or "linalg" in name or "ufunc" in name:
            return "numpy C calls"
        return "Python builtins"
    if "/fractions" in fn or "/decimal" in fn:
        return "fractions/decimal"
    return "other Python (stdlib/shapely)"


if __name__ == "__main__":
    name = sys.argv[1]
    run = W[name]()
    if "--noprof" in sys.argv:
        t = time.perf_counter(); r = run(); dt = time.perf_counter() - t
        print(json.dumps({"workload": name, "seconds": round(dt, 3), "faces": len(r.faces)})); sys.exit()
    pr = cProfile.Profile(); pr.enable(); t = time.perf_counter(); r = run(); dt = time.perf_counter() - t; pr.disable()
    st = pstats.Stats(pr); tot = defaultdict(float); calls = defaultdict(int)
    for key, (cc, nc, tt, ct, callers) in st.stats.items():
        tot[layer(key)] += tt; calls[layer(key)] += nc
    T = sum(tot.values())
    print(f"== {name}: {dt:.2f} s profiled, faces {len(r.faces)}")
    for k, v in sorted(tot.items(), key=lambda kv: -kv[1]):
        print(f"  {k:32s} {100*v/T:5.1f}%  calls {calls[k]:>10,d}")
    total_calls = sum(calls.values())
    print(f"  total function calls {total_calls:,d}  -> {1e6*T/total_calls:.2f} us/call")
    # top self-time functions
    items = sorted(st.stats.items(), key=lambda kv: -kv[1][2])[:18]
    for key, (cc, nc, tt, ct, _) in items:
        fn = os.path.basename(key[0]); print(f"    {100*tt/T:5.1f}%  {nc:>9,d}  {fn}:{key[1]} {key[2]}")
