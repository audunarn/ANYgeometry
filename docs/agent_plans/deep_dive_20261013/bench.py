"""Cylinder.local_uv: current numpy vs pure Python vs C (per call and batched)."""
import math, time
import numpy as np
import kern
from anygeometry import Cylinder

cyl = Cylinder(origin=(0.1, 0.2, 0.3), axis=(0.1, 0.0, 1.0), radial_direction=(1, 0, 0), radius=0.8, height=2.0,
               start_angle=0.3, sweep_angle=4.0)
N = 20000
rng = np.random.default_rng(1)
P = rng.standard_normal((N, 3))
pts = [tuple(map(float, p)) for p in P]
frame = tuple(float(x) for x in (*cyl.origin, *cyl.axis, *cyl.radial_direction, *cyl.circumferential_direction,
                                  cyl.start_angle, cyl.sweep_angle, cyl.height))
o, a, r, c = frame[0:3], frame[3:6], frame[6:9], frame[9:12]; st, sw, h = frame[12:15]
TW = 2 * math.pi


def py_uv(x):
    d0, d1, d2 = x[0] - o[0], x[1] - o[1], x[2] - o[2]
    ax = d0 * a[0] + d1 * a[1] + d2 * a[2]
    q0, q1, q2 = d0 - ax * a[0], d1 - ax * a[1], d2 - ax * a[2]
    ang = math.atan2(q0 * c[0] + q1 * c[1] + q2 * c[2], q0 * r[0] + q1 * r[1] + q2 * r[2])
    raw = ang - st
    delta = math.fmod(raw, TW)
    if delta < 0: delta += TW
    return delta / sw, ax / h


def bench(label, fn, reps=3):
    best = min(_t(fn) for _ in range(reps))
    print(f"{label:34s} {1e9 * best / N:9.0f} ns/point")
    return best


def _t(fn):
    t = time.perf_counter(); fn(); return time.perf_counter() - t


ref = [cyl.local_uv(p) for p in pts]
out = np.empty((N, 2))
kern.many(frame, np.ascontiguousarray(P), out)
assert max(abs(x - y) for p, q in zip(ref, out.tolist()) for x, y in zip(p, q)) < 1e-12
assert max(abs(x - y) for p, q in zip(ref, map(py_uv, pts)) for x, y in zip(p, q)) < 1e-12
t0 = bench("numpy per point (current)", lambda: [cyl.local_uv(p) for p in pts])
bench("pure Python math per point", lambda: [py_uv(p) for p in pts])
bench("C per point (Python loop)", lambda: [kern.one(frame, p) for p in pts])


def np_batch():
    d = P - cyl.origin; ax = d @ cyl.axis; q = d - ax[:, None] * cyl.axis
    ang = np.arctan2(q @ cyl.circumferential_direction, q @ cyl.radial_direction)
    return np.mod(ang - cyl.start_angle, TW) / cyl.sweep_angle, ax / cyl.height


bench("numpy vectorised batch", np_batch)
bench("C batch (one call)", lambda: kern.many(frame, P, out))
