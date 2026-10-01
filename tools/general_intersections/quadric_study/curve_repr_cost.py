"""Cost of the curve representation inside the exact arrangement: analytic branch vs fitted Bezier chain.

Same cylinder x cylinder (10 deg off perpendicular) facets, same arrangement engine, same tolerances;
only the trace curve class differs.
"""
import sys, math, time
sys.path.insert(0, sys.argv[1])
import numpy as np
from math import comb
from anygeometry.generators import cylinder
from anygeometry.material_arrangement import MaterialDomain, ArrangementPath, arrange_material
from anygeometry.arrangement_geometry import BezierPath
from anygeometry.analytic_supports import cylinder_cylinder_support

DEGREE = int(sys.argv[2]) if len(sys.argv) > 2 else 5
TOL = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-9
off = 10.0
segs = 8
m = cylinder(2., 6., origin=(0., 0., -3.), circumferential_segments=segs)
base = sorted(m.faces)
a = math.radians(off)
m.insert_model(cylinder(0.75, 5., origin=(0., 0., 0.), axis=(math.cos(a), 0., math.sin(a)), radial_direction=(0., 1., 0.),
                        circumferential_segments=segs))
other = sorted(set(m.faces) - set(base))


def bern(ts, n):
    return np.array([[comb(n, i) * t ** i * (1 - t) ** (n - i) for i in range(n + 1)] for t in ts])


def fit(curve, tol, degree):
    pieces = 1
    while True:
        edges = np.linspace(0, 1, pieces + 1)
        out, worst = [], 0.
        for i in range(pieces):
            lo, hi = edges[i], edges[i + 1]
            nodes = np.sort(.5 * (1 + np.cos(np.pi * (2 * np.arange(degree + 1) + 1) / (2 * (degree + 1)))))
            ts = lo + (hi - lo) * nodes; ts[0], ts[-1] = lo, hi
            local = (ts - lo) / (hi - lo)
            controls = np.linalg.solve(bern(local, degree), curve.evaluate(ts))
            probe = np.linspace(0, 1, 200)
            err = np.max(np.linalg.norm(BezierPath(tuple(map(tuple, controls))).evaluate(probe) - curve.evaluate(lo + (hi - lo) * probe), axis=1))
            worst = max(worst, float(err)); out.append(controls)
        if worst <= tol or pieces >= 64:
            return [BezierPath(tuple(map(tuple, c))) for c in out]
        pieces *= 2


exact_traces = {f: [] for f in base + other}
bezier_traces = {f: [] for f in base + other}
for a_face in base:
    for b_face in other:
        for curve in cylinder_cylinder_support(m.faces[a_face].surface, m.faces[b_face].surface).curves:
            for f in (a_face, b_face):
                exact_traces[f].append(ArrangementPath(curve, owners=(a_face, b_face)))
            for bz in fit(curve, TOL, DEGREE):
                for f in (a_face, b_face):
                    bezier_traces[f].append(ArrangementPath(bz, owners=(a_face, b_face)))

rows = []
for fid in base[:3] + other[:3]:
    if not exact_traces[fid]:
        continue
    domain = MaterialDomain.from_model(m, fid)
    boxes = [p.curve.bounds() for loop in domain.boundaries for p in loop]
    length = float(np.linalg.norm(np.max([b[1] for b in boxes], axis=0) - np.min([b[0] for b in boxes], axis=0)))
    kw = dict(tolerance=m.tolerance.effective_length(length), area_tolerance=m.tolerance.effective_area(length))
    t0 = time.perf_counter(); ea = arrange_material(domain, exact_traces[fid], **kw); te = time.perf_counter() - t0
    t0 = time.perf_counter()
    try:
        bz = arrange_material(domain, bezier_traces[fid], **kw, cancellation_check=lambda: time.perf_counter() - t0 > 120)
        tb = time.perf_counter() - t0; bcells = len(bz.cells)
    except Exception as e:
        tb = time.perf_counter() - t0; bcells = type(e).__name__
    print(f'face {fid:3d} {type(domain.support).__name__:9s} analytic traces={len(exact_traces[fid]):2d} cells={len(ea.cells)} {te:7.3f}s | '
          f'Bezier(deg {DEGREE}, tol {TOL:g}) pieces={len(bezier_traces[fid]):3d} cells={bcells} {tb:7.3f}s | slowdown x{tb / max(te, 1e-9):.0f}', flush=True)
