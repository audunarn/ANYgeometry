"""Tracer: do cone and cylinder facets arrange correctly when cut by the exact cone x cylinder curve?

Exact branches (prototype) are fitted by Bernstein polynomials to 1e-13 so the existing Bezier
junction/arrangement machinery can consume them; this isolates the *chart* generalization
(Cone domain) from the *curve-class* generalization.
"""
import sys, math, time, os
sys.path.insert(0, sys.argv[1])
import numpy as np
from math import comb
from anygeometry import GeometryModel
from anygeometry.generators import cylinder, cone
from anygeometry.material_arrangement import MaterialDomain, ArrangementPath, arrange_material
from anygeometry.arrangement_geometry import BezierPath
import anygeometry._quadric_branch as qb

off = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
SEG_CYL = int(sys.argv[3]) if len(sys.argv) > 3 else 12
SEG_CONE = int(sys.argv[4]) if len(sys.argv) > 4 else 8
R, R0, R1, H = 2.0, 0.5, 1.0, 5.0
FIT_TOL = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-12
FIT_DEGREE = int(sys.argv[6]) if len(sys.argv) > 6 else 9
BUDGET = float(sys.argv[7]) if len(sys.argv) > 7 else 120.

m = cylinder(R, 6., origin=(0., 0., -3.), circumferential_segments=SEG_CYL)
cyl_faces = sorted(m.faces)
a = math.radians(off)
m.insert_model(cone(R0, R1, H, origin=(0., 0., 0.), axis=(math.cos(a), 0., math.sin(a)), radial_direction=(0., 1., 0.),
                    circumferential_segments=SEG_CONE))
cone_faces = sorted(set(m.faces) - set(cyl_faces))


def bernstein_matrix(ts, n):
    return np.array([[comb(n, i) * t ** i * (1 - t) ** (n - i) for i in range(n + 1)] for t in ts])


def fit_piece(curve, lo, hi, degree):
    nodes = .5 * (1 + np.cos(np.pi * (2 * np.arange(degree + 1) + 1) / (2 * (degree + 1))))   # Chebyshev nodes in [0,1]
    ts = lo + (hi - lo) * np.sort(nodes)
    ts[0], ts[-1] = lo, hi
    pts = curve.evaluate(ts)
    local = (ts - lo) / (hi - lo)
    controls = np.linalg.solve(bernstein_matrix(local, degree), pts)
    return controls


def bezier_error(controls, curve, lo, hi, n=400):
    ts = np.linspace(0, 1, n)
    b = BezierPath(tuple(map(tuple, controls))).evaluate(ts)
    return float(np.max(np.linalg.norm(b - curve.evaluate(lo + (hi - lo) * ts), axis=1)))


def fit_chart(curve, tol=1e-12, degree=9):
    pieces = 1
    while True:
        edges = np.linspace(0, 1, pieces + 1)
        fits = [fit_piece(curve, edges[i], edges[i + 1], degree) for i in range(pieces)]
        if max(bezier_error(f, curve, edges[i], edges[i + 1]) for i, f in enumerate(fits)) <= tol or pieces >= 64:
            return [BezierPath(tuple(map(tuple, f))) for f in fits], pieces
        pieces *= 2


# traces: every cone-facet x cylinder-facet support intersection (exact), fitted
t0 = time.perf_counter()
traces_for = {f: [] for f in cone_faces + cyl_faces}
n_charts = n_pieces = 0
for cf in cyl_faces:
    for kf in cone_faces:
        charts, _info = qb.support_intersection(m.faces[kf].surface, m.faces[cf].surface)
        for chart in charts:
            n_charts += 1
            beziers, pieces = fit_chart(chart, tol=FIT_TOL, degree=FIT_DEGREE)
            n_pieces += len(beziers)
            for bz in beziers:
                path = ArrangementPath(bz, owners=(kf, cf))
                traces_for[kf].append(path); traces_for[cf].append(path)
t_traces = time.perf_counter() - t0
print(f'exact charts {n_charts} -> {n_pieces} fitted Bezier pieces (tol {FIT_TOL}, degree {FIT_DEGREE}) in {t_traces:.2f}s', flush=True)
print('traces per face', {f: len(v) for f, v in traces_for.items() if v}, flush=True)

# arrangement per face
report = []
for fid in (cone_faces + cyl_faces if os.environ.get('TRACER_CYL') else cone_faces):
    if not traces_for[fid]:
        continue
    domain = MaterialDomain.from_model(m, fid)
    t0 = time.perf_counter()
    try:
        boxes = [path.curve.bounds() for loop in domain.boundaries for path in loop]
        length = float(np.linalg.norm(np.max([b[1] for b in boxes], axis=0) - np.min([b[0] for b in boxes], axis=0)))
        arr = arrange_material(domain, traces_for[fid], tolerance=m.tolerance.effective_length(length),
                               area_tolerance=m.tolerance.effective_area(length),
                               cancellation_check=lambda: time.perf_counter() - t0 > BUDGET)
    except Exception as e:
        report.append((fid, type(domain.support).__name__, 'FAILED', f'{type(e).__name__}: {str(e)[:100]}', time.perf_counter() - t0))
        print('face %3d %-9s %s %s %.2fs' % report[-1], flush=True)
        continue
    holes = sum(len(c.holes) for c in arr.cells)
    report.append((fid, type(domain.support).__name__, len(arr.cells), holes, time.perf_counter() - t0))
    print('face %3d %-9s cells=%s holes=%s  %.2fs' % report[-1], flush=True)
ok = sum(1 for r in report if r[2] != 'FAILED')
print(f'{ok}/{len(report)} faces arranged (off={off} deg)')
