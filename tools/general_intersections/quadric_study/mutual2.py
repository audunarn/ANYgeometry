import sys, math, time
sys.path.insert(0, sys.argv[1])
import numpy as np
from anygeometry import GeometryModel, plan_intersections, apply_intersections, ConnectionIntent

def build(n, concurrent, seed=1):
    rng = np.random.default_rng(seed)
    m = GeometryModel(); faces = []
    for i in range(n):
        ang = math.pi * i / n
        nrm = np.array([math.cos(ang), math.sin(ang), 0.0]); t = np.array([-math.sin(ang), math.cos(ang), 0.0])
        z = np.array([0, 0, 1.0]) + 0.3 * math.sin(3 * ang) * nrm
        shift = np.zeros(3) if concurrent else rng.uniform(-1.2, 1.2, 3)
        pts = [tuple(shift + 5 * (a * t + b * z)) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        faces.append(m.add_plate(m.add_points(pts)))
    return m, faces

for n in (12, 14, 16, 20):
    for concurrent in (True, False):
        m, f = build(n, concurrent)
        t0 = time.perf_counter()
        try:
            plan = plan_intersections(m, f, policy=ConnectionIntent.CONNECT)
            apply_intersections(m, plan, policy=ConnectionIntent.CONNECT)
            print(f'n={n:2d} {"through one point" if concurrent else "generic position  "}: OK {time.perf_counter()-t0:.2f}s faces={len(m.faces)}', flush=True)
        except Exception as e:
            print(f'n={n:2d} {"through one point" if concurrent else "generic position  "}: {type(e).__name__}: {str(e)[:90]}  ({time.perf_counter()-t0:.1f}s)', flush=True)
