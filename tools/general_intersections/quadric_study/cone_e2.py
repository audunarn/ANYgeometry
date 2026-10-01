"""E2: how good is the certified numeric engine on cone x cylinder (0 and 10 deg off perpendicular)?"""
import sys, time, math
sys.path.insert(0, sys.argv[1])
import numpy as np
from cone_probe import build
from anygeometry import query_intersection
from anygeometry.surfaces import Cylinder, Cone

off = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
m, cf, kf = build(off)
rows = []
t_total = 0
for c in cf:
    for k in kf:
        t0 = time.perf_counter()
        r = query_intersection(m, m.handle('face', c), m.handle('face', k))
        dt = time.perf_counter() - t0
        t_total += dt
        if r.kind.name == 'CROSS':
            rows.append((c, k, r, dt))
print(f'off={off}: {len(rows)} crossing facet pairs, total {t_total:.1f}s')
c, k, r, dt = rows[0]
print('first crossing pair', c, k, 'time', f'{dt:.2f}s', 'kind', r.kind, 'dim', r.dimension, 'tol', r.tolerance_used)
print('diagnostics', r.diagnostics)
print('n components', len(r.components))
comp = r.components[0]
print(type(comp).__name__)
for name in comp.__dataclass_fields__:
    v = getattr(comp, name)
    s = repr(v)
    print('  ', name, ':', s[:200])
print('certificate:', r.certificate)
