"""E1b: quality of what the legacy imprint path persists for a cone x cylinder facet pair."""
import sys, time, math
sys.path.insert(0, sys.argv[1])
import numpy as np
from cone_probe import build
from anygeometry import query_intersection, plan_imprint, apply_imprint
from anygeometry.structural import ConnectionIntent
from anygeometry.surfaces import Cylinder, Cone

off = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
m, cf, kf = build(off)
# pick the first crossing pair
pair = None
for c in cf:
    for k in kf:
        r = query_intersection(m, m.handle('face', c), m.handle('face', k))
        if r.kind.name == 'CROSS':
            pair = (c, k); break
    if pair: break
c, k = pair
print('pair', pair)
before_edges = set(m.edges)
t0 = time.perf_counter()
plan = plan_imprint(m, m.handle('face', c), m.handle('face', k), policy=ConnectionIntent.IMPRINT)
t1 = time.perf_counter()
app = apply_imprint(m, plan, policy=ConnectionIntent.IMPRINT)
t2 = time.perf_counter()
print(f'plan {t1-t0:.2f}s apply {t2-t1:.2f}s; batch_plan={plan.batch_plan is not None}')
print('validate_topology:', m.validate_topology()[:3])
new_edges = sorted(set(m.edges) - before_edges)
print('new edges', len(new_edges), [type(m.edges[e].curve).__name__ for e in new_edges])


def residual_cyl(p, s):  # radial distance error
    off_ = p - s.origin
    ax = off_ @ s.axis
    rad = off_ - ax * s.axis
    return abs(np.linalg.norm(rad) - s.radius)


def residual_cone(p, s):
    off_ = p - s.origin
    ax = float(off_ @ s.axis)
    rad = off_ - ax * s.axis
    v = ax / s.height
    r = (1 - v) * s.radius_start + v * s.radius_end
    return abs(np.linalg.norm(rad) - r) * math.cos(math.atan2(abs(s.radius_end - s.radius_start), abs(s.height)))


cyl = m.faces[c].surface
con = m.faces[k].surface
worst_c = worst_k = 0.
for e in new_edges:
    pts = m.sample_edge(e, np.linspace(0, 1, 201))
    worst_c = max(worst_c, max(residual_cyl(p, cyl) for p in pts))
    worst_k = max(worst_k, max(residual_cone(p, con) for p in pts))
print(f'legacy-persisted edges: max distance to cylinder {worst_c:.3e}, to cone {worst_k:.3e}')
