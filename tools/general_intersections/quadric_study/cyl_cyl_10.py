"""Reference: cylinder x cylinder 10 deg off perpendicular through the exact batch engine (supported today)."""
import sys, time, math
sys.path.insert(0, sys.argv[1])
import numpy as np
from anygeometry import to_dict, query_trimmed_surface_charts, plan_intersections, apply_intersections, ConnectionIntent
from anygeometry.generators import cylinder

off = float(sys.argv[2])
segs = int(sys.argv[3]) if len(sys.argv) > 3 else 12
m = cylinder(2., 6., origin=(0., 0., -3.), circumferential_segments=segs)
a = math.radians(off)
d = (math.cos(a), 0., math.sin(a))
m.insert_model(cylinder(0.75, 5., origin=(0., 0., 0.), axis=d, radial_direction=(0., 1., 0.), circumferential_segments=segs))
t0 = time.perf_counter()
plan = plan_intersections(m, tuple(m.faces), policy=ConnectionIntent.CONNECT)
t1 = time.perf_counter()
app = apply_intersections(m, plan, policy=ConnectionIntent.CONNECT)
t2 = time.perf_counter()
print(f'cyl x cyl, axis {off} deg off perpendicular, {segs} segments each: plan {t1-t0:.2f}s apply {t2-t1:.2f}s '
      f'faces={len(m.faces)} joint_edges={len(app.joint_edges)} validate={m.validate_topology()[:2]}')
curves = {}
for e in m.edges.values():
    curves[type(e.curve).__name__] = curves.get(type(e.curve).__name__, 0) + 1
print('edge curve types:', curves)
charts = query_trimmed_surface_charts(m)
area = sum(c.material_area for c in charts.charts)
print('material area', area)
