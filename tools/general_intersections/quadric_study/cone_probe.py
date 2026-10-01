"""What does the current kernel do with a cone meeting a cylinder, and at what angle offset?"""
import sys, time, math
import numpy as np
from anygeometry import GeometryModel, to_dict, query_intersection
from anygeometry.generators import cylinder, cone
from anygeometry.batch_intersections import plan_intersections, apply_intersections
from anygeometry.structural import ConnectionIntent


def build(off_deg, R=2.0, r0=0.5, r1=1.0, seg_cyl=12, seg_cone=8, lift=0.0, ax_off=0.0):
    """Cylinder axis z, radius R.  Cone axis in the x-z plane, `off_deg` off the perpendicular."""
    m = cylinder(R, 6., origin=(0., 0., -3.), circumferential_segments=seg_cyl)
    cyl_faces = set(m.faces)
    a = math.radians(off_deg)
    d = np.array([math.cos(a), 0., math.sin(a)])
    radial = np.array([0., 1., 0.])
    origin = np.array([0., ax_off, lift])
    m.insert_model(cone(r0, r1, 5., origin=tuple(origin), axis=tuple(d), radial_direction=tuple(radial),
                        circumferential_segments=seg_cone))
    cone_faces = set(m.faces) - cyl_faces
    return m, sorted(cyl_faces), sorted(cone_faces)


def pair_probe(off_deg):
    m, cf, kf = build(off_deg)
    out = {}
    # find facets that really cross: try every pair with a bounded count
    t0 = time.perf_counter()
    kinds = {}
    for c in cf:
        for k in kf:
            try:
                r = query_intersection(m, m.handle('face', c), m.handle('face', k))
            except Exception as e:
                kinds[f'EXC {type(e).__name__}'] = kinds.get(f'EXC {type(e).__name__}', 0) + 1
                continue
            key = f'{r.kind.name}/{r.dimension.name if r.dimension else None}' + ('' if not r.diagnostics else ' [' + r.diagnostics[0][:70] + ']')
            kinds[key] = kinds.get(key, 0) + 1
    return kinds, time.perf_counter() - t0


if __name__ == '__main__':
    for off in (0.0, 10.0):
        kinds, t = pair_probe(off)
        print(f'off={off:4.1f}deg  query_intersection over all cylinder x cone facet pairs: {kinds}  ({t:.1f}s)', flush=True)
        m, cf, kf = build(off)
        for policy in (ConnectionIntent.CONNECT,):
            t0 = time.perf_counter()
            try:
                plan = plan_intersections(m, [m.handle('face', f) for f in cf + kf], policy=policy)
                plan_t = time.perf_counter() - t0
                print(f'   plan_intersections OK in {plan_t:.2f}s', flush=True)
            except Exception as e:
                print(f'   plan_intersections -> {type(e).__name__}: {str(e)[:300]}  ({time.perf_counter()-t0:.2f}s)', flush=True)
