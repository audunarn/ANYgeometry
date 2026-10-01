"""Does the generalized branch reproduce the production CylinderIntersectionCurve for cylinder x cylinder?"""
import sys, math, time
sys.path.insert(0, sys.argv[1])
import numpy as np
import anygeometry._quadric_branch as qb
from anygeometry.analytic_supports import cylinder_cylinder_support
from anygeometry.generators import cylinder
from anygeometry.surfaces import Cylinder

cases = {
    'perpendicular': dict(radius=1., origin=(0., -1.5, 0.), axis=(0., 1., 0.), radial_direction=(1., 0., 0.)),
    'skew': dict(radius=.8, origin=(-.5, -1., -.5), axis=(1., 2., 1.), radial_direction=(1., 0., 0.)),
    'tangent': dict(radius=1., origin=(2., 0., -1.5), axis=(0., 0., 1.), radial_direction=(1., 0., 0.)),
    '10deg': dict(radius=.75, origin=(0., 0., 0.), axis=(math.cos(math.radians(10)), 0., math.sin(math.radians(10))), radial_direction=(0., 1., 0.)),
    'equal-radius perpendicular': dict(radius=1., origin=(0., -1.5, 0.), axis=(0., 1., 0.), radial_direction=(1., 0., 0.)),
}
def _inv(mode_left, mode_right, curve, angle):
    w = (angle - curve.start_angle) / curve.sweep_angle
    w = min(1.0, max(0.0, w))
    if mode_left and mode_right:
        return 2 / math.pi * math.asin(math.sqrt(w)) if w <= .5 else 1 - 2 / math.pi * math.asin(math.sqrt(1 - w))
    if mode_left:
        return math.sqrt(w)
    if mode_right:
        return 1 - math.sqrt(1 - w)
    return w

def invert_old(curve, angle):
    mode = curve.parameterization
    return _inv(mode in ('left_square', 'both_sine'), mode in ('right_square', 'both_sine'), curve, angle)

def invert_new(curve, angle):
    return _inv(curve.left_fold, curve.right_fold, curve, angle)

total = dict(pairs=0, old=0, new=0, same_set=0, mismatched=0, worst=0.0)
for name, spec in cases.items():
    base = cylinder(1., 3., origin=(0., 0., -1.5), circumferential_segments=8)
    other = cylinder(height=3., circumferential_segments=8, **spec)
    A = [f.surface for f in base.faces.values()]
    B = [f.surface for f in other.faces.values()]
    stat = dict(pairs=0, old=0, new=0, same=0, worst=0.0, time_old=0., time_new=0.)
    for a in A:
        for b in B:
            t0 = time.perf_counter()
            old = cylinder_cylinder_support(a, b).curves
            t1 = time.perf_counter()
            try:
                new, _ = qb.support_intersection(a, b, mid_split=True)
            except Exception as e:
                print('   new raised', type(e).__name__, e); new = []
            t2 = time.perf_counter()
            stat['time_old'] += t1 - t0; stat['time_new'] += t2 - t1
            stat['pairs'] += 1; stat['old'] += len(old); stat['new'] += len(new)
            # match charts by (start, end, branch)
            def key(c):
                lo, hi = sorted((c.start_angle, c.start_angle + c.sweep_angle))
                return (round(lo, 9), round(hi, 9), c.branch)
            ko = {key(c): c for c in old}
            kn = {key(c): c for c in new}
            if set(ko) == set(kn):
                stat['same'] += 1
                for k in ko:
                    # compare the same angles: each class inverts its own reparametrization
                    lo, hi = k[0], k[1]
                    for w in np.linspace(.03, .97, 9):
                        angle = lo + w * (hi - lo)
                        t_old = invert_old(ko[k], angle); t_new = invert_new(kn[k], angle)
                        po = ko[k].evaluate(np.array([t_old]))[0]; pn = kn[k].evaluate(np.array([t_new]))[0]
                        stat['worst'] = max(stat['worst'], float(np.linalg.norm(po - pn)))
            else:
                only_old = sorted(set(ko) - set(kn)); only_new = sorted(set(kn) - set(ko))
                if only_old or only_new:
                    print(f'   {name}: mismatch  only-old={only_old[:3]} only-new={only_new[:3]}')
    print(f"{name:28s} facet pairs={stat['pairs']:3d} old charts={stat['old']:3d} new charts={stat['new']:3d} identical chart sets={stat['same']:3d}/{stat['pairs']} "
          f"max point difference={stat['worst']:.2e}  time old {stat['time_old']*1000:.0f} ms new {stat['time_new']*1000:.0f} ms", flush=True)
