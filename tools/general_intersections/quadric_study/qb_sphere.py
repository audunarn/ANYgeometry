import sys, math, time
sys.path.insert(0, sys.argv[1])
import numpy as np
import qb_verify as V
import anygeometry._quadric_branch as qb
from anygeometry.surfaces import Cylinder, Cone

rng = np.random.default_rng(int(sys.argv[2]))
unit = lambda v: v / np.linalg.norm(v)
stats = dict(runs=0, charts=0, folds=0, poles=0, contour=0, far=0, bad=0)
for k in range(int(sys.argv[3])):
    axis = unit(rng.normal(size=3)); helper = unit(np.cross(axis, rng.normal(size=3)))
    if k % 2:
        first = Cylinder(rng.uniform(-1, 1, 3), axis, helper, rng.uniform(.5, 2.), rng.choice([-1, 1]) * rng.uniform(3., 7.), rng.uniform(0, 6), math.tau)
    else:
        r0, r1 = rng.uniform(.2, 1.), rng.uniform(1.2, 2.5)
        first = Cone(rng.uniform(-1, 1, 3), axis, helper, r0, r1, rng.choice([-1, 1]) * rng.uniform(2., 6.), rng.uniform(0, 6), math.tau)
    sphere = qb.Quadric.sphere(rng.uniform(-1.5, 1.5, 3), rng.uniform(.8, 3.0))
    r = V.coverage(first, sphere, n_angle=721, n_s=361)
    stats['runs'] += 1
    for key, val in (('charts', r['curves']), ('folds', r['folds']), ('poles', r['poles']), ('contour', r['contour'])):
        stats[key] += val
    curves, info = qb.support_intersection(first, sphere)
    edge = np.array([*info['folds'], *info['events']])
    far = [1 for angle, s in r['uncovered'] if not len(edge) or np.min(np.abs(((edge - angle + np.pi) % (2 * np.pi)) - np.pi)) > 4.5 * r['h']]
    stats['far'] += len(far)
    if far or r['worst_resid'] > 1e-10:
        stats['bad'] += 1
        print('CHECK', k, r['curves'], len(far), r['worst_resid'])
print('sphere as second support:', stats)
