import sys, math, time
sys.path.insert(0, sys.argv[1])
import numpy as np
import qb_verify as V
import anygeometry._quadric_branch as qb
from anygeometry.surfaces import Cylinder, Cone, Plane

seed = int(sys.argv[2]); count = int(sys.argv[3]); pairing = sys.argv[4]   # e.g. cone:plane
rng = np.random.default_rng(seed)
unit = lambda v: v / np.linalg.norm(v)


def make(kind, near):
    axis = unit(rng.normal(size=3))
    helper = unit(np.cross(axis, rng.normal(size=3)))
    full = rng.random() < .5
    sweep = math.tau if full else rng.uniform(.6, 5.5)
    start = rng.uniform(0, math.tau)
    origin = rng.uniform(-1, 1, 3) * near
    if kind == 'cyl':
        return Cylinder(origin, axis, helper, rng.uniform(.5, 2.5), rng.choice([-1, 1]) * rng.uniform(2., 7.), start, sweep)
    if kind == 'cone':
        r0, r1 = rng.uniform(0., 1.5), rng.uniform(0., 2.5)
        if abs(r1 - r0) < .1:
            r1 = r0 + .6
        return Cone(origin, axis, helper, r0, r1, rng.choice([-1, 1]) * rng.uniform(1.5, 6.), start, sweep)
    if kind == 'plane':
        u = unit(np.cross(axis, helper)) * rng.uniform(3, 8)
        v = helper * rng.uniform(3, 8)
        return Plane(origin - .5 * u - .5 * v, u, v)


fk, sk = pairing.split(':')
stats = dict(runs=0, charts=0, folds=0, poles=0, contour=0, far=0, errors=0, bad=0, degenerate=0)
t0 = time.perf_counter()
for k in range(count):
    a, b = make(fk, 1.0), make(sk, 1.0)
    try:
        r = V.coverage(a, b, n_angle=721, n_s=361)
    except GeometryError if False else Exception as e:
        if 'asymptotic for every angle' in str(e):
            stats['degenerate'] += 1; continue
        stats['errors'] += 1
        print(f'#{k} EXCEPTION {type(e).__name__}: {str(e)[:150]}')
        continue
    stats['runs'] += 1
    for key, val in (('charts', r['curves']), ('folds', r['folds']), ('poles', r['poles']), ('contour', r['contour'])):
        stats[key] += val
    curves, info = qb.support_intersection(a, b)
    edge = np.array([*info['folds'], *info['poles'], *info['events']])
    far = []
    for angle, s in r['uncovered']:
        d = np.min(np.abs(((edge - angle + np.pi) % (2 * np.pi)) - np.pi)) if len(edge) else np.inf
        if d > 4.5 * r['h']:
            far.append((angle, s, d))
    stats['far'] += len(far)
    flag = ''
    if far or r['worst_resid'] > 1e-10:
        stats['bad'] += 1; flag = '  <== CHECK'
        print(f"#{k} charts={r['curves']} folds={r['folds']} poles={r['poles']} contour={r['contour']} unc={len(r['uncovered'])}/{len(far)} resid={r['worst_resid']:.1e}{flag}", flush=True)
print(pairing, stats, f'{time.perf_counter() - t0:.0f}s')
