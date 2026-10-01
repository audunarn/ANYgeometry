import sys, math, time, os
sys.path.insert(0, sys.argv[1])
import numpy as np
import qb_verify as V
import anygeometry._quadric_branch as qb
from anygeometry.surfaces import Cylinder, Cone

seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
count = int(sys.argv[3]) if len(sys.argv) > 3 else 40
rng = np.random.default_rng(seed)


def unit(v):
    return v / np.linalg.norm(v)


MODE = sys.argv[4] if len(sys.argv) > 4 else 'generic'


def random_pair():
    cyl_axis = unit(rng.normal(size=3))
    helper = unit(np.cross(cyl_axis, rng.normal(size=3)))
    full = rng.random() < .5
    cyl = Cylinder(rng.uniform(-1, 1, 3), cyl_axis, helper, rng.uniform(.6, 3.0), rng.choice([-1, 1]) * rng.uniform(2., 8.),
                   rng.uniform(0, math.tau), math.tau if full else rng.uniform(.5, 5.5))
    # cone axis: random direction biased to be near-perpendicular to cylinder axis
    tilt = math.radians(rng.choice([0., 10., 25., 45., 70., 89.]) + rng.uniform(-3, 3))
    ortho = unit(np.cross(cyl_axis, rng.normal(size=3)))
    cone_axis = math.cos(tilt) * ortho + math.sin(tilt) * cyl_axis
    helper2 = unit(np.cross(cone_axis, rng.normal(size=3)))
    r0, r1 = rng.uniform(0., 2.), rng.uniform(0., 3.)
    if abs(r1 - r0) < .05 or max(r0, r1) <= 0:
        r1 = r0 + .5
    # keep the apex-side radius non-negative as the kernel requires
    full2 = rng.random() < .5
    start_offset = rng.uniform(-2.5, 2.5) * cyl_axis * 0.0
    origin = cyl.origin + rng.uniform(-1.2, 1.2) * cyl_axis * 0 + rng.uniform(-1., 1., 3) * .8
    height = rng.choice([-1, 1]) * rng.uniform(1.5, 7.)
    if MODE == 'flat':          # wide, nearly flat cones: generators approach parallelism with the cylinder axis
        r0, r1 = (0., rng.uniform(3., 25.)) if rng.random() < .5 else (rng.uniform(0., 1.), rng.uniform(5., 25.))
        height = rng.choice([-1, 1]) * rng.uniform(.5, 3.)
    if MODE == 'coaxial':
        cone_axis = cyl_axis * rng.choice([-1, 1])
        origin = cyl.origin + cyl_axis * rng.uniform(-1, 1)
    if MODE == 'parallel':      # parallel axes, offset
        cone_axis = cyl_axis * rng.choice([-1, 1])
        origin = cyl.origin + cyl_axis * rng.uniform(-1, 1) + unit(np.cross(cyl_axis, rng.normal(size=3))) * rng.uniform(.1, 1.6)
    if MODE == 'apex_on_cyl':   # apex placed on the cylinder wall
        ang = rng.uniform(0, math.tau)
        e1c = np.asarray(cyl.radial_direction); e2c = np.asarray(cyl.circumferential_direction)
        apex = cyl.origin + cyl.radius * (math.cos(ang) * e1c + math.sin(ang) * e2c) + rng.uniform(min(0, cyl.height), max(0, cyl.height)) * np.asarray(cyl.axis)
        r0 = 0.
        origin = apex
    cone = Cone(origin, cone_axis, helper2, r0, r1, height, rng.uniform(0, math.tau),
                math.tau if full2 else rng.uniform(.5, 5.5))
    return cone, cyl


bad = 0
stats = dict(pairs=0, charts=0, folds=0, poles=0, contour=0, uncovered_far=0, errors=0)
t0 = time.perf_counter()
for k in range(count):
    cone, cyl = random_pair()
    for name, a, b in (('cone-first', cone, cyl), ('cyl-first', cyl, cone)):
        try:
            r = V.coverage(a, b, n_angle=721, n_s=361)
        except Exception as e:
            if 'explicit decision' in str(e):
                stats.setdefault('refused_apex', 0); stats['refused_apex'] += 1; continue
            stats['errors'] += 1
            print(f'#{k} {name}: EXCEPTION {type(e).__name__}: {str(e)[:120]}')
            continue
        stats['pairs'] += 1
        stats['charts'] += r['curves']; stats['folds'] += r['folds']; stats['poles'] += r['poles']; stats['contour'] += r['contour']
        # an uncovered point matters only away from fold angles
        curves, info = qb.support_intersection(a, b)
        edge_angles = np.array([*info['folds'], *info['events']])
        far = []
        for angle, s in r['uncovered']:
            d = np.min(np.abs(((edge_angles - angle + np.pi) % (2 * np.pi)) - np.pi)) if len(edge_angles) else np.inf
            if d > 3 * r['h'] * 1.5:
                far.append((angle, s, d))
        stats['uncovered_far'] += len(far)
        flag = ''
        if far or r['worst_resid'] > 1e-11:
            bad += 1
            flag = '  <== CHECK'
        if flag:
            import pickle
            pickle.dump((a, b), open(os.path.join(sys.argv[1], f'fail_{MODE}_{seed}_{k}_{name}.pkl'), 'wb'))
        if flag or k < 3:
            print(f"#{k} {name:10s} charts={r['curves']:2d} folds={r['folds']} poles={r['poles']} contour={r['contour']:5d} "
                  f"unc={len(r['uncovered'])}/{len(far)} off={r['worst_match']:.1e} resid={r['worst_resid']:.1e}{flag}", flush=True)
print(stats, f'bad={bad}', f'{time.perf_counter() - t0:.0f}s')
