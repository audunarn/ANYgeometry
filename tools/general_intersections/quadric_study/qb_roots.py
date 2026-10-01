"""Exact branch-vs-quadric roots against dense sign-change detection along the curve."""
import sys, math, time
sys.path.insert(0, sys.argv[1])
import numpy as np
import anygeometry._quadric_branch as qb
from anygeometry.surfaces import Cylinder, Cone, Plane

seed = int(sys.argv[2]); count = int(sys.argv[3])
rng = np.random.default_rng(seed)
unit = lambda v: v / np.linalg.norm(v)


def make(kind, near=1.0):
    axis = unit(rng.normal(size=3))
    helper = unit(np.cross(axis, rng.normal(size=3)))
    full = rng.random() < .6
    sweep = math.tau if full else rng.uniform(.8, 5.5)
    start = rng.uniform(0, math.tau)
    origin = rng.uniform(-1, 1, 3) * near
    if kind == 'cyl':
        return Cylinder(origin, axis, helper, rng.uniform(.5, 2.5), rng.choice([-1, 1]) * rng.uniform(2., 7.), start, sweep)
    if kind == 'cone':
        r0, r1 = rng.uniform(0.2, 1.5), rng.uniform(0.2, 2.5)
        if abs(r1 - r0) < .1:
            r1 = r0 + .6
        return Cone(origin, axis, helper, r0, r1, rng.choice([-1, 1]) * rng.uniform(1.5, 6.), start, sweep)
    u = unit(np.cross(axis, helper)) * rng.uniform(3, 8)
    v = helper * rng.uniform(3, 8)
    return Plane(origin - .5 * u - .5 * v, u, v)


def sign_change_roots(curve, other, n=20001):
    t = np.linspace(0, 1, n)
    x = curve.evaluate(t)
    f = other.value(x) / np.maximum(other.gradient_norm(x), 1e-12)
    ok = np.isfinite(f)
    roots = []
    idx = np.nonzero(ok[:-1] & ok[1:] & (np.sign(f[:-1]) * np.sign(f[1:]) < 0))[0]
    for i in idx:
        lo, hi = t[i], t[i + 1]
        flo = f[i]
        for _ in range(60):
            mid = .5 * (lo + hi)
            xm = curve.evaluate(mid)
            fm = other.value(xm) / max(float(other.gradient_norm(xm)), 1e-12)
            if np.sign(fm) == np.sign(flo):
                lo, flo = mid, fm
            else:
                hi = mid
        roots.append(.5 * (lo + hi))
    return roots


stats = dict(charts=0, checked=0, exact_roots=0, numeric_roots=0, missed=0, tangent_or_end=0, none=0, worst=0.0)
t0 = time.perf_counter()
solve_time = 0.
kinds = [('cone', 'cyl'), ('cyl', 'cone'), ('cone', 'cone'), ('cyl', 'cyl'), ('cone', 'plane'), ('cyl', 'plane')]
for k in range(count):
    fk, sk = kinds[k % len(kinds)]
    a, b = make(fk), make(sk)
    try:
        curves, info = qb.support_intersection(a, b)
    except Exception:
        continue
    for curve in curves:
        stats['charts'] += 1
        pt = curve.evaluate(rng.uniform(.1, .9))
        choice = rng.integers(0, 3)
        if choice == 0:
            n = unit(rng.normal(size=3))
            u = unit(np.cross(n, rng.normal(size=3)))
            other_surface = Plane(pt, u, unit(np.cross(n, u)))
        elif choice == 1:
            other_surface = make('cyl', .5)
        else:
            other_surface = make('cone', .5)
        other = qb.Quadric.from_surface(other_surface)
        t1 = time.perf_counter()
        exact = qb.branch_roots(curve, other)
        solve_time += time.perf_counter() - t1
        stats['checked'] += 1
        if exact is None:
            stats['none'] += 1
            continue
        numeric = sign_change_roots(curve, other)
        stats['exact_roots'] += len(exact); stats['numeric_roots'] += len(numeric)
        for r in numeric:
            if not any(abs(r - e) <= 2e-4 for e in exact):
                stats['missed'] += 1
                if stats['missed'] <= 5:
                    print(f'MISSED sign-change root t={r:.6f}; exact {exact}; chart [{curve.start_angle:.3f},{curve.start_angle + curve.sweep_angle:.3f}] folds=({curve.left_fold},{curve.right_fold}) b={curve.branch}')
            else:
                e = min(exact, key=lambda e: abs(e - r)); stats['worst'] = max(stats['worst'], abs(e - r))
        for e in exact:
            if not any(abs(r - e) <= 2e-4 for r in numeric):
                stats['tangent_or_end'] += 1
print(stats, f'{time.perf_counter() - t0:.0f}s, exact solve {solve_time / max(stats["checked"], 1) * 1000:.1f} ms per chart/quadric')
