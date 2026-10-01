"""Independent coverage oracle for exact branch charts.

A dense grid in the first support's (angle, s) chart is contoured with the
second support's implicit equation (marching-squares edge crossings, no use of
A, B, C, discriminants or events). Every contour point that lies in both finite
patches must be reproduced by a selected exact chart, and every selected chart
must lie on both surfaces.
"""
import sys, math, time
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else '.')
import numpy as np
import anygeometry._quadric_branch as qb
from anygeometry.surfaces import Cylinder, Cone, Plane


def param_for_angle(curve, angle):
    """Invert the chart's angular reparametrization (squares/sines at fold ends)."""
    lo, sweep = curve.start_angle, curve.sweep_angle
    w = (angle - lo) / sweep
    w = min(1.0, max(0.0, w))
    if curve.left_fold and curve.right_fold:
        # delta = sweep*sin^2(pi t/2) for t<=.5 ; sweep*(1-cos^2) otherwise; w in [0,1]
        if w <= .5:
            return 2 / math.pi * math.asin(math.sqrt(min(1.0, w)))   # sin^2 = w
        return 1 - 2 / math.pi * math.asin(math.sqrt(min(1.0, 1 - w)))
    if curve.left_fold:
        return math.sqrt(w)
    if curve.right_fold:
        return 1 - math.sqrt(1 - w)
    return w


def contour_points(first_surface, second_surface, n_angle=1441, n_s=721):
    first = qb.RuledSupport.from_surface(first_surface)
    second = qb.Quadric.from_surface(second_surface)
    start, sweep = first.start_angle, first.sweep_angle
    angles = start + sweep * np.linspace(0, 1, n_angle)
    s_lo, s_hi = first.s_range
    ss = np.linspace(s_lo, s_hi, n_s)
    pts = first.point(angles[:, None], ss[None, :])
    F = second.value(pts)
    out = []
    # vertical edges (fixed angle): sign change in s
    sgn = np.sign(F)
    change = (sgn[:, :-1] * sgn[:, 1:]) < 0
    i, j = np.nonzero(change)
    t = F[i, j] / (F[i, j] - F[i, j + 1])
    out.append(np.column_stack((angles[i], ss[j] + t * (ss[j + 1] - ss[j]))))
    # horizontal edges (fixed s): sign change in angle
    change = (sgn[:-1, :] * sgn[1:, :]) < 0
    i, j = np.nonzero(change)
    t = F[i, j] / (F[i, j] - F[i + 1, j])
    out.append(np.column_stack((angles[i] + t * (angles[i + 1] - angles[i]), ss[j])))
    return first, np.vstack(out)


def coverage(first_surface, second_surface, n_angle=1441, n_s=721, tolerance=1e-10):
    curves, info = qb.support_intersection(first_surface, second_surface, tolerance=tolerance)
    first, contour = contour_points(first_surface, second_surface, n_angle, n_s)
    h = max(abs(first.sweep_angle) / (n_angle - 1), (first.s_range[1] - first.s_range[0]) / (n_s - 1))
    worst_match = 0.0
    uncovered = []
    matched = 0
    inside_count = 0
    apex = np.asarray(first.p[0]) if isinstance(first_surface, Cone) else None
    for angle, s in contour:
        point = first.point(angle, s)
        if apex is not None and np.linalg.norm(point - apex) < 1e-6 * max(1.0, abs(first.s_range[1] - first.s_range[0])):
            continue                       # the apex row: a singular point, not a contour
        if not (qb._inside(first_surface, point, 1e-9) and qb._inside(second_surface, point, 1e-9)):
            continue
        inside_count += 1
        best = np.inf
        for curve in curves:
            lo, hi = sorted((curve.start_angle, curve.start_angle + curve.sweep_angle))
            if angle < lo - 1e-12 or angle > hi + 1e-12:
                continue
            t = param_for_angle(curve, angle)
            d = np.linalg.norm(curve.evaluate(np.array([t]))[0] - point)
            best = min(best, d)
        if best < np.inf:
            matched += 1
            worst_match = max(worst_match, best)
        else:
            uncovered.append((angle, s))
    # soundness: every chart lies on both surfaces
    q1, q2 = first_surface, second_surface
    Q1 = qb.Quadric.from_surface(q1); Q2 = qb.Quadric.from_surface(q2)
    worst_resid = 0.0
    for curve in curves:
        x = curve.evaluate(np.linspace(0, 1, 41))
        for Q, surf in ((Q1, q1), (Q2, q2)):
            g = Q.gradient_norm(x)
            ok = g > 1e-6                      # the apex is a singular point of a cone
            d = np.abs(Q.value(x))[ok] / g[ok]
            if d.size:
                worst_resid = max(worst_resid, float(d.max()))
    return dict(curves=len(curves), folds=len(info['folds']), poles=len(info['poles']), contour=inside_count,
                matched=matched, uncovered=uncovered, worst_match=worst_match, h=h, worst_resid=worst_resid)


def report(name, a, b, **kw):
    t0 = time.perf_counter()
    r = coverage(a, b, **kw)
    dt = time.perf_counter() - t0
    print(f"{name:34s} charts={r['curves']:2d} folds={r['folds']} poles={r['poles']} contour-pts={r['contour']:6d} "
          f"uncovered={len(r['uncovered']):4d} worst-3D-offset={r['worst_match']:.2e} (grid h={r['h']:.1e}) "
          f"resid-dist={r['worst_resid']:.1e}  [{dt:.1f}s]", flush=True)
    return r


def cone_at(off_deg, r0=.5, r1=1., h=5., origin=(0., 0., 0.), roll=0., sweep=math.tau, start=0.):
    a = math.radians(off_deg)
    axis = (math.cos(a), 0., math.sin(a))
    return Cone(origin, axis, (0., 1., 0.), r0, r1, h, start, sweep)


if __name__ == '__main__':
    cyl = Cylinder((0., 0., -3.), (0., 0., 1.), (1., 0., 0.), 2., 6., 0., math.tau)
    for off in (0., 10.):
        cone = cone_at(off)
        report(f'cone-first  small cone @{off:4.1f} deg', cone, cyl)
        report(f'cyl-first   small cone @{off:4.1f} deg', cyl, cone)
    big = cone_at(10., 1.5, 3.0, 5.0)
    report('cone-first  wide cone  @10 deg', big, cyl)
    report('cyl-first   wide cone  @10 deg', cyl, big)
    fat = cone_at(10., 1.0, 1.0001 + 3.0, 5.0)           # half-angle ~31 deg
    report('cone-first  fat cone   @10 deg', fat, cyl)
    report('cyl-first   fat cone   @10 deg', cyl, fat)
