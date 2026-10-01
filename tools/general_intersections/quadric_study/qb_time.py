import sys, math, time
sys.path.insert(0, sys.argv[1])
import numpy as np
from cone_probe import build
import anygeometry._quadric_branch as qb

for off in (0., 10.):
    m, cf, kf = build(off)
    for label, (A, B) in (('cone-first', (kf, cf)), ('cylinder-first', (cf, kf))):
        best = 1e9
        for rep in range(3):
            t0 = time.perf_counter(); charts = 0; crossing = 0
            for a in A:
                for b in B:
                    c, info = qb.support_intersection(m.faces[a].surface, m.faces[b].surface)
                    charts += len(c); crossing += bool(c)
            best = min(best, time.perf_counter() - t0)
        print(f'{off:4.1f} deg {label:15s}: {len(A)*len(B)} facet pairs, {crossing} with curves, {charts} charts, {best*1000:.0f} ms total, {best/(len(A)*len(B))*1000:.1f} ms per pair')
