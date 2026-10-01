import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from cone_probe import build
from anygeometry import query_intersection
off=float(sys.argv[2])
m,cf,kf=build(off)
def dist_to_face(fid,p):
    f=m.faces[fid]
    return min(m.closest_edge_point(o.edge,p)[2] for loop in (f.loop,)+f.holes for o in loop)
for c in cf:
    for k in kf:
        r=query_intersection(m,m.handle('face',c),m.handle('face',k))
        if r.kind.name!='CROSS': continue
        sizes=[len(comp.witnesses) for comp in r.components]
        interior=0
        for comp in r.components:
            w=[np.asarray(p) for p in comp.witnesses]
            for p in (w[0],w[-1]):
                if min(dist_to_face(c,p),dist_to_face(k,p))>1e-6 and True:
                    interior+=1
        cert=r.certificate
        print(f'pair ({c:2d},{k:2d}) components={len(sizes):3d} witness-counts={sizes[:8]}{"..." if len(sizes)>8 else ""} interior-ended-endpoints={interior} complete={cert.complete if cert else None} resid={cert.max_residual if cert else None}')
