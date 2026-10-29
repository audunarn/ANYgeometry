"""Generic signed-linear clipping of a triangle: no wave physics, only the region where a linear field is positive.

A linear scalar field ``g`` is given by its values at the three vertices of a triangle (or, for a general
convex polygon, at the polygon vertices).  The clipped region ``g > 0`` is returned as a convex polygon in
barycentric (triangle) or caller-supplied parametric coordinates, plus a triangle fan of that polygon.  Because
barycentric coordinates are an affine map of any physical or reference triangle, the same result serves the
physical triangle and its reference-element image.  Wet means ``g > 0`` (``g <= 0`` everywhere is dry/empty);
an edge with ``g == 0`` is dry unless the rest of the polygon is wet.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class SignedClip:
    state: str                 # 'dry' | 'partial' | 'wet'
    parameters: np.ndarray     # (m, k) polygon vertices in the supplied parametric coordinates (m=0 when dry)
    values: np.ndarray         # (m,) field value at each polygon vertex (interpolated)
    fan: tuple                 # triangles of vertex indices covering the polygon
    area_fraction: float       # polygon area / input polygon area (affine invariant)


def _finite_vector(values,name):
    v=np.asarray(values,dtype=float)
    if v.ndim!=1 or not np.isfinite(v).all():raise ValueError(f'{name} must be a finite one-dimensional array')
    return v


def clip_polygon_positive(parameters,values):
    """Clip a convex polygon with vertex ``parameters`` (n,k) and linear field ``values`` (n,) to ``values > 0``."""
    p=np.asarray(parameters,dtype=float);g=_finite_vector(values,'field values')
    if p.ndim!=2 or len(p)<3 or len(p)!=len(g) or not np.isfinite(p).all():
        raise ValueError('a convex polygon with one finite field value per vertex is required')
    if np.all(g<=0.) or not np.any(g>0.):
        return SignedClip('dry',np.zeros((0,p.shape[1])),np.zeros(0),(),0.)
    if np.all(g>=0.):
        out_p,out_g,state=p.copy(),g.copy(),'wet'
    else:
        out_p=[];out_g=[]
        n=len(p)
        for i in range(n):
            a,b=i,(i+1)%n
            ga,gb=g[a],g[b]
            if ga>0.:
                out_p.append(p[a]);out_g.append(ga)
            if (ga>0.)!=(gb>0.):
                # Crossing of the zero level: ga and gb differ in sign or one is exactly zero.
                t=ga/(ga-gb)
                out_p.append(p[a]+t*(p[b]-p[a]));out_g.append(0.)
        out_p=np.array(out_p);out_g=np.array(out_g);state='partial'
    fan=tuple((0,i,i+1) for i in range(1,len(out_p)-1))
    area=_fan_area(out_p,fan);full=_fan_area(p,tuple((0,i,i+1) for i in range(1,len(p)-1)))
    if not area>0.:
        return SignedClip('dry',np.zeros((0,p.shape[1])),np.zeros(0),(),0.)
    fraction=min(1.,area/full) if full>0. else 1.
    if state=='wet' or math.isclose(fraction,1.,rel_tol=0.,abs_tol=1e-15):state='wet'
    return SignedClip(state,out_p,out_g,fan,fraction)


def _fan_area(points,fan):
    """Area measure in the first two parametric coordinates (affine-invariant ratio is all that is used)."""
    total=0.
    for a,b,c in fan:
        u=points[b]-points[a];v=points[c]-points[a]
        if points.shape[1]==2:total+=.5*abs(u[0]*v[1]-u[1]*v[0])
        else:total+=.5*float(np.linalg.norm(np.cross(np.r_[u,0.] if len(u)==2 else u,np.r_[v,0.] if len(v)==2 else v)))
    return total


def clip_triangle_positive(values):
    """Wet region of a triangle where the linear field with vertex ``values`` (3,) is positive.

    ``parameters`` of the result are barycentric coordinates (m,3) summing to one in each row.
    """
    g=_finite_vector(values,'field values')
    if g.shape!=(3,):raise ValueError('exactly three vertex values required')
    return clip_polygon_positive(np.eye(3),g)
