"""Independent B1 analytic stations and horizontal-slice reference.

Only the public B1 control net supplies geometry; no certified implementation imports.
Reference error estimates are numerical evidence, never certificate validity.
"""
from __future__ import annotations
import math
import numpy as np

NAMES = ('V', 'Jx', 'Jy', 'Jz', 'A', 'Mu', 'Mv', 'Iuu', 'Ivv', 'Iuv')
# Public b1_bow_surface(): exact binary64 controls and middle weight.
W = math.cos(math.pi / 4)
_GAUSS = {}


def station_width(x, z):
    """Positive y at fixed x,z, derived by solving the rational quadratic.

    t=u/(1-u), q=x/L(z): t²+2wt=q/(1-q).
    y/B=(1+2wt)/(1+2wt+t²). Float controls are retained, not idealized.
    """
    v = z / 4
    length = 5.4 * (1-v) + 6.0 * v
    breadth = 6.0 * (1-v) + 7.5 * v
    q = x / length
    if not 0 <= q < 1:
        raise ValueError('station outside public B1 bow')
    # Rationalized quadratic root avoids cancellation at small q.
    rhs = q / (1-q)
    t = rhs / (math.sqrt(W*W + rhs) + W)
    return breadth * (1+2*W*t) / (1+2*W*t+t*t)


def analytic_stations():
    return [dict(x_m=x, z_m=z, half_breadth_m=station_width(x,z))
            for x in (1.5,2.5,3.5) for z in (.5,1.,1.75,2.5,3.)]


def _root(f, lo, hi):
    a,b=f(lo),f(hi)
    if a == 0: return lo
    if b == 0: return hi
    if a*b > 0: return None
    for _ in range(60):
        mid=(lo+hi)/2
        if mid == lo or mid == hi: break
        c=f(mid)
        if a*c <= 0: hi=mid
        else: lo,a=mid,c
    return (lo+hi)/2


def integrate(f, lo, hi, tol=2e-12, depth=0):
    """Adaptive paired 12/24-point Gauss; discrepancy is an estimator, not a proof."""
    if hi <= lo: return np.zeros_like(f(lo)), 0.0
    values=[]
    for order in (12,24):
        if order not in _GAUSS:
            _GAUSS[order]=np.polynomial.legendre.leggauss(order)
        nodes,weights=_GAUSS[order]
        mid,half=(hi+lo)/2,(hi-lo)/2
        values.append(half*sum(w*np.asarray(f(mid+half*t)) for t,w in zip(nodes,weights)))
    err=float(np.max(np.abs(values[1]-values[0])))
    if err <= tol or depth >= 10:
        if depth >= 10 and err > tol:
            raise ArithmeticError('reference quadrature did not converge')
        return values[1],err
    mid=(lo+hi)/2
    a,ea=integrate(f,lo,mid,tol/2,depth+1)
    b,eb=integrate(f,mid,hi,tol/2,depth+1)
    return a+b,ea+eb


def reference(normal, offset, origin, axes, *, half=False):
    """All ten quantities for public default B1, n.r<=d and dot-product u,v.

    Horizontal slices clip analytic y intervals; their zeroth/first y integrals
    give V,J. Waterplane uses z=(d-nx*x-ny*y)/nz with physical area factor.
    This is not the certified boundary/divergence integration algorithm.
    """
    nx,ny,nz=map(float,normal); d=float(offset)
    o=np.asarray(origin,dtype=float); e=np.asarray(axes,dtype=float)
    if nz <= .9 or abs(nx) > .11 or abs(ny) > .11:
        raise ValueError('reference supports only the frozen near-upright B1 pilot')
    inner_error=0.0
    def station(x):
        nonlocal inner_error
        splits=[.5,3.]
        for sign in ((1,) if half else (-1,1)):
            r=_root(lambda z: nx*x+ny*sign*station_width(x,z)+nz*z-d,.5,3.)
            if r is not None: splits.append(r)
        if half:
            r=(d-nx*x)/nz
            if .5 < r < 3: splits.append(r)
        splits=sorted(set(splits))
        def slab(z):
            b=station_width(x,z); low=0. if half else -b; high=b
            if ny == 0:
                if nx*x+nz*z > d: return np.zeros(4)
            elif ny > 0: high=min(high,(d-nx*x-nz*z)/ny)
            else: low=max(low,(d-nx*x-nz*z)/ny)
            if high <= low: return np.zeros(4)
            width=high-low
            return np.array([width,x*width,(high*high-low*low)/2,z*width])
        total=np.zeros(4); errors=0.
        for lo,hi in zip(splits,splits[1:]):
            val,err=integrate(slab,lo,hi); total+=val; errors+=err
        inner_error=max(inner_error,errors)
        return total
    volume,outer_error=integrate(station,1.5,3.5)
    def plane_station(x):
        z=lambda y: (d-nx*x-ny*y)/nz
        # Bounds encompass every public B1 half breadth, with no geometry execution.
        bounds=[]
        for sign in (-1,1):
            root=_root(lambda y:y-sign*station_width(x,z(y)),-9.,9.)
            if root is None: raise ArithmeticError('waterplane station root unbracketed')
            bounds.append(root)
        low,high=bounds
        if half: low=max(0.,low)
        if ny == 0:
            if not .5 < z(0) < 3.: return np.zeros(6)
        else:
            za=(d-nx*x-nz*.5)/ny; zb=(d-nx*x-nz*3.)/ny
            low=max(low,min(za,zb)); high=min(high,max(za,zb))
        if high <= low: return np.zeros(6)
        # u=a0+a1*y and v=b0+b1*y at fixed x, using the exact supplied dot basis.
        point=np.array([x,0.,z(0)])-o
        constant=e@point; slope=e@np.array([0.,1.,-ny/nz])
        m0=high-low; m1=(high**2-low**2)/2; m2=(high**3-low**3)/3
        a,b=constant; c,f=slope
        return np.array([m0,a*m0+c*m1,b*m0+f*m1,
                         a*a*m0+2*a*c*m1+c*c*m2,
                         b*b*m0+2*b*f*m1+f*f*m2,
                         a*b*m0+(a*f+b*c)*m1+c*f*m2])
    area,area_error=integrate(plane_station,1.5,3.5)
    factor=math.sqrt(1+(nx/nz)**2+(ny/nz)**2)
    area*=factor
    absent=bool(not area[0])
    values={name:float(v) for name,v in zip(NAMES[:4],volume)}
    values.update({name:None if absent else float(v) for name,v in zip(NAMES[4:],area)})
    # Declared estimated numerical allowance includes root/float accumulation;
    # its status remains estimated, separate from certified enclosure semantics.
    err=max(outer_error+2*inner_error,area_error*factor,2e-11)
    return dict(method='independent analytic-station horizontal-slice integration',
                values=values,estimated_absolute_uncertainty=err,
                uncertainty_is_rigorous=False,station_derivation='public rational B1 control net',
                half=half)
