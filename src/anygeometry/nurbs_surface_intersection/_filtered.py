"""Directed binary64 interval filter; every arithmetic primitive rounds out.

No BLAS result supplies an enclosure. Nonfinite arithmetic or unsupported
conversion raises Unknown and the caller must use its Fraction reference.
"""
from __future__ import annotations

import math
from fractions import Fraction

from ._intervals import Interval, float_down, float_up


class Unknown(ArithmeticError):
    pass


def _finite(value):
    if not math.isfinite(value):
        raise Unknown("nonfinite directed interval operation")
    return value


class Bound:
    __slots__=("lo","hi")

    def __init__(self,lo,hi=None):
        self.lo=_finite(lo)
        self.hi=_finite(lo if hi is None else hi)
        if self.lo>self.hi:
            raise Unknown("reversed interval")

    @classmethod
    def exact(cls,value):
        value=Fraction(value)
        return cls(float_down(value),float_up(value))

    def __add__(self,other):
        if self.lo==self.hi==0:
            return other
        if other.lo==other.hi==0:
            return self
        return Bound(math.nextafter(self.lo+other.lo,-math.inf),
                     math.nextafter(self.hi+other.hi,math.inf))

    def __sub__(self,other):
        if other.lo==other.hi==0:
            return self
        return Bound(math.nextafter(self.lo-other.hi,-math.inf),
                     math.nextafter(self.hi-other.lo,math.inf))

    def __mul__(self,other):
        if self.lo==self.hi==0 or other.lo==other.hi==0:
            return ZERO
        if self.lo==self.hi==1:
            return other
        if other.lo==other.hi==1:
            return self
        a,b,c,d=self.lo,self.hi,other.lo,other.hi
        if a>=0:
            if c>=0:
                lo,hi=a*c,b*d
            elif d<=0:
                lo,hi=b*c,a*d
            else:
                lo,hi=b*c,b*d
        elif b<=0:
            if c>=0:
                lo,hi=a*d,b*c
            elif d<=0:
                lo,hi=b*d,a*c
            else:
                lo,hi=a*d,a*c
        elif c>=0:
            lo,hi=a*d,b*d
        elif d<=0:
            lo,hi=b*c,a*c
        else:
            lo,hi=min(a*d,b*c),max(a*c,b*d)
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise Unknown("nonfinite interval product")
        return Bound(math.nextafter(lo,-math.inf),math.nextafter(hi,math.inf))

    def reciprocal(self):
        if self.lo<=0<=self.hi:
            raise Unknown("reciprocal includes zero")
        return Bound(math.nextafter(1/self.hi,-math.inf),math.nextafter(1/self.lo,math.inf))

    def scale(self,value):
        return self*Bound.exact(value)

    def fraction(self):
        return Interval(Fraction(self.lo),Fraction(self.hi))

    def excludes_zero(self):
        return self.lo>0 or self.hi<0


ONE=Bound(1.0)
ZERO=Bound(0.0)


def _sum(values):
    values=iter(values)
    result=next(values,ZERO)
    for value in values:
        result=result+value
    return result


def convert(grid):
    return tuple(tuple(tuple(Bound.exact(x) for x in p) for p in row) for row in grid)


def _lerp(a,b,t):
    one_minus=ONE-t
    return _lerp_terms(a,b,t,one_minus)


def _lerp_terms(a,b,t,one_minus):
    return tuple(one_minus*x+t*y for x,y in zip(a,b))


def _axis_reference(vectors,a,b):
    n=len(vectors)-1
    if a.lo==b.lo and a.hi==b.hi:
        work=list(vectors)
        one_minus=ONE-a
        for _ in range(n):
            work=[_lerp_terms(work[i],work[i+1],a,one_minus) for i in range(len(work)-1)]
        return [work[0]]*(n+1)
    out=[]
    for k in range(n+1):
        work=list(vectors)
        for tau in [a]*(n-k)+[b]*k:
            one_minus=ONE-tau
            work=[_lerp_terms(work[i],work[i+1],tau,one_minus) for i in range(len(work)-1)]
        out.append(work[0])
    return out


def _split_axis(vectors,tau):
    work=list(vectors)
    left,right=[work[0]],[work[-1]]
    one_minus=ONE-tau
    while len(work)>1:
        work=[_lerp_terms(work[i],work[i+1],tau,one_minus) for i in range(len(work)-1)]
        left.append(work[0])
        right.append(work[-1])
    return left,list(reversed(right))


def _axis(vectors,a,b):
    """Quadratic directed restriction with conservative original fallback.

    For true parameters a0,b0 inside the supplied outward intervals, both
    split stages contain the exact restricted coefficients. The ratio bound
    includes (b0-a0)/(1-a0); correlations discarded between stages only widen.
    """
    if len(vectors)==1:
        return list(vectors)
    if a.lo==b.lo and a.hi==b.hi:
        return _axis_reference(vectors,a,b)
    if not (0<=a.lo<=a.hi<b.lo<=b.hi<=1):
        return _axis_reference(vectors,a,b)
    try:
        starts_zero=a.lo==a.hi==0
        right=list(vectors) if starts_zero else _split_axis(vectors,a)[1]
        if b.lo==b.hi==1:
            return right
        tau=b if starts_zero else (b-a)*(ONE-a).reciprocal()
        return _split_axis(right,tau)[0]
    except (Unknown,OverflowError,ZeroDivisionError):
        return _axis_reference(vectors,a,b)


def subbox(grid,u0,u1,v0,v1):
    u0,u1,v0,v1=(Bound.exact(x) for x in (u0,u1,v0,v1))
    columns=[_axis([grid[i][j] for i in range(len(grid))],u0,u1) for j in range(len(grid[0]))]
    rows=[[columns[j][i] for j in range(len(columns))] for i in range(len(columns[0]))]
    return tuple(tuple(_axis(row,v0,v1)) for row in rows)


def ranges(grid):
    return tuple(Bound(min(p[k].lo for row in grid for p in row),
                       max(p[k].hi for row in grid for p in row)) for k in range(len(grid[0][0])))


def transform(grid,origin=None,matrix=None):
    origin=None if origin is None else tuple(Bound.exact(x) for x in origin)
    raw_matrix=matrix
    matrix=None if matrix is None else tuple(tuple(Bound.exact(x) for x in row) for row in matrix)
    out=[]
    for row in grid:
        made=[]
        for p in row:
            xyz=p[:3] if origin is None else tuple(p[k]-origin[k]*p[3] for k in range(3))
            if matrix is not None:
                xyz=tuple(_sum(matrix[i][k]*xyz[k] for k in range(3) if raw_matrix[i][k]) for i in range(3))
            made.append(tuple(xyz)+(p[3],))
        out.append(tuple(made))
    return tuple(out)


class Pair:
    def __init__(self,pair,matrix=None):
        if pair.float_cache[0] is None:
            pair.float_cache[0]=tuple(convert(g) for g in (pair.a,pair.b,*pair.da,*pair.db))
        self.grids=tuple(transform(g,pair.origin,matrix) for g in pair.float_cache[0])
        self.original=pair
        self.enclosure_cache={}
        self.cache_hits=0
        self.cache_misses=0

    def enclose(self,box):
        key=tuple(tuple(axis) for axis in box)
        if key in self.enclosure_cache:
            result=self.enclosure_cache.pop(key)
            self.enclosure_cache[key]=result
            self.cache_hits+=1
            return result
        self.cache_misses+=1
        (u0,u1),(v0,v1),(s0,s1),(t0,t1)=box
        bounds=[]
        for i,g in enumerate(self.grids):
            bounds.append(ranges(subbox(g,*( (u0,u1,v0,v1) if i in (0,2,3) else (s0,s1,t0,t1)))))
        result=bounds[0],bounds[1],(bounds[2],bounds[3]),(bounds[4],bounds[5])
        if len(self.enclosure_cache)>=4:
            self.enclosure_cache.pop(next(iter(self.enclosure_cache)))
        self.enclosure_cache[key]=result
        return result

    def section_value(self,free,y0,phi):
        it=iter(y0)
        box=[(phi.lo,phi.hi) if k==free else (lambda x:(x,x))(next(it)) for k in range(4)]
        (u0,u1),(v0,v1),(s0,s1),(t0,t1)=box
        a=subbox(self.grids[0],u0,u1,v0,v1)
        b=subbox(self.grids[1],s0,s1,t0,t1)
        varying,point=(a,b[0][0]) if free<2 else (b,a[0][0])
        coefficients=[]
        for row in varying:
            for p in row:
                value=tuple(p[k]*point[3]-point[k]*p[3] for k in range(3))
                coefficients.append(value if free<2 else tuple(ZERO-x for x in value))
        return tuple(Bound(min(p[k].lo for p in coefficients),max(p[k].hi for p in coefficients))
                     for k in range(3))


def prepared(pair):
    view=getattr(pair,"filtered_view",None)
    if view is None:
        view=Pair(pair)
        pair.filtered_view=view
    return view


def free_order(pair,cell):
    """Numerical chart proposal ONLY; no enclosure or classification authority."""
    midpoint=tuple(((a+b)/2,)*2 for a,b in cell)
    _,jac=pair.value_and_jacobian(prepared(pair).enclose(midpoint))
    m=[[(x.lo/2+x.hi/2) for x in row] for row in jac]
    def det(a):
        return (a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
                -a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
                +a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]))
    scores={}
    for free in range(4):
        if cell[free][0]==cell[free][1]:
            continue
        width=float(cell[free][1]-cell[free][0])
        if width<=0 or not math.isfinite(width):
            raise Unknown("chart proposal width unavailable")
        score=abs(det([[row[k] for k in range(4) if k!=free] for row in m]))/width
        if not math.isfinite(score):
            raise Unknown("chart proposal cofactor nonfinite")
        scores[free]=score
    if scores and not any(scores.values()):
        raise Unknown("chart proposal minors unresolved")
    return sorted(scores,key=lambda k:(-scores[k],k))


def certificate(pair,box,free,y,phi,*,for_root_filter=False):
    """Directed-float K image or Unknown; C is a proposal frozen to exact rationals."""
    import numpy as np
    from ._certificate import LocalArcCertificate
    original=prepared(pair)
    it=iter(y)
    region=[(phi.lo,phi.hi) if k==free else (lambda x:(x.lo,x.hi))(next(it)) for k in range(4)]
    enc=original.enclose(region)
    if enc[0][3].lo<=0 or enc[1][3].lo<=0:
        raise Unknown("weight positivity unresolved")
    immersed=pair.immersed(enc)
    if not immersed and not for_root_filter:
        raise Unknown("immersion unresolved")
    columns=[k for k in range(4) if k!=free]
    y0=[x.mid for x in y]
    it=iter(y0)
    midpoint=[(phi.mid,)*2 if k==free else (lambda x:(x,x))(next(it)) for k in range(4)]
    _,jac=pair.value_and_jacobian(original.enclose(midpoint))
    matrix=np.array([[(jac[i][k].lo/2+jac[i][k].hi/2) for k in columns] for i in range(3)])
    largest=float(np.max(np.abs(matrix)))
    if not math.isfinite(largest) or largest==0:
        raise Unknown("midpoint preconditioner unavailable")
    exponent=math.frexp(largest)[1]-1
    scale=math.ldexp(1.,exponent)
    try:
        inverse=np.linalg.inv(matrix/scale)
    except np.linalg.LinAlgError as error:
        raise Unknown("midpoint minor singular") from error
    if not np.all(np.isfinite(inverse)):
        raise Unknown("preconditioner nonfinite")
    c=tuple(tuple(Fraction(float(inverse[i,k]))/Fraction(scale) for k in range(3)) for i in range(3))
    conditioned=Pair(pair,c)
    cf=conditioned.section_value(free,y0,phi)
    _,cj=pair.value_and_jacobian(conditioned.enclose(region))
    image=[]
    for i in range(3):
        result=Bound.exact(y0[i])-cf[i]
        for j,col in enumerate(columns):
            residual=Bound(1. if i==j else 0.)-cj[i][col]
            delta=Bound(float_down(y[j].lo),float_up(y[j].hi))-Bound.exact(y0[j])
            result=result+residual*delta
        image.append(result.fraction())
    strict=all(image[i].strictly_inside(y[i]) for i in range(3))
    good=strict and immersed
    return LocalArcCertificate(good,"directed-float strict graph" if good else "directed-float K image only",
        free,box,(phi.lo,phi.hi),tuple(image),True,immersed,good,False,c)
