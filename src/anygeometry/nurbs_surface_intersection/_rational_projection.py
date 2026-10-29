"""Exact admission for shared injective separable rational projections."""
from fractions import Fraction as Q
from math import comb

from ..nurbs import _exact_refine
from ._records import RationalProjectionCertificate
from ._structural import _cross, _dot, _greville


def _poll(values,tick):
    for value in values:
        tick()
        yield value


def _product(a, b, tick):
    n, m = len(a)-1, len(b)-1
    result=[]
    for k in range(n+m+1):
        tick()
        result.append(sum((a[i]*b[k-i]*Q(comb(n,i)*comb(m,k-i),comb(n+m,k))
                           for i in range(max(0,k-m),min(n,k)+1)),Q(0)))
    return tuple(result)


def _monotonicity(degree, knots, coordinates, weights, tick=lambda:None):
    """Full active domain, exact derivative numerator signs and rational joins."""
    breaks = sorted(set(Q(k) for k in knots[degree:len(coordinates)+1]))
    points = [(w*x,w) for x,w in zip(coordinates,weights)]
    proofs, sign, previous = [], None, None
    for lo,hi in zip(breaks,breaks[1:]):
        tick()
        _, refined, active = _exact_refine(degree,list(map(Q,knots)),points,lo,hi)
        piece = [refined[i] for i in active]
        if len(piece) != degree+1 or any(p[1] <= 0 for p in piece):
            return None
        a, w = tuple(p[0] for p in piece), tuple(p[1] for p in piece)
        da = tuple(degree*(a[i+1]-a[i]) for i in range(degree))
        dw = tuple(degree*(w[i+1]-w[i]) for i in range(degree))
        numerator = tuple(x-y for x,y in zip(_product(da,w,tick),_product(a,dw,tick)))
        current = 1 if all(x>0 for x in numerator) else -1 if all(x<0 for x in numerator) else 0
        start,end = a[0]/w[0],a[-1]/w[-1]
        if not current or (sign is not None and current != sign) or (previous is not None and start != previous):
            return None
        proofs.append(((lo,hi),numerator,(start,end)))
        sign,previous = current,end
    return tuple(proofs) if proofs else None


def derive_rational_projection(first, second, tick=None):
    tick = tick or (lambda:None)
    tick()
    pu,pv,ku,kv,ca,wa,ur_a,vr_a = first.values
    qu,qv,lu,lv,cb,wb,ur_b,vr_b = second.values
    if (pu,pv,ku,kv)!=(qu,qv,lu,lv) or pu<1 or pv<1:
        return None,"rational projection degrees or knots differ"
    n,m = len(ca),len(ca[0])
    for degree,knots,count in ((pu,ku,n),(pv,kv,m)):
        if any(knots.count(k)>degree-1 for k in set(knots) if knots[degree]<k<knots[count]):
            return None,"rational projection requires C1 original supports at interior knots"
    if len(cb)!=n or any(len(row)!=m for row in cb):
        return None,"rational projection control grids differ"
    weights = tuple(tuple(Q(1) if wa is None else Q(wa[i][j]) for j in range(m)) for i in _poll(range(n),tick))
    other = tuple(tuple(Q(1) if wb is None else Q(wb[i][j]) for j in range(m)) for i in _poll(range(n),tick))
    if any(w<=0 for grid in (weights,other) for row in grid for w in row):
        return None,"rational projection weights are not positive"
    normalized = tuple(tuple(w/weights[0][0] for w in row) for row in _poll(weights,tick))
    if any(normalized[i][j] != other[i][j]/other[0][0] for i in range(n) for j in range(m)):
        return None,"rational projection weights are not proportional"
    wu,wv = tuple(row[0] for row in normalized),normalized[0]
    if any(normalized[i][j] != wu[i]*wv[j] for i in range(n) for j in range(m)):
        return None,"rational projection weights do not factor"
    a = tuple(tuple(tuple(map(Q,p)) for p in row) for row in _poll(ca,tick))
    differences = tuple(tuple(tuple(Q(y)-x for x,y in zip(a[i][j],cb[i][j])) for j in range(m)) for i in _poll(range(n),tick))
    direction = next((d for row in differences for d in row if any(d)),None)
    if direction is None or any(any(_cross(direction,d)) for row in differences for d in row):
        return None,"rational projection differences are coincident or not parallel"
    candidates = [x for x in (_cross(direction,e) for e in ((1,0,0),(0,1,0),(0,0,1))) if any(x)]
    row1 = candidates[0]
    row2 = next(x for x in candidates[1:] if any(_cross(row1,x)))
    projected = tuple(tuple((_dot(row1,p),_dot(row2,p)) for p in row) for row in _poll(a,tick))
    origin = projected[0][0]
    def delta(p):
        return tuple(x-y for x,y in zip(p,origin))
    u = next((delta(projected[i][0]) for i in range(n) if any(delta(projected[i][0]))),None)
    v = next((delta(projected[0][j]) for j in range(m) if any(delta(projected[0][j]))),None)
    if u is None or v is None:
        return None,"rational projection has a constant coordinate"
    det = u[0]*v[1]-u[1]*v[0]
    if not det:
        return None,"rational projection basis is singular"
    def solve(p):
        x,y = delta(p)
        return ((x*v[1]-y*v[0])/det,(u[0]*y-u[1]*x)/det)
    xs = tuple(solve(projected[i][0])[0] for i in range(n))
    ys = tuple(solve(projected[0][j])[1] for j in range(m))
    if any(projected[i][j] != tuple(origin[k]+u[k]*xs[i]+v[k]*ys[j] for k in range(2))
           for i in range(n) for j in range(m)):
        return None,"rational projection controls are not separable"
    proofs = _monotonicity(pu,ku,xs,wu,tick),_monotonicity(pv,kv,ys,wv,tick)
    if any(p is None for p in proofs):
        return None,"rational projection monotonicity or continuity is not certified"
    norm = _dot(direction,direction)
    field = tuple(tuple(normalized[i][j]*_dot(differences[i][j],direction)/norm for j in range(m)) for i in _poll(range(n),tick))
    lo_u,hi_u = max(min(ur_a),min(ur_b)),min(max(ur_a),max(ur_b))
    lo_v,hi_v = max(min(vr_a),min(vr_b)),min(max(vr_a),max(vr_b))
    if lo_u==hi_u or lo_v==hi_v:
        return None,"selected rectangles touch only at a support boundary"
    domain = None if lo_u>hi_u or lo_v>hi_v else ((Q(lo_u),Q(hi_u)),(Q(lo_v),Q(hi_v)))
    return RationalProjectionCertificate(direction,(row1,row2),origin,(u,v),(xs,ys),(wu,wv),proofs,
        _greville(ku,pu,n),_greville(kv,pv,m),field,domain),""
