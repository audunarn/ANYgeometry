"""Exact stored-circle Jordan domains and positive oriented boundary census."""
from dataclasses import dataclass
from fractions import Fraction as F
from functools import cmp_to_key
from math import isqrt
from .errors import GeometryError
from .cylinder_charts import _Refusal
from .edge_subcurve_preimages import _arc_circle


def sign(value):
    return value.sign() if isinstance(value,Q) else (value>0)-(value<0)


class Q:
    """Exact a+b sqrt(d), used only within a single intersection field."""
    __slots__=('a','b','d')
    def __init__(self,a,b=0,d=0):
        self.a,self.b,self.d=F(a),F(b),F(d)
        if any(max(abs(x.numerator).bit_length(),x.denominator.bit_length())>8192 for x in (self.a,self.b,self.d)):
            raise _Refusal('rational_bit_budget')
        if self.d<0:raise GeometryError('negative algebraic radicand')
        n,m=isqrt(self.d.numerator),isqrt(self.d.denominator)
        if n*n==self.d.numerator and m*m==self.d.denominator:
            self.a+=self.b*F(n,m);self.b=F(0);self.d=F(0)
    def pair(self,other):
        if not isinstance(other,Q):return Q(other,0,self.d)
        if other.b and self.b and other.d!=self.d:raise GeometryError('mixed algebraic fields')
        return other
    def __add__(self,other):
        other=self.pair(other)
        return Q(self.a+other.a,self.b+other.b,self.d if self.b else other.d)
    __radd__=__add__
    def __neg__(self):return Q(-self.a,-self.b,self.d)
    def __sub__(self,other):return self+-self.pair(other)
    def __rsub__(self,other):return -self+other
    def __mul__(self,other):
        other=self.pair(other);d=self.d if self.b else other.d
        return Q(self.a*other.a+self.b*other.b*d,self.a*other.b+self.b*other.a,d)
    __rmul__=__mul__
    def __truediv__(self,other):return Q(self.a/F(other),self.b/F(other),self.d)
    def sign(self):
        a,b=sign(self.a),sign(self.b)
        if not b:return a
        if not a:return b
        if a==b:return a
        return a*sign(self.a*self.a-self.b*self.b*self.d)
    def __eq__(self,other):return sign(self-other)==0
    def __lt__(self,other):return sign(self-other)<0
    def __le__(self,other):return sign(self-other)<=0
    def __gt__(self,other):return sign(self-other)>0
    def __ge__(self,other):return sign(self-other)>=0


def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def add(a,b):return tuple(x+y for x,y in zip(a,b))
def mul(a,k):return tuple(x*k for x in a)
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def cross(a,b):return a[0]*b[1]-a[1]*b[0]
def equal(a,b):return all(sign(x-y)==0 for x,y in zip(a,b))


def angle_compare(anchor,a,b):
    def half(v):
        y=sign(cross(anchor,v));x=sign(dot(anchor,v))
        return int(y<0 or y==0 and x<0)
    x,y=half(a),half(b)
    return (x>y)-(x<y) if x!=y else -sign(cross(a,b))


@dataclass(frozen=True)
class Curve:
    kind:str
    p:tuple
    q:tuple
    center:tuple=()
    radius2:F=F(0)
    ccw:bool=True
    via:tuple=()


def contains(curve,point,strict=False):
    if equal(point,curve.p) or equal(point,curve.q):return not strict
    if curve.kind=='line':
        d=sub(curve.q,curve.p);v=sub(point,curve.p)
        return sign(cross(d,v))==0 and 0<dot(v,d)<dot(d,d)
    p,q,v=(sub(x,curve.center) for x in (curve.p,curve.q,point))
    if sign(dot(v,v)-curve.radius2):return False
    order=angle_compare(p,v,q)
    return order<0 if curve.ccw else order>0


def intersections(a,b):
    if a.kind=='line' and b.kind=='line':
        d,e=sub(a.q,a.p),sub(b.q,b.p);offset=sub(b.p,a.p);det=cross(d,e)
        if det:
            t,u=cross(offset,e)/det,cross(offset,d)/det
            return [add(a.p,mul(d,t))] if 0<=t<=1 and 0<=u<=1 else []
        if cross(d,offset):return []
        points=[p for p in (a.p,a.q,b.p,b.q) if contains(a,p) and contains(b,p)]
        unique=[]
        for point in points:
            if not any(equal(point,p) for p in unique):unique.append(point)
        if len(unique)>1:raise GeometryError('curved material boundary overlaps/backtracks')
        return unique
    if a.kind=='circle' and b.kind=='line':a,b=b,a
    if a.kind=='line':
        d=sub(a.q,a.p);length=dot(d,d);offset=sub(a.p,b.center)
        base=add(a.p,mul(d,-dot(offset,d)/length))
        rad=(b.radius2-dot(sub(base,b.center),sub(base,b.center)))/length
        direction=d
    else:
        d=sub(b.center,a.center);length=dot(d,d)
        if not length:
            if a.radius2!=b.radius2:return []
            if (contains(a,b.via,True) or contains(b,a.via,True) or
                any(contains(a,p,True) for p in (b.p,b.q)) or any(contains(b,p,True) for p in (a.p,a.q))):
                raise GeometryError('curved material circular spans overlap')
            return [p for p in (a.p,a.q) if contains(b,p)]
        t=(a.radius2-b.radius2+length)/(2*length)
        base=add(a.center,mul(d,t));direction=(-d[1],d[0]);rad=a.radius2/length-t*t
    if rad<0:return []
    points=[tuple(Q(x,y,rad) for x,y in zip(base,mul(direction,k))) for k in ((1,) if rad==0 else (1,-1))]
    return [p for p in points if contains(a,p) and contains(b,p)]


def compile_loops(document,face,frame,proof):
    normal=frame[3];active=[i for i,x in enumerate(normal) if x]
    if len(active)!=1:raise GeometryError('curved planar material requires axis-aligned physical Plane')
    fixed=active[0];axes=tuple(i for i in range(3) if i!=fixed);height=frame[0][fixed]
    def project(point):
        if point[fixed]!=height:raise GeometryError('curved trim is not exactly on common Plane')
        return tuple(point[i] for i in axes)
    result=[]
    for loop in (face['loop'],*face['holes']):
        curves=[]
        for identifier,forward in loop:
            proof.charge();edge=document['edges'][identifier]
            first,last=(document['vertices'][edge[key]] for key in ('start','end'))
            if not forward:first,last=last,first
            p,q=project(first),project(last)
            if p==q:raise GeometryError('zero curved boundary edge')
            kind=edge['curve']['type']
            if kind=='straight':curve=Curve('line',p,q)
            elif kind=='arc':
                via=document['vertices'][edge['curve']['via_vertex']];v=project(via)
                center,radius2,_=_arc_circle((first,via,last));c=project(center)
                ccw=angle_compare(sub(p,c),sub(v,c),sub(q,c))<0
                curve=Curve('circle',p,q,c,radius2,ccw,v)
            else:raise GeometryError('unsupported curved planar trim family: '+kind)
            curves.append(curve)
        if len(curves)<2 or len({c.p for c in curves})!=len(curves):
            raise GeometryError('degenerate/repeated curved loop vertex')
        if any(a.q!=b.p for a,b in zip(curves,(*curves[1:],curves[0]))):
            raise GeometryError('curved document loop is not closed')
        result.append(tuple(curves))
    return tuple(result)


def candidate_pairs(curves,proof):
    boxes=[]
    for index,c in enumerate(curves):
        proof.charge()
        if c.kind=='line':lo=tuple(min(a,b) for a,b in zip(c.p,c.q));hi=tuple(max(a,b) for a,b in zip(c.p,c.q))
        else:
            radius=proof.sqrt(c.radius2).hi;lo=tuple(x-radius for x in c.center);hi=tuple(x+radius for x in c.center)
        boxes.append((lo[0],hi[0],lo[1],hi[1],index))
    active=[]
    for x0,x1,y0,y1,index in sorted(boxes):
        proof.charge();retained=[]
        for entry in active:
            proof.charge()
            right,bottom,top,previous=entry
            if right<x0:continue
            retained.append(entry)
            if min(y1,top)>=max(y0,bottom):yield previous,index
        active=retained
        active.append((x1,y0,y1,index))


def loop_sign(loop,proof):
    integral=proof.i(0)
    for c in loop:
        proof.charge()
        if c.kind=='line':term=proof.i(cross(c.p,c.q))
        else:
            p,q=sub(c.p,c.center),sub(c.q,c.center)
            theta=proof.atan2(cross(p,q),dot(p,q))
            if c.ccw and cross(p,q)<0:theta=proof.add(theta,proof.mul(2,proof.pi_bound()))
            if not c.ccw and cross(p,q)>=0:theta=proof.sub(theta,proof.mul(2,proof.pi_bound()))
            term=proof.add(cross(c.center,sub(c.q,c.p)),proof.mul(c.radius2,theta))
        integral=proof.add(integral,term)
    if integral.lo>0:return 1
    if integral.hi<0:return -1
    raise GeometryError('curved material orientation interval contains zero')


def winding(loop,point,proof):
    # One exact generic ray determines nesting only AFTER complete Jordan and
    # disjoint-boundary certification. It is never a partition witness oracle.
    k=0
    while True:
        proof.charge();direction=(F(1),F(k));k+=1
        rejected=False
        for c in loop:
            proof.charge()
            if cross(direction,sub(c.p,point))==0:
                rejected=True;break
        if rejected:continue
        for c in loop:
            proof.charge()
            if c.kind=='circle' and cross(direction,sub(c.center,point))**2==c.radius2*dot(direction,direction):
                rejected=True;break
        if not rejected:break
    total=0
    for c in loop:
        proof.charge()
        if c.kind=='line':
            edge=sub(c.q,c.p);det=cross(direction,edge)
            if not det:continue
            offset=sub(c.p,point);t,u=cross(offset,edge)/det,cross(offset,direction)/det
            if t>0 and 0<u<1:total+=sign(cross(direction,edge))
        else:
            length=dot(direction,direction);offset=sub(point,c.center)
            base=add(point,mul(direction,-dot(offset,direction)/length))
            rad=(c.radius2-dot(sub(base,c.center),sub(base,c.center)))/length
            if rad<0:continue
            for side in (-1,1):
                hit=tuple(Q(x,y,rad) for x,y in zip(base,mul(direction,side)))
                if dot(sub(hit,point),direction)>0 and contains(c,hit):
                    total+=sign(dot(direction,sub(hit,c.center)))*(1 if c.ccw else -1)
    return total


def certify_domain(loops,proof):
    signs=[]
    for loop in loops:
        for i,j in candidate_pairs(loop,proof):
            hits=intersections(loop[i],loop[j]);allowed=[]
            if (i-j)%len(loop) in (1,len(loop)-1):
                allowed=[p for p in (loop[i].p,loop[i].q) if p in (loop[j].p,loop[j].q)]
            if any(not any(equal(hit,p) for p in allowed) for hit in hits):
                raise GeometryError('curved material boundary is not Jordan')
        signs.append(loop_sign(loop,proof))
    for i,hole in enumerate(loops[1:],1):
        for j in range(i):
            combined=loops[j]+hole;offset=len(loops[j])
            for a,b in candidate_pairs(combined,proof):
                if (a<offset)==(b<offset):continue
                if intersections(combined[a],combined[b]):raise GeometryError('curved hole boundaries intersect')
            if j and (winding(loops[j],hole[0].p,proof) or winding(hole,loops[j][0].p,proof)):
                raise GeometryError('curved holes overlap or nest')
        if not winding(loops[0],hole[0].p,proof):raise GeometryError('curved hole lies outside outer domain')
    return tuple(signs)


def positive_chain_census(domains,proof):
    carriers={};loop_counts=0
    for domain_index,loops in enumerate(domains):
        signs=certify_domain(loops,proof);loop_counts+=len(loops)
        for loop_index,(loop,orientation) in enumerate(zip(loops,signs)):
            weight=(-1 if domain_index==0 else 1)*(1 if loop_index==0 else -1)*orientation
            for c in loop:
                proof.charge()
                if c.kind=='circle':key=('circle',c.center,c.radius2)
                else:
                    d=sub(c.q,c.p);a,b=-d[1],d[0];scale=a or b
                    key=('line',a/scale,b/scale,-dot((a,b),c.p)/scale)
                carriers.setdefault(key,[]).append((c,weight))
    atoms=0
    for key,spans in carriers.items():
        proof.charge();events={p for c,_ in spans for p in (c.p,c.q)}
        center=key[1] if key[0]=='circle' else None
        anchor=sub(min(events),center) if center is not None else None
        def compare(a,b):
            proof.charge()
            return angle_compare(anchor,sub(a,center),sub(b,center)) if center is not None else (a>b)-(a<b)
        ordered=sorted(events,key=cmp_to_key(compare))
        ranks={point:index for index,point in enumerate(ordered)}
        deltas=[0]*len(ordered);initial=0
        for c,weight in spans:
            proof.charge()
            if c.kind=='line':start,end=min(c.p,c.q),max(c.p,c.q);weight*=1 if c.p<c.q else -1
            else:
                start,end=(c.p,c.q) if c.ccw else (c.q,c.p)
                weight*=1 if c.ccw else -1
            lo,hi=ranks[start],ranks[end]
            deltas[lo]+=weight;deltas[hi]-=weight
            if lo>hi:initial+=weight
        count=initial
        for index,delta in enumerate(deltas):
            proof.charge();count+=delta
            if key[0]=='line' and index==len(deltas)-1:break
            if count:raise GeometryError('positive curved material boundary multiplicity differs from source')
            atoms+=1
    return dict(proof='positive Jordan winding linearity and exact directed atomic carrier census',
                jordan_loops=loop_counts,carrier_count=len(carriers),atomic_spans=atoms,
                orientation_semantics='positive outer and negative holes; stored FaceUse orientation remains separate')
