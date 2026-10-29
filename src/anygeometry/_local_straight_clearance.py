"""Exact bounded straight-branch stars; all outside portions compete."""
from fractions import Fraction as Q
from ._local_curvature import LocalApproximationNotCertified
from .nurbs import _sqrt_down
from .plane_nurbs import charge_box
from .predicates import DEFAULT_INTERSECTION_QUALIFICATION_POLICY

def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def dot(a,b):return sum((x*y for x,y in zip(a,b)),Q(0))
def at(segment,t):return tuple(a+t*(b-a) for a,b in zip(*segment))
def part(segment,a,b):return at(segment,a),at(segment,b)

def segment_distance_squared(first,second):
    p,r=first;q,s=second;u=sub(r,p);v=sub(s,q);w=sub(p,q)
    a,b,c,d,e=dot(u,u),dot(u,v),dot(v,v),dot(u,w),dot(v,w)
    if a<=0 or c<=0:raise LocalApproximationNotCertified('degenerate trim segment')
    clamp=lambda t:max(Q(0),min(Q(1),t))
    candidates=[(Q(0),clamp(e/c)),(Q(1),clamp((e+b)/c)),(clamp(-d/a),Q(0)),(clamp((b-d)/a),Q(1))]
    det=a*c-b*b
    if det>0:
        t,z=(b*e-c*d)/det,(a*e-b*d)/det
        if 0<=t<=1 and 0<=z<=1:candidates.append((t,z))
    return min(dot(delta,delta) for t,z in candidates for delta in (sub(at(first,t),at(second,z)),))

def clearance_of_straight_loops(loops,*,check=lambda:None):
    segments=[];stars=[];count=0
    for loop in loops:
        if len(loop)<3:raise LocalApproximationNotCertified('straight loop needs three regular branches')
        offset=len(segments);segments.extend(loop)
        for k,segment in enumerate(loop):
            if segment[1]!=loop[(k+1)%len(loop)][0]:raise LocalApproximationNotCertified('corner anchor not exact')
            u=sub(segment[1],segment[0])
            if dot(u,u)<=0:raise LocalApproximationNotCertified('degenerate branch')
            stars.append(((part(segment,Q(1,8),Q(7,8)),),{offset+k:(Q(0),Q(1))}))
        for k,segment in enumerate(loop):
            previous=(k-1)%len(loop);incoming=loop[previous]
            u=sub(incoming[1],incoming[0]);v=sub(segment[1],segment[0]);a,b,c=dot(u,u),dot(u,v),dot(v,v);det=a*c-b*b
            if det>0:q=tuple((c-b)*x/det+(a-b)*y/det for x,y in zip(u,v))
            elif b>0:q=tuple(x/a for x in u)
            else:raise LocalApproximationNotCertified('corner star cusp unresolved')
            if dot(q,u)<=0 or dot(q,v)<=0:raise LocalApproximationNotCertified('corner monotone projection unresolved')
            stars.append(((part(incoming,Q(7,8),Q(1)),part(segment,Q(0),Q(1,8))),
                {offset+previous:(Q(3,4),Q(1)),offset+k:(Q(0),Q(1,4))}))
    best=None
    for inner,excluded in stars:
        for index,segment in enumerate(segments):
            interval=excluded.get(index)
            pieces=[segment] if interval is None else [part(segment,a,b) for a,b in
                ((Q(0),interval[0]),(interval[1],Q(1))) if a<b]
            for first in inner:
                for second in pieces:
                    check();charge_box();count+=1
                    if count>DEFAULT_INTERSECTION_QUALIFICATION_POLICY.max_boxes_per_pair:
                        raise LocalApproximationNotCertified('clearance pair budget exhausted')
                    value=segment_distance_squared(first,second)
                    if value<=0:raise LocalApproximationNotCertified('nonlocal/contact trim clearance unresolved')
                    best=value if best is None else min(best,value)
    if best is None:raise LocalApproximationNotCertified('trim star cover has no exterior')
    return _sqrt_down(best)
